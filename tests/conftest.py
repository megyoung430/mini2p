"""Synthetic recordings: tiny interleaved TIFFs with a known phase error per chunk."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
from scipy.ndimage import gaussian_filter
from suite2p.registration import bidiphase as s2p_bidiphase

LY, LX = 48, 96


def textured_image(rng: np.random.Generator, shape=(LY, LX), sigma=1.5) -> np.ndarray:
    """Isotropic blob texture, zero mean and unit SD."""
    image = gaussian_filter(rng.standard_normal(shape), sigma)
    return (image - image.mean()) / image.std()


def misalign(frames: np.ndarray, true_phase: int) -> np.ndarray:
    """Introduce a scan-phase error that a correction of ``true_phase`` undoes."""
    return s2p_bidiphase.shift(frames.astype(np.float32, copy=True), -int(true_phase))


def make_recording(
    rng: np.random.Generator,
    true_phase_by_chunk: list[int],
    chunk_frames: int,
    n_channels: int = 2,
    phase_by_channel: list[list[int]] | None = None,
    noise_by_channel: list[list[float]] | None = None,
) -> list[np.ndarray]:
    """One int16 (frames, Ly, Lx) stack per channel, ScanImage-like offset and noise.

    ``phase_by_channel`` / ``noise_by_channel`` give per-channel, per-chunk phase
    errors and noise SDs; by default every channel shares ``true_phase_by_chunk``.
    """
    anatomy = textured_image(rng)
    n_chunks = len(true_phase_by_chunk)
    phases = phase_by_channel or [true_phase_by_chunk] * n_channels
    noises = noise_by_channel or [[40.0] * n_chunks] * n_channels
    stacks = []
    for channel in range(n_channels):
        gain = 400.0 if channel == 0 else 250.0
        chunks = []
        for true_phase, noise in zip(phases[channel], noises[channel], strict=True):
            frames = 200 + gain * anatomy + noise * rng.standard_normal((chunk_frames, LY, LX))
            chunks.append(misalign(frames, true_phase))
        stacks.append(np.clip(np.concatenate(chunks), -32768, 32767).astype(np.int16))
    return stacks


def write_interleaved_tiff(path: Path, stacks: list[np.ndarray]) -> Path:
    pages = np.stack(stacks, axis=1).reshape(-1, *stacks[0].shape[1:])
    tifffile.imwrite(path, pages, photometric="minisblack")
    return path


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture
def recording(tmp_path, rng):
    """Three 20-frame chunks (the last one short) with true phases +3, -2, +5."""
    stacks = make_recording(rng, [3, -2, 5], chunk_frames=20)
    stacks = [s[:55] for s in stacks]
    return write_interleaved_tiff(tmp_path / "raw_00001.tif", stacks), stacks


@pytest.fixture
def mixed_recording(tmp_path, rng):
    """Channels disagree on phase in chunk 1, and each channel is cleaner in different chunks.

    Red (structural) phases +3, -2, +5; green (functional) +3, +1, +5. Green is far
    cleaner in chunk 1 and red in chunks 0 and 2, so 'best_score' must switch sources.
    """
    stacks = make_recording(
        rng,
        [0, 0, 0],
        chunk_frames=20,
        phase_by_channel=[[3, -2, 5], [3, 1, 5]],
        noise_by_channel=[[20.0, 3000.0, 20.0], [3000.0, 5.0, 3000.0]],
    )
    return write_interleaved_tiff(tmp_path / "mixed_00001.tif", stacks), stacks


def soma_field(
    rng: np.random.Generator, shape=(256, 256), n_cells=60, vignette=0.0, dark_quadrant=False
) -> np.ndarray:
    """Mean-image-like picture: sparse 5-8 px somata on smooth neuropil, plus noise.

    ``vignette`` dims the edges (0 = none, 0.6 = strong); ``dark_quadrant`` blanks
    the lower-right quadrant, like a large vessel or the edge of the window.
    """
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    image = 40 + 25 * gaussian_filter(rng.standard_normal(shape), 12)
    for cy, cx, radius in zip(
        rng.uniform(0, h, n_cells),
        rng.uniform(0, w, n_cells),
        rng.uniform(2.5, 4.0, n_cells),
        strict=True,
    ):
        image += 120 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * radius**2))
    if vignette:
        r2 = ((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2
        image *= 1 - vignette * np.clip(r2, 0, 1)
    if dark_quadrant:
        image[h // 2 :, w // 2 :] = image[h // 2 :, w // 2 :].mean() * 0.1
    return image + rng.normal(0, 4, shape)


def as_contrast_png(image: np.ndarray, path: Path, black=5.0, white=99.5, gamma=1.2) -> Path:
    """Save the way the QC export does: percentile clip, gamma, grey colormap PNG."""
    import matplotlib.pyplot as plt

    lo, hi = np.percentile(image, [black, white])
    plt.imsave(path, np.clip((image - lo) / (hi - lo), 0, 1) ** gamma, cmap="gray", vmin=0, vmax=1)
    return path


def fake_calibration(shape=(48, 96)):
    """Stand-in for distortion_correct_full_recording: a gentle barrel-like warp."""

    def load_transform(path, z_um):
        return {"k": 0.05}, 0.0, "fake-schema", 12

    def make_maps(transform, image_shape):
        yy, xx = np.mgrid[0 : image_shape[0], 0 : image_shape[1]].astype(np.float64)
        cy, cx = (image_shape[0] - 1) / 2, (image_shape[1] - 1) / 2
        r2 = ((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2
        scale = 1 + transform["k"] * r2
        return cy + (yy - cy) * scale, cx + (xx - cx) * scale

    def crop_bounds(map_y, map_x, image_shape, padding):
        return (
            2 + padding,
            image_shape[0] - 2 - padding,
            3 + padding,
            image_shape[1] - 3 - padding,
        )

    return SimpleNamespace(
        load_transform=load_transform, make_maps=make_maps, crop_bounds=crop_bounds
    )
