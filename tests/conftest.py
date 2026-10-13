"""Synthetic recordings: tiny interleaved TIFFs with a known phase error per chunk."""

from __future__ import annotations

from pathlib import Path

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
