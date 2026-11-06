"""Per-frame photon-free level: the reading of a pixel that caught no photon.

ScanImage subtracts a detector offset it measures once, at the start of an acquisition.
The true dark level then drifts (by tens of counts within a minute on the Mini2P), so
pixels with no photon do not read 0 and every pixel of a frame is shifted by the same
amount. At the photon rates of these recordings (well under one photon per pixel per
frame in green) most readings in a frame hold no photon, so the mode of the frame's
values is that level. Subtracting it per frame puts F, Fneu and the background traces on
a true zero, which a ΔF/F0 needs; it removes no light. For a bright channel (a dense
structural label) the mode is taken over its dimmest pixels, where photon-free readings
still dominate.

Traces must be weighted means of pixels with weights summing to 1 (Suite2p's F with its
normalized lam, Fneu, the background traces) for the level to subtract exactly.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

# Histogram smoothing (counts) for the first guess, the mode: photon-free readings spread
# over tens of counts of electronic noise, so single-count bins are too noisy.
HIST_SMOOTH_COUNTS = 3
# Readings within this many noise SDs of the level count as photon-free; a photon adds
# hundreds of counts, so photon readings stay outside.
WINDOW_SDS = 2.5
REFINE_STEPS = 5
# Fraction of the readings that must be photon-free for the level to be measurable;
# brighter pixels need a dim-pixel mask.
MIN_PHOTON_FREE_FRACTION = 0.3


def photon_free_level(values: np.ndarray) -> float:
    """The level of integer detector readings that hold no photon.

    First guess: the mode. Then, a few times over: the noise SD from the readings just
    below the level (no photon reading is lower than the level), and the level as the mean
    of the readings within WINDOW_SDS of it, which converges on the centre of the
    photon-free peak with far less scatter than the mode.
    """
    values = np.asarray(values, np.float64).ravel()
    if values.size == 0:
        raise ValueError("There are no values to take a photon-free level from")
    low = int(values.min())
    counts = np.bincount((values - low).astype(np.int64), minlength=1).astype(np.float64)
    smooth = gaussian_filter1d(counts, HIST_SMOOTH_COUNTS)
    peak = int(np.argmax(smooth))
    left = np.flatnonzero(smooth[:peak] < smooth[peak] / 2)
    half_width = peak - left[-1] if left.size else HIST_SMOOTH_COUNTS
    level, sd = float(peak + low), half_width / 1.1774  # Gaussian half width at half max
    for _ in range(REFINE_STEPS):
        below = values[(values < level) & (values > level - 2 * WINDOW_SDS * sd)]
        if below.size:
            sd = max(float(np.sqrt(np.mean((below - level) ** 2))), 1.0)
        near = values[np.abs(values - level) <= WINDOW_SDS * sd]
        level = float(near.mean())
    fraction = near.size / values.size
    if fraction < MIN_PHOTON_FREE_FRACTION:
        raise ValueError(
            f"Only {fraction:.0%} of the readings lie near their most common value, so it is "
            "not a photon-free level (too much light); measure it over dimmer pixels"
        )
    return level


def photon_free_levels(frames: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Photon-free level of every frame of a (frames, Ly, Lx) stack, over *mask* pixels."""
    frames = np.asarray(frames)
    if mask is not None and np.shape(mask) != frames.shape[1:]:
        raise ValueError(f"Mask shape {np.shape(mask)} vs frames {frames.shape[1:]}")
    pick = (lambda f: f[mask]) if mask is not None else (lambda f: f)
    return np.array([photon_free_level(pick(frame)) for frame in frames])


def dim_pixel_mask(
    mean_image: np.ndarray, percentile: float, region: tuple[slice, slice] | None = None
) -> np.ndarray:
    """The dimmest *percentile* % of the pixels of *mean_image* within *region*."""
    if not 0 < percentile <= 100:
        raise ValueError(f"percentile must be in (0, 100], got {percentile}")
    image = np.asarray(mean_image, np.float64)
    inside = np.zeros(image.shape, bool)
    inside[region if region is not None else (slice(None), slice(None))] = True
    if not inside.any():
        raise ValueError("The region holds no pixels")
    threshold = np.percentile(image[inside], percentile)
    return inside & (image <= threshold)


def subtract_level(traces: np.ndarray, level: np.ndarray) -> np.ndarray:
    """*traces* (n_frames,) or (n_rows, n_frames) minus the per-frame *level*."""
    traces, level = np.asarray(traces, np.float64), np.asarray(level, np.float64)
    if traces.ndim not in (1, 2):
        raise ValueError(f"Traces must be (n_frames,) or (n_rows, n_frames), got {traces.shape}")
    if level.ndim != 1 or traces.shape[-1] != level.size:
        raise ValueError(f"Level has {level.size} frames, traces {traces.shape[-1]}")
    if not np.isfinite(level).all():
        raise ValueError("The photon-free level has values that are not finite")
    return traces - level
