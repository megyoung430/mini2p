"""Per-frame photon-free level: the reading of a pixel that caught no photon.

ScanImage subtracts a detector offset it measures once, at the start of an acquisition.
The true dark level then drifts (by tens of counts within a minute on the Mini2P), so
pixels with no photon do not read 0 and every pixel of a frame is shifted by the same
amount. At the photon rates of these recordings (well under one photon per pixel per
frame in green) most readings in a frame hold no photon, so the photon-free readings form
the lowest peak of the frame's histogram, and its centre is the level. Subtracting it per
frame puts F, Fneu and the background traces on a true zero, which a ΔF/F0 needs; it
removes no light. For a bright channel (a dense structural label) the level is taken over
its dimmest pixels, where photon-free readings still dominate.

One level per frame is enough: on the Mini2P (7 Aug 2026 session) bands of columns or
rows differ from the frame's level by a fixed 3-5 counts at most and follow it to within
1-2 counts over time, while the level itself moves by over 100 counts.

Traces must be weighted means of pixels with weights summing to 1 (Suite2p's F with its
normalized lam, Fneu, the background traces) for the level to subtract exactly.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

# Histogram smoothing (counts) for the first guess: photon-free readings spread over tens
# of counts of electronic noise, so single-count bins are too noisy.
HIST_SMOOTH_COUNTS = 3
# Readings within this many noise SDs of the level count as photon-free; a photon adds
# hundreds of counts, so photon readings stay outside.
WINDOW_SDS = 2.5
REFINE_STEPS = 5
# Fraction of the readings that must be photon-free for the level to be measurable;
# brighter pixels need a dim-pixel mask.
MIN_PHOTON_FREE_FRACTION = 0.3
# Readings this many noise SDs below a correct level are rare (3e-5 of a Gaussian; real
# frames have occasional glitches, up to ~2% on the Mini2P). An estimate on a photon peak
# has every photon-free reading below it, at least ~15% of them whenever at least
# MIN_PHOTON_FREE_FRACTION lie near it, so more than MAX_FAR_BELOW means a photon peak.
FAR_BELOW_SDS = 4
MAX_FAR_BELOW = 0.05
# Fewest pixels a frame's level is measured on (scatter ~20 / sqrt(n) counts).
MIN_PIXELS = 500
# The level moves by a few counts per frame; a photon adds ~300-400 counts on the Mini2P,
# so a larger step between frames means one frame's estimate is wrong.
MAX_STEP_COUNTS = 100
# What the estimate depends on, for cache keys: change any and cached levels are stale.
ESTIMATOR = {
    "version": 3,
    "hist_smooth_counts": HIST_SMOOTH_COUNTS,
    "window_sds": WINDOW_SDS,
    "refine_steps": REFINE_STEPS,
    "min_photon_free_fraction": MIN_PHOTON_FREE_FRACTION,
    "far_below_sds": FAR_BELOW_SDS,
    "max_far_below": MAX_FAR_BELOW,
    "min_pixels": MIN_PIXELS,
    "max_step_counts": MAX_STEP_COUNTS,
}
_INT16 = np.iinfo(np.int16)


def photon_free_level(values: np.ndarray) -> float:
    """The level of integer detector readings that hold no photon (see the module text).

    First guess: the mode. Refined by taking, a few times over, the noise SD from the
    readings just below the level (no photon reading is lower than the level) and the
    level as the mean of the readings within WINDOW_SDS of it. If that settled on a photon
    peak, it starts again from the readings below it.
    """
    values = _readings(values)
    level, sd = _refine(values, *_mode_guess(values))
    if _fraction_far_below(values, level, sd) > MAX_FAR_BELOW:
        lower = values[values < level - FAR_BELOW_SDS * sd]
        level, sd = _refine(values, *_mode_guess(lower))
    _check_photon_free(values, level, sd)
    return level


def _readings(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, np.float64).ravel()
    if values.size == 0:
        raise ValueError("There are no values to take a photon-free level from")
    if not np.isfinite(values).all():
        raise ValueError("Readings must be finite detector values (got NaN or inf)")
    return values


def _mode_guess(values: np.ndarray) -> tuple[float, float]:
    """The most common reading and a noise SD from its peak's half width on the left."""
    low = int(values.min())
    counts = np.bincount((values - low).astype(np.int64)).astype(np.float64)
    smooth = gaussian_filter1d(counts, HIST_SMOOTH_COUNTS)
    peak = int(np.argmax(smooth))
    left = np.flatnonzero(smooth[:peak] < smooth[peak] / 2)
    if not left.size:
        raise ValueError(
            "The most common reading is at the bottom of the range, with no readings below "
            "it: the values look clipped at a floor, so the photon-free level is unknown"
        )
    return float(peak + low), (peak - left[-1]) / 1.1774  # Gaussian half width at half max


def _refine(values: np.ndarray, level: float, sd: float) -> tuple[float, float]:
    for _ in range(REFINE_STEPS):
        below = values[(values < level) & (values > level - 2 * WINDOW_SDS * sd)]
        if below.size:
            sd = max(float(np.sqrt(np.mean((below - level) ** 2))), 1.0)
        level = float(values[np.abs(values - level) <= WINDOW_SDS * sd].mean())
    return level, sd


def _fraction_far_below(values: np.ndarray, level: float, sd: float) -> float:
    return float(np.mean(values < level - FAR_BELOW_SDS * sd))


def _check_photon_free(values: np.ndarray, level: float, sd: float) -> None:
    near = float(np.mean(np.abs(values - level) <= WINDOW_SDS * sd))
    if near < MIN_PHOTON_FREE_FRACTION:
        raise ValueError(
            f"Only {near:.0%} of the readings lie near their most common value, so it is "
            "not a photon-free level (too much light); measure it over dimmer pixels"
        )
    far = _fraction_far_below(values, level, sd)
    if far > MAX_FAR_BELOW:
        raise ValueError(
            f"{far:.0%} of the readings lie far below the level found ({level:.0f}), so it "
            "is a photon peak, not the photon-free one"
        )
    if level - WINDOW_SDS * sd <= _INT16.min or level + WINDOW_SDS * sd >= _INT16.max:
        raise ValueError(f"The level {level:.0f} is at the int16 limit: the values are clipped")


def photon_free_levels(frames: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Photon-free level of every frame of a (frames, Ly, Lx) stack, over *mask* pixels."""
    frames = np.asarray(frames)
    if mask is not None:
        mask = np.asarray(mask)
        if mask.dtype != bool or mask.shape != frames.shape[1:]:
            raise ValueError(
                f"Mask must be boolean of the frame shape {frames.shape[1:]}, got "
                f"{mask.dtype} {mask.shape}"
            )
    n_pixels = int(mask.sum()) if mask is not None else int(np.prod(frames.shape[1:]))
    if n_pixels < MIN_PIXELS:
        raise ValueError(f"{n_pixels} pixels are too few for a level (need {MIN_PIXELS})")
    levels = np.array([photon_free_level(f[mask] if mask is not None else f) for f in frames])
    check_steps(levels)
    return levels


def check_steps(levels: np.ndarray) -> None:
    """Refuse a level series that jumps by about a photon between frames."""
    steps = np.abs(np.diff(levels))
    if steps.size and steps.max() > MAX_STEP_COUNTS:
        frame = int(np.argmax(steps)) + 1
        raise ValueError(
            f"The photon-free level jumps by {steps.max():.0f} counts at frame {frame}; "
            "one of the two estimates is probably a photon peak"
        )


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
    """*traces* (n_frames,) or (n_rows, n_frames) minus the per-frame *level*.

    Rows that are 0 in every frame (Suite2p's F of a ROI left with no pixels) stay 0, so
    they remain recognizable as empty.
    """
    traces, level = np.asarray(traces, np.float64), np.asarray(level, np.float64)
    if traces.ndim not in (1, 2):
        raise ValueError(f"Traces must be (n_frames,) or (n_rows, n_frames), got {traces.shape}")
    if level.ndim != 1 or traces.shape[-1] != level.size:
        raise ValueError(f"Level has {level.size} frames, traces {traces.shape[-1]}")
    if not np.isfinite(level).all():
        raise ValueError("The photon-free level has values that are not finite")
    empty = ~traces.any(axis=-1, keepdims=True)
    return np.where(empty, 0.0, traces - level)
