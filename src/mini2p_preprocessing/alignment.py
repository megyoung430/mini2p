"""Moving Cellpose masks onto a plane whose registration differs by a uniform shift.

Two Suite2p registrations of the same recording (for example the first 1000 frames
and the full session) each align to their own reference image, so their mean
images, and anything drawn on them, can be offset. The offset is measured by
registering the image the masks were drawn on to the plane's own mean image.

Cross-correlation (not phase correlation) of background-subtracted images is used:
the mask image is usually a contrast-adjusted PNG, and gamma, clipping and noise
corrupt exactly the high-frequency phase that phase correlation relies on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
from matplotlib.image import imread
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

QUADRANT_MIN_TEXTURE = 0.3  # quadrants with less structure than this x the richest are skipped
QUADRANT_MIN_FIT = 0.2  # Pearson a structured quadrant must reach at its best nearby shift
QUADRANT_SEARCH_EXTRA_PX = 3  # quadrants are searched within tolerance + this of the overall shift
MIN_STRUCTURED_QUADRANTS = 2


def load_reference_image(path: Path) -> np.ndarray:
    """A 2-D float image from a grey PNG (as saved by the contrast export) or a TIFF."""
    path = Path(path)
    image = tifffile.imread(path) if path.suffix.lower() in {".tif", ".tiff"} else imread(path)
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3:
        rgb = image[..., :3]
        if np.abs(rgb - rgb.mean(axis=2, keepdims=True)).max() > 1e-3:
            raise ValueError(
                f"{path} is a colour image, not grey; use the green contrast PNG the masks "
                "were drawn on, not an overlay"
            )
        image = rgb.mean(axis=2)
    if image.ndim != 2:
        raise ValueError(f"{path} is not a 2-D image (shape {image.shape})")
    return image


def shift_image(image: np.ndarray, dy: int, dx: int, fill: float | int) -> np.ndarray:
    """``image`` moved down by ``dy`` and right by ``dx``; uncovered pixels get ``fill``."""
    out = np.full_like(image, fill)
    h, w = image.shape
    out[max(dy, 0) : h + min(dy, 0), max(dx, 0) : w + min(dx, 0)] = image[
        max(-dy, 0) : h + min(-dy, 0), max(-dx, 0) : w + min(-dx, 0)
    ]
    return out


def _prepare(image: np.ndarray, background_sigma_px: float) -> np.ndarray:
    """Remove slow background (vignetting, neuropil glow), unit variance, tapered edges."""
    image = np.asarray(image, dtype=np.float64)
    detail = image - gaussian_filter(image, background_sigma_px)
    std = detail.std()
    if not np.isfinite(std) or std == 0:
        raise ValueError("Image has no structure to register (constant or blank)")
    window = np.outer(np.hanning(image.shape[0]), np.hanning(image.shape[1]))
    return (detail - detail.mean()) / std * window


def _peak_shift(fixed: np.ndarray, moving: np.ndarray) -> tuple[int, int]:
    """Whole-pixel (dy, dx) maximising the circular cross-correlation of prepared images."""
    correlation = np.real(np.fft.ifft2(np.fft.fft2(fixed) * np.conj(np.fft.fft2(moving))))
    iy, ix = np.unravel_index(np.argmax(correlation), correlation.shape)
    dy_all = np.fft.fftfreq(fixed.shape[0], 1 / fixed.shape[0]).astype(int)
    dx_all = np.fft.fftfreq(fixed.shape[1], 1 / fixed.shape[1]).astype(int)
    return int(dy_all[iy]), int(dx_all[ix])


def best_shift(
    fixed: np.ndarray, moving: np.ndarray, background_sigma_px: float = 24.0
) -> tuple[int, int]:
    """Whole-pixel (dy, dx) that moves ``moving`` onto ``fixed``, with no size limit.

    Positive dy moves down and positive dx moves right.
    """
    if fixed.shape != moving.shape:
        raise ValueError(f"Image shapes differ: {fixed.shape} vs {moving.shape}")
    return _peak_shift(_prepare(fixed, background_sigma_px), _prepare(moving, background_sigma_px))


def check_shift_limit(dy: int, dx: int, max_shift_px: int) -> None:
    if max(abs(dy), abs(dx)) > max_shift_px:
        raise ValueError(
            f"Best shift ({dy}, {dx}) px exceeds fluorescence.mask_max_shift_px={max_shift_px}; "
            "raise it if that offset is plausible"
        )


def estimate_shift(
    fixed: np.ndarray, moving: np.ndarray, max_shift_px: int, background_sigma_px: float = 24.0
) -> tuple[int, int]:
    """``best_shift``, refusing a shift beyond ``max_shift_px`` rather than shrinking it."""
    dy, dx = best_shift(fixed, moving, background_sigma_px)
    check_shift_limit(dy, dx, max_shift_px)
    return dy, dx


def shift_labels(labels: np.ndarray, dy: int, dx: int) -> tuple[np.ndarray, list[int]]:
    """Labels moved by (dy, dx), zero-filled; also the labels still in the image."""
    moved = shift_image(np.asarray(labels), dy, dx, 0)
    return moved, [int(v) for v in np.unique(moved) if v != 0]


def _quadrants(shape: tuple[int, int]) -> list[tuple[slice, slice]]:
    h, w = shape
    return [
        (slice(0, h // 2), slice(0, w // 2)),
        (slice(0, h // 2), slice(w // 2, w)),
        (slice(h // 2, h), slice(0, w // 2)),
        (slice(h // 2, h), slice(w // 2, w)),
    ]


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(b)
    a, b = a[valid], b[valid]
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _structure(detail: np.ndarray) -> float:
    """Spread of lightly smoothed detail, so pixel noise does not count as structure."""
    return float(gaussian_filter(detail, 1.5).std())


def quadrant_fits(
    fixed: np.ndarray,
    moving: np.ndarray,
    dy: int,
    dx: int,
    tolerance_px: int,
    background_sigma_px: float,
) -> list[dict | None]:
    """Each structured quadrant's best shift near the overall one, and how well it fits.

    Shifts are searched within ``tolerance_px + QUADRANT_SEARCH_EXTRA_PX`` of the
    overall shift and scored by unwindowed Pearson correlation, so a quadrant's shift
    is the one that fits it best. Quadrants without structure in both images are None.
    """
    fixed_detail = _detail(fixed, background_sigma_px)
    moving_detail = _detail(moving, background_sigma_px)
    windows = _quadrants(fixed.shape)
    fixed_structure = [_structure(fixed_detail[w]) for w in windows]
    moving_structure = [_structure(moving_detail[w]) for w in windows]
    radius = tolerance_px + QUADRANT_SEARCH_EXTRA_PX
    candidates = [
        (dy + sy, dx + sx) for sy in range(-radius, radius + 1) for sx in range(-radius, radius + 1)
    ]
    fits: list[dict | None] = []
    for q, window in enumerate(windows):
        if fixed_structure[q] < QUADRANT_MIN_TEXTURE * max(fixed_structure) or moving_structure[
            q
        ] < QUADRANT_MIN_TEXTURE * max(moving_structure):
            fits.append(None)
            continue
        scores = {
            c: _pearson(fixed_detail[window], shift_image(moving_detail, *c, np.nan)[window])
            for c in candidates
        }
        finite = {c: v for c, v in scores.items() if np.isfinite(v)}
        best = max(finite, key=finite.get) if finite else (dy, dx)
        fits.append(
            {
                "shift": list(best),
                "fit": finite.get(best, float("nan")),
                "fit_at_overall": scores[(dy, dx)],
            }
        )
    return fits


def _detail(image: np.ndarray, background_sigma_px: float) -> np.ndarray:
    image = np.asarray(image, np.float64)
    return image - gaussian_filter(image, background_sigma_px)


def rank_correlation(
    fixed: np.ndarray, moving: np.ndarray, dy: int, dx: int, background_sigma_px: float
) -> float:
    """Spearman correlation of background-subtracted images after the shift.

    Ranks make it insensitive to contrast and gamma; removing the background stops
    shared vignetting from making unrelated images look alike.
    """
    moved = shift_image(_detail(moving, background_sigma_px), dy, dx, np.nan)
    valid = np.isfinite(moved)
    return float(spearmanr(_detail(fixed, background_sigma_px)[valid], moved[valid]).statistic)


def _drop_partial(original: np.ndarray, moved: np.ndarray, min_fraction: float):
    """Remove masks keeping less than ``min_fraction`` of their pixels after the shift."""
    labels, before = np.unique(original[original != 0], return_counts=True)
    after = dict(zip(*np.unique(moved[moved != 0], return_counts=True), strict=True))
    retained = {int(lab): after.get(lab, 0) / n for lab, n in zip(labels, before, strict=True)}
    dropped = sorted(lab for lab, frac in retained.items() if frac < min_fraction)
    if dropped:
        moved = np.where(np.isin(moved, dropped), 0, moved)
    return moved, retained, dropped


def align_labels(
    labels: np.ndarray,
    mask_image: np.ndarray,
    plane_image: np.ndarray,
    max_shift_px: int,
    tolerance_px: int,
    min_correlation: float,
    min_retained_fraction: float,
    background_sigma_px: float = 24.0,
) -> tuple[np.ndarray, dict]:
    """Shift masks drawn on ``mask_image`` onto ``plane_image``, or refuse with a reason.

    Refuses when the images do not match after the shift (wrong or blank image),
    when fewer than two quadrants have structure to check, or when a structured
    quadrant needs a shift more than ``tolerance_px`` from the overall one (a
    non-rigid difference a single shift cannot fix). A smooth warp such as a slight
    rotation is averaged into one shift, so masks far from the centre can be off by
    more than the tolerance; two rigid registrations of one recording should not
    differ that way. Masks keeping less than ``min_retained_fraction`` of their
    pixels are dropped.
    """
    if labels.shape != mask_image.shape:
        raise ValueError(f"Mask shape {labels.shape} differs from its image {mask_image.shape}")
    plane_image = np.asarray(plane_image, np.float64)
    mask_image = np.asarray(mask_image, np.float64)
    dy, dx = best_shift(plane_image, mask_image, background_sigma_px)
    after = rank_correlation(plane_image, mask_image, dy, dx, background_sigma_px)
    fits = quadrant_fits(plane_image, mask_image, dy, dx, tolerance_px, background_sigma_px)
    structured = [f for f in fits if f is not None]
    if len(structured) < MIN_STRUCTURED_QUADRANTS:
        raise ValueError(
            f"Only {len(structured)} image quadrant(s) have enough structure to check that the "
            "shift is uniform; align these masks by re-segmenting this plane instead"
        )
    matched = [f for f in structured if np.isfinite(f["fit"]) and f["fit"] >= QUADRANT_MIN_FIT]
    quadrant_shifts = [f["shift"] if f else None for f in fits]
    if len(matched) * 2 <= len(structured):
        raise ValueError(
            f"Most of the image matches nothing near the best shift ({dy}, {dx}) px "
            f"(quadrant fits {[round(f['fit'], 2) for f in structured]}); is mask_image the "
            "mean image these masks were drawn on?"
        )
    off = [
        f for f in matched if max(abs(f["shift"][0] - dy), abs(f["shift"][1] - dx)) > tolerance_px
    ]
    poor_match = not np.isfinite(after) or after < min_correlation
    differing_parts = len({tuple(f["shift"]) for f in matched}) > 1
    if off or len(matched) < len(structured) or (poor_match and differing_parts):
        raise ValueError(
            f"Offset is not uniform across the image: overall ({dy}, {dx}) px but quadrants "
            f"need {quadrant_shifts}; a single shift cannot align these masks. Check that "
            "mask_image is the mean image they were drawn on; if it is, re-segment this plane"
        )
    if poor_match:
        raise ValueError(
            f"Images still differ after the best shift (rank correlation {after:.2f} < "
            f"{min_correlation}); is mask_image the mean image these masks were drawn on?"
        )
    check_shift_limit(dy, dx, max_shift_px)
    moved, _ = shift_labels(labels, dy, dx)
    moved, retained, dropped = _drop_partial(np.asarray(labels), moved, min_retained_fraction)
    report = {
        "dy": dy,
        "dx": dx,
        "quadrant_shifts": quadrant_shifts,
        "quadrant_fits": [f["fit"] if f else None for f in fits],
        "quadrant_fits_at_overall_shift": [f["fit_at_overall"] if f else None for f in fits],
        "quadrants_used": len(structured),
        "rank_correlation_before": rank_correlation(
            plane_image, mask_image, 0, 0, background_sigma_px
        ),
        "rank_correlation_after": after,
        "retained_fraction": {str(k): float(v) for k, v in retained.items()},
        "kept_labels": [int(v) for v in np.unique(moved) if v != 0],
        "dropped_labels": dropped,
    }
    return moved, report
