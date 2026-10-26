"""Ways to correct a ROI's raw F before ΔF/F0, and numbers to compare them by (computation
only; the extraction comparison notebook loads the run and draws).

Suite2p's correction ``F - 0.7 * Fneu`` assumes the ring of neuropil around a ROI carries
contaminating calcium signal about as bright as the halo inside it. With an indicator kept in
the soma (ribosome-tethered GCaMP), the neuropil holds little indicator: its signal is mostly
out-of-focus somata, autofluorescence and any detector offset, so subtracting 0.7 of it can
take most of a dim cell's own baseline and leave F0 <= 0. The corrections compared:

- ``Neuropil <c>``: the fixed coefficient (``[fluorescence] neuropil_coefficient``);
- ``Neuropil Fitted``: each ROI's own coefficient, the least-squares slope of F on its
  neuropil outside its transients (``fitted_neuropil_coefficients``), kept in [0, 1];
- ``No Neuropil``: raw F;
- ``No Neuropil, <source> Bg Regressed``: raw F with a background trace (vessel, or every
  pixel outside the ROIs) regressed out, each ROI with its own coefficient, mean kept;
- ``Neuropil <c>, <source> Bg Regressed``: the background removed from F and Fneu before the
  fixed neuropil correction, as in the QC notebook (``background.dff_versions``).
"""

from __future__ import annotations

import numpy as np

from .background import regress_out
from .fluorescence import DffResult, neuropil_correct

MAD_TO_SD = 1.4826  # scales a median absolute deviation to a normal standard deviation
FIT_ITERATIONS = 3
PAIRWISE_MAX_ROIS = 300  # mean pairwise correlation on at most this many ROIs (evenly spaced)


# ---------------------------------------------------------------- neuropil coefficient


def _robust_sd(values: np.ndarray) -> float:
    return MAD_TO_SD * float(np.median(np.abs(values - np.median(values))))


def _fit_one(f: np.ndarray, fneu: np.ndarray, transient_sd: float) -> float:
    """Slope of F on Fneu, refitted without the frames far above the fit (transients)."""
    keep = np.ones(len(f), bool)
    for _ in range(FIT_ITERATIONS):
        slope, intercept = np.polyfit(fneu[keep], f[keep], 1)
        residual = f - (slope * fneu + intercept)
        kept = residual[keep]
        keep = residual <= np.median(kept) + transient_sd * _robust_sd(kept)
    return float(slope)


def fitted_neuropil_coefficients(
    f: np.ndarray,
    fneu: np.ndarray,
    transient_sd: float = 2.0,
    bounds: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """Each ROI's neuropil coefficient: the least-squares slope of its F on its Fneu, refitted
    ``FIT_ITERATIONS`` times without frames more than *transient_sd* robust SDs above the fit
    (the cell's own transients, which would otherwise pull the slope), then kept in *bounds*.

    A heuristic: the slope measures how much of the neuropil's fluctuation appears in the ROI,
    which is what a neuropil correction should remove. A slow signal shared by the whole field
    raises it, as it should.

    Raises:
        ValueError: If *transient_sd* is not positive, or a ROI's neuropil is flat.
    """
    f, fneu = np.asarray(f, np.float64), np.asarray(fneu, np.float64)
    if transient_sd <= 0:
        raise ValueError(f"transient_sd must be positive, got {transient_sd}")
    if f.shape != fneu.shape or f.ndim != 2:
        raise ValueError(f"F {f.shape} and Fneu {fneu.shape} must be the same (ROIs, frames)")
    flat = np.flatnonzero(np.ptp(fneu, axis=1) == 0)
    if flat.size:
        raise ValueError(f"Neuropil is flat for ROI(s) {flat[:10].tolist()}; no slope to fit")
    slopes = np.array([_fit_one(a, b, transient_sd) for a, b in zip(f, fneu, strict=True)])
    return np.clip(slopes, *bounds)


def contrast_ratio(f: np.ndarray, fneu: np.ndarray) -> np.ndarray:
    """Median F over median Fneu per ROI: how far each ROI stands above its surroundings.
    NaN where the neuropil's median is not positive (no ratio to give; a detector offset at
    or below zero shows up this way)."""
    f_med = np.median(np.asarray(f, np.float64), axis=1)
    n_med = np.median(np.asarray(fneu, np.float64), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(n_med > 0, f_med / n_med, np.nan)


# ---------------------------------------------------------------- corrections


def corrected_traces(
    f: np.ndarray,
    fneu: np.ndarray,
    backgrounds: dict[str, np.ndarray],
    neuropil_coefficient: float,
    fitted: np.ndarray,
) -> dict[str, np.ndarray]:
    """The corrected F of every method (see the module docstring), by method name.

    Args:
        backgrounds: background trace (n_frames,) of each source, by label (e.g. "Vessel").
        fitted: each ROI's coefficient, from ``fitted_neuropil_coefficients``.
    """
    f = np.asarray(f, np.float32)
    fitted = np.asarray(fitted, np.float32)
    if fitted.shape != (len(f),):
        raise ValueError(f"{len(fitted)} fitted coefficients for {len(f)} ROIs")
    fixed = f"Neuropil {neuropil_coefficient:g}"
    traces = {
        fixed: neuropil_correct(f, fneu, neuropil_coefficient),
        "Neuropil Fitted": f - fitted[:, None] * np.asarray(fneu, np.float32),
        "No Neuropil": f,
    }
    for label, background in backgrounds.items():
        traces[f"No Neuropil, {label} Bg Regressed"] = regress_out(f, background).cleaned
    for label, background in backgrounds.items():
        f_bg = regress_out(f, background).cleaned
        fneu_bg = regress_out(fneu, background).cleaned
        traces[f"{fixed}, {label} Bg Regressed"] = neuropil_correct(
            f_bg, fneu_bg, neuropil_coefficient
        )
    return traces


# ---------------------------------------------------------------- comparing them


def _noise(dff: np.ndarray) -> np.ndarray:
    """Frame-to-frame SD of each ROI's ΔF/F0 (robust SD of the differences / sqrt 2)."""
    out = np.full(len(dff), np.nan)
    for i, row in enumerate(dff):
        steps = np.diff(row)
        steps = steps[np.isfinite(steps)]
        if steps.size > 1:
            out[i] = _robust_sd(steps) / np.sqrt(2)
    return out


def _skew(dff: np.ndarray) -> np.ndarray:
    out = np.full(len(dff), np.nan)
    for i, row in enumerate(dff):
        row = row[np.isfinite(row)]
        if row.size > 2 and row.std() > 0:
            out[i] = np.mean(((row - row.mean()) / row.std()) ** 3)
    return out


def _mean_pairwise_r(traces: np.ndarray) -> float:
    rows = np.unique(np.linspace(0, len(traces) - 1, PAIRWISE_MAX_ROIS).round().astype(int))
    r = np.corrcoef(traces[rows])
    return float(np.nanmean(r[~np.eye(len(rows), dtype=bool)]))


def method_metrics(result: DffResult, background: np.ndarray) -> dict[str, float]:
    """Numbers to compare one correction by.

    - ``rois_f0_le_0_anywhere`` / ``rois_f0_le_0_median``: ROIs whose F0 is <= 0 in any frame /
      in at least half of the frames;
    - ``fraction_frames_f0_le_0``: of all ROI-frames, those without a ΔF/F0;
    - ``median_f0``: median over ROIs of each ROI's median F0;
    - ``median_noise``: median frame-to-frame SD of ΔF/F0;
    - ``median_skew``: median skewness of ΔF/F0 (calcium transients make it positive);
    - ``median_abs_r_background``: median |r| of the corrected F with the background trace
      (what of the background is left);
    - ``mean_pairwise_r``: mean correlation between the ROIs' corrected F (a field-wide
      signal raises it), over up to ``PAIRWISE_MAX_ROIS`` ROIs.
    """
    background = np.asarray(background, np.float64)
    f_corr = np.asarray(result.f_corr, np.float64)
    r_background = [np.corrcoef(row, background)[0, 1] for row in f_corr]
    invalid = np.asarray(result.invalid_f0)
    return {
        "rois_f0_le_0_anywhere": int(invalid.any(axis=1).sum()),
        "rois_f0_le_0_median": int((invalid.mean(axis=1) >= 0.5).sum()),
        "fraction_frames_f0_le_0": float(invalid.mean()),
        "median_f0": float(np.median(np.median(result.f0, axis=1))),
        "median_noise": float(np.nanmedian(_noise(result.dff))),
        "median_skew": float(np.nanmedian(_skew(result.dff))),
        "median_abs_r_background": float(np.nanmedian(np.abs(r_background))),
        "mean_pairwise_r": _mean_pairwise_r(f_corr),
    }


def background_channel_check(green: np.ndarray, red: np.ndarray) -> dict[str, float]:
    """How the red background follows the green one, as fractions of each channel's mean.

    Laser power or shared detection drift moves both channels by the same fraction (slope
    near 1); blood volume dims green more than red (hemoglobin absorbs green more): slope
    between 0 and 1; activity in green indicator alone: slope near 0. A noisy red channel
    pulls the slope towards 0 too, so read it with ``r``.

    Returns:
        ``red_vs_green_slope`` (least squares of red's fractional change on green's), ``r``,
        and each channel's SD as a fraction of its mean.

    Raises:
        ValueError: If a channel's mean is not positive (no fraction to take).
    """
    fractions = {}
    for name, trace in (("green", green), ("red", red)):
        trace = np.asarray(trace, np.float64)
        if trace.mean() <= 0:
            raise ValueError(f"The {name} background's mean is {trace.mean():g}; must be > 0")
        fractions[name] = trace / trace.mean() - 1
    g, r = fractions["green"], fractions["red"]
    return {
        "red_vs_green_slope": float(g @ r / (g @ g)),
        "r": float(np.corrcoef(g, r)[0, 1]),
        "green_sd_fraction": float(g.std()),
        "red_sd_fraction": float(r.std()),
    }
