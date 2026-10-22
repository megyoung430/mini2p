"""NATEX's calcium-trace processing, ported from the MINI2P toolbox (Zong et al., 2022).

NATEX (``Analysis/Applications/NATEX/NATEXplorer.mlapp`` and ``Analysis/+preprocessing`` in
https://github.com/kavli-ntnu/MINI2P_toolbox) turns Suite2p traces into, per cell:

- **ΔF/F** with a baseline F0 that is the 8th percentile of the neuropil-corrected trace in a
  running window (30 s), lifted by the mean distance between the trace and that percentile in
  the quietest stretches of the recording (``DetaF_T_baselinecorrect``, after Low et al.,
  PNAS 2014);
- **significant transients**: frames where the low-passed ΔF/F stays above twice its local
  standard deviation for at least ``transient_s`` (0.75 s), widened by ``extend_s`` (0.4 s)
  on either side;
- **SNR**: the median of the top 10 % of ΔF/F within transients, minus the median ΔF/F outside
  them, over the standard deviation of the whole ΔF/F (``CalculateSNR``);
- **cleaned events**: the deconvolved trace (``spks``) kept only within transients, minus its
  mean outside them, clipped at 0, outliers capped, and scaled to the cell's peak ΔF/F; the
  event count is the number of frames with an event left;
- **bad cells**: cells whose raw F is zero throughout.

Each function follows the MATLAB code line by line (indices shifted to 0-based); where it
cannot be followed exactly, its docstring says so. Not ported: NATEX's tracking
(DeepLabCut) analysis and its combination with the imaging, its detection of one cell seen
in several imaging planes (these recordings have one plane), and its GUI.

The pure functions work on one cell's trace; ``process_plane`` runs them over a Suite2p
plane, and ``write_natex`` saves the results next to it.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from pydantic import BaseModel, ConfigDict, Field
from scipy import signal

from .config import run_dir_of
from .plane import PlaneTraces, load_plane_traces

log = logging.getLogger(__name__)

OUTPUT_FOLDER = "natex"
BASELINE_PERCENTILE = 8.0  # NATEX's F0 percentile (DetaF_T_baselinecorrect)
QUIET_FRACTION = 0.1  # the quietest stretches: local SD in the lowest 10 % of its range
SIGNIFICANCE_SD = 2.0  # a transient frame exceeds 2 x the local SD of the low-passed ΔF/F
PEAK_PERCENTILE = 90.0  # SNR signal: frames above this percentile within transients
LOWPASS_STEEPNESS = 0.85  # MATLAB lowpass defaults
LOWPASS_ATTENUATION_DB = 60.0


class NatexSettings(BaseModel):
    """The ``[natex]`` section of the pipeline config. The timing defaults are NATEX's; unlike
    NATEX (Suite2p "cells" only, neuropil coefficient 0.7), every ROI is kept and the
    neuropil coefficient is ``[fluorescence]``'s unless set here."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    moving_window_s: float = Field(30.0, gt=0)  # running window for F0 and the local SD
    lowpass_hz: float = Field(2.0, gt=0)  # low-pass for detecting transients only
    transient_s: float = Field(0.75, gt=0)  # a significant transient lasts at least this
    extend_s: float = Field(0.4, ge=0)  # margin added on both sides of a transient
    # None: [fluorescence] neuropil_coefficient, so NATEX and the QC audit correct alike
    neuropil_coefficient: float | None = Field(None, ge=0)
    only_iscell: bool = False  # NATEX keeps Suite2p "cells" only; Cellpose ROIs are all kept


def load_natex_settings(path: Path, **overrides: object) -> NatexSettings:
    """Read ``[natex]`` from a pipeline config (NATEX's defaults if there is none)."""
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle).get("natex", {})
    return NatexSettings.model_validate({**raw, **overrides})


# ---------------------------------------------------------------- MATLAB building blocks


def matlab_percentile(values: np.ndarray, percentile: float, axis: int = -1) -> np.ndarray:
    """MATLAB's ``prctile``: the sorted values sit at percentiles 100 (i - 0.5) / n, with
    linear interpolation between them (numpy's ``hazen`` method)."""
    return np.percentile(values, percentile, axis=axis, method="hazen")


def frames_in(seconds: float, fs: float) -> int:
    """``ceil(seconds * fs)``, as NATEX converts times to frames."""
    return math.ceil(seconds * fs - 1e-9)


def percentile_baseline(trace: np.ndarray, window_frames: int, percentile: float) -> np.ndarray:
    """``smooth_percentile``: the *percentile* of a running window around each frame.

    The window holds ``2 * ceil(window_frames / 2) + 1`` frames (2H + 1). It is centred on
    the frame, except within H frames of either end: there, as in the MATLAB code, a frame's
    window starts at that frame (the first H frames) or ends at it (the last H frames).

    Raises:
        ValueError: If the trace is shorter than one window.
    """
    trace = np.asarray(trace, dtype=float)
    half = math.ceil(window_frames / 2)
    length = len(trace)
    if length < 3 * half:  # MATLAB's first and last windows reach 3H frames from the end
        raise ValueError(
            f"A {length}-frame trace is shorter than {3 * half} frames, which a "
            f"{window_frames}-frame window needs"
        )
    windows = matlab_percentile(sliding_window_view(trace, 2 * half + 1), percentile)
    index = np.arange(length)
    start = np.where(index < half, index, index - half)  # first H frames: window starts there
    start = np.where(index >= length - half, index - 2 * half, start)  # last H: ends there
    return windows[start]


def moving_std(trace: np.ndarray, window: float) -> np.ndarray:
    """MATLAB's ``movstd(x, k)``: SD (n - 1 normalised) of the frames j with
    ``t - k/2 <= j < t + k/2`` around each frame t, shrinking at the ends.

    That half-open window is MATLAB's documented rule for whole k (odd: centred; even: one
    more frame before than after) and, as far as is known, also how it treats a non-integer
    k, which NATEX passes (0.75 s and 30 s x 7.52 Hz = 5.64 and 225.6 frames: 5 and 225
    frames, centred).
    """
    trace = np.asarray(trace, dtype=float)
    if window <= 0:
        raise ValueError(f"The window must be positive, not {window}")
    before, after = math.floor(window / 2), math.ceil(window / 2) - 1
    index = np.arange(len(trace))
    start = np.maximum(index - before, 0)
    stop = np.minimum(index + after + 1, len(trace))
    centred = trace - trace.mean()
    sums = np.concatenate([[0.0], np.cumsum(centred)])
    squares = np.concatenate([[0.0], np.cumsum(centred**2)])
    count = stop - start
    total = sums[stop] - sums[start]
    variance = (squares[stop] - squares[start] - total**2 / count) / np.maximum(count - 1, 1)
    variance[count == 1] = 0.0  # one frame: MATLAB gives 0
    return np.sqrt(np.clip(variance, 0, None))


def lowpass(trace: np.ndarray, cutoff_hz: float, fs: float) -> np.ndarray:
    """MATLAB's ``lowpass(x, wpass, fs)``, approximately: a linear-phase FIR filter with its
    defaults (steepness 0.85, 60 dB stopband), with its delay removed.

    MATLAB designs a minimum-order equiripple filter; this one is Kaiser-windowed to the same
    passband edge, transition width and attenuation. The trace is reflected at both ends,
    where MATLAB pads with zeros (so its filtered trace falls towards 0 within half a filter,
    about 7 s at 7.52 Hz, of either end).
    """
    nyquist = fs / 2
    if not 0 < cutoff_hz < nyquist:
        raise ValueError(f"Low-pass cut-off {cutoff_hz} Hz must be below Nyquist ({nyquist} Hz)")
    width_hz = (1 - LOWPASS_STEEPNESS) * (nyquist - cutoff_hz)
    taps, beta = signal.kaiserord(LOWPASS_ATTENUATION_DB, width_hz / nyquist)
    taps |= 1  # odd: a whole-frame delay
    kernel = signal.firwin(taps, cutoff_hz + width_hz / 2, window=("kaiser", beta), fs=fs)
    half = taps // 2
    trace = np.asarray(trace, dtype=float)
    if len(trace) <= half:
        raise ValueError(f"A {len(trace)}-frame trace is shorter than the filter ({taps} taps)")
    filtered = np.convolve(np.pad(trace, half, mode="reflect"), kernel, mode="same")
    return filtered[half : half + len(trace)]


def interval_mask(length: int, starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    """True on frames ``starts[i] .. stops[i]`` (both included) for every i."""
    edges = np.zeros(length + 1, int)
    np.add.at(edges, np.clip(starts, 0, length), 1)
    np.add.at(edges, np.clip(np.asarray(stops) + 1, 0, length), -1)
    return np.cumsum(edges[:-1]) > 0


def significant_frames(above: np.ndarray, lasting: int, extend: int) -> np.ndarray:
    """NATEX's two passes over the frames where ΔF/F exceeds its threshold.

    Forward, for every start frame j up to ``length - lasting - extend``: if at least
    *lasting* of the ``lasting + 1`` frames ``j .. j + lasting`` are above, frames
    ``j .. j + lasting + extend`` are significant. Backward, for every end frame j from
    ``lasting + extend``: if at least *lasting* of ``j - lasting .. j`` are above, frames
    ``j - lasting - extend .. j`` are significant.
    """
    above = np.asarray(above, bool)
    length = len(above)
    if length <= lasting:
        return np.zeros(length, bool)
    counts = sliding_window_view(above.astype(int), lasting + 1).sum(axis=1)  # frames k..k+lasting
    k = np.arange(len(counts))
    hit = counts >= lasting
    forward = k[hit & (k <= length - lasting - extend - 1)]
    backward = k[hit & (k >= extend)]  # window k..k+lasting ends at frame j = k + lasting
    starts = np.concatenate([forward, backward - extend])
    stops = np.concatenate([forward + lasting + extend, backward + lasting])
    return interval_mask(length, starts, stops)


# ---------------------------------------------------------------- one cell


@dataclass(frozen=True)
class CellResult:
    f0: np.ndarray  # baseline (corrected F units)
    dff: np.ndarray  # ΔF/F0 of the corrected trace
    dff_lowpassed: np.ndarray  # ΔF/F0 of the low-passed trace, used to find transients
    significant: np.ndarray  # bool per frame


def natex_dff(fcorr: np.ndarray, fs: float, settings: NatexSettings) -> CellResult:
    """``DetaF_T_baselinecorrect`` for one neuropil-corrected trace.

    F0 is the running 8th percentile plus F_M, the mean of trace - percentile over the frames
    whose local SD is in the lowest 10 % of its range. Unlike the MATLAB code, ΔF/F is NaN
    (with a warning) wherever F0 <= 0, where the ratio has no meaning.
    """
    fcorr = np.asarray(fcorr, dtype=float)
    window = settings.moving_window_s * fs
    f0_uncorrected = percentile_baseline(fcorr, math.ceil(window), BASELINE_PERCENTILE)
    local_sd = moving_std(fcorr, window)
    threshold = local_sd.min() + QUIET_FRACTION * (local_sd.max() - local_sd.min())
    quiet = local_sd < threshold
    if not quiet.any():  # a constant local SD: MATLAB's mean of nothing would be NaN
        log.warning("No quiet stretch (constant local SD): F0 is lifted by the whole trace")
        quiet = np.ones_like(quiet)
    f0 = f0_uncorrected + np.mean(fcorr[quiet] - f0_uncorrected[quiet])
    valid = f0 > 0  # ΔF/F is NaN where F0 <= 0 (process_traces reports which ROIs)
    with np.errstate(divide="ignore", invalid="ignore"):
        dff = np.where(valid, (fcorr - f0) / f0, np.nan)
        dff_lowpassed = np.where(valid, (lowpass(fcorr, settings.lowpass_hz, fs) - f0) / f0, np.nan)
    local_dff_sd = moving_std(np.nan_to_num(dff_lowpassed), settings.transient_s * fs)
    above = np.nan_to_num(dff_lowpassed) > SIGNIFICANCE_SD * local_dff_sd
    significant = significant_frames(
        above, frames_in(settings.transient_s, fs), frames_in(settings.extend_s, fs)
    )
    return CellResult(f0, dff, dff_lowpassed, significant)


def snr(dff: np.ndarray, significant: np.ndarray) -> float:
    """``CalculateSNR``: (median of the top 10 % of ΔF/F within transients - median ΔF/F
    outside them) / SD of the whole ΔF/F; 0 when undefined (no transients, as in MATLAB).

    Frames where ΔF/F is NaN (F0 <= 0) are left out.
    """
    dff = np.asarray(dff, dtype=float)
    significant = np.asarray(significant, bool)
    defined = np.isfinite(dff)
    dff, significant = dff[defined], significant[defined]
    inside, outside = dff[significant], dff[~significant]
    if not len(inside) or not len(outside):
        return 0.0
    peaks = inside[inside >= matlab_percentile(inside, PEAK_PERCENTILE)]
    noise = np.std(dff, ddof=1)
    value = (np.median(peaks) - np.median(outside)) / noise if noise > 0 else np.nan
    return 0.0 if np.isnan(value) else float(value)


def clean_events(spks: np.ndarray, significant: np.ndarray, dff: np.ndarray) -> np.ndarray:
    """NATEX's clean-up of one cell's deconvolved trace.

    Events outside significant transients are dropped; the rest lose the mean of the trace
    outside transients and are clipped at 0. Events above twice the 99.9th percentile are
    set to the 99.95th, and all are scaled so the largest equals the largest ΔF/F within
    transients. A cell with no event left gets zeros (MATLAB would give NaN).
    """
    spks = np.asarray(spks, dtype=float)
    significant = np.asarray(significant, bool)
    baseline = spks[~significant].mean() if (~significant).any() else 0.0
    events = np.clip((spks - baseline) * significant, 0, None)
    outliers = events > 2 * matlab_percentile(events, 99.9)
    events[outliers] = matlab_percentile(events, 99.95)
    peak_event = events.max()
    # max(detaF_F .* Significant): the zeros outside transients count, so never below 0
    peak_dff = max(0.0, float(np.nanmax(np.where(significant, dff, 0.0))))
    if peak_event <= 0 or peak_dff <= 0:
        return np.zeros_like(events)
    return events * peak_dff / peak_event


# ---------------------------------------------------------------- one plane


@dataclass(frozen=True)
class PlaneResult:
    rois: np.ndarray  # Suite2p ROI index of each row
    f0: np.ndarray  # (n_rois, n_frames)
    dff: np.ndarray
    dff_lowpassed: np.ndarray
    significant: np.ndarray  # bool
    events: np.ndarray
    snr: np.ndarray  # (n_rois,)
    undefined_fraction: np.ndarray  # fraction of frames with F0 <= 0 (ΔF/F NaN)
    event_count: np.ndarray
    transient_fraction: np.ndarray  # fraction of frames within significant transients
    bad: np.ndarray  # raw F is zero throughout
    fs: float
    neuropil_coefficient: float


def process_traces(
    f: np.ndarray,
    fneu: np.ndarray,
    spks: np.ndarray,
    fs: float,
    neuropil_coefficient: float,
    settings: NatexSettings,
    rois: np.ndarray | None = None,
) -> PlaneResult:
    """NATEX over every ROI: neuropil correction, ΔF/F, transients, SNR and cleaned events."""
    f, fneu, spks = (np.asarray(a, dtype=float) for a in (f, fneu, spks))
    if not f.shape == fneu.shape == spks.shape:
        raise ValueError(f"F {f.shape}, Fneu {fneu.shape} and spks {spks.shape} differ")
    fcorr = f - neuropil_coefficient * fneu
    cells = [natex_dff(trace, fs, settings) for trace in fcorr]
    stacked = {
        name: np.stack([getattr(c, name) for c in cells]) for name in CellResult.__annotations__
    }
    events = np.stack(
        [clean_events(s, c.significant, c.dff) for s, c in zip(spks, cells, strict=True)]
    )
    rois = np.arange(len(f)) if rois is None else np.asarray(rois)
    undefined = np.isnan(stacked["dff"]).mean(axis=1)
    if undefined.any():
        affected = rois[undefined > 0]
        log.warning(
            "F0 <= 0 in some frames of %d ROIs (ΔF/F NaN there; SNR uses the rest): %s",
            len(affected),
            affected[:30].tolist(),
        )
    return PlaneResult(
        rois=rois,
        undefined_fraction=undefined,
        events=events,
        snr=np.array([snr(c.dff, c.significant) for c in cells]),
        event_count=(events > 0).sum(axis=1),
        transient_fraction=stacked["significant"].mean(axis=1),
        bad=(f.sum(axis=1) == 0),
        fs=fs,
        neuropil_coefficient=neuropil_coefficient,
        **stacked,
    )


def process_plane(
    plane_dir: Path, settings: NatexSettings, neuropil_coefficient: float
) -> tuple[PlaneResult, PlaneTraces]:
    """NATEX for one Suite2p plane folder (``F``, ``Fneu``, ``spks``, ``iscell``, ``ops``);
    also returns the plane's traces, for its mean image and ROI outlines."""
    plane_dir = Path(plane_dir)
    traces = load_plane_traces(plane_dir)
    spks = np.load(plane_dir / "spks.npy")
    rois = np.arange(len(traces.f))
    if settings.only_iscell:
        rois = rois[np.asarray(traces.iscell)[:, 0] > 0]
    log.info("NATEX on %d ROIs of %s at %.3g Hz", len(rois), plane_dir, traces.fs)
    result = process_traces(
        traces.f[rois],
        traces.fneu[rois],
        spks[rois],
        traces.fs,
        neuropil_coefficient,
        settings,
        rois,
    )
    return result, traces


ARRAYS = ("f0", "dff", "dff_lowpassed", "significant", "events")
TABLE = "natex_cells.csv"
RECORD = "natex_settings.json"
REPORT = "natex_qc.pdf"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_natex(
    result: PlaneResult,
    plane_dir: Path,
    settings: NatexSettings,
    output_dir: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Save *result*: one ``.npy`` per array (ROIs x frames), a per-ROI table and a record of
    the settings and of the traces it came from. Returns the output folder."""
    plane_dir = Path(plane_dir)
    out = Path(output_dir) if output_dir else plane_dir / OUTPUT_FOLDER
    if out.is_dir() and any(out.glob("natex_*")) and not overwrite:
        raise FileExistsError(f"{out} already holds NATEX results; pass overwrite to replace them")
    out.mkdir(parents=True, exist_ok=True)
    for name in ARRAYS:
        array = getattr(result, name)
        np.save(out / f"natex_{name}.npy", array if array.dtype == bool else array.astype("f4"))
    np.save(out / "natex_rois.npy", result.rois)
    with (out / TABLE).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["roi", "snr", "event_count", "transient_fraction", "undefined_fraction", "bad"]
        )
        for row in zip(
            result.rois,
            result.snr,
            result.event_count,
            result.transient_fraction,
            result.undefined_fraction,
            result.bad,
            strict=True,
        ):
            roi, value, count, fraction, undefined, bad = row
            writer.writerow(
                [
                    int(roi),
                    f"{value:.4f}",
                    int(count),
                    f"{fraction:.4f}",
                    f"{undefined:.4f}",
                    bool(bad),
                ]
            )
    record = {
        "settings": settings.model_dump(),
        "neuropil_coefficient": result.neuropil_coefficient,
        "fs": result.fs,
        "plane_dir": str(plane_dir),
        "inputs_sha256": {
            name: _sha256(plane_dir / name)
            for name in ("F.npy", "Fneu.npy", "spks.npy", "iscell.npy", "ops.npy")
        },
        "source": "Python port of NATEX (kavli-ntnu/MINI2P_toolbox)",
    }
    (out / RECORD).write_text(json.dumps(record, indent=2))
    return out


def example_rows(result: PlaneResult, count: int = 5) -> list[int]:
    """Rows spanning the SNR range of ROIs with signal: highest to lowest, evenly in rank."""
    good = np.flatnonzero(~result.bad)
    if not len(good):
        return []
    ranked = good[np.argsort(-result.snr[good], kind="stable")]
    picks = np.unique(np.linspace(0, len(ranked) - 1, min(count, len(ranked))).round()).astype(int)
    return [int(ranked[i]) for i in picks]


def write_report(result: PlaneResult, traces: PlaneTraces, path: Path, title: str) -> Path:
    """A PDF: SNR and event-count histograms, ROIs coloured by SNR, and example cells."""
    # Figures made without pyplot, so the caller's backend (e.g. a notebook's) is untouched.
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure

    from . import plots

    with PdfPages(path) as pdf:
        fig = Figure(figsize=(15, 4.5), layout="constrained")
        axes = fig.subplots(1, 3)
        plots.plot_natex_overview(result, np.asarray(traces.ops["meanImg"]), traces.labels, axes)
        fig.suptitle(f"{title}: NATEX, neuropil coefficient {result.neuropil_coefficient:g}")
        pdf.savefig(fig)
        rows = example_rows(result)
        if rows:
            fig = Figure(figsize=(14, 1.8 * len(rows)), layout="constrained")
            axes = fig.subplots(len(rows), 1, sharex=True, squeeze=False)
            plots.plot_natex_cells(result, rows, axes)
            fig.suptitle(f"{title}: ROIs from highest to lowest SNR (transients shaded)")
            pdf.savefig(fig)
    return path


def run_natex(
    plane_dir: Path,
    settings: NatexSettings,
    neuropil_coefficient: float,
    output_dir: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """NATEX on one plane: results and ``natex_qc.pdf`` in ``<plane>/natex`` (or
    *output_dir*). Returns the output folder."""
    if settings.neuropil_coefficient is not None:
        neuropil_coefficient = settings.neuropil_coefficient
    result, traces = process_plane(plane_dir, settings, neuropil_coefficient)
    out = write_natex(result, plane_dir, settings, output_dir, overwrite)
    variant_dir = Path(plane_dir).resolve().parents[1]
    run_dir = run_dir_of(variant_dir)
    title = " ".join(run_dir.parts[-3:] + variant_dir.relative_to(run_dir).parts)
    write_report(result, traces, out / REPORT, title)
    log.info("NATEX results in %s", out)
    return out
