"""The QC notebook's plots for one run, saved without the notebook (batch runs).

:func:`save_run_qc` draws, for a run the pipeline has just processed, the plots the QC
notebook (``notebooks/qc_single_run.ipynb``) saves for its preprocessing and registration,
into the same ``quality_control/<step>/`` folders (``qc_figures``): the raw mean, the
bidiphase phases and before/after, destriping, distortion correction, the registration's
before/after, shift traces, nonrigid offsets, registered means and their crops, the focus QC
and (optionally, as it rereads every stage) the frame order. :func:`save_fluorescence_qc`
draws the notebook's fluorescence plots once the run's Cellpose masks are extracted: the
photon-free levels, F0 overview, mask alignment, the non-ROI background and the top ROIs.

What only the notebook can do is left out: the plots behind sliders, the hand-chosen
contrast, and the vessel background (each recording needs its own vessel masks). Each plot
is drawn on its own; one that fails is logged and listed with the error in the
registration's ``quality_control/batch_qc_summary.json`` (:func:`summary_path`), and the
others are still saved. QC is advisory, so a
failed plot never fails the run.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import plots, qc, raw_previews
from .config import (
    PipelineSettings,
    corrected_tiff_path,
    distortion_corrected,
    variant_dir_of,
)
from .contrast import scale_contrast
from .qc_figures import QC_FOLDER, qc_step_dir, save_qc_figure

log = logging.getLogger(__name__)
SUMMARY_FILE = "batch_qc_summary.json"
RAW_MEAN_FRAMES = 500
STAGE_FRAMES = 100  # the stretch shown before/after bidiphase and destriping
REGISTRATION_FRAMES = 500
DISTORTION_FRAMES = 100
N_TOP_ROIS = 20


@dataclass(frozen=True)
class RunQc:
    """One processed run, as the QC plots need it."""

    input_tiff: Path  # the acquired TIFF
    variant_dir: Path  # run-XX/<distortion>/<destriping>
    corrected_tiff: Path  # phase-corrected (and destriped)
    suite2p_input: Path  # what Suite2p registered (undistorted, or the corrected TIFF)
    plane_dir: Path  # the registration's plane0
    settings: PipelineSettings

    @property
    def channel_names(self) -> dict[int, str]:
        """Channel number to name (structural/functional), from the run's settings."""
        return self.settings.channels.names


def run_qc_for_plane(plane_dir: Path, extraction_settings: PipelineSettings) -> RunQc:
    """The :class:`RunQc` of an already processed plane, from the settings its variant
    folder recorded, with the ``[fluorescence]`` of *extraction_settings* (the ΔF/F0 the
    extraction is checked with).

    Raises:
        FileNotFoundError: If the variant has no ``pipeline_settings.json`` or no record of
            its acquired TIFF.
    """
    from .movies import acquired_tiff, load_run_settings
    from .pipeline import suite2p_input_for

    variant_dir = variant_dir_of(plane_dir)
    recorded = load_run_settings(variant_dir)
    if recorded is None:
        raise FileNotFoundError(f"{variant_dir} has no pipeline_settings.json")
    settings = recorded.model_copy(update={"fluorescence": extraction_settings.fluorescence})
    input_tiff = acquired_tiff(variant_dir, settings)
    return RunQc(
        input_tiff, variant_dir, corrected_tiff_path(variant_dir, input_tiff, settings),
        suite2p_input_for(variant_dir, input_tiff, settings), Path(plane_dir), settings,
    )  # fmt: skip


def _caption(run: RunQc, text: str, registration: bool) -> str:
    """*text* followed by the recording, output folder and (for a registration plot) the
    registration, as the QC notebook captions its plots."""
    caption = f"{text.strip()}\nRecording: {run.input_tiff}\nOutput folder: {run.variant_dir}\n"
    if registration:
        caption += f"Registration: {run.plane_dir.parent}\n"
    return caption + "Saved by the batch (mini2p_preprocessing.run_qc).\n"


def _save(run: RunQc, drawn, step: str, name: str, text: str, registration: bool) -> Path:
    """Saves *drawn* where the QC notebook would (:func:`save_qc_figure`), closes it, and
    returns the saved path without extension."""
    stem = save_qc_figure(
        drawn, run.variant_dir, step, name, _caption(run, text, registration),
        plane_dir=run.plane_dir if registration else None,
        save_folder=run.settings.registration.save_folder if registration else None,
    )  # fmt: skip
    plt.close("all")
    return stem


# ---- preprocessing --------------------------------------------------------------------


def raw_mean_plots(run: RunQc) -> list[Path]:
    """The mean of the first frames of the acquired TIFF, before any correction."""
    raw_mean, n_raw = qc.raw_mean_images(
        run.input_tiff, RAW_MEAN_FRAMES, run.settings.channels.n_channels
    )
    stem = qc_step_dir(run.variant_dir, "raw_mean") / f"raw_meanImg_first{n_raw}"
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": plots.QC_FONTS}):
        fig = raw_previews.draw_preview(raw_mean, n_raw, run.input_tiff.name, run.channel_names)
        caption = raw_previews.preview_caption(run.input_tiff, n_raw, RAW_MEAN_FRAMES)
        raw_previews.save_preview(fig, stem, caption, dpi=200)
    plt.close("all")
    return [stem]


def bidiphase_plots(run: RunQc, frame_phases) -> list[Path]:
    """The phase of each frame (per-frame method) or chunk (chunked), and the
    before/after of the bidiphase correction and destriping over the first frames."""
    settings, n_channels = run.settings, run.settings.channels.n_channels
    saved = []
    raw = qc.load_chunk(run.input_tiff, 0, STAGE_FRAMES, n_channels)
    n = len(raw[0])
    if frame_phases is not None:
        saved.append(_save(run, plots.plot_frame_phases(frame_phases), "bidiphase", "frame_phases",
                           "The bidiphase (odd/even-line) phase chosen for each frame, in px, and "
                           "which channel set it: the structural channel where its per-frame phase "
                           "was consistent and agreed with the functional channel, else the "
                           "functional channel averaged over "
                           f"{settings.phase.fallback_window_frames} frames.", False))  # fmt: skip
        from .frame_phase import apply_frame_phases

        after = [apply_frame_phases(c, frame_phases.phase_px[:n]) for c in raw]
        title = "After per-frame bidiphase"
    else:
        trace = qc.phase_trace(run.input_tiff, settings)
        saved.append(_save(run, plots.plot_phase_trace(trace, settings), "bidiphase",
                           "phase_trace", "The bidiphase phase and its search score for each chunk "
                           f"of {settings.phase.chunk_frames} timepoints (phase.method = "
                           '"chunked").', False))  # fmt: skip
        after = qc.load_chunk(run.corrected_tiff, 0, STAGE_FRAMES, n_channels)
        title = "After bidiphase" + (" and destriping" if settings.destripe.enabled else "")
    saved.append(_save(run, plots.plot_channel_pairs(
        qc.mean_images(raw), qc.mean_images(after), ("Before", title), run.channel_names),
        "bidiphase", "before_after_bidiphase",
        f"Mean of timepoints 0:{n} of each channel before and after the bidiphase correction.",
        False))  # fmt: skip
    if settings.destripe.enabled and frame_phases is not None:
        from .destripe import detrend_chunk

        destriped = [detrend_chunk(c, settings.destripe.preserve_frame_mean) for c in after]
        saved.append(_save(run, plots.plot_channel_pairs(
            qc.mean_images(after), qc.mean_images(destriped),
            ("Before destriping", "After destriping"), run.channel_names),
            "destriping", "before_after_destriping",
            f"Mean of timepoints 0:{n} of each channel, bidiphase-corrected, before and after "
            "destriping.", False))  # fmt: skip
    return saved


def distortion_plots(run: RunQc) -> list[Path]:
    """The first frames before and after distortion correction."""
    if not distortion_corrected(run.settings):
        return []
    n_channels = run.settings.channels.n_channels
    before = qc.mean_images(qc.load_chunk(run.corrected_tiff, 0, DISTORTION_FRAMES, n_channels))
    after = qc.mean_images(qc.load_chunk(run.suite2p_input, 0, DISTORTION_FRAMES, n_channels))
    return [_save(run, plots.plot_channel_pairs(
        before, after, ("Before distortion correction", "After"), run.channel_names),
        "distortion_correction", "before_after_distortion_correction",
        f"Mean of the first {DISTORTION_FRAMES} timepoints of each channel before and after "
        f"distortion correction (calibration {run.settings.distortion.transform_mat}, mode "
        f"{run.settings.distortion.mode}).", False)]  # fmt: skip


# ---- registration ------------------------------------------------------------------


def registration_plots(run: RunQc) -> list[Path]:
    """Before/after registration, the shift traces, nonrigid offsets and the registered
    means with their crops to the region no frame's shifts wrap into."""
    from .registration import load_registered_outputs

    settings, n_channels = run.settings, run.settings.channels.n_channels
    registered = load_registered_outputs(run.plane_dir)
    ops, fs = registered.ops, settings.registration.fs_hz
    before = qc.mean_images(qc.load_chunk(run.suite2p_input, 0, REGISTRATION_FRAMES, n_channels))
    first_n = qc.registered_first_n_means(registered, REGISTRATION_FRAMES, settings)
    saved = [
        _save(run, plots.plot_channel_pairs(before, first_n, ("Before Suite2p registration",
              "After"), run.channel_names), "motion_registration", "before_after_registration",
              f"Mean of the first {REGISTRATION_FRAMES} timepoints of each channel before and "
              "after Suite2p registration.", True),
        _save(run, plots.plot_registration_traces(ops, fs), "motion_registration",
              "registration_traces", "Suite2p's rigid shifts and registration correlation for "
              "every frame.", True),
    ]  # fmt: skip
    if np.ndim(ops.get("xoff1", [])) >= 2:
        saved.append(_save(run, plots.plot_nonrigid_offsets(ops, fs), "motion_registration",
                           "nonrigid_offsets", "Suite2p's nonrigid block shifts for every frame.",
                           True))  # fmt: skip
    functional, structural = qc.suite2p_mean_images(ops)
    by_channel = {settings.channels.functional_channel: functional,
                  settings.channels.structural_channel: structural}  # fmt: skip
    recording = [by_channel[ch] for ch in range(1, n_channels + 1)]
    rows, cols = qc.unwrapped_region(ops)
    saved.append(_save(run, plots.plot_channel_columns(
        [first_n, recording, [i[rows, cols] for i in first_n], [i[rows, cols] for i in recording]],
        (f"First {REGISTRATION_FRAMES} Registered Frames", "Suite2p Recording-Wide Mean",
         f"First {REGISTRATION_FRAMES}, Cropped", "Recording-Wide, Cropped"),
        run.channel_names, limit_columns=(2, 3), box=(rows, cols), box_columns=(0, 1)),
        "motion_registration", "registered_means_cropped",
        f"Registered mean images: the first {REGISTRATION_FRAMES} frames and Suite2p's "
        f"recording-wide mean, then each cropped to rows {rows.start}-{rows.stop - 1}, columns "
        f"{cols.start}-{cols.stop - 1} (dashed box), the region no frame's rigid or block shifts "
        "wrap into. Contrast is set from the cropped images.", True))  # fmt: skip
    return saved


def focus_plots(run: RunQc) -> list[Path]:
    """How well each registered frame matches the mean image, with the frames flagged."""
    from .registration import load_registered_outputs

    settings = run.settings
    registered = load_registered_outputs(run.plane_dir)
    focus = qc.focus_qc(registered, settings)
    axes = plots.plot_focus_qc(focus, settings.registration.fs_hz, registered.ops,
                               settings.focus.merge_gap_frames)  # fmt: skip
    return [_save(run, axes, "motion_registration", "focus_qc",
                  f"How well each registered frame of the {settings.focus.channel} channel "
                  f"matches the mean image, against its running median over "
                  f"{settings.focus.window_frames} frames; frames more than "
                  f"{settings.focus.drop_threshold:g} robust SDs below it are flagged.",
                  True)]  # fmt: skip


def frame_order_plots(run: RunQc) -> list[Path]:
    """Every stage's frames against the acquired TIFF (reads each stage again)."""
    from .frame_order import check_frame_order

    stages = ("bidiphase", *(("undistorted",) if distortion_corrected(run.settings) else ()),
              "registered")  # fmt: skip
    reports = check_frame_order(run.plane_dir, stages, run.settings, raw_tiff=run.input_tiff)
    for report in reports:
        log.info("Frame order: %s", report.summary())
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": plots.QC_FONTS}):
        axes = plots.plot_frame_order(reports)
        return [_save(run, axes, "frame_order", "frame_order",
                      "Each stage's frames against the acquired TIFF, by the mean of each frame "
                      "per channel minus its ~4 s running mean: for each window of 200 frames, "
                      "the position of its best match in the acquired TIFF relative to its own "
                      "(0 = in place), and its correlation with its own position. Summaries: "
                      + " | ".join(r.summary() for r in reports), True)]  # fmt: skip


# ---- running them --------------------------------------------------------------------


def _run_steps(steps: dict[str, Callable[[], list[Path]]]) -> dict[str, list[str] | str]:
    """Each step's saved plots, or ``"failed: <error>"``: a failed plot is advisory, so it
    is logged and recorded, and the other steps still run."""
    results: dict[str, list[str] | str] = {}
    for name, step in steps.items():
        try:
            results[name] = [str(path) for path in step()]
        except Exception as error:  # noqa: BLE001 - deliberate: QC never fails the run
            log.warning("QC %s failed: %r", name, error)
            results[name] = f"failed: {error!r}"
        finally:
            plt.close("all")
    return results


def summary_path(run: RunQc) -> Path:
    """``batch_qc_summary.json`` in the registration's ``quality_control/``: each registration
    of a variant keeps its own."""
    return run.plane_dir.parent / QC_FOLDER / SUMMARY_FILE


def _write_summary(run: RunQc, part: str, results: dict) -> Path:
    """Records *results* (plot paths, or why a step saved none) under *part* in the run's
    :func:`summary_path`, keeping the other parts."""
    path = summary_path(run)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = json.loads(path.read_text()) if path.is_file() else {}
    summary[part] = {"plane_dir": str(run.plane_dir), "plots": results}
    path.write_text(json.dumps(summary, indent=2))
    return path


def save_run_qc(run: RunQc, frame_phases=None, frame_order: bool = False) -> dict:
    """The QC notebook's preprocessing and registration plots for *run*.

    Args:
        frame_phases: The run's per-frame phases (``CorrectionSummary.frame_phases``); None
            for the chunked method.
        frame_order: Also check every stage's frame order (rereads each stage).

    Returns:
        dict: Each step's saved plots, or why it failed; also written to the run's
        :func:`summary_path`.
    """
    steps = {
        "raw_mean": lambda: raw_mean_plots(run),
        "bidiphase": lambda: bidiphase_plots(run, frame_phases),
        "distortion_correction": lambda: distortion_plots(run),
        "motion_registration": lambda: registration_plots(run),
        "focus": lambda: focus_plots(run),
    }
    if frame_order:
        steps["frame_order"] = lambda: frame_order_plots(run)
    results = _run_steps(steps)
    _write_summary(run, "preprocessing_and_registration", results)
    return results


# ---- fluorescence (after the Cellpose extraction) ------------------------------------------


def fluorescence_plots(run: RunQc, contrast) -> list[Path]:
    """Photon-free levels, F0 overview, mask alignment, the non-ROI background and the top
    ROIs, as the QC notebook draws them, on all frames, with the mean image at *contrast*."""
    from .background import dff_versions, top_rois
    from .background_io import SOURCE_LABELS, load_non_roi_traces
    from .extraction import EXTRACTION_RECORD, load_cellpose_labels
    from .fluorescence import audit_table, compute_dff
    from .photon_free_io import background_without_level, raw_levels, traces_without_level
    from .plane import load_plane_traces
    from .registration import NoPerFrameRecord, load_registered_outputs, red_usable

    settings, fl = run.settings, run.settings.fluorescence
    record = json.loads((run.plane_dir / EXTRACTION_RECORD).read_text())
    traces = load_plane_traces(run.plane_dir)
    saved, levels = [], None
    if fl.subtract_photon_free_level:
        levels = raw_levels(run.input_tiff, run.plane_dir, traces.f.shape[1], settings)
        traces = traces_without_level(traces, levels)
        with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": plots.QC_FONTS}):
            fig = plots.plot_photon_free_levels(levels.green, levels.red, traces.fs).figure
            saved.append(_save(run, fig, "fluorescence", "photon_free_levels",
                               "Each frame's photon-free level (the reading of pixels that caught "
                               "no photon), in counts, measured on the raw TIFF over each "
                               f"channel's dimmest {fl.photon_free_pixel_percentile:g}% of pixels; "
                               "subtracted from F, Fneu and the background before ΔF/F0.",
                               True))  # fmt: skip
    green = compute_dff(traces.f, traces.fneu, traces.fs, fl)
    table = audit_table(traces.f, traces.fneu, green)
    saved.append(_save(run, plots.plot_f0_overview(table, fl.min_f0), "fluorescence",
                       "f0_overview", f"F0 of every ROI ({fl.baseline} baseline) and how often it "
                       f"is at or below min_f0 = {fl.min_f0:g}, where ΔF/F0 is undefined.",
                       True))  # fmt: skip
    mean_img, _, _ = scale_contrast(traces.ops["meanImg"], contrast.black_percentile,
                                    contrast.white_percentile, contrast.gamma)  # fmt: skip
    shift = record.get("mask_alignment") or {"dy": 0, "dx": 0}
    saved.append(_save(run, plots.plot_mask_alignment(
        mean_img, load_cellpose_labels(record["seg_file"]), traces.labels,
        title=f"dy={shift['dy']}, dx={shift['dx']}"), "fluorescence", "mask_alignment",
        f"Cellpose masks from {record['seg_file']} as segmented and as used on this plane, "
        f"shifted by dy={shift['dy']}, dx={shift['dx']} px, on the green mean image "
        f"({contrast.label()}).", True))  # fmt: skip

    registered = load_registered_outputs(run.plane_dir)
    background = load_non_roi_traces(registered, run.plane_dir, traces.labels, settings)
    trace_background = background_without_level(background, levels) if levels else background
    has_red = traces.f_chan2 is not None
    try:
        red_ok, red_why = red_usable(run.suite2p_input, settings) if has_red else (False, "no red")
    except NoPerFrameRecord as error:
        red_ok, red_why = False, f"cannot tell whether red is usable: {error}"
    label = SOURCE_LABELS["non_roi"]
    versions = dff_versions(
        traces.f, traces.fneu, traces.f_chan2, traces.fneu_chan2,
        trace_background.green, trace_background.red, traces.fs, fl,
        red_usable=red_ok, standard=green,
    )  # fmt: skip
    coefficients = {f"{label} In F": versions.background_coef["F"]}
    if versions.red_coef is not None:
        coefficients[f"Red In Green ({label})"] = versions.red_coef
    saved.append(_save(run, plots.plot_background(
        mean_img, {label: background.mask},
        {label: {"green": trace_background.green, "red": trace_background.red}},
        coefficients, traces.fs), "fluorescence", "background",
        f"The {label} background mask ({int(background.mask.sum())} px) on the green mean image, "
        "its green and red traces, and the coefficients with which it is regressed out of each "
        f"ROI; red regression {'used' if red_ok else 'skipped'} ({red_why}). The vessel "
        "background needs the recording's own vessel masks (QC notebook).", True))  # fmt: skip

    rois = top_rois(green.dff, N_TOP_ROIS)
    dffs = {"Standard Green": green.dff, f"{label} Bg Removed": versions.background.dff}
    if versions.red is not None:
        dffs[f"{label} Bg + Red Removed"] = versions.red.dff
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": plots.QC_FONTS}):
        axes = plots.plot_top_rois(dffs, rois, xlabel="Frames")
        axes[0].figure.suptitle(f"Top {N_TOP_ROIS} ROIs By Maximum Standard ΔF/F0")
        saved.append(_save(run, axes[0].figure, "fluorescence", "top_rois_dff",
                           f"ΔF/F0 of the {N_TOP_ROIS} ROIs with the largest maximum standard "
                           "green ΔF/F0 (all frames), largest on top: standard, with the "
                           f"{label} background regressed out of F and Fneu first, and "
                           + ("with each ROI's red trace regressed out as well."
                              if versions.red is not None else f"no red regression ({red_why}).")
                           + " Every panel uses the same vertical scale.", True))  # fmt: skip
    return saved


def save_fluorescence_qc(run: RunQc, contrast) -> dict:
    """:func:`fluorescence_plots` for a run whose Cellpose masks are extracted; also written
    to the run's :func:`summary_path`."""
    results = _run_steps({"fluorescence": lambda: fluorescence_plots(run, contrast)})
    _write_summary(run, "fluorescence", results)
    return results
