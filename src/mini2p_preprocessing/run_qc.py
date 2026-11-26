"""The QC notebook's plots for one run, saved without the notebook (batch runs).

:func:`save_run_qc` draws, for a run the pipeline has processed, the plots the QC notebook
(``notebooks/qc_single_run.ipynb``) saves for its preprocessing and registration, into the
same ``quality_control/<step>/`` folders (``qc_figures``): the raw mean, the bidiphase
phases and before/after, destriping, distortion correction, the registration's
before/after, shift traces, nonrigid offsets, registered means and their crops, the focus QC
and (optionally, as it rereads every stage) the frame order. :func:`save_fluorescence_qc`
draws the notebook's fluorescence plots once the run's Cellpose masks are extracted: the
photon-free levels, F0 overview, mask alignment, the non-ROI background and the top ROIs.

Everything is read back from the run's folder (its recorded settings, TIFFs, phase sidecar
and registration), so QC can run after the batch has recorded the run as complete, in its
own process (``python -m mini2p_preprocessing.run_qc``, or ``mini2p-run-qc``).

What only the notebook can do is left out: the plots behind sliders, the hand-chosen
contrast, and the vessel background (each recording needs its own vessel masks). Each plot
is its own step; one that fails is logged with its traceback and listed with the error in
the registration's ``quality_control/batch_qc_summary.json`` (:func:`summary_path`), and
the others are still saved. QC is advisory: a failed plot never fails the run.

The modules behind the fluorescence plots (Suite2p's extraction and torch), the run's
records (``movies``) and the pipeline are imported inside the functions that need them, so
the preprocessing QC and ``cellpose_batch`` do not load them just by importing this one.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import plots, qc, raw_previews, registration
from .batch import QC_EXIT_SOME_FAILED
from .bidiphase import select_alignment
from .config import (
    PipelineSettings,
    QCRun,
    corrected_tiff_path,
    distortion_corrected,
    find_registered_plane,
    variant_dir_of,
)
from .contrast import scale_contrast
from .correction import correction_stages, load_frame_phases
from .destripe import detrend_chunk
from .exports import Contrast
from .frame_phase import FramePhases, apply_frame_phases
from .qc_figures import QC_FOLDER, REGISTRATION_STEPS, figure_of, qc_step_dir, save_qc_figure

log = logging.getLogger(__name__)
SUMMARY_FILE = "batch_qc_summary.json"
RAW_MEAN_FRAMES = 500  # as the notebook's raw-mean cell
# The notebook's [qc] defaults: a batch has no [qc] of its own to read
REGISTRATION_FRAMES = QCRun.model_fields["qc_registration_frames"].default
DISTORTION_FRAMES = QCRun.model_fields["qc_distortion_frames"].default
N_TOP_ROIS = 20
EXIT_SOME_FAILED = QC_EXIT_SOME_FAILED  # the QC process's exit code when some plots fail


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

    @property
    def frame_phases_file(self) -> Path:
        """The per-frame phases the pipeline saved beside the corrected TIFF."""
        return self.corrected_tiff.with_suffix(".phase_frames.npz")


class NotDrawn(Exception):
    """A plot that does not apply to this run (e.g. nonrigid offsets of a rigid one)."""


# ---- finding the run ------------------------------------------------------------------


def _run_qc(variant_dir: Path, plane_dir: Path, settings: PipelineSettings) -> RunQc:
    """The :class:`RunQc` of *plane_dir*, with the TIFF paths the pipeline gave this variant."""
    from .movies import acquired_tiff
    from .pipeline import suite2p_input_for

    input_tiff = acquired_tiff(variant_dir, settings)
    return RunQc(
        input_tiff,
        variant_dir,
        corrected_tiff_path(variant_dir, input_tiff, settings),
        suite2p_input_for(variant_dir, input_tiff, settings),
        Path(plane_dir),
        settings,
    )


def _recorded_settings(variant_dir: Path) -> PipelineSettings:
    """The settings the variant was processed with (``pipeline_settings.json``).

    Raises:
        FileNotFoundError: If it has none.
    """
    from .movies import load_run_settings

    recorded = load_run_settings(variant_dir)
    if recorded is None:
        raise FileNotFoundError(f"{variant_dir} has no pipeline_settings.json")
    return recorded


def run_qc_for_variant(variant_dir: Path) -> RunQc:
    """The :class:`RunQc` of a processed variant folder, from the settings it recorded
    (``pipeline_settings.json``), with the registration those settings name.

    Raises:
        FileNotFoundError: If the variant has no settings record, no record of its acquired
            TIFF, or no registration.
    """
    variant_dir = Path(variant_dir)
    settings = _recorded_settings(variant_dir)
    registration = settings.registration
    plane = find_registered_plane(
        variant_dir, registration.motion_alignment_source, registration.save_folder
    )
    if plane is None:
        raise FileNotFoundError(f"{variant_dir} has no {registration.save_folder} registration")
    return _run_qc(variant_dir, plane, settings)


def run_qc_for_plane(plane_dir: Path, extraction_settings: PipelineSettings) -> RunQc:
    """The :class:`RunQc` of an already processed plane, from the settings its variant
    folder recorded, with the ``[fluorescence]`` of *extraction_settings* (the ΔF/F0 the
    extraction is checked with) and the ``save_folder`` of *plane_dir* itself (the variant's
    record names whichever registration ran last).

    Raises:
        FileNotFoundError: If the variant has no ``pipeline_settings.json`` or no record of
            its acquired TIFF.
    """
    plane_dir = Path(plane_dir)
    variant_dir = variant_dir_of(plane_dir)
    recorded = _recorded_settings(variant_dir)
    registration = recorded.registration.model_copy(update={"save_folder": plane_dir.parent.name})
    settings = recorded.model_copy(
        update={"fluorescence": extraction_settings.fluorescence, "registration": registration}
    )
    return _run_qc(variant_dir, plane_dir, settings)


# ---- saving -----------------------------------------------------------------------------


def _caption(run: RunQc, text: str, registration: bool) -> str:
    """*text* followed by the recording, output folder and (for a registration plot) the
    registration, as the QC notebook captions its plots."""
    caption = f"{text.strip()}\nRecording: {run.input_tiff}\nOutput folder: {run.variant_dir}\n"
    if registration:
        caption += f"Registration: {run.plane_dir.parent}\n"
    return caption + "Saved by the batch (mini2p_preprocessing.run_qc).\n"


def _save(run: RunQc, drawn, step: str, name: str, text: str) -> Path:
    """Saves *drawn* where the QC notebook would (:func:`save_qc_figure`), closes its
    figure, and returns the saved path without extension."""
    registration = step in REGISTRATION_STEPS
    figure = figure_of(drawn)
    try:
        return save_qc_figure(
            figure,
            run.variant_dir,
            step,
            name,
            _caption(run, text, registration),
            plane_dir=run.plane_dir if registration else None,
            save_folder=run.settings.registration.save_folder if registration else None,
        )
    finally:
        plt.close(figure)


def _qc_fonts():
    """The QC notebook's sans-serif fonts, for the plots it draws in them."""
    return plt.rc_context({"font.family": "sans-serif", "font.sans-serif": plots.QC_FONTS})


@dataclass
class _Loads:
    """Data several plots share, each loaded once; a load that fails fails (with the same
    error) every plot that needs it, and only those."""

    loaders: dict[str, Callable[[], object]]
    _done: dict[str, tuple[object, Exception | None]] = field(default_factory=dict)

    def __getitem__(self, name: str):
        """The loaded *name*, loading it the first time; a failed load raises again."""
        if name not in self._done:
            try:
                self._done[name] = (self.loaders[name](), None)
            except Exception as error:  # noqa: BLE001 - re-raised to every plot needing it
                self._done[name] = (None, error)
        value, error = self._done[name]
        if error is not None:
            raise error
        return value


def _run_steps(steps: dict[str, Callable[[], Path]]) -> dict[str, str]:
    """Each step's saved plot, ``"not drawn: <why>"``, or ``"failed: <error>"``: QC is
    advisory, so a failure is logged (with its traceback) and recorded, and the other steps
    still run."""
    results = {}
    for name, step in steps.items():
        open_before = set(plt.get_fignums())
        try:
            results[name] = str(step())
        except NotDrawn as why:
            results[name] = f"not drawn: {why}"
        except Exception as error:  # noqa: BLE001 - deliberate: QC never fails the run
            log.exception("QC plot %s failed", name)
            results[name] = f"failed: {error!r}"
        finally:  # a plot that failed half-drawn (others close their own figure)
            for number in set(plt.get_fignums()) - open_before:
                plt.close(number)
    return results


def failures(results: dict[str, str]) -> list[str]:
    """The steps of a QC result that failed."""
    return [name for name, result in results.items() if result.startswith("failed")]


def summary_path(run: RunQc) -> Path:
    """``batch_qc_summary.json`` in the registration's ``quality_control/``: each registration
    of a variant keeps its own."""
    return run.plane_dir.parent / QC_FOLDER / SUMMARY_FILE


def _write_summary(run: RunQc, part: str, results: dict[str, str], **notes: str) -> Path:
    """Records *results* (and any *notes*) under *part* in the run's :func:`summary_path`,
    keeping the other parts (an unreadable summary is replaced: it only lists plots that are
    on disk)."""
    path = summary_path(run)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        summary = json.loads(path.read_text()) if path.is_file() else {}
    except json.JSONDecodeError:
        log.warning("Replacing unreadable %s", path)
        summary = {}
    summary[part] = {"plane_dir": str(run.plane_dir), "plots": results, **notes}
    path.write_text(json.dumps(summary, indent=2))
    return path


# ---- preprocessing --------------------------------------------------------------------


def raw_mean_plot(run: RunQc) -> Path:
    """The mean of the first frames of the acquired TIFF, before any correction."""
    raw_mean, n_raw = qc.raw_mean_images(
        run.input_tiff, RAW_MEAN_FRAMES, run.settings.channels.n_channels
    )
    stem = qc_step_dir(run.variant_dir, "raw_mean") / f"raw_meanImg_first{n_raw}"
    with _qc_fonts():
        figure = raw_previews.draw_preview(raw_mean, n_raw, run.input_tiff.name, run.channel_names)
        try:
            raw_previews.save_preview(
                figure,
                stem,
                raw_previews.preview_caption(run.input_tiff, n_raw, RAW_MEAN_FRAMES),
                dpi=200,
            )
        finally:
            plt.close(figure)
    return stem


def frame_phases_plot(run: RunQc, phases: FramePhases) -> Path:
    """The phase chosen for each frame and which channel set it (per-frame method)."""
    text = (
        "The bidiphase (odd/even-line) phase chosen for each frame, in px, and which channel "
        "set it: the structural channel where its per-frame phase was consistent and agreed "
        "with the functional channel, else the functional channel averaged over "
        f"{run.settings.phase.fallback_window_frames} frames."
    )
    return _save(run, plots.plot_frame_phases(phases), "bidiphase", "frame_phases", text)


def phase_trace_plot(run: RunQc, trace: list[dict]) -> Path:
    """The phase and search score of each chunk (chunked method)."""
    text = (
        "The bidiphase phase and its search score for each chunk of "
        f'{run.settings.phase.chunk_frames} timepoints (phase.method = "chunked").'
    )
    return _save(run, plots.plot_phase_trace(trace, run.settings), "bidiphase", "phase_trace", text)


def score_curve_plot(run: RunQc, trace: list[dict]) -> Path:
    """The score against the phase tried, for the chunk that scored lowest ("wide"
    estimator only)."""
    settings = run.settings
    if settings.phase.estimator != "wide":
        raise NotDrawn(f"phase.estimator = {settings.phase.estimator!r}, not 'wide'")
    worst = qc.lowest_score_chunk(trace)
    chunks = qc.load_chunk(
        run.input_tiff, worst["start"], worst["end"], settings.channels.n_channels
    )
    choice = select_alignment(chunks, settings.phase, settings.channels)
    span = f"{worst['start']}:{worst['end']}"
    ax = plots.plot_score_curve(
        choice.estimate, f"Lowest-score chunk {span} ({choice.selected_source})"
    )
    return _save(
        run,
        ax,
        "bidiphase",
        f"score_curve_lowest_chunk_{worst['start']}",
        f"Adjacent-line score against the phase tried, for the lowest-scoring chunk "
        f"(timepoints {span}, {choice.selected_source} channel).",
    )


def stage_pair_plot(run: RunQc, before, after, titles, step, name, text) -> Path:
    """Mean images of two stages of the same frames, each channel side by side."""
    figure = plots.plot_channel_pairs(
        qc.mean_images(before), qc.mean_images(after), titles, run.channel_names
    )
    return _save(run, figure, step, name, text)


def per_frame_stages(
    raw: list[np.ndarray], phases: FramePhases, settings: PipelineSettings
) -> dict:
    """The first frames after the per-frame bidiphase correction, and destriped."""
    after = [apply_frame_phases(c, phases.phase_px[: len(raw[0])]) for c in raw]
    destriped = [detrend_chunk(c, settings.destripe.preserve_frame_mean) for c in after]
    return {"raw": raw, "after_phase": after, "destriped": destriped}


def chunked_stages(raw: list[np.ndarray], settings: PipelineSettings) -> dict:
    """The first frames after each pass of the chunked correction, as the notebook's
    ``correction_stages``. With ``second_pass_mode = "whole"`` only the first pass is
    worked out (its second phase comes from the whole recording): the later stages are then
    the first pass's, and their plots are refused (:func:`bidiphase_steps`)."""
    phase = settings.phase
    if phase.run_second_pass and phase.second_pass_mode == "whole":
        settings = settings.model_copy(
            update={"phase": phase.model_copy(update={"run_second_pass": False})}
        )
    return {"raw": raw, "stages": correction_stages(raw, 0, settings)}


def bidiphase_steps(run: RunQc, loads: _Loads) -> dict[str, Callable[[], Path]]:
    """The notebook's bidiphase and destriping plots for the run's phase method, over its
    first ``phase.chunk_frames`` timepoints (the notebook's ``qc_start_timepoint`` = 0)."""
    settings = run.settings

    def span() -> str:  # the frames actually read: fewer in a shorter recording
        return f"Mean of timepoints 0:{len(loads['raw'][0])} of each channel"

    if settings.phase.method == "per_frame":
        steps = {
            "bidiphase/frame_phases": lambda: frame_phases_plot(run, loads["phases"]),
            "bidiphase/before_after_bidiphase": lambda: stage_pair_plot(
                run,
                loads["per_frame"]["raw"],
                loads["per_frame"]["after_phase"],
                ("Before", "After per-frame bidiphase"),
                "bidiphase",
                "before_after_bidiphase",
                f"{span()} before and after the per-frame bidiphase correction.",
            ),
        }
        if settings.destripe.enabled:
            steps["destriping/before_after_destriping"] = lambda: stage_pair_plot(
                run,
                loads["per_frame"]["after_phase"],
                loads["per_frame"]["destriped"],
                ("Before destriping", "After destriping"),
                "destriping",
                "before_after_destriping",
                f"{span()}, bidiphase-corrected, before and after destriping.",
            )
        return steps

    def stages():
        return loads["chunked"]["stages"]

    def refuse_whole_second_pass():
        """The plots after a ``second_pass_mode = "whole"`` second pass need its phase from
        the whole recording (the QC notebook works it out)."""
        if settings.phase.run_second_pass and settings.phase.second_pass_mode == "whole":
            raise NotDrawn('second_pass_mode = "whole" needs the whole-recording phase')

    def second_pass_plot() -> Path:
        refuse_whole_second_pass()
        mode = settings.phase.second_pass_mode
        return stage_pair_plot(
            run,
            stages().after_first,
            stages().after_second,
            ("After first bidiphase", f"After second ({mode})"),
            "bidiphase",
            "before_after_second_bidiphase",
            f"{span()} after the first bidiphase pass and after the second ({mode}, "
            f"{stages().second_phase_px:+d} px).",
        )

    def destriping_plot() -> Path:
        refuse_whole_second_pass()
        return stage_pair_plot(
            run,
            stages().after_second,
            stages().corrected,
            ("Before destriping", "After destriping"),
            "destriping",
            "before_after_destriping",
            f"{span()}, bidiphase-corrected, before and after destriping.",
        )

    steps = {
        "bidiphase/phase_trace": lambda: phase_trace_plot(run, loads["trace"]),
        "bidiphase/score_curve": lambda: score_curve_plot(run, loads["trace"]),
        "bidiphase/before_after_first_bidiphase": lambda: stage_pair_plot(
            run,
            loads["chunked"]["raw"],
            stages().after_first,
            ("Before", "After first bidiphase"),
            "bidiphase",
            "before_after_first_bidiphase",
            f"{span()} before and after the first bidiphase pass "
            f"({stages().first_phase_px:+d} px).",
        ),
    }
    if settings.phase.run_second_pass:
        steps["bidiphase/before_after_second_bidiphase"] = second_pass_plot
    if settings.destripe.enabled:
        steps["destriping/before_after_destriping"] = destriping_plot
    return steps


def distortion_plot(run: RunQc) -> Path:
    """The first frames before and after distortion correction."""
    if not distortion_corrected(run.settings):
        raise NotDrawn('[distortion] mode = "off"')
    n_channels = run.settings.channels.n_channels
    before = qc.first_frames(run.corrected_tiff, DISTORTION_FRAMES, n_channels)
    after = qc.first_frames(run.suite2p_input, DISTORTION_FRAMES, n_channels)
    distortion = run.settings.distortion
    return stage_pair_plot(
        run,
        before,
        after,
        ("Before distortion correction", "After"),
        "distortion_correction",
        "before_after_distortion_correction",
        f"Mean of the first {DISTORTION_FRAMES} timepoints of each channel before and after "
        f"distortion correction (calibration {distortion.transform_mat}, mode {distortion.mode}).",
    )


# ---- registration ------------------------------------------------------------------


def before_after_registration_plot(run: RunQc, registered, first_n: list[np.ndarray]) -> Path:
    """The first frames of what Suite2p registered, before and after registration."""
    before = qc.mean_images(
        qc.first_frames(run.suite2p_input, REGISTRATION_FRAMES, run.settings.channels.n_channels)
    )
    figure = plots.plot_channel_pairs(
        before, first_n, ("Before Suite2p registration", "After"), run.channel_names
    )
    return _save(
        run,
        figure,
        "motion_registration",
        "before_after_registration",
        f"Mean of the first {REGISTRATION_FRAMES} timepoints of each channel before and "
        "after Suite2p registration.",
    )


def registration_traces_plot(run: RunQc, ops: dict) -> Path:
    """Suite2p's rigid shifts and registration correlation for every frame."""
    figure = plots.plot_registration_traces(ops, run.settings.registration.fs_hz)
    return _save(
        run,
        figure,
        "motion_registration",
        "registration_traces",
        "Suite2p's rigid shifts and registration correlation for every frame.",
    )


def nonrigid_offsets_plot(run: RunQc, ops: dict) -> Path:
    """Suite2p's nonrigid block shifts for every frame (nonrigid registrations only)."""
    if np.ndim(ops.get("xoff1", [])) < 2:
        raise NotDrawn("rigid registration: no block shifts")
    figure = plots.plot_nonrigid_offsets(ops, run.settings.registration.fs_hz)
    return _save(
        run,
        figure,
        "motion_registration",
        "nonrigid_offsets",
        "Suite2p's nonrigid block shifts for every frame.",
    )


def registered_means_cropped_plot(run: RunQc, ops: dict, first_n: list[np.ndarray]) -> Path:
    """The registered means (first frames and recording-wide), whole and cropped to the
    region no frame's shifts wrap into."""
    channels = run.settings.channels
    functional, structural = qc.suite2p_mean_images(ops)
    by_channel = {channels.functional_channel: functional, channels.structural_channel: structural}
    recording = [by_channel[ch] for ch in range(1, channels.n_channels + 1)]
    rows, cols = qc.unwrapped_region(ops)
    n = REGISTRATION_FRAMES
    figure = plots.plot_channel_columns(
        [first_n, recording, [i[rows, cols] for i in first_n], [i[rows, cols] for i in recording]],
        (
            f"First {n} Registered Frames",
            "Suite2p Recording-Wide Mean",
            f"First {n}, Cropped",
            "Recording-Wide, Cropped",
        ),
        run.channel_names,
        limit_columns=(2, 3),
        box=(rows, cols),
        box_columns=(0, 1),
    )
    return _save(
        run,
        figure,
        "motion_registration",
        "registered_means_cropped",
        f"Registered mean images: the first {n} frames and Suite2p's recording-wide "
        f"mean, then each cropped to rows {rows.start}-{rows.stop - 1}, columns "
        f"{cols.start}-{cols.stop - 1} (dashed box), the region no frame's rigid or "
        "block shifts wrap into. Contrast is set from the cropped images.",
    )


def focus_plot(run: RunQc, registered) -> Path:
    """How well each registered frame matches the mean image, with the frames flagged
    (reads the whole registered binary, in chunks)."""
    focus = qc.focus_qc(registered, run.settings)
    settings = run.settings.focus
    axes = plots.plot_focus_qc(
        focus, run.settings.registration.fs_hz, registered.ops, settings.merge_gap_frames
    )
    return _save(
        run,
        axes,
        "motion_registration",
        "focus_qc",
        f"How well each registered frame of the {settings.channel} channel matches the "
        f"mean image, against its running median over {settings.window_frames} frames; "
        f"frames more than {settings.drop_threshold:g} robust SDs below it are "
        "flagged.",
    )


def frame_order_plot(run: RunQc) -> Path:
    """Every stage's frames against the acquired TIFF (reads each stage again)."""
    from .frame_order import check_frame_order

    stages = (
        "bidiphase",
        *(("undistorted",) if distortion_corrected(run.settings) else ()),
        "registered",
    )
    reports = check_frame_order(run.plane_dir, stages, run.settings, raw_tiff=run.input_tiff)
    for report in reports:
        log.info("Frame order: %s", report.summary())
    with _qc_fonts():
        axes = plots.plot_frame_order(reports)
        return _save(
            run,
            axes,
            "frame_order",
            "frame_order",
            "Each stage's frames against the acquired TIFF, by the mean of each frame "
            "per channel minus its ~4 s running mean: for each window of 200 frames, the "
            "position of its best match in the acquired TIFF relative to its own (0 = in "
            "place), and its correlation with its own position. Summaries: "
            + " | ".join(r.summary() for r in reports),
        )


# ---- running them --------------------------------------------------------------------


def _preprocessing_loads(run: RunQc) -> _Loads:
    """What the preprocessing and registration plots share: the per-frame phases, the
    first raw frames and their corrected stages, the chunk phase trace, the registration
    and its first registered frames, each loaded when a plot first needs it."""
    settings = run.settings
    n_channels = settings.channels.n_channels
    loads = _Loads({})

    def raw():
        return qc.first_frames(run.input_tiff, settings.phase.chunk_frames, n_channels)

    def registered():
        return registration.load_registered_outputs(run.plane_dir)

    loads.loaders.update(
        phases=lambda: load_frame_phases(run.frame_phases_file),
        raw=raw,
        per_frame=lambda: per_frame_stages(loads["raw"], loads["phases"], settings),
        chunked=lambda: chunked_stages(loads["raw"], settings),
        trace=lambda: qc.phase_trace(run.input_tiff, settings),  # reads the whole TIFF
        registered=registered,
        first_n=lambda: qc.registered_first_n_means(
            loads["registered"], REGISTRATION_FRAMES, settings
        ),
    )
    return loads


def save_run_qc(run: RunQc, frame_order: bool = False) -> dict[str, str]:
    """The QC notebook's preprocessing and registration plots for *run*, each its own step.

    Args:
        frame_order: Also check every stage's frame order (rereads each stage).

    Returns:
        Each step's saved plot, why it was not drawn, or why it failed; also written to the
        run's :func:`summary_path`.
    """
    loads = _preprocessing_loads(run)
    steps = {
        "raw_mean": lambda: raw_mean_plot(run),
        **bidiphase_steps(run, loads),
        "distortion_correction": lambda: distortion_plot(run),
        "motion_registration/before_after_registration": lambda: before_after_registration_plot(
            run, loads["registered"], loads["first_n"]
        ),
        "motion_registration/registration_traces": lambda: registration_traces_plot(
            run, loads["registered"].ops
        ),
        "motion_registration/nonrigid_offsets": lambda: nonrigid_offsets_plot(
            run, loads["registered"].ops
        ),
        "motion_registration/registered_means_cropped": lambda: registered_means_cropped_plot(
            run, loads["registered"].ops, loads["first_n"]
        ),
        "motion_registration/focus_qc": lambda: focus_plot(run, loads["registered"]),
    }
    if frame_order:
        steps["frame_order"] = lambda: frame_order_plot(run)
    results = _run_steps(steps)
    _write_summary(run, "preprocessing_and_registration", results)
    return results


# ---- fluorescence (after the Cellpose extraction) ------------------------------------------


@dataclass(frozen=True)
class _Fluorescence:
    """A plane's traces, ΔF/F0 and backgrounds as the fluorescence plots use them."""

    traces: object  # PlaneTraces, with the photon-free level removed when it is subtracted
    green: object  # the standard green ΔF/F0 (DffResult)
    mean_img: np.ndarray  # the green mean image at the export contrast
    background: object  # the non-ROI background traces and mask
    trace_background: object  # the same, with the photon-free level removed
    versions: object  # the background- and red-corrected ΔF/F0 (dff_versions)
    red_ok: bool
    red_why: str


def _fluorescence(run: RunQc, contrast: Contrast, traces, levels) -> _Fluorescence:
    """A plane's traces (less the photon-free *levels*, unless None), standard ΔF/F0,
    contrast-scaled mean image, non-ROI background and its corrected ΔF/F0."""
    from .background import dff_versions
    from .background_io import load_non_roi_traces
    from .fluorescence import compute_dff
    from .photon_free_io import background_without_level, traces_without_level

    settings, fl = run.settings, run.settings.fluorescence
    if levels is not None:
        traces = traces_without_level(traces, levels)
    green = compute_dff(traces.f, traces.fneu, traces.fs, fl)
    mean_img, _, _ = scale_contrast(
        traces.ops["meanImg"], contrast.black_percentile, contrast.white_percentile, contrast.gamma
    )
    background = load_non_roi_traces(
        registration.load_registered_outputs(run.plane_dir), run.plane_dir, traces.labels, settings
    )
    trace_background = (
        background_without_level(background, levels) if levels is not None else background
    )
    if traces.f_chan2 is None:
        red_ok, red_why = False, "no red"
    else:
        try:
            red_ok, red_why = registration.red_usable(run.suite2p_input, settings)
        except registration.NoPerFrameRecord as error:
            red_ok, red_why = False, f"cannot tell whether red is usable: {error}"
    versions = dff_versions(
        traces.f,
        traces.fneu,
        traces.f_chan2,
        traces.fneu_chan2,
        trace_background.green,
        trace_background.red,
        traces.fs,
        fl,
        red_usable=red_ok,
        standard=green,
    )
    return _Fluorescence(
        traces, green, mean_img, background, trace_background, versions, red_ok, red_why
    )


def photon_free_levels_plot(run: RunQc, levels, fs: float) -> Path:
    """Each frame's photon-free level, as subtracted before ΔF/F0."""
    pct = run.settings.fluorescence.photon_free_pixel_percentile
    with _qc_fonts():
        axes = plots.plot_photon_free_levels(levels.green, levels.red, fs)
        return _save(
            run,
            axes,
            "fluorescence",
            "photon_free_levels",
            "Each frame's photon-free level (the reading of pixels that caught no "
            f"photon), in counts, measured on the raw TIFF over each channel's dimmest "
            f"{pct:g}% of pixels; subtracted from F, Fneu and the background before "
            "ΔF/F0.",
        )


def photon_free_note(run: RunQc, loads: _Loads) -> str:
    """Whether the photon-free level was subtracted: ``off``, ``subtracted``, or ``not
    subtracted: <why>`` when it could not be measured."""
    if not run.settings.fluorescence.subtract_photon_free_level:
        return "off"
    try:
        loads["levels"]
    except Exception as error:  # noqa: BLE001 - its own plot fails with it
        return f"not subtracted: {error!r}"
    return "subtracted"


def caption_note(photon_free: str) -> str:
    """What the fluorescence captions add about the photon-free level."""
    if photon_free == "off":
        return ""
    if photon_free == "subtracted":
        return " The photon-free level is subtracted first."
    return (
        " The photon-free level could not be measured "
        f"({photon_free.removeprefix('not subtracted: ')}), so it is NOT subtracted here, "
        "unlike the standard ΔF/F0."
    )


def f0_overview_plot(run: RunQc, data: _Fluorescence, note: str) -> Path:
    """F0 of every ROI and how often ΔF/F0 is undefined."""
    from .fluorescence import audit_table

    fl = run.settings.fluorescence
    table = audit_table(data.traces.f, data.traces.fneu, data.green)
    return _save(
        run,
        plots.plot_f0_overview(table, fl.min_f0),
        "fluorescence",
        "f0_overview",
        f"F0 of every ROI ({fl.baseline} baseline) and how often it is at or below "
        f"min_f0 = {fl.min_f0:g}, where ΔF/F0 is undefined.{note}",
    )


def mask_alignment_plot(run: RunQc, data: _Fluorescence, contrast: Contrast) -> Path:
    """The Cellpose masks as segmented and as shifted onto this plane."""
    from .extraction import EXTRACTION_RECORD, load_cellpose_labels

    record = json.loads((run.plane_dir / EXTRACTION_RECORD).read_text())
    shift = record.get("mask_alignment") or {"dy": 0, "dx": 0}
    figure = plots.plot_mask_alignment(
        data.mean_img,
        load_cellpose_labels(record["seg_file"]),
        data.traces.labels,
        title=f"dy={shift['dy']}, dx={shift['dx']}",
    )
    return _save(
        run,
        figure,
        "fluorescence",
        "mask_alignment",
        f"Cellpose masks from {record['seg_file']} as segmented and as used on this "
        f"plane, shifted by dy={shift['dy']}, dx={shift['dx']} px, on the green mean "
        f"image ({contrast.label()}).",
    )


def background_plot(run: RunQc, data: _Fluorescence, note: str) -> Path:
    """The non-ROI background mask, its traces and how much of it is regressed out."""
    from .background_io import SOURCE_LABELS

    label = SOURCE_LABELS["non_roi"]
    coefficients = {f"{label} In F": data.versions.background_coef["F"]}
    if data.versions.red_coef is not None:
        coefficients[f"Red In Green ({label})"] = data.versions.red_coef
    figure = plots.plot_background(
        data.mean_img,
        {label: data.background.mask},
        {label: {"green": data.trace_background.green, "red": data.trace_background.red}},
        coefficients,
        data.traces.fs,
    )
    return _save(
        run,
        figure,
        "fluorescence",
        "background",
        f"The {label} background mask ({int(data.background.mask.sum())} px) on the "
        "green mean image, its green and red traces, and the coefficients with which it "
        f"is regressed out of each ROI; red regression "
        f"{'used' if data.red_ok else 'skipped'} ({data.red_why}). The vessel background "
        f"needs the recording's own vessel masks (QC notebook).{note}",
    )


def top_rois_plot(run: RunQc, data: _Fluorescence, note: str) -> Path:
    """ΔF/F0 of the ROIs with the largest standard ΔF/F0, with each correction."""
    from .background import top_rois
    from .background_io import SOURCE_LABELS

    label = SOURCE_LABELS["non_roi"]
    rois = top_rois(data.green.dff, N_TOP_ROIS)
    dffs = {"Standard Green": data.green.dff, f"{label} Bg Removed": data.versions.background.dff}
    if data.versions.red is not None:
        dffs[f"{label} Bg + Red Removed"] = data.versions.red.dff
    red = (
        "with each ROI's red trace regressed out as well."
        if data.versions.red is not None
        else f"no red regression ({data.red_why})."
    )
    with _qc_fonts():
        axes = plots.plot_top_rois(dffs, rois, xlabel="Frames")
        axes[0].figure.suptitle(f"Top {N_TOP_ROIS} ROIs By Maximum Standard ΔF/F0")
        return _save(
            run,
            axes[0].figure,
            "fluorescence",
            "top_rois_dff",
            f"ΔF/F0 of the {N_TOP_ROIS} ROIs with the largest maximum standard green "
            f"ΔF/F0 (all frames), largest on top: standard, with the {label} background "
            f"regressed out of F and Fneu first, and {red} Every panel uses the same "
            f"vertical scale.{note}",
        )


def save_fluorescence_qc(run: RunQc, contrast: Contrast) -> dict[str, str]:
    """The QC notebook's fluorescence plots for a run whose Cellpose masks are extracted, on
    all frames, with the mean image at *contrast*; also written to the run's
    :func:`summary_path`, with ``photon_free_level`` (:func:`photon_free_note`). A
    photon-free level that cannot be measured fails its own plot; the others are drawn
    without subtracting it, and their captions and the summary say so."""
    from .photon_free_io import raw_levels
    from .plane import load_plane_traces

    loads = _Loads({"plane": lambda: load_plane_traces(run.plane_dir)})
    loads.loaders["levels"] = lambda: raw_levels(
        run.input_tiff, run.plane_dir, loads["plane"].f.shape[1], run.settings
    )
    photon_free = photon_free_note(run, loads)
    levels = loads["levels"] if photon_free == "subtracted" else None
    loads.loaders["data"] = lambda: _fluorescence(run, contrast, loads["plane"], levels)
    note = caption_note(photon_free)
    steps = {}
    if photon_free != "off":
        steps["fluorescence/photon_free_levels"] = lambda: photon_free_levels_plot(
            run, loads["levels"], loads["plane"].fs
        )
    steps |= {
        "fluorescence/f0_overview": lambda: f0_overview_plot(run, loads["data"], note),
        "fluorescence/mask_alignment": lambda: mask_alignment_plot(run, loads["data"], contrast),
        "fluorescence/background": lambda: background_plot(run, loads["data"], note),
        "fluorescence/top_rois_dff": lambda: top_rois_plot(run, loads["data"], note),
    }
    results = _run_steps(steps)
    _write_summary(run, "fluorescence", results, photon_free_level=photon_free)
    return results


# ---- command line ------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """``mini2p-run-qc``: the preprocessing and registration QC of one processed variant
    folder. Exits with :data:`EXIT_SOME_FAILED` when some plots could not be drawn (the
    others are saved)."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "--variant-dir",
        type=Path,
        required=True,
        help="run-XX/<distortion>/<destriping> folder the pipeline wrote",
    )
    parser.add_argument(
        "--frame-order",
        action="store_true",
        help="also check every stage's frame order (rereads each stage)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    results = save_run_qc(run_qc_for_variant(args.variant_dir), frame_order=args.frame_order)
    failed = failures(results)
    if failed:
        log.warning("QC plots that failed: %s", ", ".join(failed))
        raise SystemExit(EXIT_SOME_FAILED)


if __name__ == "__main__":
    main()
