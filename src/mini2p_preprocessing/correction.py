"""Bidiphase correction (per frame, or per chunk) and optional destriping of an interleaved TIFF.

``phase.method = "per_frame"`` reads the TIFF twice: once to measure every frame's line
correlation in each channel, once to shift each frame by the phase chosen for it
(``frame_phase.py``). ``"chunked"`` estimates and applies one phase per chunk.
"""

from __future__ import annotations

import gc
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

from .bidiphase import AlignmentChoice, apply_bidiphase, select_alignment
from .config import PhaseSettings, PipelineSettings
from .destripe import detrend_chunk
from .frame_phase import (
    FramePhases,
    apply_frame_phases,
    choose_frame_phases,
    fallback_stretches,
    line_correlation_curves,
)
from .progress import PROGRESS_EVERY_S, ProgressLog
from .tiff_io import chunk_bounds, count_timepoints, read_all_channels, write_interleaved

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GlobalResidual:
    """One second-pass phase for the whole recording (second_pass_mode='whole')."""

    phase_px: int
    sample_count: int
    choice: AlignmentChoice


@dataclass(frozen=True)
class CorrectionSummary:
    chunks: list[dict]  # empty for method = "per_frame"
    global_residual: GlobalResidual | None
    frame_phases: FramePhases | None = None  # method = "per_frame" only

    @property
    def global_second_phase_px(self) -> int | None:
        return self.global_residual.phase_px if self.global_residual else None

    @property
    def global_second_phase_sample_count(self) -> int | None:
        return self.global_residual.sample_count if self.global_residual else None


# ---- analysis: pure functions on in-memory chunks --------------------------------


def first_pass(
    chunks: Sequence[np.ndarray], start: int, settings: PipelineSettings
) -> tuple[list[np.ndarray], int, AlignmentChoice]:
    """Estimate and apply the first phase; a manual override for ``start`` wins."""
    choice = select_alignment(chunks, settings.phase, settings.channels)
    phase_px = int(settings.phase.manual_first_phase_by_start.get(start, choice.phase_px))
    return [apply_bidiphase(chunk, phase_px) for chunk in chunks], phase_px, choice


@dataclass(frozen=True)
class CorrectionStages:
    """Every intermediate of one chunk's correction, for writing and for QC."""

    after_first: list[np.ndarray]
    after_second: list[np.ndarray]
    corrected: list[np.ndarray]
    first_phase_px: int
    second_phase_px: int
    first_choice: AlignmentChoice
    second_choice: AlignmentChoice | None


def correction_stages(
    chunks: Sequence[np.ndarray],
    start: int,
    settings: PipelineSettings,
    global_residual: GlobalResidual | None = None,
) -> CorrectionStages:
    """Phase-correct (one or two passes) then optionally destripe one chunk.

    ``global_residual`` is required when ``second_pass_mode='whole'``.
    """
    phase = settings.phase
    after_first, first_phase, first_choice = first_pass(chunks, start, settings)
    second_phase, second_choice, after_second = 0, None, after_first
    if phase.run_second_pass:
        if phase.second_pass_mode == "whole":
            if global_residual is None:
                raise ValueError("second_pass_mode='whole' needs a global_residual")
            second_choice, estimated = global_residual.choice, global_residual.phase_px
        else:
            second_choice = select_alignment(after_first, phase, settings.channels)
            estimated = second_choice.phase_px
        second_phase = int(phase.manual_second_phase_by_start.get(start, estimated))
        after_second = [apply_bidiphase(chunk, second_phase) for chunk in after_first]
    corrected = (
        [detrend_chunk(c, settings.destripe.preserve_frame_mean) for c in after_second]
        if settings.destripe.enabled
        else after_second
    )
    return CorrectionStages(
        after_first, after_second, corrected, first_phase, second_phase, first_choice, second_choice
    )


def correct_chunk(
    chunks: Sequence[np.ndarray],
    start: int,
    settings: PipelineSettings,
    global_residual: GlobalResidual | None = None,
) -> tuple[list[np.ndarray], dict]:
    """Corrected per-channel chunks plus the chunk's sidecar record."""
    stages = correction_stages(chunks, start, settings, global_residual)
    record = chunk_record(
        start,
        len(chunks[0]),
        stages.first_phase_px,
        stages.second_phase_px,
        stages.first_choice,
        stages.second_choice,
    )
    return stages.corrected, record


def chunk_record(
    start: int,
    n_frames: int,
    first_phase: int,
    second_phase: int,
    first: AlignmentChoice,
    second: AlignmentChoice | None,
) -> dict:
    return {
        "start": start,
        "end": start + n_frames,
        "first_phase_px": int(first_phase),
        "second_phase_px": int(second_phase),
        "first_peak_score": float(first.estimate.peak_score),
        "first_score_margin": float(first.estimate.score_margin),
        "first_at_search_boundary": bool(first.estimate.at_boundary),
        "first_selected_source": first.selected_source,
        "first_score_by_source": first.score_by_source,
        "second_selected_source": second.selected_source if second is not None else None,
        "second_score_by_source": second.score_by_source if second is not None else None,
    }


def residual_sample_indices(n_timepoints: int, max_frames: int | None) -> np.ndarray:
    """Timepoints used for the whole-recording residual, spread evenly in time."""
    if max_frames is None:
        return np.arange(n_timepoints)
    count = min(int(max_frames), n_timepoints)
    return np.unique(np.linspace(0, n_timepoints - 1, count, dtype=int))


def phase_metadata(
    settings: PipelineSettings,
    run_name: str,
    input_tiff: Path,
    output_tiff: Path,
    summary: CorrectionSummary,
) -> dict:
    """JSON sidecar describing every chunk and the settings that produced it."""
    phase = settings.phase
    channels = settings.channels
    alignment_channel = settings.phase_alignment_channel
    return {
        "run_name": run_name,
        "input_tiff": str(input_tiff),
        "output_tiff": str(output_tiff),
        "n_channels": channels.n_channels,
        "structural_channel_one_based": channels.structural_channel,
        "functional_channel_one_based": channels.functional_channel,
        "phase_alignment_source": phase.alignment_source,
        "phase_alignment_channel_one_based": alignment_channel,
        "phase_score_switch_margin": phase.score_switch_margin,
        "motion_alignment_source": settings.registration.motion_alignment_source,
        "registration_channel_one_based": settings.registration_channel,
        "phase_reference_channel_zero_based": (alignment_channel or channels.structural_channel)
        - 1,
        "phase_chunk_frames": phase.chunk_frames,
        "phase_method": phase.estimator,
        "phase_max_shift_px": phase.max_shift_px if phase.estimator == "wide" else 10,
        "phase_score_crop_px": phase.score_crop_px,
        "phase_boundary_warning_px": phase.boundary_warning_px,
        "manual_first_phase_by_start": phase.manual_first_phase_by_start,
        "manual_second_phase_by_start": phase.manual_second_phase_by_start,
        "second_pass_enabled": phase.run_second_pass,
        "second_pass_mode": phase.second_pass_mode if phase.run_second_pass else None,
        "global_second_phase_px": summary.global_second_phase_px,
        "global_second_phase_sample_count": summary.global_second_phase_sample_count,
        "destriping_enabled": settings.destripe.enabled,
        "operation_order": (
            ["first_biphase"]
            + (["second_biphase"] if phase.run_second_pass else [])
            + (["destriping"] if settings.destripe.enabled else [])
        ),
        "preserve_frame_mean": settings.destripe.preserve_frame_mean,
        "chunks": summary.chunks,
    }


def frame_phases_from_curves(curves: np.ndarray, settings: PipelineSettings) -> FramePhases:
    """Choose every frame's phase from (frames, channels, shifts) line-correlation curves."""
    channels = settings.channels
    if curves.ndim != 3 or curves.shape[1] != channels.n_channels:
        raise ValueError(
            f"Expected (frames, {channels.n_channels} channels, shifts) curves, got {curves.shape}"
        )
    functional = curves[:, channels.functional_channel - 1]
    structural = (
        None
        if channels.structural_channel == channels.functional_channel
        else curves[:, channels.structural_channel - 1]
    )
    return choose_frame_phases(structural, functional, settings.phase)


def frame_phase_record(curves: np.ndarray, chosen: FramePhases, settings: PipelineSettings) -> dict:
    """Arrays for the ``.phase_frames.npz`` sidecar: the choice, its evidence, the curves.

    The ``structural_*`` arrays are absent when there is no separate structural channel.
    """
    max_shift = settings.phase.max_shift_px
    record = {
        "phase_px": chosen.phase_px,
        "used_structural": chosen.used_structural,
        "functional_px": chosen.functional_px,
        "fallback_px": chosen.fallback_px,
        "functional_loss": chosen.functional_loss,
        "line_curves": curves.astype(np.float32),
        "shifts_px": np.arange(-max_shift, max_shift + 1),
    }
    if chosen.has_structural:
        record["structural_px"] = chosen.structural_px
        record["structural_consistency"] = chosen.structural_consistency
        record["structural_loss"] = chosen.structural_loss
    return record


def frames_at_search_boundary(phases_px: np.ndarray, phase: PhaseSettings) -> np.ndarray:
    """Frames whose phase is within ``boundary_warning_px`` of ±``max_shift_px``: the true
    phase may lie outside the searched range."""
    return np.flatnonzero(np.abs(phases_px) >= phase.max_shift_px - phase.boundary_warning_px)


# Settings that only method = "chunked" reads; set under "per_frame" they do nothing.
CHUNKED_ONLY_SETTINGS = (
    "chunk_frames",
    "estimator",
    "score_crop_px",
    "alignment_source",
    "score_switch_margin",
    "run_second_pass",
    "second_pass_mode",
    "second_pass_global_max_frames",
)


def ignored_chunked_settings(phase: PhaseSettings) -> list[str]:
    """Chunked-only settings the config sets explicitly although the method is per_frame."""
    if phase.method != "per_frame":
        return []
    return [name for name in CHUNKED_ONLY_SETTINGS if name in phase.model_fields_set]


def _source_metadata(
    settings: PipelineSettings, run_name: str, input_tiff: Path, output_tiff: Path
) -> dict:
    channels = settings.channels
    return {
        "run_name": run_name,
        "input_tiff": str(input_tiff),
        "output_tiff": str(output_tiff),
        "n_channels": channels.n_channels,
        "structural_channel_one_based": channels.structural_channel,
        "functional_channel_one_based": channels.functional_channel,
    }


def frame_phase_metadata(
    settings: PipelineSettings,
    run_name: str,
    input_tiff: Path,
    output_tiff: Path,
    chosen: FramePhases,
) -> dict:
    """JSON sidecar for method = "per_frame": settings, a summary, where functional was used."""
    phase = settings.phase
    stretches = fallback_stretches(chosen.used_structural)
    return {
        **_source_metadata(settings, run_name, input_tiff, output_tiff),
        "phase_method": "per_frame",
        "phase_settings": phase.model_dump(
            include={
                "max_shift_px",
                "consistency_window_frames",
                "consistency_tolerance_px",
                "consistency_threshold",
                "agreement_px",
                "trust_window_frames",
                "fallback_window_frames",
            }
        ),
        "motion_alignment_source": settings.registration.motion_alignment_source,
        "registration_channel_one_based": settings.registration_channel,
        "destriping_enabled": settings.destripe.enabled,
        "operation_order": ["per_frame_biphase"]
        + (["destriping"] if settings.destripe.enabled else []),
        "preserve_frame_mean": settings.destripe.preserve_frame_mean,
        "n_frames": int(len(chosen.phase_px)),
        "frames_at_search_boundary": int(len(frames_at_search_boundary(chosen.phase_px, phase))),
        "phase_range_px": [int(chosen.phase_px.min()), int(chosen.phase_px.max())],
        "fraction_structural": float(np.mean(chosen.used_structural)),
        "functional_stretches": [list(stretch) for stretch in stretches],
        "frames_sidecar": output_tiff.with_suffix(".phase_frames.npz").name,
    }


# ---- I/O: stream the TIFF through the pure functions above -----------------------


def estimate_global_residual(
    tif: tifffile.TiffFile, n_timepoints: int, settings: PipelineSettings
) -> GlobalResidual:
    """Second-pass phase from first-pass-corrected frames spread across the recording."""
    n_channels = settings.channels.n_channels
    sample_indices = residual_sample_indices(
        n_timepoints, settings.phase.second_pass_global_max_frames
    )
    sampled: list[list[np.ndarray]] = [[] for _ in range(n_channels)]
    bounds = chunk_bounds(n_timepoints, settings.phase.chunk_frames)
    progress = ProgressLog("Whole-recording residual", len(bounds), log.info, PROGRESS_EVERY_S)
    for start, end in bounds:
        selected = sample_indices[(sample_indices >= start) & (sample_indices < end)]
        if not len(selected):
            progress.step_done()  # nearly instant; counted, as skips spread through the loop
            continue
        after_first, _, _ = first_pass(
            read_all_channels(tif, start, end, n_channels), start, settings
        )
        for channel in range(n_channels):
            sampled[channel].append(after_first[channel][selected - start])
        progress.step_done()
    samples = [np.concatenate(parts, axis=0) for parts in sampled]
    choice = select_alignment(samples, settings.phase, settings.channels)
    log.info(
        "Whole-recording second phase: %+d px from %d frames; source=%s; scores=%s",
        choice.phase_px,
        len(sample_indices),
        choice.selected_source,
        choice.score_by_source,
    )
    return GlobalResidual(int(choice.phase_px), int(len(sample_indices)), choice)


def measure_line_curves(tiff: Path, settings: PipelineSettings) -> np.ndarray:
    """Line-correlation curves of every frame of every channel, (frames, channels, shifts)."""
    n_channels = settings.channels.n_channels
    max_shift = settings.phase.max_shift_px
    parts = []
    with tifffile.TiffFile(tiff) as tif:
        n_timepoints = count_timepoints(tif, n_channels, settings.max_timepoints)
        bounds = chunk_bounds(n_timepoints, settings.phase.frames_per_read)
        progress = ProgressLog("Measuring frame phases", len(bounds), log.info, PROGRESS_EVERY_S)
        for start, end in bounds:
            chunks = read_all_channels(tif, start, end, n_channels)
            parts.append(np.stack([line_correlation_curves(c, max_shift) for c in chunks], 1))
            progress.step_done()
    return np.concatenate(parts, axis=0)


def _write_per_frame(
    input_tiff: Path, output_tiff: Path, phases_px: np.ndarray, settings: PipelineSettings
) -> None:
    """Shift, destripe and write every frame; ``output_tiff`` is replaced only when complete."""
    n_channels = settings.channels.n_channels
    partial = output_tiff.with_name(output_tiff.name + ".partial")
    try:
        with (
            tifffile.TiffFile(input_tiff) as tif,
            tifffile.TiffWriter(partial, bigtiff=True) as writer,
        ):
            n_timepoints = count_timepoints(tif, n_channels, settings.max_timepoints)
            bounds = chunk_bounds(n_timepoints, settings.phase.frames_per_read)
            progress = ProgressLog("Phase correction", len(bounds), log.info, PROGRESS_EVERY_S)
            for start, end in bounds:
                shifted = [
                    apply_frame_phases(chunk, phases_px[start:end])
                    for chunk in read_all_channels(tif, start, end, n_channels)
                ]
                if settings.destripe.enabled:
                    keep_mean = settings.destripe.preserve_frame_mean
                    shifted = [detrend_chunk(c, keep_mean) for c in shifted]
                write_interleaved(writer, shifted, settings.dtype)
                progress.step_done()
        partial.replace(output_tiff)
    finally:
        partial.unlink(missing_ok=True)


def check_line_curves(curves: np.ndarray, input_tiff: Path, settings: PipelineSettings) -> None:
    """Refuse curves that do not match this recording and these settings, before writing."""
    with tifffile.TiffFile(input_tiff) as tif:
        n_timepoints = count_timepoints(tif, settings.channels.n_channels, settings.max_timepoints)
    expected = (n_timepoints, settings.channels.n_channels, 2 * settings.phase.max_shift_px + 1)
    if np.shape(curves) != expected:
        raise ValueError(
            f"line_curves have shape {np.shape(curves)}, but {input_tiff.name} with these "
            f"settings needs {expected} (frames, channels, 2 * max_shift_px + 1); measure "
            "them again with the current settings"
        )


def warn_about_frame_phases(chosen: FramePhases, settings: PipelineSettings) -> None:
    """Log what a per-frame run cannot fix silently: boundary hits, little structural use,
    and chunked-only settings that have no effect."""
    phase = settings.phase
    at_boundary = frames_at_search_boundary(chosen.phase_px, phase)
    if len(at_boundary):
        log.warning(
            "%d frame(s) have a phase within %d px of the ±%d px search limit (first: %s); "
            "raise phase.max_shift_px",
            len(at_boundary),
            phase.boundary_warning_px,
            phase.max_shift_px,
            at_boundary[:10].tolist(),
        )
    if chosen.has_structural and np.mean(chosen.used_structural) < 0.5:
        log.warning(
            "The structural channel set the phase for only %.0f%% of frames; the rest use "
            "the functional channel averaged over %d frames, which smooths over fast jumps. "
            "Check the .phase_frames.png",
            100 * np.mean(chosen.used_structural),
            phase.fallback_window_frames,
        )
    ignored = ignored_chunked_settings(phase)
    if ignored:
        log.warning(
            'phase.method = "per_frame" ignores these chunked-only settings: %s',
            ", ".join(ignored),
        )


def preprocess_per_frame(
    input_tiff: Path,
    output_tiff: Path,
    settings: PipelineSettings,
    run_name: str,
    line_curves: np.ndarray | None = None,
) -> CorrectionSummary:
    """Measure, choose and apply a phase per frame; write the TIFF and both sidecars.

    ``line_curves`` (from ``measure_line_curves``, e.g. a notebook preview) skips the
    measuring pass; they must match the recording and the current ``max_shift_px``.
    """
    if line_curves is None:
        curves = measure_line_curves(input_tiff, settings)
    else:
        check_line_curves(line_curves, input_tiff, settings)
        curves = line_curves
    chosen = frame_phases_from_curves(curves, settings)
    log.info(
        "Per-frame phase %+d..%+d px; structural channel used for %.0f%% of %d frames; "
        "functional channel in %d stretch(es)",
        chosen.phase_px.min(),
        chosen.phase_px.max(),
        100 * np.mean(chosen.used_structural),
        len(chosen.phase_px),
        len(fallback_stretches(chosen.used_structural)),
    )
    warn_about_frame_phases(chosen, settings)
    _write_per_frame(input_tiff, output_tiff, chosen.phase_px, settings)
    np.savez_compressed(
        output_tiff.with_suffix(".phase_frames.npz"),
        **frame_phase_record(curves, chosen, settings),
    )
    metadata = frame_phase_metadata(settings, run_name, input_tiff, output_tiff, chosen)
    output_tiff.with_suffix(".phase.json").write_text(json.dumps(metadata, indent=2))
    log.info("Saved %s and its .phase.json / .phase_frames.npz sidecars", output_tiff)
    return CorrectionSummary(chunks=[], global_residual=None, frame_phases=chosen)


# Per-frame sidecars; a chunked rerun of the same folder removes them so none go stale.
PER_FRAME_SIDECAR_SUFFIXES = (
    ".phase_frames.npz",
    ".phase_frames.png",
    ".phase_frames.pdf",
    ".phase_frames.svg",
    ".phase_frames.caption.txt",
)


def preprocess_tiff(
    input_tiff: Path,
    output_tiff: Path,
    settings: PipelineSettings,
    run_name: str,
    line_curves: np.ndarray | None = None,
) -> CorrectionSummary:
    """Write the phase-corrected (optionally destriped) TIFF and its .phase.json sidecar.

    ``line_curves`` is used only by ``phase.method = "per_frame"``.
    """
    output_tiff = Path(output_tiff)
    if output_tiff.exists() and not settings.overwrite:
        raise FileExistsError(f"{output_tiff} exists; set overwrite=true to replace it")
    if settings.phase.method == "per_frame":
        return preprocess_per_frame(input_tiff, output_tiff, settings, run_name, line_curves)
    if line_curves is not None:
        raise ValueError('line_curves are only used by phase.method = "per_frame"')
    for suffix in PER_FRAME_SIDECAR_SUFFIXES:
        output_tiff.with_suffix(suffix).unlink(missing_ok=True)
    phase = settings.phase
    n_channels = settings.channels.n_channels
    records: list[dict] = []
    with tifffile.TiffFile(input_tiff) as tif:
        n_timepoints = count_timepoints(tif, n_channels, settings.max_timepoints)
        global_residual = (
            estimate_global_residual(tif, n_timepoints, settings)
            if phase.run_second_pass and phase.second_pass_mode == "whole"
            else None
        )
        with tifffile.TiffWriter(output_tiff, bigtiff=True) as writer:
            bounds = chunk_bounds(n_timepoints, phase.chunk_frames)
            progress = ProgressLog("Phase correction", len(bounds), log.info, PROGRESS_EVERY_S)
            for start, end in bounds:
                chunks = read_all_channels(tif, start, end, n_channels)
                corrected, record = correct_chunk(chunks, start, settings, global_residual)
                write_interleaved(writer, corrected, settings.dtype)
                records.append(record)
                log.info(
                    "%7d/%d; first=%+d px; second=%+d px; source=%s; score=%.3f; "
                    "boundary=%s; destriping=%s",
                    end,
                    n_timepoints,
                    record["first_phase_px"],
                    record["second_phase_px"],
                    record["first_selected_source"],
                    record["first_peak_score"],
                    record["first_at_search_boundary"],
                    settings.destripe.enabled,
                )
                del chunks, corrected
                gc.collect()
                progress.step_done()
    summary = CorrectionSummary(chunks=records, global_residual=global_residual)
    sidecar = output_tiff.with_suffix(".phase.json")
    metadata = phase_metadata(settings, run_name, input_tiff, output_tiff, summary)
    sidecar.write_text(json.dumps(metadata, indent=2))
    log.info("Saved %s and %s", output_tiff, sidecar)
    return summary
