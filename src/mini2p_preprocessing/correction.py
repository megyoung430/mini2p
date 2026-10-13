"""Chunked bidiphase correction and optional destriping of an interleaved TIFF."""

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
from .config import PipelineSettings
from .destripe import detrend_chunk
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
    chunks: list[dict]
    global_residual: GlobalResidual | None

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
    for start, end in chunk_bounds(n_timepoints, settings.phase.chunk_frames):
        selected = sample_indices[(sample_indices >= start) & (sample_indices < end)]
        if not len(selected):
            continue
        after_first, _, _ = first_pass(
            read_all_channels(tif, start, end, n_channels), start, settings
        )
        for channel in range(n_channels):
            sampled[channel].append(after_first[channel][selected - start])
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


def preprocess_tiff(
    input_tiff: Path, output_tiff: Path, settings: PipelineSettings, run_name: str
) -> CorrectionSummary:
    """Write the phase-corrected (optionally destriped) TIFF and its .phase.json sidecar."""
    output_tiff = Path(output_tiff)
    if output_tiff.exists() and not settings.overwrite:
        raise FileExistsError(f"{output_tiff} exists; set overwrite=true to replace it")
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
            for start, end in chunk_bounds(n_timepoints, phase.chunk_frames):
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
    summary = CorrectionSummary(chunks=records, global_residual=global_residual)
    sidecar = output_tiff.with_suffix(".phase.json")
    metadata = phase_metadata(settings, run_name, input_tiff, output_tiff, summary)
    sidecar.write_text(json.dumps(metadata, indent=2))
    log.info("Saved %s and %s", output_tiff, sidecar)
    return summary
