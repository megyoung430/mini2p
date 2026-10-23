"""Computations behind the notebook's QC figures (no plotting here)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

from .bidiphase import select_alignment
from .config import PipelineSettings
from .correction import frame_phases_from_curves, measure_line_curves
from .frame_phase import FramePhases, phase_loss
from .registration import RegisteredOutputs
from .tiff_io import chunk_bounds, count_timepoints, read_all_channels


def load_chunk(tiff: Path, start: int, end: int, n_channels: int) -> list[np.ndarray]:
    with tifffile.TiffFile(tiff) as tif:
        n_timepoints = count_timepoints(tif, n_channels)
        if not 0 <= start < min(end, n_timepoints):
            raise ValueError(
                f"Timepoints {start}:{end} are outside a {n_timepoints}-frame recording"
            )
        return read_all_channels(tif, start, min(end, n_timepoints), n_channels)


def phase_trace(tiff: Path, settings: PipelineSettings) -> list[dict]:
    """First-pass phase estimate for every chunk, without writing anything."""
    n_channels = settings.channels.n_channels
    records = []
    with tifffile.TiffFile(tiff) as tif:
        n_timepoints = count_timepoints(tif, n_channels, settings.max_timepoints)
        for start, end in chunk_bounds(n_timepoints, settings.phase.chunk_frames):
            choice = select_alignment(
                read_all_channels(tif, start, end, n_channels), settings.phase, settings.channels
            )
            records.append(
                {
                    "start": start,
                    "end": end,
                    "phase_px": choice.phase_px,
                    "peak_score": choice.estimate.peak_score,
                    "score_margin": choice.estimate.score_margin,
                    "at_boundary": choice.estimate.at_boundary,
                    "selected_source": choice.selected_source,
                    "selected_channel_zero_based": choice.selected_channel_zero_based,
                    "score_by_source": choice.score_by_source,
                }
            )
    return records


@dataclass(frozen=True)
class FramePhasePreview:
    curves: np.ndarray  # (frames, channels, shifts); pass to preprocess_tiff as line_curves
    chosen: FramePhases
    structural_loss: np.ndarray  # structural correlation lost at the chosen phase


def frame_phase_preview(tiff: Path, settings: PipelineSettings) -> FramePhasePreview:
    """Per-frame phases as ``phase.method = "per_frame"`` would apply them, nothing written."""
    curves = measure_line_curves(tiff, settings)
    chosen = frame_phases_from_curves(curves, settings)
    structural = curves[:, settings.channels.structural_channel - 1]
    loss = phase_loss(structural, chosen.phase_px, settings.phase.max_shift_px)
    return FramePhasePreview(curves, chosen, loss)


def lowest_score_chunk(trace: list[dict]) -> dict:
    finite = [row for row in trace if np.isfinite(row["peak_score"])]
    if not finite:
        raise ValueError("No chunk has a finite phase score")
    return min(finite, key=lambda row: row["peak_score"])


def mean_images(stacks: list[np.ndarray]) -> list[np.ndarray]:
    return [np.mean(stack, axis=0) for stack in stacks]


def registered_first_n_means(
    outputs: RegisteredOutputs, n_frames: int, settings: PipelineSettings
) -> list[np.ndarray]:
    """Mean of the first ``n_frames`` registered frames, in 0-based physical channel order.

    Suite2p stores the functional channel in data.bin and the other in data_chan2.bin.
    """
    by_channel = {
        settings.channels.functional_channel: np.mean(outputs.functional[:n_frames], axis=0),
        settings.channels.structural_channel: np.mean(outputs.structural[:n_frames], axis=0),
    }
    return [by_channel[ch] for ch in range(1, settings.channels.n_channels + 1)]


def suite2p_mean_images(ops: dict) -> tuple[np.ndarray, np.ndarray]:
    """(functional, structural) recording-wide means stored by Suite2p."""
    return (
        np.squeeze(np.asarray(ops["meanImg"], dtype=np.float32)),
        np.squeeze(np.asarray(ops["meanImg_chan2"], dtype=np.float32)),
    )
