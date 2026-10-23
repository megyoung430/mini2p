"""Computations behind the notebook's QC figures (no plotting here)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

from .bidiphase import select_alignment
from .config import PipelineSettings
from .correction import frame_phases_from_curves, measure_line_curves
from .frame_phase import FramePhases
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


def frame_phase_preview(tiff: Path, settings: PipelineSettings) -> FramePhasePreview:
    """Per-frame phases as ``phase.method = "per_frame"`` would apply them, nothing written."""
    curves = measure_line_curves(tiff, settings)
    return FramePhasePreview(curves, frame_phases_from_curves(curves, settings))


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


def unwrapped_region(ops: dict, n_frames: int | None = None) -> tuple[slice, slice]:
    """(rows, columns) of the registered frame that no rigid shift wraps into.

    Suite2p shifts each frame with ``roll(frame, (-yoff, -xoff))``, a circular shift: a
    frame with ``yoff > 0`` moves up and its top ``yoff`` rows reappear at the bottom, one
    with ``yoff < 0`` the reverse, and likewise for ``xoff`` and columns. The margins come
    from every frame (or the first ``n_frames``). Suite2p's own ``ops['yrange']`` /
    ``['xrange']`` leave out frames it flags as bad, which are still averaged into its
    mean images, so they are not used here.
    """
    for key in ("yoff", "xoff", "Ly", "Lx"):
        if key not in ops:
            raise KeyError(f"ops has no '{key}'; it needs Suite2p's rigid registration output")
    yoff = np.asarray(ops["yoff"], float).ravel()[:n_frames]
    xoff = np.asarray(ops["xoff"], float).ravel()[:n_frames]
    region = []
    for offsets, length, axis in ((yoff, int(ops["Ly"]), "rows"), (xoff, int(ops["Lx"]), "cols")):
        start = int(np.ceil(max(0.0, -offsets.min())))  # wrapped in from the far edge
        stop = length - int(np.ceil(max(0.0, offsets.max())))
        if start >= stop:
            raise ValueError(
                f"Shifts of {offsets.min():+.0f}..{offsets.max():+.0f} px wrap every one of "
                f"the {length} {axis}: nothing is free of wrapped pixels"
            )
        region.append(slice(start, stop))
    return region[0], region[1]


def suite2p_mean_images(ops: dict) -> tuple[np.ndarray, np.ndarray]:
    """(functional, structural) recording-wide means stored by Suite2p."""
    return (
        np.squeeze(np.asarray(ops["meanImg"], dtype=np.float32)),
        np.squeeze(np.asarray(ops["meanImg_chan2"], dtype=np.float32)),
    )
