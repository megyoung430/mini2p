"""Computations behind the notebook's QC figures (no plotting here)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

from .bidiphase import select_alignment
from .config import PipelineSettings
from .correction import frame_phases_from_curves, measure_line_curves
from .focus import FocusQC, assess_focus, frame_match, smooth_frames
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


def _offsets(ops: dict, key: str, n_dims: int, n_frames_total: int) -> np.ndarray:
    values = np.asarray(ops[key], dtype=float)
    if values.ndim != n_dims:
        raise ValueError(f"ops['{key}'] should be {n_dims}-D, got shape {values.shape}")
    if len(values) != n_frames_total:
        raise ValueError(f"ops['{key}'] covers {len(values)} frames, ops['yoff'] {n_frames_total}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"ops['{key}'] has values that are not finite")
    return values


def _edge_margins(rigid: np.ndarray, blocks: np.ndarray | None) -> tuple[int, int]:
    """(near-edge, far-edge) margins, in pixels, that some frame wraps into along one axis.

    Suite2p rolls each frame by -rigid, so rigid > 0 wraps rows (columns) from the near
    edge into the far edge and rigid < 0 the reverse; nonrigid registration then samples
    that rolled frame up to the block shifts away (bilinear, edge-padded), reaching
    further into the wrapped band by its most outward block shift.
    """
    toward_near = np.maximum(0.0, -rigid)
    toward_far = np.maximum(0.0, rigid)
    if blocks is not None:
        toward_near = toward_near - np.min(blocks, axis=1)
        toward_far = toward_far + np.max(blocks, axis=1)
    near = int(np.ceil(max(0.0, toward_near.max())))
    far = int(np.ceil(max(0.0, toward_far.max())))
    return near, far


def unwrapped_region(ops: dict, n_frames: int | None = None) -> tuple[slice, slice]:
    """(rows, columns) of the registered frame that no frame's shifts wrap into.

    Suite2p shifts each frame with ``roll(frame, (-yoff, -xoff))``, a circular shift, then
    (nonrigid registration) resamples the rolled frame by the block shifts ``yoff1`` /
    ``xoff1``, which can reach further into the wrapped band. The margins come from every
    frame (or the first ``n_frames``), per edge. Suite2p's own ``ops['yrange']`` /
    ``['xrange']`` leave out frames it flags as bad, which are still averaged into its mean
    images, and ignore the block shifts, so they are not used here.
    """
    for key in ("yoff", "xoff", "Ly", "Lx"):
        if key not in ops:
            raise KeyError(f"ops has no '{key}'; it needs Suite2p's registration output")
    if n_frames is not None and n_frames < 1:
        raise ValueError(f"n_frames must be at least 1, got {n_frames}")
    has_blocks = [ops.get(key) is not None for key in ("yoff1", "xoff1")]
    if any(has_blocks) and not all(has_blocks):
        raise ValueError("ops has only one of 'yoff1' / 'xoff1'; nonrigid output needs both")
    n_total = len(np.atleast_1d(ops["yoff"]))
    region = []
    for rigid_key, block_key, size_key in (("yoff", "yoff1", "Ly"), ("xoff", "xoff1", "Lx")):
        rigid = _offsets(ops, rigid_key, 1, n_total)[:n_frames]
        blocks = _offsets(ops, block_key, 2, n_total)[:n_frames] if all(has_blocks) else None
        near, far = _edge_margins(rigid, blocks)
        length = int(ops[size_key])
        if near >= length - far:
            raise ValueError(
                f"Shifts wrap {near} + {far} of the {length} px along {size_key}: nothing "
                "is free of wrapped pixels"
            )
        region.append(slice(near, length - far))
    return region[0], region[1]


def suite2p_mean_images(ops: dict) -> tuple[np.ndarray, np.ndarray]:
    """(functional, structural) recording-wide means stored by Suite2p."""
    return (
        np.squeeze(np.asarray(ops["meanImg"], dtype=np.float32)),
        np.squeeze(np.asarray(ops["meanImg_chan2"], dtype=np.float32)),
    )


def focus_qc(outputs: RegisteredOutputs, settings: PipelineSettings) -> FocusQC:
    """Per-frame focus QC of the registered binary, inside the region no shift wraps into.

    Each frame of ``settings.focus.channel`` is matched against Suite2p's mean image of that
    channel; both are smoothed first. Reads the binary in chunks, so memory stays at one
    chunk of frames.
    """
    focus = settings.focus
    rows, cols = unwrapped_region(outputs.ops)
    functional_mean, structural_mean = suite2p_mean_images(outputs.ops)
    stack, mean = {
        "structural": (outputs.structural, structural_mean),
        "functional": (outputs.functional, functional_mean),
    }[focus.channel]
    if mean.shape != stack.shape[1:]:
        raise ValueError(f"Suite2p mean image {mean.shape} vs frames {stack.shape[1:]}")
    reference = smooth_frames(mean[None, rows, cols], focus.smooth_px)[0]

    def chunk_stats(bounds: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        frames = np.asarray(stack[bounds[0] : bounds[1], rows, cols], np.float32)
        return frame_match(smooth_frames(frames, focus.smooth_px), reference), frames.mean(
            axis=(1, 2)
        )

    stats = [chunk_stats(b) for b in chunk_bounds(len(stack), focus.chunk_frames)]
    return assess_focus(
        np.concatenate([m for m, _ in stats]), np.concatenate([b for _, b in stats]), focus
    )
