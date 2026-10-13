"""Reading and writing channel-interleaved ScanImage TIFFs, one chunk at a time."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import tifffile


def chunk_bounds(n_timepoints: int, chunk_frames: int) -> list[tuple[int, int]]:
    """Half-open (start, end) timepoint ranges covering the recording."""
    if chunk_frames < 1:
        raise ValueError(f"chunk_frames must be >= 1, got {chunk_frames}")
    return [
        (start, min(start + chunk_frames, n_timepoints))
        for start in range(0, n_timepoints, chunk_frames)
    ]


def count_timepoints(
    tif: tifffile.TiffFile, n_channels: int, max_timepoints: int | None = None
) -> int:
    n_pages = len(tif.pages)
    if n_pages % n_channels:
        raise ValueError(f"{n_pages} pages is not divisible by {n_channels} channels")
    n_timepoints = n_pages // n_channels
    return n_timepoints if max_timepoints is None else min(n_timepoints, int(max_timepoints))


def read_channel_chunk(
    tif: tifffile.TiffFile, start: int, end: int, channel: int, n_channels: int
) -> np.ndarray:
    """Frames start:end of one 0-based channel, shape (frames, Ly, Lx)."""
    pages = channel + n_channels * np.arange(start, end)
    return np.stack([tif.pages[int(page)].asarray() for page in pages], axis=0)


def read_all_channels(
    tif: tifffile.TiffFile, start: int, end: int, n_channels: int
) -> list[np.ndarray]:
    """One (frames, Ly, Lx) array per 0-based channel."""
    return [read_channel_chunk(tif, start, end, ch, n_channels) for ch in range(n_channels)]


def write_interleaved(
    writer: tifffile.TiffWriter, chunks: Sequence[np.ndarray], dtype: np.dtype
) -> None:
    """Append chunks in the input page order: ch0 t0, ch1 t0, ch0 t1, ..."""
    for frame_index in range(len(chunks[0])):
        for channel_chunk in chunks:
            writer.write(channel_chunk[frame_index].astype(dtype), contiguous=False)
