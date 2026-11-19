"""Reading and writing channel-interleaved ScanImage TIFFs, one chunk at a time."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import tifffile

log = logging.getLogger(__name__)


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
    """Whole timepoints in a channel-interleaved TIFF (at most *max_timepoints*).

    A ScanImage acquisition stopped during a frame can leave that frame with only its first
    channels saved. Those leftover pages are left out (with a warning) when ScanImage's
    frame numbers show they are exactly that; any other page count that is not a multiple
    of *n_channels* (a page missing earlier shifts the channel of every later page) is
    refused.
    """
    n_pages = len(tif.pages)
    n_timepoints, leftover = divmod(n_pages, n_channels)
    if leftover:
        _check_unfinished_last_frame(tif, n_timepoints, n_channels)
        log.warning(
            "%s: %d pages is not divisible by %d channels; leaving out the unfinished last "
            "frame (%d page(s) of frame %d), so %d timepoints remain",
            tif.filename, n_pages, n_channels, leftover, n_timepoints + 1, n_timepoints,
        )  # fmt: skip
    return n_timepoints if max_timepoints is None else min(n_timepoints, int(max_timepoints))


FRAME_NUMBER = re.compile(r"^frameNumbers = (\d+)\s*$", re.MULTILINE)


def frame_number(tif: tifffile.TiffFile, page: int) -> int | None:
    """ScanImage's frame number in a page's header, or None for a page without one."""
    match = FRAME_NUMBER.search(tif.pages[page].description or "")
    return int(match.group(1)) if match else None


def _check_unfinished_last_frame(tif: tifffile.TiffFile, n_timepoints: int, n_channels: int):
    """Raise unless the pages after the last whole timepoint are one later, unfinished frame:
    the last whole timepoint's pages share a frame number, and the leftover pages share a
    higher one."""
    n_pages = len(tif.pages)
    whole = range(max(n_timepoints - 1, 0) * n_channels, n_timepoints * n_channels)
    numbers_whole = {frame_number(tif, page) for page in whole}
    numbers_left = {frame_number(tif, page) for page in range(n_timepoints * n_channels, n_pages)}
    if not (
        len(numbers_whole) == 1
        and len(numbers_left) == 1
        and None not in numbers_whole | numbers_left
        and min(numbers_left) > min(numbers_whole)
    ):
        raise ValueError(
            f"{n_pages} pages is not divisible by {n_channels} channels, and ScanImage's frame "
            "numbers do not show the leftover pages to be one unfinished last frame "
            f"(last whole timepoint: frame(s) {sorted(numbers_whole, key=str)}, leftover "
            f"pages: frame(s) {sorted(numbers_left, key=str)}); a page may be missing"
        )


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


class TooMuchClipped(ValueError):
    """More values than ``max_clipped_fraction`` did not fit the output type."""


def to_output_dtype(frames: np.ndarray, dtype: np.dtype) -> tuple[np.ndarray, int]:
    """``frames`` as ``dtype``, and how many values had to be clipped to fit it.

    Integer output is rounded to the nearest count (halves to even) and clipped to the
    type's range. A plain ``astype`` (what Suite2p does to any non-int16 TIFF) truncates
    towards zero and wraps values beyond the range around silently, e.g. 40000 to -25536.
    Values that are not finite are refused for every output type.
    """
    dtype = np.dtype(dtype)
    frames = np.asarray(frames)
    if dtype.kind not in "fiu":
        raise ValueError(f"Output dtype must be float or integer, not {dtype}")
    if frames.dtype.kind == "f" and not np.all(np.isfinite(frames)):
        raise ValueError("Frames have values that are not finite; they cannot be written")
    if dtype.kind == "f":
        return frames.astype(dtype, copy=False), 0
    if frames.dtype == dtype:
        return frames, 0
    limits = np.iinfo(dtype)
    if frames.dtype.kind == "f":
        if frames.dtype.itemsize < 2 * dtype.itemsize:
            # a float at least twice as wide holds every limit of the integer type exactly
            # (float16 cannot even hold 32767); float32 suffices for int16
            frames = frames.astype(np.float64)
        frames = np.rint(frames)
    clipped = int(np.count_nonzero((frames < limits.min) | (frames > limits.max)))
    return np.clip(frames, limits.min, limits.max).astype(dtype), clipped


def write_interleaved(
    writer: tifffile.TiffWriter, chunks: Sequence[np.ndarray], dtype: np.dtype
) -> int:
    """Append chunks in the input page order: ch0 t0, ch1 t0, ch0 t1, ...

    Each frame is converted as it is written (see ``to_output_dtype``), so only one
    converted frame is held at a time. Returns how many values were clipped.
    """
    clipped = 0
    for frame_index in range(len(chunks[0])):
        for channel_chunk in chunks:
            frame, n = to_output_dtype(channel_chunk[frame_index], dtype)
            writer.write(frame, contiguous=False)
            clipped += n
    return clipped


def n_written_values(n_timepoints: int, n_channels: int, frame_shape: Sequence[int]) -> int:
    return int(n_timepoints) * int(n_channels) * int(np.prod(frame_shape))


def check_clipped(
    clipped: int, n_values: int, path, dtype: np.dtype, max_fraction: float, log=log
) -> None:
    """Warn when values had to be clipped to fit the output type; raise ``TooMuchClipped``
    when more than *max_fraction* of them were (the data do not fit the type)."""
    if not clipped:
        return
    fraction = clipped / n_values
    message = (
        f"{path}: {clipped} of {n_values} values ({100 * fraction:.2g}%) were beyond the "
        f"{np.dtype(dtype)} range and clipped to it"
    )
    if fraction > max_fraction:
        raise TooMuchClipped(
            f"{message}, more than max_clipped_fraction = {max_fraction:g}; the data do not "
            f'fit {np.dtype(dtype)} (check the input, or set output_dtype = "float32")'
        )
    log.warning(message)


def output_record(dtype_name: str, clipped: int) -> dict:
    """What a sidecar records about how its TIFF was written."""
    return {"output_dtype": dtype_name, "clipped_values": int(clipped)}


@contextmanager
def replaced_when_complete(output: Path) -> Iterator[Path]:
    """Yield a ``.partial`` path to write; it replaces *output* only if the block finishes.

    A run that fails or is refused part-way leaves the previous *output* (and the sidecars
    written after it) as they were, so a file and its record always belong together.
    """
    output = Path(output)
    partial = output.with_name(output.name + ".partial")
    try:
        yield partial
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)
