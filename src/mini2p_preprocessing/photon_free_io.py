"""The per-frame photon-free level of a run's raw ScanImage TIFF, cached in the plane folder
(computation in ``photon_free``).

It is measured on the raw TIFF, not the registered binaries: interpolation in distortion
correction and registration mixes photon counts into photon-free pixels and lifts their
mode (by ~8 counts on the Mini2P). Each channel's level comes from its dimmest pixels in
the mean of the first frames, so a bright structural channel can be measured too. A
per-frame offset is the same in every pixel, so it is unchanged by those later steps and
can be subtracted from traces extracted from the registered binaries, frame for frame.

Two settings change that offset before Suite2p and are refused: destriping without
``preserve_frame_mean`` (it already drives every column to zero mean) and distortion
correction without ``crop_black_borders`` (its zero-filled borders carry no offset, so
the non-ROI background would hold only part of it).

Only the QC notebook's traces use the level: the extraction outputs (``iscell``, ``spks``)
and NATEX still work on F and Fneu as extracted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import tifffile

from .background_io import BackgroundTraces
from .config import PipelineSettings
from .photon_free import (
    ESTIMATOR,
    check_steps,
    dim_pixel_mask,
    photon_free_levels,
    subtract_level,
)
from .plane import PlaneTraces
from .tiff_io import chunk_bounds, count_timepoints, read_channel_chunk, replaced_when_complete

LEVELS_FILE = "photon_free_levels.npz"
# Raw-frame border left out of the dim-pixel masks (the resonant scan's turnarounds).
RAW_MARGIN_PX = 16
CACHE_VERSION = 2
_FIELDS = ("green", "red", "green_mask", "red_mask")  # PhotonFreeLevels order, as cached


@dataclass(frozen=True)
class PhotonFreeLevels:
    green: np.ndarray  # photon-free level per frame (counts), functional channel
    red: np.ndarray  # structural channel
    green_mask: np.ndarray  # raw-frame pixels each level was measured on
    red_mask: np.ndarray
    key: dict  # what the levels depend on (the cache key)


def levels_key(input_tiff: Path, n_frames: int, settings: PipelineSettings) -> dict:
    stat = Path(input_tiff).stat()
    fl, ch = settings.fluorescence, settings.channels
    return {
        "version": CACHE_VERSION,
        "tiff": str(Path(input_tiff).resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "n_frames": int(n_frames),
        "channels": [ch.n_channels, ch.functional_channel, ch.structural_channel],
        "percentile": fl.photon_free_pixel_percentile,
        "mask_frames": fl.photon_free_mask_frames,
        "margin_px": RAW_MARGIN_PX,
        "max_timepoints": settings.max_timepoints,
        "estimator": ESTIMATOR,
    }


def raw_levels(
    input_tiff: Path, plane_dir: Path, n_frames: int, settings: PipelineSettings
) -> PhotonFreeLevels:
    """Photon-free level of every frame of each channel of *input_tiff*.

    *n_frames* is the registered recording's frame count; a TIFF with another number of
    timepoints is refused, since the levels are subtracted frame for frame. Reading the
    TIFF takes minutes, so the levels are kept in ``photon_free_levels.npz`` in
    *plane_dir* and reused while the TIFF and settings are unchanged.
    """
    check_offset_kept(settings)
    key = levels_key(input_tiff, n_frames, settings)
    cache = Path(plane_dir) / LEVELS_FILE
    if cache.is_file():
        with np.load(cache) as saved:
            if str(saved["key"]) == json.dumps(key, sort_keys=True):
                return PhotonFreeLevels(*(saved[k] for k in _FIELDS), key)
    levels = _measure(Path(input_tiff), n_frames, settings)
    with replaced_when_complete(cache) as partial:
        with partial.open("wb") as handle:  # np.savez would add .npz to a .partial name
            np.savez(
                handle,
                key=json.dumps(key, sort_keys=True),
                **dict(zip(_FIELDS, levels, strict=True)),
            )
    return PhotonFreeLevels(*levels, key)


def check_offset_kept(settings: PipelineSettings) -> None:
    """Refuse settings under which the Suite2p input no longer carries the raw offset."""
    if settings.destripe.enabled and not settings.destripe.preserve_frame_mean:
        raise ValueError(
            "destripe.preserve_frame_mean = false already removes each column's mean, offset "
            "included; subtracting the photon-free level as well would remove it twice"
        )
    if settings.distortion.mode != "off" and not settings.distortion.crop_black_borders:
        raise ValueError(
            "distortion.crop_black_borders = false leaves zero-filled borders that carry no "
            "offset, so the level cannot be subtracted from every pixel alike"
        )


def _measure(input_tiff: Path, n_frames: int, settings: PipelineSettings) -> tuple:
    ch, fl = settings.channels, settings.fluorescence
    channels = (ch.functional_channel - 1, ch.structural_channel - 1)
    with tifffile.TiffFile(input_tiff) as tif:
        # the same first timepoints the pipeline processed (max_timepoints for test runs)
        n_timepoints = count_timepoints(tif, ch.n_channels, settings.max_timepoints)
        if n_timepoints != n_frames:
            raise ValueError(
                f"{input_tiff.name} holds {n_timepoints} timepoints but the registered "
                f"recording {n_frames} frames; the levels would not line up frame for frame"
            )
        # The masks come from the mean of the first frames, which are measured too; their
        # dimmest pixels are picked by structure (vessels, dark tissue), so how much a
        # frame's own noise sways the pick is negligible with hundreds of frames.
        masks = []
        for channel in channels:
            first = read_channel_chunk(
                tif, 0, min(fl.photon_free_mask_frames, n_timepoints), channel, ch.n_channels
            )
            region = (slice(RAW_MARGIN_PX, -RAW_MARGIN_PX), slice(RAW_MARGIN_PX, -RAW_MARGIN_PX))
            masks.append(
                dim_pixel_mask(first.mean(axis=0), fl.photon_free_pixel_percentile, region)
            )
        levels = [[], []]
        for start, end in chunk_bounds(n_timepoints, fl.background_chunk_frames):
            for i, channel in enumerate(channels):
                frames = read_channel_chunk(tif, start, end, channel, ch.n_channels)
                levels[i].append(photon_free_levels(frames, masks[i], first_frame=start))
    green, red = np.concatenate(levels[0]), np.concatenate(levels[1])
    check_steps(green)  # chunks were checked inside; this covers their boundaries
    check_steps(red)
    return green, red, masks[0], masks[1]


def traces_without_level(traces: PlaneTraces, levels: PhotonFreeLevels) -> PlaneTraces:
    """*traces* with each frame's photon-free level subtracted from F and Fneu of each
    channel (exact: F is a lam-weighted mean with weights summing to 1, Fneu a mean)."""
    _check_frames(traces.f.shape[1], levels)

    def without(rows: np.ndarray | None, level: np.ndarray) -> np.ndarray | None:
        return None if rows is None else subtract_level(rows, level).astype(np.float32)

    return replace(
        traces,
        f=without(traces.f, levels.green),
        fneu=without(traces.fneu, levels.green),
        f_chan2=without(traces.f_chan2, levels.red),
        fneu_chan2=without(traces.fneu_chan2, levels.red),
    )


def background_without_level(
    background: BackgroundTraces, levels: PhotonFreeLevels
) -> BackgroundTraces:
    """*background* with each frame's photon-free level subtracted in each channel."""
    _check_frames(len(background.green), levels)
    return replace(
        background,
        green=subtract_level(background.green, levels.green),
        red=subtract_level(background.red, levels.red),
    )


def _check_frames(n_frames: int, levels: PhotonFreeLevels) -> None:
    if len(levels.green) != n_frames or len(levels.red) != n_frames:
        raise ValueError(
            f"Photon-free levels cover {len(levels.green)} frames, the traces {n_frames}"
        )
