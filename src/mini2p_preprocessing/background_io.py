"""Files for the background: the vessel mask on this plane or the pixels outside the ROIs,
their cached traces from the registered binaries, and the background-corrected TIFF (see
``background``)."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import tifffile

from .background import (
    centred_regressor,
    corrected_chunk,
    mask_mean_trace,
    non_roi_mask,
    pixel_coefficients,
)
from .background import vessel_mask as build_vessel_mask
from .config import PipelineSettings
from .extraction import align_to_plane, file_sha256, load_cellpose_labels, registered_binaries
from .qc import unwrapped_region
from .registration import RegisteredOutputs
from .tiff_io import (
    check_clipped,
    chunk_bounds,
    n_written_values,
    output_record,
    replaced_when_complete,
    write_interleaved,
)

log = logging.getLogger(__name__)

BackgroundSource = Literal["vessel", "non_roi"]
SOURCE_LABELS = {"vessel": "Vessel", "non_roi": "Non-ROI"}


def traces_cache(source: BackgroundSource) -> str:
    """Cached background traces of one source, in the plane folder."""
    return f"background_{source}_traces.npz"


def background_tiff(source: BackgroundSource) -> str:
    return f"data_background_removed_{source}.tif"


def background_record(source: BackgroundSource) -> str:
    return f"data_background_removed_{source}.json"


@dataclass(frozen=True)
class BackgroundTraces:
    source: BackgroundSource
    green: np.ndarray  # mean background fluorescence per frame, every frame of the session
    red: np.ndarray
    mask: np.ndarray  # the background pixels used, on this plane
    alignment: dict | None  # the shift onto this plane, when drawn on another image
    key: dict  # what the traces depend on (the cache key)


def binaries_identity(ops: dict, plane_dir: Path) -> dict:
    """Size and modification time of each registered binary, so a registration replaced in
    any way (not only by run_registration) makes what was computed from it stale."""
    green, red = registered_binaries(ops, Path(plane_dir))
    return {
        name: {
            "file": path.name,
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for name, path in (("green", green), ("red", red))
        if path is not None
    }


def drawn_on(seg_file: Path) -> Path | None:
    """The image a Cellpose segmentation was drawn on, as Cellpose recorded it."""
    name = np.load(seg_file, allow_pickle=True).item().get("filename")
    return Path(name) if name else None


def vessel_labels_on_plane(
    seg_file: Path,
    ops: dict,
    plane_dir: Path,
    settings: PipelineSettings,
    image: Path | None = None,
) -> tuple[np.ndarray, dict | None]:
    """The vessel labels on this plane: shifted onto its mean image when *image* (the image
    they were drawn on) is given, as for the ROI masks; otherwise they must have been drawn
    on an image in this plane folder, which the segmentation's own record is checked for."""
    labels = load_cellpose_labels(seg_file)
    if image is not None:
        return align_to_plane(labels, image, ops, settings)
    source = drawn_on(seg_file)
    if source is None:
        log.warning("%s does not record its image; assuming it is this plane's", seg_file)
    elif source.parent.resolve() != Path(plane_dir).resolve():
        raise ValueError(
            f"{Path(seg_file).name} was drawn on {source}, not on an image of this plane "
            f"({plane_dir}); set [qc] background_image to that image so the vessel mask is "
            "shifted onto this plane, as the ROI masks are"
        )
    return labels, None


def mask_sha256(mask: np.ndarray) -> str:
    return hashlib.sha256(np.packbits(np.asarray(mask, bool))).hexdigest()


def vessel_cache_key(
    seg_file: Path,
    ops: dict,
    plane_dir: Path,
    image: Path | None,
    exclude: np.ndarray | None,
) -> dict:
    """What the vessel traces depend on: the segmentation, the image it was drawn on, the
    pixels left out, and the registered binaries."""
    return {
        "source": "vessel",
        "seg_file": str(Path(seg_file).resolve()),
        "seg_sha256": file_sha256(seg_file),
        "image_sha256": file_sha256(image) if image is not None else None,
        "excluded_sha256": mask_sha256(exclude) if exclude is not None else None,
        "shape": [int(ops["nframes"]), int(ops["Ly"]), int(ops["Lx"])],
        "binaries": binaries_identity(ops, plane_dir),
    }


def cached_mask_traces(
    outputs: RegisteredOutputs,
    plane_dir: Path,
    source: BackgroundSource,
    mask: np.ndarray,
    key: dict,
    settings: PipelineSettings,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean fluorescence of *mask* per frame in each channel (green, red), over every frame.

    Reading both binaries takes minutes, so the traces are kept in the source's cache file
    and reused while *key* is unchanged.
    """
    cache = Path(plane_dir) / traces_cache(source)
    if cache.is_file():
        with np.load(cache) as saved:
            if str(saved["key"]) == json.dumps(key, sort_keys=True):
                return saved["green"], saved["red"]
    chunk = settings.fluorescence.background_chunk_frames
    green = mask_mean_trace(outputs.functional, mask, chunk)
    red = mask_mean_trace(outputs.structural, mask, chunk)
    with replaced_when_complete(cache) as partial:
        with partial.open("wb") as handle:  # np.savez would add .npz to a .partial name
            np.savez(handle, key=json.dumps(key, sort_keys=True), green=green, red=red)
    return green, red


def load_vessel_traces(
    outputs: RegisteredOutputs,
    plane_dir: Path,
    seg_file: Path,
    settings: PipelineSettings,
    image: Path | None = None,
    exclude: np.ndarray | None = None,
) -> BackgroundTraces:
    """The background from a blood-vessel segmentation; *exclude* (e.g. the ROI pixels) is
    left out of the vessel mask."""
    labels, alignment = vessel_labels_on_plane(seg_file, outputs.ops, plane_dir, settings, image)
    mask = build_vessel_mask(labels, outputs.functional.shape[1:], exclude)
    if exclude is not None and (overlap := int(((labels > 0) & exclude).sum())):
        log.info("Left %d ROI pixels out of the vessel mask", overlap)
    key = vessel_cache_key(seg_file, outputs.ops, plane_dir, image, exclude)
    green, red = cached_mask_traces(outputs, plane_dir, "vessel", mask, key, settings)
    return BackgroundTraces("vessel", green, red, mask, alignment, key)


def load_non_roi_traces(
    outputs: RegisteredOutputs, plane_dir: Path, roi_labels: np.ndarray, settings: PipelineSettings
) -> BackgroundTraces:
    """The background from every pixel outside the cell ROIs, within the region no frame's
    registration shifts wrap into."""
    rows, cols = unwrapped_region(outputs.ops)
    mask = non_roi_mask(roi_labels, (rows, cols))
    key = {
        "source": "non_roi",
        "rois_sha256": mask_sha256(np.asarray(roi_labels) > 0),
        "region": [rows.start, rows.stop, cols.start, cols.stop],
        "shape": [int(outputs.ops["nframes"]), int(outputs.ops["Ly"]), int(outputs.ops["Lx"])],
        "binaries": binaries_identity(outputs.ops, plane_dir),
    }
    green, red = cached_mask_traces(outputs, plane_dir, "non_roi", mask, key, settings)
    return BackgroundTraces("non_roi", green, red, mask, None, key)


def write_background_tiff(
    outputs: RegisteredOutputs,
    background: BackgroundTraces,
    plane_dir: Path,
    settings: PipelineSettings,
) -> Path:
    """``data_background_removed_<source>.tif``: the registered movie with each channel's
    background trace regressed out of every pixel (its own coefficient per pixel, fitted on
    every frame of the session), interleaved in the acquired channel order, int16; and a
    ``.json`` record of the background it was made from. Needs disk space for a second copy
    while it is written (``.partial``)."""
    channels = settings.channels
    chunk = settings.fluorescence.background_chunk_frames
    by_channel = {
        channels.functional_channel: ("green", outputs.functional, background.green),
        channels.structural_channel: ("red", outputs.structural, background.red),
    }
    order = [by_channel[ch] for ch in sorted(by_channel)]
    centred = [centred_regressor(trace, len(stack)) for _, stack, trace in order]
    coefs = [pixel_coefficients(stack, trace, chunk) for _, stack, trace in order]
    path = Path(plane_dir) / background_tiff(background.source)
    record_path = Path(plane_dir) / background_record(background.source)
    record_path.unlink(missing_ok=True)  # never beside a TIFF it does not describe
    n_frames, frame_shape = len(outputs.functional), outputs.functional.shape[1:]
    with replaced_when_complete(path) as partial:
        clipped = 0
        with tifffile.TiffWriter(partial, bigtiff=True) as writer:
            for start, end in chunk_bounds(n_frames, chunk):
                chunks = [
                    corrected_chunk(stack, start, end, coef, c)
                    for (_, stack, _), coef, c in zip(order, coefs, centred, strict=True)
                ]
                clipped += write_interleaved(writer, chunks, np.dtype(np.int16))
        n_values = n_written_values(n_frames, len(order), frame_shape)
        check_clipped(clipped, n_values, path, np.int16, settings.max_clipped_fraction)
    record = {
        "background": background.key,
        "background_alignment": background.alignment,
        "method": "per-pixel least squares of the background trace over every frame, means kept",
        "channel_order_one_based": sorted(by_channel),
        "coefficient_median": {
            role: float(np.median(coef)) for (role, _, _), coef in zip(order, coefs, strict=True)
        },
        **output_record("int16", clipped),
    }
    with replaced_when_complete(record_path) as partial:
        partial.write_text(json.dumps(record, indent=2))
    return path


def check_background_tiff(
    plane_dir: Path, ops: dict, source: BackgroundSource, key: dict | None = None
) -> Path:
    """``data_background_removed_<source>.tif`` of this plane, refused if it is missing,
    was made from other registered binaries than the plane's current ones, or (given *key*)
    from another background (vessel mask, image, ROI set)."""
    path = Path(plane_dir) / background_tiff(source)
    record_path = Path(plane_dir) / background_record(source)
    if not path.is_file() or not record_path.is_file():
        raise FileNotFoundError(
            f"No {path} with its {record_path.name}: write it with the background step of "
            "the QC notebook first"
        )
    made_from = json.loads(record_path.read_text())["background"]
    if made_from["binaries"] != binaries_identity(ops, plane_dir):
        raise ValueError(
            f"{path} was made from another registration of this plane (or the plane folder "
            "was copied without keeping file times); write it again with the background step "
            "of the QC notebook"
        )
    if key is not None and made_from != key:
        raise ValueError(
            f"{path} was made with another background (vessel segmentation, image or ROI "
            "set); write it again with the background step of the QC notebook"
        )
    return path
