"""Fixed-calibration distortion correction applied before Suite2p registration.

The calibration maths (``load_transform``, ``make_maps``, ``crop_bounds``) lives in
``distortion_correct_full_recording.py``, shared with the SLURM worker, and is
loaded from ``DistortionSettings.calibration_script_dir``.
"""

from __future__ import annotations

import gc
import importlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import tifffile
from scipy.ndimage import map_coordinates

from .config import DistortionSettings, PipelineSettings
from .progress import PROGRESS_EVERY_S, ProgressLog
from .tiff_io import (
    check_clipped,
    chunk_bounds,
    count_timepoints,
    n_written_values,
    output_record,
    read_all_channels,
    write_interleaved,
)

log = logging.getLogger(__name__)
CALIBRATION_SCRIPT = "distortion_correct_full_recording.py"


@dataclass(frozen=True)
class DistortionMaps:
    """One full-frame spatial map and crop, fixed for the whole recording."""

    map_y: np.ndarray
    map_x: np.ndarray
    crop: tuple[int, int, int, int]  # top, bottom, left, right
    image_shape: tuple[int, int]
    matched_z_um: float
    calibration_schema: str
    control_points: int


# ---- analysis ---------------------------------------------------------------------


def apply_distortion_to_chunk(chunk: np.ndarray, maps: DistortionMaps, order: int) -> np.ndarray:
    """Warp every frame of (frames, Ly, Lx) with the fixed map, then crop."""
    top, bottom, left, right = maps.crop
    output = np.empty((len(chunk), bottom - top, right - left), dtype=np.float32)
    for frame_index, frame in enumerate(chunk):
        warped = map_coordinates(
            frame.astype(np.float32),
            [maps.map_y, maps.map_x],
            order=int(order),
            mode="constant",
            cval=0,
            prefilter=int(order) > 1,
        )
        output[frame_index] = warped[top:bottom, left:right]
    return output


def distortion_metadata(
    input_tiff: Path,
    output_tiff: Path,
    maps: DistortionMaps,
    distortion: DistortionSettings,
    n_timepoints: int,
) -> dict:
    top, bottom, left, right = maps.crop
    return {
        "input_tiff": str(input_tiff),
        "output_tiff": str(output_tiff),
        "mode": "local_batched_before_registration",
        "batch_frames": distortion.batch_frames,
        "transform_reestimated_per_batch": False,
        "transform_mat": str(distortion.transform_mat),
        "requested_z_um": float(distortion.z_um),
        "matched_z_um": maps.matched_z_um,
        "calibration_schema": maps.calibration_schema,
        "control_points": int(maps.control_points),
        "interpolation_order": distortion.interpolation_order,
        "crop_bounds_top_bottom_left_right": [int(v) for v in maps.crop],
        "input_shape_yx": [int(v) for v in maps.image_shape],
        "output_shape_yx": [bottom - top, right - left],
        "n_timepoints": int(n_timepoints),
        "applied_before": "suite2p_registration",
    }


# ---- I/O --------------------------------------------------------------------------


def load_calibration_module(script_dir: Path | None) -> ModuleType:
    """Import the shared calibration script; None searches ./scripts then ./.

    The folder goes on sys.path and the script is imported by name, so it can
    import its own sibling modules exactly as it does in the SLURM worker.
    """
    candidates = [Path(script_dir)] if script_dir else [Path.cwd() / "scripts", Path.cwd()]
    module_name = Path(CALIBRATION_SCRIPT).stem
    for folder in candidates:
        script = folder / CALIBRATION_SCRIPT
        if script.is_file():
            cached = sys.modules.get(module_name)
            if cached is not None and Path(cached.__file__).resolve() != script.resolve():
                del sys.modules[module_name]  # a different folder was loaded earlier
            if str(folder) not in sys.path:
                sys.path.insert(0, str(folder))
            return importlib.import_module(module_name)
    raise FileNotFoundError(
        f"Could not find {CALIBRATION_SCRIPT} in {[str(c) for c in candidates]}; "
        "set distortion.calibration_script_dir"
    )


def build_distortion_maps(
    distortion: DistortionSettings,
    image_shape: tuple[int, int],
    calibration: ModuleType | None = None,
) -> DistortionMaps:
    """Load the calibration and build the map; pass ``calibration`` to skip the import."""
    if distortion.transform_mat is None:
        raise ValueError("distortion.transform_mat is not set")
    calibration = calibration or load_calibration_module(distortion.calibration_script_dir)
    transform, matched_z, schema, point_count = calibration.load_transform(
        distortion.transform_mat, distortion.z_um
    )
    map_y, map_x = calibration.make_maps(transform, image_shape)
    crop = (
        calibration.crop_bounds(map_y, map_x, image_shape, distortion.crop_padding_px)
        if distortion.crop_black_borders
        else (0, image_shape[0], 0, image_shape[1])
    )
    return DistortionMaps(
        map_y,
        map_x,
        tuple(int(v) for v in crop),
        tuple(image_shape),
        matched_z,
        schema,
        point_count,
    )


def write_distortion_corrected(
    input_tiff: Path,
    output_tiff: Path,
    settings: PipelineSettings,
    calibration: ModuleType | None = None,
) -> DistortionMaps:
    """Write the distortion-corrected TIFF and its .distortion.json sidecar."""
    output_tiff = Path(output_tiff)
    if output_tiff.exists() and not settings.overwrite:
        raise FileExistsError(f"{output_tiff} exists; set overwrite=true to replace it")
    distortion = settings.distortion
    n_channels = settings.channels.n_channels
    with tifffile.TiffFile(input_tiff) as tif:
        n_timepoints = count_timepoints(tif, n_channels, settings.max_timepoints)
        maps = build_distortion_maps(distortion, tuple(tif.pages[0].shape), calibration)
        with tifffile.TiffWriter(output_tiff, bigtiff=True) as writer:
            bounds = chunk_bounds(n_timepoints, distortion.batch_frames)
            progress = ProgressLog("Distortion correction", len(bounds), log.info, PROGRESS_EVERY_S)
            clipped = 0
            for start, end in bounds:
                raw = read_all_channels(tif, start, end, n_channels)
                warped = [
                    apply_distortion_to_chunk(c, maps, distortion.interpolation_order) for c in raw
                ]
                clipped += write_interleaved(writer, warped, settings.dtype)
                del raw, warped
                gc.collect()
                progress.step_done()
    top, bottom, left, right = maps.crop
    n_values = n_written_values(n_timepoints, n_channels, (bottom - top, right - left))
    check_clipped(clipped, n_values, output_tiff, settings.dtype, settings.max_clipped_fraction)
    sidecar = output_tiff.with_suffix(".distortion.json")
    metadata = distortion_metadata(
        input_tiff, output_tiff, maps, distortion, n_timepoints
    ) | output_record(settings.output_dtype, clipped)
    sidecar.write_text(json.dumps(metadata, indent=2))
    log.info("Saved %s and %s", output_tiff, sidecar)
    return maps
