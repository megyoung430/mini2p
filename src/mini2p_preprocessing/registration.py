"""Suite2p motion registration of the corrected TIFF, plus loaders for its outputs."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import suite2p
import tifffile

from .config import VARIANT_SUBDIRS, PipelineSettings

log = logging.getLogger(__name__)


# ---- analysis: build Suite2p inputs (pure apart from Suite2p's defaults) ----------


def build_suite2p_settings(settings: PipelineSettings) -> dict:
    """Suite2p >= 1.0 nested settings for registration-only (or full) runs."""
    reg = settings.registration
    s2p = suite2p.default_settings()
    s2p["fs"] = reg.fs_hz
    s2p["diameter"] = reg.diameter_px
    s2p["registration"].update(
        {
            # True aligns on Suite2p's second (non-functional) binary.
            "align_by_chan2": settings.registration_channel != settings.channels.functional_channel,
            # Scan phase is already corrected chunk by chunk before registration.
            "do_bidiphase": False,
            "bidiphase": 0,
            "nonrigid": reg.nonrigid,
            "norm_frames": reg.norm_frames,
            "nimg_init": reg.nimg_init,
            "th_badframes": reg.th_badframes,
            "two_step_registration": reg.two_step_registration,
            "batch_size": reg.batch_size,
        }
    )
    s2p["run"].update(
        {
            "do_registration": 1,
            "do_detection": reg.run_roi_detection,
            "do_deconvolution": reg.run_roi_detection,
        }
    )
    s2p["io"]["save_mat"] = False
    return s2p


def build_suite2p_db(
    output_dir: Path, suite2p_input_tiff: Path, settings: PipelineSettings
) -> dict:
    # An explicit file_list stops Suite2p from also picking up the other TIFFs in
    # output_dir, e.g. the uncropped corrected TIFF beside the cropped undistorted one.
    return {
        "data_path": [str(output_dir)],
        "file_list": [Path(suite2p_input_tiff).name],
        "look_one_level_down": False,
        "save_path0": str(output_dir),
        "save_folder": settings.registration.save_folder,
        "nplanes": 1,
        "nchannels": settings.channels.n_channels,
        "functional_chan": settings.channels.functional_channel,
        "keep_movie_raw": settings.registration.keep_raw_binary,
    }


# ---- I/O --------------------------------------------------------------------------


def check_existing_suite2p_folder(
    output_dir: Path, suite2p_input_tiff: Path, save_folder: str
) -> None:
    """Refuse to reuse a Suite2p folder made from different files or a different shape."""
    suite2p_dir = Path(output_dir) / save_folder
    with tifffile.TiffFile(suite2p_input_tiff) as tif:
        page_shape = tuple(tif.pages[0].shape)
    db_path = suite2p_dir / "db.npy"
    if db_path.is_file():
        existing = np.load(db_path, allow_pickle=True).item()
        files = [Path(p).name for p in existing.get("file_list", [])]
        if files != [Path(suite2p_input_tiff).name]:
            raise RuntimeError(
                f"{suite2p_dir} was created from a different or mixed file list {files}; "
                "move it before rerunning"
            )
    ops_path = suite2p_dir / "plane0" / "ops.npy"
    if ops_path.is_file():
        ops = np.load(ops_path, allow_pickle=True).item()
        existing_shape = (int(ops["Ly"]), int(ops["Lx"]))
        if existing_shape != page_shape:
            raise RuntimeError(
                f"Existing Suite2p shape {existing_shape} does not match {page_shape}; "
                f"move {suite2p_dir} before rerunning"
            )


def clear_earlier_registration(suite2p_dir: Path, settings: PipelineSettings) -> None:
    """Remove an earlier registration of the same file so Suite2p registers afresh.

    Suite2p reuses an existing binary and skips registration, which would keep the old
    result. Only a folder inside a with_/without_destriping subfolder is removed, never
    ``run-XX/suite2p``; refuses without ``overwrite``, or if Cellpose masks are inside.
    """
    suite2p_dir = Path(suite2p_dir)
    if not suite2p_dir.exists():
        return
    if suite2p_dir.parent.name not in VARIANT_SUBDIRS:
        raise RuntimeError(
            f"Not replacing {suite2p_dir}: only registrations inside a variant subfolder "
            f"{VARIANT_SUBDIRS} are replaced; this may be the one Cellpose masks were drawn on"
        )
    if not settings.overwrite:
        raise FileExistsError(f"{suite2p_dir} exists; set overwrite=true to register again")
    masks = sorted(suite2p_dir.rglob("*_seg.npy"))
    if masks:
        raise RuntimeError(
            f"Not replacing {suite2p_dir}: it holds Cellpose masks {[m.name for m in masks]}; "
            "move them out first"
        )
    log.info("Removing earlier registration %s", suite2p_dir)
    shutil.rmtree(suite2p_dir)


def run_registration(
    suite2p_input_tiff: Path, output_dir: Path, settings: PipelineSettings
) -> Path:
    """Run Suite2p on one TIFF inside ``output_dir``; returns the plane0 folder."""
    suite2p_input_tiff = Path(suite2p_input_tiff)
    if not suite2p_input_tiff.is_file():
        raise FileNotFoundError(f"Suite2p input does not exist: {suite2p_input_tiff}")
    check_existing_suite2p_folder(output_dir, suite2p_input_tiff, settings.registration.save_folder)
    clear_earlier_registration(Path(output_dir) / settings.registration.save_folder, settings)
    db = build_suite2p_db(output_dir, suite2p_input_tiff, settings)
    log.info("Suite2p file_list: %s", db["file_list"])
    from suite2p.run_s2p import run_s2p

    run_s2p(settings=build_suite2p_settings(settings), db=db)
    return Path(output_dir) / settings.registration.save_folder / "plane0"


@dataclass(frozen=True)
class RegisteredOutputs:
    ops: dict
    functional: np.memmap  # Suite2p's data.bin
    structural: np.memmap  # Suite2p's data_chan2.bin


def resolve_registered_binary(ops: dict, plane_dir: Path, ops_key: str, filename: str) -> Path:
    # Binaries can sit outside save_folder (fast_disk, reused conversions), so the
    # path recorded in ops is tried first, then the default suite2p/ folder.
    plane_dir = Path(plane_dir)
    candidates = [Path(ops[ops_key])] if ops.get(ops_key) else []
    candidates += [plane_dir / filename, plane_dir.parents[1] / "suite2p" / "plane0" / filename]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find {filename}; checked {[str(c) for c in candidates]}")


def load_registered_outputs(plane_dir: Path) -> RegisteredOutputs:
    plane_dir = Path(plane_dir)
    ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
    shape = (int(ops["nframes"]), int(ops["Ly"]), int(ops["Lx"]))
    functional = resolve_registered_binary(ops, plane_dir, "reg_file", "data.bin")
    structural = resolve_registered_binary(ops, plane_dir, "reg_file_chan2", "data_chan2.bin")
    return RegisteredOutputs(
        ops=ops,
        functional=np.memmap(functional, dtype=np.int16, mode="r", shape=shape),
        structural=np.memmap(structural, dtype=np.int16, mode="r", shape=shape),
    )
