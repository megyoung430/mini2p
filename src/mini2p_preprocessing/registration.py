"""Suite2p motion registration of the corrected TIFF, plus loaders for its outputs."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import suite2p
import tifffile

from .config import (
    ALIGNMENT_SUBDIRS,
    VARIANT_SUBDIRS,
    ChannelRole,
    PipelineSettings,
    alignment_dir,
    run_dir_of,
)

log = logging.getLogger(__name__)

MOTION_ALIGNMENT_RECORD = "motion_alignment.json"  # in aligned_<source>/: why that channel


# ---- analysis: build Suite2p inputs (pure apart from Suite2p's defaults) ----------


def choose_motion_source(fraction_structural: float, min_fraction: float) -> ChannelRole:
    """ "structural" if the bidiphase step trusted it for at least *min_fraction* of frames."""
    if not 0 <= fraction_structural <= 1:
        raise ValueError(f"fraction_structural must be in [0, 1], got {fraction_structural}")
    return "structural" if fraction_structural >= min_fraction else "functional"


def with_motion_source(settings: PipelineSettings, source: ChannelRole) -> PipelineSettings:
    registration = settings.registration.model_copy(update={"motion_alignment_source": source})
    return settings.model_copy(update={"registration": registration})


def resolved_channel(settings: PipelineSettings) -> ChannelRole:
    source = settings.registration.motion_alignment_source
    if source == "auto":
        raise ValueError('motion_alignment_source "auto" must be resolved before registration')
    return source


def build_suite2p_settings(settings: PipelineSettings) -> dict:
    """Suite2p >= 1.0 nested settings for registration-only (or full) runs."""
    resolved_channel(settings)
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
    registration_dir = alignment_dir(output_dir, resolved_channel(settings))
    db = {
        "data_path": [str(output_dir)],
        "file_list": [Path(suite2p_input_tiff).name],
        "look_one_level_down": False,
        "save_path0": str(registration_dir),
        "save_folder": settings.registration.save_folder,
        "nplanes": 1,
        "nchannels": settings.channels.n_channels,
        "functional_chan": settings.channels.functional_channel,
        "keep_movie_raw": settings.registration.keep_raw_binary,
    }
    if settings.registration.fast_disk is not None:
        db["fast_disk"] = str(fast_disk_for(registration_dir, settings.registration.fast_disk))
    return db


def fast_disk_for(registration_dir: Path, fast_disk: Path) -> Path:
    """This registration's folder under *fast_disk*, named after its session, run, variant
    and alignment (``registration_dir`` is ``<variant>/aligned_<source>``)."""
    registration_dir = Path(registration_dir)
    run_dir = run_dir_of(registration_dir.parent)
    return Path(fast_disk) / "_".join(
        run_dir.parts[-3:] + registration_dir.relative_to(run_dir).parts
    )


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
    result. Only a folder inside ``<variant>/aligned_<source>/`` (the variant a
    with_/without_destriping folder) is removed, never ``run-XX/suite2p``; refuses without
    ``overwrite``, or if Cellpose masks are inside.
    """
    suite2p_dir = Path(suite2p_dir)
    if not suite2p_dir.exists():
        return
    in_alignment = suite2p_dir.parent.name in ALIGNMENT_SUBDIRS.values()
    if not (in_alignment and suite2p_dir.parents[1].name in VARIANT_SUBDIRS):
        raise RuntimeError(
            f"Not replacing {suite2p_dir}: only registrations inside <variant>/"
            f"{' or '.join(ALIGNMENT_SUBDIRS.values())}/ are replaced; this may be the one "
            "Cellpose masks were drawn on"
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


def check_suite2p_input(suite2p_input_tiff: Path, settings: PipelineSettings) -> None:
    """Suite2p registers int16 and converts anything else with a plain astype(int16), which
    truncates towards zero and wraps values beyond the int16 range.

    With ``output_dtype = "int16"`` a different input can only be a stale file or one
    written elsewhere (a cluster distortion script), so it is refused; with ``"float32"``,
    asked for deliberately, it is only reported.
    """
    with tifffile.TiffFile(suite2p_input_tiff) as tif:
        dtype = tif.pages[0].dtype
    if dtype == np.int16:
        return
    message = (
        f"Suite2p input {suite2p_input_tiff} is {dtype}, not int16: Suite2p would convert it "
        "with astype(int16), truncating towards zero and wrapping values beyond the int16 range"
    )
    if settings.output_dtype == "int16":
        raise ValueError(
            f'{message}. Rewrite it with the pipeline, or set output_dtype = "float32" to '
            "accept that"
        )
    log.warning(message)


def run_registration(
    suite2p_input_tiff: Path, output_dir: Path, settings: PipelineSettings
) -> Path:
    """Run Suite2p on one TIFF in ``output_dir``; returns the plane0 folder, in
    ``output_dir/aligned_<source>/``, *source* being the channel it aligned on ("auto"
    decides it from this variant's per-frame bidiphase record)."""
    suite2p_input_tiff = Path(suite2p_input_tiff)
    if not suite2p_input_tiff.is_file():
        raise FileNotFoundError(f"Suite2p input does not exist: {suite2p_input_tiff}")
    check_suite2p_input(suite2p_input_tiff, settings)
    source, why = motion_source_for(output_dir, settings)
    settings = with_motion_source(settings, source)
    registration_dir = alignment_dir(output_dir, source)
    save_folder = settings.registration.save_folder
    check_existing_suite2p_folder(registration_dir, suite2p_input_tiff, save_folder)
    clear_earlier_registration(registration_dir / save_folder, settings)
    registration_dir.mkdir(parents=True, exist_ok=True)
    db = build_suite2p_db(output_dir, suite2p_input_tiff, settings)
    log.info("Suite2p aligns on the %s channel (%s); file_list: %s", source, why, db["file_list"])
    from suite2p.run_s2p import run_s2p

    run_s2p(settings=build_suite2p_settings(settings), db=db)
    record = {
        "motion_alignment_source": source,
        "reason": why,
        "suite2p_input": str(suite2p_input_tiff),
    }
    (registration_dir / MOTION_ALIGNMENT_RECORD).write_text(json.dumps(record, indent=2))
    return registration_dir / save_folder / "plane0"


def phase_record(output_dir: Path, settings: PipelineSettings) -> tuple[Path, dict]:
    """The per-frame bidiphase sidecar (``<stem>_<correction suffix>.phase.json``) in
    *output_dir*; there must be exactly one."""
    found = sorted(Path(output_dir).glob(f"*_{settings.correction_suffix}.phase.json"))
    if len(found) != 1:
        raise FileNotFoundError(
            f'motion_alignment_source = "auto" needs one *_{settings.correction_suffix}'
            f".phase.json in {output_dir}, found {[p.name for p in found]}; run the phase "
            'correction first, or set "structural" or "functional"'
        )
    return found[0], json.loads(found[0].read_text())


def motion_source_for(output_dir: Path, settings: PipelineSettings) -> tuple[ChannelRole, str]:
    """The channel to align on and why: the configured one, or for "auto" the choice from
    how often the per-frame bidiphase step trusted the structural channel."""
    source = settings.registration.motion_alignment_source
    if source != "auto":
        return source, "set in the config"
    path, record = phase_record(output_dir, settings)
    if record.get("phase_method") != "per_frame" or "fraction_structural" not in record:
        raise ValueError(
            f"{path} is not a per-frame bidiphase record with fraction_structural; rerun the "
            'phase correction with method = "per_frame"'
        )
    fraction = float(record["fraction_structural"])
    threshold = settings.registration.auto_structural_min_fraction
    chosen = choose_motion_source(fraction, threshold)
    return chosen, (
        f"auto: the bidiphase step trusted the structural channel for {fraction:.0%} of frames, "
        f"threshold {threshold:.0%} ({path.name})"
    )


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
