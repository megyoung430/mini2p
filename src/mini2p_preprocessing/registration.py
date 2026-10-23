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
    if suite2p_dir.exists():
        check_removable(suite2p_dir.parent, settings, f"Not replacing {suite2p_dir}")
        log.info("Removing earlier registration %s", suite2p_dir)
        shutil.rmtree(suite2p_dir)


def check_removable(alignment_folder: Path, settings: PipelineSettings, refusal: str) -> None:
    """Raise unless ``alignment_folder`` is a ``<variant>/aligned_<source>/`` folder whose
    contents may be removed: never ``run-XX/suite2p`` (the registration Cellpose masks are
    drawn on), never without ``overwrite``, never with Cellpose masks inside."""
    alignment_folder = Path(alignment_folder)
    in_variant = (
        alignment_folder.name in ALIGNMENT_SUBDIRS.values()
        and alignment_folder.parent.name in VARIANT_SUBDIRS
    )
    if not in_variant:
        raise RuntimeError(
            f"{refusal}: only registrations inside <variant>/"
            f"{' or '.join(ALIGNMENT_SUBDIRS.values())}/ are replaced; this may be the one "
            "Cellpose masks were drawn on"
        )
    if not settings.overwrite:
        raise FileExistsError(f"{refusal}: it exists; set overwrite=true to register again")
    masks = sorted(alignment_folder.rglob("*_seg.npy"))
    if masks:
        raise RuntimeError(
            f"{refusal}: it holds Cellpose masks {[m.name for m in masks]}; move them out first"
        )


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
    decides it from the TIFF's per-frame bidiphase record).

    A registration aligned on the other channel is kept only if it registered this same
    TIFF (same size and modification time); otherwise it is stale and is removed, under the
    same guards as re-registering.
    """
    suite2p_input_tiff = Path(suite2p_input_tiff)
    if not suite2p_input_tiff.is_file():
        raise FileNotFoundError(f"Suite2p input does not exist: {suite2p_input_tiff}")
    check_suite2p_input(suite2p_input_tiff, settings)
    choice = motion_choice(suite2p_input_tiff, settings)
    settings = with_motion_source(settings, choice.source)
    identity = input_identity(suite2p_input_tiff)
    registration_dir = alignment_dir(output_dir, choice.source)
    save_folder = settings.registration.save_folder
    check_existing_suite2p_folder(registration_dir, suite2p_input_tiff, save_folder)
    other = "functional" if choice.source == "structural" else "structural"
    other_dir = alignment_dir(output_dir, other)
    stale = other_dir.exists() and not same_input(read_alignment_record(other_dir), identity)
    # Every refusal comes before anything is removed, so a refused run changes nothing.
    if (registration_dir / save_folder).exists():
        check_removable(registration_dir, settings, f"Not replacing {registration_dir}")
    if stale:
        check_removable(
            other_dir, settings, f"Not removing {other_dir}, a registration of another TIFF"
        )
        log.warning("Removing %s and everything in it: it registered another TIFF", other_dir)
        shutil.rmtree(other_dir)
    elif other_dir.exists():
        log.info("Keeping %s: it registered the same TIFF", other_dir)
    clear_earlier_registration(registration_dir / save_folder, settings)
    # A run that fails must not leave the previous run's record describing it.
    (registration_dir / MOTION_ALIGNMENT_RECORD).unlink(missing_ok=True)
    registration_dir.mkdir(parents=True, exist_ok=True)
    db = build_suite2p_db(output_dir, suite2p_input_tiff, settings)
    log.info(
        "Suite2p aligns on the %s channel (%s); file_list: %s",
        choice.source,
        choice.reason,
        db["file_list"],
    )
    from suite2p.run_s2p import run_s2p

    run_s2p(settings=build_suite2p_settings(settings), db=db)
    write_alignment_record(registration_dir, alignment_record(choice, identity))
    return registration_dir / save_folder / "plane0"


@dataclass(frozen=True)
class MotionChoice:
    source: ChannelRole
    reason: str
    fraction_structural: float | None = None  # "auto" only
    threshold: float | None = None
    phase_record: str | None = None


def phase_record_path(suite2p_input_tiff: Path, settings: PipelineSettings) -> Path:
    """The ``.phase.json`` written beside the corrected TIFF that *suite2p_input_tiff* is
    (or, ``*_undistorted.tif``, was made from)."""
    tiff = Path(suite2p_input_tiff)
    stem = tiff.stem.removesuffix("_undistorted")
    if not stem.endswith(f"_{settings.correction_suffix}"):
        raise ValueError(
            f"{tiff.name} is not a corrected TIFF of these settings (*_"
            f"{settings.correction_suffix}[_undistorted].tif), so its phase record is unknown"
        )
    return tiff.with_name(f"{stem}.phase.json")


def motion_choice(suite2p_input_tiff: Path, settings: PipelineSettings) -> MotionChoice:
    """The channel to align on and why: the configured one, or for "auto" the choice from
    how often the per-frame bidiphase step trusted the structural channel in this TIFF."""
    source = settings.registration.motion_alignment_source
    if source != "auto":
        return MotionChoice(source, "set in the config")
    path = phase_record_path(suite2p_input_tiff, settings)
    if not path.is_file():
        raise FileNotFoundError(
            f'motion_alignment_source = "auto" needs {path.name} beside the TIFF; run the '
            'phase correction first, or set "structural" or "functional"'
        )
    record = json.loads(path.read_text())
    if record.get("phase_method") != "per_frame" or "fraction_structural" not in record:
        raise ValueError(
            f"{path} is not a per-frame bidiphase record with fraction_structural; rerun the "
            'phase correction with method = "per_frame"'
        )
    fraction = float(record["fraction_structural"])
    threshold = settings.registration.auto_structural_min_fraction
    reason = (
        f"auto: the bidiphase step trusted the structural channel for {fraction:.0%} of "
        f"frames, threshold {threshold:.0%}"
    )
    return MotionChoice(
        choose_motion_source(fraction, threshold), reason, fraction, threshold, path.name
    )


def input_identity(tiff: Path) -> dict:
    stat = Path(tiff).stat()
    return {"suite2p_input": str(tiff), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


# The file name, not its full path: a session folder moved or copied (share <-> laptop)
# keeps its registrations valid.
SAME_INPUT_KEYS = ("size_bytes", "mtime_ns")


def alignment_record(choice: MotionChoice, identity: dict) -> dict:
    return {
        "motion_alignment_source": choice.source,
        "reason": choice.reason,
        "fraction_structural": choice.fraction_structural,
        "auto_structural_min_fraction": choice.threshold,
        "phase_record": choice.phase_record,
        **identity,
    }


def write_alignment_record(registration_dir: Path, record: dict) -> None:
    path = Path(registration_dir) / MOTION_ALIGNMENT_RECORD
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(record, indent=2))
    partial.replace(path)


def read_alignment_record(registration_dir: Path) -> dict | None:
    path = Path(registration_dir) / MOTION_ALIGNMENT_RECORD
    return json.loads(path.read_text()) if path.is_file() else None


def same_input(record: dict | None, identity: dict) -> bool:
    """Whether a registration's record shows it registered the TIFF *identity* describes:
    same file name, size and modification time."""
    if record is None or "suite2p_input" not in record:
        return False
    same_name = Path(record["suite2p_input"]).name == Path(identity["suite2p_input"]).name
    return same_name and all(record.get(k) == identity[k] for k in SAME_INPUT_KEYS)


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
