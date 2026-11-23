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
    DEFAULT_SAVE_FOLDER,
    VARIANT_SUBDIRS,
    ChannelRole,
    PipelineSettings,
    alignment_dir,
    run_dir_of,
)

log = logging.getLogger(__name__)
MOTION_ALIGNMENT_RECORD = "motion_alignment.json"  # in <save_folder>/: why that channel


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
            "smooth_sigma": reg.smooth_sigma,
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
    save_folder = settings.registration.save_folder
    if settings.registration.fast_disk is not None:
        db["fast_disk"] = str(
            fast_disk_for(registration_dir, settings.registration.fast_disk, save_folder)
        )
    elif save_folder != DEFAULT_SAVE_FOLDER:
        # Suite2p writes its binaries in <fast_disk>/suite2p/plane0 whatever save_folder is
        # (fast_disk defaulting to save_path0), so another save_folder would overwrite the
        # default registration's data.bin. Here they go inside this registration's folder,
        # <save_folder>/suite2p/plane0 (run_registration then moves them into its plane0),
        # and are removed with it when it is registered again.
        db["fast_disk"] = str(registration_dir / save_folder)
    return db


def fast_disk_for(
    registration_dir: Path, fast_disk: Path, save_folder: str = DEFAULT_SAVE_FOLDER
) -> Path:
    """This registration's folder under *fast_disk*, named after its cohort, subject, session,
    run, variant, alignment (``registration_dir`` is ``<variant>/aligned_<source>``) and,
    other than the default, its save_folder."""
    registration_dir = Path(registration_dir)
    run_dir = run_dir_of(registration_dir.parent)
    parts = run_dir.parts[-5:] + registration_dir.relative_to(run_dir).parts
    if save_folder != DEFAULT_SAVE_FOLDER:
        parts += (save_folder,)
    return Path(fast_disk) / "_".join(parts)


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
    ``overwrite``, or if Cellpose masks are anywhere in that ``aligned_<source>/`` folder.
    """
    suite2p_dir = Path(suite2p_dir)
    if suite2p_dir.exists():
        check_removable(suite2p_dir.parent, settings, f"Not replacing {suite2p_dir}")
        log.info("Removing earlier registration %s", suite2p_dir)
        shutil.rmtree(suite2p_dir)


def check_removable(
    alignment_folder: Path, settings: PipelineSettings, refusal: str, action: str = "register again"
) -> None:
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
        raise FileExistsError(f"{refusal}: it exists; set overwrite=true to {action}")
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
    stale = other_dir.exists() and not any(
        same_input(record, identity) for record in alignment_records(other_dir)
    )
    # Every refusal comes before anything is removed, so a refused run changes nothing.
    if (registration_dir / save_folder).exists():
        check_removable(registration_dir, settings, f"Not replacing {registration_dir}")
    if stale:
        check_removable(
            other_dir,
            settings,
            f"Not removing {other_dir}, a registration of another TIFF",
            action="remove it",
        )
        log.warning("Removing %s and everything in it: it registered another TIFF", other_dir)
        shutil.rmtree(other_dir)
    elif other_dir.exists():
        log.info("Keeping %s: it registered the same TIFF", other_dir)
    # Removes this registration's earlier record too: a run that fails must not leave the
    # previous run's record describing it.
    clear_earlier_registration(registration_dir / save_folder, settings)
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
    plane_dir = registration_dir / save_folder / "plane0"
    if db.get("fast_disk") == str(registration_dir / save_folder):  # set by build_suite2p_db
        move_binaries_into_plane(plane_dir)
    record = alignment_record(choice, identity) | {"binaries": binaries_identity(plane_dir)}
    write_alignment_record(registration_dir / save_folder, record)
    return plane_dir


class NoPerFrameRecord(ValueError):
    """The session has no per-frame bidiphase record (``.phase.json`` missing, or written by
    the chunked method), so how often the structural channel was usable is unknown."""


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
    fraction, path = structural_trust(suite2p_input_tiff, settings)
    threshold = settings.registration.auto_structural_min_fraction
    reason = (
        f"auto: the bidiphase step trusted the structural channel for {fraction:.0%} of "
        f"frames, threshold {threshold:.0%}"
    )
    return MotionChoice(
        choose_motion_source(fraction, threshold), reason, fraction, threshold, path.name
    )


def structural_trust(suite2p_input_tiff: Path, settings: PipelineSettings) -> tuple[float, Path]:
    """Fraction of frames the per-frame bidiphase step trusted the structural channel in,
    from the ``.phase.json`` of the corrected TIFF *suite2p_input_tiff* is (or came from)."""
    path = phase_record_path(suite2p_input_tiff, settings)
    if not path.is_file():
        raise NoPerFrameRecord(
            f"No {path.name} beside the TIFF, which records how often the structural channel "
            "was usable; run the phase correction first"
        )
    record = json.loads(path.read_text())
    if record.get("phase_method") != "per_frame" or "fraction_structural" not in record:
        raise NoPerFrameRecord(
            f"{path} is not a per-frame bidiphase record with fraction_structural; rerun the "
            'phase correction with method = "per_frame"'
        )
    return float(record["fraction_structural"]), path


def red_usable(suite2p_input_tiff: Path, settings: PipelineSettings) -> tuple[bool, str]:
    """Whether the structural (red) channel is usable for this session, by the rule "auto"
    alignment uses, and why."""
    fraction, path = structural_trust(suite2p_input_tiff, settings)
    threshold = settings.registration.auto_structural_min_fraction
    usable = choose_motion_source(fraction, threshold) == "structural"
    return usable, (
        f"the bidiphase step trusted the structural channel for {fraction:.0%} of frames "
        f"({'at least' if usable else 'below'} the {threshold:.0%} of "
        f"auto_structural_min_fraction; {path.name})"
    )


def input_identity(tiff: Path) -> dict:
    stat = Path(tiff).stat()
    return {"suite2p_input": str(tiff), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


# The file name, not its full path: a session folder moved or copied (share <-> laptop)
# keeps its registrations valid.
SAME_INPUT_KEYS = ("size_bytes", "mtime_ns")


BINARY_KEYS = ("reg_file", "reg_file_chan2", "raw_file", "raw_file_chan2")


def move_binaries_into_plane(plane_dir: Path) -> None:
    """Move the binaries Suite2p left in ``<save_folder>/suite2p/plane0`` into the
    registration's ``plane0``, beside its results, and point ``ops.npy`` and ``db.npy`` at
    them. Suite2p always writes binaries to a ``suite2p/plane0`` folder, so for a save_folder
    other than the default they would otherwise sit one folder down."""
    plane_dir = Path(plane_dir)
    nested = plane_dir.parent / "suite2p" / "plane0"
    for record_name in ("ops.npy", "db.npy"):
        record_file = plane_dir / record_name
        if not record_file.is_file():
            continue
        record = np.load(record_file, allow_pickle=True).item()
        for key in BINARY_KEYS:
            path = Path(record[key]) if record.get(key) else None
            if path is None or path.parent == plane_dir:
                continue
            if path.parent != nested:
                log.warning("%s: %s is %s, not in %s; left where it is", record_file, key,
                            path, nested)  # fmt: skip
                continue
            if path.is_file():
                path.replace(plane_dir / path.name)
            elif not (plane_dir / path.name).is_file():
                log.warning("%s: %s %s does not exist", record_file, key, path)
            record[key] = str(plane_dir / path.name)
        if record.get("fast_disk") == str(nested):  # the folder holding the binaries
            record["fast_disk"] = str(plane_dir)
        partial = record_file.with_name(record_file.stem + ".partial.npy")
        np.save(partial, record, allow_pickle=True)
        partial.replace(record_file)
    for folder in (nested, nested.parent):
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()


def binaries_identity(plane_dir: Path) -> dict:
    """Path, size and modification time of each binary Suite2p wrote for this registration,
    as its ops records them, so a binary outside the registration's folder (fast_disk) can be
    told apart from another one written there later."""
    ops_file = Path(plane_dir) / "ops.npy"
    if not ops_file.is_file():
        log.warning("No %s after Suite2p: the binaries are not recorded", ops_file)
        return {}
    ops = np.load(ops_file, allow_pickle=True).item()
    paths = {key: Path(ops[key]) for key in BINARY_KEYS if ops.get(key)}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        log.warning("Binaries named in %s do not exist, not recorded: %s", ops_file, missing)
    return {key: binary_identity(path) for key, path in paths.items() if path.is_file()}


def binary_identity(path: Path) -> dict:
    stat = Path(path).stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def alignment_record(choice: MotionChoice, identity: dict) -> dict:
    return {
        "motion_alignment_source": choice.source,
        "reason": choice.reason,
        "fraction_structural": choice.fraction_structural,
        "auto_structural_min_fraction": choice.threshold,
        "phase_record": choice.phase_record,
        **identity,
    }


def write_alignment_record(suite2p_dir: Path, record: dict) -> None:
    """The record of one registration, in its ``<save_folder>/``."""
    path = Path(suite2p_dir) / MOTION_ALIGNMENT_RECORD
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(record, indent=2))
    partial.replace(path)


def alignment_records(alignment_folder: Path) -> list[dict]:
    """The records of every registration in ``aligned_<source>/``: one per save_folder,
    and the one earlier versions wrote beside them, in ``aligned_<source>/`` itself (only
    read, never removed: it describes the TIFF the folder's earlier registrations used)."""
    alignment_folder = Path(alignment_folder)
    paths = [alignment_folder / MOTION_ALIGNMENT_RECORD]
    paths += sorted(alignment_folder.glob(f"*/{MOTION_ALIGNMENT_RECORD}"))
    return [json.loads(path.read_text()) for path in paths if path.is_file()]


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


# How much newer than its registration's ops.npy a binary may be (file times can be coarse,
# e.g. on SMB shares); Suite2p itself saves ops.npy only after closing the binaries.
BINARY_AFTER_OPS_TOLERANCE_S = 2.0


def resolve_registered_binary(ops: dict, plane_dir: Path, ops_key: str, filename: str) -> Path:
    """This registration's binary: in its plane0, else in <save_folder>/suite2p/plane0 (where
    registrations into a save_folder other than the default kept it between PR #36 and PR
    #37), else where ops records it (fast_disk). Never another registration's.

    Raises:
        FileNotFoundError: If there is no such binary.
        RuntimeError: If the binary found is not shown to be this registration's (see
            ``check_registration_binary``).
    """
    plane_dir = Path(plane_dir)
    candidates = [plane_dir / filename, plane_dir.parent / "suite2p" / "plane0" / filename]
    if ops.get(ops_key) and Path(ops[ops_key]) not in candidates:
        candidates.append(Path(ops[ops_key]))
    for candidate in candidates:
        if candidate.is_file():
            check_registration_binary(candidate, plane_dir, ops_key)
            return candidate
    raise FileNotFoundError(f"Could not find {filename}; checked {[str(c) for c in candidates]}")


def check_registration_binary(binary: Path, plane_dir: Path, ops_key: str) -> None:
    """Raise unless *binary* is shown to belong to the registration in *plane_dir*.

    A registration with its own ``<save_folder>/motion_alignment.json`` keeps its binaries to
    itself: one inside its folder is its own; one outside it (fast_disk) must have the size
    and modification time the record gives. A registration without the record was made
    before each save_folder kept its own binaries, when a registration into a save_folder
    other than the default wrote its binaries into the variant's ``suite2p/plane0``: its
    binary is checked against its ``ops.npy`` if it is outside its own folder, or if it is the
    default registration of an ``aligned_<channel>/`` folder.
    """
    own = Path(plane_dir).parent
    binary = Path(binary)
    record_file = own / MOTION_ALIGNMENT_RECORD
    if record_file.is_file():
        if binary.is_relative_to(own):
            return
        recorded = json.loads(record_file.read_text()).get("binaries", {}).get(ops_key)
        if recorded is None:
            raise RuntimeError(
                f"{record_file} records no {ops_key} binary, so {binary}, outside "
                f"{own.name}/, cannot be shown to be this registration's; register again"
            )
        if any(recorded[k] != binary_identity(binary)[k] for k in SAME_INPUT_KEYS):
            raise RuntimeError(
                f"{binary} is not the {ops_key} binary {own.name} wrote (its size or "
                f"modification time differ from {record_file}): another registration may have "
                "written it, or it was copied without keeping file times; register again, or "
                "copy it again with cp -p or rsync -t"
            )
        return
    shared_place = not binary.is_relative_to(own) or (
        own.name == DEFAULT_SAVE_FOLDER and own.parent.name in ALIGNMENT_SUBDIRS.values()
    )
    if shared_place:
        check_binary_belongs(binary, Path(plane_dir) / "ops.npy")


def check_binary_belongs(binary: Path, ops_file: Path) -> None:
    """Refuse a binary written after the registration's ops.npy (no ops.npy: not checked):
    another registration wrote it, or the folder was copied without keeping file times."""
    if not ops_file.is_file():
        return
    lag = binary.stat().st_mtime - ops_file.stat().st_mtime
    if lag > BINARY_AFTER_OPS_TOLERANCE_S:
        raise RuntimeError(
            f"{binary} was written {lag:.0f} s after {ops_file}. Either another registration "
            "into a save_folder of this variant wrote over it (register this one again), or "
            "the folder was copied without keeping file times (copy it again with cp -p or "
            "rsync -t)"
        )


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
