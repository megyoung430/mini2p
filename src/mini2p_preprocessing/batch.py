"""Plan and run the pipeline over every matching raw TIFF in a range of sessions.

Raw TIFFs within a session get ``run-01``, ``run-02``, ... from their ScanImage
acquisition counter. Each run directory holds a ``source_tiff.json`` manifest that
records which raw file it came from and whether processing finished, so reruns
skip completed work and stop on conflicting assignments instead of overwriting.

``[batch] aligned_runs_csv`` limits the batch to the TIFFs that can be put on pyControl
time, as listed by magnitude-bandit-analysis's ``scripts/funcimg/list_aligned_runs.py``.
The batch runs one TIFF after another (:func:`run_batch`), or as a SLURM array, one job
per TIFF (:func:`prepare_array`, then :func:`run_array_task` in each job).
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict

from .config import distortion_corrected, load_settings, settings_from_toml, variant_subdir
from .progress import ProgressLog

log = logging.getLogger(__name__)
MANIFEST_NAME = "source_tiff.json"
SESSION_PATTERN = re.compile(r"^ses-(\d+)_")
ACQUISITION_PATTERN = re.compile(r"(\d+)$")
Action = Literal["run", "skip", "conflict"]


class BatchSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    raw_subject_dir: Path
    derivatives_subject_dir: Path
    session_first: int
    session_last: int
    source_substring: str
    # CSV from magnitude-bandit-analysis's scripts/funcimg/list_aligned_runs.py: only the
    # TIFFs it marks aligned are run. Sessions missing from the range are then allowed.
    aligned_runs_csv: Path | None = None
    destripe: bool = False
    stop_on_error: bool = True
    retry_incomplete_runs: bool = True  # retries failed/interrupted runs
    # Register again, replacing them, the runs that finished before this time (with a
    # time zone, e.g. 2026-10-21T09:00:00+01:00). Relaunching with the same time never
    # redoes a run the rerun has already replaced.
    rerun_completed_before: AwareDatetime | None = None
    # Cellpose masks for run-XX and the image they were drawn on, relative to run-XX/.
    cellpose_seg: str = "suite2p/plane0/meanImg_green_functional_contrast_seg.npy"
    cellpose_image: str = "suite2p/plane0/meanImg_green_functional_contrast.png"


def load_batch_settings(path: Path) -> BatchSettings:
    """The ``[batch]`` table of a config.

    Raises:
        ValueError: If the config has no ``[batch]`` table, or it is invalid.
    """
    with Path(path).open("rb") as handle:
        return batch_settings_from_toml(tomllib.load(handle), path)


def batch_settings_from_toml(document: dict, source: Path | str) -> BatchSettings:
    """The ``[batch]`` table of a parsed config read from *source* (named in errors)."""
    if "batch" not in document:
        raise ValueError(f"{source} has no [batch] table")
    return BatchSettings.model_validate(document["batch"])


@dataclass(frozen=True)
class ExistingRun:
    """What an existing run directory says about its source TIFF."""

    run_dir: Path
    source: Path
    metadata: dict
    evidence: Literal["manifest", "phase sidecar"]


@dataclass(frozen=True)
class PlannedRun:
    session: int
    source: Path
    run_name: str
    run_dir: Path
    identity: dict
    action: Action
    reason: str


@dataclass(frozen=True)
class AlignedRun:
    """One row of the aligned-runs CSV: whether a raw TIFF can be put on pyControl time."""

    session_name: str  # the raw session folder's name, ses-<number>_...
    run_name: str  # run-XX as the listing numbered the TIFF
    aligned: bool
    reason: str  # why it cannot be aligned; empty when it can


ALIGNED_RUNS_COLUMNS = ("session", "run", "tiff", "aligned", "reason")


# ---- analysis: pure planning decisions --------------------------------------------


def alignment_decision(
    listed: AlignedRun | None, run_name: str, csv_name: str
) -> tuple[Action, str] | None:
    """What the aligned-runs CSV says about one planned TIFF: None to leave the plan's
    decision, ``skip`` for a TIFF it does not mark aligned, or ``conflict`` when it numbers
    the TIFF differently, since then the two repositories disagree on which run is which."""
    if listed is None:
        return "skip", f"not in {csv_name}"
    if listed.run_name != run_name:
        return "conflict", (
            f"{csv_name} numbers this TIFF {listed.run_name}, the batch {run_name}; the TIFFs "
            "of the session differ between them"
        )
    if not listed.aligned:
        return "skip", f"not aligned to behavior: {listed.reason}"
    return None


def acquisition_key(tiff_path: Path) -> tuple:
    """Sort ScanImage names by their trailing acquisition counter."""
    match = ACQUISITION_PATTERN.search(tiff_path.stem)
    if match:
        return (0, int(match.group(1)), tiff_path.name.lower())
    return (1, tiff_path.name.lower())


def run_folder_conflict(here: ExistingRun | None, source: Path) -> str | None:
    """Why output directly in ``run-XX/`` blocks this run, or None if it doesn't.

    ``run-XX/suite2p`` holds the registration the Cellpose masks were drawn on, so it
    must come from the same TIFF as the run's variant subfolders.
    """
    if here is None or here.source == source:
        return None
    return (
        f"{here.run_dir.name}/ holds output of {here.source.name} ({here.evidence}), "
        "a different TIFF; check the run numbering"
    )


def numbering_shift(run_name: str, recorded_in: set[str]) -> str | None:
    """Why a TIFF cannot become *run_name* when output of it is already recorded in the
    run folders *recorded_in* (any variant), or None: the run numbers have shifted."""
    elsewhere = sorted(recorded_in - {run_name})
    if not elsewhere:
        return None
    return (
        f"this TIFF is already in {', '.join(elsewhere)} (another variant there): the run "
        f"numbering has shifted, so {run_name} would name a second run folder for it"
    )


def old_layout(variant_dir: Path) -> str | None:
    """Why a variant folder from before the aligned_<channel>/ layout cannot be reused: its
    registration is in <variant>/suite2p, which nothing reads any more."""
    if (Path(variant_dir) / "suite2p").is_dir():
        return (
            f"{variant_dir.name}/ has the old layout (suite2p/ directly inside); move or "
            "delete it to register again into aligned_<channel>/"
        )
    return None


def decide_action(
    source: Path,
    identity: dict,
    run_dir: Path,
    run_dir_exists: bool,
    prior_elsewhere: ExistingRun | None,
    prior_here: ExistingRun | None,
    retry_incomplete: bool,
    rerun_before: datetime | None,
    job_ended: bool = False,
) -> tuple[Action, str]:
    """Whether to run, skip or flag one source/run-directory assignment.

    The same TIFF already processed under another ``run-XX`` conflicts: the run numbers
    have shifted (a TIFF added or removed), so a run folder no longer names its TIFF.
    *job_ended* says a manifest still marked running belongs to a SLURM job that has ended
    (killed by a time or memory limit), so the run is retried like a failed one.
    """
    if prior_elsewhere is not None and prior_elsewhere.run_dir != run_dir:
        return (
            "conflict",
            f"this TIFF is already in {run_folder_name(prior_elsewhere.run_dir)} "
            f"({prior_elsewhere.evidence}): the run numbering has shifted; check it",
        )
    if not run_dir_exists:
        return "run", "new source/run assignment"
    if prior_here is None:
        return "conflict", "run directory exists but its source cannot be identified"
    if prior_here.source != source:
        return (
            "conflict",
            f"run directory belongs to {prior_here.source.name} ({prior_here.evidence})",
        )
    if prior_here.evidence != "manifest":
        if rerun_before is not None:
            return "run", f"rerun_completed_before: replacing older output ({prior_here.evidence})"
        return "skip", f"matching older output found via {prior_here.evidence}"
    meta = prior_here.metadata
    if (meta.get("source_size_bytes"), meta.get("source_mtime_ns")) != (
        identity["source_size_bytes"],
        identity["source_mtime_ns"],
    ):
        return "conflict", "source path matches but file size/mtime changed"
    status = meta.get("status")
    if status == "complete":
        finished = meta.get("finished_utc")
        if rerun_before is not None and (
            finished is None or datetime.fromisoformat(finished) < rerun_before
        ):
            return "run", f"rerun_completed_before: replacing the run finished {finished}"
        return "skip", "completed manifest already exists"
    if retry_incomplete and status in {"failed", "interrupted"}:
        return "run", f"retrying manifest status={status}"
    if retry_incomplete and status == "running" and job_ended:
        return "run", f"retrying: its SLURM job {meta.get('slurm_job_id')} ended while running"
    return "skip", f"existing manifest status={status or 'unknown'}"


def run_folder_name(variant_dir: Path) -> str:
    """The ``run-XX`` folder a variant folder lies in, for messages."""
    return next(
        (part for part in reversed(Path(variant_dir).parts) if part.startswith("run-")),
        str(variant_dir),
    )


# ---- I/O: discovery and manifests -------------------------------------------------


def load_aligned_runs(path: Path) -> dict[Path, AlignedRun]:
    """The aligned-runs CSV, keyed by each raw TIFF's resolved path.

    Raises:
        ValueError: If a column is missing, ``aligned`` is not True/False, or a TIFF is
            listed twice.
    """
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in ALIGNED_RUNS_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} lacks the columns {missing}")
        rows = list(reader)
    runs = {}
    for row in rows:
        if row["aligned"] not in ("True", "False"):
            raise ValueError(f"{path}: aligned must be True or False, not {row['aligned']!r}")
        if not SESSION_PATTERN.match(row["session"]):
            raise ValueError(f"{path}: {row['session']!r} is not a ses-<number>_... folder name")
        tiff = Path(row["tiff"]).resolve()
        if tiff in runs:
            raise ValueError(f"{path} lists {tiff} twice")
        runs[tiff] = AlignedRun(row["session"], row["run"], row["aligned"] == "True", row["reason"])
    return runs


def _write_json(path: Path, data: dict) -> None:
    """Write *data* to *path* whole or not at all, so a killed job never leaves a
    half-written manifest that would stop every later plan."""
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    partial.write_text(json.dumps(data, indent=2))
    os.replace(partial, path)


def _read_json(path: Path) -> dict | None:
    """Parsed JSON, or None if the file does not exist; a corrupt file raises."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except ValueError as exc:
        raise ValueError(f"Corrupt JSON in {path}; inspect or move it before planning") from exc


def read_existing_run(run_dir: Path) -> ExistingRun | None:
    """Source identity from the manifest, or from an older .phase.json sidecar."""
    manifest = _read_json(run_dir / MANIFEST_NAME)
    if manifest and manifest.get("source_tiff"):
        return ExistingRun(run_dir, Path(manifest["source_tiff"]).resolve(), manifest, "manifest")
    for sidecar in sorted(run_dir.glob("*.phase.json")):
        phase = _read_json(sidecar)
        if phase and phase.get("input_tiff"):
            return ExistingRun(run_dir, Path(phase["input_tiff"]).resolve(), phase, "phase sidecar")
    return None


def run_folders_by_source(funcimg_dir: Path) -> dict[Path, set[str]]:
    """For each raw TIFF with output under ``funcimg_dir/run-*`` (in ``run-XX/`` itself or
    any variant folder, old or new layout), the names of the run folders holding it."""
    found: dict[Path, set[str]] = {}
    for run_folder in sorted(Path(funcimg_dir).glob("run-*")) if Path(funcimg_dir).is_dir() else []:
        if not run_folder.is_dir():
            continue
        candidates = [run_folder, *run_folder.glob("*/"), *run_folder.glob("*/*/")]
        for folder in candidates:
            if folder.name in ("suite2p",) or folder.name.startswith("aligned_"):
                continue
            existing = read_existing_run(folder)
            if existing is not None:
                found.setdefault(existing.source, set()).add(run_folder.name)
    return found


def run_state(variant_dir: Path, since: datetime | None) -> str:
    """Where the batch is with this run folder, for deciding whether its output is final.

    ``"ready"``: completed (at or after *since*, the rerun cutoff, if given). Otherwise
    ``"not started"`` (no folder), ``"no batch record"`` (older output), ``"running"``,
    ``"failed"``, ``"interrupted"``, or ``"awaiting rerun"`` (completed before the
    cutoff, so still the old output, about to be replaced).
    """
    variant_dir = Path(variant_dir)
    if not variant_dir.is_dir():
        return "not started"
    run = read_existing_run(variant_dir)
    if run is None or run.evidence != "manifest":
        return "no batch record"
    status = run.metadata.get("status")
    if status != "complete":
        return str(status)
    finished = run.metadata.get("finished_utc")
    if since is not None and (finished is None or datetime.fromisoformat(finished) < since):
        return "awaiting rerun"
    return "ready"


def finished_by_batch(variant_dir: Path, since: datetime | None) -> bool:
    """Whether :func:`run_state` is ``"ready"``."""
    return run_state(variant_dir, since) == "ready"


def launch_cutoff(derivatives_dir: Path) -> datetime | None:
    """``rerun_completed_before`` of the newest batch launch, from its frozen config.

    A running batch uses the config it froze at launch, not the live ``pipeline.toml``,
    which may have been edited since.

    Raises:
        FileNotFoundError: If no batch has been launched into *derivatives_dir*.
    """
    return load_batch_settings(_newest_snapshot(derivatives_dir)).rerun_completed_before


def _newest_snapshot(derivatives_dir: Path) -> Path:
    snapshots = sorted(Path(derivatives_dir).glob("batch_config_*.toml"))
    if not snapshots:
        raise FileNotFoundError(
            f"No batch_config_*.toml in {derivatives_dir}; launch the batch first"
        )
    return snapshots[-1]


def source_identity(tiff_path: Path) -> dict:
    stat = tiff_path.stat()
    return {
        "source_tiff": str(tiff_path.resolve()),
        "source_size_bytes": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
    }


def find_sessions(
    raw_subject_dir: Path, first: int, last: int, allow_gaps: bool = False
) -> list[tuple[int, Path]]:
    """The numbered raw sessions from *first* to *last*; a missing number raises unless
    *allow_gaps*."""
    sessions = sorted(
        (int(m.group(1)), d)
        for d in Path(raw_subject_dir).iterdir()
        if d.is_dir() and (m := SESSION_PATTERN.match(d.name)) and first <= int(m.group(1)) <= last
    )
    missing = sorted(set(range(first, last + 1)) - {n for n, _ in sessions})
    if missing and not allow_gaps:
        raise FileNotFoundError(f"Missing raw session directories: {missing}")
    return sessions


def batch_distortion_corrected(config_path: Path) -> bool:
    """Whether the config's ``[distortion] mode`` corrects distortion (anything but "off")."""
    return distortion_corrected(load_settings(config_path))


def launch_variant(derivatives_dir: Path) -> str:
    """The variant folder the newest batch launch writes to, from its frozen config."""
    snapshot = _newest_snapshot(derivatives_dir)
    settings = settings_from_toml(tomllib.loads(snapshot.read_text()))
    return variant_subdir(
        load_batch_settings(snapshot).destripe,
        distortion_corrected=distortion_corrected(settings),
    )


def plan_batch(batch: BatchSettings, *, corrected: bool) -> list[PlannedRun]:
    """What the batch would do with every raw TIFF in the session range. Runs go to
    ``run-XX/<variant>``, the variant set by ``batch.destripe`` and *corrected* (the
    config's ``[distortion] mode`` is not "off")."""
    subdir = variant_subdir(batch.destripe, distortion_corrected=corrected)
    aligned = None if batch.aligned_runs_csv is None else load_aligned_runs(batch.aligned_runs_csv)
    plan = []
    for session, session_dir in find_sessions(
        batch.raw_subject_dir,
        batch.session_first,
        batch.session_last,
        allow_gaps=aligned is not None,
    ):
        sources = session_sources(session_dir, batch.source_substring)
        derivative_funcimg = batch.derivatives_subject_dir / session_dir.name / "funcimg"
        # The same raw TIFF may already sit under a different run number.
        existing_by_source = {}
        if derivative_funcimg.is_dir():
            for existing_run in sorted(derivative_funcimg.glob("run-*")):
                found = read_existing_run(existing_run / subdir) if existing_run.is_dir() else None
                if found is not None:
                    existing_by_source[found.source] = found
        # ... recorded in any of its variants: run folders are paired by number alone.
        folders_by_source = run_folders_by_source(derivative_funcimg)
        for index, source in enumerate(sources, start=1):
            run_name = f"run-{index:02d}"
            run_folder = derivative_funcimg / run_name
            run_dir = run_folder / subdir
            identity = source_identity(source)
            resolved = source.resolve()
            here = read_existing_run(run_folder) if run_folder.is_dir() else None
            blocked = (
                run_folder_conflict(here, resolved)
                or numbering_shift(run_name, folders_by_source.get(resolved, set()))
                or old_layout(run_dir)
            )
            if blocked:
                action, reason = "conflict", blocked
            else:
                action, reason = decide_action(
                    resolved,
                    identity,
                    run_dir,
                    run_dir.exists(),
                    existing_by_source.get(resolved),
                    read_existing_run(run_dir) if run_dir.exists() else None,
                    batch.retry_incomplete_runs,
                    batch.rerun_completed_before,
                    job_ended=running_job_ended(run_dir),
                )
            if aligned is not None and action != "conflict":
                listed = aligned.get(resolved)
                override = alignment_decision(listed, run_name, batch.aligned_runs_csv.name)
                action, reason = override or (action, reason)
            plan.append(PlannedRun(session, source, run_name, run_dir, identity, action, reason))
    if aligned is not None:
        check_listing_covered(aligned, plan, batch)
    return plan


def session_sources(session_dir: Path, source_substring: str) -> list[Path]:
    """The session's raw TIFFs (``funcimg/*.tif`` naming *source_substring*, any case), in
    acquisition order. macOS AppleDouble files (``._*``) are metadata, not TIFFs."""
    return sorted(
        (
            f
            for f in (Path(session_dir) / "funcimg").iterdir()
            if f.is_file()
            and f.suffix.lower() == ".tif"
            and not f.name.startswith("._")
            and source_substring.lower() in f.name.lower()
        ),
        key=acquisition_key,
    )


def check_listing_covered(
    aligned: dict[Path, AlignedRun], plan: list[PlannedRun], batch: BatchSettings
) -> None:
    """Refuse an aligned-runs CSV that does not cover the planned sessions: one that marks
    aligned a TIFF of the range the batch did not find (made from other raw data, or the
    TIFF has gone), or that lists none of a planned session's TIFFs (made before that
    session was recorded, so its runs would all be skipped unnoticed).

    Raises:
        ValueError: Naming the first few such TIFFs or sessions.
    """
    planned = {p.source.resolve() for p in plan}

    def in_range(row: AlignedRun) -> bool:
        number = int(SESSION_PATTERN.match(row.session_name).group(1))
        return batch.session_first <= number <= batch.session_last

    unplanned = sorted(
        str(tiff)
        for tiff, row in aligned.items()
        if row.aligned and in_range(row) and tiff not in planned
    )
    if unplanned:
        raise ValueError(
            f"{batch.aligned_runs_csv} marks aligned {len(unplanned)} TIFF(s) the batch did not "
            f"find under {batch.raw_subject_dir} (e.g. {unplanned[:3]}); list the runs again "
            "from this raw folder"
        )
    unlisted = sorted(
        {p.session for p in plan} - {p.session for p in plan if p.source.resolve() in aligned}
    )
    if unlisted:
        raise ValueError(
            f"{batch.aligned_runs_csv} lists no TIFF of session(s) {unlisted[:5]}, which have "
            "TIFFs; list the runs again so it covers them"
        )


def format_plan(plan: list[PlannedRun]) -> str:
    lines = [f"{'session':>7}  {'run':>6}  {'action':>8}  source / reason"]
    lines += [
        f"{p.session:>7}  {p.run_name:>6}  {p.action:>8}  {p.source.name} — {p.reason}"
        for p in plan
    ]
    counts = {a: sum(p.action == a for p in plan) for a in ("run", "skip", "conflict")}
    return "\n".join(lines + [f"Summary: {counts}"])


# ---- I/O: workers -----------------------------------------------------------------


def _now() -> str:
    return datetime.now(UTC).isoformat()


def worker_command(row: PlannedRun, config_path: Path, destripe: bool) -> list[str]:
    return [
        sys.executable,
        "-m",
        "mini2p_preprocessing.cli",
        "--config",
        str(Path(config_path).resolve()),
        "--input",
        str(row.source.resolve()),
        "--output-dir",
        str(row.run_dir.resolve()),
        "--destripe",
        "1" if destripe else "0",
    ]


def run_one(row: PlannedRun, config_path: Path, destripe: bool) -> None:
    """Run the pipeline for one source in a fresh process, tracking it in the manifest."""
    row.run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = row.run_dir / MANIFEST_NAME
    log_path = row.run_dir / "preprocess.log"
    command = worker_command(row, config_path, destripe)
    manifest = {
        **row.identity,
        "session": row.session,
        "run": row.run_name,
        "destriping_enabled": destripe,
        "output_dir": str(row.run_dir),
        "status": "running",
        "started_utc": _now(),
        "pipeline_command": command,
        "pipeline_config": str(config_path),
        "worker_log": str(log_path),
        **slurm_ids(),
    }
    _write_json(manifest_path, manifest)
    try:
        with log_path.open("w") as worker_log:
            subprocess.run(
                command, check=True, stdout=worker_log, stderr=subprocess.STDOUT, text=True
            )
    except subprocess.CalledProcessError as exc:
        tail = "\n".join(log_path.read_text().splitlines()[-80:]) if log_path.exists() else ""
        manifest.update(status="failed", finished_utc=_now(), error=repr(exc), error_tail=tail)
        _write_json(manifest_path, manifest)
        raise
    except KeyboardInterrupt:
        manifest.update(status="interrupted", finished_utc=_now())
        _write_json(manifest_path, manifest)
        raise
    manifest.update(status="complete", finished_utc=_now())
    _write_json(manifest_path, manifest)


def snapshot_config(config: Path | bytes, destination_dir: Path) -> Path:
    """Freeze the config (a file, or its bytes as read) for this launch so later edits
    cannot reach running workers."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    destination_dir.mkdir(parents=True, exist_ok=True)
    snapshot = destination_dir / f"batch_config_{stamp}.toml"
    with snapshot.open("xb") as frozen:  # never replaces a snapshot another launch uses
        frozen.write(config if isinstance(config, bytes) else Path(config).read_bytes())
    return snapshot


@dataclass(frozen=True)
class Launch:
    """A planned launch: the plan, its pending rows and the frozen config (None when
    nothing is pending)."""

    plan: list[PlannedRun]
    pending: list[PlannedRun]
    config: Path | None


def _launch(batch: BatchSettings, config_bytes: bytes) -> Launch:
    """Plan, check and freeze a launch from one reading of the config (*config_bytes*),
    so they cannot disagree.

    Raises:
        RuntimeError: If any run folder conflicts.
        ValueError: If a pending run would replace output and the config sets overwrite =
            false.
    """
    settings = settings_from_toml(tomllib.loads(config_bytes.decode()))
    plan = plan_batch(batch, corrected=distortion_corrected(settings))
    conflicts = [p for p in plan if p.action == "conflict"]
    if conflicts:
        details = "\n".join(f"  {p.run_dir}: {p.reason}" for p in conflicts)
        raise RuntimeError(f"Resolve run-directory conflicts before launching:\n{details}")
    pending = [p for p in plan if p.action == "run"]
    if not pending:
        return Launch(plan, pending, None)
    replacing = [p.run_dir for p in pending if p.run_dir.exists()]
    if replacing and not settings.overwrite:
        raise ValueError(
            f"{len(replacing)} run(s) would replace existing output (e.g. {replacing[0]}) but "
            "the config sets overwrite = false"
        )
    return Launch(plan, pending, snapshot_config(config_bytes, batch.derivatives_subject_dir))


def run_batch(batch: BatchSettings, config_path: Path) -> list[PlannedRun]:
    """Plan afresh from the manifests on disk, then run every pending source in turn.

    Planning here rather than reusing an earlier plan means relaunching after a
    failure never reprocesses runs that completed in the meantime.
    """
    launch = _launch(batch, Path(config_path).read_bytes())
    if not launch.pending:
        log.info("Nothing to run.")
        return launch.plan
    log.info("Workers use config snapshot %s", launch.config)
    progress = ProgressLog("Batch", len(launch.pending), log.info)
    for index, row in enumerate(launch.pending, start=1):
        log.info("[%d/%d] %s -> %s", index, len(launch.pending), row.source, row.run_dir)
        try:
            run_one(row, launch.config, batch.destripe)
        except subprocess.CalledProcessError:
            log.error("FAILED: %s (log: %s)", row.source, row.run_dir / "preprocess.log")
            if batch.stop_on_error:
                raise
            progress.step_done(succeeded=False)
        else:
            log.info("COMPLETE: %s", row.run_dir)
            progress.step_done()
    return launch.plan


# ---- SLURM ------------------------------------------------------------------------

ARRAY_JOB_NAME = "mini2p-batch"  # the --job-name of scripts/slurm/batch_array.sh
TASKS_PREFIX = "batch_tasks_"
# Folders that are private to one compute node: binaries left there cannot be read later.
NODE_LOCAL_PREFIXES = ("/tmp", "/var/tmp", "/dev/shm")


def slurm_ids() -> dict:
    """The SLURM job (and array task) running this process, for the manifest; empty
    outside SLURM."""
    names = {
        "slurm_job_id": "SLURM_JOB_ID",
        "slurm_array_job_id": "SLURM_ARRAY_JOB_ID",
        "slurm_array_task_id": "SLURM_ARRAY_TASK_ID",
    }
    return {key: os.environ[env] for key, env in names.items() if env in os.environ}


# States in which a job may still write to its run folder.
LIVE_STATES = {
    "PENDING", "CONFIGURING", "RUNNING", "COMPLETING", "SUSPENDED", "REQUEUED",
    "REQUEUE_HOLD", "REQUEUE_FED", "RESIZING", "SIGNALING", "STAGE_OUT", "STOPPED",
}  # fmt: skip


def _squeue(*args: str) -> subprocess.CompletedProcess | None:
    """``squeue -h`` with *args*, or None where SLURM is not available."""
    if shutil.which("squeue") is None:
        return None
    return subprocess.run(["squeue", "-h", *args], capture_output=True, text=True)


def slurm_job_alive(job_id: str) -> bool | None:
    """Whether SLURM job *job_id* may still be writing: pending, running or finishing.

    A job SLURM no longer knows ("Invalid job id", some minutes after it ends) has ended,
    as has one listed in a final state. None if SLURM cannot be asked (no ``squeue``, or
    it fails otherwise), and a warning is logged.
    """
    result = _squeue("-j", str(job_id), "-o", "%T")
    if result is None:
        log.warning("Cannot ask SLURM about job %s: no squeue here", job_id)
        return None
    if result.returncode != 0:
        if "invalid job id" in (result.stderr + result.stdout).lower():
            return False
        log.warning("Cannot ask SLURM about job %s: %s", job_id, result.stderr.strip())
        return None
    states = {line.strip().upper() for line in result.stdout.splitlines() if line.strip()}
    return bool(states & LIVE_STATES)


def running_job_ended(run_dir: Path) -> bool:
    """Whether *run_dir*'s manifest says running for a SLURM job that has ended. False if
    it is not running, records no job, or SLURM cannot be asked (e.g. on a laptop)."""
    manifest = _read_json(Path(run_dir) / MANIFEST_NAME)
    if not manifest or manifest.get("status") != "running" or "slurm_job_id" not in manifest:
        return False
    return slurm_job_alive(manifest["slurm_job_id"]) is False


def queued_array_jobs(derivatives_dir: Path) -> list[str] | None:
    """Job ids of this user's batch array jobs, submitted for *derivatives_dir* (their
    ``--comment``, as ``prepare`` prints it), that SLURM still lists; None if SLURM cannot
    be asked."""
    user = os.environ.get("USER", "")
    result = _squeue("-u", user, "-n", ARRAY_JOB_NAME, "-o", "%i|%k") if user else None
    if result is None or result.returncode != 0:
        return None
    wanted = str(Path(derivatives_dir))
    return [
        job
        for job, _, comment in (line.partition("|") for line in result.stdout.splitlines())
        if job.strip() and comment.strip() == wanted
    ]


def node_local_fast_disk(settings) -> Path | None:
    """``[registration] fast_disk`` if it is a node-private folder, else None."""
    fast_disk = settings.registration.fast_disk
    if fast_disk is None:
        return None
    text = str(fast_disk)
    tmpdir = os.environ.get("TMPDIR")
    prefixes = NODE_LOCAL_PREFIXES + ((tmpdir,) if tmpdir else ())
    return fast_disk if any(text == p or text.startswith(p + "/") for p in prefixes) else None


# ---- SLURM array: one job per pending TIFF -----------------------------------------


def prepare_array(config_path: Path, allow_queued: bool = False) -> tuple[Path | None, Launch]:
    """Plan and freeze a launch of the config's ``[batch]`` as :func:`run_batch` does, and
    write its pending runs to ``batch_tasks_<UTC time>.json`` beside the frozen config,
    for a SLURM array whose task ``i`` (1-based) runs row ``i``.

    Returns the task list (None if nothing is pending) and the launch.

    Raises:
        RuntimeError: If batch array jobs are still queued or running (unless
            *allow_queued*): their runs have no complete manifest yet, so they would be
            planned again and two jobs would write one run folder.
        ValueError: If ``fast_disk`` is a node-private folder, where each job's binaries
            would be stranded; and as :func:`_launch`.
    """
    config_bytes = Path(config_path).read_bytes()
    document = tomllib.loads(config_bytes.decode())
    batch = batch_settings_from_toml(document, config_path)
    queued = queued_array_jobs(batch.derivatives_subject_dir)
    if queued is None:
        log.warning("Could not ask SLURM for queued batch jobs; not checked")
    elif queued and not allow_queued:
        raise RuntimeError(
            f"{len(queued)} {ARRAY_JOB_NAME} job(s) for {batch.derivatives_subject_dir} are "
            f"still queued or running ({', '.join(queued[:5])}); wait for them, or cancel "
            "them, before preparing again"
        )
    local = node_local_fast_disk(settings_from_toml(document))
    if local is not None:
        raise ValueError(
            f"[registration] fast_disk = {local} is private to each compute node, so the "
            "binaries of each job would be left on its node; unset it or use a shared folder"
        )
    launch = _launch(batch, config_bytes)
    if not launch.pending:
        return None, launch
    tasks = {
        "created_utc": _now(),
        "config": str(launch.config),
        "destripe": batch.destripe,
        "runs": [
            {
                "session": row.session,
                "source": str(row.source),
                "run_name": row.run_name,
                "run_dir": str(row.run_dir),
                "identity": row.identity,
            }
            for row in launch.pending
        ],
    }
    path = launch.config.with_name(
        launch.config.name.replace("batch_config_", TASKS_PREFIX).replace(".toml", ".json")
    )
    with path.open("x") as handle:  # each launch has its own task list
        json.dump(tasks, handle, indent=2)
    return path, launch


def array_task_row(tasks: dict, index: int) -> PlannedRun:
    """Row *index* (1-based, as ``SLURM_ARRAY_TASK_ID``) of a task list.

    Raises:
        IndexError: If the list has no such row.
    """
    runs = tasks["runs"]
    if not 1 <= index <= len(runs):
        raise IndexError(f"Task {index} is outside the list's 1-{len(runs)}")
    row = runs[index - 1]
    return PlannedRun(
        row["session"],
        Path(row["source"]),
        row["run_name"],
        Path(row["run_dir"]),
        row["identity"],
        "run",
        "array task",
    )


def task_state(manifest: dict | None, tasks: dict, own_job: str | None) -> str | None:
    """Why an array task should not run its row, from the row's manifest, or None to run
    it: ``"done"`` when this launch completed it, ``"superseded"`` when a later launch
    (another frozen config) completed it since this task list was written, ``"running"``
    when another SLURM job that may still be alive is running it."""
    if manifest is None:
        return None
    status = manifest.get("status")
    started = manifest.get("started_utc")
    if (
        status == "complete"
        and started is not None
        and datetime.fromisoformat(started) >= datetime.fromisoformat(tasks["created_utc"])
    ):
        return "done" if manifest.get("pipeline_config") == tasks["config"] else "superseded"
    job = manifest.get("slurm_job_id")
    if status == "running" and job is not None and job != own_job:
        return "running" if slurm_job_alive(job) is not False else None
    return None


class _Terminated(Exception):
    """SIGTERM reached an array task: SLURM's time limit, or scancel."""


def _raise_terminated(signum, frame):
    """Signal handler: turn SIGTERM into :class:`_Terminated`."""
    raise _Terminated(f"signal {signum}")


def run_array_task(tasks_path: Path, index: int) -> str:
    """Run row *index* of the task list in *tasks_path*. Returns ``"complete"``, or
    ``"skipped: ..."`` when this launch already completed the row or another live job is
    running it. SIGTERM (a time limit, scancel) marks the row interrupted, so the next
    launch retries it.

    Raises:
        ValueError: If the raw TIFF changed since the launch (size or modification time).
        subprocess.CalledProcessError: If the pipeline fails (recorded in the manifest).
    """
    tasks = json.loads(Path(tasks_path).read_text())
    row = array_task_row(tasks, index)
    manifest = _read_json(row.run_dir / MANIFEST_NAME)
    state = task_state(manifest, tasks, os.environ.get("SLURM_JOB_ID"))
    if state == "done":
        return "skipped: already complete in this launch"
    if state == "superseded":
        return f"skipped: a later launch ({manifest['pipeline_config']}) completed it"
    if state == "running":
        return f"skipped: SLURM job {manifest['slurm_job_id']} is running it"
    identity = source_identity(row.source)
    if identity != row.identity:
        raise ValueError(f"{row.source} changed since the launch ({row.identity} -> {identity})")
    log.info("Task %d: %s -> %s", index, row.source, row.run_dir)
    previous = signal.signal(signal.SIGTERM, _raise_terminated)
    try:
        run_one(row, Path(tasks["config"]), tasks["destripe"])
    except _Terminated:
        _mark_interrupted(row.run_dir)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
    return "complete"


def _mark_interrupted(run_dir: Path) -> None:
    """Record in the run's manifest that its task was terminated, so it is retried."""
    manifest_path = Path(run_dir) / MANIFEST_NAME
    manifest = _read_json(manifest_path) or {}
    manifest.update(status="interrupted", finished_utc=_now(), error="terminated (SIGTERM)")
    _write_json(manifest_path, manifest)
