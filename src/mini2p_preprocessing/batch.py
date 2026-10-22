"""Plan and run the pipeline over every matching raw TIFF in a range of sessions.

Raw TIFFs within a session get ``run-01``, ``run-02``, ... from their ScanImage
acquisition counter. Each run directory holds a ``source_tiff.json`` manifest that
records which raw file it came from and whether processing finished, so reruns
skip completed work and stop on conflicting assignments instead of overwriting.
"""

from __future__ import annotations

import json
import logging
import re
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
    with Path(path).open("rb") as handle:
        return BatchSettings.model_validate(tomllib.load(handle)["batch"])


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


# ---- analysis: pure planning decisions --------------------------------------------


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


def decide_action(
    source: Path,
    identity: dict,
    run_dir: Path,
    run_dir_exists: bool,
    prior_elsewhere: ExistingRun | None,
    prior_here: ExistingRun | None,
    retry_incomplete: bool,
    rerun_before: datetime | None,
) -> tuple[Action, str]:
    """Whether to run, skip or flag one source/run-directory assignment."""
    if prior_elsewhere is not None and prior_elsewhere.run_dir != run_dir:
        return (
            "skip",
            f"already assigned to {prior_elsewhere.run_dir.name} ({prior_elsewhere.evidence})",
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
    return "skip", f"existing manifest status={status or 'unknown'}"


# ---- I/O: discovery and manifests -------------------------------------------------


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


def find_sessions(raw_subject_dir: Path, first: int, last: int) -> list[tuple[int, Path]]:
    sessions = sorted(
        (int(m.group(1)), d)
        for d in Path(raw_subject_dir).iterdir()
        if d.is_dir() and (m := SESSION_PATTERN.match(d.name)) and first <= int(m.group(1)) <= last
    )
    missing = sorted(set(range(first, last + 1)) - {n for n, _ in sessions})
    if missing:
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
    plan = []
    for session, session_dir in find_sessions(
        batch.raw_subject_dir, batch.session_first, batch.session_last
    ):
        sources = sorted(
            (
                f
                for f in (session_dir / "funcimg").iterdir()
                if f.is_file()
                and f.suffix.lower() == ".tif"
                and batch.source_substring.lower() in f.name.lower()
            ),
            key=acquisition_key,
        )
        derivative_funcimg = batch.derivatives_subject_dir / session_dir.name / "funcimg"
        # The same raw TIFF may already sit under a different run number.
        existing_by_source = {}
        if derivative_funcimg.is_dir():
            for existing_run in sorted(derivative_funcimg.glob("run-*")):
                found = read_existing_run(existing_run / subdir) if existing_run.is_dir() else None
                if found is not None:
                    existing_by_source[found.source] = found
        for index, source in enumerate(sources, start=1):
            run_name = f"run-{index:02d}"
            run_folder = derivative_funcimg / run_name
            run_dir = run_folder / subdir
            identity = source_identity(source)
            resolved = source.resolve()
            here = read_existing_run(run_folder) if run_folder.is_dir() else None
            blocked = run_folder_conflict(here, resolved)
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
                )
            plan.append(PlannedRun(session, source, run_name, run_dir, identity, action, reason))
    return plan


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
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    try:
        with log_path.open("w") as worker_log:
            subprocess.run(
                command, check=True, stdout=worker_log, stderr=subprocess.STDOUT, text=True
            )
    except subprocess.CalledProcessError as exc:
        tail = "\n".join(log_path.read_text().splitlines()[-80:]) if log_path.exists() else ""
        manifest.update(status="failed", finished_utc=_now(), error=repr(exc), error_tail=tail)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        raise
    except KeyboardInterrupt:
        manifest.update(status="interrupted", finished_utc=_now())
        manifest_path.write_text(json.dumps(manifest, indent=2))
        raise
    manifest.update(status="complete", finished_utc=_now())
    manifest_path.write_text(json.dumps(manifest, indent=2))


def snapshot_config(config: Path | bytes, destination_dir: Path) -> Path:
    """Freeze the config (a file, or its bytes as read) for this launch so later edits
    cannot reach running workers."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    destination_dir.mkdir(parents=True, exist_ok=True)
    snapshot = destination_dir / f"batch_config_{stamp}.toml"
    with snapshot.open("xb") as frozen:  # never replaces a snapshot another launch uses
        frozen.write(config if isinstance(config, bytes) else Path(config).read_bytes())
    return snapshot


def run_batch(batch: BatchSettings, config_path: Path) -> list[PlannedRun]:
    """Plan afresh from the manifests on disk, then run every pending source.

    Planning here rather than reusing an earlier plan means relaunching after a
    failure never reprocesses runs that completed in the meantime.
    """
    # Plan, check and freeze from one reading of the config, so they cannot disagree.
    config_bytes = Path(config_path).read_bytes()
    settings = settings_from_toml(tomllib.loads(config_bytes.decode()))
    plan = plan_batch(batch, corrected=distortion_corrected(settings))
    conflicts = [p for p in plan if p.action == "conflict"]
    if conflicts:
        details = "\n".join(f"  {p.run_dir}: {p.reason}" for p in conflicts)
        raise RuntimeError(f"Resolve run-directory conflicts before launching:\n{details}")
    pending = [p for p in plan if p.action == "run"]
    if not pending:
        log.info("Nothing to run.")
        return plan
    replacing = [p.run_dir for p in pending if p.run_dir.exists()]
    if replacing and not settings.overwrite:
        raise ValueError(
            f"{len(replacing)} run(s) would replace existing output (e.g. {replacing[0]}) but "
            "the config sets overwrite = false"
        )
    frozen_config = snapshot_config(config_bytes, batch.derivatives_subject_dir)
    log.info("Workers use config snapshot %s", frozen_config)
    progress = ProgressLog("Batch", len(pending), log.info)
    for index, row in enumerate(pending, start=1):
        log.info("[%d/%d] %s -> %s", index, len(pending), row.source, row.run_dir)
        try:
            run_one(row, frozen_config, batch.destripe)
        except subprocess.CalledProcessError:
            log.error("FAILED: %s (log: %s)", row.source, row.run_dir / "preprocess.log")
            if batch.stop_on_error:
                raise
            progress.step_done(succeeded=False)
        else:
            log.info("COMPLETE: %s", row.run_dir)
            progress.step_done()
    return plan
