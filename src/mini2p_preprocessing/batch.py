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

from pydantic import BaseModel, ConfigDict

from .config import variant_output_dir

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
    retry_incomplete_runs: bool = True  # retries failed/interrupted, never completed runs


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


def older_layout_conflict(older: ExistingRun | None, source: Path, destripe: bool) -> str | None:
    """Why output found directly in ``run-XX/`` blocks this variant, or None if it doesn't.

    Output there predates the with_/without_destriping subfolders. It blocks the
    variant it belongs to (or both, if its record does not say), and any run whose
    folder it shares with another TIFF.
    """
    if older is None:
        return None
    where = f"{older.run_dir.name}/ ({older.evidence})"
    if older.source != source:
        return f"{where} holds older output of {older.source.name}; move it out of the run folder"
    enabled = older.metadata.get("destriping_enabled")
    if enabled is None:
        return (
            f"older output of this TIFF sits directly in {where} and its destriping is "
            "unknown; check it and put it in the variant subfolder it belongs to"
        )
    if enabled != destripe:
        return None
    variant = variant_output_dir(Path(), enabled).name
    return (
        f"older output of this TIFF sits directly in {where}; move it into {variant}/ to "
        "keep it, or out of the run folder to redo it"
    )


def decide_action(
    source: Path,
    identity: dict,
    run_dir: Path,
    run_dir_exists: bool,
    prior_elsewhere: ExistingRun | None,
    prior_here: ExistingRun | None,
    retry_incomplete: bool,
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
        return "skip", f"matching older output found via {prior_here.evidence}"
    meta = prior_here.metadata
    if (meta.get("source_size_bytes"), meta.get("source_mtime_ns")) != (
        identity["source_size_bytes"],
        identity["source_mtime_ns"],
    ):
        return "conflict", "source path matches but file size/mtime changed"
    status = meta.get("status")
    if status == "complete":
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


def plan_batch(batch: BatchSettings) -> list[PlannedRun]:
    subdir = variant_output_dir(Path(), batch.destripe).name
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
            run_dir = derivative_funcimg / run_name / subdir
            identity = source_identity(source)
            resolved = source.resolve()
            older = read_existing_run(run_dir.parent) if run_dir.parent.is_dir() else None
            blocked = older_layout_conflict(older, resolved, batch.destripe)
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


def snapshot_config(config_path: Path, destination_dir: Path) -> Path:
    """Freeze the config for this launch so later edits cannot reach running workers."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    destination_dir.mkdir(parents=True, exist_ok=True)
    snapshot = destination_dir / f"batch_config_{stamp}.toml"
    with snapshot.open("xb") as frozen:  # never replaces a snapshot another launch uses
        frozen.write(Path(config_path).read_bytes())
    return snapshot


def run_batch(batch: BatchSettings, config_path: Path) -> list[PlannedRun]:
    """Plan afresh from the manifests on disk, then run every pending source.

    Planning here rather than reusing an earlier plan means relaunching after a
    failure never reprocesses runs that completed in the meantime.
    """
    plan = plan_batch(batch)
    conflicts = [p for p in plan if p.action == "conflict"]
    if conflicts:
        details = "\n".join(f"  {p.run_dir}: {p.reason}" for p in conflicts)
        raise RuntimeError(f"Resolve run-directory conflicts before launching:\n{details}")
    pending = [p for p in plan if p.action == "run"]
    if not pending:
        log.info("Nothing to run.")
        return plan
    frozen_config = snapshot_config(config_path, batch.derivatives_subject_dir)
    log.info("Workers use config snapshot %s", frozen_config)
    for index, row in enumerate(pending, start=1):
        log.info("[%d/%d] %s -> %s", index, len(pending), row.source, row.run_dir)
        try:
            run_one(row, frozen_config, batch.destripe)
        except subprocess.CalledProcessError:
            log.error("FAILED: %s (log: %s)", row.source, row.run_dir / "preprocess.log")
            if batch.stop_on_error:
                raise
        else:
            log.info("COMPLETE: %s", row.run_dir)
    return plan
