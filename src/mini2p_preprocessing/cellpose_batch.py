"""Cellpose extraction for every batch run, with a summary and a mask-alignment review PDF.

The masks for ``run-XX`` are drawn on the registration in ``run-XX/suite2p`` (e.g. the
first frames of the recording); each variant subfolder, ``run-XX/<variant>``, has its
own full registration. The masks are aligned onto each variant before extraction.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from .alignment import load_reference_image
from .batch import read_existing_run
from .config import VARIANTS, PipelineSettings, run_dir_of
from .contrast import scale_contrast
from .exports import Contrast, find_run_planes, require_run_count
from .extraction import (
    EXTRACTION_RECORD,
    ExtractionRefused,
    file_sha256,
    load_cellpose_labels,
    run_cellpose_extraction,
)
from .plane import load_plane_traces
from .plots import plot_mask_alignment, plot_masks_on
from .progress import PROGRESS_EVERY_S, ProgressLog

log = logging.getLogger(__name__)

ALIGNMENT_FIELDS = (
    "dy",
    "dx",
    "rank_correlation_before",
    "rank_correlation_after",
    "quadrants_used",
    "dropped_labels",
)
# Statuses that need a look: no traces were written from these masks.
ATTENTION = ("failed", "no masks", "incomplete run", "other extraction")


# ---- analysis: decisions on one run (pure) ---------------------------------------


def summary_row(session: int, run: str, variant: str, plane_dir: Path, **fields) -> dict:
    row = {"session": session, "run": run, "variant": variant, "plane_dir": str(plane_dir)}
    row.update({"status": None, "error": None, "n_rois": None, "seg_file": None})
    row.update({"mask_image": None, "source_check": None, **dict.fromkeys(ALIGNMENT_FIELDS)})
    return {**row, **fields}


def row_from_record(row: dict, record: dict, status: str) -> dict:
    alignment = record.get("mask_alignment") or {}
    found = {key: alignment.get(key) for key in ALIGNMENT_FIELDS}
    return {**row, **found, "status": status, "n_rois": record.get("n_rois")}


def extraction_key(seg_file: Path, mask_image: Path, settings: PipelineSettings) -> dict:
    """What an extraction depends on, as stored in its record."""
    return {
        "seg_file": str(Path(seg_file).resolve()),
        "seg_sha256": file_sha256(seg_file),
        "mask_image": str(Path(mask_image).resolve()),
        "mask_image_sha256": file_sha256(mask_image),
        "fluorescence_settings": json.loads(settings.fluorescence.model_dump_json()),
        "diameter_px": settings.registration.diameter_px,
    }


def compare_record(record: dict | None, key: dict) -> str:
    """'none', 'current' (same inputs and settings), 'stale' (same files since
    changed, or other settings) or 'other' (different masks or no alignment)."""
    if record is None:
        return "none"
    if all(record.get(name) == value for name, value in key.items()):
        return "current"
    same_files = (record.get("seg_file"), record.get("mask_image")) == (
        key["seg_file"],
        key["mask_image"],
    )
    return "stale" if same_files else "other"


def source_mismatch(plane_source: Path | None, mask_source: Path | None) -> str | None:
    if plane_source is None or mask_source is None or plane_source == mask_source:
        return None
    return (
        f"The masks' registration is of {mask_source.name} but this plane is of "
        f"{plane_source.name}; check the run numbering"
    )


# ---- I/O: one run, all runs --------------------------------------------------------


def _read_record(plane_dir: Path) -> dict | None:
    path = Path(plane_dir) / EXTRACTION_RECORD
    return json.loads(path.read_text()) if path.is_file() else None


def extract_run(
    plane_dir: Path,
    run_dir: Path,
    seg_file: Path,
    mask_image: Path,
    settings: PipelineSettings,
    overwrite: bool,
    row: dict,
) -> dict:
    """Align and extract one variant plane. Refusals are reported in the row; anything
    else (a bug, out of memory) is raised."""
    row = {**row, "seg_file": str(seg_file), "mask_image": str(mask_image)}
    plane_run = read_existing_run(Path(plane_dir).parents[1])
    if plane_run is not None and plane_run.evidence == "manifest":
        status = plane_run.metadata.get("status")
        if status not in (None, "complete"):
            return {**row, "status": "incomplete run", "error": f"batch status {status}"}
    if not seg_file.is_file():
        return {**row, "status": "no masks"}
    if not mask_image.is_file():
        return {**row, "status": "failed", "error": f"Mask image not found: {mask_image}"}
    mask_run = read_existing_run(run_dir)
    plane_source = plane_run.source if plane_run else None
    mismatch = source_mismatch(plane_source, mask_run.source if mask_run else None)
    if mismatch:
        return {**row, "status": "failed", "error": mismatch}
    row["source_check"] = "same TIFF" if plane_run and mask_run else "unverified"

    record = _read_record(plane_dir)
    state = compare_record(record, extraction_key(seg_file, mask_image, settings))
    if state == "current" and not overwrite:
        return row_from_record(row, record, "existing")
    if state == "other" and not overwrite:
        return {
            **row,
            "status": "other extraction",
            "error": f"kept the extraction from {record.get('seg_file')}; overwrite to replace",
        }
    try:
        run_cellpose_extraction(seg_file, plane_dir, settings, mask_image=mask_image)
    except ExtractionRefused as refusal:
        log.error("Extraction refused for %s: %s", plane_dir, refusal)
        return {**row, "status": "failed", "error": str(refusal)}
    return row_from_record(row, _read_record(plane_dir), "extracted")


def extract_all(
    subject_dir: Path,
    first_session: int,
    last_session: int,
    variant: str,
    seg_relative: str,
    image_relative: str,
    settings: PipelineSettings,
    expected_runs: int,
    overwrite: bool = False,
) -> list[dict]:
    """Extract every registered ``run-XX/<variant>`` plane from ``run-XX/<seg_relative>``."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, not {variant!r}")
    runs = find_run_planes(subject_dir, first_session, last_session, variant)
    require_run_count(runs, expected_runs)
    rows = []
    progress = ProgressLog(f"Cellpose extraction ({variant})", len(runs), log.info)
    for run in sorted(runs, key=lambda r: (r.session, r.run)):
        run_dir = run_dir_of(run.plane_dir.parents[1])
        row = summary_row(run.session, run.run_name, variant, run.plane_dir)
        rows.append(
            extract_run(
                run.plane_dir,
                run_dir,
                run_dir / seg_relative,
                run_dir / image_relative,
                settings,
                overwrite,
                row,
            )
        )
        log.info("ses-%d %s %s: %s", run.session, run.run_name, variant, rows[-1]["status"])
        progress.step_done()
    return rows


def write_extraction_summary(rows: list[dict], path: Path) -> Path:
    statuses = sorted({r["status"] for r in rows})
    counts = {status: sum(r["status"] == status for r in rows) for status in statuses}
    payload = {"created_utc": datetime.now(UTC).isoformat(), "counts": counts, "runs": rows}
    Path(path).write_text(json.dumps(payload, indent=2))
    return Path(path)


def format_summary(rows: list[dict]) -> str:
    header = f"{'session':>7}  {'run':>6}  {'status':<16} ROIs  shift   r_after  source"
    lines = [header]
    for r in rows:
        shift = f"{r['dy']},{r['dx']}" if r["dy"] is not None else "-"
        r_after = r["rank_correlation_after"]
        after = f"{r_after:.2f}" if r_after is not None else "-"
        rois = r["n_rois"] if r["n_rois"] is not None else "-"
        lines.append(
            f"{r['session']:>7}  {r['run']:>6}  {r['status']:<16} {rois:>4}  {shift:<6}  "
            f"{after:<7}  {r['source_check'] or '-'}"
        )
        if r["error"]:
            lines.append(f"{'':>17}{r['error']}")
    return "\n".join(lines)


# ---- review PDF: load, then draw ---------------------------------------------------


def load_review_images(row: dict, contrast: Contrast) -> dict:
    """What a review page shows, loaded from disk; missing pieces are None."""
    plane_dir = Path(row["plane_dir"])
    found = {"plane": None, "reference": None, "segmented": None, "extracted": None}
    ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
    if "meanImg" in ops:
        found["plane"], *_ = scale_contrast(
            ops["meanImg"], contrast.black_percentile, contrast.white_percentile, contrast.gamma
        )
    if row["seg_file"] and Path(row["seg_file"]).is_file():
        found["segmented"] = load_cellpose_labels(row["seg_file"])
    if row["mask_image"] and Path(row["mask_image"]).is_file():
        found["reference"] = load_reference_image(row["mask_image"])
    if row["status"] in {"extracted", "existing"}:
        found["extracted"] = load_plane_traces(plane_dir).labels
    return found


def draw_review_page(row: dict, images: dict):
    fig, axes = plt.subplots(1, 2, figsize=(11.69, 7.0), facecolor="white")
    shift = f"shift dy={row['dy']}, dx={row['dx']} px"
    if images["extracted"] is not None and images["segmented"] is not None:
        plot_mask_alignment(
            images["plane"], images["segmented"], images["extracted"], axes=axes, title=shift
        )
    elif images["segmented"] is not None and images["plane"] is not None:
        if images["reference"] is not None:
            plot_masks_on(images["reference"], images["segmented"], axes[0], "Masks as drawn")
        else:
            axes[0].axis("off")
        plot_masks_on(images["plane"], images["segmented"], axes[1], "Same masks, unshifted")
    else:
        for ax in axes:
            ax.axis("off")
    if row["error"]:
        fig.text(0.5, 0.04, row["error"], ha="center", wrap=True, fontsize=10, color="firebrick")
    r_after = row["rank_correlation_after"]
    fig.suptitle(
        f"Session {row['session']} | {row['run']} | {row['variant']} | {row['status']}"
        + (f" | r = {r_after:.2f}" if r_after is not None else ""),
        fontsize=14,
        fontweight="bold",
    )
    return fig


def write_mask_alignment_pdf(
    rows: list[dict], output: Path, contrast: Contrast, title: str, overwrite: bool = False
) -> Path:
    """One page per run: masks as segmented and as shifted onto the variant's mean image;
    for a refused run, the masks on their own image and unshifted on this plane."""
    if not rows:
        raise ValueError("No runs to review")
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} already exists; pass overwrite=True to replace it")
    with PdfPages(output) as pdf:
        pdf.infodict()["Title"] = title
        progress = ProgressLog("Mask-alignment PDF", len(rows), log.info, PROGRESS_EVERY_S)
        for row in rows:
            fig = draw_review_page(row, load_review_images(row, contrast))
            pdf.savefig(fig)
            plt.close(fig)
            progress.step_done()
    return output
