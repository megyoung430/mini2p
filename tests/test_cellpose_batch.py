import json
import re

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from mini2p_preprocessing.batch import MANIFEST_NAME  # noqa: E402
from mini2p_preprocessing.cellpose_batch import (  # noqa: E402
    extract_all,
    write_extraction_summary,
    write_mask_alignment_pdf,
)
from mini2p_preprocessing.exports import Contrast  # noqa: E402
from mini2p_preprocessing.extraction import run_cellpose_extraction  # noqa: E402
from tests.conftest import as_contrast_png  # noqa: E402
from tests.test_rois_extraction import SETTINGS, label_image, make_plane, offset  # noqa: E402

SEG = "suite2p/plane0/meanImg_green_functional_contrast_seg.npy"
IMAGE = "suite2p/plane0/meanImg_green_functional_contrast.png"


def add_segmentation(run_dir, plane, dy, dx, image=None):
    """Masks and their image from the run's own (other) registration, offset by (dy, dx)."""
    reference = run_dir / "suite2p" / "plane0"
    reference.mkdir(parents=True)
    ops = np.load(plane / "ops.npy", allow_pickle=True).item()
    picture = offset(ops["meanImg"], dy, dx) if image is None else image
    as_contrast_png(picture, run_dir / IMAGE)
    np.save(run_dir / SEG, {"masks": offset(label_image(), dy, dx)}, allow_pickle=True)


@pytest.fixture
def derivatives(tmp_path, rng):
    funcimg = tmp_path / "der" / "ses-001_date-x" / "funcimg"
    planes = {}
    for run in ("run-01", "run-02", "run-03"):
        planes[run], _ = make_plane(funcimg / run / "without_destriping", rng, texture=True)
    add_segmentation(funcimg / "run-01", planes["run-01"], -2, 3)
    add_segmentation(funcimg / "run-03", planes["run-03"], 0, 0, image=rng.normal(size=(40, 56)))
    return tmp_path / "der", planes


def extract(subject_dir, overwrite=False, expected_runs=3):
    return extract_all(
        subject_dir, 1, 1, "without_destriping", SEG, IMAGE, SETTINGS, expected_runs, overwrite
    )


def write_manifest(folder, source):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / MANIFEST_NAME).write_text(json.dumps({"source_tiff": str(source)}))


def test_each_run_reports_its_alignment_or_why_it_has_none(derivatives):
    subject_dir, planes = derivatives
    rows = {row["run"]: row for row in extract(subject_dir)}
    assert rows["run-01"]["status"] == "extracted"
    assert (rows["run-01"]["dy"], rows["run-01"]["dx"]) == (2, -3)
    assert rows["run-01"]["n_rois"] == 3 and rows["run-01"]["rank_correlation_after"] > 0.5
    assert rows["run-02"]["status"] == "no masks"
    assert rows["run-03"]["status"] == "failed" and rows["run-03"]["error"]
    assert not (planes["run-03"] / "F.npy").exists()  # a refused alignment writes nothing


def test_extraction_is_reused_unless_overwritten(derivatives):
    subject_dir, planes = derivatives
    extract(subject_dir)
    written = (planes["run-01"] / "F.npy").stat().st_mtime_ns
    again = {row["run"]: row for row in extract(subject_dir)}
    assert again["run-01"]["status"] == "existing" and again["run-01"]["dx"] == -3
    assert (planes["run-01"] / "F.npy").stat().st_mtime_ns == written
    redone = {row["run"]: row for row in extract(subject_dir, overwrite=True)}
    assert redone["run-01"]["status"] == "extracted"


def test_summary_and_review_pdf_cover_every_run(derivatives, tmp_path):
    subject_dir, _ = derivatives
    rows = extract(subject_dir)
    summary = write_extraction_summary(rows, tmp_path / "summary.json")
    assert [r["status"] for r in json.loads(summary.read_text())["runs"]] == [
        "extracted",
        "no masks",
        "failed",
    ]
    pdf = write_mask_alignment_pdf(rows, tmp_path / "masks.pdf", Contrast(), "title")
    assert len(re.findall(rb"/Type\s*/Page\b", pdf.read_bytes())) == 3
    with pytest.raises(FileExistsError):
        write_mask_alignment_pdf(rows, pdf, Contrast(), "title")


def test_a_different_extraction_is_kept_unless_overwritten(derivatives):
    subject_dir, planes = derivatives
    plane = planes["run-01"]
    run_cellpose_extraction(plane.parents[1] / "meanImg_seg.npy", plane, SETTINGS)  # e.g. QC
    kept = {row["run"]: row for row in extract(subject_dir)}["run-01"]
    assert kept["status"] == "other extraction"
    assert json.loads((plane / "cellpose_extraction.json").read_text())["mask_image"] is None
    replaced = {row["run"]: row for row in extract(subject_dir, overwrite=True)}["run-01"]
    assert replaced["status"] == "extracted"


def test_resaved_masks_or_new_settings_are_extracted_again(derivatives):
    subject_dir, planes = derivatives
    extract(subject_dir)
    seg = planes["run-01"].parents[2] / SEG
    masks = np.load(seg, allow_pickle=True).item()["masks"]
    masks[masks == 7] = 0  # edited in the Cellpose GUI, saved under the same name
    np.save(seg, {"masks": masks}, allow_pickle=True)
    row = {r["run"]: r for r in extract(subject_dir)}["run-01"]
    assert row["status"] == "extracted" and row["n_rois"] == 2


def test_masks_from_another_recording_are_refused(derivatives, tmp_path):
    subject_dir, planes = derivatives
    run_dir = planes["run-01"].parents[2]
    write_manifest(run_dir / "without_destriping", tmp_path / "71rtask_00001.tif")
    write_manifest(run_dir, tmp_path / "71rtask_00002.tif")
    row = {r["run"]: r for r in extract(subject_dir)}["run-01"]
    assert row["status"] == "failed" and "71rtask_00002.tif" in row["error"]
    assert not (planes["run-01"] / "F.npy").exists()
    write_manifest(run_dir, tmp_path / "71rtask_00001.tif")
    row = {r["run"]: r for r in extract(subject_dir)}["run-01"]
    assert row["status"] == "extracted" and row["source_check"] == "same TIFF"
    assert {r["run"]: r for r in extract(subject_dir)}["run-03"]["source_check"] == "unverified"


def test_a_plane_without_a_mean_image_fails_alone(derivatives):
    subject_dir, planes = derivatives
    ops = np.load(planes["run-01"] / "ops.npy", allow_pickle=True).item()
    del ops["meanImg"]
    np.save(planes["run-01"] / "ops.npy", ops, allow_pickle=True)
    rows = {r["run"]: r for r in extract(subject_dir)}
    assert rows["run-01"]["status"] == "failed" and "meanImg" in rows["run-01"]["error"]
    assert rows["run-03"]["status"] == "failed"  # the loop carried on


def test_unexpected_errors_are_not_reported_as_refusals(derivatives, monkeypatch):
    from mini2p_preprocessing import extraction

    def broken(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(extraction, "extract_traces", broken)
    with pytest.raises(RuntimeError, match="out of memory"):
        extract(derivatives[0])


def test_run_count_and_variant_are_checked(derivatives):
    subject_dir, _ = derivatives
    with pytest.raises(RuntimeError, match="Expected 4 runs"):
        extract(subject_dir, expected_runs=4)
    with pytest.raises(ValueError, match="variant"):
        extract_all(subject_dir, 1, 1, "", SEG, IMAGE, SETTINGS, 3)


def test_review_pdf_needs_rows(tmp_path):
    with pytest.raises(ValueError, match="No runs"):
        write_mask_alignment_pdf([], tmp_path / "m.pdf", Contrast(), "t")


def test_review_page_survives_masks_deleted_after_extraction(derivatives, tmp_path):
    subject_dir, planes = derivatives
    rows = extract(subject_dir)  # rows from before the masks were deleted
    (planes["run-01"].parents[2] / SEG).unlink()
    pdf = write_mask_alignment_pdf(rows, tmp_path / "m.pdf", Contrast(), "t")
    assert pdf.is_file()
