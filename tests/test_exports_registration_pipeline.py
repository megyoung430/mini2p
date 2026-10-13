import json
import re

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from mini2p_preproc import pipeline as pipeline_module  # noqa: E402
from mini2p_preproc.config import PipelineSettings  # noqa: E402
from mini2p_preproc.contrast import scale_contrast  # noqa: E402
from mini2p_preproc.exports import (  # noqa: E402
    Contrast,
    export_mean_images,
    find_run_planes,
    require_run_count,
    write_comparison_pdf,
    write_high_resolution_pdf,
)
from mini2p_preproc.registration import check_existing_suite2p_folder  # noqa: E402
from tests.conftest import write_interleaved_tiff  # noqa: E402


def fake_plane(root, session, run, rng, subdir=""):
    plane = (
        root / f"ses-{session:03d}_x" / "funcimg" / f"run-{run:02d}" / subdir / "suite2p" / "plane0"
    )
    plane.mkdir(parents=True)
    ops = {"meanImg": rng.normal(100, 10, (12, 16)), "meanImg_chan2": rng.normal(50, 5, (12, 16))}
    np.save(plane / "ops.npy", ops, allow_pickle=True)
    return plane


def test_scale_contrast_maps_percentiles_to_unit_range(rng):
    image = rng.normal(size=(20, 20))
    display, black, white = scale_contrast(image, 5, 95, 1.0)
    assert display.min() == 0 and display.max() == 1
    assert black == pytest.approx(np.percentile(image, 5))
    with pytest.raises(ValueError, match="2D"):
        scale_contrast(np.zeros((2, 3, 4)), 5, 95, 1)


def test_find_run_planes_orders_and_respects_subdir(tmp_path, rng):
    fake_plane(tmp_path, 7, 10, rng)
    fake_plane(tmp_path, 7, 2, rng)
    fake_plane(tmp_path, 7, 3, rng, subdir="without_destriping")
    fake_plane(tmp_path, 99, 1, rng)
    runs = find_run_planes(tmp_path, 1, 10)
    assert [(r.session, r.run) for r in runs] == [(7, 2), (7, 10)]
    assert [r.run for r in find_run_planes(tmp_path, 1, 10, "without_destriping")] == [3]
    with pytest.raises(RuntimeError, match="Expected 3 runs"):
        require_run_count(runs, 3)


def test_export_writes_files_and_never_overwrites_by_default(tmp_path, rng):
    plane = fake_plane(tmp_path, 1, 1, rng)
    states = export_mean_images(plane, Contrast())
    assert set(states.values()) == {"saved"}
    settings = json.loads((plane / "meanImg_fixed_contrast_settings.json").read_text())
    assert settings["green_functional_physical_channel_2"]["suite2p_key"] == "meanImg"
    assert set(export_mean_images(plane, Contrast()).values()) == {"existing"}


def page_count(pdf_path):
    return len(re.findall(rb"/Type\s*/Page\b", pdf_path.read_bytes()))


def test_pdfs_have_expected_pages(tmp_path, rng):
    for run in (1, 2):
        export_mean_images(fake_plane(tmp_path, 1, run, rng), Contrast())
    runs = find_run_planes(tmp_path, 1, 1)
    small = write_comparison_pdf(runs, tmp_path / "a.pdf", Contrast(), "t")
    large = write_high_resolution_pdf(runs, tmp_path / "b.pdf", Contrast(), "t", dpi=50)
    assert page_count(small) == 2
    assert page_count(large) == 4
    with pytest.raises(FileExistsError):
        write_comparison_pdf(runs, small, Contrast(), "t")


def test_existing_suite2p_folder_from_other_file_is_refused(tmp_path, rng):
    tiff = write_interleaved_tiff(tmp_path / "x.tif", [np.zeros((2, 4, 6), np.int16)] * 2)
    (tmp_path / "suite2p").mkdir()
    np.save(tmp_path / "suite2p" / "db.npy", {"file_list": ["other.tif"]}, allow_pickle=True)
    with pytest.raises(RuntimeError, match="different or mixed"):
        check_existing_suite2p_folder(tmp_path, tiff, "suite2p")


def test_pipeline_wires_stages_together(tmp_path, recording, monkeypatch):
    raw_tiff, _ = recording
    calls = {}

    def fake_registration(tiff, output_dir, settings):
        calls["tiff"] = tiff
        return output_dir / "suite2p" / "plane0"

    monkeypatch.setattr(pipeline_module, "run_registration", fake_registration)
    settings = PipelineSettings.model_validate(
        {"phase": {"chunk_frames": 20, "max_shift_px": 8}, "distortion": {"mode": "off"}}
    )
    out = tmp_path / "run-01" / "without_destriping"
    result = pipeline_module.run_pipeline(raw_tiff, out, settings)
    assert calls["tiff"] == result.corrected_tiff  # distortion off: register the corrected TIFF
    meta = json.loads(result.corrected_tiff.with_suffix(".phase.json").read_text())
    assert meta["run_name"] == "run-01"
    assert [c["first_phase_px"] for c in meta["chunks"]] == [3, -2, 5]


def test_cluster_mode_requires_existing_undistorted_tiff(tmp_path, recording, monkeypatch):
    raw_tiff, _ = recording
    monkeypatch.setattr(pipeline_module, "run_registration", lambda *a: None)
    settings = PipelineSettings.model_validate(
        {
            "phase": {"chunk_frames": 20, "max_shift_px": 8},
            "distortion": {"mode": "cluster_full"},
        }
    )
    with pytest.raises(FileNotFoundError, match="Cluster distortion output"):
        pipeline_module.run_pipeline(raw_tiff, tmp_path / "out", settings)


def test_refused_overwrite_keeps_settings_record_consistent(tmp_path, recording, monkeypatch):
    raw_tiff, _ = recording
    monkeypatch.setattr(pipeline_module, "run_registration", lambda tiff, out, s: out)
    first = PipelineSettings.model_validate(
        {"phase": {"chunk_frames": 20, "max_shift_px": 8}, "distortion": {"mode": "off"}}
    )
    out = tmp_path / "run-01"
    pipeline_module.run_pipeline(raw_tiff, out, first)
    refused = first.model_copy(
        update={"overwrite": False, "phase": first.phase.model_copy(update={"chunk_frames": 10})}
    )
    with pytest.raises(FileExistsError):
        pipeline_module.run_pipeline(raw_tiff, out, refused)
    record = json.loads((out / pipeline_module.SETTINGS_RECORD).read_text())
    assert record["phase"]["chunk_frames"] == 20
