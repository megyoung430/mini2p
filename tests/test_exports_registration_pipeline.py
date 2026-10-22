import json
import re

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.image  # noqa: E402

from mini2p_preprocessing import pipeline as pipeline_module  # noqa: E402
from mini2p_preprocessing.config import PipelineSettings  # noqa: E402
from mini2p_preprocessing.contrast import scale_contrast  # noqa: E402
from mini2p_preprocessing.exports import (  # noqa: E402
    Contrast,
    export_mean_images,
    find_run_planes,
    load_contrast_choice,
    require_run_count,
    save_contrast_choice,
    with_saved_contrast,
    write_comparison_pdf,
    write_high_resolution_pdf,
)
from mini2p_preprocessing.registration import check_existing_suite2p_folder  # noqa: E402
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
    out = tmp_path / "run-01" / "without_distortion_correction" / "without_destriping"
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


def test_both_qc_variants_coexist_with_correct_run_name(tmp_path, recording, monkeypatch):
    from mini2p_preprocessing.config import variant_output_dir, with_destriping

    raw_tiff, _ = recording
    monkeypatch.setattr(pipeline_module, "run_registration", lambda tiff, out, s: out)
    base = PipelineSettings.model_validate(
        {"phase": {"chunk_frames": 20, "max_shift_px": 8}, "distortion": {"mode": "off"}}
    )
    run_dir = tmp_path / "run-02"
    for enabled in (False, True):
        pipeline_module.run_pipeline(
            raw_tiff, variant_output_dir(run_dir, enabled, False), with_destriping(base, enabled)
        )
    sidecars = sorted(run_dir.glob("without_distortion_correction/*/*.phase.json"))
    assert [p.parent.name for p in sidecars] == ["with_destriping", "without_destriping"]
    metas = [json.loads(p.read_text()) for p in sidecars]
    assert [m["run_name"] for m in metas] == ["run-02", "run-02"]
    assert [m["destriping_enabled"] for m in metas] == [True, False]


def test_display_with_saved_contrast_reproduces_the_saved_png(tmp_path, rng):
    green_raw, red_raw = rng.gamma(2, 50, (2, 20, 24))
    green, red = Contrast(2.0, 99.0, 0.8), Contrast(1.0, 99.8, 1.5)
    paths = save_contrast_choice(tmp_path, green_raw, red_raw, green, red)
    assert load_contrast_choice(tmp_path) == {"green": green, "red": red}
    png = matplotlib.image.imread(paths["green PNG"])[..., 0]
    shown = with_saved_contrast(tmp_path, green_raw, "green")
    assert np.allclose(png, shown, atol=2 / 255)  # 8-bit grey levels


def test_saved_contrast_refuses_a_different_mean_image(tmp_path, rng):
    green_raw, red_raw = rng.gamma(2, 50, (2, 20, 24))
    save_contrast_choice(tmp_path, green_raw, red_raw, Contrast(), Contrast())
    with pytest.raises(ValueError, match="different mean image"):
        with_saved_contrast(tmp_path, green_raw * 1.5, "green")


def test_missing_contrast_choice_names_the_cell_that_writes_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="Save the contrast"):
        with_saved_contrast(tmp_path, np.ones((4, 4)), "green")


def test_each_variant_gets_its_own_summary_and_pdf_names(tmp_path):
    from mini2p_preprocessing.exports import ExportSettings

    common = {"subject_dir": tmp_path, "session_first": 1, "session_last": 4, "expected_runs": 1}
    labels = {
        sub: ExportSettings(**common, run_subdir=sub).session_label
        for sub in ("without_destriping", "with_destriping")
    }
    assert labels == {
        "without_destriping": "sessions-1-4_without_destriping",
        "with_destriping": "sessions-1-4_with_destriping",
    }
    with pytest.raises(ValueError, match="run_subdir"):  # never export into run-XX/suite2p
        ExportSettings(**common, run_subdir="")


def test_export_summary_is_named_by_label(tmp_path, rng):
    from mini2p_preprocessing.exports import export_all_mean_images

    export_mean_images(fake_plane(tmp_path, 1, 1, rng, "with_destriping"), Contrast())
    runs = find_run_planes(tmp_path, 1, 1, "with_destriping")
    path = export_all_mean_images(runs, tmp_path, "sessions-1-1_with_destriping", Contrast())
    assert path.name == "meanImg_export_summary_sessions-1-1_with_destriping.json"


def test_output_folder_must_match_the_destriping_setting(tmp_path, recording, monkeypatch):
    raw_tiff, _ = recording
    monkeypatch.setattr(pipeline_module, "run_registration", lambda tiff, out, s: out)
    off = PipelineSettings.model_validate(
        {"phase": {"chunk_frames": 20, "max_shift_px": 8}, "distortion": {"mode": "off"}}
    )
    with pytest.raises(ValueError, match="with_destriping.*destriping is off"):
        pipeline_module.run_pipeline(raw_tiff, tmp_path / "run-01" / "with_destriping", off)
    assert not (tmp_path / "run-01").exists()


def _registered_folder(tmp_path, tiff_name="x.tif", shape=(4, 6)):
    """A variant folder, run-01/without_destriping, holding an earlier registration."""
    import tifffile

    out = tmp_path / "run-01" / "without_destriping"
    plane = out / "suite2p" / "plane0"
    plane.mkdir(parents=True)
    tiff = out / tiff_name
    tifffile.imwrite(tiff, np.zeros((2, *shape), np.int16))
    np.save(out / "suite2p" / "db.npy", {"file_list": [tiff_name]}, allow_pickle=True)
    np.save(plane / "ops.npy", {"Ly": shape[0], "Lx": shape[1]}, allow_pickle=True)
    (plane / "data.bin").write_bytes(b"old registration")
    return tiff, plane


def _stub_suite2p(monkeypatch):
    import importlib

    module = importlib.import_module("suite2p.run_s2p")  # the package re-exports a function
    calls = []
    monkeypatch.setattr(module, "run_s2p", lambda settings, db: calls.append(db))
    return calls


def test_overwrite_replaces_an_earlier_registration(tmp_path, monkeypatch):
    # Suite2p reuses an existing binary and skips registration, so it must be cleared.
    from mini2p_preprocessing.registration import run_registration

    tiff, plane = _registered_folder(tmp_path)
    calls = _stub_suite2p(monkeypatch)
    settings = PipelineSettings.model_validate({"distortion": {"mode": "off"}})
    run_registration(tiff, tiff.parent, settings)
    assert len(calls) == 1 and not (plane / "data.bin").exists()


def test_earlier_registration_is_kept_without_overwrite_or_with_masks(tmp_path, monkeypatch):
    from mini2p_preprocessing.registration import run_registration

    tiff, plane = _registered_folder(tmp_path)
    calls = _stub_suite2p(monkeypatch)
    keep = PipelineSettings.model_validate({"overwrite": False, "distortion": {"mode": "off"}})
    with pytest.raises(FileExistsError, match="overwrite"):
        run_registration(tiff, tiff.parent, keep)
    (plane / "meanImg_seg.npy").write_bytes(b"masks")
    replace = keep.model_copy(update={"overwrite": True})
    with pytest.raises(RuntimeError, match="meanImg_seg.npy"):
        run_registration(tiff, tiff.parent, replace)
    assert calls == [] and (plane / "data.bin").exists()


def test_registration_outside_a_variant_folder_is_never_cleared(tmp_path, monkeypatch):
    # run-XX/suite2p holds the registration the Cellpose masks were drawn on.
    import shutil

    from mini2p_preprocessing.registration import run_registration

    tiff, plane = _registered_folder(tmp_path)
    run_dir = tmp_path / "run-01"
    shutil.move(str(plane.parent), run_dir / "suite2p")
    shutil.move(str(tiff), run_dir / tiff.name)
    calls = _stub_suite2p(monkeypatch)
    settings = PipelineSettings.model_validate({"distortion": {"mode": "off"}})
    with pytest.raises(RuntimeError, match="variant"):
        run_registration(run_dir / tiff.name, run_dir, settings)
    assert calls == [] and (run_dir / "suite2p" / "plane0" / "data.bin").exists()


@pytest.mark.parametrize("folder", ["", ".", "..", "a/b", "/abs"])
def test_suite2p_save_folder_must_be_a_plain_name(folder):
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="save_folder"):
        PipelineSettings.model_validate(
            {"registration": {"save_folder": folder}, "distortion": {"mode": "off"}}
        )


def test_fast_disk_gives_each_run_its_own_local_binary_folder(tmp_path):
    from mini2p_preprocessing.registration import build_suite2p_db

    output_dir = (
        tmp_path / "share" / "ses-224_date-20260809" / "funcimg" / "run-01" / "with_destriping"
    )
    tiff = output_dir / "a_undistorted.tif"
    plain = PipelineSettings.model_validate({"distortion": {"mode": "off"}})
    assert "fast_disk" not in build_suite2p_db(output_dir, tiff, plain)
    local = PipelineSettings.model_validate(
        {"distortion": {"mode": "off"}, "registration": {"fast_disk": str(tmp_path / "local")}}
    )
    db = build_suite2p_db(output_dir, tiff, local)
    assert db["fast_disk"] == str(
        tmp_path / "local" / "ses-224_date-20260809_funcimg_run-01_with_destriping"
    )
    assert db["save_path0"] == str(output_dir)  # results still go beside the run
