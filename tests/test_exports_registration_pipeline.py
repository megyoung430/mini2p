import importlib
import json
import re
from pathlib import Path

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
    """run-XX/suite2p/plane0, or for a variant *subdir* its aligned_structural registration."""
    run_dir = root / f"ses-{session:03d}_x" / "funcimg" / f"run-{run:02d}"
    registration = run_dir / subdir / "aligned_structural" if subdir else run_dir
    plane = registration / "suite2p" / "plane0"
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
        {
            "phase": {"method": "chunked", "chunk_frames": 20, "max_shift_px": 8},
            "registration": {"motion_alignment_source": "functional"},  # "auto" needs per_frame
            "distortion": {"mode": "off"},
        }
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
            raw_tiff,
            variant_output_dir(run_dir, enabled, distortion_corrected=False),
            with_destriping(base, enabled),
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
    subdirs = (
        "with_distortion_correction/without_destriping",
        "with_distortion_correction/with_destriping",
    )
    labels = {sub: ExportSettings(**common, run_subdir=sub).session_label for sub in subdirs}
    assert labels == {
        subdirs[0]: "sessions-1-4_with_distortion_correction_without_destriping_aligned_auto",
        subdirs[1]: "sessions-1-4_with_distortion_correction_with_destriping_aligned_auto",
    }
    chosen = ExportSettings(**common, run_subdir=subdirs[0], alignment="structural")
    assert chosen.session_label.endswith("_without_destriping_aligned_structural")
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


# These tests are about clearing, so the channel is fixed rather than "auto".
STRUCTURAL = {
    "distortion": {"mode": "off"},
    "registration": {"motion_alignment_source": "structural"},
}


def _registered_folder(tmp_path, tiff_name="x.tif", shape=(4, 6), variant=True):
    """A variant folder holding an earlier registration aligned on the structural channel
    (or, with *variant* False, run-01 itself)."""
    import tifffile

    out = tmp_path / "run-01" / "without_distortion_correction" / "without_destriping"
    out = out if variant else tmp_path / "run-01"
    plane = out / "aligned_structural" / "suite2p" / "plane0"
    plane.mkdir(parents=True)
    tiff = out / tiff_name
    tifffile.imwrite(tiff, np.zeros((2, *shape), np.int16))
    np.save(plane.parent / "db.npy", {"file_list": [tiff_name]}, allow_pickle=True)
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
    settings = PipelineSettings.model_validate(STRUCTURAL)
    run_registration(tiff, tiff.parent, settings)
    assert len(calls) == 1 and not (plane / "data.bin").exists()


def test_earlier_registration_is_kept_without_overwrite_or_with_masks(tmp_path, monkeypatch):
    from mini2p_preprocessing.registration import run_registration

    tiff, plane = _registered_folder(tmp_path)
    calls = _stub_suite2p(monkeypatch)
    keep = PipelineSettings.model_validate({**STRUCTURAL, "overwrite": False})
    with pytest.raises(FileExistsError, match="overwrite"):
        run_registration(tiff, tiff.parent, keep)
    (plane / "meanImg_seg.npy").write_bytes(b"masks")
    replace = keep.model_copy(update={"overwrite": True})
    with pytest.raises(RuntimeError, match="meanImg_seg.npy"):
        run_registration(tiff, tiff.parent, replace)
    assert calls == [] and (plane / "data.bin").exists()


def test_registration_outside_a_variant_folder_is_never_cleared(tmp_path, monkeypatch):
    # run-XX/ holds the registration the Cellpose masks were drawn on; never clear there.
    from mini2p_preprocessing.registration import run_registration

    tiff, plane = _registered_folder(tmp_path, variant=False)
    calls = _stub_suite2p(monkeypatch)
    settings = PipelineSettings.model_validate(STRUCTURAL)
    with pytest.raises(RuntimeError, match="Not replacing"):
        run_registration(tiff, tiff.parent, settings)
    assert calls == [] and (plane / "data.bin").exists()


@pytest.mark.parametrize("folder", ["", ".", "..", "a/b", "/abs"])
def test_suite2p_save_folder_must_be_a_plain_name(folder):
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="save_folder"):
        PipelineSettings.model_validate(
            {"registration": {"save_folder": folder}, "distortion": {"mode": "off"}}
        )


def test_fast_disk_gives_each_run_its_own_local_binary_folder(tmp_path):
    from mini2p_preprocessing.registration import build_suite2p_db

    subject = tmp_path / "share" / "cohort-01" / "sub-03_id-MY_71_R"
    run_dir = subject / "ses-224_date-20260809" / "funcimg" / "run-01"
    output_dir = run_dir / "with_distortion_correction" / "with_destriping"
    tiff = output_dir / "a_undistorted.tif"
    plain = PipelineSettings.model_validate(STRUCTURAL)
    assert "fast_disk" not in build_suite2p_db(output_dir, tiff, plain)
    local = PipelineSettings.model_validate(
        {
            "distortion": {"mode": "off"},
            "registration": {
                "fast_disk": str(tmp_path / "local"),
                "motion_alignment_source": "functional",
            },
        }
    )
    db = build_suite2p_db(output_dir, tiff, local)
    assert db["fast_disk"] == str(
        tmp_path / "local" / "cohort-01_sub-03_id-MY_71_R_ses-224_date-20260809_funcimg_run-01_"
        "with_distortion_correction_with_destriping_aligned_functional"
    )
    assert db["save_path0"] == str(output_dir / "aligned_functional")  # results beside the run
    with pytest.raises(ValueError, match="resolved"):
        build_suite2p_db(
            output_dir, tiff, PipelineSettings.model_validate({"distortion": {"mode": "off"}})
        )


def test_per_frame_pipeline_saves_its_qc_plot(tmp_path, recording, monkeypatch):
    raw_tiff, _ = recording
    monkeypatch.setattr(
        pipeline_module, "run_registration", lambda tiff, out, s: out / "suite2p" / "plane0"
    )
    settings = PipelineSettings.model_validate(
        {
            "phase": {"max_shift_px": 8, "trust_window_frames": 11, "fallback_window_frames": 21},
            "distortion": {"mode": "off"},
        }
    )
    out = tmp_path / "run-01" / "without_distortion_correction" / "without_destriping"
    result = pipeline_module.run_pipeline(raw_tiff, out, settings)
    assert list(result.correction.frame_phases.phase_px[[5, 25, 50]]) == [3, -2, 5]
    for suffix in (".json", ".npz", ".png", ".pdf", ".svg", ".caption.txt"):
        path = result.corrected_tiff.with_suffix(".phase_frames" + suffix)
        if suffix == ".json":
            path = result.corrected_tiff.with_suffix(".phase.json")
        assert path.is_file(), path


def test_auto_alignment_reads_the_phase_record_the_pipeline_just_wrote(
    tmp_path, recording, monkeypatch
):
    module = importlib.import_module("suite2p.run_s2p")
    calls = []
    monkeypatch.setattr(module, "run_s2p", lambda settings, db: calls.append(settings))
    raw_tiff, _ = recording
    settings = PipelineSettings.model_validate(
        {
            "phase": {"max_shift_px": 8, "trust_window_frames": 11, "fallback_window_frames": 21},
            "distortion": {"mode": "off"},
        }
    )
    out = tmp_path / "run-01" / "without_distortion_correction" / "without_destriping"
    result = pipeline_module.run_pipeline(raw_tiff, out, settings)
    fraction = float(result.correction.frame_phases.used_structural.mean())
    chosen = "structural" if fraction >= 0.5 else "functional"
    assert result.plane_dir == out / f"aligned_{chosen}" / "suite2p" / "plane0"
    assert calls[0]["registration"]["align_by_chan2"] is (chosen == "structural")


def suite2p_binary_dir(db: dict) -> Path:
    """Where Suite2p itself puts the binaries for this db (it creates the folders)."""
    from suite2p.io.utils import init_dbs

    return Path(init_dbs(dict(db))[0]["reg_file"]).parent


@pytest.mark.parametrize("keep_raw", [False, True])
def test_each_save_folder_keeps_its_binaries_inside_it(tmp_path, keep_raw):
    # Suite2p writes its binaries in <fast_disk>/suite2p/plane0 whatever save_folder is; each
    # registration of a variant must still have binaries of its own
    from suite2p.io.utils import init_dbs

    from mini2p_preprocessing.registration import build_suite2p_db

    output_dir = tmp_path / "run-02" / "without_distortion_correction" / "with_destriping"
    tiff = output_dir / "a_destriped.tif"
    aligned = output_dir / "aligned_structural"
    folders = {}
    for save_folder in ("suite2p", "suite2p_rigid_sigma3", "suite2p_withoutnonrigid_sigma1.15"):
        registration = {
            **STRUCTURAL["registration"],
            "save_folder": save_folder,
            "keep_raw_binary": keep_raw,
        }
        settings = PipelineSettings.model_validate({**STRUCTURAL, "registration": registration})
        (db,) = init_dbs(dict(build_suite2p_db(output_dir, tiff, settings)))
        keys = ("reg_file", "reg_file_chan2", "raw_file", "raw_file_chan2")
        files = [Path(db[key]) for key in keys if key in db]
        assert len(files) == (4 if keep_raw else 2)
        assert all(f.is_relative_to(aligned / save_folder) for f in files)
        folders[save_folder] = files[0].parent
    assert folders["suite2p"] == aligned / "suite2p" / "plane0"  # unchanged for the default
    assert len(set(folders.values())) == len(folders)


def test_fast_disk_binaries_are_kept_apart_per_save_folder(tmp_path):
    from mini2p_preprocessing.registration import build_suite2p_db

    subject = tmp_path / "cohort-01" / "sub-03_id-MY_71_R"
    output_dir = subject / "ses-223_date-20260807" / "funcimg" / "run-02" / "with_destriping"
    tiff = output_dir / "a_destriped.tif"

    def db(save_folder):
        settings = PipelineSettings.model_validate(
            {
                "distortion": {"mode": "off"},
                "registration": {
                    "fast_disk": str(tmp_path / "local"),
                    "save_folder": save_folder,
                    "motion_alignment_source": "structural",
                },
            }  # fmt: skip
        )
        return build_suite2p_db(output_dir, tiff, settings)

    default, other = db("suite2p"), db("suite2p_rigid_sigma3")
    name = (
        "cohort-01_sub-03_id-MY_71_R_ses-223_date-20260807_funcimg_run-02_with_destriping_"
        "aligned_structural"
    )
    assert default["fast_disk"] == str(tmp_path / "local" / name)  # the default's name is kept
    assert other["fast_disk"] == str(tmp_path / "local" / f"{name}_suite2p_rigid_sigma3")
    assert suite2p_binary_dir(default) != suite2p_binary_dir(other)


def test_a_registration_never_reads_another_registrations_binary(tmp_path):
    from mini2p_preprocessing.registration import resolve_registered_binary

    aligned = tmp_path / "with_destriping" / "aligned_structural"
    plane = aligned / "suite2p_rigid_sigma3" / "plane0"
    plane.mkdir(parents=True)
    (aligned / "suite2p" / "plane0").mkdir(parents=True)
    (aligned / "suite2p" / "plane0" / "data.bin").write_bytes(b"other registration")
    with pytest.raises(FileNotFoundError, match="data.bin"):
        resolve_registered_binary({}, plane, "reg_file", "data.bin")
    own = aligned / "suite2p_rigid_sigma3" / "suite2p" / "plane0" / "data.bin"
    own.parent.mkdir(parents=True)
    own.write_bytes(b"this registration")
    assert resolve_registered_binary({}, plane, "reg_file", "data.bin") == own


def registration(aligned, save_folder, record=True):
    """A registration as this version leaves it (record=True: its motion_alignment.json) or as
    earlier versions did (no record; a non-default save_folder's binary in suite2p/plane0)."""
    from mini2p_preprocessing.registration import MOTION_ALIGNMENT_RECORD

    plane = aligned / save_folder / "plane0"
    plane.mkdir(parents=True)
    np.save(plane / "ops.npy", {})
    if save_folder == "suite2p":
        binary = plane / "data.bin"
    elif record:
        binary = aligned / save_folder / "suite2p" / "plane0" / "data.bin"
    else:
        binary = aligned / "suite2p" / "plane0" / "data.bin"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(b"frames")
    if record:
        (aligned / save_folder / MOTION_ALIGNMENT_RECORD).write_text("{}")
    return plane, binary


def age(path, seconds_before_binary, binary):
    import os

    t = binary.stat().st_mtime - seconds_before_binary
    os.utime(path, (t, t))


def test_an_old_default_registration_rewritten_by_a_sweep_stays_refused(tmp_path):
    # also after the sweep is registered again with binaries of its own
    from mini2p_preprocessing.registration import resolve_registered_binary

    aligned = tmp_path / "with_destriping" / "aligned_structural"
    default, binary = registration(aligned, "suite2p", record=False)
    age(default / "ops.npy", 600, binary)  # a sweep wrote over it 10 min after it registered
    with pytest.raises(RuntimeError, match="register this one again"):
        resolve_registered_binary({}, default, "reg_file", "data.bin")
    registration(aligned, "suite2p_sigma1.15")  # the sweep, registered again
    with pytest.raises(RuntimeError, match="written 600 s after"):
        resolve_registered_binary({}, default, "reg_file", "data.bin")


def test_an_old_sweep_reading_the_shared_binary_is_refused_unless_it_wrote_it_last(tmp_path):
    from mini2p_preprocessing.registration import resolve_registered_binary

    aligned = tmp_path / "with_destriping" / "aligned_structural"
    registration(aligned, "suite2p", record=False)
    sweep, shared = registration(aligned, "suite2p_sigma3", record=False)
    ops = {"reg_file": str(shared)}
    age(sweep / "ops.npy", 600, shared)
    with pytest.raises(RuntimeError):
        resolve_registered_binary(ops, sweep, "reg_file", "data.bin")
    age(sweep / "ops.npy", -30, shared)
    assert resolve_registered_binary(ops, sweep, "reg_file", "data.bin") == shared


@pytest.mark.parametrize(("lag", "refused"), [(1, False), (3, True)])
def test_the_tolerance_allows_coarse_file_times(tmp_path, lag, refused):
    from mini2p_preprocessing.registration import resolve_registered_binary

    aligned = tmp_path / "with_destriping" / "aligned_structural"
    default, binary = registration(aligned, "suite2p", record=False)
    age(default / "ops.npy", lag, binary)
    if refused:
        with pytest.raises(RuntimeError):
            resolve_registered_binary({}, default, "reg_file", "data.bin")
    else:
        assert resolve_registered_binary({}, default, "reg_file", "data.bin") == binary


def record_binary(save_folder_dir, key, binary):
    """The binary's identity in the registration's record, as run_registration writes it."""
    import json

    from mini2p_preprocessing.registration import MOTION_ALIGNMENT_RECORD, binary_identity

    path = save_folder_dir / MOTION_ALIGNMENT_RECORD
    record = json.loads(path.read_text())
    record.setdefault("binaries", {})[key] = binary_identity(binary)
    path.write_text(json.dumps(record))


def test_registrations_with_their_own_record_are_read_whatever_their_file_times(tmp_path):
    # e.g. copied without keeping file times
    from mini2p_preprocessing.registration import resolve_registered_binary

    aligned = tmp_path / "with_destriping" / "aligned_structural"
    default, binary = registration(aligned, "suite2p")
    sweep, sweep_binary = registration(aligned, "suite2p_sigma3")
    age(default / "ops.npy", 60, binary)
    age(sweep / "ops.npy", 60, sweep_binary)
    assert resolve_registered_binary({}, default, "reg_file", "data.bin") == binary
    assert resolve_registered_binary({}, sweep, "reg_file", "data.bin") == sweep_binary


def test_a_fast_disk_binary_must_be_the_one_its_registration_recorded(tmp_path):
    import os

    from mini2p_preprocessing.registration import resolve_registered_binary

    aligned = tmp_path / "with_destriping" / "aligned_structural"
    sweep, _ = registration(aligned, "suite2p_sigma3")
    (aligned / "suite2p_sigma3" / "suite2p").rename(tmp_path / "unused")  # binaries elsewhere
    local = tmp_path / "local" / "run_suite2p_sigma3" / "suite2p" / "plane0" / "data.bin"
    local.parent.mkdir(parents=True)
    local.write_bytes(b"frames")
    record_binary(aligned / "suite2p_sigma3", "reg_file", local)
    ops = {"reg_file": str(local)}
    age(sweep / "ops.npy", 60, local)  # the laptop's clock differs: no matter
    assert resolve_registered_binary(ops, sweep, "reg_file", "data.bin") == local
    local.write_bytes(b"another registration's frames")
    later = local.stat().st_mtime + 100
    os.utime(local, (later, later))
    with pytest.raises(RuntimeError, match="not the reg_file binary"):
        resolve_registered_binary(ops, sweep, "reg_file", "data.bin")


def test_a_copy_reads_its_own_binary_not_the_one_its_ops_names(tmp_path):
    # a laptop copy whose ops still names the share's binary, which is still there
    from mini2p_preprocessing.registration import resolve_registered_binary

    share, laptop = tmp_path / "share", tmp_path / "laptop"
    _, share_binary = registration(share / "aligned_structural", "suite2p_sigma3")
    plane, own = registration(laptop / "aligned_structural", "suite2p_sigma3")
    ops = {"reg_file": str(share_binary)}
    assert resolve_registered_binary(ops, plane, "reg_file", "data.bin") == own


def test_registrations_outside_an_aligned_folder_are_not_checked(tmp_path):
    # run-XX/suite2p, the registration masks were drawn on, never shared its binary
    from mini2p_preprocessing.registration import resolve_registered_binary

    plane = tmp_path / "run-01" / "suite2p" / "plane0"
    plane.mkdir(parents=True)
    np.save(plane / "ops.npy", {})
    (plane / "data.bin").write_bytes(b"frames")
    age(plane / "ops.npy", 60, plane / "data.bin")
    assert resolve_registered_binary({}, plane, "reg_file", "data.bin") == plane / "data.bin"


def test_a_binary_without_its_ops_file_is_not_checked(tmp_path):
    from mini2p_preprocessing.registration import check_binary_belongs

    binary = tmp_path / "data.bin"
    binary.write_bytes(b"frames")
    check_binary_belongs(binary, tmp_path / "ops.npy")


def test_a_record_without_the_binary_says_to_register_again(tmp_path):
    from mini2p_preprocessing.registration import resolve_registered_binary

    aligned = tmp_path / "with_destriping" / "aligned_structural"
    sweep, _ = registration(aligned, "suite2p_sigma3")
    (aligned / "suite2p_sigma3" / "suite2p").rename(tmp_path / "unused")
    local = tmp_path / "local" / "data.bin"
    local.parent.mkdir()
    local.write_bytes(b"frames")
    with pytest.raises(RuntimeError, match="records no reg_file binary"):
        resolve_registered_binary({"reg_file": str(local)}, sweep, "reg_file", "data.bin")
