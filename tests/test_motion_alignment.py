"""motion_alignment_source = "auto": the channel Suite2p aligns on, chosen per run from how
often the per-frame bidiphase step trusted the structural channel, and the aligned_<channel>
folder its output goes in."""

import importlib
import json

import numpy as np
import pytest
import tifffile
from pydantic import ValidationError

from mini2p_preprocessing.config import PipelineSettings
from mini2p_preprocessing.registration import (
    MOTION_ALIGNMENT_RECORD,
    build_suite2p_settings,
    choose_motion_source,
    motion_choice,
    run_registration,
)

VARIANT = ("run-01", "with_distortion_correction", "with_destriping")


def settings(**registration) -> PipelineSettings:
    return PipelineSettings.model_validate(
        {
            "distortion": {"mode": "cluster_full"},
            "destripe": {"enabled": True},
            "registration": registration,
        }
    )


def variant_with_phase_record(tmp_path, fraction_structural: float, method="per_frame"):
    """A variant folder as the phase step leaves it: the corrected TIFF and its .phase.json."""
    out = tmp_path.joinpath(*VARIANT)
    out.mkdir(parents=True)
    tiff = out / "71rtask_00036_dynamic_bidiphase_destriped_undistorted.tif"
    tifffile.imwrite(tiff, np.zeros((2, 4, 6), np.int16))
    record = {"phase_method": method, "fraction_structural": fraction_structural}
    (out / "71rtask_00036_dynamic_bidiphase_destriped.phase.json").write_text(json.dumps(record))
    return out, tiff


@pytest.fixture
def suite2p_calls(monkeypatch):
    module = importlib.import_module("suite2p.run_s2p")
    calls = []
    monkeypatch.setattr(
        module, "run_s2p", lambda settings, db: calls.append((settings["registration"], db))
    )
    return calls


@pytest.mark.parametrize(
    ("fraction", "threshold", "expected"),
    [(0.9, 0.5, "structural"), (0.5, 0.5, "structural"), (0.49, 0.5, "functional"),
     (0.0, 0.5, "functional"), (1.0, 1.0, "structural")],
)  # fmt: skip
def test_structural_is_chosen_when_trusted_often_enough(fraction, threshold, expected):
    assert choose_motion_source(fraction, threshold) == expected


def test_a_fraction_outside_zero_to_one_is_refused():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        choose_motion_source(1.5, 0.5)


@pytest.mark.parametrize(("fraction", "chosen"), [(0.8, "structural"), (0.1, "functional")])
def test_auto_registers_into_the_folder_of_the_channel_it_chose(
    tmp_path, suite2p_calls, fraction, chosen
):
    out, tiff = variant_with_phase_record(tmp_path, fraction)
    plane = run_registration(tiff, out, settings())
    assert plane == out / f"aligned_{chosen}" / "suite2p" / "plane0"
    ((registration, db),) = suite2p_calls
    assert registration["align_by_chan2"] is (chosen == "structural")
    assert db["save_path0"] == str(out / f"aligned_{chosen}")
    record = json.loads((out / f"aligned_{chosen}" / MOTION_ALIGNMENT_RECORD).read_text())
    assert record["motion_alignment_source"] == chosen
    assert record["fraction_structural"] == fraction
    assert record["auto_structural_min_fraction"] == 0.5
    assert record["suite2p_input"] == str(tiff) and record["size_bytes"] == tiff.stat().st_size
    assert not (
        out / f"aligned_{'functional' if chosen == 'structural' else 'structural'}"
    ).exists()


def test_the_threshold_comes_from_the_settings(tmp_path):
    _, tiff = variant_with_phase_record(tmp_path, 0.3)
    assert motion_choice(tiff, settings()).source == "functional"
    lenient = settings(auto_structural_min_fraction=0.25)
    assert motion_choice(tiff, lenient).source == "structural"


def test_a_fixed_channel_ignores_the_phase_record(tmp_path, suite2p_calls):
    out, tiff = variant_with_phase_record(tmp_path, 0.9)
    plane = run_registration(tiff, out, settings(motion_alignment_source="functional"))
    assert plane.parents[1].name == "aligned_functional"
    assert json.loads((plane.parents[1] / MOTION_ALIGNMENT_RECORD).read_text())["reason"] == (
        "set in the config"
    )


def test_auto_without_a_per_frame_record_fails_before_suite2p(tmp_path, suite2p_calls):
    out, tiff = variant_with_phase_record(tmp_path, 0.9)
    next(out.glob("*.phase.json")).unlink()
    with pytest.raises(FileNotFoundError, match="phase.json"):
        run_registration(tiff, out, settings())
    chunked = {"chunks": []}
    (out / "71rtask_00036_dynamic_bidiphase_destriped.phase.json").write_text(json.dumps(chunked))
    with pytest.raises(ValueError, match="per-frame"):
        run_registration(tiff, out, settings())
    assert suite2p_calls == []


def test_the_record_is_the_one_of_the_tiff_being_registered(tmp_path):
    out, tiff = variant_with_phase_record(tmp_path, 0.9)
    record = {"phase_method": "per_frame", "fraction_structural": 0.1}
    (out / "other_dynamic_bidiphase_destriped.phase.json").write_text(json.dumps(record))
    assert motion_choice(tiff, settings()).source == "structural"  # not other's 10%
    corrected = out / "71rtask_00036_dynamic_bidiphase_destriped.tif"  # distortion off
    assert motion_choice(corrected, settings()).phase_record == (
        "71rtask_00036_dynamic_bidiphase_destriped.phase.json"
    )
    without_destriping = settings().model_copy(
        update={"destripe": settings().destripe.model_copy(update={"enabled": False})}
    )
    with pytest.raises(ValueError, match="not a corrected TIFF"):
        motion_choice(tiff, without_destriping)


def set_fraction(out, fraction):
    record = {"phase_method": "per_frame", "fraction_structural": fraction}
    (out / "71rtask_00036_dynamic_bidiphase_destriped.phase.json").write_text(json.dumps(record))


def test_switching_channel_on_a_rewritten_tiff_removes_the_stale_registration(
    tmp_path, suite2p_calls
):
    out, tiff = variant_with_phase_record(tmp_path, 0.8)
    run_registration(tiff, out, settings())
    (out / "aligned_structural" / "suite2p" / "plane0").mkdir(parents=True)
    tifffile.imwrite(tiff, np.ones((3, 4, 6), np.int16))  # the phase step ran again
    set_fraction(out, 0.2)
    run_registration(tiff, out, settings())
    assert not (out / "aligned_structural").exists()
    assert (out / "aligned_functional" / MOTION_ALIGNMENT_RECORD).is_file()


def test_both_alignments_of_the_same_tiff_are_kept(tmp_path, suite2p_calls):
    out, tiff = variant_with_phase_record(tmp_path, 0.8)
    run_registration(tiff, out, settings(motion_alignment_source="structural"))
    run_registration(tiff, out, settings(motion_alignment_source="functional"))
    assert (out / "aligned_structural" / MOTION_ALIGNMENT_RECORD).is_file()
    assert (out / "aligned_functional" / MOTION_ALIGNMENT_RECORD).is_file()


def test_a_stale_registration_is_only_removed_with_overwrite_and_without_masks(
    tmp_path, suite2p_calls
):
    out, tiff = variant_with_phase_record(tmp_path, 0.2)
    stale = out / "aligned_structural" / "suite2p" / "plane0"
    stale.mkdir(parents=True)  # no record: it cannot be shown to match this TIFF
    keep = settings().model_copy(update={"overwrite": False})
    with pytest.raises(FileExistsError, match="another TIFF.*overwrite"):
        run_registration(tiff, out, keep)
    (stale / "meanImg_seg.npy").write_bytes(b"masks")
    with pytest.raises(RuntimeError, match="Cellpose masks"):
        run_registration(tiff, out, settings())
    assert suite2p_calls == [] and stale.is_dir()


def test_a_failed_registration_leaves_no_record(tmp_path, monkeypatch):
    out, tiff = variant_with_phase_record(tmp_path, 0.8)
    module = importlib.import_module("suite2p.run_s2p")
    monkeypatch.setattr(module, "run_s2p", lambda settings, db: None)
    run_registration(tiff, out, settings())

    def fail(settings, db):
        raise RuntimeError("Suite2p crashed")

    monkeypatch.setattr(module, "run_s2p", fail)
    with pytest.raises(RuntimeError, match="crashed"):
        run_registration(tiff, out, settings())
    assert not (out / "aligned_structural" / MOTION_ALIGNMENT_RECORD).exists()


def test_auto_needs_the_per_frame_phase_method():
    with pytest.raises(ValidationError, match="per_frame"):
        PipelineSettings.model_validate(
            {"phase": {"method": "chunked"}, "distortion": {"mode": "off"}}
        )
    PipelineSettings.model_validate(
        {
            "phase": {"method": "chunked"},
            "distortion": {"mode": "off"},
            "registration": {"motion_alignment_source": "structural"},
        }
    )


def test_suite2p_settings_need_a_resolved_channel():
    with pytest.raises(ValueError, match="resolved"):
        build_suite2p_settings(settings())
    assert settings().registration_channel is None


def registered_on_both(tmp_path):
    out, tiff = variant_with_phase_record(tmp_path, 0.8)
    for source in ("structural", "functional"):
        run_registration(tiff, out, settings(motion_alignment_source=source))
        (out / f"aligned_{source}" / "suite2p" / "plane0").mkdir(parents=True)
    return out, tiff


def test_a_refused_run_keeps_the_record_and_both_registrations(tmp_path, suite2p_calls):
    out, tiff = registered_on_both(tmp_path)
    tifffile.imwrite(tiff, np.ones((3, 4, 6), np.int16))  # functional's record is now stale
    keep = settings(motion_alignment_source="structural").model_copy(update={"overwrite": False})
    with pytest.raises(FileExistsError, match="overwrite"):
        run_registration(tiff, out, keep)
    (out / "aligned_structural" / "suite2p" / "meanImg_seg.npy").write_bytes(b"masks")
    with pytest.raises(RuntimeError, match="Cellpose masks"):
        run_registration(tiff, out, settings(motion_alignment_source="structural"))
    for source in ("structural", "functional"):
        assert (out / f"aligned_{source}" / MOTION_ALIGNMENT_RECORD).is_file()
        assert (out / f"aligned_{source}" / "suite2p" / "plane0").is_dir()


def test_registrations_stay_valid_when_the_session_folder_moves(tmp_path, suite2p_calls):
    import shutil

    out, _ = registered_on_both(tmp_path / "share")
    moved = tmp_path / "laptop"
    shutil.move(str(tmp_path / "share"), moved)  # keeps size and mtime
    out = moved.joinpath(*VARIANT)
    tiff = out / "71rtask_00036_dynamic_bidiphase_destriped_undistorted.tif"
    run_registration(tiff, out, settings(motion_alignment_source="structural"))
    assert (out / "aligned_functional" / "suite2p" / "plane0").is_dir()
