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
    motion_source_for,
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
    assert f"{fraction:.0%}" in record["reason"]
    assert not (
        out / f"aligned_{'functional' if chosen == 'structural' else 'structural'}"
    ).exists()


def test_the_threshold_comes_from_the_settings(tmp_path):
    out, _ = variant_with_phase_record(tmp_path, 0.3)
    assert motion_source_for(out, settings())[0] == "functional"
    lenient = settings(auto_structural_min_fraction=0.25)
    assert motion_source_for(out, lenient)[0] == "structural"


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
    (out / "x_dynamic_bidiphase_destriped.phase.json").write_text(json.dumps({"chunks": []}))
    with pytest.raises(ValueError, match="per-frame"):
        run_registration(tiff, out, settings())
    assert suite2p_calls == []


def test_two_phase_records_are_ambiguous(tmp_path):
    out, _ = variant_with_phase_record(tmp_path, 0.9)
    record = {"phase_method": "per_frame", "fraction_structural": 0.1}
    (out / "other_dynamic_bidiphase_destriped.phase.json").write_text(json.dumps(record))
    with pytest.raises(FileNotFoundError, match="needs one"):
        motion_source_for(out, settings())


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
