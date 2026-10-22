"""Variant folders: with/without destriping, and runs without distortion correction nested
one level down in run-XX/without_distortion_correction/."""

from __future__ import annotations

from pathlib import Path

import pytest

from mini2p_preprocessing.config import (
    VARIANTS,
    PipelineSettings,
    check_variant_folder,
    run_dir_of,
    run_name_from_output_dir,
    settings_variant,
    variant_label,
    variant_output_dir,
    variant_subdir,
)
from mini2p_preprocessing.exports import ExportSettings, find_run_planes
from mini2p_preprocessing.registration import fast_disk_for

RUN = Path("/d/sub-03/ses-224_date-x/funcimg/run-01")


@pytest.mark.parametrize(
    ("destripe", "corrected", "expected"),
    [
        (False, True, "without_destriping"),
        (True, True, "with_destriping"),
        (False, False, "without_distortion_correction/without_destriping"),
        (True, False, "without_distortion_correction/with_destriping"),
    ],
)
def test_every_combination_has_its_own_folder(destripe, corrected, expected):
    assert variant_subdir(destripe, corrected) == expected
    assert variant_output_dir(RUN, destripe, corrected) == RUN / expected
    assert expected in VARIANTS
    assert run_dir_of(RUN / expected) == RUN
    assert run_name_from_output_dir(RUN / expected) == "run-01"
    check_variant_folder(RUN / expected, destripe, corrected)


def test_the_settings_choose_the_folder():
    off = PipelineSettings.model_validate(
        {"distortion": {"mode": "off"}, "destripe": {"enabled": True}}
    )
    assert settings_variant(off) == "without_distortion_correction/with_destriping"
    on = PipelineSettings.model_validate({"distortion": {"mode": "cluster_full"}})
    assert settings_variant(on) == "without_destriping"


@pytest.mark.parametrize(
    ("folder", "destripe", "corrected"),
    [
        ("without_destriping", False, False),  # distortion off belongs one level down
        ("without_distortion_correction/without_destriping", False, True),
        ("without_distortion_correction/with_destriping", False, False),
    ],
)
def test_a_folder_that_disagrees_with_the_settings_is_refused(folder, destripe, corrected):
    with pytest.raises(ValueError, match="which write to"):
        check_variant_folder(RUN / folder, destripe, corrected)


def test_a_plain_output_folder_is_not_checked():
    check_variant_folder(Path("/tmp/anything"), True, False)
    assert run_name_from_output_dir(Path("/tmp/anything")) == "anything"


def test_fast_disk_folders_name_session_run_and_variant():
    fast = Path("/fast")
    assert fast_disk_for(RUN / "without_destriping", fast) == fast / (
        "ses-224_date-x_funcimg_run-01_without_destriping"
    )
    nested = fast_disk_for(RUN / "without_distortion_correction" / "with_destriping", fast)
    assert (
        nested
        == fast / "ses-224_date-x_funcimg_run-01_without_distortion_correction_with_destriping"
    )


def test_labels_and_export_names_have_no_slashes():
    nested = "without_distortion_correction/with_destriping"
    assert variant_label(nested) == "without_distortion_correction_with_destriping"
    exports = ExportSettings(
        subject_dir="/d", session_first=1, session_last=2, expected_runs=1, run_subdir=nested
    )
    assert "/" not in exports.session_label


def test_nested_runs_are_found_for_exports_and_extraction(tmp_path):
    import numpy as np

    plane = (
        tmp_path / "ses-224_date-x" / "funcimg" / "run-01" / "without_distortion_correction"
        / "without_destriping" / "suite2p" / "plane0"
    )  # fmt: skip
    plane.mkdir(parents=True)
    np.save(plane / "ops.npy", {}, allow_pickle=True)
    (run,) = find_run_planes(tmp_path, 224, 224, "without_distortion_correction/without_destriping")
    assert run.plane_dir == plane and run.run_name == "run-01"
    assert run_dir_of(plane.parents[1]).name == "run-01"
