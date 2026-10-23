"""Variant folders: run-XX/<with_/without_distortion_correction>/<with_/without_destriping>,
and Suite2p's output in <variant>/aligned_<channel>/."""

from __future__ import annotations

from pathlib import Path

import pytest

from mini2p_preprocessing.config import (
    VARIANTS,
    PipelineSettings,
    check_variant_folder,
    find_registered_plane,
    plane_dir_of,
    run_dir_of,
    run_name_from_output_dir,
    settings_variant,
    variant_dir_of,
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
        (False, True, "with_distortion_correction/without_destriping"),
        (True, True, "with_distortion_correction/with_destriping"),
        (False, False, "without_distortion_correction/without_destriping"),
        (True, False, "without_distortion_correction/with_destriping"),
    ],
)
def test_every_combination_has_its_own_folder(destripe, corrected, expected):
    assert variant_subdir(destripe, distortion_corrected=corrected) == expected
    assert variant_output_dir(RUN, destripe, distortion_corrected=corrected) == RUN / expected
    assert expected in VARIANTS
    assert run_dir_of(RUN / expected) == RUN
    assert run_name_from_output_dir(RUN / expected) == "run-01"
    check_variant_folder(RUN / expected, destripe, distortion_corrected=corrected)


def test_the_settings_choose_the_folder():
    off = PipelineSettings.model_validate(
        {"distortion": {"mode": "off"}, "destripe": {"enabled": True}}
    )
    assert settings_variant(off) == "without_distortion_correction/with_destriping"
    on = PipelineSettings.model_validate({"distortion": {"mode": "cluster_full"}})
    assert settings_variant(on) == "with_distortion_correction/without_destriping"


@pytest.mark.parametrize(
    ("folder", "destripe", "corrected"),
    [
        ("with_distortion_correction/without_destriping", False, False),
        ("without_distortion_correction/without_destriping", False, True),
        ("without_distortion_correction/with_destriping", False, False),
    ],
)
def test_a_folder_that_disagrees_with_the_settings_is_refused(folder, destripe, corrected):
    with pytest.raises(ValueError, match="which write to"):
        check_variant_folder(RUN / folder, destripe, distortion_corrected=corrected)


def test_a_plain_output_folder_is_not_checked():
    check_variant_folder(Path("/tmp/anything"), True, distortion_corrected=False)
    assert run_name_from_output_dir(Path("/tmp/anything")) == "anything"


def test_fast_disk_folders_name_session_run_variant_and_alignment():
    fast = Path("/fast")
    registration = RUN / "without_distortion_correction" / "with_destriping" / "aligned_structural"
    assert fast_disk_for(registration, fast) == fast / (
        "ses-224_date-x_funcimg_run-01_without_distortion_correction_with_destriping_"
        "aligned_structural"
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
        / "without_destriping" / "aligned_functional" / "suite2p" / "plane0"
    )  # fmt: skip
    plane.mkdir(parents=True)
    np.save(plane / "ops.npy", {}, allow_pickle=True)
    (run,) = find_run_planes(tmp_path, 224, 224, "without_distortion_correction/without_destriping")
    assert run.plane_dir == plane and run.run_name == "run-01"
    assert run_dir_of(variant_dir_of(plane)).name == "run-01"


def test_the_distortion_setting_must_be_given():
    with pytest.raises(TypeError):
        variant_subdir(True)  # no default: a forgotten argument must not mean "corrected"


@pytest.mark.parametrize("corrected", [True, False])
def test_the_distortion_folders_are_not_output_folders(corrected):
    folder = "with_distortion_correction" if corrected else "without_distortion_correction"
    with pytest.raises(ValueError, match="only holds variant folders"):
        check_variant_folder(RUN / folder, False, distortion_corrected=corrected)


def test_a_nested_registration_is_replaced_but_not_the_one_above(tmp_path):
    from mini2p_preprocessing.registration import clear_earlier_registration

    settings = PipelineSettings.model_validate({"distortion": {"mode": "off"}})
    variant = tmp_path / "run-01" / "without_distortion_correction" / "without_destriping"
    nested = variant / "aligned_structural" / "suite2p"
    nested.mkdir(parents=True)
    clear_earlier_registration(nested, settings)
    assert not nested.exists()
    for not_a_registration in (
        variant / "suite2p",  # beside the TIFFs, not inside an aligned_* folder
        tmp_path / "run-01" / "without_distortion_correction" / "suite2p",
        tmp_path / "run-01" / "aligned_structural" / "suite2p",  # not inside a variant
    ):
        not_a_registration.mkdir(parents=True)
        with pytest.raises(RuntimeError, match="Not replacing"):
            clear_earlier_registration(not_a_registration, settings)


def test_variant_choices_are_the_same_everywhere():
    import typing

    from mini2p_preprocessing.movies import MovieSettings

    for model, field in ((ExportSettings, "run_subdir"), (MovieSettings, "variant")):
        assert set(typing.get_args(model.model_fields[field].annotation)) == set(VARIANTS)


def test_the_launched_batch_names_its_variant(tmp_path):
    from mini2p_preprocessing.batch import launch_variant, snapshot_config

    config = (
        '[distortion]\nmode = "off"\n[batch]\nraw_subject_dir = "/r"\n'
        'derivatives_subject_dir = "/d"\nsession_first = 1\nsession_last = 1\n'
        'source_substring = "x"\ndestripe = true\n'
    )
    snapshot_config(config.encode(), tmp_path)
    assert launch_variant(tmp_path) == "without_distortion_correction/with_destriping"


def _registered(variant_dir: Path, source: str) -> Path:
    import numpy as np

    plane = plane_dir_of(variant_dir, source)
    plane.mkdir(parents=True)
    np.save(plane / "ops.npy", {}, allow_pickle=True)
    return plane


def test_the_registration_is_found_by_alignment(tmp_path):
    variant = tmp_path / "run-01" / "with_distortion_correction" / "with_destriping"
    assert find_registered_plane(variant) is None
    structural = _registered(variant, "structural")
    assert structural == variant / "aligned_structural" / "suite2p" / "plane0"
    assert find_registered_plane(variant) == structural
    assert find_registered_plane(variant, "functional") is None
    functional = _registered(variant, "functional")
    with pytest.raises(ValueError, match="both channels"):
        find_registered_plane(variant)  # "auto" does not guess
    assert find_registered_plane(variant, "functional") == functional
    assert variant_dir_of(functional) == variant
    with pytest.raises(ValueError, match="aligned_"):
        variant_dir_of(variant / "suite2p" / "plane0")


def test_a_path_too_short_to_hold_a_variant_is_refused():
    with pytest.raises(ValueError, match="aligned_"):
        variant_dir_of(Path("plane0"))
