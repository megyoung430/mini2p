import numpy as np
import pytest
from scipy.io import savemat

from mini2p_preprocessing import calibration
from mini2p_preprocessing.config import DistortionSettings
from mini2p_preprocessing.distortion import build_distortion_maps
from tests.reference import distortion_correct_full_recording as reference

SHAPE = (40, 60)


def anchors(shift=1.5):
    """Calibration points as (x, y) on a grid over the frame, and where the scope puts them."""
    ys, xs = np.meshgrid(np.linspace(0, SHAPE[0] - 1, 5), np.linspace(0, SHAPE[1] - 1, 6))
    fixed = np.column_stack([xs.ravel(), ys.ravel()])
    centre = np.array([(SHAPE[1] - 1) / 2, (SHAPE[0] - 1) / 2])
    r2 = (((fixed - centre) / centre) ** 2).sum(axis=1, keepdims=True)
    return fixed, fixed + shift * r2 * (fixed - centre) / centre


def top_level_mat(path):
    fixed, moving = anchors()
    savemat(path, {"Anchor_pairs_nodistortion": fixed, "Anchor_pairs_withdistortion": moving})
    return path


def z_indexed_mat(path, zs=(0.0, 50.0, 100.0)):
    records = np.empty(len(zs), dtype=[("Z", "O"), ("GridParameter", "O")])
    for i, z in enumerate(zs):
        fixed, moving = anchors(shift=1.0 + i)
        records[i] = (
            z,
            {"Anchor_pairs_nodistortion": fixed, "Anchor_pairs_withdistortion": moving},
        )
    savemat(path, {"TransfMatrix": records})
    return path


@pytest.mark.filterwarnings("ignore:`estimate` is deprecated:FutureWarning")
@pytest.mark.parametrize(
    ("make_mat", "z_um"),
    [(top_level_mat, 0), (z_indexed_mat, 0), (z_indexed_mat, 60), (z_indexed_mat, 500)],
)
def test_maps_and_crop_match_the_calibration_script(tmp_path, make_mat, z_um):
    mat = make_mat(tmp_path / "calib.mat")
    old_tf, *old_info = reference.load_transform(mat, z_um)
    new_tf, *new_info = calibration.load_transform(mat, z_um)
    assert new_info == old_info
    old_maps = reference.make_maps(old_tf, SHAPE)
    new_maps = calibration.make_maps(new_tf, SHAPE)
    for old, new in zip(old_maps, new_maps, strict=True):
        assert np.array_equal(old, new, equal_nan=True)
    for padding in (0, 1, 3):
        assert calibration.crop_bounds(*new_maps, SHAPE, padding) == reference.crop_bounds(
            *old_maps, SHAPE, padding
        )


def test_the_nearest_depth_is_used(tmp_path):
    mat = z_indexed_mat(tmp_path / "calib.mat")
    assert calibration.load_transform(mat, 60)[1:3] == (50.0, "z_indexed_TransfMatrix")
    assert calibration.load_transform(mat, 80)[1] == 100.0


def test_a_calibration_without_anchor_pairs_names_what_it_holds(tmp_path):
    savemat(tmp_path / "other.mat", {"something_else": np.zeros(3)})
    with pytest.raises(KeyError, match="something_else"):
        calibration.load_transform(tmp_path / "other.mat", 0)


def test_anchor_pairs_must_be_matching_lists_of_points():
    fixed, moving = anchors()
    with pytest.raises(ValueError, match="anchor shapes"):
        calibration.fit_transform(fixed, moving[:-1])
    with pytest.raises(ValueError, match="anchor shapes"):
        calibration.fit_transform(fixed[:, :1], moving[:, :1])


def test_the_largest_valid_rectangle_is_found():
    mask = np.ones((6, 8), bool)
    mask[0, :] = False  # a black border along the top
    mask[:, 7] = False  # and down the right
    mask[4, 2] = False  # a hole that still leaves rows 1-3 whole
    assert calibration.largest_valid_rectangle(mask) == (1, 4, 0, 7)
    with pytest.raises(RuntimeError, match="no valid"):
        calibration.largest_valid_rectangle(np.zeros((3, 3), bool))


def test_padding_that_removes_everything_is_refused():
    yy, xx = np.mgrid[0:4, 0:4].astype(np.float32)
    with pytest.raises(ValueError, match="padding"):
        calibration.crop_bounds(yy, xx, (4, 4), 2)


def test_distortion_maps_are_built_with_no_script_to_find(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no scripts/ folder anywhere near
    mat = top_level_mat(tmp_path / "c.mat")
    settings = DistortionSettings(mode="local_batched", transform_mat=mat)
    maps = build_distortion_maps(settings, SHAPE)
    assert maps.calibration_schema == "top_level_anchor_pairs"
    assert maps.control_points == 30
    top, bottom, left, right = maps.crop
    assert 0 < top < bottom < SHAPE[0] and 0 < left < right < SHAPE[1]


def test_an_old_config_naming_the_script_folder_still_loads(caplog):
    with caplog.at_level("WARNING"):
        settings = DistortionSettings(mode="off", calibration_script_dir="/old/scripts")
    assert not hasattr(settings, "calibration_script_dir")
    assert "calibration_script_dir" in caplog.text
