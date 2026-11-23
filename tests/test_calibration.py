import json

import numpy as np
import pytest
from scipy.io import savemat

from mini2p_preprocessing import calibration
from mini2p_preprocessing.config import (
    DistortionSettings,
    PipelineSettings,
    load_settings,
    settings_from_record,
)
from mini2p_preprocessing.distortion import build_distortion_maps
from tests.reference import distortion_correct_full_recording as reference

SHAPE = (40, 60)


def anchors(shift=1.5, inset=0):
    """Calibration points as (x, y) on a grid over the frame (``inset`` px in from its edges),
    and where the scope puts them."""
    ys, xs = np.meshgrid(np.linspace(inset, SHAPE[0] - 1 - inset, 5),
                         np.linspace(inset, SHAPE[1] - 1 - inset, 6))  # fmt: skip
    fixed = np.column_stack([xs.ravel(), ys.ravel()])
    centre = np.array([(SHAPE[1] - 1) / 2, (SHAPE[0] - 1) / 2])
    r2 = (((fixed - centre) / centre) ** 2).sum(axis=1, keepdims=True)
    return fixed, fixed + shift * r2 * (fixed - centre) / centre


def top_level_mat(path, inset=0):
    fixed, moving = anchors(inset=inset)
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


def single_depth_mat(path):
    """One depth saved by MATLAB as a scalar struct, its Z a 1x1 int16 array."""
    fixed, moving = anchors()
    grid = {"Anchor_pairs_nodistortion": fixed, "Anchor_pairs_withdistortion": moving}
    savemat(path, {"TransfMatrix": {"Z": np.array([[30]], np.int16), "GridParameter": grid}})
    return path


def inset_mat(path):
    """Anchors that stop 6 px short of the frame's edges: pixels outside them sample nothing."""
    return top_level_mat(path, inset=6)


@pytest.mark.filterwarnings("ignore:`estimate` is deprecated:FutureWarning")
@pytest.mark.parametrize(
    ("make_mat", "z_um"),
    [(top_level_mat, 0), (z_indexed_mat, 0), (z_indexed_mat, 60), (z_indexed_mat, 500),
     (single_depth_mat, 0), (inset_mat, 0)],
)  # fmt: skip
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


def test_pixels_outside_the_anchors_are_cropped_away(tmp_path):
    transform, *_ = calibration.load_transform(inset_mat(tmp_path / "calib.mat"), 0)
    map_y, map_x = calibration.make_maps(transform, SHAPE)
    assert map_y[0, 0] == map_x[0, 0] == -1  # scikit-image's value outside the anchors
    top, bottom, left, right = calibration.crop_bounds(map_y, map_x, SHAPE, 0)
    assert top >= 6 and left >= 6 and bottom <= SHAPE[0] - 6 and right <= SHAPE[1] - 6


def test_a_single_depth_struct_is_read(tmp_path):
    info = calibration.load_transform(single_depth_mat(tmp_path / "calib.mat"), 0)[1:]
    assert info == (30.0, "z_indexed_TransfMatrix", 30)


def test_the_warp_moves_pixels_the_way_the_anchors_say(tmp_path):
    # the scope puts each point 2 px right and 3 px up of where it belongs: output pixel
    # (y, x) must sample raw pixel (y - 3, x + 2)
    fixed, _ = anchors()
    moving = fixed + np.array([2.0, -3.0])
    path = tmp_path / "shift.mat"
    savemat(path, {"Anchor_pairs_nodistortion": fixed, "Anchor_pairs_withdistortion": moving})
    map_y, map_x = calibration.make_maps(calibration.load_transform(path, 0)[0], SHAPE)
    yy, xx = np.mgrid[10:20, 10:30]
    np.testing.assert_allclose(map_y[10:20, 10:30], yy - 3, atol=1e-4)
    np.testing.assert_allclose(map_x[10:20, 10:30], xx + 2, atol=1e-4)


def test_a_failed_fit_is_refused(monkeypatch):
    class Failed:
        def __bool__(self):
            return False

        def __str__(self):
            return "degenerate anchors"

    monkeypatch.setattr(calibration.PiecewiseAffineTransform, "from_estimate",
                        classmethod(lambda cls, src, dst: Failed()))  # fmt: skip
    fixed, moving = anchors()
    with pytest.raises(RuntimeError, match="degenerate anchors"):
        calibration.fit_transform(fixed, moving)


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


def test_a_config_toml_with_the_old_setting_loads_with_a_warning(tmp_path, caplog):
    config = tmp_path / "p.toml"
    config.write_text('[distortion]\nmode = "off"\ncalibration_script_dir = "/old"\n')
    with caplog.at_level("WARNING"):
        load_settings(config)
    assert "calibration_script_dir" in caplog.text


def test_run_records_with_the_old_setting_load_quietly(caplog):
    # every pipeline_settings.json written before the port has the key, set or null
    for value in ("/Users/me/scripts", None):
        record = PipelineSettings.model_validate({"distortion": {"mode": "off"}}).model_dump(
            mode="json"
        )
        record["distortion"]["calibration_script_dir"] = value
        with caplog.at_level("WARNING"):
            settings = settings_from_record(json.dumps(record))
        assert settings.distortion.mode == "off"
    assert "calibration_script_dir" not in caplog.text


def test_a_null_setting_is_dropped_without_a_warning(caplog):
    with caplog.at_level("WARNING"):
        DistortionSettings(mode="off", calibration_script_dir=None)
    assert "calibration_script_dir" not in caplog.text
