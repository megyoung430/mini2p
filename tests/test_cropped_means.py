"""Saving the registration QC's cropped mean images."""

import json

import numpy as np
import pytest
import tifffile

from mini2p_preprocessing.qc import CROPPED_MEANS_RECORD, cropped_mean_path, save_cropped_means

REGION = (slice(2, 8), slice(1, 9))


def images(seed=0):
    rng = np.random.default_rng(seed)
    return {"green": rng.random((10, 12)), "red": rng.random((10, 12))}


def test_each_image_is_written_cropped_with_its_values(tmp_path):
    means = {"": images(0), "first500": images(1)}
    written = save_cropped_means(tmp_path, means, REGION)
    assert sorted(p.name for p in written) == [
        "meanImg_green_cropped.tif",
        "meanImg_green_first500_cropped.tif",
        "meanImg_red_cropped.tif",
        "meanImg_red_first500_cropped.tif",
    ]
    for label, by_channel in means.items():
        for channel, image in by_channel.items():
            saved = tifffile.imread(cropped_mean_path(tmp_path, channel, label))
            assert saved.dtype == np.float32
            np.testing.assert_allclose(saved, image[REGION], rtol=1e-6)


def test_the_crop_is_recorded(tmp_path):
    save_cropped_means(tmp_path, {"": images()}, REGION)
    record = json.loads((tmp_path / CROPPED_MEANS_RECORD).read_text())
    assert record["rows"] == [2, 8] and record["columns"] == [1, 9]
    assert record["full_shape"] == [10, 12]
    assert record["files"] == ["meanImg_green_cropped.tif", "meanImg_red_cropped.tif"]


def test_refuses_an_empty_crop(tmp_path):
    with pytest.raises(ValueError, match="leaves nothing"):
        save_cropped_means(tmp_path, {"": images()}, (slice(5, 5), slice(0, 12)))


def test_refuses_images_of_different_shapes(tmp_path):
    means = {"": images(), "first500": {"green": np.zeros((9, 12))}}
    with pytest.raises(ValueError, match="differ in shape"):
        save_cropped_means(tmp_path, means, REGION)


def test_refuses_a_stack_instead_of_an_image(tmp_path):
    with pytest.raises(ValueError, match="2-D"):
        save_cropped_means(tmp_path, {"": {"green": np.zeros((2, 10, 12))}}, REGION)
