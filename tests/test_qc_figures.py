"""Saving the QC notebook's plots in quality_control/<step>/, and the cropped mean images."""

import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import tifffile

from mini2p_preprocessing.contrast import scale_contrast
from mini2p_preprocessing.exports import Contrast, save_cropped_contrast_choice
from mini2p_preprocessing.qc_figures import (
    QC_FOLDER,
    QC_STEPS,
    figure_of,
    qc_step_dir,
    save_qc_figure,
)

REGION = (slice(2, 18), slice(3, 21))


def test_each_step_has_its_own_folder_under_quality_control(tmp_path):
    asked_for = {"bidiphase", "destriping", "distortion_correction", "motion_registration"}
    assert asked_for <= set(QC_STEPS)
    assert len({qc_step_dir(tmp_path, step) for step in QC_STEPS}) == len(QC_STEPS)
    folder = qc_step_dir(tmp_path, "distortion_correction")
    assert folder == tmp_path / QC_FOLDER / "distortion_correction"
    with pytest.raises(ValueError, match="motion_registration"):
        qc_step_dir(tmp_path, "registration")


@pytest.mark.parametrize("drawn", ["figure", "axes", "axes array", "one-axes array"])
def test_a_figure_is_saved_as_png_pdf_svg_and_caption(tmp_path, drawn):
    fig, axes = plt.subplots(2, 2)
    given = {"figure": fig, "axes": axes[1, 0], "axes array": axes,
             "one-axes array": np.array([axes[0, 1]])}[drawn]  # fmt: skip
    assert figure_of(given) is fig
    name = "before_after_bidiphase"
    stem = save_qc_figure(given, tmp_path, "bidiphase", name, "What it shows.\n")
    plt.close(fig)
    assert stem == tmp_path / QC_FOLDER / "bidiphase" / "before_after_bidiphase"
    names = sorted(p.name for p in stem.parent.iterdir())
    assert names == [f"before_after_bidiphase.{e}" for e in ("caption.txt", "pdf", "png", "svg")]
    assert (stem.parent / "before_after_bidiphase.caption.txt").read_text() == "What it shows.\n"


@pytest.mark.parametrize("name", ["", "a/b", "../up", "x.png"])
def test_a_figure_name_must_be_a_plain_file_stem(tmp_path, name):
    fig = plt.figure()
    with pytest.raises(ValueError, match="name"):
        save_qc_figure(fig, tmp_path, "bidiphase", name, "caption\n")
    plt.close(fig)


def test_only_drawn_things_have_a_figure():
    with pytest.raises(TypeError, match="figure"):
        figure_of("not a plot")


def test_cropped_means_are_written_with_their_own_contrast(tmp_path, rng):
    green_raw, red_raw = rng.gamma(2, 50, (2, 20, 24))
    green_raw[:2] = 1e4  # a wrapped band: bright edge rows the crop leaves out
    green, red = Contrast(2.0, 99.0, 0.8), Contrast(1.0, 99.8, 1.5)
    paths = save_cropped_contrast_choice(tmp_path, green_raw, red_raw, green, red, REGION)
    assert sorted(p.name for p in paths.values()) == [
        "meanImg_cropped.json",
        "meanImg_cropped_green_contrast.png",
        "meanImg_cropped_green_raw.tif",
        "meanImg_cropped_red_contrast.png",
        "meanImg_cropped_red_raw.tif",
    ]
    assert tifffile.imread(paths["green raw TIFF"]).dtype == np.float32
    for name, image, contrast in (("green", green_raw, green), ("red", red_raw, red)):
        np.testing.assert_allclose(tifffile.imread(paths[f"{name} raw TIFF"]), image[REGION],
                                   rtol=1e-6)  # fmt: skip
        expected, *_ = scale_contrast(image[REGION], contrast.black_percentile,
                                      contrast.white_percentile, contrast.gamma)  # fmt: skip
        png = matplotlib.image.imread(paths[f"{name} PNG"])[..., 0]
        assert png.shape == expected.shape
        assert np.allclose(png, expected, atol=2 / 255)  # 8-bit grey levels


def test_the_crop_and_contrast_are_recorded(tmp_path, rng):
    green_raw, red_raw = rng.gamma(2, 50, (2, 20, 24))
    paths = save_cropped_contrast_choice(
        tmp_path, green_raw, red_raw, Contrast(), Contrast(), REGION
    )
    record = json.loads(paths["crop JSON"].read_text())
    assert record["rows"] == [2, 18] and record["columns"] == [3, 21]
    assert record["full_shape"] == [20, 24]
    assert record["green"]["black_percentile"] == Contrast().black_percentile
    assert record["green"]["white_value"] == pytest.approx(np.percentile(green_raw[REGION], 99.5))
    assert record["red"]["gamma"] == Contrast().gamma


def test_a_crop_given_as_numpy_integers_or_open_ends_is_recorded_as_plain_bounds(tmp_path, rng):
    green_raw, red_raw = rng.gamma(2, 50, (2, 20, 24))
    region = (slice(np.int64(2), np.int64(18)), slice(None, None))
    paths = save_cropped_contrast_choice(tmp_path, green_raw, red_raw, Contrast(), Contrast(),
                                         region)  # fmt: skip
    record = json.loads(paths["crop JSON"].read_text())
    assert record["rows"] == [2, 18] and record["columns"] == [0, 24]
    assert tifffile.imread(paths["green raw TIFF"]).shape == (16, 24)


@pytest.mark.parametrize(
    ("region", "message"),
    [((slice(5, 5), slice(0, 24)), "leaves nothing"),
     ((slice(2, 100), slice(0, 24)), "outside"),
     ((slice(-5, None), slice(0, 24)), "outside"),
     ((slice(0, 20), slice(0, 24, 2)), "skip"),
     ((slice(2.7, 10), slice(0, 24)), "integer"),
     ((slice(True, 10), slice(0, 24)), "integer"),
     ((slice(0, 20),), "rows, columns"),
     ((3, slice(0, 24)), "rows, columns")],
)  # fmt: skip
def test_cropped_means_refuse_a_crop_that_is_not_a_block_of_the_image(
    tmp_path, rng, region, message
):
    green_raw, red_raw = rng.gamma(2, 50, (2, 20, 24))
    with pytest.raises(ValueError, match=message):
        save_cropped_contrast_choice(tmp_path, green_raw, red_raw, Contrast(), Contrast(), region)
    assert list(tmp_path.iterdir()) == []  # refused before anything was written


def test_cropped_means_refuse_unequal_images(tmp_path, rng):
    green_raw, red_raw = rng.gamma(2, 50, (2, 20, 24))
    with pytest.raises(ValueError, match="shape"):
        save_cropped_contrast_choice(tmp_path, green_raw, red_raw[:10], Contrast(), Contrast(),
                                     REGION)  # fmt: skip
