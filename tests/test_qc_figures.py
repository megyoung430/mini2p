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
    PREPROCESSING_STEPS,
    QC_FOLDER,
    REGISTRATION_STEPS,
    figure_of,
    qc_step_dir,
    save_qc_figure,
)

REGION = (slice(2, 18), slice(3, 21))


def test_preprocessing_plots_go_in_the_variant_folder(tmp_path):
    for step in PREPROCESSING_STEPS:
        assert qc_step_dir(tmp_path, step) == tmp_path / QC_FOLDER / step
    assert {"bidiphase", "destriping", "distortion_correction"} <= set(PREPROCESSING_STEPS)
    with pytest.raises(ValueError, match="motion_registration"):
        qc_step_dir(tmp_path, "registration")


VARIANT = ("run-02", "without_distortion_correction", "with_destriping")


def plane(tmp_path, save_folder="suite2p", channel="aligned_structural"):
    return tmp_path.joinpath(*VARIANT, channel, save_folder, "plane0")


def test_registration_plots_go_in_their_registration_folder(tmp_path):
    variant = tmp_path.joinpath(*VARIANT)
    one, other = plane(tmp_path), plane(tmp_path, "suite2p_rigid_sigma3")
    assert {"motion_registration", "frame_order", "fluorescence"} == set(REGISTRATION_STEPS)
    for step in REGISTRATION_STEPS:
        assert qc_step_dir(variant, step, one) == one.parent / QC_FOLDER / step
        assert qc_step_dir(variant, step, other, "suite2p_rigid_sigma3") == (
            other.parent / QC_FOLDER / step
        )
    with pytest.raises(ValueError, match="registration cell"):
        qc_step_dir(variant, "motion_registration")


def test_a_plane_of_another_registration_than_the_configured_one_is_refused(tmp_path):
    # e.g. save_folder changed in the config while the notebook still holds the old plane
    variant = tmp_path.joinpath(*VARIANT)
    with pytest.raises(ValueError, match="suite2p_rigid_sigma3"):
        qc_step_dir(variant, "fluorescence", plane(tmp_path), "suite2p_rigid_sigma3")


@pytest.mark.parametrize(
    "wrong",
    [
        lambda t: t.joinpath(*VARIANT),  # the variant itself
        lambda t: plane(t).parent,  # the save folder rather than its plane0
        lambda t: t / "run-03" / "with_destriping" / "aligned_structural" / "suite2p" / "plane0",
        lambda t: t.joinpath(*VARIANT, "suite2p", "plane0"),  # not inside aligned_<channel>/
    ],
)
def test_a_plane_that_is_not_a_registration_of_the_variant_is_refused(tmp_path, wrong):
    with pytest.raises(ValueError, match="not a registration of"):
        qc_step_dir(tmp_path.joinpath(*VARIANT), "fluorescence", wrong(tmp_path))


def test_preprocessing_plots_take_no_plane(tmp_path):
    with pytest.raises(ValueError, match="variant"):
        qc_step_dir(tmp_path.joinpath(*VARIANT), "bidiphase", plane(tmp_path))


def test_a_registration_plot_is_saved_beside_its_registration(tmp_path):
    variant = tmp_path.joinpath(*VARIANT)
    plane0 = plane(tmp_path, channel="aligned_functional")
    fig = plt.figure()
    stem = save_qc_figure(fig, variant, "fluorescence", "f0_overview", "F0.\n",
                          plane_dir=plane0, save_folder="suite2p")  # fmt: skip
    plt.close(fig)
    assert stem == plane0.parent / QC_FOLDER / "fluorescence" / "f0_overview"
    assert (stem.parent / "f0_overview.png").is_file()


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
