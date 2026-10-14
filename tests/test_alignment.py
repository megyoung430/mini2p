import numpy as np
import pytest

from mini2p_preprocessing.alignment import (
    align_labels,
    estimate_shift,
    load_reference_image,
    shift_image,
    shift_labels,
)
from tests.conftest import as_contrast_png, soma_field

SETTINGS = {
    "max_shift_px": 30,
    "tolerance_px": 1,
    "min_correlation": 0.5,
    "min_retained_fraction": 0.5,
}
SHIFTS = [(0, 0), (3, -2), (-5, 4), (1, 7), (-9, -6), (12, 0)]


def moved_copy(image, dy, dx, noise_rng):
    """A second registration: same tissue, offset so that shifting it by (dy, dx) aligns it,
    with its own noise (different frames averaged)."""
    return shift_image(image, -dy, -dx, fill=float(image.mean())) + noise_rng.normal(
        0, 4, image.shape
    )


@pytest.mark.parametrize("dy, dx", SHIFTS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_shift_is_recovered_from_a_contrast_png_of_realistic_tissue(tmp_path, dy, dx, seed):
    rng = np.random.default_rng(seed)
    plane = soma_field(rng)
    png = as_contrast_png(moved_copy(plane, dy, dx, rng), tmp_path / "other.png", gamma=0.6)
    assert estimate_shift(plane, load_reference_image(png), max_shift_px=30) == (dy, dx)


@pytest.mark.parametrize("kind", ["vignette", "dark_quadrant"])
def test_good_data_with_weak_regions_is_accepted(tmp_path, kind):
    rng = np.random.default_rng(5)
    plane = soma_field(
        rng, vignette=0.7 if kind == "vignette" else 0.0, dark_quadrant=kind == "dark_quadrant"
    )
    labels = np.zeros(plane.shape, np.int32)
    labels[60:66, 60:66] = 3
    png = as_contrast_png(moved_copy(plane, 4, -3, rng), tmp_path / "other.png")
    moving_labels, _ = shift_labels(labels, -4, 3)
    aligned, report = align_labels(moving_labels, load_reference_image(png), plane, **SETTINGS)
    assert (report["dy"], report["dx"]) == (4, -3)
    assert np.array_equal(aligned == 3, labels == 3)


def test_nonuniform_offset_is_refused():
    rng = np.random.default_rng(7)
    plane = soma_field(rng)
    moving = plane.copy()
    moving[:, 128:] = shift_image(plane, 5, 0, fill=float(plane.mean()))[:, 128:]
    with pytest.raises(ValueError, match="not uniform"):
        align_labels(np.zeros(plane.shape, np.int32), moving, plane, **SETTINGS)


def test_blank_or_unrelated_image_is_refused():
    rng = np.random.default_rng(8)
    plane = soma_field(rng)
    labels = np.zeros(plane.shape, np.int32)
    with pytest.raises(ValueError, match="no structure"):
        align_labels(labels, np.zeros_like(plane), plane, **SETTINGS)
    with pytest.raises(ValueError, match="mask_image the mean image"):
        align_labels(labels, soma_field(np.random.default_rng(99)), plane, **SETTINGS)


def test_shift_at_the_search_limit_names_the_setting():
    rng = np.random.default_rng(9)
    plane = soma_field(rng)
    with pytest.raises(ValueError, match="mask_max_shift_px"):
        estimate_shift(plane, moved_copy(plane, 25, 0, rng), max_shift_px=10)


def test_masks_mostly_pushed_off_the_edge_are_dropped_and_reported():
    rng = np.random.default_rng(10)
    plane = soma_field(rng)
    labels = np.zeros(plane.shape, np.int32)
    labels[0:6, 100:106] = 1  # top edge: moving up by 4 keeps 2 of 6 rows
    labels[100:106, 100:106] = 2
    moving = moved_copy(plane, -4, 0, rng)
    moving_labels, _ = shift_labels(labels, 0, 0)
    aligned, report = align_labels(moving_labels, moving, plane, **SETTINGS)
    assert report["dropped_labels"] == [1] and report["kept_labels"] == [2]
    assert report["retained_fraction"]["1"] == pytest.approx(2 / 6)
    assert set(np.unique(aligned)) == {0, 2}


def test_shift_labels_moves_and_zero_fills():
    labels = np.zeros((10, 10), np.int32)
    labels[2:4, 2:4] = 1
    moved, kept = shift_labels(labels, 3, 3)
    assert np.array_equal(np.argwhere(moved == 1), np.argwhere(labels == 1) + 3)
    assert kept == [1] and moved.dtype == labels.dtype


def test_colour_images_are_refused_unless_grey(tmp_path):
    import matplotlib.pyplot as plt

    rgb = np.zeros((20, 30, 3))
    rgb[..., 0] = 1.0  # red/green overlay style image
    plt.imsave(tmp_path / "overlay.png", rgb)
    with pytest.raises(ValueError, match="grey"):
        load_reference_image(tmp_path / "overlay.png")
    image = np.random.default_rng(0).random((20, 30))
    plt.imsave(tmp_path / "grey.png", image, cmap="gray", vmin=0, vmax=1)
    grey = load_reference_image(tmp_path / "grey.png")
    assert grey.shape == (20, 30) and np.corrcoef(grey.ravel(), image.ravel())[0, 1] > 0.99


def test_shapes_must_match():
    with pytest.raises(ValueError, match="shape"):
        estimate_shift(np.zeros((10, 10)), np.zeros((10, 12)), max_shift_px=3)


def test_unrelated_image_is_refused_even_with_shared_vignetting():
    rng = np.random.default_rng(11)
    plane = soma_field(rng, vignette=0.7)
    other = soma_field(np.random.default_rng(111), vignette=0.7)
    with pytest.raises(ValueError, match="mask_image the mean image"):
        align_labels(np.zeros(plane.shape, np.int32), other, plane, **SETTINGS)


@pytest.mark.parametrize("kind", ["sparse", "dark_quadrant"])
@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("dy, dx", [(3, -4), (-10, 8), (12, 0)])
def test_sparse_or_partly_dark_fields_are_accepted_across_seeds(tmp_path, kind, seed, dy, dx):
    rng = np.random.default_rng(seed)
    plane = soma_field(
        rng, n_cells=15 if kind == "sparse" else 60, dark_quadrant=kind == "dark_quadrant"
    )
    png = as_contrast_png(moved_copy(plane, dy, dx, rng), tmp_path / "other.png", gamma=0.6)
    _, report = align_labels(
        np.zeros(plane.shape, np.int32), load_reference_image(png), plane, **SETTINGS
    )
    assert (report["dy"], report["dx"]) == (dy, dx)


def _part_moved(seed, where, k, tmp_path):
    rng = np.random.default_rng(seed)
    plane = soma_field(rng)
    moving = plane.copy()
    part = (slice(None), slice(128, None)) if where == "half" else (slice(128, None),) * 2
    moving[part] = shift_image(plane, k, 0, fill=float(plane.mean()))[part]
    png = as_contrast_png(moving + rng.normal(0, 4, plane.shape), tmp_path / "m.png")
    return plane, load_reference_image(png)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("where, k", [("half", 3), ("half", 4), ("quadrant", 3), ("quadrant", 4)])
def test_non_rigid_offsets_beyond_the_tolerance_are_refused(tmp_path, seed, where, k):
    plane, moving = _part_moved(seed, where, k, tmp_path)
    with pytest.raises(ValueError, match="not uniform"):
        align_labels(np.zeros(plane.shape, np.int32), moving, plane, **SETTINGS)


@pytest.mark.parametrize("seed", range(5))
def test_two_pixel_split_is_refused_or_lands_within_one_pixel_of_both_halves(tmp_path, seed):
    plane, moving = _part_moved(seed, "half", 2, tmp_path)
    try:
        _, report = align_labels(np.zeros(plane.shape, np.int32), moving, plane, **SETTINGS)
    except ValueError as error:
        assert "not uniform" in str(error)
    else:  # parts need 0 and -2 px: only the midpoint keeps every mask within 1 px
        assert (report["dy"], report["dx"]) == (-1, 0)


@pytest.mark.parametrize("seed", range(8))
def test_unrelated_images_sharing_vignetting_and_an_edge_are_refused(tmp_path, seed):
    plane = soma_field(np.random.default_rng(seed), vignette=0.7, dark_quadrant=True)
    other = soma_field(np.random.default_rng(seed + 50), vignette=0.7, dark_quadrant=True)
    png = as_contrast_png(other, tmp_path / "o.png")
    with pytest.raises(ValueError):
        align_labels(np.zeros(plane.shape, np.int32), load_reference_image(png), plane, **SETTINGS)


def test_report_records_how_well_each_quadrant_fits_the_applied_shift(tmp_path):
    rng = np.random.default_rng(3)
    plane = soma_field(rng)
    png = as_contrast_png(moved_copy(plane, 2, 5, rng), tmp_path / "m.png")
    _, report = align_labels(
        np.zeros(plane.shape, np.int32), load_reference_image(png), plane, **SETTINGS
    )
    fits = report["quadrant_fits_at_overall_shift"]
    assert len(fits) == 4 and all(f is None or 0.3 < f <= 1 for f in fits)
    assert report["quadrants_used"] == sum(f is not None for f in fits)
