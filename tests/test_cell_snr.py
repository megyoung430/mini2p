"""Resting SNR and transients of putative cells, on synthetic images and traces."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from mini2p_preprocessing.cell_snr import (
    GEOMETRY_512,
    TRANSIENT_SD,
    bin_frames,
    binned_traces,
    cell_measures,
    pick_cells,
    rising_edges,
    roi_weights,
    scaled_geometry,
    summary,
)

N_BINS, N_CELLS = 2000, 20


def test_geometry_scales_with_the_scan_width():
    assert scaled_geometry(256)["disc_r"] == GEOMETRY_512["disc_r"] / 2
    assert scaled_geometry() == pytest.approx(GEOMETRY_512)


def test_cells_are_the_blobs_of_the_mean_image_away_from_its_edge(rng):
    image = rng.normal(0, 0.05, (160, 160))
    blobs = [(50, 50), (50, 110), (110, 50), (110, 110), (80, 80)]
    for row, column in blobs:
        image[row, column] += 300
    image = gaussian_filter(image, 3)
    image[5, 5] += 50  # brighter but inside the edge margin
    whole = ((0, 160), (0, 160))
    cells = pick_cells(image, whole, scaled_geometry(), n_cells=5)
    assert sorted(map(tuple, cells)) == sorted(blobs)
    with pytest.raises(ValueError, match="putative cells"):
        pick_cells(image, whole, scaled_geometry(), n_cells=10_000)


def test_disc_and_annulus_weights_average_their_pixels():
    weights = roi_weights(np.array([[10, 10]]), (21, 21), 0, 2)
    image = np.zeros((21, 21))
    image[8:13, 8:13] = 1  # covers the disc of radius 2
    assert (weights @ image.ravel())[0] == pytest.approx(1)
    ring = roi_weights(np.array([[10, 10]]), (21, 21), 4, 6)
    assert (ring @ image.ravel())[0] == 0


def test_binned_traces_average_frames_in_bins_across_chunks(rng):
    movie = rng.normal(size=(12, 4, 4))
    weights = roi_weights(np.array([[1, 1]]), (4, 4), 0, 0)  # the single pixel (1, 1)
    (trace,) = binned_traces([movie[:6], movie[6:]], [weights], 3)
    assert np.allclose(trace[:, 0], movie[:, 1, 1].reshape(4, 3).mean(axis=1))
    assert bin_frames(movie, 3).shape == (4, 16)


def test_rising_edges_mark_where_each_run_starts():
    mask = np.array([[0, 1, 1, 0, 1]], bool).T
    assert list(rising_edges(mask)[:, 0]) == [False, True, False, False, True]


def synthetic_traces(rng, active=10, amplitude=12.0, events=8):
    """Cells 5 counts above their annulus, with noise of SD 1 of their own; the first *active*
    cells have *events* decaying transients each."""
    ring = 100 + rng.normal(0, 1, (N_BINS, N_CELLS))
    disc = ring + 5 + rng.normal(0, 1, (N_BINS, N_CELLS))
    for cell in range(active):
        for start in rng.choice(np.arange(50, N_BINS - 50, 60), events, replace=False):
            disc[start : start + 10, cell] += amplitude * np.exp(-np.arange(10) / 3)
    return disc, ring


def test_resting_snr_and_transients_of_active_and_quiet_cells(rng):
    disc, ring = synthetic_traces(rng)
    measures = cell_measures(disc, ring, bin_s=0.5)
    # the disc carries the annulus's noise, which cancels: contrast = 5 + noise of SD 1
    assert np.median(measures["resting_snr"][10:]) == pytest.approx(5, rel=0.1)
    # a transient's noisy decay can dip under the threshold and cross it again, as in the
    # 71R / AB08 measure, which this keeps; never fewer than the events
    assert all(8 <= n <= 11 for n in measures["transients"][:10])
    assert measures["transients"][10:].sum() <= 1
    assert measures["transients_flipped"].sum() <= 2
    numbers = summary(measures, minutes=N_BINS * 0.5 / 60)
    assert numbers["cells_with_transient"] >= 10
    assert numbers["transients_per_min"] == pytest.approx(
        measures["transients"].sum() / (N_BINS * 0.5 / 60)
    )


def test_a_flash_around_every_cell_is_not_a_transient(rng):
    disc, ring = synthetic_traces(rng, active=0)
    jump = 20 * TRANSIENT_SD
    disc[1000], ring[1000] = disc[1000] + 1.3 * jump, ring[1000] + jump  # brighter on cells
    measures = cell_measures(disc, ring, bin_s=0.5)
    assert measures["n_flash_bins"] == 1
    assert measures["transients"].sum() <= 1


# ---- the same as the original script ----------------------------------------------


def textured_movie(rng, n_frames=400, size=128):
    """A mean image with blob-like cells, some active, on noise."""
    image = gaussian_filter(rng.normal(size=(size, size)), 2.0)
    image = 200 + 400 * (image - image.min()) / np.ptp(image)
    movie = image + rng.normal(0, 8, (n_frames, size, size))
    activity = np.stack([np.convolve((rng.random(n_frames) < 0.02) * 60.0,
                                     np.exp(-np.arange(20) / 4))[:n_frames]
                         for _ in range(6)])  # fmt: skip
    for (row, column), trace in zip([(40, 40), (40, 88), (88, 40), (88, 88), (64, 64), (50, 70)],
                                    activity, strict=True):  # fmt: skip
        movie[:, row - 3 : row + 4, column - 3 : column + 4] += trace[:, None, None]
    return movie.astype(np.int16).astype(np.float32)


def test_cells_traces_and_measures_match_plane_snr(rng):
    from tests.reference import plane_snr as original

    movie = textured_movie(rng)
    mean_image = movie.mean(axis=0).astype(np.float64)
    valid = ((2, 126), (3, 125))
    geometry = scaled_geometry(256)
    cells = pick_cells(mean_image, valid, geometry, n_cells=12)
    assert np.array_equal(cells, original.pick_cells(mean_image, valid, geometry, n_cells=12))
    shape = mean_image.shape
    weights = [roi_weights(cells, shape, 0, geometry["disc_r"]),
               roi_weights(cells, shape, geometry["ring_in"], geometry["ring_out"])]  # fmt: skip
    fpb, n_bins = 4, 100
    disc, ring = binned_traces([movie[:200], movie[200:]], weights, fpb)
    assert np.allclose(disc, original.bin_traces(movie, weights[0], fpb, n_bins))
    assert np.allclose(ring, original.bin_traces(movie, weights[1], fpb, n_bins))
    bin_s = 0.5
    ours = cell_measures(disc.copy(), ring, bin_s)
    theirs = original.cell_measures(disc.copy(), ring, original.HIGHPASS_S / bin_s)
    assert np.allclose(ours["resting_snr"], theirs["resting_snr"])
    assert np.allclose(ours["noise"], theirs["noise"])
    assert ours["transients"].sum() == theirs["transients"]
    assert ours["transients_flipped"].sum() == theirs["transients_flipped"]
    assert (ours["transients"] >= 1).sum() == theirs["cells_ge1"]
    assert ours["n_flash_bins"] == theirs["n_flash_bins"]


def test_ragged_bins_and_constant_cells_are_refused(rng):
    with pytest.raises(ValueError, match="whole bins"):
        bin_frames(np.zeros((7, 2, 2)), 3)
    disc, ring = synthetic_traces(rng, active=0)
    disc[:, 0] = ring[:, 0] = 5.0
    with pytest.raises(ValueError, match="no noise"):
        cell_measures(disc, ring, bin_s=0.5)
