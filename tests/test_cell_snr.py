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
    cells = pick_cells(image, scaled_geometry(), n_cells=5)
    assert sorted(map(tuple, cells)) == sorted(blobs)
    with pytest.raises(ValueError, match="putative cells"):
        pick_cells(image, scaled_geometry(), n_cells=10_000)


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
