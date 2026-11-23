"""Resting SNR and large transients of putative cells, without segmentation.

Ported from the ``plane_snr.py`` script that compared mice 71R, AB08 and M; the tests check it
gives the same cells, counts and SNR (``tests/reference/plane_snr.py``). Putative cells
are the strongest local maxima of the mean image's cell-scale structure (a difference of
Gaussians); each one's trace is the mean of a disc on it minus the mean of an annulus around
it, so a drifting detector offset, common to both, cancels; traces are averaged in about
0.5 s bins.

- noise: robust SD of successive bin differences / sqrt 2;
- resting SNR: the trace's mean (how far the cell sits above its surroundings) over its
  noise, a measure of contrast, not activity;
- large transients: rising crossings of 4.5 noise SDs after removing a Gaussian running mean
  (sigma 16 s); the same count on the sign-flipped traces estimates how many are noise.

Bins where the brightness around every cell jumps together (stray light reaching the
detector) are left out; such a bin inside a transient splits it into two crossings, as in the
original.

Pixel geometry is given for the mini2p's 512 px scan and scaled for other scan widths of the
same field of view (``scaled_geometry``). The original scaled it by the registered frame's
width instead; the two agree for uncropped frames, and differ for distortion-corrected crops,
whose pixels are the scan's size. The frame rate is Suite2p's ``fs`` here, given on the
command line in the original.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from scipy import sparse
from scipy.ndimage import gaussian_filter, gaussian_filter1d, maximum_filter

N_CELLS = 200
# px at a 512 px scan: DoG sigmas, peak spacing, margin from the crop edge, disc, annulus
GEOMETRY_512 = {"dog_small": 1.5, "dog_large": 6.0, "spacing": 11, "edge": 30,
                "disc_r": 5.0, "ring_in": 8.0, "ring_out": 14.0}  # fmt: skip
SCAN_PX = 512
TARGET_BIN_S = 0.5
HIGHPASS_S = 16.0
TRANSIENT_SD = 4.5  # the threshold used for 71R / AB08, kept so counts are comparable
FLASH_SD = 6.0
MAD_TO_SD = 1.4826


# ---- geometry and cells -----------------------------------------------------------


def scaled_geometry(scan_px: int = SCAN_PX) -> dict:
    """The cell geometry in pixels of a *scan_px*-wide scan of the same field of view."""
    scale = scan_px / SCAN_PX
    return {key: value * scale for key, value in GEOMETRY_512.items()}


def pick_cells(
    mean_image: np.ndarray,
    valid: tuple[Sequence[int], Sequence[int]],
    geometry: dict,
    n_cells: int = N_CELLS,
) -> np.ndarray:
    """(cells, 2) row and column of the strongest local maxima of the mean image's
    cell-scale structure, at least ``edge`` px inside the *valid* ((row start, stop),
    (column start, stop)) region, strongest first."""
    image = np.asarray(mean_image, np.float64)
    dog = gaussian_filter(image, geometry["dog_small"]) - gaussian_filter(
        image, geometry["dog_large"]
    )
    peaks = dog == maximum_filter(dog, size=int(round(geometry["spacing"])) | 1)
    (top, bottom), (left, right) = ((int(a), int(b)) for a, b in valid)
    edge = int(np.ceil(geometry["edge"]))
    inside = np.zeros_like(peaks)
    inside[top + edge : bottom - edge, left + edge : right - edge] = True
    rows, columns = np.nonzero(peaks & inside)
    order = np.argsort(dog[rows, columns])[::-1][:n_cells]
    if len(order) < n_cells:
        raise ValueError(
            f"Only {len(order)} putative cells in a {image.shape[0]} x {image.shape[1]} px "
            f"image, need {n_cells}"
        )
    return np.stack([rows[order], columns[order]], axis=1)


def roi_weights(
    cells: np.ndarray, shape: tuple[int, int], r_low: float, r_high: float
) -> sparse.csr_matrix:
    """(cells, pixels) averaging weights over the pixels r_low <= distance <= r_high."""
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    rows, columns, values = [], [], []
    for index, (cy, cx) in enumerate(cells):
        distance = np.hypot(yy - cy, xx - cx)
        pixels = np.flatnonzero((distance >= r_low) & (distance <= r_high))
        rows.append(np.full(pixels.size, index))
        columns.append(pixels)
        values.append(np.full(pixels.size, 1 / pixels.size))
    return sparse.csr_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=(len(cells), shape[0] * shape[1]),
    )


def bin_frames(block: np.ndarray, frames_per_bin: int) -> np.ndarray:
    """(bins, pixels): the mean of each run of *frames_per_bin* frames of a block whose
    length is a multiple of it."""
    if len(block) % frames_per_bin:
        raise ValueError(f"{len(block)} frames are not whole bins of {frames_per_bin}")
    n_bins = len(block) // frames_per_bin
    flat = np.asarray(block[: n_bins * frames_per_bin], np.float32).reshape(n_bins, -1)
    return flat.reshape(n_bins, frames_per_bin, -1).mean(axis=1)


# ---- measures ---------------------------------------------------------------------


def robust_sd_of_steps(traces: np.ndarray) -> np.ndarray:
    """Per column (bins along axis 0): robust SD of successive differences / sqrt 2."""
    steps = np.diff(traces, axis=0)
    centred = steps - np.nanmedian(steps, axis=0)
    return MAD_TO_SD * np.nanmedian(np.abs(centred), axis=0) / np.sqrt(2)


def flash_bins(ring: np.ndarray) -> np.ndarray:
    """Bins where the brightness around all cells jumps together (stray light)."""
    common = np.median(ring, axis=1)
    residual = common - gaussian_filter1d(common, 5, mode="nearest")
    return residual > FLASH_SD * robust_sd_of_steps(common[:, None])[0]


def rising_edges(mask: np.ndarray) -> np.ndarray:
    """Where each run of True along axis 0 starts."""
    padded = np.vstack([np.zeros((1, mask.shape[1]), bool), mask]).astype(int)
    return np.diff(padded, axis=0) == 1


def cell_measures(disc: np.ndarray, ring: np.ndarray, bin_s: float) -> dict:
    """Per-cell resting SNR and noise, and transient counts, from (bins, cells) disc and
    annulus traces."""
    flash = flash_bins(ring)
    contrast = disc - ring
    contrast[flash] = np.nan
    noise = robust_sd_of_steps(contrast)
    if not (noise > 0).all():
        raise ValueError(f"{int((~(noise > 0)).sum())} putative cells have no noise; their "
                         "traces are constant (empty or saturated pixels?)")  # fmt: skip
    filled = np.where(np.isnan(contrast), np.nanmedian(contrast, axis=0), contrast)
    high_passed = contrast - gaussian_filter1d(filled, HIGHPASS_S / bin_s, axis=0, mode="nearest")
    z = high_passed / noise
    up = rising_edges(np.nan_to_num(z) > TRANSIENT_SD).sum(axis=0)
    down = rising_edges(np.nan_to_num(-z) > TRANSIENT_SD).sum(axis=0)
    return {
        "resting_snr": np.nanmean(contrast, axis=0) / noise,
        "noise": noise,
        "transients": up,
        "transients_flipped": down,
        "n_flash_bins": int(flash.sum()),
    }


def summary(measures: dict, minutes: float) -> dict:
    """The numbers the 71R / AB08 comparison reported, for one recording and channel."""
    snr = measures["resting_snr"]
    return {
        "resting_snr_median": float(np.nanmedian(snr)),
        "resting_snr_p25": float(np.nanpercentile(snr, 25)),
        "resting_snr_p75": float(np.nanpercentile(snr, 75)),
        "transients_per_min": float(measures["transients"].sum() / minutes),
        "transients_flipped_per_min": float(measures["transients_flipped"].sum() / minutes),
        "cells_with_transient": int((measures["transients"] >= 1).sum()),
        "flash_bins": measures["n_flash_bins"],
    }


def frames_per_bin(fs_hz: float) -> int:
    return max(1, int(round(TARGET_BIN_S * fs_hz)))


def binned_traces(
    chunks: Iterable[np.ndarray], weights: Sequence[sparse.csr_matrix], fpb: int
) -> list[np.ndarray]:
    """(bins, cells) traces for each weight matrix, from (frames, rows, columns) chunks whose
    lengths are multiples of *fpb*."""
    out = [[] for _ in weights]
    for chunk in chunks:
        binned = bin_frames(chunk, fpb)
        for traces, weight in zip(out, weights, strict=True):
            traces.append(np.asarray(weight @ binned.T).T)
    return [np.concatenate(traces) for traces in out]
