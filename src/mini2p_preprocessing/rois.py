"""ROI label images <-> Suite2p ``stat`` entries, and outlines for display (pure)."""

from __future__ import annotations

import numpy as np


def masks_to_stat(labels: np.ndarray) -> list[dict]:
    """One minimal Suite2p stat entry per non-zero label, in increasing label order.

    Every pixel gets weight 1 (Cellpose masks are binary), and ``med`` is the
    mean pixel position. ``suite2p.detection.roi_stats`` fills in the rest.
    """
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError(f"Expected a 2-D label image, got shape {labels.shape}")
    flat = labels.ravel()
    pixels = np.flatnonzero(flat)
    if not len(pixels):
        raise ValueError("Label image contains no ROIs")
    order = np.argsort(flat[pixels], kind="stable")
    pixels = pixels[order]
    _, starts = np.unique(flat[pixels], return_index=True)
    stat = []
    for roi_pixels in np.split(pixels, starts[1:]):
        ypix, xpix = np.unravel_index(roi_pixels, labels.shape)
        npix = len(ypix)
        stat.append(
            {
                "ypix": ypix,
                "xpix": xpix,
                "npix": npix,
                "lam": np.ones(npix, np.float32),
                "med": [float(np.mean(ypix)), float(np.mean(xpix))],
            }
        )
    return stat


def labels_from_stat(stat, ly: int, lx: int) -> np.ndarray:
    """Label image with ROI i drawn as i + 1; where ROIs overlap, the earlier one wins."""
    labels = np.zeros((ly, lx), np.int32)
    for index, roi in enumerate(stat):
        ypix, xpix = np.asarray(roi["ypix"]), np.asarray(roi["xpix"])
        free = labels[ypix, xpix] == 0
        labels[ypix[free], xpix[free]] = index + 1
    return labels


def outline_labels(labels: np.ndarray) -> np.ndarray:
    """Keep each ROI's inner boundary: labelled pixels with a 4-neighbour of another value."""
    labels = np.asarray(labels)
    padded = np.pad(labels, 1, mode="constant")
    centre = padded[1:-1, 1:-1]
    differs = (
        (padded[:-2, 1:-1] != centre)
        | (padded[2:, 1:-1] != centre)
        | (padded[1:-1, :-2] != centre)
        | (padded[1:-1, 2:] != centre)
    )
    return np.where((labels != 0) & differs, labels, 0)
