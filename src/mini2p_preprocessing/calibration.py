"""The mini2p's fixed distortion calibration: anchor pairs, the warp map, and its crop.

A calibration ``.mat`` holds pairs of anchor points, as (x, y): where each point is in an
undistorted image and where the scope puts it. A piecewise-affine fit of the pairs gives,
for every output pixel, the raw pixel to sample (``make_maps``), and ``crop_bounds`` keeps
the largest rectangle every output pixel of which samples inside the raw frame.

Ported from mini2p-preprocessing's ``scripts/distortion_correct_full_recording.py``; the
tests check it gives the same transform, maps and crop.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat
from skimage.transform import PiecewiseAffineTransform

ANCHOR_KEYS = ("Anchor_pairs_nodistortion", "Anchor_pairs_withdistortion")
TOP_LEVEL_SCHEMA = "top_level_anchor_pairs"
Z_INDEXED_SCHEMA = "z_indexed_TransfMatrix"


# ---- analysis ---------------------------------------------------------------------


def anchor_pairs(mat: dict, z_um: float) -> tuple[np.ndarray, np.ndarray, float | None, str]:
    """(fixed, moving, matched depth, schema) from a loaded calibration.

    Either the pairs sit at the top level, or ``TransfMatrix`` holds one set per depth
    ``Z`` and the one nearest ``z_um`` is used (its depth is returned; None otherwise).
    """
    fixed_key, moving_key = ANCHOR_KEYS
    if set(ANCHOR_KEYS).issubset(mat):
        fixed, moving = mat[fixed_key], mat[moving_key]
        matched_z, schema = None, TOP_LEVEL_SCHEMA
    elif "TransfMatrix" in mat:
        transforms = np.atleast_1d(mat["TransfMatrix"])
        zs = np.asarray([float(item.Z) for item in transforms])
        index = int(np.argmin(np.abs(zs - z_um)))
        grid = transforms[index].GridParameter
        fixed, moving = getattr(grid, fixed_key), getattr(grid, moving_key)
        matched_z, schema = float(zs[index]), Z_INDEXED_SCHEMA
    else:
        available = sorted(key for key in mat if not key.startswith("__"))
        raise KeyError(f"No supported anchor pairs found. Variables: {available}")
    return np.asarray(fixed, np.float32), np.asarray(moving, np.float32), matched_z, schema


def fit_transform(fixed: np.ndarray, moving: np.ndarray) -> PiecewiseAffineTransform:
    """Piecewise-affine map from distorted (moving) to undistorted (fixed) points."""
    if fixed.shape != moving.shape or fixed.ndim != 2 or fixed.shape[1] != 2:
        raise ValueError(
            f"Unexpected anchor shapes: with distortion {moving.shape}, "
            f"without distortion {fixed.shape}; expected two equal (points, 2) arrays"
        )
    transform = PiecewiseAffineTransform.from_estimate(moving, fixed)
    if not transform:
        raise RuntimeError(f"Piecewise-affine transform estimation failed: {transform}")
    return transform


def make_maps(
    transform: PiecewiseAffineTransform, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """(map_y, map_x): the raw pixel each output pixel samples.

    Outside the convex hull of the anchors scikit-image gives -1 in both maps, which
    ``crop_bounds`` treats as outside the frame.
    """
    height, width = shape
    row, column = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    source = transform.inverse(np.column_stack([column.ravel(), row.ravel()]))
    return (
        source[:, 1].reshape(shape).astype(np.float32),
        source[:, 0].reshape(shape).astype(np.float32),
    )


def largest_valid_rectangle(mask: np.ndarray) -> tuple[int, int, int, int]:
    """(top, bottom, left, right), end-exclusive, of the largest all-True rectangle.

    Row by row, each column's run of True above it is a histogram bar; the largest
    rectangle under the histogram is found with a stack of rising bars.
    """
    heights = np.zeros(mask.shape[1], dtype=int)
    best_area, best = 0, None
    for row in range(mask.shape[0]):
        heights = np.where(mask[row], heights + 1, 0)
        stack: list[int] = []
        for column in range(len(heights) + 1):
            height = int(heights[column]) if column < len(heights) else 0
            while stack and int(heights[stack[-1]]) > height:
                h = int(heights[stack.pop()])
                left = stack[-1] + 1 if stack else 0
                area = h * (column - left)
                if area > best_area:
                    best_area = area
                    best = (row - h + 1, row + 1, left, column)
            stack.append(column)
    if best is None:
        raise RuntimeError("Distortion map has no valid rectangular output region")
    return best


def crop_bounds(
    map_y: np.ndarray, map_x: np.ndarray, shape: tuple[int, int], padding: int
) -> tuple[int, int, int, int]:
    """The crop, less ``padding`` px on every side, that keeps only pixels mapped in-frame."""
    height, width = shape
    valid = (
        np.isfinite(map_y)
        & np.isfinite(map_x)
        & (map_y >= 0)
        & (map_y <= height - 1)
        & (map_x >= 0)
        & (map_x <= width - 1)
    )
    top, bottom, left, right = largest_valid_rectangle(valid)
    bounds = (top + padding, bottom - padding, left + padding, right - padding)
    if bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
        raise ValueError("Crop padding removes the complete valid region")
    return bounds


# ---- I/O --------------------------------------------------------------------------


def load_transform(
    path: Path, z_um: float
) -> tuple[PiecewiseAffineTransform, float | None, str, int]:
    """(transform, matched depth, schema, number of anchor pairs) from a calibration .mat."""
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    fixed, moving, matched_z, schema = anchor_pairs(mat, z_um)
    return fit_transform(fixed, moving), matched_z, schema, len(fixed)
