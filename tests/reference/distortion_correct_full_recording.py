#!/usr/bin/env python3
"""Apply one calibration-based distortion transform to a complete interleaved TIFF.

Verbatim copy of mini2p-preprocessing's scripts/distortion_correct_full_recording.py,
used only as a reference: tests check mini2p_preprocessing.calibration gives the same
transform, maps and crop.
"""

# ruff: noqa

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy.io import loadmat
from scipy.ndimage import map_coordinates
from skimage.transform import PiecewiseAffineTransform


def load_transform(path, z_um):
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    keys = {"Anchor_pairs_nodistortion", "Anchor_pairs_withdistortion"}
    if keys.issubset(mat):
        fixed = np.asarray(mat["Anchor_pairs_nodistortion"], np.float32)
        moving = np.asarray(mat["Anchor_pairs_withdistortion"], np.float32)
        matched_z, schema = None, "top_level_anchor_pairs"
    elif "TransfMatrix" in mat:
        transforms = np.atleast_1d(mat["TransfMatrix"])
        zs = np.asarray([float(item.Z) for item in transforms])
        index = int(np.argmin(np.abs(zs - z_um)))
        grid = transforms[index].GridParameter
        fixed = np.asarray(grid.Anchor_pairs_nodistortion, np.float32)
        moving = np.asarray(grid.Anchor_pairs_withdistortion, np.float32)
        matched_z, schema = float(zs[index]), "z_indexed_TransfMatrix"
    else:
        available = sorted(key for key in mat if not key.startswith("__"))
        raise KeyError(f"No supported anchor pairs found. Variables: {available}")
    if fixed.shape != moving.shape or fixed.ndim != 2 or fixed.shape[1] != 2:
        raise ValueError(f"Unexpected anchor shapes: {moving.shape}, {fixed.shape}")
    transform = PiecewiseAffineTransform()
    if not transform.estimate(src=moving, dst=fixed):
        raise RuntimeError("Piecewise-affine transform estimation failed")
    return transform, matched_z, schema, len(fixed)


def make_maps(transform, shape):
    height, width = shape
    row, column = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    source = transform.inverse(np.column_stack([column.ravel(), row.ravel()]))
    return (source[:, 1].reshape(shape).astype(np.float32),
            source[:, 0].reshape(shape).astype(np.float32))


def largest_valid_rectangle(mask):
    heights = np.zeros(mask.shape[1], dtype=int)
    best_area, best = 0, None
    for row in range(mask.shape[0]):
        heights = np.where(mask[row], heights + 1, 0)
        stack = []
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


def crop_bounds(map_y, map_x, shape, padding):
    height, width = shape
    valid = (np.isfinite(map_y) & np.isfinite(map_x) &
             (map_y >= 0) & (map_y <= height - 1) &
             (map_x >= 0) & (map_x <= width - 1))
    top, bottom, left, right = largest_valid_rectangle(valid)
    bounds = (top + padding, bottom - padding, left + padding, right - padding)
    if bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
        raise ValueError("Crop padding removes the complete valid region")
    return bounds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--transform", required=True, type=Path)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--z-um", type=float, default=0)
    parser.add_argument("--order", type=int, default=1, choices=(0, 1, 2, 3))
    parser.add_argument("--crop-padding", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for path in (args.input, args.transform):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite to replace it")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    transform, matched_z, schema, point_count = load_transform(
        args.transform, args.z_um)
    print("Loading complete TIFF into RAM:", args.input, flush=True)
    raw = tifffile.imread(args.input)
    if raw.ndim != 3:
        raise ValueError(f"Expected pages x height x width; got {raw.shape}")
    if len(raw) % args.channels:
        raise ValueError(f"{len(raw)} pages is not divisible by {args.channels} channels")
    height, width = raw.shape[1:]
    map_y, map_x = make_maps(transform, (height, width))
    top, bottom, left, right = crop_bounds(
        map_y, map_x, (height, width), max(0, args.crop_padding))
    output_shape = (len(raw), bottom - top, right - left)
    estimated_gb = (raw.nbytes + np.prod(output_shape) * 4) / 1e9
    print(f"Input shape: {raw.shape}; output shape: {output_shape}", flush=True)
    print(f"Approximate input + output array memory: {estimated_gb:.2f} GB", flush=True)

    corrected = np.empty(output_shape, dtype=np.float32)
    for page in range(len(raw)):
        warped = map_coordinates(
            raw[page].astype(np.float32), [map_y, map_x], order=args.order,
            mode="constant", cval=0, prefilter=args.order > 1)
        corrected[page] = warped[top:bottom, left:right]
        if (page + 1) % 1000 == 0 or page + 1 == len(raw):
            print(f"Corrected {page + 1}/{len(raw)} pages", flush=True)

    print("Writing:", args.output, flush=True)
    tifffile.imwrite(args.output, corrected, bigtiff=True, photometric="minisblack")
    metadata = {
        "input": str(args.input), "output": str(args.output),
        "transform": str(args.transform), "channels": args.channels,
        "requested_z_um": args.z_um, "matched_z_um": matched_z,
        "calibration_schema": schema, "control_points": point_count,
        "interpolation_order": args.order,
        "crop_bounds_top_bottom_left_right": [top, bottom, left, right],
        "input_shape": list(raw.shape), "output_shape": list(corrected.shape),
        "processing": "complete_recording_in_memory_before_suite2p",
    }
    sidecar = args.output.with_suffix(".distortion.json")
    sidecar.write_text(json.dumps(metadata, indent=2))
    print("Saved:", args.output, flush=True)
    print("Saved:", sidecar, flush=True)


if __name__ == "__main__":
    main()
