"""Blood-vessel background and red-channel regression before ΔF/F0 (pure functions).

A blood vessel has no labelled cells, so its fluorescence over time is background: light
that reaches every pixel (out-of-focus fluorescence, scattered light, brightness changes
of the whole field). Its trace is regressed out of each ROI's F and of its neuropil Fneu,
each with its own coefficient, before the usual neuropil correction; regressing it out of
F alone would let ``F - c * Fneu`` add part of it back.

The red (structural) label does not follow activity, so what a ROI's red trace shares with
its green trace is brightness change from motion or focus. With a usable red channel, each
ROI's red trace (background removed and neuropil corrected the same way) is regressed out
of its green trace.

Regression keeps each trace's mean: only the fluctuations explained by the regressor are
removed, so F0 stays on the original scale.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

from .config import FluorescenceSettings, PipelineSettings
from .extraction import file_sha256, load_cellpose_labels
from .fluorescence import DffResult, compute_dff, dff_from_corrected, neuropil_correct
from .registration import RegisteredOutputs
from .tiff_io import (
    check_clipped,
    chunk_bounds,
    n_written_values,
    output_record,
    replaced_when_complete,
    write_interleaved,
)


@dataclass(frozen=True)
class Regression:
    cleaned: np.ndarray  # (n_rois, n_frames), mean of each row unchanged
    coef: np.ndarray  # (n_rois,) regressor coefficient per row


@dataclass(frozen=True)
class DffVersions:
    standard: DffResult  # neuropil correction only, as before
    background: DffResult  # vessel background regressed out of F and Fneu first
    red: DffResult | None  # and each ROI's red trace regressed out; None when not usable
    background_coef: dict[str, np.ndarray]  # "F", "Fneu" (and "F_chan2", "Fneu_chan2")
    red_coef: np.ndarray | None


def vessel_mask(labels: np.ndarray, frame_shape: tuple[int, int]) -> np.ndarray:
    """Every labelled pixel of a Cellpose segmentation of blood vessels, as one mask."""
    labels = np.asarray(labels)
    if labels.shape != tuple(frame_shape):
        raise ValueError(
            f"Vessel mask shape {labels.shape} differs from the registered frames "
            f"{tuple(frame_shape)}; segment this plane's mean image"
        )
    mask = labels > 0
    if not mask.any():
        raise ValueError("The vessel segmentation has no labelled pixels")
    return mask


def _centred_regressor(regressor: np.ndarray, n_frames: int) -> np.ndarray:
    regressor = np.asarray(regressor, np.float64)
    if regressor.shape[-1] != n_frames:
        raise ValueError(f"Regressor has {regressor.shape[-1]} frames, traces {n_frames}")
    if not np.isfinite(regressor).all():
        raise ValueError("Regressor has values that are not finite")
    return regressor - regressor.mean(axis=-1, keepdims=True)


def regress_out(traces: np.ndarray, regressor: np.ndarray) -> Regression:
    """Least-squares removal of one regressor (n_frames,) from every row of *traces*."""
    traces = np.asarray(traces, np.float64)
    _check_traces(traces)
    centred = _centred_regressor(regressor, traces.shape[1])
    power = centred @ centred
    if power == 0:
        raise ValueError("The regressor is flat, so nothing can be regressed out")
    coef = (traces - traces.mean(axis=1, keepdims=True)) @ centred / power
    return Regression((traces - coef[:, None] * centred).astype(np.float32), coef)


def regress_out_each(traces: np.ndarray, regressors: np.ndarray) -> Regression:
    """Least-squares removal of row i of *regressors* from row i of *traces*."""
    traces = np.asarray(traces, np.float64)
    _check_traces(traces)
    if np.shape(regressors) != traces.shape:
        raise ValueError(f"Regressors of shape {np.shape(regressors)} vs traces {traces.shape}")
    centred = _centred_regressor(regressors, traces.shape[1])
    power = np.einsum("rt,rt->r", centred, centred)
    flat = np.flatnonzero(power == 0)
    if flat.size:
        raise ValueError(f"The regressor is flat for row(s) {flat[:10].tolist()}")
    traces_centred = traces - traces.mean(axis=1, keepdims=True)
    coef = np.einsum("rt,rt->r", traces_centred, centred) / power
    return Regression((traces - coef[:, None] * centred).astype(np.float32), coef)


def _check_traces(traces: np.ndarray) -> None:
    if traces.ndim != 2:
        raise ValueError(f"Traces must be (n_rois, n_frames), got shape {traces.shape}")
    if not np.isfinite(traces).all():
        raise ValueError("Traces have values that are not finite")


def dff_versions(
    f: np.ndarray,
    fneu: np.ndarray,
    f_chan2: np.ndarray | None,
    fneu_chan2: np.ndarray | None,
    background_green: np.ndarray,
    background_red: np.ndarray | None,
    fs_hz: float,
    settings: FluorescenceSettings,
    red_usable: bool,
) -> DffVersions:
    """ΔF/F0 three ways: standard, vessel background removed, and red regressed out too.

    The red version needs red traces, a red background trace and *red_usable*.
    """
    standard = compute_dff(f, fneu, fs_hz, settings)
    f_bg, fneu_bg = regress_out(f, background_green), regress_out(fneu, background_green)
    background = compute_dff(f_bg.cleaned, fneu_bg.cleaned, fs_hz, settings)
    coefs = {"F": f_bg.coef, "Fneu": fneu_bg.coef}
    have_red = f_chan2 is not None and fneu_chan2 is not None and background_red is not None
    if not (red_usable and have_red):
        return DffVersions(standard, background, None, coefs, None)
    f2_bg, fneu2_bg = regress_out(f_chan2, background_red), regress_out(fneu_chan2, background_red)
    coefs |= {"F_chan2": f2_bg.coef, "Fneu_chan2": fneu2_bg.coef}
    red_corrected = neuropil_correct(f2_bg.cleaned, fneu2_bg.cleaned, settings.neuropil_coefficient)
    without_red = regress_out_each(background.f_corr, red_corrected)
    red = dff_from_corrected(without_red.cleaned, fs_hz, settings)
    return DffVersions(standard, background, red, coefs, without_red.coef)


# ---- reading the registered binaries ------------------------------------------------


def mask_mean_trace(stack: np.ndarray, mask: np.ndarray, chunk_frames: int) -> np.ndarray:
    """Mean of the *mask* pixels in every frame of a (frames, Ly, Lx) stack, read in chunks."""
    if np.shape(stack)[1:] != np.shape(mask):
        raise ValueError(f"Mask shape {np.shape(mask)} vs frames {np.shape(stack)[1:]}")
    return np.concatenate(
        [
            np.asarray(stack[start:end], np.float64)[:, mask].mean(axis=1)
            for start, end in chunk_bounds(len(stack), chunk_frames)
        ]
    )


def pixel_coefficients(stack: np.ndarray, regressor: np.ndarray, chunk_frames: int) -> np.ndarray:
    """Least-squares coefficient of *regressor* (n_frames,) for every pixel of the stack."""
    centred = _centred_regressor(regressor, len(stack))
    power = centred @ centred
    if power == 0:
        raise ValueError("The regressor is flat, so nothing can be regressed out")
    total = np.zeros(np.shape(stack)[1:], np.float64)
    for start, end in chunk_bounds(len(stack), chunk_frames):
        # sum(pixel * centred regressor) needs no centred pixel: the regressor sums to zero
        total += np.tensordot(centred[start:end], np.asarray(stack[start:end], np.float64), 1)
    return total / power


def corrected_chunk(
    stack: np.ndarray, start: int, end: int, coef: np.ndarray, centred: np.ndarray
) -> np.ndarray:
    """Frames start..end with the regressor removed from each pixel (means unchanged)."""
    frames = np.asarray(stack[start:end], np.float32)
    return frames - centred[start:end, None, None].astype(np.float32) * coef.astype(np.float32)


# ---- files: cached vessel traces and the background-corrected TIFF -----------------

BACKGROUND_TRACES = "background_traces.npz"  # in the plane folder
BACKGROUND_TIFF = "data_background_removed.tif"
BACKGROUND_RECORD = "data_background_removed.json"


def vessel_cache_key(seg_file: Path, ops: dict) -> dict:
    """What the vessel traces depend on: the segmentation and the registered movie (a new
    registration clears the plane folder, cache included)."""
    return {
        "seg_file": str(Path(seg_file).resolve()),
        "seg_sha256": file_sha256(seg_file),
        "shape": [int(ops["nframes"]), int(ops["Ly"]), int(ops["Lx"])],
    }


def load_vessel_traces(
    outputs: RegisteredOutputs, plane_dir: Path, seg_file: Path, settings: PipelineSettings
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Mean vessel fluorescence per frame in each channel ({"green", "red"}), and the mask.

    Reading both binaries takes minutes, so the traces are kept in ``background_traces.npz``
    and reused while the segmentation and the registration are unchanged.
    """
    mask = vessel_mask(load_cellpose_labels(seg_file), outputs.functional.shape[1:])
    key = json.dumps(vessel_cache_key(seg_file, outputs.ops), sort_keys=True)
    cache = Path(plane_dir) / BACKGROUND_TRACES
    if cache.is_file():
        with np.load(cache) as saved:
            if str(saved["key"]) == key:
                return {"green": saved["green"], "red": saved["red"]}, mask
    chunk = settings.fluorescence.background_chunk_frames
    traces = {
        "green": mask_mean_trace(outputs.functional, mask, chunk),
        "red": mask_mean_trace(outputs.structural, mask, chunk),
    }
    with replaced_when_complete(cache) as partial:
        with partial.open("wb") as handle:  # np.savez would add .npz to a .partial name
            np.savez(handle, key=key, **traces)
    return traces, mask


def write_background_tiff(
    outputs: RegisteredOutputs,
    traces: dict[str, np.ndarray],
    plane_dir: Path,
    settings: PipelineSettings,
    source: dict,
) -> Path:
    """``data_background_removed.tif``: the registered movie with each channel's vessel
    trace regressed out of every pixel (its own coefficient per pixel), interleaved in the
    acquired channel order, int16; and a JSON record of how it was made (*source*: e.g.
    the vessel cache key)."""
    channels = settings.channels
    chunk = settings.fluorescence.background_chunk_frames
    by_channel = {
        channels.functional_channel: (outputs.functional, traces["green"]),
        channels.structural_channel: (outputs.structural, traces["red"]),
    }
    roles = {channels.functional_channel: "green", channels.structural_channel: "red"}
    order = [by_channel[ch] for ch in sorted(by_channel)]
    centred = [_centred_regressor(trace, len(stack)) for stack, trace in order]
    coefs = [pixel_coefficients(stack, trace, chunk) for stack, trace in order]
    path = Path(plane_dir) / BACKGROUND_TIFF
    Path(plane_dir, BACKGROUND_RECORD).unlink(missing_ok=True)  # never beside another TIFF
    n_frames, frame_shape = len(outputs.functional), outputs.functional.shape[1:]
    with replaced_when_complete(path) as partial:
        clipped = 0
        with tifffile.TiffWriter(partial, bigtiff=True) as writer:
            for start, end in chunk_bounds(n_frames, chunk):
                chunks = [
                    corrected_chunk(stack, start, end, coef, c)
                    for (stack, _), coef, c in zip(order, coefs, centred, strict=True)
                ]
                clipped += write_interleaved(writer, chunks, np.dtype(np.int16))
        n_values = n_written_values(n_frames, len(order), frame_shape)
        check_clipped(clipped, n_values, path, np.int16, settings.max_clipped_fraction)
    order_roles = [roles[ch] for ch in sorted(by_channel)]
    record = {
        **source,
        "method": "per-pixel least squares of the vessel trace, means kept",
        "channel_order_one_based": sorted(by_channel),
        "coefficient_median": {
            role: float(np.median(coef)) for role, coef in zip(order_roles, coefs, strict=True)
        },
        **output_record("int16", clipped),
    }
    Path(plane_dir, BACKGROUND_RECORD).write_text(json.dumps(record, indent=2))
    return path


def top_rois(dff: np.ndarray, n: int) -> np.ndarray:
    """The *n* ROIs with the largest maximum ΔF/F0, largest first (all-NaN rows last)."""
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    peaks = np.full(len(dff), -np.inf)
    defined = np.isfinite(dff).any(axis=1)
    peaks[defined] = np.nanmax(dff[defined], axis=1)
    return np.argsort(-peaks, kind="stable")[:n]
