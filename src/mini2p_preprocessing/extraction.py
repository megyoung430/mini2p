"""Fluorescence extraction from Cellpose masks on the Suite2p registered binaries.

Third step of the workflow, after registration and Cellpose segmentation of the
registered mean image. Suite2p's own outputs in the plane folder (F, Fneu, stat,
iscell, spks and, with two channels, F_chan2 and Fneu_chan2) are replaced.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import suite2p
import torch
from scipy.stats import skew
from suite2p.detection import roi_stats
from suite2p.extraction import oasis
from suite2p.extraction.extract import extraction_wrapper
from suite2p.extraction.masks import create_masks

from .config import FluorescenceSettings, PipelineSettings
from .fluorescence import baseline, neuropil_correct
from .registration import resolve_registered_binary
from .rois import masks_to_stat

log = logging.getLogger(__name__)
EXTRACTION_RECORD = "cellpose_extraction.json"
RED_TRACE_FILES = ("F_chan2", "Fneu_chan2")


@dataclass(frozen=True)
class ExtractionResult:
    stat: np.ndarray
    f: np.ndarray
    fneu: np.ndarray
    f_chan2: np.ndarray | None
    fneu_chan2: np.ndarray | None
    iscell: np.ndarray
    spks: np.ndarray


# ---- analysis: pure steps ---------------------------------------------------------


def build_stat(labels: np.ndarray, ly: int, lx: int, diameter: int) -> np.ndarray:
    """Suite2p stat for every Cellpose ROI, with none removed.

    ``median=True`` normalises ROI sizes to the median of all ROIs, as Suite2p does
    for its own Cellpose detection. The default overlap limit stays because it is
    what fills in each ROI's ``overlap`` field; Cellpose labels cannot overlap, so
    it never removes one, and the count check below makes sure of that.
    """
    entries = masks_to_stat(labels)
    stat = roi_stats(np.array(entries), ly, lx, diameter=[diameter, diameter], median=True)
    if len(stat) != len(entries):
        raise RuntimeError(f"roi_stats kept {len(stat)} of {len(entries)} Cellpose ROIs")
    return stat


def classify_rois(
    stat: np.ndarray, f: np.ndarray, fneu: np.ndarray, coefficient: float
) -> np.ndarray:
    """Suite2p's built-in classifier, with the activity features Suite2p adds first.

    Like Suite2p's pipeline, this stores ``snr``, ``skew`` and ``std`` of the
    neuropil-corrected trace in each stat entry; without them the classifier
    silently uses ROI shape alone. It was trained on Suite2p-detected ROIs, so
    its scores on Cellpose ROIs are advisory.
    """
    delta = neuropil_correct(f, fneu, coefficient)
    flat = np.flatnonzero(delta.var(axis=1) == 0)
    if flat.size:
        # A constant trace gives NaN features that the classifier still scores highly;
        # it usually means the ROI sits in a cropped or saturated part of the image.
        raise ValueError(
            f"ROIs {flat.tolist()} have constant traces; remove them from the Cellpose masks"
        )
    snr = 1 - 0.5 * np.diff(delta, axis=1).var(axis=1) / delta.var(axis=1)
    sk, sd = skew(delta, axis=1), np.std(delta, axis=1)
    for roi, values in zip(stat, zip(snr, sk, sd, strict=True), strict=True):
        roi["snr"], roi["skew"], roi["std"] = (float(v) for v in values)
    return suite2p.classify(stat=stat, classfile=suite2p.classification.builtin_classfile)


def deconvolve(
    f: np.ndarray, fneu: np.ndarray, fs_hz: float, fl: FluorescenceSettings
) -> np.ndarray:
    """OASIS spikes from Suite2p's baseline-subtracted trace ΔF (not ΔF/F0)."""
    f_corr = neuropil_correct(f, fneu, fl.neuropil_coefficient)
    delta_f = f_corr - baseline(f_corr, fs_hz, fl)
    return oasis(F=delta_f, batch_size=fl.batch_size, tau=fl.tau_s, fs=fs_hz)


def updated_ops(ops: dict, fl: FluorescenceSettings) -> dict:
    """``ops`` with the settings used here, nested as Suite2p 1.x and its GUI read them.

    The flat keys are kept in step for analysis code that reads ``ops["baseline"]``.
    """
    ops = dict(ops)
    ops["extraction"] = {
        **ops.get("extraction", {}),
        "neuropil_coefficient": fl.neuropil_coefficient,
    }
    preprocess = {
        "baseline": fl.baseline,
        "win_baseline": fl.win_baseline_s,
        "sig_baseline": fl.sig_baseline_frames,
        "prctile_baseline": fl.prctile_baseline,
    }
    ops["dcnv_preprocess"] = {**ops.get("dcnv_preprocess", {}), **preprocess}
    ops.update(preprocess, neucoeff=fl.neuropil_coefficient, tau=fl.tau_s)
    return ops


# ---- I/O --------------------------------------------------------------------------


def load_cellpose_labels(seg_file: Path) -> np.ndarray:
    """The label image from a Cellpose ``*_seg.npy`` file (0 = background)."""
    seg = np.load(seg_file, allow_pickle=True).item()
    if "masks" not in seg:
        raise KeyError(f"{seg_file} has no 'masks' entry; is it a Cellpose _seg.npy file?")
    return np.asarray(seg["masks"])


def registered_binaries(ops: dict, plane_dir: Path) -> tuple[Path, Path | None]:
    """Green and (for two-channel planes) red registered binaries, wherever ops puts them."""
    if "nchannels" not in ops:
        raise KeyError(f"ops.npy in {plane_dir} has no 'nchannels'; cannot tell if red exists")
    green = resolve_registered_binary(ops, plane_dir, "reg_file", "data.bin")
    if int(ops["nchannels"]) < 2:
        return green, None
    return green, resolve_registered_binary(ops, plane_dir, "reg_file_chan2", "data_chan2.bin")


def _check_plane(labels: np.ndarray, ops: dict, settings: PipelineSettings, plane_dir: Path):
    shape = (int(ops["Ly"]), int(ops["Lx"]))
    if labels.shape != shape:
        raise ValueError(
            f"Mask shape {labels.shape} differs from the registered movie shape {shape} in "
            f"{plane_dir}; segment the mean image of this plane"
        )
    fs_config = settings.registration.fs_hz
    if not np.isclose(float(ops["fs"]), fs_config):
        raise ValueError(f"ops fs={ops['fs']} but config registration.fs_hz={fs_config}")


def extract_traces(
    labels: np.ndarray, ops: dict, plane_dir: Path, settings: PipelineSettings
) -> ExtractionResult:
    """Extract both channels with identical masks, then classify and deconvolve."""
    _check_plane(labels, ops, settings, plane_dir)
    fl = settings.fluorescence
    ly, lx, fs = int(ops["Ly"]), int(ops["Lx"]), float(ops["fs"])
    green_bin, red_bin = registered_binaries(ops, Path(plane_dir))
    stat = build_stat(labels, ly, lx, settings.registration.diameter_px)
    cell_masks, neuropil_masks = create_masks(stat, Ly=ly, Lx=lx)
    f, fneu, f2, fneu2 = extraction_wrapper(
        stat,
        suite2p.io.BinaryFile(ly, lx, green_bin),
        f_reg_chan2=suite2p.io.BinaryFile(ly, lx, red_bin) if red_bin else None,
        cell_masks=cell_masks,
        neuropil_masks=neuropil_masks,
        settings={"batch_size": fl.batch_size},
        device=torch.device(fl.torch_device),
    )
    iscell = classify_rois(stat, f, fneu, fl.neuropil_coefficient)
    spks = deconvolve(f, fneu, fs, fl)
    return ExtractionResult(stat, f, fneu, f2, fneu2, iscell, spks)


def _replace_files(
    plane_dir: Path, arrays: dict[str, object], record: dict, remove: tuple[str, ...] = ()
) -> None:
    """Write every file under a temporary name, then swap them in.

    A failure while writing removes the temporary files and leaves the existing
    outputs untouched. Files named in ``remove`` (stale outputs that would no
    longer match) are deleted only once everything is written, just before the
    swap; the record is swapped in last.
    """
    staged: list[tuple[Path, Path]] = []
    try:
        for name, value in arrays.items():
            tmp = plane_dir / f"{name}.tmp.npy"
            staged.append((tmp, plane_dir / f"{name}.npy"))
            np.save(tmp, value, allow_pickle=True)
        tmp = plane_dir / f"{EXTRACTION_RECORD}.tmp"
        staged.append((tmp, plane_dir / EXTRACTION_RECORD))
        tmp.write_text(json.dumps(record, indent=2))
    except BaseException:
        for tmp, _ in staged:
            tmp.unlink(missing_ok=True)
        raise
    for name in remove:
        (plane_dir / f"{name}.npy").unlink(missing_ok=True)
    for tmp, final in staged:
        os.replace(tmp, final)


def write_extraction(
    result: ExtractionResult,
    ops: dict,
    plane_dir: Path,
    seg_file: Path,
    settings: PipelineSettings,
) -> None:
    plane_dir = Path(plane_dir)
    fl = settings.fluorescence
    arrays = {
        "F": result.f,
        "Fneu": result.fneu,
        "iscell": result.iscell,
        "spks": result.spks,
        "stat": result.stat,
        "ops": updated_ops(ops, fl),
    }
    two_channel = result.f_chan2 is not None
    if two_channel:
        arrays.update({"F_chan2": result.f_chan2, "Fneu_chan2": result.fneu_chan2})
    record = {
        "seg_file": str(Path(seg_file).resolve()),
        "n_rois": int(len(result.stat)),
        "channels": 2 if two_channel else 1,
        "fluorescence_settings": fl.model_dump(),
        "diameter_px": settings.registration.diameter_px,
        "created_utc": datetime.now(UTC).isoformat(),
    }
    # Red traces from an earlier segmentation would no longer match these ROIs.
    _replace_files(plane_dir, arrays, record, remove=() if two_channel else RED_TRACE_FILES)


def run_cellpose_extraction(
    seg_file: Path, plane_dir: Path, settings: PipelineSettings
) -> ExtractionResult:
    """Extract from Cellpose masks and replace the plane's Suite2p trace files."""
    plane_dir = Path(plane_dir)
    ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
    result = extract_traces(load_cellpose_labels(seg_file), ops, plane_dir, settings)
    write_extraction(result, ops, plane_dir, seg_file, settings)
    log.info("Extracted %d Cellpose ROIs into %s", len(result.stat), plane_dir)
    return result
