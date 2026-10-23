"""One recording end to end: phase/destripe -> distortion -> Suite2p registration."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from .config import (
    SETTINGS_RECORD,
    PipelineSettings,
    check_variant_folder,
    corrected_tiff_path,
    distortion_corrected,
    run_name_from_output_dir,
    undistorted_tiff_path,
)
from .correction import CorrectionSummary, preprocess_tiff
from .distortion import write_distortion_corrected
from .plots import plot_frame_phases
from .progress import format_duration
from .registration import run_registration

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineResult:
    corrected_tiff: Path
    suite2p_input_tiff: Path
    plane_dir: Path
    correction: CorrectionSummary


def suite2p_input_for(output_dir: Path, input_tiff: Path, settings: PipelineSettings) -> Path:
    """The TIFF Suite2p registers: undistorted unless distortion correction is off."""
    if settings.distortion.mode == "off":
        return corrected_tiff_path(output_dir, input_tiff, settings)
    return undistorted_tiff_path(output_dir, input_tiff, settings)


def _log_stage(stage: str, started: float) -> float:
    now = time.monotonic()
    log.info("%s took %s", stage, format_duration(now - started))
    return now


def save_frame_phase_plot(corrected_tiff: Path, summary: CorrectionSummary, run_name: str) -> Path:
    """``<stem>.phase_frames.png`` beside the corrected TIFF, for screening runs by eye."""
    record = np.load(corrected_tiff.with_suffix(".phase_frames.npz"))
    figure = Figure(figsize=(12, 6), layout="constrained")  # no pyplot: safe in batch workers
    plot_frame_phases(
        summary.frame_phases,
        record["structural_loss"],
        axes=figure.subplots(2, 1),
        run_name=run_name,
    )
    path = corrected_tiff.with_suffix(".phase_frames.png")
    figure.savefig(path, dpi=120)
    return path


def run_pipeline(input_tiff: Path, output_dir: Path, settings: PipelineSettings) -> PipelineResult:
    # Absolute paths go into the sidecars, which the batch planner compares later.
    input_tiff, output_dir = Path(input_tiff).resolve(), Path(output_dir).resolve()
    if not input_tiff.is_file():
        raise FileNotFoundError(f"Input TIFF not found: {input_tiff}")
    check_variant_folder(
        output_dir, settings.destripe.enabled, distortion_corrected=distortion_corrected(settings)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    corrected = corrected_tiff_path(output_dir, input_tiff, settings)
    started = time.monotonic()
    summary = preprocess_tiff(input_tiff, corrected, settings, run_name_from_output_dir(output_dir))
    if summary.frame_phases is not None:
        save_frame_phase_plot(corrected, summary, run_name_from_output_dir(output_dir))
    started = _log_stage("Phase correction", started)
    # Written only once outputs were (re)written, so it always describes what is on disk.
    (output_dir / SETTINGS_RECORD).write_text(settings.model_dump_json(indent=2))

    suite2p_input = suite2p_input_for(output_dir, input_tiff, settings)
    if settings.distortion.mode == "local_batched":
        write_distortion_corrected(corrected, suite2p_input, settings)
        started = _log_stage("Distortion correction", started)
    elif settings.distortion.mode == "cluster_full" and not suite2p_input.is_file():
        raise FileNotFoundError(f"Cluster distortion output not found: {suite2p_input}")
    log.info("Suite2p input (%s distortion): %s", settings.distortion.mode, suite2p_input)

    plane_dir = run_registration(suite2p_input, output_dir, settings)
    _log_stage("Suite2p registration", started)
    return PipelineResult(corrected, suite2p_input, plane_dir, summary)
