"""One recording end to end: phase/destripe -> distortion -> Suite2p registration."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from matplotlib import rc_context
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


QC_FONTS = ["Arial", "Liberation Sans", "DejaVu Sans"]  # Arial where installed

FRAME_PHASE_CAPTION = """Per-frame bidiphase correction of {run_name}; the structural channel
set the phase for {structural_pct:.0f}% of {n_frames} frames.
Top: best odd-row shift of each single frame in each channel (dots) and the shift applied
(black). Grey stretches: the structural channel was not trusted there (its per-frame
phases were inconsistent, or the functional channel did not back them: it neither
preferred the structural phase over {support} frames nor agreed within {agreement_px} px
without contradicting it), so the functional channel averaged over {window} frames set
the phase.
Bottom: even/odd line correlation lost at the applied shift in the {lost_in} channel,
relative to that frame's best shift; 0 = fixed, spikes = frames not fully fixed.
"""


def save_frame_phase_plot(
    corrected_tiff: Path, summary: CorrectionSummary, settings: PipelineSettings, run_name: str
) -> Path:
    """``<stem>.phase_frames.{png,pdf,svg}`` and a caption beside the corrected TIFF."""
    chosen = summary.frame_phases
    stem = corrected_tiff.with_suffix(".phase_frames")
    with rc_context({"font.family": "sans-serif", "font.sans-serif": QC_FONTS}):
        figure = Figure(figsize=(13, 6), layout="constrained")  # no pyplot: safe in workers
        plot_frame_phases(chosen, axes=figure.subplots(2, 1), run_name=run_name)
        for extension in ("png", "pdf", "svg"):
            figure.savefig(stem.with_name(f"{stem.name}.{extension}"), dpi=120)
    caption = FRAME_PHASE_CAPTION.format(
        run_name=run_name,
        structural_pct=100 * float(chosen.used_structural.mean()),
        n_frames=len(chosen.phase_px),
        agreement_px=settings.phase.agreement_px,
        support=settings.phase.support_window_frames,
        window=settings.phase.fallback_window_frames,
        lost_in="structural" if chosen.has_structural else "functional",
    )
    stem.with_name(f"{stem.name}.caption.txt").write_text(caption)
    return stem.with_name(f"{stem.name}.png")


def warn_unless_int16(suite2p_input: Path) -> None:
    """Suite2p registers int16 and converts anything else with a plain astype, which
    truncates and wraps; say so when its input is not int16 (an older float32 run, or a
    cluster distortion script writing floats)."""
    with tifffile.TiffFile(suite2p_input) as tif:
        dtype = tif.pages[0].dtype
    if dtype != np.int16:
        log.warning(
            "Suite2p input %s is %s, not int16: Suite2p will convert it with astype(int16), "
            "truncating towards zero and wrapping values beyond the int16 range",
            suite2p_input,
            dtype,
        )


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
        save_frame_phase_plot(corrected, summary, settings, run_name_from_output_dir(output_dir))
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
    warn_unless_int16(suite2p_input)

    plane_dir = run_registration(suite2p_input, output_dir, settings)
    _log_stage("Suite2p registration", started)
    return PipelineResult(corrected, suite2p_input, plane_dir, summary)
