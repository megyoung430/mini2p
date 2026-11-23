"""Where the QC notebook saves its plots: ``<variant>/quality_control/<step>/``.

Each plot is saved as png, pdf and svg, with ``<name>.caption.txt`` beside it saying what
it shows and which data it was drawn from.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from .raw_previews import save_preview

QC_FOLDER = "quality_control"
QC_STEPS = (
    "raw_mean",
    "bidiphase",
    "destriping",
    "distortion_correction",
    "motion_registration",
    "frame_order",
    "fluorescence",
)
QC_DPI = 200


def qc_step_dir(output_dir: Path, step: str) -> Path:
    """``<output_dir>/quality_control/<step>``; the step must be one of ``QC_STEPS``."""
    if step not in QC_STEPS:
        raise ValueError(f"Unknown QC step {step!r}; one of {', '.join(QC_STEPS)}")
    return Path(output_dir) / QC_FOLDER / step


def figure_of(drawn) -> plt.Figure:
    """The figure a plotting function drew on, from the figure, an axes or an axes array."""
    if isinstance(drawn, plt.Figure):
        return drawn
    if isinstance(drawn, Axes):
        return drawn.figure
    if isinstance(drawn, np.ndarray) and drawn.size and isinstance(drawn.flat[0], Axes):
        return drawn.flat[0].figure
    raise TypeError(f"Expected a matplotlib figure or axes, got {type(drawn).__name__}")


def save_qc_figure(
    drawn, output_dir: Path, step: str, name: str, caption: str, dpi: int = QC_DPI
) -> Path:
    """Save the figure in its step's folder as ``<name>.png/.pdf/.svg`` + ``.caption.txt``.

    Returns the path without extension. ``name`` is a file stem: no folders, no extension.
    """
    if not name or Path(name).name != name or Path(name).suffix or name in (".", ".."):
        raise ValueError(f"A QC figure name must be a plain file stem, not {name!r}")
    stem = qc_step_dir(output_dir, step) / name
    save_preview(figure_of(drawn), stem, caption, dpi=dpi)
    return stem
