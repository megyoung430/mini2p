"""Where the QC notebook saves its plots.

Plots of the preprocessing go in ``<variant>/quality_control/<step>/``. Plots that depend on
a registration go beside it, in ``<variant>/aligned_<channel>/<save_folder>/quality_control/
<step>/``, so registrations of one variant with different settings keep their own, and
registering again (which replaces that folder) removes the plots of the old registration.

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
PREPROCESSING_STEPS = ("raw_mean", "bidiphase", "destriping", "distortion_correction")
REGISTRATION_STEPS = ("motion_registration", "frame_order", "fluorescence")
QC_STEPS = PREPROCESSING_STEPS + REGISTRATION_STEPS
QC_DPI = 200


def qc_step_dir(variant_dir: Path, step: str, registration_dir: Path | None = None) -> Path:
    """The folder for one QC step's plots.

    ``<variant_dir>/quality_control/<step>`` for a preprocessing step;
    ``<registration_dir>/quality_control/<step>`` for a step that depends on the
    registration, where ``registration_dir`` is that registration's Suite2p folder inside the
    variant (the parent of its ``plane0``).

    Raises:
        ValueError: For an unknown step, or a registration step without a registration
            folder inside ``variant_dir``.
    """
    if step in PREPROCESSING_STEPS:
        return Path(variant_dir) / QC_FOLDER / step
    if step not in REGISTRATION_STEPS:
        raise ValueError(f"Unknown QC step {step!r}; one of {', '.join(QC_STEPS)}")
    if registration_dir is None:
        raise ValueError(f"The {step} plots go in their registration folder; none was given")
    if not Path(registration_dir).is_relative_to(variant_dir):
        raise ValueError(
            f"The registration folder {registration_dir} must be inside the variant "
            f"folder {variant_dir}"
        )
    return Path(registration_dir) / QC_FOLDER / step


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
    drawn,
    variant_dir: Path,
    step: str,
    name: str,
    caption: str,
    registration_dir: Path | None = None,
    dpi: int = QC_DPI,
) -> Path:
    """Save the figure in its step's folder (``qc_step_dir``) as ``<name>.png/.pdf/.svg``
    + ``.caption.txt``.

    Returns the path without extension. ``name`` is a file stem: no folders, no extension.
    """
    if not name or Path(name).name != name or Path(name).suffix or name in (".", ".."):
        raise ValueError(f"A QC figure name must be a plain file stem, not {name!r}")
    stem = qc_step_dir(variant_dir, step, registration_dir) / name
    save_preview(figure_of(drawn), stem, caption, dpi=dpi)
    return stem
