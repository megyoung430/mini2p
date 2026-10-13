"""Fixed-contrast mean-image exports and review PDFs across many runs."""

from __future__ import annotations

import json
import logging
import re
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.backends.backend_pdf import PdfPages
from pydantic import BaseModel, ConfigDict

from .contrast import red_green_overlay, scale_contrast
from .qc import suite2p_mean_images

log = logging.getLogger(__name__)
SESSION_PATTERN = re.compile(r"^ses-(\d+)_")
RUN_PATTERN = re.compile(r"^run-(\d+)$")

GREEN_PNG = "meanImg_green_functional_contrast.png"
RED_PNG = "meanImg_red_structural_contrast.png"
OLDER_SETTINGS_JSON = "meanImg_contrast_settings.json"
FIXED_SETTINGS_JSON = "meanImg_fixed_contrast_settings.json"


@dataclass(frozen=True)
class Contrast:
    black_percentile: float = 5.0
    white_percentile: float = 99.5
    gamma: float = 1.2

    def label(self) -> str:
        return (
            f"black {self.black_percentile:g}%, white {self.white_percentile:g}%, "
            f"gamma {self.gamma:g}"
        )


class ExportSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    subject_dir: Path
    session_first: int
    session_last: int
    expected_runs: int  # exports and PDFs refuse to run on any other count
    # Folder between run-XX/ and suite2p/, e.g. "without_destriping".
    run_subdir: str = ""
    black_percentile: float = 5.0
    white_percentile: float = 99.5
    gamma: float = 1.2
    high_resolution_dpi: int = 400
    overwrite: bool = False  # per-run TIFF/PNG/JSON exports
    pdf_overwrite: bool = False

    @property
    def contrast(self) -> Contrast:
        return Contrast(self.black_percentile, self.white_percentile, self.gamma)

    @property
    def session_label(self) -> str:
        return f"sessions-{self.session_first}-{self.session_last}"


def load_export_settings(path: Path) -> ExportSettings:
    with Path(path).open("rb") as handle:
        return ExportSettings.model_validate(tomllib.load(handle)["exports"])


@dataclass(frozen=True)
class RunPlane:
    session: int
    run: int
    session_name: str
    run_name: str
    plane_dir: Path


# ---- I/O: discovery ---------------------------------------------------------------


def find_run_planes(
    subject_dir: Path, first_session: int, last_session: int, run_subdir: str = ""
) -> list[RunPlane]:
    """Every ``ses-*/funcimg/run-*/<run_subdir>/suite2p/plane0`` with an ops.npy, in order."""
    found = []
    for session_dir in Path(subject_dir).glob("ses-*"):
        session_match = SESSION_PATTERN.match(session_dir.name)
        if not session_match or not first_session <= int(session_match.group(1)) <= last_session:
            continue
        for run_dir in (session_dir / "funcimg").glob("run-*"):
            run_match = RUN_PATTERN.match(run_dir.name)
            plane_dir = run_dir / run_subdir / "suite2p" / "plane0"
            if run_match and (plane_dir / "ops.npy").is_file():
                found.append(
                    RunPlane(
                        int(session_match.group(1)),
                        int(run_match.group(1)),
                        session_dir.name,
                        run_dir.name,
                        plane_dir,
                    )
                )
    return sorted(found, key=lambda r: (r.session, r.run))


def require_run_count(runs: list[RunPlane], expected: int) -> None:
    if len(runs) != expected:
        raise RuntimeError(
            f"Expected {expected} runs but found {len(runs)}; inspect missing or extra runs first"
        )


def load_means(plane_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    ops = np.load(Path(plane_dir) / "ops.npy", allow_pickle=True).item()
    missing = [key for key in ("meanImg", "meanImg_chan2") if key not in ops]
    if missing:
        raise KeyError(f"{plane_dir / 'ops.npy'} is missing {missing}")
    green, red = suite2p_mean_images(ops)
    if green.shape != red.shape:
        raise ValueError(f"{plane_dir}: mean image shapes differ: {green.shape} vs {red.shape}")
    return green, red


# ---- I/O: per-run mean-image export -----------------------------------------------


def _save_unless_present(path: Path, save: Callable[[Path], None], overwrite: bool) -> str:
    if path.exists() and not overwrite:
        return "existing"
    save(path)
    return "saved"


def export_mean_images(
    plane_dir: Path, contrast: Contrast, overwrite: bool = False
) -> dict[str, str]:
    """Raw float TIFFs, contrast PNGs, an overlay and a settings JSON beside ops.npy.

    With functional_chan=2 Suite2p stores physical green channel 2 in meanImg and
    physical red channel 1 in meanImg_chan2.
    """
    plane_dir = Path(plane_dir)
    green_raw, red_raw = load_means(plane_dir)
    args = (contrast.black_percentile, contrast.white_percentile, contrast.gamma)
    green, green_black, green_white = scale_contrast(green_raw, *args)
    red, red_black, red_white = scale_contrast(red_raw, *args)
    outputs = {
        "green raw TIFF": plane_dir / "meanImg_green_functional_raw.tif",
        "red raw TIFF": plane_dir / "meanImg_red_structural_raw.tif",
        "green contrast PNG": plane_dir / GREEN_PNG,
        "red contrast PNG": plane_dir / RED_PNG,
        "overlay contrast PNG": plane_dir / "meanImg_red_green_overlay_contrast.png",
    }
    savers = {
        "green raw TIFF": lambda p: tifffile.imwrite(p, green_raw, photometric="minisblack"),
        "red raw TIFF": lambda p: tifffile.imwrite(p, red_raw, photometric="minisblack"),
        "green contrast PNG": lambda p: plt.imsave(p, green, cmap="gray", vmin=0, vmax=1),
        "red contrast PNG": lambda p: plt.imsave(p, red, cmap="gray", vmin=0, vmax=1),
        "overlay contrast PNG": lambda p: plt.imsave(
            p, red_green_overlay(red, green), vmin=0, vmax=1
        ),
    }
    states = {
        label: _save_unless_present(outputs[label], savers[label], overwrite) for label in outputs
    }
    record = {
        "ops_path": str(plane_dir / "ops.npy"),
        "black_percentile": contrast.black_percentile,
        "white_percentile": contrast.white_percentile,
        "gamma": contrast.gamma,
        "green_functional_physical_channel_2": {
            "suite2p_key": "meanImg",
            "black_value": green_black,
            "white_value": green_white,
        },
        "red_structural_physical_channel_1": {
            "suite2p_key": "meanImg_chan2",
            "black_value": red_black,
            "white_value": red_white,
        },
        "outputs": {label: str(path) for label, path in outputs.items()},
    }
    states["settings JSON"] = _save_unless_present(
        plane_dir / FIXED_SETTINGS_JSON,
        lambda p: p.write_text(json.dumps(record, indent=2)),
        overwrite,
    )
    return states


def export_all_mean_images(
    runs: list[RunPlane],
    subject_dir: Path,
    first_session: int,
    last_session: int,
    contrast: Contrast,
    overwrite: bool = False,
) -> Path:
    summary = []
    for index, run in enumerate(runs, start=1):
        states = export_mean_images(run.plane_dir, contrast, overwrite)
        summary.append(
            {
                "session": run.session,
                "run": run.run_name,
                "plane_dir": str(run.plane_dir),
                "states": states,
            }
        )
        log.info("[%02d/%02d] ses-%d %s: %s", index, len(runs), run.session, run.run_name, states)
    path = (
        Path(subject_dir) / f"meanImg_export_summary_sessions-{first_session}-{last_session}.json"
    )
    path.write_text(
        json.dumps(
            {
                "run_count": len(summary),
                "black_percentile": contrast.black_percentile,
                "white_percentile": contrast.white_percentile,
                "gamma": contrast.gamma,
                "runs": summary,
            },
            indent=2,
        )
    )
    return path


def save_contrast_choice(
    plane_dir: Path,
    green_raw: np.ndarray,
    red_raw: np.ndarray,
    green: Contrast,
    red: Contrast,
) -> dict[str, Path]:
    """Save hand-tuned display images beside ops.npy, plus the unscaled means as TIFFs."""
    plane_dir = Path(plane_dir)
    green_display, green_black, green_white = scale_contrast(
        green_raw, green.black_percentile, green.white_percentile, green.gamma
    )
    red_display, red_black, red_white = scale_contrast(
        red_raw, red.black_percentile, red.white_percentile, red.gamma
    )
    paths = {
        "green PNG": plane_dir / "meanImg_green_contrast.png",
        "red PNG": plane_dir / "meanImg_red_contrast.png",
        "overlay PNG": plane_dir / "meanImg_red_green_overlay.png",
        "green raw TIFF": plane_dir / "meanImg_green_raw.tif",
        "red raw TIFF": plane_dir / "meanImg_red_raw.tif",
        "settings JSON": plane_dir / OLDER_SETTINGS_JSON,
    }
    plt.imsave(paths["green PNG"], green_display, cmap="gray", vmin=0, vmax=1)
    plt.imsave(paths["red PNG"], red_display, cmap="gray", vmin=0, vmax=1)
    plt.imsave(paths["overlay PNG"], red_green_overlay(red_display, green_display), vmin=0, vmax=1)
    tifffile.imwrite(paths["green raw TIFF"], np.asarray(green_raw, dtype=np.float32))
    tifffile.imwrite(paths["red raw TIFF"], np.asarray(red_raw, dtype=np.float32))
    record = {
        name: {
            "black_percentile": c.black_percentile,
            "white_percentile": c.white_percentile,
            "gamma": c.gamma,
            "black_value": black,
            "white_value": white,
        }
        for name, c, black, white in [
            ("green", green, green_black, green_white),
            ("red", red, red_black, red_white),
        ]
    }
    paths["settings JSON"].write_text(json.dumps(record, indent=2))
    return paths


# ---- I/O: review PDFs -------------------------------------------------------------


def runs_with_both_settings_files(runs: list[RunPlane]) -> list[RunPlane]:
    return [
        r
        for r in runs
        if (r.plane_dir / OLDER_SETTINGS_JSON).is_file()
        and (r.plane_dir / FIXED_SETTINGS_JSON).is_file()
    ]


def _refuse_overwrite(path: Path, overwrite: bool) -> None:
    if Path(path).exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass overwrite=True to replace it")


def write_comparison_pdf(
    runs: list[RunPlane], output: Path, contrast: Contrast, title: str, overwrite: bool = False
) -> Path:
    """One landscape page per run: exported green and red contrast PNGs side by side."""
    missing = [
        str(r.plane_dir / name)
        for r in runs
        for name in (GREEN_PNG, RED_PNG)
        if not (r.plane_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError("Run export_all_mean_images first. Missing:\n" + "\n".join(missing))
    _refuse_overwrite(output, overwrite)
    with PdfPages(output) as pdf:
        info = pdf.infodict()
        info["Title"], info["Author"] = title, "Mini2P preprocessing pipeline"
        info["Subject"] = f"Suite2p green and red mean images; {contrast.label()}"
        for page, run in enumerate(runs, start=1):
            fig, axes = plt.subplots(1, 2, figsize=(11.69, 7.25), facecolor="white")
            fig.subplots_adjust(left=0.035, right=0.965, bottom=0.075, top=0.84, wspace=0.045)
            fig.suptitle(
                f"Session {run.session} | {run.run_name}", fontsize=19, fontweight="bold", y=0.955
            )
            fig.text(
                0.5,
                0.905,
                f"{run.session_name}  |  fixed contrast: {contrast.label()}",
                ha="center",
                va="center",
                fontsize=10.5,
                color="0.30",
            )
            panels = [
                (GREEN_PNG, "Green functional\nphysical channel 2 | Suite2p meanImg"),
                (RED_PNG, "Red structural\nphysical channel 1 | Suite2p meanImg_chan2"),
            ]
            for ax, (name, panel_title) in zip(axes, panels, strict=True):
                ax.imshow(
                    plt.imread(run.plane_dir / name),
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                    interpolation="nearest",
                )
                ax.set_title(panel_title, fontsize=12, pad=10)
                ax.axis("off")
            fig.text(
                0.5,
                0.025,
                f"Page {page} of {len(runs)}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="0.45",
            )
            pdf.savefig(fig, dpi=180, facecolor="white")
            plt.close(fig)
    return Path(output)


def write_high_resolution_pdf(
    runs: list[RunPlane],
    output: Path,
    contrast: Contrast,
    title: str,
    dpi: int = 400,
    overwrite: bool = False,
) -> Path:
    """Two portrait pages per run (green then red), contrast recomputed from ops.npy."""
    _refuse_overwrite(output, overwrite)
    n_pages = 2 * len(runs)
    args = (contrast.black_percentile, contrast.white_percentile, contrast.gamma)
    with PdfPages(output) as pdf:
        info = pdf.infodict()
        info["Title"], info["Author"] = title, "Mini2P preprocessing pipeline"
        info["Subject"] = (
            f"{n_pages} pages: green then red for each of {len(runs)} runs; {contrast.label()}"
        )
        page = 0
        for run in runs:
            green_raw, red_raw = load_means(run.plane_dir)
            channel_pages = [
                (
                    *scale_contrast(green_raw, *args),
                    "Green functional",
                    "Physical channel 2 | Suite2p meanImg",
                    "#2d7f4f",
                ),
                (
                    *scale_contrast(red_raw, *args),
                    "Red structural",
                    "Physical channel 1 | Suite2p meanImg_chan2",
                    "#9b3a3a",
                ),
            ]
            for image, black, white, channel_title, detail, accent in channel_pages:
                page += 1
                fig = plt.figure(figsize=(8.5, 11), facecolor="white")
                ax = fig.add_axes([0.045, 0.115, 0.91, 0.73])
                ax.imshow(
                    image, cmap="gray", vmin=0, vmax=1, interpolation="nearest", aspect="equal"
                )
                ax.axis("off")
                fig.text(
                    0.5,
                    0.955,
                    f"Session {run.session} | {run.run_name}",
                    ha="center",
                    va="top",
                    fontsize=22,
                    fontweight="bold",
                )
                fig.text(
                    0.5,
                    0.905,
                    channel_title,
                    ha="center",
                    va="top",
                    fontsize=18,
                    fontweight="bold",
                    color=accent,
                )
                fig.text(0.5, 0.865, detail, ha="center", va="top", fontsize=11, color="0.25")
                fig.text(
                    0.5,
                    0.075,
                    f"{run.session_name} | values {black:.3g}-{white:.3g} | {contrast.label()}",
                    ha="center",
                    va="center",
                    fontsize=9.5,
                    color="0.35",
                )
                fig.text(
                    0.5,
                    0.035,
                    f"Page {page} of {n_pages}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="0.50",
                )
                pdf.savefig(fig, dpi=dpi, facecolor="white", bbox_inches=None)
                plt.close(fig)
    return Path(output)
