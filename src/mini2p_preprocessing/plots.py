"""QC figures. Each function draws into axes it is given and creates them only if not."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

from .bidiphase import PhaseEstimate
from .config import PipelineSettings
from .contrast import red_green_overlay

ROLE_COLORS = {"structural": "tab:red", "functional": "tab:green"}


def _axes_grid(axes, nrows: int, ncols: int, figsize) -> np.ndarray:
    if axes is not None:
        return np.asarray(axes).reshape(nrows, ncols)
    _, created = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True, squeeze=False)
    return created


def plot_phase_trace(trace: list[dict], settings: PipelineSettings, axes=None) -> np.ndarray:
    """Top: phase per chunk, coloured by source, boundary hits ringed. Bottom: scores."""
    axes = _axes_grid(axes, 2, 1, (12, 6)).ravel()
    centers = np.array([(r["start"] + r["end"] - 1) / 2 for r in trace])
    values = np.array([r["phase_px"] for r in trace])
    boundary = np.array([r["at_boundary"] for r in trace], dtype=bool)
    sources = np.array([r["selected_source"] for r in trace])
    for role, color in ROLE_COLORS.items():
        chosen = sources == role
        axes[0].scatter(
            centers[chosen], values[chosen], color=color, s=35, label=f"{role} selected", zorder=3
        )
    axes[0].plot(centers, values, color="0.6", linewidth=1)
    axes[0].scatter(
        centers[boundary],
        values[boundary],
        facecolors="none",
        edgecolors="black",
        s=70,
        label="near search boundary",
        zorder=4,
    )
    axes[0].axhline(0, color="0.7", linewidth=1)
    if settings.phase.estimator == "wide":
        for limit in (settings.phase.max_shift_px, -settings.phase.max_shift_px):
            axes[0].axhline(limit, color="red", linestyle=":", alpha=0.5)
    axes[0].set(
        ylabel="Odd-row shift (pixels)",
        title=f"Dynamic biphase ({settings.phase.chunk_frames}-frame chunks; "
        f"{settings.phase.estimator}; source={settings.phase.alignment_source})",
    )
    axes[0].legend(loc="best")
    for role, color in ROLE_COLORS.items():
        scores = np.array([r["score_by_source"].get(role, np.nan) for r in trace])
        if np.any(np.isfinite(scores)):
            axes[1].plot(centers, scores, ".-", color=color, label=role)
    axes[1].set(xlabel="Timepoint", ylabel="Best normalized score")
    axes[1].legend(loc="best")
    return axes


def plot_score_curve(estimate: PhaseEstimate, title: str, ax=None):
    if estimate.candidates is None:
        raise ValueError("Score curves exist only for the 'wide' estimator")
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(estimate.candidates, estimate.scores, ".-")
    ax.axvline(
        estimate.phase_px, color="red", linestyle="--", label=f"selected {estimate.phase_px:+d} px"
    )
    ax.set(
        xlabel="Candidate odd-row shift (pixels)",
        ylabel="Normalized adjacent-line score",
        title=title,
    )
    ax.legend()
    return ax


def plot_channel_pairs(
    left: Sequence[np.ndarray],
    right: Sequence[np.ndarray],
    titles: tuple[str, str],
    channel_names: dict[int, str],
    axes=None,
    percentiles=(1, 99.8),
    x_slice=slice(None),
) -> np.ndarray:
    """One row per channel: two 2-D images with shared display limits within the row."""
    axes = _axes_grid(axes, len(left), 2, (12, 5 * len(left)))
    for ch, (img_a, img_b) in enumerate(zip(left, right, strict=True)):
        finite = np.concatenate([img_a[np.isfinite(img_a)], img_b[np.isfinite(img_b)]])
        vmin, vmax = np.percentile(finite, percentiles)
        for ax, image, title in zip(axes[ch], (img_a, img_b), titles, strict=True):
            ax.imshow(image[:, x_slice], cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_title(f"{channel_names[ch + 1]}\n{title}")
            ax.axis("off")
    return axes


def plot_registration_traces(ops: dict, fs_hz: float, axes=None) -> np.ndarray:
    """Rigid offsets, frame-to-reference correlation and bad-frame flags over time."""
    xoff = np.asarray(ops.get("xoff", []), dtype=float).ravel()
    yoff = np.asarray(ops.get("yoff", []), dtype=float).ravel()
    n = min(len(xoff), len(yoff))
    if n == 0:
        raise ValueError("ops has no rigid registration offsets")
    corr = np.asarray(ops.get("corrXY", []), dtype=float).ravel()[:n]
    bad = np.asarray(ops.get("badframes", np.zeros(n, bool)), dtype=bool).ravel()[:n]
    minutes = np.arange(n) / fs_hz / 60
    axes = _axes_grid(axes, 3, 1, (13, 8)).ravel()
    axes[0].plot(minutes, xoff[:n], linewidth=0.7, label="xoff")
    axes[0].plot(minutes, yoff[:n], linewidth=0.7, label="yoff")
    axes[0].set_ylabel("Rigid shift (px)")
    axes[0].legend(loc="upper right")
    axes[1].plot(minutes[: len(corr)], corr, color="tab:green", linewidth=0.7)
    axes[1].set_ylabel("corrXY")
    axes[2].vlines(minutes[np.flatnonzero(bad)], 0, 1, color="tab:red", linewidth=0.7)
    axes[2].set(ylim=(-0.05, 1.05), ylabel="Bad frame", xlabel="Time (minutes)")
    return axes


def plot_nonrigid_offsets(ops: dict, fs_hz: float, ax=None):
    xoff1 = np.asarray(ops.get("xoff1", []), dtype=float)
    yoff1 = np.asarray(ops.get("yoff1", []), dtype=float)
    if xoff1.ndim < 2 or yoff1.ndim < 2:
        raise ValueError("ops has no nonrigid block offsets")
    if ax is None:
        _, ax = plt.subplots(figsize=(13, 3.5))
    minutes = np.arange(len(xoff1)) / fs_hz / 60
    ax.plot(minutes, np.nanmedian(xoff1, axis=1), linewidth=0.7, label="median nonrigid xoff1")
    ax.plot(minutes, np.nanmedian(yoff1, axis=1), linewidth=0.7, label="median nonrigid yoff1")
    ax.set(
        xlabel="Time (minutes)",
        ylabel="Median block shift (px)",
        title="Nonrigid offsets across spatial blocks",
    )
    ax.legend()
    return ax


def plot_contrast_panels(
    green: np.ndarray,
    red: np.ndarray,
    green_range: tuple[float, float],
    red_range: tuple[float, float],
    axes=None,
) -> np.ndarray:
    """Green, red and red-green overlay of display images already scaled to [0, 1]."""
    axes = _axes_grid(axes, 1, 3, (17, 6)).ravel()
    panels = [
        (green, f"Green / functional\nvalues {green_range[0]:.3g}–{green_range[1]:.3g}"),
        (red, f"Red / structural\nvalues {red_range[0]:.3g}–{red_range[1]:.3g}"),
        (red_green_overlay(red, green), "Red–green overlay"),
    ]
    for ax, (image, title) in zip(axes, panels, strict=True):
        ax.imshow(
            image, cmap="gray" if image.ndim == 2 else None, vmin=0, vmax=1, interpolation="nearest"
        )
        ax.set_title(title)
        ax.axis("off")
    return axes
