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


def plot_trace_audit(
    f: np.ndarray,
    fneu: np.ndarray,
    result,
    roi: int,
    fs_hz: float,
    frames: slice = slice(None),
    axes=None,
    time_label: str = "Time (s)",
) -> np.ndarray:
    """One ROI: raw F and Fneu; corrected F with its F0; ΔF/F0 (gaps where F0 is invalid)."""
    axes = _axes_grid(axes, 3, 1, (12, 8)).ravel()
    t = np.arange(f.shape[1])[frames] / fs_hz
    axes[0].plot(t, f[roi, frames], color="tab:green", linewidth=0.7, label="F")
    axes[0].plot(t, fneu[roi, frames], color="0.4", linewidth=0.7, label="Fneu")
    axes[0].set(ylabel="Raw fluorescence", title=f"ROI {roi}")
    axes[1].plot(
        t, result.f_corr[roi, frames], color="tab:green", linewidth=0.7, label="Fcorr = F − c·Fneu"
    )
    axes[1].plot(t, result.f0[roi, frames], color="black", linewidth=1.2, label="F0")
    axes[1].axhline(0, color="tab:red", linewidth=0.8, linestyle=":")
    invalid = result.invalid_f0[roi, frames]
    if invalid.any():
        axes[1].fill_between(
            t,
            0,
            1,
            where=invalid,
            color="tab:red",
            alpha=0.15,
            transform=axes[1].get_xaxis_transform(),
            label="F0 ≤ min_f0",
        )
    axes[1].set_ylabel("Corrected F")
    axes[2].plot(t, result.dff[roi, frames], color="tab:blue", linewidth=0.7)
    axes[2].axhline(0, color="0.6", linewidth=0.8)
    axes[2].set(ylabel="ΔF/F0", xlabel=time_label)
    _note_if_undefined(axes[2], result.dff[roi, frames])
    for ax in axes:
        ax.set_xlim(t[0], t[-1])
    for ax in axes[:2]:
        ax.legend(loc="upper right", fontsize=8)
    return axes


def plot_f0_overview(table: dict, min_f0: float, axes=None) -> np.ndarray:
    """Left: median F0 per ROI. Right: how often corrected F goes negative vs F0."""
    axes = _axes_grid(axes, 1, 2, (12, 4)).ravel()
    axes[0].hist(table["median_f0"], bins=40, color="0.5")
    axes[0].axvline(min_f0, color="tab:red", linestyle="--", label=f"min_f0 = {min_f0:g}")
    axes[0].set(xlabel="Median F0 per ROI", ylabel="ROIs")
    axes[0].legend()
    flagged = table["frac_invalid_f0"] > 0
    axes[1].scatter(
        table["median_f0"][~flagged],
        table["frac_negative_fcorr"][~flagged],
        s=12,
        color="0.4",
        label="F0 valid throughout",
    )
    axes[1].scatter(
        table["median_f0"][flagged],
        table["frac_negative_fcorr"][flagged],
        s=12,
        color="tab:red",
        label="F0 invalid somewhere",
    )
    axes[1].set(xlabel="Median F0 per ROI", ylabel="Fraction of frames with Fcorr < 0")
    axes[1].legend()
    return axes


def plot_cell_fov(
    roi: int, mean_img: np.ndarray, labels: np.ndarray, axes=None, margin_px: int = 10
) -> np.ndarray:
    """ROI ``roi`` (red outline) and the other ROIs (blue), zoomed and in the full field."""
    from .rois import outline_labels

    ys, xs = np.nonzero(labels == roi + 1)
    if not len(ys):
        raise ValueError(f"ROI {roi} is not in the label image")
    axes = _axes_grid(axes, 1, 2, (10, 5)).ravel()
    outline = outline_labels(labels)
    overlay = np.zeros((*labels.shape, 4), np.float32)
    overlay[(outline != 0) & (outline != roi + 1)] = (0.2, 0.4, 1.0, 0.8)
    overlay[outline == roi + 1] = (1.0, 0.1, 0.1, 1.0)
    y0, y1 = max(0, ys.min() - margin_px), min(labels.shape[0], ys.max() + margin_px + 1)
    x0, x1 = max(0, xs.min() - margin_px), min(labels.shape[1], xs.max() + margin_px + 1)
    views = [
        ((slice(y0, y1), slice(x0, x1)), f"ROI {roi} zoomed in"),
        ((slice(None), slice(None)), f"ROI {roi} in the full field of view"),
    ]
    for ax, (window, title) in zip(axes, views, strict=True):
        ax.imshow(mean_img[window], cmap="gray", interpolation="nearest")
        ax.imshow(overlay[window], interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
    return axes


def plot_dff_trace(
    result,
    roi: int,
    fs_hz: float,
    frames: slice = slice(None),
    ax=None,
    label: str = "ΔF/F0",
    color: str = "tab:blue",
    time_label: str = "Time (s)",
):
    """A single ΔF/F0-style trace, e.g. the green/red ratio from ``green_red_dff``."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 2.5))
    t = np.arange(result.dff.shape[1])[frames] / fs_hz
    ax.plot(t, result.dff[roi, frames], color=color, linewidth=0.7, label=label)
    ax.axhline(0, color="0.6", linewidth=0.8)
    ax.set(xlabel=time_label, ylabel=label, xlim=(t[0], t[-1]))
    _note_if_undefined(ax, result.dff[roi, frames])
    return ax


def _note_if_undefined(ax, values: np.ndarray) -> None:
    """An all-NaN ΔF/F0 trace would otherwise look like an empty, broken panel."""
    if not np.isfinite(values).any():
        ax.text(
            0.5,
            0.5,
            "undefined here: F0 ≤ min_f0 in every frame shown",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="tab:red",
        )
