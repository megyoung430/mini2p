"""QC figures. Each function draws into axes it is given and creates them only if not."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from .bidiphase import PhaseEstimate
from .config import PipelineSettings
from .contrast import red_green_overlay
from .focus import FocusQC, flagged_stretches
from .frame_phase import FramePhases, fallback_stretches

ROLE_COLORS = {"structural": "tab:red", "functional": "tab:green"}
# Okabe-Ito, readable with colour-vision deficiencies.
OKABE_ITO = {"blue": "#0072B2", "orange": "#E69F00", "vermillion": "#D55E00", "grey": "#999999"}


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


def plot_frame_phases(chosen: FramePhases, axes=None, run_name: str = "") -> np.ndarray:
    """Top: each frame's phase in each channel and the applied phase; grey where the
    functional channel set it. Bottom: line correlation lost at the applied phase in the
    structural channel (the functional one if there is no separate structural channel);
    0 = fixed, spikes = frames the correction did not fully fix."""
    axes = _axes_grid(axes, 2, 1, (12, 6)).ravel()
    frames = np.arange(len(chosen.phase_px))
    for start, end in fallback_stretches(chosen.used_structural):
        for ax in axes:
            ax.axvspan(start - 0.5, end - 0.5, color="0.88", linewidth=0, zorder=0)
    axes[0].plot(
        frames,
        chosen.functional_px,
        ".",
        ms=1.5,
        alpha=0.4,
        color=ROLE_COLORS["functional"],
        label="Functional, Single Frames",
    )
    if chosen.has_structural:
        axes[0].plot(
            frames,
            chosen.structural_px,
            ".",
            ms=1.5,
            alpha=0.6,
            color=ROLE_COLORS["structural"],
            label="Structural, Single Frames",
        )
    axes[0].plot(frames, chosen.phase_px, color="black", linewidth=0.8, label="Applied")
    axes[0].set(
        ylabel="Odd-Row Shift (px)",
        title=f"{run_name + ': ' if run_name else ''}Per-Frame Biphase",
    )
    axes[0].legend(
        loc="upper left", bbox_to_anchor=(1.01, 1), markerscale=6, frameon=False, fontsize=8
    )
    lost_in = "Structural" if chosen.has_structural else "Functional"
    axes[1].plot(frames, chosen.loss, color=ROLE_COLORS[lost_in.lower()], linewidth=0.6)
    axes[1].set(xlabel="Frame", ylabel=f"{lost_in} Corr Lost")
    axes[0].sharex(axes[1])
    axes[0].tick_params(labelbottom=False)
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


def _outline(ax, box: tuple[slice, slice]) -> None:
    """Dashed outline around the pixels in ``box`` (rows, columns) of an imshow image."""
    rows, cols = box
    ax.add_patch(
        Rectangle(
            (cols.start - 0.5, rows.start - 0.5),  # imshow pixel centres are at integers
            cols.stop - cols.start,
            rows.stop - rows.start,
            fill=False,
            edgecolor="#E69F00",
            linestyle="--",
            linewidth=1,
        )
    )


def _check_columns(columns, titles, limit_columns, box, x_slice) -> None:
    if len(titles) != len(columns):
        raise ValueError(f"{len(titles)} titles for {len(columns)} columns")
    counts = {len(images) for images in columns}
    if len(counts) != 1:
        raise ValueError(f"Every column needs the same number of channels, got {sorted(counts)}")
    if limit_columns is not None and len(limit_columns) == 0:
        raise ValueError("limit_columns is empty: no images to set the display range from")
    if box is not None:
        if any(s.start is None or s.stop is None or s.start < 0 or s.stop <= s.start for s in box):
            raise ValueError("box needs explicit, non-negative (start, stop) rows and columns")
        if x_slice not in (slice(None), slice(0, None)):
            raise ValueError("box is in full-frame columns, so it cannot be used with x_slice")


def plot_channel_columns(
    columns: Sequence[Sequence[np.ndarray]],
    titles: Sequence[str],
    channel_names: dict[int, str],
    axes=None,
    percentiles=(1, 99.8),
    x_slice=slice(None),
    limit_columns: Sequence[int] | None = None,
    box: tuple[slice, slice] | None = None,
    box_columns: Sequence[int] = (),
) -> np.ndarray:
    """One row per channel, one column per entry of ``columns`` (each a per-channel list of
    2-D images); every image in a row shares one display range.

    ``limit_columns`` picks which columns set that range (default: all), e.g. cropped
    images, so a bright edge artefact in the others cannot wash out the contrast. ``box``
    (rows, columns) is outlined, dashed, on the ``box_columns``.
    """
    _check_columns(columns, titles, limit_columns, box, x_slice)
    n_channels = len(columns[0])
    axes = _axes_grid(axes, n_channels, len(columns), (6 * len(columns), 5 * n_channels))
    limit_columns = range(len(columns)) if limit_columns is None else limit_columns
    for ch in range(n_channels):
        values = np.concatenate([np.ravel(columns[c][ch]) for c in limit_columns])
        values = values[np.isfinite(values)]
        if not len(values):
            raise ValueError(f"Channel row {ch + 1} has no finite pixels to set its display range")
        vmin, vmax = np.percentile(values, percentiles)
        for col, (images, title) in enumerate(zip(columns, titles, strict=True)):
            ax = axes[ch, col]
            ax.imshow(
                images[ch][:, x_slice], cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest"
            )
            ax.set_title(f"{channel_names[ch + 1]}\n{title}")
            ax.axis("off")
            if box is not None and col in box_columns:
                _outline(ax, box)
    return axes


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
    return plot_channel_columns(
        [left, right],
        titles,
        channel_names,
        axes=axes,
        percentiles=percentiles,
        x_slice=x_slice,
    )


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
    grey_range = {"vmin": np.nanmin(mean_img), "vmax": np.nanmax(mean_img)}  # zoom keeps it
    for ax, (window, title) in zip(axes, views, strict=True):
        ax.imshow(mean_img[window], cmap="gray", interpolation="nearest", **grey_range)
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


def plot_masks_on(
    image: np.ndarray, labels: np.ndarray, ax, title: str = "", color=(1.0, 0.3, 0.1, 1.0)
):
    """Mask outlines on ``image``, on one axis."""
    from .rois import outline_labels

    overlay = np.zeros((*labels.shape, 4), np.float32)
    overlay[outline_labels(labels) != 0] = color
    ax.imshow(image, cmap="gray", interpolation="nearest")
    ax.imshow(overlay, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")
    return ax


def plot_mask_alignment(
    plane_image: np.ndarray, original: np.ndarray, aligned: np.ndarray, axes=None, title: str = ""
) -> np.ndarray:
    """Outlines as segmented (left) and after shifting onto this plane (right)."""
    from .rois import outline_labels

    axes = _axes_grid(axes, 1, 2, (14, 7)).ravel()
    for ax, labels, label, color in zip(
        axes,
        (original, aligned),
        ("Masks as segmented", "Masks shifted onto this plane"),
        ((1.0, 0.3, 0.1, 1.0), (0.1, 1.0, 0.3, 1.0)),
        strict=True,
    ):
        overlay = np.zeros((*labels.shape, 4), np.float32)
        overlay[outline_labels(labels) != 0] = color
        ax.imshow(plane_image, cmap="gray", interpolation="nearest")
        ax.imshow(overlay, interpolation="nearest")
        ax.set_title(f"{label}\n{title}".strip())
        ax.axis("off")
    return axes


def plot_natex_overview(result, mean_image: np.ndarray, labels: np.ndarray, axes=None):
    """NATEX per ROI: SNR and event-count histograms, and the ROIs coloured by SNR.

    Args:
        result: A ``natex.PlaneResult``.
        mean_image (np.ndarray): The plane's mean image.
        labels (np.ndarray): ROI label image, ROI i drawn as i + 1.
        axes: Three axes, or None for a new figure.

    Returns:
        np.ndarray: The three axes.
    """
    axes = _axes_grid(axes, 1, 3, (15, 4.5)).ravel()
    good = ~result.bad
    axes[0].hist(result.snr[good], bins=30, color="0.3")
    axes[0].set(xlabel="SNR", ylabel="ROIs", title=f"SNR ({good.sum()} ROIs with signal)")
    axes[1].hist(result.event_count[good], bins=30, color="0.3")
    axes[1].set(xlabel="frames with a cleaned event", ylabel="ROIs", title="Event count")
    low, high = np.nanpercentile(mean_image, [1, 99.8])
    axes[2].imshow(mean_image, cmap="gray", vmin=low, vmax=high)
    snr_image = np.full(labels.shape, np.nan)
    for roi, value in zip(result.rois, result.snr, strict=True):
        snr_image[labels == roi + 1] = value
    shown = axes[2].imshow(snr_image, cmap="viridis", alpha=0.8)
    axes[2].figure.colorbar(shown, ax=axes[2], label="SNR", fraction=0.046)
    axes[2].set_axis_off()
    axes[2].set_title("ROIs coloured by SNR")
    return axes


def plot_natex_cells(result, rows: Sequence[int], axes=None) -> np.ndarray:
    """ΔF/F of chosen rows of a NATEX result, significant transients shaded, cleaned events
    beneath; time in seconds from the start of the run."""
    axes = _axes_grid(axes, len(rows), 1, (14, 1.8 * len(rows))).ravel()
    t = np.arange(result.dff.shape[1]) / result.fs
    for ax, row in zip(axes, rows, strict=True):
        ax.plot(t, result.dff[row], color="0.2", linewidth=0.5)
        ax.fill_between(
            t,
            0,
            1,
            where=result.significant[row],
            color="tab:orange",
            alpha=0.25,
            transform=ax.get_xaxis_transform(),
            linewidth=0,
        )
        ax.plot(t, -result.events[row] * 0.3, color="tab:blue", linewidth=0.5)
        ax.set_ylabel("ΔF/F")
        ax.set_title(
            f"ROI {result.rois[row]}: SNR {result.snr[row]:.2f}, "
            f"{result.event_count[row]} event frames, "
            f"{100 * result.transient_fraction[row]:.1f} % in transients",
            fontsize=9,
            loc="left",
        )
    axes[-1].set_xlabel("Time from the start of the run (s)")
    return axes


def plot_focus_qc(
    result: FocusQC, fs_hz: float, ops: dict, merge_gap_frames: int, axes=None
) -> np.ndarray:
    """Frame match to the mean image, relative brightness and rigid shift over time, with
    flagged stretches shaded in every panel.

    The shift panel shows whether a flagged stretch came with a large, fast excursion. It
    cannot by itself separate z-motion from movement during the scan: one shift per frame
    hides movement within the frame, and z-motion often comes with x/y motion.
    """
    n = len(result.match)
    yoff = np.asarray(ops["yoff"], float).ravel()
    xoff = np.asarray(ops["xoff"], float).ravel()
    if len(yoff) != n or len(xoff) != n:
        raise ValueError(f"ops has shifts for {len(yoff)} frames, the focus QC {n}")
    minutes = np.arange(n) / fs_hz / 60
    axes = _axes_grid(axes, 3, 1, (13, 7)).ravel()
    axes[0].plot(minutes, result.match, color=OKABE_ITO["blue"], linewidth=0.6, label="Frame")
    axes[0].plot(minutes, result.baseline, color="black", linewidth=1, label="Running Median")
    axes[0].set_ylabel("Match To Mean Image")
    axes[1].plot(minutes, result.brightness, color=OKABE_ITO["orange"], linewidth=0.6)
    axes[1].axhline(1, color=OKABE_ITO["grey"], linewidth=0.8)
    axes[1].set_ylabel("Relative Brightness")
    axes[2].plot(minutes, yoff, color=OKABE_ITO["blue"], linewidth=0.6, label="y")
    axes[2].plot(minutes, xoff, color=OKABE_ITO["orange"], linewidth=0.6, label="x")
    axes[2].set(ylabel="Rigid Shift (px)", xlabel="Time (minutes)")
    for start, end in flagged_stretches(result.flagged, merge_gap_frames):
        for ax in axes:
            ax.axvspan(
                minutes[start],
                minutes[end - 1] + 1 / fs_hz / 60,
                color=OKABE_ITO["vermillion"],
                alpha=0.25,
                linewidth=0,
            )
    for ax in (axes[0], axes[2]):
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    flagged_pct = 100 * float(np.mean(result.flagged))
    axes[0].set_title(
        f"Frames Not Matching The Mean After Registration: {flagged_pct:.1f}% Flagged (Shaded)"
    )
    return axes
