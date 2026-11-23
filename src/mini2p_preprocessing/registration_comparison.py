"""Compare registrations of one recording side by side (a registration sweep), without cells.

Every registration in ``<aligned_folder>/<save_folder>/plane0`` is measured the same way, on
the same frames and in the same crop (inside every registration's unwrapped region), so the
numbers differ only because the registrations do. Nothing needs cell segmentation.

Alignment, on evenly spaced frames (both channels):

- frame-to-mean correlation: each frame against the mean of the sampled frames, both smoothed
  with the same Gaussian (higher, and a higher 5th percentile = frames line up better);
- residual shift: the subpixel shift that still best aligns each frame to that mean (lower);
- crispness: gradient energy of the mean image over its intensity (higher = less blur);
- Suite2p's own rigid shifts: their median frame-to-frame change (jitter), and bad frames.

Signal, on a few contiguous blocks of frames, from small patches of pixels (``patch_px``) and
single pixels, each high-passed over ``detrend_s`` to remove bleaching and drift:

- green peak SNR: how far a patch's largest fluctuations (its 99th percentile, after a
  3-frame running mean) rise above its median, in units of its frame-to-frame noise (the
  robust SD of its first differences / sqrt 2). Higher = clearer activity. Motion that jitters
  from frame to frame raises that noise as much as the peaks, so this is not a motion measure;
- red fluctuation: a patch's high-passed SD over its mean, in %. The red signal should not
  change, so beyond the noise floor (the same for every registration) this is motion left in
  it, from frame to frame or slower. Lower is better;
- noise level (Rupprecht et al., Eur J Neurosci 2021): the median absolute frame-to-frame
  change of a patch's ΔF/F over sqrt(frame rate), in %/sqrt(Hz), over the brighter half of the
  patches (F0: their 20th percentile). Lower = less noise per unit of baseline. F0 includes any
  detector offset, so it compares registrations, not recordings;
- local correlation image: each pixel's correlation with the mean of its 8 neighbours. Cells
  stand out as bright patches; its 99th percentile measures them, and its contrast (99th
  percentile minus median) discounts the correlation that resampling alone gives neighbouring
  pixels (nonrigid registration and subpixel shifts interpolate, which smooths noise);
- green-red correlation: the median correlation of a patch's green and red fluctuations.
  Lower = less of what moves the red (motion, brain movement in z) left in the green.

Red (structural) is the fairer test of motion, as its signal does not change with activity.
Read the green signal measures only among registrations the red measures find equally well
aligned: motion left in a movie raises a pixel's correlation with its neighbours and a patch's
largest fluctuations too, so on its own a gain in green peak SNR or in the local correlation's
99th percentile can be motion. The contrast and the noise level are not fooled that way.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter, uniform_filter1d
from skimage.registration import phase_cross_correlation

from .plots import QC_FONTS, SERIES_COLORS
from .qc import unwrapped_region
from .raw_previews import save_preview
from .registration import load_registered_outputs

logger = logging.getLogger(__name__)

CHANNELS = ("structural", "functional")  # Suite2p's data_chan2.bin and data.bin
SHORT = {"structural": "red", "functional": "green"}
SMOOTH_PX = 1.0  # the same spatial smoothing for every registration, against shot noise
UPSAMPLE = 10  # residual shifts to 0.1 px
N_FRAMES = 1000  # evenly spaced frames for the alignment measures
N_BLOCKS = 5  # contiguous blocks, evenly spread, for the signal measures
BLOCK_FRAMES = 600
MIN_BLOCK_FRAMES = 50
PATCH_PX = 4
DETREND_S = 10.0  # keeps calcium transients (seconds), removes bleaching and drift
SIGNAL_SMOOTH_FRAMES = 3
PEAK_PERCENTILE = 99
BASELINE_PERCENTILE = 20  # F0 of the noise level
MAD_TO_SD = 1.4826  # scales a median absolute deviation to a normal standard deviation
OUTPUT_FOLDER = "registration_comparison"
TABLE_NAME = "registration_comparison"
IMAGES_NAME = "registration_comparison_images"


# ---- analysis: alignment ----------------------------------------------------------


def common_crop(regions: Sequence[tuple[slice, slice]]) -> tuple[slice, slice]:
    """The rows and columns inside every registration's unwrapped region."""
    top = max(rows.start for rows, _ in regions)
    bottom = min(rows.stop for rows, _ in regions)
    left = max(columns.start for _, columns in regions)
    right = min(columns.stop for _, columns in regions)
    if bottom <= top or right <= left:
        raise ValueError(f"The registrations' unwrapped regions do not overlap: {regions}")
    return slice(top, bottom), slice(left, right)


def sample_frames(n_frames: int, n_sample: int) -> np.ndarray:
    """Evenly spaced frame indices over the recording."""
    if n_frames < 1 or n_sample < 1:
        raise ValueError(f"Need at least one frame to sample, got {n_frames=}, {n_sample=}")
    return np.unique(np.linspace(0, n_frames - 1, min(n_sample, n_frames)).astype(int))


def block_slices(n_frames: int, n_blocks: int, block_frames: int) -> list[slice]:
    """*n_blocks* contiguous, non-overlapping blocks evenly spread over the recording, each
    *block_frames* long or as long as fits."""
    if n_blocks < 1:
        raise ValueError(f"n_blocks must be at least 1, got {n_blocks}")
    length = min(block_frames, n_frames // n_blocks)
    if length < MIN_BLOCK_FRAMES:
        raise ValueError(
            f"{n_blocks} blocks of {n_frames} frames are {length} frames long; the signal "
            f"measures need at least {MIN_BLOCK_FRAMES}: use fewer blocks"
        )
    starts = np.linspace(0, n_frames - length, n_blocks).astype(int)
    return [slice(int(start), int(start) + length) for start in starts]


def crispness(image: np.ndarray) -> float:
    """Gradient energy of the image relative to its intensity (scale-free sharpness)."""
    gy, gx = np.gradient(image.astype(np.float64))
    return float(np.sqrt(np.sum(gy**2 + gx**2)) / np.linalg.norm(image))


def frame_alignment(frames: np.ndarray, mean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per frame: correlation with the mean, and the residual shift to it (px)."""
    reference = gaussian_filter(mean.astype(np.float32), SMOOTH_PX)
    reference_z = (reference - reference.mean()) / reference.std()
    correlations, shifts = [], []
    for frame in frames:
        smooth = gaussian_filter(frame.astype(np.float32), SMOOTH_PX)
        z = (smooth - smooth.mean()) / (smooth.std() + 1e-9)
        correlations.append(float(np.mean(z * reference_z)))
        # plain, not whitened, cross-correlation: whitening lets shot noise pick the peak
        shift, _, _ = phase_cross_correlation(
            reference, smooth, upsample_factor=UPSAMPLE, normalization=None
        )
        shifts.append(float(np.hypot(*shift)))
    return np.asarray(correlations), np.asarray(shifts)


def alignment_measures(frames: np.ndarray) -> tuple[dict, np.ndarray]:
    """Alignment measures of one channel's sampled frames, and their mean image."""
    mean = frames.mean(axis=0)
    correlation, residual = frame_alignment(frames, mean)
    measures = {
        "corr_median": float(np.median(correlation)),
        "corr_p5": float(np.percentile(correlation, 5)),
        "residual_shift_px": float(np.sqrt(np.mean(residual**2))),
        "residual_shift_median_px": float(np.median(residual)),
        "crispness": crispness(mean),
    }
    return measures, mean


def shift_jitter(ops: dict) -> float:
    """Median frame-to-frame change of Suite2p's rigid shift (px)."""
    dy = np.diff(np.asarray(ops["yoff"], float))
    dx = np.diff(np.asarray(ops["xoff"], float))
    return float(np.median(np.hypot(dy, dx))) if dy.size else float("nan")


def registration_settings(ops: dict) -> dict:
    """nonrigid, smooth_sigma and norm_frames as the registration used them (None if not
    recorded). Suite2p 1.x keeps them in ops['registration'], older versions at the top."""
    settings = ops.get("registration") if isinstance(ops.get("registration"), dict) else ops
    return {key: settings.get(key) for key in ("nonrigid", "smooth_sigma", "norm_frames")}


# ---- analysis: signal -------------------------------------------------------------


def detrend_window(fs_hz: float, detrend_s: float) -> int:
    """Odd window (frames) of the running mean that high-passes traces."""
    if fs_hz <= 0 or detrend_s <= 0:
        raise ValueError(f"fs_hz and detrend_s must be positive, got {fs_hz}, {detrend_s}")
    return max(3, int(round(detrend_s * fs_hz)) | 1)


def high_pass(traces: np.ndarray, window: int, axis: int = -1) -> np.ndarray:
    """Traces minus their running mean over *window* frames (float32)."""
    traces = np.asarray(traces, np.float32)
    return traces - uniform_filter1d(traces, window, axis=axis, mode="nearest")


def patch_traces(block: np.ndarray, patch_px: int) -> np.ndarray:
    """(patches, frames): the mean of each *patch_px* x *patch_px* patch of a (frames, rows,
    columns) block; edge rows and columns that do not fill a patch are left out."""
    n_frames, rows, columns = block.shape
    n_rows, n_columns = rows // patch_px, columns // patch_px
    if n_rows < 1 or n_columns < 1:
        raise ValueError(f"A {rows} x {columns} crop holds no {patch_px} px patch")
    trimmed = np.asarray(block[:, : n_rows * patch_px, : n_columns * patch_px], np.float32)
    patches = trimmed.reshape(n_frames, n_rows, patch_px, n_columns, patch_px).mean(axis=(2, 4))
    return patches.reshape(n_frames, -1).T


def robust_noise(traces: np.ndarray) -> np.ndarray:
    """Each trace's frame-to-frame noise SD: the robust SD of its first differences / sqrt 2
    (slow signal barely changes from one frame to the next)."""
    diffs = np.diff(traces, axis=-1)
    centred = diffs - np.median(diffs, axis=-1, keepdims=True)
    return MAD_TO_SD * np.median(np.abs(centred), axis=-1) / np.sqrt(2)


def peak_snr(high_passed: np.ndarray) -> np.ndarray:
    """Each trace's 99th percentile, after a 3-frame running mean, above its median, over its
    frame-to-frame noise."""
    smooth = uniform_filter1d(high_passed, SIGNAL_SMOOTH_FRAMES, axis=-1, mode="nearest")
    rise = np.percentile(smooth, PEAK_PERCENTILE, axis=-1) - np.median(smooth, axis=-1)
    noise = robust_noise(high_passed)
    return np.divide(rise, noise, out=np.full(rise.shape, np.nan), where=noise > 0)


def noise_level(traces: np.ndarray, fs_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """(noise level in %/sqrt(Hz), F0) of each raw trace: the median absolute frame-to-frame
    change of its ΔF/F over sqrt(frame rate). NaN where F0 is not positive."""
    f0 = np.percentile(traces, BASELINE_PERCENTILE, axis=-1)
    safe = np.where(f0 > 0, f0, np.nan)
    dff = (traces - safe[:, None]) / safe[:, None]
    level = 100 * np.median(np.abs(np.diff(dff, axis=-1)), axis=-1) / np.sqrt(fs_hz)
    return level, f0


def bright_half_median(values: np.ndarray, f0: np.ndarray) -> float:
    """Median of *values* over the traces whose F0 is positive and at least the median F0."""
    usable = np.isfinite(values) & (f0 > 0)
    if not usable.any():
        return float("nan")
    return float(np.median(values[usable & (f0 >= np.median(f0[usable]))]))


def local_correlation_image(block: np.ndarray, window: int) -> np.ndarray:
    """Each pixel's correlation with the mean of its 8 neighbours, over the high-passed block
    (frames, rows, columns)."""
    movie = high_pass(block, window, axis=0)
    movie -= movie.mean(axis=0)
    movie /= movie.std(axis=0) + 1e-9
    # mirror: an edge pixel's missing neighbours are its inner ones, never itself
    neighbours = (9 * uniform_filter(movie, size=(1, 3, 3), mode="mirror") - movie) / 8
    covariance = np.mean(movie * neighbours, axis=0)
    return covariance / (np.sqrt(np.mean(neighbours**2, axis=0)) + 1e-9)


def trace_correlations(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation of each row of *a* with the same row of *b*."""
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    denominator = np.sqrt(np.sum(a**2, axis=-1) * np.sum(b**2, axis=-1))
    numerator = np.sum(a * b, axis=-1)
    return np.divide(numerator, denominator, out=np.full(len(a), np.nan), where=denominator > 0)


def fluctuation(traces: np.ndarray, high_passed: np.ndarray) -> np.ndarray:
    """Each trace's high-passed SD over its mean, in % (NaN where the mean is not positive)."""
    mean = traces.mean(axis=-1)
    sd = high_passed.std(axis=-1)
    return np.divide(100 * sd, mean, out=np.full(mean.shape, np.nan), where=mean > 0)


def block_signal(green: np.ndarray, red: np.ndarray, fs_hz: float, window: int, patch_px: int):
    """Per patch (green peak SNR, red fluctuation, noise level, F0, green-red correlation)
    and the green local correlation image, of one block of each channel."""
    traces = {name: patch_traces(block, patch_px) for name, block in (("g", green), ("r", red))}
    passed = {name: high_pass(t, window) for name, t in traces.items()}
    level, f0 = noise_level(traces["g"], fs_hz)
    per_patch = np.stack([
        peak_snr(passed["g"]), fluctuation(traces["r"], passed["r"]), level, f0,
        trace_correlations(passed["g"], passed["r"]),
    ])  # fmt: skip
    return per_patch, local_correlation_image(green, window)


def signal_measures(
    blocks: Sequence[tuple[np.ndarray, np.ndarray]],
    fs_hz: float,
    detrend_s: float = DETREND_S,
    patch_px: int = PATCH_PX,
) -> tuple[dict, np.ndarray]:
    """Signal measures over (green, red) blocks, each measure averaged per patch (or pixel)
    over the blocks, and the green local correlation image."""
    window = detrend_window(fs_hz, detrend_s)
    results = [block_signal(g, r, fs_hz, window, patch_px) for g, r in blocks]
    snr_green, red_fluctuation, level, f0, green_red = np.nanmean([p for p, _ in results], axis=0)
    correlation_image = np.mean([image for _, image in results], axis=0)
    p99 = float(np.percentile(correlation_image, 99))
    measures = {
        "green_peak_snr_median": float(np.nanmedian(snr_green)),
        "green_peak_snr_p95": float(np.nanpercentile(snr_green, 95)),
        "red_fluctuation_pct": float(np.nanmedian(red_fluctuation)),
        "green_noise_level": bright_half_median(level, f0),
        "green_local_corr_p99": p99,
        "green_local_corr_contrast": p99 - float(np.median(correlation_image)),
        "green_red_corr_median": float(np.nanmedian(green_red)),
    }
    return measures, correlation_image


# ---- I/O --------------------------------------------------------------------------


@dataclass
class Plan:
    """What every registration is measured on."""

    crop: tuple[slice, slice]
    frames: np.ndarray
    blocks: list[slice]
    fs_hz: float


def find_registrations(aligned_folder: Path) -> list[Path]:
    """Every <save_folder>/plane0 with an ops.npy in *aligned_folder*."""
    planes = sorted(p for p in Path(aligned_folder).glob("*/plane0") if (p / "ops.npy").is_file())
    if not planes:
        raise FileNotFoundError(f"No <save_folder>/plane0/ops.npy in {aligned_folder}")
    return planes


def load_ops(plane: Path) -> dict:
    return np.load(Path(plane) / "ops.npy", allow_pickle=True).item()


def only_value(opses: Sequence[dict], key: str, planes: Sequence[Path]):
    """The value of *key* that every registration shares; raises if they differ."""
    if any(key not in ops for ops in opses):
        missing = [p.parent.name for p, o in zip(planes, opses, strict=True) if key not in o]
        raise KeyError(f"ops has no '{key}' in {missing}")
    values = {ops[key] if np.isscalar(ops[key]) else str(ops[key]) for ops in opses}
    if len(values) != 1:
        raise ValueError(f"The registrations differ in {key}: {sorted(map(str, values))}; "
                         "only registrations of the same recording can be compared")  # fmt: skip
    return values.pop()


def plan_comparison(
    planes: Sequence[Path], n_frames: int, n_blocks: int, block_frames: int
) -> Plan:
    """The shared crop, sampled frames and blocks of these registrations."""
    opses = [load_ops(plane) for plane in planes]
    total = int(only_value(opses, "nframes", planes))
    for key in ("Ly", "Lx"):
        only_value(opses, key, planes)
    return Plan(
        crop=common_crop([unwrapped_region(ops) for ops in opses]),
        frames=sample_frames(total, n_frames),
        blocks=block_slices(total, n_blocks, block_frames),
        fs_hz=float(only_value(opses, "fs", planes)),
    )


def measure_registration(
    plane: Path, plan: Plan, detrend_s: float = DETREND_S, patch_px: int = PATCH_PX
) -> tuple[dict, dict]:
    """One registration's row of measures, and its images (red mean, green local
    correlation)."""
    outputs = load_registered_outputs(plane)
    rows, columns = plan.crop
    row = {"registration": Path(plane).parent.name, **registration_settings(outputs.ops)}
    row["bad_frames"] = int(np.sum(outputs.ops.get("badframes", [])))
    row["rigid_shift_jitter_px"] = shift_jitter(outputs.ops)
    means = {}
    for name in CHANNELS:
        sampled = np.asarray(getattr(outputs, name)[plan.frames])[:, rows, columns]
        measures, means[name] = alignment_measures(sampled)
        row |= {f"{SHORT[name]}_{key}": value for key, value in measures.items()}
    blocks = [
        tuple(np.asarray(getattr(outputs, name)[block])[:, rows, columns]
              for name in ("functional", "structural"))
        for block in plan.blocks
    ]  # fmt: skip
    signal, correlation_image = signal_measures(blocks, plan.fs_hz, detrend_s, patch_px)
    row |= signal
    return row, {"red_mean": means["structural"], "green_local_corr": correlation_image}


def write_table(rows: Sequence[dict], path: Path) -> None:
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def output_stems(aligned_folder: Path) -> tuple[Path, Path]:
    folder = Path(aligned_folder) / OUTPUT_FOLDER
    return folder / TABLE_NAME, folder / IMAGES_NAME


def compare_registrations(
    aligned_folder: Path,
    n_frames: int = N_FRAMES,
    n_blocks: int = N_BLOCKS,
    block_frames: int = BLOCK_FRAMES,
    patch_px: int = PATCH_PX,
    detrend_s: float = DETREND_S,
    overwrite: bool = False,
) -> list[dict]:
    """Measure every registration in *aligned_folder* and save the table, the measures
    figure and the image figure in ``<aligned_folder>/registration_comparison/``. Reads the
    registrations only. Returns the rows."""
    table_stem, images_stem = output_stems(aligned_folder)
    if table_stem.with_suffix(".csv").exists() and not overwrite:
        raise FileExistsError(f"{table_stem}.csv exists; pass overwrite to replace it")
    planes = find_registrations(aligned_folder)
    plan = plan_comparison(planes, n_frames, n_blocks, block_frames)
    rows, images = [], []
    for plane in planes:
        logger.info("Measuring %s", plane.parent.name)
        row, image = measure_registration(plane, plan, detrend_s, patch_px)
        rows.append(row)
        images.append(image)
    table_stem.parent.mkdir(parents=True, exist_ok=True)
    write_table(rows, table_stem.with_suffix(".csv"))
    labels = [short_label(row["registration"]) for row in rows]
    details = caption_details(aligned_folder, plan, patch_px, detrend_s)
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": QC_FONTS}):
        figure = plot_measures(rows).flat[0].figure
        save_preview(figure, table_stem, MEASURES_CAPTION.format(**details))
        plt.close(figure)
        figure = plot_images(labels, images).flat[0].figure
        save_preview(figure, images_stem, IMAGES_CAPTION.format(**details))
        plt.close(figure)
    return rows


# ---- plotting ---------------------------------------------------------------------


# (column, title, which way is better); one row of panels per line
PANELS = [
    [("red_corr_median", "Red Frame-To-Mean r", "higher"),
     ("red_corr_p5", "Red r, 5th Percentile", "higher"),
     ("red_residual_shift_px", "Red Residual Shift (px)", "lower"),
     ("red_crispness", "Red Mean Crispness", "higher"),
     ("red_fluctuation_pct", "Red Fluctuation (%)", "lower")],
    [("green_corr_median", "Green Frame-To-Mean r", "higher"),
     ("green_crispness", "Green Mean Crispness", "higher"),
     ("green_red_corr_median", "Green-Red Correlation", "lower"),
     ("rigid_shift_jitter_px", "Rigid Shift Jitter (px)", "lower"),
     ("bad_frames", "Bad Frames", "lower")],
    [("green_peak_snr_median", "Green Peak SNR, Median", "higher"),
     ("green_peak_snr_p95", "Green Peak SNR, 95th Percentile", "higher"),
     ("green_residual_shift_px", "Green Residual Shift (px)", "lower"),
     ("green_local_corr_contrast", "Green Local Correlation Contrast", "higher"),
     ("green_noise_level", "Green Noise Level (%/√Hz)", "lower")],
]  # fmt: skip


def short_label(registration: str) -> str:
    return registration.removeprefix("suite2p_") or registration


def zoomed_limits(values: np.ndarray) -> tuple[float, float] | None:
    """Y limits around the finite values, never below 0 for values that are all >= 0."""
    finite = values[np.isfinite(values)]
    if not finite.size:
        return None
    spread = max(float(np.ptp(finite)), 1e-3 * max(abs(float(finite.max())), 1.0))
    low = float(finite.min()) - 0.4 * spread
    return (max(low, 0.0) if finite.min() >= 0 else low), float(finite.max()) + 0.6 * spread


def plot_measure(ax, labels: Sequence[str], values: np.ndarray, title: str, better: str):
    """One measure: a dot per registration, labelled with its value."""
    x = np.arange(len(labels))
    colors = [SERIES_COLORS[i % len(SERIES_COLORS)] for i in x]
    ax.scatter(x, values, c=colors, s=50, zorder=3)
    for xi, value in zip(x, values, strict=True):
        if np.isfinite(value):
            ax.annotate(f"{value:.3g}", (xi, value), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7)  # fmt: skip
    ax.set_title(f"{title}\n({better} is better)", fontsize=9)
    ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=7)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    limits = zoomed_limits(values)
    if limits is not None:
        ax.set_ylim(*limits)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def plot_measures(rows: Sequence[dict], axes=None) -> np.ndarray:
    """Every measure, one panel each (``PANELS``), registrations along x."""
    n_rows, n_columns = len(PANELS), len(PANELS[0])
    if axes is None:
        _, axes = plt.subplots(n_rows, n_columns, figsize=(3.6 * n_columns, 3.6 * n_rows),
                               constrained_layout=True, squeeze=False)  # fmt: skip
    axes = np.asarray(axes).reshape(n_rows, n_columns)
    labels = [short_label(row["registration"]) for row in rows]
    for panel_row, ax_row in zip(PANELS, axes, strict=True):
        for (key, title, better), ax in zip(panel_row, ax_row, strict=True):
            values = np.asarray([np.nan if r[key] is None else r[key] for r in rows], float)
            plot_measure(ax, labels, values, title, better)
    axes.flat[0].figure.suptitle("Registration Comparison")
    return axes


def plot_images(labels: Sequence[str], images: Sequence[dict], axes=None) -> np.ndarray:
    """Per registration (columns): the red mean image, and the green local correlation image
    (one color scale for all)."""
    if axes is None:
        _, axes = plt.subplots(2, len(labels), figsize=(3.2 * len(labels), 6.6),
                               constrained_layout=True, squeeze=False)  # fmt: skip
    axes = np.asarray(axes).reshape(2, len(labels))
    maps = np.stack([image["green_local_corr"] for image in images])
    vmin, vmax = np.percentile(maps, [1, 99.5])
    for column, (label, image) in enumerate(zip(labels, images, strict=True)):
        red = image["red_mean"]
        axes[0, column].imshow(red, cmap="gray", vmin=np.percentile(red, 1),
                               vmax=np.percentile(red, 99.5))  # fmt: skip
        shown = axes[1, column].imshow(image["green_local_corr"], cmap="magma",
                                       vmin=vmin, vmax=vmax)  # fmt: skip
        axes[0, column].set_title(label, fontsize=9)
        for ax in axes[:, column]:
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].set_ylabel("Red Mean")
    axes[1, 0].set_ylabel("Green Local Correlation")
    axes.flat[0].figure.colorbar(shown, ax=axes[1, :].tolist(), shrink=0.8, label="r")
    axes.flat[0].figure.suptitle("Registration Comparison Images")
    return axes


def caption_details(aligned_folder: Path, plan: Plan, patch_px: int, detrend_s: float) -> dict:
    rows, columns = plan.crop
    lengths = {block.stop - block.start for block in plan.blocks}
    return {
        "recording": "/".join(Path(aligned_folder).resolve().parts[-7:]),
        "n": len(plan.frames), "rows": f"{rows.start}-{rows.stop - 1}",
        "columns": f"{columns.start}-{columns.stop - 1}", "sigma": SMOOTH_PX,
        "n_blocks": len(plan.blocks), "block_frames": lengths.pop(), "patch": patch_px,
        "detrend": detrend_s, "fs": plan.fs_hz,
    }  # fmt: skip


MEASURES_CAPTION = """Registrations of {recording}, each measured the same way in rows \
{rows} and columns {columns} (inside every registration's unwrapped region); no cell \
segmentation.
Alignment, on the same {n} evenly spaced frames: frame-to-mean r, each frame \
against the mean of those frames, both smoothed with a {sigma} px Gaussian (median and 5th \
percentile across frames); residual shift, root-mean-square subpixel shift that still best \
aligns each frame to that mean; crispness, gradient energy of the mean over its intensity; \
rigid shift jitter, median frame-to-frame change of Suite2p's rigid shift; bad frames, \
Suite2p's count.
Signal, on {n_blocks} contiguous blocks of {block_frames} frames ({fs:g} Hz) spread over the \
recording, from {patch} x {patch} px patches high-passed over {detrend:g} s: peak SNR, a \
patch's 99th percentile (3-frame running mean) above its median over its frame-to-frame \
noise (robust SD of first differences / sqrt 2), median or 95th percentile across patches; \
red fluctuation, median across patches of the high-passed SD over the mean, in % (the red \
signal is activity-independent, so beyond the shared noise floor this is motion left in it); \
green-red correlation, median correlation of each patch's green and red fluctuations (shared \
motion); local correlation, each pixel's correlation with the mean of its 8 neighbours \
(99th percentile, and contrast = 99th percentile minus median, which discounts the \
correlation that interpolation alone gives neighbouring pixels); noise level (Rupprecht et \
al. 2021), median absolute frame-to-frame change of ΔF/F (F0: 20th percentile) over \
sqrt(frame rate), median over the brighter half of the patches; F0 includes any detector \
offset, so compare registrations, not recordings.
Red is the structural channel (activity-independent, the fairer test of motion); green the \
functional one. Motion also raises green peak SNR and local correlation, so compare green \
signal measures only among registrations the red measures find equally well aligned. Y-axes \
are zoomed on each measure's range (dots, not bars). Table: registration_comparison.csv \
(also the local correlation's 99th percentile, and median residual shifts).
"""

IMAGES_CAPTION = """Registrations of {recording}, rows {rows} and columns {columns}. Top: red \
mean of the {n} evenly spaced frames (1st-99.5th percentile of each). Bottom: green local \
correlation image, each pixel's correlation with the mean of its 8 neighbours over \
{n_blocks} blocks of {block_frames} frames high-passed over {detrend:g} s, on one color scale \
for all registrations; sharper, brighter cell bodies mean better-aligned activity.
"""
