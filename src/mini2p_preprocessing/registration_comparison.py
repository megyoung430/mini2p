"""Compare registrations of one recording side by side (a registration sweep), without cells.

Every registration in ``<aligned_folder>/<save_folder>/plane0`` is measured the same way, on
the same frames and, for the alignment and signal measures, in the same crop (inside every
registration's unwrapped region). The crop depends on which registrations are in the folder,
so numbers compare within one comparison, not between comparisons of different folders.

Alignment, on evenly spaced frames (both channels):

- frame-to-mean correlation: each frame against the mean of the sampled frames, both smoothed
  with the same Gaussian (higher, and a higher 5th percentile = frames line up better);
- residual shift: the subpixel shift that still best aligns each frame to that mean (lower);
- crispness: gradient energy of the mean image over its intensity (higher = less blur);
- Suite2p's regDX (combined rigid + nonrigid shift between the top and bottom of its
  principal components, when computed) and bad frames.

Signal, on a few contiguous blocks of frames, from small patches of pixels (``patch_px``) and
single pixels, each high-passed over ``detrend_s`` to remove bleaching and drift:

- green peak SNR: how far a patch's largest fluctuations (its 99th percentile, after a
  3-frame running mean) rise above its median, in units of its frame-to-frame noise (the
  robust SD of its first differences / sqrt 2);
- red fluctuation: a patch's high-passed SD over its mean, in %. The red signal should not
  change, so beyond its noise this is motion left in it, from frame to frame or slower;
- noise level (Rupprecht et al., Eur J Neurosci 2021): the median absolute frame-to-frame
  change of a patch's ΔF/F over sqrt(frame rate), in %/sqrt(Hz), over the brighter half of the
  patches (F0: their 20th percentile). F0 includes any detector offset, so it compares
  registrations, not recordings;
- local correlation image: each pixel's correlation with the mean of its 8 neighbours. Cells
  stand out as bright patches; reported as its 99th percentile and its contrast (99th
  percentile minus median). Under nonrigid registration it follows the layout of Suite2p's
  blocks: each pixel is interpolated by its own fraction of a pixel, which correlates it with
  its neighbours by an amount that varies across the block grid, and the grid changes with the
  settings, so it does not compare even nonrigid registrations of different settings;
- green-red correlation: the median correlation of a patch's green and red fluctuations.
  Lower = less of what moves the red (motion, brain movement in z) left in the green.

Putative cells (``cell_snr``, the measure of the earlier 71R / AB08 / M comparison): the 200
strongest cell-scale peaks of each registration's green mean image, each a disc-minus-annulus
trace in 0.5 s bins over the first ``cell_minutes``; green resting SNR (contrast over noise),
large transients per minute (and on flipped traces, the noise count), cells with at least
one; and the same count in red at the same places, which should be at the noise count (above
it = motion or stray light).

What these can and cannot tell apart:

- Red (structural) is the fairer test of motion, as its signal does not change with activity.
  Motion left in a movie also raises green peak SNR, local correlation and transient counts,
  so compare green measures only among registrations the red measures find equally well
  aligned.
- Registrations that resample frames differently cannot be compared cleanly on the measures
  that depend on pixel noise. Nonrigid registration interpolates every frame by its own
  subpixel block shifts; rigid registration moves frames by whole pixels. Interpolation
  smooths noise (lowering noise level and fluctuation, raising local correlation and resting
  SNR) and blurs each frame by a different amount (raising fluctuation, lowering crispness),
  with no change in alignment. Comparing settings within one mode (smooth_sigma, norm_frames,
  rigid or nonrigid) is clean; between modes, rely on residual shift, the red transient count
  and the transient counts against their flipped noise counts.
- Suite2p computes regDX with each registration's own smooth_sigma, so it does not compare
  registrations of different smooth_sigma cleanly either.
"""

from __future__ import annotations

import csv
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter, uniform_filter1d
from skimage.registration import phase_cross_correlation

from . import cell_snr
from .alignment import load_reference_image
from .config import FluorescenceSettings, RegistrationSettings
from .extraction import align_masks
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
CELL_MINUTES = 4.3  # as in the 71R / AB08 comparison
CHUNK_FRAMES = 600  # frames read at a time for the cell traces
SEGMENTATION_FOLDER = "cell_segmentation"  # looked for in the aligned folder and above it
NOT_CELLS = ("vessel", "background")  # _seg.npy files of these names segment something else
RUN_FOLDER = re.compile(r"run-\d+")  # the segmentation folder is looked for up to here
IMAGE_SUFFIXES = (".png", ".tif", ".tiff")
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


REG_DX_COMBINED = 3  # regDX columns: rigid, mean nonrigid, max nonrigid, combined


def reg_dx(ops: dict) -> float:
    """Suite2p's regDX: the combined rigid + nonrigid shift (px) between the top and bottom of
    each principal component of the registered movie, averaged over the components; NaN when
    Suite2p did not compute it (do_regmetrics off, or fewer than 1500 frames)."""
    value = ops.get("regDX")
    if value is None or not np.size(value):
        return float("nan")
    value = np.atleast_2d(np.asarray(value, float))
    if value.shape[1] <= REG_DX_COMBINED:  # Suite2p 0.x: no combined column
        logger.warning("ops['regDX'] has shape %s, without Suite2p 1.x's combined column; "
                       "regDX is left blank", value.shape)  # fmt: skip
        return float("nan")
    return float(np.mean(value[:, REG_DX_COMBINED]))


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
class CellMasks:
    """Cellpose masks and the image they were drawn on."""

    labels: np.ndarray
    image: np.ndarray
    seg_file: Path
    image_file: Path


@dataclass
class SegmentedCells:
    """The masks on each registration, and the cells every registration keeps."""

    masks: CellMasks
    on_plane: dict[str, tuple[np.ndarray, int, int]]  # plane folder -> (labels, dy, dx)
    ids: list[int]  # labels inside every registration's crop, with a ring, in all of them
    left_out: int  # labels left out (lost in at least one registration)
    dropped: dict[str, int]  # left-out labels with each reason in at least one registration


@dataclass
class Plan:
    """What every registration is measured on."""

    crop: tuple[slice, slice]
    frames: np.ndarray
    blocks: list[slice]
    fs_hz: float
    cell_frames: int  # the first frames, a whole number of bins, for the cell measures
    scan_px: int = cell_snr.SCAN_PX
    n_cells: int = cell_snr.N_CELLS
    segmented: SegmentedCells | None = None  # None: putative cells


def run_folder_of(folder: Path) -> Path | None:
    """The nearest ``run-XX`` folder at or above *folder*, or None."""
    return next((p for p in [folder, *folder.parents] if RUN_FOLDER.fullmatch(p.name)), None)


def find_segmentation(aligned_folder: Path) -> Path | None:
    """The ``*_seg.npy`` in the nearest ``cell_segmentation/`` folder at or above
    *aligned_folder*, looking no higher than its ``run-XX`` folder (only the folder itself
    outside a run), or None when there is no such folder or it holds none. Files whose names
    contain "vessel" or "background" segment something else and are ignored.

    Raises:
        ValueError: If that folder holds several.
    """
    folder = Path(aligned_folder).absolute()
    run = run_folder_of(folder)
    parents = [folder, *folder.parents]
    for parent in parents[: parents.index(run) + 1] if run else [folder]:
        segmentation = parent / SEGMENTATION_FOLDER
        if not segmentation.is_dir():
            continue
        found = sorted(
            f for f in segmentation.glob("*_seg.npy") if not any(w in f.name for w in NOT_CELLS)
        )
        if len(found) > 1:
            raise ValueError(f"Several _seg.npy files in {segmentation}; choose one with --seg")
        if not found:
            logger.warning("%s holds no _seg.npy; using putative cells", segmentation)
        return found[0] if found else None
    return None


def mask_image_for(seg_file: Path, seg: dict) -> Path:
    """The image the masks of *seg* (the loaded ``_seg.npy``) were drawn on: beside it with the
    same name (PNG or TIFF), or the file it names, in its folder.

    Raises:
        FileNotFoundError: If neither is there.
    """
    seg_file = Path(seg_file)
    stem = seg_file.name.removesuffix("_seg.npy")
    names = [stem + suffix for suffix in IMAGE_SUFFIXES]
    if seg.get("filename"):
        names.append(Path(str(seg["filename"])).name)
    for name in names:
        if (seg_file.parent / name).is_file():
            return seg_file.parent / name
    raise FileNotFoundError(
        f"No image the masks of {seg_file.name} were drawn on beside it (looked for {names}); "
        "pass --mask-image"
    )


def load_masks(seg_file: Path, image_file: Path | None = None) -> CellMasks:
    """The masks of a Cellpose ``_seg.npy`` and the image they were drawn on."""
    seg = np.load(seg_file, allow_pickle=True).item()
    if "masks" not in seg:
        raise KeyError(f"{seg_file} has no 'masks' entry; is it a Cellpose _seg.npy file?")
    image_file = Path(image_file) if image_file else mask_image_for(seg_file, seg)
    labels = np.asarray(seg["masks"])
    image = load_reference_image(image_file)
    if labels.shape != image.shape:
        raise ValueError(f"The masks are {labels.shape} but {image_file.name} is {image.shape}")
    return CellMasks(labels, image, Path(seg_file), image_file)


def ring_geometry(scan_px: int) -> tuple[float, float]:
    """(gap, width) of the ring around a mask: as far from its edge, and as wide, as the
    putative cells' annulus is from their disc."""
    geometry = cell_snr.scaled_geometry(scan_px)
    return geometry["ring_in"] - geometry["disc_r"], geometry["ring_out"] - geometry["ring_in"]


def segment_planes(
    planes: Sequence[Path],
    masks: CellMasks,
    crop: tuple[slice, slice],
    scan_px: int,
    fl: FluorescenceSettings,
    diameter_px: int,
) -> SegmentedCells:
    """The masks shifted onto every registration's own mean image (each aligns to its own
    reference), and the cells every registration keeps: aligned, inside the crop with their
    ring, and with ring pixels. Every registration is then measured on the same cells.

    Raises:
        ValueError: If a registration cannot be aligned by one shift, or no cell is kept.
    """
    gap, width = ring_geometry(scan_px)
    all_ids = {int(v) for v in np.unique(masks.labels[masks.labels > 0])}
    on_plane, kept, reasons = {}, set(all_ids), {}
    for plane in planes:
        labels, report = align_masks(masks.labels, masks.image, load_ops(plane), fl, diameter_px)
        usable, dropped = cell_snr.usable_masks(labels, gap, width, crop)
        dropped["dropped by alignment"] = list(report["dropped_labels"])
        on_plane[str(plane)] = (labels, int(report["dy"]), int(report["dx"]))
        kept &= set(usable)
        for reason, ids in dropped.items():
            reasons.setdefault(reason, set()).update(ids)
    counts = {reason: len(ids - kept) for reason, ids in reasons.items() if ids - kept}
    if not kept:
        raise ValueError(
            f"No mask of {masks.seg_file.name} is usable in every registration: {counts}"
        )
    left_out = len(all_ids - kept)
    if left_out:
        logger.warning("%d of %d masks left out; reasons, each in at least one registration "
                       "(a mask can have several): %s", left_out, len(all_ids), counts)  # fmt: skip
    return SegmentedCells(masks, on_plane, sorted(kept), left_out, counts)


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


def cell_frame_count(total: int, fs_hz: float, minutes: float | None) -> int:
    """The first frames (a whole number of bins) of *minutes*, or of the whole recording."""
    wanted = total if minutes is None else int(minutes * 60 * fs_hz)
    if wanted > total:
        raise ValueError(f"{minutes} min is {wanted} frames; the recording has {total}")
    fpb = cell_snr.frames_per_bin(fs_hz)
    if wanted // fpb < 3:
        raise ValueError(f"{minutes} min is fewer than 3 bins of {fpb} frames")
    return wanted // fpb * fpb


def plan_comparison(
    planes: Sequence[Path],
    n_frames: int,
    n_blocks: int,
    block_frames: int,
    cell_minutes: float | None = CELL_MINUTES,
    scan_px: int = cell_snr.SCAN_PX,
    n_cells: int = cell_snr.N_CELLS,
) -> Plan:
    """The shared crop, sampled frames and blocks of these registrations."""
    opses = [load_ops(plane) for plane in planes]
    total = int(only_value(opses, "nframes", planes))
    for key in ("Ly", "Lx"):
        only_value(opses, key, planes)
    fs_hz = float(only_value(opses, "fs", planes))
    return Plan(
        crop=common_crop([unwrapped_region(ops) for ops in opses]),
        frames=sample_frames(total, n_frames),
        blocks=block_slices(total, n_blocks, block_frames),
        fs_hz=fs_hz,
        cell_frames=cell_frame_count(total, fs_hz, cell_minutes),
        scan_px=scan_px,
        n_cells=n_cells,
    )


def movie_chunks(movie, n_frames: int, chunk_frames: int):
    """The first *n_frames* of a (frames, rows, columns) movie in chunks of *chunk_frames*
    (the last may be shorter)."""
    for start in range(0, n_frames, chunk_frames):
        yield np.asarray(movie[start : min(start + chunk_frames, n_frames)])


def putative_cells(ops: dict, plan: Plan) -> np.ndarray:
    """The putative cells, picked as in ``cell_snr``'s source on the registration's own green
    mean image within its own valid region (Suite2p's yrange and xrange)."""
    for key in ("meanImg", "yrange", "xrange"):
        if key not in ops:
            raise KeyError(f"ops has no '{key}', which the putative cells are picked with")
    valid = (tuple(ops["yrange"]), tuple(ops["xrange"]))
    geometry = cell_snr.scaled_geometry(plan.scan_px)
    mean_image = np.asarray(ops["meanImg"], np.float64)
    return cell_snr.pick_cells(mean_image, valid, geometry, plan.n_cells)


def cell_weights(ops: dict, plane: Path, plan: Plan) -> tuple[list, dict]:
    """The (cell, ring) averaging weights of this registration's cells, and what they are:
    the segmented masks shifted onto it, or else the putative cells."""
    if plan.segmented is not None:
        cells = plan.segmented
        labels, dy, dx = cells.on_plane[str(plane)]
        weights = list(cell_snr.mask_weights(labels, cells.ids, *ring_geometry(plan.scan_px)))
        about = {"cells": f"cellpose {cells.masks.seg_file.name}", "n_cells": len(cells.ids),
                 "mask_dy": dy, "mask_dx": dx}  # fmt: skip
        return weights, about
    geometry = cell_snr.scaled_geometry(plan.scan_px)
    centres = putative_cells(ops, plan)
    shape = np.shape(ops["meanImg"])
    disc = cell_snr.roi_weights(centres, shape, 0, geometry["disc_r"])
    ring = cell_snr.roi_weights(centres, shape, geometry["ring_in"], geometry["ring_out"])
    return [disc, ring], {"cells": "putative", "n_cells": len(centres), "mask_dy": None,
                          "mask_dx": None}  # fmt: skip


def cell_snr_measures(outputs, plane: Path, plan: Plan) -> dict:
    """Resting SNR and transients of the cells (segmented, or else putative) in green, and
    transients at the same places in red."""
    weights, about = cell_weights(outputs.ops, plane, plan)
    fpb = cell_snr.frames_per_bin(plan.fs_hz)
    chunk = max(fpb, CHUNK_FRAMES // fpb * fpb)
    minutes = plan.cell_frames / plan.fs_hz / 60
    row = dict(about)
    for name in ("functional", "structural"):
        chunks = movie_chunks(getattr(outputs, name), plan.cell_frames, chunk)
        disc, ring = cell_snr.binned_traces(chunks, weights, fpb)
        measures = cell_snr.cell_measures(disc, ring, fpb / plan.fs_hz)
        row |= {f"{SHORT[name]}_cell_{key}": value
                for key, value in cell_snr.summary(measures, minutes).items()}  # fmt: skip
    return row


def measure_registration(
    plane: Path, plan: Plan, detrend_s: float = DETREND_S, patch_px: int = PATCH_PX
) -> tuple[dict, dict]:
    """One registration's row of measures, and its red and green mean images."""
    outputs = load_registered_outputs(plane)
    rows, columns = plan.crop
    row = {"registration": Path(plane).parent.name, **registration_settings(outputs.ops)}
    row["bad_frames"] = int(np.sum(outputs.ops["badframes"]))
    row["suite2p_regDX_px"] = reg_dx(outputs.ops)
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
    signal, _ = signal_measures(blocks, plan.fs_hz, detrend_s, patch_px)
    row |= signal | cell_snr_measures(outputs, plane, plan)
    return row, {"red_mean": means["structural"], "green_mean": means["functional"]}


def write_table(rows: Sequence[dict], path: Path) -> None:
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def output_stems(aligned_folder: Path) -> tuple[Path, Path]:
    folder = Path(aligned_folder) / OUTPUT_FOLDER
    return folder / TABLE_NAME, folder / IMAGES_NAME


def prepare_comparison(
    aligned_folder: Path,
    n_frames: int = N_FRAMES,
    n_blocks: int = N_BLOCKS,
    block_frames: int = BLOCK_FRAMES,
    overwrite: bool = False,
    cell_minutes: float | None = CELL_MINUTES,
    scan_px: int = cell_snr.SCAN_PX,
    n_cells: int = cell_snr.N_CELLS,
    seg_file: Path | None = None,
    mask_image: Path | None = None,
    putative: bool = False,
    fluorescence: FluorescenceSettings | None = None,
    diameter_px: int | None = None,
) -> tuple[list[Path], Plan]:
    """The registrations in *aligned_folder* and what they are measured on.

    Cells are the Cellpose masks of *seg_file* (drawn on *mask_image*, by default the image of
    the same name beside it), or of the ``_seg.npy`` in the nearest ``cell_segmentation/``
    folder at or above *aligned_folder* within its run (``find_segmentation``); with neither,
    or with *putative*, putative cells. Masks are aligned with the ``[fluorescence]`` mask
    limits of *fluorescence* and the cell diameter *diameter_px* (the defaults if None).

    Raises:
        FileExistsError: If a comparison is there already and *overwrite* is not set.
        FileNotFoundError: If the folder holds no registration, or the masks' image is missing.
        KeyError, ValueError: If the registrations are not of one recording, or the request
            does not fit it (e.g. too few putative cells, or masks that do not align or that
            no registration keeps).
    """
    table_stem, _ = output_stems(aligned_folder)
    if table_stem.with_suffix(".csv").exists() and not overwrite:
        raise FileExistsError(f"{table_stem}.csv exists; pass overwrite to replace it")
    planes = find_registrations(aligned_folder)
    if putative and (seg_file or mask_image):
        raise ValueError("Choose putative cells or a segmentation, not both")
    seg_file = None if putative else seg_file or find_segmentation(aligned_folder)
    if mask_image and not seg_file:
        raise ValueError("A mask image needs its segmentation (--seg)")
    plan = plan_comparison(planes, n_frames, n_blocks, block_frames, cell_minutes, scan_px, n_cells)
    if seg_file:
        masks = load_masks(seg_file, mask_image)
        logger.info("Cells: %s, drawn on %s", masks.seg_file, masks.image_file.name)
        plan.segmented = segment_planes(
            planes, masks, plan.crop, scan_px, fluorescence or FluorescenceSettings(),
            diameter_px or RegistrationSettings().diameter_px,
        )  # fmt: skip
    for plane in planes:  # seconds now, rather than failing after measuring the others
        cell_weights(load_ops(plane), plane, plan)
    return planes, plan


def run_comparison(
    aligned_folder: Path,
    planes: Sequence[Path],
    plan: Plan,
    patch_px: int = PATCH_PX,
    detrend_s: float = DETREND_S,
) -> list[dict]:
    """Measure the registrations and save the table, the measures figure and the image
    figure in ``<aligned_folder>/registration_comparison/``. Returns the rows."""
    table_stem, images_stem = output_stems(aligned_folder)
    rows, images = [], []
    for plane in planes:
        logger.info("Measuring %s", plane.parent.name)
        row, image = measure_registration(plane, plan, detrend_s, patch_px)
        rows.append(row)
        images.append(image)
    table_stem.parent.mkdir(parents=True, exist_ok=True)
    write_table(rows, table_stem.with_suffix(".csv"))
    labels = [short_label(row) for row in rows]
    details = caption_details(aligned_folder, plan, patch_px, detrend_s)
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": QC_FONTS}):
        figure = plot_measures(rows).flat[0].figure
        save_preview(figure, table_stem, MEASURES_CAPTION.format(**details))
        plt.close(figure)
        figure = plot_images(labels, images).flat[0].figure
        save_preview(figure, images_stem, IMAGES_CAPTION.format(**details))
        plt.close(figure)
    return rows


def compare_registrations(
    aligned_folder: Path,
    n_frames: int = N_FRAMES,
    n_blocks: int = N_BLOCKS,
    block_frames: int = BLOCK_FRAMES,
    patch_px: int = PATCH_PX,
    detrend_s: float = DETREND_S,
    overwrite: bool = False,
    cell_minutes: float | None = CELL_MINUTES,
    scan_px: int = cell_snr.SCAN_PX,
    n_cells: int = cell_snr.N_CELLS,
    seg_file: Path | None = None,
    mask_image: Path | None = None,
    putative: bool = False,
    fluorescence: FluorescenceSettings | None = None,
    diameter_px: int | None = None,
) -> list[dict]:
    """Measure every registration in *aligned_folder* and save the results (see
    ``prepare_comparison`` and ``run_comparison``). Reads the registrations only."""
    planes, plan = prepare_comparison(
        aligned_folder, n_frames, n_blocks, block_frames, overwrite, cell_minutes, scan_px,
        n_cells, seg_file, mask_image, putative, fluorescence, diameter_px,
    )  # fmt: skip
    return run_comparison(aligned_folder, planes, plan, patch_px, detrend_s)


# ---- plotting ---------------------------------------------------------------------


HIGHER, LOWER = "higher is better", "lower is better"
IF_RED_QUIET = "higher is better if red is at noise"
WITHIN_MODE = "higher is better, within a mode"
# (column, title, how to read it); one row of panels per line
PANELS = [
    [("red_corr_median", "Red Frame-To-Mean r", HIGHER),
     ("red_corr_p5", "Red r, 5th Percentile", HIGHER),
     ("red_residual_shift_median_px", "Red Residual Shift, Median (px)", LOWER),
     ("red_crispness", "Red Mean Crispness", HIGHER),
     ("red_fluctuation_pct", "Red Fluctuation (%)", LOWER)],
    [("green_corr_median", "Green Frame-To-Mean r", HIGHER),
     ("green_crispness", "Green Mean Crispness", HIGHER),
     ("green_red_corr_median", "Green-Red Correlation", LOWER),
     ("suite2p_regDX_px", "Suite2p regDX (px)", LOWER),
     ("bad_frames", "Bad Frames", LOWER)],
    [("green_peak_snr_median", "Green Peak SNR, Median", IF_RED_QUIET),
     ("green_peak_snr_p95", "Green Peak SNR, 95th Percentile", IF_RED_QUIET),
     ("green_residual_shift_median_px", "Green Residual Shift, Median (px)", LOWER),
     ("green_local_corr_contrast", "Green Local Correlation Contrast", IF_RED_QUIET),
     ("green_noise_level", "Green Noise Level (%/√Hz)", LOWER)],
    [("green_cell_resting_snr_median", "Green Resting SNR, Median", WITHIN_MODE),
     ("green_cell_transients_per_min", "Green Transients / Min", IF_RED_QUIET),
     ("green_cell_transients_flipped_per_min", "Green Flipped Transients / Min", "noise count"),
     ("green_cell_cells_with_transient", "Cells With >= 1 Transient", IF_RED_QUIET),
     ("red_cell_transients_per_min", "Red Transients / Min", "motion; noise count is best")],
]  # fmt: skip


def short_label(row: dict) -> str:
    """The save_folder without its 'suite2p_' prefix, and the registration's mode."""
    name = row["registration"].removeprefix("suite2p_") or row["registration"]
    mode = {True: "nonrigid", False: "rigid"}.get(row.get("nonrigid"), "mode unknown")
    return f"{name}\n({mode})"


def zoomed_limits(values: np.ndarray) -> tuple[float, float] | None:
    """Y limits around the finite values, never below 0 for values that are all >= 0."""
    finite = values[np.isfinite(values)]
    if not finite.size:
        return None
    spread = max(float(np.ptp(finite)), 1e-3 * max(abs(float(finite.max())), 1.0))
    low = float(finite.min()) - 0.4 * spread
    return (max(low, 0.0) if finite.min() >= 0 else low), float(finite.max()) + 0.6 * spread


def plot_measure(ax, labels: Sequence[str], values: np.ndarray, title: str, reading: str):
    """One measure: a dot per registration, labelled with its value."""
    x = np.arange(len(labels))
    colors = [SERIES_COLORS[i % len(SERIES_COLORS)] for i in x]
    ax.scatter(x, values, c=colors, s=50, zorder=3)
    for xi, value in zip(x, values, strict=True):
        if np.isfinite(value):
            ax.annotate(f"{value:.3g}", (xi, value), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7)  # fmt: skip
    ax.set_title(f"{title}\n({reading})", fontsize=9)
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
    labels = [short_label(row) for row in rows]
    for panel_row, ax_row in zip(PANELS, axes, strict=True):
        for (key, title, reading), ax in zip(panel_row, ax_row, strict=True):
            values = np.asarray([np.nan if r[key] is None else r[key] for r in rows], float)
            plot_measure(ax, labels, values, title, reading)
    axes.flat[0].figure.suptitle("Registration Comparison")
    return axes


IMAGE_ROWS = (("red_mean", "Red Mean"), ("green_mean", "Green Mean"))
DISPLAY_PERCENTILES = (1, 99.5)


def plot_images(labels: Sequence[str], images: Sequence[dict], axes=None) -> np.ndarray:
    """Per registration (columns): the red and the green mean image, each shown from its own
    1st to 99.5th percentile."""
    if axes is None:
        _, axes = plt.subplots(len(IMAGE_ROWS), len(labels), figsize=(3.2 * len(labels), 6.6),
                               constrained_layout=True, squeeze=False)  # fmt: skip
    axes = np.asarray(axes).reshape(len(IMAGE_ROWS), len(labels))
    for column, (label, image) in enumerate(zip(labels, images, strict=True)):
        for ax, (key, _) in zip(axes[:, column], IMAGE_ROWS, strict=True):
            low, high = np.percentile(image[key], DISPLAY_PERCENTILES)
            ax.imshow(image[key], cmap="gray", vmin=low, vmax=high)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[0, column].set_title(label, fontsize=9)
    for ax, (_, name) in zip(axes[:, 0], IMAGE_ROWS, strict=True):
        ax.set_ylabel(name)
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
        "detrend": detrend_s, "fs": plan.fs_hz, "n_cells": plan.n_cells,
        "cell_minutes": plan.cell_frames / plan.fs_hz / 60, "scan_px": plan.scan_px,
        "bin_s": cell_snr.frames_per_bin(plan.fs_hz) / plan.fs_hz,
        "threshold": cell_snr.TRANSIENT_SD, "highpass": cell_snr.HIGHPASS_S,
        "cells": cells_caption(plan),
    }  # fmt: skip


def cells_caption(plan: Plan) -> str:
    """How the cells of the bottom row were found, for the caption."""
    cells = plan.segmented
    if cells is not None:
        reasons = ", ".join(f"{n} {r}" for r, n in cells.dropped.items())
        dropped = (f"; {cells.left_out} more were left out because at least one registration "
                   f"lost them (reasons, a mask can have several: {reasons})"
                   if cells.left_out else "")  # fmt: skip
        masks = cells.masks
        return (f"Cells (bottom row): {len(cells.ids)} Cellpose masks of {masks.seg_file.name}, "
                f"drawn on {masks.image_file.name} and shifted onto each registration's green "
                "mean image (mask_dy, mask_dx in the table), the same masks in every "
                f"registration, each with its ring inside the crop{dropped}; trace = mask minus "
                "a ring around it outside every mask")  # fmt: skip
    return (f"Putative cells (bottom row; the measure of the earlier 71R / AB08 / M comparison): "
            f"the {plan.n_cells} strongest local maxima of the cell-scale structure (difference "
            "of Gaussians) of each registration's green mean image, within its valid region "
            f"less a margin, geometry for a {plan.scan_px} px scan; trace = disc minus "
            "surrounding annulus")  # fmt: skip


MEASURES_CAPTION = """Registrations of {recording}. Alignment and signal measures in rows \
{rows} and columns {columns} (inside every registration's unwrapped region), without cell \
segmentation.
Alignment, on the same {n} evenly spaced frames: frame-to-mean r, each frame against the mean \
of those frames, both smoothed with a {sigma} px Gaussian (median and 5th percentile across \
frames); residual shift, median across frames of the subpixel shift that still best aligns \
each frame to that mean; crispness, gradient energy of the mean over its intensity; Suite2p's \
regDX, combined rigid + nonrigid shift between the top and bottom of each principal component, \
averaged over components (computed with each registration's own smooth_sigma; blank when \
Suite2p did not compute it); bad frames, Suite2p's count.
Signal, on {n_blocks} contiguous blocks of {block_frames} frames ({fs:g} Hz) spread over the \
recording, from {patch} x {patch} px patches high-passed over {detrend:g} s: peak SNR, a \
patch's 99th percentile (3-frame running mean) above its median over its frame-to-frame \
noise (robust SD of first differences / sqrt 2), median or 95th percentile across patches; \
red fluctuation, median across patches of the high-passed SD over the mean, in % (the red \
signal is activity-independent, so beyond its noise this is motion left in it); green-red \
correlation, median correlation of each patch's green and red fluctuations (shared motion); \
local correlation contrast, 99th percentile minus median of each pixel's correlation with the \
mean of its 8 neighbours; noise level (Rupprecht et al. 2021), median absolute frame-to-frame \
change of ΔF/F (F0: 20th percentile) over sqrt(frame rate), median over the brighter half of \
the patches; F0 includes any detector offset, so compare registrations, not recordings.
{cells} in {bin_s:.2g} s bins over the first \
{cell_minutes:.1f} min, without bins where the brightness around every cell jumps; noise = \
robust SD of successive bin differences / sqrt 2. Resting SNR: a cell's mean excess over its \
surroundings over its noise (median across cells; contrast, not activity). Transients: rising \
crossings of {threshold:g} SD after removing a Gaussian running mean (sigma {highpass:g} s), \
summed over cells, per minute; flipped: the same on sign-flipped traces (noise count); red: \
the same in red at the same places (at the noise count when no motion or stray light is left).
How to read: red is the structural channel (activity-independent, the fairer test of \
motion); motion also raises green peak SNR, local correlation and transient counts, so read \
those only among registrations the red measures find equally well aligned. Nonrigid \
registration interpolates each frame by its own subpixel shifts and rigid registration moves \
frames by whole pixels; interpolation alone changes the measures that depend on pixel noise \
(fluctuation, noise level, local correlation, crispness, frame-to-mean r, resting SNR), in \
either direction, so compare those within one mode (rigid or nonrigid; local correlation only \
between rigid registrations, as under nonrigid registration the interpolation varies across \
Suite2p's block grid, which changes with the settings) and, between modes, rely on residual \
shift and on transient counts against the flipped and red counts. The crop \
depends on which registrations are compared, so numbers compare within this comparison only. \
Y-axes are zoomed on each measure's range (dots, not bars). Table: \
registration_comparison.csv (also root-mean-square residual shifts and the local \
correlation's 99th percentile).
"""

IMAGES_CAPTION = """Registrations of {recording}, rows {rows} and columns {columns}. Mean of \
the same {n} evenly spaced frames of each registration, red (structural) on top and green \
(functional) below, each image shown from its own 1st to 99.5th percentile. Misalignment \
blurs a mean image; differences of a fraction of a pixel may not be visible, and the \
measures figure quantifies them.
"""
