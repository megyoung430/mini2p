"""Are a stage's frames in the acquired order? Checked against the acquired TIFF by content.

Each step that works in chunks (bidiphase, destriping, distortion correction, registration)
must put every frame back where it came from. Bookkeeping errors (a chunk written twice or
never, a block moved, an off-by-one at a seam, swapped channels) leave the frame count right,
so they are checked by content: the mean of each frame, per channel, minus its running mean
over ``FAST_WINDOW`` frames (``fast_part``), is a signature that these steps barely change (a
rigid shift with wrap-around keeps the frame sum; bidiphase, destriping and resampling change
it slightly), while laser, detector and motion fluctuations make it differ from frame to
frame. Two checks use it:

- **windows**: every ``window`` frames of a stage (the last window ends at the last frame)
  are matched against every position of the acquired signature (normalised correlation):
  *in place* (best match at its own position), *moved* (another position matches at least
  ``min_r`` and better than its own by ``margin``: the frames came from there), *ambiguous*
  (best elsewhere, but not clearly; its own position still matches at least ``min_r``) or
  *unclear* (no position matches well: e.g. frames never written, or from beyond the
  frames compared);
- **frames**: each frame of the stage against the acquired frame at the same position. This
  finds blocks shorter than a window (e.g. one of the pipeline's 50- or 100-frame chunks
  written over its neighbour), which a window match can still place at its own position.
  The stage's signature is scaled to the acquired one (least squares); a frame is out of
  place when its difference exceeds both ``MISMATCH_SD`` x the acquired signature's SD and
  ``NOISE_SDS`` x the noise of the differences (from their lower quartile), so a noisier
  stage is judged against its own noise; scale and noise are refitted without the frames
  found out of place. Stretches where at least half of ``MISMATCH_RUN`` frames are out of
  place are reported. A frame from elsewhere differs by about 1.4 acquired SDs, so in a
  stage whose differences are about that noisy (r with the acquired signature below about
  0.9) only the window check remains.

A stage passes with no moved or unclear window, no mismatched stretch, the same frame count
as the run, and a better match with the acquired same channel than with the other one.
The background-removed TIFF is not checked: removing the background takes most of the
frame-to-frame fluctuation out of the frame means.
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import uniform_filter1d

from .movies import open_stage

FAST_WINDOW = 31  # frames (about 4 s at 7.5 Hz): slower changes (bleaching, drift) removed
WINDOW = 200  # frames per matched window: long enough that chance matches stay well below MIN_R
MIN_R = 0.5  # a window matching a position less well than this is not placed there
MARGIN = 0.1  # how much better another position must match to call a window moved
# On a good session (7 Aug 2026) every stage's per-frame difference stayed below 1.0 x the
# acquired signature's SD (99.9th percentile 0.1-0.5); a frame from elsewhere differs by ~1.4
MISMATCH_SD = 0.75
NOISE_SDS = 4.0  # robust SDs of the stage's own differences: chance exceedances ~1 in 10^4
MISMATCH_RUN = 9  # frames: single noisy frames are not a misplaced block
FIT_ROUNDS = 3
LOWER_QUARTILE_SD = 0.3186  # 25th percentile of |x| for normal x, in SDs
READ_FRAMES = 256  # frames read at a time when taking frame means
SIGNATURE_CACHE = "frame_signatures.npz"
CHECKED_STAGES = ("bidiphase", "undistorted", "registered")


# ---------------------------------------------------------------- signatures (pure)


def fast_part(signal: np.ndarray, window: int = FAST_WINDOW) -> np.ndarray:
    """*signal* minus its centred running mean over *window* (odd) frames."""
    window = max(3, window | 1)
    signal = np.asarray(signal, np.float64)
    return signal - uniform_filter1d(signal, window, mode="nearest")


def _zscored_windows(signal: np.ndarray, window: int) -> np.ndarray:
    """Every window of *signal*, centred and scaled to unit norm; a window without any
    fluctuation (e.g. frames never written) is all zeros, so it matches nothing."""
    windows = sliding_window_view(signal, window)
    centred = windows - windows.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(centred, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(norm > 0, centred / norm, 0.0)


def window_starts(n_frames: int, window: int) -> np.ndarray:
    """Non-overlapping windows from frame 0, and one more ending at the last frame."""
    starts = np.arange(0, n_frames - window + 1, window)
    if starts[-1] + window < n_frames:
        starts = np.append(starts, n_frames - window)
    return starts


@dataclass(frozen=True)
class WindowMatch:
    start: np.ndarray  # first frame of each window of the checked stage
    best: np.ndarray  # position in the acquired signature it matches best
    r_best: np.ndarray
    r_own: np.ndarray  # correlation with its own position
    status: np.ndarray  # "in place", "moved", "ambiguous" or "unclear"
    window: int


def match_windows(
    reference: np.ndarray,
    checked: np.ndarray,
    window: int = WINDOW,
    min_r: float = MIN_R,
    margin: float = MARGIN,
) -> WindowMatch:
    """Match the windows of *checked* (a stage's fast signature) against every position of
    *reference* (the acquired one), as described in the module docstring.

    Raises:
        ValueError: If either signature is shorter than *window* or *checked* is longer.
    """
    reference, checked = np.asarray(reference, float), np.asarray(checked, float)
    if len(checked) > len(reference):
        raise ValueError(f"{len(checked)} frames checked against only {len(reference)} acquired")
    if len(checked) < window:
        raise ValueError(f"{len(checked)} frames are fewer than one window of {window}")
    candidates = _zscored_windows(reference, window)
    starts = window_starts(len(checked), window)
    r = _zscored_windows(checked, window)[starts] @ candidates.T  # (windows, positions)
    rows = np.arange(len(starts))
    best = np.argmax(r, axis=1)
    r_best, r_own = r[rows, best], r[rows, starts]
    in_place = best == starts
    moved = ~in_place & (r_best >= min_r) & (r_best - r_own >= margin)
    ambiguous = ~in_place & ~moved & (r_own >= min_r)
    status = np.select(
        [in_place & (r_own >= min_r), moved, ambiguous], ["in place", "moved", "ambiguous"],
        "unclear",
    )  # fmt: skip
    return WindowMatch(starts, best, r_best, r_own, status, window)


def mismatched_stretches(
    reference: np.ndarray,
    checked: np.ndarray,
    threshold_sd: float = MISMATCH_SD,
    noise_sds: float = NOISE_SDS,
    run: int = MISMATCH_RUN,
) -> list[tuple[int, int]]:
    """Half-open frame ranges of *checked* (fast signature) that differ from *reference* at
    the same frames, as described in the module docstring.

    Raises:
        ValueError: If *reference* has no fluctuation to compare with.
    """
    reference = np.asarray(reference, float)[: len(checked)]
    checked = np.asarray(checked, float)
    power = reference @ reference
    if power == 0:
        raise ValueError("The acquired signature has no frame-to-frame fluctuation")
    off = np.zeros(len(checked), bool)
    for _ in range(FIT_ROUNDS):  # refit without the frames found off, so they do not set it
        kept = ~off
        scale = (reference[kept] @ checked[kept]) / (reference[kept] @ reference[kept])
        difference = checked - scale * reference
        # the lower quartile of |difference| (0.3186 SD for normal noise) stays an estimate
        # of the noise even when many frames (blocks of misplaced ones) are far off
        noise = np.percentile(np.abs(difference[kept]), 25) / LOWER_QUARTILE_SD
        limit = max(threshold_sd * abs(scale) * reference.std(), noise_sds * noise)
        off = np.abs(difference) > limit
    flagged = uniform_filter1d(off.astype(float), run, mode="constant") >= 0.5
    edges = np.flatnonzero(np.diff(np.r_[0, flagged.astype(int), 0]))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)]


@dataclass(frozen=True)
class OrderReport:
    stage: str
    channel: str
    n_frames: int  # frames compared (the run's)
    frames_in_file: int | None  # frames the stage's file holds, when counted (TIFF stages)
    match: WindowMatch
    mismatched: list[tuple[int, int]]
    r_same_channel: float  # whole-signature correlation with the acquired same channel
    r_other_channel: float  # ... with the acquired other channel (NaN with one channel)

    def count(self, status: str) -> int:
        return int((self.match.status == status).sum())

    @property
    def channels_swapped(self) -> bool:
        return bool(self.r_other_channel > self.r_same_channel)

    @property
    def wrong_length(self) -> bool:
        return self.frames_in_file is not None and self.frames_in_file != self.n_frames

    @property
    def passed(self) -> bool:
        return not (
            self.count("moved") or self.count("unclear") or self.mismatched
            or self.channels_swapped or self.wrong_length
        )  # fmt: skip

    def summary(self) -> str:
        windows = len(self.match.status)
        counts = ", ".join(
            f"{self.count(s)} {s}" for s in ("moved", "ambiguous", "unclear") if self.count(s)
        )
        text = (
            f"{'OK' if self.passed else 'PROBLEM'}: {self.stage} {self.channel}, "
            f"{self.n_frames} frames: {self.count('in place')} of {windows} windows in place"
            + (f" ({counts})" if counts else "")
            + f"; r with acquired {self.channel} {self.r_same_channel:.2f}"
            + (
                f", other channel {self.r_other_channel:.2f}"
                if np.isfinite(self.r_other_channel)
                else ""
            )
        )
        if self.count("moved"):
            first = int(np.flatnonzero(self.match.status == "moved")[0])
            start = int(self.match.start[first])
            text += (
                f"; first moved window: frames {start}-{start + self.match.window - 1} match "
                f"acquired frame {self.match.best[first]}"
            )
        if self.count("unclear"):
            first = int(self.match.start[np.flatnonzero(self.match.status == "unclear")[0]])
            text += f"; first unclear window from frame {first} (frames never written?)"
        if self.mismatched:
            shown = ", ".join(f"{a}-{b - 1}" for a, b in self.mismatched[:5])
            text += f"; frames unlike the acquired ones at the same position: {shown}"
        if self.wrong_length:
            text += f"; the file holds {self.frames_in_file} frames, the run {self.n_frames}"
        if self.channels_swapped:
            text += "; it matches the other channel better: channels swapped?"
        return text


def _r(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    return float(np.corrcoef(a[:n], b[:n])[0, 1])


def order_report(
    stage: str,
    channel: str,
    checked: np.ndarray,
    acquired: dict[str, np.ndarray],
    frames_in_file: int | None = None,
    **match_options,
) -> OrderReport:
    """The order of one channel of one stage, from frame-mean signatures (before
    ``fast_part``): *checked*, and *acquired* by channel name."""
    fast = {name: fast_part(signal) for name, signal in acquired.items()}
    mine = fast_part(checked)
    others = [name for name in fast if name != channel]
    return OrderReport(
        stage,
        channel,
        len(checked),
        frames_in_file,
        match_windows(fast[channel], mine, **match_options),
        mismatched_stretches(fast[channel], mine),
        _r(mine, fast[channel]),
        _r(mine, fast[others[0]]) if others else float("nan"),
    )


# ---------------------------------------------------------------- reading


def frame_means(source, read_frames: int = READ_FRAMES) -> np.ndarray:
    """Mean of every frame of a stage's channel (``movies.Frames``), read in chunks."""
    return np.concatenate(
        [
            source.read(start, min(start + read_frames, source.n_frames)).mean(axis=(1, 2))
            for start in range(0, source.n_frames, read_frames)
        ]
    )


def _cached(cache: Path, path: Path, key: str, compute) -> np.ndarray:
    """*compute()* for *path*, kept in *cache* (one npz for every file checked from a plane)
    under the file's path, size and modification time and *key*: a replaced file is
    computed again."""
    stat = Path(path).stat()
    full_key = json.dumps([str(Path(path).resolve()), stat.st_size, stat.st_mtime_ns, key])
    saved = {}
    if Path(cache).is_file():
        with np.load(cache) as npz:
            saved = {name: npz[name] for name in npz.files}
    keys = [str(k) for k in saved.get("keys", np.array([], str))]
    if full_key in keys:
        return saved[f"value_{keys.index(full_key)}"]
    value = np.asarray(compute())
    saved[f"value_{len(keys)}"] = value
    saved["keys"] = np.array([*keys, full_key])
    partial = Path(cache).with_suffix(".partial.npz")
    np.savez(partial, **saved)
    partial.replace(cache)
    return value


def cached_frame_means(source, cache: Path, key: str) -> np.ndarray:
    """``frame_means`` of *source* (its first ``n_frames``; *key* says which page of the file:
    channel index and channel count), cached: reading a 14 GB TIFF takes tens of seconds."""
    return _cached(cache, source.path, f"means:{source.n_frames}:{key}",
                   lambda: frame_means(source))  # fmt: skip


def tiff_frames(path: Path, n_channels: int) -> int:
    """Frames in a channel-interleaved TIFF (pages / channels).

    Raises:
        ValueError: If the pages are not whole frames (a last frame half written).
    """
    with tifffile.TiffFile(path) as tif:
        pages = len(tif.pages)
    if pages % n_channels:
        raise ValueError(
            f"{path} has {pages} pages, not a whole number of {n_channels}-channel frames"
        )
    return pages // n_channels


def cached_tiff_frames(path: Path, cache: Path, n_channels: int) -> int:
    """``tiff_frames``, cached: counting the pages walks the whole file."""
    return int(_cached(cache, path, f"frames:{n_channels}", lambda: tiff_frames(path, n_channels)))


def check_frame_order(
    plane_dir: Path,
    stages: tuple[str, ...],
    run,
    raw_tiff: Path | None = None,
    **match_options,
) -> list[OrderReport]:
    """Each channel of each of *stages* (``CHECKED_STAGES``) of a registered plane, checked
    against the first ``ops['nframes']`` frames of the acquired TIFF (the frames the run
    processed). Frame means are cached in the plane folder (``SIGNATURE_CACHE``).

    Args:
        run: the settings the run was written with (``movies.load_run_settings``, or
            ``movies.settings_from_records`` for a QC-notebook run), for the TIFF names and
            channel order.
        raw_tiff: the acquired TIFF, if not where the run's records say.

    Raises:
        ValueError: For a stage this check does not cover.
    """
    unknown = set(stages) - set(CHECKED_STAGES)
    if unknown:
        raise ValueError(f"Frame order is checked for {CHECKED_STAGES}, not {sorted(unknown)}")
    plane_dir = Path(plane_dir)
    ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
    cache = plane_dir / SIGNATURE_CACHE
    channels = run.channels
    shown = "side_by_side" if channels.n_channels > 1 else "green"
    page = {"green": channels.functional_channel - 1, "red": channels.structural_channel - 1}
    with ExitStack() as stack:

        def signatures(stage: str) -> tuple[dict[str, np.ndarray], int | None]:
            sources = open_stage(stage, plane_dir, ops, shown, run, raw_tiff, stack)
            means = {
                name: cached_frame_means(src, cache, f"{name}:{page[name]}/{channels.n_channels}")
                for name, src in sources.items()
            }
            path = next(iter(sources.values())).path
            in_file = (
                None if stage == "registered"
                else cached_tiff_frames(path, cache, channels.n_channels)
            )  # fmt: skip
            return means, in_file

        acquired, _ = signatures("raw")
        reports = []
        for stage in stages:
            means, in_file = signatures(stage)
            reports += [
                order_report(stage, name, signal, acquired, in_file, **match_options)
                for name, signal in means.items()
            ]
        return reports
