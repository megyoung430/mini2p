"""Are a stage's frames in the acquired order? Checked against the acquired TIFF by content.

Each step that works in chunks (bidiphase, destriping, distortion correction, registration)
must put every frame back where it came from. Bookkeeping errors (a chunk written twice, a
block moved, an off-by-one at a seam, swapped channels) leave the frame count right, so they
are checked by content instead: the mean of each frame, per channel, with its slow trend
removed (``fast_part``), is a signature that these steps barely change (a rigid shift with
wrap-around keeps the frame sum; bidiphase, destriping and resampling change it slightly),
while laser, detector and motion fluctuations make it differ from frame to frame.

Every ``window`` frames of a stage's signature are matched against every position of the
acquired one (normalised correlation): a window in place matches best at its own position.

- **in place**: its best match is its own position (within ``tolerance`` frames);
- **moved**: another position matches at least ``min_r`` and clearly better than its own
  (by ``margin``): the frames came from elsewhere;
- **unclear**: no position matches well (the signature is too weak there to tell).

The background-removed TIFF is not checked this way: removing the background takes most of
the frame-to-frame fluctuation out of the frame means.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import uniform_filter1d

FAST_WINDOW = 31  # frames: the running mean removed from a signature (about 4 s at 7.5 Hz)
WINDOW = 200  # frames per matched window
MIN_R = 0.5
MARGIN = 0.1
TOLERANCE = 0  # frames: a frame off by one is out of place
READ_FRAMES = 256
SIGNATURE_CACHE = "frame_signatures.npz"


# ---------------------------------------------------------------- signatures (pure)


def fast_part(signal: np.ndarray, window: int = FAST_WINDOW) -> np.ndarray:
    """*signal* minus its centred running mean over *window* (odd) frames."""
    window = max(3, window | 1)
    signal = np.asarray(signal, np.float64)
    return signal - uniform_filter1d(signal, window, mode="nearest")


def _zscored_windows(signal: np.ndarray, window: int) -> np.ndarray:
    windows = sliding_window_view(signal, window)
    centred = windows - windows.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(centred, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(norm > 0, centred / norm, 0.0)


@dataclass(frozen=True)
class WindowMatch:
    start: np.ndarray  # first frame of each window of the checked stage
    best: np.ndarray  # position in the acquired signature it matches best
    r_best: np.ndarray
    r_own: np.ndarray  # correlation with its own position
    status: np.ndarray  # "in place", "moved" or "unclear"
    window: int


def match_windows(
    reference: np.ndarray,
    checked: np.ndarray,
    window: int = WINDOW,
    min_r: float = MIN_R,
    margin: float = MARGIN,
    tolerance: int = TOLERANCE,
) -> WindowMatch:
    """Match every non-overlapping *window* of *checked* (a stage's fast signature) against
    every position of *reference* (the acquired one), as described in the module docstring.

    Raises:
        ValueError: If either signature is shorter than *window* or *checked* is longer.
    """
    reference, checked = np.asarray(reference, float), np.asarray(checked, float)
    if len(checked) > len(reference):
        raise ValueError(f"{len(checked)} frames checked against only {len(reference)} acquired")
    if len(checked) < window:
        raise ValueError(f"{len(checked)} frames are fewer than one window of {window}")
    candidates = _zscored_windows(reference, window)
    starts = np.arange(0, len(checked) - window + 1, window)
    pieces = _zscored_windows(checked, window)[starts]
    r = pieces @ candidates.T  # (windows, positions)
    best = np.argmax(r, axis=1)
    r_best = r[np.arange(len(starts)), best]
    r_own = r[np.arange(len(starts)), starts]
    in_place = np.abs(best - starts) <= tolerance
    moved = ~in_place & (r_best >= min_r) & (r_best - r_own >= margin)
    status = np.where(in_place | ~moved & (r_own >= min_r), "in place", "unclear")
    status = np.where(moved, "moved", status)
    return WindowMatch(starts, best, r_best, r_own, status, window)


@dataclass(frozen=True)
class OrderReport:
    stage: str
    channel: str
    n_frames: int
    n_acquired: int
    match: WindowMatch
    r_same_channel: float  # whole-signature correlation with the acquired same channel
    r_other_channel: float  # ... with the acquired other channel

    @property
    def moved(self) -> int:
        return int((self.match.status == "moved").sum())

    @property
    def unclear(self) -> int:
        return int((self.match.status == "unclear").sum())

    @property
    def channels_swapped(self) -> bool:
        return self.r_other_channel > self.r_same_channel

    @property
    def passed(self) -> bool:
        return self.moved == 0 and not self.channels_swapped

    def summary(self) -> str:
        windows = len(self.match.status)
        verdict = "OK" if self.passed else "PROBLEM"
        text = (
            f"{verdict}: {self.stage} {self.channel}, {self.n_frames} frames: "
            f"{windows - self.moved - self.unclear} of {windows} windows in place, "
            f"{self.moved} moved, {self.unclear} unclear; r with acquired {self.channel} "
            f"{self.r_same_channel:.2f}, with the other channel {self.r_other_channel:.2f}"
        )
        if self.moved:
            first = int(np.flatnonzero(self.match.status == "moved")[0])
            text += (
                f"; first moved window: frames {self.match.start[first]}-"
                f"{self.match.start[first] + self.match.window - 1} match acquired frame "
                f"{self.match.best[first]}"
            )
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
    **match_options,
) -> OrderReport:
    """The order of one channel of one stage, from frame-mean signatures (raw, before
    ``fast_part``): *checked*, and *acquired* by channel name ("green", "red")."""
    fast = {name: fast_part(signal) for name, signal in acquired.items()}
    mine = fast_part(checked)
    other = next(name for name in fast if name != channel)
    return OrderReport(
        stage,
        channel,
        len(checked),
        len(acquired[channel]),
        match_windows(fast[channel], mine, **match_options),
        _r(mine, fast[channel]),
        _r(mine, fast[other]),
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


def _file_key(path: Path, n_frames: int) -> str:
    stat = Path(path).stat()
    return json.dumps([str(Path(path).resolve()), stat.st_size, stat.st_mtime_ns, n_frames])


def cached_frame_means(source, cache: Path, channel: str) -> np.ndarray:
    """``frame_means`` of *source* (the *channel* of a stage), kept in *cache* (one npz for
    every file checked from a plane) under its path, size, modification time, frame count
    and channel: reading a 14 GB TIFF takes minutes, and a replaced file is read again."""
    key = f"{_file_key(source.path, source.n_frames)}#{channel}"
    saved = dict(np.load(cache)) if Path(cache).is_file() else {}
    keys = [str(k) for k in saved.get("keys", np.array([], str))]
    if key in keys:
        return saved[f"means_{keys.index(key)}"]
    means = frame_means(source)
    saved[f"means_{len(keys)}"] = means
    saved["keys"] = np.array([*keys, key])
    tmp = Path(cache).with_suffix(".partial.npz")
    np.savez(tmp, **saved)
    tmp.replace(cache)
    return means


CHECKED_STAGES = ("bidiphase", "undistorted", "registered")


def check_frame_order(
    plane_dir: Path,
    stages: tuple[str, ...],
    run,
    raw_tiff: Path | None = None,
    **match_options,
) -> list[OrderReport]:
    """Each channel of each of *stages* (``CHECKED_STAGES``) of a registered plane, checked
    against the acquired TIFF. Frame means are cached in the plane folder
    (``SIGNATURE_CACHE``).

    Args:
        run: the settings the run was processed with (``movies.load_run_settings``), for the
            TIFF names and channel order.
        raw_tiff: the acquired TIFF, if not where the run's records say.

    Raises:
        ValueError: For a stage this check does not cover.
    """
    from contextlib import ExitStack

    from .movies import open_stage

    unknown = set(stages) - set(CHECKED_STAGES)
    if unknown:
        raise ValueError(f"Frame order is checked for {CHECKED_STAGES}, not {sorted(unknown)}")
    plane_dir = Path(plane_dir)
    ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
    cache = plane_dir / SIGNATURE_CACHE
    with ExitStack() as stack:

        def means(stage: str) -> dict[str, np.ndarray]:
            sources = open_stage(stage, plane_dir, ops, "side_by_side", run, raw_tiff, stack)
            return {name: cached_frame_means(src, cache, name) for name, src in sources.items()}

        acquired = means("raw")
        return [
            order_report(stage, channel, signal, acquired, **match_options)
            for stage in stages
            for channel, signal in means(stage).items()
        ]
