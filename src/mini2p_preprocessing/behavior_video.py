"""The behavior video of a run, paired with its imaging frames by pyControl time.

magnitude-bandit-analysis writes two files beside each run folder (``run-XX/``), both on
pyControl's clock (s, NaN outside the sync pulses):

- ``frame_times.npz``: ``time_s``, the start of each imaging frame;
- ``video_frame_times.npz``: ``time_s``, the start of each frame of the run's behavior video,
  and ``video``, the video's path.

A movie frame shows the video frame that was on screen at the middle of its imaging window
(from the start of its first imaging frame to the end of its last): the last video frame to
start at or before that moment. Where the movie frame or the video has no time there, the
behavior panel is black.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

FRAME_TIMES_FILE = "frame_times.npz"
VIDEO_FRAME_TIMES_FILE = "video_frame_times.npz"
WRITTEN_BY = (
    "magnitude-bandit-analysis's scripts/funcimg/reward_responses.ipynb (steps 1 and 1b) writes it"
)


@dataclass(frozen=True)
class BehaviorSync:
    frame_time_s: np.ndarray  # pyControl time of each imaging frame's start, one per frame
    video_time_s: np.ndarray  # pyControl time of each video frame's start
    video: Path


# ---------------------------------------------------------------- pairing frames (pure)


def movie_frame_times(frame_time_s: np.ndarray, starts: np.ndarray, n: int) -> np.ndarray:
    """pyControl time of the middle of each movie frame's window (imaging frames ``s`` to
    ``s + n - 1``, the last lasting one frame period); NaN if any of them has no time."""
    frame_time_s = np.asarray(frame_time_s, np.float64)
    period = np.nanmedian(np.diff(frame_time_s))
    window = frame_time_s[np.asarray(starts)[:, None] + np.arange(n)]
    middle = (window[:, 0] + window[:, -1] + period) / 2
    return np.where(np.isnan(window).any(axis=1), np.nan, middle)


def check_frame_times(time_s: np.ndarray, what: str, min_timed: int = 1) -> None:
    """Refuse frame times with fewer than *min_timed* finite values, or whose finite values
    do not increase."""
    timed = np.asarray(time_s, np.float64)
    timed = timed[np.isfinite(timed)]
    if len(timed) < min_timed:
        raise ValueError(f"{what} has {len(timed)} frames with a time; needs {min_timed}")
    if np.any(np.diff(timed) <= 0):
        raise ValueError(f"{what}: frame times must increase")


def video_frame_for(times: np.ndarray, video_time_s: np.ndarray) -> np.ndarray:
    """Index of the video frame on screen at each time: the last one to start at or before
    it. -1 where the time is NaN, before the first timed video frame, or more than one video
    frame period after the last timed frame.

    Raises:
        ValueError: If fewer than two video frames have a time, or they do not increase.
    """
    check_frame_times(video_time_s, "The video", min_timed=2)
    video_time_s = np.asarray(video_time_s, np.float64)
    timed = np.flatnonzero(np.isfinite(video_time_s))
    starts = video_time_s[timed]
    period = np.median(np.diff(starts))
    times = np.asarray(times, np.float64)
    position = np.searchsorted(starts, times, side="right") - 1
    clipped = np.clip(position, 0, len(starts) - 1)
    before_last = position < len(starts) - 1  # a later frame starts after it: still shown
    shown = np.isfinite(times) & (position >= 0) & (before_last | (times - starts[-1] < period))
    return np.where(shown, timed[clipped], -1)


def behavior_width(video_size: tuple[int, int], height: int) -> int:
    """Width of a behavior panel *height* pixels high with the video's aspect, rounded to
    an even number (H.264)."""
    width, video_height = video_size
    return 2 * round(height * width / video_height / 2)


def behavior_panel(frame: np.ndarray | None, height: int, width: int) -> np.ndarray:
    """``(height, width, 3)`` uint8 grey: the video frame resized, or black for no frame."""
    if frame is None:
        return np.zeros((height, width, 3), np.uint8)
    from PIL import Image

    resized = np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS))
    return np.repeat(resized[..., None], 3, axis=-1)


# ---------------------------------------------------------------- files


def _load_times(path: Path, keys: tuple[str, ...]) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"No {path}: {WRITTEN_BY}")
    with np.load(path) as saved:
        missing = [key for key in keys if key not in saved.files]
        if missing:
            raise ValueError(f"{path} lacks {missing}; {WRITTEN_BY.replace('writes it', 'writes')}")
        return {key: saved[key] for key in keys}


def video_frame_count(path: Path) -> int:
    """Frames in a video by its duration and frame rate (quick; exact for a constant rate)."""
    import imageio_ffmpeg

    reader = imageio_ffmpeg.read_frames(str(path))
    try:
        meta = next(reader)
    finally:
        reader.close()
    return round(meta["duration"] * meta["fps"])


def check_frame_count(n_video: int, n_times: int, video: Path) -> None:
    """Refuse a video whose frame count is off from its frame times by more than 1 % (+2),
    the precision of a count from duration and frame rate: the frames would be paired with
    the wrong times."""
    if abs(n_video - n_times) > 0.01 * n_times + 2:
        raise ValueError(
            f"{video} has about {n_video} frames but its frame times list {n_times}: they are "
            "not one recording, or the video is incomplete"
        )


def load_behavior_sync(run_dir: Path, n_frames: int, video: Path | None = None) -> BehaviorSync:
    """The imaging and video frame times of a run, and its video (*video* if given, else the
    one ``video_frame_times.npz`` names).

    The run's frames are the first *n_frames* of the acquisition (the pipeline may have
    processed only the first ``max_timepoints``), so the first *n_frames* imaging times are
    used.

    Raises:
        FileNotFoundError: If either file or the video is missing.
        ValueError: If there are fewer than *n_frames* imaging frame times.
    """
    run_dir = Path(run_dir)
    frame_time_s = _load_times(run_dir / FRAME_TIMES_FILE, ("time_s",))["time_s"]
    if len(frame_time_s) < n_frames:
        raise ValueError(
            f"{run_dir / FRAME_TIMES_FILE} has {len(frame_time_s)} frame times, fewer than the "
            f"run's {n_frames} frames: it belongs to another acquisition"
        )
    saved = _load_times(run_dir / VIDEO_FRAME_TIMES_FILE, ("time_s", "video"))
    video_time_s, recorded = saved["time_s"], Path(str(saved["video"]))
    video = Path(video) if video is not None else recorded
    if not video.is_file():
        raise FileNotFoundError(
            f"No behavior video {video}; set behavior_video in [movies] if it has moved"
        )
    check_frame_times(frame_time_s[:n_frames], str(run_dir / FRAME_TIMES_FILE))
    check_frame_times(video_time_s, str(run_dir / VIDEO_FRAME_TIMES_FILE), min_timed=2)
    check_frame_count(video_frame_count(video), len(video_time_s), video)
    return BehaviorSync(frame_time_s[:n_frames], video_time_s, video)


class VideoFrames:
    """A video's frames as greyscale arrays, decoded once from the start, so they can only be
    asked for in order (the same frame again is fine)."""

    def __init__(self, path: Path):
        import imageio_ffmpeg

        self.path = Path(path)
        # passthrough: every decoded frame once, so frame k is row k of the camera's table
        self._reader = imageio_ffmpeg.read_frames(
            str(path), pix_fmt="gray", bits_per_pixel=8, output_params=["-fps_mode", "passthrough"]
        )
        meta = next(self._reader)
        self.size: tuple[int, int] = tuple(meta["size"])  # width, height
        self._index, self._frame = -1, None

    def frame(self, index: int) -> np.ndarray:
        if index < self._index:
            raise ValueError(f"Video frames are read forward only: {index} after {self._index}")
        width, height = self.size
        while self._index < index:
            try:
                raw = next(self._reader)
            except StopIteration:
                raise ValueError(
                    f"{self.path} has only {self._index + 1} frames; frame {index} was asked "
                    "for (the video is shorter than its frame times)"
                ) from None
            self._index += 1
            self._frame = np.frombuffer(raw, np.uint8).reshape(height, width)
        return self._frame

    def close(self) -> None:
        self._reader.close()

    def __enter__(self) -> VideoFrames:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
