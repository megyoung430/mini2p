"""Movies of each preprocessing stage of a run: the whole recording or a range of frames.

A run can be shown at four stages, each from the file the pipeline wrote:

- ``raw``: the acquired TIFF (from the run's ``source_tiff.json``);
- ``bidiphase``: after scan-phase correction (and destriping, in ``with_destriping``);
- ``undistorted``: after distortion correction too;
- ``registered``: after Suite2p motion registration (its ``data.bin`` / ``data_chan2.bin``).

Each stage gives up to two MP4s (H.264, yuv420p, which PowerPoint, Keynote and QuickTime play):

- **unaveraged**: every frame, unsmoothed;
- **averaged**: each movie frame is the mean of ``average_frames`` consecutive frames, either
  in non-overlapping bins (``"bin"``: fewer frames, each covering a longer time) or as a
  running mean (``"rolling"``: one movie frame per acquired frame, smoothed in time).

Movies are greyscale. By default the functional (green) channel is on the left and the
structural (red) channel on the right, each labelled. Each channel's contrast uses the QC
notebook's formula (black and white percentiles, then a gamma), but with the percentiles
taken over a sample of the movie's own frames rather than of the mean image, and fixed for
the whole movie, so brightness changes over the recording stay visible. A label gives the
stage, the time from the start of the run (frame index / frame rate) and the playback speed.
The run is found from cohort, subject, session and run number under the derivatives folder;
all of it, including which frames, stages and versions, is set in the ``[movies]`` section of
the config.

Frame numbers are 0-based, like Suite2p's, and ``end_frame`` is included. All stages have the
same frames, so a frame number means the same moment in every stage's movie.
"""

from __future__ import annotations

import json
import logging
import re
import tomllib
import typing
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import tifffile
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from scipy.ndimage import gaussian_filter

from .background_io import SOURCE_LABELS, BackgroundSource, check_background_tiff
from .batch import MANIFEST_NAME, SESSION_PATTERN
from .config import (
    SETTINGS_RECORD,
    VARIANTS,
    WITH_DESTRIPING_SUBDIR,
    ChannelSettings,
    MotionSource,
    PipelineSettings,
    corrected_tiff_path,
    find_registered_plane,
    run_dir_of,
    settings_from_record,
    undistorted_tiff_path,
    variant_dir_of,
    variant_label,
    with_destriping,
)
from .progress import ProgressLog
from .qc import unwrapped_region
from .registration import resolve_registered_binary
from .tiff_io import read_channel_chunk

log = logging.getLogger(__name__)

Channel = Literal["side_by_side", "green", "red"]
AverageMode = Literal["bin", "rolling"]
Version = Literal["unaveraged", "averaged"]
Stage = Literal["raw", "bidiphase", "undistorted", "registered", "background_removed"]
STAGES: tuple[Stage, ...] = (
    "raw",
    "bidiphase",
    "undistorted",
    "registered",
    "background_removed",
)
# Stages in the registered frame, which suite2p_crop crops
REGISTERED_STAGES = ("registered", "background_removed")
BINARIES = {"green": ("reg_file", "data.bin"), "red": ("reg_file_chan2", "data_chan2.bin")}
CHANNEL_TITLES = {"green": "functional", "red": "structural"}
PANEL_GAP_PX = 4
MOVIE_FOLDER = "movies"
MIN_FPS = 1.0  # imageio-ffmpeg rounds the rate to 0.01 fps
SOURCE_FRAMES_PER_BLOCK = 128  # acquired frames held in memory per channel at a time


class MovieSettings(BaseModel):
    """The ``[movies]`` section of the pipeline config: which run, which frames, which stages
    and versions, where to read and write, and how the movies look. Command-line options and
    keyword arguments to ``load_movie_settings`` override it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Which run: by name, under derivatives_root/<cohort>/<subject>/ses-<session>_*/funcimg/
    # run-<run>, or by path, run_dir (a folder laid out like run-XX/:
    # <variant>/aligned_<channel>/suite2p/...).
    # run_dir, when set, is used and the four by-name settings are ignored.
    cohort: str | None = Field(None, min_length=1)  # e.g. "cohort-01"
    subject: str | None = Field(None, min_length=1)  # subject folder or part of it: "sub-03"
    session: int | None = Field(None, ge=0)  # e.g. 224
    run: int | None = Field(None, ge=1)  # e.g. 1 for run-01
    run_dir: Path | None = None  # e.g. "/Users/you/Desktop/example_ses/img"
    variant: Literal[
        "with_distortion_correction/without_destriping",
        "with_distortion_correction/with_destriping",
        "without_distortion_correction/without_destriping",
        "without_distortion_correction/with_destriping",
    ] = "with_distortion_correction/without_destriping"
    # Which registration: "auto" takes the one aligned_* folder there is (refusing a variant
    # that has both); or "structural" / "functional".
    alignment: MotionSource = "auto"
    # Which frames: the whole recording, or start_frame to end_frame (0-based, both included).
    # No default: a whole TIFF stage is ~14 GB to read, so it is never chosen by omission.
    whole_recording: bool
    start_frame: int | None = Field(None, ge=0)
    end_frame: int | None = Field(None, ge=0)
    # Which stages of the preprocessing to make movies of
    stage_raw: bool = False  # the acquired TIFF
    stage_bidiphase: bool = False  # after phase correction (and destriping)
    stage_undistorted: bool = False  # after distortion correction too (not if it was off)
    stage_registered: bool = True  # after Suite2p motion registration too
    # the registered movie with the vessel background regressed out of every pixel, from
    # the QC notebook's data_background_removed_<source>.tif in the plane folder
    stage_background_removed: bool = False
    background_source: BackgroundSource = "vessel"  # or "non_roi": pixels outside the ROIs
    # Which versions of each: every frame, and frames averaged average_frames at a time
    unaveraged_movie: bool = True
    averaged_movie: bool = True
    # Where to read and write
    derivatives_root: Path | None = None  # holds cohort-XX/<subject>/ses-*/funcimg/run-XX
    raw_tiff: Path | None = None  # the acquired TIFF, if not where source_tiff.json says
    output_dir: Path | None = None  # None: run-XX/<variant>/aligned_<source>/movies
    overwrite: bool = False  # replace existing movies
    save_folder: str = "suite2p"  # Suite2p output folder inside aligned_<channel>/
    # How they look
    channel: Channel = "side_by_side"  # functional left, structural right; or one alone
    average_frames: int = Field(5, ge=2, le=256)
    average_mode: AverageMode = "bin"
    # Playback speed, times real time. A movie must play at >= MIN_FPS (1 fps), so with
    # bins of many frames raise speed: at 7.5 Hz, bin 10 needs speed >= 1.4.
    speed: float = Field(4.0, gt=0)
    # Contrast, as in the QC notebook: percentiles mapped to black and white, then a gamma.
    black_percentile: float = Field(1.0, ge=0, le=100)
    white_percentile: float = Field(99.8, ge=0, le=100)
    gamma: float = Field(1.0, gt=0)
    # Gaussian smoothing within each averaged frame; 0 = none. Unaveraged movies never are.
    spatial_sigma_px: float = Field(0.0, ge=0)
    upscale: int = Field(1, ge=1, le=8)  # each pixel drawn as upscale x upscale pixels
    label: bool = True  # stage, time, speed and channel burnt into the frame
    # Also a copy of each registered movie cropped to the part of the frame no frame's
    # rigid or nonrigid shifts wrap into (qc.unwrapped_region). Only registered frames
    # have wrapped edges, so the other stages are not cropped.
    suite2p_crop: bool = False
    contrast_sample_frames: int = Field(200, ge=1)
    quality: float = Field(8.0, ge=0, le=10)  # H.264 quality, 0-10

    @field_validator("run_dir", "derivatives_root", "raw_tiff", "output_dir", mode="before")
    @classmethod
    def _expand_home(cls, value):
        if value is None:
            return None
        if not str(value).strip():
            raise ValueError("a folder must be named, not left empty")
        return Path(value).expanduser()

    @model_validator(mode="after")
    def _consistent(self) -> MovieSettings:
        by_name = {"cohort": self.cohort, "subject": self.subject, "session": self.session,
                   "run": self.run, "derivatives_root": self.derivatives_root}  # fmt: skip
        if self.run_dir is not None:
            given = [k for k, v in by_name.items() if v is not None]
            if given:
                log.warning("run_dir is set, so %s are ignored", ", ".join(given))
        elif missing := [k for k, v in by_name.items() if v is None]:
            raise ValueError(
                f"Set run_dir, or all of cohort, subject, session, run and derivatives_root "
                f"(missing: {', '.join(missing)})"
            )
        if self.black_percentile >= self.white_percentile:
            raise ValueError(
                f"black_percentile ({self.black_percentile}) must be below white_percentile "
                f"({self.white_percentile})"
            )
        if not self.stages:
            raise ValueError("Choose at least one stage (stage_raw, ..., stage_background_removed)")
        if not self.versions:
            raise ValueError("Choose unaveraged_movie and/or averaged_movie")
        if self.suite2p_crop and not set(self.stages) & set(REGISTERED_STAGES):
            raise ValueError(
                "suite2p_crop crops the registered movies, so it needs stage_registered or "
                "stage_background_removed = true"
            )
        if not self.whole_recording:
            if self.start_frame is None or self.end_frame is None:
                raise ValueError("With whole_recording = false, set start_frame and end_frame")
            if self.start_frame > self.end_frame:
                raise ValueError(
                    f"start_frame ({self.start_frame}) is after end_frame ({self.end_frame})"
                )
        elif self.start_frame is not None or self.end_frame is not None:
            log.warning(
                "whole_recording = true: start_frame %s and end_frame %s are ignored",
                self.start_frame,
                self.end_frame,
            )
        return self

    @property
    def stages(self) -> tuple[Stage, ...]:
        return tuple(stage for stage in STAGES if getattr(self, f"stage_{stage}"))

    @property
    def versions(self) -> tuple[Version, ...]:
        chosen = {"unaveraged": self.unaveraged_movie, "averaged": self.averaged_movie}
        return tuple(version for version, on in chosen.items() if on)

    @property
    def frames_requested(self) -> tuple[bool, int | None, int | None]:
        """``(whole_recording, start_frame, end_frame)``; with the whole recording the frame
        numbers are ignored (with a warning), so they can stay in the config."""
        if self.whole_recording:
            return True, None, None
        return False, self.start_frame, self.end_frame


def load_movie_settings(path: Path, **overrides: object) -> MovieSettings:
    """Read ``[movies]`` from a pipeline config; keyword arguments replace its values."""
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle).get("movies")
    if raw is None:
        raise ValueError(f"No [movies] section in {path}; see config/pipeline.example.toml")
    return MovieSettings.model_validate({**raw, **overrides})


# ---------------------------------------------------------------- finding the run


def _one(found: list[Path], what: str, where: Path) -> Path:
    if len(found) == 1:
        return found[0]
    available = sorted(p.name for p in where.iterdir() if p.is_dir()) if where.is_dir() else []
    problem = "No" if not found else f"{len(found)} folders match:"
    raise FileNotFoundError(
        f"{problem} {what} in {where} {[p.name for p in found] if found else ''}; "
        f"folders there: {available}"
    )


def subject_matches(folder_name: str, subject: str) -> bool:
    """``sub-03``, ``MY_71_R`` and ``sub-03_id-MY_71_R`` all name ``sub-03_id-MY_71_R``."""
    return (
        folder_name == subject
        or folder_name.startswith(f"{subject}_")
        or folder_name.endswith(f"-{subject}")
    )


def find_plane_dir(
    derivatives_root: Path,
    cohort: str,
    subject: str,
    session: int,
    run: int,
    variant: str,
    save_folder: str = "suite2p",
    alignment: MotionSource = "auto",
) -> Path:
    """``<root>/<cohort>/<subject>/ses-<session>_*/funcimg/run-<run>/<variant>/aligned_<source>/
    <save_folder>/plane0`` (see ``find_registered_plane`` for *alignment*).

    Raises:
        FileNotFoundError: If a level is missing or ambiguous, listing what is there.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, not {variant!r}")
    cohort_dir = Path(derivatives_root) / cohort
    subjects = [
        d for d in sorted(cohort_dir.glob("*")) if d.is_dir() and subject_matches(d.name, subject)
    ]
    subject_dir = _one(subjects, f"subject {subject!r}", cohort_dir)
    sessions = [
        d
        for d in sorted(subject_dir.glob("ses-*"))
        if d.is_dir() and (m := SESSION_PATTERN.match(d.name)) and int(m.group(1)) == session
    ]
    session_dir = _one(sessions, f"session {session}", subject_dir)
    funcimg = session_dir / "funcimg"
    run_dir = funcimg / f"run-{run:02d}"
    if not run_dir.is_dir():
        _one([], f"run-{run:02d}", funcimg)
    return registered_plane(run_dir / variant, alignment, save_folder)


def registered_plane(variant_dir: Path, alignment: MotionSource, save_folder: str) -> Path:
    plane_dir = find_registered_plane(variant_dir, alignment, save_folder)
    if plane_dir is None:
        started = sorted(Path(variant_dir).glob(f"aligned_*/{save_folder}/plane0"))
        if started:
            raise FileNotFoundError(
                f"No ops.npy in {[str(p) for p in started]}: Suite2p has not finished (it "
                "writes ops.npy at the end)"
            )
        raise FileNotFoundError(
            f"No registered output in {variant_dir} (no aligned_*/{save_folder}/plane0/ops.npy "
            f"for alignment {alignment!r})"
        )
    return plane_dir


# ---------------------------------------------------------------- frames (pure)


def frame_span(
    n_frames: int, whole_recording: bool, start_frame: int | None, end_frame: int | None
) -> tuple[int, int]:
    """``(start, stop)`` of the frames to show, ``stop`` exclusive.

    ``end_frame`` is included, so frames 10 to 150 are 141 frames.
    """
    if whole_recording:
        if start_frame is not None or end_frame is not None:
            raise ValueError("Give start_frame and end_frame only with whole_recording=False")
        return 0, n_frames
    if start_frame is None or end_frame is None:
        raise ValueError("With whole_recording=False, give both start_frame and end_frame")
    if not 0 <= start_frame <= end_frame < n_frames:
        raise ValueError(
            f"Need 0 <= start_frame <= end_frame < {n_frames} (frames are 0-based), "
            f"got {start_frame} and {end_frame}"
        )
    return start_frame, end_frame + 1


def window_starts(start: int, stop: int, n: int, mode: AverageMode) -> np.ndarray:
    """First source frame of each movie frame; each averages frames ``s .. s + n - 1``.

    ``"bin"`` steps by *n* (a last partial bin is dropped), ``"rolling"`` by one; ``n = 1``
    gives every frame.
    """
    if stop - start < n:
        raise ValueError(f"{stop - start} frames cannot be averaged {n} at a time")
    step = n if mode == "bin" else 1
    return np.arange(start, stop - n + 1, step)


def average_windows(frames: np.ndarray, first: np.ndarray, n: int) -> np.ndarray:
    """Mean of ``frames[f - first[0] : f - first[0] + n]`` for each ``f`` in *first* (float32)."""
    offsets = np.asarray(first) - first[0]
    total = np.zeros((len(offsets), *frames.shape[1:]), np.float32)
    for k in range(n):
        total += frames[offsets + k]
    return total / n


def smooth(frames: np.ndarray, sigma_px: float) -> np.ndarray:
    """Gaussian smoothing within each frame (not across frames)."""
    return gaussian_filter(frames, (0, sigma_px, sigma_px)) if sigma_px > 0 else frames


def display_limits(sample: np.ndarray, black: float, white: float) -> tuple[float, float]:
    """Values shown as black and white: percentiles of *sample*; never an empty range."""
    lo, hi = (float(v) for v in np.nanpercentile(sample, [black, white]))
    if hi > lo:
        return lo, hi
    log.warning("Black and white percentiles are both %g: the channel is flat in this movie", lo)
    return lo, lo + 1.0


def to_display(frames: np.ndarray, lo: float, hi: float, gamma: float) -> np.ndarray:
    """uint8 display frames, as the QC notebook's ``scale_contrast``: clip to [lo, hi], scale
    to [0, 1], raise to *gamma*."""
    scaled = np.clip((np.asarray(frames, np.float32) - lo) / (hi - lo), 0, 1) ** gamma
    return (np.nan_to_num(scaled) * 255).round().astype(np.uint8)


def channels_shown(channel: Channel) -> tuple[str, ...]:
    return ("green", "red") if channel == "side_by_side" else (channel,)


def compose_panels(panels: dict[str, np.ndarray]) -> np.ndarray:
    """``(..., H, W, 3)`` uint8 greyscale: the channels left to right in the order given,
    separated by ``PANEL_GAP_PX`` black columns."""
    first = next(iter(panels.values()))
    *lead, height, width = first.shape
    total = len(panels) * width + (len(panels) - 1) * PANEL_GAP_PX
    grey = np.zeros((*lead, height, total), np.uint8)
    for i, frames in enumerate(panels.values()):
        left = i * (width + PANEL_GAP_PX)
        grey[..., left : left + width] = frames
    return np.repeat(grey[..., None], 3, axis=-1)


def stage_title(
    stage: Stage, run: PipelineSettings | None, background_source: BackgroundSource = "vessel"
) -> str:
    """What had been done to the frames by *stage*, from the settings the run used."""
    removed = f"{SOURCE_LABELS[background_source].lower()} background removed"
    if run is None:
        return removed if stage == "background_removed" else stage
    steps = ["bidiphase" + (" + destriping" if run.destripe.enabled else "")]
    if run.distortion.mode != "off":
        steps.append("distortion")
    steps.append("registration")
    if stage == "background_removed":
        return " + ".join([*steps, removed])
    shown = {"raw": 0, "bidiphase": 1, "undistorted": 2, "registered": len(steps)}[stage]
    return " + ".join(steps[:shown]) if shown else "raw"


def upscale_and_pad(rgb: np.ndarray, factor: int) -> np.ndarray:
    """Each pixel as *factor* x *factor*, then padded to even height and width (for H.264)."""
    if factor > 1:
        rgb = rgb.repeat(factor, axis=-3).repeat(factor, axis=-2)
    h, w = rgb.shape[-3:-1]
    pad = [(0, 0)] * (rgb.ndim - 3) + [(0, h % 2), (0, w % 2), (0, 0)]
    return np.pad(rgb, pad)


def frame_label(first: int, n: int, fs: float, speed: float) -> str:
    """Time of the window's centre from the run start, its frames, and the playback speed."""
    centre_s = (first + (n - 1) / 2) / fs
    frames = f"frame {first}" if n == 1 else f"frames {first}-{first + n - 1}"
    return f"{centre_s:7.1f} s  {frames}  {speed:g}x"


@dataclass(frozen=True)
class MoviePlan:
    """What one movie shows: ``n`` frames per movie frame, starting at each of ``starts``."""

    stage: Stage
    name: str
    n: int
    starts: np.ndarray
    fps: float
    sigma_px: float
    first_frame: int  # first and last source frames actually shown
    last_frame: int


def plan_movie(
    version: Version,
    stage: Stage,
    span: tuple[int, int],
    fs: float,
    settings: MovieSettings,
    names: tuple[str, ...],
    cropped: bool = False,
    alignment: str | None = None,
) -> MoviePlan:
    """The frames, playback rate and file name of one stage's unaveraged or averaged movie
    (pure).

    *names* start the file name: the subject, session and run folder names, or the last two
    folders of a run given by path. The unaveraged movie is never
    smoothed.
    The name gives the frames actually shown, so a dropped partial bin does not appear in it.

    Raises:
        ValueError: If *version* is unknown, or the playback rate would be under ``MIN_FPS``.
    """
    if version not in typing.get_args(Version):
        raise ValueError(f"version must be 'unaveraged' or 'averaged', not {version!r}")
    unaveraged = version == "unaveraged"
    n = 1 if unaveraged else settings.average_frames
    starts = window_starts(*span, n, settings.average_mode)
    step = n if settings.average_mode == "bin" else 1
    fps = fs / step * settings.speed
    if fps < MIN_FPS:
        raise ValueError(
            f"Playback at {fps:.3g} fps is too slow to encode accurately; raise speed "
            f"(now {settings.speed:g}) or average fewer frames"
        )
    sigma = 0.0 if unaveraged else settings.spatial_sigma_px
    first, last = int(starts[0]), int(starts[-1]) + n - 1
    parts = [
        *names,
        variant_label(settings.variant),
        *([alignment] if alignment else []),  # e.g. aligned_structural
        f"{stage}-{settings.background_source}" if stage == "background_removed" else stage,
        settings.channel,
        *(["cropped"] if cropped else []),
        f"frames-{first}-{last}",
        "unaveraged" if unaveraged else f"{settings.average_mode}{n}",
    ]
    if sigma > 0:
        parts.append(f"sigma{sigma:g}")
    parts.append(f"{settings.speed:g}x")
    name = "_".join(parts) + ".mp4"
    return MoviePlan(stage, name, n, starts, fps, sigma, first, last)


def block_size(n: int, mode: AverageMode, source_frames: int = SOURCE_FRAMES_PER_BLOCK) -> int:
    """Movie frames per read so that a read holds about *source_frames* acquired frames."""
    # rolling: at least n movie frames per block, so each acquired frame is read at most twice
    return max(1, source_frames // n) if mode == "bin" else max(n, source_frames - n + 1)


# ---------------------------------------------------------------- reading stages


class Frames(Protocol):
    """One channel of one stage: ``n_frames`` frames of ``frame_shape``."""

    path: Path
    n_frames: int
    frame_shape: tuple[int, int]

    def read(self, start: int, stop: int) -> np.ndarray: ...


class BinaryFrames:
    """A Suite2p registered binary (int16, frames x Ly x Lx)."""

    def __init__(self, path: Path, n_frames: int, frame_shape: tuple[int, int]):
        self.path, self.n_frames, self.frame_shape = Path(path), n_frames, frame_shape
        expected = n_frames * frame_shape[0] * frame_shape[1] * 2
        if self.path.stat().st_size != expected:
            raise ValueError(
                f"{self.path} is {self.path.stat().st_size} bytes, not {expected} for "
                f"{n_frames} x {frame_shape} int16 frames; is the registration complete?"
            )
        self._data = np.memmap(self.path, np.int16, "r", shape=(n_frames, *frame_shape))

    def read(self, start: int, stop: int) -> np.ndarray:
        return np.asarray(self._data[start:stop], dtype=np.float32)


class TiffFrames:
    """One channel of a channel-interleaved TIFF (page = channel + n_channels * frame).

    The frame count is given, not counted: counting walks every page of the file, minutes on a
    14 GB TIFF on the share. A frame past the end of the file fails when it is read.
    """

    def __init__(
        self, tif: tifffile.TiffFile, path: Path, channel: int, n_channels: int, n_frames: int
    ):
        self._tif, self.path = tif, Path(path)
        self._channel, self._n_channels = channel, n_channels
        self.n_frames = n_frames
        self.frame_shape = tuple(tif.pages[0].shape)

    def read(self, start: int, stop: int) -> np.ndarray:
        try:
            chunk = read_channel_chunk(self._tif, start, stop, self._channel, self._n_channels)
        except IndexError as error:
            raise ValueError(
                f"{self.path} has fewer than {stop} frames of {self._n_channels} channels"
            ) from error
        return chunk.astype(np.float32)


class CroppedFrames:
    """The (rows, columns) region of another stage's frames."""

    def __init__(self, frames: Frames, rows: slice, cols: slice):
        self._frames, self._rows, self._cols = frames, rows, cols
        self.path, self.n_frames = frames.path, frames.n_frames
        self.frame_shape = (rows.stop - rows.start, cols.stop - cols.start)

    def read(self, start: int, stop: int) -> np.ndarray:
        return self._frames.read(start, stop)[:, self._rows, self._cols]


def load_run_settings(variant_dir: Path) -> PipelineSettings | None:
    """The settings the run was processed with (its ``pipeline_settings.json``), if recorded."""
    record = Path(variant_dir) / SETTINGS_RECORD
    return settings_from_record(record.read_text()) if record.is_file() else None


def phase_record(variant_dir: Path, run: PipelineSettings) -> tuple[Path, dict] | None:
    """The ``.phase.json`` the pipeline wrote beside the variant's bidiphase TIFF, if any.

    Raises:
        ValueError: If there are several.
    """
    found = sorted(Path(variant_dir).glob(f"*_{run.correction_suffix}.phase.json"))
    if len(found) > 1:
        raise ValueError(f"{variant_dir} has {len(found)} bidiphase records: {found}")
    return (found[0], json.loads(found[0].read_text())) if found else None


def acquired_tiff(variant_dir: Path, run: PipelineSettings) -> Path:
    """The acquired TIFF of a run: from its ``source_tiff.json`` (batch runs) or its
    bidiphase TIFF's ``.phase.json``; if the recorded path is not there (the folder was
    copied), the file of that name in the run folder.

    Raises:
        FileNotFoundError: If neither record names it.
        ValueError: If ``source_tiff.json`` lacks ``source_tiff``.
    """
    variant_dir = Path(variant_dir)
    manifest = variant_dir / MANIFEST_NAME
    if manifest.is_file():
        recorded = json.loads(manifest.read_text()).get("source_tiff")
        if not recorded:
            raise ValueError(f"{manifest} has no source_tiff; set raw_tiff in [movies]")
        return Path(recorded)
    record = phase_record(variant_dir, run)
    if record is None:
        raise FileNotFoundError(
            f"{variant_dir} has neither {MANIFEST_NAME} nor a .phase.json to say which TIFF was "
            "acquired; set raw_tiff in [movies]"
        )
    recorded = Path(record[1]["input_tiff"])
    beside = run_dir_of(variant_dir) / recorded.name
    return recorded if recorded.is_file() or not beside.is_file() else beside


def stage_tiff(
    stage: Stage, variant_dir: Path, run: PipelineSettings, raw_tiff: Path | None
) -> Path:
    """The TIFF the pipeline read (``raw``) or wrote (``bidiphase``, ``undistorted``).

    The acquired TIFF is *raw_tiff* if given, else the one the run's records name
    (``acquired_tiff``).
    """
    variant_dir = Path(variant_dir)
    if raw_tiff is None:
        raw_tiff = acquired_tiff(variant_dir, run)
    if stage == "undistorted" and run.distortion.mode == "off":
        raise ValueError(
            "Distortion correction was off for this run, so it has no undistorted TIFF; "
            "set stage_undistorted = false"
        )
    path = {
        "raw": lambda: Path(raw_tiff),
        "bidiphase": lambda: corrected_tiff_path(variant_dir, raw_tiff, run),
        "undistorted": lambda: undistorted_tiff_path(variant_dir, raw_tiff, run),
    }[stage]()
    if not path.is_file():
        hint = "; set raw_tiff in [movies] if it has moved" if stage == "raw" else ""
        raise FileNotFoundError(f"No {stage} TIFF: {path}{hint}")
    return path


def open_stage(
    stage: Stage,
    plane_dir: Path,
    ops: dict,
    channel: Channel,
    run: PipelineSettings | None,
    raw_tiff: Path | None,
    stack: ExitStack,
    background_source: BackgroundSource = "vessel",
) -> dict[str, Frames]:
    """The frames of each channel shown, for one stage; TIFFs stay open until *stack* closes.

    Every stage has ``ops["nframes"]`` frames: the pipeline processed the first
    ``max_timepoints`` of the acquired TIFF, so frame *i* is the same moment in each.
    """
    names = channels_shown(channel)
    n_frames = int(ops["nframes"])
    if stage == "registered":
        shape = (int(ops["Ly"]), int(ops["Lx"]))
        return {
            name: BinaryFrames(
                resolve_registered_binary(ops, plane_dir, *BINARIES[name]), n_frames, shape
            )
            for name in names
        }
    if run is None:
        raise FileNotFoundError(
            f"No {SETTINGS_RECORD} in {variant_dir_of(plane_dir)}: the {stage} stage needs "
            "the channel order and file names the run was processed with"
        )
    if stage == "background_removed":
        path = check_background_tiff(plane_dir, ops, background_source)
    else:
        path = stage_tiff(stage, variant_dir_of(plane_dir), run, raw_tiff)
    tif = stack.enter_context(tifffile.TiffFile(path))
    index = {
        "green": run.channels.functional_channel - 1,
        "red": run.channels.structural_channel - 1,
    }
    return {
        name: TiffFrames(tif, path, index[name], run.channels.n_channels, n_frames)
        for name in names
    }


def _windows(source: Frames, first: np.ndarray, n: int, sigma: float) -> np.ndarray:
    frames = source.read(int(first[0]), int(first[-1]) + n)
    return smooth(average_windows(frames, first, n), sigma)


def channel_limits(
    sources: dict[str, Frames], plan: MoviePlan, settings: MovieSettings
) -> dict[str, tuple[float, float]]:
    """Each channel's black and white values, from up to ``contrast_sample_frames`` of this
    movie's frames (every 4th pixel of each, to keep the sample small)."""
    count = settings.contrast_sample_frames
    picks = np.unique(np.linspace(0, len(plan.starts) - 1, count).round()).astype(int)
    limits = {}
    for name, source in sources.items():
        sample = [
            _windows(source, plan.starts[[i]], plan.n, plan.sigma_px)[0, ::4, ::4] for i in picks
        ]
        limits[name] = display_limits(
            np.stack(sample), settings.black_percentile, settings.white_percentile
        )
    return limits


def rendered_frames(
    sources: dict[str, Frames],
    plan: MoviePlan,
    limits: dict[str, tuple[float, float]],
    settings: MovieSettings,
    fs: float,
    title: str = "",
    block: int | None = None,
) -> Iterator[np.ndarray]:
    """RGB uint8 movie frames, read *block* movie frames at a time (default: ``block_size``)."""
    block = block or block_size(plan.n, settings.average_mode)
    for b in range(0, len(plan.starts), block):
        first = plan.starts[b : b + block]
        panels = {
            name: to_display(
                _windows(source, first, plan.n, plan.sigma_px), *limits[name], settings.gamma
            )
            for name, source in sources.items()
        }
        rgb = compose_panels(panels)
        width = next(iter(panels.values())).shape[-1]
        lefts = [i * (width + PANEL_GAP_PX) * settings.upscale for i in range(len(panels))]
        for frame, f in zip(rgb, first, strict=True):
            frame = upscale_and_pad(frame, settings.upscale)
            if settings.label:
                lines = [title, frame_label(int(f), plan.n, fs, settings.speed).strip()]
                frame = _draw_labels(frame, lines, dict(zip(panels, lefts, strict=True)))
            yield frame


def _draw_labels(rgb: np.ndarray, lines: list[str], panel_lefts: dict[str, int]) -> np.ndarray:
    """*lines* at the top left; each channel's name at the bottom left of its panel."""
    from PIL import Image, ImageDraw, ImageFont

    size = max(10, rgb.shape[0] // 36)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    style = {
        "fill": (255, 255, 255),
        "font": ImageFont.load_default(size),
        "stroke_width": max(1, size // 10),
        "stroke_fill": (0, 0, 0),
    }
    for row, text in enumerate(line for line in lines if line):
        draw.text((size // 2, size // 2 + row * (size + size // 3)), text, **style)
    for name, left in panel_lefts.items():
        draw.text((left + size // 2, rgb.shape[0] - 2 * size), CHANNEL_TITLES[name], **style)
    return np.asarray(image)


def _encoded_frames(path: Path) -> int:
    import imageio_ffmpeg

    return imageio_ffmpeg.count_frames_and_secs(str(path))[0]


def write_mp4(
    path: Path, frames: Iterator[np.ndarray], n_frames: int, fps: float, quality: float
) -> None:
    """Write RGB frames to an H.264 MP4 under a temporary name, and rename it only once the
    file decodes to *n_frames* frames; anything that fails leaves no movie behind.

    imageio-ffmpeg does not report an ffmpeg failure while it finishes the file, hence the
    frame count.
    """
    import imageio_ffmpeg

    path = Path(path)
    partial = path.with_name(path.stem + ".partial.mp4")
    progress = ProgressLog(path.name, n_frames, log.info, min_interval_s=30)
    writer = None
    try:
        for frame in frames:
            if writer is None:
                h, w = frame.shape[:2]
                writer = imageio_ffmpeg.write_frames(
                    str(partial),
                    (w, h),
                    fps=fps,
                    quality=quality,
                    codec="libx264",
                    pix_fmt_out="yuv420p",
                    macro_block_size=2,
                )
                writer.send(None)
            writer.send(np.ascontiguousarray(frame))
            progress.step_done()
        if writer is None:
            raise ValueError(f"No frames to write to {path}")
        writer.close()
        writer = None
        encoded = _encoded_frames(partial)
        if encoded != n_frames:
            raise OSError(f"{partial} holds {encoded} frames, not {n_frames}; the encode failed")
    except BaseException:
        try:
            if writer is not None:
                writer.close()
        finally:
            partial.unlink(missing_ok=True)
        raise
    partial.replace(path)


@dataclass(frozen=True)
class MovieJob:
    plan: MoviePlan
    sources: dict[str, Frames]
    span: tuple[int, int]
    title: str
    fs: float


def settings_from_records(variant_dir: Path, config: PipelineSettings) -> PipelineSettings:
    """Settings for a run folder without ``pipeline_settings.json``: the config's, with the
    channel order from the run's ``.phase.json``, destriping from the variant folder's name,
    and distortion correction as the run's ``.distortion.json`` shows.

    Raises:
        FileNotFoundError: If the run has no ``.phase.json`` to give its channel order.
    """
    variant_dir = Path(variant_dir)
    run = with_destriping(config, variant_dir.name == WITH_DESTRIPING_SUBDIR)
    record = phase_record(variant_dir, run)
    if record is None:
        raise FileNotFoundError(
            f"{variant_dir} has neither {SETTINGS_RECORD} nor a .phase.json, so its channel "
            "order is unknown"
        )
    path, phase = record
    try:
        channels = ChannelSettings.model_validate(
            {
                "n_channels": phase["n_channels"],
                "structural_channel": phase["structural_channel_one_based"],
                "functional_channel": phase["functional_channel_one_based"],
            }
        )
    except (KeyError, ValidationError) as error:
        raise ValueError(f"{path} does not give a usable channel order: {error}") from error
    if channels != run.channels:
        log.warning("%s: channel order from %s, not the config's", variant_dir, path.name)
    undistorted = any(variant_dir.glob(f"*_{run.correction_suffix}_undistorted.distortion.json"))
    distortion = run.distortion
    if undistorted and distortion.mode == "off":
        distortion = distortion.model_copy(update={"mode": "cluster_full"})
    elif not undistorted and distortion.mode != "off":
        distortion = distortion.model_copy(update={"mode": "off"})
    log.warning(
        "No %s in %s: channel order from %s, file names from the config",
        SETTINGS_RECORD,
        variant_dir,
        path.name,
    )
    return run.model_copy(update={"channels": channels, "distortion": distortion})


def open_jobs(
    settings: MovieSettings,
    plane_dir: Path,
    names: tuple[str, ...],
    stack: ExitStack,
    config: Callable[[], PipelineSettings] | None = None,
) -> list[MovieJob]:
    """Open every stage's files and plan every movie, before anything is written.

    The TIFF stages use the run's ``pipeline_settings.json``. A run given by path
    (``run_dir``) without one, e.g. made by the QC notebook, uses ``settings_from_records``
    with the config's settings, which *config* loads only then.
    """
    variant_dir = variant_dir_of(plane_dir)
    alignment = plane_dir.parents[1].name  # in every file name: the folder can be left
    stages = settings.stages
    if set(stages) - {"registered"}:
        run = load_run_settings(variant_dir)  # the TIFF stages need it: fail if unreadable
        if run is None and settings.run_dir is not None and config is not None:
            run = settings_from_records(variant_dir, config())
    else:
        run = _settings_for_title(variant_dir)
    if not (plane_dir / "ops.npy").is_file():
        raise FileNotFoundError(
            f"No {plane_dir / 'ops.npy'}: Suite2p has not finished (it writes ops.npy at the end)"
        )
    ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
    fs = float(ops["fs"])
    span = frame_span(int(ops["nframes"]), *settings.frames_requested)
    region = crop_region(ops) if settings.suite2p_crop else None
    if region is not None and set(stages) - {"registered"}:
        log.info(
            "suite2p_crop: only the registered stage has wrapped edges, so only the "
            "registered stage is cropped"
        )
    jobs = []
    for stage in stages:
        sources = open_stage(
            stage, plane_dir, ops, settings.channel, run, settings.raw_tiff, stack,
            settings.background_source,
        )  # fmt: skip
        title = stage_title(stage, run, settings.background_source)
        for version in settings.versions:
            plan = plan_movie(version, stage, span, fs, settings, names, alignment=alignment)
            jobs.append(MovieJob(plan, sources, span, title, fs))
            if region is not None and stage in REGISTERED_STAGES:
                cropped = {name: CroppedFrames(src, *region) for name, src in sources.items()}
                plan = plan_movie(
                    version, stage, span, fs, settings, names, cropped=True, alignment=alignment
                )
                jobs.append(MovieJob(plan, cropped, span, f"{title}, cropped", fs))
    return jobs


def crop_region(ops: dict) -> tuple[slice, slice]:
    """The registered frame's unwrapped region, from the whole recording's shifts, so every
    cropped movie of a run has the same size whatever frames it shows."""
    try:
        return unwrapped_region(ops)
    except KeyError as error:
        raise ValueError(f"suite2p_crop needs the run's registration shifts: {error}") from error


def _settings_for_title(variant_dir: Path) -> PipelineSettings | None:
    """The run's settings if they still validate; they only name the stage in the label."""
    try:
        return load_run_settings(variant_dir)
    except ValidationError as error:
        log.warning(
            "%s no longer matches the settings (%s); plain stage titles", variant_dir, error
        )
        return None


RUN_FOLDER = re.compile(r"^run-\d+$")


def run_folder(settings: MovieSettings) -> tuple[Path, tuple[str, ...]]:
    """The run's plane folder, and the folder names that start its movies' file names:
    subject, session and run, also for a batch run given by path; else the path's last two
    folders."""
    if settings.run_dir is None:
        plane_dir = find_plane_dir(
            settings.derivatives_root,
            settings.cohort,
            settings.subject,
            settings.session,
            settings.run,
            settings.variant,
            settings.save_folder,
            settings.alignment,
        )
        run_dir = run_dir_of(variant_dir_of(plane_dir))
    else:
        run_dir = Path(settings.run_dir).resolve()
        variant_dir = run_dir / settings.variant
        if not variant_dir.is_dir():
            folders = (
                sorted(p.name for p in run_dir.iterdir() if p.is_dir()) if run_dir.is_dir() else []
            )
            raise FileNotFoundError(f"No {variant_dir}; folders in {run_dir}: {folders}")
        plane_dir = registered_plane(variant_dir, settings.alignment, settings.save_folder)
    in_batch_layout = (
        RUN_FOLDER.match(run_dir.name)
        and run_dir.parent.name == "funcimg"
        and SESSION_PATTERN.match(run_dir.parents[1].name)
    )
    if in_batch_layout:
        return plane_dir, (run_dir.parents[2].name, run_dir.parents[1].name, run_dir.name)
    return plane_dir, (run_dir.parent.name, run_dir.name)


def make_movies(
    settings: MovieSettings, config: Callable[[], PipelineSettings] | None = None
) -> list[Path]:
    """Write the movies *settings* ask for (one per stage and version), and return their paths.

    The run is ``settings.run_dir`` if set, else found by cohort, subject, session and run.
    Every stage's files are found and every movie is planned before anything is written, so
    a missing file or an existing movie stops the call before it starts (a TIFF shorter than
    the run is only found when its frames are read). The TIFF stages use the channel order
    and file names in the run's ``pipeline_settings.json``; a run given by path without one
    uses its ``.phase.json`` and the settings *config* loads (see ``settings_from_records``).
    Each movie gets its own display range per channel (averaging narrows the range of
    values, and the stages are on different scales).
    """
    plane_dir, names = run_folder(settings)
    # Beside the registration they show, so movies of two alignments never collide.
    alignment_folder = plane_dir.parents[1]
    out_dir = Path(settings.output_dir) if settings.output_dir else alignment_folder / MOVIE_FOLDER
    with ExitStack() as stack:
        jobs = open_jobs(settings, plane_dir, names, stack, config)
        for job in jobs:
            if (out_dir / job.plan.name).exists() and not settings.overwrite:
                raise FileExistsError(
                    f"{out_dir / job.plan.name} exists; set overwrite = true to replace it"
                )
        out_dir.mkdir(parents=True, exist_ok=True)
        written = [_write_job(job, out_dir, settings) for job in jobs]
    return written


def _write_job(job: MovieJob, out_dir: Path, settings: MovieSettings) -> Path:
    plan = job.plan
    if (plan.first_frame, plan.last_frame + 1) != job.span:
        log.info(
            "%s: frames %d-%d requested; %d-%d shown (whole windows of %d only)",
            plan.name,
            job.span[0],
            job.span[1] - 1,
            plan.first_frame,
            plan.last_frame,
            plan.n,
        )
    limits = channel_limits(job.sources, plan, settings)
    log.info(
        "%s: %d movie frames at %.3g fps from %s; display ranges %s",
        plan.name,
        len(plan.starts),
        plan.fps,
        sorted({str(source.path) for source in job.sources.values()}),
        limits,
    )
    path = out_dir / plan.name
    frames = rendered_frames(job.sources, plan, limits, settings, job.fs, job.title)
    write_mp4(path, frames, len(plan.starts), plan.fps, settings.quality)
    return path
