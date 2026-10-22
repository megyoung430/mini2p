"""Movies of each preprocessing stage of a run: the whole recording or a range of frames.

A run can be shown at four stages, each from the file the pipeline wrote:

- ``raw``: the acquired TIFF (from the run's ``source_tiff.json``);
- ``bidiphase``: after scan-phase correction (and destriping, in ``with_destriping``);
- ``undistorted``: after distortion correction too;
- ``registered``: after Suite2p motion registration (its ``data.bin`` / ``data_chan2.bin``).

Each stage gives two MP4s (H.264, yuv420p, which PowerPoint, Keynote and QuickTime play):

- **raw**: every frame, unsmoothed;
- **averaged**: each movie frame is the mean of ``average_frames`` consecutive frames, either
  in non-overlapping bins (``"bin"``: fewer frames, each covering a longer time) or as a
  running mean (``"rolling"``: one movie frame per acquired frame, smoothed in time).

By default the green (functional) channel is on the left and the red (structural) channel on
the right. Each channel's contrast uses the QC notebook's formula (black and white percentiles,
then a gamma), but with the percentiles taken over a sample of the movie's own frames rather
than of the mean image, and fixed for the whole movie, so brightness changes over the
recording stay visible. A label gives the stage, the time from the
start of the run (frame index / frame rate) and the playback speed. The run is found from
cohort, subject, session and run number under the derivatives folder set in ``[movies]``.

Frame numbers are 0-based, like Suite2p's, and ``end_frame`` is included. All stages have the
same frames, so a frame number means the same moment in every stage's movie.
"""

from __future__ import annotations

import json
import logging
import tomllib
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import tifffile
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from scipy.ndimage import gaussian_filter

from .batch import MANIFEST_NAME, SESSION_PATTERN
from .config import (
    VARIANT_SUBDIRS,
    PipelineSettings,
    corrected_tiff_path,
    undistorted_tiff_path,
)
from .pipeline import SETTINGS_RECORD
from .progress import ProgressLog
from .registration import resolve_registered_binary
from .tiff_io import read_channel_chunk

log = logging.getLogger(__name__)

Channel = Literal["side_by_side", "green", "red"]
AverageMode = Literal["bin", "rolling"]
Stage = Literal["raw", "bidiphase", "undistorted", "registered"]
STAGES: tuple[Stage, ...] = ("raw", "bidiphase", "undistorted", "registered")
BINARIES = {"green": ("reg_file", "data.bin"), "red": ("reg_file_chan2", "data_chan2.bin")}
CHANNEL_TITLES = {"green": "green (functional)", "red": "red (structural)"}
TINTS = {"green": (0, 1, 0), "red": (1, 0, 0)}  # which of R, G, B each channel lights
PANEL_GAP_PX = 4
MOVIE_FOLDER = "movies"
MIN_FPS = 1.0  # imageio-ffmpeg rounds the rate to 0.01 fps
SOURCE_FRAMES_PER_BLOCK = 128  # acquired frames held in memory per channel at a time


class MovieSettings(BaseModel):
    """The ``[movies]`` section of the pipeline config; arguments to ``make_movies`` override it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    derivatives_root: Path  # holds cohort-XX/<subject>/ses-*/funcimg/run-XX
    variant: Literal["without_destriping", "with_destriping"] = "without_destriping"
    channel: Channel = "side_by_side"  # green left, red right; or one channel alone
    tint: bool = True  # green channel drawn in green, red in red; False for grey
    average_frames: int = Field(5, ge=2, le=256)
    average_mode: AverageMode = "bin"
    # Playback speed, times real time. A movie must play at >= MIN_FPS (1 fps), so with
    # bins of many frames raise speed: at 7.5 Hz, bin 10 needs speed >= 1.4.
    speed: float = Field(4.0, gt=0)
    # Contrast, as in the QC notebook: percentiles mapped to black and white, then a gamma.
    black_percentile: float = Field(1.0, ge=0, le=100)
    white_percentile: float = Field(99.8, ge=0, le=100)
    gamma: float = Field(1.0, gt=0)
    # Gaussian smoothing within each averaged frame; 0 = none. The raw movie is never smoothed.
    spatial_sigma_px: float = Field(0.0, ge=0)
    upscale: int = Field(1, ge=1, le=8)  # each pixel drawn as upscale x upscale pixels
    label: bool = True  # stage, time and speed burnt into the frame
    contrast_sample_frames: int = Field(200, ge=1)
    quality: float = Field(8.0, ge=0, le=10)  # H.264 quality, 0-10
    output_dir: Path | None = None  # None: run-XX/<variant>/movies
    save_folder: str = "suite2p"  # Suite2p output folder inside the variant folder

    @model_validator(mode="after")
    def _black_below_white(self) -> MovieSettings:
        if self.black_percentile >= self.white_percentile:
            raise ValueError(
                f"black_percentile ({self.black_percentile}) must be below white_percentile "
                f"({self.white_percentile})"
            )
        return self


def load_movie_settings(path: Path, **overrides: object) -> MovieSettings:
    """Read ``[movies]`` from a pipeline config; keyword arguments replace its values."""
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle).get("movies")
    if raw is None:
        raise ValueError(f"No [movies] section in {path}; add one with derivatives_root")
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
) -> Path:
    """``<root>/<cohort>/<subject>/ses-<session>_*/funcimg/run-<run>/<variant>/<save_folder>/plane0``.

    Raises:
        FileNotFoundError: If a level is missing or ambiguous, listing what is there.
    """
    if variant not in VARIANT_SUBDIRS:
        raise ValueError(f"variant must be one of {VARIANT_SUBDIRS}, not {variant!r}")
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
    plane_dir = run_dir / variant / save_folder / "plane0"
    if not (plane_dir / "ops.npy").is_file():
        raise FileNotFoundError(
            f"No registered {variant} output in {run_dir} (no {plane_dir}/ops.npy)"
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


def compose_panels(panels: dict[str, np.ndarray], tint: bool) -> np.ndarray:
    """``(..., H, W, 3)`` uint8: the channels left to right in the order given, each in its
    colour (or grey), separated by ``PANEL_GAP_PX`` black columns."""
    first = next(iter(panels.values()))
    *lead, height, width = first.shape
    total = len(panels) * width + (len(panels) - 1) * PANEL_GAP_PX
    rgb = np.zeros((*lead, height, total, 3), np.uint8)
    for i, (name, frames) in enumerate(panels.items()):
        left = i * (width + PANEL_GAP_PX)
        colours = [c for c, on in enumerate(TINTS[name]) if on] if tint else [0, 1, 2]
        for c in colours:
            rgb[..., left : left + width, c] = frames
    return rgb


def stage_title(stage: Stage, run: PipelineSettings | None) -> str:
    """What had been done to the frames by *stage*, from the settings the run used."""
    if run is None:
        return stage
    steps = ["bidiphase" + (" + destriping" if run.destripe.enabled else "")]
    if run.distortion.mode != "off":
        steps.append("distortion")
    steps.append("registration")
    shown = {"raw": 0, "bidiphase": 1, "undistorted": 2, "registered": len(steps)}[stage]
    return " + ".join(steps[:shown]) if shown else "raw"


def run_stages(run: PipelineSettings) -> tuple[Stage, ...]:
    """The stages the run has files for: no ``undistorted`` TIFF if distortion was off."""
    return tuple(s for s in STAGES if s != "undistorted" or run.distortion.mode != "off")


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
    version: str,
    stage: Stage,
    span: tuple[int, int],
    fs: float,
    settings: MovieSettings,
    names: tuple[str, str, str],
) -> MoviePlan:
    """The frames, playback rate and file name of one stage's raw or averaged movie (pure).

    *names* are the subject, session and run folder names. The raw movie is never smoothed.
    The name gives the frames actually shown, so a dropped partial bin does not appear in it.

    Raises:
        ValueError: If the playback rate would be under ``MIN_FPS``.
    """
    raw = version == "raw"
    n = 1 if raw else settings.average_frames
    starts = window_starts(*span, n, settings.average_mode)
    step = n if settings.average_mode == "bin" else 1
    fps = fs / step * settings.speed
    if fps < MIN_FPS:
        raise ValueError(
            f"Playback at {fps:.3g} fps is too slow to encode accurately; raise speed "
            f"(now {settings.speed:g}) or average fewer frames"
        )
    sigma = 0.0 if raw else settings.spatial_sigma_px
    first, last = int(starts[0]), int(starts[-1]) + n - 1
    subject, session, run = names
    parts = [
        subject,
        session,
        run,
        settings.variant,
        stage,
        settings.channel,
        f"frames-{first}-{last}",
        "raw" if raw else f"{settings.average_mode}{n}",
    ]
    if sigma > 0:
        parts.append(f"sigma{sigma:g}")
    parts.append(f"{settings.speed:g}x")
    return MoviePlan(stage, "_".join(parts) + ".mp4", n, starts, fps, sigma, first, last)


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


def load_run_settings(variant_dir: Path) -> PipelineSettings | None:
    """The settings the run was processed with (its ``pipeline_settings.json``), if recorded."""
    record = Path(variant_dir) / SETTINGS_RECORD
    return PipelineSettings.model_validate_json(record.read_text()) if record.is_file() else None


def stage_tiff(
    stage: Stage, variant_dir: Path, run: PipelineSettings, raw_tiff: Path | None
) -> Path:
    """The TIFF the pipeline read (``raw``) or wrote (``bidiphase``, ``undistorted``).

    The acquired TIFF comes from *raw_tiff*, else from the run's ``source_tiff.json``.
    """
    variant_dir = Path(variant_dir)
    if raw_tiff is None:
        manifest = variant_dir / MANIFEST_NAME
        recorded = (
            json.loads(manifest.read_text()).get("source_tiff") if manifest.is_file() else None
        )
        if recorded is None:
            raise FileNotFoundError(
                f"{manifest} does not say which TIFF was acquired; pass raw_tiff (--raw-tiff)"
            )
        raw_tiff = Path(recorded)
    if stage == "undistorted" and run.distortion.mode == "off":
        raise ValueError(
            "Distortion correction was off for this run, so it has no undistorted TIFF"
        )
    path = {
        "raw": lambda: Path(raw_tiff),
        "bidiphase": lambda: corrected_tiff_path(variant_dir, raw_tiff, run),
        "undistorted": lambda: undistorted_tiff_path(variant_dir, raw_tiff, run),
    }[stage]()
    if not path.is_file():
        hint = "; pass raw_tiff (--raw-tiff) if it has moved" if stage == "raw" else ""
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
            f"No {SETTINGS_RECORD} in {Path(plane_dir).parents[1]}: the {stage} stage needs "
            "the channel order and file names the run was processed with"
        )
    path = stage_tiff(stage, Path(plane_dir).parents[1], run, raw_tiff)
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
        rgb = compose_panels(panels, settings.tint)
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


def open_jobs(
    settings: MovieSettings,
    plane_dir: Path,
    stages: tuple[Stage, ...] | None,
    versions: tuple[str, ...],
    span_request: tuple[bool, int | None, int | None],
    raw_tiff: Path | None,
    stack: ExitStack,
) -> list[MovieJob]:
    """Open every stage's files and plan every movie, before anything is written."""
    unknown = set(versions) - {"raw", "averaged"}
    if unknown or not versions:
        raise ValueError(f"versions must be 'raw' and/or 'averaged', not {versions}")
    variant_dir = plane_dir.parents[1]
    if stages is None or set(stages) - {"registered"}:
        run = load_run_settings(variant_dir)  # the TIFF stages need it: fail if unreadable
    else:
        run = _settings_for_title(variant_dir)
    if stages is None:
        if run is None:
            raise FileNotFoundError(
                f"No {SETTINGS_RECORD} in {variant_dir} to say which stages ran"
            )
        stages = run_stages(run)
    if not stages or set(stages) - set(STAGES):
        raise ValueError(f"stages must be some of {STAGES}, not {stages}")
    if raw_tiff is not None and set(stages) == {"registered"}:
        raise ValueError("raw_tiff is only used by the raw, bidiphase and undistorted stages")
    ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
    fs = float(ops["fs"])
    names = (variant_dir.parents[3].name, variant_dir.parents[2].name, variant_dir.parent.name)
    span = frame_span(int(ops["nframes"]), *span_request)
    jobs = []
    for stage in dict.fromkeys(stages):
        sources = open_stage(stage, plane_dir, ops, settings.channel, run, raw_tiff, stack)
        title = stage_title(stage, run)
        for version in versions:
            plan = plan_movie(version, stage, span, fs, settings, names)
            jobs.append(MovieJob(plan, sources, span, title, fs))
    return jobs


def _settings_for_title(variant_dir: Path) -> PipelineSettings | None:
    """The run's settings if they still validate; they only name the stage in the label."""
    try:
        return load_run_settings(variant_dir)
    except ValidationError as error:
        log.warning(
            "%s no longer matches the settings (%s); plain stage titles", variant_dir, error
        )
        return None


def make_movies(
    settings: MovieSettings,
    cohort: str,
    subject: str,
    session: int,
    run: int,
    whole_recording: bool = True,
    start_frame: int | None = None,
    end_frame: int | None = None,
    stages: tuple[Stage, ...] | None = ("registered",),
    versions: tuple[str, ...] = ("raw", "averaged"),
    overwrite: bool = False,
    raw_tiff: Path | None = None,
) -> list[Path]:
    """Write the raw and/or averaged movie of each stage of one run, and return their paths.

    Every stage's files are found and every movie is planned before anything is written, so
    a missing file or an existing movie stops the call before it starts (a TIFF shorter than
    the run is only found when its frames are read). The TIFF stages use
    the channel order and file names in the run's ``pipeline_settings.json``. Each movie gets
    its own display range per channel (averaging narrows the range of values, and the stages
    are on different scales).

    Args:
        settings (MovieSettings): From ``load_movie_settings`` (with any overrides).
        cohort (str): e.g. ``"cohort-01"``.
        subject (str): The subject folder or part of it, e.g. ``"sub-03"`` or ``"MY_71_R"``.
        session (int): Session number, e.g. ``224``.
        run (int): Run number, e.g. ``1`` for ``run-01``.
        whole_recording (bool): Every frame of the run; otherwise give the two frames below.
        start_frame (int | None): First frame shown (0-based).
        end_frame (int | None): Last frame shown (included).
        stages (tuple[Stage, ...] | None): Any of ``STAGES``; None for every stage the run has.
        versions (tuple[str, ...]): ``"raw"`` and/or ``"averaged"``.
        overwrite (bool): Replace movies that already exist; otherwise they are refused.
        raw_tiff (Path | None): The acquired TIFF, if not where ``source_tiff.json`` says.

    Returns:
        list[Path]: The movies written.
    """
    plane_dir = find_plane_dir(
        settings.derivatives_root,
        cohort,
        subject,
        session,
        run,
        settings.variant,
        settings.save_folder,
    )
    variant_dir = plane_dir.parents[1]
    out_dir = Path(settings.output_dir) if settings.output_dir else variant_dir / MOVIE_FOLDER
    with ExitStack() as stack:
        jobs = open_jobs(
            settings,
            plane_dir,
            stages,
            versions,
            (whole_recording, start_frame, end_frame),
            raw_tiff,
            stack,
        )
        for job in jobs:
            if (out_dir / job.plan.name).exists() and not overwrite:
                raise FileExistsError(
                    f"{out_dir / job.plan.name} exists; pass overwrite=True to replace it"
                )
        out_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for job in jobs:
            written.append(_write_job(job, out_dir, settings))
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
