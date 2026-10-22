"""Movies of a registered run: the whole recording or a range of frames, raw and averaged.

Frames come from Suite2p's registered binaries (``data.bin``, ``data_chan2.bin``), so the
movie shows the motion-corrected data the traces were extracted from. Each run gives two
MP4s (H.264, yuv420p, which PowerPoint, Keynote and QuickTime play):

- **raw**: every registered frame, unsmoothed;
- **averaged**: each movie frame is the mean of ``average_frames`` consecutive frames, either
  in non-overlapping bins (``"bin"``: fewer frames, each covering a longer time) or as a
  running mean (``"rolling"``: one movie frame per acquired frame, smoothed in time).

Contrast is fixed for the whole movie (percentiles of a sample of its own frames), so
brightness changes over the recording stay visible instead of being normalised away. A
label gives the time from the start of the run (frame index / frame rate) and the playback
speed. The run is found from cohort, subject, session and run number under the
derivatives folder set in the ``[movies]`` section of the config.

Frame numbers are 0-based, like Suite2p's, and ``end_frame`` is included.
"""

from __future__ import annotations

import logging
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.ndimage import gaussian_filter

from .batch import SESSION_PATTERN
from .config import VARIANT_SUBDIRS
from .progress import ProgressLog
from .registration import resolve_registered_binary

log = logging.getLogger(__name__)

Channel = Literal["green", "red", "merge"]
AverageMode = Literal["bin", "rolling"]
BINARIES = {"green": ("reg_file", "data.bin"), "red": ("reg_file_chan2", "data_chan2.bin")}
MOVIE_FOLDER = "movies"
MIN_FPS = 1.0  # imageio-ffmpeg rounds the rate to 0.01 fps
SOURCE_FRAMES_PER_BLOCK = 256  # acquired frames held in memory per channel at a time


class MovieSettings(BaseModel):
    """The ``[movies]`` section of the pipeline config; arguments to ``make_movies`` override it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    derivatives_root: Path  # holds cohort-XX/<subject>/ses-*/funcimg/run-XX
    variant: Literal["without_destriping", "with_destriping"] = "without_destriping"
    channel: Channel = "green"  # "merge": green in green, red in magenta
    average_frames: int = Field(5, ge=2, le=256)
    average_mode: AverageMode = "bin"
    # Playback speed, times real time. A movie must play at >= MIN_FPS (1 fps), so with
    # bins of many frames raise speed: at 7.5 Hz, bin 10 needs speed >= 1.4.
    speed: float = Field(4.0, gt=0)
    low_percentile: float = Field(0.5, ge=0, le=100)
    high_percentile: float = Field(99.8, ge=0, le=100)
    # Gaussian smoothing within each averaged frame; 0 = none. The raw movie is never smoothed.
    spatial_sigma_px: float = Field(0.0, ge=0)
    upscale: int = Field(1, ge=1, le=8)  # each pixel drawn as upscale x upscale pixels
    label: bool = True  # time and speed burnt into the frame
    contrast_sample_frames: int = Field(200, ge=1)
    quality: float = Field(8.0, ge=0, le=10)  # H.264 quality, 0-10
    output_dir: Path | None = None  # None: run-XX/<variant>/movies
    save_folder: str = "suite2p"  # Suite2p output folder inside the variant folder


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


def contrast_limits(sample: np.ndarray, low: float, high: float) -> tuple[float, float]:
    """Display range from percentiles of *sample*; never empty."""
    lo, hi = (float(v) for v in np.nanpercentile(sample, [low, high]))
    return lo, hi if hi > lo else lo + 1.0


def to_uint8(frames: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (np.clip((frames - lo) / (hi - lo), 0, 1) * 255).round().astype(np.uint8)


def compose_rgb(green: np.ndarray | None, red: np.ndarray | None) -> np.ndarray:
    """``(..., 3)`` uint8: one channel in grey, or green in green and red in magenta."""
    if green is not None and red is not None:
        return np.stack([red, green, red], axis=-1)
    single = green if green is not None else red
    return np.repeat(single[..., None], 3, axis=-1)


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

    name: str
    n: int
    starts: np.ndarray
    fps: float
    sigma_px: float
    first_frame: int  # first and last source frames actually shown
    last_frame: int


def plan_movie(
    version: str,
    span: tuple[int, int],
    fs: float,
    settings: MovieSettings,
    names: tuple[str, str, str],
) -> MoviePlan:
    """The frames, playback rate and file name of the raw or averaged movie (pure).

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
        settings.channel,
        f"frames-{first}-{last}",
        "raw" if raw else f"{settings.average_mode}{n}",
    ]
    if sigma > 0:
        parts.append(f"sigma{sigma:g}")
    parts.append(f"{settings.speed:g}x")
    return MoviePlan("_".join(parts) + ".mp4", n, starts, fps, sigma, first, last)


def block_size(n: int, mode: AverageMode, source_frames: int = SOURCE_FRAMES_PER_BLOCK) -> int:
    """Movie frames per read so that a read holds about *source_frames* acquired frames."""
    # rolling: at least n movie frames per block, so each acquired frame is read at most twice
    return max(1, source_frames // n) if mode == "bin" else max(n, source_frames - n + 1)


# ---------------------------------------------------------------- reading and writing


def open_channels(plane_dir: Path, ops: dict, channel: Channel) -> dict[str, np.memmap]:
    """Memory-mapped registered binaries of the channels *channel* needs."""
    names = ("green", "red") if channel == "merge" else (channel,)
    shape = (int(ops["nframes"]), int(ops["Ly"]), int(ops["Lx"]))
    opened = {}
    for name in names:
        key, filename = BINARIES[name]
        path = resolve_registered_binary(ops, plane_dir, key, filename)
        expected = int(np.prod(shape)) * 2
        if path.stat().st_size != expected:
            raise ValueError(
                f"{path} is {path.stat().st_size} bytes, not {expected} for {shape} int16 frames; "
                "is the registration complete?"
            )
        opened[name] = np.memmap(path, dtype=np.int16, mode="r", shape=shape)
    return opened


def _windows(source: np.memmap, first: np.ndarray, n: int, sigma: float) -> np.ndarray:
    frames = np.asarray(source[first[0] : first[-1] + n], dtype=np.float32)
    return smooth(average_windows(frames, first, n), sigma)


def channel_limits(
    sources: dict[str, np.memmap], plan: MoviePlan, settings: MovieSettings
) -> dict[str, tuple[float, float]]:
    """Each channel's display range, from up to ``contrast_sample_frames`` of this movie's frames
    (every 4th pixel of each, to keep the sample small)."""
    count = settings.contrast_sample_frames
    picks = np.unique(np.linspace(0, len(plan.starts) - 1, count).round()).astype(int)
    limits = {}
    for name, source in sources.items():
        sample = [
            _windows(source, plan.starts[[i]], plan.n, plan.sigma_px)[0, ::4, ::4] for i in picks
        ]
        limits[name] = contrast_limits(
            np.stack(sample), settings.low_percentile, settings.high_percentile
        )
    return limits


def rendered_frames(
    sources: dict[str, np.memmap],
    plan: MoviePlan,
    limits: dict[str, tuple[float, float]],
    settings: MovieSettings,
    fs: float,
    block: int | None = None,
) -> Iterator[np.ndarray]:
    """RGB uint8 movie frames, read *block* movie frames at a time (default: ``block_size``)."""
    block = block or block_size(plan.n, settings.average_mode)
    for b in range(0, len(plan.starts), block):
        first = plan.starts[b : b + block]
        scaled = {
            name: to_uint8(_windows(source, first, plan.n, plan.sigma_px), *limits[name])
            for name, source in sources.items()
        }
        rgb = compose_rgb(scaled.get("green"), scaled.get("red"))
        for frame, f in zip(rgb, first, strict=True):
            frame = upscale_and_pad(frame, settings.upscale)
            if settings.label:
                frame = _draw_label(frame, frame_label(int(f), plan.n, fs, settings.speed))
            yield frame


def _draw_label(rgb: np.ndarray, text: str) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    size = max(10, rgb.shape[0] // 36)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    draw.text(
        (size // 2, size // 2),
        text,
        fill=(255, 255, 255),
        font=ImageFont.load_default(size),
        stroke_width=max(1, size // 10),
        stroke_fill=(0, 0, 0),
    )
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


def make_movies(
    settings: MovieSettings,
    cohort: str,
    subject: str,
    session: int,
    run: int,
    whole_recording: bool = True,
    start_frame: int | None = None,
    end_frame: int | None = None,
    versions: tuple[str, ...] = ("raw", "averaged"),
    overwrite: bool = False,
) -> list[Path]:
    """Write the raw and/or averaged movie of one run, and return their paths.

    Each movie gets its own display range, so the raw and averaged movies are not on the same
    brightness scale (averaging narrows the range of values).

    Args:
        settings (MovieSettings): From ``load_movie_settings`` (with any overrides).
        cohort (str): e.g. ``"cohort-01"``.
        subject (str): The subject folder or part of it, e.g. ``"sub-03"`` or ``"MY_71_R"``.
        session (int): Session number, e.g. ``224``.
        run (int): Run number, e.g. ``1`` for ``run-01``.
        whole_recording (bool): Every frame of the run; otherwise give the two frames below.
        start_frame (int | None): First frame shown (0-based).
        end_frame (int | None): Last frame shown (included).
        versions (tuple[str, ...]): ``"raw"`` and/or ``"averaged"``.
        overwrite (bool): Replace movies that already exist; otherwise they are refused.

    Returns:
        list[Path]: The movies written.
    """
    unknown = set(versions) - {"raw", "averaged"}
    if unknown or not versions:
        raise ValueError(f"versions must be 'raw' and/or 'averaged', not {versions}")
    plane_dir = find_plane_dir(
        settings.derivatives_root,
        cohort,
        subject,
        session,
        run,
        settings.variant,
        settings.save_folder,
    )
    ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
    fs = float(ops["fs"])
    span = frame_span(int(ops["nframes"]), whole_recording, start_frame, end_frame)
    variant_dir = plane_dir.parents[1]
    names = (variant_dir.parents[3].name, variant_dir.parents[2].name, variant_dir.parent.name)
    plans = [plan_movie(version, span, fs, settings, names) for version in versions]
    out_dir = Path(settings.output_dir) if settings.output_dir else variant_dir / MOVIE_FOLDER
    for plan in plans:
        if (out_dir / plan.name).exists() and not overwrite:
            raise FileExistsError(
                f"{out_dir / plan.name} exists; pass overwrite=True to replace it"
            )
    sources = open_channels(plane_dir, ops, settings.channel)

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for plan in plans:
        if (plan.first_frame, plan.last_frame + 1) != span:
            log.info(
                "%s: frames %d-%d requested; %d-%d shown (whole windows of %d only)",
                plan.name,
                span[0],
                span[1] - 1,
                plan.first_frame,
                plan.last_frame,
                plan.n,
            )
        limits = channel_limits(sources, plan, settings)
        log.info(
            "%s: %d movie frames at %.3g fps; display ranges %s",
            plan.name,
            len(plan.starts),
            plan.fps,
            limits,
        )
        path = out_dir / plan.name
        frames = rendered_frames(sources, plan, limits, settings, fs)
        write_mp4(path, frames, len(plan.starts), plan.fps, settings.quality)
        written.append(path)
    return written
