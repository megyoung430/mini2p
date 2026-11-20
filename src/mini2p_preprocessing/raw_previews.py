"""Raw mean images of every acquired TIFF of a subject, before any preprocessing.

For deciding which recordings are worth preprocessing: the mean of the first frames of
each channel of every ``ses-*/funcimg/*.tif``, read straight from the acquired TIFF (no
phase correction, destriping, distortion correction or registration).

Each TIFF's figure (png, pdf, svg and a caption, named after the TIFF) goes in its own
folder in the derivatives tree, ``<derivatives subject>/<session>/funcimg/preview-all-tifs/``,
apart from the run folders, so nothing the batch or the QC notebook wrote is touched. Each
subject also gets one PDF with a page per TIFF, in session order and, within a session, in
order of acquisition start (ScanImage's ``epoch`` in the TIFF header).

A TIFF whose preview is already in that folder (a caption naming it, with the same file
size and frame count, and the figure files beside it) is not read again.
"""

from __future__ import annotations

import glob
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.backends.backend_pdf import PdfPages

from . import plots
from .batch import SESSION_PATTERN
from .plots import QC_FONTS
from .progress import ProgressLog
from .qc import raw_mean_images

log = logging.getLogger(__name__)

PREVIEW_FOLDER = "preview-all-tifs"
PREVIEW_SUFFIX = "_raw_meanImg_first"
SUMMARY_PREFIX = "raw_mean_previews_first"  # + <frames>.json
DISPLAY_PERCENTILES = (1, 99.8)
FIGURE_SUFFIXES = (".png", ".pdf", ".svg")
# ScanImage writes the acquisition's start time in every page header:
# "epoch = [2026  8 13 15 27 33.497]" (year month day hour minute seconds)
EPOCH = re.compile(r"^epoch = \[\s*([\d.\s]+?)\s*\]\s*$", re.MULTILINE)
FRAMES_IN_CAPTION = re.compile(r"Mean of the first (\d+) frames")
SIZE_IN_CAPTION = re.compile(r"Source file size: (\d+) bytes\.")


@dataclass(frozen=True)
class RawTiff:
    session: str
    path: Path
    started: datetime
    started_from: str  # "ScanImage epoch" or "file modification time"


def parse_epoch(header: str) -> datetime | None:
    """ScanImage's acquisition start from a page header, or None when it has no ``epoch``.

    Raises:
        ValueError: If the ``epoch`` line does not hold year, month, day, hour, minute and
            seconds.
    """
    match = EPOCH.search(header)
    if not match:
        return None
    values = match.group(1).split()
    if len(values) != 6:
        raise ValueError(f"ScanImage epoch {values} does not hold 6 values")
    year, month, day, hour, minute, seconds = (float(v) for v in values)
    return datetime(int(year), int(month), int(day), int(hour), int(minute)) + timedelta(
        seconds=seconds
    )


def first_header(path: Path) -> str | None:
    """The first page's header, or None for a file tifffile cannot read (its preview then
    fails and is reported)."""
    try:
        with tifffile.TiffFile(path) as tif:
            return tif.pages[0].aspage().description or ""
    except Exception as error:  # a corrupt TIFF makes tifffile raise many different types
        log.warning("%s: header unreadable (%s)", path, error)
        return None


def acquisition_start(path: Path) -> tuple[datetime, str]:
    """When the acquisition started: ScanImage's ``epoch`` in the first page header, or the
    file's modification time for a TIFF without one or with an unreadable header (noted in
    the second value)."""
    header = first_header(path)
    started = parse_epoch(header) if header is not None else None
    if started is not None:
        return started, "ScanImage epoch"
    return datetime.fromtimestamp(path.stat().st_mtime), "file modification time"


def _tiffs_in(folder: Path) -> list[Path]:
    """The ``.tif`` files of a folder, without the ``._`` files macOS leaves on network
    shares."""
    if not folder.is_dir():
        return []
    return [
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() == ".tif" and not p.name.startswith("._")
    ]


def session_tiffs(session_dir: Path) -> list[RawTiff]:
    """The TIFFs in ``<session>/funcimg``, in order of acquisition start (ties by name)."""
    started = [(acquisition_start(path), path) for path in _tiffs_in(session_dir / "funcimg")]
    ordered = sorted(started, key=lambda item: (item[0][0], item[1].name))
    by_file_time = [path.name for (_, source), path in ordered if source != "ScanImage epoch"]
    if by_file_time:
        log.warning(
            "%s: %s have no ScanImage start time and are ordered by their file time, which "
            "on a share may be when they were copied",
            session_dir, ", ".join(by_file_time),
        )  # fmt: skip
    stems = [path.stem for _, path in ordered]
    if len(set(stems)) != len(stems):
        raise ValueError(f"{session_dir}: two TIFFs share a name apart from the extension's case")
    return [RawTiff(session_dir.name, path, when, source) for (when, source), path in ordered]


def find_raw_tiffs(subject_dir: Path) -> list[RawTiff]:
    """Every raw TIFF of a subject, sessions in number order.

    Raises:
        FileNotFoundError: If the folder holds no ``ses-*/funcimg/*.tif``.
    """
    subject_dir = Path(subject_dir)
    sessions = sorted(
        (int(match.group(1)), folder)
        for folder in subject_dir.iterdir()
        if folder.is_dir() and (match := SESSION_PATTERN.match(folder.name))
    )
    found = [raw for _, folder in sessions for raw in session_tiffs(folder)]
    if not found:
        raise FileNotFoundError(f"No raw TIFFs in {subject_dir}/ses-*/funcimg")
    return found


def derivatives_subject_dir(raw_subject_dir: Path) -> Path:
    """The raw subject folder with its ``rawdata`` part replaced by ``derivatives``.

    Raises:
        ValueError: If the path has no ``rawdata`` part (give the folder explicitly).
    """
    parts = Path(raw_subject_dir).parts
    if parts.count("rawdata") != 1:
        raise ValueError(
            f"Cannot tell the derivatives folder of {raw_subject_dir}: it needs exactly one "
            "'rawdata' part; give the derivatives subject folder explicitly"
        )
    return Path(*("derivatives" if part == "rawdata" else part for part in parts))


def preview_folder(derivatives_dir: Path, raw: RawTiff) -> Path:
    """``<derivatives subject>/<session>/funcimg/preview-all-tifs``."""
    return Path(derivatives_dir) / raw.session / "funcimg" / PREVIEW_FOLDER


def preview_stem(folder: Path, path: Path, n_used: int) -> Path:
    """``<folder>/<tiff stem>_raw_meanImg_first<n_used>`` (extensions added to it)."""
    return Path(folder) / f"{Path(path).stem}{PREVIEW_SUFFIX}{n_used}"


def tiff_label(path: Path) -> str:
    """``<session>/funcimg/<name>``: names a raw TIFF the same way on every mount."""
    return "/".join(Path(path).parts[-3:])


def existing_preview(folder: Path, path: Path, n_frames: int) -> Path | None:
    """This TIFF's preview already in *folder*: a caption naming it (by session, folder and
    file name) with its current file size, averaged over ``n_frames`` frames or all of a
    shorter recording, with every figure file beside it. Returns its stem, or None."""
    path = Path(path)
    pattern = f"{glob.escape(path.stem)}{PREVIEW_SUFFIX}*.caption.txt"
    for caption in sorted(Path(folder).glob(pattern)):
        text = caption.read_text()
        frames, size = FRAMES_IN_CAPTION.search(text), SIZE_IN_CAPTION.search(text)
        if not frames or not size or tiff_label(path) not in text:
            continue
        if int(size.group(1)) != path.stat().st_size:
            continue  # previewed from another version of the file
        n_used = int(frames.group(1))
        shorter = f"The recording has only {n_used} whole frames." in text
        stem = caption.parent / caption.name.removesuffix(".caption.txt")
        if (n_used == n_frames or (n_used < n_frames and shorter)) and all(
            with_extension(stem, extension).exists() for extension in FIGURE_SUFFIXES
        ):
            return stem
    return None


def preview_caption(path: Path, n_used: int, n_asked: int, size: int | None = None) -> str:
    """The caption of a preview; it names the TIFF, how many frames were averaged and the
    file's *size* (taken before reading, so a file still being copied is previewed again;
    default: its size now), which is how ``existing_preview`` recognizes it."""
    size = Path(path).stat().st_size if size is None else size
    low, high = DISPLAY_PERCENTILES
    shorter = f" The recording has only {n_used} whole frames." if n_used < n_asked else ""
    return (
        f"Mean of the first {n_used} frames of each channel of {path}, read straight from the "
        "acquired TIFF before any phase correction, destriping, distortion correction or "
        f"registration.{shorter} Each channel is shown from its percentile {low} to its "
        f"percentile {high}. Values are ScanImage counts after its start-of-acquisition "
        "offset subtraction, so the dark level is not zero and drifts during the recording; "
        f"they are not photon counts. Source file size: {size} bytes.\n"
    )


def draw_preview(
    means: list[np.ndarray], n_used: int, tiff_name: str, names: dict[int, str]
) -> plt.Figure:
    """Each channel's raw mean image, titled by the TIFF's file name."""
    axes = plots.plot_channel_columns(
        [means],
        [f"Raw Mean, First {n_used} Frames, No Preprocessing"],
        names,
        percentiles=DISPLAY_PERCENTILES,
    )
    figure = axes[0, 0].figure
    figure.suptitle(tiff_name, fontsize=12)
    return figure


def with_extension(stem: Path, extension: str) -> Path:
    """*stem* plus *extension*; unlike ``Path.with_suffix`` it keeps dots in the name."""
    return stem.parent / f"{stem.name}{extension}"


def save_preview(figure: plt.Figure, stem: Path, caption: str, dpi: int = 150) -> None:
    """The figure as png, pdf and svg, then its caption (written last: an interrupted save
    has no caption and is made again)."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in FIGURE_SUFFIXES:
        figure.savefig(with_extension(stem, extension), dpi=dpi, bbox_inches="tight")
    with_extension(stem, ".caption.txt").write_text(caption)


def preview_one(
    raw: RawTiff, folder: Path, n_frames: int, n_channels: int, names: dict[int, str]
) -> dict:
    """The preview of one TIFF in *folder*, unless it is already there. Returns its summary
    row (status "written", "exists", or "failed" for a TIFF that cannot be read)."""
    row = {"tiff": str(raw.path), "session": raw.session,
           "started": raw.started.isoformat(), "started_from": raw.started_from}  # fmt: skip
    stem = existing_preview(folder, raw.path, n_frames)
    if stem is not None:
        return {**row, "status": "exists", "figure": str(with_extension(stem, ".png"))}
    try:
        size = raw.path.stat().st_size
        means, n_used = raw_mean_images(raw.path, n_frames, n_channels)
    except Exception as error:  # a corrupt TIFF makes tifffile raise many different types
        log.error("%s: %s: %s", raw.path, type(error).__name__, error)
        return {**row, "status": "failed", "error": f"{type(error).__name__}: {error}"}
    stem = preview_stem(folder, raw.path, n_used)
    figure = draw_preview(means, n_used, raw.path.name, names)
    try:
        save_preview(figure, stem, preview_caption(raw.path, n_used, n_frames, size))
    finally:
        plt.close(figure)
    return {**row, "status": "written", "figure": str(with_extension(stem, ".png"))}


def write_overview(rows: list[dict], pdf_path: Path) -> Path:
    """One PDF page per preview figure (by its png), labelled with session and file."""
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        for row in rows:
            if "figure" not in row:
                continue
            figure, ax = plt.subplots(figsize=(8.27, 11.69))
            try:
                ax.imshow(plt.imread(row["figure"]))
                ax.set_axis_off()
                ax.set_title(f"{row['session']}   {Path(row['tiff']).name}", fontsize=11)
                pdf.savefig(figure)
            finally:
                plt.close(figure)
    return pdf_path


def plan_previews(
    subject_dirs: list[Path],
    derivatives_dirs: list[Path] | None,
    n_frames: int,
    n_channels: int,
    names: dict[int, str],
) -> list[tuple[Path, list[RawTiff]]]:
    """Each subject's derivatives folder and raw TIFFs, after checking the request.

    *derivatives_dirs* gives each subject's derivatives folder, or None to take
    ``derivatives_subject_dir`` of each raw folder.

    Raises:
        ValueError: If *n_frames* is below 1, *names* does not name every channel, or the
            derivatives folders do not pair up with the subjects.
        FileNotFoundError: If a subject folder holds no raw TIFF.
    """
    if n_frames < 1:
        raise ValueError(f"Ask for at least one frame, got {n_frames}")
    if set(names) != set(range(1, n_channels + 1)):
        raise ValueError(f"Channel names {sorted(names)} do not cover channels 1-{n_channels}")
    if derivatives_dirs is None:
        derivatives_dirs = [derivatives_subject_dir(d) for d in subject_dirs]
    if len(derivatives_dirs) != len(subject_dirs):
        raise ValueError("Give one derivatives folder per raw subject folder")
    for raw, derived in zip(subject_dirs, derivatives_dirs, strict=True):
        if "rawdata" in Path(derived).parts or Path(derived).name != Path(raw).name:
            raise ValueError(
                f"{derived} is not the derivatives folder of {raw}: it must be outside "
                "rawdata and end in the same subject folder"
            )
    return [
        (Path(d), find_raw_tiffs(r)) for r, d in zip(subject_dirs, derivatives_dirs, strict=True)
    ]


def run_previews(
    plan: list[tuple[Path, list[RawTiff]]],
    n_frames: int,
    n_channels: int,
    names: dict[int, str],
    report=log.info,
) -> list[dict]:
    """Previews of every planned TIFF, in its session's preview folder; per subject also an
    overview PDF and a JSON summary in the derivatives subject folder
    (``preview-all-tifs_raw_meanImg_first<n>.pdf``, ``raw_mean_previews.json``).

    Returns:
        One summary row per TIFF, in order; status "written", "exists" or "failed".
    """
    progress = ProgressLog("Raw mean previews", sum(len(raws) for _, raws in plan), report)
    rows = []
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": QC_FONTS}):
        for derivatives_dir, raws in plan:
            rows += preview_subject(raws, derivatives_dir, n_frames, n_channels, names, progress)
    return rows


def write_raw_previews(
    subject_dirs: list[Path],
    derivatives_dirs: list[Path] | None,
    n_frames: int,
    n_channels: int,
    names: dict[int, str],
    report=log.info,
) -> list[dict]:
    """``plan_previews`` then ``run_previews``."""
    plan = plan_previews(subject_dirs, derivatives_dirs, n_frames, n_channels, names)
    return run_previews(plan, n_frames, n_channels, names, report)


def preview_subject(
    raws: list[RawTiff],
    derivatives_dir: Path,
    n_frames: int,
    n_channels: int,
    names: dict[int, str],
    progress: ProgressLog,
) -> list[dict]:
    """Previews of one subject's TIFFs, then its overview PDF and summary. The summary is
    written also when an error stops the run, marked incomplete."""
    rows, complete = [], False
    try:
        for raw in raws:
            row = preview_one(
                raw, preview_folder(derivatives_dir, raw), n_frames, n_channels, names
            )
            rows.append(row)
            progress.step_done(succeeded=row["status"] in ("written", "exists"))
        write_overview(rows, derivatives_dir / f"{PREVIEW_FOLDER}{PREVIEW_SUFFIX}{n_frames}.pdf")
        complete = True
    finally:
        derivatives_dir.mkdir(parents=True, exist_ok=True)
        summary = {"n_frames": n_frames, "complete": complete, "tiffs": rows}
        summary_path = derivatives_dir / f"{SUMMARY_PREFIX}{n_frames}.json"
        summary_path.write_text(json.dumps(summary, indent=2))
    return rows
