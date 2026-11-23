"""Command-line entry points.

``mini2p-preprocess --config ... --input ... --output-dir ...`` runs phase correction,
distortion correction and registration; ``mini2p-extract --config ... --plane-dir ...
--seg ...`` extracts fluorescence from Cellpose masks into a Suite2p plane folder;
``mini2p-movie --config ...`` writes the movies of a run's preprocessing stages set in the
config's ``[movies]`` section (any setting can be overridden, e.g. ``--session 224``);
``mini2p-natex --config ... --plane-dir ...`` runs NATEX's calcium processing on a plane;
``mini2p-compare-registrations <aligned folder>`` compares the registrations of a sweep.
"""

from __future__ import annotations

import argparse
import logging
import types
import typing
from pathlib import Path

from .config import load_settings, with_destriping
from .extraction import run_cellpose_extraction
from .movies import MovieSettings, load_movie_settings, make_movies
from .natex import load_natex_settings, run_natex
from .pipeline import run_pipeline
from .raw_previews import plan_previews, run_previews
from .registration_comparison import (
    BLOCK_FRAMES,
    CELL_MINUTES,
    DETREND_S,
    N_BLOCKS,
    N_FRAMES,
    PATCH_PX,
    TABLE_NAME,
    compare_registrations,
    output_stems,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="pipeline TOML file")
    parser.add_argument("--input", type=Path, required=True, help="raw interleaved TIFF")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="final output folder, e.g. RUN_DIR/with_distortion_correction/without_destriping "
        "(RUN_DIR/without_distortion_correction/... when [distortion] mode is off)",
    )
    parser.add_argument(
        "--destripe", choices=["0", "1"], help="override destripe.enabled from the config"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    settings = load_settings(args.config)
    if args.destripe is not None:
        settings = with_destriping(settings, args.destripe == "1")
    run_pipeline(args.input, args.output_dir, settings)


def extract_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract traces from Cellpose masks")
    parser.add_argument("--config", type=Path, required=True, help="pipeline TOML file")
    parser.add_argument("--plane-dir", type=Path, required=True, help="suite2p/plane0 folder")
    parser.add_argument("--seg", type=Path, required=True, help="Cellpose *_seg.npy file")
    parser.add_argument(
        "--mask-image",
        type=Path,
        help="image the masks were drawn on, if from another registration",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    run_cellpose_extraction(args.seg, args.plane_dir, load_settings(args.config), args.mask_image)


# Every [movies] setting, with its command-line help; choices come from the settings model.
MOVIE_HELP = {
    "cohort": "cohort folder, e.g. cohort-01",
    "subject": "subject folder or part of it, e.g. sub-03 or MY_71_R",
    "session": "session number, e.g. 224",
    "run": "run number, e.g. 1 for run-01",
    "run_dir": "a run folder by path (laid out like run-XX/), instead of cohort ... run",
    "variant": "variant folder, e.g. with_distortion_correction/with_destriping",
    "alignment": "registration to use: auto (the one aligned_* folder), structural or functional",
    "whole_recording": "every frame (--no-whole-recording: start to end frame)",
    "start_frame": "first frame (0-based); implies --no-whole-recording",
    "end_frame": "last frame, included; implies --no-whole-recording",
    "stage_raw": "movie of the acquired TIFF",
    "stage_bidiphase": "movie after phase correction (and destriping)",
    "stage_undistorted": "movie after distortion correction",
    "stage_registered": "movie after motion registration",
    "stage_background_removed": "registered movie with the background regressed out",
    "background_source": "background of that movie: vessel or non_roi (outside the ROIs)",
    "unaveraged_movie": "every frame",
    "averaged_movie": "average_frames frames per movie frame",
    "behavior": "also each movie with the behavior video on the left, in sync",
    "derivatives_root": "folder holding cohort-XX/<subject>/ses-*/funcimg/run-XX",
    "raw_tiff": "the acquired TIFF, if not where source_tiff.json says",
    "behavior_video": "the behavior video, if not where video_frame_times.npz says",
    "output_dir": (
        "where to write (default: run-XX/<variant>/aligned_<channel>/<save_folder>/movies)"
    ),
    "overwrite": "replace existing movies",
    "save_folder": "Suite2p output folder inside the aligned_<channel> folder",
    "channel": "both side by side (functional left), or one alone",
    "average_frames": "frames per averaged movie frame (2-256)",
    "average_mode": "bin (non-overlapping) or rolling (running mean)",
    "speed": "playback speed, x real time (a movie must play at >= 1 fps)",
    "black_percentile": "percentile shown black",
    "white_percentile": "percentile shown white",
    "gamma": "display gamma",
    "spatial_sigma_px": "Gaussian smoothing of averaged frames (px)",
    "upscale": "pixels drawn upscale x upscale",
    "label": "stage, time, frames, speed and channel burnt in",
    "suite2p_crop": "also a copy of each registered movie cropped to where shifts never wrap",
    "contrast_sample_frames": "frames sampled for the display range",
    "quality": "H.264 quality, 0-10",
}


def _movie_option(name: str) -> dict:
    """argparse keywords for one MovieSettings field, from its type."""
    kind = MovieSettings.model_fields[name].annotation
    if typing.get_origin(kind) in (typing.Union, types.UnionType):  # X | None
        kind = next(a for a in typing.get_args(kind) if a is not type(None))
    if kind is bool:
        return {"action": argparse.BooleanOptionalAction}
    if typing.get_origin(kind) is typing.Literal:
        return {"choices": list(typing.get_args(kind))}
    return {"type": _named_path if kind is Path else kind}


def _named_path(text: str) -> Path:
    """A folder or file on the command line; empty (an unset shell variable) is refused."""
    if not text.strip():
        raise argparse.ArgumentTypeError("a path must be given, not left empty")
    return Path(text)


def movie_main(argv: list[str] | None = None) -> None:
    """Movies of a run's preprocessing stages, as set in [movies]; options override it."""
    parser = argparse.ArgumentParser(
        description="Movies of a run's preprocessing stages. Everything is set in the [movies] "
        "section of the config; each option below overrides one of its settings."
    )
    parser.add_argument("--config", type=Path, required=True, help="pipeline TOML file")
    for name, text in MOVIE_HELP.items():
        parser.add_argument(f"--{name.replace('_', '-')}", help=text, **_movie_option(name))
    args = parser.parse_args(argv)
    chosen = {k: getattr(args, k) for k in MOVIE_HELP if getattr(args, k) is not None}
    by_name = {"cohort", "subject", "session", "run", "derivatives_root"} & set(chosen)
    if by_name and "run_dir" in chosen:
        parser.error("Give --run-dir or the run by name (--cohort ... --run), not both")
    if by_name:  # the command line chooses the run by name: a run_dir in the config yields
        chosen["run_dir"] = None
    if {"start_frame", "end_frame"} & set(chosen):
        if chosen.get("whole_recording"):
            parser.error("--whole-recording and --start-frame/--end-frame contradict each other")
        chosen["whole_recording"] = False
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    try:
        settings = load_movie_settings(args.config, **chosen)
        # Only a run given by path without its own settings record needs the config's.
        written = make_movies(settings, lambda: load_settings(args.config))
    except (ValueError, FileNotFoundError, FileExistsError) as error:  # incl. ValidationError
        parser.error(str(error))
    for path in written:
        print(path)


def natex_main(argv: list[str] | None = None) -> None:
    """NATEX (MINI2P toolbox) on Suite2p plane folders: ΔF/F, transients, SNR, events."""
    parser = argparse.ArgumentParser(
        description="NATEX's calcium processing on a plane: ΔF/F, significant transients, SNR "
        "and cleaned events per ROI, saved with a QC PDF in <plane>/natex (settings: [natex])"
    )
    parser.add_argument("--config", type=Path, required=True, help="pipeline TOML file")
    parser.add_argument(
        "--plane-dir", type=Path, required=True, nargs="+", help="suite2p/plane0 folder(s)"
    )
    parser.add_argument("--output-dir", type=Path, help="instead of <plane>/natex (one plane)")
    parser.add_argument("--overwrite", action="store_true", help="replace earlier results")
    args = parser.parse_args(argv)
    if args.output_dir and len(args.plane_dir) > 1:
        parser.error("--output-dir takes one plane folder")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    try:
        settings = load_natex_settings(args.config)
        coefficient = load_settings(args.config).fluorescence.neuropil_coefficient
        for plane_dir in args.plane_dir:
            print(run_natex(plane_dir, settings, coefficient, args.output_dir, args.overwrite))
    except (ValueError, FileNotFoundError, FileExistsError) as error:  # incl. ValidationError
        parser.error(str(error))


def raw_means_main(argv: list[str] | None = None) -> None:
    """Raw mean images of the first frames of every raw TIFF of one or more subjects."""
    parser = argparse.ArgumentParser(
        description="Mean image of the first frames of each channel of every "
        "<subject>/ses-*/funcimg/*.tif, before any preprocessing, saved in "
        "<derivatives subject>/<session>/funcimg/preview-all-tifs/ with one overview PDF per "
        "subject; TIFFs already previewed there are not read again"
    )
    parser.add_argument("--config", type=Path, required=True, help="pipeline TOML ([channels])")
    parser.add_argument(
        "--subject-dir", type=_named_path, required=True, nargs="+", help="raw subject folder(s)"
    )
    parser.add_argument(
        "--derivatives-subject-dir", type=_named_path, nargs="+",
        help="derivatives folder of each subject (default: 'rawdata' replaced by 'derivatives')",
    )  # fmt: skip
    parser.add_argument("--frames", type=int, default=500, help="first frames to average")
    args = parser.parse_args(argv)
    if args.frames < 1:
        parser.error(f"--frames must be at least 1, got {args.frames}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    try:  # only the request is checked here; errors during the run raise as they are
        channels = load_settings(args.config).channels
        plan = plan_previews(
            args.subject_dir, args.derivatives_subject_dir, args.frames,
            channels.n_channels, channels.names,
        )  # fmt: skip
    except (ValueError, FileNotFoundError) as error:  # incl. ValidationError
        parser.error(str(error))
    rows = run_previews(plan, args.frames, channels.n_channels, channels.names)
    failed = [row for row in rows if row["status"] == "failed"]
    counts = {s: sum(r["status"] == s for r in rows) for s in ("written", "exists")}
    print(f"{len(rows) - len(failed)} of {len(rows)} TIFFs previewed "
          f"({counts['written']} new, {counts['exists']} already there)")  # fmt: skip
    for row in failed:
        print(f"FAILED {row['tiff']}: {row['error']}")
    if failed:
        raise SystemExit(1)


def compare_registrations_main(argv: list[str] | None = None) -> None:
    """Alignment and signal quality of every registration of one recording, side by side."""
    parser = argparse.ArgumentParser(
        description="Measure every <save_folder>/plane0 in an aligned_<channel> folder the "
        "same way (alignment, and signal quality without cell segmentation) and save a "
        f"table and figures in <aligned folder>/registration_comparison/{TABLE_NAME}.*"
    )
    parser.add_argument("aligned_folder", type=Path, help="aligned_<channel> folder")
    parser.add_argument("--frames", type=int, default=N_FRAMES,
                        help="evenly spaced frames for the alignment measures")  # fmt: skip
    parser.add_argument("--blocks", type=int, default=N_BLOCKS,
                        help="contiguous blocks for the signal measures")  # fmt: skip
    parser.add_argument("--block-frames", type=int, default=BLOCK_FRAMES,
                        help="frames per block (shorter if the blocks do not fit)")  # fmt: skip
    parser.add_argument("--patch-px", type=int, default=PATCH_PX,
                        help="side of the pixel patches the signal measures use")  # fmt: skip
    parser.add_argument("--detrend-s", type=float, default=DETREND_S,
                        help="high-pass window of the signal measures (s)")  # fmt: skip
    parser.add_argument("--cell-minutes", type=float, default=CELL_MINUTES,
                        help="first minutes for the putative-cell SNR and transients "
                        "(0: the whole recording)")  # fmt: skip
    parser.add_argument("--scan-px", type=int, default=512,
                        help="scan width in px (sets the putative-cell geometry)")  # fmt: skip
    parser.add_argument("--cells", type=int, default=200, help="putative cells per registration")
    parser.add_argument("--overwrite", action="store_true", help="replace an earlier comparison")
    args = parser.parse_args(argv)
    if args.cell_minutes < 0:
        parser.error("--cell-minutes must be 0 (whole recording) or positive")
    for name in ("frames", "blocks", "block_frames", "patch_px", "scan_px", "cells"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if args.detrend_s <= 0:
        parser.error("--detrend-s must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    try:
        rows = compare_registrations(
            args.aligned_folder, args.frames, args.blocks, args.block_frames, args.patch_px,
            args.detrend_s, args.overwrite, args.cell_minutes or None, args.scan_px,
            args.cells,
        )  # fmt: skip
    except (ValueError, KeyError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))
    print_summary(rows)
    print(f"Saved {output_stems(args.aligned_folder)[0].parent}")


SUMMARY_COLUMNS = [
    "registration", "red_corr_median", "red_residual_shift_px", "red_fluctuation_pct",
    "green_local_corr_contrast", "green_noise_level", "green_cell_resting_snr_median",
    "green_cell_transients_per_min", "green_cell_transients_flipped_per_min",
    "red_cell_transients_per_min",
]  # fmt: skip


def print_summary(rows: list[dict]) -> None:
    """The main measures, one registration per line."""
    first = max(len(row["registration"]) for row in rows + [{"registration": "registration"}])
    widths = [max(len(column), 10) for column in SUMMARY_COLUMNS[1:]]
    header = [f"{'registration':<{first}}"]
    print(
        "  ".join(header + [f"{c:>{w}}" for c, w in zip(SUMMARY_COLUMNS[1:], widths, strict=True)])
    )
    for row in rows:
        values = [f"{row[c]:>{w}.4g}" for c, w in zip(SUMMARY_COLUMNS[1:], widths, strict=True)]
        print("  ".join([f"{row['registration']:<{first}}"] + values))


if __name__ == "__main__":
    main()
