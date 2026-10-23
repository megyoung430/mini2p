"""Command-line entry points.

``mini2p-preprocess --config ... --input ... --output-dir ...`` runs phase correction,
distortion correction and registration; ``mini2p-extract --config ... --plane-dir ...
--seg ...`` extracts fluorescence from Cellpose masks into a Suite2p plane folder;
``mini2p-movie --config ...`` writes the movies of a run's preprocessing stages set in the
config's ``[movies]`` section (any setting can be overridden, e.g. ``--session 224``);
``mini2p-natex --config ... --plane-dir ...`` runs NATEX's calcium processing on a plane.
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="pipeline TOML file")
    parser.add_argument("--input", type=Path, required=True, help="raw interleaved TIFF")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="final output folder, e.g. RUN_DIR/without_destriping or RUN_DIR/with_destriping "
        "(inside RUN_DIR/without_distortion_correction/ when [distortion] mode is off)",
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
    "variant": "variant folder, e.g. with_destriping or without_distortion_correction/...",
    "whole_recording": "every frame (--no-whole-recording: start to end frame)",
    "start_frame": "first frame (0-based); implies --no-whole-recording",
    "end_frame": "last frame, included; implies --no-whole-recording",
    "stage_raw": "movie of the acquired TIFF",
    "stage_bidiphase": "movie after phase correction (and destriping)",
    "stage_undistorted": "movie after distortion correction",
    "stage_registered": "movie after motion registration",
    "unaveraged_movie": "every frame",
    "averaged_movie": "average_frames frames per movie frame",
    "derivatives_root": "folder holding cohort-XX/<subject>/ses-*/funcimg/run-XX",
    "raw_tiff": "the acquired TIFF, if not where source_tiff.json says",
    "output_dir": "where to write (default: run-XX/<variant>/movies)",
    "overwrite": "replace existing movies",
    "save_folder": "Suite2p output folder inside the variant folder",
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
    "suite2p_crop": "also a copy of each movie cropped to where Suite2p's shifts never wrap",
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


if __name__ == "__main__":
    main()
