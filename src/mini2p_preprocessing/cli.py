"""Command-line entry points.

``mini2p-preprocess --config ... --input ... --output-dir ...`` runs phase correction,
distortion correction and registration; ``mini2p-extract --config ... --plane-dir ...
--seg ...`` extracts fluorescence from Cellpose masks into a Suite2p plane folder;
``mini2p-movie --config ... --cohort ... --subject ... --session ... --run ...`` writes the
raw and averaged movies of a registered run, and ``mini2p-stage-movies`` (same options) those
of every preprocessing stage.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import load_settings, with_destriping
from .extraction import run_cellpose_extraction
from .movies import STAGES, load_movie_settings, make_movies
from .pipeline import run_pipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="pipeline TOML file")
    parser.add_argument("--input", type=Path, required=True, help="raw interleaved TIFF")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="final output folder, e.g. RUN_DIR/without_destriping or RUN_DIR/with_destriping",
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


def _movie_command(
    argv: list[str] | None, default_stages: tuple[str, ...] | None, description: str
):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, required=True, help="pipeline TOML file")
    parser.add_argument("--cohort", required=True, help="e.g. cohort-01")
    parser.add_argument("--subject", required=True, help="e.g. sub-03 or MY_71_R")
    parser.add_argument("--session", type=int, required=True, help="e.g. 224")
    parser.add_argument("--run", type=int, required=True, help="e.g. 1 for run-01")
    span = parser.add_mutually_exclusive_group(required=True)
    span.add_argument("--whole-recording", action="store_true", help="every frame of the run")
    span.add_argument("--start-frame", type=int, help="first frame (0-based); needs --end-frame")
    parser.add_argument("--end-frame", type=int, help="last frame, included")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGES,
        default=default_stages,
        help="preprocessing stages to show (default: "
        + (" ".join(default_stages) if default_stages else "every stage the run has")
        + ")",
    )
    parser.add_argument("--raw-tiff", type=Path, help="acquired TIFF, if no source_tiff.json")
    parser.add_argument("--only", choices=["raw", "averaged"], help="write just one version")
    parser.add_argument("--overwrite", action="store_true", help="replace existing movies")
    overrides = {
        "variant": dict(choices=["without_destriping", "with_destriping"]),
        "channel": dict(choices=["side_by_side", "green", "red"]),
        "average_frames": dict(type=int),
        "average_mode": dict(choices=["bin", "rolling"]),
        "speed": dict(type=float),
        "black_percentile": dict(type=float),
        "white_percentile": dict(type=float),
        "gamma": dict(type=float),
        "spatial_sigma_px": dict(type=float),
        "upscale": dict(type=int),
        "output_dir": dict(type=Path),
    }
    for name, kwargs in overrides.items():
        parser.add_argument(
            f"--{name.replace('_', '-')}", help=f"override [movies] {name}", **kwargs
        )
    args = parser.parse_args(argv)
    if (args.start_frame is None) != (args.end_frame is None):
        parser.error("--start-frame and --end-frame go together")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    chosen = {k: getattr(args, k) for k in overrides if getattr(args, k) is not None}
    stages = tuple(args.stages) if args.stages else None
    try:
        settings = load_movie_settings(args.config, **chosen)
        written = make_movies(
            settings,
            args.cohort,
            args.subject,
            args.session,
            args.run,
            whole_recording=args.whole_recording,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            stages=stages,
            versions=(args.only,) if args.only else ("raw", "averaged"),
            overwrite=args.overwrite,
            raw_tiff=args.raw_tiff,
        )
    except (ValueError, FileNotFoundError, FileExistsError) as error:  # incl. ValidationError
        parser.error(str(error))
    for path in written:
        print(path)


def movie_main(argv: list[str] | None = None) -> None:
    """Raw and averaged movies of a run; by default its registered frames."""
    _movie_command(
        argv, ("registered",), "Raw and averaged movies of a run (settings from [movies])"
    )


def stage_movies_main(argv: list[str] | None = None) -> None:
    """Raw and averaged movies of every preprocessing stage of a run."""
    _movie_command(
        argv,
        None,
        "Movies of each preprocessing stage of a run: raw TIFF, bidiphase, + distortion "
        "correction, + registration (settings from [movies])",
    )


if __name__ == "__main__":
    main()
