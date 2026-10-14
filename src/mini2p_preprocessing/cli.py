"""Command-line entry points.

``mini2p-preprocess --config ... --input ... --output-dir ...`` runs phase correction,
distortion correction and registration; ``mini2p-extract --config ... --plane-dir ...
--seg ...`` extracts fluorescence from Cellpose masks into a Suite2p plane folder.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import load_settings, with_destriping
from .extraction import run_cellpose_extraction
from .pipeline import run_pipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="pipeline TOML file")
    parser.add_argument("--input", type=Path, required=True, help="raw interleaved TIFF")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="final output folder, including any without_destriping subfolder",
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


if __name__ == "__main__":
    main()
