"""Reading the repo and the hand-written inputs of the code map (the I/O edge)."""

import json
import subprocess
import tomllib
from pathlib import Path

from src.code_map.assemble import Overviews, overviews_from_toml, parse_descriptions
from src.code_map.explanations import parse_explanations
from src.code_map.extract import ModuleRecord, NotebookRecord, module_record, notebook_record

SOURCE_FOLDERS = ("src", "scripts", "notebooks")


def skipped(path: Path, root: Path) -> bool:
    """True for files inside caches and hidden folders (``__pycache__``, ``.ipynb_checkpoints``)."""
    parts = path.relative_to(root).parts[:-1]
    return any(p == "__pycache__" or p.startswith(".") for p in parts)


def source_paths(root: Path) -> tuple[list[Path], list[Path]]:
    """The Python files and notebooks under the source folders, sorted by path."""
    root = Path(root)
    python, notebooks = [], []
    for folder in (f for f in SOURCE_FOLDERS if (root / f).is_dir()):
        python += [p for p in (root / folder).rglob("*.py") if not skipped(p, root)]
        notebooks += [p for p in (root / folder).rglob("*.ipynb") if not skipped(p, root)]
    return sorted(python), sorted(notebooks)


def read_records(root: Path) -> tuple[list[ModuleRecord], list[NotebookRecord]]:
    """Every Python file and notebook of the repo at *root*, read and parsed."""
    root = Path(root)
    python, notebooks = source_paths(root)

    def name(p: Path) -> str:
        return p.relative_to(root).as_posix()

    modules = [module_record(name(p), p.read_text(encoding="utf-8")) for p in python]
    books = [notebook_record(name(p), json.loads(p.read_text(encoding="utf-8"))) for p in notebooks]
    return modules, books


def load_overviews(path: Path) -> Overviews:
    """The hand-written page text from ``overviews.toml``, with each topic's schematic read from
    the SVG file it names (relative to the TOML file).

    Raises:
        FileNotFoundError: If a topic names a schematic that does not exist.
        ValueError: If a schematic file is not an SVG element.
    """
    path = Path(path)
    with open(path, "rb") as f:
        parsed = tomllib.load(f)
    for topic in parsed.get("topic", []):
        if "svg" in topic:
            topic["svg"] = read_schematic(path.parent / topic["svg"])
    return overviews_from_toml(parsed)


def read_schematic(path: Path) -> str:
    """The text of one schematic, which must be a single ``<svg>`` element."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if not (text.startswith("<svg") and text.endswith("</svg>")):
        raise ValueError(f"{path} is not one <svg> element")
    return text


def load_descriptions(path: Path) -> dict[str, str]:
    """Written descriptions from ``descriptions.tsv``."""
    return parse_descriptions(Path(path).read_text(encoding="utf-8"))


def load_explanations(path: Path) -> dict[str, str]:
    """Plain-language explanations from ``explanations.md`` ({} if the file does not exist)."""
    path = Path(path)
    return parse_explanations(path.read_text(encoding="utf-8")) if path.exists() else {}


def git_meta(root: Path) -> str:
    """Branch, short commit and commit date of *root*, as "<branch> @ <commit>, <date>".

    Raises:
        RuntimeError: If *root* is not a git checkout with at least one commit.
    """
    try:
        branch = git_output(root, "rev-parse", "--abbrev-ref", "HEAD")
        commit, date = git_output(root, "log", "-1", "--format=%h %cs").split()
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as err:
        raise RuntimeError(f"cannot read the git commit of {root}: {err}") from err
    return f"{branch} @ {commit}, {date}"


def git_output(root: Path, *args: str) -> str:
    """stdout of one git command run in *root*."""
    done = subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)
    return done.stdout.strip()


def write_page(path: Path, html: str) -> None:
    """Write the page, creating its folder."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
