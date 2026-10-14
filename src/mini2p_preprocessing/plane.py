"""Loading one Suite2p plane folder's traces, ROIs and frame selections."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .rois import labels_from_stat


@dataclass(frozen=True)
class PlaneTraces:
    f: np.ndarray
    fneu: np.ndarray
    f_chan2: np.ndarray | None
    fneu_chan2: np.ndarray | None
    iscell: np.ndarray
    stat: np.ndarray
    ops: dict
    labels: np.ndarray  # ROI i drawn as i + 1

    @property
    def fs(self) -> float:
        return float(self.ops["fs"])


def _load_optional(path: Path) -> np.ndarray | None:
    return np.load(path) if path.is_file() else None


def load_plane_traces(plane_dir: Path) -> PlaneTraces:
    plane_dir = Path(plane_dir)
    if not (plane_dir / "F.npy").is_file():
        raise FileNotFoundError(
            f"No F.npy in {plane_dir}; run Cellpose extraction (or Suite2p detection) first"
        )
    ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
    stat = np.load(plane_dir / "stat.npy", allow_pickle=True)
    traces = PlaneTraces(
        f=np.load(plane_dir / "F.npy"),
        fneu=np.load(plane_dir / "Fneu.npy"),
        f_chan2=_load_optional(plane_dir / "F_chan2.npy"),
        fneu_chan2=_load_optional(plane_dir / "Fneu_chan2.npy"),
        iscell=np.load(plane_dir / "iscell.npy"),
        stat=stat,
        ops=ops,
        labels=labels_from_stat(stat, int(ops["Ly"]), int(ops["Lx"])),
    )
    _check_consistent(traces, plane_dir)
    return traces


def _check_consistent(traces: PlaneTraces, plane_dir: Path) -> None:
    """Every per-ROI file must describe the same ROIs, and traces the same frames."""
    counts = {
        "stat": len(traces.stat),
        "F": traces.f.shape[0],
        "Fneu": traces.fneu.shape[0],
        "iscell": traces.iscell.shape[0],
    }
    shapes = {"Fneu": traces.fneu.shape}
    if (traces.f_chan2 is None) != (traces.fneu_chan2 is None):
        raise ValueError(f"{plane_dir} has only one of F_chan2.npy and Fneu_chan2.npy")
    if traces.f_chan2 is not None:
        shapes.update({"F_chan2": traces.f_chan2.shape, "Fneu_chan2": traces.fneu_chan2.shape})
    if len(set(counts.values())) > 1 or any(s != traces.f.shape for s in shapes.values()):
        raise ValueError(
            f"ROI files in {plane_dir} disagree: counts {counts}, F shape {traces.f.shape}, "
            f"others {shapes}; rerun extraction"
        )


def load_valid_frames(npz_file: Path, key: str = "valid_frames") -> np.ndarray:
    """Frame mask or indices saved by the behaviour alignment (an .npz file)."""
    with np.load(npz_file) as data:
        if key not in data:
            raise KeyError(f"{npz_file} has no '{key}' array; it has {list(data.keys())}")
        return data[key]
