"""Per-cell resting SNR and large-transient counts for each plane of a multi-plane Suite2p run.

Same measure as the 71R / AB08 comparison: cells are the 200 strongest local maxima of the mean
image's cell-scale structure (difference of Gaussians); each cell's trace is the mean of a disc
minus the mean of an annulus around it (so the drifting detector offset, common to both, cancels),
averaged in ~0.5 s bins; noise is the robust SD of successive bin differences / sqrt(2).
Pixel-based geometry is given for 512 px frames and scaled for other frame sizes of the same FOV.

Usage: python plane_snr.py <out_dir> <run_dir>... --fs HZ... [--labels NAME...] [--minutes 4.3]
  each <run_dir> holds suite2p/plane*/ (flyback planes in ops["ignore_flyback"] are skipped);
  --labels names each run in the results and cache files (default: the run folder's name).
Writes <out_dir>/cache/<run>_plane<N>_<stage>.npz (binned traces) and <out_dir>/plane_snr.json.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.ndimage import gaussian_filter, gaussian_filter1d, maximum_filter

N_CELLS = 200
GEOMETRY_512 = dict(dog_small=1.5, dog_large=6.0, spacing=11, edge=30, disc_r=5.0, ring_in=8.0, ring_out=14.0)
TARGET_BIN_S = 0.5
HIGHPASS_S = 16.0
NOISE_CEILING_SD = 4.5  # ceiling used for 71R / AB08, kept so counts are comparable
SMALL_BAND_SD = (2.0, 4.0)
FLASH_SD = 6.0
STAGE_FILES = {"registered": "data.bin", "raw": "data_raw.bin"}  # functional channel (functional_chan = 2)


# ---------- I/O ----------
def load_ops(plane_dir):
    return np.load(plane_dir / "ops.npy", allow_pickle=True).item()


def imaging_planes(run_dir):
    dirs = sorted((run_dir / "suite2p").glob("plane[0-9]*"), key=lambda d: int(d.name[5:]))
    dirs = [d for d in dirs if (d / "ops.npy").exists()]
    if not dirs:
        raise FileNotFoundError(f"No plane*/ops.npy under {run_dir}/suite2p")
    ignore = load_ops(dirs[0]).get("ignore_flyback")  # None or [] when there is no flyback plane
    flyback = set() if ignore is None else {int(p) for p in np.atleast_1d(ignore)}
    return [d for d in dirs if int(d.name[5:]) not in flyback]


def open_movie(plane_dir, stage, ops):
    shape = (int(ops["nframes"]), int(ops["Ly"]), int(ops["Lx"]))
    path = plane_dir / STAGE_FILES[stage]
    if path.stat().st_size != np.prod(shape) * 2:
        raise ValueError(f"{path} size does not match int16 {shape}")
    return np.memmap(path, dtype=np.int16, mode="r", shape=shape)


def cached_bintraces(cache_path, key, compute):
    """Return the cached npz contents if its key matches, else compute, save and return."""
    if cache_path.exists():
        data = dict(np.load(cache_path))
        if str(data.get("key")) == key:
            return data
    data = compute()
    np.savez(cache_path, key=key, **data)
    return data


# ---------- analysis: cells ----------
def scaled_geometry(lx):
    s = lx / 512
    return {k: v * s for k, v in GEOMETRY_512.items()}


def pick_cells(mean_img, valid_yx, geom, n_cells=N_CELLS):
    """Strongest local maxima of the cell-scale (DoG) structure, inside the valid region."""
    dog = gaussian_filter(mean_img, geom["dog_small"]) - gaussian_filter(mean_img, geom["dog_large"])
    size = int(round(geom["spacing"])) | 1
    peaks = (dog == maximum_filter(dog, size=size))
    (y0, y1), (x0, x1) = valid_yx
    e = int(np.ceil(geom["edge"]))
    inside = np.zeros_like(peaks)
    inside[y0 + e:y1 - e, x0 + e:x1 - e] = True
    ys, xs = np.nonzero(peaks & inside)
    order = np.argsort(dog[ys, xs])[::-1][:n_cells]
    if len(order) < n_cells:
        raise ValueError(f"Only {len(order)} candidate cells, need {n_cells}")
    return np.stack([ys[order], xs[order]], axis=1)


def roi_weights(cells, shape, r_lo, r_hi):
    """Sparse (n_cells x n_pixels) averaging matrix over pixels at r_lo <= distance <= r_hi."""
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    rows, cols, vals = [], [], []
    for i, (cy, cx) in enumerate(cells):
        d = np.hypot(yy - cy, xx - cx)
        idx = np.flatnonzero((d >= r_lo) & (d <= r_hi))
        rows.append(np.full(idx.size, i)); cols.append(idx); vals.append(np.full(idx.size, 1 / idx.size))
    return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(len(cells), shape[0] * shape[1]))


def bin_traces(movie, weights, frames_per_bin, n_bins, chunk_bins=50):
    out = []
    for b0 in range(0, n_bins, chunk_bins):
        b1 = min(b0 + chunk_bins, n_bins)
        block = np.asarray(movie[b0 * frames_per_bin:b1 * frames_per_bin], dtype=np.float32)
        binned = block.reshape(b1 - b0, frames_per_bin, -1).mean(1)
        out.append(np.asarray(weights @ binned.T).T)
    return np.concatenate(out)  # bins x cells


# ---------- analysis: measures ----------
def robust_sd_of_steps(x):
    d = np.diff(x, axis=0)
    return 1.4826 * np.nanmedian(np.abs(d - np.nanmedian(d, 0)), 0) / np.sqrt(2)


def flash_bins(ring):
    """Bins where the brightness around all cells jumps (light reaching the detector)."""
    common = np.median(ring, axis=1)
    resid = common - gaussian_filter1d(common, 5, mode="nearest")
    sd = robust_sd_of_steps(common[:, None])[0]
    return resid > FLASH_SD * sd


def rising_edges(mask):
    return np.diff(np.vstack([np.zeros((1, mask.shape[1]), bool), mask]).astype(int), axis=0) == 1


def cell_measures(disc, ring, hp_bins, ceiling=NOISE_CEILING_SD):
    flash = flash_bins(ring)
    c = disc - ring
    c[flash] = np.nan
    noise = robust_sd_of_steps(c)
    filled = np.where(np.isnan(c), np.nanmedian(c, 0), c)
    hp = c - gaussian_filter1d(filled, hp_bins, axis=0, mode="nearest")
    z = hp / noise
    up, down = np.nan_to_num(z) > ceiling, np.nan_to_num(-z) > ceiling
    n_up, n_down = rising_edges(up).sum(0), rising_edges(down).sum(0)
    lo, hi = SMALL_BAND_SD
    zf = z[np.isfinite(z)]
    return dict(
        resting_snr=np.nanmean(c, 0) / noise, noise=noise, peak=np.nanmax(z, 0), trough=-np.nanmin(z, 0),
        transients=int(n_up.sum()), transients_flipped=int(n_down.sum()),
        cells_ge1=int((n_up >= 1).sum()), cells_ge3=int((n_up >= 3).sum()),
        surplus_small=int(((zf > lo) & (zf <= hi)).sum() - ((zf < -lo) & (zf >= -hi)).sum()),
        surplus_large=int((zf > hi).sum() - (zf < -hi).sum()), n_flash_bins=int(flash.sum()), z=z)


def frame_jumps(ops):
    return float(np.median(np.abs(np.diff(ops["yoff"])))), float(np.median(np.abs(np.diff(ops["xoff"]))))


# ---------- workflow ----------
def plane_bintraces(plane_dir, stage, fs, minutes, cache_dir, run_name):
    ops = load_ops(plane_dir)
    geom = scaled_geometry(int(ops["Lx"]))
    fpb = max(1, int(round(TARGET_BIN_S * fs)))
    n_bins = int(minutes * 60 * fs) // fpb
    movie = open_movie(plane_dir, stage, ops)
    if n_bins * fpb > movie.shape[0]:
        raise ValueError(f"{plane_dir}: {minutes} min needs {n_bins * fpb} frames, only {movie.shape[0]} recorded")
    key = hashlib.sha1(json.dumps([str(plane_dir), stage, fpb, n_bins, GEOMETRY_512, N_CELLS,
                                   (plane_dir / STAGE_FILES[stage]).stat().st_mtime]).encode()).hexdigest()

    def compute():
        if stage == "registered":
            mean_img, valid = ops["meanImg"], (ops["yrange"], ops["xrange"])
        else:
            mean_img, valid = movie[::4].astype(np.float32).mean(0), ((0, ops["Ly"]), (0, ops["Lx"]))
        cells = pick_cells(np.asarray(mean_img, np.float64), valid, geom)
        shape = (int(ops["Ly"]), int(ops["Lx"]))
        disc = bin_traces(movie, roi_weights(cells, shape, 0, geom["disc_r"]), fpb, n_bins)
        ring = bin_traces(movie, roi_weights(cells, shape, geom["ring_in"], geom["ring_out"]), fpb, n_bins)
        return dict(disc=disc, ring=ring, cells=cells, frames_per_bin=fpb, bin_s=fpb / fs)

    data = cached_bintraces(cache_dir / f"{run_name}_{plane_dir.name}_{stage}.npz", key, compute)
    return data, ops


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out_dir", type=Path)
    p.add_argument("run_dirs", type=Path, nargs="+")
    p.add_argument("--fs", type=float, nargs="+", required=True, help="per-plane rate (Hz) for each run")
    p.add_argument("--labels", nargs="+", help="one name per run (default: each run folder's name)")
    p.add_argument("--minutes", type=float, default=4.3)
    p.add_argument("--stages", nargs="+", default=list(STAGE_FILES))
    args = p.parse_args()
    if len(args.fs) != len(args.run_dirs):
        raise ValueError("--fs needs one rate per run")
    labels = args.labels or [d.name for d in args.run_dirs]
    if len(labels) != len(args.run_dirs) or len(set(labels)) != len(labels):
        raise ValueError("--labels needs one unique name per run")
    cache = args.out_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    results = []
    for run_dir, fs, label in zip(args.run_dirs, args.fs, labels):
        for plane_dir in imaging_planes(run_dir):
            for stage in args.stages:
                data, ops = plane_bintraces(plane_dir, stage, fs, args.minutes, cache, label)
                bin_s = float(data["bin_s"])
                m = cell_measures(data["disc"], data["ring"], HIGHPASS_S / bin_s)
                m.pop("z")
                results.append(dict(run=label, plane=plane_dir.name, fs_hz=fs, minutes=args.minutes, stage=stage, bin_s=bin_s,
                                    n_bins=int(data["disc"].shape[0]), frame_jumps_px=frame_jumps(ops),
                                    **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in m.items()}))
                print(f"{label} {plane_dir.name} {stage}: SNR {np.median(m['resting_snr']):.2f} "
                      f"transients {m['transients']} ({m['transients_flipped']}) cells>=1 {m['cells_ge1']}", flush=True)
    (args.out_dir / "plane_snr.json").write_text(json.dumps(results))


if __name__ == "__main__":
    main()
