# mini2p-preprocessing

Scan-phase (bidiphase) correction, optional destriping, distortion correction and
Suite2p registration for two-channel Mini2P recordings, plus batch processing and
mean-image review exports.

## Setup

```bash
uv sync                      # reproducible environment from uv.lock
uv sync --extra notebook     # adds JupyterLab and ipywidgets for the QC notebook
cp config/pipeline.example.toml config/pipeline.toml   # git-ignored; fill in your paths
uv run pytest                # synthetic-data tests, no lab data needed
```

`config/pipeline.toml` holds every setting and machine path. The notebooks, the
CLI and the batch workers all read it, so they cannot drift apart.

## Running

- **One recording, interactively:** `notebooks/qc_single_run.ipynb`. Previews the
  phase trace, offers a manual-offset slider, writes each stage and shows QC
  figures for each.
- **One recording, headless:**
  `uv run mini2p-preprocess --config config/pipeline.toml --input RAW.tif --output-dir RUN_DIR/without_destriping`
- **Many sessions:** `notebooks/batch_and_exports.ipynb`. Builds a dry-run plan,
  launches one CLI process per pending TIFF, then exports mean images and PDFs.

## From registration to ΔF/F0

1. **Register:** the pipeline above (bidiphase, optional destriping, distortion, Suite2p).
2. **Segment:** run Cellpose on the registered mean image, e.g.
   `meanImg_green_functional_contrast.png`, and save its `*_seg.npy`.
3. **Extract:**
   `uv run mini2p-extract --config config/pipeline.toml --plane-dir RUN_DIR/.../suite2p/plane0 --seg MEAN_seg.npy`,
   or the extraction cell in the QC notebook. This builds Suite2p `stat` from the masks,
   extracts both channels with identical masks, classifies, deconvolves, and
   **replaces** `F`, `Fneu`, `stat`, `iscell`, `spks` (and `F_chan2`, `Fneu_chan2`) in
   that folder, recording what it did in `cellpose_extraction.json`. Nothing is written
   if the mask shape or frame rate does not match the plane.
4. **Audit:** the "Fluorescence audit" section of the QC notebook shows F, Fneu,
   corrected F with its F0, ΔF/F0 and the green/red ratio for each ROI, and flags ROIs
   whose F0 is ever ≤ `fluorescence.min_f0`.

**What ΔF/F0 means here.** Suite2p's `preprocess` returns ΔF = Fcorr − F0, where
Fcorr = F − c·Fneu. The package recovers F0 explicitly and returns ΔF/F0, with ΔF and
F0 kept alongside. ΔF/F0 is NaN wherever F0 ≤ `min_f0`, since the ratio has no
meaning there. ΔF itself stays finite. The same applies to the green/red ratio,
Δ(G/R)/(G/R)0.

## Layout

| Module | Layer | Contents |
|---|---|---|
| `config.py` | boundary | pydantic settings; output naming |
| `tiff_io.py` | I/O | chunked reads/writes of interleaved TIFFs |
| `bidiphase.py` | analysis | phase score, estimators, channel selection |
| `destripe.py` | analysis | even/odd line-profile removal |
| `correction.py` | analysis + I/O | per-chunk correction stages; TIFF + `.phase.json` writer |
| `distortion.py` | analysis + I/O | fixed-map warp; TIFF + `.distortion.json` writer |
| `registration.py` | I/O | Suite2p settings/db, run, output loaders |
| `pipeline.py`, `cli.py` | I/O | one recording end to end |
| `batch.py` | analysis + I/O | run planning (pure `decide_action`), manifests, workers |
| `qc.py`, `contrast.py`, `plots.py` | analysis / plotting | QC computations; figures accept `ax` |
| `exports.py` | I/O | fixed-contrast exports and review PDFs |
| `rois.py` | analysis | Cellpose labels ↔ Suite2p `stat`, ROI outlines |
| `fluorescence.py` | analysis | neuropil correction, F0, ΔF, ΔF/F0, G/R ratio, audit table |
| `extraction.py`, `plane.py` | I/O | Cellpose-mask extraction; loading a plane's traces and valid frames |

Distortion calibration still comes from your `distortion_correct_full_recording.py`,
which is shared with the SLURM worker. Set `distortion.calibration_script_dir` to
the folder that contains it.

## Relation to the original notebook

`tests/test_equivalence.py` runs verbatim copies of the original notebook functions
(`tests/reference/original_notebook.py`) and the package on the same synthetic
TIFFs. It requires byte-identical corrected and undistorted TIFFs, identical
`.phase.json` / `.distortion.json` sidecars, and identical Suite2p settings and db,
across several setting combinations. Deliberate differences, none of which touch the
written pixels:

- Batch workers run `mini2p-preprocess` instead of re-executing the notebook with
  nbconvert, so there is no `preprocessing_log.ipynb`. Each run folder has
  `preprocess.log` and `pipeline_settings.json` (the exact settings used). The
  manifest records `pipeline_command`, `pipeline_config` and `worker_log`.
- Launching a batch re-plans from the manifests on disk, then copies the config to
  `batch_config_<UTC time>.toml` in the derivatives folder. Every worker reads that
  frozen copy, so editing `pipeline.toml` mid-batch cannot change runs already
  queued.
- A corrupt `source_tiff.json` stops planning with an error instead of being
  silently ignored.
- `distortion.mode` and `exports.expected_runs` must be stated in the config. The
  PDFs have their own `pdf_overwrite` flag.
- The QC figures use the same function that writes the TIFF, so in
  `second_pass_mode = "whole"` they now honour `manual_second_phase_by_start`, as
  the written data always did.
- The three copies of the contrast scaling are one function. For finite mean
  images the output is unchanged.

## Relation to `Convert_seg_to_stat` and `cellTV_functions_cohort3.py`

The extraction and ΔF/F0 code ports those two scripts. The differences are deliberate:

- **ΔF/F0 is a true ratio.** `get_dff` saved Suite2p's `preprocess` output as
  `DF_F0.npy`, but that output is ΔF, not divided by F0. `get_dff_GR` likewise
  returned Δ(G/R), not Δ(G/R)/(G/R)0.
- **The neuropil coefficient comes from the config.** `get_dff` and `get_dff_GR` used a
  hard-coded 0.7.
- **Suite2p's `prctile` baseline modifies its input in place,** so it is always given a
  copy. Its `constant` baseline takes one minimum across all ROIs together, so it is
  rejected rather than silently giving a pooled F0.
- **Extraction checks before writing.** It checks that the mask matches the plane's
  image size and that `ops["fs"]` matches the config, and it writes nothing if either
  differs. The script overwrote the files unconditionally.
- **Settings come from the config.** The settings used are written into `ops` from the
  config, both nested (`ops["extraction"]`, `ops["dcnv_preprocess"]`, which Suite2p
  1.x and its GUI read) and as the flat keys (`neucoeff`, `baseline`, ...) that older
  analysis code such as `get_dff` reads. The script only filled in missing flat keys,
  with `fs = 7.52` and `tau = 1.0` hard-coded, and left the nested values the GUI uses
  unchanged.
- **The classifier gets its activity features.** Like Suite2p's own pipeline,
  extraction now stores `snr`, `skew` and `std` in `stat` before classifying. Without
  them, `iscell` was computed from ROI shape alone. ROI sizes are also normalised to
  the median of all ROIs (`median=True`), as Suite2p does for Cellpose.
- **Extraction writes safely.** Registered binaries are found where `ops` says, and
  both channels are required when `ops["nchannels"]` is 2. Files are written under
  temporary names and then moved into place, so an interrupted write leaves the
  previous outputs intact. A single-channel extraction removes stale `F_chan2` and
  `Fneu_chan2`. `load_plane_traces` refuses folders whose files disagree on the ROI
  count.
- **Frame selections are checked.** Valid frames must be a boolean mask or strictly
  increasing indices, so a 0/1 mask saved as integers is rejected rather than read as
  frames 0 and 1. Traces containing NaN or inf are rejected before any baseline is
  computed.
- **The `plot_fluorescence_traces` crash is fixed.** It failed for one-channel data,
  because `plt.subplots(1, 1)` returns a single axis that it then indexed. Its
  replacements are `plot_trace_audit` and `plot_dff_trace`, which accept axes.
- **Not ported:** `get_funcimg_path`, `load_valid_frames`'s session-folder lookup and
  `load_dF_data`'s caching. They depend on the analysis repo's folder layout. Pass a
  `valid_frames.npz` path through `[qc] valid_frames` instead. ΔF/F0 for one plane is
  quick to recompute.

## Keep QC runs out of the batch tree

The batch planner treats any run folder that holds a `.phase.json` but no
`source_tiff.json` manifest as finished output, and skips it. The original batch
cell did the same. So point `[qc] run_dir` somewhere outside
`derivatives_subject_dir`, and don't run the CLI by hand inside it.

## Known issues, deliberately left for the follow-up fix step

1. `adjacent_line_score` pairs each odd row only with the row above it, which
   biases the estimate on oblique structure. Line-locked stripes, which are still
   present during estimation when destriping is off, can also capture the peak.
2. `adjacent_line_score` returns `-inf` for a flat or non-finite image. If every
   candidate scores `-inf`, the chunk silently gets `-max_shift_px`.
3. `score_margin` compares the best shift with its immediate neighbour, so it
   measures how sharp the peak is rather than whether a rival peak exists.
4. Integer-only shifts make the chunked second pass nearly always return 0.
5. `two_step_registration = true` with `keep_raw_binary = false`. Suite2p's
   two-step registration needs the raw binary, so check whether it is being
   skipped.
6. Float output can exceed the int16 range when Suite2p converts it. Consider
   clipping before writing.
7. The batch writes outputs to `run-XX/without_destriping/`, but the export
   default `run_subdir = ""` looks in `run-XX/`. Set `run_subdir` to match the
   batch.
