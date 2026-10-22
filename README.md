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
  (or `RUN_DIR/with_destriping` with `[destripe] enabled = true`; a folder that disagrees
  with the setting is refused)
- **Many sessions:** `notebooks/batch_and_exports.ipynb`. Builds a dry-run plan,
  launches one CLI process per pending TIFF, then exports mean images and PDFs.
- **Sessions the batch has finished, while it runs:** `notebooks/post_batch_sessions.ipynb`,
  in a second kernel. Exports, Cellpose extraction and an alignment review for the chosen
  sessions, only for runs the batch completed since `rerun_completed_before`.

### With and without destriping

Each variant has its own subfolder of `run-XX/`: `without_destriping/` and
`with_destriping/`. The batch picks it from `[batch] destripe`, the QC notebook and
the CLI from `[destripe] enabled` (the CLI's `--destripe` overrides it).

For both variants over many sessions, run the batch once with `destripe = false` and
once with `destripe = true`, in either order. Each launch freezes its own copy of the
config, so the setting can be changed as soon as a batch has started. To export, set
`[exports] run_subdir` to the variant and run the export cells once per variant; the
summary JSON and PDFs carry the variant in their names.

`run-XX/suite2p/` itself is the registration the Cellpose masks were drawn on, and
the batch never changes it. The planner only checks that whatever is recorded
directly in `run-XX/` came from the same TIFF as the run, and flags a conflict if not.

### Suite2p binaries on a local disk

Suite2p writes each run's registered movie (`data.bin`, `data_chan2.bin`, about 3.4 GB each
for a 7,000-frame two-channel run) and rereads it several times while registering. Over a
network share this is slow and can fail with `OSError: [Errno 5] Input/output error`. Set
`[registration] fast_disk` to a local folder: the binaries go to
`<fast_disk>/<session>_funcimg_<run>_<variant>/`, everything else still goes beside the run,
and `ops["reg_file"]` records where the binary is. Cellpose extraction reads the binary, so
keep a run's folder until it has been extracted.

### Re-running, extraction and outputs

`[batch] rerun_completed_before = <time>` registers again every run that finished
before that time, e.g. `2026-10-21T09:00:00+01:00` (a time zone is required). Keep the
same time when relaunching after a failure, so runs the rerun already replaced are
skipped. Each run's `suite2p/` folder is cleared first, because Suite2p would otherwise
reuse the old binary and skip registration. Clearing needs `overwrite = true` (checked
at launch), happens only inside a `with_destriping/` or `without_destriping/` folder,
and is refused if the folder holds any `*_seg.npy`.

After a batch, the last cells of the batch notebook (run once per variant, with
`[exports] run_subdir` set to it):

1. **Contrast exports:** green and red mean images per run, as raw TIFFs and as
   contrast PNGs with a red/green overlay, using the `[exports]` contrast.
2. **Review PDFs** of those mean images across runs.
3. **Cellpose extraction:** `run-XX/<cellpose_seg>`, drawn on `run-XX/<cellpose_image>`,
   is aligned onto the variant's mean image and extracted into its `suite2p/plane0`
   (`stat`, `F`, `Fneu`, `iscell`, `spks`, red traces, `cellpose_extraction.json`).
   Nothing is written for a run whose alignment is refused or whose masks' registration
   is of a different TIFF (`failed`). An extraction from the same masks and settings is
   kept; one from other masks is kept unless `EXTRACT_OVERWRITE`.
4. **Alignment review:** `cellpose_extraction_summary_<label>.json` and
   `mask_alignment_<label>.pdf`, one page per run with the masks as segmented and as
   shifted onto the variant's mean image.

### Movies

`mini2p-movie` writes MP4s (H.264, which PowerPoint, Keynote and QuickTime play) of a
registered run, from Suite2p's registered binaries, into `run-XX/<variant>/movies/`
(or `[movies] output_dir`):

```bash
# the whole run
uv run mini2p-movie --config config/pipeline.toml --cohort cohort-01 --subject sub-03 \
    --session 224 --run 1 --whole-recording
# frames 10 to 150 (0-based, both included), green and red merged, bins of 10 frames
uv run mini2p-movie --config config/pipeline.toml --cohort cohort-01 --subject MY_71_R \
    --session 224 --run 1 --start-frame 10 --end-frame 150 --channel merge --average-frames 10
```

or from Python, `movies.make_movies(load_movie_settings(CONFIG), "cohort-01", "sub-03", 224,
1, whole_recording=False, start_frame=10, end_frame=150)`. Each call writes two movies:

- **raw**: every frame, unsmoothed;
- **averaged**: each movie frame the mean of `average_frames` frames, in bins (`bin`:
  fewer frames, each covering more time, the usual choice for talks) or as a running mean
  (`rolling`: smooth in time at the full frame rate), optionally with spatial Gaussian
  smoothing (`spatial_sigma_px`, e.g. 1).

Defaults are in `[movies]` (see `config/pipeline.example.toml`); any of them can be given
on the command line (`--average-frames`, `--average-mode`, `--speed`, `--channel`,
`--variant`, `--spatial-sigma-px`, `--upscale`, `--output-dir`). Contrast is fixed per
movie (percentiles of its own frames), so slow brightness changes stay visible; the raw
and averaged movies therefore have different display ranges. Each frame is labelled with
the time from the start of the run, its frames and the playback speed. Existing movies are
refused unless `--overwrite`. Single 7.5 Hz frames are mostly shot noise: for slides,
bins of 5-10 frames with `spatial_sigma_px = 1` at 4-10x speed usually read best. A raw
movie of a whole run is large (about 0.9 GB per channel for 7000 frames).

## From registration to ΔF/F0

1. **Register:** the pipeline above (bidiphase, optional destriping, distortion, Suite2p).
2. **Segment:** run Cellpose on the registered mean image, either
   `meanImg_green_contrast.png` (the contrast chosen in the QC notebook) or
   `meanImg_green_functional_contrast.png` (the batch export's fixed contrast), and save
   its `*_seg.npy`. The QC notebook's mask and ROI views use the contrast saved in the
   QC notebook.
3. **Extract:**
   `uv run mini2p-extract --config config/pipeline.toml --plane-dir RUN_DIR/.../suite2p/plane0 --seg MEAN_seg.npy`,
   or the extraction cell in the QC notebook. This builds Suite2p `stat` from the masks,
   extracts both channels with identical masks, classifies, deconvolves, and
   **replaces** `F`, `Fneu`, `stat`, `iscell`, `spks` (and `F_chan2`, `Fneu_chan2`) in
   that folder, recording what it did in `cellpose_extraction.json`. Nothing is written
   if the mask shape or frame rate does not match the plane.

   **Masks drawn on another registration** (e.g. one run on the first 1000 frames):
   pass that registration's mean image with `--mask-image`, or set `[qc] cellpose_image`.
   The masks are then shifted rigidly onto this plane's `meanImg`. Extraction refuses
   if the images do not match, if parts of the image need different shifts (more than
   `fluorescence.mask_shift_tolerance_px`), or if the shift exceeds
   `fluorescence.mask_max_shift_px`. Masks pushed mostly off the edge are dropped. The
   shift and match scores are saved in `cellpose_extraction.json`.
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
| `alignment.py` | analysis | rigid shift of Cellpose masks between two mean images |
| `fluorescence.py` | analysis | neuropil correction, F0, ΔF, ΔF/F0, G/R ratio, audit table |
| `extraction.py`, `plane.py` | I/O | Cellpose-mask extraction; loading a plane's traces and valid frames |
| `cellpose_batch.py` | I/O | extraction for every batch run; summary and mask-alignment review PDF |
| `movies.py` | analysis + I/O | movie frames and names (pure `plan_movie`); MP4 writing |

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
cell did the same. QC and batch runs now share the variant subfolder names, so a QC
run inside the derivatives tree would be taken as that run's batch output. Point
`[qc] run_dir` somewhere outside `derivatives_subject_dir`, and don't run the CLI by
hand inside it.

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
