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
  per-frame phase (or, with `method = "chunked"`, the phase trace and a manual-offset
  slider), writes each stage and shows QC figures for each.
- **One recording, headless:**
  `uv run mini2p-preprocess --config config/pipeline.toml --input RAW.tif --output-dir RUN_DIR/with_distortion_correction/without_destriping`
  (or `.../with_destriping` with `[destripe] enabled = true`; a folder that disagrees
  with the settings is refused)
- **Many sessions:** `notebooks/batch_and_exports.ipynb`. Builds a dry-run plan,
  launches one CLI process per pending TIFF, then exports mean images and PDFs.
- **Sessions the batch has finished, while it runs:** `notebooks/post_batch_sessions.ipynb`,
  in a second kernel. Exports, Cellpose extraction and an alignment review for the chosen
  sessions, only for runs the batch completed since `rerun_completed_before`.

### With and without destriping

Each variant has its own subfolder of `run-XX/`: `with_distortion_correction/` or
`without_distortion_correction/` (from `[distortion] mode`), then `without_destriping/`
or `with_destriping/`. The batch picks destriping from `[batch] destripe`, the QC
notebook and the CLI from `[destripe] enabled` (the CLI's `--destripe` overrides it).
Suite2p's output goes one level further down, in a folder named for the channel it
aligned on, so registrations aligned on either channel never share a folder; the
corrected TIFFs do not depend on that channel and stay in the variant folder:

```
run-01/
  with_distortion_correction/
    with_destriping/                  corrected TIFFs and their sidecars
      aligned_structural/
        suite2p/plane0/
        motion_alignment.json         which channel, and why
        movies/
      aligned_functional/
        suite2p/plane0/
    without_destriping/
  without_distortion_correction/
    with_destriping/
    without_destriping/
```

`[registration] motion_alignment_source = "auto"` (the default) aligns each run on the
structural channel when the per-frame bidiphase step trusted it for at least
`auto_structural_min_fraction` of frames, and on the functional channel otherwise, so
the folder is the one it chose. Wherever a variant is named (`[exports] run_subdir`,
`[movies] variant`) it is written `with_distortion_correction/with_destriping`; in file
names (export summaries, PDFs, movies) the `/` becomes `_`. `[exports] alignment` and
`[movies] alignment` pick the registration: `"auto"` takes the one `aligned_*` folder a
run has and refuses a run that has both. Registering on one channel removes the other
channel's `aligned_*/` folder if it registered a different or since-rewritten TIFF (it is
stale; same guards as re-registering), and keeps it if it registered the same TIFF, so
the two can be compared. "Same TIFF" means the same file name, size and modification
time, so copy session folders with a tool that keeps file times (`rsync -a`, `cp -p`,
Finder); a plain `cp` makes the other channel's registration look stale. A variant folder in the old layout (`suite2p/` directly inside)
is flagged by the batch plan rather than skipped: move or delete it to register again.

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
at launch), happens only inside `<variant>/aligned_<channel>/`, and is refused if the
folder holds any `*_seg.npy`.

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

`mini2p-movie` writes greyscale MP4s (H.264, which PowerPoint, Keynote and QuickTime play) of
a run at each stage of the preprocessing:

| Stage (`[movies]` switch) | File |
|---|---|
| raw (`stage_raw`) | the acquired TIFF (from the run's `source_tiff.json`, or `raw_tiff`) |
| bidiphase (`stage_bidiphase`) | `*_dynamic_bidiphase[_destriped].tif` (phase correction, and destriping) |
| undistorted (`stage_undistorted`) | `*_undistorted.tif` (+ distortion correction; cropped) |
| registered (`stage_registered`) | Suite2p's `data.bin` / `data_chan2.bin` (+ motion registration) |
| background removed (`stage_background_removed`) | `data_background_removed_<source>.tif` in the plane folder, written by the QC notebook's background step (+ background regressed out of every pixel; `background_source`: `vessel` mask or `non_roi`, every pixel outside the cell ROIs) |

Everything is set in the `[movies]` section of the config (see
`config/pipeline.example.toml`):

- **which run**: by name (`cohort`, `subject`, `session`, `run` under `derivatives_root`), or
  by path (`run_dir`: any folder laid out like a `run-XX/` folder, e.g. one the QC notebook
  made). If the config has both, `run_dir` is used and the by-name settings are ignored, with
  a warning; on the command line, `--run-dir` or a run by name (`--session` ...) overrides
  whichever the config uses. Then `variant`, and `alignment` (`"auto"`: the one
  `aligned_*` folder the variant has);
- **which frames**: `whole_recording` (it must be set), or `start_frame` to `end_frame`,
  0-based and both included (with the whole recording these two are ignored, with a warning);
- **which stages**: the five switches above (only `stage_registered` is on unless the
  config says otherwise), and **versions**: `unaveraged_movie`, `averaged_movie`;
- **behavior**: with `behavior = true`, every movie also gets a twin (`..._behavior_...`)
  with the run's behavior video on the left of the imaging, scaled to its height and in
  sync: each movie frame shows the video frame on screen at the middle of its imaging
  window, by pyControl time. It reads `run-XX/frame_times.npz` (imaging frames) and
  `run-XX/video_frame_times.npz` (video frames, and the video's path), which
  magnitude-bandit-analysis's `scripts/funcimg/reward_responses.ipynb` writes (steps 1 and
  1b); `behavior_video` gives the video if it has moved. Where a movie frame or the video
  has no time (outside the sync pulses, or after the video ends), the panel is black and
  labelled "behavior (no video)". The video is decoded from its start for every movie, so
  a frame range late in a long run waits for the video to reach it;
- **where to read and write**: `derivatives_root`, `raw_tiff`, `output_dir` (default
  `run-XX/<variant>/aligned_<channel>/movies/`; `~` works in all of them);
- **how they look**.

Suite2p must have finished on the run: it writes `ops.npy` only at the end. Then:

```bash
uv run mini2p-movie --config config/pipeline.toml
# a run folder by path instead of cohort/subject/session/run
uv run mini2p-movie --config config/pipeline.toml --run-dir ~/Desktop/example_ses/img
# any other setting can be overridden for one call (see --help), e.g. another session and run,
# or other frames (--start-frame / --end-frame imply --no-whole-recording)
uv run mini2p-movie --config config/pipeline.toml --session 225 --run 2 \
    --no-stage-raw --no-stage-bidiphase --no-stage-undistorted
```

or from Python, `movies.make_movies(movies.load_movie_settings(CONFIG, session=225))`.

- **unaveraged**: every frame, unsmoothed;
- **averaged**: each movie frame the mean of `average_frames` frames, in bins (`bin`:
  fewer frames, each covering more time, the usual choice for talks) or as a running mean
  (`rolling`: smooth in time at the full frame rate), optionally with spatial Gaussian
  smoothing (`spatial_sigma_px`, e.g. 1).

The functional (green) channel is on the left and the structural (red) channel on the right,
both in grey and labelled (`channel = "green"` or `"red"` for one alone). Contrast uses the QC
notebook's formula and defaults (black 1 %, white 99.8 %, gamma 1.0), but the percentiles are
taken per channel over a sample of the movie's own *frames*, not of the mean image, and fixed
for the whole movie, so slow brightness changes stay visible. Single frames are noisier than
the mean image, so the same percentiles give a wider range (a dimmer picture) than in the QC
panels, most of all in the unaveraged movie; each stage and version has its own range. Each
frame is labelled with the stage, the time from the start of the run, its frames and the
playback speed.

The TIFF stages take the channel order and file names from the run's own
`pipeline_settings.json`, not the current config. A run given by path without one (made by
the QC notebook, say) takes its channel order and acquired TIFF from its bidiphase TIFF's
`.phase.json`, whether distortion correction ran from its `.distortion.json`, and the rest
from the config, with a warning; if the acquired TIFF is no longer where the record says, the
file of that name beside the variant folder is used (`raw_tiff` overrides both). A run
processed without distortion correction has no undistorted stage (set
`stage_undistorted = false` for it). Every stage has the run's frames (with `max_timepoints`,
the first frames of the acquired TIFF), so a frame number is the same moment in each movie.
Existing movies are refused unless `overwrite = true`, and every stage's file must exist
before anything is written.

With `suite2p_crop = true` each registered movie also gets a copy (`_cropped_` in its name,
"cropped" in its label) cropped to the part of the frame that no frame's rigid or nonrigid
shifts wrap into: Suite2p shifts frames with a circular roll, so with large motion the edges
of the registered movie show pixels from the opposite edge. The region is computed from the
run's `ops.npy` (`qc.unwrapped_region`, as in the QC notebook) over the whole recording, so
every cropped movie of a run has the same size, and the copy takes its contrast from the
cropped frames. Only the registered stage is cropped: the earlier stages are not
motion-corrected, so they have no wrapped edges, and a fixed crop would not follow the
tissue. The cropped copy reads the registered binaries once more.

Single 7.5 Hz frames are mostly shot noise: for slides, bins of 5-10 frames with
`spatial_sigma_px = 1` at 4-10x speed usually read best. Movies must play at >= 1 frame/s.
Each movie reads its frames once more for itself (the unaveraged and averaged versions
separately), so whole-run movies read about 2 x 14 GB per TIFF stage over the share, and a
TIFF is read page by page from its start, so frames late in a run take longer to reach. Prefer
frame ranges for the TIFF stages; an unaveraged movie of a whole run is also large (1-2 GB).

### NATEX: transients, SNR and cleaned events per ROI

`mini2p-natex` runs the calcium-trace part of NATEX, the MINI2P toolbox's post-Suite2p app
(Zong et al., Cell 2022; [kavli-ntnu/MINI2P_toolbox](https://github.com/kavli-ntnu/MINI2P_toolbox)),
ported from MATLAB:

```bash
uv run mini2p-natex --config config/pipeline.toml \
    --plane-dir /path/to/run-01/with_distortion_correction/without_destriping/aligned_structural/suite2p/plane0
```

Per ROI, from the neuropil-corrected trace (`[fluorescence] neuropil_coefficient`, unless
`[natex] neuropil_coefficient` is set):

- **ΔF/F**: F0 is the 8th percentile in a running 30 s window, lifted by the mean distance
  between the trace and that percentile in the quietest stretches (local SD in the lowest
  10 % of its range);
- **significant transients**: the 2 Hz low-passed ΔF/F above twice its local SD for at least
  0.75 s, widened by 0.4 s on either side;
- **SNR**: median of the top 10 % of ΔF/F within transients minus the median outside them,
  over the SD of the whole ΔF/F;
- **cleaned events**: `spks` kept only within transients, minus their mean outside them,
  outliers capped, scaled to the peak ΔF/F; the **event count** is the frames with an event;
- **bad**: raw F zero throughout.

Results go to `<plane>/natex/` (or `--output-dir`): `natex_dff.npy`, `natex_f0.npy`,
`natex_dff_lowpassed.npy`, `natex_significant.npy`, `natex_events.npy` (ROIs x frames,
rows in `natex_rois.npy`), `natex_cells.csv` (SNR, event count, fraction of time in
transients, fraction of frames with F0 <= 0, bad) and `natex_settings.json`, with
`natex_qc.pdf`: SNR and event-count histograms, the ROIs coloured by SNR, and example
cells from highest to lowest SNR with
their transients shaded. The parameters are NATEX's defaults, set in `[natex]`; all ROIs are
used (`only_iscell = true` keeps Suite2p "cells" only, as NATEX does). Low SNR or few
transients flag ROIs to look at, or to leave out; the ΔF/F here is an alternative to the
maximin ΔF/F0 of the QC notebook, not a replacement.

Each step follows the MATLAB code. `smooth_percentile`, the transient loops of
`DetaF_T_baselinecorrect`, `CalculateSNR` and NATEXplorer's event clean-up are tested against
line-by-line transcriptions; `movstd` and `lowpass` against their documented behaviour. The
differences:

- MATLAB's `lowpass` (a minimum-order equiripple FIR) is approximated by a Kaiser-window FIR
  with the same band edges and attenuation, and the trace is reflected at its ends where
  MATLAB pads with zeros, so the first and last ~7 s can differ;
- ΔF/F is NaN where F0 <= 0 (MATLAB gives infinities); SNR then uses the other frames, and
  `undefined_fraction` in `natex_cells.csv` says how many frames were left out;
- arrays are saved as float32.

`movstd` with NATEX's non-integer windows (0.75 s and 30 s x 7.52 Hz) is taken to use MATLAB's
half-open window rule, t - k/2 <= j < t + k/2 (5 and 225 frames). One property of NATEX is kept
on purpose: a cell with events in fewer than ~0.1 % of frames loses them all to the outlier cap,
so its event count is 0. NATEX's tracking analysis (DeepLabCut), its combination with imaging,
its detection of one cell in several planes, and its GUI are not ported.

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

## Output data type

Suite2p registers int16 frames: given any other type, its TIFF reader converts each frame
with a plain `astype(np.int16)`, which truncates towards zero (9.7 becomes 9, -0.7
becomes 0) and wraps values beyond -32768..32767 around silently (40000 becomes -25536).
So the pipeline writes the corrected and undistorted TIFFs as int16 itself
(`output_dtype = "int16"`, the default): each value is rounded to the nearest count and
clipped to the int16 range, one frame at a time. If anything is clipped the log warns
with the count, and above `max_clipped_fraction` (default 1e-4) the run stops instead.
Every writer writes to a `.partial` file that replaces the TIFF only once the run has
passed, so a stopped run keeps the previous TIFF and its sidecar together. The
`.phase.json` / `.distortion.json` sidecars record `output_dtype` and `clipped_values`.
Registration (the pipeline and the QC notebook alike) refuses a Suite2p input that is not
int16 when `output_dtype = "int16"`: it can only be a stale file or one written elsewhere
(with `distortion.mode = "cluster_full"` the cluster script, which this repository does
not control); with `"float32"` it only warns. The files are
half the size of float32 ones. `output_dtype = "float32"` writes the floats unchanged and
leaves the conversion to Suite2p (the equivalence tests use it, as the original notebook
did).

Runs processed with float32 TIFFs went through Suite2p's truncation, about -0.5 counts
on average wherever destriping or distortion correction made values non-integer, so they
differ slightly from runs processed since. Their sidecars have no `output_dtype` key; to
compare sessions processed before and after, rerun the older ones (`[batch]
rerun_completed_before`).

## Scan-phase correction

`[phase] method = "per_frame"` (the default) gives every frame its own odd-row shift:

1. **Measure.** One pass over the TIFF computes, for every frame of both channels, the
   even/odd row correlation at every shift within ±`max_shift_px` (each row's mean
   removed first, so stripes and brightness do not count).
2. **Choose.** The structural channel sets a frame's phase where its per-frame phases are
   *consistent* (at least `consistency_threshold` of the `consistency_window_frames`
   around it lie within `consistency_tolerance_px` of their running median) **and** the
   functional channel backs that running median. Backing is a local vote: over the
   `support_window_frames` around a frame, each functional frame's correlation at the
   structural phase minus at the functional long-window phase, as a t-like statistic
   (mean over its standard error). The structural phase is backed where that is at least
   `support_t_threshold`, or where it is within `agreement_px` of the long-window phase,
   unless the vote is at most -`contradict_t_threshold` and the two differ by more than
   `contradict_min_px`. The vote lets short excursions to another phase (the minority
   within the long window) follow the structural channel; a functional stretch with no
   signal gives no vote either way. Each frame then follows the majority of the
   `trust_window_frames` around it, so single doubtful frames do not switch channel.
   Everywhere else the functional channel's `fallback_window_frames` average is used.
   This is decided stretch by stretch, so a structural channel that is good in some
   parts of a session and pure noise in others is handled.
3. **Apply.** A second pass shifts each frame by its phase, destripes if enabled, and
   writes `<stem>_dynamic_bidiphase[_destriped].tif`. The output replaces an existing
   file only once it is complete.

Why both conditions: the structural channel's single frames carry about 15× more line
texture than GCaMP, so where it has tissue it follows phase jumps exactly. Where it
has none, its per-frame phases scatter, which consistency catches; but tissue-free
noise can also sit steadily near 0 px, which only the functional check catches. Picking
between two given shifts needs far less signal than finding the best shift, so a short
functional window is enough for that check.

Outputs beside the TIFF: `.phase.json` (settings, phase range, fraction of frames set
by the structural channel, `functional_stretches` as half-open frame ranges),
`.phase_frames.npz` (per frame: applied phase, which channel, each channel's own
phase, consistency, the functional fallback, correlation lost at the applied phase in
each channel, and the measured curves), and `.phase_frames.{png,pdf,svg}` with a
`.caption.txt`, for screening runs by eye (grey = functional channel used; spikes in
the lower panel = frames not fully fixed). The run log warns when phases reach the
search limit, when the structural channel set fewer than half the frames, and when the
config sets chunked-only settings, which per_frame ignores.

The default thresholds are provisional: they were set from two sessions, one with a
clean structural channel and one with none. A higher
`consistency_threshold` (0.8) or lower `agreement_px` rejects noise more reliably but
also rejects good structural frames around phase jumps, where the functional average
lags. Limitations: where the functional fallback is used, a jump inside its window is
smoothed over; and where the functional channel has too little signal to vote (e.g. it
is only noise), it can neither back nor contradict the structural channel, so the
structural phase is used only where it is also within `agreement_px` of the functional
long-window phase; the log warns when the structural channel set under half the frames.

`method = "chunked"` estimates one phase per chunk of `chunk_frames` (its settings and
the manual overrides apply only to it). A run's `pipeline_settings.json` without a
`method` is read as chunked.

## Layout

| Module | Layer | Contents |
|---|---|---|
| `config.py` | boundary | pydantic settings; output naming |
| `tiff_io.py` | I/O | chunked reads/writes of interleaved TIFFs |
| `bidiphase.py` | analysis | phase score, estimators, channel selection (chunked) |
| `frame_phase.py` | analysis | per-frame line correlation, consistency, channel choice, per-frame shifts |
| `destripe.py` | analysis | even/odd line-profile removal |
| `correction.py` | analysis + I/O | per-frame (two-pass) and per-chunk correction; TIFF, `.phase.json`, `.phase_frames.npz` writers |
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
| `natex.py` | analysis + I/O | NATEX port: ΔF/F, transients, SNR, cleaned events; results and QC PDF |
| `movies.py` | analysis + I/O | movies of each stage: frames, contrast, names (pure `plan_movie`); stage readers; MP4 writing |

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

1. Both the chunked `adjacent_line_score` and the per-frame curves pair each odd row
   only with the row above it, which biases the estimate on oblique structure. Line-locked stripes, which are still
   present during estimation when destriping is off, can also capture the peak.
2. `adjacent_line_score` returns `-inf` for a flat or non-finite image. If every
   candidate scores `-inf`, the chunk silently gets `-max_shift_px`.
3. `score_margin` compares the best shift with its immediate neighbour, so it
   measures how sharp the peak is rather than whether a rival peak exists.
4. Integer-only shifts make the chunked second pass nearly always return 0.
5. `two_step_registration = true` with `keep_raw_binary = false`. Suite2p's
   two-step registration needs the raw binary, so check whether it is being
   skipped.
6. With `output_dtype = "float32"`, values beyond the int16 range would wrap when
   Suite2p converts them (it uses a plain `astype(np.int16)`). The default `"int16"`
   rounds and clips them in the pipeline instead, and records how many were clipped.
