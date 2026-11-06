# Settings reference

Every setting lives in one TOML file, `config/pipeline.toml` (git-ignored; copy
`config/pipeline.example.toml` and fill in your paths). The commands and notebooks all read
the same file, so a setting means the same thing everywhere. Settings you leave out take
the default shown here. A misspelled or unknown key is refused when the file is loaded,
so a typo never silently falls back to a default.

Top-level keys must come before the first `[table]` header.

## Which part reads which section

| Section | Read by |
|---|---|
| top level, `[channels]`, `[phase]`, `[destripe]`, `[distortion]`, `[registration]` | the preprocessing: `mini2p-preprocess`, the batch notebook, the QC notebook |
| `[fluorescence]` | extraction (`mini2p-extract`, QC notebook) and the QC notebook's fluorescence audit and background steps |
| `[focus]` | the QC notebook's "frames that still do not match" step |
| `[qc]` | the QC notebook only: which recording to inspect |
| `[batch]` | `batch_and_exports.ipynb` and `post_batch_sessions.ipynb` |
| `[exports]` | the export part of those two notebooks |
| `[movies]` | `mini2p-movie` and the QC notebook's movie cell |
| `[natex]` | `mini2p-natex` |

Each preprocessed run saves the exact settings it used in `pipeline_settings.json`, so
changing the file later never changes what an existing run means.

## What to rerun after changing a setting

| You changed | Rerun from |
|---|---|
| `[phase]`, `[destripe]`, `[channels]`, `output_dtype`, `max_timepoints` | writing the corrected TIFF (everything after it too) |
| `[distortion]` | distortion correction, then registration onwards |
| `[registration]` | registration onwards (the corrected TIFFs are reused) |
| `[fluorescence]` mask alignment (`mask_*`), `tau_s`, `batch_size`, `torch_device` | extraction onwards |
| any other `[fluorescence]` setting, including the photon-free level | only the QC notebook's fluorescence audit and background cells |
| `[focus]` | only the focus QC cell |
| `[movies]`, `[natex]`, `[exports]` | only that command or export |

Preprocessing folders depend on two settings: `[destripe] enabled` and
`[distortion] mode`. Each combination writes to its own variant folder inside `run-XX/`:
`with_distortion_correction/` or `without_distortion_correction/`, then `with_destriping/`
or `without_destriping/`. So changing either never overwrites the other variant.
Suite2p's output sits one level further down, in `aligned_structural/` or
`aligned_functional/`, named for the channel it aligned on.

---

## Top level

| Key | Default | What it does |
|---|---|---|
| `overwrite` | `true` | Replace outputs that already exist. With `false`, a stage whose output exists stops with an error instead. |
| `max_timepoints` | none (all) | Process only the first N timepoints. For quick test runs; every later stage, including the photon-free level, uses the same N. |
| `output_dtype` | `"int16"` | Data type of the TIFFs written for Suite2p. `"int16"` rounds and clips here and reports how much was clipped; `"float32"` leaves the conversion to Suite2p, which truncates and wraps values silently. |
| `max_clipped_fraction` | `1e-4` | If more than this fraction of values had to be clipped to fit `output_dtype`, the run stops instead of warning. |

## `[channels]`

The raw ScanImage TIFF interleaves channels page by page: page = channel + n_channels ×
timepoint.

| Key | Default | What it does |
|---|---|---|
| `n_channels` | `2` | Channels saved in the TIFF. |
| `structural_channel` | `1` | The red, structural channel (1-based, as in Suite2p). |
| `functional_channel` | `2` | The green, calcium channel: the ROI channel. |

## `[phase]`: odd/even-line scan-phase (bidiphase) correction

A resonant scanner draws alternate lines in opposite directions; if their timing is off,
odd and even lines are shifted against each other. This step measures and corrects that
shift.

| Key | Default | What it does |
|---|---|---|
| `method` | `"per_frame"` | `"per_frame"`: every frame gets its own phase. `"chunked"`: one phase per chunk of `chunk_frames` timepoints (the original notebook's method). `motion_alignment_source = "auto"` needs `"per_frame"`. |
| `max_shift_px` | `40` | Largest phase searched, in pixels. Raise it only if phases reach this boundary (the QC notebook warns). |
| `boundary_warning_px` | `2` | Flag frames or chunks whose phase is within this many pixels of ±`max_shift_px`. |

**`per_frame` settings.** Each frame's phase is measured in both channels. The structural
channel sets a frame's phase where it is consistent over time and the functional channel
agrees; elsewhere the functional channel's phase, averaged over a longer window, is used.
All windows are in frames and must be odd, so they are centred on the frame.

| Key | Default | What it does |
|---|---|---|
| `consistency_window_frames` | `9` | Frames whose structural phases are compared to judge consistency. |
| `consistency_tolerance_px` | `2` | A frame "agrees" if its phase is within this many pixels of the running median. |
| `consistency_threshold` | `0.7` | Fraction of the window that must agree for the structural channel to be consistent (tissue gives ~0.9, noise ~0.4). |
| `agreement_px` | `6` | The structural phase is backed if the functional phase is this close to it. |
| `support_window_frames` | `15` | Frames over which the functional channel votes on the structural phase... |
| `support_t_threshold` | `1.5` | ...the vote (t-like) needed to back it... |
| `contradict_t_threshold` | `2.0` | ...and the vote against it that overrules `agreement_px`... |
| `contradict_min_px` | `2` | ...counted only when the two phases differ by more than this (oriented structure can bias the functional peak by a pixel or two). |
| `trust_window_frames` | `25` | A frame follows the majority decision of this many frames around it. |
| `fallback_window_frames` | `101` | Where the structural channel is not trusted, the functional channel's phase is averaged over this many frames. Smaller follows fast changes better but is noisier. |
| `frames_per_read` | `50` | Frames read at once. Memory only (~300 MB per channel at 512 × 512); the result is the same. |

**`chunked` settings** (ignored with `per_frame`; setting the manual tables with
`per_frame` is an error).

| Key | Default | What it does |
|---|---|---|
| `chunk_frames` | `100` | Timepoints per chunk. 10–25 helps when the phase changes quickly. |
| `estimator` | `"wide"` | `"wide"` searches ±`max_shift_px`; `"suite2p"` uses Suite2p's estimator, which searches only ±10 px. |
| `score_crop_px` | none | Pixels cropped from both image edges when scoring. None crops `max_shift_px` (`"wide"`) or 10 px (`"suite2p"`). |
| `alignment_source` | `"functional"` | Channel the phase is measured on: `"functional"`, `"structural"`, or `"best_score"` (whichever scores better per chunk). |
| `score_switch_margin` | `0.0` | With `"best_score"`, the functional channel wins only by more than this margin. |
| `run_second_pass` | `true` | Measure and correct the residual phase after the first correction. |
| `second_pass_mode` | `"chunked"` | Residual per chunk (`"chunked"`) or one for the whole recording (`"whole"`). |
| `second_pass_global_max_frames` | none | Frames sampled for the `"whole"` residual; none uses every frame (can use a lot of memory). |
| `[phase.manual_first_phase_by_start]` | empty | Override a chunk's first-pass phase: `chunk start timepoint = phase in px`, e.g. `200 = -17`. |
| `[phase.manual_second_phase_by_start]` | empty | The same for the second pass. |

## `[destripe]`

| Key | Default | What it does |
|---|---|---|
| `enabled` | `false` | Remove line-synchronous stripes (patterns that repeat in every line, differing by column). Writes to the `with_destriping/` variant, so both versions can coexist. |
| `preserve_frame_mean` | `true` | Keep each frame's mean fluorescence. Must stay `true` with the photon-free level: `false` drives every column to zero mean, which removes the detector offset as well. |

## `[distortion]`: the Mini2P's fixed optical distortion

| Key | Default | What it does |
|---|---|---|
| `mode` | required | `"local_batched"`: correct here, in batches. `"cluster_full"`: use a file corrected on the cluster (it must already exist). `"off"`: no correction (output goes to `without_distortion_correction/`). No default, so registering uncorrected data never happens by accident. |
| `transform_mat` | none | The calibration `.mat` file. Required for `"local_batched"`. |
| `calibration_script_dir` | none | Folder holding `distortion_correct_full_recording.py`; none searches `./scripts`, then `./`. |
| `z_um` | `0` | Imaging depth used to pick the calibration, in µm. |
| `interpolation_order` | `1` | Spline order used to resample the frames (0–5; 1 = bilinear). |
| `crop_black_borders` | `true` | Crop the zero-filled borders the correction leaves. Must stay `true` with the photon-free level, whose offset those borders do not carry. |
| `crop_padding_px` | `1` | Extra pixels cropped beyond the black border. |
| `batch_frames` | `500` | Frames corrected at once (memory only). |

## `[registration]`: Suite2p motion correction

| Key | Default | What it does |
|---|---|---|
| `fs_hz` | `7.52` | Frame rate per channel (Hz). |
| `diameter_px` | `12` | Expected cell diameter (Suite2p's `diameter`). |
| `nonrigid` | `true` | Non-rigid registration: each block of the frame gets its own shift. |
| `norm_frames` | `false` | Suite2p's `norm_frames`: clip frames to the reference's intensity range before computing shifts. |
| `nimg_init` | `1000` | Frames, spread over the recording, used to build the reference image. |
| `th_badframes` | `0.7` | Suite2p's threshold for flagging frames whose registration failed. |
| `two_step_registration` | `true` | Register twice, the second time to a reference built from the once-registered frames. |
| `batch_size` | `500` | Frames registered at once (memory only). |
| `keep_raw_binary` | `false` | Keep Suite2p's unregistered binary as well (another ~4 GB per channel). |
| `save_folder` | `"suite2p"` | Name of Suite2p's output folder. It is cleared when re-registering. |
| `fast_disk` | none | A local folder (e.g. on the laptop's SSD) for Suite2p's working binaries. Much faster and more reliable than a network share; results still go beside the run. |
| `run_roi_detection` | `false` | Run Suite2p's own ROI detection. Off, because ROIs come from Cellpose masks. |
| `motion_alignment_source` | `"auto"` | Channel the motion is measured on: `"structural"`, `"functional"`, or `"auto"`: structural where the per-frame phase step trusted it for at least `auto_structural_min_fraction` of frames, else functional. |
| `auto_structural_min_fraction` | `0.5` | That fraction, for `"auto"`. |

## `[fluorescence]`: extraction, neuropil, F0 and ΔF/F0

**Neuropil and baseline**

| Key | Default | What it does |
|---|---|---|
| `neuropil_coefficient` | `0.7` | Fcorr = F − c · Fneu. |
| `baseline` | `"maximin"` | How F0 is computed from Fcorr: `"maximin"` (Gaussian smoothing, then the maximum of running minima) or `"prctile"` (a running percentile). Suite2p's single-value `"constant"` is not offered, since it is not a per-cell F0. |
| `win_baseline_s` | `60.0` | Window for the running minimum/maximum or percentile, in seconds. |
| `sig_baseline_frames` | `10.0` | Gaussian smoothing before `"maximin"`, in frames. |
| `prctile_baseline` | `8.0` | Percentile used by `"prctile"`. |
| `min_f0` | `0.0` | ΔF/F0 is NaN wherever F0 ≤ this, since the ratio means nothing there. |

**Photon-free level.** A pixel that caught no photon should read 0, but the detector's
zero drifts during a recording (see the README, "Photon-free level").

| Key | Default | What it does |
|---|---|---|
| `subtract_photon_free_level` | `true` | Measure each frame's photon-free level on the raw TIFF and subtract it from F, Fneu and the background traces of each channel before ΔF/F0 (QC notebook). Cached in `photon_free_levels.npz` in the plane folder. |
| `photon_free_pixel_percentile` | `5.0` | The level is measured over each channel's dimmest pixels, this percent of them. |
| `photon_free_mask_frames` | `500` | Those pixels are picked in the mean of this many first frames. |

**Background (QC notebook)**

| Key | Default | What it does |
|---|---|---|
| `background_chunk_frames` | `100` | Frames read at once for the background traces, the background-corrected TIFF and the photon-free level (memory only; ~1 GB of float64 per 500 frames of 512 × 512). |

**Green/red ratio**

| Key | Default | What it does |
|---|---|---|
| `gr_median_window_s` | `0.25` | Median filter on green and red before the ratio, in seconds; its width is int(fs × window) rounded up to odd, so 0.25 s at 7.52 Hz is 1 frame (no filtering). |
| `gr_red_floor_percentile` | `5.0` | Red is floored at this percentile of its own values before dividing, so near-zero red cannot blow up the ratio... |
| `gr_red_floor_min` | `0.001` | ...and never below this. |

**Extraction from Cellpose masks**

| Key | Default | What it does |
|---|---|---|
| `tau_s` | `1.0` | Indicator decay time used by OASIS deconvolution (`spks`), in seconds. |
| `batch_size` | `500` | Frames processed at once during extraction (memory only). |
| `torch_device` | `"cpu"` | `"cuda"` on a GPU machine. |
| `mask_max_shift_px` | `50` | Masks drawn on another registration's mean image are shifted onto this one; the largest shift allowed. |
| `mask_shift_tolerance_px` | `1` | The four quadrants of the image may disagree about that shift by at most this much; otherwise extraction refuses, since one shift cannot fit. |
| `mask_min_correlation` | `0.5` | Rank correlation the two images must reach after the shift. |
| `mask_min_retained_fraction` | `0.5` | Masks keeping less than this fraction of their pixels after the shift (pushed off the edge) are dropped. |

## `[focus]`: frames that still do not match after registration (QC notebook)

| Key | Default | What it does |
|---|---|---|
| `channel` | `"structural"` | Channel compared to the mean image; structural, because its signal does not follow activity. |
| `smooth_px` | `2.0` | Gaussian smoothing before comparing, so shot noise does not dominate. |
| `window_frames` | `301` | Running-median baseline of the match score (odd; ~40 s at 7.52 Hz). Slower changes, and the middle of a bad stretch longer than about half of it, are not flagged. |
| `drop_threshold` | `4.0` | Frames this many robust SDs below the baseline are flagged. |
| `merge_gap_frames` | `8` | Flagged runs this close together count as one stretch. |
| `chunk_frames` | `100` | Frames read at once (memory only; ~0.5 GB for 100 frames of 478 × 501). |

## `[qc]`: the recording the QC notebook inspects

| Key | Default | What it does |
|---|---|---|
| `input_tiff` | required | The raw ScanImage TIFF. Also where the photon-free level is measured. |
| `run_dir` | required | The run folder outputs go into (the variant subfolders are added). Keep it outside `[batch] derivatives_subject_dir`: the batch planner would treat a folder there as a finished run and skip it. |
| `qc_start_timepoint` | `0` | First timepoint of the stretch shown in the stage-by-stage QC figures. |
| `qc_registration_frames` | `500` | Frames in the before/after-registration comparison. |
| `qc_distortion_frames` | `100` | Frames in the before/after-distortion comparison. |
| `cellpose_seg` | none | Cellpose `*_seg.npy` with the cell masks, for extraction. |
| `cellpose_image` | none | Only if the masks were drawn on another registration's mean image: that image. The masks are then shifted to fit this plane. |
| `background_seg` | none | Cellpose `*_seg.npy` of blood vessels: their fluorescence is the vessel background. Without it, only the non-ROI background is used. |
| `background_image` | none | Only if the vessels were drawn on another image: that image. |
| `valid_frames` | none | `.npz` from behavior alignment listing the frames to keep; none audits every frame. |
| `valid_frames_key` | `"valid_frames"` | The array's name in that file. |

## `[batch]`: many sessions

| Key | Default | What it does |
|---|---|---|
| `raw_subject_dir` | required | The subject's raw-data folder. |
| `derivatives_subject_dir` | required | Where the subject's runs are written. |
| `session_first`, `session_last` | required | Session numbers to process (inclusive). |
| `source_substring` | required | Only TIFFs whose names contain this are processed (e.g. `"71rtask"`). |
| `destripe` | `false` | Which destriping variant to write. |
| `stop_on_error` | `true` | Stop the batch at the first failed run. |
| `retry_incomplete_runs` | `true` | Retry runs that failed or were interrupted. |
| `rerun_completed_before` | none | Register again, replacing them, the runs that finished before this time (with a time zone, e.g. `2026-10-21T09:00:00+01:00`). Relaunching with the same time never redoes a run already replaced. Needs `overwrite = true`. |
| `cellpose_seg` | `"suite2p/plane0/meanImg_green_functional_contrast_seg.npy"` | Each run's Cellpose masks, relative to `run-XX/`. |
| `cellpose_image` | `"suite2p/plane0/meanImg_green_functional_contrast.png"` | The image they were drawn on, relative to `run-XX/`. |

## `[exports]`: mean images and PDFs per run

| Key | Default | What it does |
|---|---|---|
| `subject_dir` | required | The subject's derivatives folder. |
| `session_first`, `session_last` | required | Sessions to export. |
| `expected_runs` | required | Exports and PDFs refuse to run on any other number of runs. |
| `run_subdir` | required | The variant to export, e.g. `"with_distortion_correction/without_destriping"`. |
| `alignment` | `"auto"` | Which registration: the one `aligned_*` folder each run has, or `"structural"` / `"functional"`. |
| `black_percentile`, `white_percentile`, `gamma` | `5.0`, `99.5`, `1.2` | Contrast of the exported images. |
| `high_resolution_dpi` | `400` | Resolution of the high-resolution PNGs. |
| `overwrite` | `false` | Replace existing per-run exports. |
| `pdf_overwrite` | `false` | Replace existing PDFs. |

## `[movies]`: `mini2p-movie`

Command-line options override these.

**Which run.** Either by name, all of `cohort`, `subject`, `session`, `run` and
`derivatives_root`, or by path, `run_dir` (which then overrides the others).

| Key | Default | What it does |
|---|---|---|
| `cohort` | none | e.g. `"cohort-01"`. |
| `subject` | none | The subject folder or part of it (`"sub-03"`, `"MY_71_R"`). |
| `session` | none | e.g. `224`. |
| `run` | none | e.g. `1` for `run-01`. |
| `derivatives_root` | none | Holds `cohort-XX/<subject>/ses-*/funcimg/run-XX`. |
| `run_dir` | none | A folder laid out like `run-XX/`. |
| `variant` | `"with_distortion_correction/without_destriping"` | Which preprocessing variant. |
| `alignment` | `"auto"` | Which registration (as in `[exports]`). |
| `save_folder` | `"suite2p"` | Suite2p's output folder inside `aligned_<channel>/`. |
| `raw_tiff` | none | The acquired TIFF, if not where the run's `source_tiff.json` says. |

**Which frames and stages**

| Key | Default | What it does |
|---|---|---|
| `whole_recording` | required | `true`: every frame. `false`: `start_frame` to `end_frame`. No default, because a whole TIFF stage is ~14 GB to read. |
| `start_frame`, `end_frame` | none | 0-based, both included. |
| `stage_raw` | `false` | The acquired TIFF. |
| `stage_bidiphase` | `false` | After phase correction (and destriping). |
| `stage_undistorted` | `false` | After distortion correction too. |
| `stage_registered` | `true` | After Suite2p registration too. |
| `stage_background_removed` | `false` | The registered movie with the background regressed out of every pixel (the QC notebook's `data_background_removed_<source>.tif`). |
| `background_source` | `"vessel"` | That background: `"vessel"` or `"non_roi"`. |
| `unaveraged_movie` | `true` | A movie of every frame. |
| `averaged_movie` | `true` | A movie of frames averaged `average_frames` at a time. |
| `behavior` | `false` | Also each movie with the behavior video on the left, in sync (added by the behavior-movies pull request). |
| `behavior_video` | none | The behavior video, if not where `video_frame_times.npz` says (same pull request). |

**Where to write, and how they look**

| Key | Default | What it does |
|---|---|---|
| `output_dir` | none | Default: `run-XX/<variant>/aligned_<channel>/movies`. |
| `overwrite` | `false` | Replace existing movies. |
| `channel` | `"side_by_side"` | Functional left and structural right, or `"green"` / `"red"` alone. |
| `average_frames` | `5` | Frames per averaged movie frame (2–256). |
| `average_mode` | `"bin"` | `"bin"` (non-overlapping) or `"rolling"`. |
| `speed` | `4.0` | Playback speed, times real time. A movie must play at ≥ 1 fps, so large bins need higher speed. |
| `black_percentile`, `white_percentile`, `gamma` | `1.0`, `99.8`, `1.0` | Contrast, per channel, as in the QC notebook. |
| `contrast_sample_frames` | `200` | Frames sampled to set that contrast. |
| `spatial_sigma_px` | `0.0` | Gaussian smoothing within each averaged frame (unaveraged movies never are). |
| `upscale` | `1` | Each pixel drawn as upscale × upscale pixels (1–8). |
| `label` | `true` | Stage, time, speed and channel written into each frame. |
| `suite2p_crop` | `false` | Also a copy of each registered movie cropped to the region no frame's registration shifts wrap into. |
| `quality` | `8.0` | H.264 quality, 0–10. |

## `[natex]`: `mini2p-natex`

Timing defaults are NATEX's. Unlike NATEX, every Cellpose ROI is kept, and the neuropil
coefficient is `[fluorescence]`'s unless set here. NATEX works on F and Fneu as extracted,
without the photon-free level.

| Key | Default | What it does |
|---|---|---|
| `moving_window_s` | `30.0` | Running window for F0 and the local SD, in seconds. |
| `lowpass_hz` | `2.0` | Low-pass filter used only to find transients. |
| `transient_s` | `0.75` | A significant transient lasts at least this long. |
| `extend_s` | `0.4` | Margin added on both sides of a transient. |
| `neuropil_coefficient` | none | None uses `[fluorescence] neuropil_coefficient`. |
| `only_iscell` | `false` | `true` keeps only Suite2p "cells", as NATEX does. |
