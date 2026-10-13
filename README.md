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
