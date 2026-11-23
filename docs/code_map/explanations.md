# Plain-language explanations for the code map

These show as "In plain words" dropdowns on the code map page (`scripts/build_code_map.py`). They
are for readers new to the methods: what problem a file solves, how, and how to read what it
returns. Each entry starts with a `## <key>` heading: a file path for a primer on the whole file,
or `path::name` for one function. Write for someone who knows the experiment but not the image
processing; define each term the first time it is used. Keep entries short and accurate, and
update an entry when the code it describes changes.

A few words used throughout:

- **Run**: one raw ScanImage TIFF, its frames interleaving the red (structural) and green
  (functional) channels.
- **Variant**: one way of processing a run, named for its corrections, e.g.
  `with_distortion_correction/with_destriping/`.
- **Registration**: Suite2p's motion correction of a run. Each `save_folder` inside
  `aligned_<channel>/` is one registration, with its own settings and outputs.
- **Plane folder**: a registration's `plane0/`, holding `ops.npy` (Suite2p's record of the
  registration: shifts, mean images, settings), the registered movies `data.bin` (green) and
  `data_chan2.bin` (red), and, after extraction, the traces.
- **ROI**: one cell's mask. **F** is the mean fluorescence inside it per frame; **Fneu** the mean in
  a ring around it (the neuropil).

## src/mini2p_preprocessing/bidiphase.py

A resonant scanner sweeps left to right on one line and right to left on the next. If the timing
of the two directions is off, odd and even lines are shifted sideways against each other, and
every edge in the image looks combed. This file measures that shift (the **bidirectional phase**,
in pixels) on a stretch of frames: it tries each shift, undoes it on the mean frame, and scores
how alike neighbouring lines then are. The best-scoring shift wins. `phase.method = "chunked"`
uses one phase per chunk of frames; the per-frame method (`frame_phase.py`) builds on the same
idea.

## src/mini2p_preprocessing/frame_phase.py

The scan phase can jump from one frame to the next, so the per-frame method measures it in every
frame. The red channel usually has enough line-to-line texture to give each frame's phase
exactly. Where red has no tissue (pure noise) its per-frame answers scatter, so those frames use
the green channel instead, summed over a longer window to have enough signal. The result says,
frame by frame, which phase and which channel were used and why.

## src/mini2p_preprocessing/correction.py

Writes the phase-corrected TIFF. With the per-frame method it reads the raw TIFF twice: once to
measure every frame's line similarity in each channel, once to shift each frame by the phase
chosen for it. Destriping, if on, runs in the same pass. Beside the TIFF it writes `.phase.json`
(settings and summary) and `.phase_frames.npz` (every frame's phase), so the choices can be
checked later.

## src/mini2p_preprocessing/destripe.py

Some recordings carry faint stripes that repeat on every line: electrical pickup locked to the
scan. Forward (even) and reverse (odd) lines pick up different waveforms. Averaged down the
lines of a frame, the stripe pattern stands out from the tissue, which varies from line to line, so
destriping subtracts each frame's mean even-line and mean odd-line column profile, by default only
their shape, keeping the frame's mean brightness.

## src/mini2p_preprocessing/calibration.py

The Mini2P's lens bends the image, more towards the edges. The scope's calibration file holds
pairs of points: where a point really is, and where the scope puts it. A piecewise-affine fit (the
image cut into triangles between the points, each stretched on its own) turns those pairs into a
map saying, for every output pixel, which raw pixel to read. The crop then keeps the largest
rectangle whose every pixel reads from inside the raw frame.

## src/mini2p_preprocessing/distortion.py

Applies the calibration map to every frame (bilinear interpolation by default) and crops it,
writing the `*_undistorted.tif` that Suite2p registers, with a `.distortion.json` saying which
calibration, depth and crop were used.

## src/mini2p_preprocessing/registration.py

Runs Suite2p's motion correction on the corrected TIFF. **Rigid** registration moves each frame by
one whole-pixel shift; **nonrigid** registration also shifts small blocks of the frame by their own
subpixel amounts, interpolating the frame as it goes. Motion is estimated on one channel (usually
red, whose brightness does not change with activity) and the same shifts are applied to the other.
Each `save_folder` keeps its own binaries and its own `motion_alignment.json`, which records which
channel was used and why, and identifies the binaries, so a loader never reads another
registration's movie by mistake.

## src/mini2p_preprocessing/alignment.py

Two registrations of the same recording each align to their own reference image, so their mean
images, and masks drawn on one of them, can be offset by a few pixels. This file finds that offset
by sliding the image the masks were drawn on over the plane's own mean image and keeping the shift
where the two match best (cross-correlation on images with their smooth background removed). It
checks that each quadrant of the image agrees on that shift, and refuses when they do not, when the
images still differ after the shift, or when the shift is larger than allowed: one uniform shift
cannot fix those cases, and the plane should be segmented again.

## src/mini2p_preprocessing/extraction.py

Turns Cellpose masks into traces. Each mask becomes a Suite2p ROI; F is the mean of its pixels in
each registered frame and Fneu the mean of a ring around it, for both channels. The deconvolved
activity (`spks`) comes from Suite2p's own deconvolution (OASIS). Suite2p's own trace files in the plane
folder are replaced, and `cellpose_extraction.json` records which masks and settings were used.

## src/mini2p_preprocessing/fluorescence.py

From raw traces to ΔF/F0. **Neuropil correction** subtracts a fraction of the ring around each cell
(F − c × Fneu), because out-of-focus light from the surroundings also falls on the cell. **F0** is
the cell's baseline, from Suite2p's baseline estimate over a sliding window (by default
"maximin": the running maximum of the running minimum of the smoothed trace). **ΔF/F0** is
how far the trace rises above that baseline, as a fraction of it. Where F0 is at or below
`min_f0` (zero by default), ΔF/F0 is meaningless, so those frames are set to NaN and flagged rather
than divided. A small positive F0 still gives large ΔF/F0 values, so check the F0 overview.

## src/mini2p_preprocessing/background.py

Brightness changes of the whole field (stray light, focus, motion) reach every cell. This file
removes them by regression: a background trace, from a blood vessel (no labelled cells) or from
every pixel outside the ROIs, is fitted to each cell's trace and the fitted part subtracted,
keeping the trace's mean. It can also regress each cell's own red trace out of its green trace,
since red should not follow activity. Each correction can also remove real shared signal, so the
docstring lists its caveats.

## src/mini2p_preprocessing/photon_free.py

A pixel that caught no photon should read zero, but ScanImage's offset is measured once and the
true dark level then drifts, by over 100 counts within a session on the Mini2P. Most pixels in a
frame catch no photon at these light levels, so their readings form the lowest peak of the frame's
histogram, and its centre is the **photon-free level**. Subtracting it frame by frame puts F, Fneu
and the backgrounds on a true zero, which ΔF/F0 needs; it removes no light.

## src/mini2p_preprocessing/natex.py

A port of the calcium-trace part of NATEX, the MINI2P toolbox's analysis app. Per cell it computes
ΔF/F with its own baseline method, finds **significant transients** (stretches where the smoothed
ΔF/F stays well above its local noise for long enough), an **SNR** (how large the transients are
against the trace's spread) and **cleaned events** (deconvolved activity kept only inside
transients). Each function follows the MATLAB code line by line.

## src/mini2p_preprocessing/focus.py

After registration every frame should look like the session's mean image. This file scores each
frame by its correlation with the mean and flags frames that match much worse than their
neighbours. A low score can mean the plane moved in depth, the brain moved during the frame's scan,
or registration picked the wrong shift; the score alone cannot tell which.

## src/mini2p_preprocessing/frame_order.py

Several steps process a run in chunks, and a bookkeeping error (a chunk written twice, a block
moved, swapped channels) would leave the frame count right but the frames out of order. This file
checks the order by content: each frame's mean brightness fluctuates from frame to frame in a way
the processing barely changes, so each stage's fluctuation pattern is matched against the raw
TIFF's, window by window and frame by frame.

## src/mini2p_preprocessing/registration_comparison.py

Measures several registrations of the same recording the same way, so a settings sweep can be
compared without segmenting cells. **Alignment**: how well frames match their mean image, how far
they would still have to move, and how sharp the mean is. **Signal**: from small patches of pixels,
how far the largest green fluctuations rise above the noise, how much the red still fluctuates
(red should not change, so that is motion), and how alike neighbouring pixels' activity is.
**Putative cells** (`cell_snr.py`): resting SNR and transient counts at bright cell-sized spots.
Read the green measures only among registrations the red ones find equally well aligned, and
compare noise-based measures within rigid or within nonrigid registrations: interpolation alone
changes them.

## src/mini2p_preprocessing/cell_snr.py

A segmentation-free stand-in for cells. The spots of the mean image that stand out most from their
surroundings at the size of a cell (peaks of a difference of Gaussians) are taken as cells; each one's trace is the mean of a small disc on it minus the mean of a ring around it, so
anything both share (the drifting dark level, field-wide light) cancels. **Resting SNR** is how far
the disc sits above the ring, in units of the trace's noise: contrast, not activity. **Transients**
are moments when the trace jumps more than 4.5 noise SDs above its slow baseline; the same count on
the upside-down trace says how many such jumps noise alone would give.

## src/mini2p_preprocessing/movies.py

Makes MP4 movies of any processing stage of a run (raw, phase-corrected, undistorted, registered,
background-removed),
of the whole recording or a range of frames, optionally averaged over a few frames and smoothed,
with both channels side by side and the behavior video beside them. Watching a stage is often the
quickest way to see what a correction did.

## src/mini2p_preprocessing/batch.py

Runs the pipeline over every raw TIFF in a range of sessions. Each TIFF becomes `run-01`, `run-02`,
… in acquisition order, and each run's variant folder holds a `source_tiff.json` recording which
TIFF it came from and whether it finished. Relaunching therefore skips finished runs, and a TIFF that would land on
another TIFF's run folder stops the batch instead of overwriting it.
