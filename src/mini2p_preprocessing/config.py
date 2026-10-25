"""Validated pipeline settings, loaded once at the boundary and passed explicitly."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ChannelRole = Literal["structural", "functional"]
# "auto": structural where the bidiphase step trusted it often enough, else functional.
MotionSource = Literal["structural", "functional", "auto"]
WITHOUT_DESTRIPING_SUBDIR = "without_destriping"
WITH_DESTRIPING_SUBDIR = "with_destriping"
VARIANT_SUBDIRS = (WITHOUT_DESTRIPING_SUBDIR, WITH_DESTRIPING_SUBDIR)
# The level above the destriping folder: run-XX/<distortion folder>/<destriping folder>.
WITH_DISTORTION_SUBDIR = "with_distortion_correction"
NO_DISTORTION_SUBDIR = "without_distortion_correction"
DISTORTION_SUBDIRS = (WITH_DISTORTION_SUBDIR, NO_DISTORTION_SUBDIR)
# Every variant folder, relative to run-XX/
VARIANTS = tuple(f"{d}/{v}" for d in DISTORTION_SUBDIRS for v in VARIANT_SUBDIRS)
# Suite2p's output sits one level below the variant folder, named for the channel it was
# aligned on: <variant>/aligned_structural/suite2p/plane0. The corrected TIFFs do not depend
# on that channel, so both alignments share them.
ALIGNMENT_SUBDIRS: dict[str, str] = {
    "structural": "aligned_structural",
    "functional": "aligned_functional",
}
SETTINGS_RECORD = "pipeline_settings.json"  # the exact settings each run used


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ChannelSettings(_Frozen):
    """Interleaved page order: page index = channel + n_channels * timepoint."""

    n_channels: int = Field(2, ge=1)
    structural_channel: int = Field(1, ge=1)  # red, 1-based (Suite2p numbering)
    functional_channel: int = Field(2, ge=1)  # green, the calcium/ROI channel

    @model_validator(mode="after")
    def _channels_exist(self) -> ChannelSettings:
        for name in ("structural_channel", "functional_channel"):
            if getattr(self, name) > self.n_channels:
                raise ValueError(
                    f"{name}={getattr(self, name)} exceeds n_channels={self.n_channels}"
                )
        return self

    @property
    def channel_by_role(self) -> dict[str, int]:
        return {"functional": self.functional_channel, "structural": self.structural_channel}

    @property
    def names(self) -> dict[int, str]:
        return {
            self.structural_channel: f"Channel {self.structural_channel} — red / structural",
            self.functional_channel: f"Channel {self.functional_channel} — green / functional",
        }


class PhaseSettings(_Frozen):
    """Odd/even-line scan-phase estimation.

    ``method = "per_frame"`` measures every frame's phase in both channels and uses the
    structural channel wherever it is trustworthy, the functional channel over a longer
    window elsewhere (see ``frame_phase.py``). ``method = "chunked"`` estimates one phase
    per chunk of ``chunk_frames`` timepoints; the remaining fields apply only to it.
    """

    method: Literal["per_frame", "chunked"] = "per_frame"
    # ---- per_frame
    # Structural is trusted where at least consistency_threshold of the
    # consistency_window_frames around a frame lie within consistency_tolerance_px of
    # their running median, and the functional channel backs that median: over
    # support_window_frames it prefers that phase to its own phase over
    # fallback_window_frames with t >= support_t_threshold, or the two are within
    # agreement_px and it does not prefer its own phase with t >= contradict_t_threshold
    # (when they differ by more than contradict_min_px).
    # Most of trust_window_frames must agree.
    consistency_window_frames: int = Field(9, ge=3)
    consistency_tolerance_px: int = Field(2, ge=0)
    consistency_threshold: float = Field(0.7, gt=0, le=1)
    agreement_px: int = Field(6, ge=0)
    support_window_frames: int = Field(15, ge=3)
    support_t_threshold: float = Field(1.5, gt=0)
    contradict_t_threshold: float = Field(2.0, gt=0)
    # A contradiction counts only when the two phases differ by more than this: oriented
    # structure can bias the functional channel's own peak by a pixel or two.
    contradict_min_px: int = Field(2, ge=0)
    trust_window_frames: int = Field(25, ge=1)
    fallback_window_frames: int = Field(101, ge=1)
    # Frames read at once; memory only, the result is the same. Measuring uses about
    # 3 x frames_per_read x frame size x 8 bytes per channel (512 x 512: ~300 MB at 50).
    frames_per_read: int = Field(50, ge=1)
    # Frames (per_frame) or chunks (chunked) within this many px of ±max_shift_px are flagged.
    boundary_warning_px: int = Field(2, ge=0)
    # ---- chunked
    chunk_frames: int = Field(100, ge=1)
    estimator: Literal["wide", "suite2p"] = "wide"
    max_shift_px: int = Field(40, ge=1)
    # None crops max_shift_px ('wide') or 10 px ('suite2p') from both image edges.
    score_crop_px: int | None = Field(None, ge=0)
    alignment_source: Literal["best_score", "structural", "functional"] = "functional"
    # In 'best_score' mode the functional channel wins only by more than this margin.
    score_switch_margin: float = 0.0
    manual_first_phase_by_start: dict[int, int] = Field(default_factory=dict)
    manual_second_phase_by_start: dict[int, int] = Field(default_factory=dict)
    run_second_pass: bool = True
    second_pass_mode: Literal["chunked", "whole"] = "chunked"
    # None samples every frame for the 'whole' residual (can use a lot of RAM).
    second_pass_global_max_frames: int | None = Field(None, ge=1)

    @model_validator(mode="after")
    def _per_frame_settings_make_sense(self) -> PhaseSettings:
        windows = (
            "consistency_window_frames",
            "support_window_frames",
            "trust_window_frames",
            "fallback_window_frames",
        )
        for name in windows:
            if getattr(self, name) % 2 == 0:
                raise ValueError(
                    f"phase.{name} must be odd (centred on the frame), got {getattr(self, name)}"
                )
        manual = self.manual_first_phase_by_start or self.manual_second_phase_by_start
        if self.method == "per_frame" and manual:
            raise ValueError(
                "phase.manual_*_phase_by_start sets a phase per chunk, which only "
                'method = "chunked" uses; remove them or set method = "chunked"'
            )
        return self


class DestripeSettings(_Frozen):
    enabled: bool = False
    preserve_frame_mean: bool = True


class DistortionSettings(_Frozen):
    # Required: registering undistorted data by accident must not be a silent default.
    mode: Literal["local_batched", "cluster_full", "off"]
    transform_mat: Path | None = None
    # Folder holding distortion_correct_full_recording.py; None searches ./scripts then ./
    calibration_script_dir: Path | None = None
    z_um: float = 0.0
    interpolation_order: int = Field(1, ge=0, le=5)
    crop_black_borders: bool = True
    crop_padding_px: int = Field(1, ge=0)
    batch_frames: int = Field(500, ge=1)

    @model_validator(mode="after")
    def _local_mode_needs_transform(self) -> DistortionSettings:
        if self.mode == "local_batched" and self.transform_mat is None:
            raise ValueError("distortion.mode='local_batched' requires distortion.transform_mat")
        return self


class RegistrationSettings(_Frozen):
    fs_hz: float = Field(7.52, gt=0)  # frames/s per channel
    diameter_px: int = Field(12, ge=1)
    nonrigid: bool = True
    norm_frames: bool = False
    nimg_init: int = Field(1000, ge=1)
    th_badframes: float = Field(0.7, gt=0)
    two_step_registration: bool = True
    batch_size: int = Field(500, ge=1)
    keep_raw_binary: bool = False
    # One plain folder name inside the output folder; it is cleared when re-registering.
    save_folder: str = Field("suite2p", pattern=r"^[A-Za-z0-9_.-]+$")
    run_roi_detection: bool = False
    motion_alignment_source: MotionSource = "auto"
    # "auto" aligns on the structural channel when the per-frame bidiphase step trusted it
    # for at least this fraction of frames (it is textured and consistent there).
    auto_structural_min_fraction: float = Field(0.5, gt=0, le=1)
    # Local folder for Suite2p's working binaries (data.bin, data_chan2.bin), e.g. on the
    # laptop's SSD; results still go beside the run. Each run gets its own subfolder.
    # Unset: the binaries sit in the run's suite2p folder.
    fast_disk: Path | None = None

    @field_validator("save_folder")
    @classmethod
    def _plain_folder_name(cls, value: str) -> str:
        if value in {".", ".."}:
            raise ValueError("save_folder must name a folder inside the output folder")
        return value


class FluorescenceSettings(_Frozen):
    """Neuropil correction, F0 baseline, ΔF/F0 and Cellpose-mask extraction."""

    neuropil_coefficient: float = Field(0.7, ge=0)
    # Suite2p's "constant" baseline takes one minimum over all ROIs together, so it
    # is not a per-cell F0 and is deliberately not offered here.
    baseline: Literal["maximin", "prctile"] = "maximin"
    win_baseline_s: float = Field(60.0, gt=0)  # maximin min/max window
    sig_baseline_frames: float = Field(10.0, gt=0)  # maximin Gaussian smoothing
    prctile_baseline: float = Field(8.0, ge=0, le=100)
    # ΔF/F0 is NaN wherever F0 <= min_f0; a non-positive F0 has no meaningful ratio.
    min_f0: float = 0.0
    tau_s: float = Field(1.0, gt=0)  # indicator decay time used by OASIS deconvolution
    batch_size: int = Field(500, ge=1)
    torch_device: str = "cpu"  # "cuda" on a GPU machine; "cpu" works everywhere
    # Green/red median filter before the ratio. Its width is int(fs * window) | 1
    # frames, so at 7.52 Hz the default 0.25 s is 1 frame: no smoothing.
    gr_median_window_s: float = Field(0.25, gt=0)
    gr_red_floor_percentile: float = Field(5.0, ge=0, le=100)
    gr_red_floor_min: float = Field(1e-3, gt=0)
    # Aligning masks drawn on another registration's mean image (see alignment.py).
    mask_max_shift_px: int = Field(50, ge=1)  # Suite2p allows ~0.1 x frame size
    mask_shift_tolerance_px: int = Field(1, ge=0)  # allowed quadrant disagreement
    mask_min_correlation: float = Field(0.5, ge=0, le=1)  # rank correlation after the shift
    mask_min_retained_fraction: float = Field(0.5, gt=0, le=1)  # else the mask is dropped
    # Frames read at once for the background traces and the background-corrected TIFF
    # (~1 GB of float64 per 500 frames of 512 x 512).
    background_chunk_frames: int = Field(100, ge=1)


class FocusSettings(_Frozen):
    """Per-frame focus QC: registered frames that still do not match the mean (focus.py)."""

    channel: ChannelRole = "structural"  # its signal does not follow activity
    smooth_px: float = Field(2.0, ge=0)  # Gaussian sigma, so shot noise does not dominate
    # Running-median baseline (odd), ~40 s at 7.52 Hz. Changes slower than this, and the
    # middle of a bad stretch longer than about half of it, are not flagged.
    window_frames: int = Field(301, ge=3)
    drop_threshold: float = Field(4.0, gt=0)  # robust SDs below the baseline to flag
    merge_gap_frames: int = Field(8, ge=0)  # flagged runs this close count as one stretch
    # Frames read from the binary at once; peak memory is ~20 bytes per pixel per frame
    # (~0.5 GB for 100 frames of 478 x 501).
    chunk_frames: int = Field(100, ge=1)

    @field_validator("window_frames")
    @classmethod
    def _odd_window(cls, value: int) -> int:
        if value % 2 == 0:
            raise ValueError(f"window_frames must be odd so the window is centred, got {value}")
        return value


class PipelineSettings(_Frozen):
    channels: ChannelSettings = ChannelSettings()
    phase: PhaseSettings = PhaseSettings()
    destripe: DestripeSettings = DestripeSettings()
    distortion: DistortionSettings
    registration: RegistrationSettings = RegistrationSettings()
    fluorescence: FluorescenceSettings = FluorescenceSettings()
    focus: FocusSettings = FocusSettings()
    # TIFFs written for Suite2p, which registers int16: "int16" rounds and clips here (and
    # says how much was clipped) instead of leaving Suite2p to truncate and wrap silently.
    output_dtype: Literal["int16", "float32"] = "int16"
    # Above this fraction of clipped values a run stops instead of warning.
    max_clipped_fraction: float = Field(1e-4, ge=0, le=1)
    max_timepoints: int | None = Field(None, ge=1)  # small values make quick test runs
    overwrite: bool = True

    @model_validator(mode="after")
    def _auto_alignment_needs_per_frame_phase(self) -> PipelineSettings:
        # "auto" reads how often the per-frame bidiphase step trusted the structural channel.
        if self.registration.motion_alignment_source == "auto" and self.phase.method != "per_frame":
            raise ValueError(
                'registration.motion_alignment_source = "auto" needs phase.method = '
                '"per_frame"; with "chunked", set "structural" or "functional"'
            )
        return self

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.output_dtype)

    @property
    def correction_suffix(self) -> str:
        return "dynamic_bidiphase" + ("_destriped" if self.destripe.enabled else "")

    @property
    def phase_alignment_channel(self) -> int | None:
        """1-based channel used for phase, or None when chosen per chunk."""
        return self.channels.channel_by_role.get(self.phase.alignment_source)

    @property
    def registration_channel(self) -> int | None:
        """1-based channel Suite2p aligns on; None for "auto", decided at registration."""
        source = self.registration.motion_alignment_source
        return None if source == "auto" else self.channels.channel_by_role[source]


NON_PIPELINE_SECTIONS = ("batch", "exports", "qc", "movies", "natex")


def settings_from_record(text: str) -> PipelineSettings:
    """Settings from a run's ``pipeline_settings.json``.

    Records written before ``phase.method`` existed describe chunked runs, so a record
    without it is read as ``method = "chunked"`` rather than today's default.
    """
    record = json.loads(text)
    record.setdefault("phase", {}).setdefault("method", "chunked")
    # Records from before "auto" existed aligned on the then-default functional channel.
    record.setdefault("registration", {}).setdefault("motion_alignment_source", "functional")
    return PipelineSettings.model_validate(record)


def load_settings(path: Path, **overrides: object) -> PipelineSettings:
    """Read the pipeline sections of a TOML config; keyword overrides replace whole sections.

    The [batch], [exports], [qc], [movies] and [natex] tables are read by their own loaders.
    """
    with Path(path).open("rb") as handle:
        return settings_from_toml(tomllib.load(handle), **overrides)


def settings_from_toml(raw: dict, **overrides: object) -> PipelineSettings:
    """The pipeline settings of an already-parsed config (see ``load_settings``)."""
    raw = {k: v for k, v in raw.items() if k not in NON_PIPELINE_SECTIONS}
    raw.update(overrides)
    return PipelineSettings.model_validate(raw)


def with_destriping(settings: PipelineSettings, enabled: bool) -> PipelineSettings:
    destripe = settings.destripe.model_copy(update={"enabled": enabled})
    return settings.model_copy(update={"destripe": destripe})


def distortion_corrected(settings: PipelineSettings) -> bool:
    """Whether *settings* correct distortion (``[distortion] mode`` is not "off")."""
    return settings.distortion.mode != "off"


def variant_subdir(destripe_enabled: bool, *, distortion_corrected: bool) -> str:
    """The variant folder, relative to ``run-XX/``: ``with_destriping`` or
    ``without_destriping`` inside ``with_distortion_correction/`` or
    ``without_distortion_correction/``, so every combination has its own folder."""
    subdir = WITH_DESTRIPING_SUBDIR if destripe_enabled else WITHOUT_DESTRIPING_SUBDIR
    distortion = WITH_DISTORTION_SUBDIR if distortion_corrected else NO_DISTORTION_SUBDIR
    return f"{distortion}/{subdir}"


def settings_variant(settings: PipelineSettings) -> str:
    """The variant folder that *settings* write to."""
    return variant_subdir(
        settings.destripe.enabled, distortion_corrected=distortion_corrected(settings)
    )


def variant_output_dir(
    run_dir: Path, destripe_enabled: bool, *, distortion_corrected: bool
) -> Path:
    """``run_dir/<variant_subdir>``, so every variant coexists."""
    return Path(run_dir) / variant_subdir(
        destripe_enabled, distortion_corrected=distortion_corrected
    )


def variant_label(variant: str) -> str:
    """A variant as one file-name part: ``without_distortion_correction_with_destriping``."""
    return variant.replace("/", "_")


def run_dir_of(variant_dir: Path) -> Path:
    """The ``run-XX/`` folder of a variant folder (two levels up), or the parent of a plain
    output folder."""
    variant_dir = Path(variant_dir)
    parent = variant_dir.parent
    return parent.parent if parent.name in DISTORTION_SUBDIRS else parent


def alignment_dir(variant_dir: Path, source: ChannelRole) -> Path:
    """``<variant>/aligned_<source>``, which holds Suite2p's output aligned on *source*."""
    if source not in ALIGNMENT_SUBDIRS:
        raise ValueError(f"source must be one of {sorted(ALIGNMENT_SUBDIRS)}, not {source!r}")
    return Path(variant_dir) / ALIGNMENT_SUBDIRS[source]


def plane_dir_of(variant_dir: Path, source: ChannelRole, save_folder: str = "suite2p") -> Path:
    return alignment_dir(variant_dir, source) / save_folder / "plane0"


def variant_dir_of(plane_dir: Path) -> Path:
    """The variant folder of ``<variant>/aligned_<source>/<save_folder>/plane0``."""
    plane_dir = Path(plane_dir)
    if len(plane_dir.parents) < 3 or plane_dir.parents[1].name not in ALIGNMENT_SUBDIRS.values():
        raise ValueError(
            f"{plane_dir} is not inside an {' or '.join(ALIGNMENT_SUBDIRS.values())} folder"
        )
    return plane_dir.parents[2]


def find_registered_plane(
    variant_dir: Path, alignment: MotionSource = "auto", save_folder: str = "suite2p"
) -> Path | None:
    """The registered plane folder (with an ops.npy) of a variant, or None if there is none.

    "auto" takes whichever alignment exists and refuses to guess when both do.
    """
    sources = list(ALIGNMENT_SUBDIRS) if alignment == "auto" else [alignment]
    found = [
        plane
        for plane in (plane_dir_of(variant_dir, s, save_folder) for s in sources)
        if (plane / "ops.npy").is_file()
    ]
    if len(found) > 1:
        raise ValueError(
            f"{variant_dir} has registrations aligned on both channels ({found}); set "
            "alignment to 'structural' or 'functional' to pick one"
        )
    return found[0] if found else None


def check_variant_folder(
    output_dir: Path, destripe_enabled: bool, *, distortion_corrected: bool
) -> None:
    """Refuse a variant folder that disagrees with the destriping or distortion setting, and
    the ``with_/without_distortion_correction/`` folders themselves, which only hold
    variants."""
    output_dir = Path(output_dir)
    expected = variant_subdir(destripe_enabled, distortion_corrected=distortion_corrected)
    if output_dir.name in DISTORTION_SUBDIRS:
        raise ValueError(f"{output_dir} only holds variant folders; write to .../{expected}/")
    if output_dir.name not in VARIANT_SUBDIRS:
        return  # a plain output folder, as for tests or one-off runs
    actual = output_dir.relative_to(run_dir_of(output_dir)).as_posix()
    if actual != expected:
        destripe = "on" if destripe_enabled else "off"
        distortion = "on" if distortion_corrected else "off"
        raise ValueError(
            f"Output folder {output_dir} is {actual}/ but destriping is {destripe} and "
            f"distortion correction is {distortion}, which write to {expected}/"
        )


def run_name_from_output_dir(output_dir: Path) -> str:
    output_dir = Path(output_dir)
    return run_dir_of(output_dir).name if output_dir.name in VARIANT_SUBDIRS else output_dir.name


def corrected_tiff_path(output_dir: Path, input_tiff: Path, settings: PipelineSettings) -> Path:
    return Path(output_dir) / f"{Path(input_tiff).stem}_{settings.correction_suffix}.tif"


def undistorted_tiff_path(output_dir: Path, input_tiff: Path, settings: PipelineSettings) -> Path:
    return (
        Path(output_dir) / f"{Path(input_tiff).stem}_{settings.correction_suffix}_undistorted.tif"
    )


class QCRun(_Frozen):
    """The single recording inspected interactively in the QC notebook."""

    input_tiff: Path
    run_dir: Path  # e.g. derivatives/.../funcimg/run-02; the variant subfolder is added
    qc_start_timepoint: int = Field(0, ge=0)
    qc_registration_frames: int = Field(500, ge=1)
    qc_distortion_frames: int = Field(100, ge=1)
    cellpose_seg: Path | None = None  # Cellpose *_seg.npy file
    # Image the masks were drawn on, when it comes from a different registration
    # than this run (e.g. first 1000 frames vs full session); masks are shifted to fit.
    cellpose_image: Path | None = None
    valid_frames: Path | None = None  # .npz from behaviour alignment; None audits every frame
    # Cellpose *_seg.npy of blood vessels: their fluorescence is the background regressed
    # out of F and Fneu before ΔF/F0. Drawn on this registration's mean image, or set
    # background_image to the image it was drawn on to shift it onto this plane.
    background_seg: Path | None = None
    background_image: Path | None = None
    valid_frames_key: str = "valid_frames"


def load_qc_run(path: Path) -> QCRun:
    with Path(path).open("rb") as handle:
        return QCRun.model_validate(tomllib.load(handle)["qc"])
