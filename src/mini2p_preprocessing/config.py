"""Validated pipeline settings, loaded once at the boundary and passed explicitly."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ChannelRole = Literal["structural", "functional"]
WITHOUT_DESTRIPING_SUBDIR = "without_destriping"
WITH_DESTRIPING_SUBDIR = "with_destriping"
VARIANT_SUBDIRS = (WITHOUT_DESTRIPING_SUBDIR, WITH_DESTRIPING_SUBDIR)


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
    """Odd/even-line scan-phase estimation, applied per chunk of timepoints."""

    chunk_frames: int = Field(100, ge=1)
    estimator: Literal["wide", "suite2p"] = "wide"
    max_shift_px: int = Field(40, ge=1)
    # None crops max_shift_px ('wide') or 10 px ('suite2p') from both image edges.
    score_crop_px: int | None = Field(None, ge=0)
    boundary_warning_px: int = Field(2, ge=0)
    alignment_source: Literal["best_score", "structural", "functional"] = "functional"
    # In 'best_score' mode the functional channel wins only by more than this margin.
    score_switch_margin: float = 0.0
    manual_first_phase_by_start: dict[int, int] = Field(default_factory=dict)
    manual_second_phase_by_start: dict[int, int] = Field(default_factory=dict)
    run_second_pass: bool = True
    second_pass_mode: Literal["chunked", "whole"] = "chunked"
    # None samples every frame for the 'whole' residual (can use a lot of RAM).
    second_pass_global_max_frames: int | None = Field(None, ge=1)


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
    motion_alignment_source: ChannelRole = "functional"
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


class PipelineSettings(_Frozen):
    channels: ChannelSettings = ChannelSettings()
    phase: PhaseSettings = PhaseSettings()
    destripe: DestripeSettings = DestripeSettings()
    distortion: DistortionSettings
    registration: RegistrationSettings = RegistrationSettings()
    fluorescence: FluorescenceSettings = FluorescenceSettings()
    output_dtype: Literal["float32"] = "float32"
    max_timepoints: int | None = Field(None, ge=1)  # small values make quick test runs
    overwrite: bool = True

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
    def registration_channel(self) -> int:
        return self.channels.channel_by_role[self.registration.motion_alignment_source]


NON_PIPELINE_SECTIONS = ("batch", "exports", "qc")


def load_settings(path: Path, **overrides: object) -> PipelineSettings:
    """Read the pipeline sections of a TOML config; keyword overrides replace whole sections.

    The [batch], [exports] and [qc] tables in the same file are read by their own loaders.
    """
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    for section in NON_PIPELINE_SECTIONS:
        raw.pop(section, None)
    raw.update(overrides)
    return PipelineSettings.model_validate(raw)


def with_destriping(settings: PipelineSettings, enabled: bool) -> PipelineSettings:
    destripe = settings.destripe.model_copy(update={"enabled": enabled})
    return settings.model_copy(update={"destripe": destripe})


def variant_output_dir(run_dir: Path, destripe_enabled: bool) -> Path:
    """``run_dir/with_destriping`` or ``run_dir/without_destriping``, so both variants coexist."""
    subdir = WITH_DESTRIPING_SUBDIR if destripe_enabled else WITHOUT_DESTRIPING_SUBDIR
    return Path(run_dir) / subdir


def check_variant_folder(output_dir: Path, destripe_enabled: bool) -> None:
    """Refuse a with_/without_destriping folder that disagrees with the destriping setting."""
    name = Path(output_dir).name
    if name in VARIANT_SUBDIRS and name != variant_output_dir(Path(), destripe_enabled).name:
        state = "on" if destripe_enabled else "off"
        raise ValueError(f"Output folder {output_dir} is {name}/ but destriping is {state}")


def run_name_from_output_dir(output_dir: Path) -> str:
    output_dir = Path(output_dir)
    return output_dir.parent.name if output_dir.name in VARIANT_SUBDIRS else output_dir.name


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
    valid_frames_key: str = "valid_frames"


def load_qc_run(path: Path) -> QCRun:
    with Path(path).open("rb") as handle:
        return QCRun.model_validate(tomllib.load(handle)["qc"])
