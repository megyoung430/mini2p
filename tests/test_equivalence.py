"""The refactored package must reproduce the original notebook's outputs exactly."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import suite2p
import tifffile

from mini2p_preproc.config import PipelineSettings
from mini2p_preproc.correction import preprocess_tiff
from mini2p_preproc.distortion import write_distortion_corrected
from mini2p_preproc.registration import build_suite2p_db, build_suite2p_settings
from tests.reference import original_notebook as original

SCENARIOS = {
    "defaults": {},
    "best_score_whole_destripe_manual": {
        "phase": {
            "alignment_source": "best_score",
            "score_switch_margin": 0.01,
            "second_pass_mode": "whole",
            "second_pass_global_max_frames": 17,
            "manual_first_phase_by_start": {20: -4},
            "manual_second_phase_by_start": {40: 1},
        },
        "destripe": {"enabled": True},
    },
    "suite2p_structural_single_pass_no_mean": {
        "phase": {
            "estimator": "suite2p",
            "alignment_source": "structural",
            "run_second_pass": False,
            "score_crop_px": 4,
        },
        "destripe": {"enabled": True, "preserve_frame_mean": False},
        "max_timepoints": 45,
    },
}


def settings_for(overrides: dict) -> PipelineSettings:
    base = {"phase": {"chunk_frames": 20, "max_shift_px": 8}, "distortion": {"mode": "off"}}
    merged = {**base, **overrides, "phase": {**base["phase"], **overrides.get("phase", {})}}
    return PipelineSettings.model_validate(merged)


def configure_original(s: PipelineSettings, run_name: str = "run-01") -> None:
    p, d, r = s.phase, s.destripe, s.registration
    original.configure(
        N_CHANNELS=s.channels.n_channels,
        STRUCTURAL_CHANNEL=s.channels.structural_channel,
        FUNCTIONAL_CHANNEL=s.channels.functional_channel,
        PHASE_ALIGNMENT_SOURCE=p.alignment_source,
        PHASE_SCORE_SWITCH_MARGIN=p.score_switch_margin,
        MOTION_ALIGNMENT_SOURCE=r.motion_alignment_source,
        PHASE_CHUNK_FRAMES=p.chunk_frames,
        PHASE_ESTIMATOR=p.estimator,
        PHASE_MAX_SHIFT_PX=p.max_shift_px,
        PHASE_SCORE_CROP_PX=p.score_crop_px,
        PHASE_BOUNDARY_WARNING_PX=p.boundary_warning_px,
        MANUAL_FIRST_PHASE_BY_START=dict(p.manual_first_phase_by_start),
        MANUAL_SECOND_PHASE_BY_START=dict(p.manual_second_phase_by_start),
        RUN_SECOND_BIDIPHASE_PASS=p.run_second_pass,
        SECOND_PASS_MODE=p.second_pass_mode,
        SECOND_PASS_GLOBAL_MAX_FRAMES=p.second_pass_global_max_frames,
        RUN_DESTRIPING=d.enabled,
        PRESERVE_FRAME_MEAN=d.preserve_frame_mean,
        OUTPUT_DTYPE=np.float32,
        MAX_TIMEPOINTS=s.max_timepoints,
        OVERWRITE=True,
        RUN_NAME=run_name,
    )


def read_pages(path: Path) -> np.ndarray:
    with tifffile.TiffFile(path) as tif:
        return np.stack([page.asarray() for page in tif.pages])


@pytest.mark.parametrize("name", SCENARIOS)
def test_preprocess_matches_original(tmp_path, recording, name):
    raw_tiff, _ = recording
    settings = settings_for(SCENARIOS[name])
    configure_original(settings)
    (tmp_path / "old").mkdir()
    (tmp_path / "new").mkdir()
    old_out, new_out = tmp_path / "old" / "out.tif", tmp_path / "new" / "out.tif"

    old_summary = original.preprocess_tiff(raw_tiff, old_out)
    new_summary = preprocess_tiff(raw_tiff, new_out, settings, run_name="run-01")

    assert np.array_equal(read_pages(old_out), read_pages(new_out))
    assert old_summary["global_second_phase_px"] == new_summary.global_second_phase_px
    old_meta = json.loads(old_out.with_suffix(".phase.json").read_text())
    new_meta = json.loads(new_out.with_suffix(".phase.json").read_text())
    assert old_meta.pop("output_tiff") != new_meta.pop("output_tiff")
    assert old_meta == new_meta


@pytest.mark.parametrize("second_pass_mode", ["chunked", "whole"])
def test_best_score_switching_matches_original(tmp_path, mixed_recording, second_pass_mode):
    raw_tiff, _ = mixed_recording
    settings = settings_for(
        {"phase": {"alignment_source": "best_score", "second_pass_mode": second_pass_mode}}
    )
    configure_original(settings)
    old_out, new_out = tmp_path / "old.tif", tmp_path / "new.tif"
    original.preprocess_tiff(raw_tiff, old_out)
    new_summary = preprocess_tiff(raw_tiff, new_out, settings, run_name="run-01")

    sources = [c["first_selected_source"] for c in new_summary.chunks]
    assert sources == ["structural", "functional", "structural"]  # the scenario must switch
    assert [c["first_phase_px"] for c in new_summary.chunks] == [3, 1, 5]
    assert np.array_equal(read_pages(old_out), read_pages(new_out))
    old_meta = json.loads(old_out.with_suffix(".phase.json").read_text())
    new_meta = json.loads(new_out.with_suffix(".phase.json").read_text())
    old_meta.pop("output_tiff"), new_meta.pop("output_tiff")
    assert old_meta == new_meta


def fake_calibration(shape=(48, 96)):
    """Stand-in for distortion_correct_full_recording: a gentle barrel-like warp."""

    def load_transform(path, z_um):
        return {"k": 0.05}, 0.0, "fake-schema", 12

    def make_maps(transform, image_shape):
        yy, xx = np.mgrid[0 : image_shape[0], 0 : image_shape[1]].astype(np.float64)
        cy, cx = (image_shape[0] - 1) / 2, (image_shape[1] - 1) / 2
        r2 = ((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2
        scale = 1 + transform["k"] * r2
        return cy + (yy - cy) * scale, cx + (xx - cx) * scale

    def crop_bounds(map_y, map_x, image_shape, padding):
        return (
            2 + padding,
            image_shape[0] - 2 - padding,
            3 + padding,
            image_shape[1] - 3 - padding,
        )

    return SimpleNamespace(
        load_transform=load_transform, make_maps=make_maps, crop_bounds=crop_bounds
    )


def test_distortion_matches_original(tmp_path, recording):
    raw_tiff, _ = recording
    settings = PipelineSettings.model_validate(
        {"distortion": {"mode": "local_batched", "transform_mat": "calib.mat", "batch_frames": 16}}
    )
    calibration = fake_calibration()
    configure_original(settings)
    original.configure.__globals__.update(
        load_transform=calibration.load_transform,
        make_maps=calibration.make_maps,
        crop_bounds=calibration.crop_bounds,
        DISTORTION_TRANSFORM_MAT=Path("calib.mat"),
        DISTORTION_Z_UM=0,
        DISTORTION_INTERPOLATION_ORDER=1,
        CROP_DISTORTION_BLACK_BORDERS=True,
        DISTORTION_CROP_PADDING_PX=1,
        DISTORTION_BATCH_FRAMES=16,
    )
    old_out, new_out = tmp_path / "old.tif", tmp_path / "new.tif"
    original.write_local_batched_distortion(raw_tiff, old_out)
    write_distortion_corrected(raw_tiff, new_out, settings, calibration=calibration)

    assert np.array_equal(read_pages(old_out), read_pages(new_out))
    old_meta = json.loads(old_out.with_suffix(".distortion.json").read_text())
    new_meta = json.loads(new_out.with_suffix(".distortion.json").read_text())
    old_meta.pop("output_tiff"), new_meta.pop("output_tiff")
    assert old_meta == new_meta


@pytest.mark.parametrize("motion_source", ["functional", "structural"])
def test_suite2p_settings_match_original(tmp_path, motion_source):
    settings = PipelineSettings.model_validate(
        {"registration": {"motion_alignment_source": motion_source}, "distortion": {"mode": "off"}}
    )
    configure_original(settings)
    reg = settings.registration
    original.configure.__globals__.update(
        FS_HZ=reg.fs_hz,
        DIAMETER_PX=reg.diameter_px,
        DO_NONRIGID=reg.nonrigid,
        NORMALIZE_REGISTRATION_FRAMES=reg.norm_frames,
        NIMG_INIT=reg.nimg_init,
        BAD_FRAME_THRESHOLD=reg.th_badframes,
        TWO_STEP_REGISTRATION=reg.two_step_registration,
        REGISTRATION_BATCH_SIZE=reg.batch_size,
        RUN_ROI_DETECTION=reg.run_roi_detection,
        SUITE2P_SAVE_FOLDER=reg.save_folder,
        KEEP_RAW_BINARY=reg.keep_raw_binary,
    )
    tiff = tmp_path / "x_undistorted.tif"
    old_settings, old_db = original.reference_suite2p_settings_and_db(
        suite2p.default_settings, tiff, tmp_path
    )
    assert build_suite2p_settings(settings) == old_settings
    assert build_suite2p_db(tmp_path, tiff, settings) == old_db
