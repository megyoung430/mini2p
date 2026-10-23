"""Per-frame bidiphase on a TIFF: two passes, the written frames and the sidecars."""

import json

import numpy as np
import pytest
import tifffile
from pydantic import ValidationError

from mini2p_preprocessing import correction as correction_module
from mini2p_preprocessing.config import (
    ChannelSettings,
    DestripeSettings,
    PhaseSettings,
    PipelineSettings,
    settings_from_record,
)
from mini2p_preprocessing.correction import measure_line_curves, preprocess_tiff
from mini2p_preprocessing.destripe import detrend_chunk
from mini2p_preprocessing.frame_phase import apply_frame_phases
from tests.conftest import write_interleaved_tiff
from tests.test_frame_phase import frames_with_phases

TRUE = np.r_[np.full(37, -3), np.full(53, 5)]  # a jump that is not on any read boundary


def settings(**phase) -> PipelineSettings:
    base = {
        "method": "per_frame",
        "max_shift_px": 8,
        "consistency_window_frames": 9,
        "trust_window_frames": 11,
        "fallback_window_frames": 31,
        "frames_per_read": 16,
    }
    return PipelineSettings.model_validate(
        {"phase": {**base, **phase}, "distortion": {"mode": "off"}}
    )


@pytest.fixture
def jump_recording(tmp_path, rng):
    """Channel 1 structural (clean tissue), channel 2 functional (noisy), shared jump."""
    structural = 200 + 400 * frames_with_phases(rng, TRUE, noise_sd=0.1)
    functional = 200 + 250 * frames_with_phases(rng, TRUE, noise_sd=2.0)
    stacks = [np.clip(s, -32768, 32767).astype(np.int16) for s in (structural, functional)]
    return write_interleaved_tiff(tmp_path / "raw_00001.tif", stacks), stacks


def read_pages(path):
    with tifffile.TiffFile(path) as tif:
        return np.stack([page.asarray() for page in tif.pages])


def read_channels(path, n_channels=2):
    pages = read_pages(path)
    return [pages[ch::n_channels] for ch in range(n_channels)]


def test_every_frame_is_shifted_by_its_own_phase(tmp_path, jump_recording):
    raw, stacks = jump_recording
    out = tmp_path / "corrected.tif"
    summary = preprocess_tiff(raw, out, settings(), run_name="run-01")
    assert np.array_equal(summary.frame_phases.phase_px, TRUE)
    written = read_channels(out)
    for channel, stack in enumerate(stacks):
        np.testing.assert_array_equal(written[channel], apply_frame_phases(stack, TRUE))


def test_destriping_follows_the_per_frame_shift(tmp_path, jump_recording):
    raw, stacks = jump_recording
    out = tmp_path / "corrected.tif"
    destriped = settings().model_copy(update={"destripe": DestripeSettings(enabled=True)})
    preprocess_tiff(raw, out, destriped, run_name="run-01")
    expected = detrend_chunk(apply_frame_phases(stacks[0], TRUE))
    np.testing.assert_allclose(read_channels(out)[0], expected, rtol=1e-6, atol=1e-3)


def test_reading_in_smaller_pieces_gives_the_same_result(tmp_path, jump_recording):
    raw, _ = jump_recording
    outputs = []
    for frames_per_read in (7, 90):
        out = tmp_path / f"corrected_{frames_per_read}.tif"
        preprocess_tiff(raw, out, settings(frames_per_read=frames_per_read), run_name="run-01")
        outputs.append(read_pages(out))
    np.testing.assert_array_equal(outputs[0], outputs[1])


def test_sidecars_record_the_choice_and_its_evidence(tmp_path, jump_recording):
    raw, _ = jump_recording
    out = tmp_path / "corrected.tif"
    preprocess_tiff(raw, out, settings(), run_name="run-01")
    meta = json.loads(out.with_suffix(".phase.json").read_text())
    assert meta["phase_method"] == "per_frame"
    assert meta["input_tiff"] == str(raw)
    assert meta["structural_channel_one_based"] == 1
    assert meta["functional_channel_one_based"] == 2
    assert meta["n_frames"] == len(TRUE)
    assert meta["phase_range_px"] == [-3, 5]
    assert meta["fraction_structural"] == 1.0
    assert meta["functional_stretches"] == []
    frames = np.load(out.with_suffix(".phase_frames.npz"))
    assert np.array_equal(frames["phase_px"], TRUE)
    assert frames["line_curves"].shape == (len(TRUE), 2, 17)
    assert np.allclose(frames["structural_loss"], 0, atol=1e-6)  # structural was fixed exactly
    for key in (
        "used_structural",
        "structural_px",
        "functional_px",
        "structural_consistency",
        "fallback_px",
        "functional_loss",
        "shifts_px",
    ):
        assert key in frames


def test_curves_measured_beforehand_are_used_as_given(tmp_path, jump_recording):
    raw, stacks = jump_recording
    curves = measure_line_curves(raw, settings())
    assert curves.shape == (len(TRUE), 2, 17)
    flat = np.zeros_like(curves)
    flat[:, :, 8 + 2] = 1  # every frame "measured" at +2 px
    out = tmp_path / "corrected.tif"
    summary = preprocess_tiff(raw, out, settings(), run_name="run-01", line_curves=flat)
    assert np.all(summary.frame_phases.phase_px == 2)


def test_max_timepoints_limits_both_passes(tmp_path, jump_recording):
    raw, _ = jump_recording
    out = tmp_path / "corrected.tif"
    short = settings().model_copy(update={"max_timepoints": 20})
    summary = preprocess_tiff(raw, out, short, run_name="run-01")
    assert len(summary.frame_phases.phase_px) == 20
    assert len(read_pages(out)) == 40


def test_curves_of_the_wrong_length_are_refused(tmp_path, jump_recording):
    raw, _ = jump_recording
    curves = measure_line_curves(raw, settings())[:-1]
    with pytest.raises(ValueError, match="line_curves have shape"):
        preprocess_tiff(raw, tmp_path / "c.tif", settings(), run_name="run-01", line_curves=curves)


def test_line_curves_are_refused_by_the_chunked_method(tmp_path, jump_recording):
    raw, _ = jump_recording
    chunked = settings(method="chunked")
    with pytest.raises(ValueError, match="per_frame"):
        preprocess_tiff(
            raw, tmp_path / "c.tif", chunked, run_name="run-01", line_curves=np.zeros((90, 2, 17))
        )


def test_per_frame_is_the_default():
    assert PhaseSettings().method == "per_frame"


@pytest.mark.parametrize(
    "field", ["consistency_window_frames", "trust_window_frames", "fallback_window_frames"]
)
def test_windows_must_be_odd(field):
    with pytest.raises(ValidationError, match="odd"):
        PhaseSettings(**{field: 10})


def test_manual_chunk_phases_need_the_chunked_method():
    with pytest.raises(ValidationError, match="chunked"):
        PhaseSettings(manual_first_phase_by_start={0: 3})
    PhaseSettings(method="chunked", manual_first_phase_by_start={0: 3})


# ---- regressions -----------------------------------------------------------------------


def test_curves_measured_with_another_search_range_are_refused(tmp_path, jump_recording):
    raw, _ = jump_recording
    curves = measure_line_curves(raw, settings(max_shift_px=6))
    with pytest.raises(ValueError, match="max_shift_px"):
        preprocess_tiff(raw, tmp_path / "c.tif", settings(), run_name="run-01", line_curves=curves)
    assert not (tmp_path / "c.tif").exists()


def test_a_failed_rerun_leaves_the_previous_output_intact(tmp_path, jump_recording, monkeypatch):
    raw, _ = jump_recording
    out = tmp_path / "corrected.tif"
    preprocess_tiff(raw, out, settings(), run_name="run-01")
    before = out.read_bytes()

    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(correction_module, "write_interleaved", broken)
    with pytest.raises(OSError, match="disk full"):
        preprocess_tiff(raw, out, settings(), run_name="run-01")
    assert out.read_bytes() == before
    assert not list(tmp_path.glob("*.partial"))


def test_frames_at_the_search_boundary_are_flagged(tmp_path, rng, caplog):
    true = np.full(60, 7)  # max_shift 8, boundary_warning 2: 7 is within the margin
    stacks = [
        np.clip(200 + 400 * frames_with_phases(rng, true, noise_sd=0.1), -32768, 32767).astype(
            np.int16
        )
        for _ in range(2)
    ]
    raw = write_interleaved_tiff(tmp_path / "raw_00002.tif", stacks)
    out = tmp_path / "corrected.tif"
    with caplog.at_level("WARNING"):
        preprocess_tiff(raw, out, settings(), run_name="run-01")
    assert json.loads(out.with_suffix(".phase.json").read_text())["frames_at_search_boundary"] == 60
    assert "max_shift_px" in caplog.text


def test_chunked_only_settings_are_reported_as_ignored(tmp_path, jump_recording, caplog):
    raw, _ = jump_recording
    with caplog.at_level("WARNING"):
        preprocess_tiff(
            raw, tmp_path / "c.tif", settings(alignment_source="functional"), run_name="run-01"
        )
    assert "ignores these chunked-only settings: alignment_source" in caplog.text


def test_a_mostly_untrusted_structural_channel_is_reported(tmp_path, rng, caplog):
    structural = 200 + 50 * rng.standard_normal((90, 48, 96))  # no tissue at all
    functional = 200 + 250 * frames_with_phases(rng, TRUE, noise_sd=2.0)
    stacks = [np.clip(s, -32768, 32767).astype(np.int16) for s in (structural, functional)]
    raw = write_interleaved_tiff(tmp_path / "raw_00003.tif", stacks)
    with caplog.at_level("WARNING"):
        preprocess_tiff(raw, tmp_path / "c.tif", settings(), run_name="run-01")
    assert "structural channel set the phase for only" in caplog.text


def test_a_chunked_rerun_removes_the_per_frame_sidecars(tmp_path, jump_recording):
    raw, _ = jump_recording
    out = tmp_path / "corrected.tif"
    preprocess_tiff(raw, out, settings(), run_name="run-01")
    out.with_suffix(".phase_frames.png").write_bytes(b"plot")
    preprocess_tiff(raw, out, settings(method="chunked", chunk_frames=30), run_name="run-01")
    assert not out.with_suffix(".phase_frames.npz").exists()
    assert not out.with_suffix(".phase_frames.png").exists()


def test_one_channel_recordings_use_that_channel(tmp_path, rng):
    stack = np.clip(200 + 400 * frames_with_phases(rng, TRUE, noise_sd=0.1), -32768, 32767)
    raw = write_interleaved_tiff(tmp_path / "raw_00004.tif", [stack.astype(np.int16)])
    one = settings().model_copy(
        update={
            "channels": ChannelSettings(n_channels=1, structural_channel=1, functional_channel=1)
        }
    )
    out = tmp_path / "c.tif"
    summary = preprocess_tiff(raw, out, one, run_name="run-01")
    assert not summary.frame_phases.has_structural
    assert np.all(np.abs(summary.frame_phases.phase_px - TRUE) <= 1)
    record = np.load(out.with_suffix(".phase_frames.npz"))
    assert "structural_px" not in record and "functional_loss" in record


def test_old_settings_records_are_read_as_chunked():
    old = {
        "phase": {"chunk_frames": 50, "manual_first_phase_by_start": {"0": 3}},
        "distortion": {"mode": "off"},
    }
    loaded = settings_from_record(json.dumps(old))
    assert loaded.phase.method == "chunked"
    assert loaded.phase.manual_first_phase_by_start == {0: 3}


def test_new_settings_records_keep_their_method():
    assert settings_from_record(settings().model_dump_json()).phase.method == "per_frame"


def test_a_noise_only_functional_channel_is_reported(tmp_path, caplog):
    # Nothing can say which channel is right when they disagree, so it must at least be loud.
    rng = np.random.default_rng(3)  # a draw where the noise disagrees with structural mostly
    structural = 200 + 400 * frames_with_phases(rng, np.full(90, 7), noise_sd=0.1)
    functional = 200 + 50 * rng.standard_normal((90, 48, 96))
    stacks = [np.clip(s, -32768, 32767).astype(np.int16) for s in (structural, functional)]
    raw = write_interleaved_tiff(tmp_path / "raw_00005.tif", stacks)
    with caplog.at_level("WARNING"):
        summary = preprocess_tiff(raw, tmp_path / "c.tif", settings(), run_name="run-01")
    chosen = summary.frame_phases
    assert chosen.used_structural.mean() < 0.5
    assert "structural channel set the phase for only" in caplog.text
    assert np.all(chosen.phase_px[chosen.used_structural] == 7)
