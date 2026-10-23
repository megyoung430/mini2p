"""Writing frames as int16 for Suite2p: rounded and clipped here, never truncated or wrapped."""

import json

import numpy as np
import pytest
import tifffile

from mini2p_preprocessing.config import PipelineSettings
from mini2p_preprocessing.correction import preprocess_tiff
from mini2p_preprocessing.destripe import detrend_chunk
from mini2p_preprocessing.distortion import write_distortion_corrected
from mini2p_preprocessing.frame_phase import apply_frame_phases
from mini2p_preprocessing.registration import check_suite2p_input, run_registration
from mini2p_preprocessing.tiff_io import (
    TooMuchClipped,
    check_clipped,
    read_all_channels,
    to_output_dtype,
    write_interleaved,
)
from tests.conftest import fake_calibration, write_interleaved_tiff
from tests.test_frame_phase import frames_with_phases


def read_pages(path):
    with tifffile.TiffFile(path) as tif:
        return np.stack([page.asarray() for page in tif.pages])


# ---- conversion -----------------------------------------------------------------------


def test_int16_output_rounds_to_the_nearest_count():
    out, clipped = to_output_dtype(np.array([[9.7, -0.7, 3.2, -3.2]], np.float32), np.int16)
    assert out.dtype == np.int16 and clipped == 0
    assert out.tolist() == [[10, -1, 3, -3]]  # Suite2p's own astype would give 9, 0, 3, -3


def test_values_that_round_into_range_are_not_clipped():
    values = np.array([40000.0, -40000.0, 32767.4, -32768.4, 32767.5, 100.0], np.float32)
    out, clipped = to_output_dtype(values, np.int16)
    assert out.tolist() == [32767, -32768, 32767, -32768, 32767, 100]
    assert clipped == 3  # 40000, -40000 and 32767.5 (which rounds to 32768)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_no_float_type_can_wrap(dtype):
    big = np.array([np.finfo(dtype).max, -np.finfo(dtype).max], dtype)
    out, clipped = to_output_dtype(big, np.int16)
    assert out.tolist() == [32767, -32768] and clipped == 2


@pytest.mark.parametrize("dtype", [np.uint16, np.int32])
def test_wider_integers_are_clipped_not_wrapped(dtype):
    out, clipped = to_output_dtype(np.array([40000, 5], dtype), np.int16)
    assert out.tolist() == [32767, 5] and clipped == 1


def test_int16_input_passes_through_unchanged():
    values = np.array([-32768, 0, 32767], np.int16)
    out, clipped = to_output_dtype(values, np.int16)
    assert np.array_equal(out, values) and clipped == 0


def test_float32_output_is_unchanged():
    values = np.array([1.25, 40000.5], np.float32)
    out, clipped = to_output_dtype(values, np.float32)
    assert out.dtype == np.float32 and np.array_equal(out, values) and clipped == 0


@pytest.mark.parametrize("dtype", [np.int16, np.float32])
def test_values_that_are_not_finite_are_refused(dtype):
    with pytest.raises(ValueError, match="finite"):
        to_output_dtype(np.array([1.0, np.nan], np.float32), dtype)


def test_write_interleaved_reports_and_clips_per_value(tmp_path):
    stacks = [np.full((2, 3, 4), 50000.0, np.float32), np.full((2, 3, 4), 7.6, np.float32)]
    out = tmp_path / "out.tif"
    with tifffile.TiffWriter(out) as writer:
        clipped = write_interleaved(writer, stacks, np.dtype("int16"))
    assert clipped == 2 * 3 * 4
    with tifffile.TiffFile(out) as tif:
        assert tif.pages[0].dtype == np.int16
        back = read_all_channels(tif, 0, 2, 2)
    assert np.all(back[0] == 32767) and np.all(back[1] == 8)


def test_a_few_clipped_values_warn_and_many_stop_the_run(caplog):
    with caplog.at_level("WARNING"):
        check_clipped(3, 1_000_000, "x.tif", np.int16, max_fraction=1e-4)
    assert "3 of 1000000" in caplog.text
    with pytest.raises(TooMuchClipped, match="max_clipped_fraction"):
        check_clipped(1000, 1_000_000, "x.tif", np.int16, max_fraction=1e-4)
    check_clipped(0, 10, "x.tif", np.int16, max_fraction=0.0)  # nothing clipped: fine


def test_int16_is_the_default_output():
    assert PipelineSettings.model_validate({"distortion": {"mode": "off"}}).dtype == np.int16


# ---- every writer -----------------------------------------------------------------------


def per_frame_settings(**changes) -> PipelineSettings:
    phase = {"max_shift_px": 8, "trust_window_frames": 11, "fallback_window_frames": 21}
    return PipelineSettings.model_validate(
        {"phase": phase, "distortion": {"mode": "off"}, **changes}
    )


def float_recording(tmp_path, rng, bright=()):
    """Two float32 channels; *bright* lists (frame, row, col) set to 40000 in channel 2."""
    true = np.zeros(40, int)
    structural = (200 + 400 * frames_with_phases(rng, true, noise_sd=0.1)).astype(np.float32)
    functional = (200 + 250 * frames_with_phases(rng, true, noise_sd=2.0)).astype(np.float32)
    for index in bright:
        functional[index] = 40000.0
    return write_interleaved_tiff(tmp_path / "raw_00006.tif", [structural, functional])


def test_per_frame_correction_clips_records_and_reads_back(tmp_path, rng, caplog):
    raw = float_recording(tmp_path, rng, bright=[(5, 10, 20)])
    out = tmp_path / "corrected.tif"
    with caplog.at_level("WARNING"):
        summary = preprocess_tiff(raw, out, per_frame_settings(), run_name="run-01")
    pages = read_pages(out)
    assert pages.dtype == np.int16
    assert pages[2 * 5 + 1, 10, 20] == 32767  # frame 5, channel 2: clipped, not wrapped
    meta = json.loads(out.with_suffix(".phase.json").read_text())
    assert (meta["output_dtype"], meta["clipped_values"]) == ("int16", 1)
    assert summary.clipped_values == 1
    assert f"1 of {40 * 2 * 48 * 96}" in caplog.text


def test_destriped_output_is_the_rounded_float_result(tmp_path, recording):
    raw, stacks = recording
    out = tmp_path / "corrected.tif"
    settings = per_frame_settings(destripe={"enabled": True})
    summary = preprocess_tiff(raw, out, settings, run_name="run-01")
    expected = detrend_chunk(apply_frame_phases(stacks[0], summary.frame_phases.phase_px))
    np.testing.assert_array_equal(read_pages(out)[0::2], to_output_dtype(expected, np.int16)[0])


def test_too_much_clipping_stops_the_run_and_keeps_the_previous_output(tmp_path, rng):
    raw = float_recording(tmp_path, rng, bright=[(f, slice(None), slice(None)) for f in range(4)])
    out = tmp_path / "corrected.tif"
    out.write_bytes(b"previous")
    with pytest.raises(TooMuchClipped):
        preprocess_tiff(raw, out, per_frame_settings(), run_name="run-01")
    assert out.read_bytes() == b"previous"
    assert not out.with_suffix(".phase.json").exists()


def test_chunked_correction_writes_int16_and_records_it(tmp_path, rng, caplog):
    raw = float_recording(tmp_path, rng, bright=[(3, 1, 2), (30, 4, 5)])
    out = tmp_path / "corrected.tif"
    settings = PipelineSettings.model_validate(
        {
            "phase": {"method": "chunked", "chunk_frames": 20, "max_shift_px": 8},
            "distortion": {"mode": "off"},
        }
    )
    with caplog.at_level("WARNING"):
        summary = preprocess_tiff(raw, out, settings, run_name="run-01")
    assert read_pages(out).dtype == np.int16
    meta = json.loads(out.with_suffix(".phase.json").read_text())
    assert (meta["output_dtype"], meta["clipped_values"]) == ("int16", 2)
    assert summary.clipped_values == 2
    assert f"2 of {40 * 2 * 48 * 96}" in caplog.text


def test_distortion_correction_clips_exactly_what_overflows(tmp_path, recording, caplog):
    raw, stacks = recording
    bright = [s.astype(np.float32) for s in stacks]
    bright[0][3, 20, 40] = 50000.0  # inside the cropped frame
    source = write_interleaved_tiff(tmp_path / "float_in.tif", bright)
    distortion = {"mode": "local_batched", "transform_mat": "calib.mat"}
    float_out, int_out = tmp_path / "float.tif", tmp_path / "int16.tif"
    write_distortion_corrected(
        source,
        float_out,
        PipelineSettings.model_validate({"distortion": distortion, "output_dtype": "float32"}),
        calibration=fake_calibration(),
    )
    reference = read_pages(float_out)
    with caplog.at_level("WARNING"):
        write_distortion_corrected(
            source,
            int_out,
            PipelineSettings.model_validate({"distortion": distortion}),
            calibration=fake_calibration(),
        )
    written = read_pages(int_out)
    expected, n_clipped = to_output_dtype(reference, np.int16)
    assert n_clipped >= 1 and np.array_equal(written, expected)
    assert written.min() >= -32768 and np.all(written[reference > 32767.5] == 32767)
    meta = json.loads(int_out.with_suffix(".distortion.json").read_text())
    assert (meta["output_dtype"], meta["clipped_values"]) == ("int16", n_clipped)
    assert f"{n_clipped} of {reference.size}" in caplog.text


# ---- a stopped run leaves the previous output and its record together ---------------------


def chunked_settings() -> PipelineSettings:
    return PipelineSettings.model_validate(
        {
            "phase": {"method": "chunked", "chunk_frames": 20, "max_shift_px": 8},
            "distortion": {"mode": "off"},
        }
    )


def test_a_stopped_chunked_run_keeps_the_previous_tiff_and_record(tmp_path, rng, recording):
    clean, _ = recording
    out = tmp_path / "corrected.tif"
    preprocess_tiff(clean, out, chunked_settings(), run_name="run-01")
    before_tiff, before_meta = out.read_bytes(), out.with_suffix(".phase.json").read_text()
    flooded = float_recording(
        tmp_path, rng, bright=[(f, slice(None), slice(None)) for f in range(4)]
    )
    with pytest.raises(TooMuchClipped):
        preprocess_tiff(flooded, out, chunked_settings(), run_name="run-01")
    assert out.read_bytes() == before_tiff
    assert out.with_suffix(".phase.json").read_text() == before_meta
    assert not list(tmp_path.glob("*.partial"))


def test_a_stopped_distortion_run_keeps_the_previous_tiff_and_record(tmp_path, recording):
    raw, stacks = recording
    distortion = {"mode": "local_batched", "transform_mat": "calib.mat"}
    settings = PipelineSettings.model_validate({"distortion": distortion})
    out = tmp_path / "undistorted.tif"
    write_distortion_corrected(raw, out, settings, calibration=fake_calibration())
    before_tiff = out.read_bytes()
    before_meta = out.with_suffix(".distortion.json").read_text()
    flooded = [np.full(s.shape, 50000.0, np.float32) for s in stacks]
    source = write_interleaved_tiff(tmp_path / "flooded.tif", flooded)
    with pytest.raises(TooMuchClipped):
        write_distortion_corrected(source, out, settings, calibration=fake_calibration())
    assert out.read_bytes() == before_tiff
    assert out.with_suffix(".distortion.json").read_text() == before_meta
    assert not list(tmp_path.glob("*.partial"))


# ---- what Suite2p is given --------------------------------------------------------------


def suite2p_input(tmp_path, dtype):
    frames = np.full((4, 2, 8, 8), 100, dtype).reshape(8, 8, 8)
    return write_interleaved_tiff(tmp_path / f"in_{np.dtype(dtype).name}.tif", [frames])


def test_suite2p_input_must_be_int16_when_int16_is_asked_for(tmp_path):
    settings = per_frame_settings()
    check_suite2p_input(suite2p_input(tmp_path, np.int16), settings)
    with pytest.raises(ValueError, match="float32"):
        check_suite2p_input(suite2p_input(tmp_path, np.float32), settings)


def test_a_float32_suite2p_input_only_warns_when_float32_is_asked_for(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        check_suite2p_input(
            suite2p_input(tmp_path, np.float32), per_frame_settings(output_dtype="float32")
        )
    assert "truncat" in caplog.text


def test_registration_checks_its_input_before_running_suite2p(tmp_path, monkeypatch):
    import importlib

    module = importlib.import_module("suite2p.run_s2p")
    calls = []
    monkeypatch.setattr(module, "run_s2p", lambda settings, db: calls.append(db))
    tiff = suite2p_input(tmp_path, np.float32)
    with pytest.raises(ValueError, match="int16"):
        run_registration(tiff, tmp_path, per_frame_settings())
    assert calls == []
