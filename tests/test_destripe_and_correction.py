import numpy as np
import pytest

from mini2p_preprocessing.config import PipelineSettings
from mini2p_preprocessing.correction import correct_chunk, residual_sample_indices
from mini2p_preprocessing.destripe import detrend_chunk
from tests.conftest import make_recording


def striped_chunk(rng, n_frames=3, ly=8, lx=30):
    base = rng.normal(100, 5, size=(n_frames, ly, lx))
    waveform = 20 * np.sin(np.arange(lx) / 2.0)
    base[:, ::2] += waveform
    base[:, 1::2] -= waveform
    return base.astype(np.float32), waveform


def test_detrend_preserves_frame_mean(rng):
    chunk, _ = striped_chunk(rng)
    out = detrend_chunk(chunk, preserve_frame_mean=True)
    assert np.allclose(out.mean(axis=(1, 2)), chunk.mean(axis=(1, 2)), atol=1e-3)


def test_detrend_removes_line_locked_waveform(rng):
    chunk, waveform = striped_chunk(rng)
    out = detrend_chunk(chunk)
    residual = out[:, ::2].mean(axis=1) - out[:, 1::2].mean(axis=1)
    # Each row set keeps its own mean, so only the waveform's shape must be gone.
    shape_left = residual - residual.mean(axis=1, keepdims=True)
    assert np.abs(shape_left).max() < 1e-3 < np.abs(waveform).max()


def test_detrend_without_preserving_mean_zeroes_every_column(rng):
    chunk, _ = striped_chunk(rng)
    out = detrend_chunk(chunk, preserve_frame_mean=False)
    assert np.allclose(out[:, ::2].mean(axis=1), 0, atol=1e-3)
    assert np.allclose(out[:, 1::2].mean(axis=1), 0, atol=1e-3)


def settings(**phase):
    return PipelineSettings.model_validate(
        {
            "phase": {"method": "chunked", "chunk_frames": 10, "max_shift_px": 8, **phase},
            "registration": {"motion_alignment_source": "functional"},  # "auto" needs per_frame
            "distortion": {"mode": "off"},
        }
    )


def test_correct_chunk_finds_phase_and_records_it(rng):
    chunks = make_recording(rng, [4], chunk_frames=10)
    corrected, record = correct_chunk(chunks, 0, settings())
    assert record["first_phase_px"] == 4
    assert record["second_phase_px"] == 0  # nothing left after an exact integer fix
    assert (record["start"], record["end"]) == (0, 10)
    assert len(corrected) == 2 and corrected[0].dtype == np.float32


def test_manual_overrides_win(rng):
    chunks = make_recording(rng, [4], chunk_frames=10)
    s = settings(manual_first_phase_by_start={30: 1}, manual_second_phase_by_start={30: -2})
    _, record = correct_chunk(chunks, 30, s)
    assert (record["first_phase_px"], record["second_phase_px"]) == (1, -2)


def test_whole_mode_requires_global_residual(rng):
    chunks = make_recording(rng, [0], chunk_frames=10)
    with pytest.raises(ValueError, match="global_residual"):
        correct_chunk(chunks, 0, settings(second_pass_mode="whole"))


def test_residual_sample_indices_spread_and_bounded():
    assert np.array_equal(residual_sample_indices(5, None), np.arange(5))
    idx = residual_sample_indices(1000, 7)
    assert len(idx) == 7 and idx[0] == 0 and idx[-1] == 999
