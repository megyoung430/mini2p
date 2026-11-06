"""Per-frame photon-free level: synthetic photon-counting frames with known offsets."""

import numpy as np
import pytest
import tifffile

from mini2p_preprocessing import photon_free_io
from mini2p_preprocessing.config import PipelineSettings
from mini2p_preprocessing.photon_free import (
    dim_pixel_mask,
    photon_free_level,
    photon_free_levels,
    subtract_level,
)

GAIN, READ_NOISE = 350, 20  # counts per photon, electronic noise SD


def photon_frames(levels, photons_per_px, shape=(96, 96), seed=0):
    """Frames whose photon-free readings sit at *levels* (one per frame)."""
    rng = np.random.default_rng(seed)
    rate = np.broadcast_to(photons_per_px, shape)
    photons = rng.poisson(rate, (len(levels), *shape))
    values = photons * GAIN + rng.normal(0, READ_NOISE, photons.shape)
    return np.round(values + np.asarray(levels)[:, None, None]).astype(np.int16)


def test_level_finds_the_photon_free_peak_in_sparse_photons():
    frame = photon_frames([-64], 0.15)[0]
    assert photon_free_level(frame) == pytest.approx(-64, abs=3)


def test_levels_follow_a_drifting_offset_frame_by_frame():
    levels = np.array([-100, -40, 0, 25, 80])
    np.testing.assert_allclose(photon_free_levels(photon_frames(levels, 0.15)), levels, atol=3)


def test_level_refuses_values_with_no_photon_free_peak():
    bright = photon_frames([0], 3.0)[0]  # most pixels hold photons
    with pytest.raises(ValueError, match="photon-free"):
        photon_free_level(bright)


def test_level_refuses_empty_values():
    with pytest.raises(ValueError, match="no values"):
        photon_free_level(np.array([], np.int16))


def test_dim_mask_lets_a_bright_channel_be_measured():
    rate = np.full((96, 96), 3.0)
    rate[:, :12] = 0.1  # a dim strip, e.g. a vessel (12.5% of the pixels)
    frames = photon_frames([30, -20], rate)
    mask = dim_pixel_mask(frames.mean(axis=0), percentile=10)
    assert mask[:, :12].mean() > 0.7 and mask[:, 12:].mean() < 0.02
    np.testing.assert_allclose(photon_free_levels(frames, mask), [30, -20], atol=4)


def test_dim_mask_stays_inside_the_region():
    image = np.arange(100.0).reshape(10, 10)
    mask = dim_pixel_mask(image, percentile=50, region=(slice(5, 10), slice(0, 10)))
    assert not mask[:5].any() and mask[5:].sum() == 25


@pytest.mark.parametrize("percentile", [0, 101])
def test_dim_mask_refuses_bad_percentile(percentile):
    with pytest.raises(ValueError, match="percentile"):
        dim_pixel_mask(np.ones((4, 4)), percentile)


def test_levels_refuse_mask_of_wrong_shape():
    with pytest.raises(ValueError, match="shape"):
        photon_free_levels(photon_frames([0], 0.15), np.ones((3, 3), bool))


def test_subtract_level_removes_the_offset_from_every_trace():
    traces = np.array([[10.0, 20.0, 30.0], [1.0, 2.0, 3.0]])
    np.testing.assert_allclose(
        subtract_level(traces, np.array([5.0, -5.0, 0.0])), [[5, 25, 30], [-4, 7, 3]]
    )
    np.testing.assert_allclose(subtract_level(np.array([4.0, 4.0]), np.array([1.0, 3.0])), [3, 1])


def test_subtract_level_refuses_mismatched_frames_and_nan():
    with pytest.raises(ValueError, match="frames"):
        subtract_level(np.zeros((2, 3)), np.zeros(4))
    with pytest.raises(ValueError, match="finite"):
        subtract_level(np.zeros((2, 2)), np.array([0.0, np.nan]))


# ---- reading the raw TIFF -----------------------------------------------------------


def write_raw(path, red, green):
    """Interleaved pages: red (channel 1) then green (channel 2) for each timepoint."""
    pages = np.empty((2 * len(red), *red.shape[1:]), np.int16)
    pages[0::2], pages[1::2] = red, green
    tifffile.imwrite(path, pages)


def settings(**fluorescence):
    return PipelineSettings.model_validate(
        {
            "distortion": {"mode": "off"},
            "fluorescence": {
                "photon_free_mask_frames": MASK_FRAMES,
                "photon_free_pixel_percentile": 20,  # ~800 px per frame here
                "background_chunk_frames": 4,
                **fluorescence,
            },
        }
    )


@pytest.fixture
def raw_tiff(tmp_path):
    green_levels, red_levels = np.linspace(-90, 10, 10), np.linspace(40, 400, 10)
    red_rate = np.full((96, 96), 2.0)
    red_rate[:, :30] = 0.1  # wide enough to stay dim inside the 16 px margin
    write_raw(
        tmp_path / "run.tif",
        photon_frames(red_levels, red_rate, seed=1),
        photon_frames(green_levels, np.where(np.arange(96) < 30, 0.05, 0.3), seed=2),
    )
    return tmp_path / "run.tif", green_levels, red_levels


MASK_FRAMES = 6


def test_raw_levels_measure_each_channel_per_frame(raw_tiff, tmp_path):
    """Frames after the ones the dim pixels were picked from are measured without bias;
    those frames read a little low here, since among this test's identical strip pixels
    the pick follows their noise (real masks follow structure, over hundreds of frames)."""
    tiff, green, red = raw_tiff
    levels = photon_free_io.raw_levels(tiff, tmp_path, n_frames=10, settings=settings())
    later = slice(MASK_FRAMES, None)
    np.testing.assert_allclose(levels.green[later], green[later], atol=3)
    np.testing.assert_allclose(levels.red[later], red[later], atol=3)
    np.testing.assert_allclose(levels.green, green, atol=8)
    np.testing.assert_allclose(levels.red, red, atol=8)
    assert levels.red_mask[:, 30:].sum() == 0


def test_raw_levels_are_cached_and_recomputed_when_the_tiff_changes(
    raw_tiff, tmp_path, monkeypatch
):
    tiff, *_ = raw_tiff
    first = photon_free_io.raw_levels(tiff, tmp_path, n_frames=10, settings=settings())
    assert (tmp_path / photon_free_io.LEVELS_FILE).is_file()

    def no_read(*args, **kwargs):
        raise AssertionError("read the TIFF again")

    monkeypatch.setattr(photon_free_io, "read_channel_chunk", no_read)
    np.testing.assert_array_equal(
        photon_free_io.raw_levels(tiff, tmp_path, n_frames=10, settings=settings()).green,
        first.green,
    )
    monkeypatch.undo()
    write_raw(
        tiff, photon_frames(np.zeros(10), 0.1, seed=3), photon_frames(np.zeros(10), 0.1, seed=4)
    )
    changed = photon_free_io.raw_levels(tiff, tmp_path, n_frames=10, settings=settings())
    np.testing.assert_allclose(changed.green[MASK_FRAMES:], 0, atol=3)


def test_raw_levels_refuse_a_tiff_with_other_frame_count(raw_tiff, tmp_path):
    tiff, *_ = raw_tiff
    with pytest.raises(ValueError, match="10 timepoints"):
        photon_free_io.raw_levels(tiff, tmp_path, n_frames=12, settings=settings())


def test_traces_and_background_lose_the_level_frame_for_frame(raw_tiff, tmp_path):
    from mini2p_preprocessing.background_io import BackgroundTraces
    from mini2p_preprocessing.plane import PlaneTraces

    tiff, *_ = raw_tiff
    levels = photon_free_io.raw_levels(tiff, tmp_path, n_frames=10, settings=settings())
    ones = np.ones((3, 10), np.float32)
    traces = PlaneTraces(
        ones, 2 * ones, 3 * ones, None, np.ones(3), np.array([]), {}, np.zeros((2, 2))
    )
    cleaned = photon_free_io.traces_without_level(traces, levels)

    def expected(value, level):
        return np.broadcast_to(value - level, (3, 10))

    np.testing.assert_allclose(cleaned.f, expected(1, levels.green), atol=1e-3)
    np.testing.assert_allclose(cleaned.fneu, expected(2, levels.green), atol=1e-3)
    np.testing.assert_allclose(cleaned.f_chan2, expected(3, levels.red), atol=1e-3)
    assert cleaned.fneu_chan2 is None
    background = BackgroundTraces(
        "vessel", np.zeros(10), np.ones(10), np.zeros((2, 2), bool), None, {}
    )
    cleaned_bg = photon_free_io.background_without_level(background, levels)
    np.testing.assert_allclose(cleaned_bg.red, 1 - levels.red)
    with pytest.raises(ValueError, match="cover 10 frames"):
        photon_free_io.traces_without_level(
            PlaneTraces(
                ones[:, :9], ones[:, :9], None, None, np.ones(3), np.array([]), {}, np.zeros((2, 2))
            ),
            levels,
        )


def test_raw_levels_follow_max_timepoints(raw_tiff, tmp_path):
    tiff, green, _ = raw_tiff
    short = settings().model_copy(update={"max_timepoints": 8})
    levels = photon_free_io.raw_levels(tiff, tmp_path, n_frames=8, settings=short)
    assert len(levels.green) == 8
    np.testing.assert_allclose(levels.green[MASK_FRAMES:], green[MASK_FRAMES:8], atol=3)
