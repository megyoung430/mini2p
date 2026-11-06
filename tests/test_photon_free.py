"""Per-frame photon-free level: synthetic photon-counting frames with known offsets."""

import numpy as np
import pytest
import tifffile

from mini2p_preprocessing import photon_free_io
from mini2p_preprocessing.background_io import BackgroundTraces
from mini2p_preprocessing.config import PipelineSettings
from mini2p_preprocessing.photon_free import (
    check_steps,
    dim_pixel_mask,
    photon_free_level,
    photon_free_levels,
    subtract_level,
)
from mini2p_preprocessing.plane import PlaneTraces

GAIN, READ_NOISE = 350, 20  # mean counts per photon, electronic noise SD


def photon_frames(levels, photons_per_px, shape=(96, 96), seed=0, pulse_cv=0.0):
    """Frames whose photon-free readings sit at *levels* (one per frame).

    *pulse_cv* spreads single-photon pulse heights (gamma, mean GAIN), as a real PMT does.
    """
    rng = np.random.default_rng(seed)
    rate = np.broadcast_to(photons_per_px, shape)
    photons = rng.poisson(rate, (len(levels), *shape))
    if pulse_cv:
        k = 1 / pulse_cv**2  # sum of n gamma(k, GAIN/k) pulses is gamma(n k, GAIN/k)
        signal = np.where(photons > 0, rng.gamma(np.maximum(photons, 1) * k, GAIN / k), 0.0)
    else:
        signal = photons * GAIN
    values = signal + rng.normal(0, READ_NOISE, photons.shape)
    return np.round(values + np.asarray(levels)[:, None, None]).astype(np.int16)


def test_level_finds_the_photon_free_peak_in_sparse_photons():
    frame = photon_frames([-64], 0.15)[0]
    assert photon_free_level(frame) == pytest.approx(-64, abs=2)


def test_levels_follow_a_drifting_offset_frame_by_frame():
    levels = np.array([-100, -70, -40, -10, 25, 60, 80])
    np.testing.assert_allclose(photon_free_levels(photon_frames(levels, 0.15)), levels, atol=2)


@pytest.mark.parametrize("rate, bias", [(0.15, 1.5), (0.5, 3.0)])
def test_level_with_spread_pulse_heights_is_biased_by_little(rate, bias):
    """Small pulses of a real PMT fall inside the window and lift the level slightly."""
    frame = photon_frames([0], rate, shape=(128, 128), pulse_cv=1.0)[0]
    assert abs(photon_free_level(frame)) < bias


@pytest.mark.parametrize("rate", [1.0, 1.1, 1.2, 1.3, 1.5])
def test_level_never_returns_a_one_photon_peak(rate):
    """Near 1 photon per pixel the one-photon peak rivals the photon-free one: the level
    must be the true zero or refused, never ~350 counts off."""
    for seed in range(5):
        frame = photon_frames([0], rate, shape=(128, 128), seed=seed)[0]
        try:
            level = photon_free_level(frame)
        except ValueError:
            assert rate > 1.1  # at <= 1.1 enough readings are photon-free to measure
            continue
        assert level == pytest.approx(0, abs=3)


@pytest.mark.parametrize("rate", [1.6, 3.0])
def test_level_refuses_values_with_too_few_photon_free_readings(rate):
    with pytest.raises(ValueError, match="photon"):
        photon_free_level(photon_frames([0], rate, shape=(128, 128))[0])


def test_level_refuses_values_clipped_at_a_floor():
    frame = np.maximum(photon_frames([-50], 0.15)[0], -40)
    with pytest.raises(ValueError, match="clipped at a floor"):
        photon_free_level(frame)


def test_level_refuses_saturated_values():
    with pytest.raises(ValueError, match="clipped"):
        photon_free_level(np.full(1000, np.iinfo(np.int16).max, np.int16))


@pytest.mark.parametrize(
    "values, match", [(np.array([], np.int16), "no values"), (np.array([1.0, np.nan]), "finite")]
)
def test_level_refuses_empty_or_non_finite_values(values, match):
    with pytest.raises(ValueError, match=match):
        photon_free_level(values)


def test_dim_mask_lets_a_bright_channel_be_measured():
    rate = np.full((96, 96), 3.0)
    rate[:, :12] = 0.1  # a dim strip, e.g. a vessel (12.5% of the pixels)
    frames = photon_frames([30, -20], rate)
    mask = dim_pixel_mask(frames.mean(axis=0), percentile=10)
    assert mask[:, :12].mean() > 0.7 and mask[:, 12:].mean() < 0.02
    np.testing.assert_allclose(photon_free_levels(frames, mask), [30, -20], atol=3)


def test_dim_mask_stays_inside_the_region():
    image = np.arange(100.0).reshape(10, 10)
    mask = dim_pixel_mask(image, percentile=50, region=(slice(5, 10), slice(0, 10)))
    assert not mask[:5].any() and mask[5:].sum() == 25


@pytest.mark.parametrize("percentile", [0, 101])
def test_dim_mask_refuses_bad_percentile(percentile):
    with pytest.raises(ValueError, match="percentile"):
        dim_pixel_mask(np.ones((4, 4)), percentile)


@pytest.mark.parametrize(
    "mask, match",
    [
        (np.ones((3, 3), bool), "boolean of the frame shape"),
        (np.ones((96, 96), int), "boolean of the frame shape"),
        (np.pad(np.ones((10, 10), bool), ((0, 86), (0, 86))), "too few"),
    ],
)
def test_levels_refuse_unusable_masks(mask, match):
    with pytest.raises(ValueError, match=match):
        photon_free_levels(photon_frames([0], 0.15), mask)


def test_check_steps_refuses_a_jump_of_about_a_photon():
    check_steps(np.array([0.0, -3, 2, 5]))
    with pytest.raises(ValueError, match="jumps by 350 counts at frame 2"):
        check_steps(np.array([0.0, 0, 350, 352]))


def test_subtract_level_removes_the_offset_from_every_trace():
    traces = np.array([[10.0, 20.0, 30.0], [1.0, 2.0, 3.0]])
    np.testing.assert_allclose(
        subtract_level(traces, np.array([5.0, -5.0, 0.0])), [[5, 25, 30], [-4, 7, 3]]
    )
    np.testing.assert_allclose(subtract_level(np.array([4.0, 4.0]), np.array([1.0, 3.0])), [3, 1])


def test_subtract_level_keeps_empty_rows_empty():
    traces = np.array([[0.0, 0.0], [5.0, 5.0]])
    np.testing.assert_allclose(subtract_level(traces, np.array([2.0, 3.0])), [[0, 0], [3, 2]])


def test_subtract_level_refuses_mismatched_frames_and_nan():
    with pytest.raises(ValueError, match="frames"):
        subtract_level(np.zeros((2, 3)), np.zeros(4))
    with pytest.raises(ValueError, match="finite"):
        subtract_level(np.ones((2, 2)), np.array([0.0, np.nan]))


# ---- reading the raw TIFF -----------------------------------------------------------

MASK_FRAMES = 6


def write_raw(path, red, green):
    """Interleaved pages: red (channel 1) then green (channel 2) for each timepoint."""
    pages = np.empty((2 * len(red), *red.shape[1:]), np.int16)
    pages[0::2], pages[1::2] = red, green
    tifffile.imwrite(path, pages)


def settings(fluorescence=None, **top):
    return PipelineSettings.model_validate(
        {
            "distortion": {"mode": "off"},
            "fluorescence": {
                "photon_free_mask_frames": MASK_FRAMES,
                "photon_free_pixel_percentile": 20,  # ~800 px per frame here
                "background_chunk_frames": 4,
                **(fluorescence or {}),
            },
            **top,
        }
    )


@pytest.fixture
def raw_tiff(tmp_path):
    # steps of ~11 (green) and 20 (red) counts per frame: a fast real drift
    green_levels, red_levels = np.linspace(-90, 10, 10), np.linspace(40, 220, 10)
    red_rate = np.full((96, 96), 2.0)
    red_rate[:, :30] = 0.1  # wide enough to stay dim inside the 16 px margin
    write_raw(
        tmp_path / "run.tif",
        photon_frames(red_levels, red_rate, seed=1),
        photon_frames(green_levels, np.where(np.arange(96) < 30, 0.05, 0.3), seed=2),
    )
    return tmp_path / "run.tif", green_levels, red_levels


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


def test_raw_levels_are_cached_and_recomputed_when_tiff_or_settings_change(
    raw_tiff, tmp_path, monkeypatch
):
    tiff, *_ = raw_tiff
    first = photon_free_io.raw_levels(tiff, tmp_path, n_frames=10, settings=settings())
    assert (tmp_path / photon_free_io.LEVELS_FILE).is_file()

    def no_read(*args, **kwargs):
        raise AssertionError("read the TIFF again")

    monkeypatch.setattr(photon_free_io, "read_channel_chunk", no_read)
    again = photon_free_io.raw_levels(tiff, tmp_path, n_frames=10, settings=settings())
    np.testing.assert_array_equal(again.green, first.green)
    with pytest.raises(AssertionError, match="again"):
        photon_free_io.raw_levels(
            tiff, tmp_path, 10, settings({"photon_free_pixel_percentile": 25})
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


def test_raw_levels_follow_max_timepoints(raw_tiff, tmp_path):
    tiff, green, _ = raw_tiff
    levels = photon_free_io.raw_levels(tiff, tmp_path, 8, settings(max_timepoints=8))
    assert len(levels.green) == 8
    np.testing.assert_allclose(levels.green[MASK_FRAMES:], green[MASK_FRAMES:8], atol=3)


@pytest.mark.parametrize(
    "top, match",
    [
        ({"destripe": {"enabled": True, "preserve_frame_mean": False}}, "twice"),
        (
            {
                "distortion": {
                    "mode": "local_batched",
                    "transform_mat": "t.mat",
                    "crop_black_borders": False,
                }
            },
            "zero-filled",
        ),
    ],
)
def test_raw_levels_refuse_settings_that_remove_the_offset(raw_tiff, tmp_path, top, match):
    tiff, *_ = raw_tiff
    with pytest.raises(ValueError, match=match):
        photon_free_io.raw_levels(tiff, tmp_path, 10, settings(**top))


def plane_traces(f, fneu, f_chan2=None, fneu_chan2=None):
    return PlaneTraces(
        f=f,
        fneu=fneu,
        f_chan2=f_chan2,
        fneu_chan2=fneu_chan2,
        iscell=np.ones(len(f)),
        stat=np.array([]),
        ops={},
        labels=np.zeros((2, 2)),
    )


def test_traces_and_background_lose_the_level_frame_for_frame(raw_tiff, tmp_path):
    tiff, *_ = raw_tiff
    levels = photon_free_io.raw_levels(tiff, tmp_path, n_frames=10, settings=settings())
    ones = np.ones((3, 10), np.float32)
    cleaned = photon_free_io.traces_without_level(plane_traces(ones, 2 * ones, 3 * ones), levels)

    def expected(value, level):
        return np.broadcast_to(value - level, (3, 10))

    np.testing.assert_allclose(cleaned.f, expected(1, levels.green), atol=1e-3)
    np.testing.assert_allclose(cleaned.fneu, expected(2, levels.green), atol=1e-3)
    np.testing.assert_allclose(cleaned.f_chan2, expected(3, levels.red), atol=1e-3)
    assert cleaned.fneu_chan2 is None
    background = BackgroundTraces(
        source="vessel",
        green=np.zeros(10) + 5,
        red=np.ones(10),
        mask=np.zeros((2, 2), bool),
        alignment=None,
        key={},
    )
    cleaned_bg = photon_free_io.background_without_level(background, levels)
    np.testing.assert_allclose(cleaned_bg.red, 1 - levels.red)
    with pytest.raises(ValueError, match="cover 10 frames"):
        photon_free_io.traces_without_level(plane_traces(ones[:, :9], ones[:, :9]), levels)


def test_errors_name_the_recording_frame(raw_tiff, tmp_path):
    """A jump inside a later chunk is reported at its frame in the recording."""
    tiff, *_ = raw_tiff
    with tifffile.TiffFile(tiff) as tif:
        pages = tif.asarray()
    pages[2 * 6 + 1] += 350  # green, frame 6: a level one photon too high
    tifffile.imwrite(tiff, pages)
    with pytest.raises(ValueError, match="at frame 6"):
        photon_free_io.raw_levels(tiff, tmp_path, n_frames=10, settings=settings())
