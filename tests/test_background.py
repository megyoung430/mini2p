"""Blood-vessel background and red-channel regression on synthetic traces with known parts."""

import json

import numpy as np
import pytest

from mini2p_preprocessing.background import (
    dff_versions,
    regress_out,
    regress_out_each,
    vessel_mask,
)
from mini2p_preprocessing.config import FluorescenceSettings

FS = 7.52
SETTINGS = FluorescenceSettings(win_baseline_s=20.0)


def slow_wave(rng, n, scale):
    """A smooth, zero-mean fluctuation, like background or motion-driven brightness."""
    kernel = np.hanning(31) / np.hanning(31).sum()
    wave = np.convolve(rng.standard_normal(n + 30), kernel, mode="valid")[:n]
    return scale * (wave - wave.mean()) / wave.std()


def transients(rng, n, n_events=6):
    """Calcium-like events: sparse jumps with exponential decay."""
    spikes = np.zeros(n)
    spikes[rng.choice(n - 50, n_events, replace=False)] = 1.0
    return np.convolve(spikes, np.exp(-np.arange(40) / 8), mode="full")[:n]


def test_regress_out_removes_a_shared_regressor_and_keeps_the_mean():
    rng = np.random.default_rng(0)
    n = 2000
    background = 50 + slow_wave(rng, n, 10)
    signal = np.stack([transients(rng, n), transients(rng, n)])
    traces = np.array([[200.0], [300.0]]) + 40 * signal + np.array([[1.5], [0.4]]) * background
    result = regress_out(traces, background)
    np.testing.assert_allclose(result.coef, [1.5, 0.4], atol=0.15)
    np.testing.assert_allclose(result.cleaned.mean(axis=1), traces.mean(axis=1), rtol=1e-6)
    residual = result.cleaned - 40 * signal
    assert np.std(residual[0]) < 0.1 * np.std(1.5 * background)


def test_regress_out_each_uses_each_rows_own_regressor():
    rng = np.random.default_rng(1)
    n = 1500
    reds = np.stack([100 + slow_wave(rng, n, 8), 80 + slow_wave(rng, n, 5)])
    greens = (
        300
        + np.array([[0.7], [2.0]]) * reds
        + 30 * np.stack([transients(rng, n), transients(rng, n)])
    )
    result = regress_out_each(greens, reds)
    np.testing.assert_allclose(result.coef, [0.7, 2.0], atol=0.2)


def test_regression_fails_loudly_on_unusable_input():
    with pytest.raises(ValueError, match="flat"):
        regress_out(np.ones((2, 10)), np.full(10, 3.0))
    with pytest.raises(ValueError, match="frames"):
        regress_out(np.ones((2, 10)), np.arange(9.0))
    with pytest.raises(ValueError, match="finite"):
        regress_out(np.ones((2, 10)), np.r_[np.nan, np.arange(9.0)])
    with pytest.raises(ValueError, match="flat.*row"):
        regress_out_each(np.ones((2, 10)), np.stack([np.arange(10.0), np.ones(10)]))
    with pytest.raises(ValueError, match="shape"):
        regress_out_each(np.ones((2, 10)), np.ones((3, 10)))


def test_vessel_mask_is_every_labelled_pixel():
    labels = np.zeros((5, 6), int)
    labels[1, 1:3] = 1
    labels[3, 4] = 2
    mask = vessel_mask(labels, (5, 6))
    assert mask.sum() == 3 and mask[3, 4]
    with pytest.raises(ValueError, match="no labelled"):
        vessel_mask(np.zeros((5, 6), int), (5, 6))
    with pytest.raises(ValueError, match="shape"):
        vessel_mask(labels, (6, 5))


def synthetic_plane(rng, n=3000, n_rois=3):
    """F and Fneu sharing a vessel background, plus a red-correlated (motion) brightness."""
    background = {"green": 40 + slow_wave(rng, n, 8), "red": 60 + slow_wave(rng, n, 6)}
    motion = np.stack([slow_wave(rng, n, 15) for _ in range(n_rois)])
    activity = np.stack([transients(rng, n) for _ in range(n_rois)])
    f2 = 200 + motion + 0.8 * background["red"] + rng.normal(0, 1, (n_rois, n))
    fneu2 = 100 + 0.8 * background["red"] + rng.normal(0, 1, (n_rois, n))
    f = 400 + 60 * activity + 0.5 * motion + 1.2 * background["green"]
    f = f + rng.normal(0, 1, (n_rois, n))
    fneu = 150 + 1.2 * background["green"] + rng.normal(0, 1, (n_rois, n))
    return f, fneu, f2, fneu2, background, activity


def correlation(a, b):
    return np.array([np.corrcoef(x, y)[0, 1] for x, y in zip(a, b, strict=True)])


def test_each_correction_brings_the_dff_closer_to_the_activity():
    rng = np.random.default_rng(2)
    f, fneu, f2, fneu2, background, activity = synthetic_plane(rng)
    versions = dff_versions(
        f, fneu, f2, fneu2, background["green"], background["red"], FS, SETTINGS, red_usable=True
    )
    standard = correlation(versions.standard.dff, activity)
    removed = correlation(versions.background.dff, activity)
    red = correlation(versions.red.dff, activity)
    assert np.all(removed > standard) and np.all(red > removed)
    # F also holds activity and motion, which correlate with the background by chance.
    np.testing.assert_allclose(versions.background_coef["F"], 1.2, atol=0.25)
    np.testing.assert_allclose(versions.background_coef["Fneu"], 1.2, atol=0.1)
    assert versions.red_coef.shape == (3,)


def test_red_regression_is_skipped_when_red_is_not_usable():
    rng = np.random.default_rng(3)
    f, fneu, f2, fneu2, background, _ = synthetic_plane(rng)
    versions = dff_versions(
        f, fneu, f2, fneu2, background["green"], background["red"], FS, SETTINGS, red_usable=False
    )
    assert versions.red is None and versions.red_coef is None
    single_channel = dff_versions(
        f, fneu, None, None, background["green"], None, FS, SETTINGS, red_usable=True
    )
    assert single_channel.red is None


def test_versions_need_matching_frames():
    rng = np.random.default_rng(4)
    f, fneu, f2, fneu2, background, _ = synthetic_plane(rng, n=500)
    with pytest.raises(ValueError, match="frames"):
        dff_versions(f, fneu, f2, fneu2, background["green"][:-1], background["red"], FS,
                     SETTINGS, red_usable=True)  # fmt: skip


# ---------------------------------------------------------------- the registered binaries


def registered_movie(rng, plane, n=400, shape=(12, 16), drawn_on="meanImg.png"):
    """A plane folder with registered binaries: a static image per channel plus a
    vessel-driven background whose strength varies across the field, and a vessel
    segmentation drawn on *drawn_on* in that folder."""
    from mini2p_preprocessing.registration import RegisteredOutputs

    plane.mkdir(parents=True, exist_ok=True)
    ly, lx = shape
    vessel = np.zeros(shape, int)
    vessel[2:4, :] = 1  # one horizontal vessel
    weight = np.linspace(0.5, 1.5, lx)[None, :] * np.ones((ly, 1))
    stacks, backgrounds = {}, {}
    for name, level, binary in (("green", 300, "data.bin"), ("red", 800, "data_chan2.bin")):
        bg = slow_wave(rng, n, 20)
        image = level + 50 * rng.random(shape)
        stack = image[None] + bg[:, None, None] * weight[None] + rng.normal(0, 1, (n, *shape))
        stack[:, vessel > 0] = level / 3 + bg[:, None]  # the vessel is background only
        stack.astype(np.int16).tofile(plane / binary)
        stacks[name] = np.memmap(plane / binary, np.int16, "r", shape=(n, *shape))
        backgrounds[name] = bg
    ops = {"nframes": n, "Ly": ly, "Lx": lx, "fs": FS, "nchannels": 2}
    seg = plane / "vessels_seg.npy"
    np.save(seg, {"masks": vessel, "filename": str(plane / drawn_on)}, allow_pickle=True)
    outputs = RegisteredOutputs(ops=ops, functional=stacks["green"], structural=stacks["red"])
    return outputs, seg, backgrounds, weight


def pipeline_settings(**fluorescence):
    from mini2p_preprocessing.config import PipelineSettings

    return PipelineSettings.model_validate(
        {"distortion": {"mode": "off"}, "fluorescence": {"background_chunk_frames": 64,
                                                         **fluorescence}}
    )  # fmt: skip


def test_vessel_traces_follow_the_background_and_are_cached(tmp_path, monkeypatch):
    from mini2p_preprocessing import background_io as module

    rng = np.random.default_rng(5)
    outputs, seg, backgrounds, _ = registered_movie(rng, tmp_path)
    vessel = module.load_vessel_traces(outputs, tmp_path, seg, pipeline_settings())
    assert vessel.mask.sum() == 2 * 16 and vessel.alignment is None
    assert np.corrcoef(vessel.green, backgrounds["green"])[0, 1] > 0.99
    assert np.corrcoef(vessel.red, backgrounds["red"])[0, 1] > 0.99
    monkeypatch.setattr(module, "mask_mean_trace", lambda *a: pytest.fail("cache not used"))
    again = module.load_vessel_traces(outputs, tmp_path, seg, pipeline_settings())
    np.testing.assert_array_equal(again.green, vessel.green)


@pytest.mark.parametrize("change", ["segmentation", "excluded pixels", "binaries"])
def test_the_vessel_cache_is_redone_when_its_inputs_change(tmp_path, monkeypatch, change):
    import os

    from mini2p_preprocessing import background_io as module

    rng = np.random.default_rng(9)
    outputs, seg, _, _ = registered_movie(rng, tmp_path)
    module.load_vessel_traces(outputs, tmp_path, seg, pipeline_settings())
    exclude = None
    if change == "segmentation":
        record = np.load(seg, allow_pickle=True).item()
        record["masks"][8, 3] = 2
        np.save(seg, record, allow_pickle=True)
    elif change == "excluded pixels":
        exclude = np.zeros((12, 16), bool)
        exclude[2, :4] = True  # a ROI over part of the vessel
    else:  # a registration replaced without clearing the folder: same size, new mtime
        stat = (tmp_path / "data.bin").stat()
        os.utime(tmp_path / "data.bin", ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    monkeypatch.setattr(module, "mask_mean_trace", lambda *a: pytest.fail("read again"))
    with pytest.raises(pytest.fail.Exception, match="read again"):
        module.load_vessel_traces(outputs, tmp_path, seg, pipeline_settings(), exclude=exclude)


def test_roi_pixels_are_left_out_of_the_vessel_mask(tmp_path):
    from mini2p_preprocessing.background_io import load_vessel_traces

    rng = np.random.default_rng(10)
    outputs, seg, _, _ = registered_movie(rng, tmp_path)
    exclude = np.zeros((12, 16), bool)
    exclude[2, :4] = True
    vessel = load_vessel_traces(outputs, tmp_path, seg, pipeline_settings(), exclude=exclude)
    assert vessel.mask.sum() == 2 * 16 - 4 and not vessel.mask[2, :4].any()


def test_a_vessel_mask_drawn_on_another_image_needs_that_image(tmp_path):
    from mini2p_preprocessing.background_io import load_vessel_traces

    rng = np.random.default_rng(11)
    outputs, seg, _, _ = registered_movie(rng, tmp_path, drawn_on="../other/meanImg.png")
    with pytest.raises(ValueError, match="background_image"):
        load_vessel_traces(outputs, tmp_path, seg, pipeline_settings())


def test_pixel_coefficients_follow_the_background_strength(tmp_path):
    from mini2p_preprocessing.background import pixel_coefficients

    rng = np.random.default_rng(6)
    outputs, _, backgrounds, weight = registered_movie(rng, tmp_path)
    coef = pixel_coefficients(outputs.functional, backgrounds["green"], 64)
    np.testing.assert_allclose(coef[6:, :], weight[6:, :], atol=0.05)


def test_the_corrected_tiff_has_the_background_removed_and_the_means_kept(tmp_path):
    import tifffile

    from mini2p_preprocessing.background_io import (
        BACKGROUND_RECORD,
        VesselTraces,
        check_background_tiff,
        load_vessel_traces,
        write_background_tiff,
    )

    rng = np.random.default_rng(7)
    outputs, seg, backgrounds, _ = registered_movie(rng, tmp_path)
    settings = pipeline_settings()
    vessel = load_vessel_traces(outputs, tmp_path, seg, settings)
    exact = VesselTraces(backgrounds["green"], backgrounds["red"], vessel.mask, None, vessel.key)
    path = write_background_tiff(outputs, exact, tmp_path, settings)
    with tifffile.TiffFile(path) as tif:  # one page per frame and channel
        pages = np.stack([page.asarray() for page in tif.pages])
    assert pages.dtype == np.int16 and pages.shape == (2 * 400, 12, 16)
    red, green = pages[0::2], pages[1::2]  # channel 1 red, channel 2 green
    for name, movie, original in (("green", green, outputs.functional),
                                  ("red", red, outputs.structural)):  # fmt: skip
        pixel = movie[:, 8, 10].astype(float)
        assert abs(np.corrcoef(pixel, backgrounds[name])[0, 1]) < 0.1
        assert abs(np.corrcoef(original[:, 8, 10], backgrounds[name])[0, 1]) > 0.9
        np.testing.assert_allclose(movie.mean(axis=0), original.mean(axis=0), atol=0.6)
    record = json.loads((tmp_path / BACKGROUND_RECORD).read_text())
    assert record["vessel"]["seg_file"] == str(seg.resolve()) and record["clipped_values"] == 0
    assert set(record["coefficient_median"]) == {"green", "red"}
    assert check_background_tiff(tmp_path, outputs.ops) == path
    (tmp_path / "data_chan2.bin").write_bytes((tmp_path / "data_chan2.bin").read_bytes() + b"x")
    with pytest.raises(ValueError, match="another registration"):
        check_background_tiff(tmp_path, outputs.ops)


# ---------------------------------------------------------------- choosing and drawing ROIs


def test_top_rois_rank_by_peak_and_put_undefined_rows_last():
    from mini2p_preprocessing.background import top_rois

    dff = np.array([[0.1, 0.2], [np.nan, np.nan], [0.5, 0.0], [0.3, np.nan]])
    assert top_rois(dff, 3).tolist() == [2, 3, 0]
    assert top_rois(dff, 10).tolist() == [2, 3, 0, 1]
    with pytest.raises(ValueError, match="at least 1"):
        top_rois(dff, 0)


def test_background_and_top_roi_plots_draw():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mini2p_preprocessing import plots

    rng = np.random.default_rng(8)
    f, fneu, f2, fneu2, background, _ = synthetic_plane(rng, n=600)
    versions = dff_versions(
        f, fneu, f2, fneu2, background["green"], background["red"], FS, SETTINGS, red_usable=True
    )
    mask = np.zeros((10, 12), bool)
    mask[3:5] = True
    axes = plots.plot_background(rng.random((10, 12)), mask, background,
                                 versions.background_coef, FS)  # fmt: skip
    assert len(axes) == 3 and len(axes[1].lines) == 2
    dffs = {"Standard Green": versions.standard.dff, "Background Removed": versions.background.dff,
            "Red Regressed Out": versions.red.dff}  # fmt: skip
    axes = plots.plot_top_rois(dffs, [2, 0], frames=slice(100, 400))
    assert [ax.get_title() for ax in axes] == list(dffs)
    labels = [t.get_text() for t in axes[0].get_yticklabels()]
    by_height = sorted(zip(axes[0].get_yticks(), labels, strict=True))
    assert [label for _, label in by_height] == ["ROI 0", "ROI 2"]  # first ROI on top
    assert axes[0].get_xlim() == (100, 399)
    plots.plot_corrected_trace(versions.background, 1, FS, slice(0, 300))
    with pytest.raises(ValueError, match="No finite"):
        plots.plot_top_rois({"x": np.full((3, 50), np.nan)}, [0, 1])
    plt.close("all")


def test_every_coefficient_set_gets_its_own_histogram():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mini2p_preprocessing import plots

    rng = np.random.default_rng(12)
    coefficients = {f"Set {i}": rng.normal(size=20) for i in range(5)}
    traces = {"green": rng.normal(size=50), "red": rng.normal(size=50)}
    axes = plots.plot_background(rng.random((6, 6)), np.eye(6, dtype=bool), traces,
                                 coefficients, FS)  # fmt: skip
    assert len(axes[2].patches) == 5
    with pytest.raises(ValueError, match="told apart"):
        plots.plot_background(rng.random((6, 6)), np.eye(6, dtype=bool), traces,
                              {f"Set {i}": np.ones(3) for i in range(9)}, FS)  # fmt: skip
    plt.close("all")


def test_green_bleeding_into_red_lets_red_regression_remove_activity():
    # A documented limit: with clean red, a few % of green in red is enough to take a
    # sizeable part of each transient with it.
    rng = np.random.default_rng(13)
    n = 4000
    activity = 60 * transients(rng, n, n_events=30)
    green = 400 + activity + rng.normal(0, 1, n)
    red = 200 + 0.05 * activity + rng.normal(0, 0.5, n)
    cleaned = regress_out_each(green[None], red[None]).cleaned[0]
    kept = np.polyfit(activity, cleaned - cleaned.mean(), 1)[0]
    assert kept < 0.9
