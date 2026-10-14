import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from mini2p_preprocessing import plots  # noqa: E402
from mini2p_preprocessing.config import FluorescenceSettings, PipelineSettings  # noqa: E402
from mini2p_preprocessing.extraction import run_cellpose_extraction  # noqa: E402
from mini2p_preprocessing.fluorescence import audit_table, compute_dff  # noqa: E402
from mini2p_preprocessing.plane import load_plane_traces, load_valid_frames  # noqa: E402
from tests.test_rois_extraction import label_image, make_plane  # noqa: E402

SETTINGS = FluorescenceSettings(baseline="prctile", prctile_baseline=10.0)


def test_audit_table_summarises_each_roi():
    f = np.array([[10.0] * 8, [2.0] * 8], np.float32)
    fneu = np.array([[0.0] * 8, [10.0] * 8], np.float32)  # roi 1: 2 - 0.7*10 < 0
    result = compute_dff(f, fneu, 7.52, SETTINGS)
    table = audit_table(f, fneu, result)
    assert np.allclose(table["median_f"], [10.0, 2.0])
    assert np.allclose(table["median_f0"], [10.0, -5.0])
    assert np.allclose(table["lowest_f0"], [10.0, -5.0])
    assert np.allclose(table["frac_negative_fcorr"], [0.0, 1.0])
    assert np.allclose(table["frac_invalid_f0"], [0.0, 1.0])
    assert np.isnan(table["dff_p99"][1]) and table["dff_p99"][0] == pytest.approx(0.0)


def test_load_plane_traces_and_valid_frames(tmp_path, rng):
    plane, seg = make_plane(tmp_path, rng)
    with pytest.raises(FileNotFoundError, match="F.npy"):
        load_plane_traces(plane)
    run_cellpose_extraction(
        seg, plane, PipelineSettings.model_validate({"distortion": {"mode": "off"}})
    )
    traces = load_plane_traces(plane)
    assert traces.f.shape == traces.f_chan2.shape == (3, 200)
    assert traces.fs == pytest.approx(7.52) and traces.labels.shape == label_image().shape

    frames = tmp_path / "valid_frames.npz"
    np.savez(frames, valid_frames=np.arange(0, 200, 2))
    assert len(load_valid_frames(frames)) == 100
    with pytest.raises(KeyError, match="valid_frames"):
        load_valid_frames(frames, key="missing")


def test_audit_figures_draw_into_given_axes(tmp_path, rng):
    plane, seg = make_plane(tmp_path, rng)
    run_cellpose_extraction(
        seg, plane, PipelineSettings.model_validate({"distortion": {"mode": "off"}})
    )
    traces = load_plane_traces(plane)
    result = compute_dff(traces.f, traces.fneu, traces.fs, SETTINGS)
    table = audit_table(traces.f, traces.fneu, result)

    fig, axes = plt.subplots(3, 1)
    drawn = plots.plot_trace_audit(traces.f, traces.fneu, result, roi=1, fs_hz=traces.fs, axes=axes)
    assert drawn[0].figure is fig and len(drawn[1].lines) >= 2
    plots.plot_f0_overview(table, min_f0=SETTINGS.min_f0)
    fig, axes = plt.subplots(1, 2)
    plots.plot_cell_fov(1, np.ones(traces.labels.shape), traces.labels, axes=axes)
    with pytest.raises(ValueError, match="ROI 9"):
        plots.plot_cell_fov(9, np.ones(traces.labels.shape), traces.labels)
    plt.close("all")


def test_dff_trace_plot_uses_given_axis():
    from mini2p_preprocessing.fluorescence import compute_dff

    f = np.full((2, 30), 10.0, np.float32)
    result = compute_dff(f, np.zeros_like(f), 7.52, SETTINGS)
    fig, ax = plt.subplots()
    assert plots.plot_dff_trace(result, 1, 7.52, ax=ax) is ax and len(ax.lines) == 2
    plt.close("all")


def test_roi_with_no_valid_f0_says_so_and_keeps_the_time_axis():
    from mini2p_preprocessing.fluorescence import compute_dff

    f = np.full((1, 75), 2.0, np.float32)
    result = compute_dff(f, np.full_like(f, 10.0), 7.5, SETTINGS)  # F0 < 0 everywhere
    axes = plots.plot_trace_audit(f, np.full_like(f, 10.0), result, roi=0, fs_hz=7.5)
    assert axes[2].get_xlim()[1] == pytest.approx(74 / 7.5, rel=0.06)
    assert any("undefined" in text.get_text() for text in axes[2].texts)
    ax = plots.plot_dff_trace(result, 0, 7.5)
    assert any("undefined" in text.get_text() for text in ax.texts)
    plt.close("all")


def test_time_axis_label_can_say_frames_were_selected():
    from mini2p_preprocessing.fluorescence import compute_dff

    f = np.full((1, 40), 10.0, np.float32)
    result = compute_dff(f, np.zeros_like(f), 7.5, SETTINGS)
    axes = plots.plot_trace_audit(
        f, np.zeros_like(f), result, 0, 7.5, time_label="Valid-frame time (s)"
    )
    assert axes[2].get_xlabel() == "Valid-frame time (s)"
    ax = plots.plot_dff_trace(result, 0, 7.5, time_label="Valid-frame time (s)")
    assert ax.get_xlabel() == "Valid-frame time (s)"
    plt.close("all")


def test_mask_alignment_plot_draws_both_panels():
    labels = np.zeros((20, 20), np.int32)
    labels[5:9, 5:9] = 1
    fig, axes = plt.subplots(1, 2)
    drawn = plots.plot_mask_alignment(
        np.ones((20, 20)), labels, np.roll(labels, 2, axis=0), axes=axes, title="dy=2, dx=0"
    )
    assert drawn[0].figure is fig and "dy=2" in drawn[1].get_title()
    plt.close("all")
