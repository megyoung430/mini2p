import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mini2p_preprocessing import plots, qc  # noqa: E402
from mini2p_preprocessing.bidiphase import estimate_phase  # noqa: E402
from mini2p_preprocessing.config import PipelineSettings  # noqa: E402
from mini2p_preprocessing.correction import correction_stages  # noqa: E402

SETTINGS = PipelineSettings.model_validate(
    {
        "phase": {"chunk_frames": 20, "max_shift_px": 8, "alignment_source": "best_score"},
        "distortion": {"mode": "off"},
    }
)


def test_phase_trace_matches_known_phases_and_plots(recording):
    tiff, _ = recording
    trace = qc.phase_trace(tiff, SETTINGS)
    assert [row["phase_px"] for row in trace] == [3, -2, 5]
    assert qc.lowest_score_chunk(trace) in trace
    fig, axes = plt.subplots(2, 1)
    assert plots.plot_phase_trace(trace, SETTINGS, axes=axes) is not None
    plt.close("all")


def test_qc_stages_and_figures_draw_into_given_axes(recording):
    tiff, stacks = recording
    raw = qc.load_chunk(tiff, 20, 40, 2)
    stages = correction_stages(raw, 20, SETTINGS)
    fig, axes = plt.subplots(2, 2)
    drawn = plots.plot_channel_pairs(
        qc.mean_images(raw),
        qc.mean_images(stages.corrected),
        ("Before", "After"),
        SETTINGS.channels.names,
        axes=axes,
    )
    assert drawn[0, 0].figure is fig
    plots.plot_score_curve(estimate_phase(raw[1], SETTINGS.phase), "chunk 20:40")
    green = np.random.default_rng(0).random((12, 16))
    plots.plot_contrast_panels(green, green, (0, 1), (0, 1))
    plt.close("all")


def test_registration_figures_from_ops():
    n = 50
    ops = {
        "xoff": np.zeros(n),
        "yoff": np.ones(n),
        "corrXY": np.full(n, 0.9),
        "badframes": np.arange(n) % 10 == 0,
        "xoff1": np.zeros((n, 4)),
        "yoff1": np.zeros((n, 4)),
    }
    plots.plot_registration_traces(ops, fs_hz=7.52)
    plots.plot_nonrigid_offsets(ops, fs_hz=7.52)
    plt.close("all")


def _per_frame_settings():
    return PipelineSettings.model_validate(
        {
            "phase": {
                "max_shift_px": 8,
                "consistency_window_frames": 9,
                "trust_window_frames": 11,
                "fallback_window_frames": 21,
            },
            "distortion": {"mode": "off"},
        }
    )


def test_frame_phase_preview_and_plot(recording):
    raw_tiff, _ = recording
    preview = qc.frame_phase_preview(raw_tiff, _per_frame_settings())
    assert preview.curves.shape == (55, 2, 17)
    assert list(preview.chosen.phase_px[[5, 25, 50]]) == [3, -2, 5]
    _, axes = plt.subplots(2, 1)
    drawn = plots.plot_frame_phases(preview.chosen, axes=axes)
    assert drawn[0] is axes[0] and len(drawn) == 2
    assert drawn[0].get_xlabel() == "" and drawn[1].get_xlabel() == "Frame"
    assert drawn[1].get_ylabel() == "Structural Corr Lost"
    plt.close("all")


def test_frame_phase_plot_makes_its_own_axes(recording):
    raw_tiff, _ = recording
    preview = qc.frame_phase_preview(raw_tiff, _per_frame_settings())
    axes = plots.plot_frame_phases(preview.chosen)
    assert len(axes) == 2
    plt.close("all")


# ---- registered means cropped to the part Suite2p's wrap-around never reaches ---------


def test_unwrapped_region_follows_suite2p_ranges():
    rows, cols = qc.unwrapped_region({"yrange": [46, 466], "xrange": [22, 490]})
    assert (rows, cols) == (slice(46, 466), slice(22, 490))
    assert np.zeros((512, 512))[rows, cols].shape == (420, 468)


def test_unwrapped_region_needs_the_ranges():
    import pytest

    with pytest.raises(KeyError, match="yrange"):
        qc.unwrapped_region({"xrange": [0, 10]})


def test_channel_columns_share_limits_from_the_chosen_columns():
    rng = np.random.default_rng(0)
    full = [rng.random((40, 50)) for _ in range(2)]
    for image in full:
        image[-5:] = 100.0  # a bright wrapped stripe outside the region
    rows, cols = slice(0, 35), slice(5, 45)
    cropped = [image[rows, cols] for image in full]
    _, axes = plt.subplots(2, 4)
    drawn = plots.plot_channel_columns(
        [full, full, cropped, cropped],
        ("A", "B", "A cropped", "B cropped"),
        {1: "red", 2: "green"},
        axes=axes,
        limit_columns=(2, 3),
        box=(rows, cols),
        box_columns=(0, 1),
    )
    assert drawn.shape == (2, 4)
    for row in drawn:
        limits = {ax.images[0].get_clim() for ax in row}
        assert len(limits) == 1  # one display range per channel row
        assert max(limits.pop()) < 100  # set by the cropped columns, not the stripe
    assert len(drawn[0, 0].patches) == 1 and len(drawn[0, 2].patches) == 0
    assert drawn[1, 3].get_title() == "green\nB cropped"
    plt.close("all")


def test_channel_columns_need_one_title_per_column():
    import pytest

    image = np.zeros((4, 4))
    with pytest.raises(ValueError, match="titles"):
        plots.plot_channel_columns([[image], [image]], ("only one",), {1: "red"})
