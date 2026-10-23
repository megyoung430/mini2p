import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

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


def rolled_like_suite2p(base, yoff, xoff):
    """Suite2p's rigid shift: torch.roll(frame, shifts=(-dy, -dx)), i.e. circular."""
    return np.stack(
        [np.roll(base, (-dy, -dx), axis=(0, 1)) for dy, dx in zip(yoff, xoff, strict=True)]
    )


def test_unwrapped_region_excludes_every_wrapped_pixel_and_no_more():
    rng = np.random.default_rng(0)
    ly, lx = 40, 50
    yoff = rng.integers(-3, 6, 200)
    xoff = rng.integers(-7, 2, 200)
    yoff[17], xoff[90] = 11, -9  # single jolts, the kind Suite2p flags as bad frames
    rows, cols = qc.unwrapped_region({"yoff": yoff, "xoff": xoff, "Ly": ly, "Lx": lx})
    row_of = rolled_like_suite2p(np.repeat(np.arange(ly)[:, None], lx, 1), yoff, xoff)
    col_of = rolled_like_suite2p(np.repeat(np.arange(lx)[None], ly, 0), yoff, xoff)
    i, j = np.arange(ly)[:, None], np.arange(lx)[None]
    wrapped = (row_of != (i + yoff[:, None, None])) | (col_of != (j + xoff[:, None, None]))
    assert not wrapped[:, rows, cols].any()
    # and the region is as large as it can be: each edge just outside it is wrapped somewhere
    assert wrapped[:, rows.start - 1].any() and wrapped[:, rows.stop].any()
    assert wrapped[:, :, cols.start - 1].any() and wrapped[:, :, cols.stop].any()


def test_unwrapped_region_can_be_limited_to_the_first_frames():
    ops = {"yoff": np.r_[np.zeros(10), 12], "xoff": np.zeros(11), "Ly": 40, "Lx": 50}
    assert qc.unwrapped_region(ops)[0] == slice(0, 28)
    assert qc.unwrapped_region(ops, n_frames=10)[0] == slice(0, 40)


def test_unwrapped_region_needs_offsets_and_room():
    with pytest.raises(KeyError, match="yoff"):
        qc.unwrapped_region({"xoff": [0], "Ly": 10, "Lx": 10})
    with pytest.raises(ValueError, match="nothing"):
        qc.unwrapped_region({"yoff": [6, -6], "xoff": [0, 0], "Ly": 10, "Lx": 10})


def full_and_cropped(rng):
    full = [rng.random((40, 50)) for _ in range(2)]
    for image in full:
        image[-5:] = 100.0  # a bright wrapped stripe outside the region
    rows, cols = slice(0, 35), slice(5, 45)
    return full, [image[rows, cols] for image in full], (rows, cols)


def test_channel_columns_share_limits_from_the_chosen_columns():
    full, cropped, box = full_and_cropped(np.random.default_rng(0))
    _, axes = plt.subplots(2, 4)
    drawn = plots.plot_channel_columns(
        [full, full, cropped, cropped],
        ("A", "B", "A Cropped", "B Cropped"),
        {1: "red", 2: "green"},
        axes=axes,
        limit_columns=(2, 3),
        box=box,
        box_columns=(0, 1),
    )
    assert drawn.shape == (2, 4)
    for row in drawn:
        limits = {ax.images[0].get_clim() for ax in row}
        assert len(limits) == 1  # one display range per channel row
        assert max(limits.pop()) < 100  # set by the cropped columns, not the stripe
    assert len(drawn[0, 0].patches) == 1 and len(drawn[0, 2].patches) == 0
    assert drawn[1, 3].get_title() == "green\nB Cropped"
    plt.close("all")


def test_the_box_outlines_exactly_the_region():
    full, _, (rows, cols) = full_and_cropped(np.random.default_rng(0))
    _, axes = plt.subplots(2, 1)
    drawn = plots.plot_channel_columns(
        [full], ("A",), {1: "r", 2: "g"}, axes=axes, box=(rows, cols), box_columns=(0,)
    )
    (x0, y0), (x1, y1) = drawn[0, 0].patches[0].get_bbox().get_points()
    assert (x0, x1) == (cols.start - 0.5, cols.stop - 0.5)  # outer edges of the pixels
    assert (y0, y1) == (rows.start - 0.5, rows.stop - 0.5)
    plt.close("all")


def test_channel_columns_refuse_inconsistent_input():
    image = np.zeros((4, 4))
    names = {1: "red", 2: "green"}
    with pytest.raises(ValueError, match="titles"):
        plots.plot_channel_columns([[image], [image]], ("only one",), names)
    with pytest.raises(ValueError, match="channels"):
        plots.plot_channel_columns([[image], [image, image]], ("a", "b"), names)
    with pytest.raises(ValueError, match="x_slice"):
        plots.plot_channel_columns(
            [[image]],
            ("a",),
            names,
            x_slice=slice(1, None),
            box=(slice(0, 2), slice(0, 2)),
            box_columns=(0,),
        )
    with pytest.raises(ValueError, match="limit_columns"):
        plots.plot_channel_columns([[image]], ("a",), names, limit_columns=())
    with pytest.raises(ValueError, match="finite"):
        plots.plot_channel_columns([[np.full((4, 4), np.nan)]], ("a",), names)


def test_channel_pairs_still_refuse_mismatched_channels():
    image = np.zeros((4, 4))
    with pytest.raises(ValueError, match="channels"):
        plots.plot_channel_pairs([image], [image, image], ("a", "b"), {1: "r", 2: "g"})
