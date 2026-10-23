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


def suite2p_shifted(base, yoff, xoff, yoff1=None, xoff1=None, block=64):
    """Frames shifted by Suite2p's own shift_frames: rigid roll, then nonrigid resampling."""
    import torch
    from suite2p.registration import nonrigid
    from suite2p.registration.register import shift_frames

    ly, lx = base.shape
    blocks = nonrigid.make_blocks(ly, lx, (block, block))
    frames = torch.from_numpy(np.repeat(base[None], len(yoff), 0))
    return np.asarray(
        shift_frames(
            frames,
            torch.from_numpy(yoff),
            torch.from_numpy(xoff),
            yoff1,
            xoff1,
            blocks,
            device=torch.device("cpu"),
        ),
        dtype=float,
    )


def marked(shape, edge, band=40):
    """1000 everywhere except a band of 30000 along one edge, which only wraps around."""
    image = np.full(shape, 1000, np.int16)
    image[
        {
            "top": np.s_[:band],
            "bottom": np.s_[-band:],
            "left": np.s_[:, :band],
            "right": np.s_[:, -band:],
        }[edge]
    ] = 30000
    return image


@pytest.mark.parametrize("nonrigid", [False, True])
def test_unwrapped_region_excludes_every_wrapped_pixel(nonrigid):
    rng = np.random.default_rng(0)
    shape, n = (128, 128), 30
    yoff, xoff = rng.integers(-6, 7, n), rng.integers(-6, 7, n)
    yoff[3], xoff[7] = 11, -10  # single jolts, the kind Suite2p flags as bad frames
    n_blocks = 9 if nonrigid else 0  # 64-px blocks on 128 px overlap: 3 x 3
    yoff1 = rng.uniform(-3, 3, (n, n_blocks)).astype(np.float32) if nonrigid else None
    xoff1 = rng.uniform(-3, 3, (n, n_blocks)).astype(np.float32) if nonrigid else None
    ops = {"yoff": yoff, "xoff": xoff, "Ly": 128, "Lx": 128}
    if nonrigid:
        ops |= {"yoff1": yoff1, "xoff1": xoff1}
    rows, cols = qc.unwrapped_region(ops)
    mid_r, mid_c = (rows.start + rows.stop) // 2, (cols.start + cols.stop) // 2
    # a band wrapped from the opposite edge is the only way a marker reaches each half
    checks = {
        "bottom": np.s_[:, rows.start : mid_r, cols],
        "top": np.s_[:, mid_r : rows.stop, cols],
        "right": np.s_[:, rows, cols.start : mid_c],
        "left": np.s_[:, rows, mid_c : cols.stop],
    }
    for edge, inside in checks.items():
        shifted = suite2p_shifted(marked(shape, edge), yoff, xoff, yoff1, xoff1)
        # 999/1000 is integer rounding of the resampled interior; a marker would add ~30000
        assert np.all(np.abs(shifted[inside] - 1000) <= 1), edge


def test_unwrapped_region_is_no_smaller_than_the_rigid_shifts_need():
    yoff, xoff = np.array([-4, 9, 0]), np.array([3, -2, 0])
    rows, cols = qc.unwrapped_region({"yoff": yoff, "xoff": xoff, "Ly": 50, "Lx": 60})
    assert (rows, cols) == (slice(4, 41), slice(2, 57))


def test_nonrigid_shifts_widen_the_margins():
    ops = {
        "yoff": np.array([-4, 0]),
        "xoff": np.array([0, 0]),
        "Ly": 50,
        "Lx": 60,
        "yoff1": np.array([[-2.5, 0.0], [1.0, 3.2]]),
        "xoff1": np.zeros((2, 2)),
    }
    rows, _ = qc.unwrapped_region(ops)
    assert rows == slice(7, 46)  # top: 4 + 2.5 -> 7; bottom: 0 + 3.2 -> 4


def test_unwrapped_region_can_be_limited_to_the_first_frames():
    ops = {"yoff": np.r_[np.zeros(10), 12], "xoff": np.zeros(11), "Ly": 40, "Lx": 50}
    assert qc.unwrapped_region(ops)[0] == slice(0, 28)
    assert qc.unwrapped_region(ops, n_frames=10)[0] == slice(0, 40)


@pytest.mark.parametrize(
    "change, match",
    [
        ({"yoff": [0.0, np.nan]}, "finite"),
        ({"yoff": [0.0, 1.0, 2.0]}, "frames"),
        ({"yoff": [[0.0, 1.0]]}, "1-D"),
        ({"yoff1": np.zeros((3, 4)), "xoff1": np.zeros((3, 4))}, "frames"),
        ({"yoff1": np.zeros((2, 4))}, "both"),
        ({"yoff": [6, -6]}, "nothing"),
    ],
)
def test_unwrapped_region_checks_its_offsets(change, match):
    ops = {"yoff": [0.0, 0.0], "xoff": [0.0, 0.0], "Ly": 10, "Lx": 10} | change
    with pytest.raises(ValueError, match=match):
        qc.unwrapped_region(ops)


def test_block_shifts_must_cover_the_same_frames_before_trimming():
    ops = {
        "yoff": [0.0] * 3,
        "xoff": [0.0] * 3,
        "Ly": 10,
        "Lx": 10,
        "yoff1": np.zeros((5, 2)),
        "xoff1": np.zeros((5, 2)),
    }
    with pytest.raises(ValueError, match="frames"):
        qc.unwrapped_region(ops, n_frames=2)


def test_unwrapped_region_needs_offsets_and_at_least_one_frame():
    with pytest.raises(KeyError, match="yoff"):
        qc.unwrapped_region({"xoff": [0], "Ly": 10, "Lx": 10})
    with pytest.raises(ValueError, match="n_frames"):
        qc.unwrapped_region({"yoff": [0], "xoff": [0], "Ly": 10, "Lx": 10}, n_frames=0)


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
    with pytest.raises(ValueError, match="non-negative"):
        plots.plot_channel_columns(
            [[image]], ("a",), names, box=(slice(1, -1), slice(0, 2)), box_columns=(0,)
        )
    with pytest.raises(ValueError, match="limit_columns"):
        plots.plot_channel_columns([[image]], ("a",), names, limit_columns=())
    with pytest.raises(ValueError, match="finite"):
        plots.plot_channel_columns([[np.full((4, 4), np.nan)]], ("a",), names)


def test_channel_pairs_still_refuse_mismatched_channels():
    image = np.zeros((4, 4))
    with pytest.raises(ValueError, match="channels"):
        plots.plot_channel_pairs([image], [image, image], ("a", "b"), {1: "r", 2: "g"})
