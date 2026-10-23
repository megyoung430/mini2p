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
    drawn = plots.plot_frame_phases(preview.chosen, preview.structural_loss, axes=axes)
    assert drawn[0] is axes[0] and len(drawn) == 2
    assert drawn[0].get_xlabel() == "" and drawn[1].get_xlabel() == "Frame"
    plt.close("all")


def test_frame_phase_plot_makes_its_own_axes(recording):
    raw_tiff, _ = recording
    preview = qc.frame_phase_preview(raw_tiff, _per_frame_settings())
    axes = plots.plot_frame_phases(preview.chosen, preview.structural_loss)
    assert len(axes) == 2
    plt.close("all")
