"""The QC notebook's plots saved for a batch run (run_qc)."""

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from mini2p_preprocessing import pipeline as pipeline_module
from mini2p_preprocessing import run_qc
from mini2p_preprocessing.config import PipelineSettings
from mini2p_preprocessing.exports import Contrast
from mini2p_preprocessing.qc_figures import QC_FOLDER

SETTINGS = {
    "phase": {"method": "chunked", "chunk_frames": 20, "max_shift_px": 8},
    "destripe": {"enabled": True},
    "registration": {"motion_alignment_source": "functional"},  # "auto" needs per_frame
    "distortion": {"mode": "off"},
    "fluorescence": {"subtract_photon_free_level": False},
}


def fake_registration(tiff, output_dir, settings):
    """A registration of *tiff* as Suite2p leaves it, with no motion: the corrected frames as
    the binaries, their means and zero shifts."""
    with tifffile.TiffFile(tiff) as tif:
        pages = np.stack([page.asarray() for page in tif.pages])
    green, red = pages[1::2], pages[0::2]  # pages alternate ch1 (red), ch2 (green)
    plane = output_dir / "aligned_functional" / "suite2p" / "plane0"
    plane.mkdir(parents=True)
    green.astype(np.int16).tofile(plane / "data.bin")
    red.astype(np.int16).tofile(plane / "data_chan2.bin")
    n, ly, lx = green.shape
    ops = {
        "nframes": n, "Ly": ly, "Lx": lx, "fs": 7.52, "nchannels": 2, "tau": 1.0,
        "meanImg": green.mean(axis=0).astype(np.float32),
        "meanImg_chan2": red.mean(axis=0).astype(np.float32),
        "yoff": np.zeros(n), "xoff": np.zeros(n), "corrXY": np.ones(n),
        "badframes": np.zeros(n, bool),
    }  # fmt: skip
    np.save(plane / "ops.npy", ops, allow_pickle=True)
    (plane.parent / "motion_alignment.json").write_text("{}")
    return plane


@pytest.fixture
def processed_run(tmp_path, recording, monkeypatch):
    raw_tiff, _ = recording
    monkeypatch.setattr(pipeline_module, "run_registration", fake_registration)
    settings = PipelineSettings.model_validate(SETTINGS)
    out = tmp_path / "run-01" / "without_distortion_correction" / "with_destriping"
    result = pipeline_module.run_pipeline(raw_tiff, out, settings)
    run = run_qc.RunQc(raw_tiff.resolve(), out.resolve(), result.corrected_tiff,
                       result.suite2p_input_tiff, result.plane_dir, settings)  # fmt: skip
    return run, result


def saved_names(results: dict) -> set[str]:
    return {Path(p).name for plots in results.values() if isinstance(plots, list) for p in plots}


def test_the_notebook_s_preprocessing_and_registration_plots_are_saved(processed_run):
    run, result = processed_run
    results = run_qc.save_run_qc(run, result.correction.frame_phases)
    assert not [step for step, plots in results.items() if isinstance(plots, str)]
    assert {
        "raw_meanImg_first55", "phase_trace", "before_after_bidiphase", "before_after_registration",
        "registration_traces", "registered_means_cropped", "focus_qc",
    } <= saved_names(results)  # fmt: skip
    for step in ("raw_mean", "bidiphase"):
        assert list((run.variant_dir / QC_FOLDER / step).glob("*.png"))
    registration_qc = run.plane_dir.parent / QC_FOLDER / "motion_registration"
    for name in ("before_after_registration", "focus_qc"):
        for extension in ("png", "pdf", "svg", "caption.txt"):
            assert (registration_qc / f"{name}.{extension}").is_file()
    assert "Saved by the batch" in (registration_qc / "focus_qc.caption.txt").read_text()
    summary = json.loads(run_qc.summary_path(run).read_text())
    assert summary["preprocessing_and_registration"]["plots"] == results


def test_a_failing_plot_is_recorded_and_the_others_still_saved(processed_run, monkeypatch):
    run, result = processed_run

    def broken(run):
        raise ValueError("no focus today")

    monkeypatch.setattr(run_qc, "focus_plots", broken)
    results = run_qc.save_run_qc(run, result.correction.frame_phases)
    assert results["focus"].startswith("failed: ValueError('no focus today')")
    assert isinstance(results["motion_registration"], list) and results["motion_registration"]


def test_frame_order_is_only_checked_when_asked(processed_run, monkeypatch):
    run, result = processed_run
    called = []
    monkeypatch.setattr(run_qc, "frame_order_plots", lambda run: called.append(run) or [])
    assert "frame_order" not in run_qc.save_run_qc(run, result.correction.frame_phases)
    assert "frame_order" in run_qc.save_run_qc(run, None, frame_order=True) and called


def test_the_fluorescence_plots_are_saved_after_the_extraction(processed_run):
    from mini2p_preprocessing.extraction import run_cellpose_extraction

    run, _ = processed_run
    ops = np.load(run.plane_dir / "ops.npy", allow_pickle=True).item()
    labels = np.zeros((ops["Ly"], ops["Lx"]), np.int32)
    labels[5:9, 5:9], labels[15:19, 20:24], labels[25:29, 10:14] = 1, 2, 3
    seg = run.variant_dir / "masks_seg.npy"
    np.save(seg, {"masks": labels}, allow_pickle=True)
    run_cellpose_extraction(seg, run.plane_dir, run.settings)
    results = run_qc.save_fluorescence_qc(run, Contrast(1, 99.8, 1.0))
    assert isinstance(results["fluorescence"], list), results
    assert {"f0_overview", "mask_alignment", "background", "top_rois_dff"} <= saved_names(results)
    summary = json.loads(run_qc.summary_path(run).read_text())
    assert set(summary) == {"fluorescence"}


def test_the_worker_saves_the_qc_only_when_the_batch_asks(tmp_path):
    from mini2p_preprocessing.batch import qc_flags

    config = tmp_path / "pipeline.toml"
    batch = (
        f'[batch]\nraw_subject_dir = "{tmp_path}"\nderivatives_subject_dir = "{tmp_path}"\n'
        'session_first = 1\nsession_last = 2\nsource_substring = ""\n'
    )
    config.write_text('[distortion]\nmode = "off"\n' + batch)
    assert qc_flags(config) == ["--qc"]
    config.write_text('[distortion]\nmode = "off"\n' + batch + "qc_frame_order = true\n")
    assert qc_flags(config) == ["--qc", "--qc-frame-order"]
    config.write_text('[distortion]\nmode = "off"\n' + batch + "save_qc = false\n")
    assert qc_flags(config) == []
    config.write_text('[distortion]\nmode = "off"\n')
    assert qc_flags(config) == []


def test_a_processed_plane_gives_back_the_run_the_pipeline_made(processed_run):
    run, _ = processed_run
    extraction = run.settings.model_copy(
        update={"fluorescence": run.settings.fluorescence.model_copy(update={"min_f0": 7.0})}
    )
    rebuilt = run_qc.run_qc_for_plane(run.plane_dir, extraction)
    assert (rebuilt.input_tiff, rebuilt.corrected_tiff, rebuilt.suite2p_input) == (
        run.input_tiff, run.corrected_tiff, run.suite2p_input,
    )  # fmt: skip
    assert rebuilt.settings.fluorescence.min_f0 == 7.0
    assert rebuilt.settings.phase == run.settings.phase
