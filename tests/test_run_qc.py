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
    "phase": {"chunk_frames": 20, "max_shift_px": 8},
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


@pytest.fixture(params=["per_frame", "chunked"])
def processed_run(request, tmp_path, recording, monkeypatch):
    """A run the pipeline processed (registration faked), read back from its folder as the
    batch's QC process reads it."""
    raw_tiff, _ = recording
    monkeypatch.setattr(pipeline_module, "run_registration", fake_registration)
    settings = PipelineSettings.model_validate(
        SETTINGS | {"phase": SETTINGS["phase"] | {"method": request.param}}
    )
    out = tmp_path / "run-01" / "without_distortion_correction" / "with_destriping"
    pipeline_module.run_pipeline(raw_tiff, out, settings)
    return run_qc.run_qc_for_variant(out)


def saved_names(results: dict) -> set[str]:
    return {Path(r).name for r in results.values() if not r.startswith(("failed", "not drawn"))}


def test_the_notebook_s_preprocessing_and_registration_plots_are_saved(processed_run):
    run = processed_run
    results = run_qc.save_run_qc(run)
    assert run_qc.failures(results) == []
    common = {
        "raw_meanImg_first55", "before_after_destriping", "before_after_registration",
        "registration_traces", "registered_means_cropped", "focus_qc",
    }  # fmt: skip
    if run.settings.phase.method == "per_frame":
        assert common | {"frame_phases", "before_after_bidiphase"} == saved_names(results)
    else:
        assert common | {"phase_trace", "before_after_first_bidiphase"} <= saved_names(results)
    assert results["motion_registration/nonrigid_offsets"].startswith("not drawn: rigid")
    assert results["distortion_correction"].startswith("not drawn")
    for step in ("raw_mean", "bidiphase", "destriping"):
        assert list((run.variant_dir / QC_FOLDER / step).glob("*.png"))
    registration_qc = run.plane_dir.parent / QC_FOLDER / "motion_registration"
    for name in ("before_after_registration", "focus_qc"):
        for extension in ("png", "pdf", "svg", "caption.txt"):
            assert (registration_qc / f"{name}.{extension}").is_file()
    assert "Saved by the batch" in (registration_qc / "focus_qc.caption.txt").read_text()
    summary = json.loads(run_qc.summary_path(run).read_text())
    assert summary["preprocessing_and_registration"]["plots"] == results


def test_a_failing_plot_is_recorded_and_the_others_still_saved(processed_run, monkeypatch):
    def broken(*args):
        raise ValueError("no focus today")

    monkeypatch.setattr(run_qc, "focus_plot", broken)
    results = run_qc.save_run_qc(processed_run)
    assert run_qc.failures(results) == ["motion_registration/focus_qc"]
    assert results["motion_registration/focus_qc"] == "failed: ValueError('no focus today')"
    assert Path(results["motion_registration/registration_traces"] + ".png").is_file()


def test_a_failing_load_fails_only_the_plots_that_need_it(processed_run, monkeypatch):
    from mini2p_preprocessing import registration

    def unreadable(plane_dir):
        raise OSError("binary unreadable")

    monkeypatch.setattr(registration, "load_registered_outputs", unreadable)
    results = run_qc.save_run_qc(processed_run)
    assert {f for f in run_qc.failures(results)} == {
        "motion_registration/before_after_registration", "motion_registration/registration_traces",
        "motion_registration/nonrigid_offsets", "motion_registration/registered_means_cropped",
        "motion_registration/focus_qc",
    }  # fmt: skip
    assert Path(results["raw_mean"] + ".png").is_file()


def test_frame_order_is_only_checked_when_asked(processed_run, monkeypatch):
    called = []
    monkeypatch.setattr(run_qc, "frame_order_plot", lambda run: called.append(run) or "drawn")
    assert "frame_order" not in run_qc.save_run_qc(processed_run)
    assert run_qc.save_run_qc(processed_run, frame_order=True)["frame_order"] == "drawn"
    assert called


def extract_masks(run):
    from mini2p_preprocessing.extraction import run_cellpose_extraction

    ops = np.load(run.plane_dir / "ops.npy", allow_pickle=True).item()
    labels = np.zeros((ops["Ly"], ops["Lx"]), np.int32)
    labels[5:9, 5:9], labels[15:19, 20:24], labels[25:29, 10:14] = 1, 2, 3
    seg = run.variant_dir / "masks_seg.npy"
    np.save(seg, {"masks": labels}, allow_pickle=True)
    run_cellpose_extraction(seg, run.plane_dir, run.settings)


FLUORESCENCE_PLOTS = {"f0_overview", "mask_alignment", "background", "top_rois_dff"}


def test_the_fluorescence_plots_are_saved_after_the_extraction(processed_run):
    run = processed_run
    extract_masks(run)
    results = run_qc.save_fluorescence_qc(run, Contrast(1, 99.8, 1.0))
    assert run_qc.failures(results) == []
    assert saved_names(results) == FLUORESCENCE_PLOTS
    summary = json.loads(run_qc.summary_path(run).read_text())
    assert set(summary) == {"fluorescence"}


def test_a_photon_free_level_that_cannot_be_measured_leaves_the_other_plots(processed_run):
    run = processed_run
    extract_masks(run)
    fluorescence = run.settings.fluorescence.model_copy(update={"subtract_photon_free_level": True})
    run = run_qc.run_qc_for_plane(
        run.plane_dir, run.settings.model_copy(update={"fluorescence": fluorescence})
    )
    results = run_qc.save_fluorescence_qc(run, Contrast(1, 99.8, 1.0))
    assert run_qc.failures(results) == ["fluorescence/photon_free_levels"]  # too few pixels
    assert saved_names(results) == FLUORESCENCE_PLOTS
    caption = Path(results["fluorescence/f0_overview"] + ".caption.txt").read_text()
    assert "could not be measured" in caption and "not subtracted" in caption


def test_a_plane_gives_back_the_run_with_the_extraction_s_fluorescence(processed_run):
    run = processed_run
    fluorescence = run.settings.fluorescence.model_copy(update={"min_f0": 7.0})
    rebuilt = run_qc.run_qc_for_plane(
        run.plane_dir, run.settings.model_copy(update={"fluorescence": fluorescence})
    )
    assert (rebuilt.input_tiff, rebuilt.corrected_tiff, rebuilt.suite2p_input) == (
        run.input_tiff, run.corrected_tiff, run.suite2p_input,
    )  # fmt: skip
    assert rebuilt.settings.fluorescence.min_f0 == 7.0
    assert rebuilt.settings.phase == run.settings.phase


def test_the_qc_process_exits_with_its_own_code_when_some_plots_fail(processed_run, monkeypatch):
    def broken(*args):
        raise ValueError("no focus today")

    monkeypatch.setattr(run_qc, "focus_plot", broken)
    with pytest.raises(SystemExit) as stopped:
        run_qc.main(["--variant-dir", str(processed_run.variant_dir)])
    assert stopped.value.code == run_qc.EXIT_SOME_FAILED
