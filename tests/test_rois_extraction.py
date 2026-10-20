import json

import numpy as np
import pytest

from mini2p_preprocessing import extraction as extraction_module
from mini2p_preprocessing.config import PipelineSettings
from mini2p_preprocessing.extraction import (
    EXTRACTION_RECORD,
    ExtractionRefused,
    build_stat,
    classify_rois,
    deconvolve,
    run_cellpose_extraction,
)
from mini2p_preprocessing.plane import load_plane_traces
from mini2p_preprocessing.rois import labels_from_stat, masks_to_stat, outline_labels

LY, LX, T = 40, 56, 200
CENTERS = {1: (10, 12), 3: (10, 42), 7: (28, 26)}  # label gaps are allowed
BRIGHTNESS = {1: 250.0, 3: 300.0, 7: 350.0}
SETTINGS = PipelineSettings.model_validate({"distortion": {"mode": "off"}})


def label_image():
    yy, xx = np.mgrid[:LY, :LX]
    labels = np.zeros((LY, LX), np.int32)
    for label, (cy, cx) in CENTERS.items():
        labels[(yy - cy) ** 2 + (xx - cx) ** 2 <= 12] = label
    return labels


def suite2p_like_ops(fs, n_channels):
    """The nested layout Suite2p 1.x writes: {**db, **settings, ...}."""
    return {
        "Ly": LY,
        "Lx": LX,
        "fs": fs,
        "nframes": T,
        "nchannels": n_channels,
        "tau": 1.0,
        "extraction": {"neuropil_coefficient": 0.7, "batch_size": 500},
        "dcnv_preprocess": {
            "baseline": "maximin",
            "win_baseline": 60.0,
            "sig_baseline": 10.0,
            "prctile_baseline": 8.0,
        },
    }


def make_plane(tmp_path, rng, two_channels=True, fs=7.52, texture=False):
    plane = tmp_path / "suite2p" / "plane0"
    plane.mkdir(parents=True)
    labels = label_image()
    movie = 100.0 + rng.normal(0, 1, (T, LY, LX))
    if texture:  # tissue-like structure everywhere, so image registration has something to lock on
        from tests.conftest import textured_image

        movie += 30 * textured_image(rng, shape=(LY, LX))
    for label, value in BRIGHTNESS.items():
        movie[:, labels == label] += value
    movie.astype(np.int16).tofile(plane / "data.bin")
    if two_channels:
        (0.5 * movie).astype(np.int16).tofile(plane / "data_chan2.bin")
    ops = suite2p_like_ops(fs, 2 if two_channels else 1)
    ops["meanImg"] = movie.astype(np.int16).mean(axis=0).astype(np.float32)
    np.save(plane / "ops.npy", ops, allow_pickle=True)
    seg = tmp_path / "meanImg_seg.npy"
    np.save(seg, {"masks": labels}, allow_pickle=True)
    return plane, seg


def test_masks_to_stat_lists_each_label_in_order_with_pixels_and_centre():
    labels = label_image()
    stat = masks_to_stat(labels)
    assert len(stat) == 3
    ypix, xpix = np.nonzero(labels == 3)
    assert np.array_equal(
        np.sort(stat[1]["ypix"] * LX + stat[1]["xpix"]), np.sort(ypix * LX + xpix)
    )
    assert stat[1]["npix"] == len(ypix)
    assert np.allclose(stat[1]["med"], [ypix.mean(), xpix.mean()])
    assert stat[1]["lam"].dtype == np.float32 and np.all(stat[1]["lam"] == 1)


def test_masks_to_stat_rejects_empty_or_non_2d():
    with pytest.raises(ValueError, match="no ROIs"):
        masks_to_stat(np.zeros((5, 5), int))
    with pytest.raises(ValueError, match="2-D"):
        masks_to_stat(np.zeros((2, 5, 5), int))


def test_outlines_keep_only_inner_boundary_with_roi_labels():
    labels = np.zeros((7, 7), int)
    labels[1:6, 1:6] = 4
    outline = outline_labels(labels)
    assert outline[3, 3] == 0 and outline[1, 1] == 4 and outline[1, 3] == 4
    assert set(np.unique(outline)) == {0, 4}


def test_labels_from_stat_round_trips():
    labels = label_image()
    rebuilt = labels_from_stat(masks_to_stat(labels), LY, LX)
    assert np.array_equal(rebuilt > 0, labels > 0)
    assert len(np.unique(rebuilt)) == 4  # background + 3 ROIs, renumbered 1..3


def test_extraction_recovers_roi_brightness_and_writes_suite2p_files(tmp_path, rng):
    plane, seg = make_plane(tmp_path, rng)
    result = run_cellpose_extraction(seg, plane, SETTINGS)

    f = np.load(plane / "F.npy")
    assert f.shape == (3, T)
    assert np.allclose(f.mean(axis=1), [100 + v for v in BRIGHTNESS.values()], atol=2)
    f2 = np.load(plane / "F_chan2.npy")
    assert np.allclose(f2.mean(axis=1), 0.5 * f.mean(axis=1), atol=2)
    for name in ("Fneu.npy", "Fneu_chan2.npy", "spks.npy"):
        assert np.load(plane / name).shape == (3, T)
    assert np.load(plane / "iscell.npy").shape == (3, 2)
    assert len(np.load(plane / "stat.npy", allow_pickle=True)) == 3
    ops = np.load(plane / "ops.npy", allow_pickle=True).item()
    assert ops["extraction"]["neuropil_coefficient"] == 0.7
    record = json.loads((plane / EXTRACTION_RECORD).read_text())
    assert record["n_rois"] == 3 and record["seg_file"] == str(seg.resolve())
    assert result.f.shape == (3, T)


def test_single_channel_plane_writes_no_red_traces(tmp_path, rng):
    plane, seg = make_plane(tmp_path, rng, two_channels=False)
    run_cellpose_extraction(seg, plane, SETTINGS)
    assert (plane / "F.npy").is_file() and not (plane / "F_chan2.npy").exists()


def test_mask_from_a_different_image_size_writes_nothing(tmp_path, rng):
    plane, _ = make_plane(tmp_path, rng)
    wrong = tmp_path / "wrong_seg.npy"
    np.save(wrong, {"masks": np.ones((LY + 2, LX), int)}, allow_pickle=True)
    with pytest.raises(ValueError, match="shape"):
        run_cellpose_extraction(wrong, plane, SETTINGS)
    assert not (plane / "F.npy").exists()


def test_frame_rate_disagreement_between_ops_and_config_fails(tmp_path, rng):
    plane, seg = make_plane(tmp_path, rng, fs=30.0)
    with pytest.raises(ValueError, match="fs"):
        run_cellpose_extraction(seg, plane, SETTINGS)


def test_extract_cli_reads_config_and_writes_traces(tmp_path, rng):
    from mini2p_preprocessing.cli import extract_main

    plane, seg = make_plane(tmp_path, rng)
    config = tmp_path / "pipeline.toml"
    config.write_text('[distortion]\nmode = "off"\n[fluorescence]\nneuropil_coefficient = 0.5\n')
    extract_main(["--config", str(config), "--plane-dir", str(plane), "--seg", str(seg)])
    ops = np.load(plane / "ops.npy", allow_pickle=True).item()
    assert ops["extraction"]["neuropil_coefficient"] == 0.5


def test_pure_steps_work_without_disk(rng):
    labels = label_image()
    stat = build_stat(labels, LY, LX, diameter=12)
    assert len(stat) == len(CENTERS)  # Cellpose labels never overlap, so none are removed
    f = 300 + rng.normal(0, 5, (len(stat), T)).astype(np.float32)
    f[:, 50:55] += 200
    fneu = np.full_like(f, 100.0)
    iscell = classify_rois(stat, f, fneu, 0.7)
    assert iscell.shape == (len(stat), 2)
    assert {"skew", "std", "snr"} <= set(stat[0])  # activity features the classifier uses
    spks = deconvolve(f, fneu, 7.52, SETTINGS.fluorescence)
    assert spks.shape == f.shape


def test_classifier_sees_skew_after_full_extraction(tmp_path, rng):
    plane, seg = make_plane(tmp_path, rng)
    run_cellpose_extraction(seg, plane, SETTINGS)
    stat = np.load(plane / "stat.npy", allow_pickle=True)
    assert all({"skew", "std", "snr"} <= set(roi) for roi in stat)


def test_suite2p_settings_nested_in_ops_are_updated(tmp_path, rng):
    plane, seg = make_plane(tmp_path, rng)
    settings = SETTINGS.model_copy(
        update={
            "fluorescence": SETTINGS.fluorescence.model_copy(
                update={"neuropil_coefficient": 0.5, "baseline": "prctile", "tau_s": 0.7}
            )
        }
    )
    run_cellpose_extraction(seg, plane, settings)
    ops = np.load(plane / "ops.npy", allow_pickle=True).item()
    assert ops["extraction"]["neuropil_coefficient"] == 0.5
    assert ops["dcnv_preprocess"]["baseline"] == "prctile"
    assert ops["tau"] == 0.7


def test_registered_binaries_are_found_where_ops_says(tmp_path, rng):
    plane, seg = make_plane(tmp_path, rng)
    elsewhere = tmp_path / "fast_disk"
    elsewhere.mkdir()
    (plane / "data_chan2.bin").rename(elsewhere / "data_chan2.bin")
    ops = np.load(plane / "ops.npy", allow_pickle=True).item()
    ops["reg_file_chan2"] = str(elsewhere / "data_chan2.bin")
    np.save(plane / "ops.npy", ops, allow_pickle=True)
    run_cellpose_extraction(seg, plane, SETTINGS)
    assert (plane / "F_chan2.npy").is_file()


def test_missing_red_binary_for_two_channel_plane_fails_before_writing(tmp_path, rng):
    plane, seg = make_plane(tmp_path, rng)
    (plane / "data_chan2.bin").unlink()
    with pytest.raises(ExtractionRefused, match="data_chan2.bin"):
        run_cellpose_extraction(seg, plane, SETTINGS)
    assert not (plane / "F.npy").exists()


def test_single_channel_extraction_removes_stale_red_traces(tmp_path, rng):
    plane, seg = make_plane(tmp_path, rng, two_channels=False)
    np.save(plane / "F_chan2.npy", np.zeros((9, T)))
    np.save(plane / "Fneu_chan2.npy", np.zeros((9, T)))
    run_cellpose_extraction(seg, plane, SETTINGS)
    assert not (plane / "F_chan2.npy").exists() and not (plane / "Fneu_chan2.npy").exists()


def test_failed_write_leaves_previous_outputs_untouched(tmp_path, rng, monkeypatch):
    plane, seg = make_plane(tmp_path, rng)
    run_cellpose_extraction(seg, plane, SETTINGS)
    before = {p.name: p.read_bytes() for p in plane.glob("*.npy")}
    real_save, calls = np.save, []

    def flaky_save(file, *args, **kwargs):
        calls.append(file)
        if len(calls) == 3:
            raise OSError("disk full")
        return real_save(file, *args, **kwargs)

    monkeypatch.setattr(extraction_module.np, "save", flaky_save)
    with pytest.raises(OSError, match="disk full"):
        run_cellpose_extraction(seg, plane, SETTINGS)
    monkeypatch.undo()
    after = {p.name: p.read_bytes() for p in plane.glob("*.npy")}
    assert after == before
    assert not [p for p in plane.iterdir() if ".tmp" in p.name]


def test_loading_mismatched_trace_files_fails(tmp_path, rng):
    plane, seg = make_plane(tmp_path, rng)
    run_cellpose_extraction(seg, plane, SETTINGS)
    np.save(plane / "iscell.npy", np.zeros((5, 2)))
    with pytest.raises(ValueError, match="ROI"):
        load_plane_traces(plane)


def test_constant_traces_are_refused_by_the_classifier_step():
    stat = build_stat(label_image(), LY, LX, diameter=12)
    f = np.full((len(stat), T), 300.0, np.float32)
    f[0] += np.random.default_rng(0).normal(0, 5, T)  # ROI 0 varies; ROIs 1 and 2 are flat
    with pytest.raises(ExtractionRefused, match=r"\[1, 2\].*constant"):
        classify_rois(stat, f, np.zeros_like(f), 0.7)


def test_ops_without_channel_count_fails_instead_of_guessing(tmp_path, rng):
    plane, seg = make_plane(tmp_path, rng)
    ops = np.load(plane / "ops.npy", allow_pickle=True).item()
    del ops["nchannels"]
    np.save(plane / "ops.npy", ops, allow_pickle=True)
    np.save(plane / "F_chan2.npy", np.ones((3, T)))
    with pytest.raises(ExtractionRefused, match="nchannels"):
        run_cellpose_extraction(seg, plane, SETTINGS)
    assert (plane / "F_chan2.npy").is_file()


def offset(array, dy, dx):
    """``array`` moved by (dy, dx); labels are zero-filled, images mean-filled."""
    from mini2p_preprocessing.alignment import shift_image

    fill = 0 if array.dtype.kind in "iu" else float(array.mean())
    return shift_image(array, dy, dx, fill)


def test_masks_from_another_registration_are_shifted_onto_the_plane(tmp_path, rng):
    from tests.conftest import as_contrast_png

    plane, seg = make_plane(tmp_path, rng, texture=True)
    run_cellpose_extraction(seg, plane, SETTINGS)
    f_reference = np.load(plane / "F.npy")

    ops = np.load(plane / "ops.npy", allow_pickle=True).item()
    other_image = offset(ops["meanImg"], -2, 3)  # the other registration sits 2 up, 3 right
    image_file = as_contrast_png(other_image, tmp_path / "meanImg_green_contrast.png")
    other_seg = tmp_path / "other_seg.npy"
    np.save(other_seg, {"masks": offset(label_image(), -2, 3)}, allow_pickle=True)

    run_cellpose_extraction(other_seg, plane, SETTINGS, mask_image=image_file)
    assert np.allclose(np.load(plane / "F.npy"), f_reference)
    record = json.loads((plane / EXTRACTION_RECORD).read_text())
    assert (record["mask_alignment"]["dy"], record["mask_alignment"]["dx"]) == (2, -3)
    assert record["mask_image"] == str(image_file.resolve())

    run_cellpose_extraction(other_seg, plane, SETTINGS)  # no alignment: wrong pixels
    assert not np.allclose(np.load(plane / "F.npy"), f_reference)
    assert json.loads((plane / EXTRACTION_RECORD).read_text())["mask_alignment"] is None


def test_mask_image_of_a_different_size_fails_before_writing(tmp_path, rng):
    import matplotlib.pyplot as plt

    plane, seg = make_plane(tmp_path, rng, texture=True)
    image_file = tmp_path / "wrong.png"
    plt.imsave(image_file, np.zeros((LY, LX + 4)), cmap="gray")
    with pytest.raises(ValueError, match="shape"):
        run_cellpose_extraction(seg, plane, SETTINGS, mask_image=image_file)
    assert not (plane / "F.npy").exists()
