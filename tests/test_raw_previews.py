import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import tifffile  # noqa: E402

from mini2p_preprocessing import raw_previews  # noqa: E402
from mini2p_preprocessing.config import ChannelSettings  # noqa: E402

NAMES = ChannelSettings().names


def acquired_tiff(path, epoch, rng, n_frames=6, value=None):
    """A 2-channel ScanImage-like TIFF whose headers carry the acquisition start *epoch*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = " ".join(str(v) for v in epoch)
    with tifffile.TiffWriter(path) as writer:
        for timepoint in range(n_frames):
            for _ in range(2):
                page = (
                    rng.integers(0, 300, (8, 8)) if value is None else np.full((8, 8), value)
                ).astype(np.int16)
                header = f"frameNumbers = {timepoint + 1}\nepoch = [{stamp}]\n"
                writer.write(page, photometric="minisblack", description=header)
    return path


def raw_subject(root):
    return root / "rawdata" / "cohort-01" / "sub-03_id-MY_71_R"


def derived_subject(root):
    return root / "derivatives" / "cohort-01" / "sub-03_id-MY_71_R"


def preview_dir(root, session="ses-1_date-20260101"):
    return derived_subject(root) / session / "funcimg" / "preview-all-tifs"


def test_tiffs_of_a_session_are_ordered_by_acquisition_start_not_by_name(tmp_path, rng):
    funcimg = raw_subject(tmp_path) / "ses-137_date-20260526" / "funcimg"
    acquired_tiff(funcimg / "task2_00001.tif", [2026, 5, 26, 14, 5, 2.5], rng)
    acquired_tiff(funcimg / "zz_task_00009.tif", [2026, 5, 26, 9, 30, 0], rng)
    acquired_tiff(funcimg / "a_test_00003.tif", [2026, 5, 26, 16, 0, 0], rng)
    raws = raw_previews.find_raw_tiffs(raw_subject(tmp_path))
    assert [r.path.name for r in raws] == [
        "zz_task_00009.tif",
        "task2_00001.tif",
        "a_test_00003.tif",
    ]
    assert raws[1].started == datetime(2026, 5, 26, 14, 5, 2, 500000)
    assert {r.started_from for r in raws} == {"ScanImage epoch"}


def test_a_tiff_without_an_epoch_is_ordered_by_its_file_time(tmp_path):
    path = raw_subject(tmp_path) / "ses-1_date-20260101" / "funcimg" / "plain.tif"
    path.parent.mkdir(parents=True)
    tifffile.imwrite(path, np.zeros((4, 8, 8), np.int16), photometric="minisblack")
    (raw,) = raw_previews.find_raw_tiffs(raw_subject(tmp_path))
    assert raw.started_from == "file modification time"


def test_find_raw_tiffs_skips_non_sessions_and_refuses_an_empty_subject(tmp_path, rng):
    subject = raw_subject(tmp_path)
    funcimg = subject / "ses-99_date-20260501" / "funcimg"
    acquired_tiff(funcimg / "b.tif", [2026, 5, 1, 9, 0, 0], rng)
    (subject / "ses-100_date-20260502").mkdir()  # no funcimg folder
    (subject / "notes").mkdir()
    assert [r.session for r in raw_previews.find_raw_tiffs(subject)] == ["ses-99_date-20260501"]
    with pytest.raises(FileNotFoundError, match="No raw TIFFs"):
        raw_previews.find_raw_tiffs(tmp_path / "rawdata")


def test_derivatives_subject_dir_replaces_rawdata(tmp_path):
    assert raw_previews.derivatives_subject_dir(raw_subject(tmp_path)) == derived_subject(tmp_path)
    with pytest.raises(ValueError, match="rawdata"):
        raw_previews.derivatives_subject_dir(tmp_path / "elsewhere" / "sub-03")


def test_previews_go_in_the_preview_folder_and_are_not_redone(tmp_path, rng):
    subject = raw_subject(tmp_path)
    funcimg = subject / "ses-137_date-20260526" / "funcimg"
    acquired_tiff(funcimg / "task2_00001.tif", [2026, 5, 26, 14, 0, 0], rng, value=7)
    acquired_tiff(funcimg / "pretask2_00004.tif", [2026, 5, 26, 10, 0, 0], rng, value=3)
    run_folder = derived_subject(tmp_path) / "ses-137_date-20260526" / "funcimg" / "run-01"
    run_folder.mkdir(parents=True)
    (run_folder / "source_tiff.json").write_text("{}")  # the batch's: left as it is
    rows = raw_previews.write_raw_previews([subject], None, 4, 2, NAMES)

    assert [(Path(r["tiff"]).name, r["status"]) for r in rows] == [
        ("pretask2_00004.tif", "written"), ("task2_00001.tif", "written")]  # fmt: skip
    folder = preview_dir(tmp_path, "ses-137_date-20260526")
    for suffix in (".png", ".pdf", ".svg", ".caption.txt"):
        assert (folder / f"task2_00001_raw_meanImg_first4{suffix}").exists()
    assert "task2_00001.tif" in (folder / "task2_00001_raw_meanImg_first4.caption.txt").read_text()
    assert sorted(p.name for p in run_folder.iterdir()) == ["source_tiff.json"]
    derived = derived_subject(tmp_path)
    assert (derived / "preview-all-tifs_raw_meanImg_first4.pdf").exists()
    summary = json.loads((derived / "raw_mean_previews.json").read_text())
    assert summary["complete"] and len(summary["tiffs"]) == 2

    rows = raw_previews.write_raw_previews([subject], None, 4, 2, NAMES)  # nothing read again
    assert [r["status"] for r in rows] == ["exists", "exists"]
    rows = raw_previews.write_raw_previews([subject], None, 5, 2, NAMES)  # other frame count
    assert [r["status"] for r in rows] == ["written", "written"]


def test_names_with_dots_keep_their_own_files(tmp_path, rng):
    subject = raw_subject(tmp_path)
    funcimg = subject / "ses-1_date-20260101" / "funcimg"
    acquired_tiff(funcimg / "fov1.2_00001.tif", [2026, 1, 1, 9, 0, 0], rng)
    acquired_tiff(funcimg / "fov1.3_00002.tif", [2026, 1, 1, 9, 5, 0], rng)
    rows = raw_previews.write_raw_previews([subject], None, 4, 2, NAMES)
    assert len({r["figure"] for r in rows}) == 2
    assert (preview_dir(tmp_path) / "fov1.2_00001_raw_meanImg_first4.png").exists()
    rows = raw_previews.write_raw_previews([subject], None, 4, 2, NAMES)
    assert [r["status"] for r in rows] == ["exists", "exists"]


def test_a_shorter_recording_is_not_read_again(tmp_path, rng):
    subject = raw_subject(tmp_path)
    funcimg = subject / "ses-1_date-20260101" / "funcimg"
    acquired_tiff(funcimg / "s.tif", [2026, 1, 1, 9, 0, 0], rng, n_frames=3)
    (row,) = raw_previews.write_raw_previews([subject], None, 500, 2, NAMES)
    assert row["figure"].endswith("s_raw_meanImg_first3.png")
    (row,) = raw_previews.write_raw_previews([subject], None, 500, 2, NAMES)
    assert row["status"] == "exists"


def test_a_preview_of_another_version_of_the_file_is_redone(tmp_path, rng):
    subject = raw_subject(tmp_path)
    funcimg = subject / "ses-1_date-20260101" / "funcimg"
    acquired_tiff(funcimg / "only_red.tif", [2026, 1, 1, 9, 0, 0], rng, n_frames=3)
    (row,) = raw_previews.write_raw_previews([subject], None, 500, 2, NAMES)
    assert row["status"] == "written"
    # the copy was still growing: the full file is larger, so its preview is made again
    acquired_tiff(funcimg / "only_red.tif", [2026, 1, 1, 9, 0, 0], rng, n_frames=8)
    (row,) = raw_previews.write_raw_previews([subject], None, 500, 2, NAMES)
    assert row["status"] == "written" and row["figure"].endswith("first8.png")


def test_an_unreadable_tiff_is_reported_and_the_rest_go_ahead(tmp_path, rng):
    subject = raw_subject(tmp_path)
    funcimg = subject / "ses-1_date-20260101" / "funcimg"
    acquired_tiff(funcimg / "good.tif", [2026, 1, 1, 9, 0, 0], rng)
    (funcimg / "broken.tif").write_bytes(b"not a tiff")
    rows = raw_previews.write_raw_previews([subject], None, 4, 2, NAMES)
    assert sorted(r["status"] for r in rows) == ["failed", "written"]


def test_a_tiff_that_tifffile_cannot_parse_in_any_way_is_reported(tmp_path, rng, monkeypatch):
    subject = raw_subject(tmp_path)
    funcimg = subject / "ses-1_date-20260101" / "funcimg"
    acquired_tiff(funcimg / "odd.tif", [2026, 1, 1, 9, 0, 0], rng)

    def unexpected(*_):
        raise TypeError("odd header")  # tifffile raises this for some corrupt headers

    monkeypatch.setattr(raw_previews, "raw_mean_images", unexpected)
    (row,) = raw_previews.write_raw_previews([subject], None, 4, 2, NAMES)
    assert row["status"] == "failed" and row["error"] == "TypeError: odd header"


def test_a_plotting_error_stops_the_run_and_leaves_a_partial_summary(tmp_path, rng, monkeypatch):
    subject = raw_subject(tmp_path)
    funcimg = subject / "ses-1_date-20260101" / "funcimg"
    acquired_tiff(funcimg / "a.tif", [2026, 1, 1, 9, 0, 0], rng)

    def broken(*_):
        raise RuntimeError("plot bug")

    monkeypatch.setattr(raw_previews, "draw_preview", broken)
    with pytest.raises(RuntimeError, match="plot bug"):
        raw_previews.write_raw_previews([subject], None, 4, 2, NAMES)
    summary = json.loads((derived_subject(tmp_path) / "raw_mean_previews.json").read_text())
    assert summary["complete"] is False


def test_write_raw_previews_checks_frames_and_channel_names_first(tmp_path, rng):
    subject = raw_subject(tmp_path)
    acquired_tiff(subject / "ses-1_date-20260101" / "funcimg" / "a.tif", [2026, 1, 1, 9, 0, 0], rng)
    with pytest.raises(ValueError, match="at least one frame"):
        raw_previews.write_raw_previews([subject], None, 0, 2, NAMES)
    with pytest.raises(ValueError, match="do not cover"):
        raw_previews.write_raw_previews([subject], None, 4, 3, NAMES)


def test_ties_are_broken_by_name_and_macos_dot_files_are_skipped(tmp_path, rng):
    funcimg = raw_subject(tmp_path) / "ses-1_date-20260101" / "funcimg"
    acquired_tiff(funcimg / "b.tif", [2026, 1, 1, 9, 0, 0], rng)
    acquired_tiff(funcimg / "a.tif", [2026, 1, 1, 9, 0, 0], rng)
    (funcimg / "._a.tif").write_bytes(b"AppleDouble")
    raws = raw_previews.find_raw_tiffs(raw_subject(tmp_path))
    assert [r.path.name for r in raws] == ["a.tif", "b.tif"]


def test_a_session_mixing_start_times_and_file_times_warns(tmp_path, rng, caplog):
    funcimg = raw_subject(tmp_path) / "ses-1_date-20260101" / "funcimg"
    acquired_tiff(funcimg / "a.tif", [2026, 1, 1, 9, 0, 0], rng)
    tifffile.imwrite(funcimg / "plain.tif", np.zeros((4, 8, 8), np.int16), photometric="minisblack")
    with caplog.at_level("WARNING"):
        raw_previews.find_raw_tiffs(raw_subject(tmp_path))
    assert "file time" in caplog.text


def test_epoch_seconds_just_below_a_minute_are_read(tmp_path, rng):
    path = acquired_tiff(tmp_path / "e.tif", [2026, 1, 1, 9, 0, 59.9999996], rng)
    start, source = raw_previews.acquisition_start(path)
    assert source == "ScanImage epoch" and start == datetime(2026, 1, 1, 9, 1, 0)


def test_raw_means_command_reports_failures_with_exit_code_1(tmp_path, rng, capsys):
    from mini2p_preprocessing.cli import raw_means_main

    subject = raw_subject(tmp_path)
    funcimg = subject / "ses-1_date-20260101" / "funcimg"
    acquired_tiff(funcimg / "r.tif", [2026, 1, 1, 9, 0, 0], rng)
    config = tmp_path / "pipeline.toml"
    config.write_text("[distortion]\nmode = 'off'\n")
    args = ["--config", str(config), "--subject-dir", str(subject), "--frames", "4"]
    raw_means_main(args)
    assert "1 of 1 TIFFs" in capsys.readouterr().out
    (funcimg / "broken.tif").write_bytes(b"not a tiff")
    with pytest.raises(SystemExit) as stopped:
        raw_means_main(args)
    assert stopped.value.code == 1
    assert "FAILED" in capsys.readouterr().out
