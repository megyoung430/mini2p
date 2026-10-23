"""The NATEX port, against line-by-line transcriptions of the MATLAB code and synthetic cells."""

from __future__ import annotations

import csv
import json
import math

import numpy as np
import pytest
from scipy import signal

from mini2p_preprocessing.natex import (
    NatexSettings,
    clean_events,
    frames_in,
    load_natex_settings,
    lowpass,
    matlab_percentile,
    moving_std,
    natex_dff,
    percentile_baseline,
    process_traces,
    significant_frames,
    snr,
    write_natex,
)

FS = 7.52
RNG = np.random.default_rng(7)

# ---------------------------------------------------------------- MATLAB, transcribed
# Each reference below is the MATLAB loop with 1-based indices kept as written
# (``x[i - 1]`` for MATLAB's ``x(i)``), so the port can be checked against it directly.


def matlab_prctile(values, p):
    """MATLAB prctile, from its definition: sorted values at 100 (i - 0.5) / n."""
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    positions = 100 * (np.arange(1, n + 1) - 0.5) / n
    return float(np.interp(p, positions, x))


def reference_smooth_percentile(F, TimeWindow, percentile):
    Length = len(F)
    F_zero = np.array(F, dtype=float)
    HalfSpan = math.ceil(TimeWindow / 2)
    for i in range(1 + HalfSpan, Length - HalfSpan + 1):
        F_zero[i - 1] = matlab_prctile(F[i - HalfSpan - 1 : i + HalfSpan], percentile)
    for i in range(1, HalfSpan + 1):
        F_zero[i - 1] = matlab_prctile(F[i - 1 : i + 2 * HalfSpan], percentile)
    for i in range(Length - HalfSpan + 1, Length + 1):
        F_zero[i - 1] = matlab_prctile(F[i - 2 * HalfSpan - 1 : i], percentile)
    return F_zero


def reference_movstd(x, k):
    """MATLAB movstd from its window rule: frames j with t - k/2 <= j < t + k/2, shrunk at
    the ends (k odd: centred; k even: one more before; k non-integer: the same rule)."""
    index = np.arange(len(x))
    out = np.empty(len(x))
    for t in range(len(x)):
        window = x[(index >= t - k / 2) & (index < t + k / 2)]
        out[t] = np.std(window, ddof=1) if len(window) > 1 else 0.0
    return out


def reference_significant(Significant_tem, n, e):
    """The two loops of DetaF_T_baselinecorrect, n = ceil(T fs), e = ceil(E fs)."""
    L = len(Significant_tem)
    Significant = np.zeros(L)
    for j in range(1, L - n - e + 1):
        if np.sum(Significant_tem[j - 1 : j + n]) >= n:
            Significant[j - 1 : j + n + e] = 1
    for j in range(1 + n + e, L + 1):
        if np.sum(Significant_tem[j - n - 1 : j]) >= n:
            Significant[j - n - e - 1 : j] = 1
    return Significant.astype(bool)


def reference_snr(RawTrancient, signalTime, restingTime):
    SignalPeriod = RawTrancient[signalTime]
    Peaks = SignalPeriod[SignalPeriod >= matlab_prctile(SignalPeriod, 90)]
    Signal = np.median(Peaks) - np.median(RawTrancient[restingTime])
    noise = np.std(RawTrancient, ddof=1)
    SNR = Signal / noise
    return 0.0 if np.isnan(SNR) else SNR


def reference_events(EventTrain, Significant, detaF_F):
    Event_baseline = ~Significant
    Event_uncertainty = np.mean(EventTrain[Event_baseline])
    FilteredEvent = (EventTrain - Event_uncertainty) * Significant
    FilteredEvent[FilteredEvent <= 0] = 0
    Outliers = FilteredEvent > 2 * matlab_prctile(FilteredEvent, 99.9)
    FilteredEvent[Outliers] = matlab_prctile(FilteredEvent, 99.95)
    dF_F_MAX = np.max(detaF_F * Significant)  # detaF_F_filtered = detaF_F .* Significant
    return FilteredEvent * dF_F_MAX / np.max(FilteredEvent)


def calcium_cell(n_frames=3000, n_events=12, amplitude=1.0, noise=0.08, baseline=200.0):
    """A cell's corrected F: a slow drift, decaying transients and noise."""
    t = np.arange(n_frames) / FS
    drift = 1 + 0.1 * np.sin(2 * np.pi * t / 300)
    kernel = np.exp(-np.arange(int(6 * FS)) / (1.5 * FS))
    spikes = np.zeros(n_frames)
    onsets = np.sort(RNG.choice(np.arange(200, n_frames - 100), n_events, replace=False))
    spikes[onsets] = amplitude
    calcium = np.convolve(spikes, kernel)[:n_frames]
    trace = baseline * drift * (1 + calcium) + baseline * noise * RNG.standard_normal(n_frames)
    return trace, onsets, spikes


# ---------------------------------------------------------------- building blocks


def test_percentile_is_matlab_prctile():
    for n in (1, 2, 5, 37):
        values = RNG.normal(size=n)
        for p in (0.0, 8.0, 50.0, 90.0, 99.95, 100.0):
            assert matlab_percentile(values, p) == pytest.approx(matlab_prctile(values, p))


@pytest.mark.parametrize(
    ("length", "window"), [(50, 7), (50, 8), (41, 20), (30, 20), (29, 20), (300, 226), (400, 226)]
)
def test_percentile_baseline_matches_smooth_percentile(length, window):
    trace = RNG.normal(size=length).cumsum()
    if length < 3 * math.ceil(window / 2):  # MATLAB indexes past the end
        with pytest.raises(ValueError, match="shorter"):
            percentile_baseline(trace, window, 8)
        return
    np.testing.assert_allclose(
        percentile_baseline(trace, window, 8), reference_smooth_percentile(trace, window, 8)
    )


@pytest.mark.parametrize("window", [1, 2, 5, 6, 5.64, 225.6, 226])
def test_moving_std_matches_movstd(window):
    trace = RNG.normal(size=400) * 3 + 1000  # an offset, as raw F has
    np.testing.assert_allclose(
        moving_std(trace, window), reference_movstd(trace, window), atol=1e-8
    )


def test_moving_std_windows_at_the_natex_frame_rate():
    """0.75 s and 30 s at 7.52 Hz are 5 and 225 frames, centred."""
    trace = np.zeros(400)
    trace[200] = 1.0
    spread = np.flatnonzero(moving_std(trace, 0.75 * FS) > 1e-6)  # cumulative-sum rounding
    assert spread.tolist() == [198, 199, 200, 201, 202]
    assert np.count_nonzero(moving_std(trace, 30 * FS) > 1e-6) == 225


@pytest.mark.parametrize(("n", "e"), [(6, 4), (1, 0), (3, 0), (6, 0)])
def test_significant_frames_match_both_matlab_loops(n, e):
    for density in (0.2, 0.6, 0.9):
        above = RNG.random(300) < density
        np.testing.assert_array_equal(
            significant_frames(above, n, e), reference_significant(above, n, e)
        )


def test_frames_are_rounded_up_as_in_natex():
    assert frames_in(0.75, FS) == 6  # 5.64
    assert frames_in(0.4, FS) == 4  # 3.008


def test_lowpass_keeps_slow_signals_and_removes_fast_ones_without_delay():
    t = np.arange(4000) / FS
    slow, fast = np.sin(2 * np.pi * 0.2 * t), np.sin(2 * np.pi * 3.5 * t)
    out = lowpass(slow + fast, 2.0, FS)
    core = slice(300, -300)
    np.testing.assert_allclose(out[core], slow[core], atol=0.01)  # in phase: no delay
    taps = signal.kaiserord(60, (0.15 * (FS / 2 - 2.0)) / (FS / 2))[0]
    assert taps < len(t)
    with pytest.raises(ValueError, match="Nyquist"):
        lowpass(slow, 4.0, FS)
    with pytest.raises(ValueError, match="shorter than the filter"):
        lowpass(slow[:40], 2.0, FS)


# ---------------------------------------------------------------- one cell


def reference_baselinecorrect(F_raw, fs, s: NatexSettings):
    """DetaF_T_baselinecorrect up to the significance step, with the port's low-pass."""
    window = s.moving_window_s * fs
    F_zero_uncorrected = reference_smooth_percentile(F_raw, math.ceil(window), 8)
    F_std_local = reference_movstd(F_raw, window)
    lo, hi = F_std_local.min(), F_std_local.max()
    quiet = F_std_local < lo + 0.1 * (hi - lo)
    F_M = np.mean(F_raw[quiet] - F_zero_uncorrected[quiet])
    F_zero = F_zero_uncorrected + F_M
    F_lowpassed = lowpass(F_raw, s.lowpass_hz, fs)
    DetaF_F_filtered = (F_lowpassed - F_zero) / F_zero
    DetaF_raw = (F_raw - F_zero) / F_zero
    local = reference_movstd(DetaF_F_filtered, s.transient_s * fs)
    Significant = reference_significant(
        DetaF_F_filtered > 2 * local, math.ceil(s.transient_s * fs), math.ceil(s.extend_s * fs)
    )
    return F_zero, DetaF_raw, DetaF_F_filtered, Significant


def test_natex_dff_follows_detaf_t_baselinecorrect():
    trace, _, _ = calcium_cell(n_frames=1200)
    settings = NatexSettings()
    result = natex_dff(trace, FS, settings)
    f0, dff, dff_lp, significant = reference_baselinecorrect(trace, FS, settings)
    np.testing.assert_allclose(result.f0, f0, rtol=1e-10)
    np.testing.assert_allclose(result.dff, dff, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(result.dff_lowpassed, dff_lp, rtol=1e-8, atol=1e-12)
    np.testing.assert_array_equal(result.significant, significant)


def test_transients_are_found_where_the_cell_fired():
    trace, onsets, _ = calcium_cell(n_frames=3000, n_events=10, amplitude=1.5)
    result = natex_dff(trace, FS, NatexSettings())
    hit = [result.significant[o : o + int(2 * FS)].any() for o in onsets]
    assert np.mean(hit) >= 0.9
    assert result.significant.mean() < 0.4  # not the whole trace
    quiet = natex_dff(200 + 10 * RNG.standard_normal(3000), FS, NatexSettings())
    assert quiet.significant.mean() < 0.05  # noise alone rarely passes


def test_baseline_tracks_slow_drift_not_transients():
    trace, _, _ = calcium_cell(n_frames=3000, n_events=6, amplitude=2.0, noise=0.03)
    result = natex_dff(trace, FS, NatexSettings())
    assert np.median(result.dff[~result.significant]) == pytest.approx(0, abs=0.05)


def test_dff_is_nan_where_f0_is_not_positive_and_snr_uses_the_rest(caplog):
    trace, _, _ = calcium_cell(n_frames=1500)
    trace[:300] = -5.0
    result = natex_dff(trace, FS, NatexSettings())
    assert np.isnan(result.dff[:100]).all() and np.isfinite(result.dff[-100:]).all()
    defined = np.isfinite(result.dff)
    expected = reference_snr(
        result.dff[defined], result.significant[defined], ~result.significant[defined]
    )
    assert snr(result.dff, result.significant) == pytest.approx(expected)
    assert expected > 0
    plane = process_traces(
        np.stack([trace, trace + 1]), np.zeros((2, 1500)), np.zeros((2, 1500)), FS, 0.0,
        NatexSettings(),
    )  # fmt: skip
    assert plane.undefined_fraction[0] == pytest.approx(np.isnan(result.dff).mean())
    assert "F0 <= 0 in some frames of 2 ROIs" in caplog.text


def test_snr_matches_calculatesnr_and_ranks_cells():
    trace, _, _ = calcium_cell()
    result = natex_dff(trace, FS, NatexSettings())
    expected = reference_snr(result.dff, result.significant, ~result.significant)
    assert snr(result.dff, result.significant) == pytest.approx(expected)
    weak = natex_dff(calcium_cell(amplitude=0.3)[0], FS, NatexSettings())
    assert snr(result.dff, result.significant) > snr(weak.dff, weak.significant)
    assert snr(result.dff, np.zeros(len(result.dff), bool)) == 0.0  # no transients


def test_clean_events_matches_natex():
    trace, _, spikes = calcium_cell()
    result = natex_dff(trace, FS, NatexSettings())
    spks = spikes * 50 + RNG.random(len(spikes))
    spks[RNG.integers(0, len(spks))] = 5000  # an outlier
    np.testing.assert_allclose(
        clean_events(spks, result.significant, result.dff),
        reference_events(spks, result.significant, result.dff),
    )
    cleaned = clean_events(spks, result.significant, result.dff)
    assert np.all(cleaned[~result.significant] == 0)
    assert cleaned.max() == pytest.approx(np.max(result.dff[result.significant]))
    assert not clean_events(np.zeros(len(spks)), result.significant, result.dff).any()


def test_clean_events_never_go_negative():
    """MATLAB's scale, max(detaF_F .* Significant), counts the zeros outside transients."""
    significant = np.zeros(200, bool)
    significant[50:60] = True
    dff = np.full(200, -0.1)
    spks = np.zeros(200)
    spks[55] = 10
    cleaned = clean_events(spks, significant, dff)
    np.testing.assert_allclose(cleaned, np.nan_to_num(reference_events(spks, significant, dff)))
    assert not cleaned.any()


def test_sparse_events_are_lost_as_in_natex():
    """With events in under 0.1 % of frames the 99.9th percentile is 0, so NATEX caps them all
    at the 99.95th, also 0: a known property of the original, kept."""
    significant = np.ones(7000, bool)
    spks = np.zeros(7000)
    spks[[100, 2000, 5000]] = 50
    assert not clean_events(spks, significant, np.ones(7000)).any()


# ---------------------------------------------------------------- a plane


def plane_arrays(n_cells=4, n_frames=1500):
    traces = [calcium_cell(n_frames=n_frames, amplitude=a) for a in np.linspace(0.3, 2, n_cells)]
    f = np.stack([t for t, _, _ in traces])
    fneu = 0.3 * f + 20 * RNG.standard_normal(f.shape)
    spks = np.stack([s for _, _, s in traces]) * 40
    return f, fneu, spks


def test_process_traces_corrects_neuropil_and_marks_bad_cells():
    f, fneu, spks = plane_arrays()
    f[2] = 0  # a cell with no signal
    fneu[2] = 0
    result = process_traces(f, fneu, spks, FS, 0.7, NatexSettings())
    assert result.bad.tolist() == [False, False, True, False]
    single = natex_dff(f[0] - 0.7 * fneu[0], FS, NatexSettings())
    np.testing.assert_allclose(result.dff[0], single.dff)
    assert result.event_count[0] == (result.events[0] > 0).sum()
    assert result.snr[3] > result.snr[0]  # the stronger cell
    with pytest.raises(ValueError, match="differ"):
        process_traces(f, fneu[:, :-1], spks, FS, 0.7, NatexSettings())


def test_written_results_and_record(tmp_path):
    f, fneu, spks = plane_arrays(n_cells=3, n_frames=900)
    for name, array in (("F.npy", f), ("Fneu.npy", fneu), ("spks.npy", spks)):
        np.save(tmp_path / name, array)
    np.save(tmp_path / "iscell.npy", np.ones((3, 2)))
    np.save(tmp_path / "ops.npy", {"fs": FS}, allow_pickle=True)
    settings = NatexSettings()
    result = process_traces(f, fneu, spks, FS, 0.9, settings)
    out = write_natex(result, tmp_path, settings)
    assert out == tmp_path / "natex"
    assert np.load(out / "natex_dff.npy").shape == (3, 900)
    assert np.load(out / "natex_dff.npy").dtype == np.float32
    assert np.load(out / "natex_significant.npy").dtype == bool
    with (out / "natex_cells.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [int(r["roi"]) for r in rows] == [0, 1, 2]
    assert float(rows[1]["snr"]) == pytest.approx(result.snr[1], abs=1e-4)
    record = json.loads((out / "natex_settings.json").read_text())
    assert record["neuropil_coefficient"] == 0.9 and record["settings"]["transient_s"] == 0.75
    assert "undefined_fraction" in rows[0]
    assert set(record["inputs_sha256"]) == {
        "F.npy",
        "Fneu.npy",
        "spks.npy",
        "iscell.npy",
        "ops.npy",
    }
    with pytest.raises(FileExistsError):
        write_natex(result, tmp_path, settings)
    (out / "natex_settings.json").unlink()  # a run that stopped before its record
    with pytest.raises(FileExistsError):
        write_natex(result, tmp_path, settings)
    write_natex(result, tmp_path, settings, overwrite=True)


def test_settings_come_from_the_natex_section(tmp_path):
    config = tmp_path / "pipeline.toml"
    config.write_text("[natex]\ntransient_s = 1.0\n")
    assert load_natex_settings(config).transient_s == 1.0
    config.write_text('[distortion]\nmode = "off"\n')
    assert load_natex_settings(config) == NatexSettings()


# ---------------------------------------------------------------- a plane folder


def make_plane(root, n_cells=4, n_frames=900, iscell=None):
    """A Suite2p plane folder with Cellpose-style ROIs: 4x4-pixel squares on a 30 x 40 image."""
    run_dir = root / "cohort-01" / "sub-03" / "ses-224_x" / "funcimg" / "run-01"
    plane = (
        run_dir / "with_distortion_correction" / "v" / "aligned_structural" / "suite2p" / "plane0"
    )
    plane.mkdir(parents=True)
    f, fneu, spks = plane_arrays(n_cells, n_frames)
    stat = np.array(
        [{"ypix": np.repeat(np.arange(4), 4) + 2, "xpix": np.tile(np.arange(4), 4) + 8 * i + 2}
         for i in range(n_cells)],
        dtype=object,
    )  # fmt: skip
    flags = np.ones(n_cells) if iscell is None else np.asarray(iscell, float)
    arrays = {"F": f, "Fneu": fneu, "spks": spks, "iscell": np.column_stack([flags, flags])}
    for name, array in arrays.items():
        np.save(plane / f"{name}.npy", array)
    np.save(plane / "stat.npy", stat, allow_pickle=True)
    ops = {"fs": FS, "Ly": 30, "Lx": 40, "meanImg": RNG.random((30, 40)), "nframes": n_frames}
    np.save(plane / "ops.npy", ops, allow_pickle=True)
    return plane, f, fneu


def test_run_natex_writes_results_and_a_report(tmp_path):
    from mini2p_preprocessing.natex import run_natex

    plane, f, fneu = make_plane(tmp_path)
    out = run_natex(plane, NatexSettings(), 0.8)
    assert out == plane / "natex"
    assert (out / "natex_qc.pdf").stat().st_size > 10_000
    dff = np.load(out / "natex_dff.npy")
    np.testing.assert_allclose(dff[1], natex_dff(f[1] - 0.8 * fneu[1], FS, NatexSettings()).dff)
    with pytest.raises(FileExistsError):
        run_natex(plane, NatexSettings(), 0.8)


def test_natex_settings_can_set_their_own_neuropil_coefficient_and_cells(tmp_path):
    from mini2p_preprocessing.natex import run_natex

    plane, _, _ = make_plane(tmp_path, iscell=[1, 0, 1, 0])
    settings = NatexSettings(neuropil_coefficient=0.5, only_iscell=True)
    out = run_natex(plane, settings, 0.8)
    record = json.loads((out / "natex_settings.json").read_text())
    assert record["neuropil_coefficient"] == 0.5
    assert np.load(out / "natex_rois.npy").tolist() == [0, 2]


def test_example_rows_span_the_snr_range():
    from mini2p_preprocessing.natex import example_rows

    f, fneu, spks = plane_arrays(n_cells=6)
    f[1] = 0
    result = process_traces(f, fneu, spks, FS, 0.7, NatexSettings())
    rows = example_rows(result, count=3)
    assert 1 not in rows and len(rows) == 3
    snrs = result.snr[rows]
    assert snrs[0] == result.snr[~result.bad].max() and snrs[-1] == result.snr[~result.bad].min()


def test_natex_command(tmp_path):
    from mini2p_preprocessing.cli import natex_main

    plane, _, _ = make_plane(tmp_path)
    config = tmp_path / "pipeline.toml"
    config.write_text('[distortion]\nmode = "off"\n\n[fluorescence]\nneuropil_coefficient = 0.9\n')
    natex_main(["--config", str(config), "--plane-dir", str(plane)])
    record = json.loads((plane / "natex" / "natex_settings.json").read_text())
    assert record["neuropil_coefficient"] == 0.9  # from [fluorescence]
    with pytest.raises(SystemExit):  # results exist: a usage error, not a traceback
        natex_main(["--config", str(config), "--plane-dir", str(plane)])
    natex_main(["--config", str(config), "--plane-dir", str(plane), "--overwrite"])


def test_natex_runs_on_a_plane_outside_a_variant(tmp_path):
    # run-XX/suite2p/plane0, the registration Cellpose masks are drawn on, is a valid target.
    from mini2p_preprocessing.natex import report_title, run_natex

    plane, _, _ = make_plane(tmp_path)
    run_dir = plane.parents[4]  # run-01
    plain = run_dir / "suite2p" / "plane0"
    plain.parent.mkdir()
    plane.rename(plain)
    assert report_title(plain) == "ses-224_x funcimg run-01 suite2p"
    assert (run_natex(plain, NatexSettings(), 0.8) / "natex_qc.pdf").is_file()
    variant_plane = run_dir / "with_distortion_correction" / "v" / "aligned_structural"
    assert report_title(variant_plane / "suite2p" / "plane0") == (
        "ses-224_x funcimg run-01 with_distortion_correction v aligned_structural"
    )
