"""Verbatim copies of the original notebook's functions, used only as a reference.

Copied from mini2p_dynamic_bidiphase_destripe_suite2p.ipynb (cells 3, 9, 20, 24) so
tests can check the refactored package produces identical outputs. Notebook
globals are set with ``configure``.
"""

# ruff: noqa
import gc
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import map_coordinates
from suite2p.registration import bidiphase as s2p_bidiphase


def configure(**values):
    globals().update(values)
    globals()["channel_by_role"] = {
        "functional": values["FUNCTIONAL_CHANNEL"],
        "structural": values["STRUCTURAL_CHANNEL"],
    }
    globals()["PHASE_ALIGNMENT_CHANNEL"] = channel_by_role.get(values["PHASE_ALIGNMENT_SOURCE"])
    globals()["REGISTRATION_CHANNEL"] = channel_by_role[values["MOTION_ALIGNMENT_SOURCE"]]
    globals()["PHASE_REFERENCE_CHANNEL"] = (
        PHASE_ALIGNMENT_CHANNEL or values["STRUCTURAL_CHANNEL"]
    ) - 1


def apply_bidiphase(chunk, phase_px):
    # Suite2p modifies its input in place, so protect the raw chunk.
    corrected = chunk.astype(np.float32, copy=True)
    return s2p_bidiphase.shift(corrected, int(phase_px))


def _adjacent_line_score(shifted_mean, crop_px):
    """Normalized agreement between neighboring even/odd rows after a candidate shift."""
    even = shifted_mean[::2]
    odd = shifted_mean[1::2]
    n_pairs = min(len(even), len(odd))
    even, odd = even[:n_pairs], odd[:n_pairs]
    if crop_px:
        if 2 * crop_px >= even.shape[1] - 2:
            raise ValueError("PHASE_SCORE_CROP_PX is too large for the image width")
        even, odd = even[:, crop_px:-crop_px], odd[:, crop_px:-crop_px]
    # Remove each row's baseline so line brightness/stripe offsets do not drive the score.
    even = even - np.mean(even, axis=1, keepdims=True)
    odd = odd - np.mean(odd, axis=1, keepdims=True)
    numerator = float(np.sum(even * odd))
    denominator = float(np.sqrt(np.sum(even**2) * np.sum(odd**2)))
    return numerator / denominator if denominator > 0 else -np.inf


def estimate_bidiphase_diagnostics(chunk):
    """Return the phase estimate plus candidate scores and boundary diagnostics."""
    chunk = np.asarray(chunk)
    if chunk.ndim != 3 or chunk.shape[1] < 2:
        raise ValueError("Expected frames x height x width with at least two rows")
    if PHASE_ESTIMATOR == "suite2p":
        phase = int(s2p_bidiphase.compute(chunk))
        mean_frame = np.mean(chunk.astype(np.float32), axis=0)
        crop_px = 10 if PHASE_SCORE_CROP_PX is None else int(PHASE_SCORE_CROP_PX)
        peak_score = _adjacent_line_score(apply_bidiphase(mean_frame[None], phase)[0], crop_px)
        return {
            "phase_px": phase,
            "peak_score": float(peak_score),
            "score_margin": np.nan,
            "at_boundary": abs(phase) >= 10,
            "candidates": None,
            "scores": None,
        }
    if PHASE_ESTIMATOR != "wide":
        raise ValueError("PHASE_ESTIMATOR must be 'wide' or 'suite2p'")
    max_shift = int(PHASE_MAX_SHIFT_PX)
    if max_shift < 1:
        raise ValueError("PHASE_MAX_SHIFT_PX must be at least 1")
    crop_px = max_shift if PHASE_SCORE_CROP_PX is None else int(PHASE_SCORE_CROP_PX)
    mean_frame = np.mean(chunk.astype(np.float32), axis=0)
    candidates = np.arange(-max_shift, max_shift + 1, dtype=int)
    scores = np.array(
        [
            _adjacent_line_score(apply_bidiphase(mean_frame[None], phase)[0], crop_px)
            for phase in candidates
        ]
    )
    best_index = int(np.nanargmax(scores))
    phase = int(candidates[best_index])
    finite_scores = np.sort(scores[np.isfinite(scores)])
    margin = float(finite_scores[-1] - finite_scores[-2]) if len(finite_scores) > 1 else np.nan
    boundary_distance = max_shift - abs(phase)
    return {
        "phase_px": phase,
        "peak_score": float(scores[best_index]),
        "score_margin": margin,
        "at_boundary": boundary_distance <= int(PHASE_BOUNDARY_WARNING_PX),
        "candidates": candidates,
        "scores": scores,
    }


def estimate_bidiphase(chunk):
    return int(estimate_bidiphase_diagnostics(chunk)["phase_px"])


def detrend_chunk(chunk, preserve_frame_mean=True):
    """Remove separate even/odd line waveforms (existing method)."""
    chunk = chunk.astype(np.float32, copy=False)
    avg_even = np.mean(chunk[:, ::2, :], axis=1, keepdims=True)
    avg_odd = np.mean(chunk[:, 1::2, :], axis=1, keepdims=True)
    if preserve_frame_mean:
        avg_even -= np.mean(avg_even, axis=2, keepdims=True)
        avg_odd -= np.mean(avg_odd, axis=2, keepdims=True)
    output = np.empty_like(chunk, dtype=np.float32)
    output[:, ::2, :] = chunk[:, ::2, :] - avg_even
    output[:, 1::2, :] = chunk[:, 1::2, :] - avg_odd
    return output


def read_channel_chunk(tif, start, end, channel, n_channels):
    pages = channel + n_channels * np.arange(start, end)
    return np.stack([tif.pages[int(page)].asarray() for page in pages], axis=0)


def select_bidiphase_alignment(chunks):
    """Choose the phase-estimation channel for one chunk and return diagnostics."""
    if len(chunks) != N_CHANNELS:
        raise ValueError(f"Expected {N_CHANNELS} channel chunks, received {len(chunks)}")
    sources_to_score = (
        ["structural", "functional"]
        if PHASE_ALIGNMENT_SOURCE == "best_score"
        else [PHASE_ALIGNMENT_SOURCE]
    )
    diagnostics = {
        role: estimate_bidiphase_diagnostics(chunks[channel_by_role[role] - 1])
        for role in sources_to_score
    }
    if PHASE_ALIGNMENT_SOURCE == "best_score":
        red_score = diagnostics["structural"]["peak_score"]
        green_score = diagnostics["functional"]["peak_score"]
        selected_source = (
            "functional"
            if green_score > red_score + float(PHASE_SCORE_SWITCH_MARGIN)
            else "structural"
        )
    else:
        selected_source = PHASE_ALIGNMENT_SOURCE
    selected = dict(diagnostics[selected_source])
    selected.update(
        {
            "selected_source": selected_source,
            "selected_channel_one_based": channel_by_role[selected_source],
            "selected_channel_zero_based": channel_by_role[selected_source] - 1,
            "score_by_source": {
                role: float(diagnostic["peak_score"]) for role, diagnostic in diagnostics.items()
            },
            "diagnostic_by_source": diagnostics,
        }
    )
    return selected


OVERWRITE = True


def estimate_whole_recording_residual_phase(tif, n_timepoints):
    """Estimate pass-2 residual after adaptive pass 1, before destriping."""
    if SECOND_PASS_GLOBAL_MAX_FRAMES is None:
        sample_indices = np.arange(n_timepoints)
    else:
        sample_count = min(int(SECOND_PASS_GLOBAL_MAX_FRAMES), n_timepoints)
        sample_indices = np.unique(np.linspace(0, n_timepoints - 1, sample_count, dtype=int))
    sampled_after_first = [[] for _ in range(N_CHANNELS)]
    for chunk_start in range(0, n_timepoints, PHASE_CHUNK_FRAMES):
        chunk_end = min(chunk_start + PHASE_CHUNK_FRAMES, n_timepoints)
        selected_indices = sample_indices[
            (sample_indices >= chunk_start) & (sample_indices < chunk_end)
        ]
        if not len(selected_indices):
            continue
        chunks = [
            read_channel_chunk(tif, chunk_start, chunk_end, channel, N_CHANNELS)
            for channel in range(N_CHANNELS)
        ]
        first_selection = select_bidiphase_alignment(chunks)
        first_phase = int(MANUAL_FIRST_PHASE_BY_START.get(chunk_start, first_selection["phase_px"]))
        after_first = [apply_bidiphase(chunk, first_phase) for chunk in chunks]
        relative = selected_indices - chunk_start
        for channel in range(N_CHANNELS):
            sampled_after_first[channel].append(after_first[channel][relative])
        del chunks, after_first
    samples = [np.concatenate(parts, axis=0) for parts in sampled_after_first]
    selection = select_bidiphase_alignment(samples)
    residual = int(selection["phase_px"])
    print(
        f"Whole-recording second phase: {residual:+d} px from "
        f"{len(sample_indices)} frames; source={selection['selected_source']}; "
        f"scores={selection['score_by_source']}"
    )
    del samples, sampled_after_first
    gc.collect()
    return residual, int(len(sample_indices)), selection


def preprocess_tiff(input_tiff, output_tiff):
    if output_tiff.exists() and not OVERWRITE:
        raise FileExistsError(f"{output_tiff} exists; set OVERWRITE=True to replace it")
    phase_records = []
    with tifffile.TiffFile(input_tiff) as tif:
        n_pages = len(tif.pages)
        if n_pages % N_CHANNELS:
            raise ValueError(f"{n_pages} pages is not divisible by {N_CHANNELS} channels")
        n_timepoints = n_pages // N_CHANNELS
        if MAX_TIMEPOINTS is not None:
            n_timepoints = min(n_timepoints, int(MAX_TIMEPOINTS))
        if SECOND_PASS_MODE not in {"chunked", "whole"}:
            raise ValueError("SECOND_PASS_MODE must be 'chunked' or 'whole'")
        global_second_phase = None
        global_sample_count = None
        global_second_selection = None
        if RUN_SECOND_BIDIPHASE_PASS and SECOND_PASS_MODE == "whole":
            global_second_phase, global_sample_count, global_second_selection = (
                estimate_whole_recording_residual_phase(tif, n_timepoints)
            )

        with tifffile.TiffWriter(output_tiff, bigtiff=True) as writer:
            for start in range(0, n_timepoints, PHASE_CHUNK_FRAMES):
                end = min(start + PHASE_CHUNK_FRAMES, n_timepoints)
                chunks = [
                    read_channel_chunk(tif, start, end, ch, N_CHANNELS) for ch in range(N_CHANNELS)
                ]
                first_diagnostic = select_bidiphase_alignment(chunks)
                first_phase = int(
                    MANUAL_FIRST_PHASE_BY_START.get(start, first_diagnostic["phase_px"])
                )
                after_first = [apply_bidiphase(chunk, first_phase) for chunk in chunks]
                second_phase = 0
                second_diagnostic = None
                after_second = after_first
                if RUN_SECOND_BIDIPHASE_PASS:
                    second_diagnostic = (
                        global_second_selection
                        if SECOND_PASS_MODE == "whole"
                        else select_bidiphase_alignment(after_first)
                    )
                    second_phase = int(
                        MANUAL_SECOND_PHASE_BY_START.get(
                            start,
                            global_second_phase
                            if SECOND_PASS_MODE == "whole"
                            else second_diagnostic["phase_px"],
                        )
                    )
                    after_second = [apply_bidiphase(chunk, second_phase) for chunk in after_first]
                corrected = (
                    [detrend_chunk(chunk, PRESERVE_FRAME_MEAN) for chunk in after_second]
                    if RUN_DESTRIPING
                    else after_second
                )

                for frame_index in range(end - start):
                    for channel in range(N_CHANNELS):
                        writer.write(
                            corrected[channel][frame_index].astype(OUTPUT_DTYPE), contiguous=False
                        )
                phase_records.append(
                    {
                        "start": start,
                        "end": end,
                        "first_phase_px": int(first_phase),
                        "second_phase_px": int(second_phase),
                        "first_peak_score": float(first_diagnostic["peak_score"]),
                        "first_score_margin": float(first_diagnostic["score_margin"]),
                        "first_at_search_boundary": bool(first_diagnostic["at_boundary"]),
                        "first_selected_source": first_diagnostic["selected_source"],
                        "first_score_by_source": first_diagnostic["score_by_source"],
                        "second_selected_source": (
                            second_diagnostic["selected_source"]
                            if second_diagnostic is not None
                            else None
                        ),
                        "second_score_by_source": (
                            second_diagnostic["score_by_source"]
                            if second_diagnostic is not None
                            else None
                        ),
                    }
                )
                print(
                    f"{end:>7}/{n_timepoints}; first={first_phase:+d} px; "
                    f"second={second_phase:+d} px; source={first_diagnostic['selected_source']}; "
                    f"score={first_diagnostic['peak_score']:.3f}; "
                    f"boundary={first_diagnostic['at_boundary']}; "
                    f"destriping={RUN_DESTRIPING}"
                )
                del chunks, after_first, after_second, corrected
                gc.collect()

    metadata = {
        "run_name": RUN_NAME,
        "input_tiff": str(input_tiff),
        "output_tiff": str(output_tiff),
        "n_channels": N_CHANNELS,
        "structural_channel_one_based": STRUCTURAL_CHANNEL,
        "functional_channel_one_based": FUNCTIONAL_CHANNEL,
        "phase_alignment_source": PHASE_ALIGNMENT_SOURCE,
        "phase_alignment_channel_one_based": PHASE_ALIGNMENT_CHANNEL,
        "phase_score_switch_margin": PHASE_SCORE_SWITCH_MARGIN,
        "motion_alignment_source": MOTION_ALIGNMENT_SOURCE,
        "registration_channel_one_based": REGISTRATION_CHANNEL,
        "phase_reference_channel_zero_based": PHASE_REFERENCE_CHANNEL,
        "phase_chunk_frames": PHASE_CHUNK_FRAMES,
        "phase_method": PHASE_ESTIMATOR,
        "phase_max_shift_px": PHASE_MAX_SHIFT_PX if PHASE_ESTIMATOR == "wide" else 10,
        "phase_score_crop_px": PHASE_SCORE_CROP_PX,
        "phase_boundary_warning_px": PHASE_BOUNDARY_WARNING_PX,
        "manual_first_phase_by_start": MANUAL_FIRST_PHASE_BY_START,
        "manual_second_phase_by_start": MANUAL_SECOND_PHASE_BY_START,
        "second_pass_enabled": RUN_SECOND_BIDIPHASE_PASS,
        "second_pass_mode": SECOND_PASS_MODE if RUN_SECOND_BIDIPHASE_PASS else None,
        "global_second_phase_px": global_second_phase,
        "global_second_phase_sample_count": global_sample_count,
        "destriping_enabled": RUN_DESTRIPING,
        "operation_order": (
            ["first_biphase"]
            + (["second_biphase"] if RUN_SECOND_BIDIPHASE_PASS else [])
            + (["destriping"] if RUN_DESTRIPING else [])
        ),
        "preserve_frame_mean": PRESERVE_FRAME_MEAN,
        "chunks": phase_records,
    }
    sidecar = output_tiff.with_suffix(".phase.json")
    sidecar.write_text(json.dumps(metadata, indent=2))
    print("Saved:", output_tiff)
    print("Saved:", sidecar)
    return {
        "chunks": phase_records,
        "global_second_phase_px": global_second_phase,
        "global_second_phase_sample_count": global_sample_count,
    }


def apply_distortion_to_chunk(chunk, map_y, map_x, crop, order):
    """Apply one full-frame spatial map to all frames in one memory batch."""
    top, bottom, left, right = crop
    output = np.empty((len(chunk), bottom - top, right - left), dtype=np.float32)
    for frame_index, frame in enumerate(chunk):
        warped = map_coordinates(
            frame.astype(np.float32),
            [map_y, map_x],
            order=int(order),
            mode="constant",
            cval=0,
            prefilter=int(order) > 1,
        )
        output[frame_index] = warped[top:bottom, left:right]
    return output


def write_local_batched_distortion(input_tiff, output_tiff):
    if output_tiff.exists() and not OVERWRITE:
        raise FileExistsError(f"{output_tiff} exists; set OVERWRITE=True to replace it")
    transform, matched_z, schema, point_count = load_transform(
        DISTORTION_TRANSFORM_MAT, DISTORTION_Z_UM
    )
    with tifffile.TiffFile(input_tiff) as tif:
        n_pages = len(tif.pages)
        if n_pages % N_CHANNELS:
            raise ValueError(f"{n_pages} pages is not divisible by {N_CHANNELS} channels")
        n_timepoints = n_pages // N_CHANNELS
        if MAX_TIMEPOINTS is not None:
            n_timepoints = min(n_timepoints, int(MAX_TIMEPOINTS))
        image_shape = tif.pages[0].shape
        map_y, map_x = make_maps(transform, image_shape)
        if CROP_DISTORTION_BLACK_BORDERS:
            crop = crop_bounds(map_y, map_x, image_shape, int(DISTORTION_CROP_PADDING_PX))
        else:
            crop = (0, image_shape[0], 0, image_shape[1])
        batch_frames = int(DISTORTION_BATCH_FRAMES)
        if batch_frames < 1:
            raise ValueError("DISTORTION_BATCH_FRAMES must be at least 1")

        with tifffile.TiffWriter(output_tiff, bigtiff=True) as writer:
            for start in range(0, n_timepoints, batch_frames):
                end = min(start + batch_frames, n_timepoints)
                raw_chunks = [
                    read_channel_chunk(tif, start, end, channel, N_CHANNELS)
                    for channel in range(N_CHANNELS)
                ]
                corrected_chunks = [
                    apply_distortion_to_chunk(
                        chunk, map_y, map_x, crop, DISTORTION_INTERPOLATION_ORDER
                    )
                    for chunk in raw_chunks
                ]
                for frame_index in range(end - start):
                    for channel in range(N_CHANNELS):
                        writer.write(
                            corrected_chunks[channel][frame_index].astype(OUTPUT_DTYPE),
                            contiguous=False,
                        )
                print(f"Distortion correction: {end:>7}/{n_timepoints} timepoints")
                del raw_chunks, corrected_chunks
                gc.collect()

    sidecar = output_tiff.with_suffix(".distortion.json")
    sidecar.write_text(
        json.dumps(
            {
                "input_tiff": str(input_tiff),
                "output_tiff": str(output_tiff),
                "mode": "local_batched_before_registration",
                "batch_frames": int(DISTORTION_BATCH_FRAMES),
                "transform_reestimated_per_batch": False,
                "transform_mat": str(DISTORTION_TRANSFORM_MAT),
                "requested_z_um": float(DISTORTION_Z_UM),
                "matched_z_um": matched_z,
                "calibration_schema": schema,
                "control_points": int(point_count),
                "interpolation_order": int(DISTORTION_INTERPOLATION_ORDER),
                "crop_bounds_top_bottom_left_right": list(map(int, crop)),
                "input_shape_yx": list(map(int, image_shape)),
                "output_shape_yx": [crop[1] - crop[0], crop[3] - crop[2]],
                "n_timepoints": int(n_timepoints),
                "applied_before": "suite2p_registration",
            },
            indent=2,
        )
    )
    print("Saved:", output_tiff)
    print("Saved:", sidecar)


def reference_suite2p_settings_and_db(default_settings, SUITE2P_INPUT_TIFF, OUTPUT_DIR):
    settings = default_settings()
    settings["fs"] = FS_HZ
    settings["diameter"] = DIAMETER_PX
    settings["registration"].update(
        {
            # True selects Suite2p's non-functional binary as the alignment source.
            "align_by_chan2": REGISTRATION_CHANNEL != FUNCTIONAL_CHANNEL,
            # Already corrected separately for every short chunk above.
            "do_bidiphase": False,
            "bidiphase": 0,
            "nonrigid": DO_NONRIGID,
            "norm_frames": NORMALIZE_REGISTRATION_FRAMES,
            "nimg_init": NIMG_INIT,
            "th_badframes": BAD_FRAME_THRESHOLD,
            "two_step_registration": TWO_STEP_REGISTRATION,
            "batch_size": REGISTRATION_BATCH_SIZE,
        }
    )
    settings["run"].update(
        {
            "do_registration": 1,
            "do_detection": RUN_ROI_DETECTION,
            "do_deconvolution": RUN_ROI_DETECTION,
        }
    )
    settings["io"]["save_mat"] = False

    db = {
        "data_path": [str(OUTPUT_DIR)],
        "file_list": [SUITE2P_INPUT_TIFF.name],
        "look_one_level_down": False,
        "save_path0": str(OUTPUT_DIR),
        "save_folder": SUITE2P_SAVE_FOLDER,
        "nplanes": 1,
        "nchannels": N_CHANNELS,
        "functional_chan": FUNCTIONAL_CHANNEL,
        "keep_movie_raw": KEEP_RAW_BINARY,
    }
    return settings, db
