"""Odd/even scan-line phase estimation and correction (pure functions, no I/O).

Shifts use Suite2p's ``bidiphase.shift`` so the sign convention matches Suite2p:
a positive phase moves odd rows towards higher column indices.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from suite2p.registration import bidiphase as s2p_bidiphase

from .config import ChannelSettings, PhaseSettings

SUITE2P_SEARCH_PX = 10  # Suite2p's compute() only searches about ±10 px


@dataclass(frozen=True)
class PhaseEstimate:
    phase_px: int
    peak_score: float
    score_margin: float  # best minus second-best score; NaN for the 'suite2p' estimator
    at_boundary: bool  # the true phase may lie outside the searched range
    candidates: np.ndarray | None = None
    scores: np.ndarray | None = None


@dataclass(frozen=True)
class AlignmentChoice:
    """Phase estimate from the channel chosen to drive correction for one chunk."""

    estimate: PhaseEstimate
    selected_source: str
    selected_channel_one_based: int
    estimate_by_source: dict[str, PhaseEstimate]

    @property
    def phase_px(self) -> int:
        return self.estimate.phase_px

    @property
    def selected_channel_zero_based(self) -> int:
        return self.selected_channel_one_based - 1

    @property
    def score_by_source(self) -> dict[str, float]:
        return {role: float(e.peak_score) for role, e in self.estimate_by_source.items()}


def apply_bidiphase(chunk: np.ndarray, phase_px: int) -> np.ndarray:
    """Float32 copy of ``chunk`` (frames, Ly, Lx) with odd rows shifted by ``phase_px``."""
    # Suite2p shifts in place; the copy keeps the caller's raw data intact.
    corrected = chunk.astype(np.float32, copy=True)
    return s2p_bidiphase.shift(corrected, int(phase_px))


def adjacent_line_score(shifted_mean: np.ndarray, crop_px: int) -> float:
    """Normalized correlation between rows 2k and 2k+1 of a 2-D image, in [-1, 1].

    ``crop_px`` columns are dropped from both edges so columns left stale by the
    shift do not count. Returns -inf for an image with no row texture, which the
    estimator ranks below every real candidate.
    """
    even = shifted_mean[::2]
    odd = shifted_mean[1::2]
    n_pairs = min(len(even), len(odd))
    even, odd = even[:n_pairs], odd[:n_pairs]
    if crop_px:
        if 2 * crop_px >= even.shape[1] - 2:
            raise ValueError(f"crop_px={crop_px} is too large for image width {even.shape[1]}")
        even, odd = even[:, crop_px:-crop_px], odd[:, crop_px:-crop_px]
    # Remove each row's baseline so line brightness/stripe offsets do not drive the score.
    even = even - np.mean(even, axis=1, keepdims=True)
    odd = odd - np.mean(odd, axis=1, keepdims=True)
    numerator = float(np.sum(even * odd))
    denominator = float(np.sqrt(np.sum(even**2) * np.sum(odd**2)))
    return numerator / denominator if denominator > 0 else -np.inf


def _score_shift(mean_frame: np.ndarray, phase_px: int, crop_px: int) -> float:
    return adjacent_line_score(apply_bidiphase(mean_frame[None], phase_px)[0], crop_px)


def _estimate_suite2p(chunk: np.ndarray, phase: PhaseSettings) -> PhaseEstimate:
    phase_px = int(s2p_bidiphase.compute(chunk))
    mean_frame = np.mean(chunk.astype(np.float32), axis=0)
    crop_px = SUITE2P_SEARCH_PX if phase.score_crop_px is None else phase.score_crop_px
    return PhaseEstimate(
        phase_px=phase_px,
        peak_score=float(_score_shift(mean_frame, phase_px, crop_px)),
        score_margin=np.nan,
        at_boundary=abs(phase_px) >= SUITE2P_SEARCH_PX,
    )


def _estimate_wide(chunk: np.ndarray, phase: PhaseSettings) -> PhaseEstimate:
    max_shift = phase.max_shift_px
    crop_px = max_shift if phase.score_crop_px is None else phase.score_crop_px
    mean_frame = np.mean(chunk.astype(np.float32), axis=0)
    candidates = np.arange(-max_shift, max_shift + 1, dtype=int)
    scores = np.array([_score_shift(mean_frame, int(c), crop_px) for c in candidates])
    best_index = int(np.nanargmax(scores))
    phase_px = int(candidates[best_index])
    finite_scores = np.sort(scores[np.isfinite(scores)])
    margin = float(finite_scores[-1] - finite_scores[-2]) if len(finite_scores) > 1 else np.nan
    return PhaseEstimate(
        phase_px=phase_px,
        peak_score=float(scores[best_index]),
        score_margin=margin,
        at_boundary=max_shift - abs(phase_px) <= phase.boundary_warning_px,
        candidates=candidates,
        scores=scores,
    )


def estimate_phase(chunk: np.ndarray, phase: PhaseSettings) -> PhaseEstimate:
    """Estimate the odd-row phase of one channel's chunk (frames, Ly, Lx)."""
    chunk = np.asarray(chunk)
    if chunk.ndim != 3 or chunk.shape[1] < 2:
        raise ValueError(f"Expected frames x height x width with >= 2 rows, got {chunk.shape}")
    if phase.estimator == "suite2p":
        return _estimate_suite2p(chunk, phase)
    return _estimate_wide(chunk, phase)


def _choose_source(estimates: dict[str, PhaseEstimate], phase: PhaseSettings) -> str:
    if phase.alignment_source != "best_score":
        return phase.alignment_source
    red = estimates["structural"].peak_score
    green = estimates["functional"].peak_score
    # Structural is the default; it usually has steadier texture than calcium signal.
    return "functional" if green > red + phase.score_switch_margin else "structural"


def select_alignment(
    chunks: Sequence[np.ndarray], phase: PhaseSettings, channels: ChannelSettings
) -> AlignmentChoice:
    """Pick the channel whose phase estimate drives correction for this chunk.

    ``chunks`` holds one (frames, Ly, Lx) array per channel, in 0-based order.
    """
    if len(chunks) != channels.n_channels:
        raise ValueError(f"Expected {channels.n_channels} channel chunks, got {len(chunks)}")
    roles = (
        ["structural", "functional"]
        if phase.alignment_source == "best_score"
        else [phase.alignment_source]
    )
    by_role = channels.channel_by_role
    estimates = {role: estimate_phase(chunks[by_role[role] - 1], phase) for role in roles}
    source = _choose_source(estimates, phase)
    return AlignmentChoice(
        estimate=estimates[source],
        selected_source=source,
        selected_channel_one_based=by_role[source],
        estimate_by_source=estimates,
    )
