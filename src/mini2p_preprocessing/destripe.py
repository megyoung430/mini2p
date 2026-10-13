"""Removal of line-synchronous stripe waveforms (pure functions, no I/O)."""

from __future__ import annotations

import numpy as np


def detrend_chunk(chunk: np.ndarray, preserve_frame_mean: bool = True) -> np.ndarray:
    """Subtract each frame's mean even-row and mean odd-row column profile.

    Forward (even) and reverse (odd) scan lines pick up different electrical
    waveforms, so each set gets its own profile, estimated per frame. With
    ``preserve_frame_mean`` only each profile's shape is removed, keeping the
    frame's mean fluorescence; without it every column of every frame is driven
    to zero mean, which discards the fluorescence baseline.
    """
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
