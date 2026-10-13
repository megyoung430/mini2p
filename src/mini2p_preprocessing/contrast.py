"""Display contrast for mean images (pure)."""

from __future__ import annotations

import numpy as np


def scale_contrast(
    image: np.ndarray, black_percentile: float, white_percentile: float, gamma: float
) -> tuple[np.ndarray, float, float]:
    """Map [black, white] percentiles to [0, 1] with a gamma curve.

    Returns the display image plus the black and white values used. Non-finite
    pixels are shown black.
    """
    image = np.squeeze(np.asarray(image, dtype=np.float32))
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D mean image, got shape {image.shape}")
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        raise ValueError("Mean image contains no finite pixels")
    black, white = np.percentile(finite, [black_percentile, white_percentile])
    if white <= black:
        raise ValueError(f"Invalid contrast range: black={black}, white={white}")
    display = np.clip((image - black) / (white - black), 0, 1) ** gamma
    display[~np.isfinite(display)] = 0
    return display.astype(np.float32), float(black), float(white)


def red_green_overlay(red_display: np.ndarray, green_display: np.ndarray) -> np.ndarray:
    overlay = np.zeros((*green_display.shape, 3), dtype=np.float32)
    overlay[..., 0] = red_display
    overlay[..., 1] = green_display
    return overlay
