"""Turning points into heatmaps for training, and heatmaps back into points.

A heatmap is a small image with a bright Gaussian blob where the point is.
Predicting blobs is easier for a network to learn than predicting raw x, y.
"""

import numpy as np
import torch


def render_heatmaps(
    points: np.ndarray, visibility: np.ndarray, size: tuple[int, int], stride: int, sigma: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """Builds target heatmaps.

    Args:
        points: (C, 2) point positions in input-image pixels.
        visibility: (C,) COCO visibility flags; 0 means no label.
        size: (height, width) of the heatmap.
        stride: input pixels per heatmap pixel.
        sigma: blob radius in heatmap pixels.

    Returns:
        heatmaps (C, H, W) float32 and weights (C,) float32, where weight 0
        tells the loss to ignore a channel.
    """
    height, width = size
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    channels = points.shape[0]
    heatmaps = np.zeros((channels, height, width), dtype=np.float32)
    weights = np.zeros(channels, dtype=np.float32)
    for c in range(channels):
        if visibility[c] <= 0:
            continue
        x, y = points[c] / stride
        if not (-3 * sigma <= x < width + 3 * sigma and -3 * sigma <= y < height + 3 * sigma):
            continue
        heatmaps[c] = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma**2))
        weights[c] = 1.0
    return heatmaps, weights


def decode_heatmaps(heatmaps: torch.Tensor, stride: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Finds the peak of each heatmap.

    Args:
        heatmaps: (N, C, H, W) network output.
        stride: input pixels per heatmap pixel.

    Returns:
        points (N, C, 2) in input-image pixels and scores (N, C), the peak
        heights, which work as a confidence.
    """
    n, c, h, w = heatmaps.shape
    flat = heatmaps.reshape(n, c, -1)
    scores, index = flat.max(dim=-1)
    x = (index % w).float()
    y = torch.div(index, w, rounding_mode="floor").float()

    # Nudge a quarter pixel toward the higher neighbour for sub-pixel accuracy.
    xi = x.long().clamp(1, w - 2)
    yi = y.long().clamp(1, h - 2)
    rows = heatmaps.reshape(n * c, h, w)
    k = torch.arange(n * c, device=heatmaps.device)
    xi_f, yi_f = xi.reshape(-1), yi.reshape(-1)
    dx = rows[k, yi_f, xi_f + 1] - rows[k, yi_f, xi_f - 1]
    dy = rows[k, yi_f + 1, xi_f] - rows[k, yi_f - 1, xi_f]
    x = x + 0.25 * torch.sign(dx).reshape(n, c)
    y = y + 0.25 * torch.sign(dy).reshape(n, c)

    points = torch.stack([x, y], dim=-1) * stride + (stride - 1) / 2
    return points, scores
