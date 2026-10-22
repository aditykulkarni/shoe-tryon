"""How good the predictions are.

PCK (percentage of correct keypoints) counts a point as correct when it lands
within a fraction of the foot's length from the label. PCK@0.05 on a 260 mm
foot means within 13 mm.
"""

import torch

from .keypoints import NUM_KEYPOINTS, SIDES, channel


def foot_lengths(points: torch.Tensor, visibility: torch.Tensor) -> torch.Tensor:
    """Heel-to-toe length per foot, (N, 2); NaN where either end is unlabelled."""
    lengths = []
    for side in SIDES:
        toe, heel = channel(side, "toe_tip"), channel(side, "heel_back")
        d = torch.linalg.norm(points[:, toe] - points[:, heel], dim=-1)
        labelled = (visibility[:, toe] > 0) & (visibility[:, heel] > 0)
        lengths.append(torch.where(labelled, d, torch.full_like(d, float("nan"))))
    return torch.stack(lengths, dim=1)


def pck_counts(
    predicted: torch.Tensor, target: torch.Tensor, visibility: torch.Tensor, threshold: float = 0.05
) -> tuple[int, int]:
    """Correct and total labelled points in a batch.

    Args:
        predicted, target: (N, 16, 2) points in pixels.
        visibility: (N, 16) flags; only labelled points on feet with a known
            length are counted.
    """
    lengths = foot_lengths(target, visibility)  # (N, 2)
    per_channel = lengths.repeat_interleave(NUM_KEYPOINTS, dim=1)  # (N, 16)
    error = torch.linalg.norm(predicted - target, dim=-1)
    counted = (visibility > 0) & ~torch.isnan(per_channel)
    correct = counted & (error <= threshold * per_channel)
    return int(correct.sum()), int(counted.sum())
