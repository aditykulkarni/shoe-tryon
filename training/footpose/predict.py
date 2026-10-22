"""Runs a checkpoint on one photo and saves a picture of the predicted points.

Run with:  uv run footpose-predict runs/latest/best.pt my_feet.jpg
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from .dataset import letterbox_matrix, normalise
from .export import load_model
from .heatmaps import decode_heatmaps
from .keypoints import KEYPOINT_NAMES, NUM_KEYPOINTS
from .model import STRIDE

COLOURS = {"left": (80, 200, 120), "right": (240, 160, 60)}


def predict(model: torch.nn.Module, image_bgr: np.ndarray, input_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Points (16, 2) in original image pixels and scores (16,)."""
    h, w = image_bgr.shape[:2]
    m = letterbox_matrix(w, h, input_size)
    crop = cv2.warpAffine(
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), m, (input_size, input_size), borderValue=(114, 114, 114)
    )
    with torch.no_grad():
        heatmaps = model(normalise(crop)[None])
    points, scores = decode_heatmaps(heatmaps, STRIDE)
    inverse = cv2.invertAffineTransform(m)
    pts = points[0].numpy() @ inverse[:, :2].T + inverse[:, 2]
    return pts, scores[0].numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw predicted foot points on a photo.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("--out", type=Path, help="defaults to <image>_points.jpg")
    parser.add_argument("--min-score", type=float, default=0.3)
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"Could not read {args.image}")
    points, scores = predict(load_model(args.checkpoint), image)
    for i, ((x, y), score) in enumerate(zip(points, scores, strict=True)):
        if score < args.min_score:
            continue
        side = "left" if i < NUM_KEYPOINTS else "right"
        cv2.circle(image, (int(x), int(y)), 6, COLOURS[side], -1, cv2.LINE_AA)
        cv2.putText(
            image,
            KEYPOINT_NAMES[i % NUM_KEYPOINTS],
            (int(x) + 8, int(y) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            COLOURS[side],
            1,
            cv2.LINE_AA,
        )
    out = args.out or args.image.with_name(f"{args.image.stem}_points.jpg")
    cv2.imwrite(str(out), image)
    print(f"Saved {out}")
