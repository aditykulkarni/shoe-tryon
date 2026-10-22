"""Fake training data, so the whole pipeline can be tried before real data exists.

Draws two foot-shaped blobs on a random background, as seen from above, and
writes images plus an annotation file in the format footpose.dataset reads.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .keypoints import KEYPOINT_NAMES, VISIBLE

# Point layout of a right foot in "foot units": heel at the origin, toe up the
# negative y axis (image up), length 1.0. The left foot mirrors x.
RIGHT_FOOT = {
    "toe_tip": (0.02, -1.0),
    "big_toe_joint": (-0.17, -0.72),
    "little_toe_joint": (0.2, -0.66),
    "heel_back": (0.0, 0.0),
    "inner_ankle": (-0.12, -0.12),
    "outer_ankle": (0.12, -0.1),
    "instep": (0.0, -0.42),
    "ankle_front": (0.0, -0.22),
}
OUTLINE = [
    (0.0, 0.0),
    (0.14, -0.15),
    (0.21, -0.66),
    (0.12, -0.93),
    (0.02, -1.0),
    (-0.1, -0.95),
    (-0.18, -0.72),
    (-0.13, -0.2),
]


def _place(points: list[tuple[float, float]], side: str, centre: np.ndarray, length: float, angle: float) -> np.ndarray:
    p = np.array(points, dtype=np.float32)
    if side == "left":
        p[:, 0] *= -1
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return (p * length) @ rot.T + centre


def make_image(rng: np.random.Generator, size: tuple[int, int] = (480, 640)) -> tuple[np.ndarray, dict]:
    h, w = size
    background = rng.integers(40, 220, size=3).tolist()
    image = np.full((h, w, 3), background, dtype=np.uint8)
    noise = rng.normal(0, 12, size=image.shape)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)

    length = rng.uniform(0.35, 0.55) * h
    base_angle = rng.uniform(-0.6, 0.6)
    gap = rng.uniform(0.18, 0.3) * length
    centre = np.array([w / 2 + rng.uniform(-0.15, 0.15) * w, h * 0.5 + length * 0.5], dtype=np.float32)
    skin = rng.integers(60, 255, size=3).tolist()

    annotations = {}
    for side, offset in (("left", -gap), ("right", gap)):
        angle = base_angle + rng.uniform(-0.25, 0.25)
        heel = centre + np.array([offset * np.cos(base_angle), offset * np.sin(base_angle)], dtype=np.float32)
        outline = _place(OUTLINE, side, heel, length, angle)
        cv2.fillPoly(image, [outline.astype(np.int32)], skin, lineType=cv2.LINE_AA)
        pts = _place([RIGHT_FOOT[n] for n in KEYPOINT_NAMES], side, heel, length, angle)
        annotations[side] = pts
    return image, annotations


def write_toy_dataset(out_dir: Path, count: int, seed: int = 0) -> Path:
    """Writes count images and annotations.json into out_dir; returns the JSON path."""
    rng = np.random.default_rng(seed)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    for i in range(count):
        image, feet = make_image(rng)
        name = f"toy_{i:05d}.jpg"
        cv2.imwrite(str(image_dir / name), image)
        images.append({"id": i, "file_name": name, "width": image.shape[1], "height": image.shape[0]})
        for side, pts in feet.items():
            kp = [v for x, y in pts.tolist() for v in (round(x, 2), round(y, 2), VISIBLE)]
            annotations.append({"image_id": i, "side": side, "keypoints": kp})
    path = out_dir / "annotations.json"
    path.write_text(json.dumps({"images": images, "annotations": annotations}))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a toy dataset of drawn feet.")
    parser.add_argument("--out", type=Path, default=Path("data/toy"))
    parser.add_argument("--train", type=int, default=800, help="training images")
    parser.add_argument("--val", type=int, default=200, help="validation images")
    args = parser.parse_args()
    write_toy_dataset(args.out / "train", args.train, seed=1)
    write_toy_dataset(args.out / "val", args.val, seed=2)
    print(f"Wrote {args.train} training and {args.val} validation images to {args.out}")
