"""Loading labelled images and preparing them for training.

Annotation files are JSON, close to the COCO keypoint format:

    {
      "images": [{"id": 1, "file_name": "frame_0001.jpg"}],
      "annotations": [
        {"image_id": 1, "side": "left", "keypoints": [x1, y1, v1, ..., x8, y8, v8]}
      ]
    }

Keypoints follow the order in footpose.keypoints.KEYPOINT_NAMES. The flag v
is 0 (not labelled), 1 (labelled but hidden) or 2 (visible). Each image has
at most one annotation per side.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .heatmaps import render_heatmaps
from .keypoints import NUM_CHANNELS, NUM_KEYPOINTS, SIDES, flip_permutation
from .model import STRIDE

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class Sample:
    path: Path
    points: np.ndarray  # (16, 2) pixels in the original image
    visibility: np.ndarray  # (16,)


def load_annotations(annotation_file: Path, image_dir: Path) -> list[Sample]:
    """Reads an annotation file into one Sample per image."""
    data = json.loads(Path(annotation_file).read_text())
    by_image: dict[int, Sample] = {}
    for image in data["images"]:
        by_image[image["id"]] = Sample(
            path=Path(image_dir) / image["file_name"],
            points=np.zeros((NUM_CHANNELS, 2), dtype=np.float32),
            visibility=np.zeros(NUM_CHANNELS, dtype=np.float32),
        )
    for ann in data["annotations"]:
        sample = by_image[ann["image_id"]]
        kp = np.asarray(ann["keypoints"], dtype=np.float32).reshape(NUM_KEYPOINTS, 3)
        start = SIDES.index(ann["side"]) * NUM_KEYPOINTS
        sample.points[start : start + NUM_KEYPOINTS] = kp[:, :2]
        sample.visibility[start : start + NUM_KEYPOINTS] = kp[:, 2]
    return list(by_image.values())


def letterbox_matrix(width: int, height: int, size: int) -> np.ndarray:
    """2x3 affine that fits an image inside a size x size square, centred."""
    scale = size / max(width, height)
    return np.array(
        [[scale, 0, (size - width * scale) / 2], [0, scale, (size - height * scale) / 2]],
        dtype=np.float32,
    )


def random_augment_matrix(size: int, rng: np.random.Generator) -> np.ndarray:
    """Random rotation, zoom and shift around the image centre, as a 2x3 affine."""
    angle = rng.uniform(-30, 30)
    scale = rng.uniform(0.75, 1.25)
    shift = rng.uniform(-0.1, 0.1, size=2) * size
    m = cv2.getRotationMatrix2D((size / 2, size / 2), angle, scale)
    m[:, 2] += shift
    return m.astype(np.float32)


def compose(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """Affine that applies inner first, then outer."""
    a = np.vstack([outer, [0, 0, 1]])
    b = np.vstack([inner, [0, 0, 1]])
    return (a @ b)[:2]


def transform_points(m: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ m[:, :2].T + m[:, 2]


def normalise(image_rgb: np.ndarray) -> torch.Tensor:
    """uint8 HWC RGB image to a float CHW tensor with ImageNet statistics."""
    x = (image_rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


class FootDataset(Dataset):
    """Images and heatmap targets, with augmentation when train=True."""

    def __init__(self, samples: list[Sample], input_size: int = 256, train: bool = False, seed: int = 0) -> None:
        self.samples = samples
        self.input_size = input_size
        self.train = train
        self.rng = np.random.default_rng(seed)
        self.flip = flip_permutation()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        image = cv2.imread(str(sample.path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image {sample.path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        size = self.input_size

        m = letterbox_matrix(width, height, size)
        points = sample.points.copy()
        visibility = sample.visibility.copy()
        if self.train:
            m = compose(random_augment_matrix(size, self.rng), m)
        image = cv2.warpAffine(image, m, (size, size), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
        points = transform_points(m, points)

        if self.train:
            if self.rng.random() < 0.5:
                image = image[:, ::-1]
                points[:, 0] = size - 1 - points[:, 0]
                points, visibility = points[self.flip], visibility[self.flip]
            image = self._colour_jitter(image)

        # Points pushed outside the crop by augmentation are no longer labelled.
        outside = (points < 0).any(axis=1) | (points >= size).any(axis=1)
        visibility = np.where(outside, 0, visibility)

        heatmaps, weights = render_heatmaps(points, visibility, (size // STRIDE, size // STRIDE), STRIDE)
        return {
            "image": normalise(np.ascontiguousarray(image)),
            "heatmaps": torch.from_numpy(heatmaps),
            "weights": torch.from_numpy(weights),
            "points": torch.from_numpy(points.astype(np.float32)),
            "visibility": torch.from_numpy(visibility.astype(np.float32)),
        }

    def _colour_jitter(self, image: np.ndarray) -> np.ndarray:
        contrast = self.rng.uniform(0.7, 1.3)
        brightness = self.rng.uniform(-30, 30)
        out = image.astype(np.float32) * contrast + brightness
        return np.clip(out, 0, 255).astype(np.uint8)
