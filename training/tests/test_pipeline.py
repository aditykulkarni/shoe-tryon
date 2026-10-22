"""End-to-end checks on a small toy dataset: data loads, the model learns, export works."""

import numpy as np
import pytest
import torch

from footpose.dataset import FootDataset, load_annotations
from footpose.export import export
from footpose.keypoints import NUM_CHANNELS
from footpose.metrics import pck_counts
from footpose.toydata import write_toy_dataset
from footpose.train import TrainConfig, train


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    root = tmp_path_factory.mktemp("toy")
    write_toy_dataset(root / "train", 48, seed=1)
    write_toy_dataset(root / "val", 16, seed=2)
    return root


def test_dataset_shapes_and_augmented_points_stay_labelled(toy):
    samples = load_annotations(toy / "train" / "annotations.json", toy / "train" / "images")
    item = FootDataset(samples, input_size=128, train=True)[0]
    assert item["image"].shape == (3, 128, 128)
    assert item["heatmaps"].shape == (NUM_CHANNELS, 32, 32)
    assert item["visibility"].sum() > 0


def test_unaugmented_points_match_labels_after_letterbox(toy):
    samples = load_annotations(toy / "val" / "annotations.json", toy / "val" / "images")
    item = FootDataset(samples, input_size=320, train=False)[0]
    # The 640x480 toy image fits a 320 square at half scale, shifted down 40 px.
    expected = samples[0].points * 0.5 + np.array([0, 40])
    assert np.allclose(item["points"].numpy(), expected, atol=1e-3)


def test_pck_counts_perfect_prediction(toy):
    samples = load_annotations(toy / "val" / "annotations.json", toy / "val" / "images")
    pts = torch.from_numpy(np.stack([s.points for s in samples]))
    vis = torch.from_numpy(np.stack([s.visibility for s in samples]))
    correct, total = pck_counts(pts, pts, vis)
    assert correct == total == len(samples) * NUM_CHANNELS


def test_short_training_run_learns_and_exports(toy, tmp_path):
    cfg = TrainConfig(
        data=toy,
        out=tmp_path / "run",
        epochs=3,
        batch_size=8,
        lr=2e-3,
        input_size=128,
        workers=0,
        device="cpu",
        pretrained=False,
    )
    train(cfg)
    history = (cfg.out / "history.csv").read_text().splitlines()
    first, last = (float(line.split(",")[1]) for line in (history[1], history[-1]))
    assert last < first, "training loss should fall"
    assert (cfg.out / "best.pt").exists()
    diff = export(cfg.out / "best.pt", tmp_path / "model.onnx", input_size=128)
    assert diff < 1e-3
