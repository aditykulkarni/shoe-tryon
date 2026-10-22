"""Training loop.

Run with:  uv run footpose-train --data data/toy --epochs 20
"""

import argparse
import csv
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import FootDataset, load_annotations
from .device import best_device
from .heatmaps import decode_heatmaps
from .metrics import pck_counts
from .model import STRIDE, FootPoseNet

log = logging.getLogger("footpose.train")

FOREGROUND_WEIGHT = 10.0


@dataclass
class TrainConfig:
    data: Path
    out: Path = Path("runs/latest")
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    input_size: int = 256
    workers: int = 4
    device: str = "auto"
    pretrained: bool = True
    seed: int = 0


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Mean squared error per heatmap, ignoring channels with weight 0.

    Pixels near a point count up to FOREGROUND_WEIGHT times more. Almost every
    heatmap pixel is zero, so without this the network can score well early on
    by predicting blank heatmaps everywhere, and gets stuck there.
    """
    pixel_weight = 1 + (FOREGROUND_WEIGHT - 1) * target
    per_channel = (pixel_weight * (pred - target) ** 2).mean(dim=(2, 3))
    return (per_channel * weights).sum() / weights.sum().clamp(min=1)


def make_loaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader]:
    def split(name: str, train: bool) -> FootDataset:
        folder = cfg.data / name
        samples = load_annotations(folder / "annotations.json", folder / "images")
        return FootDataset(samples, cfg.input_size, train=train, seed=cfg.seed)

    common = {"batch_size": cfg.batch_size, "num_workers": cfg.workers, "pin_memory": torch.cuda.is_available()}
    return (
        DataLoader(split("train", True), shuffle=True, drop_last=True, persistent_workers=cfg.workers > 0, **common),
        DataLoader(split("val", False), shuffle=False, **common),
    )


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    loss_sum, batches, correct, total = 0.0, 0, 0, 0
    for batch in loader:
        images = batch["image"].to(device)
        heatmaps = model(images)
        loss_sum += weighted_mse(heatmaps, batch["heatmaps"].to(device), batch["weights"].to(device)).item()
        batches += 1
        points, _ = decode_heatmaps(heatmaps.float(), STRIDE)
        c, t = pck_counts(points.cpu(), batch["points"], batch["visibility"])
        correct, total = correct + c, total + t
    return {"val_loss": loss_sum / max(batches, 1), "pck05": correct / max(total, 1)}


def train(cfg: TrainConfig) -> dict[str, float]:
    torch.manual_seed(cfg.seed)
    device = best_device(cfg.device)
    cfg.out.mkdir(parents=True, exist_ok=True)
    log.info("Training on %s, writing to %s", device, cfg.out)

    train_loader, val_loader = make_loaders(cfg)
    model = FootPoseNet(pretrained=cfg.pretrained).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=cfg.lr, total_steps=max(steps, 1))
    # Mixed precision roughly doubles speed on NVIDIA GPUs.
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best = {"pck05": -1.0}
    history_path = cfg.out / "history.csv"
    with history_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "pck05", "seconds"])
        writer.writeheader()
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            start, loss_sum = time.time(), 0.0
            for batch in train_loader:
                images = batch["image"].to(device, non_blocking=True)
                targets = batch["heatmaps"].to(device, non_blocking=True)
                weights = batch["weights"].to(device, non_blocking=True)
                with torch.autocast(device.type, enabled=use_amp):
                    loss = weighted_mse(model(images), targets, weights)
                optimiser.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()
                scheduler.step()
                loss_sum += loss.item()

            row = {"epoch": epoch, "train_loss": loss_sum / len(train_loader), **evaluate(model, val_loader, device)}
            row["seconds"] = round(time.time() - start, 1)
            writer.writerow(row)
            f.flush()
            log.info(
                "epoch %d/%d  train %.5f  val %.5f  PCK@0.05 %.1f%%  (%ss)",
                epoch,
                cfg.epochs,
                row["train_loss"],
                row["val_loss"],
                100 * row["pck05"],
                row["seconds"],
            )

            checkpoint = {
                "model": model.state_dict(),
                "config": {k: str(v) for k, v in asdict(cfg).items()},
                "epoch": epoch,
            }
            torch.save(checkpoint, cfg.out / "last.pt")
            if row["pck05"] > best["pck05"]:
                best = row
                torch.save(checkpoint, cfg.out / "best.pt")
    log.info("Best PCK@0.05 %.1f%% at epoch %d", 100 * best["pck05"], best["epoch"])
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the foot keypoint model.")
    parser.add_argument("--data", type=Path, required=True, help="folder with train/ and val/ subfolders")
    parser.add_argument("--out", type=Path, default=TrainConfig.out)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--workers", type=int, default=TrainConfig.workers)
    parser.add_argument("--device", default=TrainConfig.device, help="auto, cuda, mps or cpu")
    parser.add_argument("--no-pretrained", action="store_true", help="start from random weights")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    train(
        TrainConfig(
            data=args.data,
            out=args.out,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            workers=args.workers,
            device=args.device,
            pretrained=not args.no_pretrained,
        )
    )
