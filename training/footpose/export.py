"""Exports a trained checkpoint to ONNX for the browser, and checks it.

Run with:  uv run footpose-export runs/latest/best.pt --out ../public/models/footpose.onnx
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from .model import FootPoseNet

log = logging.getLogger("footpose.export")


def load_model(checkpoint: Path) -> FootPoseNet:
    model = FootPoseNet(pretrained=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    return model.eval()


def export(checkpoint: Path, out: Path, input_size: int = 256) -> float:
    """Writes the ONNX file and returns the largest difference from PyTorch."""
    model = load_model(checkpoint)
    dummy = torch.randn(1, 3, input_size, input_size)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, (dummy,), str(out), input_names=["image"], output_names=["heatmaps"], opset_version=18, dynamo=True
    )
    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"image": dummy.numpy()})[0]
    with torch.no_grad():
        torch_out = model(dummy).numpy()
    diff = float(np.abs(onnx_out - torch_out).max())
    log.info("Wrote %s (%.1f MB); max difference from PyTorch %.2e", out, out.stat().st_size / 1e6, diff)
    return diff


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a checkpoint to ONNX.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, default=Path("runs/latest/footpose.onnx"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    diff = export(args.checkpoint, args.out)
    if diff > 1e-3:
        raise SystemExit(f"ONNX output differs from PyTorch by {diff:.2e}; do not ship this file.")
