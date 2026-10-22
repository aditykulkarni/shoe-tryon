"""Checks the setup: Python, PyTorch, and whether the GPU is usable.

Run with:  uv run footpose-check
"""

import platform
import sys

import torch

from .device import best_device


def main() -> None:
    print(f"Python   {sys.version.split()[0]} on {platform.system()} {platform.release()}")
    print(f"PyTorch  {torch.__version__}")
    device = best_device()
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU      {props.name}, {props.total_memory / 2**30:.0f} GB, CUDA {torch.version.cuda}")
        x = torch.randn(2048, 2048, device=device)
        torch.cuda.synchronize()
        print(f"Test     matrix multiply on GPU OK ({float((x @ x).abs().mean()):.1f})")
    elif device.type == "mps":
        print("GPU      Apple GPU (MPS)")
    else:
        print("GPU      none found, training will use the CPU (slow).")
        if platform.system() in ("Windows", "Linux"):
            print("         If you have an NVIDIA card: update its driver, then run")
            print("         `uv sync --reinstall-package torch`.")
    print("Ready." if device.type != "cpu" else "Setup works, but without a GPU.")
