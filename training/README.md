# Foot keypoint model

Trains the model that finds 8 points on each foot from the phone's looking-down view. The trained model is exported to ONNX and runs in the browser. Setup on Windows: [docs/SETUP-WINDOWS.md](../docs/SETUP-WINDOWS.md). The plan behind it: [Foot keypoint model plan](https://claude.ai/artifact/RBbRZosA5iwBB8cRNcg1Ex).

## Commands

| Command | What it does |
| --- | --- |
| `uv run footpose-check` | Confirms Python, PyTorch and the GPU work |
| `uv run footpose-toy-data` | Writes a toy dataset of drawn feet to `data/toy` |
| `uv run footpose-train --data data/toy` | Trains; writes `runs/latest/best.pt` and `history.csv` |
| `uv run footpose-predict runs/latest/best.pt photo.jpg` | Draws predicted points on a photo |
| `uv run footpose-export runs/latest/best.pt` | Writes `footpose.onnx` and checks it matches PyTorch |
| `uv run pytest` | Runs the tests |
| `uv run ruff check .` and `uv run ruff format .` | Lints and formats the code |

Add `--help` to any `footpose-` command to see its options.

## How it works, file by file

Read them in this order. Each one introduces a few Python ideas.

| File | What it does | Python ideas to look for |
| --- | --- | --- |
| `footpose/keypoints.py` | Names the 8 points and maps them to 16 output channels | Tuples, constants, functions with type hints |
| `footpose/heatmaps.py` | Turns points into blurry "heatmap" targets and back | NumPy arrays, broadcasting, PyTorch tensors |
| `footpose/model.py` | The network: MobileNetV3 plus an upsampling head | Classes, inheritance (`nn.Module`), `__init__` and `forward` |
| `footpose/dataset.py` | Loads images and labels, augments them | `@dataclass`, JSON, `pathlib`, the `Dataset` protocol (`__len__`, `__getitem__`) |
| `footpose/metrics.py` | PCK, the accuracy score | Boolean masks, vectorised maths instead of loops |
| `footpose/train.py` | The training loop, checkpoints, the history file | `argparse`, `logging`, `with` blocks, the optimiser and learning-rate schedule |
| `footpose/export.py` | ONNX export and a parity check | Raising errors, `SystemExit` |
| `footpose/toydata.py` | The toy dataset | Random numbers with a seed, drawing with OpenCV |
| `tests/` | Checks each piece and a short end-to-end run | `pytest` fixtures, `assert` |

## The model in one paragraph

A MobileNetV3 backbone, pretrained on ImageNet, turns a 256 x 256 photo into coarse 8 x 8 features. Three upsampling layers grow those into 64 x 64 heatmaps, one per point per foot (16 in all). Training compares them with Gaussian blobs drawn at the labelled points. At inference the brightest pixel of each heatmap is the point, and its brightness is the confidence. Flipping an image left to right during training swaps the left and right channels, which teaches the model which foot is which.
