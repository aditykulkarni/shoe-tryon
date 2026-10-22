# Data

Datasets live here and are not committed to Git (they are large and may carry licence or privacy terms). Each dataset is a folder with `train/` and `val/`, and each of those has `images/` and `annotations.json`:

```
data/
  toy/            made by `uv run footpose-toy-data`
    train/
      images/
      annotations.json
    val/
      images/
      annotations.json
```

The annotation format is described at the top of [footpose/dataset.py](../footpose/dataset.py). Keep the people in `val/` out of `train/` entirely, otherwise the validation score is flattering.
