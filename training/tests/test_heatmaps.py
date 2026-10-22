import numpy as np
import torch

from footpose.heatmaps import decode_heatmaps, render_heatmaps
from footpose.keypoints import NUM_CHANNELS, channel, flip_permutation


def test_round_trip_is_within_one_input_pixel():
    rng = np.random.default_rng(0)
    points = rng.uniform(20, 230, size=(NUM_CHANNELS, 2)).astype(np.float32)
    visibility = np.full(NUM_CHANNELS, 2.0, dtype=np.float32)
    heatmaps, weights = render_heatmaps(points, visibility, (64, 64), stride=4)
    decoded, scores = decode_heatmaps(torch.from_numpy(heatmaps)[None], stride=4)
    assert weights.tolist() == [1.0] * NUM_CHANNELS
    assert (decoded[0].numpy() - points).__abs__().max() < 2.5
    assert scores.min() > 0.8


def test_unlabelled_points_get_zero_weight():
    points = np.full((NUM_CHANNELS, 2), 100.0, dtype=np.float32)
    visibility = np.zeros(NUM_CHANNELS, dtype=np.float32)
    visibility[3] = 1
    heatmaps, weights = render_heatmaps(points, visibility, (64, 64), stride=4)
    assert weights.sum() == 1 and weights[3] == 1
    assert heatmaps[0].max() == 0


def test_flip_swaps_left_and_right():
    perm = flip_permutation()
    assert perm[channel("left", "heel_back")] == channel("right", "heel_back")
    assert sorted(perm) == list(range(NUM_CHANNELS))
