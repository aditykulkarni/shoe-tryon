"""The keypoints the model predicts, and how they map to output channels.

Each foot has the same 8 points. The model outputs one heatmap per point per
side, so 16 channels: channels 0-7 are the left foot, 8-15 the right foot.
"""

KEYPOINT_NAMES: tuple[str, ...] = (
    "toe_tip",
    "big_toe_joint",
    "little_toe_joint",
    "heel_back",
    "inner_ankle",
    "outer_ankle",
    "instep",
    "ankle_front",
)
SIDES: tuple[str, ...] = ("left", "right")

NUM_KEYPOINTS = len(KEYPOINT_NAMES)
NUM_CHANNELS = NUM_KEYPOINTS * len(SIDES)

# Visibility flags, as in COCO.
NOT_LABELLED = 0
HIDDEN = 1
VISIBLE = 2


def channel(side: str, name: str) -> int:
    """Output channel index for one point of one foot."""
    return SIDES.index(side) * NUM_KEYPOINTS + KEYPOINT_NAMES.index(name)


def flip_permutation() -> list[int]:
    """Channel order after a horizontal flip.

    Mirroring the image turns a left foot into a right foot, so the left and
    right channel blocks swap. The points themselves keep their names.
    """
    left = list(range(NUM_KEYPOINTS))
    right = list(range(NUM_KEYPOINTS, NUM_CHANNELS))
    return right + left
