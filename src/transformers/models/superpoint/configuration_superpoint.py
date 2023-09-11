from typing import List

from transformers import PretrainedConfig


class SuperPointConfig(PretrainedConfig):
    def __init__(
        self,
        conv_layers_sizes: List[int] = [64, 64, 128, 128, 256],
        descriptor_dim: int = 256,
        keypoint_threshold: float = 0.005,
        max_keypoints: int = -1,
        nms_radius: int = 4,
        border_removal_distance: int = 4,
        initializer_range=0.02,
        **kwargs,
    ):
        self.conv_layers_sizes = conv_layers_sizes
        self.descriptor_dim = descriptor_dim
        self.keypoint_threshold = keypoint_threshold
        self.max_keypoints = max_keypoints
        self.nms_radius = nms_radius
        self.remove_borders = border_removal_distance
        self.initializer_range = initializer_range

        super().__init__(**kwargs)
