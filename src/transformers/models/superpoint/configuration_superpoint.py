from typing import List

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

SUPERPOINT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "stevenbucaille/superpoint": "https://huggingface.co/stevenbucaille/superpoint/blob/main/config.json"
}


class SuperPointConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SuperPointModel`]. It is used to instantiate a
    SuperPoint model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the SuperPoint
    [stevenbucaille/superpoint](https://huggingface.co/stevenbucaille/superpoint) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_sizes (`List[int]`, *optional*, defaults to `[64, 64, 128, 128, 256]`):
            The number of channels in each convolutional layer.
        descriptor_dim (`int`, *optional*, defaults to 256):
            The dimension of the descriptor.
        keypoint_threshold (`float`, *optional*, defaults to 0.005):
            The threshold to use for extracting keypoints.
        max_keypoints (`int`, *optional*, defaults to -1):
            The maximum number of keypoints to extract. If `-1`, will extract all keypoints.
        nms_radius (`int`, *optional*, defaults to 4):
            The radius for non-maximum suppression.
        border_removal_distance (`int`, *optional*, defaults to 4):
            The distance from the border to remove keypoints.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    Example:
    ```python
    >>> from transformers import SuperPointConfig, SuperPointModel

    >>> # Initializing a SuperPoint superpoint style configuration
    >>> configuration = SuperPointConfig()
    >>> # Initializing a model from the superpoint style configuration
    >>> model = SuperPointModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "superpoint"

    def __init__(
        self,
        hidden_sizes: List[int] = [64, 64, 128, 128, 256],
        descriptor_dim: int = 256,
        keypoint_threshold: float = 0.005,
        max_keypoints: int = -1,
        nms_radius: int = 4,
        border_removal_distance: int = 4,
        initializer_range=0.02,
        **kwargs,
    ):
        self.hidden_sizes = hidden_sizes
        self.descriptor_dim = descriptor_dim
        self.keypoint_threshold = keypoint_threshold
        self.max_keypoints = max_keypoints
        self.nms_radius = nms_radius
        self.border_removal_distance = border_removal_distance
        self.initializer_range = initializer_range

        super().__init__(**kwargs)
