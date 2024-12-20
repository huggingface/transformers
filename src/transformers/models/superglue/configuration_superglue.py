# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import TYPE_CHECKING, List

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING


if TYPE_CHECKING:
    from ..superpoint import SuperPointConfig

logger = logging.get_logger(__name__)


class SuperGlueConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SuperGlueModel`]. It is used to instantiate a
    SuperGlue model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the SuperGlue
    [magic-leap-community/superglue_indoor](https://huggingface.co/magic-leap-community/superglue_indoor) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        keypoint_detector_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `SuperPointConfig`):
            The config object or dictionary of the keypoint detector.
        hidden_size (`int`, *optional*, defaults to 256):
            The dimension of the descriptors.
        keypoint_encoder_sizes (`List[int]`, *optional*, defaults to `[32, 64, 128, 256]`):
            The sizes of the keypoint encoder layers.
        gnn_layers_types (`List[str]`, *optional*, defaults to `['self', 'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross']`):
            The types of the GNN layers. Must be either 'self' or 'cross'.
        num_attention_heads (`int`, *optional*, defaults to 4):
            The number of heads in the GNN layers.
        sinkhorn_iterations (`int`, *optional*, defaults to 100):
            The number of Sinkhorn iterations.
        matching_threshold (`float`, *optional*, defaults to 0.0):
            The matching threshold.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    Examples:
        ```python
        >>> from transformers import SuperGlueConfig, SuperGlueModel

        >>> # Initializing a SuperGlue superglue style configuration
        >>> configuration = SuperGlueConfig()

        >>> # Initializing a model from the superglue style configuration
        >>> model = SuperGlueModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
        ```
    """

    model_type = "superglue"

    def __init__(
        self,
        keypoint_detector_config: "SuperPointConfig" = None,
        hidden_size: int = 256,
        keypoint_encoder_sizes: List[int] = None,
        gnn_layers_types: List[str] = None,
        num_attention_heads: int = 4,
        sinkhorn_iterations: int = 100,
        matching_threshold: float = 0.0,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        self.gnn_layers_types = gnn_layers_types if gnn_layers_types is not None else ["self", "cross"] * 9
        # Check whether all gnn_layers_types are either 'self' or 'cross'
        if not all(layer_type in ["self", "cross"] for layer_type in self.gnn_layers_types):
            raise ValueError("All gnn_layers_types must be either 'self' or 'cross'")

        if hidden_size % num_attention_heads != 0:
            raise ValueError("hidden_size % num_attention_heads is different from zero")

        self.keypoint_encoder_sizes = (
            keypoint_encoder_sizes if keypoint_encoder_sizes is not None else [32, 64, 128, 256]
        )
        self.hidden_size = hidden_size
        self.keypoint_encoder_sizes = keypoint_encoder_sizes
        self.gnn_layers_types = gnn_layers_types
        self.num_attention_heads = num_attention_heads
        self.sinkhorn_iterations = sinkhorn_iterations
        self.matching_threshold = matching_threshold

        if isinstance(keypoint_detector_config, dict):
            keypoint_detector_config["model_type"] = (
                keypoint_detector_config["model_type"] if "model_type" in keypoint_detector_config else "superpoint"
            )
            keypoint_detector_config = CONFIG_MAPPING[keypoint_detector_config["model_type"]](
                **keypoint_detector_config
            )
        if keypoint_detector_config is None:
            keypoint_detector_config = CONFIG_MAPPING["superpoint"]()

        self.keypoint_detector_config = keypoint_detector_config
        self.initializer_range = initializer_range
        self.attention_probs_dropout_prob = 0
        self.is_decoder = False

        super().__init__(**kwargs)


__all__ = ["SuperGlueConfig"]
