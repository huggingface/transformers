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

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="magic-leap-community/superglue_indoor")
@strict
class SuperGlueConfig(PreTrainedConfig):
    r"""
    keypoint_detector_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `SuperPointConfig`):
        The config object or dictionary of the keypoint detector.
    keypoint_encoder_sizes (`list[int]`, *optional*, defaults to `[32, 64, 128, 256]`):
        The sizes of the keypoint encoder layers.
    gnn_layers_types (`list[str]`, *optional*, defaults to `['self', 'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross']`):
        The types of the GNN layers. Must be either 'self' or 'cross'.
    sinkhorn_iterations (`int`, *optional*, defaults to 100):
        The number of Sinkhorn iterations.
    matching_threshold (`float`, *optional*, defaults to 0.0):
        The matching threshold.

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
    sub_configs = {"keypoint_detector_config": AutoConfig}

    keypoint_detector_config: dict | PreTrainedConfig | None = None
    hidden_size: int = 256
    keypoint_encoder_sizes: list[int] | None = None
    gnn_layers_types: list[str] | None = None
    num_attention_heads: int = 4
    sinkhorn_iterations: int = 100
    matching_threshold: float = 0.0
    initializer_range: float = 0.02
    is_decoder: bool = False
    attention_probs_dropout_prob: int | float = 0.0

    def __post_init__(self, **kwargs):
        self.gnn_layers_types = self.gnn_layers_types if self.gnn_layers_types is not None else ["self", "cross"] * 9
        self.keypoint_encoder_sizes = (
            self.keypoint_encoder_sizes if self.keypoint_encoder_sizes is not None else [32, 64, 128, 256]
        )

        if isinstance(self.keypoint_detector_config, dict):
            self.keypoint_detector_config["model_type"] = self.keypoint_detector_config.get("model_type", "superpoint")
            self.keypoint_detector_config = CONFIG_MAPPING[self.keypoint_detector_config["model_type"]](
                **self.keypoint_detector_config
            )
        elif self.keypoint_detector_config is None:
            self.keypoint_detector_config = CONFIG_MAPPING["superpoint"]()

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        # Check whether all gnn_layers_types are either 'self' or 'cross'
        if not all(layer_type in ["self", "cross"] for layer_type in self.gnn_layers_types):
            raise ValueError("All gnn_layers_types must be either 'self' or 'cross'")

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size % num_attention_heads is different from zero")


__all__ = ["SuperGlueConfig"]
