# Copyright 2025 The HuggingFace Team. All rights reserved.
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


@auto_docstring(checkpoint="zju-community/efficientloftr")
@strict
class EfficientLoFTRConfig(PreTrainedConfig):
    r"""
    stage_num_blocks (`List`, *optional*, defaults to [1, 2, 4, 14]):
        The number of blocks in each stages
    stage_stride (`List`, *optional*, defaults to [2, 1, 2, 2]):
        The stride used in each stage
    q_aggregation_kernel_size (`int`, *optional*, defaults to 4):
        The kernel size of the aggregation of query states in the fusion network
    kv_aggregation_kernel_size (`int`, *optional*, defaults to 4):
        The kernel size of the aggregation of key and value states in the fusion network
    q_aggregation_stride (`int`, *optional*, defaults to 4):
        The stride of the aggregation of query states in the fusion network
    kv_aggregation_stride (`int`, *optional*, defaults to 4):
        The stride of the aggregation of key and value states in the fusion network
    num_attention_layers (`int`, *optional*, defaults to 4):
        Number of attention layers in the LocalFeatureTransformer
    mlp_activation_function (`str`, *optional*, defaults to `"leaky_relu"`):
        Activation function used in the attention mlp layer.
    coarse_matching_skip_softmax (`bool`, *optional*, defaults to `False`):
        Whether to skip softmax or not at the coarse matching step.
    coarse_matching_threshold (`float`, *optional*, defaults to 0.2):
        The threshold for the minimum score required for a match.
    coarse_matching_temperature (`float`, *optional*, defaults to 0.1):
        The temperature to apply to the coarse similarity matrix
    coarse_matching_border_removal (`int`, *optional*, defaults to 2):
        The size of the border to remove during coarse matching
    fine_kernel_size (`int`, *optional*, defaults to 8):
        Kernel size used for the fine feature matching
    batch_norm_eps (`float`, *optional*, defaults to 1e-05):
        The epsilon used by the batch normalization layers
    fine_matching_slice_dim (`int`, *optional*, defaults to 8):
        The size of the slice used to divide the fine features for the first and second fine matching stages.
    fine_matching_regress_temperature (`float`, *optional*, defaults to 10.0):
        The temperature to apply to the fine similarity matrix

    Examples:
        ```python
        >>> from transformers import EfficientLoFTRConfig, EfficientLoFTRForKeypointMatching

        >>> # Initializing a EfficientLoFTR configuration
        >>> configuration = EfficientLoFTRConfig()

        >>> # Initializing a model from the EfficientLoFTR configuration
        >>> model = EfficientLoFTRForKeypointMatching(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
        ```
    """

    model_type = "efficientloftr"

    stage_num_blocks: list[int] | None = None
    out_features: list[int] | None = None
    stage_stride: list[int] | None = None
    hidden_size: int = 256
    activation_function: str = "relu"
    q_aggregation_kernel_size: int = 4
    kv_aggregation_kernel_size: int = 4
    q_aggregation_stride: int = 4
    kv_aggregation_stride: int = 4
    num_attention_layers: int = 4
    num_attention_heads: int = 8
    attention_dropout: float | int = 0.0
    attention_bias: bool = False
    mlp_activation_function: str = "leaky_relu"
    coarse_matching_skip_softmax: bool = False
    coarse_matching_threshold: float = 0.2
    coarse_matching_temperature: float = 0.1
    coarse_matching_border_removal: int = 2
    fine_kernel_size: int = 8
    batch_norm_eps: float = 1e-5
    rope_parameters: dict | None = None
    fine_matching_slice_dim: int = 8
    fine_matching_regress_temperature: float = 10.0
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        # Stage level of RepVGG
        self.stage_num_blocks = self.stage_num_blocks if self.stage_num_blocks is not None else [1, 2, 4, 14]
        self.stage_stride = self.stage_stride if self.stage_stride is not None else [2, 1, 2, 2]
        self.out_features = self.out_features if self.out_features is not None else [64, 64, 128, 256]
        self.stage_in_channels = [1] + self.out_features[:-1]

        # Block level of RepVGG
        self.stage_block_stride = [
            [stride] + [1] * (num_blocks - 1) for stride, num_blocks in zip(self.stage_stride, self.stage_num_blocks)
        ]
        self.stage_block_out_channels = [
            [self.out_features[stage_idx]] * num_blocks for stage_idx, num_blocks in enumerate(self.stage_num_blocks)
        ]
        self.stage_block_in_channels = [
            [self.stage_in_channels[stage_idx]] + self.stage_block_out_channels[stage_idx][:-1]
            for stage_idx in range(len(self.stage_num_blocks))
        ]

        self.num_key_value_heads = self.num_attention_heads
        self.fine_fusion_dims = list(reversed(self.out_features))[:-1]
        self.intermediate_size = self.hidden_size * 2
        kwargs.setdefault("partial_rotary_factor", 4.0)  # assign default for BC
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.hidden_size != self.out_features[-1]:
            raise ValueError(
                f"hidden_size should be equal to the last value in out_features. hidden_size = {self.hidden_size}, out_features = {self.out_features[-1]}"
            )


__all__ = ["EfficientLoFTRConfig"]
