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
from typing import Optional

from ...configuration_utils import PretrainedConfig
from ...modeling_rope_utils import rope_config_validation


class EfficientLoFTRConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`EffientLoFTRFromKeypointMatching`].
    It is used to instantiate a EfficientLoFTR model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the
    EfficientLoFTR [zju-community/efficientloftr](https://huggingface.co/zju-community/efficientloftr) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        stage_num_blocks (`List`, *optional*, defaults to [1, 2, 4, 14]):
            The number of blocks in each stages
        out_features (`List`, *optional*, defaults to [64, 64, 128, 256]):
            The number of channels in each stage
        stage_stride (`List`, *optional*, defaults to [2, 1, 2, 2]):
            The stride used in each stage
        hidden_size (`int`, *optional*, defaults to 256):
            The dimension of the descriptors.
        activation_function (`str`, *optional*, defaults to `"relu"`):
            The activation function used in the backbone
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
        num_attention_heads (`int`, *optional*, defaults to 8):
            The number of heads in the GNN layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during attention.
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
            The epsilon used by the batch normalization layers.
        embedding_size (`List`, *optional*, defaults to [15, 20]):
            The size (height, width) of the embedding for the position embeddings.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        partial_rotary_factor (`float`, *optional*, defaults to 4.0):
            Dim factor for the RoPE embeddings, in EfficientLoFTR, frequencies should be generated for
            the whole hidden_size, so this factor is used to compensate.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3', '2d'], with 'default' being the original RoPE implementation.
                `dim` (`int`): The dimension of the RoPE embeddings.
        fine_matching_slice_dim (`int`, *optional*, defaults to 8):
            The size of the slice used to divide the fine features for the first and second fine matching stages.
        fine_matching_regress_temperature (`float`, *optional*, defaults to 10.0):
            The temperature to apply to the fine similarity matrix
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

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

    def __init__(
        self,
        stage_num_blocks: Optional[list[int]] = None,
        out_features: Optional[list[int]] = None,
        stage_stride: Optional[list[int]] = None,
        hidden_size: int = 256,
        activation_function: str = "relu",
        q_aggregation_kernel_size: int = 4,
        kv_aggregation_kernel_size: int = 4,
        q_aggregation_stride: int = 4,
        kv_aggregation_stride: int = 4,
        num_attention_layers: int = 4,
        num_attention_heads: int = 8,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        mlp_activation_function: str = "leaky_relu",
        coarse_matching_skip_softmax: bool = False,
        coarse_matching_threshold: float = 0.2,
        coarse_matching_temperature: float = 0.1,
        coarse_matching_border_removal: int = 2,
        fine_kernel_size: int = 8,
        batch_norm_eps: float = 1e-5,
        embedding_size: Optional[list[int]] = None,
        rope_theta: float = 10000.0,
        partial_rotary_factor: float = 4.0,
        rope_scaling: Optional[dict] = None,
        fine_matching_slice_dim: int = 8,
        fine_matching_regress_temperature: float = 10.0,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        # Stage level of RepVGG
        self.stage_num_blocks = stage_num_blocks if stage_num_blocks is not None else [1, 2, 4, 14]
        self.stage_stride = stage_stride if stage_stride is not None else [2, 1, 2, 2]
        self.out_features = out_features if out_features is not None else [64, 64, 128, 256]
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

        # Fine matching level of EfficientLoFTR
        self.fine_fusion_dims = list(reversed(self.out_features))[:-1]

        self.hidden_size = hidden_size
        if self.hidden_size != self.out_features[-1]:
            raise ValueError(
                f"hidden_size should be equal to the last value in out_features. hidden_size = {self.hidden_size}, out_features = {self.stage_out_channels}"
            )

        self.activation_function = activation_function
        self.q_aggregation_kernel_size = q_aggregation_kernel_size
        self.kv_aggregation_kernel_size = kv_aggregation_kernel_size
        self.q_aggregation_stride = q_aggregation_stride
        self.kv_aggregation_stride = kv_aggregation_stride
        self.num_attention_layers = num_attention_layers
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.intermediate_size = self.hidden_size * 2
        self.mlp_activation_function = mlp_activation_function
        self.coarse_matching_skip_softmax = coarse_matching_skip_softmax
        self.coarse_matching_threshold = coarse_matching_threshold
        self.coarse_matching_temperature = coarse_matching_temperature
        self.coarse_matching_border_removal = coarse_matching_border_removal
        self.fine_kernel_size = fine_kernel_size
        self.batch_norm_eps = batch_norm_eps
        self.fine_matching_slice_dim = fine_matching_slice_dim
        self.fine_matching_regress_temperature = fine_matching_regress_temperature

        self.num_key_value_heads = num_attention_heads
        self.embedding_size = embedding_size if embedding_size is not None else [15, 20]
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling if rope_scaling is not None else {"rope_type": "default"}

        # for compatibility with "default" rope type
        self.partial_rotary_factor = partial_rotary_factor
        rope_config_validation(self)

        self.initializer_range = initializer_range

        super().__init__(**kwargs)


__all__ = ["EfficientLoFTRConfig"]
