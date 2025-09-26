# coding=utf-8
# Copyright 2025 Baidu and HuggingFace Inc. team. All rights reserved.
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
"""Ernie4.5-VL model configuration"""

from ...modeling_rope_utils import rope_config_validation
from ...modeling_utils import PretrainedConfig


class Ernie4_5_VLVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of the [`Ernie4_5_VLVisionTransformerPretrainedModel`] and the
    [`Ernie4_5_VLVariableResolutionResamplerModel`]. It is used to instantiate the vision models portion of the complete
    Ernie4.5-VL model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        depth (`int`, *optional*, defaults to 32):
            Number of layers (depth) in the model.
        hidden_size (`int`, *optional*, defaults to 1280):
            Dimensionality of the encoder layers and the pooler layer.
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler.
        intermediate_size (`int`, *optional*, defaults to 5120):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        patch_size (`int`, *optional*, defaults to `14`):
            The size (resolution) of each patch.
        spatial_merge_size (`int`, *optional*, defaults to 2):
            The size used for merging spatial dimensions.
        temporal_merge_size (`int`, *optional*, defaults to 2):
            The size used for merge along the temporal dimension.
        text_hidden_size (`int`, *optional*, defaults to 2560):
            Dimensionality of the subsequent text model.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        vision_rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers in certain vision portions.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    """

    model_type = "ernie4_5_vl_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=32,
        hidden_size=1280,
        hidden_act="quick_gelu",
        intermediate_size=4 * 1280,
        num_heads=16,
        in_channels=3,
        patch_size=14,
        spatial_merge_size=2,
        temporal_merge_size=2,
        text_hidden_size=2560,
        rms_norm_eps=1e-5,
        vision_rms_norm_eps=1e-6,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # vision projection
        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size

        # resampler
        self.text_hidden_size = text_hidden_size
        self.temporal_merge_size = temporal_merge_size
        self.rms_norm_eps = rms_norm_eps
        self.vision_rms_norm_eps = vision_rms_norm_eps

        self.initializer_range = initializer_range


class Ernie4_5_VLTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Ernie4_5_VLTextModel`]. It is used to instantiate a
    the text model portion of the complete Ernie4.5-VL model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 103424):
            Vocabulary size of the Ernie 4.5 VL model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Ernie4_5_VLTextModel`]
        hidden_size (`int`, *optional*, defaults to 2560):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 12288):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 28):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 20):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 4):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `4`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 131072):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        use_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in any of the projections including mlp and attention for example.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 500000.0):
            The base period of the RoPE embeddings.
        freq_allocation (`int`, *optional*, defaults to 20):
            The absolute size allocated to the time dimension in 3D RoPE.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3', 'ernie_3d'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`list[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`list[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
                `freq_allocation` (`int`, *optional*):
                    Only used with 'ernie_3d'. The absolute size allocated to the time dimension in 3D RoPE
        moe_intermediate_size (`list[int]`, *optional*, defaults to [1536, 512]):
            Intermediate size of the routed experts; differs between text (first) and image (second) experts.
        moe_k (`int`, *optional*, defaults to 6):
            Number of selected experts.
        moe_num_experts (`int`, *optional*, defaults to 64):
            Number of routed experts.
        moe_num_shared_experts (`int`, *optional*, defaults to 2):
            The number of experts that are shared for all MoE forwards.
        moe_layer_start_index (`int`, *optional*, defaults to 1):
            The first index at which MoE layers start to appear.
        moe_layer_end_index (`int`, *optional*, defaults to -1):
            The last possible index for a MoE layer.
        moe_layer_interval (`int`, *optional*, defaults to 1):
            The intervals between MoE layers to appear.
        moe_norm_min (`float`, *optional*, defaults to 1e-12):
            Minimum division value during routing normalization.
        output_router_logits (`bool`, *optional*, defaults to `False`):
            Whether or not the router logits should be returned by the model. Enabling this will also
            allow the model to output the auxiliary loss, including load balancing loss and router z-loss.
        router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
            The aux loss factor for the total loss.
    """

    model_type = "ernie4_5_vl_text"
    base_config_key = "text_config"
    attribute_map = {"num_experts": "moe_num_experts", "num_experts_per_tok": "moe_k"}

    def __init__(
        self,
        vocab_size=103424,
        hidden_size=2560,
        intermediate_size=12288,
        num_hidden_layers=28,
        num_attention_heads=20,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        use_bias=False,
        tie_word_embeddings=True,
        rope_theta=500_000.0,
        freq_allocation=20,
        rope_scaling=None,
        moe_intermediate_size=[1536, 512],
        moe_k=6,
        moe_num_experts=64,
        moe_num_shared_experts=2,
        moe_layer_start_index=1,
        moe_layer_end_index=29,
        moe_layer_interval=1,
        moe_norm_min=1e-12,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.use_bias = use_bias
        self.rope_theta = rope_theta
        self.freq_allocation = freq_allocation
        self.rope_scaling = rope_scaling
        if rope_scaling is None:
            self.rope_scaling = {"rope_type": "ernie_3d", "freq_allocation": freq_allocation}
        rope_config_validation(self)
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_k = moe_k
        self.moe_num_experts = moe_num_experts
        self.moe_num_shared_experts = moe_num_shared_experts
        self.moe_layer_start_index = moe_layer_start_index
        self.moe_layer_end_index = moe_layer_end_index
        self.moe_layer_interval = moe_layer_interval
        self.moe_norm_min = moe_norm_min
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class Ernie4_5_VLConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Ernie4_5_VLModel`]. It is used to instantiate a
    Ernie4.5-VL model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    TODO [TODO](TODO) <-- 28B models hf style.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `Ernie4_5_VLTextConfig`):
            The config object or dictionary of the text backbone.
        vision_config (`Union[PreTrainedConfig, dict]`,  *optional*, defaults to `Ernie4_5_VLVisionConfig`):
            The config object or dictionary of the vision backbone.
        image_start_token_id (`int`, *optional*, defaults to 101304):
            The image token index to encode the start of image.
        image_end_token_id (`int`, *optional*, defaults to 101305):
            The image token index to encode the end of image.
        image_token_id (`int`, *optional*, defaults to 100295):
            The image token index to encode the image prompt.
        video_start_token_id (`int`, *optional*, defaults to 101306):
            The video token index to encode the start of video.
        video_end_token_id (`int`, *optional*, defaults to 101307):
            The video token index to encode the end of video.
        video_token_id (`int`, *optional*, defaults to 100296):
            The video token index to encode the video prompt.

    ```python
    >>> from transformers import Ernie4_5_VLForConditionalGeneration, Ernie4_5_VLConfig

    >>> # Initializing a Ernie4_5_VL style configuration
    >>> configuration = Ernie4_5_VLConfig()

    >>> # Initializing a model from the TODO configuration
    >>> model = Ernie4_5_VLForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "ernie4_5_vl"
    sub_configs = {"vision_config": Ernie4_5_VLVisionConfig, "text_config": Ernie4_5_VLTextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_start_token_id=101304,
        image_end_token_id=101305,
        image_token_id=100295,
        video_start_token_id=101306,
        video_end_token_id=101307,
        video_token_id=100296,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            # For BC use all kwargs to init `TextConfig`
            self.text_config = self.sub_configs["text_config"](**kwargs)

        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.image_token_id = image_token_id
        self.video_start_token_id = video_start_token_id
        self.video_end_token_id = video_end_token_id
        self.video_token_id = video_token_id

        super().__init__(**kwargs)


__all__ = [
    "Ernie4_5_VLConfig",
    "Ernie4_5_VLTextConfig",
    "Ernie4_5_VLVisionConfig",
]
