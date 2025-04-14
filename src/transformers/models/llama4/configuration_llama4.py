# coding=utf-8
# Copyright 2025 The LLAMA4 and HuggingFace Inc. team. All rights reserved.
#
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


from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class Llama4VisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Llama4VisionModel`]. It is used to instantiate a
    Llama4 vision model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Llama4 109B.

    e.g. [meta-llama/Llama-4-Scout-17B-16E](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        num_hidden_layers (`int`, *optional*, defaults to 34):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            Number of channels in the input image.
        intermediate_size (`int`, *optional*, defaults to 5632):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        vision_output_dim (`int`, *optional*, defaults to 7680):
            Dimensionality of the vision model output. Includes output of transformer
            encoder with intermediate layers and global transformer encoder.
        image_size (`int`, *optional*, defaults to 448):
            The size (resolution) of each image *tile*.
        patch_size (`int`, *optional*, defaults to 14):
            The size (resolution) of each patch.
        norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        vision_feature_layer (``, *optional*, defaults to -1): TODO
        vision_feature_select_strategy (`int`, *optional*, defaults to `"default"`): TODO
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        pixel_shuffle_ratio (`int`, *optional*, defaults to 0.5): TODO
        projector_input_dim (`int`, *optional*, defaults to 4096): TODO
        projector_output_dim (`int`, *optional*, defaults to 4096): TODO
        multi_modal_projector_bias (`int`, *optional*, defaults to `False`): TODO
        projector_dropout (`int`, *optional*, defaults to 0.0): TODO
        attention_dropout (`int`, *optional*, defaults to 0.0): TODO
        rope_theta (`int`, *optional*, defaults to 10000): TODO
    """

    base_model_tp_plan = {
        "model.layers.*.self_attn.q_proj": "colwise",
        "model.layers.*.self_attn.k_proj": "colwise",
        "model.layers.*.self_attn.v_proj": "colwise",
        "model.layers.*.self_attn.o_proj": "rowwise",
        "vision_adapter.mlp.fc1": "colwise",
        "vision_adapter.mlp.fc2": "rowwise",
        "patch_embedding.linear": "colwise_rep",
    }
    model_type = "llama4_vision_model"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size: int = 768,
        hidden_act: str = "gelu",
        num_hidden_layers: int = 34,
        num_attention_heads: int = 16,
        num_channels: int = 3,
        intermediate_size: int = 5632,
        vision_output_dim: int = 7680,
        image_size: int = 448,
        patch_size: int = 14,
        norm_eps: float = 1e-5,
        vision_feature_layer=-1,
        vision_feature_select_strategy="default",
        initializer_range: float = 0.02,
        pixel_shuffle_ratio=0.5,
        projector_input_dim=4096,
        projector_output_dim=4096,
        multi_modal_projector_bias=False,
        projector_dropout=0.0,
        attention_dropout=0.0,
        rope_theta=10000,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.num_hidden_layers = num_hidden_layers
        self.num_channels = num_channels
        self.intermediate_size = intermediate_size
        self.image_size = image_size
        self.vision_output_dim = vision_output_dim
        self.patch_size = patch_size
        self.norm_eps = norm_eps
        self.num_attention_heads = num_attention_heads
        self.initializer_range = initializer_range
        self.pixel_shuffle_ratio = pixel_shuffle_ratio
        self.projector_input_dim = projector_input_dim
        self.projector_output_dim = projector_output_dim
        self.multi_modal_projector_bias = multi_modal_projector_bias
        self.projector_dropout = projector_dropout
        self.attention_dropout = attention_dropout
        self.vision_feature_layer = vision_feature_layer
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.rope_theta = rope_theta
        super().__init__(**kwargs)


class Llama4TextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Llama4TextModel`]. It is used to instantiate a
    Llama4 text model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Llama4 109B.

    e.g. [meta-llama/Llama-4-Scout-17B-16E](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 202048):
            Vocabulary size of the Llama4 text model. Defines the maximum number of different tokens that can be represented
            by the `inputs_ids` passed when calling [`Llama4TextModel`].
        hidden_size (`int`, *optional*, defaults to 5120):
            Dimensionality of the embeddings and hidden states.
        intermediate_size (`int`, *optional*, defaults to 8192):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        intermediate_size_mlp (`int`, *optional*, defaults to 16384): TODO
        num_hidden_layers (`int`, *optional*, defaults to 48):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 40):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If not
            specified, will default to `num_attention_heads`.
        head_dim (`int`, *optional*, defaults to 128): TODO
        hidden_act (`str` or `Callable`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the encoder and pooler.
        max_position_embeddings (`int`, *optional*, defaults to 131072):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions.
        pad_token_id (`int`, *optional*, defaults to 128004):
            The id of the padding token.
        bos_token_id (`int`, *optional*, defaults to 1):
            The id of the beginning of sentence token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the end of sentence token.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to `500000.0`):
            The base period of the RoPE embeddings.
        attention_dropout (`int`, *optional*, defaults to 0.0): TODO
        num_experts_per_tok (`int`, *optional*, defaults to 1): TODO
        num_local_experts (`int`, *optional*, defaults to 16): TODO
        moe_layers (`int`, *optional*): TODO
        interleave_moe_layer_step (`int`, *optional*, defaults to 1): TODO
        use_qk_norm (`int`, *optional*, defaults to `True`): TODO
        output_router_logits (`int`, *optional*, defaults to `False`): TODO
        router_aux_loss_coef (`int`, *optional*, defaults to 0.001): TODO
        router_jitter_noise (`int`, *optional*, defaults to 0.0): TODO
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
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
                `short_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
            <TODO>
            <TODO>
        no_rope_layers (`int`, *optional*): TODO
        no_rope_layer_interval (`int`, *optional*, defaults to 4): TODO
        attention_chunk_size (`int`, *optional*, defaults to 8192):
            <TODO>
        attn_temperature_tuning (`int`, *optional*, defaults to 4): TODO
        floor_scale (`int`, *optional*, defaults to 8192): TODO
        attn_scale (`int`, *optional*, defaults to 0.1): TODO
        cache_implementation (`<fill_type>`, *optional*, defaults to `"hybrid"`): <fill_docstring>

    Example:
    """

    model_type = "llama4_text"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.input_layernorm.weight": "sequence_parallel",
        "layers.*.post_attention_layernorm.weight": "sequence_parallel",
        "norm.weight": "sequence_parallel",
        "layers.*.feed_forward.shared_expert.gate_proj": "local_colwise",
        "layers.*.feed_forward.shared_expert.up_proj": "local_colwise",
        "layers.*.feed_forward.shared_expert.down_proj": "local_rowwise",
        "layers.*.feed_forward.experts.gate_up_proj": "local_packed_rowwise",  # row because not linear
        "layers.*.feed_forward.experts.down_proj": "local_colwise",  # col because not linear
        "layers.*.feed_forward.experts": "local",
        "layers.*.feed_forward.gate_proj": "local_colwise",
        "layers.*.feed_forward.up_proj": "local_colwise",
        "layers.*.feed_forward.down_proj": "local_rowwise",
        "layers.*.feed_forward": "gather",
    }

    def __init__(
        self,
        vocab_size=202048,
        hidden_size=5120,
        intermediate_size=8192,
        intermediate_size_mlp=16384,
        num_hidden_layers=48,
        num_attention_heads=40,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=4096 * 32,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=500000,
        attention_dropout=0.0,
        num_experts_per_tok=1,
        num_local_experts=16,
        moe_layers=None,
        interleave_moe_layer_step=1,
        use_qk_norm=True,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        router_jitter_noise=0.0,
        rope_scaling=None,
        no_rope_layers=None,
        no_rope_layer_interval=4,
        attention_chunk_size=8192,
        attn_temperature_tuning=4,
        floor_scale=8192,
        attn_scale=0.1,
        cache_implementation="hybrid",
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.attn_temperature_tuning = attn_temperature_tuning
        self.attn_scale = attn_scale
        self.floor_scale = floor_scale
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.intermediate_size_mlp = intermediate_size_mlp
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.rope_scaling = rope_scaling
        self.attention_bias = False
        self.cache_implementation = cache_implementation
        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        self.use_qk_norm = use_qk_norm

        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts

        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_jitter_noise = router_jitter_noise
        default_no_rope_layers = [
            int((layer_idx + 1) % no_rope_layer_interval != 0) for layer_idx in range(self.num_hidden_layers)
        ]

        # no_rope_layers == [] is invalid as we cannot have 0 layers
        self.no_rope_layers = no_rope_layers if no_rope_layers else default_no_rope_layers

        self.interleave_moe_layer_step = interleave_moe_layer_step
        self.moe_layers = (
            moe_layers
            if moe_layers is not None
            else list(range(interleave_moe_layer_step - 1, num_hidden_layers, interleave_moe_layer_step))
        )
        self.attention_chunk_size = attention_chunk_size


class Llama4Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Llama4Model`]. It is used to instantiate an
    Llama4 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Llama4 109B.

    e.g. [meta-llama/Llama-4-Scout-17B-16E](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vision_config (`Llama4VisionConfig`, *optional*):
            The Llama4 Vision config.
        text_config (`Llama4TextConfig`, *optional*):
            The Llama4 Text config.
        boi_token_index (`int`, *optional*, defaults to 200080):
            The begin-of-image token index to wrap the image prompt.
        eoi_token_index (`int`, *optional*, defaults to 200081):
            The end-of-image token index to wrap the image prompt.
        image_token_index (`int`, *optional*, defaults to 200092):
            The image token index to encode the image prompt.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.

    ```python
    >>> from transformers import Llama4Model, Llama4Config

    >>> # Initializing a Llama4 7B style configuration
    >>> configuration = Llama4Config()

    >>> # Initializing a model from the Llama4 7B style configuration
    >>> model = Llama4Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "llama4"
    sub_configs = {"text_config": Llama4TextConfig, "vision_config": Llama4VisionConfig}
    base_model_tp_plan = {
        "multi_modal_projector.linear_1": "colwise_rep",
    }

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        boi_token_index=200080,
        eoi_token_index=200081,
        image_token_index=200092,
        tie_word_embeddings=False,
        **kwargs,
    ):
        if vision_config is None:
            self.vision_config = Llama4VisionConfig()
            logger.info("vision_config is None, using default llama4 vision config")
        elif isinstance(vision_config, dict):
            self.vision_config = Llama4VisionConfig(**vision_config)
        elif isinstance(vision_config, Llama4VisionConfig):
            self.vision_config = vision_config

        self.boi_token_index = boi_token_index
        self.eoi_token_index = eoi_token_index
        self.image_token_index = image_token_index
        if text_config is None:
            self.text_config = Llama4TextConfig()
            logger.info("text_config is None, using default llama4 text config")
        elif isinstance(text_config, dict):
            self.text_config = Llama4TextConfig(**text_config)
        elif isinstance(text_config, Llama4TextConfig):
            self.text_config = text_config

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = ["Llama4Config", "Llama4TextConfig", "Llama4VisionConfig"]
