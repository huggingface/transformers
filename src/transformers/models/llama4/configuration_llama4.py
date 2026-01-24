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


from ...configuration_utils import PreTrainedConfig, layer_type_validation
from ...modeling_rope_utils import RopeParameters
from ...utils import logging


logger = logging.get_logger(__name__)


class Llama4VisionConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Llama4VisionModel`]. It is used to instantiate a
    Llama4 vision model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Llama4 109B.

    e.g. [meta-llama/Llama-4-Scout-17B-16E](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

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
        vision_feature_select_strategy (`int`, *optional*, defaults to `"default"`):
            The feature selection strategy used to select the vision features from the vision model.
            Should be same as in model's config
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        pixel_shuffle_ratio (`int`, *optional*, defaults to 0.5):
            The ratio used for the pixel shuffle operation in the multi-modal projector.
        projector_input_dim (`int`, *optional*, defaults to 4096):
            Dimensionality of the input to the multi-modal projector.
        projector_output_dim (`int`, *optional*, defaults to 4096):
            Dimensionality of the output of the multi-modal projector.
        multi_modal_projector_bias (`int`, *optional*, defaults to `False`):
            Whether to use bias in the multi-modal projector layers.
        projector_dropout (`int`, *optional*, defaults to 0.0):
            The dropout probability for the multi-modal projector.
        attention_dropout (`int`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        rope_parameters (`RopeParameters`, *optional*):
            RoPE Parameters
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
        hidden_size: int | None = 768,
        hidden_act: str | None = "gelu",
        num_hidden_layers: int | None = 34,
        num_attention_heads: int | None = 16,
        num_channels: int | None = 3,
        intermediate_size: int | None = 5632,
        vision_output_dim: int | None = 7680,
        image_size: int | None = 448,
        patch_size: int | None = 14,
        norm_eps: float | None = 1e-5,
        vision_feature_select_strategy: str | None = "default",
        initializer_range: float | None = 0.02,
        pixel_shuffle_ratio: float | None = 0.5,
        projector_input_dim: int | None = 4096,
        projector_output_dim: int | None = 4096,
        multi_modal_projector_bias: bool | None = False,
        projector_dropout: float | None = 0.0,
        attention_dropout: float | None = 0.0,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
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
        self.vision_feature_select_strategy = vision_feature_select_strategy

        self.rope_parameters = rope_parameters

        super().__init__(**kwargs)


class Llama4TextConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Llama4TextModel`]. It is used to instantiate a
    Llama4 text model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Llama4 109B.

    e.g. [meta-llama/Llama-4-Scout-17B-16E](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 202048):
            Vocabulary size of the Llama4 text model. Defines the maximum number of different tokens that can be represented
            by the `inputs_ids` passed when calling [`Llama4TextModel`].
        hidden_size (`int`, *optional*, defaults to 5120):
            Dimensionality of the embeddings and hidden states.
        intermediate_size (`int`, *optional*, defaults to 8192):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        intermediate_size_mlp (`int`, *optional*, defaults to 16384):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the MoE MLP.
        num_hidden_layers (`int`, *optional*, defaults to 48):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 40):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If not
            specified, will default to `num_attention_heads`.
        head_dim (`int`, *optional*, defaults to 128):
            The attention head dimension size of the model.
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
        attention_dropout (`int`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        num_experts_per_tok (`int`, *optional*, defaults to 1):
            Number of experts to route each token to. This is the top-k value for the token-choice routing.
        num_local_experts (`int`, *optional*, defaults to 16):
            Number of experts for each Softmax router.
        moe_layers (`int`, *optional*):
            Indices of the layers that are MoE layers. If not specified, will be calculated using `interleave_moe_layer_step`.
        interleave_moe_layer_step (`int`, *optional*, defaults to 1):
            The frequency of MoE layers in the model. For example, setting it to 2 means every 2nd layer is an MoE layer.
        use_qk_norm (`int`, *optional*, defaults to `True`):
            Whether to normalize the Query and Key matrices in the attention layer.
        output_router_logits (`int`, *optional*, defaults to `False`):
            Whether or not to return the router logits of all MoE layers.
        router_aux_loss_coef (`int`, *optional*, defaults to 0.001):
            The aux loss factor for the total loss.
        router_jitter_noise (`int`, *optional*, defaults to 0.0):
            The amount of noise to add to the router logits.
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        no_rope_layers (`list[int]`, *optional*):
            List with at least the same length as the number of layers in the model.
            A `1` at an index position indicates that the corresponding layer will use RoPE,
            while a `0` indicates that it's a NoPE layer.
        no_rope_layer_interval (`int`, *optional*, defaults to 4):
            If `no_rope_layers` is `None`, it will be created using a NoPE layer every
            `no_rope_layer_interval` layers.
        attention_chunk_size (`int`, *optional*, defaults to 8192):
            Chunk size for the attention computation.
        layer_types (`list`, *optional*):
            Attention pattern for each layer.
        attn_temperature_tuning (`bool`, *optional*, defaults to `True`):
            Whether to dynamically scale the attention temperature for each query token based on sequence length.
            Recommended for long sequences (e.g., >32k tokens) to maintain stable output results.
        floor_scale (`int`, *optional*, defaults to 8192):
            Scaling factor for the floor operation in the attention mechanism.
        attn_scale (`int`, *optional*, defaults to 0.1):
            Scaling factor for the attention scores.

    Example:
    """

    model_type = "llama4_text"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 500000.0
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
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
    base_model_ep_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.feed_forward.experts.gate_up_proj": "grouped_gemm",  # row because not linear
        "layers.*.feed_forward.experts.down_proj": "grouped_gemm",  # col because not linear
        "layers.*.feed_forward.experts": "gather",  # all reduce
        "layers.*.feed_forward.gate_proj": "local_colwise",
        "layers.*.feed_forward.up_proj": "local_colwise",
        "layers.*.feed_forward.down_proj": "local_rowwise",
        "layers.*.feed_forward.router": "ep_router",
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
        attention_dropout=0.0,
        num_experts_per_tok=1,
        num_local_experts=16,
        moe_layers=None,
        interleave_moe_layer_step=1,
        use_qk_norm=True,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        router_jitter_noise=0.0,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        no_rope_layers=None,
        no_rope_layer_interval=4,
        attention_chunk_size=8192,
        layer_types=None,
        attn_temperature_tuning=True,
        floor_scale=8192,
        attn_scale=0.1,
        **kwargs,
    ):
        self.tie_word_embeddings = tie_word_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
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
        self.attention_bias = False
        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        self.use_qk_norm = use_qk_norm
        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts

        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_jitter_noise = router_jitter_noise

        # Backwards compatibility
        if no_rope_layers == []:
            no_rope_layers = None

        default_no_rope_layers = [
            int((layer_idx + 1) % no_rope_layer_interval != 0) for layer_idx in range(self.num_hidden_layers)
        ]

        self.no_rope_layers = no_rope_layers if no_rope_layers else default_no_rope_layers

        self.interleave_moe_layer_step = interleave_moe_layer_step
        self.moe_layers = (
            moe_layers
            if moe_layers is not None
            else list(
                range(
                    interleave_moe_layer_step - 1,
                    num_hidden_layers,
                    interleave_moe_layer_step,
                )
            )
        )
        self.attention_chunk_size = attention_chunk_size

        self.layer_types = layer_types
        if layer_types is None:
            self.layer_types = [
                "chunked_attention" if no_rope else "full_attention" for no_rope in self.no_rope_layers
            ]
        layer_type_validation(self.layer_types, self.num_hidden_layers)

        self.rope_parameters = rope_parameters
        super().__init__(**kwargs)


class Llama4Config(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Llama4Model`]. It is used to instantiate an
    Llama4 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Llama4 109B.

    e.g. [meta-llama/Llama-4-Scout-17B-16E](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


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
    attribute_map = {
        "image_token_id": "image_token_index",
        "boi_token_id": "boi_token_index",
        "eoi_token_id": "eoi_token_index",
    }
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

        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(**kwargs)


__all__ = ["Llama4Config", "Llama4TextConfig", "Llama4VisionConfig"]
