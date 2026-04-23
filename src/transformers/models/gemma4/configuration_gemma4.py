# Copyright 2026 the HuggingFace Team. All rights reserved.
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

from typing import Any, Literal

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging
from ...utils.type_validators import interval


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="google/gemma-4-e2b-it")
@strict
class Gemma4AudioConfig(PreTrainedConfig):
    r"""
    subsampling_conv_channels (`list[int]`, defaults to `[128, 32]`):
        Channel sizes for the convolutional layers in the Sub-sample Convolution Projection.
    residual_weight (`float`, defaults to `0.5`):
        Scaling applied to hidden_states prior to combining with the residual in the feedforward.
    attention_chunk_size (`int`, defaults to `12`):
        The sub-sequence size for attention processing.
    attention_context_left (`int`, defaults to `13`):
        The leftward context size for the attention chunk.
    attention_context_right (`int`, defaults to `0`):
        The rightward context size for the attention chunk.
    attention_logit_cap (`float`, defaults to `50.0`):
        Cap applied to attention weights.
    attention_invalid_logits_value (`float`, defaults to `1e-9`):
        Value to use for invalid logits in attention.
    use_clipped_linears (`bool`, defaults to `True`):
        If true, apply clipping to the Linear layers, drawing bounds from the model checkpoint.
    gradient_clipping (`float`, defaults to `1e10`):
        Clipping value used to stabilize extremely large gradient values.
    output_proj_dims (`int`, defaults to `1536`):
        Dimension of the final linear projection from `hidden_size` to the model's output.
    """

    model_type = "gemma4_audio"

    hidden_size: int = 1024
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    hidden_act: str = "silu"

    # subsampling parameters
    subsampling_conv_channels: list[int] | tuple[int, int] = (128, 32)

    # conformer parameters
    conv_kernel_size: int = 5
    residual_weight: float = 0.5
    attention_chunk_size: int = 12
    attention_context_left: int = 13
    attention_context_right: int = 0
    attention_logit_cap: float = 50.0
    attention_invalid_logits_value: float = -1.0e9

    use_clipped_linears: bool = True
    rms_norm_eps: float = 1e-6
    gradient_clipping: float = 1e10
    output_proj_dims: int = 1536
    initializer_range: float = interval(min=0.0, max=1.0)(default=0.02)

    def __post_init__(self, **kwargs):
        # JSON serialization converts tuples to lists, convert back
        if isinstance(self.subsampling_conv_channels, tuple):
            self.subsampling_conv_channels = list(self.subsampling_conv_channels)
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="google/gemma-4-e2b-it")
@strict
class Gemma4TextConfig(PreTrainedConfig):
    r"""
    use_bidirectional_attention (`str`, *optional*):
        Controls bidirectional attention behavior. When set to `"vision"`, vision tokens
        attend bidirectionally while text tokens use causal attention. When set to `"all"`,
        all tokens use bidirectional attention.
    vocab_size_per_layer_input (`int`, defaults to 262144):
        Vocabulary size for the per-layer input embeddings (PLE). Used by models with
        per-layer residual streams where a smaller embedding is added at each decoder layer.
    hidden_size_per_layer_input (`int`, defaults to 256):
        Per-layer hidden dimension for the PLE system. The actual embedding weight has shape
        `[vocab_size_per_layer_input, num_hidden_layers * hidden_size_per_layer_input]`
        because all layers are packed into a single table. See the [Gemma4](https://huggingface.co/docs/transformers/main/en/model_doc/gemma4#per-layer-embeddings-ple) docs
        for a description of the full PLE pipeline.
    num_global_key_value_heads (`int`, *optional*):
        Number of key-value heads for global (full) attention layers. If `None`, defaults
        to `num_key_value_heads`.
    global_head_dim (`int`, defaults to 512):
        Dimension of each attention head in global (full) attention layers.
    attention_k_eq_v (`bool`, defaults to `False`):
        Whether keys and values share the same projection weights. When `True`, the key
        projection output is reused as the value projection.
    num_kv_shared_layers (`int`, defaults to 0):
        Number of consecutive decoder layers that share the same key-value projections.
        A value of 0 means no sharing (each layer has independent KV projections).
    enable_moe_block (`bool`, defaults to `False`):
        Whether to enable Mixture-of-Experts (MoE) blocks in the decoder layers. When
        `True`, eligible layers will use a sparse MoE feed-forward network.
    use_double_wide_mlp (`bool`, defaults to `False`):
        Whether to use a double-width MLP with fused gate and up projections.
    top_k_experts (`int`, *optional*):
        Number of experts activated per token in MoE layers. Only used when
        `enable_moe_block=True`.
    moe_intermediate_size (`int`, *optional*):
        Intermediate (hidden) size of each expert's feed-forward network in MoE layers.
        Only used when `enable_moe_block=True`.
    """

    model_type = "gemma4_text"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
        "layers.*.experts.gate_up_proj": "packed_colwise",
        "layers.*.experts.down_proj": "rowwise",
        "layers.*.experts": "moe_tp_experts",
    }
    base_model_ep_plan = {
        # EP plan for google/gemma-4-26B-A4B-it: do not tp in attention (num_global_key_value_heads=2 too small to partition)
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
        "layers.*.router": "ep_router",
        "layers.*.experts.gate_up_proj": "grouped_gemm",
        "layers.*.experts.down_proj": "grouped_gemm",
        "layers.*.experts": "moe_tp_experts",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    vocab_size: int = 262_144
    hidden_size: int = 2304
    intermediate_size: int = 9216
    num_hidden_layers: int = 30
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 256
    hidden_activation: str = "gelu_pytorch_tanh"
    max_position_embeddings: int = 131_072
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    bos_token_id: int | None = 2
    tie_word_embeddings: bool = True
    rope_parameters: dict | None = None
    attention_bias: bool = False
    attention_dropout: int | float | None = 0.0
    sliding_window: int = 512
    layer_types: list[str] | None = None
    final_logit_softcapping: float | None = None
    use_bidirectional_attention: Literal["all", "vision"] | None = None
    vocab_size_per_layer_input: int = 262_144
    hidden_size_per_layer_input: int = 256
    num_global_key_value_heads: int | None = None
    global_head_dim: int = 512
    attention_k_eq_v: bool = False
    num_kv_shared_layers: int = 0
    enable_moe_block: bool = False
    use_double_wide_mlp: bool = False
    num_experts: int | None = None
    top_k_experts: int | None = None
    moe_intermediate_size: int | None = None

    def __post_init__(self, **kwargs):
        if self.use_bidirectional_attention == "all":
            self.sliding_window = (self.sliding_window // 2) + 1  # due to fa we set exclusive bounds

        if self.layer_types is None:
            sliding_window_pattern = 6  # by default 5:1
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % sliding_window_pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        if self.layer_types and (last_layer_type := self.layer_types[-1]) != "full_attention":
            logger.warning(
                f"Last layer must use `full_attention`, but got `{last_layer_type}`. Forcing last layer to `full_attention`."
            )
            self.layer_types[-1] = "full_attention"

        default_rope_params: dict[Literal["full_attention", "sliding_attention"] : dict[str, Any]] = {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
            "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1_000_000.0},
        }
        if self.rope_parameters is None:
            self.rope_parameters = default_rope_params

        super().__post_init__(**kwargs)

    def convert_rope_params_to_dict(self, **kwargs):
        # No need to handle BC for new models, because they have no old-format `rope_scaling`
        return kwargs


@auto_docstring(checkpoint="google/gemma-4-e2b-it")
@strict
class Gemma4VisionConfig(PreTrainedConfig):
    r"""
    pooling_kernel_size (`int`, *optional*):
        Spatial pooling kernel size applied after patchification.
    position_embedding_size (`int`, defaults to 10240):
        Maximum number of position embeddings for the vision encoder. Controls the size of
        the learned 2D position embedding table used by the patch embedder.
    use_clipped_linears (`bool`, defaults to `False`):
        Whether to use weight-clipped linear layers. When enabled, linear layer weights are
        clamped to a fixed range during the forward pass to improve numerical stability.
    standardize (`bool`, defaults to `False`):
        If true, applies a bias and scale to the soft tokens returned from the pooler.
    """

    model_type = "gemma4_vision"
    base_model_tp_plan = {
        "encoder.layers.*.self_attn.q_proj": "colwise",
        "encoder.layers.*.self_attn.k_proj": "colwise",
        "encoder.layers.*.self_attn.v_proj": "colwise",
        "encoder.layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "encoder.layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
        "encoder.layers.*.self_attn.o_proj": "rowwise",
        "encoder.layers.*.mlp.gate_proj": "colwise",
        "encoder.layers.*.mlp.up_proj": "colwise",
        "encoder.layers.*.mlp.down_proj": "rowwise",
    }
    default_theta = 100.0

    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 16
    num_attention_heads: int = 12
    num_key_value_heads: int = 12
    head_dim: int = 64
    hidden_activation: str = "gelu_pytorch_tanh"
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 131_072
    attention_bias: bool | None = False
    attention_dropout: float | None = 0.0
    rope_parameters: dict | None = None
    pooling_kernel_size: int = 3
    patch_size: int = 16
    position_embedding_size: int = 10 * 1024
    use_clipped_linears: bool = False
    standardize: bool = False
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if self.rope_parameters is None:
            self.rope_parameters = {"rope_type": "default", "rope_theta": 100.0}

        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="google/gemma-4-e2b-it")
@strict
class Gemma4Config(PreTrainedConfig):
    r"""
    boi_token_id (`int`, *optional*, defaults to 255999):
        The begin-of-image token index to wrap the image prompt.
    eoi_token_id (`int`, *optional*, defaults to 258882):
        The end-of-image token index to wrap the image prompt.
    boa_token_id (`int`, *optional*, defaults to 256000):
        The begin-of-audio token index to wrap the audio prompt.
    eoa_token_index (`int`, *optional*, defaults to 258883):
        The end-of-audio token index to wrap the audio prompt.

    Example:

    ```python
    >>> from transformers import (
    >>>     Gemma4AudioConfig,
    >>>     Gemma4Config,
    >>>     Gemma4ForConditionalGeneration,
    >>>     Gemma4TextConfig,
    >>>     Gemma4VisionConfig,
    >>> )

    >>> # Initializing a Gemma 4 Audio config.
    >>> audio_config = Gemma4AudioConfig()

    >>> # Initializing a Gemma 4 Text config.
    >>> text_config = Gemma4TextConfig()

    >>> # Initializing a Gemma 4 vision config.
    >>> vision_config = Gemma4VisionConfig()

    >>> # Initializing a Gemma 4 config similar to google/gemma-4-e2b-it
    >>> configuration = Gemma4Config(text_config, vision_config, audio_config)

    >>> # Initializing a model from the google/gemma-4-e2b-it configuration
    >>> model = Gemma4ForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "gemma4"
    sub_configs = {
        "text_config": Gemma4TextConfig,
        "vision_config": Gemma4VisionConfig,
        "audio_config": Gemma4AudioConfig,
    }

    text_config: Gemma4TextConfig | dict[str, Any] | None = None
    vision_config: Gemma4VisionConfig | dict[str, Any] | None = None
    audio_config: Gemma4AudioConfig | dict[str, Any] | None = None
    boi_token_id: int | None = 255_999
    eoi_token_id: int | None = 258_882
    image_token_id: int | None = 258_880
    video_token_id: int | None = 258_884
    boa_token_id: int | None = 256_000
    eoa_token_index: int | None = 258_883
    audio_token_id: int | None = 258_881
    initializer_range: float | None = 0.02
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            self.text_config = Gemma4TextConfig()
            logger.info("text_config is None. Using default Gemma4TextConfig.")
        elif isinstance(self.text_config, dict):
            self.text_config = Gemma4TextConfig(**self.text_config)

        if self.vision_config is None:
            logger.info("vision_config is None. Gemma4Model.vision_tower will not be initialized.")
        if isinstance(self.vision_config, dict):
            self.vision_config = Gemma4VisionConfig(**self.vision_config)

        if self.audio_config is None:
            logger.info("audio_config is None. Gemma4Model.audio_tower will not be initialized.")
        if isinstance(self.audio_config, dict):
            self.audio_config = Gemma4AudioConfig(**self.audio_config)

        super().__post_init__(**kwargs)


__all__ = ["Gemma4AudioConfig", "Gemma4Config", "Gemma4TextConfig", "Gemma4VisionConfig"]
