# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from typing import Callable, Optional, Union

import torch
import torch.nn as nn

from transformers.utils.generic import OutputRecorder, check_model_inputs

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, EncoderDecoderCache
from ...configuration_utils import PretrainedConfig
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_attention_mask_for_sdpa
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPast,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from ...modeling_rope_utils import rope_config_validation
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.deprecation import deprecate_kwarg
from ..glm.modeling_glm import GlmAttention, GlmRotaryEmbedding, apply_rotary_pos_emb
from ..llama.modeling_llama import LlamaDecoderLayer, LlamaModel, eager_attention_forward
from ..whisper.modeling_whisper import WhisperModel, shift_tokens_right


logger = logging.get_logger(__name__)


class MoonshineConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MoonshineModel`]. It is used to instantiate a Moonshine
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Moonshine
    [UsefulSensors/moonshine-tiny](https://huggingface.co/UsefulSensors/moonshine-tiny).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 32768):
            Vocabulary size of the Moonshine model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`MoonshineModel`].
        hidden_size (`int`, *optional*, defaults to 288):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 1152):
            Dimension of the MLP representations.
        encoder_num_hidden_layers (`int`, *optional*, defaults to 6):
            Number of hidden layers in the Transformer encoder.
        decoder_num_hidden_layers (`int`, *optional*, defaults to 6):
            Number of hidden layers in the Transformer decoder.
        encoder_num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer decoder.
        encoder_num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `encoder_num_key_value_heads=encoder_num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `encoder_num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to
            `num_attention_heads`.
        decoder_num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `decoder_num_key_value_heads=decoder_num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `decoder_num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to
            `decoder_num_attention_heads`.
        pad_head_dim_to_multiple_of (`int`, *optional*):
            Pad head dimension in encoder and decoder to the next multiple of this value. Necessary for using certain
            optimized attention implementations.
        encoder_hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder.
        decoder_hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        decoder_start_token_id (`int`, *optional*, defaults to 1):
            Corresponds to the "<|startoftranscript|>" token, which is automatically used when no `decoder_input_ids`
            are provided to the `generate` function. It is used to guide the model`s generation process depending on
            the task.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
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
        partial_rotary_factor (`float`, *optional*, defaults to 0.9):
            Percentage of the query and keys which will have rotary embedding.
        is_encoder_decoder (`bool`, *optional*, defaults to `True`):
            Whether the model is used as an encoder/decoder or not.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        bos_token_id (`int`, *optional*, defaults to 1):
            Denotes beginning of sequences token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            Denotes end of sequences token id.

    Example:

    ```python
    >>> from transformers import MoonshineModel, MoonshineConfig

    >>> # Initializing a Moonshine style configuration
    >>> configuration = MoonshineConfig().from_pretrained("UsefulSensors/moonshine-tiny")

    >>> # Initializing a model from the configuration
    >>> model = MoonshineModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "moonshine"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_key_value_heads": "encoder_num_key_value_heads",
        "num_attention_heads": "encoder_num_attention_heads",
        "num_hidden_layers": "encoder_num_hidden_layers",
    }

    def __init__(
        self,
        vocab_size=32768,
        hidden_size=288,
        intermediate_size=1152,
        encoder_num_hidden_layers=6,
        decoder_num_hidden_layers=6,
        encoder_num_attention_heads=8,
        decoder_num_attention_heads=8,
        encoder_num_key_value_heads=None,
        decoder_num_key_value_heads=None,
        pad_head_dim_to_multiple_of=None,
        encoder_hidden_act="gelu",
        decoder_hidden_act="silu",
        max_position_embeddings=512,
        initializer_range=0.02,
        decoder_start_token_id=1,
        use_cache=True,
        rope_theta=10000.0,
        rope_scaling=None,
        partial_rotary_factor=0.9,
        is_encoder_decoder=True,
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.encoder_num_hidden_layers = encoder_num_hidden_layers
        self.decoder_num_hidden_layers = decoder_num_hidden_layers
        self.encoder_num_attention_heads = encoder_num_attention_heads
        self.decoder_num_attention_heads = decoder_num_attention_heads

        if encoder_num_key_value_heads is None:
            encoder_num_key_value_heads = encoder_num_attention_heads
        self.encoder_num_key_value_heads = encoder_num_key_value_heads

        if decoder_num_key_value_heads is None:
            decoder_num_key_value_heads = decoder_num_attention_heads
        self.decoder_num_key_value_heads = decoder_num_key_value_heads

        self.pad_head_dim_to_multiple_of = pad_head_dim_to_multiple_of

        self.encoder_hidden_act = encoder_hidden_act
        self.decoder_hidden_act = decoder_hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.decoder_start_token_id = decoder_start_token_id
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.partial_rotary_factor = partial_rotary_factor
        self.is_encoder_decoder = is_encoder_decoder
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        # Validate the correctness of rotary position embeddings parameters
        rope_config_validation(self)

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            **kwargs,
        )


class MoonshineEncoderMLP(nn.Module):
    def __init__(self, config, hidden_act):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class MoonshineDecoderMLP(nn.Module):
    def __init__(self, config, hidden_act):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size * 2)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states, gate = hidden_states.chunk(2, dim=-1)
        hidden_states = self.activation_fn(gate) * hidden_states
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class MoonshineAttention(GlmAttention):
    def __init__(
        self,
        config: MoonshineConfig,
        layer_idx: int,
        is_causal: bool,
        num_attention_heads: int,
        num_key_value_heads: int,
    ):
        config.update({"num_attention_heads": num_attention_heads, "num_key_value_heads": num_key_value_heads})
        super().__init__(config, layer_idx)
        self.is_causal = is_causal
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        # Pad head dimension to the next specified multiple.
        if self.config.pad_head_dim_to_multiple_of is not None:
            target_multiple = self.config.pad_head_dim_to_multiple_of
            target_head_dim = target_multiple * ((self.head_dim + target_multiple - 1) // target_multiple)
            self.head_dim_padding = target_head_dim - self.head_dim
        else:
            self.head_dim_padding = 0

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        key_value_states: Optional[torch.Tensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        bsz, q_len = hidden_states.shape[:-1]

        query_states = (
            self.q_proj(hidden_states).view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
        )

        is_cross_attention = key_value_states is not None
        if past_key_values is not None:
            is_updated = past_key_values.is_updated.get(self.layer_idx)
            if is_cross_attention:
                # after the first generated id, we can subsequently re-use all key/value_states from cache
                past_key_values.is_updated[self.layer_idx] = True
                past_key_values = past_key_values.cross_attention_cache
            else:
                past_key_values = past_key_values.self_attention_cache

        # use key_value_states if cross attention
        current_states = key_value_states if key_value_states is not None else hidden_states
        if is_cross_attention and past_key_values and is_updated:
            key_states = past_key_values.layers[self.layer_idx].keys
            value_states = past_key_values.layers[self.layer_idx].values
        else:
            key_states = (
                self.k_proj(current_states)
                .view(bsz, -1, self.config.num_key_value_heads, self.head_dim)
                .transpose(1, 2)
            )
            value_states = (
                self.v_proj(current_states)
                .view(bsz, -1, self.config.num_key_value_heads, self.head_dim)
                .transpose(1, 2)
            )
            if is_cross_attention and past_key_values is not None:
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )

        if not is_cross_attention:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if past_key_values is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        is_causal = self.is_causal and attention_mask is None and q_len > 1

        if self.head_dim_padding > 0:
            query_states = torch.nn.functional.pad(query_states, (0, self.head_dim_padding))
            key_states = torch.nn.functional.pad(key_states, (0, self.head_dim_padding))
            value_states = torch.nn.functional.pad(value_states, (0, self.head_dim_padding))

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            is_causal=is_causal,
            **kwargs,
        )

        if self.head_dim_padding > 0:
            attn_output = attn_output[..., : -self.head_dim_padding]

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class MoonshineRotaryEmbedding(GlmRotaryEmbedding):
    pass


class MoonshineEncoderLayer(LlamaDecoderLayer):
    def __init__(self, config: MoonshineConfig, layer_idx: int):
        super().__init__(config, layer_idx)

        self.self_attn = MoonshineAttention(
            config=config,
            layer_idx=layer_idx,
            is_causal=False,
            num_attention_heads=config.encoder_num_attention_heads,
            num_key_value_heads=config.encoder_num_key_value_heads,
        )

        self.mlp = MoonshineEncoderMLP(config, config.encoder_hidden_act)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, bias=False)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, bias=False)


class MoonshineDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: MoonshineConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = MoonshineAttention(
            config=config,
            layer_idx=layer_idx,
            is_causal=True,
            num_attention_heads=config.decoder_num_attention_heads,
            num_key_value_heads=config.decoder_num_key_value_heads,
        )
        self.encoder_attn = MoonshineAttention(
            config=config,
            layer_idx=layer_idx,
            is_causal=False,
            num_attention_heads=config.decoder_num_attention_heads,
            num_key_value_heads=config.decoder_num_key_value_heads,
        )

        self.mlp = MoonshineDecoderMLP(config, config.decoder_hidden_act)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, bias=False)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, bias=False)
        self.final_layernorm = nn.LayerNorm(config.hidden_size, bias=False)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        encoder_position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        encoder_position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states, _ = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class MoonshinePreTrainedModel(PreTrainedModel):
    config: MoonshineConfig
    base_model_prefix = "model"
    main_input_name = "input_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MoonshineEncoderLayer", "MoonshineDecoderLayer"]
    _supports_flash_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = True
    # TODO arthur, how do we separate when it cross / self coming from different layer?

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """
        output_conv1_length = int((input_lengths - 127) / 64 + 1)
        output_conv2_length = int((output_conv1_length - 7) / 3 + 1)
        output_conv3_length = int((output_conv2_length - 3) / 2 + 1)

        return output_conv3_length


class MoonshineEncoder(MoonshinePreTrainedModel):
    """
    Transformer encoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MoonshineEncoderLayer`]

    Args:
        config: MoonshineConfig
    """

    main_input_name = "input_values"
    _can_record_outputs = {
        "attentions": MoonshineAttention,
        "hidden_states": MoonshineEncoderLayer,
    }

    def __init__(self, config: MoonshineConfig):
        super().__init__(config)
        self.config = config
        embed_dim = config.hidden_size

        self.conv1 = nn.Conv1d(1, embed_dim, kernel_size=127, stride=64, bias=False)
        self.conv2 = nn.Conv1d(embed_dim, 2 * embed_dim, kernel_size=7, stride=3)
        self.conv3 = nn.Conv1d(2 * embed_dim, embed_dim, kernel_size=3, stride=2)
        self.groupnorm = nn.GroupNorm(num_groups=1, num_channels=embed_dim, eps=1e-5)
        self.rotary_emb = MoonshineRotaryEmbedding(config=config)

        self.layers = nn.ModuleList(
            [MoonshineEncoderLayer(config, idx) for idx in range(config.encoder_num_hidden_layers)]
        )
        self.layer_norm = nn.LayerNorm(embed_dim, bias=False)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    @check_model_inputs
    def forward(
        self,
        input_values: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        Args:
            input_values (`torch.FloatTensor` of shape `(batch_size, audio_length)`):
                Float values of the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `list[float]`, a
                `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library (`pip install torchcodec`) or
                the soundfile library (`pip install soundfile`). To prepare the array into
                `input_values`, the [`AutoFeatureExtractor`] should be used for padding
                and conversion into a tensor of type `torch.FloatTensor`.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding indices in `input_values`. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
        """
        input_values = input_values.unsqueeze(1)
        hidden_states = nn.functional.tanh(self.conv1(input_values))
        hidden_states = self.groupnorm(hidden_states)
        hidden_states = nn.functional.gelu(self.conv2(hidden_states))
        hidden_states = nn.functional.gelu(self.conv3(hidden_states))
        hidden_states = hidden_states.permute(0, 2, 1)

        # attention mask downsampling
        if attention_mask is not None:
            mask_len = self._get_feat_extract_output_lengths(attention_mask.shape[-1])
            downsample_stride = 64 * 3 * 2  # conv strides
            attention_mask = attention_mask[..., ::downsample_stride][..., :mask_len]
            if self.config._attn_implementation == "flash_attention_2":
                attention_mask = attention_mask if (attention_mask == 0.0).any() else None
            elif self.config._attn_implementation == "sdpa":
                attention_mask = _prepare_4d_attention_mask_for_sdpa(attention_mask, hidden_states.dtype)
            else:
                attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        position_ids = torch.arange(0, hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.layer_norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
        )


class MoonshineDecoder(LlamaModel):
    main_input_name = "input_ids"
    _can_record_outputs = {
        "attentions": OutputRecorder(MoonshineAttention, index=1, layer_name="self_attn"),
        "hidden_states": MoonshineDecoderLayer,
        "cross_attentions": OutputRecorder(MoonshineAttention, index=1, layer_name="encoder_attn"),
    }

    def __init__(self, config: MoonshineConfig):
        super().__init__(config)
        self.norm = nn.LayerNorm(config.hidden_size, bias=False)
        self.layers = nn.ModuleList(
            [MoonshineDecoderLayer(config, idx) for idx in range(config.decoder_num_hidden_layers)]
        )

    @check_model_inputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        r"""
        encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            of the decoder.
        encoder_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding indices in `encoder_hidden_states`. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = EncoderDecoderCache(DynamicCache(config=self.config), DynamicCache(config=self.config))

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        if encoder_attention_mask is not None:
            mask_len = encoder_hidden_states.shape[-2]
            downsample_stride = 64 * 3 * 2  # conv strides
            encoder_attention_mask = encoder_attention_mask[..., ::downsample_stride][..., :mask_len]
            if self.config._attn_implementation == "flash_attention_2":
                encoder_attention_mask = encoder_attention_mask if (encoder_attention_mask == 0.0).any() else None
            elif self.config._attn_implementation == "sdpa":
                encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    encoder_attention_mask, hidden_states.dtype, hidden_states.shape[-2]
                )
            else:
                encoder_attention_mask = _prepare_4d_attention_mask(
                    encoder_attention_mask, hidden_states.dtype, hidden_states.shape[-2]
                )

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                causal_mask,
                encoder_hidden_states,  # as a positional argument for gradient checkpointing
                encoder_attention_mask=encoder_attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class MoonshineModel(WhisperModel):
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Union[EncoderDecoderCache, tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[tuple[torch.FloatTensor]] = None,
        decoder_position_ids: Optional[tuple[torch.LongTensor]] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Seq2SeqModelOutput:
        r"""
        input_values (`torch.FloatTensor` of shape `(batch_size, audio_length)`):
            Float values of the raw speech waveform. Raw speech waveform can be
            obtained by loading a `.flac` or `.wav` audio file into an array of type `list[float]`, a
            `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library (`pip install torchcodec`) or
            the soundfile library (`pip install soundfile`). To prepare the array into
            `input_values`, the [`AutoFeatureExtractor`] should be used for padding
            and conversion into a tensor of type `torch.FloatTensor`.
        decoder_position_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Used to calculate the position embeddings up to `config.decoder_config.max_position_embeddings`

        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoFeatureExtractor, MoonshineModel
        >>> from datasets import load_dataset

        >>> model = MoonshineModel.from_pretrained("UsefulSensors/moonshine-tiny")
        >>> feature_extractor = AutoFeatureExtractor.from_pretrained("UsefulSensors/moonshine-tiny")
        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
        >>> input_values = inputs.input_values
        >>> decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
        >>> last_hidden_state = model(input_values, decoder_input_ids=decoder_input_ids).last_hidden_state
        >>> list(last_hidden_state.shape)
        [1, 2, 288]
        ```
        """
        if encoder_outputs is None:
            encoder_outputs: BaseModelOutput = self.encoder(input_values, attention_mask=attention_mask, **kwargs)

        decoder_outputs: BaseModelOutputWithPastAndCrossAttentions = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_attention_mask=attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            position_ids=decoder_position_ids,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    The Moonshine Model with a language modeling head. Can be used for automatic speech recognition.
    """
)
class MoonshineForConditionalGeneration(MoonshinePreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["proj_out.weight"]

    def __init__(self, config: MoonshineConfig):
        super().__init__(config)
        self.model = MoonshineModel(config)
        self.proj_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def get_output_embeddings(self):
        return self.proj_out

    def set_output_embeddings(self, new_embeddings):
        self.proj_out = new_embeddings

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Union[EncoderDecoderCache, tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[tuple[torch.FloatTensor]] = None,
        decoder_position_ids: Optional[tuple[torch.LongTensor]] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Seq2SeqLMOutput:
        r"""
        input_values (`torch.FloatTensor` of shape `(batch_size, audio_length)`):
            Float values of the raw speech waveform. Raw speech waveform can be
            obtained by loading a `.flac` or `.wav` audio file into an array of type `list[float]`, a
            `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library (`pip install torchcodec`) or
            the soundfile library (`pip install soundfile`). To prepare the array into
            `input_values`, the [`AutoFeatureExtractor`] should be used for padding
            and conversion into a tensor of type `torch.FloatTensor`.
        decoder_position_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Used to calculate the position embeddings up to `config.decoder_config.max_position_embeddings`

        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, MoonshineForConditionalGeneration
        >>> from datasets import load_dataset

        >>> processor = AutoProcessor.from_pretrained("UsefulSensors/moonshine-tiny")
        >>> model = MoonshineForConditionalGeneration.from_pretrained("UsefulSensors/moonshine-tiny")

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

        >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
        >>> input_values = inputs.input_values

        >>> generated_ids = model.generate(input_values, max_new_tokens=100)

        >>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        >>> transcription
        'Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
        ```"""

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs: Seq2SeqModelOutput = self.model(
            input_values,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_position_ids=decoder_position_ids,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        logits = self.proj_out(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


__all__ = [
    "MoonshineConfig",
    "MoonshineModel",
    "MoonshinePreTrainedModel",
    "MoonshineForConditionalGeneration",
]
