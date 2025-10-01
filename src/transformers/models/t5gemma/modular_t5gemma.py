# coding=utf-8
# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.
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
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn

from ...cache_utils import Cache, DynamicCache, EncoderDecoderCache
from ...configuration_utils import PretrainedConfig
from ...generation import GenerationMixin
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    logging,
)
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import OutputRecorder, check_model_inputs
from ..gemma2.configuration_gemma2 import Gemma2Config
from ..gemma2.modeling_gemma2 import (
    Gemma2Attention,
    Gemma2MLP,
    Gemma2PreTrainedModel,
    Gemma2RMSNorm,
    Gemma2RotaryEmbedding,
    create_causal_mask,
    create_sliding_window_causal_mask,
    eager_attention_forward,
)


_CHECKPOINT_FOR_DOC = "google/t5gemma-2b-2b-prefixlm-it"


logger = logging.get_logger(__name__)


class T5GemmaModuleConfig(Gemma2Config):
    pass


class T5GemmaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`T5GemmaModel`]. It is used to instantiate an T5Gemma
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to a hypothetical balanced Gemma2 encoder-decoder model.
    e.g. [google/t5gemma-2b-2b-prefixlm-it](https://huggingface.co/google/t5gemma-2b-2b-prefixlm-it)
    ```python
    >>> from transformers import T5GemmaConfig, T5GemmaModel
    >>> t5gemma_config = T5GemmaConfig.from_pretrained("google/t5gemma-2b-2b-prefixlm-it")
    >>> model = T5GemmaModel(t5gemma_config)
    ```
    Configuration objects inherit from [PretrainedConfig] and can be used to control the model outputs. Read the
    documentation from [PretrainedConfig] for more information.
    Args:
        encoder (`Union[T5GemmaModuleConfig, dict]`, optional, *optional*):
            Configuration for the encoder.
        decoder (`Union[T5GemmaModuleConfig, dict]`, optional, *optional*):
            Configuration for the decoder.
        is_encoder_decoder (bool, optional, *optional*, defaults to `True`):
            Whether the model is used as an encoder/decoder or not.
        dropout_rate (`float`, *optional*, defaults to 0.0):
            The ratio for all dropout layers (following T5).
        classifier_dropout_rate (`float`, *optional*, defaults to 0.0):
            The dropout ratio for classifier (following T5).
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for attention.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether tie input and output embeddings.
        vocab_size (`int`, *optional*, defaults to 256000):
            Vocabulary size of the T5Gemma model (the same as Gemma 2).
        kwargs (additional keyword arguments, optional, *optional*):
            Will be passed to the PretrainedConfig base class.
    """

    model_type = "t5gemma"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        # encoder
        "encoder.layers.*.self_attn.q_proj": "colwise",
        "encoder.layers.*.self_attn.k_proj": "colwise",
        "encoder.layers.*.self_attn.v_proj": "colwise",
        "encoder.layers.*.self_attn.o_proj": "rowwise",
        "encoder.layers.*.mlp.gate_proj": "colwise",
        "encoder.layers.*.mlp.up_proj": "colwise",
        "encoder.layers.*.mlp.down_proj": "rowwise",
        # decoder
        "decoder.layers.*.self_attn.q_proj": "colwise",
        "decoder.layers.*.self_attn.k_proj": "colwise",
        "decoder.layers.*.self_attn.v_proj": "colwise",
        "decoder.layers.*.self_attn.o_proj": "rowwise",
        "decoder.layers.*.cross_attn.q_proj": "colwise",
        "decoder.layers.*.cross_attn.k_proj": "colwise",
        "decoder.layers.*.cross_attn.v_proj": "colwise",
        "decoder.layers.*.cross_attn.o_proj": "rowwise",
        "decoder.layers.*.mlp.gate_proj": "colwise",
        "decoder.layers.*.mlp.up_proj": "colwise",
        "decoder.layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        # encoder
        "encoder.embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "encoder.layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "encoder.norm": (["hidden_states"], ["hidden_states"]),
        # decoder
        "decoder.embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "decoder.layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "decoder.norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        encoder: Optional[Union[T5GemmaModuleConfig, dict[Any, Any]]] = None,
        decoder: Optional[Union[T5GemmaModuleConfig, dict[Any, Any]]] = None,
        is_encoder_decoder: bool = True,
        dropout_rate: float = 0.0,
        classifier_dropout_rate: float = 0.0,
        attention_dropout: float = 0.0,
        tie_word_embeddings: bool = True,
        vocab_size: int = 256000,
        **kwargs,
    ):
        if isinstance(encoder, dict):
            encoder = T5GemmaModuleConfig(**encoder)
        elif encoder is None:
            encoder = T5GemmaModuleConfig()
        else:
            assert isinstance(encoder, T5GemmaModuleConfig), f"{type(encoder)} is not supported."

        if isinstance(decoder, dict):
            decoder = T5GemmaModuleConfig(**decoder)
        elif decoder is None:
            decoder = encoder
        else:
            assert isinstance(decoder, T5GemmaModuleConfig), f"{type(decoder)} is not supported."

        encoder = T5GemmaModuleConfig(**encoder.to_dict())
        decoder = T5GemmaModuleConfig(**decoder.to_dict())

        encoder.is_decoder = False
        encoder.dropout_rate = dropout_rate
        encoder.attention_dropout = attention_dropout
        self.encoder = encoder

        decoder.is_decoder = True
        decoder.use_cache = True
        decoder.dropout_rate = dropout_rate
        decoder.attention_dropout = attention_dropout
        decoder.cross_attention_hidden_size = encoder.hidden_size
        self.decoder = decoder

        for special_token_key in ["bos_token_id", "pad_token_id", "eos_token_id"]:
            if special_token_key not in kwargs:
                kwargs[special_token_key] = getattr(decoder, special_token_key)

        super().__init__(**kwargs)

        self.is_encoder_decoder = is_encoder_decoder
        self.use_cache = kwargs.get("use_cache", decoder.use_cache)
        self.initializer_range = kwargs.get("initializer_range", decoder.initializer_range)
        self.dropout_rate = dropout_rate
        self.attention_dropout = attention_dropout
        self.classifier_dropout_rate = classifier_dropout_rate
        self.tie_word_embeddings = tie_word_embeddings

        # Used in pipeline generation.
        self.vocab_size = vocab_size

    def __setattr__(self, key, value):
        shared_attr_with_submodules = [
            "output_hidden_states",
            "output_attentions",
            "_attn_implementation",
            "dropout_rate",
            "attention_dropout",
            "vocab_size",
        ]

        if key in shared_attr_with_submodules:
            setattr(self.encoder, key, value)
            setattr(self.decoder, key, value)
        super().__setattr__(key, value)


class T5GemmaRMSNorm(Gemma2RMSNorm):
    pass


class T5GemmaMLP(Gemma2MLP):
    def __init__(self, config):
        super().__init__(config)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        hidden_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        hidden_states = self.dropout(hidden_states)
        down_proj = self.down_proj(hidden_states)
        return down_proj


class T5GemmaRotaryEmbedding(Gemma2RotaryEmbedding):
    def __init__(self, config, device=None):
        super().__init__(config, device)


class T5GemmaSelfAttention(Gemma2Attention):
    def __init__(self, config: T5GemmaModuleConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        # Required by flash attention: encoder selfattention is non-causal
        self.is_causal = config.is_decoder


class T5GemmaCrossAttention(Gemma2Attention):
    def __init__(self, config: T5GemmaModuleConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        del self.sliding_window
        self.is_causal = False

        if config.cross_attention_hidden_size is None:
            raise ValueError("Cross-attention needs cross_attention_hidden_size to be specified.")

        self.k_proj = nn.Linear(
            config.cross_attention_hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.cross_attention_hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        encoder_hidden_states: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        if encoder_hidden_states is None:
            raise ValueError("Encoder hidden state is required for cross attention.")

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if past_key_values is not None:
            is_updated = past_key_values.is_updated.get(self.layer_idx)
            curr_past_key_value = past_key_values.cross_attention_cache

        if past_key_values is None or not is_updated:
            encoder_input_shape = encoder_hidden_states.shape[:-1]
            encoder_hidden_shape = (*encoder_input_shape, -1, self.head_dim)
            key_states = self.k_proj(encoder_hidden_states).view(encoder_hidden_shape).transpose(1, 2)
            value_states = self.v_proj(encoder_hidden_states).view(encoder_hidden_shape).transpose(1, 2)

            if past_key_values is not None:
                key_states, value_states = curr_past_key_value.update(key_states, value_states, self.layer_idx)
                past_key_values.is_updated[self.layer_idx] = True
        else:
            key_states = curr_past_key_value.layers[self.layer_idx].keys
            value_states = curr_past_key_value.layers[self.layer_idx].values

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=None,
            softcap=self.attn_logit_softcapping,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


def bidirectional_mask_function(attention_mask: Optional[torch.Tensor]) -> Callable:
    """
    This creates bidirectional attention mask.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        if attention_mask is None:
            return torch.ones((), dtype=torch.bool)
        return attention_mask[batch_idx, kv_idx].to(torch.bool)

    return inner_mask


def sliding_window_bidirectional_mask_function(sliding_window: int) -> Callable:
    """
    This creates bidirectional attention mask with sliding window.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return (q_idx - sliding_window < kv_idx) & (kv_idx < q_idx + sliding_window)

    return inner_mask


class T5GemmaEncoderLayer(GradientCheckpointingLayer):
    """Encoder sub-layer."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        self.layer_idx = layer_idx
        self.attention_type = config.layer_types[layer_idx]

        self.self_attn = T5GemmaSelfAttention(
            config=config,
            layer_idx=layer_idx,
        )
        self.pre_self_attn_layernorm = T5GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_self_attn_layernorm = T5GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.mlp = T5GemmaMLP(config)
        self.pre_feedforward_layernorm = T5GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = T5GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor,]:
        residual = hidden_states
        hidden_states = self.pre_self_attn_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            **kwargs,
        )
        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)
        return hidden_states


class T5GemmaDecoderLayer(T5GemmaEncoderLayer):
    """Decoder sub-layer: an extra cross-attention layer."""

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.cross_attn = T5GemmaCrossAttention(config=config, layer_idx=layer_idx)
        self.pre_cross_attn_layernorm = T5GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_cross_attn_layernorm = T5GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states
        hidden_states = self.pre_self_attn_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values.self_attention_cache if past_key_values is not None else None,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)

        residual = hidden_states
        hidden_states = self.pre_cross_attn_layernorm(hidden_states)
        hidden_states, _ = self.cross_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = self.post_cross_attn_layernorm(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)
        return hidden_states


class T5GemmaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_size: int, num_labels: int, classifier_dropout_rate: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=classifier_dropout_rate)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class T5GemmaLMHead(nn.Module):
    """Head for language modeling (generation) tasks."""

    def __init__(self, hidden_size: int, vocab_size: int, bias: bool = False):
        super().__init__()
        self.out_proj = nn.Linear(hidden_size, vocab_size, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.out_proj(hidden_states)
        return logits


@auto_docstring
class T5GemmaPreTrainedModel(Gemma2PreTrainedModel):
    config: T5GemmaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["T5GemmaEncoderLayer", "T5GemmaDecoderLayer"]

    def _init_weights(self, module):
        # TODO: support initialization for encoders and decoders separately(?)
        PreTrainedModel._init_weights(self, module)
        std = self.config.initializer_range
        if isinstance(module, T5GemmaClassificationHead):
            scale = module.out_proj.weight.shape[0] ** -0.5
            module.out_proj.weight.data.normal_(mean=0.0, std=std * scale)
            if hasattr(module.out_proj, "bias") and module.out_proj.bias is not None:
                module.out_proj.bias.data.zero_()
        elif isinstance(module, T5GemmaLMHead):
            if not self.config.tie_word_embeddings:
                scale = module.out_proj.weight.shape[0] ** -0.5
                module.out_proj.weight.data.normal_(mean=0.0, std=std * scale)
        # We initialize with 0s to be 1 centered as the RMSNorm here does (1 + weight)
        elif "RMSNorm" in module.__class__.__name__:
            module.weight.data.zero_()

    def _shift_right(self, input_ids):
        """
        Shifts input_ids to the right, prepends the decoder_start_token_id, and handles
        pad_token_id replacement for labels that were -100.
        This is a common preparation step for decoder inputs in sequence-to-sequence models.
        """
        decoder_start_token_id = self.config.decoder.bos_token_id
        pad_token_id = self.config.decoder.pad_token_id

        if decoder_start_token_id is None:
            raise ValueError("self.model.config.decoder.bos_token_id has to be defined. ")

        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.decoder.pad_token_id has to be defined.")

        # Is this T5 specific?
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids


def make_default_2d_attention_mask(
    token_ids: Optional[torch.LongTensor],
    hidden_states: torch.Tensor,
    pad_token_id: Optional[int],
) -> torch.Tensor:
    """Construct the default attention mask."""
    if token_ids is not None:
        if pad_token_id is None:
            raise ValueError("`pad_token_id` is required for padding information.")
        attention_mask = (token_ids != pad_token_id).to(hidden_states.device, torch.long)
    else:
        attention_mask = torch.ones(
            (hidden_states.shape[0], hidden_states.shape[1]), device=hidden_states.device, dtype=torch.long
        )
    return attention_mask


class T5GemmaEncoder(T5GemmaPreTrainedModel):
    _can_record_outputs = {
        "attentions": T5GemmaSelfAttention,
        "hidden_states": T5GemmaEncoderLayer,
    }

    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.norm = T5GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = T5GemmaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.layers = nn.ModuleList(
            [T5GemmaEncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # As we want to pass `past_key_values=None` explicitly everywhere, we need to pop them from kwargs if present
        kwargs.pop("past_key_values", None)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if attention_mask is None:
            attention_mask = make_default_2d_attention_mask(input_ids, inputs_embeds, self.config.pad_token_id)

        if not isinstance(self_attn_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": None,
                "position_ids": position_ids,
            }
            self_attn_mask_mapping = {
                "full_attention": create_causal_mask(
                    **mask_kwargs,
                    or_mask_function=bidirectional_mask_function(attention_mask),
                ),
                "sliding_attention": create_sliding_window_causal_mask(
                    **mask_kwargs,
                    or_mask_function=sliding_window_bidirectional_mask_function(self.config.sliding_window),
                    and_mask_function=bidirectional_mask_function(attention_mask),
                ),
            }

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer
        hidden_states = self.dropout(hidden_states)

        for layer_module in self.layers[: self.config.num_hidden_layers]:
            hidden_states = layer_module(
                hidden_states,
                position_embeddings,
                self_attn_mask_mapping[layer_module.attention_type],
                position_ids,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
        )


class T5GemmaDecoder(T5GemmaEncoder):
    _can_record_outputs = {
        "attentions": OutputRecorder(T5GemmaSelfAttention, index=1),
        "cross_attentions": OutputRecorder(T5GemmaCrossAttention, index=1),
        "hidden_states": T5GemmaDecoderLayer,
    }

    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [T5GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.post_init()

    @check_model_inputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if encoder_hidden_states is None:
            raise ValueError("`encoder_hidden_states` must be given in decoder")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if not self.training and use_cache and past_key_values is None:
            past_key_values = EncoderDecoderCache(DynamicCache(config=self.config), DynamicCache(config=self.config))
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if attention_mask is None and past_key_values is None:
            attention_mask = make_default_2d_attention_mask(input_ids, inputs_embeds, self.config.pad_token_id)

        if not isinstance(self_attn_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values.self_attention_cache if past_key_values is not None else None,
                "position_ids": position_ids,
            }
            self_attn_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        if not isinstance(cross_attn_mask_mapping := encoder_attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": encoder_hidden_states,
                "attention_mask": encoder_attention_mask,
                "cache_position": cache_position,
                "past_key_values": None,
                "position_ids": None,
            }
            cross_attn_mask_mapping = {
                "full_attention": create_causal_mask(
                    **mask_kwargs,
                    or_mask_function=bidirectional_mask_function(encoder_attention_mask),
                ),
            }

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer
        hidden_states = self.dropout(hidden_states)

        for layer_module in self.layers[: self.config.num_hidden_layers]:
            hidden_states = layer_module(
                hidden_states,
                position_embeddings,
                self_attn_mask_mapping[layer_module.attention_type],
                position_ids,
                past_key_values,
                use_cache,
                cache_position,
                encoder_hidden_states,
                cross_attn_mask_mapping["full_attention"],
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring
class T5GemmaModel(T5GemmaPreTrainedModel):
    def __init__(self, config: T5GemmaConfig):
        super().__init__(config)

        if not config.is_encoder_decoder:
            raise ValueError("T5GemmaModel only support encoder-decoder modeling. Use `T5GemmaEncoderModel` instead.")

        self.encoder = T5GemmaEncoder(config.encoder)
        self.decoder = T5GemmaDecoder(config.decoder)

        self.post_init()

    def get_encoder(self):
        return self.encoder

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.encoder.set_input_embeddings(new_embeddings)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        decoder_position_ids: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[BaseModelOutput] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Seq2SeqModelOutput:
        r"""
        decoder_position_ids (`torch.LongTensor` of shape `(batch_size, decoder_sequence_length)`, *optional*):
            Indices of positions of each decoder input sequence tokens in the position embeddings. Selected in the range `[0,
            config.decoder.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
        """
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )

        encoder_hidden_states = encoder_outputs.last_hidden_state

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            position_ids=decoder_position_ids,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states
            if kwargs.get("output_hidden_states", False)
            else (decoder_outputs.last_hidden_state,),
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


@auto_docstring
class T5GemmaEncoderModel(T5GemmaPreTrainedModel):
    def __init__(self, config: T5GemmaConfig):
        super().__init__(config)

        if config.is_encoder_decoder:
            raise ValueError("T5GemmaEncoderModel only supports encoder-only model. Use `T5GemmaModel` instead.")

        self.encoder = T5GemmaEncoder(config.encoder)
        self.post_init()

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.encoder.set_input_embeddings(new_embeddings)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        return encoder_outputs


class T5GemmaForConditionalGeneration(T5GemmaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["model.decoder.embed_tokens.weight", "lm_head.out_proj.weight"]
    _tp_plan = {"lm_head.out_proj": "colwise_rep"}
    _pp_plan = {"lm_head.out_proj": (["hidden_states"], ["logits"])}

    def __init__(self, config: T5GemmaConfig):
        config.is_encoder_decoder = True
        super().__init__(config)

        self.model = T5GemmaModel(config)
        self.vocab_size = config.decoder.vocab_size
        self.lm_head = T5GemmaLMHead(config.decoder.hidden_size, self.vocab_size)
        self.loss_type = "ForMaskedLM"

        self.post_init()

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.out_proj = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head.out_proj

    def _tie_weights(self):
        # Decoder input and output embeddings are tied.
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.lm_head.out_proj, self.get_decoder().get_input_embeddings())

    def get_encoder(self):
        return self.model.encoder

    def get_decoder(self):
        return self.model.decoder

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        decoder_position_ids: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[BaseModelOutput] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        decoder_position_ids (`torch.LongTensor` of shape `(batch_size, decoder_sequence_length)`, *optional*):
            Indices of positions of each decoder input sequence tokens in the position embeddings. Selected in the range `[0,
            config.decoder.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        decoder_outputs: Seq2SeqModelOutput = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = decoder_outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        decoder_config = self.get_decoder().config
        if decoder_config.final_logit_softcapping is not None:
            logits = logits / decoder_config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * decoder_config.final_logit_softcapping

        loss = None
        if labels is not None:
            # Input has right-shifted so we directly perform masked lm loss
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.decoder_hidden_states,
            decoder_attentions=decoder_outputs.decoder_attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=decoder_outputs.encoder_last_hidden_state,
            encoder_hidden_states=decoder_outputs.encoder_hidden_states,
            encoder_attentions=decoder_outputs.encoder_attentions,
        )

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)


@auto_docstring
class T5GemmaForSequenceClassification(T5GemmaPreTrainedModel):
    def __init__(self, config: T5GemmaConfig, is_encoder_decoder: Optional[bool] = None):
        r"""
        is_encoder_decoder (`Optional`, *optional*):
            Whether use encoder_decoder for sequence classification. When set to False, only encoder is used.
        """
        if is_encoder_decoder is not None:
            config.is_encoder_decoder = is_encoder_decoder
        super().__init__(config)
        self.num_labels = config.num_labels

        if config.is_encoder_decoder:
            self.model = T5GemmaModel(config)
        else:
            self.model = T5GemmaEncoderModel(config)

        hidden_size = config.encoder.hidden_size
        if config.is_encoder_decoder:
            hidden_size = config.decoder.hidden_size

        classifier_dropout = getattr(config, "classifier_dropout_rate", 0.1)
        self.score = T5GemmaClassificationHead(hidden_size, self.num_labels, classifier_dropout)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        decoder_position_ids: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[BaseModelOutput] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> SequenceClassifierOutput:
        r"""
        decoder_position_ids (`torch.LongTensor` of shape `(batch_size, decoder_sequence_length)`, *optional*):
            Indices of positions of each decoder input sequence tokens in the position embeddings. Selected in the range `[0,
            config.decoder.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        if self.config.is_encoder_decoder and (input_ids is None and inputs_embeds is not None):
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__} in encoder-decoder mode."
            )

        # Following T5, we automatically creates decoder_input_ids from input_ids if no decoder_input_ids are provided
        if self.config.is_encoder_decoder and (decoder_input_ids is None and decoder_inputs_embeds is None):
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )
            decoder_input_ids = self._shift_right(input_ids)

        if self.config.is_encoder_decoder:
            outputs: Seq2SeqModelOutput = self.model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                decoder_position_ids=decoder_position_ids,
                encoder_outputs=encoder_outputs,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                use_cache=False,
                **kwargs,
            )
            last_hidden_state = outputs.last_hidden_state
            hidden_states = outputs.decoder_hidden_states
            attentions = outputs.decoder_attentions
        else:
            outputs: BaseModelOutput = self.model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )
            last_hidden_state = outputs.last_hidden_state
            hidden_states = outputs.hidden_states
            attentions = outputs.attentions

        logits = self.score(last_hidden_state)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
            non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)

            if self.config.is_encoder_decoder:
                last_non_pad_token += 1  # due to the right shift.
                last_non_pad_token = torch.clamp(last_non_pad_token, max=decoder_input_ids.shape[-1] - 1)
        else:
            last_non_pad_token = -1
            logger.warning_once(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)

        return SequenceClassifierOutput(
            loss=loss,
            logits=pooled_logits,
            hidden_states=hidden_states,
            attentions=attentions,
        )


@auto_docstring
class T5GemmaForTokenClassification(T5GemmaPreTrainedModel):
    def __init__(self, config: T5GemmaConfig, is_encoder_decoder: Optional[bool] = None):
        r"""
        is_encoder_decoder (`Optional`, *optional*):
            Whether use encoder_decoder for token classification. When set to False, only encoder is used.
        """
        if is_encoder_decoder is not None:
            config.is_encoder_decoder = is_encoder_decoder
        super().__init__(config)
        self.num_labels = config.num_labels

        if config.is_encoder_decoder:
            self.model = T5GemmaModel(config)
        else:
            self.model = T5GemmaEncoderModel(config)

        hidden_size = config.encoder.hidden_size
        if config.is_encoder_decoder:
            hidden_size = config.decoder.hidden_size

        classifier_dropout = getattr(config, "classifier_dropout_rate", 0.1)
        self.score = T5GemmaClassificationHead(hidden_size, self.num_labels, classifier_dropout)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        decoder_position_ids: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[BaseModelOutput] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> TokenClassifierOutput:
        r"""
        decoder_position_ids (`torch.LongTensor` of shape `(batch_size, decoder_sequence_length)`, *optional*):
            Indices of positions of each decoder input sequence tokens in the position embeddings. Selected in the range `[0,
            config.decoder.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        if self.config.is_encoder_decoder and (input_ids is None and inputs_embeds is not None):
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__} in encoder-decoder mode."
            )

        if self.config.is_encoder_decoder and (decoder_input_ids is None and decoder_inputs_embeds is None):
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )
            decoder_input_ids = self._shift_right(input_ids)

        if self.config.is_encoder_decoder:
            outputs: Seq2SeqModelOutput = self.model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                decoder_position_ids=decoder_position_ids,
                encoder_outputs=encoder_outputs,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                use_cache=False,
                **kwargs,
            )
            last_hidden_state = outputs.last_hidden_state
            hidden_states = outputs.decoder_hidden_states
            attentions = outputs.decoder_attentions
        else:
            outputs: BaseModelOutput = self.model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )
            last_hidden_state = outputs.last_hidden_state
            hidden_states = outputs.hidden_states
            attentions = outputs.attentions

        logits = self.score(last_hidden_state)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            attentions=attentions,
        )


__all__ = [
    "T5GemmaConfig",
    "T5GemmaModuleConfig",
    "T5GemmaForConditionalGeneration",
    "T5GemmaModel",
    "T5GemmaEncoderModel",
    "T5GemmaPreTrainedModel",
    "T5GemmaForSequenceClassification",
    "T5GemmaForTokenClassification",
]
