# coding=utf-8
# Copyright 2025 The Nari Labs and HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Dia model."""

from typing import Callable, Optional, Union

import torch
from torch import nn

from ...cache_utils import DynamicCache, EncoderDecoderCache
from ...masking_utils import create_causal_mask
from ...modeling_attn_mask_utils import (
    _prepare_4d_attention_mask,
    _prepare_4d_attention_mask_for_sdpa,
)
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import auto_docstring, can_return_tuple, is_torch_flex_attn_available, is_torchdynamo_compiling, logging
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    eager_attention_forward,
)
from ..phi3.modeling_phi3 import Phi3MLP
from .configuration_dia import DiaConfig, DiaDecoderConfig, DiaEncoderConfig
from .generation_dia import DiaGenerationMixin


if is_torch_flex_attn_available():
    from ...integrations.flex_attention import make_flex_block_causal_mask


logger = logging.get_logger(__name__)


@auto_docstring
class DiaPreTrainedModel(PreTrainedModel):
    config: DiaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = True
    main_input_name = "input_ids"
    _no_split_modules = ["DiaEncoderLayer", "DiaDecoderLayer"]


class DiaMultiChannelEmbedding(nn.Module):
    """In order to efficiently compute the audio embedding from the 9 different channels,
    we vectorize the embedding process by using a single embedding layer and an offset.
    Example:
    - num_embeds = 4
    - vocab_size = 8
    - num_channels = 3
    We would have offsets = [0, 8, 16]
    If audio_codes = [0, 1, 2, 3], [1, 3, 4, 7], [5, 6, 7, 8],
    then tokens = audio_codes + offsets
                = [0, 1, 2, 3, 9, 11, 12, 15, 21, 22, 23, 24]
    This allows us to use a single embedding layer for all channels.
    """

    def __init__(self, config: DiaDecoderConfig):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size * config.num_channels, config.hidden_size)
        self.hidden_size = config.hidden_size
        self.num_channels = config.num_channels
        offsets = torch.arange(config.num_channels, dtype=torch.long) * config.vocab_size  # (C,)
        self.register_buffer("offsets", offsets, persistent=False)

    def forward(self, audio_codes: torch.Tensor) -> torch.Tensor:
        tokens = (audio_codes + self.offsets.to(audio_codes.device)).squeeze(1)
        embeds = self.embed(tokens).view(tokens.shape[0], audio_codes.shape[1], -1, self.hidden_size)
        return embeds.sum(dim=2)


class DiaMLP(Phi3MLP):
    pass


class DiaRMSNorm(LlamaRMSNorm):
    pass


class DiaRotaryEmbedding(LlamaRotaryEmbedding):
    pass


class DiaSelfAttention(LlamaAttention, nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Union[DiaEncoderConfig, DiaDecoderConfig], layer_idx: int, is_causal: bool = False):
        nn.Module.__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads or self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // self.num_heads)
        self.scaling = 1
        self.attention_dropout = 0.0
        self.is_causal = is_causal

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)


class DiaCrossAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: DiaDecoderConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.cross_hidden_size = config.cross_hidden_size
        self.num_heads = self.config.cross_num_attention_heads
        self.num_key_value_heads = self.config.cross_num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.cross_head_dim
        self.scaling = 1
        self.attention_dropout = 0.0
        self.is_causal = False

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.cross_hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.cross_hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        cross_shape = (*cross_attention_states.shape[:-1], -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        is_updated = past_key_values.is_updated.get(self.layer_idx) if past_key_values is not None else False
        if past_key_values is not None and is_updated:
            # reuse k,v, cross_attentions
            key_states = past_key_values.cross_attention_cache.layers[self.layer_idx].keys
            value_states = past_key_values.cross_attention_cache.layers[self.layer_idx].values
        else:
            key_states = self.k_proj(cross_attention_states).view(cross_shape).transpose(1, 2)
            value_states = self.v_proj(cross_attention_states).view(cross_shape).transpose(1, 2)

            if past_key_values is not None:
                # save all states to the cache
                key_states, value_states = past_key_values.cross_attention_cache.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                )
                # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
                past_key_values.is_updated[self.layer_idx] = True

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape((*input_shape, -1)).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class DiaEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: DiaEncoderConfig, layer_idx: int):
        super().__init__()
        self.pre_sa_norm = DiaRMSNorm(config.hidden_size, eps=config.norm_eps)
        self.self_attention = DiaSelfAttention(config, layer_idx, is_causal=False)
        self.post_sa_norm = DiaRMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = DiaMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        normed_states = self.pre_sa_norm(hidden_states)
        self_attn_output, self_attn_weights = self.self_attention(
            normed_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = residual + self_attn_output

        residual = hidden_states
        normed_states = self.post_sa_norm(hidden_states)
        mlp_out = self.mlp(normed_states)
        hidden_states = residual + mlp_out

        return hidden_states, self_attn_weights


class DiaEncoder(DiaPreTrainedModel):
    def __init__(self, config: DiaEncoderConfig):
        super().__init__(config)
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DiaEncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = DiaRMSNorm(config.hidden_size, eps=config.norm_eps)
        self.rotary_embeddings = DiaRotaryEmbedding(config)

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[BaseModelOutput, tuple]:
        hidden_states = self.embedding(input_ids)

        # RoPE
        # Note: We expect right padding and hence always generate
        # the position ids on the fly to reduce preparation overhead
        position_ids = torch.arange(input_ids.shape[-1], device=input_ids.device)[None, :]
        position_embeddings = self.rotary_embeddings(hidden_states, position_ids)

        attention_mask = self._update_full_mask(
            attention_mask,
            hidden_states,
        )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            layer_outputs = encoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                **kwargs,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            encoder_states += (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

    # Copied from transformers.models.bart.modeling_bart.BartPreTrainedModel._update_full_mask
    def _update_full_mask(
        self,
        attention_mask: Union[torch.Tensor, None],
        inputs_embeds: torch.Tensor,
    ):
        if attention_mask is not None:
            if self.config._attn_implementation == "flash_attention_2":
                attention_mask = attention_mask if 0 in attention_mask else None
            elif self.config._attn_implementation == "sdpa":
                # output_attentions=True & head_mask can not be supported when using SDPA, fall back to
                # the manual implementation that requires a 4D causal mask in all cases.
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = _prepare_4d_attention_mask_for_sdpa(attention_mask, inputs_embeds.dtype)
            elif self.config._attn_implementation == "flex_attention":
                if isinstance(attention_mask, torch.Tensor):
                    attention_mask = make_flex_block_causal_mask(attention_mask, is_causal=False)
            else:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)

        return attention_mask


class DiaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: DiaDecoderConfig, layer_idx: int):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attention = DiaSelfAttention(config, layer_idx, is_causal=True)
        self.cross_attention = DiaCrossAttention(config, layer_idx)
        self.pre_sa_norm = DiaRMSNorm(config.hidden_size, eps=config.norm_eps)
        self.pre_ca_norm = DiaRMSNorm(config.hidden_size, eps=config.norm_eps)
        self.pre_mlp_norm = DiaRMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = DiaMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        self_attn_cache = past_key_values
        if isinstance(self_attn_cache, EncoderDecoderCache):
            self_attn_cache = self_attn_cache.self_attention_cache

        residual = hidden_states
        normed_states = self.pre_sa_norm(hidden_states)
        self_attn_output, self_attn_weights = self.self_attention(
            normed_states,
            position_embeddings,
            attention_mask,
            # Needs to be an arg in order to function properly
            # on inplace operations to be carried (e.g. compile)
            self_attn_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + self_attn_output

        residual = hidden_states
        normed_states = self.pre_ca_norm(hidden_states)
        cross_states, cross_attn_weights = self.cross_attention(
            normed_states,
            encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = residual + cross_states

        residual = hidden_states
        normed_states = self.pre_mlp_norm(hidden_states)
        mlp_out = self.mlp(normed_states)
        hidden_states = residual + mlp_out

        return hidden_states, self_attn_weights, cross_attn_weights


class DiaDecoder(DiaPreTrainedModel):
    """Transformer Decoder Stack using DenseGeneral."""

    def __init__(self, config: DiaDecoderConfig):
        super().__init__(config)
        self.num_channels = config.num_channels
        self.vocab_size = config.vocab_size
        self.embeddings = DiaMultiChannelEmbedding(config)
        self.rotary_embeddings = DiaRotaryEmbedding(config)
        self.layers = nn.ModuleList(
            [DiaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = DiaRMSNorm(config.hidden_size, eps=config.norm_eps)

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[BaseModelOutputWithPastAndCrossAttentions, tuple]:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length, num_codebooks)`):
            The original `decoder_input_ids` in 3D shape to facilitate more efficient computations.

            [What are input IDs?](../glossary#input-ids)
        """

        batch_size, seq_length = input_ids.size()[:-1]
        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(
                past_key_values_length, past_key_values_length + seq_length, device=input_ids.device
            )
        if position_ids is None:
            position_ids = cache_position[None, :]

        # RoPE
        hidden_states = self.embeddings(input_ids)
        position_embeddings = self.rotary_embeddings(hidden_states, position_ids)

        if attention_mask is None and not is_torchdynamo_compiling():
            # required mask seq length can be calculated via length of past cache
            mask_seq_length = past_key_values_length + seq_length
            attention_mask = torch.ones(batch_size, mask_seq_length, device=input_ids.device)

        attention_mask = create_causal_mask(
            config=self.config,
            input_embeds=hidden_states,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        encoder_attention_mask = self._update_cross_attn_mask(
            encoder_hidden_states,
            encoder_attention_mask,
            hidden_states.shape[:2],
            hidden_states,
        )

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(
                hidden_states,
                position_embeddings,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns = all_self_attns + (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )

    # Copied from transformers.models.bart.modeling_bart.BartPreTrainedModel._update_cross_attn_mask
    def _update_cross_attn_mask(
        self,
        encoder_hidden_states: Union[torch.Tensor, None],
        encoder_attention_mask: Union[torch.Tensor, None],
        input_shape: torch.Size,
        inputs_embeds: torch.Tensor,
    ):
        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            if self.config._attn_implementation == "flash_attention_2":
                encoder_attention_mask = encoder_attention_mask if 0 in encoder_attention_mask else None
            elif self.config._attn_implementation == "sdpa":
                # output_attentions=True & cross_attn_head_mask can not be supported when using SDPA, and we fall back on
                # the manual implementation that requires a 4D causal mask in all cases.
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    encoder_attention_mask,
                    inputs_embeds.dtype,
                    tgt_len=input_shape[-1],
                )
            elif self.config._attn_implementation == "flex_attention":
                if isinstance(encoder_attention_mask, torch.Tensor):
                    encoder_attention_mask = make_flex_block_causal_mask(
                        encoder_attention_mask,
                        query_length=input_shape[-1],
                        is_causal=False,
                    )
            else:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                encoder_attention_mask = _prepare_4d_attention_mask(
                    encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
                )

        return encoder_attention_mask


@auto_docstring(
    custom_intro="""
    The bare Dia model outputting raw hidden-states without any specific head on top.
    """
)
class DiaModel(DiaPreTrainedModel):
    def __init__(self, config: DiaConfig):
        super().__init__(config)
        self.config = config
        self.encoder = DiaEncoder(config.encoder_config)
        self.decoder = DiaDecoder(config.decoder_config)
        self.post_init()

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_position_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Union[BaseModelOutput, tuple]] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[tuple, Seq2SeqModelOutput]:
        r"""
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size * num_codebooks, target_sequence_length)
        or (batch_size, target_sequence_length, num_codebooks)`, *optional*):
            1. (batch_size * num_codebooks, target_sequence_length): corresponds to the general use case where
            the audio input codebooks are flattened into the batch dimension. This also aligns with the flat-
            tened audio logits which are used to calculate the loss.

            2. (batch_size, sequence_length, num_codebooks): corresponds to the internally used shape of
            Dia to calculate embeddings and subsequent steps more efficiently.

            If no `decoder_input_ids` are provided, it will create a tensor of `bos_token_id` with shape
            `(batch_size, 1, num_codebooks)`. Indices can be obtained using the [`DiaProcessor`]. See
            [`DiaProcessor.__call__`] for more details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)
        decoder_position_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Used to calculate the position embeddings up to `config.decoder_config.max_position_embeddings`.

            [What are position IDs?](../glossary#position-ids)
        """

        if input_ids is None and encoder_outputs is None:
            raise ValueError(
                "You should either provide text ids or the cached text encodings. Neither has been found."
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if self.is_gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                **kwargs,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput
        elif not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # On default we initialize the decoder with bos tokens if nothing has been provided
        bsz, seq_len, channels = (encoder_outputs[0].shape[0], -1, self.config.decoder_config.num_channels)
        if decoder_input_ids is None:
            decoder_input_ids = torch.full(
                size=(bsz, 1, channels), fill_value=self.config.bos_token_id, device=self.device
            )
        # Ensure 3D
        if decoder_input_ids.ndim == 2:
            decoder_input_ids = decoder_input_ids.reshape(bsz, channels, seq_len).transpose(1, 2)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            position_ids=decoder_position_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
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
            encoder_last_hidden_state=encoder_outputs[0],
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    The Dia model consisting of a (byte) text encoder and audio decoder with a prediction head on top.
    """
)
class DiaForConditionalGeneration(DiaPreTrainedModel, DiaGenerationMixin):
    base_model_prefix = "model"

    def __init__(self, config: DiaConfig):
        super().__init__(config)
        self.config = config
        self.model = DiaModel(config)

        self.num_channels = config.decoder_config.num_channels
        self.vocab_size = config.decoder_config.vocab_size
        self.logits_dense = nn.Linear(
            config.decoder_config.hidden_size, (self.num_channels * self.vocab_size), bias=False
        )
        self.loss_type = "ForMaskedLM"

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_position_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Union[BaseModelOutput, tuple]] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[tuple, Seq2SeqLMOutput]:
        r"""
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size * num_codebooks, target_sequence_length)
        or (batch_size, target_sequence_length, num_codebooks)`, *optional*):
            1. (batch_size * num_codebooks, target_sequence_length): corresponds to the general use case where
            the audio input codebooks are flattened into the batch dimension. This also aligns with the flat-
            tened audio logits which are used to calculate the loss.

            2. (batch_size, sequence_length, num_codebooks): corresponds to the internally used shape of
            Dia to calculate embeddings and subsequent steps more efficiently.

            If no `decoder_input_ids` are provided, it will create a tensor of `bos_token_id` with shape
            `(batch_size, 1, num_codebooks)`. Indices can be obtained using the [`DiaProcessor`]. See
            [`DiaProcessor.__call__`] for more details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)
        decoder_position_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Used to calculate the position embeddings up to `config.decoder_config.max_position_embeddings`.

            [What are position IDs?](../glossary#position-ids)
        labels (`torch.LongTensor` of shape `(batch_size * num_codebooks,)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in
            `[0, ..., config.decoder_config.vocab_size - 1]` or -100. Tokens with indices set to `-100`
            are ignored (masked).
        """

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_position_ids=decoder_position_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        last_hidden_state = outputs[0]
        batch_size = last_hidden_state.shape[0]
        # 3D <-> 2D makes it necessary to prioritize channel dim
        audio_logits = (
            self.logits_dense(last_hidden_state)
            .view((batch_size, -1, self.num_channels, self.vocab_size))
            .transpose(1, 2)
            .contiguous()
            .view(batch_size * self.num_channels, -1, self.vocab_size)
        )

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=audio_logits, labels=labels, vocab_size=self.vocab_size, **kwargs)

        return Seq2SeqLMOutput(
            loss=loss,
            logits=audio_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


__all__ = ["DiaModel", "DiaPreTrainedModel", "DiaForConditionalGeneration"]
