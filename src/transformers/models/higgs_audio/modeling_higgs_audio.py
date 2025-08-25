# coding=utf-8
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
"""Higgs-Audio is an end-to-end multimodal model with the capability to understand and generate text / audio."""

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from ...cache_utils import Cache, DynamicCache, StaticCache
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring, can_return_tuple, logging
from ..auto import AutoTokenizer
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from ..whisper.modeling_whisper import WhisperEncoderLayer
from .configuration_higgs_audio import HiggsAudioConfig, HiggsAudioEncoderConfig
from .generation_higgs_audio import HiggsAudioGenerationMixin, merge_input_ids_with_audio_features


logger = logging.get_logger(__name__)


@auto_docstring(
    custom_intro="""
    The bare Higgs Audio Model outputting raw hidden-states without any specific head on top.
    """
)
@auto_docstring
class HiggsAudioPreTrainedModel(PreTrainedModel):
    config_class = HiggsAudioConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def _init_weights(self, module):
        std = self.config.init_std if hasattr(self.config, "init_std") else self.config.audio_encoder_config.init_std

        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, LlamaRMSNorm):
            module.weight.data.fill_(1.0)


class HiggsAudioDecoderProjector(nn.Module):
    """Projection layers that map hidden states from the LLM component to audio / text logits.

    We support two type of audio head:
    - Basic Audio Head:
        Directly map the hidden states to audio logits for all the codebooks.
    """

    def __init__(self, config: HiggsAudioConfig):
        super().__init__()
        self.text_lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.audio_lm_head = nn.Linear(
            config.text_config.hidden_size, config.audio_num_codebooks * (config.audio_codebook_size + 2), bias=False
        )

    def forward(
        self,
        hidden_states,
        audio_out_mask,
        output_audio_hidden_states=False,
    ):
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, seq_len, hidden_size)`):
                Hidden states from the LLM component
            audio_out_mask (`torch.Tensor` of shape `(batch_size, seq_len)`):
                Mask for identifying the audio out tokens.
            output_audio_hidden_states (:obj:`bool`):
                Whether to output audio hidden states.

        Returns:
            logits (`torch.Tensor` of shape `(batch_size, seq_len, vocab_size)`):
                Logits for text tokens
            audio_logits (`torch.Tensor` of shape `(num_audio_out_tokens, audio_num_codebooks * audio_codebook_size)`):
                Logits for audio tokens. We ensure `num_text_tokens + num_audio_tokens == batch_size * seq_len`
            audio_hidden_states  (`torch.Tensor` of shape `(num_audio_out_tokens, hidden_size)`):
                Hidden states for audio out tokens
        """
        logits = self.text_lm_head(hidden_states)

        audio_logits = self.audio_lm_head(hidden_states[audio_out_mask])

        if output_audio_hidden_states:
            audio_hidden_states = hidden_states[audio_out_mask]
        else:
            audio_hidden_states = None

        return logits, audio_logits, audio_hidden_states


def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    Args:
        attention_mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`torch.dtype`):
            The dtype to use for the 4D attention mask.
        device (`torch.device`):
            The device to plcae the 4D attention mask on.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        cache_position (`torch.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`torch.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

    return causal_mask


class HiggsAudioFeatureProjector(nn.Module):
    """Projector that maps audio features extracted by Whisper to hidden state of the text model."""

    def __init__(self, config: HiggsAudioConfig):
        super().__init__()
        self.linear = nn.Linear(config.audio_encoder_config.d_model, config.text_config.hidden_size, bias=True)

    def forward(self, audio_features):
        hidden_states = self.linear(audio_features)
        return hidden_states


class HiggsAudioEncoder(HiggsAudioPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`WhisperEncoderLayer`].

    Args:
        config: HiggsAudioEncoderConfig
    """

    # Ignore copy
    config_class = HiggsAudioEncoderConfig
    main_input_name = "input_features"
    _no_split_modules = ["WhisperEncoderLayer"]

    def __init__(self, config: HiggsAudioEncoderConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)
        self.embed_positions.requires_grad_(False)

        self.layers = nn.ModuleList([WhisperEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)
        # Ignore copy
        self.avg_pooler = nn.AvgPool1d(2, stride=2)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        r"""
        Args:
            input_features (`torch.LongTensor` of shape `(batch_size, feature_size, sequence_length)`):
                Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a
                `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
                `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
                and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
            attention_mask (`torch.Tensor`)`, *optional*):
                HiggsAudio does not support masking of the `input_features`, this argument is preserved for compatibility,
                but it is not used. By default the silence in the input log mel spectrogram are ignored.
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # Ignore copy
        input_features = input_features.to(dtype=self.conv1.weight.dtype, device=self.conv1.weight.device)

        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.embed_positions.weight

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (len(self.layers)), (
                f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
            )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            # Ignore copy
            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                        output_attentions,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Ignore copy
        hidden_states = hidden_states.permute(0, 2, 1)
        # If the sequence length after average pooling is not divisible by the sequence parallel size, we would duplicate it across the sequence parallel ranks.
        # In this case, gradients need to be scaled up because the subsequent scaling up in the function _apply_audio_tower is skipped.
        hidden_states = self.avg_pooler(hidden_states)

        hidden_states = hidden_states.permute(0, 2, 1)

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

    # Ignore copy
    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths


class HiggsAudioDualFFNDecoderLayer(nn.Module):
    """We implement a dual-path FFN decoder layer where the audio tokens and text tokens go through separate FFN layers.

    The audio and text tokens share the text-attention layer, but will be encoded with separate feedforward layers.
    In addition, the audio tokens can be configured to go through separate attention layer.

    Following is an illustration:

     t    t    t    a   a     a    t    t    t
                        |
                        | (shared attention layer)
                        v
    h_t  h_t  h_t  h_a  h_a  h_a  h_t  h_t  h_t
                        |
                        | (separate text/audio hidden states)
                        v
    [h_t  h_t  h_t  h_t  h_t  h_t], [h_a, h_a, h_a]
             |                             |
             | (separate FFNs)             |
             v                             v
    [o_t  o_t  o_t  o_t  o_t  o_t], [o_a, o_a, o_a]
                        |
                        | (reorder)
                        v
    o_t  o_t  o_t  o_a  o_a  o_a  o_t  o_t  o_t

    This has a few advantages:
    1) We are able to use a smaller FFN, or even bypass the FFN for audio tokens. This accelerates the inference speed.
    2) The Audio-FFN introduces more trainable parameters to the model.
       This should have the same effect as the mixture-of-expert layer and we may expect better performance due to parameter scaling.
    3) We can replace the original FFN in LLMs with the dual-path FFN without changing the number of FLOPs.


    """

    def __init__(
        self, config: HiggsAudioConfig, layer_idx: int, fast_forward: bool = False, use_audio_attention: bool = False
    ):
        super().__init__()
        text_config = config.text_config
        self.hidden_size = text_config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = LlamaAttention(config=text_config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(text_config)

        if not fast_forward:
            if use_audio_attention:
                self.audio_attn = LlamaAttention(config=text_config, layer_idx=layer_idx + 1)
                self.audio_post_audio_attn_layer_norm = LlamaRMSNorm(
                    text_config.hidden_size, eps=text_config.rms_norm_eps
                )

            self.audio_mlp = LlamaMLP(text_config)
            self.audio_input_layernorm = LlamaRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
            self.audio_post_attention_layernorm = LlamaRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)

        self.use_audio_attention = use_audio_attention
        self.fast_forward = fast_forward
        if self.fast_forward:
            assert not self.use_audio_attention, (
                "We cannot use audio_attention if the layer is marked as fast-forward."
            )
        self.input_layernorm = LlamaRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        fast_forward_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        audio_out_mask: Optional[torch.BoolTensor] = None,
        is_decoding_audio_token: Optional[bool] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            position_ids
                IDs of positions in the input sequence
            audio_out_mask
                Mask for identifying the audio tokens. Size (batch_size, sequence_length)
                1 --> location contains audio_out
                0 --> location does not contain audio_out

                When use_cache is True and not in torch compile mode, the audio_out_mask contains audio_out masks for
                all tokens up to the current token.  That means, it has size (batch_size, sequence_length) while
                hidden_states will have size (batch_size, 1). In the torch compile mode, the audio_out_mask will have
                size (batch_size, 1).
            is_decoding_audio_token
                Used in the torch compile mode to determine if the current token is an audio token or not.
            past_key_value (`Cache`, *optional*): cached past key and value projection states. We fetch the corresponding cached key/value via the layer_idx.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states
        target_length = hidden_states.shape[1]
        use_static_cache = isinstance(past_key_value, StaticCache)

        # If we are decoding an audio token and the layer is marked as fast-forward,
        # we can skip it.
        if is_decoding_audio_token and self.fast_forward:
            return (hidden_states,)

        has_audio_out = audio_out_mask is not None and audio_out_mask.shape[0] > 0

        audio_out_mask_sq = audio_out_mask

        if self.fast_forward and has_audio_out:
            original_hidden_states = hidden_states.clone()
            min_dtype = torch.finfo(hidden_states.dtype).min
            if attention_mask is None:
                attention_mask = ~audio_out_mask

                if self.self_attn.config._attn_implementation != "flash_attention_2":
                    sequence_length = audio_out_mask.shape[1]
                    attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                        attention_mask=attention_mask,
                        sequence_length=sequence_length,
                        target_length=sequence_length,
                        dtype=hidden_states.dtype,
                        min_dtype=min_dtype,
                        device=hidden_states.device,
                        cache_position=cache_position,
                        batch_size=hidden_states.shape[0],
                    )
                    if use_cache:
                        attention_mask = attention_mask[:, :, -target_length:, :]
            elif len(attention_mask.shape) == 2:
                # Attention mask has shape (batch_size, sequence_length)
                # We should be using flash attention 2
                attention_mask = attention_mask * ~audio_out_mask
            elif len(attention_mask.shape) == 4:
                # When using static cache, the attention mask was already preprocessed in the previous layer
                if use_static_cache:
                    attention_mask = fast_forward_attention_mask
                else:
                    if use_cache:
                        # Attention mask has shape (batch_size, 1, query_length, key_length)
                        # In addition, the attention mask should be inverted, that means "1" (attend_to) --> "0", and "0" --> minimal dtype value.
                        attention_mask = attention_mask.masked_fill(
                            audio_out_mask[:, -target_length:].reshape(audio_out_mask.shape[0], 1, target_length, 1)
                            | audio_out_mask.reshape(audio_out_mask.shape[0], 1, 1, audio_out_mask.shape[1]),
                            min_dtype,
                        )
                    else:
                        attention_mask = attention_mask.masked_fill(
                            audio_out_mask.reshape(audio_out_mask.shape[0], 1, audio_out_mask.shape[1], 1)
                            | audio_out_mask.reshape(audio_out_mask.shape[0], 1, 1, audio_out_mask.shape[1]),
                            min_dtype,
                        )
            else:
                raise NotImplementedError(f"Unsupported attention_mask format, attention_mask={attention_mask}")

            if (
                self.self_attn.config._attn_implementation == "sdpa"
                and attention_mask is not None
                and attention_mask.device.type == "cuda"
                and not output_attentions
            ):
                # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
                # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
                # Details: https://github.com/pytorch/pytorch/issues/110213
                attention_mask = AttentionMaskConverter._unmask_unattended(attention_mask, min_dtype)

        if has_audio_out and not self.fast_forward:
            # Apply separate layernorm layers for audio tokens and text tokens
            if use_cache:
                hidden_states = torch.where(
                    audio_out_mask_sq[:, -target_length:].unsqueeze(-1),
                    self.audio_input_layernorm(hidden_states),
                    self.input_layernorm(hidden_states),
                )
            else:
                hidden_states = torch.where(
                    audio_out_mask_sq.unsqueeze(-1),
                    self.audio_input_layernorm(hidden_states),
                    self.input_layernorm(hidden_states),
                )
        else:
            hidden_states = self.input_layernorm(hidden_states)

        # Audio Attention
        if self.use_audio_attention and has_audio_out:
            if use_static_cache:
                assert audio_attention_mask is not None, (
                    "audio_attention_mask should not be None when using static cache."
                )

            if audio_attention_mask is None:
                no_audio_out_mask = (~audio_out_mask)[:, -target_length:].reshape(
                    audio_out_mask.shape[0], 1, target_length, 1
                ) | (~audio_out_mask).reshape(audio_out_mask.shape[0], 1, 1, audio_out_mask.shape[1])
                min_dtype = torch.finfo(hidden_states.dtype).min

                if attention_mask is None:
                    audio_attention_mask = audio_out_mask

                    if self.audio_attn.config._attn_implementation != "flash_attention_2":
                        sequence_length = audio_out_mask.shape[1]
                        audio_attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                            attention_mask=audio_attention_mask,
                            sequence_length=sequence_length,
                            target_length=sequence_length,
                            dtype=hidden_states.dtype,
                            min_dtype=min_dtype,
                            device=hidden_states.device,
                            cache_position=cache_position,
                            batch_size=hidden_states.shape[0],
                        )
                        if use_cache:
                            audio_attention_mask = audio_attention_mask[:, :, -target_length:, :]
                        audio_attention_mask = audio_attention_mask.masked_fill(no_audio_out_mask, min_dtype)
                elif len(attention_mask.shape) == 2:
                    # Attention mask has shape (batch_size, sequence_length)
                    audio_attention_mask = attention_mask * audio_out_mask
                elif len(attention_mask.shape) == 4:
                    # Attention mask has shape (batch_size, 1, query_length, key_length)
                    # In addition, the attention mask should be inverted. This means "1" (attend_to) --> "0", and "0" --> minimal dtype value.
                    audio_attention_mask = attention_mask.masked_fill(no_audio_out_mask, min_dtype)
                else:
                    raise NotImplementedError(f"Unsupported attention_mask format, attention_mask={attention_mask}")

                if (
                    self.audio_attn.config._attn_implementation == "sdpa"
                    and audio_attention_mask is not None
                    and audio_attention_mask.device.type == "cuda"
                    and not output_attentions
                ):
                    # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
                    # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
                    # Details: https://github.com/pytorch/pytorch/issues/110213
                    audio_attention_mask = AttentionMaskConverter._unmask_unattended(audio_attention_mask, min_dtype)

            audio_attention_mask = audio_attention_mask.contiguous()

            audio_hidden_states, audio_self_attn_weights = self.audio_attn(
                hidden_states=hidden_states,
                attention_mask=audio_attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            audio_hidden_states = residual + audio_hidden_states
            if use_cache:
                residual = torch.where(
                    audio_out_mask_sq[:, -target_length:].unsqueeze(-1), audio_hidden_states, residual
                )
            else:
                residual = torch.where(audio_out_mask_sq.unsqueeze(-1), audio_hidden_states, residual)
            audio_hidden_states = self.audio_post_audio_attn_layer_norm(audio_hidden_states)
            if use_cache:
                hidden_states = torch.where(
                    audio_out_mask_sq[:, -target_length:].unsqueeze(-1), audio_hidden_states, hidden_states
                )
            else:
                hidden_states = torch.where(audio_out_mask_sq.unsqueeze(-1), audio_hidden_states, hidden_states)

        # Text Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Apply Dual-path FFN
        residual = hidden_states

        if has_audio_out and not self.fast_forward:
            if use_cache:
                real_audio_out_mask = audio_out_mask_sq[:, -target_length:]
            else:
                real_audio_out_mask = audio_out_mask_sq

            # Make whole graph in decode stage
            text_hidden_states = self.post_attention_layernorm(hidden_states[~real_audio_out_mask])
            audio_hidden_states = self.audio_post_attention_layernorm(hidden_states[real_audio_out_mask])

            text_hidden_states = self.mlp(text_hidden_states)
            residual[~real_audio_out_mask] += text_hidden_states

            audio_hidden_states = self.audio_mlp(audio_hidden_states)
            residual[real_audio_out_mask] += audio_hidden_states

            hidden_states = residual
        else:
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

        if self.fast_forward and has_audio_out:
            if use_cache:
                hidden_states = torch.where(
                    audio_out_mask_sq[:, -target_length:].unsqueeze(-1), original_hidden_states, hidden_states
                )
            else:
                hidden_states = torch.where(audio_out_mask_sq.unsqueeze(-1), original_hidden_states, hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            if self.use_audio_attention:
                # The returned attn weights have shape (batch_size, num_heads + num_audio_attn_heads, seq_length, seq_length)
                outputs += (torch.concat([self_attn_weights, audio_self_attn_weights], dim=1),)
            else:
                # The returned attn weights have shape (batch_size, num_heads, seq_length, seq_length)
                outputs += (self_attn_weights,)

        if use_cache:
            outputs += (past_key_value,)

        return outputs


@dataclass
class HiggsAudioModelOutputWithPast(ModelOutput):
    """
    Output type for HiggsAudioModel when using past key values (for fast decoding)
    and multi-modal audio-text processing.

    Attributes:
        logits (Optional[torch.FloatTensor]):
            Prediction scores (before softmax) of shape `(batch_size, sequence_length, vocab_size)`.

        expanded_input_ids (Optional[torch.LongTensor]):
            Expanded text token IDs after alignment with audio features and tokens.

        expanded_labels (Optional[torch.LongTensor]):
            Expanded target labels, aligned with `expanded_input_ids` for loss computation.

        audio_in_mask (Optional[torch.BoolTensor]):
            Mask indicating which positions in the input correspond to audio features.

        audio_in_discrete_codes_mask (Optional[torch.BoolTensor]):
            Mask for discrete audio tokens in the input sequence.

        audio_out_mask (Optional[torch.BoolTensor]):
            Mask indicating which positions in the output correspond to audio predictions.

        attention_mask (Optional[torch.BoolTensor]):
            Attention mask for the Transformer to avoid attending to padding tokens.

        audio_logits (Optional[torch.FloatTensor]):
            Prediction scores for audio tokens of shape
            `(num_audio_out_tokens, audio_num_codebooks, audio_codebook_size)`.

        past_key_values (Optional[Cache]):
            Pre-computed key and value tensors for each attention layer, used to speed up
            generation by avoiding re-computation.

        last_hidden_states (Optional[torch.FloatTensor]):
            The model’s hidden state output.

        hidden_states (Optional[Tuple[torch.FloatTensor, ...]]):
            Tuple of hidden states from each layer of the transformer decoder.

        last_audio_hidden_states (Optional[Tuple[torch.FloatTensor, ...]]):
            The model’s audio hidden state output.

        attentions (Optional[Tuple[torch.FloatTensor, ...]]):
            Tuple of attention weight tensors from each layer, each of shape
            `(batch_size, num_heads, seq_length, seq_length)`.
    """

    logits: Optional[torch.FloatTensor] = None
    expanded_input_ids: Optional[torch.LongTensor] = None
    expanded_labels: Optional[torch.LongTensor] = None
    audio_in_mask: Optional[torch.BoolTensor] = None
    audio_in_discrete_codes_mask: Optional[torch.BoolTensor] = None
    audio_out_mask: Optional[torch.BoolTensor] = None
    attention_mask: Optional[torch.BoolTensor] = None
    audio_logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    last_hidden_states: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    last_audio_hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


@dataclass
class HiggsAudioOutputWithPast(ModelOutput):
    """
    Output type for HiggsAudioModel when computing losses during training,
    extending `HiggsAudioModelOutputWithPast` with additional loss fields.

    Attributes:
        loss (Optional[torch.FloatTensor]):
            Total training loss (can be a weighted sum of multiple loss components).

        llm_loss (Optional[torch.FloatTensor]):
            Loss from the language model (text) prediction component.

        audio_loss (Optional[torch.FloatTensor]):
            Loss from the audio token prediction component.

        codebook_losses (Optional[torch.FloatTensor]):
            Losses related to vector quantization codebooks, such as commitment
            or reconstruction losses in audio discretization.

        logits (Optional[torch.FloatTensor]):
            Prediction scores (before softmax) of shape `(batch_size, sequence_length, vocab_size)`.

        expanded_input_ids (Optional[torch.LongTensor]):
            Expanded text token IDs after alignment with audio features and tokens.

        expanded_labels (Optional[torch.LongTensor]):
            Expanded target labels, aligned with `expanded_input_ids` for loss computation.

        audio_in_mask (Optional[torch.BoolTensor]):
            Mask indicating which positions in the input correspond to audio features.

        audio_in_discrete_codes_mask (Optional[torch.BoolTensor]):
            Mask for discrete audio tokens in the input sequence.

        audio_out_mask (Optional[torch.BoolTensor]):
            Mask indicating which positions in the output correspond to audio predictions.

        attention_mask (Optional[torch.BoolTensor]):
            Attention mask for the Transformer to avoid attending to padding tokens.

        audio_logits (Optional[torch.FloatTensor]):
            Prediction scores for audio tokens of shape
            `(num_audio_out_tokens, audio_num_codebooks, audio_codebook_size)`.

        past_key_values (Optional[Cache]):
            Pre-computed key and value tensors for each attention

        last_hidden_states (Optional[torch.FloatTensor]):
            The model’s hidden state output.

        hidden_states (Optional[Tuple[torch.FloatTensor, ...]]):
            Tuple of hidden states from each layer of the transformer decoder.

        last_audio_hidden_states (Optional[Tuple[torch.FloatTensor, ...]]):
            The model’s audio hidden state output.

        attentions (Optional[Tuple[torch.FloatTensor, ...]]):
            Tuple of attention weight tensors from each layer, each of shape
            `(batch_size, num_heads, seq_length, seq_length)`.
    """

    loss: Optional[torch.FloatTensor] = None
    llm_loss: Optional[torch.FloatTensor] = None
    audio_loss: Optional[torch.FloatTensor] = None
    codebook_losses: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    expanded_input_ids: Optional[torch.LongTensor] = None
    expanded_labels: Optional[torch.LongTensor] = None
    audio_in_mask: Optional[torch.BoolTensor] = None
    audio_in_discrete_codes_mask: Optional[torch.BoolTensor] = None
    audio_out_mask: Optional[torch.BoolTensor] = None
    attention_mask: Optional[torch.BoolTensor] = None
    audio_logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    last_hidden_states: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    last_audio_hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


class HiggsAudioModel(HiggsAudioPreTrainedModel):
    """Higgs-Audio is an end-to-end multimodal model with the capability to understand and generate text / audio.

    Consider the following example for mixed text/audio understanding / generation:

    - input_tokens: <text_token1><|audio_bos|>[AUDIO]<|audio_eos|><text_token2><|audio_bos|>[AUDIO]<|audio_eos|><text_token4>
    - input_tokens: <text_token1><|audio_bos|>[AUDIO]<|audio_eos|><text_token2><|audio_out_bos|>[AUDIO_OUT]<|audio_eos|><text_token4>

    We will fill [AUDIO] with the audio features extracted by Whisper and fill [AUDIO_OUT] with the audio tokens.

    Consider the following example for mixed text/audio generation:

    text: <|audio_out_bos|>    MASK           MASK           MASK          MASK               MASK         <|audio_eos|> [text_token1]
    audio:     MASK    <|audio_stream_bos|> [audio_token1] [audio_token2] [audio_token3] <|audio_stream_eos|>   MASK           MASK
    token_type: 0               1              1              1             1                  1                 0              0

    """

    _supports_cache_class = True
    _supports_static_cache = True

    def __init__(self, config: HiggsAudioConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.audio_in_token_idx = config.audio_in_token_idx
        self.audio_out_token_idx = config.audio_out_token_idx
        self.audio_out_bos_token_id = config.audio_out_bos_token_id if "audio_out_bos_token_id" in config else None
        self.audio_eos_token_id = config.audio_eos_token_id if "audio_eos_token_id" in config else None
        self.vocab_size = config.text_config.vocab_size
        self.audio_num_codebooks = config.audio_num_codebooks
        self.use_delay_pattern = config.use_delay_pattern
        self.use_audio_out_embed_projector = config.use_audio_out_embed_projector
        self.use_audio_out_self_attention = config.use_audio_out_self_attention

        # A hacky solution. Not sure why _attn_implementation is None
        if config.text_config._attn_implementation is None:
            config.text_config._attn_implementation = "sdpa"
        if config.audio_encoder_config._attn_implementation is None:
            config.audio_encoder_config._attn_implementation = "sdpa"

        self.embed_tokens = nn.Embedding(self.vocab_size, config.text_config.hidden_size, self.padding_idx)

        if config.audio_adapter_type == "dual_ffn":
            layer_idx = 0
            layers = []
            for j in range(config.text_config.num_hidden_layers):
                if j in config.audio_dual_ffn_layers:
                    layers.append(
                        HiggsAudioDualFFNDecoderLayer(
                            config, layer_idx, use_audio_attention=self.use_audio_out_self_attention
                        )
                    )
                    layer_idx += 2 if self.use_audio_out_self_attention else 1
                else:
                    layers.append(LlamaDecoderLayer(config.text_config, layer_idx))
                    layer_idx += 1
            self.layers = nn.ModuleList(layers)
        elif config.audio_adapter_type == "dual_ffn_fast_forward":
            layer_idx = 0
            layers = []
            for j in range(config.text_config.num_hidden_layers):
                if j in config.audio_dual_ffn_layers:
                    layers.append(
                        HiggsAudioDualFFNDecoderLayer(
                            config,
                            layer_idx,
                            fast_forward=False,
                            use_audio_attention=self.use_audio_out_self_attention,
                        )
                    )
                    layer_idx += 2 if self.use_audio_out_self_attention else 1
                else:
                    layers.append(
                        HiggsAudioDualFFNDecoderLayer(config, layer_idx, fast_forward=True, use_audio_attention=False)
                    )
                    layer_idx += 1
            self.layers = nn.ModuleList(layers)
        elif config.audio_adapter_type == "stack":
            self.layers = nn.ModuleList(
                [
                    LlamaDecoderLayer(config.text_config, layer_idx)
                    for layer_idx in range(config.text_config.num_hidden_layers)
                ]
            )
            layer_idx = config.text_config.num_hidden_layers
        else:
            raise NotImplementedError(f"Audio adapter type {config.audio_adapter_type} not implemented.")

        self.num_activation_checkpointing_layers = len(self.layers)

        self.norm = LlamaRMSNorm(config.text_config.hidden_size, eps=config.text_config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config.text_config)

        if not config.skip_audio_tower:
            self.audio_tower = HiggsAudioEncoder(config.audio_encoder_config)
            self.audio_encoder_proj = HiggsAudioFeatureProjector(config)
        else:
            self.audio_tower = None
            self.audio_encoder_proj = None
        self.audio_decoder_proj = HiggsAudioDecoderProjector(config)
        self.audio_codebook_size = (
            config.audio_codebook_size + 2
        )  # We add 1 for the audio_stream_bos token and 1 for the audio_stream_eos token

        if config.use_audio_out_embed_projector:
            self.audio_out_embed_projector = nn.Linear(
                config.text_config.hidden_size, config.text_config.hidden_size, bias=False
            )

        self.audio_codebook_embeddings = nn.Embedding(
            config.audio_num_codebooks * self.audio_codebook_size, config.text_config.hidden_size
        )

        self.audio_codebook_weights = (
            torch.ones(config.audio_num_codebooks) / config.audio_num_codebooks
        )  # default to equal weights
        self.post_init()

    def set_audio_special_tokens(self, tokenizer: AutoTokenizer):
        self.audio_out_bos_token_id = tokenizer.convert_tokens_to_ids("<|audio_out_bos|>")
        self.audio_eos_token_id = tokenizer.convert_tokens_to_ids("<|audio_eos|>")

    def _embed_audio_ids(self, audio_ids):
        """Embed the audio ids

        Args:
            audio_ids: torch.LongTensor of shape (num_codebooks, audio_in_total_length)

        Returns:
            audio_embed: torch.LongTensor of shape (audio_in_total_length, hidden_size)
        """
        codebook_shift = (
            torch.arange(self.config.audio_num_codebooks, device=audio_ids.device) * self.audio_codebook_size
        )
        audio_embed = self.audio_codebook_embeddings(audio_ids + codebook_shift.unsqueeze(-1))
        if self.config.audio_embed_avg:
            audio_embed = torch.mean(audio_embed, dim=0)
        else:
            audio_embed = torch.sum(audio_embed, dim=0)
        if self.use_audio_out_embed_projector:
            audio_embed = self.audio_out_embed_projector(audio_embed)
        return audio_embed

    def _apply_audio_tower(self, audio_features, audio_feature_attention_mask):
        """Apply the audio tower to the audio features"""

        if audio_features.shape[0] == 0:
            if torch.is_grad_enabled():
                audio_outputs = self.audio_tower(audio_features, attention_mask=None)
                selected_audio_feature = audio_outputs.last_hidden_state
                audio_features_embed = self.audio_encoder_proj(selected_audio_feature)
                audio_feat_out_lengths = None
                return audio_features_embed, audio_feat_out_lengths
            else:
                return None, None

        audio_feat_lengths, audio_feat_out_lengths = self.audio_tower._get_feat_extract_output_lengths(
            audio_feature_attention_mask.sum(-1)
        )
        batch_size, _, max_mel_seq_len = audio_features.shape
        max_seq_len = (max_mel_seq_len - 1) // 2 + 1
        # Create a sequence tensor of shape (batch_size, max_seq_len)
        seq_range = (
            torch.arange(0, max_seq_len, dtype=audio_feat_lengths.dtype, device=audio_feat_lengths.device)
            .unsqueeze(0)
            .expand(batch_size, max_seq_len)
        )
        lengths_expand = audio_feat_lengths.unsqueeze(1).expand(batch_size, max_seq_len)
        # Create mask
        padding_mask = seq_range < lengths_expand

        if self.config._attn_implementation != "flash_attention_2":
            audio_attention_mask = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
                batch_size, 1, max_seq_len, max_seq_len
            )
        else:
            audio_attention_mask = padding_mask

        audio_outputs = self.audio_tower(audio_features, attention_mask=audio_attention_mask)
        selected_audio_feature = audio_outputs.last_hidden_state
        audio_features_embed = self.audio_encoder_proj(selected_audio_feature)

        return audio_features_embed, audio_feat_out_lengths

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    def _prepare_all_static_kv_cache_masks(self, hidden_states, attention_mask, audio_out_mask, past_key_values):
        target_length = hidden_states.shape[1]
        cur_pos = audio_out_mask.shape[1]
        min_dtype = torch.finfo(hidden_states.dtype).min
        assert len(attention_mask.shape) == 4, "Only support SDPA for now"
        kv_cache_len = past_key_values.get_max_cache_shape()
        audio_out_mask_padded = torch.nn.functional.pad(audio_out_mask, (0, kv_cache_len - cur_pos), value=True)
        fast_forward_attention_mask = attention_mask.masked_fill(
            audio_out_mask_padded[:, audio_out_mask.shape[1] - target_length : audio_out_mask.shape[1]].reshape(
                audio_out_mask_padded.shape[0], 1, target_length, 1
            )
            | audio_out_mask_padded.reshape(audio_out_mask_padded.shape[0], 1, 1, audio_out_mask_padded.shape[1]),
            min_dtype,
        )

        no_audio_out_mask = ~audio_out_mask
        no_audio_out_mask = torch.nn.functional.pad(
            no_audio_out_mask, (0, kv_cache_len - audio_out_mask.shape[1]), value=False
        )
        no_audio_out_mask = no_audio_out_mask[
            :, audio_out_mask.shape[1] - target_length : audio_out_mask.shape[1]
        ].reshape(audio_out_mask.shape[0], 1, target_length, 1) | no_audio_out_mask.reshape(
            audio_out_mask.shape[0], 1, 1, kv_cache_len
        )
        audio_attention_mask = attention_mask.masked_fill(no_audio_out_mask, min_dtype)
        return fast_forward_attention_mask, audio_attention_mask

    def _forward_core(
        self,
        hidden_states: torch.Tensor,
        causal_mask: torch.Tensor,
        position_ids: torch.Tensor,
        audio_discrete_codes_mask: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]],
        use_cache: bool,
        audio_attention_mask: torch.Tensor,
        fast_forward_attention_mask: torch.Tensor,
        output_attentions: bool,
        output_hidden_states: bool,
        is_decoding_audio_token: Optional[bool] = None,
    ):
        # create position embeddings to be shared across the decoder layers
        # When past_key_values is passed in, we need to offset the position ids when calculating the position embeddings.
        # Therefore, cache_position is used.
        position_id_offset = cache_position[0] if use_cache else 0
        position_embeddings = self.rotary_emb(hidden_states, position_ids + position_id_offset)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if isinstance(decoder_layer, HiggsAudioDualFFNDecoderLayer):
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    audio_attention_mask=audio_attention_mask,
                    fast_forward_attention_mask=fast_forward_attention_mask,
                    position_ids=position_ids,
                    audio_out_mask=audio_discrete_codes_mask,
                    is_decoding_audio_token=is_decoding_audio_token,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)
            else:
                hidden_states = layer_outputs
                if output_attentions:
                    all_self_attns += (None,)

        return hidden_states, all_hidden_states, all_self_attns

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        audio_features: Optional[torch.FloatTensor] = None,
        audio_feature_attention_mask: Optional[torch.BoolTensor] = None,
        audio_in_ids: Optional[torch.LongTensor] = None,
        audio_in_ids_start: Optional[torch.LongTensor] = None,
        audio_out_ids: Optional[torch.LongTensor] = None,
        audio_out_ids_start: Optional[torch.LongTensor] = None,
        label_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_audio_hidden_states: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        cache_audio_discrete_codes_mask: Optional[torch.LongTensor] = None,
        past_key_values_buckets: Optional[OrderedDict[int, Cache]] = None,
    ):
        r"""
        input_ids (:obj:`torch.LongTensor`):
            The input ids of the prompt. It will have shape (bsz, seq_len).
            When use_cache is enabled, the input_ids will have
            shape (bsz, 1) for incremental decode or None
        attention_mask (:obj:`torch.LongTensor`):
            The attention mask of the prompt. It will have shape (bsz, seq_len).
        audio_features (:obj:`torch.FloatTensor`):
            The audio features extracted by Whisper. It will have shape (num_audio_in, feature_dim, max_mel_seq_len).
        audio_feature_attention_mask (:obj:`torch.LongTensor`):
            The attention mask of the audio features. It will have shape (num_audio_in, max_mel_seq_len).
        audio_in_ids (:obj:`torch.LongTensor`):
            The discretized audio tokens. It will have shape (num_codebooks, audio_in_total_length).
        audio_in_ids_start (:obj:`torch.LongTensor`):
            The start indices for each audio in audio_in_ids. It will have shape (num_audio_in,)
        audio_out_ids (:obj:`torch.LongTensor`):
            The discretized audio tokens. It will have shape (num_codebooks, audio_out_total_length).
        audio_out_ids_start (:obj:`torch.LongTensor`):
            The start indices for each audio in audio_out_ids. It will have shape (num_audio_out,)
        label_ids (:obj:`torch.LongTensor`):
            The labels of the prompt. It will have shape (bsz, seq_len).
        past_key_values (:obj:`Tuple`):
            Tuple of past key values.
        use_cache (:obj:`bool`):
            Whether to use cache.
        output_attentions (:obj:`bool`):
            Whether to output attentions.
        output_hidden_states (:obj:`bool`):
            Whether to output hidden states.
        output_audio_hidden_states (:obj:`bool`):
            Whether to output audio hidden states.
        cache_position (:obj:`torch.LongTensor`):
            The position of the cache.
        cache_audio_discrete_codes_mask (:obj:`torch.LongTensor`):
            The cached audio discrete codes mask. It will only be used when use_cache is turned on.
        past_key_values_buckets (:obj:`OrderedDict`):
            The buckets of past key values.
        """
        target_device = input_ids.device

        if audio_features is not None:
            audio_features = audio_features.to(target_device)
            audio_feature_attention_mask = audio_feature_attention_mask.to(target_device)

        # 1. Extract the input embeddings
        inputs_embeds = self.embed_tokens(input_ids)

        # 2. Extract audio embeddings
        if self.config.skip_audio_tower:
            audio_features_embed = audio_features_length = None
        else:
            audio_features_embed, audio_features_length = self._apply_audio_tower(
                audio_features, audio_feature_attention_mask
            )

        if self.config.encode_audio_in_tokens:
            if audio_in_ids is not None and audio_in_ids.shape[-1] > 0:
                audio_in_ids = audio_in_ids.to(target_device)
            else:
                audio_in_ids = torch.zeros((self.audio_num_codebooks, 0), device=target_device, dtype=torch.long)
            audio_in_embed = self._embed_audio_ids(audio_in_ids)
        else:
            audio_in_embed = None

        if audio_out_ids is not None and audio_out_ids.shape[-1] > 0:
            audio_out_ids = audio_out_ids.to(target_device)
        else:
            audio_out_ids = torch.zeros((self.audio_num_codebooks, 0), device=target_device, dtype=torch.long)
        audio_out_embed = self._embed_audio_ids(audio_out_ids)

        # 3. Merge text, audio-in embeddings, and audio-out embeddings

        # use_cache is turned on during inference time, we should set round_to to 1 to avoid extra padding in the end.
        round_to = 1
        left_padding = bool(use_cache or input_ids.shape[0] == 1)
        (
            inputs_embeds,
            attention_mask,
            labels,
            position_ids,
            input_ids,
            audio_in_mask,
            audio_in_discrete_codes_mask,
            audio_out_mask,
        ) = merge_input_ids_with_audio_features(
            audio_features_embed,
            audio_features_length,
            audio_in_embed,
            audio_in_ids_start,
            audio_out_embed,
            audio_out_ids_start,
            self.audio_in_token_idx,
            self.audio_out_token_idx,
            inputs_embeds,
            input_ids,
            attention_mask,
            label_ids,
            pad_token_id=self.padding_idx,
            round_to=round_to,
            left_padding=left_padding,
        )

        # re-check if we use the correct kv cache bucket after
        # the input_embeds has been merged with audio features
        if past_key_values_buckets is not None and inputs_embeds.shape[1] > past_key_values.get_max_cache_shape():
            past_key_values, self.current_past_key_values_bucket = self._prepare_kv_cache(
                inputs_embeds.shape[1], None, past_key_values_buckets
            )

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
            if isinstance(past_key_values, StaticCache) and past_seen_tokens >= past_key_values.get_max_cache_shape():
                raise ValueError(
                    f"The current sequence length ({past_seen_tokens}) exceeds "
                    f"the maximum cache shape. "
                    f"Please consider increasing the cache size."
                )

        # Use torch compile
        use_static_cache = isinstance(past_key_values, StaticCache)

        # Apply the LLM component
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        audio_discrete_codes_mask = audio_in_discrete_codes_mask | audio_out_mask
        if cache_audio_discrete_codes_mask is not None and use_cache:
            audio_discrete_codes_mask = torch.concat(
                [cache_audio_discrete_codes_mask, audio_discrete_codes_mask], dim=1
            )

        # Generate the audio attention mask outside the layer to avoid recompilation
        if use_static_cache:
            fast_forward_attention_mask, audio_attention_mask = self._prepare_all_static_kv_cache_masks(
                hidden_states, causal_mask, audio_discrete_codes_mask, past_key_values
            )
            # Set the audio out mask to the last token
            if hidden_states.shape[1] == 1:
                audio_discrete_codes_mask = audio_discrete_codes_mask[:, -1:]
                audio_discrete_codes_mask = audio_discrete_codes_mask.reshape((-1, 1)).contiguous()
                is_decoding_audio_token = audio_discrete_codes_mask.item()
            else:
                is_decoding_audio_token = False

        _forward_core = self._forward_core

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        hidden_states, all_hidden_states, all_self_attns = _forward_core(
            hidden_states=hidden_states,
            causal_mask=causal_mask,
            position_ids=position_ids,
            audio_discrete_codes_mask=audio_discrete_codes_mask,
            is_decoding_audio_token=is_decoding_audio_token if use_static_cache else None,
            cache_position=cache_position,
            past_key_values=past_key_values,
            use_cache=use_cache,
            audio_attention_mask=audio_attention_mask if use_static_cache else None,
            fast_forward_attention_mask=fast_forward_attention_mask if use_static_cache else None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Apply the audio decoder projector
        logits, audio_logits, audio_hidden_states = self.audio_decoder_proj(
            hidden_states,
            audio_out_mask,
            output_audio_hidden_states=output_audio_hidden_states,
        )

        if audio_logits is not None:
            audio_logits = audio_logits.view(
                audio_logits.shape[0], self.audio_num_codebooks, self.audio_codebook_size
            ).float()

        next_cache = past_key_values if use_cache else None

        ret = HiggsAudioModelOutputWithPast(
            logits=logits,
            audio_logits=audio_logits,
            expanded_input_ids=input_ids,
            expanded_labels=labels,
            audio_in_mask=audio_in_mask,
            audio_in_discrete_codes_mask=audio_in_discrete_codes_mask,
            audio_out_mask=audio_out_mask,
            attention_mask=attention_mask,
            past_key_values=next_cache,
            last_hidden_states=hidden_states,
            hidden_states=all_hidden_states,
            last_audio_hidden_states=audio_hidden_states,
            attentions=all_self_attns,
        )

        return ret

    def _copy_kv_cache(self, from_cache: Cache, to_cache: Cache):
        num_layers = self.config.text_config.num_hidden_layers
        if self.config.audio_dual_ffn_layers is not None:
            num_layers += len(self.config.audio_dual_ffn_layers)
        """ Copy the key-value pairs from one cache to another. """
        for layer_idx in range(num_layers):
            from_cache_size = from_cache.get_max_cache_shape()
            assert to_cache.get_max_cache_shape() >= from_cache_size, (
                f"The target cache size {to_cache.get_max_cache_shape()} is smaller than the source cache size {from_cache_size}."
            )
            to_cache.key_cache[layer_idx][:, :, :from_cache_size, :] = from_cache.key_cache[layer_idx]
            to_cache.value_cache[layer_idx][:, :, :from_cache_size, :] = from_cache.value_cache[layer_idx]

    def _prepare_kv_cache(
        self,
        current_sequence_length: int,
        current_past_key_values_bucket: Optional[int],
        past_key_values_buckets: OrderedDict[int, Cache],
    ) -> tuple[Optional[Cache], Optional[int]]:
        """Prepare the KV cache for the current sequence length."""
        for cache_length in past_key_values_buckets.keys():
            if cache_length >= current_sequence_length:
                # Promote to the next KV cache bucket, copy the current KV cache bucket
                # to the new one.
                if current_past_key_values_bucket is not None and cache_length != current_past_key_values_bucket:
                    self._copy_kv_cache(
                        past_key_values_buckets[current_past_key_values_bucket], past_key_values_buckets[cache_length]
                    )

                return past_key_values_buckets[cache_length], cache_length

        raise ValueError(
            f"The current sequence length {current_sequence_length} is larger than "
            f"all past key values buckets {past_key_values_buckets.keys()}."
        )


@auto_docstring(
    custom_intro="""
    The Higgs Audio model, a llama-like auto-regressive transformer model with dual-FFN.
    """
)
class HiggsAudioForConditionalGeneration(HiggsAudioPreTrainedModel, HiggsAudioGenerationMixin):
    base_model_prefix = "model"

    def __init__(self, config: HiggsAudioConfig):
        super().__init__(config)
        self.config = config
        self.model = HiggsAudioModel(config)

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        audio_features: Optional[torch.FloatTensor] = None,
        audio_feature_attention_mask: Optional[torch.BoolTensor] = None,
        audio_in_ids: Optional[torch.LongTensor] = None,
        audio_in_ids_start: Optional[torch.LongTensor] = None,
        audio_out_ids: Optional[torch.LongTensor] = None,
        audio_out_ids_start: Optional[torch.LongTensor] = None,
        label_ids: Optional[torch.LongTensor] = None,
        label_audio_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_audio_hidden_states: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        cache_audio_discrete_codes_mask: Optional[torch.LongTensor] = None,
        past_key_values_buckets: Optional[OrderedDict[int, Cache]] = None,
    ):
        r"""
        input_ids (:obj:`torch.LongTensor`):
            The input ids of the prompt. It will have shape (bsz, seq_len).
            When use_cache is enabled, the input_ids will have
            shape (bsz, 1) for incremental decode or None
        attention_mask (:obj:`torch.LongTensor`):
            The attention mask of the prompt. It will have shape (bsz, seq_len).
        audio_features (:obj:`torch.FloatTensor`):
            The audio features extracted by Whisper. It will have shape (num_audio_in, feature_dim, max_mel_seq_len).
        audio_feature_attention_mask (:obj:`torch.LongTensor`):
            The attention mask of the audio features. It will have shape (num_audio_in, max_mel_seq_len).
        audio_in_ids (:obj:`torch.LongTensor`):
            The discretized audio tokens. It will have shape (num_codebooks, audio_in_total_length).
        audio_in_ids_start (:obj:`torch.LongTensor`):
            The start indices for each audio in audio_in_ids. It will have shape (num_audio_in,)
        audio_out_ids (:obj:`torch.LongTensor`):
            The discretized audio tokens. It will have shape (num_codebooks, audio_out_total_length).
        audio_out_ids_start (:obj:`torch.LongTensor`):
            The start indices for each audio in audio_out_ids. It will have shape (num_audio_out,)
        label_ids (:obj:`torch.LongTensor`):
            The labels of the prompt. It will have shape (bsz, seq_len).
        label_audio_ids (:obj:`torch.LongTensor`):
            The labels of the audio tokens. It will have the same shape as audio_out_ids, i.e., (num_codebooks, audio_out_total_length)
        past_key_values (:obj:`Tuple`):
            Tuple of past key values.
        use_cache (:obj:`bool`):
            Whether to use cache.
        output_attentions (:obj:`bool`):
            Whether to output attentions.
        output_hidden_states (:obj:`bool`):
            Whether to output hidden states.
        output_audio_hidden_states (:obj:`bool`):
            Whether to output audio hidden states.
        cache_position (:obj:`torch.LongTensor`):
            The position of the cache.
        cache_audio_discrete_codes_mask (:obj:`torch.LongTensor`):
            The cached audio discrete codes mask. It will only be used when use_cache is turned on.
        past_key_values_buckets (:obj:`OrderedDict`):
            The buckets of past key values.
        """
        target_device = input_ids.device

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_features=audio_features,
            audio_feature_attention_mask=audio_feature_attention_mask,
            audio_in_ids=audio_in_ids,
            audio_in_ids_start=audio_in_ids_start,
            audio_out_ids=audio_out_ids,
            audio_out_ids_start=audio_out_ids_start,
            label_ids=label_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_audio_hidden_states=output_audio_hidden_states,
            cache_position=cache_position,
            cache_audio_discrete_codes_mask=cache_audio_discrete_codes_mask,
            past_key_values_buckets=past_key_values_buckets,
        )

        # Loss calculation when label_ids is not None
        logits = outputs.logits
        labels = outputs.expanded_labels
        audio_in_mask = outputs.audio_in_mask
        audio_in_discrete_codes_mask = outputs.audio_in_discrete_codes_mask
        audio_out_mask = outputs.audio_out_mask
        attention_mask = outputs.attention_mask
        audio_logits = outputs.audio_logits
        past_key_values = outputs.past_key_values
        last_hidden_states = outputs.last_hidden_states
        hidden_states = outputs.hidden_states
        last_audio_hidden_states = outputs.last_audio_hidden_states
        attentions = outputs.attentions

        loss = None
        llm_loss = None
        audio_loss = None
        codebook_losses = None

        # Calculate the loss function
        # There will be two loss functions, one for the text-stream and one for the audio stream.
        if label_ids is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            if (shift_labels == -100).all():
                loss = shift_logits.sum() * 0.0  # Connect the gradient
            else:
                # Flatten the tokens
                shift_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = nn.functional.cross_entropy(shift_logits, shift_labels)

        if label_audio_ids is not None and label_audio_ids.shape[-1] > 0:
            # audio_logits have shape (num_audio_out_tokens, audio_num_codebooks * audio_codebook_size)
            audio_shift_logits = audio_logits[:-1, :, :].contiguous()

            # Ignore the first label token for each audio for proper auto-regressive training.
            # input:                    a1, a2, a3,   b1, b2, b3, b4, c1, d1
            # label (masked):           a1, a2, a3, -100, b2, b3, b4, c1, -100
            # label (shifted):          a2, a3, -100, b2, b3, b4, c1, -100
            # label_audio_ids have shape (num_codebooks, num_audio_out_tokens)
            label_audio_ids[:, audio_out_ids_start] = -100

            audio_shift_labels = label_audio_ids[:, 1:]

            audio_loss_fct = CrossEntropyLoss()
            codebook_losses = torch.zeros([self.model.audio_num_codebooks], device=target_device)
            for codebook in range(self.model.audio_num_codebooks):
                codebook_logits = (
                    audio_shift_logits[:, codebook, :].contiguous().view(-1, self.model.audio_codebook_size)
                )
                codebook_labels = audio_shift_labels[codebook, :].contiguous().to(codebook_logits.device)
                if (codebook_labels == -100).all():
                    codebook_loss = audio_shift_logits.sum() * 0.0  # connect the gradient
                else:
                    codebook_loss = audio_loss_fct(codebook_logits, codebook_labels)
                codebook_losses[codebook] = codebook_loss

            audio_loss = torch.sum(codebook_losses * self.model.audio_codebook_weights.to(target_device))
            loss += audio_loss

        if loss is not None and audio_loss is None:
            llm_loss = loss
        elif loss is not None and audio_loss is not None:
            llm_loss = loss - audio_loss

        next_cache = past_key_values if use_cache else None

        ret = HiggsAudioOutputWithPast(
            loss=loss,
            llm_loss=llm_loss,
            audio_loss=audio_loss,
            codebook_losses=codebook_losses,
            logits=logits,
            audio_logits=audio_logits,
            expanded_input_ids=input_ids,
            expanded_labels=labels,
            audio_in_mask=audio_in_mask,
            audio_in_discrete_codes_mask=audio_in_discrete_codes_mask,
            audio_out_mask=audio_out_mask,
            attention_mask=attention_mask,
            past_key_values=next_cache,
            last_hidden_states=last_hidden_states,
            hidden_states=hidden_states,
            last_audio_hidden_states=last_audio_hidden_states,
            attentions=attentions,
        )

        return ret


__all__ = ["HiggsAudioForConditionalGeneration", "HiggsAudioPreTrainedModel", "HiggsAudioModel", "HiggsAudioEncoder"]
