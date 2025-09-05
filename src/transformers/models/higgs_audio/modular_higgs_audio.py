# coding=utf-8
# Copyright 2025 Boson AI and The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from ...cache_utils import Cache, DynamicCache, StaticCache
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    ModelOutput,
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    is_torch_flex_attn_available,
    logging,
)
from ...utils.generic import check_model_inputs
from ..auto import AutoConfig
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from .configuration_higgs_audio import HiggsAudioConfig
from .generation_higgs_audio import HiggsAudioGenerationMixin


if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask

    from ...integrations.flex_attention import make_flex_block_causal_mask


logger = logging.get_logger(__name__)


def merge_input_ids_with_audio_features(
    audio_in_embed,
    audio_in_ids_start,
    audio_out_embed,
    audio_out_ids_start,
    audio_in_token_idx,
    audio_out_token_idx,
    inputs_embeds,
    input_ids,
    attention_mask,
    label_ids,
    pad_token_id,
    ignore_index=-100,
    left_padding=True,
):
    """
    Merge input_ids with audio features into final embeddings.

    Args:
        audio_in_embed (`torch.Tensor` of shape `(total_num_audio_in_tokens, embed_dim)`):
            The embeddings of audio-in tokens
        audio_in_ids_start (`torch.LongTensor` of shape `(num_audios,)`):
            The start index of the audio-in tokens for each audio
        audio_out_embed (`torch.Tensor` of shape `(total_num_audio_out_tokens, embed_dim)`):
            The embeddings of audio-out tokens
        audio_out_ids_start (`torch.LongTensor` of shape `(num_audios,)`):
            The start index of the audio-out tokens for each audio
        audio_in_token_idx
            The index of the audio-in token in the vocabulary
        audio_out_token_idx
            The index of the audio-out token in the vocabulary
        inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, embed_dim)`):
            Token embeddings before merging with audio embeddings
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Input_ids of tokens, possibly filled with audio token
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Mask to avoid performing attention on padding token indices.
        label_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*)
            labels need to be recalculated to support training (if provided)
        pad_token_id (`int`):
            The index of the pad token in the vocabulary
        ignore_index
            The index to ignore in the loss calculation
        left_padding
            Whether to apply left padding

    Returns:
        final_embedding
            The final embeddings after merging audio embeddings with text embeddings.
        final_attention_mask
            The final attention mask after merging audio embeddings with text embeddings.
        final_labels
            The labels for the text stream
        position_ids
            Positional ids for the merged data
        final_input_ids
            The final input_ids after merging audio embeddings with text embeddings.
        final_audio_in_mask
            Mask for audio-in embeddings
        final_audio_in_discrete_codes_mask
            Mask for audio-in discrete tokens
        final_audio_out_mask
            Mask for audio-out embeddings

    Explanation:
        each audio has variable length embeddings, with length specified by
        - audio_in_ids_start
        - audio_out_ids_start

        Task:
        - fill each <|AUDIO|> with audio embeddings from audio codebooks
        - fill each <|AUDIO_OUT|> with the audio-out embeddings

        Example:
            <|AUDIO_OUT|>: X (5 tokens), Y (3 tokens)
            <|AUDIO|>: Z (8 tokens)

            X, Y are in the same sequence (in-context voice-clone). Z is in a different sequence (audio understanding).
        if right padding
            input_ids: [
                a b c d e f X g h i j k Y l m
                o p q r Z s t u v _ _ _ _ _ _
            ]
            input_ids should be: [
                a b c d e f X X X X X g h i j k Y Y Y l m
                o p q r Z Z Z Z Z Z Z Z s t u v _ _ _ _ _
            ]
            labels should be: [
                a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                o p q r _ _ _ _ _ _ _ _ s t u v _ _ _ _ _
            ]
        elif left padding
            input_ids: [
                a b c d e f X g h i j k Y l m
                _ _ _ _ _ _ o p q r Z s t u v
            ]
            input_ids should be: [
                a b c d e f X X X X X g h i j k Y Y Y l m
                _ _ _ _ _ o p q r Z Z Z Z Z Z Z Z s t u v
            ]
            labels should be: [
                a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                _ _ _ _ _ o p q r _ _ _ _ _ _ _ _ s t u v
            ]

    """
    if label_ids is None:
        skip_labels = True
    else:
        skip_labels = False
    if audio_in_embed is not None and audio_in_embed.shape[0] == 0:
        audio_in_embed = None
    if audio_out_embed is not None and audio_out_embed.shape[0] == 0:
        audio_out_embed = None
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    batch_size, sequence_length, embed_dim = inputs_embeds.shape

    target_device = inputs_embeds.device
    if left_padding is None:
        left_padding = torch.any(attention_mask[:, 0] == 0)

    audio_in_token_mask = input_ids == audio_in_token_idx
    audio_out_token_mask = input_ids == audio_out_token_idx
    text_token_mask = (input_ids != audio_in_token_idx) & (input_ids != audio_out_token_idx)

    # 1. Calculate the number of tokens for each placeholder (like [<|AUDIO|>, <|AUDIO_OUT|>]).
    token_placeholder_num = torch.ones_like(input_ids)

    if audio_in_embed is not None:
        audio_in_codes_length = torch.concat(
            [
                audio_in_ids_start[1:] - audio_in_ids_start[:-1],
                torch.tensor(
                    [audio_in_embed.shape[0] - audio_in_ids_start[-1]],
                    device=audio_in_ids_start.device,
                    dtype=torch.long,
                ),
            ],
            dim=0,
        )
        token_placeholder_num[audio_in_token_mask] = audio_in_codes_length.long()

    if audio_out_embed is not None:
        audio_out_codes_length = torch.concat(
            [
                audio_out_ids_start[1:] - audio_out_ids_start[:-1],
                torch.tensor(
                    [audio_out_embed.shape[0] - audio_out_ids_start[-1]],
                    device=audio_out_ids_start.device,
                    dtype=torch.long,
                ),
            ],
            dim=0,
        )
        token_placeholder_num[audio_out_token_mask] = audio_out_codes_length.long()

    new_token_positions = torch.cumsum(token_placeholder_num, -1) - 1
    max_token_num = token_placeholder_num.sum(-1).max()
    nb_audio_pad = max_token_num - 1 - new_token_positions[:, -1]

    if left_padding:
        new_token_positions += nb_audio_pad[:, None]  # offset for left padding

    # 2. Create the full embedding, already padded to the maximum position
    final_embedding = torch.zeros(
        (batch_size, max_token_num, embed_dim), dtype=inputs_embeds.dtype, device=inputs_embeds.device
    )
    final_attention_mask = torch.zeros(
        (batch_size, max_token_num), dtype=attention_mask.dtype, device=inputs_embeds.device
    )
    final_input_ids = torch.full(
        (batch_size, max_token_num), pad_token_id, dtype=input_ids.dtype, device=inputs_embeds.device
    )
    if skip_labels:
        final_labels = None
    else:
        final_labels = torch.full(
            (batch_size, max_token_num), ignore_index, dtype=label_ids.dtype, device=inputs_embeds.device
        )

    final_audio_in_mask = torch.full((batch_size, max_token_num), False, dtype=torch.bool, device=inputs_embeds.device)
    final_audio_in_discrete_codes_mask = torch.full(
        (batch_size, max_token_num), False, dtype=torch.bool, device=inputs_embeds.device
    )
    final_audio_out_mask = torch.full(
        (batch_size, max_token_num), False, dtype=torch.bool, device=inputs_embeds.device
    )
    # 3. Get the audio-in token positions and audio-out token positions
    batch_id = torch.arange(batch_size, device=target_device).unsqueeze(1).expand(batch_size, sequence_length)
    audio_in_batch_id = batch_id[audio_in_token_mask]  # Shape (num_audio_in,)
    audio_out_batch_id = batch_id[audio_out_token_mask]  # Shape (num_audio_out,)
    audio_features_token_ends = new_token_positions[audio_in_token_mask]  # Shape (num_audio_in,)
    audio_out_embed_ends = new_token_positions[audio_out_token_mask]  # Shape (num_audio_out,)

    if audio_in_embed is not None:
        # Fill in the audio-in embeddings
        seq_indices = (
            torch.arange(max_token_num, device=target_device)
            .unsqueeze(0)
            .expand(audio_in_ids_start.shape[0], max_token_num)
        )
        audio_in_embed_token_starts = audio_features_token_ends - audio_in_codes_length + 1
        batch_indices, col_indices = torch.where(
            (seq_indices >= audio_in_embed_token_starts.unsqueeze(1))
            & (seq_indices <= audio_features_token_ends.unsqueeze(1))
        )
        batch_indices = audio_in_batch_id[batch_indices]
        final_embedding[batch_indices, col_indices] = audio_in_embed
        final_input_ids[batch_indices, col_indices] = audio_in_token_idx
        if not skip_labels:
            final_labels[batch_indices, col_indices] = ignore_index
        final_audio_in_mask[batch_indices, col_indices] = True
        final_audio_in_discrete_codes_mask[batch_indices, col_indices] = True

    if audio_out_embed is not None:
        # Fill in the audio-out embeddings
        seq_indices = (
            torch.arange(max_token_num, device=target_device)
            .unsqueeze(0)
            .expand(audio_out_ids_start.shape[0], max_token_num)
        )
        audio_out_embed_token_starts = audio_out_embed_ends - audio_out_codes_length + 1
        batch_indices, col_indices = torch.where(
            (seq_indices >= audio_out_embed_token_starts.unsqueeze(1))
            & (seq_indices <= audio_out_embed_ends.unsqueeze(1))
        )
        batch_indices = audio_out_batch_id[batch_indices]
        final_embedding[batch_indices, col_indices] = audio_out_embed
        final_input_ids[batch_indices, col_indices] = audio_out_token_idx
        if not skip_labels:
            final_labels[batch_indices, col_indices] = ignore_index
        final_audio_out_mask[batch_indices, col_indices] = True

    # Fill in the original text embeddings and labels
    batch_indices, non_audio_indices = torch.where(text_token_mask)
    text_to_overwrite = new_token_positions[batch_indices, non_audio_indices]
    final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_audio_indices]
    if not skip_labels:
        final_labels[batch_indices, text_to_overwrite] = label_ids[batch_indices, non_audio_indices]
    final_input_ids[batch_indices, text_to_overwrite] = input_ids[batch_indices, non_audio_indices]
    final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_audio_indices]
    final_attention_mask = final_attention_mask | final_audio_in_mask | final_audio_out_mask

    # Trim the tensor if there are redundant padding tokens
    if left_padding:
        first_non_zero_loc = final_attention_mask.sum(0).nonzero()[0]
        if first_non_zero_loc > 0:
            final_attention_mask = final_attention_mask[:, first_non_zero_loc:]
            final_embedding = final_embedding[:, first_non_zero_loc:]
            if not skip_labels:
                final_labels = final_labels[:, first_non_zero_loc:]
            final_input_ids = final_input_ids[:, first_non_zero_loc:]
            final_audio_in_mask = final_audio_in_mask[:, first_non_zero_loc:]
            final_audio_in_discrete_codes_mask = final_audio_in_discrete_codes_mask[:, first_non_zero_loc:]
            final_audio_out_mask = final_audio_out_mask[:, first_non_zero_loc:]
    else:
        # We have done right padding, so we need to trim the mask
        last_non_zero_loc = final_attention_mask.sum(0).nonzero()[-1] + 1
        if last_non_zero_loc < max_token_num:
            final_attention_mask = final_attention_mask[:, :last_non_zero_loc]
            final_embedding = final_embedding[:, :last_non_zero_loc]
            if not skip_labels:
                final_labels = final_labels[:, :last_non_zero_loc]
            final_input_ids = final_input_ids[:, :last_non_zero_loc]
            final_audio_in_mask = final_audio_in_mask[:, :last_non_zero_loc]
            final_audio_in_discrete_codes_mask = final_audio_in_discrete_codes_mask[:, :last_non_zero_loc]
            final_audio_out_mask = final_audio_out_mask[:, :last_non_zero_loc]

    position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)
    return (
        final_embedding,
        final_attention_mask,
        final_labels,
        position_ids,
        final_input_ids,
        final_audio_in_mask,
        final_audio_in_discrete_codes_mask,
        final_audio_out_mask,
    )


class HiggsAudioRotaryEmbedding(LlamaRotaryEmbedding):
    pass


class HiggsAudioRMSNorm(LlamaRMSNorm):
    pass


class HiggsAudioMLP(LlamaMLP):
    pass


class HiggsAudioDecoderLayer(LlamaDecoderLayer):
    pass


class HiggsAudioAttention(LlamaAttention):
    pass


class HiggsAudioDecoderProjector(nn.Module):
    """Projection layers that map hidden states from the LLM component to audio / text logits."""

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
    ):
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, seq_len, hidden_size)`):
                Hidden states from the LLM component
            audio_out_mask (`torch.Tensor` of shape `(batch_size, seq_len)`):
                Mask for identifying the audio out tokens.

        Returns:
            logits (`torch.Tensor` of shape `(batch_size, seq_len, vocab_size)`):
                Logits for text tokens
            audio_logits (`torch.Tensor` of shape `(num_audio_out_tokens, audio_num_codebooks * audio_codebook_size)`):
                Logits for audio tokens. We ensure `num_text_tokens + num_audio_tokens == batch_size * seq_len`
        """
        logits = self.text_lm_head(hidden_states)

        audio_logits = self.audio_lm_head(hidden_states[audio_out_mask])

        return logits, audio_logits


class HiggsAudioDualFFNFastDecoderLayer(nn.Module):
    """We implement a fast dual-path FFN decoder layer where the audio tokens are skipped.

    Following is an illustration:

     t    t    t    a   a    a    t    t    t
                        |
                        | (shared attention layer)
                        v
    h_t  h_t  h_t  a  a  a  h_t  h_t  h_t
                        |
                        |
                        v
    [h_t  h_t  h_t  h_t  h_t  h_t], [a, a, a]
             |                             |
             |                             |
             v                             v
    [o_t  o_t  o_t  o_t  o_t  o_t], [a, a, a]
                        |
                        | (reorder)
                        v
    o_t  o_t  o_t  a  a  a  o_t  o_t  o_t
    """

    def __init__(self, config: AutoConfig, layer_idx: int):
        super().__init__()
        text_config = config
        self.hidden_size = text_config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = HiggsAudioAttention(config=text_config, layer_idx=layer_idx)

        self.mlp = HiggsAudioMLP(text_config)

        self.input_layernorm = HiggsAudioRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
        self.post_attention_layernorm = HiggsAudioRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        fast_forward_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        audio_out_mask: Optional[torch.BoolTensor] = None,
        is_decoding_audio_token: Optional[bool] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs: Unpack[TransformersKwargs],
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
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
        """
        # If we are decoding an audio token and the layer is marked as fast-forward,
        # we can skip it.
        if is_decoding_audio_token:
            return hidden_states

        residual = hidden_states
        target_length = hidden_states.shape[1]
        use_static_cache = isinstance(past_key_value, StaticCache)

        has_audio_out = audio_out_mask is not None and audio_out_mask.shape[0] > 0

        audio_out_mask_sq = audio_out_mask

        if has_audio_out:
            original_hidden_states = hidden_states.clone()
            min_dtype = torch.finfo(hidden_states.dtype).min
            if attention_mask is None:
                attention_mask = ~audio_out_mask
                if self.self_attn.config._attn_implementation != "flash_attention_2":
                    sequence_length = audio_out_mask.shape[1]
                    attention_mask = HiggsAudioModel._prepare_4d_causal_attention_mask_with_cache_position(
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

        hidden_states = self.input_layernorm(hidden_states)

        # Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Apply Dual-path FFN
        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if has_audio_out:
            if use_cache:
                hidden_states = torch.where(
                    audio_out_mask_sq[:, -target_length:].unsqueeze(-1), original_hidden_states, hidden_states
                )
            else:
                hidden_states = torch.where(audio_out_mask_sq.unsqueeze(-1), original_hidden_states, hidden_states)

        return hidden_states


class HiggsAudioDualFFNSlowDecoderLayer(nn.Module):
    """We implement a slow dual-path FFN decoder layer where the audio tokens and text tokens go through separate FFN layers.

    The audio and text tokens share the text-attention layer, but will be encoded with separate feedforward layers.

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

    def __init__(self, config: AutoConfig, layer_idx: int):
        super().__init__()
        text_config = config
        self.hidden_size = text_config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = HiggsAudioAttention(config=text_config, layer_idx=layer_idx)

        self.mlp = HiggsAudioMLP(text_config)

        self.audio_mlp = HiggsAudioMLP(text_config)
        self.audio_input_layernorm = HiggsAudioRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
        self.audio_post_attention_layernorm = HiggsAudioRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)

        self.input_layernorm = HiggsAudioRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
        self.post_attention_layernorm = HiggsAudioRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        audio_out_mask: Optional[torch.BoolTensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs: Unpack[TransformersKwargs],
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
            past_key_value (`Cache`, *optional*): cached past key and value projection states. We fetch the corresponding cached key/value via the layer_idx.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
        """
        residual = hidden_states
        target_length = hidden_states.shape[1]

        has_audio_out = audio_out_mask is not None and audio_out_mask.shape[0] > 0

        if has_audio_out:
            # Apply separate layernorm layers for audio tokens and text tokens
            if use_cache:
                hidden_states = torch.where(
                    audio_out_mask[:, -target_length:].unsqueeze(-1),
                    self.audio_input_layernorm(hidden_states),
                    self.input_layernorm(hidden_states),
                )
            else:
                hidden_states = torch.where(
                    audio_out_mask.unsqueeze(-1),
                    self.audio_input_layernorm(hidden_states),
                    self.input_layernorm(hidden_states),
                )
        else:
            hidden_states = self.input_layernorm(hidden_states)

        # Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Apply Dual-path FFN
        residual = hidden_states

        if has_audio_out:
            if use_cache:
                real_audio_out_mask = audio_out_mask[:, -target_length:]
            else:
                real_audio_out_mask = audio_out_mask

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

        return hidden_states


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

    _can_record_outputs = {
        "hidden_states": Union[
            HiggsAudioDecoderLayer, HiggsAudioDualFFNFastDecoderLayer, HiggsAudioDualFFNSlowDecoderLayer
        ],
        "attentions": HiggsAudioAttention,
    }

    def _init_weights(self, module):
        if hasattr(self.config, "initializer_range"):
            std = self.config.initializer_range
        else:
            # 0.02 is the standard default value across the library
            std = getattr(self.config.get_text_config(), "initializer_range", 0.02)

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
        elif isinstance(module, HiggsAudioRMSNorm):
            module.weight.data.fill_(1.0)


@dataclass
class HiggsAudioModelOutputWithPast(ModelOutput):
    """
    Output type for HiggsAudioModel when using past key values (for fast decoding)
    and multi-modal audio-text processing.

    Attributes:
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

        past_key_values (Optional[Cache]):
            Pre-computed key and value tensors for each attention layer, used to speed up
            generation by avoiding re-computation.

        last_hidden_states (Optional[torch.FloatTensor]):
            The model’s hidden state output.

        hidden_states (Optional[Tuple[torch.FloatTensor, ...]]):
            Tuple of hidden states from each layer of the transformer decoder.

        attentions (Optional[Tuple[torch.FloatTensor, ...]]):
            Tuple of attention weight tensors from each layer, each of shape
            `(batch_size, num_heads, seq_length, seq_length)`.
    """

    expanded_input_ids: Optional[torch.LongTensor] = None
    expanded_labels: Optional[torch.LongTensor] = None
    audio_in_mask: Optional[torch.BoolTensor] = None
    audio_in_discrete_codes_mask: Optional[torch.BoolTensor] = None
    audio_out_mask: Optional[torch.BoolTensor] = None
    attention_mask: Optional[torch.BoolTensor] = None
    past_key_values: Optional[Cache] = None
    last_hidden_states: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
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

        self.embed_tokens = nn.Embedding(self.vocab_size, config.text_config.hidden_size, self.padding_idx)
        self.config.text_config._attn_implementation = self.config._attn_implementation

        if config.audio_adapter_type == "dual_ffn":
            layers = []
            for j in range(config.text_config.num_hidden_layers):
                if j in config.audio_dual_ffn_layers:
                    layers.append(HiggsAudioDualFFNSlowDecoderLayer(config.text_config, j))
                else:
                    layers.append(HiggsAudioDecoderLayer(config.text_config, j))
            self.layers = nn.ModuleList(layers)
        elif config.audio_adapter_type == "dual_ffn_fast_forward":
            layers = []
            for j in range(config.text_config.num_hidden_layers):
                if j in config.audio_dual_ffn_layers:
                    layers.append(
                        HiggsAudioDualFFNSlowDecoderLayer(
                            config.text_config,
                            j,
                        )
                    )
                else:
                    layers.append(HiggsAudioDualFFNFastDecoderLayer(config.text_config, j))
            self.layers = nn.ModuleList(layers)
        elif config.audio_adapter_type == "stack":
            self.layers = nn.ModuleList(
                [
                    HiggsAudioDecoderLayer(config.text_config, layer_idx)
                    for layer_idx in range(config.text_config.num_hidden_layers)
                ]
            )

        self.num_activation_checkpointing_layers = len(self.layers)

        self.norm = HiggsAudioRMSNorm(config.text_config.hidden_size, eps=config.text_config.rms_norm_eps)
        self.rotary_emb = HiggsAudioRotaryEmbedding(config=config.text_config)

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

        self.gradient_checkpointing = False

        self.post_init()

    def get_audio_features(self, audio_ids):
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
        audio_embed = torch.sum(audio_embed, dim=0)
        if self.use_audio_out_embed_projector:
            audio_embed = self.audio_out_embed_projector(audio_embed)
        return audio_embed

    def _prepare_all_static_kv_cache_masks(self, hidden_states, attention_mask, audio_out_mask, past_key_values):
        target_length = hidden_states.shape[1]
        cur_pos = audio_out_mask.shape[1]
        min_dtype = torch.finfo(hidden_states.dtype).min
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

    # Copied from transformers.models.gptj.modeling_gptj.GPTJModel._update_causal_mask
    def _update_causal_mask(
        self,
        attention_mask: Union[torch.Tensor, "BlockMask"],
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None
        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            return attention_mask

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_compilable_cache = past_key_values.is_compileable if past_key_values is not None else False

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_compilable_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype = input_tensor.dtype
        sequence_length = input_tensor.shape[1]
        if using_compilable_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu", "npu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    # Copied from transformers.models.gptj.modeling_gptj.GPTJModel._prepare_4d_causal_attention_mask_with_cache_position
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask

    @check_model_inputs
    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        audio_in_ids: Optional[torch.LongTensor] = None,
        audio_in_ids_start: Optional[torch.LongTensor] = None,
        audio_out_ids: Optional[torch.LongTensor] = None,
        audio_out_ids_start: Optional[torch.LongTensor] = None,
        label_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        cache_audio_discrete_codes_mask: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        input_ids (:obj:`torch.LongTensor`):
            The input ids of the prompt. It will have shape (bsz, seq_len).
            When use_cache is enabled, the input_ids will have
            shape (bsz, 1) for incremental decode or None
        attention_mask (:obj:`torch.LongTensor`):
            The attention mask of the prompt. It will have shape (bsz, seq_len).
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
        cache_position (:obj:`torch.LongTensor`):
            The position of the cache.
        cache_audio_discrete_codes_mask (:obj:`torch.LongTensor`):
            The cached audio discrete codes mask. It will only be used when use_cache is turned on.
        """
        target_device = input_ids.device

        # 1. Extract the input embeddings
        inputs_embeds = self.embed_tokens(input_ids)

        # 2. Extract audio embeddings
        if self.config.encode_audio_in_tokens:
            if audio_in_ids is not None and audio_in_ids.shape[-1] > 0:
                audio_in_ids = audio_in_ids.to(target_device)
            else:
                audio_in_ids = torch.zeros((self.audio_num_codebooks, 0), device=target_device, dtype=torch.long)
            audio_in_embed = self.get_audio_features(audio_in_ids)
        else:
            audio_in_embed = None

        if audio_out_ids is not None and audio_out_ids.shape[-1] > 0:
            audio_out_ids = audio_out_ids.to(target_device)
        else:
            audio_out_ids = torch.zeros((self.audio_num_codebooks, 0), device=target_device, dtype=torch.long)
        audio_out_embed = self.get_audio_features(audio_out_ids)

        # 3. Merge text, audio-in embeddings, and audio-out embeddings

        # use_cache is turned on during inference time
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
            left_padding=left_padding,
        )

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # Use torch compile
        use_static_cache = isinstance(past_key_values, StaticCache)

        # Apply the LLM component
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, kwargs.get("output_attentions", False)
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

        # Create position embeddings to be shared across the decoder layers
        # When past_key_values is passed in, we need to offset the position ids when calculating the position embeddings.
        # Therefore, cache_position is used.
        position_id_offset = cache_position[0] if use_cache else 0
        position_embeddings = self.rotary_emb(hidden_states, position_ids + position_id_offset)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                audio_attention_mask=audio_attention_mask if use_static_cache else None,
                fast_forward_attention_mask=fast_forward_attention_mask if use_static_cache else None,
                position_ids=position_ids,
                audio_out_mask=audio_discrete_codes_mask,
                is_decoding_audio_token=is_decoding_audio_token if use_static_cache else None,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return HiggsAudioModelOutputWithPast(
            expanded_input_ids=input_ids,
            expanded_labels=labels,
            audio_in_mask=audio_in_mask,
            audio_in_discrete_codes_mask=audio_in_discrete_codes_mask,
            audio_out_mask=audio_out_mask,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            last_hidden_states=hidden_states,
        )

    def _copy_kv_cache(self, from_cache: Cache, to_cache: Cache):
        num_layers = self.config.text_config.num_hidden_layers
        if self.config.audio_dual_ffn_layers is not None:
            num_layers += len(self.config.audio_dual_ffn_layers)
        """ Copy the key-value pairs from one cache to another. """
        for layer_idx in range(num_layers):
            from_cache_size = from_cache.get_max_cache_shape()
            to_cache.key_cache[layer_idx][:, :, :from_cache_size, :] = from_cache.key_cache[layer_idx]
            to_cache.value_cache[layer_idx][:, :, :from_cache_size, :] = from_cache.value_cache[layer_idx]


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
        self.audio_decoder_proj = HiggsAudioDecoderProjector(config)
        self.audio_codebook_weights = (
            torch.ones(config.audio_num_codebooks) / config.audio_num_codebooks
        )  # default to equal weights

        self.post_init()

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        audio_in_ids: Optional[torch.LongTensor] = None,
        audio_in_ids_start: Optional[torch.LongTensor] = None,
        audio_out_ids: Optional[torch.LongTensor] = None,
        audio_out_ids_start: Optional[torch.LongTensor] = None,
        label_ids: Optional[torch.LongTensor] = None,
        label_audio_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        cache_audio_discrete_codes_mask: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        input_ids (:obj:`torch.LongTensor`):
            The input ids of the prompt. It will have shape (bsz, seq_len).
            When use_cache is enabled, the input_ids will have
            shape (bsz, 1) for incremental decode or None
        attention_mask (:obj:`torch.LongTensor`):
            The attention mask of the prompt. It will have shape (bsz, seq_len).
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
        cache_position (:obj:`torch.LongTensor`):
            The position of the cache.
        cache_audio_discrete_codes_mask (:obj:`torch.LongTensor`):
            The cached audio discrete codes mask. It will only be used when use_cache is turned on.
        """
        target_device = input_ids.device

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_in_ids=audio_in_ids,
            audio_in_ids_start=audio_in_ids_start,
            audio_out_ids=audio_out_ids,
            audio_out_ids_start=audio_out_ids_start,
            label_ids=label_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            cache_audio_discrete_codes_mask=cache_audio_discrete_codes_mask,
            **kwargs,
        )

        # Loss calculation when label_ids is not None
        labels = outputs.expanded_labels
        audio_in_mask = outputs.audio_in_mask
        audio_in_discrete_codes_mask = outputs.audio_in_discrete_codes_mask
        audio_out_mask = outputs.audio_out_mask
        attention_mask = outputs.attention_mask
        past_key_values = outputs.past_key_values
        last_hidden_states = outputs.last_hidden_states
        hidden_states = outputs.hidden_states
        attentions = outputs.attentions

        # Apply the audio decoder projector
        logits, audio_logits = self.audio_decoder_proj(
            last_hidden_states,
            audio_out_mask,
        )

        if audio_logits is not None:
            audio_logits = audio_logits.view(
                audio_logits.shape[0], self.model.audio_num_codebooks, self.model.audio_codebook_size
            ).float()

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
            codebook_losses = torch.zeros([self.config.audio_num_codebooks], device=target_device)
            for codebook in range(self.config.audio_num_codebooks):
                codebook_logits = (
                    audio_shift_logits[:, codebook, :].contiguous().view(-1, self.model.audio_codebook_size)
                )
                codebook_labels = audio_shift_labels[codebook, :].contiguous().to(codebook_logits.device)
                if (codebook_labels == -100).all():
                    codebook_loss = audio_shift_logits.sum() * 0.0  # connect the gradient
                else:
                    codebook_loss = audio_loss_fct(codebook_logits, codebook_labels)
                codebook_losses[codebook] = codebook_loss

            audio_loss = torch.sum(codebook_losses * self.audio_codebook_weights.to(target_device))
            loss += audio_loss

        if loss is not None and audio_loss is None:
            llm_loss = loss
        elif loss is not None and audio_loss is not None:
            llm_loss = loss - audio_loss

        return HiggsAudioOutputWithPast(
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
            past_key_values=past_key_values,
            last_hidden_states=last_hidden_states,
            hidden_states=hidden_states,
            attentions=attentions,
        )


__all__ = ["HiggsAudioForConditionalGeneration", "HiggsAudioPreTrainedModel", "HiggsAudioModel"]
