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


class HiggsAudioRotaryEmbedding(LlamaRotaryEmbedding):
    pass


class HiggsAudioRMSNorm(LlamaRMSNorm):
    pass


class HiggsAudioMLP(LlamaMLP):
    pass


# remove
class HiggsAudioDecoderLayer:
    pass


# remove
class HiggsAudioDualFFNFastDecoderLayer:
    pass


class HiggsAudioAttention(LlamaAttention):
    pass


class HiggsAudioDecoderProjector(nn.Module):
    """Projection layers that map hidden states from the LLM component to audio / text logits."""

    def __init__(self, config: HiggsAudioConfig):
        super().__init__()
        self.text_lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.audio_lm_head = nn.Linear(
            config.hidden_size, config.audio_num_codebooks * (config.audio_codebook_size + 2), bias=False
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


class HiggsAudioDualFFNDecoderLayer(nn.Module):
    """We implement a dual-path FFN decoder layer where the audio tokens and text tokens go through separate FFN layers.

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
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = HiggsAudioAttention(config=config, layer_idx=layer_idx)

        self.mlp = HiggsAudioMLP(config)

        self.audio_mlp = HiggsAudioMLP(config)
        self.audio_input_layernorm = HiggsAudioRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.audio_post_attention_layernorm = HiggsAudioRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.input_layernorm = HiggsAudioRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = HiggsAudioRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = HiggsAudioAttention(config=config, layer_idx=layer_idx)

        self.mlp = HiggsAudioMLP(config)

        self.audio_mlp = HiggsAudioMLP(config)
        self.audio_input_layernorm = HiggsAudioRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.audio_post_attention_layernorm = HiggsAudioRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.input_layernorm = HiggsAudioRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = HiggsAudioRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
            HiggsAudioDecoderLayer, HiggsAudioDualFFNSlowDecoderLayer
        ],
        "attentions": HiggsAudioAttention,
    }

    def _init_weights(self, module):
        if hasattr(self.config, "initializer_range"):
            std = self.config.initializer_range
        else:
            # 0.02 is the standard default value across the library
            std = getattr(self.config, "initializer_range", 0.02)

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
        self.vocab_size = config.vocab_size
        self.audio_num_codebooks = config.audio_num_codebooks
        self.use_delay_pattern = config.use_delay_pattern
        self.use_audio_out_embed_projector = config.use_audio_out_embed_projector

        self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size, self.padding_idx)

        if config.audio_adapter_type == "dual_ffn_fast_forward":
            layers = []
            for j in range(config.num_hidden_layers):
                if j in config.audio_dual_ffn_layers:
                    layers.append(HiggsAudioDualFFNDecoderLayer(config, j))
                else:
                    layers.append(HiggsAudioDualFFNFastDecoderLayer(config, j))
            self.layers = nn.ModuleList(layers)
        elif config.audio_adapter_type == "stack":
            self.layers = nn.ModuleList(
                [HiggsAudioDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
            )

        self.num_activation_checkpointing_layers = len(self.layers)

        self.norm = HiggsAudioRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = HiggsAudioRotaryEmbedding(config=config)

        self.audio_codebook_size = (
            config.audio_codebook_size + 2
        )  # We add 1 for the audio_stream_bos token and 1 for the audio_stream_eos token

        if config.use_audio_out_embed_projector:
            self.audio_out_embed_projector = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.audio_codebook_embeddings = nn.Embedding(
            config.audio_num_codebooks * self.audio_codebook_size, config.hidden_size
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
        audio_embed = torch.sum(audio_embed, dim=1)
        if self.use_audio_out_embed_projector:
            audio_embed = self.audio_out_embed_projector(audio_embed)
        return audio_embed

    def get_audio_embeds(self, audio_input_ids):
        # TODO: should be removed
        return self.get_audio_features(audio_input_ids)

    @check_model_inputs
    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        audio_input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if audio_input_ids is not None:
            audio_inputs_embeds = self.get_audio_embeds(audio_input_ids)

            audio_in_token_mask = input_ids == self.config.audio_in_token_idx
            audio_out_token_mask = input_ids == self.config.audio_out_token_idx

            audio_token_mask = audio_in_token_mask | audio_out_token_mask
            inputs_embeds = inputs_embeds.masked_scatter(audio_token_mask.unsqueeze(-1), audio_inputs_embeds)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
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

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                audio_out_mask=audio_token_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
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
        audio_input_ids: Optional[torch.LongTensor] = None,
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
            audio_input_ids=audio_input_ids,
            **kwargs,
        )

        # Loss calculation when label_ids is not None
        labels = outputs.expanded_labels
        audio_in_mask = outputs.audio_in_mask
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
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

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
            audio_out_mask=audio_out_mask,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            last_hidden_states=last_hidden_states,
            hidden_states=hidden_states,
            attentions=attentions,
        )


__all__ = ["HiggsAudioForConditionalGeneration", "HiggsAudioPreTrainedModel", "HiggsAudioModel"]
