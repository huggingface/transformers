# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
""" PyTorch CLVP model."""


from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel, SequenceSummary
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_clvp import CLVPConfig, CLVPSpeechConfig, CLVPTextConfig, CLVPAutoRegressiveConfig, PretrainedConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "susnato/clvp_dev"

CLVP_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "susnato/clvp_dev",
    # See all CLVP models at https://huggingface.co/models?filter=clvp
]


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


# Copied from transformers.models.clip.modeling_clip.contrastive_loss
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


# Copied from transformers.models.clip.modeling_clip.clip_loss with clip->clvp, image_loss->speech_loss
def clvp_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    speech_loss = contrastive_loss(similarity.t())
    return (caption_loss + speech_loss) / 2.0


def rotate_half(state):
    """
    This method splits the state into two parts and then rotates the second part by 90 degrees.
    """
    state_shape = state.size()
    state = state.view([*state_shape[:-1]] + [2, state_shape[-1] // 2])
    state_part_1, state_part_2 = state.unbind(dim=-2)
    return torch.cat((-state_part_2, state_part_1), dim=-1)


def apply_rotary_pos_emb(state, freqs):
    """
    Applies rotary position embeddings on state.
    """
    pos_emb_len = freqs.shape[-1]
    state_l, state_r = state[..., :pos_emb_len], state[..., pos_emb_len:]

    freqs = freqs[:, :, -state_l.shape[-1] :]
    state_l = (state_l * freqs.cos()) + (rotate_half(state_l) * freqs.sin())

    return torch.cat([state_l, state_r], dim=-1)


class CLVPRMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization for CLVP. Please refer to the paper https://arxiv.org/abs/1910.07467 to know
    more.

    Args:
        normalized_shape(`int`):
            The dimension of the data to be normalized.
        eps(`float`):
            The epsilon used by the RMS Norm.
    """

    def __init__(self, normalized_shape, eps):
        super().__init__()
        self.scale = normalized_shape**-0.5
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, hidden_states):
        norm = torch.norm(hidden_states, dim=-1, keepdim=True) * self.scale
        return hidden_states / norm.clamp(min=self.eps) * self.gain


@dataclass
class CLVPSpeechModelOutput(ModelOutput):
    """
    Base class for speech model's outputs that also contains speech embeddings of the pooling of the last hidden
    states.

    Args:
        speech_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The speech embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Pooled output of the `last_hidden_state`.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    speech_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    pooler_output: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
# Copied from transformers.models.clip.modeling_clip.CLIPTextModelOutput with CLIP->CLVP
class CLVPTextModelOutput(ModelOutput):
    """
    Base class for text model's outputs that also contains a pooling of the last hidden states.

    Args:
        text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The text embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Pooled output of the `last_hidden_state`.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    text_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    pooler_output: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class CLVPOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for speech-text similarity.
        logits_per_speech:(`torch.FloatTensor` of shape `(speech_batch_size, text_batch_size)`):
            The scaled dot product scores between `speech_embeds` and `text_embeds`. This represents the speech-text
            similarity scores.
        logits_per_text:(`torch.FloatTensor` of shape `(text_batch_size, speech_batch_size)`):
            The scaled dot product scores between `text_embeds` and `speech_embeds`. This represents the text-speech
            similarity scores.
        text_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of the Text Model.
        speech_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The speech embeddings obtained by applying the projection layer to the pooled output of the Speech Model.
        text_model_output(`BaseModelOutputWithPooling`):
            The pooled output of the `last_hidden_state` of the Text Model.
        speech_model_output(`BaseModelOutputWithPooling`):
            The pooled output of the `last_hidden_state` of the Speech Model.
    """

    loss: Optional[torch.FloatTensor] = None
    logits_per_speech: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    speech_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    speech_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "speech_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class CLVPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    # Copied from transformers.models.clip.modeling_clip.CLIPAttention._shape
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        if rotary_pos_emb is not None:
            query_states = apply_rotary_pos_emb(query_states, rotary_pos_emb)
            key_states = apply_rotary_pos_emb(key_states, rotary_pos_emb)
            value_states = apply_rotary_pos_emb(value_states, rotary_pos_emb)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


class CLVPGatedLinearUnit(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.activation_fn = ACT2FN[config.hidden_act]
        self.proj = nn.Linear(config.hidden_size, config.intermediate_size * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.activation_fn(gate)


class CLVPMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.fc1 = CLVPGatedLinearUnit(config)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout_layer = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.dropout_layer(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLVPRotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding Class for CLVP. It was proposed in the paper 'ROFORMER: ENHANCED TRANSFORMER WITH ROTARY
    POSITION EMBEDDING', Please see https://arxiv.org/pdf/2104.09864v1.pdf .
    """

    def __init__(self, config):
        super().__init__()
        dim = max(config.projection_dim // (config.num_attention_heads * 2), 32)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))

        self.register_buffer("inv_freq", inv_freq)
        self.cached_sequence_length = None
        self.cached_rotary_positional_embedding = None

    def forward(self, hidden_states):
        sequence_length = hidden_states.shape[1]

        if sequence_length == self.cached_sequence_length and self.cached_rotary_positional_embedding is not None:
            return self.cached_rotary_positional_embedding

        self.cached_sequence_length = sequence_length
        time_stamps = torch.arange(sequence_length).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", time_stamps, self.inv_freq)
        embeddings = torch.cat((freqs, freqs), dim=-1)

        self.cached_rotary_positional_embedding = embeddings.unsqueeze(0)
        return self.cached_rotary_positional_embedding


class CLVPEncoderLayer(nn.Module):
    def __init__(self, config: CLVPConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.self_attn = CLVPAttention(config)
        self.mlp = CLVPMLP(config)

        self.pre_branch_norm1 = CLVPRMSNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.pre_branch_norm2 = CLVPRMSNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.pre_branch_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_branch_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# This is mostly ported from T5Attention with some changes to ensure it behaves as expected.
class CLVPRelativeAttention:
    """
    CLVP Relative Attention is used in `CLVPConditioningEncoder` to process log-mel spectrograms.
    This also uses relative position embeddings.
    """
    def __init__(self, config: CLVPAutoRegressiveConfig):
        super().__init__()
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.n_embd = config.n_embd
        self.n_heads = config.n_head
        self.dropout = config.dropout_rate
        self.q = nn.Linear(self.n_embd, self.n_embd, bias=True)
        self.k = nn.Linear(self.n_embd, self.n_embd, bias=True)
        self.v = nn.Linear(self.n_embd, self.n_embd, bias=True)
        self.o = nn.Linear(self.n_embd, self.n_embd, bias=True)
        self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()
        self.gradient_checkpointing = False

        def prune_heads(self, heads):
            if len(heads) == 0:
                return
            heads, index = find_pruneable_heads_and_indices(
                heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
            )
            # Prune linear layers
            self.q = prune_linear_layer(self.q, index)
            self.k = prune_linear_layer(self.k, index)
            self.v = prune_linear_layer(self.v, index)
            self.o = prune_linear_layer(self.o, index, dim=1)
            # Update hyper params
            self.n_heads = self.n_heads - len(heads)
            self.pruned_heads = self.pruned_heads.union(heads)

        @staticmethod
        def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
            """
            Adapted from Mesh Tensorflow:
            https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
            Translate relative position to a bucket number for relative attention. The relative position is defined as
            memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
            position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
            small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
            positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
            This should allow for more graceful generalization to longer sequences than the model has been trained on
            Args:
                relative_position: an int32 Tensor
                bidirectional: a boolean - whether the attention is bidirectional
                num_buckets: an integer
                max_distance: an integer
            Returns:
                a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
            """
            relative_buckets = 0
            if bidirectional:
                num_buckets //= 2
                relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
                relative_position = torch.abs(relative_position)
            else:
                relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
            # now relative_position is in the range [0, inf)

            # half of the buckets are for exact increments in positions
            max_exact = num_buckets // 2
            is_small = relative_position < max_exact

            # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
            relative_position_if_large = max_exact + (
                    torch.log(relative_position.float() / max_exact)
                    / math.log(max_distance / max_exact)
                    * (num_buckets - max_exact)
            ).to(torch.long)
            relative_position_if_large = torch.min(
                relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
            )

            relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
            return relative_buckets

        def compute_bias(self, query_length, key_length, device=None):
            """Compute binned relative position bias"""
            if device is None:
                device = self.relative_attention_bias.weight.device
            context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
            memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
            relative_position = memory_position - context_position  # shape (query_length, key_length)
            relative_position_bucket = self._relative_position_bucket(
                relative_position,  # shape (query_length, key_length)
                bidirectional=False,
                num_buckets=self.relative_attention_num_buckets,
                max_distance=self.relative_attention_max_distance,
            )
            values = self.relative_attention_bias(
                relative_position_bucket)  # shape (query_length, key_length, num_heads)
            values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
            return values

        def forward(
                self,
                hidden_states,
                mask=None,
                key_value_states=None,
                position_bias=None,
                past_key_value=None,
                layer_head_mask=None,
                query_length=None,
                use_cache=False,
                output_attentions=False,
        ):
            """
            Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
            """
            # Input is (batch_size, seq_length, dim)
            # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
            # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
            batch_size, seq_length = hidden_states.shape[:2]

            real_seq_length = seq_length

            if past_key_value is not None:
                if len(past_key_value) != 2:
                    raise ValueError(
                        f"past_key_value should have 2 past states: keys and values. Got {len(past_key_value)} past states"
                    )
                real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

            key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

            def shape(states):
                """projection"""
                return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

            def unshape(states):
                """reshape"""
                return states.transpose(1, 2).contiguous().view(batch_size, -1, self.n_embd)

            def project(hidden_states, proj_layer, key_value_states, past_key_value):
                """projects hidden states correctly to key/query states"""
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(hidden_states))
                elif past_key_value is None:
                    # cross-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(key_value_states))

                if past_key_value is not None:
                    if key_value_states is None:
                        # self-attn
                        # (batch_size, n_heads, key_length, dim_per_head)
                        hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                    elif past_key_value.shape[2] != key_value_states.shape[1]:
                        # checking that the `sequence_length` of the `past_key_value` is the same as
                        # the provided `key_value_states` to support prefix tuning
                        # cross-attn
                        # (batch_size, n_heads, seq_length, dim_per_head)
                        hidden_states = shape(proj_layer(key_value_states))
                    else:
                        # cross-attn
                        hidden_states = past_key_value
                return hidden_states

            scale = 1 / math.sqrt(self.query_dim // self.n_heads) if self.scale_qk else 1.0

            # get query states
            query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

            # get key/value states
            key_states = project(
                hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
            )
            value_states = project(
                hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
            )

            # compute scores
            scores = torch.matmul(
                query_states * scale, key_states.transpose(3, 2)
            )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

            if position_bias is None:
                if not self.has_relative_attention_bias:
                    position_bias = torch.zeros(
                        (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
                    )
                    if self.gradient_checkpointing and self.training:
                        position_bias.requires_grad = True
                else:
                    position_bias = self.compute_bias(real_seq_length, key_length, device=scores.device)

                # if key and values are already calculated
                # we want only the last query position bias
                if past_key_value is not None:
                    position_bias = position_bias[:, :, -hidden_states.size(1):, :]

                if mask is not None:
                    position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

            if self.pruned_heads:
                mask = torch.ones(position_bias.shape[1])
                mask[list(self.pruned_heads)] = 0
                position_bias_masked = position_bias[:, mask.bool()]
            else:
                position_bias_masked = position_bias

            # scores += position_bias_masked
            scores += (
                    position_bias_masked * 8
            )  # its actually root under the dimension of each attn head will be updated in the final version

            attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
                scores
            )  # (batch_size, n_heads, seq_length, key_length)
            attn_weights = nn.functional.dropout(
                attn_weights, p=self.dropout, training=self.training
            )  # (batch_size, n_heads, seq_length, key_length)

            # Mask heads if we want to
            if layer_head_mask is not None:
                attn_weights = attn_weights * layer_head_mask

            attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
            attn_output = self.o(attn_output)

            present_key_value_state = None
            outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

            if output_attentions:
                outputs = outputs + (attn_weights,)

            return outputs

class CLVPConditioningEncoder(nn.Module):
    """
    This class processes the log-mel spectrograms(generated by the Feature Extractor) and text tokens(generated by the
    tokenizer) for the Auto Regressive model as inputs.

    First each log-mel spectrogram is processed into a single vector which captures valuable characteristics from each
    of them, then the text tokens are converted into token embeddings and position embeddings are added afterwards.
    Both of these vectors are concatenated and then passed to the Auto Regressive model as inputs.

    The text tokens helps to incorporate the "text information" and the log-mel spectrogram is used to specify the
    "voice characteristics" into the generated Mel Tokens.
    """

    def __init__(self, config: CLVPConfig):
        super().__init__()

        text_config = config.text_config
        autoregressive_config = config.autoregressive_config

        self.mel_conv = nn.Conv1d(autoregressive_config.feature_size, autoregressive_config.n_embd, kernel_size=1)
        self.mel_attn_blocks = nn.Sequential([CLVPRelativeAttention(autoregressive_config) for _ in range(6)])

        self.text_token_embedding = nn.Embedding(text_config.vocab_size, autoregressive_config.n_embd)
        self.text_position_embedding = nn.Embedding(autoregressive_config.max_text_tokens, autoregressive_config.n_embd)

    def forward(self, mel_spec: torch.Tensor, text_tokens: torch.Tensor):
        # process each log-mel spectrogram into a single vector
        mel_spec = self.mel_conv(mel_spec)
        mel_spec = self.mel_attn_blocks(mel_spec)
        mel_spec = mel_spec[:, :, 0]

        # process text-tokens
        token_embeds = self.text_token_embedding(text_tokens)
        position_ids = torch.arange(0, text_tokens.shape[1], dtype=torch.int64)
        position_embeds = self.text_position_embedding(position_ids)

        text_embeds = token_embeds + position_embeds

        return  torch.concat([mel_spec, text_embeds], dim=1)


class CLVPPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = CLVPConfig
    base_model_prefix = "clvp"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=factor * 0.02)
        elif isinstance(module, CLVPAttention):
            factor = self.config.initializer_factor
            in_proj_std = (module.embed_dim**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            out_proj_std = (module.embed_dim**-0.5) * factor
            nn.init.normal_(module.q_proj.weight, std=in_proj_std)
            nn.init.normal_(module.k_proj.weight, std=in_proj_std)
            nn.init.normal_(module.v_proj.weight, std=in_proj_std)
            nn.init.normal_(module.out_proj.weight, std=out_proj_std)
        elif isinstance(module, CLVPMLP):
            factor = self.config.initializer_factor
            in_proj_std = (
                (module.config.hidden_size**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            )
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            nn.init.normal_(module.fc1.proj.weight if getattr(module.fc1, "proj") else module.fc1.weight, std=fc_std)
            nn.init.normal_(module.fc2.weight, std=in_proj_std)
        elif isinstance(module, CLVPModel):
            nn.init.normal_(
                module.text_projection.weight,
                std=module.text_embed_dim**-0.5 * self.config.initializer_factor,
            )
            nn.init.normal_(
                module.speech_projection.weight,
                std=module.speech_embed_dim**-0.5 * self.config.initializer_factor,
            )
        elif isinstance(module, CLVPSpeechModelWithProjection):
            nn.init.normal_(
                module.speech_projection.weight,
                std=self.config.hidden_size**-0.5 * self.config.initializer_factor,
            )
        elif isinstance(module, CLVPTextModelWithProjection):
            nn.init.normal_(
                module.text_projection.weight,
                std=self.config.hidden_size**-0.5 * self.config.initializer_factor,
            )

        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, CLVPEncoder):
            module.gradient_checkpointing = value


CLVP_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`CLVPConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

CLVP_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        use_causal_attention_mask (`bool`, *optional*, defaults to `False`):
            Whether to use causal attention mask.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

CLVP_SPEECH_INPUTS_DOCSTRING = r"""
    Args:
        speech_ids (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Speech Tokens. Padding will be ignored by default should you provide it.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        use_causal_attention_mask (`bool`, *optional*, defaults to `False`):
            Whether to use causal attention mask.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

CLVP_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        speech_ids (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Speech Tokens. Padding will be ignored by default should you provide it.
        text_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding text token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        speech_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding speech token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        use_causal_attention_mask (`bool`, *optional*, defaults to `False`):
            Whether to use causal attention mask.
        return_loss (`bool`, *optional*):
            Whether or not to return the contrastive loss.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class CLVPEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`CLVPEncoderLayer`].

    Args:
        config: CLVPConfig
    """

    def __init__(self, config: CLVPConfig):
        super().__init__()
        self.config = config
        self.rotary_pos_emb = CLVPRotaryPositionalEmbedding(config) if config.use_rotary_embedding else None
        self.layers = nn.ModuleList([CLVPEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        rotary_pos_emb = self.rotary_pos_emb(inputs_embeds) if self.rotary_pos_emb is not None else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(encoder_layer),
                    hidden_states,
                    rotary_pos_emb,
                    attention_mask,
                    causal_attention_mask,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    rotary_pos_emb,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class CLVPTransformer(nn.Module):
    """
    Transformer Encoder Block from 'Attention Is All You Need' paper.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.encoder = CLVPEncoder(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.sequence_summary = SequenceSummary(config)

    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=PretrainedConfig)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_causal_attention_mask: Optional[bool] = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.token_embedding(input_ids)

        # CLVP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLVP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clvp/model.py#L324
        causal_attention_mask = (
            _make_causal_mask(input_shape, hidden_states.dtype, device=hidden_states.device)
            if use_causal_attention_mask
            else None
        )
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # take the mean over axis 1 and get pooled output
        pooled_output = self.sequence_summary(last_hidden_state)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@add_start_docstrings(CLVP_START_DOCSTRING)
class CLVPModel(CLVPPreTrainedModel):
    config_class = CLVPConfig

    def __init__(self, config: CLVPConfig):
        super().__init__(config)

        if not isinstance(config.text_config, CLVPTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type CLVPTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.speech_config, CLVPSpeechConfig):
            raise ValueError(
                "config.speech_config is expected to be of type CLVPSpeechConfig but is of type"
                f" {type(config.speech_config)}."
            )

        text_config = config.text_config
        speech_config = config.speech_config

        self.projection_dim = config.projection_dim

        self.text_model = CLVPTransformerWithProjection(text_config, self.projection_dim, bias=False)
        self.speech_model = CLVPTransformerWithProjection(speech_config, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(CLVP_TEXT_INPUTS_DOCSTRING)
    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_causal_attention_mask: Optional[bool] = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim)`: The text embeddings obtained by
            applying the projection layer to the pooled output of [`CLVPTextModel`].

        Examples:

        ```python
        >>> from transformers import CLVPTokenizer, CLVPModel

        >>> model = CLVPModel.from_pretrained("susnato/clvp_dev")
        >>> tokenizer = CLVPTokenizer.from_pretrained("susnato/clvp_dev")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```"""

        # Use CLVP model's config for some fields (if specified) instead of those of speech & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_causal_attention_mask=use_causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True, # since we are returning `text_embeds` so we must use `return_dict` as True
        ).text_embeds

    @add_start_docstrings_to_model_forward(CLVP_SPEECH_INPUTS_DOCSTRING)
    def get_speech_features(
        self,
        speech_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_causal_attention_mask: Optional[bool] = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            speech_features (`torch.FloatTensor` of shape `(batch_size, output_dim)`: The speech embeddings obtained by
            applying the projection layer to the pooled output of [`CLVPSpeechModel`].

        Examples:

        ```python
        >>> from transformers import CLVPModel

        >>> model = CLVPModel.from_pretrained("susnato/clvp_dev")

        >>> # TODO : after FeatureExtractor is implemented we need to change it to something lik FE.__call__(...)
        >>> inputs = {"speech_ids": torch.tensor([[56, 8, 48, 7, 11, 23]]).long()}

        >>> speech_features = model.get_speech_features(**inputs)
        ```"""

        # Use CLVP model's config for some fields (if specified) instead of those of speech & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.speech_model(
            input_ids=speech_ids,
            attention_mask=attention_mask,
            use_causal_attention_mask=use_causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True, # since we are returning `speech_embeds` so we must use `return_dict` as True
        ).speech_embeds

    @add_start_docstrings_to_model_forward(CLVP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CLVPOutput, config_class=CLVPConfig)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        speech_ids: Optional[torch.FloatTensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        speech_attention_mask: Optional[torch.Tensor] = None,
        use_causal_attention_mask: Optional[bool] = False,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CLVPOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> import torch
        >>> from transformers import CLVPModel, CLVPTokenizer

        >>> model = CLVPModel.from_pretrained("susnato/clvp_dev")
        >>> tokenizer = CLVPTokenizer.from_pretrained("susnato/clvp_dev")

        >>> text = "This is a text."
        >>> inputs = tokenizer(text, return_tensors="pt")
        >>> # TODO : after FeatureExtractor is implemented we need to change it to something lik FE.__call__(...)
        >>> inputs["speech_ids"] = torch.tensor([[10, 55, 101, 37, 21, 102, 41]]).long()

        >>> outputs = model(**inputs)
        >>> logits_per_speech = outputs.logits_per_speech  # this is the speech-text similarity score
        >>> probs = logits_per_speech.softmax(dim=1)  # we can take the softmax to get the label probabilities
        ```
        """

        # Use CLVP model's config for some fields (if specified) instead of those of speech & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        speech_outputs = self.speech_model(
            input_ids=speech_ids,
            attention_mask=speech_attention_mask,
            use_causal_attention_mask=use_causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=text_attention_mask,
            use_causal_attention_mask=use_causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        speech_embeds = speech_outputs[0]
        text_embeds = text_outputs[0]

        # normalized features
        speech_embeds = speech_embeds / speech_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, speech_embeds.t()) * logit_scale
        logits_per_speech = logits_per_text.t()

        loss = None
        if return_loss:
            loss = clvp_loss(logits_per_text)

        if not return_dict:
            output = (logits_per_speech, logits_per_text, text_embeds, speech_embeds, text_outputs[2], speech_outputs[2])
            return ((loss,) + output) if loss is not None else output

        return CLVPOutput(
            loss=loss,
            logits_per_speech=logits_per_speech,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            speech_embeds=speech_embeds,
            text_model_output=text_outputs[2],
            speech_model_output=speech_outputs[2],
        )


@add_start_docstrings(
    """
    CLVP Text Model with a projection layer on top (a linear layer on top of the pooled output).
    """,
    CLVP_START_DOCSTRING,
)
class CLVPTextModelWithProjection(CLVPPreTrainedModel):
    config_class = CLVPTextConfig

    _no_split_modules = None

    def __init__(self, config: CLVPTextConfig):
        super().__init__(config)

        self.text_model = CLVPTransformer(config)

        self.text_projection = nn.Linear(config.hidden_size, config.projection_dim, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.token_embedding

    def set_input_embeddings(self, value):
        self.text_model.token_embedding = value

    @add_start_docstrings_to_model_forward(CLVP_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CLVPTextModelOutput, config_class=CLVPTextConfig)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_causal_attention_mask: Optional[bool] = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CLVPTextModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import CLVPTextModelWithProjection, CLVPTokenizer

        >>> model = CLVPTextModelWithProjection.from_pretrained("susnato/clvp_dev")
        >>> tokenizer = CLVPTokenizer.from_pretrained("susnato/clvp_dev")

        >>> text = "This is a sample text."
        >>> inputs = tokenizer(text, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> text_embeds = outputs.text_embeds
        ```
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_causal_attention_mask=use_causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = text_outputs[1]

        text_embeds = self.text_projection(pooled_output)

        if not return_dict:
            outputs = (text_embeds, ) + text_outputs
            return tuple(output for output in outputs if output is not None)

        return CLVPTextModelOutput(
            text_embeds=text_embeds,
            last_hidden_state=text_outputs.last_hidden_state,
            pooler_output=text_outputs.pooler_output,
            hidden_states=text_outputs.hidden_states,
            attentions=text_outputs.attentions,
        )


@add_start_docstrings(
    """
    CLVP Speech Model with a projection layer on top (a linear layer on top of the pooled output).
    """,
    CLVP_START_DOCSTRING,
)
class CLVPSpeechModelWithProjection(CLVPPreTrainedModel):
    config_class = CLVPSpeechConfig
    main_input_name = "speech_ids"

    _no_split_modules = None

    def __init__(self, config: CLVPSpeechConfig):
        super().__init__(config)

        self.speech_model = CLVPTransformer(config)

        self.speech_projection = nn.Linear(config.hidden_size, config.projection_dim, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.speech_model.token_embedding

    def set_input_embeddings(self, value):
        self.speech_model.token_embedding = value

    @add_start_docstrings_to_model_forward(CLVP_SPEECH_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CLVPSpeechModelOutput, config_class=CLVPSpeechConfig)
    def forward(
        self,
        speech_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_causal_attention_mask: Optional[bool] = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CLVPSpeechModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> import torch
        >>> from transformers import CLVPSpeechModelWithProjection

        >>> model = CLVPSpeechModelWithProjection.from_pretrained("susnato/clvp_dev")

        >>> # TODO : after FeatureExtractor is implemented we need to change it to something lik FE.__call__(...)
        >>> inputs = {"speech_ids": torch.tensor([[5, 62, 1, 5, 9, 10]]).long()}

        >>> outputs = model(**inputs)
        >>> speech_embeds = outputs.speech_embeds
        ```
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        speech_outputs = self.speech_model(
            input_ids=speech_ids,
            use_causal_attention_mask=use_causal_attention_mask,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = speech_outputs[1]  # pooled_output

        speech_embeds = self.speech_projection(pooled_output)

        if not return_dict:
            outputs = (speech_embeds, ) + speech_outputs
            return tuple(output for output in outputs if output is not None)

        return CLVPSpeechModelOutput(
            speech_embeds=speech_embeds,
            last_hidden_state=speech_outputs.last_hidden_state,
            pooler_output=speech_outputs.pooler_output,
            hidden_states=speech_outputs.hidden_states,
            attentions=speech_outputs.attentions,
        )
