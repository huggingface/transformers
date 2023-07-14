# coding=utf-8
# Copyright 2023 The OpenAI Team Authors and The HuggingFace Team. All rights reserved.
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
from .configuration_clvp import CLVPConfig, CLVPSpeechConfig, CLVPTextConfig, PretrainedConfig


try:
    from torch.nn import Identity
except ImportError:
    # Older PyTorch compatibility
    class Identity(nn.Module):
        r"""A placeholder identity operator that is argument-insensitive."""

        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, input):
            return input


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "susnato/clvp_dev"

CLVP_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "susnato/clvp_dev",
    # See all CLVP models at https://huggingface.co/models?filter=clvp
]


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


# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/2021-03-07-clvp.html
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


# Copied from transformers.models.clip.modeling_clip.clip_loss with clip->clvp, image_loss->speech_loss
def clvp_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    speech_loss = contrastive_loss(similarity.t())
    return (caption_loss + speech_loss) / 2.0


def rotate_half(x):
    x_shape = x.size()
    x = x.view([*x_shape[:-1]] + [2, x_shape[-1] // 2])
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs):
    seq_len = t.shape[-1]
    freqs = freqs[:, :, -seq_len:]
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())


class CLVPScaleNorm(nn.Module):
    def __init__(self, normalized_shape, eps):
        super().__init__()
        self.scale = normalized_shape**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class CLVPRMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps):
        super().__init__()
        self.scale = normalized_shape**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class CLVPRMSScaleShiftNorm(nn.Module):
    def __init__(self, normalized_shape, eps):
        super().__init__()
        self.scale = normalized_shape**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(normalized_shape))
        self.scale_shift_process = nn.Linear(normalized_shape * 2, normalized_shape * 2)

    def forward(self, x, norm_scale_shift_inp):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        norm = x / norm.clamp(min=self.eps) * self.g

        ss_emb = self.scale_shift_process(norm_scale_shift_inp)
        scale, shift = torch.chunk(ss_emb, 2, dim=1)
        h = norm * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return h


CLVP_NORM_TYPES = {
    "rms_norm": CLVPRMSNorm,
    "scale_norm": CLVPScaleNorm,
    "rms_scale_shift_norm": CLVPRMSScaleShiftNorm,
    "layer_norm": nn.LayerNorm,
}


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
            The text embeddings obtained by applying the projection layer to the pooled output of [`CLVPTextModel`].
        speech_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The speech embeddings obtained by applying the projection layer to the pooled output of
            [`CLVPSpeechModel`].
        text_model_output(`BaseModelOutputWithPooling`):
            The output of the [`CLVPTextModel`].
        speech_model_output(`BaseModelOutputWithPooling`):
            The output of the [`CLVPSpeechModel`].
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


class CLVPEmbeddings(nn.Module):
    """CLVP Embedding Layer for both text and speech models"""

    def __init__(self, config: CLVPTextConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        return inputs_embeds


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

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.use_attention_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.use_attention_bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.use_attention_bias)
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
            l = rotary_pos_emb.shape[-1]
            (query_states_l, query_states_r), (key_states_l, key_states_r), (value_states_l, value_states_r) = (
                (t[..., :l], t[..., l:]) for t in (query_states, key_states, value_states)
            )
            query_states_l, key_states_l, value_states_l = (
                apply_rotary_pos_emb(t, rotary_pos_emb) for t in (query_states_l, key_states_l, value_states_l)
            )
            query_states, key_states, value_states = (
                torch.cat(t, dim=-1)
                for t in (
                    (query_states_l, query_states_r),
                    (key_states_l, key_states_r),
                    (value_states_l, value_states_r),
                )
            )

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

        # Check if GLU is needed or not
        if config.use_glu_in_ff:
            self.fc1 = CLVPGatedLinearUnit(config)
        else:
            self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
            self.activation_fn = ACT2FN[config.hidden_act]

        # Check if layer norm is applied after the Activation
        if config.ff_post_act_layer_norm:
            self.post_act_layer_norm = CLVP_NORM_TYPES[config.norm_type](
                config.intermediate_size, eps=config.layer_norm_eps
            )

        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout_layer = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.config.use_glu_in_ff:
            hidden_states = self.fc1(hidden_states)
        else:
            hidden_states = self.fc1(hidden_states)
            hidden_states = self.activation_fn(hidden_states)

        if self.config.ff_post_act_layer_norm:
            hidden_states = self.post_act_layer_norm(hidden_states)

        hidden_states = self.dropout_layer(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLVPRotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding Class for CLVP. It was proposed in the paper 'ROFORMER: ENHANCED TRANSFORMER WITH ROTARY
    POSITION EMBEDDING', Please see https://arxiv.org/pdf/2104.09864v1.pdf .
    """

    def __init__(self, config):
        super().__init__()
        dim = max(config.projection_dim // (config.num_attention_heads * 2), 32)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len):
        t = torch.arange(max_seq_len).type_as(self.inv_freq)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb[None, ...]


class CLVPEncoderLayer(nn.Module):
    def __init__(self, config: CLVPConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.self_attn = CLVPAttention(config)
        self.mlp = CLVPMLP(config)

        if config.use_pre_branch_norm:
            self.pre_branch_norm1 = CLVP_NORM_TYPES[config.norm_type](self.embed_dim, eps=config.layer_norm_eps)
            self.pre_branch_norm2 = CLVP_NORM_TYPES[config.norm_type](self.embed_dim, eps=config.layer_norm_eps)
        if config.use_post_branch_norm:
            self.post_branch_norm1 = CLVP_NORM_TYPES[config.norm_type](self.embed_dim, eps=config.layer_norm_eps)
            self.post_branch_norm2 = CLVP_NORM_TYPES[config.norm_type](self.embed_dim, eps=config.layer_norm_eps)
        if config.use_post_main_norm:
            self.post_main_norm1 = CLVP_NORM_TYPES[config.norm_type](self.embed_dim, eps=config.layer_norm_eps)
            self.post_main_norm2 = CLVP_NORM_TYPES[config.norm_type](self.embed_dim, eps=config.layer_norm_eps)

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

        if self.config.use_pre_branch_norm:
            hidden_states = self.pre_branch_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        if self.config.use_post_branch_norm:
            hidden_states = self.post_branch_norm1(hidden_states)

        hidden_states = residual + hidden_states
        if self.config.use_post_main_norm:
            hidden_states = self.post_main_norm1(hidden_states)

        residual = hidden_states
        if self.config.use_pre_branch_norm:
            hidden_states = self.pre_branch_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.config.use_post_branch_norm:
            hidden_states = self.post_branch_norm2(hidden_states)
        hidden_states = residual + hidden_states
        if self.config.use_post_main_norm:
            hidden_states = self.post_main_norm2(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


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
        if isinstance(module, CLVPEmbeddings):
            module.token_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
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
        self.rotary_pos_emb = CLVPRotaryEmbedding(config) if config.use_rotary_embedding else None
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

        rotary_pos_emb = self.rotary_pos_emb(inputs_embeds.shape[1]) if self.rotary_pos_emb is not None else None

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


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


class CLVPTransformer(nn.Module):
    """
    Transformer Encoder Block from 'Attention Is All You Need' paper.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = CLVPEmbeddings(config)
        self.encoder = CLVPEncoder(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
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

        hidden_states = self.embeddings(input_ids=input_ids)

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


class CLVPTextTransformer(CLVPTransformer):
    def __init__(self, config: CLVPTextConfig):
        super().__init__(config=config)

    @add_start_docstrings_to_model_forward(CLVP_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLVPTextConfig)
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
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_causal_attention_mask=use_causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


@add_start_docstrings(
    """The text model from CLVP without any head or projection on top.""",
    CLVP_START_DOCSTRING,
)
class CLVPTextModel(CLVPPreTrainedModel):
    config_class = CLVPTextConfig

    _no_split_modules = ["CLVPEncoderLayer"]

    def __init__(self, config: CLVPTextConfig):
        super().__init__(config)
        self.text_model = CLVPTextTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        self.text_model.embeddings.token_embedding = value

    @add_start_docstrings_to_model_forward(CLVP_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLVPTextConfig)
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

        Examples:

        ```python
        >>> import torch
        >>> from transformers import CLVPTextModel

        >>> model = CLVPTextModel.from_pretrained("susnato/clvp_dev")

        >>> inputs = {"input_ids": torch.tensor([[5, 62, 1, 5, 9, 10]]).long()}

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_causal_attention_mask=use_causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class CLVPSpeechTransformer(CLVPTransformer):
    def __init__(self, config: CLVPSpeechConfig):
        super().__init__(config=config)

    @add_start_docstrings_to_model_forward(CLVP_SPEECH_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLVPSpeechConfig)
    def forward(
        self,
        speech_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_causal_attention_mask: Optional[bool] = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        return super().forward(
            input_ids=speech_ids,
            attention_mask=attention_mask,
            use_causal_attention_mask=use_causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


@add_start_docstrings(
    """The speech model from CLVP without any head or projection on top.""",
    CLVP_START_DOCSTRING,
)
class CLVPSpeechModel(CLVPPreTrainedModel):
    config_class = CLVPSpeechConfig
    main_input_name = "speech_ids"

    _no_split_modules = ["CLVPEncoderLayer"]

    def __init__(self, config: CLVPSpeechConfig):
        super().__init__(config)
        self.speech_model = CLVPSpeechTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.speech_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        self.speech_model.embeddings.token_embedding = value

    @add_start_docstrings_to_model_forward(CLVP_SPEECH_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLVPSpeechConfig)
    def forward(
        self,
        speech_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_causal_attention_mask: Optional[bool] = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Example:
        ```python
        >>> import torch
        >>> from transformers import CLVPSpeechModel

        >>> model = CLVPSpeechModel.from_pretrained("susnato/clvp_dev")

        >>> inputs = {"speech_ids": torch.tensor([[56, 8, 48, 7, 11, 23]]).long()}

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output
        ```"""

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.speech_model(
            speech_ids=speech_ids,
            attention_mask=attention_mask,
            use_causal_attention_mask=use_causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
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
        self.text_embed_dim = text_config.hidden_size
        self.speech_embed_dim = speech_config.hidden_size

        self.text_model = CLVPTextTransformer(text_config)
        self.speech_model = CLVPSpeechTransformer(speech_config)

        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.speech_projection = nn.Linear(self.speech_embed_dim, self.projection_dim, bias=False)
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

        """

        # Use CLVP model's config for some fields (if specified) instead of those of speech & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
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
        text_features = self.text_projection(pooled_output)

        return text_features

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

        """

        # Use CLVP model's config for some fields (if specified) instead of those of speech & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        speech_outputs = self.speech_model(
            speech_ids=speech_ids,
            attention_mask=attention_mask,
            use_causal_attention_mask=use_causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = speech_outputs[1]  # pooled_output
        speech_features = self.speech_projection(pooled_output)

        return speech_features

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
        >>> from transformers import CLVPModel

        >>> model = CLVPModel.from_pretrained("susnato/clvp_dev")

        >>> inputs = {
        ...     "input_ids": torch.tensor([[5, 62, 1, 9, 44]]).long(),
        ...     "speech_ids": torch.tensor([[10, 55, 101, 37, 21, 102, 41]]).long(),
        ... }

        >>> outputs = model(**inputs)
        >>> logits_per_speech = outputs.logits_per_speech  # this is the speech-text similarity score
        >>> probs = logits_per_speech.softmax(dim=1)  # we can take the softmax to get the label probabilities
        ```"""

        # Use CLVP model's config for some fields (if specified) instead of those of speech & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        speech_outputs = self.speech_model(
            speech_ids=speech_ids,
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

        speech_embeds = speech_outputs[1]
        speech_embeds = self.speech_projection(speech_embeds)

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

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
            output = (logits_per_speech, logits_per_text, text_embeds, speech_embeds, text_outputs, speech_outputs)
            return ((loss,) + output) if loss is not None else output

        return CLVPOutput(
            loss=loss,
            logits_per_speech=logits_per_speech,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            speech_embeds=speech_embeds,
            text_model_output=text_outputs,
            speech_model_output=speech_outputs,
        )


@add_start_docstrings(
    """
    CLVP Text Model with a projection layer on top (a linear layer on top of the pooled output).
    """,
    CLVP_START_DOCSTRING,
)
class CLVPTextModelWithProjection(CLVPPreTrainedModel):
    config_class = CLVPTextConfig

    _no_split_modules = ["CLVPEncoderLayer"]

    def __init__(self, config: CLVPTextConfig):
        super().__init__(config)

        self.text_model = CLVPTextTransformer(config)

        self.text_projection = nn.Linear(config.hidden_size, config.projection_dim, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        self.text_model.embeddings.token_embedding = value

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
        >>> import torch
        >>> from transformers import CLVPTextModelWithProjection

        >>> model = CLVPTextModelWithProjection.from_pretrained("susnato/clvp_dev")

        >>> inputs = {"input_ids": torch.tensor([[5, 62, 1, 5, 9, 10]]).long()}

        >>> outputs = model(**inputs)
        >>> text_embeds = outputs.text_embeds
        ```"""
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
            outputs = (text_embeds, text_outputs[0]) + text_outputs[2:]
            return tuple(output for output in outputs if output is not None)

        return CLVPTextModelOutput(
            text_embeds=text_embeds,
            last_hidden_state=text_outputs.last_hidden_state,
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

    _no_split_modules = ["CLVPEncoderLayer"]

    def __init__(self, config: CLVPSpeechConfig):
        super().__init__(config)

        self.speech_model = CLVPSpeechTransformer(config)

        self.speech_projection = nn.Linear(config.hidden_size, config.projection_dim, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.speech_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        self.speech_model.embeddings.token_embedding = value

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

        >>> inputs = {"speech_ids": torch.tensor([[5, 62, 1, 5, 9, 10]]).long()}

        >>> outputs = model(**inputs)
        >>> speech_embeds = outputs.speech_embeds
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        speech_outputs = self.speech_model(
            speech_ids=speech_ids,
            use_causal_attention_mask=use_causal_attention_mask,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = speech_outputs[1]  # pooled_output

        speech_embeds = self.speech_projection(pooled_output)

        if not return_dict:
            outputs = (speech_embeds, speech_outputs[0]) + speech_outputs[2:]
            return tuple(output for output in outputs if output is not None)

        return CLVPSpeechModelOutput(
            speech_embeds=speech_embeds,
            last_hidden_state=speech_outputs.last_hidden_state,
            hidden_states=speech_outputs.hidden_states,
            attentions=speech_outputs.attentions,
        )
