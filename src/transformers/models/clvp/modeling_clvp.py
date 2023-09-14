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


import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.generation import GenerationConfig

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, CausalLMOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel, SequenceSummary
from ...pytorch_utils import Conv1D
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_clvp import (
    CLVPAutoRegressiveConfig,
    CLVPConfig,
    CLVPSpeechConfig,
    CLVPTextConfig,
    PretrainedConfig,
)


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


class CLVPTransformerWithProjectionOutput(ModelOutput):
    """
    Base class for text and speech model's outputs that contains a pooling of the last hidden states as well as a
    projection output(a linear layer on top of the pooled output).

    Args:
        embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Pooled output of the `last_hidden_state`.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of
            the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    embeds: Optional[torch.FloatTensor] = None
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
        logits_per_speech (`torch.FloatTensor` of shape `(speech_batch_size, text_batch_size)`):
            The scaled dot product scores between `speech_embeds` and `text_embeds`. This represents the speech-text
            similarity scores.
        logits_per_text (`torch.FloatTensor` of shape `(text_batch_size, speech_batch_size)`):
            The scaled dot product scores between `text_embeds` and `speech_embeds`. This represents the text-speech
            similarity scores.
        text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of the Text Model.
        speech_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The speech embeddings obtained by applying the projection layer to the pooled output of the Speech Model.
        text_model_output (`BaseModelOutputWithPooling`):
            The pooled output of the `last_hidden_state` of the Text Model.
        speech_model_output (`BaseModelOutputWithPooling`):
            The pooled output of the `last_hidden_state` of the Speech Model.
    """

    loss: Optional[torch.FloatTensor] = None
    speech_candidates: Optional[torch.LongTensor] = None
    logits_per_speech: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    speech_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    speech_model_output: BaseModelOutputWithPooling = None


class CLVPGatedLinearUnit(nn.Module):
    """
    `CLVPGatedLinearUnit` uses the second half of the `hidden_states` to act as a gate for the first half of the
    `hidden_states` which controls the flow of data from the first of the tensor.
    """

    def __init__(self, config):
        super().__init__()
        self.activation_fn = ACT2FN[config.hidden_act]
        self.proj = nn.Linear(config.hidden_size, config.intermediate_size * 2)

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * self.activation_fn(gate)


class CLVPTransformerMLP(nn.Module):
    """
    This MLP is used in CLVP speech or text models.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.fc1 = CLVPGatedLinearUnit(config)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout_layer = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
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

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        sequence_length = hidden_states.shape[1]

        if sequence_length == self.cached_sequence_length and self.cached_rotary_positional_embedding is not None:
            return self.cached_rotary_positional_embedding

        self.cached_sequence_length = sequence_length
        time_stamps = torch.arange(sequence_length, device=hidden_states.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", time_stamps, self.inv_freq)
        embeddings = torch.cat((freqs, freqs), dim=-1)

        self.cached_rotary_positional_embedding = embeddings.unsqueeze(0)
        return self.cached_rotary_positional_embedding


class CLVPSelfAttention(nn.Module):
    """
    Multi-headed attention to combine Absolute and Rotary Positional Embeddings into a single Attention module.
    """

    def __init__(self, config, apply_hidden_states_norm=False):
        super().__init__()
        self.apply_hidden_states_norm = apply_hidden_states_norm
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

        if hasattr(config, "max_position_embeddings"):
            max_positions = config.max_position_embeddings
            self.register_buffer(
                "bias",
                torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                    1, 1, max_positions, max_positions
                ),
                persistent=False,
            )

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.use_attention_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.use_attention_bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.use_attention_bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        if self.apply_hidden_states_norm:
            num_groups, num_channels = self.compute_groupnorm_groups(self.embed_dim)
            self.norm = nn.GroupNorm(num_groups, num_channels, eps=1e-5, affine=True)

    def compute_groupnorm_groups(self, channels: int):
        """
        Calculates the value of both `num_groups` and `num_channels` for nn.GroupNorm. This logic is taken from the
        official tortoise repository. link :
        https://github.com/neonbjb/tortoise-tts/blob/4003544b6ff4b68c09856e04d3eff9da26d023c2/tortoise/models/arch_util.py#L26
        """
        groups = 32
        if channels <= 16:
            groups = 8
        elif channels <= 64:
            groups = 16
        while channels % groups != 0:
            groups = int(groups / 2)

        if groups <= 2:
            raise ValueError(
                f"Number of groups for the GroupNorm must be greater than 2, but it is {groups}."
                f"Please consider using a different `n_embd` or `hidden_size`"
            )

        return groups, channels

    # Copied from transformers.models.clip.modeling_clip.CLIPAttention._shape
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        rotary_pos_emb: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        causal_attention_mask: Optional[torch.LongTensor] = None,
        use_causal_attention_mask: Optional[bool] = True,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor], Optional[Tuple[torch.FloatTensor]]]:
        # This logic is only used for the attention in CLVPConditioningEncoder. For the attention of AutoRegressive,
        # speech and text models it is not used.
        if self.apply_hidden_states_norm:
            hidden_states = torch.permute(self.norm(torch.permute(hidden_states, (0, 2, 1))), (0, 2, 1))

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if layer_past is not None:
            past_key, past_value = layer_past
            key_states = torch.cat((past_key, key_states), dim=-2)
            value_states = torch.cat((past_value, value_states), dim=-2)

        if use_cache is True:
            present = (key_states, value_states)
        else:
            present = None

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
        if use_causal_attention_mask:
            # if causal mask if given then apply it directly
            if causal_attention_mask is not None:
                attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
                attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

            # if there is no causal mask given but the config has `max_position_embeddings`, then use the constructed
            # mask. This portion mimics the `GPT2Attention`.
            elif causal_attention_mask is None and hasattr(self.config, "max_position_embeddings"):
                query_length, key_length = query_states.size(-2), key_states.size(-2)
                causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
                mask_value = torch.finfo(attn_weights.dtype).min
                mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
                attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
                attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights * head_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

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

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights_reshaped,)

        return outputs


class CLVPConditioningEncoder(nn.Module):
    """
    This class processes the log-mel spectrograms(generated by the Feature Extractor) and text tokens(generated by the
    tokenizer) as inputs for the Auto Regressive model.

    First each log-mel spectrogram is processed into a single vector which captures valuable characteristics from each
    of them, then the text tokens are converted into token embeddings and position embeddings are added afterwards.
    Both of these vectors are concatenated and then passed to the Auto Regressive model as inputs.

    The text tokens helps to incorporate the "text information" and the log-mel spectrogram is used to specify the
    "voice characteristics" into the generated Mel Tokens.
    """

    def __init__(self, config: CLVPConfig):
        super().__init__()

        text_config = config.text_config

        self.text_start_token_id = text_config.bos_token_id
        self.text_end_token_id = text_config.eos_token_id

        self.mel_conv = nn.Conv1d(
            config.autoregressive_config.feature_size, config.autoregressive_config.n_embd, kernel_size=1
        )

        self.mel_attn_blocks = nn.ModuleList(
            [CLVPSelfAttention(config.autoregressive_config, apply_hidden_states_norm=True) for _ in range(6)]
        )

        self.text_token_embedding = nn.Embedding(text_config.vocab_size, config.autoregressive_config.n_embd)
        self.text_position_embedding = nn.Embedding(
            config.autoregressive_config.max_text_tokens, config.autoregressive_config.n_embd
        )

    def forward(self, mel_spec: torch.FloatTensor, text_tokens: torch.LongTensor):
        # process each log-mel spectrogram into a single vector
        mel_spec = self.mel_conv(mel_spec)

        mel_spec = torch.permute(mel_spec, (0, 2, 1))
        for mel_attn_block in self.mel_attn_blocks:
            mel_spec = mel_attn_block(mel_spec, use_causal_attention_mask=False)[0] + mel_spec
        mel_spec = torch.permute(mel_spec, (0, 2, 1))
        mel_spec = mel_spec[:, :, 0]

        # process text-tokens
        # we add bos and eos token ids in the modeling file instead of the tokenizer file(same as the original repo)
        text_tokens = torch.nn.functional.pad(text_tokens, (1, 0), value=self.text_start_token_id)
        text_tokens = torch.nn.functional.pad(text_tokens, (0, 1), value=self.text_end_token_id)

        token_embeds = self.text_token_embedding(text_tokens)
        position_ids = torch.arange(0, text_tokens.shape[1], dtype=torch.int64, device=text_tokens.device)
        position_embeds = self.text_position_embedding(position_ids)

        text_embeds = token_embeds + position_embeds

        mel_spec = mel_spec.unsqueeze(1)
        # repeat if there is either (1 text vs N audios) or (N texts vs 1 audio)
        if text_embeds.shape[0] == 1 and mel_spec.shape[0] != 1:
            text_embeds = text_embeds.repeat(mel_spec.shape[0], 1, 1)
        elif text_embeds.shape[0] != 1 and mel_spec.shape[0] == 1:
            mel_spec = mel_spec.repeat(text_embeds.shape[0], 1, 1)
        # If there is N texts and M audios we will raise error since the number of text and audio must be same.
        elif text_embeds.shape[0] != mel_spec.shape[0]:
            raise ValueError(
                f"The number of texts and number of audios must be same. "
                f"Found {text_embeds.shape[0]} texts vs {mel_spec.shape[0]} audios"
            )

        return torch.concat([mel_spec, text_embeds], dim=1)


class CLVPPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = CLVPConfig
    base_model_prefix = "clvp"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=factor * 0.02)
        if isinstance(module, (nn.Linear, Conv1D, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=factor * 0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, CLVPSelfAttention):
            factor = self.config.initializer_factor
            in_proj_std = (module.embed_dim**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            out_proj_std = (module.embed_dim**-0.5) * factor
            nn.init.normal_(module.q_proj.weight, std=in_proj_std)
            nn.init.normal_(module.k_proj.weight, std=in_proj_std)
            nn.init.normal_(module.v_proj.weight, std=in_proj_std)
            nn.init.normal_(module.out_proj.weight, std=out_proj_std)
        elif isinstance(module, CLVPTransformerMLP):
            factor = self.config.initializer_factor
            in_proj_std = (
                (module.config.hidden_size**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            )
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            nn.init.normal_(module.fc1.proj.weight if getattr(module.fc1, "proj") else module.fc1.weight, std=fc_std)
            nn.init.normal_(module.fc2.weight, std=in_proj_std)
        elif isinstance(module, CLVPTransformerWithProjection):
            config = self.config.text_config if hasattr(self.config, "text_config") else self.config
            module.projection.weight.data.normal_(mean=0.0, std=factor * (config.hidden_size**-0.5))
        elif isinstance(module, CLVPConditioningEncoder):
            module.mel_conv.weight.data.normal_(mean=0.0, std=factor)
            module.mel_conv.bias.data.zero_()
        elif isinstance(module, CLVPAutoRegressiveLMHeadModel):
            for name, p in module.named_parameters():
                if name == "c_proj.weight":
                    p.data.normal_(mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.config.n_layer)))
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, CLVPEncoder):
            module.gradient_checkpointing = value
        if isinstance(module, CLVPAutoRegressiveLMHeadModel):
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


CLVP_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        input_features (`torch.FloatTensor` of shape `(batch_size, feature_size, time_dim)`):
            Indicates log-melspectrogram representations for audio returned by `CLVPFeatureExtractor`.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding text token indices. Mask values selected in `[0, 1]`:

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


class CLVPEncoderLayer(nn.Module):
    def __init__(self, config: CLVPConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.self_attn = CLVPSelfAttention(config)
        self.mlp = CLVPTransformerMLP(config)

        self.pre_branch_norm1 = CLVPRMSNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.pre_branch_norm2 = CLVPRMSNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        rotary_pos_emb: torch.FloatTensor,
        attention_mask: torch.LongTensor,
        causal_attention_mask: torch.LongTensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch, seq_len, embed_dim)`):
                input to the layer.
            rotary_pos_emb (`torch.FloatTensor`):
                rotary position embeddings generated by `CLVPRotaryPositionalEmbedding` module.
            attention_mask (`torch.FloatTensor` of shape `(batch, 1, tgt_len, src_len)`):
                attention mask where padding elements are indicated by very large negative values.
            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.pre_branch_norm1(hidden_states)

        attention_outputs = self.self_attn(
            hidden_states=hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            use_causal_attention_mask=False if causal_attention_mask is None else True,
            output_attentions=output_attentions,
        )

        hidden_states = attention_outputs[0]

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_branch_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attention_outputs[-1],)

        return outputs


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
        attention_mask: Optional[torch.LongTensor] = None,
        causal_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                input embeddings for the model. This bypasses the model's internal embedding lookup matrix.
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
            return (hidden_states, encoder_states, all_attentions)
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

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        use_causal_attention_mask: Optional[bool] = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
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


@add_start_docstrings(
    """
    `CLVPTransformer` with a projection layer on top (a linear layer on top of the pooled output).
    """,
    CLVP_START_DOCSTRING,
)
class CLVPTransformerWithProjection(CLVPPreTrainedModel):
    def __init__(self, config: Union[CLVPTextConfig, CLVPSpeechConfig]):
        super().__init__(config)

        self.model_type = config.model_type

        self.transformer = CLVPTransformer(config)

        self.projection = nn.Linear(config.hidden_size, config.projection_dim, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.transformer.token_embedding

    def set_input_embeddings(self, value):
        self.transformer.token_embedding = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        use_causal_attention_mask: Optional[bool] = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CLVPTransformerWithProjectionOutput]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it. This indicates both the text ids and the speech ids.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            use_causal_attention_mask (`bool`, *optional*, defaults to `False`):
                Whether to use causal attention mask.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Examples:

        ```python
        >>> from transformers import CLVPTransformerWithProjection, CLVPTextConfig, CLVPTokenizer

        >>> text_config = CLVPTextConfig.from_pretrained("susnato/clvp_dev")
        >>> model = CLVPTransformerWithProjection(text_config)
        >>> tokenizer = CLVPTokenizer.from_pretrained("susnato/clvp_dev")

        >>> text = "This is an example text."
        >>> inputs = tokenizer(text, return_tensors="pt")

        >>> outputs = model(**inputs)
        ```
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_causal_attention_mask=use_causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = transformer_outputs[1]

        embeds = self.projection(pooled_output)

        if not return_dict:
            outputs = (embeds,) + transformer_outputs
            return outputs

        return CLVPTransformerWithProjectionOutput(
            embeds=embeds,
            last_hidden_state=transformer_outputs.last_hidden_state,
            pooler_output=transformer_outputs.pooler_output,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


# Copied from transformers.models.gpt2.modeling_gpt2.GPT2MLP with GPT2->CLVPAutoRegressive
class CLVPAutoRegressiveMLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class CLVPAutoRegressiveBlock(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = CLVPSelfAttention(config)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = CLVPAutoRegressiveMLP(inner_dim, config)

    # Copied from transformers.models.gpt2.modeling_gpt2.GPT2Block.forward
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)


CLVP_AUTOREGRESSIVE_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values[0][0].shape[-2]` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        past_key_values (`Tuple[Tuple[torch.Tensor]]` of length `config.n_layers`):
            Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see
            `past_key_values` output below). Can be used to speed up sequential decoding. The `input_ids` which have
            their past given to this model should not be passed as `input_ids` as they have already been computed.
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            If `past_key_values` is used, `attention_mask` needs to contain the masking strategy that was used for
            `past_key_values`. In other words, the `attention_mask` always has to have the length:
            `len(past_key_values) + len(input_ids)`

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.

            If `past_key_values` is used, optionally only the last `inputs_embeds` have to be input (see
            `past_key_values`).
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class CLVPAutoRegressiveLMHeadModel(CLVPPreTrainedModel):
    """
    It is an autoregressive decoder based model similar to `GPT2LMHead`.
    """

    def __init__(self, config):
        super().__init__(config)

        self.config = config.autoregressive_config if hasattr(config, "autoregressive_config") else config

        self.embed_dim = self.config.n_embd

        self.input_embeds_layer = nn.Embedding(self.config.vocab_size, self.embed_dim)
        self.position_embeds_layer = nn.Embedding(self.config.max_mel_tokens, self.embed_dim)

        self.drop = nn.Dropout(self.config.embd_pdrop)
        self.layers = nn.ModuleList(
            [CLVPAutoRegressiveBlock(self.config, layer_idx=i) for i in range(self.config.n_layer)]
        )
        self.layer_norm = nn.LayerNorm(self.embed_dim, eps=self.config.layer_norm_epsilon)

        self.final_norm = nn.LayerNorm(self.embed_dim)
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=True)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.input_embeds_layer

    def set_input_embeddings(self, new_embeddings):
        self.input_embeds_layer = new_embeddings

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, conditioning_embeds=None, **kwargs
    ):
        # for the first pass, so we must add the start token to the conditioning_embeds and return them as `inputs_embeds`
        if conditioning_embeds is not None and past_key_values is None:
            # Add the start mel token at the end
            mel_start_token_id = torch.tensor([[self.config.bos_token_id]], device=conditioning_embeds.device)
            mel_start_token_embedding = self.input_embeds_layer(mel_start_token_id) + self.position_embeds_layer(
                torch.tensor([[0]], device=conditioning_embeds.device)
            )
            mel_start_token_embedding = mel_start_token_embedding.repeat(conditioning_embeds.shape[0], 1, 1)
            conditioning_embeds = torch.concat([conditioning_embeds, mel_start_token_embedding], dim=1)

            # since we will add the position embeddings in the forward pass, we must subtract it here so that it cancells
            # out. This decision was made to make sure that `test_generate_from_inputs_embeds_decoder_only` test does not fail.
            position_ids = torch.range(0, conditioning_embeds.shape[1] - 1, device=conditioning_embeds.device)
            position_ids = position_ids.unsqueeze(0).repeat(conditioning_embeds.shape[0], 1).long()
            conditioning_embeds = conditioning_embeds - self.position_embeds_layer(position_ids)

            return {
                "inputs_embeds": conditioning_embeds,
                "past_key_values": None,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
            }

        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "token_type_ids": token_type_ids,
            }
        )
        return model_inputs

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.layers[layer].attn.prune_heads(heads)

    @add_start_docstrings_to_model_forward(CLVP_AUTOREGRESSIVE_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.layers))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.input_embeds_layer(input_ids)
        position_embeds = self.position_embeds_layer(position_ids)
        inputs_embeds = inputs_embeds + position_embeds

        hidden_states = inputs_embeds

        if token_type_ids is not None:
            token_type_embeds = self.input_embeds_layer(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.layers, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

        hidden_states = self.layer_norm(hidden_states)

        hidden_states = hidden_states.view(output_shape)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        lm_logits = self.final_norm(hidden_states)
        lm_logits = self.lm_head(lm_logits)

        loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits, presents, all_hidden_states, all_self_attentions, all_cross_attentions)
            output = ((loss,) + output) if loss is not None else output
            return tuple(v for v in output if v is not None)

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

    @staticmethod
    # Copied from transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel._reorder_cache
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
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

        if not isinstance(config.autoregressive_config, CLVPAutoRegressiveConfig):
            raise ValueError(
                "config.autoregressive_config is expected to be of type CLVPAutoRegressiveConfig but is of type"
                f" {type(config.autoregressive_config)}."
            )

        self.conditioning_encoder = CLVPConditioningEncoder(config)

        self.speech_autoregressive_model = CLVPAutoRegressiveLMHeadModel(config)

        self.text_model = CLVPTransformerWithProjection(config.text_config)
        self.speech_model = CLVPTransformerWithProjection(config.speech_config)

        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        # Initialize weights and apply final processing
        self.post_init()

    # taken from the original repo,
    # link : https://github.com/neonbjb/tortoise-tts/blob/4003544b6ff4b68c09856e04d3eff9da26d023c2/tortoise/api.py#L117
    def fix_autoregressive_speech_output(self, autoreg_output: torch.LongTensor) -> torch.LongTensor:
        """
        This method modifies the output of the autoregressive model, such as replacing the `eos_token_id` and changing
        the last few tokens of each sequence.

        Args:
            autoreg_output (`torch.LongTensor`):
                This refers to the output of the autoregressive model.
        """
        autoreg_output = autoreg_output[:, 1:]

        stop_token_indices = torch.where(autoreg_output == self.speech_autoregressive_model.config.eos_token_id, 1, 0)
        autoreg_output = torch.masked_fill(autoreg_output, mask=stop_token_indices.bool(), value=83)

        for i, each_seq_stop_token_indice in enumerate(stop_token_indices):
            # This means that no stop tokens were found so the sentence was still being generated, in that case we don't need
            # to apply any padding so just skip to the next sequence of tokens.
            if each_seq_stop_token_indice.sum() == 0:
                continue

            stm = each_seq_stop_token_indice.argmax()
            autoreg_output[i, stm:] = 83
            if stm - 3 < autoreg_output.shape[1]:
                autoreg_output[i, -3] = 45
                autoreg_output[i, -2] = 45
                autoreg_output[i, -1] = 248

        return autoreg_output

    def get_text_features(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        use_causal_attention_mask: Optional[bool] = False,
    ) -> torch.FloatTensor:
        r"""
        This method can be used to extract text_embeds from a text. The text embeddings obtained by applying the
        projection layer to the pooled output of the CLVP Text Model.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            use_causal_attention_mask (`bool`, *optional*, defaults to `False`):
                Whether to use causal attention mask.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, output_dim)`:
                The text embeddings obtained by applying the projection layer to the pooled output of the CLVP Text
                Model.

        Examples:

        ```python
        >>> from transformers import CLVPProcessor, CLVPModel

        >>> # Define the Text
        >>> text = "This is an example text."

        >>> # Define processor and model
        >>> processor = CLVPProcessor.from_pretrained("susnato/clvp_dev")
        >>> model = CLVPModel.from_pretrained("susnato/clvp_dev")

        >>> # Generate processor output and text embeds
        >>> processor_output = processor(text=text, return_tensors="pt")
        >>> text_embeds = model.get_text_features(input_ids=processor_output["input_ids"])
        ```
        """

        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_causal_attention_mask=use_causal_attention_mask,
        )[0]

    def get_speech_features(
        self,
        speech_ids: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_causal_attention_mask: Optional[bool] = False,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        r"""
        This method can be used to extract speech_embeds. The speech embeddings are obtained by applying the speech
        model on speech_ids. If speech_ids is not present but both input_ids and input_features are given then the
        autoregressive model will be used to first generate the speech_ids and then applying the speech model.

        Args:
            speech_ids (`torch.FloatTensor` of shape `(batch_size, num_speech_candidates)`, *optional*):
                Speech Tokens. Padding will be ignored by default should you provide it. If speech_ids are provided
                then input_ids and input_features will be automatically ignored.
            input_ids (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Input text Tokens. Generated from the `CLVPTokenizer`. If speech_ids is not provided, then input_ids
                and input_features will be used.
            input_features (`torch.FloatTensor` of shape `(batch_size, feature_size, time_dim)`, *optional*):
                Indicates log-melspectrogram representations for audio returned by `CLVPFeatureExtractor`. If
                speech_ids is not provided, then input_ids and input_features will be used.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding speech token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            use_causal_attention_mask (`bool`, *optional*, defaults to `False`):
                Whether to use causal attention mask.

            generation_config (`GenerationConfig`, *optional*):
                generation config to control the generation of speech_ids if they are not provided.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, output_dim)`:
                The speech embeddings obtained by applying the projection layer to the pooled output of the CLVP Speech
                Model.

        Examples:

        ```python
        >>> import torch
        >>> from transformers import CLVPModel

        >>> # Define model
        >>> model = CLVPModel.from_pretrained("susnato/clvp_dev")

        >>> # These are supposed to be the `speech_ids` generated by another model
        >>> speech_ids = {"speech_ids": torch.tensor([[56, 8, 48, 7, 11, 23]]).long()}

        >>> speech_embeds = model.get_speech_features(speech_ids=speech_ids["speech_ids"])
        ```

        ```python
        >>> import datasets
        >>> from transformers import CLVPProcessor, CLVPModel

        >>> # Define the Text and Load the Audio (We are taking an audio example from HuggingFace Hub using `datasets` library)
        >>> text = "This is an example text."
        >>> ds = datasets.load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.cast_column("audio", datasets.Audio(sampling_rate=22050))
        >>> _, audio, sr = ds.sort("id").select(range(1))[:1]["audio"][0].values()

        >>> # Define processor and model
        >>> processor = CLVPProcessor.from_pretrained("susnato/clvp_dev")
        >>> model = CLVPModel.from_pretrained("susnato/clvp_dev")

        >>> # Generate processor output and model output
        >>> processor_output = processor(raw_speech=audio, sampling_rate=sr, text=text, return_tensors="pt")
        >>> speech_embeds = model.get_speech_features(
        ...     input_ids=processor_output["input_ids"], input_features=processor_output["input_features"]
        ... )
        ```
        """

        if speech_ids is None:
            if input_ids is None and input_features is None:
                raise ValueError("Either speech_ids or input_ids and input_features must be provided.")

            if generation_config is None:
                generation_config = self.generation_config
            generation_config.update(**kwargs)

            conditioning_embeds = self.conditioning_encoder(mel_spec=input_features, text_tokens=input_ids)

            speech_ids = self.speech_autoregressive_model.generate(
                conditioning_embeds=conditioning_embeds,
                generation_config=generation_config,
            )

            speech_ids = self.fix_autoregressive_speech_output(speech_ids[0])

        return self.speech_model(
            input_ids=speech_ids,
            attention_mask=attention_mask,
            use_causal_attention_mask=use_causal_attention_mask,
        )[0]

    @add_start_docstrings_to_model_forward(CLVP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CLVPOutput, config_class=CLVPConfig)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        input_features: torch.FloatTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
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
        >>> import datasets
        >>> from transformers import CLVPProcessor, CLVPModel

        >>> # Define the Text and Load the Audio (We are taking an audio example from HuggingFace Hub using `datasets` library)
        >>> text = "This is an example text."

        >>> ds = datasets.load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.cast_column("audio", datasets.Audio(sampling_rate=22050))
        >>> _, audio, sr = ds.sort("id").select(range(1))[:1]["audio"][0].values()

        >>> # Define processor and model
        >>> processor = CLVPProcessor.from_pretrained("susnato/clvp_dev")
        >>> model = CLVPModel.from_pretrained("susnato/clvp_dev")

        >>> # processor outputs and model output s
        >>> processor_output = processor(raw_speech=audio, sampling_rate=sr, text=text, return_tensors="pt")
        >>> outputs = model(
        ...     input_ids=processor_output["input_ids"],
        ...     input_features=processor_output["input_features"],
        ...     return_dict=True,
        ... )
        ```
        """

        # Use CLVP model's config for some fields (if specified) instead of those of speech & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        conditioning_embeds = self.conditioning_encoder(mel_spec=input_features, text_tokens=input_ids)

        speech_candidates = self.speech_autoregressive_model(
            inputs_embeds=conditioning_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        speech_candidates = speech_candidates[0]

        # since we will get the embeds of shape `(batch_size, seq_len, embedding_dim)` during the forward pass
        # we must convert it to tokens, to make it compaitable with speech_transformer
        if speech_candidates.ndim == 3:
            speech_candidates = speech_candidates.argmax(2)
        speech_candidates = self.fix_autoregressive_speech_output(speech_candidates)

        speech_outputs = self.speech_model(
            input_ids=speech_candidates,
            use_causal_attention_mask=use_causal_attention_mask,
            attention_mask=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
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
            output = (
                logits_per_speech,
                logits_per_text,
                text_embeds,
                speech_embeds,
                text_outputs[2],
                speech_outputs[2],
            )
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

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        input_features: torch.FloatTensor = None,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs,
    ):
        if generation_config is None:
            generation_config = self.generation_config
        generation_config.update(**kwargs)

        conditioning_embeds = self.conditioning_encoder(mel_spec=input_features, text_tokens=input_ids)

        speech_candidates = self.speech_autoregressive_model.generate(
            conditioning_embeds=conditioning_embeds,
            generation_config=generation_config,
        )

        speech_candidates = self.fix_autoregressive_speech_output(speech_candidates[0])

        speech_outputs = self.speech_model(
            input_ids=speech_candidates,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
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

        if not generation_config.return_dict_in_generate:
            output = (
                speech_candidates,
                logits_per_speech,
                logits_per_text,
                text_embeds,
                speech_embeds,
                text_outputs[2],
                speech_outputs[2],
            )
            return output

        return CLVPOutput(
            speech_candidates=speech_candidates,
            logits_per_speech=logits_per_speech,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            speech_embeds=speech_embeds,
            text_model_output=text_outputs[2],
            speech_model_output=speech_outputs[2],
        )

    def prepare_inputs_for_generation(self, input_ids, inputs_embeds, **kwargs):
        return {"inputs_embeds": inputs_embeds, "input_ids": input_ids, **kwargs}
