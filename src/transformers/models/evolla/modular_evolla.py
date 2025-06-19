# coding=utf-8
# Copyright 2025 Westlake Representational Learning Lab (Fajie Yuan Lab) team and the HuggingFace Inc. team. All rights reserved.
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

import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import Tensor, nn

from ...cache_utils import Cache, DynamicCache, StaticCache
from ...generation import GenerationMixin
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithPast,
    ModelOutput,
)
from ...modeling_utils import ModuleUtilsMixin, get_parameter_dtype
from ...utils import (
    is_torch_flex_attn_available,
    logging,
)
from ...utils.import_utils import is_torch_fx_proxy, is_torchdynamo_compiling
from ..esm.modeling_esm import (
    EsmAttention,
    EsmEmbeddings,
    EsmEncoder,
    EsmIntermediate,
    EsmLayer,
    EsmOutput,
    EsmPooler,
    EsmSelfAttention,
    EsmSelfOutput,
)
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaMLP,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from .configuration_evolla import EvollaConfig, SaProtConfig


if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask

    from ...integrations.flex_attention import make_flex_block_causal_mask

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "EvollaConfig"


@dataclass
class EvollaProteinEncoderModelOutput(ModelOutput):
    """ """

    sequence_compressor_output: torch.FloatTensor = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class EvollaRMSNorm(LlamaRMSNorm):
    pass


class EvollaRotaryEmbedding(LlamaRotaryEmbedding):
    pass


class EvollaMLP(LlamaMLP):
    pass


class EvollaAttention(LlamaAttention):
    pass


class EvollaFeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        inner_dim = int(dim * mult)

        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, inner_dim, bias=False)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class EvollaSequenceAlignerCrossAttention(nn.Module):
    def __init__(
        self,
        config,
        protein_encoder_dim: Optional[int] = None,
        structure_encoder_dim: Optional[int] = None,
        msa_encoder_dim: Optional[int] = None,
    ):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.scale = self.num_attention_heads**-0.5
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        attention_probs_dropout_prob = (config.aligner_attention_probs_dropout_prob,)
        enable_bias = (config.aligner_enable_bias,)
        ffn_mult = (config.aligner_ffn_mult,)

        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        if protein_encoder_dim is not None:
            self.key_protein = nn.Linear(protein_encoder_dim, self.all_head_size)
            self.value_protein = nn.Linear(protein_encoder_dim, self.all_head_size)
        else:
            self.key_protein = None
            self.value_protein = None

        if structure_encoder_dim is not None:
            self.key_structure = nn.Linear(structure_encoder_dim, self.all_head_size)
            self.value_structure = nn.Linear(structure_encoder_dim, self.all_head_size)
        else:
            self.key_structure = None
            self.value_structure = None

        if msa_encoder_dim is not None:
            self.key_msa = nn.Linear(msa_encoder_dim, self.all_head_size)
            self.value_msa = nn.Linear(msa_encoder_dim, self.all_head_size)
        else:
            self.key_msa = None
            self.value_msa = None

        self.attention_norm = EvollaRMSNorm(self.hidden_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=enable_bias)

        self.ff = EvollaFeedForward(self.hidden_size, ffn_mult)
        self.gate_attention = nn.Parameter(torch.tensor([0.0]))
        self.gate_ffw = nn.Parameter(torch.tensor([0.0]))

    def cross_attention(
        self,
        query_states,
        protein_key_value_states,
        structure_key_value_states,
        msa_key_value_states,
        query_attn_mask,
        protein_kv_attn_mask,
        structure_kv_attn_mask,
        msa_kv_attn_mask,
    ):
        """
        query_states: text
        key_value_states: protein
        query_states: [bs, query_seq_len, dim]
        key_value_states: [bs, kv_seq_len, dim]
        query_attn_mask: [bs, query_seq_len]
        kv_attn_mask: [bs, kv_seq_len]
        """

        # Concatenate protein and structure
        kv_attn_mask = [protein_kv_attn_mask, structure_kv_attn_mask, msa_kv_attn_mask]
        kv_attn_mask = [_ for _ in kv_attn_mask if _ is not None]
        if not kv_attn_mask:
            raise ValueError("At least one modality should be provided for cross attention.")
        kv_attn_mask = torch.cat(kv_attn_mask, dim=1)

        query_layer = self.attention_norm(query_states)

        # Warning: This place might cause issues, refers to
        # https://discuss.pytorch.org/t/cuda-error-cublas-status-not-supported-when-calling-cublasltmatmul-from-torch-nn-functional-linear/170214/13
        # Solution: add `DISABLE_ADDMM_CUDA_LT=1` as environment variable
        # Apply linear transformation to input_query, input_key, and input_value
        query_layer = self.query(query_layer)  # [bs, querylength, dim]

        if self.key_protein is not None and self.value_protein is not None:
            protein_key_value_states = protein_key_value_states.to(query_states)
            key_layer_protein = self.key_protein(protein_key_value_states)  # [bs, keylength, dim]
            value_layer_protein = self.value_protein(protein_key_value_states)  # [bs, keylength, dim]
        else:
            key_layer_protein = None
            value_layer_protein = None

        if self.key_structure is not None and self.value_structure is not None:
            structure_key_value_states = structure_key_value_states.to(query_states)
            key_layer_structure = self.key_structure(structure_key_value_states)  # [bs, keylength, dim]
            value_layer_structure = self.value_structure(structure_key_value_states)  # [bs, keylength, dim]
        else:
            key_layer_structure = None
            value_layer_structure = None

        if self.key_msa is not None and self.value_msa is not None:
            msa_key_value_states = msa_key_value_states.to(query_states)
            key_layer_msa = self.key_msa(msa_key_value_states)  # [bs, keylength, dim]
            value_layer_msa = self.value_msa(msa_key_value_states)  # [bs, keylength, dim]
        else:
            key_layer_msa = None
            value_layer_msa = None

        key_layer = [key_layer_protein, key_layer_structure, key_layer_msa]
        key_layer = [_ for _ in key_layer if _ is not None]
        key_layer = torch.cat(key_layer, dim=1)

        value_layer = [value_layer_protein, value_layer_structure, value_layer_msa]
        value_layer = [_ for _ in value_layer if _ is not None]
        value_layer = torch.cat(value_layer, dim=1)

        new_query_layer_shape = query_layer.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        query_layer = query_layer.view(*new_query_layer_shape).permute(0, 2, 1, 3)

        new_key_layer_shape = key_layer.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        key_layer = key_layer.view(*new_key_layer_shape).permute(0, 2, 1, 3)

        new_value_layer_shape = value_layer.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        value_layer = value_layer.view(*new_value_layer_shape).permute(0, 2, 1, 3)

        query_layer = query_layer * self.scale

        # attention_mask: [bs, 1, querylength, keylength]
        attention_mask = query_attn_mask[:, None, :, None] * kv_attn_mask[:, None, None, :]
        # Compute the scaled dot-product attention scores
        attn_weights = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # [bs, numheads, querylength, keylength]
        attn_weights = attn_weights - attn_weights.amax(dim=-1, keepdim=True).detach()  # To stablize score
        attention_scores = attn_weights.masked_fill(
            (1 - attention_mask).bool(), torch.finfo(attn_weights.dtype).min
        )  # [bs, numheads, querylength, keylength]

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # attention_probs_dropped = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)  # [bs, numheads, querylength, dim/numheads]

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        context_layer = self.out_proj(context_layer)

        return context_layer

    def forward(
        self,
        query_states,
        protein_kv_states,
        structure_kv_states,
        msa_kv_states,
        query_attn_mask,
        protein_kv_attn_mask=None,
        structure_kv_attn_mask=None,
        msa_kv_attn_mask=None,
        protein_batch_mask=None,
        structure_batch_mask=None,
        msa_batch_mask=None,
        past_key_value=None,
    ):
        """
        kv_states: protein
        query_states: text

        query_states: [bs, query_seq_len, dim]
        kv_states: [bs, kv_seq_len, dim]
        query_attn_mask: [bs, query_seq_len]
        kv_attn_mask: [bs, kv_seq_len], default None
        past_key_value: [bs, past_kv_seq_len, dim], default None
        """
        if protein_kv_states is not None:
            bs, protein_kv_seq_len, dim = protein_kv_states.shape
            if protein_kv_attn_mask is None:
                protein_kv_attn_mask = (
                    torch.ones(bs, protein_kv_seq_len).to(protein_batch_mask.device)
                    * protein_batch_mask.expand(size=(protein_kv_seq_len, bs)).T
                ).to(protein_kv_states.device)
        else:
            protein_kv_attn_mask = None

        if structure_kv_states is not None:
            bs, structure_kv_seq_len, dim = structure_kv_states.shape
            if structure_kv_attn_mask is None:
                structure_kv_attn_mask = (
                    torch.ones(bs, structure_kv_seq_len).to(protein_batch_mask.device)
                    * structure_batch_mask.expand(size=(structure_kv_seq_len, bs)).T
                ).to(structure_kv_states.device)
        else:
            structure_kv_attn_mask = None

        if msa_kv_states is not None:
            bs, msa_kv_seq_len, dim = msa_kv_states.shape
            if msa_kv_attn_mask is None:
                msa_kv_attn_mask = (
                    torch.ones(bs, msa_kv_seq_len).to(protein_batch_mask.device)
                    * msa_batch_mask.expand(size=(msa_kv_seq_len, bs)).T
                ).to(msa_kv_states.device)
        else:
            msa_kv_attn_mask = None
        hidden_states = query_states
        # only when there's at least one valid modality, crossattention will be performed
        if (
            (protein_kv_states is not None and protein_kv_attn_mask.any())
            or (structure_kv_states is not None and structure_kv_attn_mask.any())
            or (msa_kv_states is not None and msa_kv_attn_mask.any())
        ):
            residual = hidden_states
            hidden_states = self.cross_attention(
                query_states=hidden_states,
                protein_key_value_states=protein_kv_states,
                structure_key_value_states=structure_kv_states,
                msa_key_value_states=msa_kv_states,
                query_attn_mask=query_attn_mask,
                protein_kv_attn_mask=protein_kv_attn_mask,
                structure_kv_attn_mask=structure_kv_attn_mask,
                msa_kv_attn_mask=msa_kv_attn_mask,
            )  # [bs, query_seq_len, dim]
            # tanh gate
            hidden_states = torch.tanh(self.gate_attention) * hidden_states

            hidden_states = residual + hidden_states  # input_query

            residual = hidden_states
            hidden_states = self.ff(hidden_states) * torch.tanh(self.gate_ffw)
            hidden_states = residual + hidden_states

        return hidden_states


# this was adapted from LlamaDecoderLayer
class EvollaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: EvollaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        if not (layer_idx + 1) % max(config.num_hidden_layers // config.aligner_num_add_layers, 1) == 0:
            self.adapter = EvollaSequenceAlignerCrossAttention(
                config,
                protein_encoder_dim=config.hidden_size,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        protein_kv_states: Optional[torch.Tensor] = None,
        structure_kv_states: Optional[torch.Tensor] = None,
        msa_kv_states: Optional[torch.Tensor] = None,
        protein_batch_mask: Optional[torch.Tensor] = None,
        structure_batch_mask: Optional[torch.Tensor] = None,
        msa_batch_mask: Optional[torch.Tensor] = None,
        query_attn_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
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

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if hasattr(self, "adapter") and self.adapter is not None:
            hidden_states = self.adapter(
                query_states=hidden_states,
                protein_kv_states=protein_kv_states,
                structure_kv_states=structure_kv_states,
                msa_kv_states=msa_kv_states,
                query_attn_mask=query_attn_mask,
                protein_batch_mask=protein_batch_mask,
                structure_batch_mask=structure_batch_mask,
                msa_batch_mask=msa_batch_mask,
            )

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class EvollaSaProtEmbeddings(EsmEmbeddings):
    def __init__(self, config):
        super().__init__()
        # remove the position_ids in EsmEmbeddings
        self.position_ids = None


def rotate_half_esm(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_esm(x, cos, sin):
    cos = cos[:, :, : x.shape[-2], :]
    sin = sin[:, :, : x.shape[-2], :]

    return (x * cos) + (rotate_half_esm(x) * sin)


class RotaryEmbedding(torch.nn.Module):
    """
    Rotary position embeddings based on those in
    [RoFormer](https://huggingface.co/docs/transformers/model_doc/roformer). Query and keys are transformed by rotation
    matrices which depend on their relative positions.
    """

    def __init__(self, dim: int):
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        inv_freq = inv_freq
        self.register_buffer("inv_freq", inv_freq)

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x, seq_dimension=2):
        seq_len = x.shape[seq_dimension]

        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if seq_len != self._seq_len_cached or self._cos_cached.device != x.device:
            self._seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dimension], device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]

        return self._cos_cached, self._sin_cached

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dimension=-2)

        return (
            apply_rotary_pos_emb_esm(q, self._cos_cached, self._sin_cached),
            apply_rotary_pos_emb_esm(k, self._cos_cached, self._sin_cached),
        )


class EvollaSaProtSelfAttention(EsmSelfAttention):
    pass


class EvollaSaProtSelfOutput(EsmSelfOutput):
    pass


class EvollaSaProtAttention(EsmAttention):
    pass


class EvollaSaProtIntermediate(EsmIntermediate):
    pass


class EvollaSaProtOutput(EsmOutput):
    pass


class EvollaSaProtLayer(EsmLayer):
    pass


class EvollaSaProtEncoder(EsmEncoder):
    pass


class EvollaSaProtPooler(EsmPooler):
    pass


class EvollaSaProtProteinEncoder(nn.Module):
    def __init__(
        self,
        config: SaProtConfig,
        add_pooling_layer: bool = False,
    ):
        super().__init__()
        self.config = config

        self.embeddings = EvollaSaProtEmbeddings(config)
        self.encoder = EvollaSaProtEncoder(config)

        self.pooler = EvollaSaProtPooler(config) if add_pooling_layer else None

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)

    def get_extended_attention_mask(
        self, attention_mask: Tensor, input_shape: Tuple[int], device: torch.device = None, dtype: torch.float = None
    ) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        if dtype is None:
            dtype = self.dtype

        if not (attention_mask.dim() == 2 and self.config.is_decoder):
            # show warning only if it won't be shown in `create_extended_attention_mask_for_decoder`
            if device is not None:
                warnings.warn(
                    "The `device` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
                )
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                extended_attention_mask = ModuleUtilsMixin.create_extended_attention_mask_for_decoder(
                    input_shape, attention_mask, device
                )
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask

    def invert_attention_mask(self, encoder_attention_mask: Tensor) -> Tensor:
        """
        Invert an attention mask (e.g., switches 0. and 1.).

        Args:
            encoder_attention_mask (`torch.Tensor`): An attention mask.

        Returns:
            `torch.Tensor`: The inverted attention mask.
        """
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
        # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
        # /transformer/transformer_layers.py#L270
        # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
        # encoder_extended_attention_mask.transpose(-1, -2))
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(self.dtype).min

        return encoder_extended_attention_mask

    def warn_if_padding_and_no_attention_mask(self, input_ids, attention_mask):
        """
        Shows a one-time warning if the input_ids appear to contain padding and no attention mask was given.
        """

        # Skip the check during tracing.
        if is_torch_fx_proxy(input_ids) or torch.jit.is_tracing() or is_torchdynamo_compiling():
            return

        if (attention_mask is not None) or (self.config.pad_token_id is None):
            return

        # Check only the first and last input IDs to reduce overhead.
        if self.config.pad_token_id in input_ids[:, [-1, 0]]:
            warn_string = (
                "We strongly recommend passing in an `attention_mask` since your input_ids may be padded. See "
                "https://huggingface.co/docs/transformers/troubleshooting"
                "#incorrect-output-when-padding-tokens-arent-masked."
            )

            # If the pad token is equal to either BOS, EOS, or SEP, we do not know whether the user should use an
            # attention_mask or not. In this case, we should still show a warning because this is a rare case.
            if (
                (self.config.bos_token_id is not None and self.config.bos_token_id == self.config.pad_token_id)
                or (self.config.eos_token_id is not None and self.config.eos_token_id == self.config.pad_token_id)
                or (self.config.sep_token_id is not None and self.config.sep_token_id == self.config.pad_token_id)
            ):
                warn_string += (
                    f"\nYou may ignore this warning if your `pad_token_id` ({self.config.pad_token_id}) is identical "
                    f"to the `bos_token_id` ({self.config.bos_token_id}), `eos_token_id` ({self.config.eos_token_id}), "
                    f"or the `sep_token_id` ({self.config.sep_token_id}), and your input is not padded."
                )

            logger.warning_once(warn_string)

    def get_head_mask(
        self, head_mask: Optional[Tensor], num_hidden_layers: int, is_attention_chunked: bool = False
    ) -> Tensor:
        """
        Prepare the head mask if needed.

        Args:
            head_mask (`torch.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (`int`):
                The number of hidden layers in the model.
            is_attention_chunked (`bool`, *optional*, defaults to `False`):
                Whether or not the attentions scores are computed by chunks or not.

        Returns:
            `torch.Tensor` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
            `[None]` for each layer.
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.to(dtype=self.dtype)  # switch to float if need + fp16 compatibility
        return head_mask


def rearrange(out):
    # [batch, head, seq, features] -> [batch, seq, head, features]
    out = out.permute(0, 2, 1, 3)

    # [batch, seq, head, features] -> [batch, seq, head*features]
    out = out.reshape(out.size(0), out.size(1), -1)

    return out


class EvollaSequenceCompressorAttention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents, mask):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D);  n2: num of latent tokens
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        h = self.heads

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(
            2, dim=-1
        )  # each: batch_size, max_protein_length+num_latents, dim_head*num_heads

        # q_raw, k_raw, v_raw = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=h)
        q = q.view(q.size(0), q.size(1), h, -1).permute(0, 2, 1, 3)
        k = k.view(k.size(0), k.size(1), h, -1).permute(0, 2, 1, 3)
        v = v.view(v.size(0), v.size(1), h, -1).permute(0, 2, 1, 3)
        # assert torch.allclose(q_raw, q)
        # assert torch.allclose(k_raw, k)
        # assert torch.allclose(v_raw, v)
        q = q * self.scale  # batch_size, num_heads, num_latents, dim_head

        # attention
        # sim = einsum("... i d, ... j d  -> ... i j", q, k)
        sim = torch.matmul(q, k.transpose(-1, -2))

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()

        bs, nh, skd, okd = sim.shape
        # mask_raw = repeat(mask, "bs okd -> bs nh skd okd", nh=nh, skd=skd)
        ones = torch.ones(nh, skd).to(mask.device)  # 创建一个全 1 的张量，形状为 (nh, skd)
        # mask = torch.einsum("bk,oj -> bojk", mask, ones)
        mask_exp = mask[:, None, None, :]
        ones_exp = ones[None, :, :, None]
        mask = mask_exp * ones_exp
        # assert torch.allclose(mask_raw, mask)

        sim = sim.masked_fill((1 - mask).bool(), -1e4)
        # sim = sim + (1 - mask) * torch.tensor(float('-inf'), dtype=sim.dtype)  # 加上mask
        attn = sim.softmax(dim=-1)

        # out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = torch.matmul(attn, v)

        out = out.permute(0, 2, 1, 3)

        # [batch, seq, head, features] -> [batch, seq, head*features]
        out = out.reshape(out.size(0), out.size(1), -1)

        # assert torch.allclose(out_raw, out)
        return self.to_out(out)


class EvollaSequenceCompressorResampler(nn.Module):
    def __init__(
        self,
        config: EvollaConfig,
    ):
        super().__init__()
        protein_repr_dim = config.protein_hidden_size
        output_repr_dim = config.hidden_size
        depth = config.resampler_depth
        dim_head = config.resampler_dim_head
        heads = config.resampler_heads
        ff_mult = config.resampler_ff_mult
        self.num_latents = config.resampler_num_latents

        self.latents = nn.Parameter(torch.randn(self.num_latents, protein_repr_dim), requires_grad=True)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        EvollaSequenceCompressorAttention(dim=protein_repr_dim, dim_head=dim_head, heads=heads),
                        EvollaFeedForward(dim=protein_repr_dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = nn.LayerNorm(output_repr_dim)

        self.protein_projector = nn.Linear(protein_repr_dim, output_repr_dim)

    @property
    def device(self):
        return self.latents.device

    @property
    def dtype(self):
        return self.latents.dtype

    def forward(self, embeds, mask):
        b = embeds.shape[0]

        bs, _ = mask.shape  # bs, max_protein_length
        latent_mask = torch.ones(bs, self.num_latents).to(mask.device)
        mask = torch.cat((mask, latent_mask), dim=1)  # bs, max_protein_length + num_latents

        # blocks
        # latents_raw = repeat(self.latents, "n d -> b n d", b=b)
        ones = torch.ones(b).to(self.latents.device)
        # latents = torch.einsum("nd, b -> bnd", self.latents, ones)
        latents = self.latents[None] * ones.view(-1, 1, 1)  # [b,n,d]
        latents = latents.to(embeds.dtype)
        # assert torch.allclose(latents_raw, latents)
        for attn, ff in self.layers:
            latents = attn(embeds, latents, mask) + latents
            latents = ff(latents) + latents

        transformed_feature = self.protein_projector(latents)

        return self.norm(transformed_feature)


# this was adapted from transformers.models.idefics.modeling_idefics.IdeficsPreTrainedModel with Idefics->Evolla
class EvollaPreTrainedModel(LlamaPreTrainedModel):
    def _init_weights(self, module):
        # important: this ported version of Evolla isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed - the m4 code
        # base should be used for training from scratch and it contains the correct code.
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
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
        elif isinstance(module, EvollaRMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, EvollaSequenceAlignerCrossAttention):
            module.gate_attention.zero_()
            module.gate_ffw.zero_()
            module.attention_norm.weight.data.fill_(1.0)
        elif isinstance(module, EvollaSequenceCompressorResampler):
            module.latents.data.normal_(mean=0.0, std=std)


class EvollaProteinEncoder(nn.Module):
    def __init__(
        self,
        config: EvollaConfig,
        add_pooling_layer: bool = False,
    ):
        super().__init__()
        protein_config = SaProtConfig(
            vocab_size=config.protein_vocab_size,
            mask_token_id=config.protein_mask_token_id,
            pad_token_id=config.protein_pad_token_id,
            hidden_size=config.protein_hidden_size,
            num_hidden_layers=config.protein_num_hidden_layers,
            num_attention_heads=config.protein_num_attention_heads,
            intermediate_size=config.protein_intermediate_size,
            hidden_dropout_prob=config.protein_hidden_dropout_prob,
            attention_probs_dropout_prob=config.protein_attention_probs_dropout_prob,
            max_position_embeddings=config.protein_max_position_embeddings,
            layer_norm_eps=config.protein_layer_norm_eps,
            position_embedding_type=config.protein_position_embedding_type,
            emb_layer_norm_before=config.protein_emb_layer_norm_before,
            token_dropout=config.protein_token_dropout,
        )
        self.model = EvollaSaProtProteinEncoder(
            config=protein_config,
            add_pooling_layer=add_pooling_layer,
        )

        self.sequence_compressor_resampler = EvollaSequenceCompressorResampler(config=config)

    def sequence_encode(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        sequence_repr = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        return sequence_repr

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        protein_output = self.sequence_encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # TODO: could be replaced by last hidden state
        protein_embeds = protein_output.last_hidden_state

        sequence_repr = self.sequence_compressor_resampler(protein_embeds, attention_mask)

        if not return_dict:
            return sequence_repr, protein_embeds, attention_mask

        return EvollaProteinEncoderModelOutput(
            sequence_compressor_output=sequence_repr,
            last_hidden_state=protein_output.last_hidden_state,
            hidden_states=protein_output.hidden_states,
            attentions=protein_output.attentions,
        )


class EvollaModel(EvollaPreTrainedModel):
    r""" """

    def __init__(
        self,
        config: EvollaConfig,
    ):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size, self.padding_idx)

        self.protein_encoder = EvollaProteinEncoder(
            config=self.config,
            add_pooling_layer=False,
        )

        self.layers = nn.ModuleList(
            [
                EvollaDecoderLayer(
                    config=config,
                    layer_idx=layer_idx,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = EvollaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = EvollaRotaryEmbedding(config=config)
        self.gradient_checkpointing = getattr(config, "gradient_checkpointing", False)
        self.config = config

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        protein_input_ids: Optional[torch.LongTensor] = None,
        protein_attention_mask: Optional[torch.Tensor] = None,
        structure_feats: Optional[torch.FloatTensor] = None,
        msa_feats: Optional[torch.FloatTensor] = None,
        structure_batch_mask: Optional[torch.Tensor] = None,
        msa_batch_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # If not provided `protein_feats`, use the `protein_encoder` to get the protein features
        if protein_input_ids is not None and protein_attention_mask is not None:
            protein_outputs = self.protein_encoder(
                input_ids=protein_input_ids,
                attention_mask=protein_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            protein_feats = protein_outputs.sequence_compressor_output
            protein_batch_mask = torch.tensor([True] * protein_input_ids.shape[0], device=protein_input_ids.device)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length, _ = inputs_embeds.shape
        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        seq_length_with_past = seq_length + past_key_values_length

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.int64, device=inputs_embeds.device
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    protein_kv_states=protein_feats,
                    structure_kv_states=structure_feats,
                    msa_kv_states=msa_feats,
                    protein_batch_mask=protein_batch_mask,
                    structure_batch_mask=structure_batch_mask,
                    msa_batch_mask=msa_batch_mask,
                    query_attn_mask=attention_mask,
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
                    protein_kv_states=protein_feats,
                    structure_kv_states=structure_feats,
                    msa_kv_states=msa_feats,
                    protein_batch_mask=protein_batch_mask,
                    structure_batch_mask=structure_batch_mask,
                    msa_batch_mask=msa_batch_mask,
                    query_attn_mask=attention_mask,
                    **kwargs,
                )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()

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
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
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
            device=device,
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
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
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
            device (`torch.device`):
                The device to place the 4D attention mask on.
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
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
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


# this was adapted from modeling_idefics.IdeficsForVisionText2Text
class EvollaForProteinText2Text(EvollaPreTrainedModel, GenerationMixin):
    # _keys_to_ignore_on_load_missing = [r"lm_head.weight"]
    _tied_weights_keys = []

    def __init__(self, config):
        super().__init__(config)
        self.model = EvollaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, self.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.model.set_input_embeddings(value)

    def forward(
        self,
        input_ids: torch.LongTensor = None,  # text input ids
        attention_mask: Optional[torch.Tensor] = None,  # text attention mask
        inputs_embeds: Optional[torch.FloatTensor] = None,  # text input embeddings
        labels: Optional[torch.LongTensor] = None,
        protein_input_ids: torch.LongTensor = None,
        protein_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        r"""
        Args:

        Returns:

        Example:

        ```python
        >>> from transformers import EvollaProcessor, EvollaForProteinText2Text
        >>> model = EvollaForProteinText2Text.from_pretrained("westlake/Evolla-10B-hf")
        >>> processor = EvollaProcessor.from_pretrained("westlake/Evolla-10B-hf")

        >>> protein_information = {
            "aa_seq": "your amino acid sequence",
            "foldseek": "your foldseek sequence",
        }
        >>> question = "What is the function of this protein?"
        >>> message = [
            {"role": "system", "content": "You are an AI expert that can answer any questions about protein."},
            {"role": "user", "content": question},
        ]

        >>> inputs = processor(proteins=[protein_information], messages_list=[message], return_tensors="pt", padding="longest")
        >>> outputs = model.generate(**inputs)

        >>> print(processor.batch_decode(outputs, skip_special_tokens=True))
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            protein_input_ids=protein_input_ids,
            protein_attention_mask=protein_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            return_dict=True,
            **kwargs,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.vocab_size, **kwargs)

        lm_outputs = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        return lm_outputs if return_dict else lm_outputs.to_tuple()


__all__ = ["EvollaForProteinText2Text", "EvollaModel", "EvollaPreTrainedModel"]
