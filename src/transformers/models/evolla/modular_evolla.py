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
from typing import Optional, Union

import torch
from torch import Tensor, nn

from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithPast,
    ModelOutput,
)
from ...modeling_utils import ModuleUtilsMixin, PreTrainedModel, get_parameter_dtype
from ...utils import (
    auto_docstring,
    can_return_tuple,
    logging,
)
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import OutputRecorder, check_model_inputs
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


logger = logging.get_logger(__name__)


class EvollaSaProtEmbeddings(EsmEmbeddings):
    def __init__(self, config):
        super().__init__(config)
        # remove the position_ids in EsmEmbeddings
        self.position_ids = None


def rotate_half_esm(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_esm(x, cos, sin):
    cos = cos[:, :, : x.shape[-2], :]
    sin = sin[:, :, : x.shape[-2], :]

    return (x * cos) + (rotate_half_esm(x) * sin)


class EvollaSaProtRotaryEmbedding(nn.Module):
    """
    Rotary position embeddings based on those in
    [RoFormer](https://huggingface.co/docs/transformers/model_doc/roformer). Query and keys are transformed by rotation
    matrices which depend on their relative positions.
    """

    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, dim: int):
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
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

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dimension=-2)

        return (
            apply_rotary_pos_emb_esm(q, self._cos_cached, self._sin_cached).to(dtype=q.dtype),
            apply_rotary_pos_emb_esm(k, self._cos_cached, self._sin_cached).to(dtype=k.dtype),
        )


class EvollaSaProtSelfAttention(EsmSelfAttention):
    def __init__(self, config, position_embedding_type=None, layer_idx=None, is_cross_attention=False):
        nn.Module.__init__(self)
        self.config = config

        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = config.attention_probs_dropout_prob
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        self.rotary_embeddings = None
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        elif self.position_embedding_type == "rotary":
            self.rotary_embeddings = EvollaSaProtRotaryEmbedding(dim=self.attention_head_size)

        self.is_decoder = config.is_decoder
        self.layer_idx = layer_idx
        self.scaling = 1.0
        self.is_causal = self.is_decoder and not is_cross_attention


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


@auto_docstring
class EvollaSaProtPreTrainedModel(PreTrainedModel):
    config: SaProtConfig
    _no_split_modules = ["EvollaSaProtLayer"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_attention_backend = True

    _can_record_outputs = {
        "hidden_states": EvollaSaProtLayer,
        "attentions": [OutputRecorder(EvollaSaProtSelfAttention, index=1, layer_name="attention")],
        "cross_attentions": [
            OutputRecorder(EvollaSaProtSelfAttention, index=1, layer_name="crossattention"),
        ],
    }

    def _init_weights(self, module):
        """Initialize the weights"""
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


class EvollaSaProtProteinEncoder(EvollaSaProtPreTrainedModel):
    def __init__(self, config: SaProtConfig):
        super().__init__(config)
        self.embeddings = EvollaSaProtEmbeddings(config)
        self.encoder = EvollaSaProtEncoder(config)

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

    @check_model_inputs
    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Union[tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape

        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)

        inputs_embeds = self.embeddings(input_ids=input_ids, attention_mask=attention_mask)
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)
        encoder_outputs = self.encoder(inputs_embeds, attention_mask=extended_attention_mask)
        sequence_output = encoder_outputs[0]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

    def get_extended_attention_mask(
        self,
        attention_mask: Tensor,
        input_shape: tuple[int],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
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
            dtype = get_parameter_dtype(self)

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

        q = q.view(q.size(0), q.size(1), h, -1).permute(0, 2, 1, 3)
        k = k.view(k.size(0), k.size(1), h, -1).permute(0, 2, 1, 3)
        v = v.view(v.size(0), v.size(1), h, -1).permute(0, 2, 1, 3)
        q = q * self.scale  # batch_size, num_heads, num_latents, dim_head

        # attention
        sim = torch.matmul(q, k.transpose(-1, -2))
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        bs, nh, skd, okd = sim.shape
        ones = torch.ones(nh, skd).to(mask.device)  # Create a tensor of ones with shape (nh, skd)
        mask_exp = mask[:, None, None, :]
        ones_exp = ones[None, :, :, None]
        mask = mask_exp * ones_exp

        sim = sim.masked_fill((1 - mask).bool(), -1e4)
        attn = sim.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3)

        # [batch, seq, head, features] -> [batch, seq, head*features]
        out = out.reshape(out.size(0), out.size(1), -1)

        return self.to_out(out)


class EvollaFeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        inner_dim = int(dim * mult)

        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, inner_dim, bias=False)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(self.norm(x))))


class EvollaSequenceCompressorResampler(nn.Module):
    def __init__(self, config: EvollaConfig):
        super().__init__()
        protein_repr_dim = config.protein_encoder_config.hidden_size
        self.num_latents = config.resampler_num_latents
        self.latents = nn.Parameter(torch.randn(self.num_latents, protein_repr_dim), requires_grad=True)
        self.layers = nn.ModuleList([])
        for _ in range(config.resampler_depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        EvollaSequenceCompressorAttention(
                            dim=protein_repr_dim, dim_head=config.resampler_dim_head, heads=config.resampler_heads
                        ),
                        EvollaFeedForward(dim=protein_repr_dim, mult=config.resampler_ff_mult),
                    ]
                )
            )

        self.norm = nn.LayerNorm(config.hidden_size)
        self.protein_projector = nn.Linear(protein_repr_dim, config.hidden_size)

    def forward(self, embeds, mask):
        b = embeds.shape[0]

        bs, _ = mask.shape  # bs, max_protein_length
        latent_mask = torch.ones(bs, self.num_latents).to(mask.device)
        mask = torch.cat((mask, latent_mask), dim=1)  # bs, max_protein_length + num_latents

        # blocks
        ones = torch.ones(b).to(self.latents.device)
        latents = self.latents[None] * ones.view(-1, 1, 1)  # [b,n,d]
        latents = latents.to(embeds.dtype)
        for attn, ff in self.layers:
            latents = attn(embeds, latents, mask) + latents
            latents = ff(latents) + latents

        transformed_feature = self.protein_projector(latents)

        return self.norm(transformed_feature)


@dataclass
@auto_docstring
class EvollaProteinEncoderModelOutput(ModelOutput):
    sequence_compressor_output: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


class EvollaProteinEncoder(nn.Module):
    def __init__(self, config: EvollaConfig):
        super().__init__()
        self.model = EvollaSaProtProteinEncoder(config=config.protein_encoder_config)
        self.sequence_compressor_resampler = EvollaSequenceCompressorResampler(config=config)

    @can_return_tuple
    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.FloatTensor, **kwargs):
        protein_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        protein_embeds = protein_output.last_hidden_state
        sequence_repr = self.sequence_compressor_resampler(protein_embeds, attention_mask)

        return EvollaProteinEncoderModelOutput(
            sequence_compressor_output=sequence_repr,
            last_hidden_state=protein_output.last_hidden_state,
        )


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

        attention_probs_dropout_prob = config.aligner_attention_probs_dropout_prob
        enable_bias = config.aligner_enable_bias
        ffn_mult = config.aligner_ffn_mult

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
        if query_attn_mask is None:
            query_attn_mask = torch.ones(query_states.size(0), query_states.size(1)).to(query_states.device)
        attention_mask = query_attn_mask[:, None, :, None] * kv_attn_mask[:, None, None, :]
        # Compute the scaled dot-product attention scores
        attn_weights = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # [bs, numheads, querylength, keylength]
        attn_weights = attn_weights - attn_weights.amax(dim=-1, keepdim=True).detach()  # To stabilize score
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

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
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
        past_key_values=None,
    ):
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


class EvollaRMSNorm(LlamaRMSNorm):
    pass


class EvollaRotaryEmbedding(LlamaRotaryEmbedding):
    pass


class EvollaMLP(LlamaMLP):
    pass


class EvollaAttention(LlamaAttention):
    pass


class EvollaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: EvollaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        if (layer_idx + 1) % max(config.num_hidden_layers // config.aligner_num_add_layers, 1) == 0:
            self.adapter = EvollaSequenceAlignerCrossAttention(
                config,
                protein_encoder_dim=config.hidden_size,
            )

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
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

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if hasattr(self, "adapter"):
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

        return hidden_states


class EvollaPreTrainedModel(LlamaPreTrainedModel):
    _supports_flash_attn = False  # see dependency on `EvollaSaProtProteinEncoder`
    _supports_flex_attn = False  # see dependency on `EvollaSaProtProteinEncoder`
    _supports_attention_backend = False
    _no_split_modules = [
        "EvollaDecoderLayer",
        "EvollaSequenceCompressorResampler",
        "EvollaSequenceAlignerCrossAttention",
    ]

    def _init_weights(self, module):
        std = self.config.initializer_range
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, EvollaSequenceAlignerCrossAttention):
            module.gate_attention.zero_()
            module.gate_ffw.zero_()
            module.attention_norm.weight.data.fill_(1.0)
        elif isinstance(module, EvollaSequenceCompressorResampler):
            module.latents.data.normal_(mean=0.0, std=std)


class EvollaModel(EvollaPreTrainedModel):
    def __init__(self, config: EvollaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size, self.padding_idx)
        self.protein_encoder = EvollaProteinEncoder(config=config)
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
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @auto_docstring
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
        protein_input_ids: Optional[torch.LongTensor] = None,
        protein_attention_mask: Optional[torch.Tensor] = None,
        structure_feats: Optional[torch.FloatTensor] = None,
        msa_feats: Optional[torch.FloatTensor] = None,
        structure_batch_mask: Optional[torch.Tensor] = None,
        msa_batch_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[tuple, BaseModelOutputWithPast]:
        r"""
        protein_input_ids (torch.LongTensor):
            The input IDs for the protein sequence in structure-aware tokens. Should be of shape `(batch_size, protein_seq_length)` and type `torch.LongTensor`.
        protein_attention_mask (torch.Tensor):
            The attention mask for the protein sequence. Should be of shape `(batch_size, protein_seq_length)` and type `torch.Tensor`.
        structure_feats (torch.FloatTensor):
            The input IDs for purely structure-based features. Should be of shape `(batch_size, structure_seq_length, structure_feat_dim)` and type `torch.FloatTensor`. Dummy input for now.
        msa_feats (torch.FloatTensor):
            The input IDs for purely MSA-based features. Should be of shape `(batch_size, msa_seq_length, msa_feat_dim)` and type `torch.FloatTensor`. Dummy input for now.
        structure_batch_mask (torch.Tensor):
            The batch mask to decide which protein sequences are purely structure-based. Should be of shape `(batch_size)` and type `torch.Tensor`. Should be paired with `structure_feats`. Dummpy input for now.
        msa_batch_mask (torch.Tensor):
            The batch mask to decide which protein sequences are purely MSA-based. Should be of shape `(batch_size)` and type `torch.Tensor`. Should be paired with `msa_feats`. Dummpy input for now.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        protein_feats = None
        protein_batch_mask = None
        # If provided, actually compute them
        if protein_input_ids is not None and protein_attention_mask is not None:
            protein_outputs = self.protein_encoder(
                input_ids=protein_input_ids,
                attention_mask=protein_attention_mask,
            )
            protein_feats = protein_outputs.sequence_compressor_output
            protein_batch_mask = torch.tensor([True] * protein_input_ids.shape[0], device=protein_input_ids.device)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
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

        hidden_states = self.norm(hidden_states)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )
        return output


class EvollaForProteinText2Text(EvollaPreTrainedModel, GenerationMixin):
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

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # text input ids
        attention_mask: Optional[torch.Tensor] = None,  # text attention mask
        inputs_embeds: Optional[torch.FloatTensor] = None,  # text input embeddings
        labels: Optional[torch.LongTensor] = None,
        protein_input_ids: Optional[torch.LongTensor] = None,
        protein_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ):
        r"""
        protein_input_ids (torch.LongTensor):
            The input IDs for the protein sequence. Should be of shape `(batch_size, protein_seq_length)` and type `torch.LongTensor`.
        protein_attention_mask (torch.Tensor):
            The attention mask for the protein sequence. Should be of shape `(batch_size, protein_seq_length)` and type `torch.Tensor`.

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

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            protein_input_ids=protein_input_ids,
            protein_attention_mask=protein_attention_mask,
            use_cache=use_cache,
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
        return lm_outputs


__all__ = ["EvollaForProteinText2Text", "EvollaModel", "EvollaPreTrainedModel"]
