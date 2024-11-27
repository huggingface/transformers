# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, StaticCache
from ...generation import GenerationMixin
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_flash_attention_utils import _flash_attention_forward
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from .configuration_llama import LlamaConfig
from .modeling_llama import (
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    apply_rotary_pos_emb,
    LlamaMLP,
    repeat_kv,
    LlamaFlashAttention2,
    LlamaSdpaAttention,
)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"

from dataclasses import dataclass

from enum import Enum

from torch.distributions.categorical import Categorical

@dataclass
class AdaptiveBaseModelOutputWithPast(BaseModelOutputWithPast):
    mean_merged_tokens: Optional[int] = None
    fan_in_merging_maps: Optional[torch.Tensor] = None
    fan_in_merging_logits: Optional[torch.Tensor] = None

@dataclass
class AdaptiveCausalLMOutputWithPast(CausalLMOutputWithPast):
    mean_merged_tokens: Optional[int] = None
    fan_in_merging_maps: Optional[torch.Tensor] = None
    fan_in_merging_logits: Optional[torch.Tensor] = None

class AdaptiveMode(Enum):
    FAN_IN = "fan_in"
    FAN_OUT = "fan_out"

@dataclass
class AdaptiveFanInOutput:
    # new_seq_len is less then input seq_len
    hidden_state: torch.Tensor # [ bs, new_seq_len, hidden_size ]
    attention_mask: torch.Tensor # [ bs, new_seq_len ]

    # mask for bos and eos embeddings that should be never merged
    # should be used in subsequent adaptive fan in modules
    special_embeddings_mask: torch.Tensor # [ bs, new_seq_len ]

    # merged_tokens_counts represents how many embeddings
    # has been merged in the corresponding output embedding
    # Eg: merged_tokens_counts = [ 1, 5, 2 ]
    # This means that:
    # * the first embedding was not merged
    # * the second one has been merged with 5 embeddings
    # * the third one has been merged with 2 embeddings
    merged_embeddings_counts: torch.Tensor # [ bs, new_seq_len ]

    merging_map: torch.Tensor
    merging_map_logits: torch.Tensor

@dataclass
class AdaptiveFanOutOutput:
    hidden_state: torch.Tensor # [ bs, restored_seq_len, hidden_size ]

class NoOpFanIn(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()

    def forward(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor, special_embeddings_mask: torch.Tensor, merging_log_probas: torch.Tensor=None, inverted_merging_map = None) -> AdaptiveFanInOutput:
        res = AdaptiveFanInOutput(
            hidden_state=hidden_state,
            attention_mask=attention_mask,
            merged_embeddings_counts=attention_mask,
            special_embeddings_mask=special_embeddings_mask,
            merging_map=None,
        )

        return res

def scaled_gumbel_softmax(
        logits,
        tau: float = 1,
        scale = 1.0,
        invert = False,
        hard = True,
        dim: int = -1,
    ):

    assert hard, 'soft mode is not supported'

    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
        .exponential_()
        .log()
    )  # ~Gumbel(0,1)
    if not gumbels.isfinite().all():
        print("gumbels are infinite")
        breakpoint()

    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim) * scale

    # Straight through.
    index = y_soft.max(dim, keepdim=True)[1]
    if invert:
        index = 1 - index

    y_hard = torch.zeros_like(
        logits, memory_format=torch.legacy_contiguous_format
    ).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft

    return ret

class AdaptiveFanInGumbel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.fan_in_mlp = nn.Linear(self.hidden_size * 2, 2)

    def generate_merges_transform(self, merging_map, attention_mask):
        # merging_map ~ [ bs, seq_len, 2 ]
        batch_size, seq_len = merging_map.shape[:2]
        device = merging_map.device

        # [
        #   [ 1, 2, 2, 3, 1, 0 ],
        #   [ 1, 2, 2, 1, 0, 0 ],
        # ]
        merged_embeddings_counts = torch.zeros([batch_size, seq_len], device=device)
        merged_embeddings_counts[:, 0] = 1

        # [
        #   [ 1, 1, 1, 1, 1, 0 ],
        #   [ 1, 1, 1, 1, 0, 0 ],
        # ]
        merged_attention_mask = torch.zeros([batch_size, seq_len], device=device)

        aggregated_embeddings_transform = torch.zeros([batch_size, seq_len, seq_len], device=device)
        aggregated_embeddings_transform[:, 0, 0] = 1

        total_initial_num_embeddings = attention_mask.sum(dim=-1).to(torch.long)

        max_new_seq_len = 0
        for batch_i in range(batch_size):
            new_seq_len_i = 1
            buffer_length = 0
            start_want_merge = 0
            total_tokens_count = total_initial_num_embeddings[batch_i].item()

            for seq_len_i in range(1, total_tokens_count):
                want_merge = merging_map[batch_i, seq_len_i, 1].item() > 0.5
                if want_merge and seq_len_i < total_tokens_count - 1:
                    if buffer_length == 0:
                        start_want_merge = seq_len_i
                    buffer_length += 1
                else:
                    if buffer_length > 0:
                        merged_embeddings_counts[batch_i, new_seq_len_i] = seq_len_i - start_want_merge
                        aggregated_embeddings_transform[batch_i, new_seq_len_i, start_want_merge:seq_len_i] = merging_map[batch_i, start_want_merge:seq_len_i, 1]
                        new_seq_len_i += 1
                        buffer_length = 0

                    aggregated_embeddings_transform[batch_i, new_seq_len_i, seq_len_i] = merging_map[batch_i, seq_len_i, 0]
                    merged_embeddings_counts[batch_i, new_seq_len_i] = 1
                    new_seq_len_i += 1


            merged_attention_mask[batch_i, :new_seq_len_i] = 1
            max_new_seq_len = max(max_new_seq_len, new_seq_len_i)

        # [ bs, new_seq_len ]
        merged_embeddings_counts = merged_embeddings_counts[:, :max_new_seq_len]
        merged_attention_mask = merged_attention_mask[:, :max_new_seq_len]
        # [ bs, new_seq_len, seq_len ]
        aggregated_embeddings_transform = aggregated_embeddings_transform[:, :max_new_seq_len, :]

        return aggregated_embeddings_transform, merged_embeddings_counts, merged_attention_mask


    def forward(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor, special_embeddings_mask: torch.Tensor, merging_log_probas: torch.Tensor=None, inverted_merging_map = None) -> AdaptiveFanInOutput:
        # TODO attention mask transforms
        # TODO return mirroring layer merging informarion to restore

        # hidden_state ~ [ bs, seq_len, hidden_size ]
        assert hidden_state.shape[-1] == self.hidden_size

        # attention_mask ~ [ bs, seq_len ]
        assert hidden_state.shape[:2] == attention_mask.shape
        assert special_embeddings_mask.shape == attention_mask.shape

        batch_size = hidden_state.shape[0]
        seq_len = hidden_state.shape[1]
        hidden_dim = hidden_state.shape[2]

        merging_mask_stub = torch.zeros([batch_size, 1, hidden_dim * 2], device=hidden_state.device)

        # joined prev and next tokens
        # each embedding could be explained as: should it be merged with the next one embedding?
        # [ bs, seq_len, hidden_size * 2 ]
        attn_output_pairs = torch.cat([ hidden_state[:, :-1], hidden_state[:, 1:] ], dim=-1)
        attn_output_pairs = torch.cat([ attn_output_pairs, merging_mask_stub ], dim=1)
        # [ bs, seq_len, 2 ] # should be merged or not (probas)?

        merging_log_probas = self.fan_in_mlp(attn_output_pairs)
        if merging_log_probas.isnan().any() or not merging_log_probas.isfinite().all():
            print("found nan merging_log_probas!")
            breakpoint()
            raise Exception("found nan merging_log_probas!")

        # merge all by default
        # merging_log_probas_bias = torch.ones_like(merging_log_probas) * 10
        # merging_log_probas_bias[:, :, 0] = 0
        # merging_log_probas += merging_log_probas_bias

        if merging_log_probas.isnan().any() or not merging_log_probas.isfinite().all():
            print("found nan merging_log_probas!")
            breakpoint()
            raise Exception("found nan merging_log_probas!")

        # OHE: [ bs, seq_len, 2 ]
        if self.training:
            merging_map = scaled_gumbel_softmax(merging_log_probas, hard=True, dim=-1, invert=inverted_merging_map)
        else:
            merging_map = torch.zeros_like(merging_log_probas)
            merging_map[:, :, 0] = (merging_log_probas[:, :, 0] > merging_log_probas[:, :, 1]).to(torch.float32)
            merging_map[:, :, 1] = 1 - merging_map[:, :, 0]
            # merging_map = torch.zeros_like(merging_log_probas)
            # merging_map.masked_fill_()

        # merging_map[:, -1, 0] = 1
        # merging_map[:, -1, 1] = 0
        merging_map[special_embeddings_mask.bool()] = torch.tensor([1., 0.], device=merging_map.device)
        merging_map[~attention_mask.bool()] = 0
        assert (merging_map.sum(dim=-1) == 1).sum().item() == attention_mask.sum().item()
        # print("attention_mask", attention_mask.sum())
        # print("merged tokens:", merging_map[:, :, 1].sum())

        # merging_log_probas[special_embeddings_mask[:, :-1]] = 0

        # [ bs, new_seq_len, seq_len ]
        merged_embeddings_transform, merged_embeddings_counts, merged_attention_mask = self.generate_merges_transform(merging_map, attention_mask)
        if merged_embeddings_transform.isnan().any() or not merged_embeddings_transform.isfinite().all():
            print("found nan merged_embeddings_transform!")
            breakpoint()
            raise Exception("found nan merged_embeddings_transform!")

        merged_special_embeddings_mask = torch.zeros([batch_size, merged_embeddings_transform.shape[1]], device=hidden_state.device)
        merged_special_embeddings_mask[:, 0] = 1
        merged_special_embeddings_mask[:, merged_attention_mask.sum(dim=-1).to(torch.long) - 1] = 1

        # [ bs, new_seq_len, emb_dim ] = [ bs, new_seq_len, seq_len ] @ [ bs, seq_len, emb_dim ]
        merged_attention_outputs = torch.bmm(merged_embeddings_transform, hidden_state)
        # Sum is better than mean
        # merged_attention_outputs = merged_attention_outputs / (merged_embeddings_counts.unsqueeze(-1) + 1e-6)

        # sum_merged_tokens = merged_embeddings_counts[(merged_embeddings_counts > 1)].sum()
        # print("sum_merged_tokens", sum_merged_tokens)
        # breakpoint()

        if merged_attention_outputs.isnan().any():
            print("found nan merged_attention_outputs!")
            breakpoint()
            raise Exception("found nan merged_attention_outputs!")

        res = AdaptiveFanInOutput(
            hidden_state=merged_attention_outputs,
            attention_mask=merged_attention_mask,
            merged_embeddings_counts=merged_embeddings_counts,
            special_embeddings_mask=merged_special_embeddings_mask,
            merging_map=merging_map,
            merging_map_logits=merging_log_probas,
        )

        return res



class AdaptiveFanOut(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        # self.fan_out_mlp = nn.Linear(self.hidden_size * 2, self.hidden_size)

    def forward(self, hidden_states, attention_mask, merged_embeddings_counts, residual_hidden_states, residual_attention_mask) -> AdaptiveFanOutOutput:
        # merged_embeddings_counts from corresponding mirroring
        # AdaptiveFanIn output
        #
        # residual_hidden_states - hidden_states from corresponding mirroring
        # residual_attention_mask - attention_mask from corresponding mirroring

        # hidden_states ~ [ batch_size, new_seq_len, hidden_size ]
        # attention_mask ~ [ batch_size, new_seq_len ]
        # merged_embeddings_counts ~ [ batch_size, new_seq_len ]
        assert hidden_states.shape[1] == attention_mask.shape[1], 'seq len mismatch'
        assert hidden_states.shape[1] == merged_embeddings_counts.shape[1], 'seq len mismatch'

        assert (merged_embeddings_counts.sum(dim=-1) == residual_attention_mask.sum(dim=-1)).all(), 'merged_embeddings_counts and residual_attention_mask mismatch'

        # residual_hidden_states ~ [ batch_size, seq_len, hidden_size ]
        # residual_hidden_states ~ [ batch_size, seq_len ]
        assert residual_hidden_states.shape[1] == residual_attention_mask.shape[1], 'seq len mismatch'

        new_seq_len = attention_mask.shape[1]
        seq_len = residual_attention_mask.shape[1]
        assert seq_len >= new_seq_len, 'residual seq len cant be less then input_embeddings seq_len'

        restored_hidden_states = torch.zeros_like(residual_hidden_states) + residual_hidden_states
        for batch_i in range(attention_mask.shape[0]):
            restored_seq_len = 0
            for seq_len_i in range(new_seq_len):
                num_repeats = merged_embeddings_counts[batch_i, seq_len_i].item()
                if num_repeats == 0:
                    break

                current_hidden_state = hidden_states[batch_i, seq_len_i]

                restored_idx = int(restored_seq_len + num_repeats - 1)
                restored_hidden_states[batch_i, restored_idx] += current_hidden_state
                restored_seq_len += num_repeats

        # DONE restore hidden states with no data leackage
        #       Будем сохранять резидуал только для последнего токена

        # TODO Но как тогда сделать мерджинг произвольного количества эмб на разных слоях?
        # TODO посмотреть RWKW и RetNet - https://datasecrets.ru/articles/19

        assert restored_hidden_states.shape == residual_hidden_states.shape

        return AdaptiveFanOutOutput(hidden_state=restored_hidden_states)


class AdaptiveLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        # TODO (joao): remove in v4.46 (RoPE is computed in the model, not in the decoder layers)
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)


        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value



class AdaptiveLlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        assert config._attn_implementation == 'eager'

        self.self_attn = AdaptiveLlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
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

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
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

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["AdaptiveLlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance, see our
            [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache);
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
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
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class AdaptiveLlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`AdaptiveLlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        assert config.num_hidden_layers % 2 == 0
        num_hidden_layers_half = config.num_hidden_layers // 2


        is_dummy_fan_in = config.dummy_adaptive_fan_in
        if is_dummy_fan_in is None:
            is_dummy_fan_in = [ False ] * num_hidden_layers_half

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        def get_fan_in_module(is_dummy):
            if is_dummy:
                return NoOpFanIn(config)

            return AdaptiveFanInGumbel(config)
        self.adaptive_down = nn.ModuleList(
            [get_fan_in_module(is_dummy_fan_in[i]) for i in range(num_hidden_layers_half)]
        )
        self.layers_down = nn.ModuleList(
            [AdaptiveLlamaDecoderLayer(config, layer_idx) for layer_idx in range(num_hidden_layers_half)]
        )
        self.layers_up = nn.ModuleList(
            [AdaptiveLlamaDecoderLayer(config, layer_idx) for layer_idx in range(num_hidden_layers_half)]
        )
        self.adaptive_up = nn.ModuleList(
            [AdaptiveFanOut(config) for _ in range(num_hidden_layers_half)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        special_embeddings_mask: Optional[torch.Tensor] = None,
        inverted_merging_map=None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, AdaptiveBaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
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
        next_decoder_cache = None

        # all_loop_down_special_embeddings_mask = [ ]
        all_loop_down_merged_embeddings_counts = [ ]
        all_loop_down_attention_mask = []
        all_loop_down_causal_mask = [ ]
        all_loop_down_position_embeddings = [ ]
        all_loop_down_position_ids = [ ]
        all_loop_down_hidden_states = []

        loop_down_special_embeddings_mask = special_embeddings_mask
        loop_down_merged_embeddings_counts = None
        loop_down_attention_mask = attention_mask
        loop_down_causal_mask = causal_mask
        loop_down_position_ids = position_ids
        loop_down_position_embeddings = position_embeddings


        fan_in_merging_maps = []
        fan_in_merging_logits = []

        for i, (decoder_layer, adaptive_down_layer) in enumerate(zip(self.layers_down, self.adaptive_down)):
            current_inverted_merging_map = None
            if inverted_merging_map is not None:
                current_inverted_merging_map = inverted_merging_map[i]

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # all_loop_down_special_embeddings_mask.append(loop_down_special_embeddings_mask)
            all_loop_down_merged_embeddings_counts.append(loop_down_merged_embeddings_counts)
            all_loop_down_attention_mask.append(loop_down_attention_mask)
            all_loop_down_causal_mask.append(loop_down_causal_mask)
            all_loop_down_position_embeddings.append(loop_down_position_embeddings)
            all_loop_down_hidden_states.append(hidden_states)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    loop_down_causal_mask,
                    loop_down_position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    loop_down_position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=loop_down_causal_mask,
                    position_ids=loop_down_position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=loop_down_position_embeddings,
                )

            hidden_states = layer_outputs[0]

            adaptive_down_layer: AdaptiveFanIn
            adaptive_down_output: AdaptiveFanInOutput = adaptive_down_layer.forward(
                hidden_state=hidden_states,
                attention_mask=loop_down_attention_mask,
                special_embeddings_mask=loop_down_special_embeddings_mask,
                inverted_merging_map=current_inverted_merging_map,
            )

            fan_in_merging_maps.append(adaptive_down_output.merging_map)
            fan_in_merging_logits.append(adaptive_down_output.merging_map_logits)

            hidden_states = adaptive_down_output.hidden_state
            loop_down_attention_mask = adaptive_down_output.attention_mask
            loop_down_merged_embeddings_counts = adaptive_down_output.merged_embeddings_counts

            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + hidden_states.shape[1], device=hidden_states.device
            )
            loop_down_position_ids = cache_position.unsqueeze(0)
            loop_down_position_embeddings = self.rotary_emb(hidden_states, loop_down_position_ids)
            all_loop_down_position_ids.append(loop_down_position_ids)

            loop_down_causal_mask = self._update_causal_mask(
                loop_down_attention_mask, hidden_states, cache_position, past_key_values, output_attentions
            )

            loop_down_special_embeddings_mask = adaptive_down_output.special_embeddings_mask

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)


        # Save the last loop objects
        # all_loop_down_special_embeddings_mask.append(loop_down_special_embeddings_mask)
        all_loop_down_merged_embeddings_counts.append(loop_down_merged_embeddings_counts)
        all_loop_down_attention_mask.append(loop_down_attention_mask)

        #     print("Total count of merged tokens:", [ (x > 1).sum() for x in all_loop_down_merged_embeddings_counts if x is not None ])
        mean_merged_tokens = sum([ x[x > 1].sum() for x in all_loop_down_merged_embeddings_counts if x is not None ]).item()

        # all_loop_down_causal_mask.append(loop_down_causal_mask)
        # all_loop_down_position_embeddings.append(loop_down_position_embeddings)

        # DO NOT ADD LAST HIDDEN STATE
        # it is already exists in hidden_states variable
        # all_loop_down_hidden_states.append(hidden_states)

        for i, decoder_layer in enumerate(self.layers_up):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            adaptive_up_layer: AdaptiveFanOut = self.adaptive_up[i]
            loop_up_attention_mask = all_loop_down_attention_mask.pop(-1)
            merged_embeddings_counts = all_loop_down_merged_embeddings_counts.pop(-1)
            residual_hidden_states = all_loop_down_hidden_states.pop(-1)
            residual_attention_mask = all_loop_down_attention_mask[-1]

            adaptive_up_output: AdaptiveFanOutOutput = adaptive_up_layer.forward(
                hidden_states,
                loop_up_attention_mask,
                merged_embeddings_counts,
                residual_hidden_states,
                residual_attention_mask,
            )

            hidden_states = adaptive_up_output.hidden_state

            # all_loop_down_special_embeddings_mask
            loop_up_causal_mask = all_loop_down_causal_mask.pop(-1)
            loop_up_position_embeddings = all_loop_down_position_embeddings.pop(-1)
            loop_up_position_ids = all_loop_down_position_ids.pop(-1)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    loop_up_causal_mask,
                    loop_up_position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    loop_up_position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=loop_up_causal_mask,
                    position_ids=loop_up_position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=loop_up_position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return AdaptiveBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            mean_merged_tokens=mean_merged_tokens,
            fan_in_merging_maps=fan_in_merging_maps,
            fan_in_merging_logits=fan_in_merging_logits,
        )

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
            and attention_mask.device.type == "cuda"
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
                The device to plcae the 4D attention mask on.
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
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


class AdaptiveLlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = AdaptiveLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    # @replace_return_docstrings(output_type=AdaptiveCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        special_embeddings_mask: Optional[torch.Tensor] = None,
        inverted_merging_map=None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, AdaptiveCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, AdaptiveLlamaForCausalLM

        >>> model = AdaptiveLlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        use_cache = False

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            special_embeddings_mask=special_embeddings_mask,
            inverted_merging_map=inverted_merging_map,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            if loss.isnan().any():
                print("Found nan loss!")
                breakpoint()

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return AdaptiveCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            mean_merged_tokens=outputs.mean_merged_tokens,
            fan_in_merging_maps=outputs.fan_in_merging_maps,
            fan_in_merging_logits=outputs.fan_in_merging_logits,
        )

