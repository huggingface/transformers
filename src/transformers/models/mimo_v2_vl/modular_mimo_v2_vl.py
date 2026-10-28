# Copyright 2026 Xiaomi Corporation and the HuggingFace Inc. team. All rights reserved.
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
"""Native image, video, and audio understanding for the MiMo V2 VL series."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import DynamicCache
from ...generation import GenerationMixin
from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature
from ...image_utils import ChannelDimension, ImageInput, PILImageResampling, SizeDict
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_mimo_v2_utils import load_original_checkpoint_if_needed
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TensorType, auto_docstring, can_return_tuple, is_torch_greater_or_equal, logging
from ...utils.generic import no_inherit_decorator
from ...utils.output_capturing import capture_outputs
from ...video_processing_utils import BaseVideoProcessor
from ...video_utils import (
    VideoInput,
    VideoMetadata,
    group_videos_by_shape,
    make_batched_metadata,
    make_batched_videos,
    reorder_videos,
)
from ..encodec.modeling_encodec import EncodecEuclideanCodebook
from ..llama.modeling_llama import LlamaMLP, LlamaRMSNorm, rotate_half
from ..qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionPatchEmbed
from ..qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor, Qwen2VLImageProcessorKwargs
from ..qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor, Qwen2VLVideoProcessorInitKwargs
from ..whisper.modeling_whisper import WhisperAttention
from .configuration_mimo_v2_vl import MiMoV2VLConfig, MiMoV2VLTextConfig


logger = logging.get_logger(__name__)


@no_inherit_decorator
class MiMoV2VLRMSNorm(LlamaRMSNorm):
    pass


class MiMoV2VLRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float):
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = self.compute_inverse_frequencies().to(torch.empty(0).device)
        self.inv_freq = nn.Buffer(inv_freq, persistent=False)

    def compute_inverse_frequencies(self):
        # Use the same CPU arithmetic during construction and meta-device reload. Small
        # CPU/CUDA pow differences otherwise change phases at long positions and can change MoE routing.
        with torch.autocast(device_type="cpu", enabled=False):
            return 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device="cpu") / self.dim))

    @torch.no_grad()
    def forward(self, position_ids):
        with torch.autocast(device_type=position_ids.device.type, enabled=False):
            frequencies = position_ids.float().unsqueeze(-1) * self.inv_freq.to(position_ids.device).float()
            embeddings = torch.cat((frequencies, frequencies), dim=-1)
        return embeddings.cos(), embeddings.sin()


def apply_rotary_pos_emb(query, key, cos, sin):
    rotary_dim = cos.shape[-1]
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    query_rotary = query[..., :rotary_dim].float()
    key_rotary = key[..., :rotary_dim].float()
    query_rotary = (query_rotary * cos + rotate_half(query_rotary) * sin).to(query.dtype)
    key_rotary = (key_rotary * cos + rotate_half(key_rotary) * sin).to(key.dtype)
    return torch.cat((query_rotary, query[..., rotary_dim:]), -1), torch.cat((key_rotary, key[..., rotary_dim:]), -1)


def _attention_forward(query, key, value, attention_mask=None, sinks=None, dropout=0.0, implementation="sdpa"):
    """Grouped-query attention with an optional zero-valued sink in the softmax denominator."""
    if attention_mask is not None and attention_mask.dtype == torch.bool:
        attention_mask = torch.zeros_like(attention_mask, dtype=query.dtype).masked_fill(~attention_mask, -torch.inf)
    groups = query.shape[1] // key.shape[1]
    if groups != 1:
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
    if sinks is not None:
        batch_size, num_heads, query_length, _ = query.shape
        if attention_mask is None:
            attention_mask = query.new_zeros(batch_size, 1, query_length, key.shape[-2])
        attention_mask = attention_mask.expand(batch_size, num_heads, query_length, key.shape[-2])
        sink_bias = sinks.to(query.dtype).view(1, num_heads, 1, 1).expand(batch_size, -1, query_length, -1)
        attention_mask = torch.cat((attention_mask, sink_bias), dim=-1)
        key = F.pad(key, (0, 0, 0, 1))
        value = F.pad(value, (0, 0, 0, 1))
    if implementation == "eager":
        weights = query @ key.transpose(-1, -2) / math.sqrt(query.shape[-1])
        if attention_mask is not None:
            weights = weights + attention_mask
        weights = weights.float().softmax(dim=-1).to(query.dtype)
        return F.dropout(weights, p=dropout, training=dropout > 0) @ value
    return F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=dropout)


def attention_forward(
    query,
    key,
    value,
    attention_mask=None,
    sinks=None,
    dropout=0.0,
    implementation="sdpa",
    window_size=None,
    query_chunk_size=512,
):
    """Bound temporary attention tensors and restrict local attention to its actual key window."""
    query_length, key_length = query.shape[-2], key.shape[-2]
    if window_size is None and query_length <= query_chunk_size:
        return _attention_forward(query, key, value, attention_mask, sinks, dropout, implementation)
    outputs = []
    for start in range(0, query_length, query_chunk_size):
        end = min(start + query_chunk_size, query_length)
        key_start, key_end = 0, key_length
        if window_size is not None:
            left, right = window_size
            key_start = max(0, key_length - query_length + start - left) if left >= 0 else 0
            key_end = min(key_length, key_length - query_length + end + right) if right >= 0 else key_length
        mask = attention_mask
        if mask is not None:
            if mask.shape[-2] != 1:
                mask = mask[..., start:end, :]
            mask = mask[..., key_start:key_end]
        if window_size is not None:
            queries = torch.arange(start, end, device=query.device) + key_length - query_length
            keys = torch.arange(key_start, key_end, device=query.device)
            allowed = torch.ones((end - start, key_end - key_start), dtype=torch.bool, device=query.device)
            if left >= 0:
                allowed = allowed & (keys[None] >= queries[:, None] - left)
            if right >= 0:
                allowed = allowed & (keys[None] <= queries[:, None] + right)
            if mask is None:
                mask = allowed
            elif mask.dtype == torch.bool:
                mask = mask & allowed
            else:
                mask = mask.masked_fill(~allowed, -torch.inf)
        outputs.append(
            _attention_forward(
                query[..., start:end, :],
                key[..., key_start:key_end, :],
                value[..., key_start:key_end, :],
                mask,
                sinks,
                dropout,
                implementation,
            )
        )
    return torch.cat(outputs, dim=-2)


class MiMoV2VLMLP(LlamaMLP):
    def __init__(self, hidden_size, intermediate_size, hidden_act="silu", bias=False):
        nn.Module.__init__(self)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.act_fn = ACT2FN[hidden_act]


class MiMoV2VLTopkRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.hidden_size))
        self.e_score_correction_bias = nn.Parameter(torch.zeros(config.n_routed_experts))

    def forward(self, hidden_states):
        router_dtype = getattr(torch, self.config.moe_router_dtype)
        if hidden_states.is_cuda and router_dtype == torch.bfloat16:
            router_input, router_weight = hidden_states.to(router_dtype), self.weight.to(router_dtype)
            if is_torch_greater_or_equal("2.9"):
                logits = torch.mm(router_input, router_weight.T, out_dtype=torch.float32)
            else:
                # Preserve BF16 operands with FP32 accumulation on releases without out_dtype.
                logits = F.linear(router_input.float(), router_weight.float())
        else:
            logits = F.linear(hidden_states.float(), self.weight.float())
        scores = logits.sigmoid()
        selection_scores = scores + self.e_score_correction_bias.float()
        if self.config.n_group > 1:
            grouped_scores = selection_scores.reshape(
                -1, self.config.n_group, self.config.n_routed_experts // self.config.n_group
            )
            group_scores = grouped_scores.topk(min(2, grouped_scores.shape[-1]), dim=-1).values.sum(-1)
            group_indices = group_scores.topk(self.config.topk_group, dim=-1).indices
            group_mask = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(1, group_indices, True)
            selection_scores = selection_scores.masked_fill(
                ~group_mask.unsqueeze(-1).expand_as(grouped_scores).flatten(1), -torch.inf
            )
        indices = selection_scores.topk(self.config.num_experts_per_tok, dim=-1).indices
        weights = scores.gather(1, indices)
        if self.config.norm_topk_prob and self.config.num_experts_per_tok > 1:
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-20)
        return indices, weights * self.config.routed_scaling_factor


class MiMoV2VLMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = MiMoV2VLTopkRouter(config)
        self.experts = nn.ModuleList(
            [
                MiMoV2VLMLP(config.hidden_size, config.moe_intermediate_size, config.hidden_act)
                for _ in range(config.n_routed_experts)
            ]
        )

    def forward(self, hidden_states):
        shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, shape[-1])
        indices, weights = self.gate(hidden_states)
        output = torch.zeros_like(hidden_states, dtype=torch.float32)
        # Only visit selected experts; decoding usually activates a small fraction of the bank.
        # Sparse eager dispatch visits only active experts; the model does not advertise full-graph compilation.
        # trf-ignore: TRF056
        for expert_index in indices.unique().tolist():
            token_indices, topk_indices = torch.where(indices == expert_index)
            expert_output = self.experts[expert_index](hidden_states[token_indices])
            output.index_add_(0, token_indices, expert_output.float() * weights[token_indices, topk_indices, None])
        return output.to(hidden_states.dtype).reshape(shape)


class MiMoV2VLTextAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.swa_num_attention_heads if is_sliding else config.num_attention_heads
        self.num_key_value_heads = config.swa_num_key_value_heads if is_sliding else config.num_key_value_heads
        self.head_dim = config.swa_head_dim if is_sliding else config.head_dim
        self.v_head_dim = config.swa_v_head_dim if is_sliding else config.v_head_dim
        self.window_size = (config.sliding_window - 1, 0) if is_sliding else None
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.v_head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, config.hidden_size, bias=False)
        use_sink = config.add_swa_attention_sink_bias if is_sliding else config.add_full_attention_sink_bias
        self.attention_sink_bias = nn.Parameter(torch.zeros(self.num_heads)) if use_sink else None

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None):
        batch_size, sequence_length, _ = hidden_states.shape
        query = (
            self.q_proj(hidden_states).view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        )
        key = (
            self.k_proj(hidden_states)
            .view(batch_size, sequence_length, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        value = (
            self.v_proj(hidden_states)
            .view(batch_size, sequence_length, self.num_key_value_heads, self.v_head_dim)
            .transpose(1, 2)
        )
        query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
        if self.config.attention_value_scale is not None:
            value = value * self.config.attention_value_scale
        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)
        if attention_mask is None and sequence_length > 1:
            query_positions = torch.arange(sequence_length, device=query.device) + key.shape[-2] - sequence_length
            key_positions = torch.arange(key.shape[-2], device=query.device)
            attention_mask = query.new_zeros(sequence_length, key.shape[-2]).masked_fill(
                key_positions[None, :] > query_positions[:, None], -torch.inf
            )[None, None]
        output = attention_forward(
            query,
            key,
            value,
            attention_mask,
            self.attention_sink_bias,
            dropout=self.config.attention_dropout if self.training else 0.0,
            implementation=self.config._attn_implementation,
            window_size=self.window_size,
        )
        return self.o_proj(output.transpose(1, 2).reshape(batch_size, sequence_length, -1))


class MiMoV2VLDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        self.self_attn = MiMoV2VLTextAttention(config, layer_idx)
        self.mlp = (
            MiMoV2VLMoE(config)
            if config.moe_layer_freq[layer_idx]
            else MiMoV2VLMLP(config.hidden_size, config.intermediate_size, config.hidden_act)
        )
        self.input_layernorm = MiMoV2VLRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = MiMoV2VLRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None):
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings, attention_mask, past_key_values
        )
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


@auto_docstring
class MiMoV2VLPreTrainedModel(PreTrainedModel):
    config_class = MiMoV2VLConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _no_split_modules = [
        "MiMoV2VLDecoderLayer",
        "MiMoV2VLVisionBlock",
        "MiMoV2VLAudioLocalLayer",
        "MiMoV2VLAudioTokenizerLayer",
    ]
    _skip_keys_device_placement = ["past_key_values"]
    _keep_in_fp32_modules_strict = ["quantizer", "e_score_correction_bias"]
    _can_record_outputs = {"hidden_states": MiMoV2VLDecoderLayer}

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = load_original_checkpoint_if_needed(cls, pretrained_model_name_or_path, **kwargs)
        if model is not None:
            return model
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    def _init_weights(self, module):
        super()._init_weights(module)
        std = getattr(self.config.get_text_config(), "initializer_range", 0.02)
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv3d)):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, (MiMoV2VLRMSNorm, nn.LayerNorm)):
            init.ones_(module.weight)
            if getattr(module, "bias", None) is not None:
                init.zeros_(module.bias)
        elif isinstance(module, MiMoV2VLTopkRouter):
            init.normal_(module.weight, mean=0.0, std=std)
            init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, MiMoV2VLRotaryEmbedding):
            init.copy_(module.inv_freq, module.compute_inverse_frequencies())
        elif isinstance(module, MiMoV2VLCodebook):
            init.ones_(module.inited)
            init.zeros_(module.cluster_size)
            init.zeros_(module.embed)
            init.zeros_(module.embed_avg)
        for name in ("sinks", "attention_sink_bias"):
            if getattr(module, name, None) is not None:
                init.zeros_(getattr(module, name))


@auto_docstring
class MiMoV2VLTextModel(MiMoV2VLPreTrainedModel):
    config_class = MiMoV2VLTextConfig
    _keep_in_fp32_modules_strict = ["e_score_correction_bias"]

    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([MiMoV2VLDecoderLayer(config, index) for index in range(config.num_hidden_layers)])
        self.norm = MiMoV2VLRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = MiMoV2VLRotaryEmbedding(
            int(config.head_dim * config.partial_rotary_factor), config.rope_theta
        )
        self.swa_rotary_emb = MiMoV2VLRotaryEmbedding(
            int(config.swa_head_dim * config.partial_rotary_factor), config.swa_rope_theta
        )
        self.gradient_checkpointing = False
        self.post_init()

    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        **kwargs,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if self.gradient_checkpointing and self.training:
            use_cache = False
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if position_ids is None:
            if attention_mask is not None and not isinstance(attention_mask, dict) and attention_mask.ndim == 2:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids = position_ids.masked_fill(attention_mask == 0, 0)[:, -inputs_embeds.shape[1] :]
            else:
                past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
                position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)[None] + past_length
        masks = attention_mask
        if not isinstance(masks, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            masks = {"full_attention": create_causal_mask(**mask_kwargs)}
            if "sliding_attention" in self.config.layer_types:
                masks["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        position_embeddings = self.rotary_emb(position_ids)
        sliding_position_embeddings = self.swa_rotary_emb(position_ids)
        hidden_states = inputs_embeds
        for layer in self.layers:
            rotary = sliding_position_embeddings if layer.layer_type == "sliding_attention" else position_embeddings
            hidden_states = layer(hidden_states, rotary, masks[layer.layer_type], past_key_values)
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class MiMoV2VLVisionPatchEmbed(Qwen2_5_VisionPatchEmbed):
    def __init__(self, config):
        patch_size = config.patch_size
        temporal_patch_size = config.temporal_patch_size
        in_channels = config.in_channels
        embed_dim = config.hidden_size
        super().__init__(patch_size, temporal_patch_size, in_channels, embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.contiguous()
        return super().forward(hidden_states)


class MiMoV2VLVisionAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.full_attention = layer_idx in config.fullatt_block_indexes
        self.qkv = nn.Linear(
            config.hidden_size, (config.num_heads + 2 * config.num_key_value_heads) * config.qk_channels
        )
        self.proj = nn.Linear(config.num_heads * config.qk_channels, config.hidden_size)
        self.sinks = (
            nn.Parameter(torch.zeros(config.num_heads)) if config.use_sink and not self.full_attention else None
        )

    def forward(self, hidden_states, position_embeddings, sequence_lengths):
        config = self.config
        q_size, kv_size = config.num_heads * config.qk_channels, config.num_key_value_heads * config.qk_channels
        query, key, value = self.qkv(hidden_states).split((q_size, kv_size, kv_size), dim=-1)
        query = query.view(1, -1, config.num_heads, config.qk_channels).transpose(1, 2)
        key = key.view(1, -1, config.num_key_value_heads, config.qk_channels).transpose(1, 2)
        value = value.view(1, -1, config.num_key_value_heads, config.qk_channels).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
        outputs = []
        for query_chunk, key_chunk, value_chunk in zip(
            query.split(sequence_lengths, 2), key.split(sequence_lengths, 2), value.split(sequence_lengths, 2)
        ):
            window_size = (
                (config.visual_token_window_size,) * 2
                if not self.full_attention and config.visual_token_window_size > 0
                else None
            )
            outputs.append(
                attention_forward(query_chunk, key_chunk, value_chunk, sinks=self.sinks, window_size=window_size)
            )
        output = torch.cat(outputs, dim=2).transpose(1, 2).reshape(hidden_states.shape[0], -1)
        return self.proj(output)


class MiMoV2VLVisionBlock(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.norm1 = MiMoV2VLRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.norm2 = MiMoV2VLRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = MiMoV2VLVisionAttention(config, layer_idx)
        self.mlp = MiMoV2VLMLP(config.hidden_size, config.intermediate_size, config.hidden_act, bias=True)

    def forward(self, hidden_states, position_embeddings, sequence_lengths):
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), position_embeddings, sequence_lengths)
        return hidden_states + self.mlp(self.norm2(hidden_states))


class MiMoV2VLVisionPatchMerger(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size * config.spatial_merge_size**2
        self.ln_q = nn.LayerNorm(config.hidden_size, eps=1e-6, bias=False)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=config.merger_bias)
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size, bias=config.merger_bias)

    def forward(self, hidden_states):
        hidden_states = self.ln_q(hidden_states).reshape(-1, self.hidden_size)
        return self.linear_fc2(F.gelu(self.linear_fc1(hidden_states)))


class MiMoV2VLVisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embed = MiMoV2VLVisionPatchEmbed(config)
        self.rotary_pos_emb = MiMoV2VLRotaryEmbedding(config.qk_channels // 2, 10000.0)
        self.blocks = nn.ModuleList([MiMoV2VLVisionBlock(config, index) for index in range(config.depth)])
        self.merger = MiMoV2VLVisionPatchMerger(config)

    def _get_positions_and_order(self, grid_thw, device):
        merge_size = self.config.spatial_merge_size
        positions, column_indices, sequence_lengths = [], [], []
        offset = 0
        for time, height, width in grid_thw.tolist():
            if min(time, height, width) <= 0 or height % merge_size or width % merge_size:
                raise ValueError(
                    "Vision grids must be positive and spatial dimensions divisible by spatial_merge_size."
                )
            rows = torch.arange(height, device=device)[:, None].expand(height, width)
            columns = torch.arange(width, device=device)[None].expand(height, width)
            grid_positions = torch.stack((rows, columns), -1).reshape(
                height // merge_size, merge_size, width // merge_size, merge_size, 2
            )
            positions.append(grid_positions.permute(0, 2, 1, 3, 4).reshape(-1, 2).repeat(time, 1))
            num_groups = time * height * width // merge_size**2
            indices = torch.arange(num_groups, device=device).reshape(time, height // merge_size, width // merge_size)
            column_indices.append(indices.transpose(1, 2).reshape(-1) + offset)
            offset += num_groups
            sequence_lengths.extend([height * width] * time)
        return torch.cat(positions), torch.cat(column_indices), sequence_lengths

    def forward(self, pixel_values, grid_thw):
        hidden_states = self.patch_embed(pixel_values)
        device = hidden_states.device
        positions, column_indices, sequence_lengths = self._get_positions_and_order(grid_thw, device)
        if sum(sequence_lengths) != hidden_states.shape[0]:
            raise ValueError("Vision grid sizes do not match the number of pixel patches.")
        # The two spatial axes each use half of the rotary frequencies, in spatial-merge patch order.
        frequencies = positions.float().unsqueeze(-1) * self.rotary_pos_emb.inv_freq.to(device)
        frequencies = frequencies.flatten(1)
        embeddings = torch.cat((frequencies, frequencies), -1)
        merge_unit = self.config.spatial_merge_size**2

        def reorder(values, indices):
            return values.unflatten(0, (-1, merge_unit))[indices.to(values.device)].flatten(0, 1)

        inverse_indices = column_indices.argsort()
        row_positions = (embeddings.cos()[None], embeddings.sin()[None])
        column_embeddings = reorder(embeddings, column_indices)
        column_positions = (column_embeddings.cos()[None], column_embeddings.sin()[None])
        in_column_order = False
        for index, block in enumerate(self.blocks):
            use_columns = self.config.vit_window_attn_types[index] == 1
            if use_columns != in_column_order:
                hidden_states = reorder(hidden_states, column_indices if use_columns else inverse_indices)
            in_column_order = use_columns
            hidden_states = block(hidden_states, column_positions if use_columns else row_positions, sequence_lengths)
        if in_column_order:
            hidden_states = reorder(hidden_states, inverse_indices)
        return self.merger(hidden_states)


class MiMoV2VLAudioLocalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        inner_size = config.input_local_attn_heads * config.input_local_head_dim
        self.q_proj = nn.Linear(config.input_local_dim, inner_size)
        self.k_proj = nn.Linear(config.input_local_dim, inner_size)
        self.v_proj = nn.Linear(config.input_local_dim, inner_size)
        self.o_proj = nn.Linear(inner_size, config.input_local_dim, bias=False)

    def forward(self, hidden_states, position_embeddings):
        batch_size, sequence_length, _ = hidden_states.shape
        shape = (batch_size, sequence_length, self.config.input_local_attn_heads, self.config.input_local_head_dim)
        query, key, value = [
            projection(hidden_states).view(shape).transpose(1, 2)
            for projection in (self.q_proj, self.k_proj, self.v_proj)
        ]
        query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=not self.config.input_full_attention,
            dropout_p=self.config.input_local_hidden_dropout if self.training else 0.0,
        )
        return self.o_proj(output.transpose(1, 2).reshape(batch_size, sequence_length, -1))


class MiMoV2VLAudioLocalLayer(GradientCheckpointingLayer):
    def __init__(self, config):
        super().__init__()
        self.self_attn = MiMoV2VLAudioLocalAttention(config)
        self.input_layernorm = MiMoV2VLRMSNorm(config.input_local_dim, config.rms_norm_eps)
        self.post_attention_layernorm = MiMoV2VLRMSNorm(config.input_local_dim, config.rms_norm_eps)
        self.mlp = MiMoV2VLMLP(config.input_local_dim, config.input_local_intermediate_size)

    def forward(self, hidden_states, position_embeddings):
        hidden_states = hidden_states + self.self_attn(self.input_layernorm(hidden_states), position_embeddings)
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class MiMoV2VLAudioLocalTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([MiMoV2VLAudioLocalLayer(config) for _ in range(config.input_local_layers)])
        self.norm = (
            MiMoV2VLRMSNorm(config.input_local_dim, config.rms_norm_eps) if config.add_post_norm else nn.Identity()
        )
        self.rotary_emb = MiMoV2VLRotaryEmbedding(config.input_local_head_dim, config.rope_theta)

    def forward(self, hidden_states):
        positions = torch.arange(hidden_states.shape[1], device=hidden_states.device)[None]
        position_embeddings = self.rotary_emb(positions)
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings)
        return self.norm(hidden_states)


class MiMoV2VLAudioProjection(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear_fc1 = nn.Linear(input_size, input_size * 4, bias=False)
        self.linear_fc2 = nn.Linear(input_size * 4, output_size, bias=False)

    def forward(self, hidden_states):
        return self.linear_fc2(F.gelu(self.linear_fc1(hidden_states)))


class MiMoV2VLAudioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_local_transformer = MiMoV2VLAudioLocalTransformer(config)
        input_size = config.input_local_dim * config.group_size
        self.projection = (
            nn.Linear(input_size, config.out_hidden_size, bias=False)
            if config.projection_layers == 1
            else MiMoV2VLAudioProjection(input_size, config.out_hidden_size)
        )

    def forward(self, speech_embeddings):
        hidden_states = self.input_local_transformer(speech_embeddings)
        return self.projection(hidden_states.flatten(1))


class MiMoV2VLCodebook(EncodecEuclideanCodebook):
    def __init__(self, dim, size):
        nn.Module.__init__(self)
        self.inited = nn.Buffer(torch.ones(1))
        self.cluster_size = nn.Buffer(torch.zeros(size))
        self.embed = nn.Buffer(torch.zeros(size, dim))
        self.embed_avg = nn.Buffer(torch.zeros(size, dim))

    def quantize(self, hidden_states):
        codebook = self.embed.float()
        distances = (
            hidden_states.square().sum(-1, keepdim=True) - 2 * hidden_states @ codebook.T + codebook.square().sum(-1)
        )
        return distances.argmin(-1)


class MiMoV2VLVectorQuantization(nn.Module):
    def __init__(self, dim, size):
        super().__init__()
        self._codebook = MiMoV2VLCodebook(dim, size)


class MiMoV2VLResidualVectorQuantization(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MiMoV2VLVectorQuantization(config.d_model, size)
                for size in config.codebook_size[: config.num_quantizers]
            ]
        )

    def forward(self, hidden_states):
        residual = hidden_states.float()
        codes = []
        for layer in self.layers:
            indices = layer._codebook.encode(residual)
            residual = residual - F.embedding(indices, layer._codebook.embed.float())
            codes.append(indices)
        return torch.stack(codes, dim=-1)


class MiMoV2VLAudioTokenizerAttention(WhisperAttention):
    def __init__(self, config, layer_idx):
        embed_dim = config.d_model
        num_heads = config.encoder_attention_heads
        dropout = 0.0
        is_decoder = False
        bias = True
        is_causal = config.encoder_causal
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            is_decoder=is_decoder,
            bias=bias,
            is_causal=is_causal,
            layer_idx=layer_idx,
            config=config,
        )
        use_window = not config.hybrid_attention or layer_idx % config.swa_per_block < config.swa_per_block - 1
        self.window_size = config.encoder_attn_window_size if use_window else (-1, -1)

    def forward(self, hidden_states, lengths, position_embeddings) -> torch.Tensor:
        """Apply RoPE and MiMo's per-layer causal/window mask to the audio tokenizer states."""
        batch_size, sequence_length, _ = hidden_states.shape
        shape = (
            batch_size,
            sequence_length,
            self.config.encoder_attention_heads,
            self.config.d_model // self.config.encoder_attention_heads,
        )
        query, key, value = [
            projection(hidden_states).view(shape).transpose(1, 2)
            for projection in (self.q_proj, self.k_proj, self.v_proj)
        ]
        query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
        positions = torch.arange(sequence_length, device=hidden_states.device)
        allowed = positions[None, None, :] < lengths[:, None, None]
        if self.config.encoder_causal:
            allowed = allowed & (positions[None, :] <= positions[:, None])
        if self.window_size[0] >= 0:
            allowed = allowed & (positions[None, :] >= positions[:, None] - self.window_size[0])
        if self.window_size[1] >= 0:
            allowed = allowed & (positions[None, :] <= positions[:, None] + self.window_size[1])
        mask = hidden_states.new_zeros(batch_size, sequence_length, sequence_length).masked_fill(~allowed, -torch.inf)
        output = F.scaled_dot_product_attention(query, key, value, attn_mask=mask[:, None])
        return self.out_proj(output.transpose(1, 2).reshape(batch_size, sequence_length, -1))


class MiMoV2VLAudioTokenizerLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = MiMoV2VLAudioTokenizerAttention(config, layer_idx)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.final_layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model)
        self.activation_fn = ACT2FN[config.activation_function]

    def forward(self, hidden_states, lengths, position_embeddings):
        hidden_states = hidden_states + self.self_attn(
            self.self_attn_layer_norm(hidden_states), lengths, position_embeddings
        )
        return hidden_states + self.fc2(self.activation_fn(self.fc1(self.final_layer_norm(hidden_states))))


class MiMoV2VLAudioTokenizerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv1d(config.n_mels, config.d_model, kernel_size=config.kernel_size, padding=1)
        self.conv2 = nn.Conv1d(
            config.d_model, config.d_model, kernel_size=config.kernel_size, stride=config.stride_size, padding=1
        )
        self.position_embedding = MiMoV2VLRotaryEmbedding(
            config.d_model // config.encoder_attention_heads, config.rope_theta
        )
        self.layers = nn.ModuleList(
            [MiMoV2VLAudioTokenizerLayer(config, index) for index in range(config.encoder_layers)]
        )
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.down_sample_layer = None
        if config.avg_pooler != 1:
            self.down_sample_layer = nn.Conv1d(
                config.d_model, config.d_model, config.avg_pooler, config.avg_pooler, bias=False
            )
            self.down_sample_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.quantizer = MiMoV2VLResidualVectorQuantization(config)

    @torch.no_grad()
    def forward(self, features, lengths):
        config = self.config
        features = features.to(dtype=self.conv1.weight.dtype)
        hidden_states = F.gelu(self.conv1(features.transpose(1, 2)))
        hidden_states = F.gelu(self.conv2(hidden_states)).transpose(1, 2)
        lengths = lengths.to(hidden_states.device)
        lengths = (lengths + 3 - config.kernel_size + 2 - config.kernel_size) // config.stride_size + 1
        positions = torch.arange(hidden_states.shape[1], device=hidden_states.device)[None]
        position_embeddings = self.position_embedding(positions)
        skip_hidden_states = None
        for index, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, lengths, position_embeddings)
            if config.encoder_skip_layer_id == index + 1:
                skip_hidden_states = hidden_states
        if skip_hidden_states is not None:
            hidden_states = hidden_states + skip_hidden_states.to(hidden_states.device)
        hidden_states = self.layer_norm(hidden_states)
        lengths = lengths.to(hidden_states.device)
        if self.down_sample_layer is not None:
            # The checkpoint's packed-to-padded unpacking repeats the last valid hidden state within a batch.
            valid = torch.arange(hidden_states.shape[1], device=lengths.device)[None] < lengths[:, None]
            last_hidden = hidden_states[torch.arange(len(lengths), device=lengths.device), lengths - 1, None]
            hidden_states = torch.where(valid[..., None], hidden_states, last_hidden)
            if pad_length := -hidden_states.shape[1] % config.avg_pooler:
                hidden_states = F.pad(hidden_states, (0, 0, 0, pad_length))
            hidden_states = F.gelu(self.down_sample_layer(hidden_states.transpose(1, 2))).transpose(1, 2)
            hidden_states = self.down_sample_norm(hidden_states)
            lengths = (lengths + config.avg_pooler - 1) // config.avg_pooler
        valid = torch.arange(hidden_states.shape[1], device=lengths.device)[None] < lengths[:, None]
        codes = self.quantizer(hidden_states[valid])
        # Restore the ragged list consumed by the audio segment alignment code.
        # trf-ignore: TRF056
        return list(codes.split(lengths.tolist(), dim=0))


class MiMoV2VLAudioTokenizerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = MiMoV2VLAudioTokenizerEncoder(config)

    @torch.no_grad()
    def forward(self, input_features, feature_attention_mask, segment_size):
        if input_features.ndim != 3 or min(input_features.shape[:2]) == 0:
            raise ValueError("input_features must have nonempty (batch, mel_time, mel_bins) dimensions.")
        if input_features.shape[-1] != self.encoder.config.n_mels:
            raise ValueError("input_features mel dimension must match audio_tokenizer_config.n_mels.")
        device = input_features.device
        if feature_attention_mask is None:
            lengths = [input_features.shape[1]] * len(input_features)
        else:
            if feature_attention_mask.shape != input_features.shape[:2]:
                raise ValueError(
                    "feature_attention_mask must match the batch and mel_time dimensions of input_features."
                )
            if feature_attention_mask.is_complex() or not torch.all(
                (feature_attention_mask == 0) | (feature_attention_mask == 1)
            ):
                raise ValueError("feature_attention_mask must contain only zero and one values.")
            # Mel segments have variable lengths and are grouped outside compiled decoder execution.
            # trf-ignore: TRF056
            lengths = feature_attention_mask.sum(-1, dtype=torch.long).tolist()
        sequences = []
        segment_counts = []
        for index, (features, length) in enumerate(zip(input_features, lengths)):
            if length <= 0:
                raise ValueError("Each audio input must contain at least one valid mel frame.")
            if feature_attention_mask is not None:
                features = features[feature_attention_mask[index].to(device=features.device, dtype=torch.bool)]
            segments = list(features[:length].split(segment_size, dim=0))
            sequences.extend(segments)
            segment_counts.append(len(segments))
        segment_lengths = torch.tensor([len(segment) for segment in sequences], device=device)
        padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True).to(device)
        code_segments = self.encoder(padded, segment_lengths)
        outputs = []
        offset = 0
        for count in segment_counts:
            outputs.append(torch.cat(code_segments[offset : offset + count]))
            offset += count
        return outputs


def as_integer_audio_metadata(values, name: str, ndim: int) -> torch.Tensor:
    """Validate lengths and offsets without silently rounding floating-point metadata."""
    try:
        values = torch.as_tensor(values)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"{name} must be a {ndim}-dimensional integer array.") from error
    if (
        values.ndim != ndim
        or values.is_floating_point()
        or values.is_complex()
        or values.is_quantized
        or values.dtype == torch.bool
    ):
        raise ValueError(f"{name} must be a {ndim}-dimensional integer array.")
    return values


@auto_docstring
class MiMoV2VLModel(MiMoV2VLPreTrainedModel):
    """MiMo V2 multimodal encoders, embedding fusion, and hybrid language decoder."""

    def __init__(self, config):
        super().__init__(config)
        self.language_model = MiMoV2VLTextModel(config.text_config)
        self.visual = MiMoV2VLVisionModel(config.vision_config)
        audio_config = config.audio_config
        self.speech_embeddings = nn.ModuleList(
            [
                nn.Embedding(vocab_size, audio_config.input_local_dim, padding_idx=padding_idx)
                for vocab_size, padding_idx in zip(audio_config.speech_vocab_size, audio_config.speech_zeroemb_idx)
            ]
        )
        self.audio_encoder = MiMoV2VLAudioEncoder(audio_config)
        self.audio_tokenizer = MiMoV2VLAudioTokenizerModel(config.audio_tokenizer_config)
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.embed_tokens

    def set_input_embeddings(self, value):
        self.language_model.embed_tokens = value

    @auto_docstring
    def get_image_features(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        return self.visual(pixel_values, image_grid_thw)

    @auto_docstring
    def get_video_features(self, pixel_values_videos: torch.Tensor, video_grid_thw: torch.Tensor) -> torch.Tensor:
        return self.visual(pixel_values_videos, video_grid_thw)

    @auto_docstring
    def get_audio_features(
        self,
        input_audio_codes=None,
        audio_feature_lens=None,
        audio_segment_offsets=None,
        input_features=None,
        feature_attention_mask=None,
        audio_codes=None,
    ):
        """
        input_audio_codes (`torch.LongTensor`, *optional*):
            Grouped codec IDs of shape `(batch, grouped_time, group_size, audio_channels)`.
        audio_feature_lens (`torch.LongTensor`, *optional*):
            Valid grouped-token length of each audio sequence before batch padding.
        audio_segment_offsets (`torch.LongTensor`, *optional*):
            `(audio_index, start, length)` rows specifying audio segments in prompt order.
        feature_attention_mask (`torch.Tensor`, *optional*):
            Mask of valid mel frames, with shape `(batch, mel_time)`.
        audio_codes (`torch.LongTensor`, *optional*):
            Legacy unpadded codec IDs of shape `(time, audio_channels)`, already ordered to match the prompt.
        """
        config = self.config.audio_config
        if sum(value is not None for value in (input_audio_codes, input_features, audio_codes)) != 1:
            raise ValueError("Specify exactly one of input_features, input_audio_codes, or audio_codes.")
        if input_features is not None:
            sequences = self.audio_tokenizer(input_features, feature_attention_mask, config.audio_segment_size)
            grouped = []
            for codes in sequences:
                if pad_length := -len(codes) % config.group_size:
                    codes = torch.cat((codes, codes[-1:].expand(pad_length, -1)))
                grouped.append(codes[:, : config.audio_channels].reshape(-1, config.group_size, config.audio_channels))
        elif audio_codes is not None:
            if (
                audio_codes.ndim != 2
                or audio_codes.shape[1] != config.audio_channels
                or len(audio_codes) % config.group_size
            ):
                raise ValueError("audio_codes must have shape (grouped_time * group_size, audio_channels).")
            grouped = [audio_codes.reshape(-1, config.group_size, config.audio_channels)]
        else:
            if input_audio_codes.ndim != 4 or tuple(input_audio_codes.shape[-2:]) != (
                config.group_size,
                config.audio_channels,
            ):
                raise ValueError(
                    "input_audio_codes must have shape (batch, grouped_time, group_size, audio_channels)."
                )
            grouped = list(input_audio_codes)
        lengths = [len(sequence) for sequence in grouped]
        if audio_feature_lens is not None and audio_codes is None:
            supplied_lengths = as_integer_audio_metadata(audio_feature_lens, "audio_feature_lens", ndim=1).tolist()
            if len(supplied_lengths) != len(grouped) or any(
                length < 0 or length > maximum for length, maximum in zip(supplied_lengths, lengths)
            ):
                raise ValueError("audio_feature_lens does not match the encoded audio sequences.")
            if input_features is not None and supplied_lengths != lengths:
                raise ValueError("Audio tokenizer output lengths differ from the processor's audio_feature_lens.")
            lengths = supplied_lengths
        if audio_segment_offsets is None or audio_codes is not None:
            audio_segment_offsets = [(index, 0, length) for index, length in enumerate(lengths)]
        audio_segment_offsets = as_integer_audio_metadata(audio_segment_offsets, "audio_segment_offsets", ndim=2)
        if audio_segment_offsets.shape[1] != 3:
            raise ValueError("audio_segment_offsets must contain (audio_index, start, length) rows.")
        segments = []
        for index, start, length in audio_segment_offsets.tolist():
            if not 0 <= index < len(grouped) or start < 0 or length < 0 or start + length > lengths[index]:
                raise ValueError("Audio segment offsets extend outside the valid codec sequence.")
            segments.append(grouped[index][start : start + length])
        if not segments:
            raise ValueError("At least one audio segment is required.")
        codes = torch.cat(segments)
        if codes.is_floating_point() or codes.is_complex() or codes.is_quantized or codes.dtype == torch.bool:
            raise ValueError("Audio codec IDs must be integers.")
        speech_embeddings = self.speech_embeddings[0](codes[..., 0].long())
        for channel, embedding in enumerate(self.speech_embeddings[1:], start=1):
            speech_embeddings = speech_embeddings + embedding(codes[..., channel].long()).to(speech_embeddings.device)
        return self.audio_encoder(speech_embeddings)

    def get_multimodal_embeddings(
        self,
        input_ids,
        inputs_embeds=None,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        video_pixel_values=None,
        input_audio_codes=None,
        audio_codes=None,
        input_features=None,
        audio_feature_lens=None,
        feature_attention_mask=None,
        audio_segment_offsets=None,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        if pixel_values_videos is not None and video_pixel_values is not None:
            raise ValueError("Specify only one of pixel_values_videos or video_pixel_values.")
        pixel_values_videos = pixel_values_videos if pixel_values_videos is not None else video_pixel_values

        def merge_embeddings(token_id, features, modality):
            nonlocal inputs_embeds
            if input_ids is None:
                raise ValueError("input_ids are required to locate multimodal placeholders.")
            mask = input_ids.to(inputs_embeds.device) == token_id
            if int(mask.sum()) != features.shape[0]:
                raise ValueError(
                    f"{modality} features and placeholder tokens do not match: {features.shape[0]} features, {int(mask.sum())} tokens."
                )
            features = features.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(mask.unsqueeze(-1), features)

        if pixel_values is not None:
            if image_grid_thw is None:
                raise ValueError("image_grid_thw is required with pixel_values.")
            merge_embeddings(
                self.config.image_token_id, self.get_image_features(pixel_values, image_grid_thw), "Image"
            )
        if pixel_values_videos is not None:
            if video_grid_thw is None:
                raise ValueError("video_grid_thw is required with pixel_values_videos.")
            merge_embeddings(
                self.config.video_token_id, self.get_video_features(pixel_values_videos, video_grid_thw), "Video"
            )
        if any(value is not None for value in (input_audio_codes, input_features, audio_codes)):
            audio_features = self.get_audio_features(
                input_audio_codes=input_audio_codes,
                audio_feature_lens=audio_feature_lens,
                audio_segment_offsets=audio_segment_offsets,
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
                audio_codes=audio_codes,
            )
            merge_embeddings(self.config.audio_token_id, audio_features, "Audio")
        return inputs_embeds

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        input_audio_codes=None,
        input_features=None,
        feature_attention_mask=None,
        audio_feature_lens=None,
        audio_segment_offsets=None,
        video_pixel_values=None,
        audio_codes=None,
        **kwargs,
    ):
        """
        input_audio_codes (`torch.LongTensor`, *optional*):
            Grouped codec IDs of shape `(batch, grouped_time, group_size, audio_channels)`.
        feature_attention_mask (`torch.Tensor`, *optional*):
            Mask identifying valid mel frames in `input_features`.
        audio_feature_lens (`torch.LongTensor`, *optional*):
            Valid grouped-token length of each audio sequence.
        audio_segment_offsets (`torch.LongTensor`, *optional*):
            `(audio_index, start, length)` rows aligning audio features with prompt spans.
        video_pixel_values (`torch.Tensor`, *optional*):
            Legacy alias for `pixel_values_videos`. Specify only one of these inputs.
        audio_codes (`torch.LongTensor`, *optional*):
            Legacy unpadded, prompt-ordered codec IDs of shape `(time, audio_channels)`.
        """
        inputs_embeds = self.get_multimodal_embeddings(
            input_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            video_pixel_values=video_pixel_values,
            input_audio_codes=input_audio_codes,
            audio_codes=audio_codes,
            input_features=input_features,
            audio_feature_lens=audio_feature_lens,
            feature_attention_mask=feature_attention_mask,
            audio_segment_offsets=audio_segment_offsets,
        )
        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
            **kwargs,
        )


@auto_docstring(checkpoint="XiaomiMiMo/MiMo-V2.6-Flash-RL")
class MiMoV2VLForConditionalGeneration(MiMoV2VLPreTrainedModel, GenerationMixin):
    """Generate text from text, images, videos, and audio using the MiMo V2 VL architecture.

    Pass the tensors returned by `MiMoV2Processor` directly to `forward` or `generate`. Both waveform mel features
    and grouped codec IDs are supported. The learned audio tokenizer is part of the native model checkpoint.
    """

    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    _multimodal_input_names = (
        "pixel_values",
        "image_grid_thw",
        "pixel_values_videos",
        "video_grid_thw",
        "video_pixel_values",
        "input_audio_codes",
        "audio_codes",
        "input_features",
        "feature_attention_mask",
        "audio_feature_lens",
        "audio_segment_offsets",
    )

    def __init__(self, config):
        super().__init__(config)
        self.model = MiMoV2VLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    @classmethod
    def from_original_checkpoint(cls, pretrained_model_name_or_path, **kwargs):
        """Load original TP-interleaved FP8/MXFP4 weights, including the separate audio tokenizer, into a native model."""
        from ...modeling_mimo_v2_utils import load_original_checkpoint

        return load_original_checkpoint(pretrained_model_name_or_path, **kwargs)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        logits_to_keep=0,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        input_audio_codes=None,
        input_features=None,
        feature_attention_mask=None,
        audio_feature_lens=None,
        audio_segment_offsets=None,
        video_pixel_values=None,
        audio_codes=None,
        **kwargs,
    ):
        """
        input_audio_codes (`torch.LongTensor`, *optional*):
            Grouped codec IDs of shape `(batch, grouped_time, group_size, audio_channels)`.
        feature_attention_mask (`torch.Tensor`, *optional*):
            Mask identifying valid mel frames in `input_features`.
        audio_feature_lens (`torch.LongTensor`, *optional*):
            Valid grouped-token length of each audio sequence.
        audio_segment_offsets (`torch.LongTensor`, *optional*):
            `(audio_index, start, length)` rows aligning audio features with prompt spans.
        video_pixel_values (`torch.Tensor`, *optional*):
            Legacy alias for `pixel_values_videos`. Specify only one of these inputs.
        audio_codes (`torch.LongTensor`, *optional*):
            Legacy unpadded, prompt-ordered codec IDs of shape `(time, audio_channels)`.
        """
        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            input_audio_codes=input_audio_codes,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            audio_feature_lens=audio_feature_lens,
            audio_segment_offsets=audio_segment_offsets,
            video_pixel_values=video_pixel_values,
            audio_codes=audio_codes,
            return_dict=True,
            **kwargs,
        )
        slices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(outputs.last_hidden_state[:, slices, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, is_first_iteration=False, **kwargs):
        cache = kwargs.get("past_key_values")
        if kwargs.get("use_cache", True) and (cache is None or cache.get_seq_length() == 0):
            is_first_iteration = True
        model_inputs = super().prepare_inputs_for_generation(
            input_ids, is_first_iteration=is_first_iteration, **kwargs
        )
        if (
            not is_first_iteration
            and kwargs.get("use_cache", True)
            and cache is not None
            and cache.get_seq_length() > 0
        ):
            for name in self._multimodal_input_names:
                model_inputs.pop(name, None)
        return model_inputs

    def _expand_inputs_for_generation(self, expand_size=1, is_encoder_decoder=False, input_ids=None, **model_kwargs):
        if expand_size > 1 and any(
            model_kwargs.get(name) is not None
            for name in (
                "pixel_values",
                "pixel_values_videos",
                "video_pixel_values",
                "input_audio_codes",
                "audio_codes",
                "input_features",
            )
        ):
            # Encode once before beam expansion so flattened patches and audio offsets retain sample boundaries.
            multimodal_inputs = {
                name: model_kwargs.pop(name) for name in self._multimodal_input_names if name in model_kwargs
            }
            model_kwargs["inputs_embeds"] = self.model.get_multimodal_embeddings(
                input_ids, inputs_embeds=model_kwargs.get("inputs_embeds"), **multimodal_inputs
            )
        return super()._expand_inputs_for_generation(
            expand_size=expand_size, is_encoder_decoder=is_encoder_decoder, input_ids=input_ids, **model_kwargs
        )


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """Apply MiMo's minimum merge-block scaling before Qwen-style pixel-budget rounding."""
    if min(height, width, factor, min_pixels) <= 0 or max_pixels < min_pixels:
        raise ValueError("Image dimensions and pixel limits must be positive and min_pixels <= max_pixels.")
    if min(height, width) < factor:
        scale = factor / min(height, width)
        height, width = round(height * scale), round(width * scale)
    elif max(height, width) / min(height, width) > 200:
        raise ValueError("Image aspect ratio must not exceed 200.")
    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor
    if resized_height * resized_width > max_pixels:
        scale = math.sqrt(height * width / max_pixels)
        resized_height = math.floor(height / scale / factor) * factor
        resized_width = math.floor(width / scale / factor) * factor
    elif resized_height * resized_width < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        resized_height = math.ceil(height * scale / factor) * factor
        resized_width = math.ceil(width * scale / factor) * factor
    if min(resized_height, resized_width) == 0:
        raise ValueError("Pixel budget is too small for the image aspect ratio.")
    return resized_height, resized_width


def resize_frames(frames: torch.Tensor, size: SizeDict, resample: PILImageResampling, factor: int) -> torch.Tensor:
    """Resize float32 frames without antialiasing, as in the original MiMo processor."""
    if not size.shortest_edge or not size.longest_edge:
        raise ValueError("size must contain shortest_edge and longest_edge pixel budgets.")
    resized_size = smart_resize(
        *frames.shape[-2:], factor=factor, min_pixels=size.shortest_edge, max_pixels=size.longest_edge
    )
    interpolation_modes = {PILImageResampling.BILINEAR: "bilinear", PILImageResampling.BICUBIC: "bicubic"}
    if resample not in interpolation_modes:
        raise ValueError("The MiMo V2 series supports bilinear and bicubic resampling.")
    frames = frames.float()
    if tuple(frames.shape[-2:]) != resized_size:
        frames = F.interpolate(
            frames, size=resized_size, mode=interpolation_modes[resample], align_corners=False, antialias=False
        )
    return frames


def normalize_frames(processor, frames, do_rescale, rescale_factor, do_normalize, image_mean, image_std):
    """Use the common backend operations with explicitly float32 source-pixel statistics."""
    frames = frames.float()
    if do_normalize:
        scale = 1 / rescale_factor if do_rescale else 1.0
        mean = torch.as_tensor(image_mean, dtype=torch.float32, device=frames.device) * scale
        std = torch.as_tensor(image_std, dtype=torch.float32, device=frames.device) * scale
        return processor.normalize(frames, mean, std)
    if do_rescale:
        return processor.rescale(frames, rescale_factor)
    return frames


class MiMoV2ImageProcessorKwargs(Qwen2VLImageProcessorKwargs, total=False):
    """
    min_pixels (`int`, *optional*, defaults to `4 * 28 * 28`):
        Minimum image pixel budget. Prefer `size["shortest_edge"]`.
    max_pixels (`int`, *optional*, defaults to `4096 * 28 * 28`):
        Maximum image pixel budget. Prefer `size["longest_edge"]`.
    patch_size (`int`, *optional*, defaults to 14):
        Spatial patch size of the vision encoder.
    temporal_patch_size (`int`, *optional*, defaults to 2):
        Number of repeated image frames in each temporal patch.
    merge_size (`int`, *optional*, defaults to 2):
        Number of spatial patches merged along each axis.
    """


@auto_docstring(
    custom_intro="Constructs an image processor for the MiMo V2 VL series.",
    checkpoint="XiaomiMiMo/MiMo-V2.6-Flash-RL",
)
class MiMoV2ImageProcessor(Qwen2VLImageProcessor):
    size = {"shortest_edge": 4 * 28 * 28, "longest_edge": 4096 * 28 * 28}
    resample = PILImageResampling.BILINEAR
    valid_kwargs = MiMoV2ImageProcessorKwargs

    def process_image(
        self,
        image: ImageInput,
        do_convert_rgb: bool | None = None,
        input_data_format: str | ChannelDimension | None = None,
        device: str | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if isinstance(image, np.ndarray) and any(stride < 0 for stride in image.strides):
            image = np.ascontiguousarray(image)
        return TorchvisionBackend.process_image(
            self, image, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format, device=device, **kwargs
        )

    def resize(
        self, images: torch.Tensor, size: SizeDict, resample: PILImageResampling, factor: int, **kwargs
    ) -> torch.Tensor:
        return resize_frames(images, size, resample, factor)

    def rescale_and_normalize(self, images, do_rescale, rescale_factor, do_normalize, image_mean, image_std):
        return normalize_frames(self, images, do_rescale, rescale_factor, do_normalize, image_mean, image_std)

    def patchify(self, images, patch_size, merge_size, temporal_patch_size):
        if min(patch_size, temporal_patch_size, merge_size) <= 0:
            raise ValueError("Patch and merge sizes must be positive.")
        if any(length % (patch_size * merge_size) for length in images.shape[-2:]):
            raise ValueError("Frame dimensions must be divisible by their temporal and spatial patch factors.")
        return super().patchify(images, patch_size, merge_size, temporal_patch_size)

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs: dict | None = None) -> int:
        images_kwargs = images_kwargs or {}
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)
        size = dict(images_kwargs.get("size", self.size))
        min_pixels = images_kwargs.get("min_pixels", size["shortest_edge"])
        max_pixels = images_kwargs.get("max_pixels", size["longest_edge"])
        if images_kwargs.get("do_resize", self.do_resize):
            height, width = smart_resize(height, width, patch_size * merge_size, min_pixels, max_pixels)
        return (height // patch_size) * (width // patch_size)


def get_video_frame_size(size: SizeDict, num_frames: int, temporal_factor: int, total_max_pixels: int) -> SizeDict:
    """Share the pre-padding pixel budget between preprocessing and patch-count estimation."""
    if min(num_frames, temporal_factor, total_max_pixels) <= 0:
        raise ValueError("Video frame count, temporal factor and total_max_pixels must be positive.")
    if not size.shortest_edge or not size.longest_edge:
        raise ValueError("size must contain shortest_edge and longest_edge pixel budgets.")
    max_pixels = max(size.shortest_edge, min(size.longest_edge, total_max_pixels * temporal_factor // num_frames))
    return SizeDict(shortest_edge=size.shortest_edge, longest_edge=max_pixels)


class MiMoV2VideoProcessorInitKwargs(Qwen2VLVideoProcessorInitKwargs, total=False):
    """
    min_pixels (`int`, *optional*, defaults to `4 * 28 * 28`):
        Minimum frame pixel budget. Prefer `size["shortest_edge"]`.
    max_pixels (`int`, *optional*, defaults to `4096 * 28 * 28`):
        Maximum frame pixel budget. Prefer `size["longest_edge"]`.
    patch_size (`int`, *optional*, defaults to 14):
        Spatial patch size of the vision encoder.
    temporal_patch_size (`int`, *optional*, defaults to 2):
        Number of consecutive frames in a temporal patch.
    merge_size (`int`, *optional*, defaults to 2):
        Spatial merge factor along each axis.
    min_frames (`int`, *optional*, defaults to 8):
        Minimum frame count when sampling at a target frame rate.
    max_frames (`int`, *optional*, defaults to 256):
        Maximum frame count when sampling at a target frame rate.
    temporal_compression_ratio (`int`, *optional*, defaults to 1):
        Number of temporal grids represented by one timestamp/token group.
    total_max_pixels (`int`, *optional*, defaults to `16384 * 28 * 28`):
        Total pixel budget, measured after temporal compression.
    """

    temporal_compression_ratio: int
    total_max_pixels: int
    cap_pixels_per_frame = AttributeError()
    max_video_tokens = AttributeError()


@auto_docstring(
    custom_intro="Constructs a video processor for the MiMo V2 VL series.",
    checkpoint="XiaomiMiMo/MiMo-V2.6-Flash-RL",
)
class MiMoV2VideoProcessor(Qwen2VLVideoProcessor):
    size = {"shortest_edge": 4 * 28 * 28, "longest_edge": 4096 * 28 * 28}
    default_to_square = False
    resample = PILImageResampling.BILINEAR
    do_sample_frames = True
    temporal_compression_ratio = 1
    total_max_pixels = 16384 * 28 * 28
    fps = 2
    num_frames: int | None = None
    min_frames = 8
    max_frames = 256
    valid_kwargs = MiMoV2VideoProcessorInitKwargs
    cap_pixels_per_frame = AttributeError()
    max_video_tokens = AttributeError()

    def _decode_and_sample_videos(self, videos, video_metadata, do_sample_frames=None, sample_indices_fn=None):
        if video_metadata is not None:
            num_videos = len(make_batched_videos(videos))
            metadata = (
                []
                if isinstance(video_metadata, list) and not video_metadata
                else make_batched_metadata(videos, video_metadata=video_metadata)
            )
            if len(metadata) != num_videos:
                raise ValueError("video_metadata must contain one entry per video.")
        return BaseVideoProcessor._decode_and_sample_videos(
            self, videos, video_metadata, do_sample_frames=do_sample_frames, sample_indices_fn=sample_indices_fn
        )

    def _prepare_input_videos(
        self,
        videos: VideoInput,
        do_convert_rgb: bool | None = None,
        input_data_format: str | ChannelDimension | None = None,
        device: str | None = None,
    ) -> list[torch.Tensor]:
        videos = [
            np.ascontiguousarray(video)
            if isinstance(video, np.ndarray) and any(stride < 0 for stride in video.strides)
            else video
            for video in videos
        ]
        return BaseVideoProcessor._prepare_input_videos(
            self, videos, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format, device=device
        )

    @auto_docstring
    def preprocess(self, videos: VideoInput, **kwargs: Unpack[MiMoV2VideoProcessorInitKwargs]) -> BatchFeature:
        if kwargs.get("num_frames") is not None and "fps" not in kwargs:
            kwargs["fps"] = None
        elif kwargs.get("fps") is not None and "num_frames" not in kwargs:
            kwargs["num_frames"] = None
        elif self.num_frames is not None and "fps" not in kwargs:
            kwargs["fps"] = None
        return BaseVideoProcessor.preprocess(self, videos, **kwargs)

    def sample_frames(
        self,
        metadata: VideoMetadata,
        num_frames: int | None = None,
        fps: float | None = None,
        temporal_patch_size: int | None = None,
        min_frames: int | None = None,
        max_frames: int | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Use rounded, endpoint-exclusive intervals, retaining repeats for short videos."""
        if num_frames is not None and fps is not None:
            raise ValueError("num_frames and fps are mutually exclusive.")
        total_num_frames = metadata.total_num_frames
        if total_num_frames <= 0:
            raise ValueError("Video must contain at least one frame.")
        temporal_patch_size = temporal_patch_size if temporal_patch_size is not None else self.temporal_patch_size
        if temporal_patch_size <= 0:
            raise ValueError("temporal_patch_size must be positive.")
        if num_frames is None and fps is None:
            num_frames = self.num_frames
        if num_frames is not None:
            if not math.isfinite(num_frames) or num_frames <= 0:
                raise ValueError("num_frames must be finite and positive.")
        else:
            fps = self.fps if fps is None else fps
            if any(rate is None or not math.isfinite(rate) or rate <= 0 for rate in (metadata.fps, fps)):
                raise ValueError("A finite positive video_metadata.fps and sampling fps are required.")
            min_frames = self.min_frames if min_frames is None else min_frames
            max_frames = self.max_frames if max_frames is None else max_frames
            if not 1 <= min_frames <= max_frames:
                raise ValueError("Require 1 <= min_frames <= max_frames.")
            num_frames = max(min(math.ceil(total_num_frames / metadata.fps * fps), max_frames), min_frames)
        num_frames = math.ceil(min(num_frames, total_num_frames) / temporal_patch_size) * temporal_patch_size
        indices = np.linspace(0, total_num_frames, num_frames + 1)[:-1].round().astype(np.int64)
        return np.clip(indices, 0, total_num_frames - 1)

    def resize(
        self,
        videos: torch.Tensor,
        size: SizeDict,
        resample: PILImageResampling,
        factor: int,
        temporal_factor: int | None = None,
        total_max_pixels: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        temporal_factor = (
            self.temporal_patch_size * self.temporal_compression_ratio if temporal_factor is None else temporal_factor
        )
        total_max_pixels = self.total_max_pixels if total_max_pixels is None else total_max_pixels
        batch_size, num_frames = videos.shape[:2]
        video_size = get_video_frame_size(size, num_frames, temporal_factor, total_max_pixels)
        frames = resize_frames(videos.flatten(0, 1), video_size, resample, factor)
        return frames.reshape(batch_size, num_frames, *frames.shape[1:])

    def rescale_and_normalize(self, images, do_rescale, rescale_factor, do_normalize, image_mean, image_std):
        return normalize_frames(self, images, do_rescale, rescale_factor, do_normalize, image_mean, image_std)

    def patchify(
        self, videos, patch_size, merge_size, temporal_patch_size, temporal_compression_ratio=1
    ) -> tuple[torch.Tensor, int, int, int]:
        if min(patch_size, temporal_patch_size, merge_size, temporal_compression_ratio) <= 0:
            raise ValueError("Patch, merge and compression sizes must be positive.")
        if any(length % (patch_size * merge_size) for length in videos.shape[-2:]):
            raise ValueError("Frame dimensions must be divisible by their temporal and spatial patch factors.")
        temporal_factor = temporal_patch_size * temporal_compression_ratio
        if pad_length := -videos.shape[1] % temporal_factor:
            videos = torch.cat((videos, videos[:, -1:].expand(-1, pad_length, -1, -1, -1)), dim=1)
        videos = videos.contiguous()
        return super().patchify(videos, patch_size, merge_size, temporal_patch_size)

    def _preprocess(
        self,
        videos: list[torch.Tensor],
        do_resize: bool,
        size: SizeDict,
        resample: PILImageResampling,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float],
        image_std: float | list[float],
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        temporal_compression_ratio: int,
        total_max_pixels: int,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        temporal_factor = temporal_patch_size * temporal_compression_ratio
        if min(temporal_patch_size, temporal_compression_ratio, total_max_pixels) <= 0:
            raise ValueError("Temporal factors and total_max_pixels must be positive.")
        grouped_videos, grouped_indices = group_videos_by_shape(videos)
        processed_videos, processed_grids = {}, {}
        for shape, stacked_videos in grouped_videos.items():
            if stacked_videos.shape[1] == 0:
                raise ValueError("Video must contain at least one frame.")
            if do_resize:
                stacked_videos = self.resize(
                    stacked_videos, size, resample, patch_size * merge_size, temporal_factor, total_max_pixels
                )
            stacked_videos = self.rescale_and_normalize(
                stacked_videos, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            patches, grid_t, grid_h, grid_w = self.patchify(
                stacked_videos, patch_size, merge_size, temporal_patch_size, temporal_compression_ratio
            )
            processed_videos[shape] = patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * len(stacked_videos)
        pixel_values = torch.cat(reorder_videos(processed_videos, grouped_indices), dim=0)
        grids = reorder_videos(processed_grids, grouped_indices)
        return BatchFeature({"pixel_values_videos": pixel_values, "video_grid_thw": grids}, tensor_type=return_tensors)

    def get_num_of_video_patches(
        self, num_frames: int, height: int, width: int, videos_kwargs: dict | None = None
    ) -> int:
        videos_kwargs = videos_kwargs or {}
        patch_size = videos_kwargs.get("patch_size", self.patch_size)
        merge_size = videos_kwargs.get("merge_size", self.merge_size)
        temporal_patch_size = videos_kwargs.get("temporal_patch_size", self.temporal_patch_size)
        compression = videos_kwargs.get("temporal_compression_ratio", self.temporal_compression_ratio)
        total_pixels = videos_kwargs.get("total_max_pixels", self.total_max_pixels)
        if min(num_frames, height, width, patch_size, merge_size, temporal_patch_size, compression, total_pixels) <= 0:
            raise ValueError("Video dimensions, patch sizes, temporal compression and pixel budget must be positive.")
        temporal_factor = temporal_patch_size * compression
        if videos_kwargs.get("do_resize", self.do_resize):
            size = dict(videos_kwargs.get("size", self.size))
            min_pixels = videos_kwargs.get("min_pixels", size["shortest_edge"])
            max_pixels = videos_kwargs.get("max_pixels", size["longest_edge"])
            frame_size = get_video_frame_size(
                SizeDict(shortest_edge=min_pixels, longest_edge=max_pixels), num_frames, temporal_factor, total_pixels
            )
            height, width = smart_resize(
                height, width, patch_size * merge_size, frame_size.shortest_edge, frame_size.longest_edge
            )
        padded_frames = num_frames + -num_frames % temporal_factor
        return (padded_frames // temporal_patch_size) * (height // patch_size) * (width // patch_size)


__all__ = [
    "MiMoV2VLPreTrainedModel",
    "MiMoV2VLTextModel",
    "MiMoV2VLModel",
    "MiMoV2VLForConditionalGeneration",
    "MiMoV2ImageProcessor",
    "MiMoV2VideoProcessor",
]
