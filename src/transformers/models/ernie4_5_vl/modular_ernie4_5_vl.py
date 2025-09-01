# coding=utf-8
# Copyright 2025 Baidu and HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Ernie4.5-VL model."""

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import OutputRecorder, check_model_inputs
from ..ernie4_5_moe.modeling_ernie4_5_moe import (
    Ernie4_5_MoeAttention,
    Ernie4_5_MoeMLP,
    Ernie4_5_MoeModel,
    Ernie4_5_MoeRMSNorm,
    Ernie4_5_MoeStatics,
)
from ..qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionPatchEmbed,
    Qwen2_5_VisionRotaryEmbedding,
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLPreTrainedModel,
    Qwen2_5_VLVisionBlock,
)
from ..qwen2_vl.modeling_qwen2_vl import VisionMlp
from .configuration_ernie4_5_vl import Ernie4_5_VLConfig, Ernie4_5_VLTextConfig


logger = logging.get_logger(__name__)


class TokenType:
    text = 0
    image = 1
    video = 2


class Ernie4_5_VLTextRotaryEmbedding(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"

        if self.rope_type != "ernie_3d":
            raise ValueError(
                f"Ernie 4.5 VL requires the `ernie_3d` rope type, but found {self.rope_type} instead."
            )

        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

        # for 3d recomposition
        t_dim = config.rope_scaling["freq_allocation"]  # time dimension
        hw_dim = inv_freq.shape[-1] - t_dim  # height and width dimension
        self.split_sizes = (hw_dim // 2, hw_dim // 2, t_dim)

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids.permute(2, 0, 1)[:, :, None, :].float()  # shape (3, bs, 1, positions)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            cos = freqs.cos() * self.attention_scaling
            sin = freqs.sin() * self.attention_scaling

        sin = self.recomposition_to_3d(sin)
        cos = self.recomposition_to_3d(cos)

        return cos, sin

    def recomposition_to_3d(self, freq):
        freq_h, freq_w, freq_t = (m[(i+1) % 3] for i, m in enumerate(freq.split([*self.split_sizes], dim=-1)))
        freq_hw = torch.stack([freq_h, freq_w], dim=-1).flatten(-2)
        freq_hwt = torch.cat([freq_hw, freq_t], dim=-1)
        return freq_hwt.repeat_interleave(2, dim=-1)


def rotate_half_text(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    original_dtype = q.dtype

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = (q.float() * cos) + (rotate_half_text(q).float() * sin)
    k_embed = (k.float() * cos) + (rotate_half_text(k).float() * sin)

    return q_embed.to(original_dtype), k_embed.to(original_dtype)


class Ernie4_5_VLTextAttention(Ernie4_5_MoeAttention):
    pass


class Ernie4_5_VLRMSNorm(Ernie4_5_MoeRMSNorm):
    pass


class Ernie4_5_VLMLP(Ernie4_5_MoeMLP):
    pass


class Ernie4_5_VLMoeStatics(Ernie4_5_MoeStatics):
    pass


class Ernie4_5_VLSparseMoeBlock(nn.Module):
    def __init__(self, config, num_experts, intermediate_size):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = config.moe_k

        # correction bias (yes it seems to be a typo with statics <> statistics)
        self.moe_statics = Ernie4_5_VLMoeStatics(num_experts_groups=1, num_experts=self.num_experts)

        # gating
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [Ernie4_5_VLMLP(config, intermediate_size) for _ in range(self.num_experts)]
        )
        self.norm_min = config.moe_norm_min

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        device_type = (
            hidden_states.device.type
            if isinstance(hidden_states.device.type, str) and hidden_states.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            # router_logits: (batch * sequence_length, n_experts)
            router_logits = self.gate(hidden_states.float())

            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            _, selected_experts = torch.topk(self.moe_statics(routing_weights), self.top_k, dim=-1)
            routing_weights = torch.gather(routing_weights, dim=-1, index=selected_experts)
            routing_weights = routing_weights / torch.clamp(
                routing_weights.sum(dim=-1, keepdim=True), min=self.norm_min
            )
            routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hitted:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        # moe results are changed to a flattened shape to ease the modality isolated assigning of results
        return final_hidden_states.flatten(), router_logits.flatten()


class Ernie4_5_VLMoeBlock(nn.Module):
    """
    Similar to `Ernie4_5_Moe` where we have modality isolated experts:
        - A set of text experts that are only run on text tokens
        - A set of vision experts that are only run on vision (image/video) tokens

    This modality isolation is unique to the Ernie 4.5 VL models.
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.moe_num_experts

        self.text_moe = Ernie4_5_VLSparseMoeBlock(
            config,
            num_experts=self.num_experts,
            intermediate_size=config.moe_intermediate_size[0]
        )
        self.vision_moe = Ernie4_5_VLSparseMoeBlock(
            config,
            num_experts=self.num_experts,
            intermediate_size=config.moe_intermediate_size[1]
        )

        self.shared_experts = None
        if config.moe_num_shared_experts > 0:
            self.shared_experts = Ernie4_5_VLMLP(config, config.moe_intermediate_size[0] * config.moe_num_shared_experts)

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        # (Optional) shared experts
        if self.shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)

        if token_type_ids is not None and token_type_ids.any():
            final_hidden_states = torch.zeros_like(hidden_states)
            router_logits = torch.zeros(
                size=(batch_size * sequence_length, self.num_experts),
                device=final_hidden_states.device, dtype=torch.float
            )

            # True (1) == vision, False (0) == text tokens
            token_type_ids = token_type_ids.bool()
            token_type_ids_router = token_type_ids.reshape(-1)[:, None].expand(-1, self.num_experts)
            token_type_ids_states = token_type_ids[..., None].expand(-1, -1, hidden_dim)

            # Extract and separate tokens into their modalities
            text_hidden_states = hidden_states[~token_type_ids_states].reshape(batch_size, -1, hidden_dim)
            vision_hidden_states = hidden_states[token_type_ids_states].reshape(batch_size, -1, hidden_dim)

            # Run moe on each modality and assign their results to the original token positions
            final_hidden_states[~token_type_ids_states], router_logits[~token_type_ids_router] = self.text_moe(text_hidden_states)
            final_hidden_states[token_type_ids_states], router_logits[token_type_ids_router] = self.vision_moe(vision_hidden_states)
        else:
            final_hidden_states, router_logits = self.text_moe(hidden_states)
            final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
            router_logits = router_logits.reshape(-1, self.num_experts)

        # Add (optional) shared experts to the result
        if self.shared_experts is not None:
            final_hidden_states = final_hidden_states + shared_output

        return final_hidden_states, router_logits


class Ernie4_5_VLDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Ernie4_5_VLTextAttention(config, layer_idx)

        if (
            ((layer_idx + 1) % config.moe_layer_interval == 0)
            and layer_idx >= config.moe_layer_start_index
            and layer_idx <= config.moe_layer_end_index
        ):
            self.mlp = Ernie4_5_VLMoeBlock(config)
        else:
            self.mlp = Ernie4_5_VLMLP(config)

        self.input_layernorm = Ernie4_5_VLRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Ernie4_5_VLRMSNorm(config.hidden_size, config.rms_norm_eps)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = hidden_states + residual

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if isinstance(self.mlp, Ernie4_5_VLMoeBlock):
            hidden_states, _ = self.mlp(hidden_states, token_type_ids)
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states


class Ernie4_5_VLPreTrainedModel(Qwen2_5_VLPreTrainedModel):
    _can_compile_fullgraph = False

    _can_record_outputs = {
        "router_logits": OutputRecorder(Ernie4_5_VLMoeBlock, index=1),
        "hidden_states": Ernie4_5_VLDecoderLayer,
        "attentions": Ernie4_5_VLTextAttention,
    }
    _keep_in_fp32_modules_strict = ["gate", "moe_statics"]


class Ernie4_5_VLTextModel(Ernie4_5_MoeModel):
    config: Ernie4_5_VLTextConfig

    def __init__(self, config: Ernie4_5_VLTextConfig):
        super().__init__(config)

        del self.padding_idx
        self.embed_tokens = nn.Embedding(
            self.vocab_size,
            config.hidden_size,
        )
        self.rotary_emb = Ernie4_5_VLTextRotaryEmbedding(config=config)

    @check_model_inputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        attention_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids[..., -1],  # TODO: check for the shape
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Ernie4_5VLVisionMLP(VisionMlp):
    pass


class Ernie4_5_VLPatchEmbed(Qwen2_5_VisionPatchEmbed):
    def __init__(
        self,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__(patch_size, in_channels, embed_dim)

        del self.temporal_patch_size
        del kernel_size  # noqa: F821
        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        return self.proj(hidden_states.to(target_dtype))


class Ernie4_5_VLVisionRotaryEmbedding(Qwen2_5_VisionRotaryEmbedding):
    pass


class Ernie4_5_VLVisionBlock(Qwen2_5_VLVisionBlock):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__(config, attn_implementation)

        self.norm1 = nn.LayerNorm(config.hidden_size, config.vision_rms_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, config.vision_rms_norm_eps)
        self.mlp = Ernie4_5VLVisionMLP(
            dim=config.hidden_size,
            hidden_dim=config.intermediate_size,
            hidden_act=config.hidden_act,
        )


class Ernie4_5_VLVisionTransformerPretrainedModel(Qwen2_5_VisionTransformerPretrainedModel):
    _no_split_modules = ["Ernie4_5_VLVisionBlock"]

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)

        del self.fullatt_block_indexes
        del self.window_size
        del self.merger

        self.patch_embed = Ernie4_5_VLPatchEmbed(
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Ernie4_5_VLVisionRotaryEmbedding(head_dim // 2)

        self.ln = nn.LayerNorm(config.hidden_size, eps=config.vision_rms_norm_eps)

    def get_window_index(self, grid_thw):
        raise AttributeError("Ernie 4.5 VL does not use windowed attention!")

    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs,
    ) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)

        seq_len, _ = hidden_states.size()
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = self.ln(hidden_states)
        return hidden_states


class Ernie4_5_VLVariableResolutionResamplerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.in_dim = config.hidden_size
        self.out_dim = config.text_hidden_size
        self.spatial_conv_size = config.spatial_conv_size
        self.temporal_conv_size = config.temporal_conv_size

        # compress 2d conv(picture) to 1d
        self.spatial_dim = self.in_dim * self.spatial_conv_size**2
        # compress 3d conv(video) to 1d
        self.temporal_dim = self.in_dim * self.spatial_conv_size**2 * self.temporal_conv_size

        self.spatial_linear = nn.Sequential(
            nn.Linear(self.spatial_dim, self.spatial_dim),
            nn.GELU(),
            nn.Linear(self.spatial_dim, self.spatial_dim),
            nn.LayerNorm(self.spatial_dim, eps=config.vision_rms_norm_eps),
        )

        self.temporal_linear = nn.Sequential(
            nn.Linear(self.temporal_dim, self.spatial_dim),
            nn.GELU(),
            nn.Linear(self.spatial_dim, self.spatial_dim),
            nn.LayerNorm(self.spatial_dim, eps=config.vision_rms_norm_eps),
        )

        self.mlp = nn.Linear(self.spatial_dim, self.out_dim)
        self.after_norm = Ernie4_5_VLRMSNorm(self.out_dim, config.rms_norm_eps)

    def _temporal_slicing(self, x, grid_thw):
        """
        Creates slices along the temporal dimension (usually if we have a video input).

        If a "real" (video) slicing happens, then we change [1,2,1,2,1,2] to [1,1,1,2,2,2] patterns.
        Otherwise, we repeat along the axis, i.e. [1,1,1] to [1,1,1,1,1,1]. NOTE: It is hard-coded
        for `temporal_conv_size == 2`.
        """
        # Calculating offsets (based on flattened tensors)
        grid_t, grid_hw = grid_thw[:, 0], grid_thw[:, 1:]
        grid_hw_after_conv = grid_hw.prod(-1) // (self.spatial_conv_size**2)

        tokens_per_img_or_vid = (grid_thw.prod(-1) // (self.spatial_conv_size**2)).flatten()
        batch_offsets = torch.empty(
            tokens_per_img_or_vid.size(), dtype=tokens_per_img_or_vid.dtype
        )
        batch_offsets[0] = 0
        batch_offsets[1:] = tokens_per_img_or_vid.cumsum(dim=0)[:-1]

        first_slice_offsets = []
        second_slice_offsets = []
        for temporal_size, spatial_size, batch_offset in zip(
            grid_t, grid_hw_after_conv, batch_offsets
        ):
            # Depending on temporal, we may interleave
            first_offset_range = range(0, temporal_size, 2)
            second_offset_range = range(1 if temporal_size > 1 else 0, temporal_size, 2)

            is_same_offset_range = first_offset_range == second_offset_range
            for temporal_offset in first_offset_range:
                first_slice_offsets.append(
                    torch.arange(
                        batch_offset + (temporal_offset) * spatial_size,
                        batch_offset + (temporal_offset + 1) * spatial_size,
                    )
                )

                # We can avoid looping another time if the ranges are the same
                if is_same_offset_range:
                    second_slice_offsets.append(
                        torch.arange(
                            batch_offset + (temporal_offset) * spatial_size,
                            batch_offset + (temporal_offset + 1) * spatial_size,
                        )
                    )

            if not is_same_offset_range:
                for temporal_offset in second_offset_range:
                    second_slice_offsets.append(
                        torch.arange(
                            batch_offset + (temporal_offset) * spatial_size,
                            batch_offset + (temporal_offset + 1) * spatial_size,
                        )
                    )

        first_slice_offsets = torch.cat(first_slice_offsets, dim=-1).to(x.device)
        second_slice_offsets = torch.cat(second_slice_offsets, dim=-1).to(x.device)

        return torch.concat(
            [
                torch.index_select(x, dim=0, index=first_slice_offsets),
                torch.index_select(x, dim=0, index=second_slice_offsets)
            ],
            dim=-1
        )

    def forward(self, x, grid_thw):
        # image spatial
        x = x.reshape([-1, x.shape[-1] * (self.spatial_conv_size**2)])
        x = self.spatial_linear(x.to(self.mlp.weight.dtype))

        # video temporal
        x = self._temporal_slicing(x, grid_thw)
        x = self.temporal_linear(x)

        # final mlp
        x = self.mlp(x)
        x = self.after_norm(x)

        return x


# TODO: refactor a bit
class Ernie4_5_VLModel(Ernie4_5_VLPreTrainedModel):
    def __init__(self, config: Ernie4_5_VLConfig):
        super().__init__(config)

        self.language_model = Ernie4_5_VLTextModel(config.text_config)
        self.vision_tower = Ernie4_5_VLVisionTransformerPretrainedModel(config.vision_config)
        self.resampler_model = Ernie4_5_VLVariableResolutionResamplerModel(config.vision_config)

        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    # TODO: same with videos
    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        if image_grid_thw is not None:
            grid_thw = image_grid_thw[image_grid_thw > 0].reshape([-1, 3])
            grid_thw = F.pad(
                torch.repeat_interleave(grid_thw[:, 1:], grid_thw[:, 0], 0),
                [1, 0, 0, 0],
                value=1,
            )
        image_embeds = self.vision_tower(pixel_values, grid_thw)
        image_embeds = self.resampler_model(image_embeds, grid_thw)
        return image_embeds

    # TODO: fixup with videos, iirc this is not handled with a token atm
    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor = None,
        video_features: torch.FloatTensor = None,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            # special_video_mask = input_ids == self.config.video_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        """n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(
                f"Videos features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
            )"""

        # return special_image_mask, special_video_mask
        return special_image_mask, None

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        images: Optional[torch.Tensor] = None,
        grid_thw: Optional[torch.Tensor] = None,
        # pixel_values: Optional[torch.Tensor] = None,
        # pixel_values_videos: Optional[torch.FloatTensor] = None,
        # image_grid_thw: Optional[torch.LongTensor] = None,
        # video_grid_thw: Optional[torch.LongTensor] = None,
        # rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # second_per_grid_ts: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        # TODO: logic change for input embeds and videos
        inputs_embeds = self.get_input_embeddings()(input_ids)

        if images is not None:
            image_embeds = self.get_image_features(images, image_grid_thw=grid_thw)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        # TODO: add and check logic with videos ("or" mask?)
        if token_type_ids is None:
            token_type_ids = image_mask.to(torch.int64)
        token_type_ids[token_type_ids == TokenType.video] = TokenType.image

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        return MoeModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


class Ernie4_5_VLForConditionalGeneration(Ernie4_5_VLPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Ernie4_5_VLConfig):
        super().__init__(config)
        self.model = Ernie4_5_VLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs,
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ):
        # TODO: rope delta approach by qwen 2.5 vl?
        position_ids = model_kwargs.pop("position_ids")
        position_ids = torch.cat(
            [position_ids, position_ids.max(dim=1, keepdim=True)[0] + 1],
            dim=1
        )

        model_kwargs = super()._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder, num_new_tokens)
        model_kwargs["position_ids"] = position_ids

        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(input_ids, **kwargs)

        if model_inputs["cache_position"][0] != 0:
            model_inputs["images"] = None  # TODO remove when refactoring preprocessing
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        images: Optional[torch.Tensor] = None,
        grid_thw: Optional[torch.Tensor] = None,
        #pixel_values: Optional[torch.Tensor] = None,
        #pixel_values_videos: Optional[torch.FloatTensor] = None,
        #image_grid_thw: Optional[torch.LongTensor] = None,
        #video_grid_thw: Optional[torch.LongTensor] = None,
        #rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        #second_per_grid_ts: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ):
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.text_config.output_router_logits
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            return_dict=True,
            images=images,
            grid_thw=grid_thw,
            #pixel_values=pixel_values,
            #pixel_values_videos=pixel_values_videos,
            #image_grid_thw=image_grid_thw,
            #video_grid_thw=video_grid_thw,
            #rope_deltas=rope_deltas,
            cache_position=cache_position,
            #second_per_grid_ts=second_per_grid_ts,
            **kwargs,
        )

        if not use_cache:
            logits = self.lm_head(outputs.last_hidden_state)
        else:
            logits = self.lm_head(outputs.last_hidden_state[:, -1:, :])

        # aka Generate Decoding
        loss = None  # TODO
        aux_loss = None  # TODO: load balancing loss

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


__all__ = [
    "Ernie4_5_VLForConditionalGeneration",
    "Ernie4_5_VLModel",
    "Ernie4_5_VLTextModel",
    "Ernie4_5_VLVisionTransformerPretrainedModel",
    "Ernie4_5_VLVariableResolutionResamplerModel",
]
