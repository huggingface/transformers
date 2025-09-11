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

import itertools
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    DefaultFastImageProcessorKwargs,
    group_images_by_shape,
    reorder_images,
)
from ...image_transforms import (
    convert_to_rgb,
    resize,
    to_channel_dimension_format,
)
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
)
from ...masking_utils import create_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ...processing_utils import ImagesKwargs, ProcessingKwargs, Unpack
from ...utils import (
    TensorType,
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    is_torchdynamo_compiling,
    logging,
)
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import OutputRecorder, check_model_inputs
from ...video_utils import VideoInput
from ..ernie4_5_moe.modeling_ernie4_5_moe import (
    Ernie4_5_MoeAttention,
    Ernie4_5_MoeMLP,
    Ernie4_5_MoeModel,
    Ernie4_5_MoeRMSNorm,
    Ernie4_5_MoeStatics,
)
from ..glm4v.modeling_glm4v import Glm4vForConditionalGeneration
from ..qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionPatchEmbed,
    Qwen2_5_VisionRotaryEmbedding,
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLModel,
    Qwen2_5_VLPreTrainedModel,
    Qwen2_5_VLVisionBlock,
)
from ..qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor, smart_resize
from ..qwen2_vl.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast
from ..qwen2_vl.modeling_qwen2_vl import VisionMlp
from ..qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
from ..qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor
from .configuration_ernie4_5_vl import Ernie4_5_VLConfig, Ernie4_5_VLTextConfig


logger = logging.get_logger(__name__)


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
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1).to(x.device)
        )
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

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
            # position_ids=position_ids[..., -1],  # TODO: needs separate text pos ids
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


class Ernie4_5_VLModel(Qwen2_5_VLModel):
    _checkpoint_conversion_mapping = {}

    def __init__(self, config: Ernie4_5_VLConfig):
        super().__init__(config)

        del self.visual
        self.vision_tower = Ernie4_5_VLVisionTransformerPretrainedModel(config.vision_config)
        self.resampler_model = Ernie4_5_VLVariableResolutionResamplerModel(config.vision_config)

    def get_position_ids(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        """TODO description"""
        attention_mask_tensor = (
            attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
        )
        if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
            attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
            # Only apply conversion for floating point tensors (inverted masks)
            if attention_mask_tensor.dtype.is_floating_point:
                attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                attention_mask_tensor = (1.0 - attention_mask_tensor).int()

        # Calculate RoPE index once per generation in the pre-fill stage only.
        # When compiling, we can't check tensor values thus we check only input length
        # It is safe to assume that `length!=1` means we're in pre-fill because compiled
        # models currently cannot do asssisted decoding
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask=attention_mask_tensor,
            )
            self.rope_deltas = rope_deltas
        # then use the prev pre-calculated rope-deltas to get the correct position ids
        else:
            batch_size, seq_length, device = input_ids.shape[0], 1, input_ids.device
            delta = (
                (cache_position[0] + self.rope_deltas).to(device)
                if cache_position is not None
                else 0
            )
            position_ids = torch.arange(seq_length, device=device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:  # otherwise `deltas` is an int `0`
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        return position_ids

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Examples:
                Temporal (Time): 3 patches, representing different segments of the video in time.
                Height: 2 patches, dividing each frame vertically.
                Width: 2 patches, dividing each frame horizontally.
                We also have some important parameters:
                fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
                tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
                temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
                interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [101, 102, 103, 104, 105]
                text height position_ids: [101, 102, 103, 104, 105]
                text width position_ids: [101, 102, 103, 104, 105]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """

        image_token_id = self.config.image_token_id
        video_start_token_id = self.config.video_start_token_id
        video_end_token_id = self.config.video_end_token_id
        # TODO: rename `conv -> merge`
        temporal_merge_size = self.config.vision_config.temporal_conv_size
        spatial_merge_size = self.config.vision_config.spatial_conv_size

        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            video_group_index = 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                input_tokens = input_ids.tolist()

                input_token_type = []
                video_check_flg = False
                for token in input_tokens:
                    if token == video_start_token_id:
                        video_check_flg = True
                    elif token == video_end_token_id:
                        video_check_flg = False

                    if token == image_token_id and not video_check_flg:
                        input_token_type.append("image")
                    elif token == image_token_id and video_check_flg:
                        input_token_type.append("video")
                    else:
                        input_token_type.append("text")

                input_type_group = []
                for key, group in itertools.groupby(enumerate(input_token_type), lambda x: x[1]):
                    group = list(group)
                    start_index = group[0][0]
                    end_index = group[-1][0] + 1
                    input_type_group.append((key, start_index, end_index))

                llm_pos_ids_list = []
                video_frame_num = 1
                for modality_type, start_idx, end_idx in input_type_group:
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0

                    if modality_type == "image":
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        llm_grid_t, llm_grid_h, llm_grid_w = (
                            t.item(),
                            h.item() // spatial_merge_size,
                            w.item() // spatial_merge_size,
                        )

                        t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                        h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                        w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                        llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + st_idx)

                        image_index += 1
                        video_frame_num = 1

                    elif modality_type == "video":
                        t, h, w = (
                            video_frame_num,  # TODO: check for correctness, og uses video idx here as well
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )

                        llm_grid_t, llm_grid_h, llm_grid_w = (
                            t.item() // temporal_merge_size,
                            h.item() // spatial_merge_size,
                            w.item() // spatial_merge_size,
                        )

                        for t_idx in range(llm_grid_t):
                            t_index = torch.tensor(t_idx).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                            h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(1, -1, llm_grid_w).flatten()
                            w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(1, llm_grid_h, -1).flatten()
                            llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + st_idx)

                        video_group_index += 1

                        if video_group_index >= video_grid_thw[video_index][0]:
                            video_index += 1
                            video_group_index = 0

                        video_frame_num += 1

                    else:
                        text_len = end_idx - start_idx
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                        video_frame_num = 1

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    # TODO: same with videos
    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        pixel_values = pixel_values.to(self.vision_tower.device)
        image_embeds = self.vision_tower(pixel_values, image_grid_thw)
        image_embeds = self.resampler_model(image_embeds, image_grid_thw)
        return image_embeds

    def get_tokentype_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
    ):
        """Return the mask indicating a multimodal token (including the start/end tokens of an image/video)"""
        def get_mask_for_token_id(token_id):
            if input_ids is not None:
                return (input_ids == token_id).bool()

            mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            return mask.all(-1).bool()

        total_mask = get_mask_for_token_id(self.config.image_token_id)
        for token_id in [
            self.config.image_start_token_id,
            self.config.image_end_token_id,
            self.config.video_token_id,
            self.config.video_start_token_id,
            self.config.video_end_token_id,
        ]:
            total_mask = torch.logical_or(total_mask, get_mask_for_token_id(token_id))
        return total_mask

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
        images: Optional[torch.Tensor] = None,  # TODO: remove after refactoring all
        grid_thw: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, MoeModelOutputWithPast]:
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # TODO: remove after refactoring all
        if pixel_values is None and images is not None:
            pixel_values = images
        if image_grid_thw is None and grid_thw is not None:
            image_grid_thw = grid_thw

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            # TODO
            #image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            # TODO
            #video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if position_ids is None:
            position_ids = self.get_position_ids(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                cache_position=cache_position,
            )

        if token_type_ids is None:
            token_type_ids = self.get_tokentype_mask(input_ids, inputs_embeds)

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


class Ernie4_5_VLForConditionalGeneration(Glm4vForConditionalGeneration, GenerationMixin):
    @property
    def visual(self):
        return self.model.vision_tower

    def prepare_inputs_for_generation(
        self,
        input_ids,
        inputs_embeds=None,
        attention_mask=None,
        cache_position=None,
        past_key_values=None,
        grid_thw=None,  # TODO remove after refactor
        image_grid_thw=None,
        video_grid_thw=None,
        # Intentionally ignore position ids and token type ids to force custom cache logic of 3D position ids
        position_ids=None,
        token_type_ids=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            grid_thw=grid_thw,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            **kwargs,
        )

        # TODO: remove after refactor
        if image_grid_thw is None and grid_thw is not None:
            image_grid_thw = grid_thw
            model_inputs["image_grid_thw"] = grid_thw

        # Using our own caching with rope delta
        model_inputs["position_ids"] = self.model.get_position_ids(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            cache_position=cache_position,
        )
        # Using our own token type logic
        model_inputs["token_type_ids"] = self.model.get_tokentype_mask(
            model_inputs["input_ids"],
            model_inputs["inputs_embeds"],
        )

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
        images: Optional[torch.Tensor] = None,  # TODO: remove after refactoring all
        grid_thw: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, MoeCausalLMOutputWithPast]:
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
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
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


class Ernie4_5_VLImageProcessor(Qwen2VLImageProcessor):
    r"""
    Constructs a Ernie 4.5 VL image processor that dynamically resizes images based on the original images.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions.
        size (`dict[str, int]`, *optional*, defaults to `{"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 6177}`):
            Size of the image after resizing. `shortest_edge` and `longest_edge` keys must be present.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use when resizing the image.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image.
        image_mean (`float` or `list[float]`, *optional*, defaults to `[0.48145466, 0.4578275, 0.40821073]`):
            Mean to use if normalizing the image. This is a float or list of floats for each channel in the image.
        image_std (`float` or `list[float]`, *optional*, defaults to `[0.26862954, 0.26130258, 0.27577711]`):
            Standard deviation to use if normalizing the image. This is a float or list of floats for each channel
            in the image.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
        min_pixels (`int`, *optional*, defaults to `56 * 56`):
            The min pixels of the image to resize the image.
        max_pixels (`int`, *optional*, defaults to `28 * 28 * 6177`):
            The max pixels of the image to resize the image.
        patch_size (`int`, *optional*, defaults to 14):
            The spatial patch size of the vision encoder.
        merge_size (`int`, *optional*, defaults to 2):
            The merge size of the vision encoder to llm encoder.
    """

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[dict[str, int]] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        do_convert_rgb: bool = True,
        min_pixels: Optional[int] = 56 * 56,
        max_pixels: Optional[int] = 6177 * 28 * 28,
        patch_size: int = 14,
        merge_size: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # Keeping it for modular compatibility
        self.temporal_patch_size = None

    def _preprocess(
        self,
        images: Union[ImageInput, VideoInput],
        do_resize: Optional[bool] = None,
        size: Optional[dict[str, int]] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        patch_size: Optional[int] = None,
        temporal_patch_size: Optional[int] = None,  # Only kept for modular
        merge_size: Optional[int] = None,
        do_convert_rgb: Optional[bool] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Preprocess an image or batch of images. Copy of the `preprocess` method from `CLIPImageProcessor`.

        Args:
            images (`ImageInput`):
                Image or batch of images to preprocess. Expects pixel values ranging from 0 to 255. If pixel values range from 0 to 1, set `do_rescale=False`.
            vision_info (`list[Dict]`, *optional*):
                Optional list of dictionaries containing additional information about vision inputs.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after resizing. `shortest_edge` and `longest_edge` keys must be present.
            resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the `PILImageResampling` enums.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Scale factor to use if rescaling the image.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `list[float]`, *optional*, defaults to `self.image_mean`):
                Mean to use if normalizing the image. Can be a float or a list of floats corresponding to the number of channels in the image.
            image_std (`float` or `list[float]`, *optional*, defaults to `self.image_std`):
                Standard deviation to use if normalizing the image. Can be a float or a list of floats corresponding to the number of channels in the image.
            patch_size (`int`, *optional*, defaults to `self.patch_size`):
                The spatial patch size of the vision encoder.
            temporal_patch_size (`int`, *optional*, defaults to `self.temporal_patch_size`):
                The temporal patch size of the vision encoder.
            merge_size (`int`, *optional*, defaults to `self.merge_size`):
                The merge size of the vision encoder to llm encoder.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            data_format (`ChannelDimension`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.   - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        images = make_list_of_images(images)

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if do_rescale and is_scaled_image(images[0]):
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )
        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        height, width = get_image_size(images[0], channel_dim=input_data_format)
        resized_height, resized_width = height, width
        processed_images = []
        for image in images:
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=patch_size * merge_size,
                    min_pixels=size["shortest_edge"],
                    max_pixels=size["longest_edge"],
                )
                image = resize(
                    image, size=(resized_height, resized_width), resample=resample, input_data_format=input_data_format
                )

            if do_rescale:
                image = self.rescale(image, scale=rescale_factor, input_data_format=input_data_format)

            if do_normalize:
                image = self.normalize(
                    image=image, mean=image_mean, std=image_std, input_data_format=input_data_format
                )

            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
            processed_images.append(image)

        patches = np.array(processed_images)
        if data_format == ChannelDimension.LAST:
            patches = patches.transpose([0, 3, 1, 2])

        # Main difference to Qwen2 VL - no temporal patches
        channel = patches.shape[1]
        grid_t = patches.shape[0]
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        patches = patches.reshape(
            [
                grid_t,
                channel,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            ]
        )
        # [grid_t, grid_h/merge, grid_w/merge, merge, merge, channel, patch, patch]
        patches = patches.transpose([0, 2, 5, 3, 6, 1, 4, 7])
        flatten_patches = patches.reshape(grid_t * grid_h * grid_w, channel * patch_size * patch_size)

        return flatten_patches, (grid_t, grid_h, grid_w)


class Ernie4_5_VLFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    """
    min_pixels (`int`, *optional*, defaults to `56 * 56`):
        The min pixels of the image to resize the image.
    max_pixels (`int`, *optional*, defaults to `28 * 28 * 6177`):
        The max pixels of the image to resize the image.
    patch_size (`int`, *optional*, defaults to 14):
        The spatial patch size of the vision encoder.
    merge_size (`int`, *optional*, defaults to 2):
        The merge size of the vision encoder to llm encoder.
    """

    min_pixels: Optional[int]
    max_pixels: Optional[int]
    patch_size: Optional[int]
    merge_size: Optional[int]


@auto_docstring
class Ernie4_5_VLImageProcessorFast(Qwen2VLImageProcessorFast):
    size = {"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 6177}
    temporal_patch_size = None  # Only kept for modular

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"],
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        patch_size: int,
        merge_size: int,
        disable_grouping: Optional[bool],
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs,
    ):
        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            height, width = stacked_images.shape[-2:]
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=patch_size * merge_size,
                    min_pixels=size["shortest_edge"],
                    max_pixels=size["longest_edge"],
                )
                stacked_images = self.resize(
                    image=stacked_images,
                    size=SizeDict(height=resized_height, width=resized_width),
                    interpolation=interpolation,
                )
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        processed_grids = {}
        for shape, stacked_images in grouped_images.items():
            resized_height, resized_width = stacked_images.shape[-2:]
            # Fused rescale and normalize
            patches = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            if patches.ndim == 4:
                # add a temporal dimension if we have images
                patches = patches.unsqueeze(1)

            # Main difference to Qwen2 VL - no temporal patches
            batch_size, grid_t, channel = patches.shape[:3]
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            patches = patches.view(
                batch_size,
                grid_t,
                channel,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
            # Reorder dimensions to group grid and patch information for subsequent flattening.
            # [batch, grid_t, grid_h/merge, grid_w/merge, merge, merge, channel, patch, patch]
            patches = patches.permute(0, 1, 3, 6, 4, 7, 2, 5, 8)

            flatten_patches = patches.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                channel * patch_size * patch_size,
            )

            processed_images_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_grids = reorder_images(processed_grids, grouped_images_index)
        pixel_values = torch.cat(processed_images, dim=0)
        image_grid_thw = torch.tensor(processed_grids)

        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}, tensor_type=return_tensors
        )


# TODO: needs customization around drawing timestamps on each frame
# See https://huggingface.co/baidu/ERNIE-4.5-VL-28B-A3B-PT/blob/main/processing_ernie4_5_vl.py#L1314-L1341
# TODO: different pixel defaults
class Ernie4_5_VLVideoProcessor(Qwen2VLVideoProcessor):
    pass


class Ernie4_5_VLImagesKwargs(ImagesKwargs):
    min_pixels: Optional[int]
    max_pixels: Optional[int]
    patch_size: Optional[int]
    merge_size: Optional[int]


class Ernie4_5_VLProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Ernie4_5_VLImagesKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": False,
        },
    }


class Ernie4_5_VLProcessor(Qwen2VLProcessor):
    r"""
    Constructs a Ernie 4.5 VL processor which wraps a Ernie 4.5 VL image processor and a Llama tokenizer into a single processor.
    [`Ernie4_5_VLProcessor`] offers all the functionalities of [`Ernie4_5_VLImageProcessor`] and [`LlamaTokenizerFast`]. See the
    [`~Ernie4_5_VLProcessor.__call__`] and [`~Ernie4_5_VLProcessor.decode`] for more information.
    Args:
        image_processor ([`Ernie4_5_VLImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`], *optional*):
            The tokenizer is a required input.
        video_processor ([`Ernie4_5_VLVideoProcessor`], *optional*):
            The video processor is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    tokenizer_class = (None, "LlamaTokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, video_processor=None, chat_template=None, **kwargs):
        super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template)
        self.image_token = "<|IMAGE_PLACEHOLDER|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.video_token = "<|VIDEO_PLACEHOLDER|>" if not hasattr(tokenizer, "video_token") else tokenizer.video_token
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = (
            tokenizer.video_token_id
            if getattr(tokenizer, "video_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.video_token)
        )


__all__ = [
    "Ernie4_5_VLProcessor",
    "Ernie4_5_VLVideoProcessor",
    "Ernie4_5_VLImageProcessor",
    "Ernie4_5_VLImageProcessorFast",
    "Ernie4_5_VLForConditionalGeneration",
    "Ernie4_5_VLModel",
    "Ernie4_5_VLTextModel",
    "Ernie4_5_VLVisionTransformerPretrainedModel",
    "Ernie4_5_VLVariableResolutionResamplerModel",
]
