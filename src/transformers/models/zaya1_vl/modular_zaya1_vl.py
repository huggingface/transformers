# Copyright 2026 Zyphra and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Zaya1-VL model."""

from typing import Any

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_outputs import MoeModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import MultiModalData, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import TransformersKwargs, auto_docstring
from ..llama.modeling_llama import repeat_kv
from ..llava.modeling_llava import LlavaModel
from ..qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
from ..qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel
from ..qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
from ..qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor, Qwen2VLProcessorKwargs
from ..qwen3_5_moe.modeling_qwen3_5_moe import apply_rotary_pos_emb
from ..zaya.configuration_zaya import ZayaConfig
from ..zaya.modeling_zaya import (
    ZayaAttention,
    ZayaCCAProjection,
    ZayaDecoderLayer,
    ZayaExperts,
    ZayaForCausalLM,
    ZayaModel,
    ZayaPreTrainedModel,
    ZayaRotaryEmbedding,
    ZayaSparseMoeBlock,
    eager_attention_forward,
)


@auto_docstring(checkpoint="Zyphra/ZAYA1-VL-8B")
@strict
class Zaya1VLVisionConfig(Qwen2_5_VLVisionConfig):
    r"""
    window_size (`int`, *optional*, defaults to 112):
        Window size used by the Qwen2.5-VL vision encoder.
    out_hidden_size (`int`, *optional*, defaults to 2048):
        Output hidden size after the vision merger.
    fullatt_block_indexes (`list[int]`, *optional*, defaults to `[7, 15, 23, 31]`):
        Vision encoder layers that use full attention.
    """

    model_type = "zaya1_vl_vision"
    base_config_key = "vision_config"

    hidden_size: int = 1280
    temporal_patch_size: int | list[int] | tuple[int, int] = 1
    out_hidden_size: int = 2048

    tokens_per_second = AttributeError()


@auto_docstring(checkpoint="Zyphra/ZAYA1-VL-8B")
@strict
class Zaya1VLTextConfig(ZayaConfig):
    r"""
    lm_head_bias (`bool`, *optional*, defaults to `False`):
        Whether to add a bias to the language modeling head.
    router_hidden_size (`int`, *optional*, defaults to 256):
        Hidden size used by the ZAYA router.
    cca_time0 (`int`, *optional*, defaults to 2):
        First temporal parameter of the CCA projection.
    cca_time1 (`int`, *optional*, defaults to 2):
        Second temporal parameter of the CCA projection.

    vision_lora (`bool`, *optional*, defaults to `True`):
        Whether to enable LoRA modules that are applied only on vision-token positions.
    vision_lora_rank_attn (`int`, *optional*, defaults to 8):
        LoRA rank for the CCA and attention output projections applied to vision-token positions.
    vision_lora_rank_mlp (`int`, *optional*, defaults to 32):
        LoRA rank for the MoE expert projections applied to vision-token positions.
    """

    model_type = "zaya1_vl_text"
    base_config_key = "text_config"

    vision_lora: bool = True
    vision_lora_rank_attn: int = 8
    vision_lora_rank_mlp: int = 32


@auto_docstring(checkpoint="Zyphra/ZAYA1-VL-8B")
@strict
class Zaya1VLConfig(Qwen2VLConfig):
    r"""
    text_config (`dict` or `Zaya1VLTextConfig`, *optional*):
        Configuration for the ZAYA text decoder.
    vision_config (`dict` or `Zaya1VLVisionConfig`, *optional*):
        Configuration for the Qwen2.5-VL vision encoder.
    image_token_id (`int`, *optional*, defaults to 262147):
        Token id used as an image placeholder.
    vision_start_token_id (`int`, *optional*, defaults to 255999):
        Token id that starts an image span.
    vision_end_token_id (`int`, *optional*, defaults to 256000):
        Token id that ends an image span.
    """

    model_type = "zaya1_vl"
    sub_configs = {"vision_config": Zaya1VLVisionConfig, "text_config": Zaya1VLTextConfig}

    image_token_id: int = 262147
    vision_start_token_id: int = 255999
    vision_end_token_id: int = 256000
    video_token_id = AttributeError()

    tie_word_embeddings: bool = True
    output_router_logits: bool = False


class Zaya1VLRotaryEmbedding(ZayaRotaryEmbedding):
    pass


def _make_lora_pair(in_features: int, rank: int, out_features: int) -> tuple[nn.Linear, nn.Linear]:
    return nn.Linear(in_features, rank, bias=False), nn.Linear(rank, out_features, bias=False)


def _apply_masked_lora(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    lora_a: nn.Linear | torch.Tensor,
    lora_b: nn.Linear | torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    if mask is None:
        return output
    indices = mask.nonzero(as_tuple=True)
    if indices[0].numel() == 0:
        return output
    hidden_states = hidden_states[indices]
    hidden_states = F.linear(hidden_states, lora_a) if isinstance(lora_a, torch.Tensor) else lora_a(hidden_states)
    hidden_states = F.linear(hidden_states, lora_b) if isinstance(lora_b, torch.Tensor) else lora_b(hidden_states)
    return output.index_put(indices, hidden_states.to(output.dtype), accumulate=True)


class Zaya1VLCCAProjection(ZayaCCAProjection):
    def __init__(self, config: Zaya1VLTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        if config.vision_lora:
            self.q_lora_a, self.q_lora_b = _make_lora_pair(
                self.hidden_size, config.vision_lora_rank_attn, self.num_attention_heads * self.head_dim
            )
            self.k_lora_a, self.k_lora_b = _make_lora_pair(
                self.hidden_size, config.vision_lora_rank_attn, self.num_key_value_heads * self.head_dim
            )
            self.v_current_lora_a, self.v_current_lora_b = _make_lora_pair(
                self.hidden_size, config.vision_lora_rank_attn, self.num_key_value_heads * self.head_dim // 2
            )
            self.v_delayed_lora_a, self.v_delayed_lora_b = _make_lora_pair(
                self.hidden_size, config.vision_lora_rank_attn, self.num_key_value_heads * self.head_dim // 2
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Cache | None,
        padding_mask: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
    ):
        if padding_mask is not None:
            hidden_states = hidden_states * padding_mask[:, :, None].to(hidden_states.dtype)

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        projected_queries = self.q_proj(hidden_states)
        projected_keys = self.k_proj(hidden_states)

        if self.config.vision_lora and image_mask is not None:
            # visual specific: apply LoRA only on vision-token positions
            projected_queries = _apply_masked_lora(
                projected_queries, hidden_states, self.q_lora_a, self.q_lora_b, image_mask
            )
            projected_keys = _apply_masked_lora(
                projected_keys, hidden_states, self.k_lora_a, self.k_lora_b, image_mask
            )

        qk_states = torch.cat([projected_queries, projected_keys], dim=-1)

        query_residual = projected_queries.view(*hidden_shape)
        key_residual = projected_keys.view(*hidden_shape).transpose(1, 2)
        key_residual = repeat_kv(key_residual, self.num_key_value_groups).transpose(1, 2)
        query_residual = (query_residual + key_residual) * 0.5
        key_residual = query_residual.view(*input_shape, -1, self.num_key_value_groups, self.head_dim).mean(dim=-2)

        qk_states = qk_states.transpose(1, 2)
        use_precomputed_states = past_key_values is not None and past_key_values.has_previous_state(self.layer_idx)
        if use_precomputed_states:
            cached_qk_states = past_key_values.layers[self.layer_idx].conv_states
            qk_states = torch.cat([cached_qk_states, qk_states], dim=-1)
        else:
            qk_states = F.pad(qk_states, (self.conv_kernel_size, 0))

        if past_key_values is not None:
            new_conv_state = qk_states[..., -self.conv_kernel_size :]
            if new_conv_state.shape[-1] < self.conv_kernel_size:
                new_conv_state = F.pad(new_conv_state, (self.conv_kernel_size - new_conv_state.shape[-1], 0))
            past_key_values.update_conv_state(new_conv_state, self.layer_idx)

        qk_states = self.conv_qk_depthwise(qk_states)
        qk_states = self.conv_qk_grouped(qk_states).transpose(1, 2)

        query_hidden_size = query_residual.shape[-2] * query_residual.shape[-1]
        query = qk_states[..., :query_hidden_size].view(*hidden_shape) + query_residual
        key = qk_states[..., query_hidden_size:].view(*hidden_shape) + key_residual

        value_current = self.v_proj_current(hidden_states)
        delayed_v_state = self.v_proj_delayed(hidden_states)

        if self.config.vision_lora and image_mask is not None:
            # visual specific: apply LoRA only on vision-token positions
            value_current = _apply_masked_lora(
                value_current, hidden_states, self.v_current_lora_a, self.v_current_lora_b, image_mask
            )
            delayed_v_state = _apply_masked_lora(
                delayed_v_state, hidden_states, self.v_delayed_lora_a, self.v_delayed_lora_b, image_mask
            )

        if use_precomputed_states:
            recurrent_v_state = past_key_values.layers[self.layer_idx].recurrent_states.unsqueeze(1)
        else:
            recurrent_v_state = self.v_proj_delayed(hidden_states.new_zeros(input_shape[0], 1, self.hidden_size))
        value_delayed = torch.cat([recurrent_v_state, delayed_v_state[:, :-1]], dim=1)

        if past_key_values is not None:
            past_key_values.update_recurrent_state(delayed_v_state[:, -1, :], self.layer_idx)

        value = torch.cat([value_current, value_delayed], dim=-1).view(*hidden_shape)

        return query, key, value


class Zaya1VLAttention(ZayaAttention):
    def __init__(self, config: Zaya1VLTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        if config.vision_lora:
            self.o_lora_a, self.o_lora_b = _make_lora_pair(
                config.num_attention_heads * self.head_dim, config.vision_lora_rank_attn, config.hidden_size
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: dict[str, Any] | None = None,
        past_key_values: Cache | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        image_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]

        mask_mapping = attention_mask or {}
        causal_mask = mask_mapping.get("causal")
        padding_mask = mask_mapping.get("padding")

        query_states, key_states, value_states = self.qkv_proj(
            hidden_states, past_key_values, padding_mask, image_mask=image_mask
        )
        query_states, key_states = self.qk_norm(query_states, key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            causal_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.view(*input_shape, -1)
        output = self.o_proj(attn_output)

        if self.config.vision_lora and image_mask is not None:
            # visual specific: apply LoRA only on vision-token positions
            output = _apply_masked_lora(output, attn_output, self.o_lora_a, self.o_lora_b, image_mask)

        return output, attn_weights


def identity_decorator(cls):
    """
    modular transformers need new decorators to overwrite the old ones e.g. use_experts_implementation;
    this decorator is just used to skip them.
    """
    return cls


@identity_decorator
class Zaya1VLExperts(ZayaExperts):
    def __init__(self, config: Zaya1VLTextConfig):
        super().__init__(config)
        self.vision_lora = config.vision_lora
        if self.vision_lora:
            self.lora_gate_up_proj_a = nn.Parameter(
                torch.empty(self.num_experts, config.vision_lora_rank_mlp, self.hidden_dim)
            )
            self.lora_gate_up_proj_b = nn.Parameter(
                torch.empty(self.num_experts, 2 * self.intermediate_dim, config.vision_lora_rank_mlp)
            )
            self.lora_down_proj_a = nn.Parameter(
                torch.empty(self.num_experts, config.vision_lora_rank_mlp, self.intermediate_dim)
            )
            self.lora_down_proj_b = nn.Parameter(
                torch.empty(self.num_experts, self.hidden_dim, config.vision_lora_rank_mlp)
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        image_mask_flat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate_up = F.linear(current_state, self.gate_up_proj[expert_idx])

            image_mask_curr_expert = None
            if self.vision_lora and image_mask_curr_expert is not None:
                image_mask_curr_expert = image_mask_flat[token_idx]
                # visual specific: apply expert LoRA only on vision-token positions
                gate_up = _apply_masked_lora(
                    gate_up,
                    current_state,
                    self.lora_gate_up_proj_a[expert_idx],
                    self.lora_gate_up_proj_b[expert_idx],
                    image_mask_curr_expert,
                )

            gate, up = gate_up.chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            down = F.linear(current_hidden_states, self.down_proj[expert_idx])

            if image_mask_curr_expert is not None:
                # visual specific: apply expert LoRA only on vision-token positions
                down = _apply_masked_lora(
                    down,
                    current_hidden_states,
                    self.lora_down_proj_a[expert_idx],
                    self.lora_down_proj_b[expert_idx],
                    image_mask_curr_expert,
                )

            down = down * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, down.to(final_hidden_states.dtype))

        return final_hidden_states


class Zaya1VLSparseMoeBlock(ZayaSparseMoeBlock):
    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_router_hidden_states: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, router_probs, router_indices, prev_router_hidden_states = self.gate(
            hidden_states, router_states=prev_router_hidden_states
        )

        batch_size, seq_length, emb_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(batch_size * seq_length, emb_dim)
        image_mask_flat = image_mask.reshape(batch_size * seq_length) if image_mask is not None else None
        expert_output = self.experts(hidden_states_flat, router_indices, router_probs, image_mask_flat=image_mask_flat)
        expert_output = expert_output.view(batch_size, seq_length, emb_dim)

        return expert_output, prev_router_hidden_states


class Zaya1VLDecoderLayer(ZayaDecoderLayer):
    def __init__(self, config: Zaya1VLTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_router_hidden_states: torch.Tensor | None = None,
        attention_mask: dict[str, Any] | None = None,
        past_key_values: Cache | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        image_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            image_mask=image_mask,
            **kwargs,
        )

        residual = self.post_attention_residual_scale(hidden_states, residual)
        hidden_states = self.post_attention_layernorm(residual)

        hidden_states, prev_router_hidden_states = self.mlp(
            hidden_states,
            prev_router_hidden_states,
            image_mask=image_mask,
        )

        hidden_states = self.post_mlp_residual_scale(hidden_states, residual)

        return hidden_states, prev_router_hidden_states


class Zaya1VLPreTrainedModel(ZayaPreTrainedModel):
    _no_split_modules = ["Zaya1VLDecoderLayer", "Qwen2_5_VLVisionBlock"]
    input_modalities = ("image", "text")

    def _init_weights(self, module):
        super()._init_weights(self, module)

        # specific for visual expert lora
        if isinstance(module, Zaya1VLExperts):
            if module.vision_lora:
                lora_param_names = "lora_gate_up_proj_a", "lora_gate_up_proj_b", "lora_down_proj_a", "lora_down_proj_b"
                for param_name in lora_param_names:
                    init.normal_(getattr(module, param_name), mean=0.0, std=0.02)


class Zaya1VLTextModel(ZayaModel):
    config: Zaya1VLTextConfig

    def __init__(self, config: Zaya1VLTextConfig):
        super().__init__(config)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        image_mask: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        r"""
        image_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Boolean mask selecting image placeholder token positions.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        if attention_mask is not None and attention_mask.ndim != 2:
            raise ValueError(
                "ZAYA CCA projection requires a 2D `attention_mask` to mask padding tokens before convolution."
            )

        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        # Original ZAYA SWA only applies the local causal pattern; padding tokens are zeroed before the CCA projection.
        sliding_mask_kwargs = {**mask_kwargs, "attention_mask": None}
        mask_creation_functions = {
            "hybrid": lambda: create_causal_mask(**mask_kwargs),
            "hybrid_sliding": lambda: create_sliding_window_causal_mask(**sliding_mask_kwargs),
        }
        causal_mask_mapping = {
            layer_type: mask_creation_functions[layer_type]() for layer_type in set(self.config.layer_types)
        }
        cca_mask = self._update_cca_mask(attention_mask, past_key_values, inputs_embeds)
        if inputs_embeds.shape[1] == 1:
            image_mask = None

        hidden_states = inputs_embeds
        position_embeddings = {
            layer_type: self.rotary_emb(hidden_states, position_ids, layer_type)
            for layer_type in set(self.config.layer_types)
        }

        hidden_states = (hidden_states + self.input_hidden_states_bias) * self.input_hidden_states_scale
        prev_router_hidden_states = None

        for layer_n, decoder_layer in enumerate(self.layers):
            layer_type = self.config.layer_types[layer_n]
            causal_mask = causal_mask_mapping[layer_type]
            if image_mask is not None and causal_mask is not None and causal_mask.shape[-1] == image_mask.shape[-1]:
                image_pair_mask = image_mask[:, None, :, None] & image_mask[:, None, None, :]
                causal_mask = causal_mask.clone().masked_fill(image_pair_mask, 0)
            mask_mapping = {"causal": causal_mask, "padding": cca_mask}
            hidden_states, prev_router_hidden_states = decoder_layer(
                hidden_states,
                prev_router_hidden_states,
                attention_mask=mask_mapping,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings[layer_type],
                image_mask=image_mask,
                **kwargs,
            )

        hidden_states = self.final_norm(hidden_states)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class Zaya1VLVisionModel(Qwen2_5_VisionTransformerPretrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)


@auto_docstring
class Zaya1VLModel(LlavaModel, Zaya1VLPreTrainedModel):
    def __init__(self, config: Zaya1VLConfig):
        Zaya1VLPreTrainedModel.__init__(self, config)
        self.visual = Zaya1VLVisionModel._from_config(config.vision_config)
        self.language_model = Zaya1VLTextModel(config.text_config)
        self.post_init()

    def get_image_features(
        self, pixel_values: torch.FloatTensor, image_grid_thw: torch.LongTensor, **kwargs: Unpack[TransformersKwargs]
    ) -> torch.FloatTensor:
        r"""
        pixel_values (`torch.FloatTensor`):
            The tensors corresponding to the input images.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
            The temporal, height and width grid of each image after image preprocessing.
        """
        pixel_values = pixel_values.type(self.visual.dtype)
        return self.visual(pixel_values, grid_thw=image_grid_thw, return_dict=True, **kwargs).pooler_output

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width grid of each image after image preprocessing.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        image_mask = None
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values, image_grid_thw, **kwargs)
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features.unsqueeze(0)
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)
            image_mask = image_mask[..., 0]

        return self.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            image_mask=image_mask,
            use_cache=use_cache,
            **kwargs,
        )


@auto_docstring
class Zaya1VLForConditionalGeneration(ZayaForCausalLM, Zaya1VLPreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    def __init__(self, config: Zaya1VLConfig):
        super().__init__(self, config)
        self.vocab_size = config.text_config.vocab_size
        self.lm_head = nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=config.text_config.lm_head_bias
        )
        self.post_init()

    def get_image_features(
        self, pixel_values: torch.FloatTensor, image_grid_thw: torch.LongTensor, **kwargs: Unpack[TransformersKwargs]
    ) -> torch.FloatTensor:
        return self.model.get_image_features(pixel_values=pixel_values, image_grid_thw=image_grid_thw, **kwargs)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        image_grid_thw=None,
        is_first_iteration=False,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=use_cache,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )
        if not is_first_iteration and use_cache:
            model_inputs["pixel_values"] = None
        return model_inputs


class Zaya1VLProcessorKwargs(Qwen2VLProcessorKwargs):
    pass


@auto_docstring
class Zaya1VLProcessor(Qwen2VLProcessor):
    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        self.image_token = getattr(tokenizer, "image_token", "<image>")
        self.image_token_id = getattr(tokenizer, "image_token_id", None) or tokenizer.convert_tokens_to_ids(
            self.image_token
        )
        ProcessorMixin.__init__(self, image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        **kwargs: Unpack[Zaya1VLProcessorKwargs],
    ) -> BatchFeature:
        r"""
        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model.
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **image_grid_thw** -- List of image 3D grid in LLM. Returned when `images` is not `None`.
        """
        output_kwargs = self._merge_kwargs(
            Zaya1VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        image_inputs = {}
        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]

        text = text.copy() if isinstance(text, list) else [text]
        if images is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    num_image_tokens = image_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids")
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"], return_tensors=None)
        self._check_special_mm_tokens(text, text_inputs, modalities=["image"])

        if return_mm_token_type_ids:
            text_inputs["mm_token_type_ids"] = self.create_mm_token_type_ids(text_inputs["input_ids"])

        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        """
        Computes the number of placeholder tokens needed for image inputs with the given sizes.
        Args:
            image_sizes (`list[list[int]]`, *optional*):
                The input sizes formatted as (height, width) per each image.
        Returns:
            `MultiModalData`: A `MultiModalData` object holding number of tokens per image.
        """
        if image_sizes is None:
            return MultiModalData()

        images_kwargs = {**Zaya1VLProcessorKwargs._defaults.get("images_kwargs", {}), **kwargs}
        merge_size = images_kwargs.get("merge_size") or self.image_processor.merge_size
        num_image_patches = [
            self.image_processor.get_number_of_image_patches(*image_size, images_kwargs) for image_size in image_sizes
        ]
        num_image_tokens = [(num_patches // merge_size**2) for num_patches in num_image_patches]
        return MultiModalData(num_image_tokens=num_image_tokens, num_image_patches=num_image_patches)

    @property
    def model_input_names(self):
        return self.image_processor.model_input_names + self.tokenizer.model_input_names


__all__ = [
    "Zaya1VLTextConfig",
    "Zaya1VLVisionConfig",
    "Zaya1VLConfig",
    "Zaya1VLVisionModel",
    "Zaya1VLModel",
    "Zaya1VLPreTrainedModel",
    "Zaya1VLTextModel",
    "Zaya1VLForConditionalGeneration",
    "Zaya1VLProcessor",
]
