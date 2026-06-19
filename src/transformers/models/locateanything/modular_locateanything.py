# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""LocateAnything: a MoonViT vision encoder + MLP projector + Qwen2.5 language model for visual grounding."""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPast, BaseModelOutputWithPooling
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel
from ..llava.modeling_llava import (
    LlavaCausalLMOutputWithPast,
    LlavaForConditionalGeneration,
    LlavaModel,
    LlavaModelOutputWithPast,
)


logger = logging.get_logger(__name__)


@auto_docstring
@strict
class LocateAnythingVisionConfig(PreTrainedConfig):
    r"""
    init_pos_emb_height (`int`, *optional*, defaults to 64):
        Height of the learnable position embedding grid that is bicubically interpolated to each image's grid.
    init_pos_emb_width (`int`, *optional*, defaults to 64):
        Width of the learnable position embedding grid.
    spatial_merge_size (`int`, *optional*, defaults to 2):
        Side length of the square patch-merge window applied before the multimodal projector.
    rope_theta (`float`, *optional*, defaults to 10000.0):
        Base period of the 2D rotary position embedding.
    """

    model_type = "locateanything_vision"
    base_config_key = "vision_config"

    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_channels: int = 3
    patch_size: int = 14
    init_pos_emb_height: int = 64
    init_pos_emb_width: int = 64
    spatial_merge_size: int = 2
    rope_theta: float = 10000.0
    layer_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    initializer_range: float = 0.02


@auto_docstring
@strict
class LocateAnythingConfig(PreTrainedConfig):
    r"""
    image_token_id (`int`, *optional*, defaults to 151665):
        Token id used as the placeholder for image patch embeddings.
    image_start_token_id (`int`, *optional*, defaults to 151666):
        Token id that opens an image span (`<img>`).
    image_end_token_id (`int`, *optional*, defaults to 151667):
        Token id that closes an image span (`</img>`).
    box_start_token_id (`int`, *optional*, defaults to 151668):
        Token id that opens a bounding-box span (`<box>`).
    box_end_token_id (`int`, *optional*, defaults to 151669):
        Token id that closes a bounding-box span (`</box>`).
    coord_start_token_id (`int`, *optional*, defaults to 151677):
        First token id of the quantized coordinate vocabulary.
    coord_end_token_id (`int`, *optional*, defaults to 152677):
        Last token id of the quantized coordinate vocabulary.
    ref_start_token_id (`int`, *optional*, defaults to 151672):
        Token id that opens a referring-expression span (`<ref>`).
    ref_end_token_id (`int`, *optional*, defaults to 151673):
        Token id that closes a referring-expression span (`</ref>`).
    none_token_id (`int`, *optional*, defaults to 4064):
        Token id emitted for an empty box.
    """

    model_type = "locateanything"
    sub_configs = {"vision_config": LocateAnythingVisionConfig, "text_config": AutoConfig}

    image_token_id: int = 151665
    image_start_token_id: int = 151666
    image_end_token_id: int = 151667
    box_start_token_id: int = 151668
    box_end_token_id: int = 151669
    coord_start_token_id: int = 151677
    coord_end_token_id: int = 152677
    ref_start_token_id: int = 151672
    ref_end_token_id: int = 151673
    none_token_id: int = 4064
    tie_word_embeddings: bool = True

    vision_config: dict | LocateAnythingVisionConfig | None = None
    text_config: dict | PreTrainedConfig | None = None

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(self.text_config, dict):
            self.text_config = CONFIG_MAPPING[self.text_config.get("model_type", "qwen2")](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["qwen2"]()

        super().__post_init__(**kwargs)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    dropout: float = 0.0,
    **kwargs,
):
    if scaling is None:
        scaling = query.size(-1) ** -0.5
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the 2D complex rotary embedding to packed query/key tensors of shape `(seq, num_heads, head_dim)`."""
    freqs_cis = freqs_cis.unsqueeze(-2)
    xq_ = torch.view_as_complex(xq.float().view(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().view(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class LocateAnythingVisionRotaryEmbedding(nn.Module):
    """2D rotary position embedding producing per-position `freqs_cis` (complex) for the packed grid."""

    def __init__(self, config: LocateAnythingVisionConfig):
        super().__init__()
        self.dim = config.hidden_size // config.num_attention_heads
        self.theta_base = config.rope_theta

    def forward(self, grid_hws: torch.Tensor) -> torch.Tensor:
        device = grid_hws.device
        dim_range = torch.arange(0, self.dim, 4, device=device)[: self.dim // 4].float()
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
        all_freqs_cis = []
        for height, width in grid_hws.tolist():
            y_pos = torch.arange(height, device=device).float()
            x_pos = torch.arange(width, device=device).float()
            x_freqs = torch.outer(x_pos, freqs)
            y_freqs = torch.outer(y_pos, freqs)
            x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
            y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
            grid_x = x_cis[None, :, :].expand(height, width, -1)
            grid_y = y_cis[:, None, :].expand(height, width, -1)
            freqs_cis = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], dim=-1)
            freqs_cis = freqs_cis.reshape(height * width, self.dim // 2)
            all_freqs_cis.append(freqs_cis)
        return torch.cat(all_freqs_cis, dim=0)


class LocateAnythingLearnable2DInterpPosEmb(nn.Module):
    def __init__(self, config: LocateAnythingVisionConfig):
        super().__init__()
        self.height = config.init_pos_emb_height
        self.width = config.init_pos_emb_width
        self.weight = nn.Parameter(torch.empty(self.height, self.width, config.hidden_size))

    def forward(self, hidden_states: torch.Tensor, grid_hws: torch.Tensor) -> torch.Tensor:
        pos_embeds = []
        for height, width in grid_hws.tolist():
            if (height, width) == (self.height, self.width):
                pos_embeds.append(self.weight.flatten(end_dim=1))
            else:
                interpolated = F.interpolate(
                    self.weight.permute(2, 0, 1).unsqueeze(0),
                    size=(height, width),
                    mode="bicubic",
                )
                pos_embeds.append(interpolated.squeeze(0).permute(1, 2, 0).flatten(end_dim=1))
        return hidden_states + torch.cat(pos_embeds)


class LocateAnythingPatchEmbed(nn.Module):
    def __init__(self, config: LocateAnythingVisionConfig):
        super().__init__()
        self.proj = nn.Conv2d(
            config.num_channels, config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size
        )
        self.pos_emb = LocateAnythingLearnable2DInterpPosEmb(config)

    def forward(self, pixel_values: torch.Tensor, grid_hws: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(pixel_values).view(pixel_values.size(0), -1)
        hidden_states = self.pos_emb(hidden_states, grid_hws)
        return hidden_states


class LocateAnythingVisionMLP(nn.Module):
    def __init__(self, config: LocateAnythingVisionConfig):
        super().__init__()
        self.fc0 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc1 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = ACT2FN["gelu_pytorch_tanh"]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc1(self.activation(self.fc0(hidden_states)))


class LocateAnythingVisionAttention(nn.Module):
    def __init__(self, config: LocateAnythingVisionConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.wqkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=True)
        self.wo = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        rope_freqs_cis: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        qkv = self.wqkv(hidden_states).view(seq_length, 3, self.num_heads, self.head_dim)
        query_states, key_states, value_states = qkv.unbind(dim=1)
        query_states, key_states = apply_rope(query_states, key_states, rope_freqs_cis)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            is_causal=False,
            **kwargs,
        )
        attn_output = attn_output.reshape(seq_length, -1)
        return self.wo(attn_output)


class LocateAnythingVisionLayer(nn.Module):
    def __init__(self, config: LocateAnythingVisionConfig):
        super().__init__()
        self.norm0 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = LocateAnythingVisionAttention(config)
        self.mlp = LocateAnythingVisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        rope_freqs_cis: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm0(hidden_states), attention_mask, rope_freqs_cis, **kwargs)
        hidden_states = hidden_states + self.mlp(self.norm1(hidden_states))
        return hidden_states


class LocateAnythingVisionEncoder(nn.Module):
    def __init__(self, config: LocateAnythingVisionConfig):
        super().__init__()
        self.config = config
        self.rotary_emb = LocateAnythingVisionRotaryEmbedding(config)
        self.blocks = nn.ModuleList([LocateAnythingVisionLayer(config) for _ in range(config.num_hidden_layers)])
        self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self, hidden_states: torch.Tensor, grid_hws: torch.Tensor, **kwargs: Unpack[TransformersKwargs]
    ) -> torch.Tensor:
        rope_freqs_cis = self.rotary_emb(grid_hws)
        seq_length = hidden_states.shape[0]
        lengths = (grid_hws[:, 0] * grid_hws[:, 1]).tolist()
        attention_mask = torch.full(
            (1, 1, seq_length, seq_length),
            torch.finfo(hidden_states.dtype).min,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        start = 0
        for length in lengths:
            attention_mask[..., start : start + length, start : start + length] = 0.0
            start += length

        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask, rope_freqs_cis, **kwargs)

        return self.final_layernorm(hidden_states)


class LocateAnythingMultiModalProjector(nn.Module):
    def __init__(self, config: LocateAnythingConfig):
        super().__init__()
        self.merge_size = config.vision_config.spatial_merge_size
        merged_dim = config.vision_config.hidden_size * self.merge_size * self.merge_size
        self.pre_norm = nn.LayerNorm(merged_dim)
        self.linear_1 = nn.Linear(merged_dim, config.text_config.hidden_size)
        self.act = ACT2FN["gelu"]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size)

    def forward(self, image_features: torch.Tensor, grid_hws: torch.Tensor) -> torch.Tensor:
        merge = self.merge_size
        dim = image_features.shape[-1]
        chunks = image_features.split((grid_hws[:, 0] * grid_hws[:, 1]).tolist(), dim=0)
        outputs = []
        for chunk, (height, width) in zip(chunks, grid_hws.tolist()):
            new_height, new_width = height // merge, width // merge
            chunk = chunk.view(new_height, merge, new_width, merge, dim).permute(0, 2, 1, 3, 4)
            chunk = chunk.reshape(new_height * new_width, merge * merge * dim)
            chunk = self.linear_2(self.act(self.linear_1(self.pre_norm(chunk))))
            outputs.append(chunk)
        return torch.cat(outputs, dim=0)


@auto_docstring
class LocateAnythingPreTrainedModel(PreTrainedModel):
    config: LocateAnythingConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LocateAnythingVisionLayer", "Qwen2DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, LocateAnythingLearnable2DInterpPosEmb):
            init.normal_(module.weight)


class LocateAnythingVisionModel(LocateAnythingPreTrainedModel):
    config: LocateAnythingVisionConfig
    main_input_name = "pixel_values"
    input_modalities = "image"

    def __init__(self, config: LocateAnythingVisionConfig):
        super().__init__(config)
        self.patch_embed = LocateAnythingPatchEmbed(config)
        self.encoder = LocateAnythingVisionEncoder(config)
        self.post_init()

    @auto_docstring
    def forward(
        self, pixel_values: torch.FloatTensor, grid_hws: torch.LongTensor, **kwargs: Unpack[TransformersKwargs]
    ) -> BaseModelOutput:
        r"""
        grid_hws (`torch.LongTensor` of shape `(num_images, 2)`):
            Patch-grid height and width of each image, used to build the packed attention mask and rotary positions.
        """
        hidden_states = self.patch_embed(pixel_values, grid_hws)
        hidden_states = self.encoder(hidden_states, grid_hws, **kwargs)
        return BaseModelOutput(last_hidden_state=hidden_states)


class LocateAnythingModelOutputWithPast(LlavaModelOutputWithPast):
    pass


class LocateAnythingCausalLMOutputWithPast(LlavaCausalLMOutputWithPast):
    pass


class LocateAnythingModel(LlavaModel):
    def __init__(self, config: LocateAnythingConfig):
        super().__init__(config)
        self.vision_tower = LocateAnythingVisionModel(config.vision_config)
        self.multi_modal_projector = LocateAnythingMultiModalProjector(config)
        self.language_model = AutoModel.from_config(config.text_config)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_hws: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        image_grid_hws (`torch.LongTensor` of shape `(num_images, 2)`):
            Patch-grid height and width of each image.
        """
        vision_outputs: BaseModelOutput = self.vision_tower(
            pixel_values=pixel_values.to(self.vision_tower.dtype), grid_hws=image_grid_hws, **kwargs
        )
        image_features = self.multi_modal_projector(vision_outputs.last_hidden_state, image_grid_hws)
        return BaseModelOutputWithPooling(
            last_hidden_state=vision_outputs.last_hidden_state, pooler_output=image_features
        )

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_grid_hws: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> LocateAnythingModelOutputWithPast:
        r"""
        image_grid_hws (`torch.LongTensor` of shape `(num_images, 2)`):
            Patch-grid height and width of each image.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_features = None
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values, image_grid_hws).pooler_output
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            special_image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        outputs: BaseModelOutputWithPast = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        return LocateAnythingModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features,
        )


class LocateAnythingForConditionalGeneration(LlavaForConditionalGeneration):
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_grid_hws: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> LocateAnythingCausalLMOutputWithPast:
        r"""
        image_grid_hws (`torch.LongTensor` of shape `(num_images, 2)`):
            Patch-grid height and width of each image.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for the language-modeling loss; indices in `[0, ..., config.text_config.vocab_size]` or -100.
        """
        outputs: LocateAnythingModelOutputWithPast = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_hws=image_grid_hws,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return LocateAnythingCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )


__all__ = [
    "LocateAnythingVisionConfig",
    "LocateAnythingConfig",
    "LocateAnythingPreTrainedModel",
    "LocateAnythingVisionModel",
    "LocateAnythingModel",
    "LocateAnythingForConditionalGeneration",
]
