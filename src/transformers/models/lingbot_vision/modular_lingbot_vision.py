# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch LingBot-Vision model."""

import math
from collections.abc import Callable
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from ... import initialization as init
from ...backbone_utils import BackboneMixin, filter_output_hidden_states
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BackboneOutput, BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import can_return_tuple, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..llama.modeling_llama import LlamaRMSNorm
from ..pixtral.modeling_pixtral import rotate_half
from ..swin.modeling_swin import SwinDropPath
from .configuration_lingbot_vision import LingbotVisionConfig


_ROPE_DTYPE_MAPPING = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _make_2tuple(value):
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError("Expected a sequence of length 2.")
        return tuple(value)
    return (value, value)


class LingbotVisionRMSNorm(LlamaRMSNorm):
    # Identical to LlamaRMSNorm, but LingBot-Vision builds RMSNorm layers with an epsilon of 1e-5.
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__(hidden_size, eps)


class LingbotVisionLayerScale(nn.Module):
    def __init__(self, hidden_size: int, init_value: float):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((hidden_size,), init_value))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * self.gamma


class LingbotVisionPatchEmbeddings(nn.Module):
    def __init__(self, config: LingbotVisionConfig):
        super().__init__()
        image_size = _make_2tuple(config.image_size)
        patch_size = _make_2tuple(config.patch_size)

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = config.num_channels
        self.num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        self.projection = nn.Conv2d(config.num_channels, config.hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        target_dtype = self.projection.weight.dtype
        embeddings = self.projection(pixel_values.to(dtype=target_dtype))
        height, width = embeddings.shape[-2:]
        embeddings = embeddings.flatten(2).transpose(1, 2)
        return embeddings, (height, width)


class LingbotVisionRotaryEmbedding(nn.Module):
    def __init__(self, config: LingbotVisionConfig):
        super().__init__()
        if config.hidden_size % (4 * config.num_attention_heads) != 0:
            raise ValueError("`hidden_size` must be divisible by `4 * num_attention_heads` for LingBot-Vision RoPE.")

        rope_parameters = config.rope_parameters or {}
        base = rope_parameters.get("rope_theta")
        both_periods = config.rope_min_period is not None and config.rope_max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError(
                "Either `rope_parameters['rope_theta']` or both `rope_min_period` and `rope_max_period` must be set."
            )

        head_dim = config.hidden_size // config.num_attention_heads
        self.base = base
        self.min_period = config.rope_min_period
        self.max_period = config.rope_max_period
        self.head_dim = head_dim
        self.normalize_coords = config.rope_normalize_coords
        self.shift_coords = config.rope_shift_coords
        self.jitter_coords = config.rope_jitter_coords
        self.rescale_coords = config.rope_rescale_coords
        self.dtype = _ROPE_DTYPE_MAPPING[config.rope_dtype]
        self.register_buffer("periods", torch.empty(head_dim // 4, dtype=self.dtype), persistent=True)
        self._init_weights()

    def _init_weights(self):
        device = self.periods.device
        dtype = self.dtype
        if self.base is not None:
            periods = self.base ** (
                2 * torch.arange(self.head_dim // 4, device=device, dtype=dtype) / (self.head_dim // 2)
            )
        else:
            base = self.max_period / self.min_period
            exponents = torch.linspace(0, 1, self.head_dim // 4, device=device, dtype=dtype)
            periods = base**exponents
            periods = periods / base
            periods = periods * self.max_period
        self.periods.data = periods

    def forward(self, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.periods.device
        dtype = self.dtype
        kwargs = {"device": device, "dtype": dtype}
        if self.normalize_coords == "max":
            max_hw = max(height, width)
            coords_h = torch.arange(0.5, height, **kwargs) / max_hw
            coords_w = torch.arange(0.5, width, **kwargs) / max_hw
        elif self.normalize_coords == "min":
            min_hw = min(height, width)
            coords_h = torch.arange(0.5, height, **kwargs) / min_hw
            coords_w = torch.arange(0.5, width, **kwargs) / min_hw
        elif self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, height, **kwargs) / height
            coords_w = torch.arange(0.5, width, **kwargs) / width
        else:
            raise ValueError(f"Unknown RoPE coordinate normalization: {self.normalize_coords}")

        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1).flatten(0, 1)
        coords = 2.0 * coords - 1.0

        if self.training and self.shift_coords is not None:
            shift = torch.empty(2, **kwargs).uniform_(-self.shift_coords, self.shift_coords)
            coords = coords + shift[None, :]
        if self.training and self.jitter_coords is not None:
            jitter_max = math.log(self.jitter_coords)
            jitter = torch.empty(2, **kwargs).uniform_(-jitter_max, jitter_max).exp()
            coords = coords * jitter[None, :]
        if self.training and self.rescale_coords is not None:
            rescale_max = math.log(self.rescale_coords)
            rescale = torch.empty(1, **kwargs).uniform_(-rescale_max, rescale_max).exp()
            coords = coords * rescale

        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2).tile(2)
        return torch.sin(angles), torch.cos(angles)


def _apply_rope(hidden_states: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    return (hidden_states * cos) + (rotate_half(hidden_states) * sin)


class LingbotVisionLinearKMaskedBias(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.bias is not None:
            self.register_buffer("bias_mask", torch.ones_like(self.bias), persistent=True)
            self.bias_mask[self.out_features // 3 : 2 * self.out_features // 3] = 0

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bias = self.bias * self.bias_mask.to(self.bias.dtype) if self.bias is not None else None
        return F.linear(input, self.weight, bias)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple[torch.Tensor, torch.Tensor]:
    if scaling is None:
        scaling = query.size(-1) ** -0.5

    attention_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attention_weights = attention_weights + attention_mask
    attention_weights = nn.functional.softmax(attention_weights, dim=-1)
    attention_weights = nn.functional.dropout(attention_weights, p=dropout, training=module.training)
    attention_output = torch.matmul(attention_weights, value).transpose(1, 2).contiguous()
    return attention_output, attention_weights


class LingbotVisionAttention(nn.Module):
    def __init__(self, config: LingbotVisionConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = False
        linear_cls = LingbotVisionLinearKMaskedBias if config.mask_k_bias else nn.Linear
        self.qkv = linear_cls(config.hidden_size, config.hidden_size * 3, bias=config.qkv_bias)
        self.attention_dropout = config.attention_probs_dropout_prob
        self.projection = nn.Linear(config.hidden_size, config.hidden_size, bias=config.proj_bias)
        self.projection_dropout = nn.Dropout(config.hidden_dropout_prob)

    def apply_rope(
        self, query: torch.Tensor, key: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sin, cos = rope
        query_dtype = query.dtype
        key_dtype = key.dtype
        rope_dtype = sin.dtype
        query = query.to(dtype=rope_dtype)
        key = key.to(dtype=rope_dtype)

        prefix_length = query.shape[-2] - sin.shape[-2]
        if prefix_length < 0:
            raise ValueError("RoPE table is longer than the token sequence.")

        query_prefix = query[:, :, :prefix_length, :]
        key_prefix = key[:, :, :prefix_length, :]
        query = _apply_rope(query[:, :, prefix_length:, :], sin, cos)
        key = _apply_rope(key[:, :, prefix_length:, :], sin, cos)
        query = torch.cat((query_prefix, query), dim=-2).to(dtype=query_dtype)
        key = torch.cat((key_prefix, key), dim=-2).to(dtype=key_dtype)
        return query, key

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_length, hidden_size = hidden_states.shape
        qkv = self.qkv(hidden_states)
        qkv = qkv.reshape(batch_size, seq_length, 3, self.num_heads, self.head_dim)
        query, key, value = torch.unbind(qkv, dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        query, key = self.apply_rope(query, key, rope)

        attention_interface: Callable = (
            eager_attention_forward
            if kwargs.get("output_attentions", False)
            else ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, eager_attention_forward)
        )
        context_layer, attention_probs = attention_interface(
            self,
            query,
            key,
            value,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            **kwargs,
        )

        context_layer = context_layer.reshape(batch_size, seq_length, hidden_size).contiguous()
        context_layer = self.projection(context_layer)
        context_layer = self.projection_dropout(context_layer)
        return context_layer, attention_probs


class LingbotVisionMlp(nn.Module):
    def __init__(self, config: LingbotVisionConfig):
        super().__init__()
        hidden_features = int(config.hidden_size * config.mlp_ratio)
        self.fc1 = nn.Linear(config.hidden_size, hidden_features, bias=config.ffn_bias)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, config.hidden_size, bias=config.ffn_bias)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class LingbotVisionSwiGLUFFN(nn.Module):
    def __init__(self, config: LingbotVisionConfig, align_to: int = 8):
        super().__init__()
        hidden_features = int(config.hidden_size * config.mlp_ratio)
        swiglu_hidden_features = int(hidden_features * 2 / 3)
        swiglu_hidden_features += -swiglu_hidden_features % align_to
        self.w1 = nn.Linear(config.hidden_size, swiglu_hidden_features, bias=config.ffn_bias)
        self.w2 = nn.Linear(config.hidden_size, swiglu_hidden_features, bias=config.ffn_bias)
        self.w3 = nn.Linear(swiglu_hidden_features, config.hidden_size, bias=config.ffn_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(hidden_states)) * self.w2(hidden_states))


_FFN_MAPPING = {
    "mlp": LingbotVisionMlp,
    "swiglu": LingbotVisionSwiGLUFFN,
    "swiglu32": partial(LingbotVisionSwiGLUFFN, align_to=32),
    "swiglu64": partial(LingbotVisionSwiGLUFFN, align_to=64),
    "swiglu128": partial(LingbotVisionSwiGLUFFN, align_to=128),
}


def _get_norm(config: LingbotVisionConfig) -> nn.Module:
    if config.norm_layer == "layernorm":
        return nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    if config.norm_layer == "layernormbf16":
        return nn.LayerNorm(config.hidden_size, eps=1e-5)
    if config.norm_layer == "rmsnorm":
        return LingbotVisionRMSNorm(config.hidden_size)
    raise ValueError(f"Unknown LingBot-Vision norm layer: {config.norm_layer}")


class LingbotVisionDropPath(SwinDropPath):
    pass


class LingbotVisionLayer(GradientCheckpointingLayer):
    def __init__(self, config: LingbotVisionConfig):
        super().__init__()
        self.norm1 = _get_norm(config)
        self.attention = LingbotVisionAttention(config)
        self.layer_scale1 = (
            LingbotVisionLayerScale(config.hidden_size, config.layer_scale_init_value)
            if config.layer_scale_init_value is not None
            else nn.Identity()
        )
        self.drop_path = LingbotVisionDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = _get_norm(config)
        self.mlp = _FFN_MAPPING[config.ffn_layer](config)
        self.layer_scale2 = (
            LingbotVisionLayerScale(config.hidden_size, config.layer_scale_init_value)
            if config.layer_scale_init_value is not None
            else nn.Identity()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        attention_output, _ = self.attention(self.norm1(hidden_states), rope, **kwargs)
        hidden_states = residual + self.drop_path(self.layer_scale1(attention_output))

        residual = hidden_states
        hidden_states = residual + self.drop_path(self.layer_scale2(self.mlp(self.norm2(hidden_states))))
        return hidden_states


class LingbotVisionEmbeddings(nn.Module):
    def __init__(self, config: LingbotVisionConfig):
        super().__init__()
        self.patch_embeddings = LingbotVisionPatchEmbeddings(config)
        self.cls_token = nn.Parameter(torch.empty(1, 1, config.hidden_size))
        self.num_storage_tokens = config.num_storage_tokens
        if self.num_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, self.num_storage_tokens, config.hidden_size))
        self.mask_token = nn.Parameter(torch.empty(1, config.hidden_size))

    def forward(
        self, pixel_values: torch.Tensor, bool_masked_pos: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        embeddings, patch_grid = self.patch_embeddings(pixel_values)
        batch_size = embeddings.shape[0]

        if bool_masked_pos is not None:
            embeddings = torch.where(
                bool_masked_pos.unsqueeze(-1), self.mask_token.to(embeddings.dtype).unsqueeze(0), embeddings
            )
            cls_token = self.cls_token
        else:
            cls_token = self.cls_token + 0 * self.mask_token

        if self.num_storage_tokens > 0:
            storage_tokens = self.storage_tokens.expand(batch_size, -1, -1)
            embeddings = torch.cat((cls_token.expand(batch_size, -1, -1), storage_tokens, embeddings), dim=1)
        else:
            embeddings = torch.cat((cls_token.expand(batch_size, -1, -1), embeddings), dim=1)
        return embeddings, patch_grid


@auto_docstring
class LingbotVisionPreTrainedModel(PreTrainedModel):
    config: LingbotVisionConfig
    base_model_prefix = "lingbot_vision"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LingbotVisionLayer"]
    _supports_sdpa = True
    _can_record_outputs = {
        "hidden_states": LingbotVisionLayer,
        "attentions": LingbotVisionAttention,
    }

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            init.trunc_normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                init.zeros_(module.bias)
            if hasattr(module, "bias_mask") and module.bias_mask is not None:
                init.ones_(module.bias_mask)
                init.zeros_(module.bias_mask[module.out_features // 3 : 2 * module.out_features // 3])
        elif isinstance(module, (nn.LayerNorm, LingbotVisionRMSNorm)):
            init.ones_(module.weight)
            if getattr(module, "bias", None) is not None:
                init.zeros_(module.bias)
        elif isinstance(module, LingbotVisionLayerScale):
            init.constant_(module.gamma, self.config.layer_scale_init_value)
        elif isinstance(module, LingbotVisionEmbeddings):
            init.normal_(module.cls_token, std=self.config.initializer_range)
            if module.num_storage_tokens > 0:
                init.normal_(module.storage_tokens, std=self.config.initializer_range)
            init.zeros_(module.mask_token)
        elif isinstance(module, LingbotVisionRotaryEmbedding):
            # The `periods` buffer is deterministic; recompute it so the module can be
            # re-initialized from scratch (e.g. when instantiated on the meta device).
            module._init_weights()


class LingbotVisionEncoder(LingbotVisionPreTrainedModel):
    def __init__(self, config: LingbotVisionConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([LingbotVisionLayer(config) for _ in range(config.num_hidden_layers)])
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    def forward(
        self,
        hidden_states: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        for layer in self.layers:
            hidden_states = layer(hidden_states, rope, **kwargs)

        return BaseModelOutput(last_hidden_state=hidden_states)


@auto_docstring
class LingbotVisionModel(LingbotVisionPreTrainedModel):
    def __init__(self, config: LingbotVisionConfig):
        super().__init__(config)
        self.embeddings = LingbotVisionEmbeddings(config)
        self.rope_embeddings = LingbotVisionRotaryEmbedding(config)
        self.encoder = LingbotVisionEncoder(config)
        self.layernorm = _get_norm(config)
        self.cls_norm = _get_norm(config) if config.untie_cls_and_patch_norms else None
        self.post_init()

    def get_input_embeddings(self) -> LingbotVisionPatchEmbeddings:
        return self.embeddings.patch_embeddings

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches should be replaced by the mask token.
        """
        embedding_output, patch_grid = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)
        rope = self.rope_embeddings(*patch_grid)
        encoder_outputs = self.encoder(embedding_output, rope=rope, **kwargs)
        sequence_output = encoder_outputs.last_hidden_state

        if self.config.untie_cls_and_patch_norms:
            prefix_length = self.config.num_storage_tokens + 1
            cls_and_storage = self.cls_norm(sequence_output[:, :prefix_length])
            patch_tokens = self.layernorm(sequence_output[:, prefix_length:])
            sequence_output = torch.cat((cls_and_storage, patch_tokens), dim=1)
        else:
            sequence_output = self.layernorm(sequence_output)

        pooled_output = sequence_output[:, 0]
        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@auto_docstring(custom_intro="LingBot-Vision backbone, to be used with dense prediction frameworks.")
class LingbotVisionBackbone(BackboneMixin, LingbotVisionPreTrainedModel):
    def __init__(self, config: LingbotVisionConfig):
        super().__init__(config)
        self.num_features = [config.hidden_size for _ in range(config.num_hidden_layers + 1)]
        self.embeddings = LingbotVisionEmbeddings(config)
        self.rope_embeddings = LingbotVisionRotaryEmbedding(config)
        self.encoder = LingbotVisionEncoder(config)
        self.layernorm = _get_norm(config)
        self.cls_norm = _get_norm(config) if config.untie_cls_and_patch_norms else None
        self.post_init()

    def get_input_embeddings(self) -> LingbotVisionPatchEmbeddings:
        return self.embeddings.patch_embeddings

    @can_return_tuple
    @filter_output_hidden_states
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BackboneOutput:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches should be replaced by the mask token.
        """
        embedding_output, patch_grid = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)
        rope = self.rope_embeddings(*patch_grid)
        kwargs["output_hidden_states"] = True
        encoder_outputs = self.encoder(embedding_output, rope=rope, **kwargs)
        hidden_states = encoder_outputs.hidden_states

        feature_maps = []
        prefix_length = self.config.num_storage_tokens + 1
        batch_size, _, height, width = pixel_values.shape
        patch_height = height // self.config.patch_size
        patch_width = width // self.config.patch_size
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage not in self.out_features:
                continue

            if self.config.apply_layernorm:
                if self.config.untie_cls_and_patch_norms:
                    hidden_state = torch.cat(
                        (
                            self.cls_norm(hidden_state[:, :prefix_length]),
                            self.layernorm(hidden_state[:, prefix_length:]),
                        ),
                        dim=1,
                    )
                else:
                    hidden_state = self.layernorm(hidden_state)
            if self.config.reshape_hidden_states:
                hidden_state = hidden_state[:, prefix_length:]
                hidden_state = hidden_state.reshape(batch_size, patch_height, patch_width, -1)
                hidden_state = hidden_state.permute(0, 3, 1, 2).contiguous()
            feature_maps.append(hidden_state)

        return BackboneOutput(
            feature_maps=tuple(feature_maps), hidden_states=hidden_states, attentions=encoder_outputs.attentions
        )


__all__ = [
    "LingbotVisionBackbone",
    "LingbotVisionModel",
    "LingbotVisionPreTrainedModel",
]
