# Copyright 2025 The HuggingFace Team. All rights reserved.
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
from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch
from torch import nn

from ...activations import ACT2CLS, ACT2FN
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BackboneOutput
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    ModelOutput,
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    torch_int,
)
from ...utils.generic import check_model_inputs
from .configuration_efficientloftr import EfficientLoFTRConfig


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for outputs of keypoint matching models. Due to the nature of keypoint detection and matching, the number
    of keypoints is not fixed and can vary from image to image, which makes batching non-trivial. In the batch of
    images, the maximum number of matches is set as the dimension of the matches and matching scores. The mask tensor is
    used to indicate which values in the keypoints, matches and matching_scores tensors are keypoint matching
    information.
    """
)
class KeypointMatchingOutput(ModelOutput):
    r"""
    matches (`torch.FloatTensor` of shape `(batch_size, 2, num_matches)`):
        Index of keypoint matched in the other image.
    matching_scores (`torch.FloatTensor` of shape `(batch_size, 2, num_matches)`):
        Scores of predicted matches.
    keypoints (`torch.FloatTensor` of shape `(batch_size, num_keypoints, 2)`):
        Absolute (x, y) coordinates of predicted keypoints in a given image.
    hidden_states (`tuple[torch.FloatTensor, ...]`, *optional*):
        Tuple of `torch.FloatTensor` (one for the output of each stage) of shape `(batch_size, 2, num_channels,
        num_keypoints)`, returned when `output_hidden_states=True` is passed or when
        `config.output_hidden_states=True`)
    attentions (`tuple[torch.FloatTensor, ...]`, *optional*):
        Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, 2, num_heads, num_keypoints,
        num_keypoints)`, returned when `output_attentions=True` is passed or when `config.output_attentions=True`)
    """

    matches: Optional[torch.FloatTensor] = None
    matching_scores: Optional[torch.FloatTensor] = None
    keypoints: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None


class EfficientLoFTRRotaryEmbedding(nn.Module):
    def __init__(self, config: EfficientLoFTRConfig, device=None):
        super().__init__()
        self.config = config
        self.rope_type = config.rope_scaling["rope_type"]
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, _ = self.rope_init_fn(self.config, device)
        inv_freq_expanded = inv_freq[None, None, None, :].float().expand(1, 1, 1, -1)

        embed_height, embed_width = config.embedding_size
        i_indices = torch.ones(embed_height, embed_width).cumsum(0).float().unsqueeze(-1)
        j_indices = torch.ones(embed_height, embed_width).cumsum(1).float().unsqueeze(-1)

        emb = torch.zeros(1, embed_height, embed_width, self.config.hidden_size // 2)
        emb[:, :, :, 0::2] = i_indices * inv_freq_expanded
        emb[:, :, :, 1::2] = j_indices * inv_freq_expanded

        self.register_buffer("inv_freq", emb, persistent=False)

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: Optional[tuple[torch.LongTensor, torch.LongTensor]] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            emb = self.inv_freq
            sin = emb.sin()
            cos = emb.cos()

        sin = sin.repeat_interleave(2, dim=-1)
        cos = cos.repeat_interleave(2, dim=-1)

        sin = sin.to(device=x.device, dtype=x.dtype)
        cos = cos.to(device=x.device, dtype=x.dtype)

        return cos, sin


# Copied from transformers.models.rt_detr_v2.modeling_rt_detr_v2.RTDetrV2ConvNormLayer with RTDetrV2->EfficientLoFTR
class EfficientLoFTRConvNormLayer(nn.Module):
    def __init__(self, config, in_channels, out_channels, kernel_size, stride, padding=None, activation=None):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels, config.batch_norm_eps)
        self.activation = nn.Identity() if activation is None else ACT2CLS[activation]()

    def forward(self, hidden_state):
        hidden_state = self.conv(hidden_state)
        hidden_state = self.norm(hidden_state)
        hidden_state = self.activation(hidden_state)
        return hidden_state


class EfficientLoFTRRepVGGBlock(GradientCheckpointingLayer):
    """
    RepVGG architecture block introduced by the work "RepVGG: Making VGG-style ConvNets Great Again".
    """

    def __init__(self, config: EfficientLoFTRConfig, stage_idx: int, block_idx: int):
        super().__init__()
        in_channels = config.stage_block_in_channels[stage_idx][block_idx]
        out_channels = config.stage_block_out_channels[stage_idx][block_idx]
        stride = config.stage_block_stride[stage_idx][block_idx]
        activation = config.activation_function
        self.conv1 = EfficientLoFTRConvNormLayer(
            config, in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.conv2 = EfficientLoFTRConvNormLayer(
            config, in_channels, out_channels, kernel_size=1, stride=stride, padding=0
        )
        self.identity = nn.BatchNorm2d(in_channels) if in_channels == out_channels and stride == 1 else None
        self.activation = nn.Identity() if activation is None else ACT2FN[activation]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.identity is not None:
            identity_out = self.identity(hidden_states)
        else:
            identity_out = 0
        hidden_states = self.conv1(hidden_states) + self.conv2(hidden_states) + identity_out
        hidden_states = self.activation(hidden_states)
        return hidden_states


class EfficientLoFTRRepVGGStage(nn.Module):
    def __init__(self, config: EfficientLoFTRConfig, stage_idx: int):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for block_idx in range(config.stage_num_blocks[stage_idx]):
            self.blocks.append(
                EfficientLoFTRRepVGGBlock(
                    config,
                    stage_idx,
                    block_idx,
                )
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return hidden_states


class EfficientLoFTRepVGG(nn.Module):
    def __init__(self, config: EfficientLoFTRConfig):
        super().__init__()

        self.stages = nn.ModuleList([])

        for stage_idx in range(len(config.stage_stride)):
            stage = EfficientLoFTRRepVGGStage(config, stage_idx)
            self.stages.append(stage)

    def forward(self, hidden_states: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        for stage in self.stages:
            hidden_states = stage(hidden_states)
            outputs.append(hidden_states)

        # Exclude first stage in outputs
        outputs = outputs[1:]
        return outputs


class EfficientLoFTRAggregationLayer(nn.Module):
    def __init__(self, config: EfficientLoFTRConfig):
        super().__init__()

        hidden_size = config.hidden_size

        self.q_aggregation = nn.Conv2d(
            hidden_size,
            hidden_size,
            kernel_size=config.q_aggregation_kernel_size,
            padding=0,
            stride=config.q_aggregation_stride,
            bias=False,
            groups=hidden_size,
        )
        self.kv_aggregation = torch.nn.MaxPool2d(
            kernel_size=config.kv_aggregation_kernel_size, stride=config.kv_aggregation_stride
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_states = hidden_states
        is_cross_attention = encoder_hidden_states is not None
        kv_states = encoder_hidden_states if is_cross_attention else hidden_states

        query_states = self.q_aggregation(query_states)
        kv_states = self.kv_aggregation(kv_states)
        query_states = query_states.permute(0, 2, 3, 1)
        kv_states = kv_states.permute(0, 2, 3, 1)
        hidden_states = self.norm(query_states)
        encoder_hidden_states = self.norm(kv_states)
        return hidden_states, encoder_hidden_states


# Copied from transformers.models.cohere.modeling_cohere.rotate_half
def rotate_half(x):
    # Split and rotate. Note that this function is different from e.g. Llama.
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rot_x = torch.stack([-x2, x1], dim=-1).flatten(-2)
    return rot_x


# Copied from transformers.models.cohere.modeling_cohere.apply_rotary_pos_emb
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
    dtype = q.dtype
    q = q.float()
    k = k.float()
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(dtype=dtype), k_embed.to(dtype=dtype)


# Copied from transformers.models.cohere.modeling_cohere.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# Copied from transformers.models.llama.modeling_llama.eager_attention_forward
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class EfficientLoFTRAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # Copied from transformers.models.llama.modeling_llama.LlamaAttention.__init__ with Llama->EfficientLoFTR
    def __init__(self, config: EfficientLoFTRConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, dim = hidden_states.shape
        input_shape = hidden_states.shape[:-1]

        query_states = self.q_proj(hidden_states).view(batch_size, seq_len, -1, dim)

        is_cross_attention = encoder_hidden_states is not None
        current_states = encoder_hidden_states if is_cross_attention else hidden_states

        key_states = self.k_proj(current_states).view(batch_size, seq_len, -1, dim)
        value_states = self.v_proj(current_states).view(batch_size, seq_len, -1, self.head_dim).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=2)

        query_states = query_states.view(batch_size, seq_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, -1, self.head_dim).transpose(1, 2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=None,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class EfficientLoFTRMLP(nn.Module):
    def __init__(self, config: EfficientLoFTRConfig):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        self.fc1 = nn.Linear(hidden_size * 2, intermediate_size, bias=False)
        self.activation = ACT2FN[config.mlp_activation_function]
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class EfficientLoFTRAggregatedAttention(nn.Module):
    def __init__(self, config: EfficientLoFTRConfig, layer_idx: int):
        super().__init__()

        self.q_aggregation_kernel_size = config.q_aggregation_kernel_size
        self.aggregation = EfficientLoFTRAggregationLayer(config)
        self.attention = EfficientLoFTRAttention(config, layer_idx)
        self.mlp = EfficientLoFTRMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        batch_size, embed_dim, _, _ = hidden_states.shape

        # Aggregate features
        aggregated_hidden_states, aggregated_encoder_hidden_states = self.aggregation(
            hidden_states, encoder_hidden_states
        )
        _, aggregated_h, aggregated_w, _ = aggregated_hidden_states.shape

        # Multi-head attention
        aggregated_hidden_states = aggregated_hidden_states.reshape(batch_size, -1, embed_dim)
        aggregated_encoder_hidden_states = aggregated_encoder_hidden_states.reshape(batch_size, -1, embed_dim)
        attn_output, _ = self.attention(
            aggregated_hidden_states,
            aggregated_encoder_hidden_states,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        # Upsample features
        # (batch_size, seq_len, embed_dim) -> (batch_size, embed_dim, h, w) with seq_len = h * w
        attn_output = attn_output.permute(0, 2, 1)
        attn_output = attn_output.reshape(batch_size, embed_dim, aggregated_h, aggregated_w)
        attn_output = torch.nn.functional.interpolate(
            attn_output, scale_factor=self.q_aggregation_kernel_size, mode="bilinear", align_corners=False
        )
        intermediate_states = torch.cat([hidden_states, attn_output], dim=1)
        intermediate_states = intermediate_states.permute(0, 2, 3, 1)
        output_states = self.mlp(intermediate_states)
        output_states = output_states.permute(0, 3, 1, 2)

        hidden_states = hidden_states + output_states

        return hidden_states


class EfficientLoFTRLocalFeatureTransformerLayer(GradientCheckpointingLayer):
    def __init__(self, config: EfficientLoFTRConfig, layer_idx: int):
        super().__init__()

        self.self_attention = EfficientLoFTRAggregatedAttention(config, layer_idx)
        self.cross_attention = EfficientLoFTRAggregatedAttention(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        batch_size, _, embed_dim, height, width = hidden_states.shape

        hidden_states = hidden_states.reshape(-1, embed_dim, height, width)
        hidden_states = self.self_attention(hidden_states, position_embeddings=position_embeddings, **kwargs)

        encoder_hidden_states = hidden_states.reshape(-1, 2, embed_dim, height, width)
        encoder_hidden_states = encoder_hidden_states.flip(1)
        encoder_hidden_states = encoder_hidden_states.reshape(-1, embed_dim, height, width)

        hidden_states = self.cross_attention(hidden_states, encoder_hidden_states, **kwargs)
        hidden_states = hidden_states.reshape(batch_size, -1, embed_dim, height, width)

        return hidden_states


class EfficientLoFTRLocalFeatureTransformer(nn.Module):
    def __init__(self, config: EfficientLoFTRConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EfficientLoFTRLocalFeatureTransformerLayer(config, layer_idx=i)
                for i in range(config.num_attention_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings=position_embeddings, **kwargs)
        return hidden_states


class EfficientLoFTROutConvBlock(nn.Module):
    def __init__(self, config: EfficientLoFTRConfig, hidden_size: int, intermediate_size: int):
        super().__init__()

        self.out_conv1 = nn.Conv2d(hidden_size, intermediate_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.out_conv2 = nn.Conv2d(
            intermediate_size, intermediate_size, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.batch_norm = nn.BatchNorm2d(intermediate_size)
        self.activation = ACT2CLS[config.mlp_activation_function]()
        self.out_conv3 = nn.Conv2d(intermediate_size, hidden_size, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, hidden_states: torch.Tensor, residual_states: torch.Tensor) -> torch.Tensor:
        residual_states = self.out_conv1(residual_states)
        residual_states = residual_states + hidden_states
        residual_states = self.out_conv2(residual_states)
        residual_states = self.batch_norm(residual_states)
        residual_states = self.activation(residual_states)
        residual_states = self.out_conv3(residual_states)
        residual_states = nn.functional.interpolate(
            residual_states, scale_factor=2.0, mode="bilinear", align_corners=False
        )
        return residual_states


class EfficientLoFTRFineFusionLayer(nn.Module):
    def __init__(self, config: EfficientLoFTRConfig):
        super().__init__()

        self.fine_kernel_size = config.fine_kernel_size

        fine_fusion_dims = config.fine_fusion_dims
        self.out_conv = nn.Conv2d(
            fine_fusion_dims[0], fine_fusion_dims[0], kernel_size=1, stride=1, padding=0, bias=False
        )
        self.out_conv_layers = nn.ModuleList()
        for i in range(1, len(fine_fusion_dims)):
            out_conv = EfficientLoFTROutConvBlock(config, fine_fusion_dims[i], fine_fusion_dims[i - 1])
            self.out_conv_layers.append(out_conv)

    def forward_pyramid(
        self,
        hidden_states: torch.Tensor,
        residual_states: list[torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = self.out_conv(hidden_states)
        hidden_states = nn.functional.interpolate(
            hidden_states, scale_factor=2.0, mode="bilinear", align_corners=False
        )
        for i, layer in enumerate(self.out_conv_layers):
            hidden_states = layer(hidden_states, residual_states[i])

        return hidden_states

    def forward(
        self,
        coarse_features: torch.Tensor,
        residual_features: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        For each image pair, compute the fine features of pixels.
        In both images, compute a patch of fine features center cropped around each coarse pixel.
        In the first image, the feature patch is kernel_size large and long.
        In the second image, it is (kernel_size + 2) large and long.
        """
        batch_size, _, embed_dim, coarse_height, coarse_width = coarse_features.shape

        coarse_features = coarse_features.reshape(-1, embed_dim, coarse_height, coarse_width)
        residual_features = list(reversed(residual_features))

        # 1. Fine feature extraction
        fine_features = self.forward_pyramid(coarse_features, residual_features)
        _, fine_embed_dim, fine_height, fine_width = fine_features.shape

        fine_features = fine_features.reshape(batch_size, 2, fine_embed_dim, fine_height, fine_width)
        fine_features_0 = fine_features[:, 0]
        fine_features_1 = fine_features[:, 1]

        # 2. Unfold all local windows in crops
        stride = int(fine_height // coarse_height)
        fine_features_0 = nn.functional.unfold(
            fine_features_0, kernel_size=self.fine_kernel_size, stride=stride, padding=0
        )
        _, _, seq_len = fine_features_0.shape
        fine_features_0 = fine_features_0.reshape(batch_size, -1, self.fine_kernel_size**2, seq_len)
        fine_features_0 = fine_features_0.permute(0, 3, 2, 1)

        fine_features_1 = nn.functional.unfold(
            fine_features_1, kernel_size=self.fine_kernel_size + 2, stride=stride, padding=1
        )
        fine_features_1 = fine_features_1.reshape(batch_size, -1, (self.fine_kernel_size + 2) ** 2, seq_len)
        fine_features_1 = fine_features_1.permute(0, 3, 2, 1)

        return fine_features_0, fine_features_1


@auto_docstring
class EfficientLoFTRPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = EfficientLoFTRConfig
    base_model_prefix = "efficientloftr"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_record_outputs = {
        "hidden_states": EfficientLoFTRRepVGGBlock,
        "attentions": EfficientLoFTRAttention,
    }

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d, nn.BatchNorm2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # Copied from transformers.models.superpoint.modeling_superpoint.SuperPointPreTrainedModel.extract_one_channel_pixel_values with SuperPoint->EfficientLoFTR
    def extract_one_channel_pixel_values(self, pixel_values: torch.FloatTensor) -> torch.FloatTensor:
        """
        Assuming pixel_values has shape (batch_size, 3, height, width), and that all channels values are the same,
        extract the first channel value to get a tensor of shape (batch_size, 1, height, width) for EfficientLoFTR. This is
        a workaround for the issue discussed in :
        https://github.com/huggingface/transformers/pull/25786#issuecomment-1730176446

        Args:
            pixel_values: torch.FloatTensor of shape (batch_size, 3, height, width)

        Returns:
            pixel_values: torch.FloatTensor of shape (batch_size, 1, height, width)

        """
        return pixel_values[:, 0, :, :][:, None, :, :]


@auto_docstring(
    custom_intro="""
    EfficientLoFTR model taking images as inputs and outputting the features of the images.
    """
)
class EfficientLoFTRModel(EfficientLoFTRPreTrainedModel):
    def __init__(self, config: EfficientLoFTRConfig):
        super().__init__(config)

        self.config = config
        self.backbone = EfficientLoFTRepVGG(config)
        self.local_feature_transformer = EfficientLoFTRLocalFeatureTransformer(config)
        self.rotary_emb = EfficientLoFTRRotaryEmbedding(config=config)

        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BackboneOutput:
        r"""
        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AutoModel
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/assets/phototourism_sample_images/london_bridge_78916675_4568141288.jpg?raw=true"
        >>> image1 = Image.open(requests.get(url, stream=True).raw)
        >>> url = "https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/assets/phototourism_sample_images/london_bridge_19481797_2295892421.jpg?raw=true"
        >>> image2 = Image.open(requests.get(url, stream=True).raw)
        >>> images = [image1, image2]

        >>> processor = AutoImageProcessor.from_pretrained("zju-community/efficient_loftr")
        >>> model = AutoModel.from_pretrained("zju-community/efficient_loftr")

        >>> with torch.no_grad():
        >>>     inputs = processor(images, return_tensors="pt")
        >>>     outputs = model(**inputs)
        ```"""
        if labels is not None:
            raise ValueError("EfficientLoFTR is not trainable, no labels should be provided.")

        if pixel_values.ndim != 5 or pixel_values.size(1) != 2:
            raise ValueError("Input must be a 5D tensor of shape (batch_size, 2, num_channels, height, width)")

        batch_size, _, channels, height, width = pixel_values.shape
        pixel_values = pixel_values.reshape(batch_size * 2, channels, height, width)
        pixel_values = self.extract_one_channel_pixel_values(pixel_values)

        # 1. Local Feature CNN
        features = self.backbone(pixel_values)
        # Last stage outputs are coarse outputs
        coarse_features = features[-1]
        # Rest is residual features used in EfficientLoFTRFineFusionLayer
        residual_features = features[:-1]
        coarse_embed_dim, coarse_height, coarse_width = coarse_features.shape[-3:]

        # 2. Coarse-level LoFTR module
        cos, sin = self.rotary_emb(coarse_features)
        cos = cos.expand(batch_size * 2, -1, -1, -1).reshape(batch_size * 2, -1, coarse_embed_dim)
        sin = sin.expand(batch_size * 2, -1, -1, -1).reshape(batch_size * 2, -1, coarse_embed_dim)
        position_embeddings = (cos, sin)

        coarse_features = coarse_features.reshape(batch_size, 2, coarse_embed_dim, coarse_height, coarse_width)
        coarse_features = self.local_feature_transformer(
            coarse_features, position_embeddings=position_embeddings, **kwargs
        )

        features = (coarse_features,) + tuple(residual_features)

        return BackboneOutput(feature_maps=features)


def mask_border(tensor: torch.Tensor, border_margin: int, value: Union[bool, float, int]) -> torch.Tensor:
    """
    Mask a tensor border with a given value

    Args:
        tensor (`torch.Tensor` of shape `(batch_size, height_0, width_0, height_1, width_1)`):
            The tensor to mask
        border_margin (`int`) :
            The size of the border
        value (`Union[bool, int, float]`):
            The value to place in the tensor's borders

    Returns:
        tensor (`torch.Tensor` of shape `(batch_size, height_0, width_0, height_1, width_1)`):
            The masked tensor
    """
    if border_margin <= 0:
        return tensor

    tensor[:, :border_margin, :border_margin, :border_margin, :border_margin] = value
    tensor[:, -border_margin:, -border_margin:, -border_margin:, -border_margin:] = value
    return tensor


def create_meshgrid(
    height: Union[int, torch.Tensor],
    width: Union[int, torch.Tensor],
    normalized_coordinates: bool = False,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Copied from kornia library : kornia/kornia/utils/grid.py:26

    Generate a coordinate grid for an image.

    When the flag ``normalized_coordinates`` is set to True, the grid is
    normalized to be in the range :math:`[-1,1]` to be consistent with the pytorch
    function :py:func:`torch.nn.functional.grid_sample`.

    Args:
        height (`int`):
            The image height (rows).
        width (`int`):
            The image width (cols).
        normalized_coordinates (`bool`):
            Whether to normalize coordinates in the range :math:`[-1,1]` in order to be consistent with the
            PyTorch function :py:func:`torch.nn.functional.grid_sample`.
        device (`torch.device`):
            The device on which the grid will be generated.
        dtype (`torch.dtype`):
            The data type of the generated grid.

    Return:
        grid (`torch.Tensor` of shape `(1, height, width, 2)`):
            The grid tensor.

    Example:
        >>> create_meshgrid(2, 2)
        tensor([[[[-1., -1.],
                  [ 1., -1.]],
        <BLANKLINE>
                 [[-1.,  1.],
                  [ 1.,  1.]]]])

        >>> create_meshgrid(2, 2, normalized_coordinates=False)
        tensor([[[[0., 0.],
                  [1., 0.]],
        <BLANKLINE>
                 [[0., 1.],
                  [1., 1.]]]])

    """
    xs = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
    ys = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
    if normalized_coordinates:
        xs = (xs / (width - 1) - 0.5) * 2
        ys = (ys / (height - 1) - 0.5) * 2
    grid = torch.stack(torch.meshgrid(ys, xs, indexing="ij"), dim=-1)
    grid = grid.permute(1, 0, 2).unsqueeze(0)
    return grid


def spatial_expectation2d(input: torch.Tensor, normalized_coordinates: bool = True) -> torch.Tensor:
    r"""
    Copied from kornia library : kornia/geometry/subpix/dsnt.py:76
    Compute the expectation of coordinate values using spatial probabilities.

    The input heatmap is assumed to represent a valid spatial probability distribution,
    which can be achieved using :func:`~kornia.geometry.subpixel.spatial_softmax2d`.

    Args:
        input (`torch.Tensor` of shape `(batch_size, embed_dim, height, width)`):
            The input tensor representing dense spatial probabilities.
        normalized_coordinates (`bool`):
            Whether to return the coordinates normalized in the range of :math:`[-1, 1]`. Otherwise, it will return
            the coordinates in the range of the input shape.

    Returns:
        output (`torch.Tensor` of shape `(batch_size, embed_dim, 2)`)
            Expected value of the 2D coordinates. Output order of the coordinates is (x, y).

    Examples:
        >>> heatmaps = torch.tensor([[[
        ... [0., 0., 0.],
        ... [0., 0., 0.],
        ... [0., 1., 0.]]]])
        >>> spatial_expectation2d(heatmaps, False)
        tensor([[[1., 2.]]])

    """
    batch_size, embed_dim, height, width = input.shape

    # Create coordinates grid.
    grid = create_meshgrid(height, width, normalized_coordinates, input.device)
    grid = grid.to(input.dtype)

    pos_x = grid[..., 0].reshape(-1)
    pos_y = grid[..., 1].reshape(-1)

    input_flat = input.view(batch_size, embed_dim, -1)

    # Compute the expectation of the coordinates.
    expected_y = torch.sum(pos_y * input_flat, -1, keepdim=True)
    expected_x = torch.sum(pos_x * input_flat, -1, keepdim=True)

    output = torch.cat([expected_x, expected_y], -1)

    return output.view(batch_size, embed_dim, 2)


@auto_docstring(
    custom_intro="""
    EfficientLoFTR model taking images as inputs and outputting the matching of them.
    """
)
class EfficientLoFTRForKeypointMatching(EfficientLoFTRPreTrainedModel):
    """EfficientLoFTR dense image matcher

    Given two images, we determine the correspondences by:
      1. Extracting coarse and fine features through a backbone
      2. Transforming coarse features through self and cross attention
      3. Matching coarse features to obtain coarse coordinates of matches
      4. Obtaining full resolution fine features by fusing transformed and backbone coarse features
      5. Refining the coarse matches using fine feature patches centered at each coarse match in a two-stage refinement

    Yifan Wang, Xingyi He, Sida Peng, Dongli Tan and Xiaowei Zhou.
    Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed
    In CVPR, 2024. https://arxiv.org/abs/2403.04765
    """

    def __init__(self, config: EfficientLoFTRConfig):
        super().__init__(config)

        self.config = config
        self.efficientloftr = EfficientLoFTRModel(config)
        self.refinement_layer = EfficientLoFTRFineFusionLayer(config)

        self.post_init()

    def _get_matches_from_scores(self, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Based on a keypoint score matrix, compute the best keypoint matches between the first and second image.
        Since each image pair can have different number of matches, the matches are concatenated together for all pair
        in the batch and a batch_indices tensor is returned to specify which match belong to which element in the batch.

        Note:
            This step can be done as a postprocessing step, because does not involve any model weights/params.
            However, we keep it in the modeling code for consistency with other keypoint matching models AND for
            easier torch.compile/torch.export (all ops are in torch).

        Args:
            scores (`torch.Tensor` of shape `(batch_size, height_0, width_0, height_1, width_1)`):
                Scores of keypoints

        Returns:
            matched_indices (`torch.Tensor` of shape `(2, num_matches)`):
                Indices representing which pixel in the first image matches which pixel in the second image
            matching_scores (`torch.Tensor` of shape `(num_matches,)`):
                Scores of each match
        """
        batch_size, height0, width0, height1, width1 = scores.shape

        scores = scores.view(batch_size, height0 * width0, height1 * width1)

        # For each keypoint, get the best match
        max_0 = scores.max(2, keepdim=True).values
        max_1 = scores.max(1, keepdim=True).values

        # 1. Thresholding
        mask = scores > self.config.coarse_matching_threshold

        # 2. Border removal
        mask = mask.reshape(batch_size, height0, width0, height1, width1)
        mask = mask_border(mask, self.config.coarse_matching_border_removal, False)
        mask = mask.reshape(batch_size, height0 * width0, height1 * width1)

        # 3. Mutual nearest neighbors
        mask = mask * (scores == max_0) * (scores == max_1)

        # 4. Fine coarse matches
        masked_scores = scores * mask
        matching_scores_0, max_indices_0 = masked_scores.max(1)
        matching_scores_1, max_indices_1 = masked_scores.max(2)

        matching_indices = torch.cat([max_indices_0, max_indices_1]).reshape(batch_size, 2, -1)
        matching_scores = torch.stack([matching_scores_0, matching_scores_1], dim=1)

        # For the keypoints not meeting the threshold score, set the indices to -1 which corresponds to no matches found
        matching_indices = torch.where(matching_scores > 0, matching_indices, -1)

        return matching_indices, matching_scores

    def _coarse_matching(
        self, coarse_features: torch.Tensor, coarse_scale: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        For each image pair, compute the matching confidence between each coarse element (by default (image_height / 8)
        * (image_width / 8 elements)) from the first image to the second image.

        Note:
            This step can be done as a postprocessing step, because does not involve any model weights/params.
            However, we keep it in the modeling code for consistency with other keypoint matching models AND for
            easier torch.compile/torch.export (all ops are in torch).

        Args:
            coarse_features (`torch.Tensor` of shape `(batch_size, 2, hidden_size, coarse_height, coarse_width)`):
                Coarse features
            coarse_scale (`float`): Scale between the image size and the coarse size

        Returns:
            keypoints (`torch.Tensor` of shape `(batch_size, 2, num_matches, 2)`):
                Keypoints coordinates.
            matching_scores (`torch.Tensor` of shape `(batch_size, 2, num_matches)`):
                The confidence matching score of each keypoint.
            matched_indices (`torch.Tensor` of shape `(batch_size, 2, num_matches)`):
                Indices which indicates which keypoint in an image matched with which keypoint in the other image. For
                both image in the pair.
        """
        batch_size, _, embed_dim, height, width = coarse_features.shape

        # (batch_size, 2, embed_dim, height, width) -> (batch_size, 2, height * width, embed_dim)
        coarse_features = coarse_features.permute(0, 1, 3, 4, 2)
        coarse_features = coarse_features.reshape(batch_size, 2, -1, embed_dim)

        coarse_features = coarse_features / coarse_features.shape[-1] ** 0.5
        coarse_features_0 = coarse_features[:, 0]
        coarse_features_1 = coarse_features[:, 1]

        similarity = coarse_features_0 @ coarse_features_1.transpose(-1, -2)
        similarity = similarity / self.config.coarse_matching_temperature

        if self.config.coarse_matching_skip_softmax:
            confidence = similarity
        else:
            confidence = nn.functional.softmax(similarity, 1) * nn.functional.softmax(similarity, 2)

        confidence = confidence.view(batch_size, height, width, height, width)
        matched_indices, matching_scores = self._get_matches_from_scores(confidence)

        keypoints = torch.stack([matched_indices % width, matched_indices // width], dim=-1) * coarse_scale

        return keypoints, matching_scores, matched_indices

    def _get_first_stage_fine_matching(
        self,
        fine_confidence: torch.Tensor,
        coarse_matched_keypoints: torch.Tensor,
        fine_window_size: int,
        fine_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        For each coarse pixel, retrieve the highest fine confidence score and index.
        The index represents the matching between a pixel position in the fine window in the first image and a pixel
        position in the fine window of the second image.
        For example, for a fine_window_size of 64 (8 * 8), the index 2474 represents the matching between the index 38
        (2474 // 64) in the fine window of the first image, and the index 42 in the second image. This means that 38
        which corresponds to the position (4, 6) (4 // 8 and 4 % 8) is matched with the position (5, 2). In this example
        the coarse matched coordinate will be shifted to the matched fine coordinates in the first and second image.

        Note:
            This step can be done as a postprocessing step, because does not involve any model weights/params.
            However, we keep it in the modeling code for consistency with other keypoint matching models AND for
            easier torch.compile/torch.export (all ops are in torch).

        Args:
            fine_confidence (`torch.Tensor` of shape `(num_matches, fine_window_size, fine_window_size)`):
                First stage confidence of matching fine features between the first and the second image
            coarse_matched_keypoints (`torch.Tensor` of shape `(2, num_matches, 2)`):
                Coarse matched keypoint between the first and the second image.
            fine_window_size (`int`):
                Size of the window used to refine matches
            fine_scale (`float`):
                Scale between the size of fine features and coarse features

        Returns:
            indices (`torch.Tensor` of shape `(2, num_matches, 1)`):
                Indices of the fine coordinate matched in the fine window
            fine_matches (`torch.Tensor` of shape `(2, num_matches, 2)`):
                Coordinates of matched keypoints after the first fine stage
        """
        batch_size, num_keypoints, _, _ = fine_confidence.shape
        fine_kernel_size = torch_int(fine_window_size**0.5)

        fine_confidence = fine_confidence.reshape(batch_size, num_keypoints, -1)
        values, indices = torch.max(fine_confidence, dim=-1)
        indices = indices[..., None]
        indices_0 = indices // fine_window_size
        indices_1 = indices % fine_window_size

        grid = create_meshgrid(
            fine_kernel_size,
            fine_kernel_size,
            normalized_coordinates=False,
            device=fine_confidence.device,
            dtype=fine_confidence.dtype,
        )
        grid = grid - (fine_kernel_size // 2) + 0.5
        grid = grid.reshape(1, 1, -1, 2).expand(batch_size, num_keypoints, -1, -1)
        delta_0 = torch.gather(grid, 1, indices_0.unsqueeze(-1).expand(-1, -1, -1, 2)).squeeze(2)
        delta_1 = torch.gather(grid, 1, indices_1.unsqueeze(-1).expand(-1, -1, -1, 2)).squeeze(2)

        fine_matches_0 = coarse_matched_keypoints[:, 0] + delta_0 * fine_scale
        fine_matches_1 = coarse_matched_keypoints[:, 1] + delta_1 * fine_scale

        indices = torch.stack([indices_0, indices_1], dim=1)
        fine_matches = torch.stack([fine_matches_0, fine_matches_1], dim=1)

        return indices, fine_matches

    def _get_second_stage_fine_matching(
        self,
        indices: torch.Tensor,
        fine_matches: torch.Tensor,
        fine_confidence: torch.Tensor,
        fine_window_size: int,
        fine_scale: float,
    ) -> torch.Tensor:
        """
        For the given position in their respective fine windows, retrieve the 3x3 fine confidences around this position.
        After applying softmax to these confidences, compute the 2D spatial expected coordinates.
        Shift the first stage fine matching with these expected coordinates.

        Note:
            This step can be done as a postprocessing step, because does not involve any model weights/params.
            However, we keep it in the modeling code for consistency with other keypoint matching models AND for
            easier torch.compile/torch.export (all ops are in torch).

        Args:
            indices (`torch.Tensor` of shape `(batch_size, 2, num_keypoints)`):
                Indices representing the position of each keypoint in the fine window
            fine_matches (`torch.Tensor` of shape `(2, num_matches, 2)`):
                Coordinates of matched keypoints after the first fine stage
            fine_confidence (`torch.Tensor` of shape `(num_matches, fine_window_size, fine_window_size)`):
                Second stage confidence of matching fine features between the first and the second image
            fine_window_size (`int`):
                Size of the window used to refine matches
            fine_scale (`float`):
                Scale between the size of fine features and coarse features

        Returns:
            fine_matches (`torch.Tensor` of shape `(2, num_matches, 2)`):
                Coordinates of matched keypoints after the second fine stage
        """
        batch_size, num_keypoints, _, _ = fine_confidence.shape
        fine_kernel_size = torch_int(fine_window_size**0.5)

        indices_0 = indices[:, 0]
        indices_1 = indices[:, 1]
        indices_1_i = indices_1 // fine_kernel_size
        indices_1_j = indices_1 % fine_kernel_size

        # matches_indices, indices_0, indices_1_i, indices_1_j of shape (num_matches, 3, 3)
        batch_indices = torch.arange(batch_size, device=indices_0.device).reshape(batch_size, 1, 1, 1)
        matches_indices = torch.arange(num_keypoints, device=indices_0.device).reshape(1, num_keypoints, 1, 1)
        indices_0 = indices_0[..., None]
        indices_1_i = indices_1_i[..., None]
        indices_1_j = indices_1_j[..., None]

        delta = create_meshgrid(3, 3, normalized_coordinates=True, device=indices_0.device).to(torch.long)
        delta = delta[None, ...]

        indices_1_i = indices_1_i + delta[..., 1]
        indices_1_j = indices_1_j + delta[..., 0]

        fine_confidence = fine_confidence.reshape(
            batch_size, num_keypoints, fine_window_size, fine_kernel_size + 2, fine_kernel_size + 2
        )
        # (batch_size, seq_len, fine_window_size, fine_kernel_size + 2, fine_kernel_size + 2) -> (batch_size, seq_len, 3, 3)
        fine_confidence = fine_confidence[batch_indices, matches_indices, indices_0, indices_1_i, indices_1_j]
        fine_confidence = fine_confidence.reshape(batch_size, num_keypoints, 9)
        fine_confidence = nn.functional.softmax(
            fine_confidence / self.config.fine_matching_regress_temperature, dim=-1
        )

        heatmap = fine_confidence.reshape(batch_size, num_keypoints, 3, 3)
        fine_coordinates_normalized = spatial_expectation2d(heatmap, True)[0]

        fine_matches_0 = fine_matches[:, 0]
        fine_matches_1 = fine_matches[:, 1] + (fine_coordinates_normalized * (3 // 2) * fine_scale)

        fine_matches = torch.stack([fine_matches_0, fine_matches_1], dim=1)

        return fine_matches

    def _fine_matching(
        self,
        fine_features_0: torch.Tensor,
        fine_features_1: torch.Tensor,
        coarse_matched_keypoints: torch.Tensor,
        fine_scale: float,
    ) -> torch.Tensor:
        """
        For each coarse pixel with a corresponding window of fine features, compute the matching confidence between fine
        features in the first image and the second image.

        Fine features are sliced in two part :
        - The first part used for the first stage are the first fine_hidden_size - config.fine_matching_slicedim (64 - 8
         = 56 by default) features.
        - The second part used for the second stage are the last config.fine_matching_slicedim (8 by default) features.

        Each part is used to compute a fine confidence tensor of the following shape :
        (batch_size, (coarse_height * coarse_width), fine_window_size, fine_window_size)
        They correspond to the score between each fine pixel in the first image and each fine pixel in the second image.

        Args:
            fine_features_0 (`torch.Tensor` of shape `(num_matches, fine_kernel_size ** 2, fine_kernel_size ** 2)`):
                Fine features from the first image
            fine_features_1 (`torch.Tensor` of shape `(num_matches, (fine_kernel_size + 2) ** 2, (fine_kernel_size + 2)
            ** 2)`):
                Fine features from the second image
            coarse_matched_keypoints (`torch.Tensor` of shape `(2, num_matches, 2)`):
                Keypoint coordinates found in coarse matching for the first and second image
            fine_scale (`int`):
                Scale between the size of fine features and coarse features

        Returns:
            fine_coordinates (`torch.Tensor` of shape `(2, num_matches, 2)`):
                Matched keypoint between the first and the second image. All matched keypoints are concatenated in the
                second dimension.

        """
        batch_size, num_keypoints, fine_window_size, fine_embed_dim = fine_features_0.shape
        fine_matching_slice_dim = self.config.fine_matching_slice_dim

        fine_kernel_size = torch_int(fine_window_size**0.5)

        # Split fine features into first and second stage features
        split_fine_features_0 = torch.split(fine_features_0, fine_embed_dim - fine_matching_slice_dim, -1)
        split_fine_features_1 = torch.split(fine_features_1, fine_embed_dim - fine_matching_slice_dim, -1)

        # Retrieve first stage fine features
        fine_features_0 = split_fine_features_0[0]
        fine_features_1 = split_fine_features_1[0]

        # Normalize first stage fine features
        fine_features_0 = fine_features_0 / fine_features_0.shape[-1] ** 0.5
        fine_features_1 = fine_features_1 / fine_features_1.shape[-1] ** 0.5

        # Compute first stage confidence
        fine_confidence = fine_features_0 @ fine_features_1.transpose(-1, -2)
        fine_confidence = nn.functional.softmax(fine_confidence, 1) * nn.functional.softmax(fine_confidence, 2)
        fine_confidence = fine_confidence.reshape(
            batch_size, num_keypoints, fine_window_size, fine_kernel_size + 2, fine_kernel_size + 2
        )
        fine_confidence = fine_confidence[..., 1:-1, 1:-1]
        first_stage_fine_confidence = fine_confidence.reshape(
            batch_size, num_keypoints, fine_window_size, fine_window_size
        )

        fine_indices, fine_matches = self._get_first_stage_fine_matching(
            first_stage_fine_confidence,
            coarse_matched_keypoints,
            fine_window_size,
            fine_scale,
        )

        # Retrieve second stage fine features
        fine_features_0 = split_fine_features_0[1]
        fine_features_1 = split_fine_features_1[1]

        # Normalize second stage fine features
        fine_features_1 = fine_features_1 / fine_matching_slice_dim**0.5

        # Compute second stage fine confidence
        second_stage_fine_confidence = fine_features_0 @ fine_features_1.transpose(-1, -2)

        fine_coordinates = self._get_second_stage_fine_matching(
            fine_indices,
            fine_matches,
            second_stage_fine_confidence,
            fine_window_size,
            fine_scale,
        )

        return fine_coordinates

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> KeypointMatchingOutput:
        r"""
        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AutoModel
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/assets/phototourism_sample_images/london_bridge_78916675_4568141288.jpg?raw=true"
        >>> image1 = Image.open(requests.get(url, stream=True).raw)
        >>> url = "https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/assets/phototourism_sample_images/london_bridge_19481797_2295892421.jpg?raw=true"
        >>> image2 = Image.open(requests.get(url, stream=True).raw)
        >>> images = [image1, image2]

        >>> processor = AutoImageProcessor.from_pretrained("zju-community/efficient_loftr")
        >>> model = AutoModel.from_pretrained("zju-community/efficient_loftr")

        >>> with torch.no_grad():
        >>>     inputs = processor(images, return_tensors="pt")
        >>>     outputs = model(**inputs)
        ```"""
        if labels is not None:
            raise ValueError("SuperGlue is not trainable, no labels should be provided.")

        # 1. Extract coarse and residual features
        model_outputs: BackboneOutput = self.efficientloftr(pixel_values, **kwargs)
        features = model_outputs.feature_maps

        # 2. Compute coarse-level matching
        coarse_features = features[0]
        coarse_embed_dim, coarse_height, coarse_width = coarse_features.shape[-3:]
        batch_size, _, channels, height, width = pixel_values.shape
        coarse_scale = height / coarse_height
        coarse_keypoints, coarse_matching_scores, coarse_matched_indices = self._coarse_matching(
            coarse_features, coarse_scale
        )

        # 3. Fine-level refinement
        residual_features = features[1:]
        fine_features_0, fine_features_1 = self.refinement_layer(coarse_features, residual_features)

        # Filter fine features with coarse matches indices
        _, _, num_keypoints = coarse_matching_scores.shape
        batch_indices = torch.arange(batch_size)[..., None]
        fine_features_0 = fine_features_0[batch_indices, coarse_matched_indices[:, 0]]
        fine_features_1 = fine_features_1[batch_indices, coarse_matched_indices[:, 1]]

        # 4. Computer fine-level matching
        fine_height = torch_int(coarse_height * coarse_scale)
        fine_scale = height / fine_height
        matching_keypoints = self._fine_matching(fine_features_0, fine_features_1, coarse_keypoints, fine_scale)

        matching_keypoints[:, :, :, 0] = matching_keypoints[:, :, :, 0] / width
        matching_keypoints[:, :, :, 1] = matching_keypoints[:, :, :, 1] / height

        return KeypointMatchingOutput(
            matches=coarse_matched_indices,
            matching_scores=coarse_matching_scores,
            keypoints=matching_keypoints,
            hidden_states=model_outputs.hidden_states,
            attentions=model_outputs.attentions,
        )


__all__ = ["EfficientLoFTRPreTrainedModel", "EfficientLoFTRModel", "EfficientLoFTRForKeypointMatching"]
