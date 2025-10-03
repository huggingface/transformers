# coding=utf-8
# Copyright 2025 Meta AI and The HuggingFace Team. All rights reserved.
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
"""PyTorch SAM 3 model."""

from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch
import torch.nn as nn

from ...activations import ACT2FN
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...utils import (
    ModelOutput,
    auto_docstring,
    logging,
)
from .configuration_sam3_refactored import (
    Sam3Config,
    Sam3GeometryEncoderConfig,
    Sam3SegmentationConfig,
    Sam3TextConfig,
    Sam3TransformerConfig,
    Sam3VisionConfig,
)


logger = logging.get_logger(__name__)


# Output classes
@dataclass
@auto_docstring(custom_intro="Base class for SAM3 vision encoder's outputs.")
class Sam3VisionEncoderOutput(ModelOutput):
    r"""
    last_hidden_state (`torch.FloatTensor` of shape `(batch_size, height, width, hidden_size)`):
        Sequence of hidden-states at the output of the last layer of the model.
    hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`):
        Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
        shape `(batch_size, height, width, hidden_size)`. Hidden-states of the model at the output of each layer.
    attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True`):
        Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, height * width, height * width)`.
        Attentions weights after the attention softmax.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


@dataclass
@auto_docstring(custom_intro="Base class for SAM3 model's output.")
class Sam3ImageSegmentationOutput(ModelOutput):
    r"""
    pred_masks (`torch.FloatTensor` of shape `(batch_size, num_masks, height, width)`):
        The predicted segmentation masks.
    iou_scores (`torch.FloatTensor` of shape `(batch_size, num_masks)`):
        The predicted IoU scores for the masks.
    pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_masks, 4)`, *optional*):
        The predicted bounding boxes in cxcywh format.
    vision_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`):
        Tuple of `torch.FloatTensor` containing the hidden states from the vision encoder.
    vision_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True`):
        Tuple of `torch.FloatTensor` containing the attention weights from the vision encoder.
    """

    pred_masks: Optional[torch.FloatTensor] = None
    iou_scores: Optional[torch.FloatTensor] = None
    pred_boxes: Optional[torch.FloatTensor] = None
    vision_hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    vision_attentions: Optional[tuple[torch.FloatTensor, ...]] = None


# Utility functions
def box_cxcywh_to_xyxy(x):
    """Convert boxes from (center_x, center_y, width, height) to (x0, y0, x1, y1) format."""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    """Convert boxes from (x0, y0, x1, y1) to (center_x, center_y, width, height) format."""
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def get_2d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
    """
    Generate 2D sine-cosine positional embeddings.

    Args:
        embed_dim: Embedding dimension
        grid_size: Height and width of the grid
        add_cls_token: Whether to add a class token

    Returns:
        pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim]
    """
    grid_h = grid_size if isinstance(grid_size, int) else grid_size[0]
    grid_w = grid_size if isinstance(grid_size, int) else grid_size[1]

    grid_h_positions = torch.arange(grid_h, dtype=torch.float32)
    grid_w_positions = torch.arange(grid_w, dtype=torch.float32)
    grid = torch.meshgrid(grid_w_positions, grid_h_positions, indexing="xy")
    grid = torch.stack(grid, dim=0)

    grid = grid.reshape([2, 1, grid_h, grid_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    if add_cls_token:
        pos_embed = torch.cat([torch.zeros([1, embed_dim]), pos_embed], dim=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """Generate sine-cosine positional embeddings from a 2D grid."""
    assert embed_dim % 2 == 0

    # Use half of dimensions for w and half for h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])

    emb = torch.cat([emb_h, emb_w], dim=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Generate 1D sine-cosine positional embeddings."""
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = torch.outer(pos, omega)

    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb


# ==============================================================================
# Vision Encoder Components
# ==============================================================================


class Sam3PatchEmbeddings(nn.Module):
    """
    Image to Patch Embedding using Conv2d.
    """

    def __init__(self, config: Sam3VisionConfig):
        super().__init__()
        image_size = config.image_size
        patch_size = config.patch_size
        num_channels = config.num_channels
        hidden_size = config.hidden_size

        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.num_patches = (self.image_size[0] // self.patch_size[0]) * (self.image_size[1] // self.patch_size[1])

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.projection(pixel_values)
        # (B, C, H, W) -> (B, H, W, C)
        embeddings = embeddings.permute(0, 2, 3, 1)
        return embeddings


class Sam3VisionEmbeddings(nn.Module):
    """
    Construct the patch embeddings and position embeddings for the vision encoder.
    """

    def __init__(self, config: Sam3VisionConfig):
        super().__init__()
        self.config = config
        self.patch_embeddings = Sam3PatchEmbeddings(config)

        num_patches = self.patch_embeddings.num_patches
        self.num_positions = num_patches

        # Position embeddings - can be absolute or RoPE
        if config.use_abs_pos:
            self.position_embeddings = nn.Parameter(
                torch.zeros(1, *self.patch_embeddings.patch_size, config.hidden_size)
            )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        embeddings = self.patch_embeddings(pixel_values)

        if self.config.use_abs_pos and hasattr(self, "position_embeddings"):
            embeddings = embeddings + self.position_embeddings

        return embeddings


class Sam3VisionAttention(nn.Module):
    """
    Multi-head attention with optional RoPE and relative position embeddings.
    """

    def __init__(self, config: Sam3VisionConfig):
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.scale = self.attention_head_size**-0.5

        self.qkv = nn.Linear(config.hidden_size, self.all_head_size * 3, bias=config.qkv_bias)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        batch_size, height, width, _ = hidden_states.shape

        # QKV projection
        qkv = self.qkv(hidden_states).reshape(
            batch_size, height * width, 3, self.num_attention_heads, self.attention_head_size
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, H*W, head_dim)
        query, key, value = qkv.unbind(0)

        # Compute attention
        attention_interface: Callable = self._eager_attention
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS.get(self.config._attn_implementation, self._eager_attention)

        attn_output, attn_weights = attention_interface(query, key, value)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, height, width, -1)
        attn_output = self.proj(attn_output)

        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs

    def _eager_attention(self, query, key, value):
        """Standard attention computation."""
        attn_weights = (query @ key.transpose(-2, -1)) * self.scale
        attn_weights = attn_weights.softmax(dim=-1)
        attn_output = attn_weights @ value
        return attn_output, attn_weights


class Sam3VisionMLP(nn.Module):
    """
    MLP block used in Vision Transformer.
    """

    def __init__(self, config: Sam3VisionConfig):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = int(hidden_size * config.mlp_ratio)

        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.act = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class Sam3VisionLayer(nn.Module):
    """
    Vision Transformer layer (block).
    """

    def __init__(self, config: Sam3VisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = Sam3VisionAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = Sam3VisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        attn_outputs = self.attn(hidden_states, output_attentions=output_attentions)
        hidden_states = attn_outputs[0]
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,) + attn_outputs[1:]
        return outputs


class Sam3VisionEncoder(nn.Module):
    """Vision encoder (ViT backbone)."""

    def __init__(self, config: Sam3VisionConfig):
        super().__init__()
        self.config = config

        self.embeddings = Sam3VisionEmbeddings(config)
        self.layers = nn.ModuleList([Sam3VisionLayer(config) for _ in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, Sam3VisionEncoderOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.embeddings(pixel_values)

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        return Sam3VisionEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class Sam3VisionNeck(nn.Module):
    """
    Vision neck that creates multi-scale feature pyramid.
    """

    def __init__(self, config: Sam3VisionConfig):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        output_channels = config.output_channels

        # Projection layers for multi-scale features
        self.proj = nn.Conv2d(hidden_size, output_channels, kernel_size=1)

        # Position encoding
        self.num_pos_feats = output_channels // 2

    def forward(self, hidden_states: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            hidden_states: (B, H, W, C) from vision encoder

        Returns:
            List of multi-scale features
        """
        batch_size, height, width, channels = hidden_states.shape

        # (B, H, W, C) -> (B, C, H, W)
        hidden_states = hidden_states.permute(0, 3, 1, 2)

        # Project to output channels
        features = self.proj(hidden_states)

        # For now, return single scale - can be extended to multi-scale
        return [features]

    def get_position_encoding(self, features: torch.Tensor) -> torch.Tensor:
        """Generate sine-cosine position encoding for features."""
        batch_size, channels, height, width = features.shape

        # Generate position encodings
        y_embed = torch.arange(height, dtype=torch.float32, device=features.device)
        x_embed = torch.arange(width, dtype=torch.float32, device=features.device)

        y_embed = y_embed.unsqueeze(1).repeat(1, width)
        x_embed = x_embed.unsqueeze(0).repeat(height, 1)

        # Normalize to [0, 1]
        y_embed = y_embed / height
        x_embed = x_embed / width

        # Create position encoding
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=features.device)
        dim_t = 10000 ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1)
        pos = pos.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        return pos


# ==============================================================================
# Text Encoder Components
# ==============================================================================


class Sam3TextEmbeddings(nn.Module):
    """Text embeddings with token + position embeddings."""

    def __init__(self, config: Sam3TextConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Parameter(torch.zeros(1, config.max_position_embeddings, config.hidden_size))

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        seq_length = input_ids.shape[1]

        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding[:, :seq_length, :]

        embeddings = token_embeds + position_embeds
        return embeddings


class Sam3TextAttention(nn.Module):
    """Causal self-attention for text encoder."""

    def __init__(self, config: Sam3TextConfig):
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.q_proj = nn.Linear(config.hidden_size, self.all_head_size)
        self.k_proj = nn.Linear(config.hidden_size, self.all_head_size)
        self.v_proj = nn.Linear(config.hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.scale = self.attention_head_size**-0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.shape

        query = (
            self.q_proj(hidden_states)
            .view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )
        key = (
            self.k_proj(hidden_states)
            .view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )
        value = (
            self.v_proj(hidden_states)
            .view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )

        # Causal attention
        attn_weights = (query @ key.transpose(-2, -1)) * self.scale

        # Apply causal mask
        causal_mask = torch.triu(torch.ones(seq_length, seq_length, device=hidden_states.device), diagonal=1).bool()
        attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = attn_weights.softmax(dim=-1)
        attn_output = attn_weights @ value

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        attn_output = self.out_proj(attn_output)

        return attn_output


class Sam3TextMLP(nn.Module):
    """MLP for text encoder with QuickGELU activation."""

    def __init__(self, config: Sam3TextConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class Sam3TextLayer(nn.Module):
    """Text transformer layer."""

    def __init__(self, config: Sam3TextConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = Sam3TextAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = Sam3TextMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Sam3TextEncoder(nn.Module):
    """Text encoder (CLIP-style)."""

    def __init__(self, config: Sam3TextConfig):
        super().__init__()
        self.config = config

        self.embeddings = Sam3TextEmbeddings(config)
        self.layers = nn.ModuleList([Sam3TextLayer(config) for _ in range(config.num_hidden_layers)])
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Projection to output dimension
        self.text_projection = nn.Linear(config.hidden_size, config.output_dim)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.embeddings(input_ids)

        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states, attention_mask)

        hidden_states = self.final_layer_norm(hidden_states)

        # Project to output dimension
        hidden_states = self.text_projection(hidden_states)

        return hidden_states


# ==============================================================================
# Geometry Encoder Components
# ==============================================================================


class Sam3PointEmbedding(nn.Module):
    """Embed points with positional encoding."""

    def __init__(self, config: Sam3GeometryEncoderConfig):
        super().__init__()
        self.config = config
        # Embeddings for positive (foreground) and negative (background) points
        self.point_embeddings = nn.Embedding(4, config.hidden_size)  # 4 types: fg, bg, topleft, bottomright

    def forward(self, points: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (B, N, 2) normalized coordinates
            labels: (B, N) point labels (1=foreground, 0=background)

        Returns:
            point_embeddings: (B, N, hidden_size)
        """
        # Get embeddings based on labels
        point_embeddings = self.point_embeddings(labels)

        # Add positional encoding (simplified - could use sine-cosine)
        # For now, just return the embeddings
        return point_embeddings


class Sam3BoxEmbedding(nn.Module):
    """Embed bounding boxes."""

    def __init__(self, config: Sam3GeometryEncoderConfig):
        super().__init__()
        self.config = config
        # Box embeddings - encode boxes as corner points
        self.box_embedding = nn.Linear(4, config.hidden_size)

    def forward(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            boxes: (B, N, 4) in cxcywh format

        Returns:
            box_embeddings: (B, N, hidden_size)
        """
        return self.box_embedding(boxes)


class Sam3GeometryEncoder(nn.Module):
    """Encode geometric prompts (points, boxes, masks)."""

    def __init__(self, config: Sam3GeometryEncoderConfig):
        super().__init__()
        self.config = config

        self.point_embedding = Sam3PointEmbedding(config)
        self.box_embedding = Sam3BoxEmbedding(config)

        # Transformer layers to process geometry
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.hidden_size,
                    nhead=config.num_attention_heads,
                    dim_feedforward=config.intermediate_size,
                    dropout=config.dropout,
                    activation=config.hidden_act,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(config.num_layers)
            ]
        )

    def forward(
        self,
        input_points: Optional[torch.Tensor] = None,
        input_points_labels: Optional[torch.Tensor] = None,
        input_boxes: Optional[torch.Tensor] = None,
        input_boxes_labels: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode geometric prompts.

        Returns:
            geometry_features: (B, num_prompts, hidden_size)
            geometry_mask: (B, num_prompts) - mask for valid prompts
        """
        embeddings_list = []

        # Process points
        if input_points is not None and input_points_labels is not None:
            point_embeds = self.point_embedding(input_points, input_points_labels)
            embeddings_list.append(point_embeds)

        # Process boxes
        if input_boxes is not None:
            box_embeds = self.box_embedding(input_boxes)
            embeddings_list.append(box_embeds)

        if not embeddings_list:
            # No prompts provided, return empty
            batch_size = 1
            return torch.zeros(batch_size, 0, self.config.hidden_size), torch.zeros(batch_size, 0, dtype=torch.bool)

        # Concatenate all embeddings
        geometry_features = torch.cat(embeddings_list, dim=1)

        # Process through transformer layers
        for layer in self.layers:
            geometry_features = layer(geometry_features)

        # Create mask (all valid for now)
        batch_size, num_prompts, _ = geometry_features.shape
        geometry_mask = torch.zeros(batch_size, num_prompts, dtype=torch.bool, device=geometry_features.device)

        return geometry_features, geometry_mask


# ==============================================================================
# Transformer Components
# ==============================================================================


class Sam3TransformerEncoder(nn.Module):
    """Transformer encoder with cross-attention to prompts."""

    def __init__(self, config: Sam3TransformerConfig):
        super().__init__()
        self.config = config

        # Self-attention layers for image features
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.hidden_size,
                    nhead=config.num_attention_heads,
                    dim_feedforward=config.intermediate_size,
                    dropout=config.dropout,
                    activation=config.hidden_act,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(config.encoder_layers)
            ]
        )

        # Cross-attention to prompts (if prompts are provided)
        self.cross_attn_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    dropout=config.dropout,
                    batch_first=True,
                )
                for _ in range(config.encoder_layers)
            ]
        )
        self.cross_attn_norm = nn.ModuleList([nn.LayerNorm(config.hidden_size) for _ in range(config.encoder_layers)])

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        prompts: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            src: (B, H*W, hidden_size) flattened image features
            src_mask: Optional attention mask for image features
            prompts: (B, num_prompts, hidden_size) optional prompt features
            prompt_mask: Optional attention mask for prompts

        Returns:
            memory: (B, H*W, hidden_size) encoded features
        """
        hidden_states = src

        for i, (self_attn_layer, cross_attn_layer, cross_attn_norm) in enumerate(
            zip(self.layers, self.cross_attn_layers, self.cross_attn_norm)
        ):
            # Self-attention on image features
            hidden_states = self_attn_layer(hidden_states, src_key_padding_mask=src_mask)

            # Cross-attention to prompts (if provided)
            if prompts is not None and prompts.shape[1] > 0:
                # Convert prompt_mask to key_padding_mask format (True = ignore)
                key_padding_mask = None
                if prompt_mask is not None:
                    # Assume prompt_mask is 1 for valid tokens, 0 for padding
                    key_padding_mask = prompt_mask == 0

                residual = hidden_states
                hidden_states = cross_attn_norm(hidden_states)
                cross_attn_output, _ = cross_attn_layer(
                    hidden_states,
                    prompts,
                    prompts,
                    key_padding_mask=key_padding_mask,
                )
                hidden_states = residual + cross_attn_output

        return hidden_states


class Sam3TransformerDecoder(nn.Module):
    """Transformer decoder generating object queries conditioned on prompts."""

    def __init__(self, config: Sam3TransformerConfig):
        super().__init__()
        self.config = config

        # Learnable object queries
        self.query_embed = nn.Parameter(torch.randn(config.num_queries, config.hidden_size))

        self.layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=config.hidden_size,
                    nhead=config.num_attention_heads,
                    dim_feedforward=config.intermediate_size,
                    dropout=config.dropout,
                    activation=config.hidden_act,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(config.decoder_layers)
            ]
        )

        # Additional cross-attention to prompts (separate from memory cross-attention)
        self.prompt_cross_attn_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    dropout=config.dropout,
                    batch_first=True,
                )
                for _ in range(config.decoder_layers)
            ]
        )
        self.prompt_cross_attn_norm = nn.ModuleList(
            [nn.LayerNorm(config.hidden_size) for _ in range(config.decoder_layers)]
        )

        # Box refinement head
        self.bbox_embed = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 4),
        )

    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
        prompts: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            memory: (B, H*W, hidden_size) from encoder
            memory_mask: Optional mask for memory
            prompts: (B, num_prompts, hidden_size) optional prompt features
            prompt_mask: Optional attention mask for prompts

        Returns:
            queries: (B, num_queries, hidden_size)
            boxes: (B, num_queries, 4) predicted boxes
        """
        batch_size = memory.shape[0]

        # Expand queries for batch
        tgt = self.query_embed.unsqueeze(0).expand(batch_size, -1, -1)

        # Decode with cross-attention to both memory (image features) and prompts
        for i, (decoder_layer, prompt_cross_attn, prompt_norm) in enumerate(
            zip(self.layers, self.prompt_cross_attn_layers, self.prompt_cross_attn_norm)
        ):
            # Standard decoder layer: self-attention + cross-attention to memory + FFN
            tgt = decoder_layer(tgt, memory, memory_key_padding_mask=memory_mask)

            # Additional cross-attention to prompts (if provided)
            if prompts is not None and prompts.shape[1] > 0:
                # Convert prompt_mask to key_padding_mask format (True = ignore)
                key_padding_mask = None
                if prompt_mask is not None:
                    # Assume prompt_mask is 1 for valid tokens, 0 for padding
                    key_padding_mask = prompt_mask == 0

                residual = tgt
                tgt = prompt_norm(tgt)
                prompt_attn_output, _ = prompt_cross_attn(
                    tgt,
                    prompts,
                    prompts,
                    key_padding_mask=key_padding_mask,
                )
                tgt = residual + prompt_attn_output

        # Predict boxes
        boxes = self.bbox_embed(tgt).sigmoid()

        return tgt, boxes


# ==============================================================================
# Segmentation Head Components
# ==============================================================================


class Sam3PixelDecoder(nn.Module):
    """Pixel decoder for generating mask features."""

    def __init__(self, config: Sam3SegmentationConfig):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size

        # Progressive upsampling
        self.upsample_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode=config.interpolation_mode),
                )
                for _ in range(config.num_upsampling_stages)
            ]
        )

        self.mask_projection = nn.Conv2d(hidden_size, hidden_size, kernel_size=1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of multi-scale features from vision neck

        Returns:
            mask_features: (B, hidden_size, H, W) upsampled features
        """
        # Use highest resolution feature
        x = features[-1] if isinstance(features, list) else features

        # Progressive upsampling
        for upsample in self.upsample_layers:
            x = upsample(x)

        mask_features = self.mask_projection(x)
        return mask_features


class Sam3MaskPredictor(nn.Module):
    """Predict masks from queries and mask features."""

    def __init__(self, config: Sam3SegmentationConfig):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size

        # MLP to generate mask embeddings from queries
        self.mask_embed = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, queries: torch.Tensor, mask_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: (B, num_queries, hidden_size)
            mask_features: (B, hidden_size, H, W)

        Returns:
            pred_masks: (B, num_queries, H, W)
        """
        # Generate mask embeddings
        mask_embed = self.mask_embed(queries)  # (B, num_queries, hidden_size)

        # Compute dot product with mask features
        batch_size, num_queries, channels = mask_embed.shape
        _, _, height, width = mask_features.shape

        # Reshape for dot product
        mask_embed = mask_embed.reshape(batch_size * num_queries, channels, 1, 1)
        mask_features_expanded = mask_features.unsqueeze(1).expand(-1, num_queries, -1, -1, -1)
        mask_features_expanded = mask_features_expanded.reshape(batch_size * num_queries, channels, height, width)

        # Compute masks
        pred_masks = (mask_embed * mask_features_expanded).sum(dim=1)
        pred_masks = pred_masks.reshape(batch_size, num_queries, height, width)

        return pred_masks


class Sam3PresenceHead(nn.Module):
    """Predict IoU scores for masks."""

    def __init__(self, config: Sam3SegmentationConfig):
        super().__init__()
        hidden_size = config.hidden_size

        self.iou_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: (B, num_queries, hidden_size)

        Returns:
            iou_scores: (B, num_queries)
        """
        iou_scores = self.iou_head(queries).squeeze(-1)
        return iou_scores


class Sam3SegmentationHead(nn.Module):
    """Complete segmentation head with optional prompt conditioning."""

    def __init__(self, config: Sam3SegmentationConfig):
        super().__init__()
        self.config = config

        self.pixel_decoder = Sam3PixelDecoder(config)
        self.mask_predictor = Sam3MaskPredictor(config)
        self.presence_head = Sam3PresenceHead(config)

        # Optional: Additional projection to combine queries with prompt information
        # This can help the mask predictor be more aware of the prompts
        self.query_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

    def forward(
        self,
        queries: torch.Tensor,
        multi_scale_features: list[torch.Tensor],
        prompts: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            queries: (B, num_queries, hidden_size) - already conditioned on prompts from decoder
            multi_scale_features: List of features from vision neck
            prompts: (B, num_prompts, hidden_size) - optional, can be used for additional conditioning

        Returns:
            pred_masks: (B, num_queries, H, W)
            iou_scores: (B, num_queries)
        """
        # Project queries (they already contain prompt information from the decoder)
        projected_queries = self.query_proj(queries)

        # Generate mask features
        mask_features = self.pixel_decoder(multi_scale_features)

        # Predict masks using projected queries
        pred_masks = self.mask_predictor(projected_queries, mask_features)

        # Predict IoU scores
        iou_scores = self.presence_head(projected_queries)

        return pred_masks, iou_scores


# ==============================================================================
# Main Model
# ==============================================================================


class Sam3PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Sam3Config
    base_model_prefix = "sam3"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Sam3VisionLayer", "Sam3TextLayer"]

    def _init_weights(self, module):
        """Initialize the weights"""
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv2d)):
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


class Sam3Model(Sam3PreTrainedModel):
    """
    SAM3 model for image segmentation with text and geometric prompts.
    """

    def __init__(self, config: Sam3Config):
        super().__init__(config)
        self.config = config

        # Vision encoder
        self.vision_encoder = Sam3VisionEncoder(config.vision_config)
        self.vision_neck = Sam3VisionNeck(config.vision_config)

        # Text encoder (optional)
        self.text_encoder = Sam3TextEncoder(config.text_config)

        # Geometry encoder
        self.geometry_encoder = Sam3GeometryEncoder(config.geometry_encoder_config)

        # Transformer
        self.transformer_encoder = Sam3TransformerEncoder(config.transformer_config)
        self.transformer_decoder = Sam3TransformerDecoder(config.transformer_config)

        # Segmentation head
        self.segmentation_head = Sam3SegmentationHead(config.segmentation_config)

        self.post_init()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_points: Optional[torch.FloatTensor] = None,
        input_points_labels: Optional[torch.LongTensor] = None,
        input_boxes: Optional[torch.FloatTensor] = None,
        input_boxes_labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, Sam3ImageSegmentationOutput]:
        """
        Forward pass of SAM3 model.

        Args:
            pixel_values: Input images of shape (batch_size, num_channels, height, width)
            input_ids: Input text token ids (optional)
            attention_mask: Attention mask for text (optional)
            input_points: Input point prompts
            input_points_labels: Labels for input points
            input_boxes: Input box prompts
            input_boxes_labels: Labels for input boxes
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return a ModelOutput object

        Returns:
            Sam3ImageSegmentationOutput or tuple
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 1. Encode images with vision encoder
        vision_outputs = self.vision_encoder(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        vision_features = vision_outputs.last_hidden_state  # (B, H, W, C)

        # 2. Get multi-scale features from neck
        multi_scale_features = self.vision_neck(vision_features)  # List[(B, 256, H, W)]

        # 3. Encode text (if provided)
        text_features = None
        text_mask = None
        if input_ids is not None:
            text_features = self.text_encoder(input_ids, attention_mask)  # (B, seq_len, 256)
            text_mask = attention_mask

        # 4. Encode geometric prompts
        geometry_features, geometry_mask = self.geometry_encoder(
            input_points=input_points,
            input_points_labels=input_points_labels,
            input_boxes=input_boxes,
            input_boxes_labels=input_boxes_labels,
        )  # (B, num_prompts, 256)

        # 5. Combine all prompts (text + geometry)
        prompt_list = []
        prompt_mask_list = []

        if text_features is not None:
            prompt_list.append(text_features)
            if text_mask is not None:
                prompt_mask_list.append(text_mask)

        if geometry_features is not None and geometry_features.shape[1] > 0:
            prompt_list.append(geometry_features)
            if geometry_mask is not None:
                prompt_mask_list.append(geometry_mask)

        # Concatenate prompts along sequence dimension
        if len(prompt_list) > 0:
            combined_prompts = torch.cat(prompt_list, dim=1)  # (B, total_prompt_len, 256)
            combined_prompt_mask = torch.cat(prompt_mask_list, dim=1) if len(prompt_mask_list) > 0 else None
        else:
            # If no prompts provided, use empty tensor
            batch_size = pixel_values.shape[0]
            combined_prompts = torch.zeros(
                (batch_size, 0, self.config.transformer_config.hidden_size),
                device=pixel_values.device,
                dtype=pixel_values.dtype,
            )
            combined_prompt_mask = None

        # 6. Prepare encoder input - flatten spatial dimensions
        batch_size, height, width, channels = vision_features.shape
        vision_features_flat = vision_features.reshape(batch_size, height * width, channels)

        # 7. Run transformer encoder with prompts for cross-attention
        # Note: The encoder should cross-attend between image features and prompts
        memory = self.transformer_encoder(
            vision_features_flat,
            prompts=combined_prompts,
            prompt_mask=combined_prompt_mask,
        )  # (B, H*W, 256)

        # 8. Run transformer decoder to generate object queries
        # The decoder cross-attends to both encoded image features (memory) and prompts
        queries, pred_boxes = self.transformer_decoder(
            memory,
            prompts=combined_prompts,
            prompt_mask=combined_prompt_mask,
        )  # (B, num_queries, 256), (B, num_queries, 4)

        # 9. Generate masks from queries and multi-scale features
        # The segmentation head uses queries conditioned on prompts
        pred_masks, iou_scores = self.segmentation_head(
            queries,
            multi_scale_features,
            prompts=combined_prompts,
        )

        if not return_dict:
            return (pred_masks, iou_scores, pred_boxes, vision_outputs.hidden_states, vision_outputs.attentions)

        return Sam3ImageSegmentationOutput(
            pred_masks=pred_masks,
            iou_scores=iou_scores,
            pred_boxes=pred_boxes,
            vision_hidden_states=vision_outputs.hidden_states,
            vision_attentions=vision_outputs.attentions,
        )


__all__ = [
    "Sam3Model",
    "Sam3PreTrainedModel",
    "Sam3ImageSegmentationOutput",
    "Sam3VisionEncoderOutput",
    "Sam3VisionEncoder",
    "Sam3TextEncoder",
    "Sam3GeometryEncoder",
]
