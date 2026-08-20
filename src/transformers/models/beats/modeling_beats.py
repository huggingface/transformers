# coding=utf-8
# Copyright 2026 Microsoft Research and The HuggingFace Inc. team.
# Licensed under the MIT License.

"""PyTorch BEATs model."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_beats import BEATsConfig

logger = logging.get_logger(__name__)


class BEATsAttention(nn.Module):
    """Multi-head self-attention with relative position bias and GRU gating."""

    def __init__(self, config: BEATsConfig):
        super().__init__()
        self.embed_dim = config.encoder_embed_dim
        self.num_heads = config.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim ** -0.5

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout = nn.Dropout(p=config.attention_dropout)

        self.relative_position_embedding = config.relative_position_embedding
        self.gru_rel_pos = config.gru_rel_pos
        self.num_buckets = config.num_buckets
        self.max_distance = config.max_distance

        if config.relative_position_embedding:
            self.relative_attention_bias = nn.Embedding(
                config.num_buckets, self.num_heads
            )

        if config.gru_rel_pos:
            self.grep_linear = nn.Linear(self.head_dim, config.grep_linear_units)
            self.grep_a = nn.Parameter(torch.ones(1, self.num_heads, 1, 1))

    def _get_relative_position_bias(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        relative_positions = relative_positions.abs().clamp(max=self.num_buckets - 1)
        bias = self.relative_attention_bias(relative_positions)
        return bias.permute(2, 0, 1).unsqueeze(0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states) * self.scaling
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1))

        if self.relative_position_embedding:
            attn_weights = attn_weights + self._get_relative_position_bias(
                seq_len, hidden_states.device
            )

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)

        if self.gru_rel_pos:
            gate = self.grep_linear(v.mean(dim=2, keepdim=True))
            gate = gate.mean(dim=-1, keepdim=True)
            gate = torch.sigmoid(self.grep_a * gate)
            v = v * gate

        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output


class BEATsEncoderLayer(nn.Module):
    """Single BEATs encoder layer — names match checkpoint keys exactly."""

    def __init__(self, config: BEATsConfig):
        super().__init__()
        self.self_attn = BEATsAttention(config)
        self.self_attn_layer_norm = LayerNorm(config.encoder_embed_dim)
        self.fc1 = nn.Linear(config.encoder_embed_dim, config.encoder_ffn_embed_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_embed_dim, config.encoder_embed_dim)
        self.final_layer_norm = LayerNorm(config.encoder_embed_dim)
        self.dropout = nn.Dropout(p=config.dropout)
        self.activation_dropout = nn.Dropout(p=config.activation_dropout)
        self.act = nn.GELU()
        self.layer_norm_first = config.layer_norm_first

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states

        if self.layer_norm_first:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        if not self.layer_norm_first:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states

        if self.layer_norm_first:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.act(self.fc1(hidden_states))
        hidden_states = self.activation_dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        if not self.layer_norm_first:
            hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class BEATsEncoder(nn.Module):
    """BEATs Transformer encoder with positional conv."""

    def __init__(self, config: BEATsConfig):
        super().__init__()
        self.pos_conv = nn.Sequential(
            nn.Conv1d(
                config.encoder_embed_dim,
                config.encoder_embed_dim,
                kernel_size=config.conv_pos,
                padding=config.conv_pos // 2,
                groups=config.conv_pos_groups,
            ),
            nn.GELU(),
        )
        self.layers = nn.ModuleList(
            [BEATsEncoderLayer(config) for _ in range(config.encoder_layers)]
        )
        self.layer_norm = LayerNorm(config.encoder_embed_dim)
        self.dropout = nn.Dropout(p=config.dropout)
        self.layer_norm_first = config.layer_norm_first

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pos_x = hidden_states.transpose(1, 2)
        pos_x = self.pos_conv(pos_x)
        pos_x = pos_x[:, :, : hidden_states.size(1)].transpose(1, 2)
        hidden_states = hidden_states + pos_x
        hidden_states = self.dropout(hidden_states)

        if not self.layer_norm_first:
            hidden_states = self.layer_norm(hidden_states)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        if self.layer_norm_first:
            hidden_states = self.layer_norm(hidden_states)

        return hidden_states


class BEATsPatchEmbedding(nn.Module):
    """2D patch embedding from Mel Filterbank — names match checkpoint keys."""

    def __init__(self, config: BEATsConfig):
        super().__init__()
        self.patch_embedding = nn.Conv2d(
            1,
            config.embed_dim,
            kernel_size=config.input_patch_size,
            stride=config.input_patch_size,
            bias=config.conv_bias,
        )
        self.layer_norm = LayerNorm(config.embed_dim)
        self.post_extract_proj = (
            nn.Linear(config.embed_dim, config.encoder_embed_dim)
            if config.embed_dim != config.encoder_embed_dim
            else None
        )
        self.dropout_input = nn.Dropout(p=config.dropout_input)

    def forward(self, fbank: torch.Tensor) -> torch.Tensor:
        fbank = fbank.unsqueeze(1)
        features = self.patch_embedding(fbank)
        features = features.reshape(features.shape[0], features.shape[1], -1)
        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)
        features = self.dropout_input(features)
        return features


class BEATsPreTrainedModel(PreTrainedModel):
    config_class = BEATsConfig
    base_model_prefix = "beats"
    supports_gradient_checkpointing = False
    _tied_weights_keys = []

    def _init_weights(self, module):
        pass

    @property
    def all_tied_weights_keys(self):
        return {}


class BEATsModel(BEATsPreTrainedModel):
    """BEATs base model — outputs hidden states."""

    def __init__(self, config: BEATsConfig):
        super().__init__(config)
        self.patch_embedding = BEATsPatchEmbedding(config)
        self.encoder = BEATsEncoder(config)

    def forward(
        self,
        fbank: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.patch_embedding(fbank)
        hidden_states = self.encoder(hidden_states, attention_mask)
        return hidden_states


class BEATsForAudioClassification(BEATsPreTrainedModel):
    """BEATs with classification head."""

    def __init__(self, config: BEATsConfig):
        super().__init__(config)
        self.beats = BEATsModel(config)
        self.classifier_dropout = nn.Dropout(p=config.dropout)
        self.classifier = nn.Linear(config.encoder_embed_dim, config.num_classes)

    def forward(
        self,
        fbank: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        hidden_states = self.beats(fbank, attention_mask)

        if attention_mask is not None:
            hidden_states[attention_mask] = 0
            hidden_states = hidden_states.sum(dim=1)
            hidden_states = hidden_states / (~attention_mask).sum(dim=1, keepdim=True)
        else:
            hidden_states = hidden_states.mean(dim=1)

        hidden_states = self.classifier_dropout(hidden_states)
        logits = self.classifier(hidden_states)
        probs = torch.sigmoid(logits)

        output = {"logits": probs}
        if labels is not None:
            output["loss"] = nn.BCELoss()(probs, labels.float())
        return output

__all__ = ['BEATsModel', 'BEATsForAudioClassification', 'BEATsPreTrainedModel']
