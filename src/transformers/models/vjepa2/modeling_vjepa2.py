# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from ...activations import ACT2FN
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...utils import ModelOutput, auto_docstring, can_return_tuple, logging
from .configuration_vjepa2 import VJEPA2Config


logger = logging.get_logger(__name__)


@dataclass
@auto_docstring(
    custom_intro="""
    VJEPA Predictor outputs that also contains the masked encoder outputs
    """
)
class VJEPA2WithMaskedInputPredictorOutput(ModelOutput):
    r"""
    masked_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, returned when `context_mask` is provided which is applied on VJEPA2Encoder outputs):
        The masked hidden state of the model.
    target_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, returned when `target_mask` is provided which is applied on VJEPA2Encoder outputs):
        The target hidden state of the model.
    """

    last_hidden_state: torch.FloatTensor
    masked_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    target_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
@auto_docstring(
    custom_intro="""
    VJEPA outputs that also contains the masked encoder outputs
    Optionally contains the predictor outputs
    """
)
class VJEPA2WithMaskedInputModelOutput(ModelOutput):
    r"""
    masked_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, returned when `context_mask` is provided which is applied on VJEPA2Encoder outputs):
        The masked hidden state of the model.
    predictor_output (`VJEPA2WithMaskedInputPredictorOutput`, *optional*):
        The output from the Predictor module.
    """

    last_hidden_state: torch.FloatTensor
    masked_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    predictor_output: Optional[VJEPA2WithMaskedInputPredictorOutput] = None

    def to_tuple(self):
        output = list(super().to_tuple())
        if isinstance(output[-1], VJEPA2WithMaskedInputPredictorOutput):
            output[-1] = output[-1].to_tuple()
        return tuple(output)


class VJEPA2PatchEmbeddings3D(nn.Module):
    """
    Image to Patch Embedding
    """

    def __init__(
        self,
        config: VJEPA2Config,
        hidden_size: int = 1024,
    ):
        super().__init__()
        self.patch_size = config.patch_size
        self.tubelet_size = config.tubelet_size
        self.hidden_size = hidden_size

        self.proj = nn.Conv3d(
            in_channels=config.in_chans,
            out_channels=hidden_size,
            kernel_size=(config.tubelet_size, config.patch_size, config.patch_size),
            stride=(config.tubelet_size, config.patch_size, config.patch_size),
        )

    @staticmethod
    def num_patches(config):
        return (
            (config.frames_per_clip // config.tubelet_size)
            * (config.crop_size // config.patch_size)
            * (config.crop_size // config.patch_size)
        )

    def forward(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        x = self.proj(pixel_values_videos).flatten(2).transpose(1, 2)
        return x


class VJEPA2Embeddings(nn.Module):
    """
    Construct mask token, position and patch embeddings.
    """

    def __init__(self, config: VJEPA2Config, hidden_size: int = 1024):
        super().__init__()

        self.config = config
        self.hidden_size = hidden_size
        self.patch_embeddings = VJEPA2PatchEmbeddings3D(config, hidden_size=hidden_size)

        self.num_patches = self.patch_embeddings.num_patches
        self.patch_size = config.patch_size

    def forward(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        num_frames = pixel_values_videos.shape[1]

        # Swap `frames` and `channels` dims, the result is:
        # (batch_size, channels, num_frames, height, width)
        pixel_values_videos = pixel_values_videos.permute(0, 2, 1, 3, 4)

        # For some cases, if the input vision (image/video) consists of num_frames < tubelet_size,
        # then embedding lookup fails. In these cases, we duplicate the frames.
        if num_frames < self.config.tubelet_size:
            pixel_values_videos = pixel_values_videos.repeat(1, 1, self.config.tubelet_size, 1, 1)

        target_dtype = self.patch_embeddings.proj.weight.dtype
        pixel_values_videos = pixel_values_videos.to(dtype=target_dtype)
        embeddings = self.patch_embeddings(pixel_values_videos)

        return embeddings


# Adapted from transformers.models.vit.modeling_vit.eager_attention_forward
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    # Take the dot product between "query" and "key" to get the raw attention scores.
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling

    # Normalize the attention scores to probabilities.
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    # Mask heads if we want to
    if attention_mask is not None:
        attn_weights = attn_weights * attention_mask

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def rotate_queries_or_keys(x, pos):
    B, num_heads, N, D = x.size()

    # similar to inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    # they are computing this every time. instead HF style is to compute the inv_freq once and store it
    # -- compute angle for each position
    omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
    omega /= D / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)
    freq = pos.unsqueeze(-1) * omega  # (..., N, D/2), outer product

    # -- build rotation matrix and apply
    emb_sin = freq.sin()  # (..., N, D/2)
    emb_cos = freq.cos()  # (..., N, D/2)

    emb_sin = emb_sin.squeeze(-1).repeat(1, 1, 1, 2)
    emb_cos = emb_cos.squeeze(-1).repeat(1, 1, 1, 2)

    # --
    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)

    y = torch.stack((-y2, y1), dim=-1)
    y = y.flatten(-2)
    return (x * emb_cos) + (y * emb_sin)


class VJEPA2RopeAttention(nn.Module):
    def __init__(
        self,
        config: VJEPA2Config,
        hidden_size: int = 1024,
        num_attention_heads: int = 16,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {(hidden_size,)} is not a multiple of the number of attention "
                f"heads {num_attention_heads}."
            )

        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.proj = nn.Linear(hidden_size, hidden_size)
        self.dropout_prob = config.attention_probs_dropout_prob
        self.dropout = nn.Dropout(self.dropout_prob)

        self.grid_size = self.config.crop_size // self.config.patch_size
        self.grid_depth = self.config.frames_per_clip // self.config.tubelet_size

        self.d_dim = int(2 * ((self.attention_head_size // 3) // 2))
        self.h_dim = int(2 * ((self.attention_head_size // 3) // 2))
        self.w_dim = int(2 * ((self.attention_head_size // 3) // 2))

        self.scaling = self.attention_head_size**-0.5
        self.is_causal = False

    def _get_frame_pos(self, ids):
        tokens_per_frame = int(self.grid_size * self.grid_size)
        return ids // tokens_per_frame

    def _get_height_pos(self, ids):
        # Remove frame component from ids
        tokens_per_frame = int(self.grid_size * self.grid_size)
        frame_ids = self._get_frame_pos(ids)
        ids = ids - tokens_per_frame * frame_ids
        # --
        tokens_per_row = self.grid_size
        return ids // tokens_per_row

    def get_position_ids(self, x, masks=None):
        device = x.device
        token_size = x.size(1)

        # Note: when masks is none, we use a 1d id instead of Bxnum_attention_heads mask,
        # as 1d vector is broadcasted to the correct shapes.
        if masks is not None:
            ids = masks.unsqueeze(1).repeat(1, self.num_attention_heads, 1)
        else:
            ids = torch.arange(token_size, device=device)
        # change to allow for extrapolation
        tokens_per_frame = int(self.grid_size * self.grid_size)
        frame_ids = self._get_frame_pos(ids)
        # --
        tokens_per_row = self.grid_size
        height_ids = self._get_height_pos(ids)
        # --
        # Remove frame component from ids (1st term) and height component (2nd term)
        width_ids = (ids - tokens_per_frame * frame_ids) - tokens_per_row * height_ids
        return frame_ids, height_ids, width_ids

    def apply_rotary_embeddings(self, qk, pos_ids):
        d_mask, h_mask, w_mask = pos_ids
        s = 0
        qkd = rotate_queries_or_keys(qk[..., s : s + self.d_dim], pos=d_mask)
        s += self.d_dim
        qkh = rotate_queries_or_keys(qk[..., s : s + self.h_dim], pos=h_mask)
        s += self.h_dim
        qkw = rotate_queries_or_keys(qk[..., s : s + self.w_dim], pos=w_mask)
        s += self.w_dim
        # Combine rotated dimension
        if s < self.attention_head_size:
            qkr = qk[..., s:]
            qk = torch.cat([qkd, qkh, qkw, qkr], dim=-1)
        else:
            qk = torch.cat([qkd, qkh, qkw], dim=-1)
        return qk

    def forward(
        self,
        hidden_states,
        position_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        head_mask: Optional[torch.Tensor] = None,
    ) -> Union[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor]]:
        batch_size, seq_length, _ = hidden_states.shape
        query_layer = (
            self.query(hidden_states)
            .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )
        key_layer = (
            self.key(hidden_states)
            .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )
        value_layer = (
            self.value(hidden_states)
            .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )

        pos_ids = self.get_position_ids(hidden_states, masks=position_mask)
        key_layer = self.apply_rotary_embeddings(key_layer, pos_ids)
        query_layer = self.apply_rotary_embeddings(query_layer, pos_ids)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        context_layer, attention_probs = attention_interface(
            self,
            query_layer,
            key_layer,
            value_layer,
            head_mask,
            is_causal=self.is_causal,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.dropout_prob,
        )

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = self.proj(context_layer.reshape(new_context_layer_shape))

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


# Adapted from transformers.models.beit.modeling_dinov2.drop_path
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


# Adapted from transformers.models.beit.modeling_beit.BeitDropPath
class VJEPA2DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"p={self.drop_prob}"


class VJEPA2MLP(nn.Module):
    def __init__(self, config: VJEPA2Config, hidden_size: int = 1024, mlp_ratio: float = 4.0):
        super().__init__()
        in_features = out_features = hidden_size
        hidden_features = int(hidden_size * mlp_ratio)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.activation = ACT2FN[config.hidden_act]
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.fc1(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.fc2(hidden_state)
        return hidden_state


class VJEPA2Layer(GradientCheckpointingLayer):
    """This corresponds to the Block class in the original implementation."""

    def __init__(
        self,
        config: VJEPA2Config,
        drop_path_rate: float = 0.0,
        hidden_size: int = 1024,
        num_attention_heads: int = 16,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.attention = VJEPA2RopeAttention(config, hidden_size, num_attention_heads)
        self.drop_path = VJEPA2DropPath(drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.mlp = VJEPA2MLP(config, hidden_size=hidden_size, mlp_ratio=mlp_ratio)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        # Self-Attention
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        self_attention_outputs = self.attention(
            hidden_states,
            position_mask=position_mask,  # position mask for context/target selection
            head_mask=head_mask,  # head mask is applied at F.scaled_dot_product_attention
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        hidden_states = self.drop_path(attention_output) + residual

        # MLP
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.drop_path(hidden_states) + residual

        # Add self attentions if we output attention weights
        outputs = self_attention_outputs[1:]
        outputs = (hidden_states,) + outputs

        return outputs


class VJEPA2Encoder(nn.Module):
    def __init__(self, config: VJEPA2Config):
        super().__init__()
        self.config = config

        self.embeddings = VJEPA2Embeddings(config, hidden_size=config.hidden_size)
        drop_path_rates = [
            (config.drop_path_rate * i / (config.num_hidden_layers - 1) if config.num_hidden_layers > 1 else 0.0)
            for i in range(config.num_hidden_layers)
        ]
        self.layer = nn.ModuleList(
            [
                VJEPA2Layer(
                    config,
                    drop_path_rate=drop_path_rates[i],
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    mlp_ratio=config.mlp_ratio,
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False

    @can_return_tuple
    def forward(
        self,
        pixel_values_videos: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> BaseModelOutput:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        hidden_states = self.embeddings(pixel_values_videos)

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(hidden_states, None, layer_head_mask, output_attentions)
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        hidden_states = self.layernorm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


def apply_masks(tensor: torch.Tensor, masks: list[torch.Tensor]) -> torch.Tensor:
    """
    Args:
        tensor (`torch.Tensor`):
            Tensor of shape [batch_size, num_patches, feature_dim]
        masks (`List[torch.Tensor]`):
            List of tensors of shape [batch_size, num_patches] containing indices of patches to keep
    """
    all_masked_tensors = []
    for mask in masks:
        mask = mask.to(tensor.device)
        mask_keep = mask.unsqueeze(-1).repeat(1, 1, tensor.size(-1))
        all_masked_tensors += [torch.gather(tensor, dim=1, index=mask_keep)]

    return torch.cat(all_masked_tensors, dim=0)


class VJEPA2PredictorEmbeddings(nn.Module):
    """
    Construct mask token, position and patch embeddings.
    """

    def __init__(self, config: VJEPA2Config):
        super().__init__()

        self.config = config
        self.predictor_embeddings = nn.Linear(config.hidden_size, config.pred_hidden_size)
        self.num_mask_tokens = 0
        self.zero_init_mask_tokens = config.pred_zero_init_mask_tokens
        self.num_mask_tokens = config.pred_num_mask_tokens
        self.mask_tokens = nn.Parameter(torch.zeros(self.num_mask_tokens, 1, 1, config.pred_hidden_size))

        self.patch_size = config.patch_size
        self.config = config

    @staticmethod
    def num_patches(config):
        if config.frames_per_clip > 1:
            return (
                (config.frames_per_clip // config.tubelet_size)
                * (config.crop_size // config.patch_size)
                * (config.crop_size // config.patch_size)
            )
        else:
            return (config.crop_size // config.patch_size) * (config.crop_size // config.patch_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        context_mask: list[torch.Tensor],
        target_mask: list[torch.Tensor],
        mask_index: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        hidden_states : encoder outputs (context)
        context_mask: tokens of the context (outputs from the encoder)
        target_mask: tokens to predict
        mask_index: index of the target mask to choose (useful for multiclip?)
        """

        B = hidden_states.size(0)
        context = self.predictor_embeddings(hidden_states)

        # Make target tokens
        mask_index = mask_index % self.num_mask_tokens
        target = self.mask_tokens[mask_index]

        # Note: this is problematic if the config isn't initialized with the right frames_per_clip value,
        # e.g. for scenarios if we want to run predictor for more tokens than in the config.
        # target = target.repeat(B, self.num_patches(self.config), 1)
        # Remedy: use the provided target mask to get the max patch num
        max_patch_num = target_mask[0].max() + 1  # one extra to include the last patch
        target = target.repeat(B, max_patch_num, 1)
        target = apply_masks(target, target_mask)

        # Concatenate context & target tokens
        context = context.repeat(len(context_mask), 1, 1)
        embeddings = torch.cat([context, target], dim=1)

        # Positions of context & target tokens
        cm = torch.cat(context_mask, dim=0)
        tm = torch.cat(target_mask, dim=0)
        masks = torch.cat([cm, tm], dim=1)

        return embeddings, masks


class VJEPA2Predictor(nn.Module):
    def __init__(self, config: VJEPA2Config):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        self.embeddings = VJEPA2PredictorEmbeddings(config)
        drop_path_rates = [
            (
                config.drop_path_rate * i / (config.pred_num_hidden_layers - 1)
                if config.pred_num_hidden_layers > 1
                else 0.0
            )
            for i in range(config.pred_num_hidden_layers)
        ]
        self.layer = nn.ModuleList(
            [
                VJEPA2Layer(
                    config,
                    drop_path_rate=drop_path_rates[i],
                    hidden_size=config.pred_hidden_size,
                    num_attention_heads=config.pred_num_attention_heads,
                    mlp_ratio=config.pred_mlp_ratio,
                )
                for i in range(config.pred_num_hidden_layers)
            ]
        )
        self.layernorm = nn.LayerNorm(config.pred_hidden_size, eps=config.layer_norm_eps)
        self.proj = nn.Linear(config.pred_hidden_size, config.hidden_size, bias=True)

    def sort_tokens(self, hidden_states, position_masks, argsort, head_mask=None):
        # gather position masks
        argsort = argsort.to(position_masks.device)
        position_masks = torch.gather(position_masks, dim=1, index=argsort)

        # gather hidden states
        argsort = argsort.to(hidden_states.device)
        hidden_states_argsort = argsort.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
        hidden_states = torch.gather(hidden_states, dim=1, index=hidden_states_argsort)

        # gather head mask
        if head_mask is not None and head_mask[0] is not None:
            argsort = argsort.to(head_mask.device)
            head_mask = head_mask.permute(1, 0, 2, 3, 4)
            argsort_4d = (
                argsort.unsqueeze(1)
                .unsqueeze(1)
                .expand(-1, head_mask.size(1), head_mask.size(2), -1)
                .unsqueeze(-1)
                .expand(-1, -1, -1, -1, head_mask.size(-1))
            )
            head_mask = torch.gather(head_mask, dim=3, index=argsort_4d)
            argsort_5d = (
                argsort.unsqueeze(1)
                .unsqueeze(1)
                .unsqueeze(1)
                .expand(-1, head_mask.size(1), head_mask.size(2), head_mask.size(3), -1)
            )
            head_mask = torch.gather(head_mask, dim=4, index=argsort_5d)
            head_mask = head_mask.permute(1, 0, 2, 3, 4)

        return hidden_states, position_masks, head_mask

    def unsort_tokens(self, hidden_states, argsort):
        argsort = argsort.to(hidden_states.device)
        reverse_argsort = torch.argsort(argsort, dim=1)
        reverse_argsort = reverse_argsort.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
        hidden_states = torch.gather(hidden_states, dim=1, index=reverse_argsort)
        return hidden_states

    @can_return_tuple
    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        context_mask: list[torch.Tensor],
        target_mask: list[torch.Tensor],
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> BaseModelOutput:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # mask out the encoder hidden states
        # this is implemented here as in VJEPA training a separate encoder is used for target
        encoder_hidden_states = apply_masks(encoder_hidden_states, context_mask)
        _, N_ctxt, D = encoder_hidden_states.shape
        hidden_states, position_masks = self.embeddings(encoder_hidden_states, context_mask, target_mask)

        # Put tokens in sorted order
        argsort = torch.argsort(position_masks, dim=1)  # [B, N]
        hidden_states, position_masks, head_mask = self.sort_tokens(hidden_states, position_masks, argsort, head_mask)

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(hidden_states, position_masks, layer_head_mask, output_attentions)
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.layernorm(hidden_states)
        # unsort and extract the predicted tokens
        hidden_states = self.unsort_tokens(hidden_states, argsort)
        hidden_states = hidden_states[:, N_ctxt:]
        # projection
        hidden_states = self.proj(hidden_states)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class VJEPA2PoolerSelfAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: VJEPA2Config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.is_causal = False

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, seq_length, embed_dim = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            queries,
            keys,
            values,
            attention_mask,
            is_causal=self.is_causal,
            scaling=self.scale,
            dropout=0.0 if not self.training else self.dropout,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, embed_dim).contiguous()
        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


class VJEPA2PoolerCrossAttention(nn.Module):
    """It's different from other cross-attention layers, doesn't have output projection layer (o_proj)"""

    # in case of modular refactoring - o_proj can be replaces with nn.Identity()

    def __init__(self, config: VJEPA2Config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.is_causal = False

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, q_seq_length, embed_dim = queries.shape
        kv_seq_length = keys.shape[1]

        queries = self.q_proj(queries)
        keys = self.k_proj(keys)
        values = self.v_proj(values)

        queries = queries.view(batch_size, q_seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, kv_seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, kv_seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            queries,
            keys,
            values,
            attention_mask,
            is_causal=self.is_causal,
            scaling=self.scale,
            dropout=0.0 if not self.training else self.dropout,
        )

        attn_output = attn_output.reshape(batch_size, q_seq_length, embed_dim).contiguous()

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


# Modified from SiglipEncoderLayer, but we have to propagate proper hidden_size to VJEPA2MLP
class VJEPA2PoolerSelfAttentionLayer(GradientCheckpointingLayer):
    def __init__(self, config: VJEPA2Config):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = VJEPA2PoolerSelfAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = VJEPA2MLP(config, hidden_size=config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> tuple[torch.Tensor, ...]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(batch, 1, q_len, k_v_seq_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class VJEPA2PoolerCrossAttentionLayer(GradientCheckpointingLayer):
    def __init__(self, config: VJEPA2Config):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.cross_attn = VJEPA2PoolerCrossAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = VJEPA2MLP(config, hidden_size=config.hidden_size)

    def forward(
        self,
        queries: torch.Tensor,
        hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        # Apply cross-attention
        residual = queries
        hidden_state = self.layer_norm1(hidden_state)
        hidden_state, *attn_weights = self.cross_attn(
            queries,
            hidden_state,
            hidden_state,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_state = residual + hidden_state

        # Apply MLP
        residual = hidden_state
        hidden_state = self.layer_norm2(hidden_state)
        hidden_state = self.mlp(hidden_state)
        hidden_state = residual + hidden_state

        outputs = (hidden_state,)
        if output_attentions:
            outputs += tuple(attn_weights)

        return outputs


class VJEPA2AttentivePooler(nn.Module):
    """Attentive Pooler"""

    def __init__(self, config: VJEPA2Config):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.cross_attention_layer = VJEPA2PoolerCrossAttentionLayer(config)
        self.self_attention_layers = nn.ModuleList(
            [VJEPA2PoolerSelfAttentionLayer(config) for _ in range(config.num_pooler_layers)]
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        for layer in self.self_attention_layers:
            hidden_state = layer(hidden_state, attention_mask=None)[0]
        queries = self.query_tokens.repeat(hidden_state.shape[0], 1, 1)
        hidden_state = self.cross_attention_layer(queries, hidden_state)[0]
        return hidden_state.squeeze(1)


@auto_docstring
class VJEPA2PreTrainedModel(PreTrainedModel):
    config: VJEPA2Config
    base_model_prefix = "vjepa2"
    main_input_name = "pixel_values_videos"
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "VJEPA2Layer",
        "VJEPA2PoolerSelfAttentionLayer",
        "VJEPA2PoolerCrossAttentionLayer",
        "VJEPA2PredictorEmbeddings",
    ]
    _supports_sdpa = True
    _supports_flash_attn = True

    def _init_weights(self, module):
        """Initialize the weights"""

        init_std = self.config.initializer_range

        # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
        # `trunc_normal_cpu` not implemented in `half` issues
        def trunc_normal_f32_(weight, std):
            data_float_32 = weight.data.to(torch.float32)
            data_init = nn.init.trunc_normal_(data_float_32, mean=0.0, std=std)
            weight.data = data_init.to(weight.dtype)

        if isinstance(module, VJEPA2AttentivePooler):
            trunc_normal_f32_(module.query_tokens, std=init_std)
            for i, layer in enumerate(module.self_attention_layers, 1):
                std = init_std / (i**0.5)
                trunc_normal_f32_(layer.self_attn.out_proj.weight, std=std)
                trunc_normal_f32_(layer.mlp.fc2.weight, std=std)
            std = init_std / (len(module.self_attention_layers) + 1) ** 0.5
            trunc_normal_f32_(module.cross_attention_layer.mlp.fc2.weight, std=std)
        elif isinstance(module, VJEPA2PredictorEmbeddings):
            if module.zero_init_mask_tokens:
                module.mask_tokens.data.zero_()
            else:
                trunc_normal_f32_(module.mask_tokens, std=init_std)
        elif isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            trunc_normal_f32_(module.weight, std=init_std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


def _convert_head_mask_to_5d(head_mask, num_hidden_layers):
    """
    Inputs:
        - head_mask: bsz x seq_length x seq_length | None
    Returns
        - [num_hidden_layers x batch x num_heads x seq_length x seq_length] | [num_hidden_layers]
    """
    if head_mask is not None:
        head_mask = head_mask.unsqueeze(1).unsqueeze(0)
        head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
    else:
        head_mask = [None] * num_hidden_layers
    return head_mask


@auto_docstring
class VJEPA2Model(VJEPA2PreTrainedModel):
    def __init__(self, config: VJEPA2Config):
        super().__init__(config)
        self.config = config

        self.encoder = VJEPA2Encoder(config)
        self.predictor = VJEPA2Predictor(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> VJEPA2PatchEmbeddings3D:
        return self.encoder.embeddings.patch_embeddings

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values_videos: torch.Tensor,
        context_head_mask: Optional[torch.Tensor] = None,
        context_mask: Optional[list[torch.Tensor]] = None,
        target_head_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[list[torch.Tensor]] = None,
        skip_predictor: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> VJEPA2WithMaskedInputModelOutput:
        r"""
        context_head_mask (`torch.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
            The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard) for the context.
        context_mask (`torch.Tensor` with shape `[batch_size, patch_size, 1]`, *optional*):
            The mask position ids indicating which encoder output patches are going to be exposed to the predictor.
            By default, this mask is created as torch.arange(N).unsqueeze(0).repeat(B,1), indicating full context
            available to the predictor.
        target_head_mask (`torch.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
            The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard) for the target.
        target_mask (`torch.Tensor` with shape `[batch_size, patch_size, 1]`, *optional*):
            The mask position ids indicating which encoder output patches are going to be used as a prediction target
            for the predictor. By default, this mask is created as torch.arange(N).unsqueeze(0).repeat(B,1), indicating
            that the predictor should predict all encoder patches.
        skip_predictor (bool):
            flag to skip the predictor forward, useful if you just need the encoder outputs
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if pixel_values_videos is None:
            raise ValueError("You have to specify pixel_values_videos")

        # Prepare head mask if needed
        context_head_mask = _convert_head_mask_to_5d(context_head_mask, self.config.num_hidden_layers)
        target_head_mask = _convert_head_mask_to_5d(target_head_mask, self.config.pred_num_hidden_layers)

        encoder_outputs: BaseModelOutput = self.encoder(
            pixel_values_videos=pixel_values_videos,
            head_mask=context_head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs.last_hidden_state

        if context_mask is None and target_mask is None:
            B = pixel_values_videos.size(0)
            N = sequence_output.size(1)  # ensure we are using dynamic patch size
            context_mask = [torch.arange(N, device=pixel_values_videos.device).unsqueeze(0).repeat((B, 1))]
            target_mask = [torch.arange(N, device=pixel_values_videos.device).unsqueeze(0).repeat((B, 1))]

        if not skip_predictor:
            predictor_outputs: BaseModelOutput = self.predictor(
                encoder_hidden_states=sequence_output,
                context_mask=context_mask,
                target_mask=target_mask,
                head_mask=target_head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            predictor_output = VJEPA2WithMaskedInputPredictorOutput(
                last_hidden_state=predictor_outputs.last_hidden_state,
                target_hidden_state=apply_masks(sequence_output, target_mask),
                hidden_states=predictor_outputs.hidden_states,
                attentions=predictor_outputs.attentions,
            )
        else:
            predictor_output = None

        encoder_output = VJEPA2WithMaskedInputModelOutput(
            last_hidden_state=sequence_output,
            masked_hidden_state=apply_masks(sequence_output, context_mask),
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            predictor_output=predictor_output,
        )

        return encoder_output

    def get_vision_features(self, pixel_values_videos) -> torch.Tensor:
        encoder_output = self.forward(pixel_values_videos, skip_predictor=True)
        return encoder_output.last_hidden_state


@auto_docstring(
    custom_intro="""
    V-JEPA 2 Model transformer with a video classification head on top (a linear layer on top of the attentive pooler).
    """
)
class VJEPA2ForVideoClassification(VJEPA2PreTrainedModel):
    def __init__(self, config: VJEPA2Config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.vjepa2 = VJEPA2Model(config)

        # Classifier head
        self.pooler = VJEPA2AttentivePooler(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels, bias=True)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values_videos: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Examples:

        ```python
        >>> import torch
        >>> import numpy as np
        >>> from transformers import AutoVideoProcessor, VJEPA2ForVideoClassification

        >>> device = "cuda"

        >>> video_processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc16-256-ssv2")
        >>> model = VJEPA2ForVideoClassification.from_pretrained("facebook/vjepa2-vitl-fpc16-256-ssv2").to(device)

        >>> video = np.ones((64, 256, 256, 3))  # 64 frames, 256x256 RGB
        >>> inputs = video_processor(video, return_tensors="pt").to(device)

        >>> # For inference
        >>> with torch.no_grad():
        ...     outputs = model(**inputs)
        >>> logits = outputs.logits

        >>> predicted_label = logits.argmax(-1).item()
        >>> print(model.config.id2label[predicted_label])

        >>> # For training
        >>> labels = torch.ones(1, dtype=torch.long, device=device)
        >>> loss = model(**inputs, labels=labels).loss

        ```"""

        outputs = self.vjepa2(
            pixel_values_videos=pixel_values_videos,
            skip_predictor=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = outputs.last_hidden_state
        pooler_output = self.pooler(last_hidden_state)
        logits = self.classifier(pooler_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(pooled_logits=logits, labels=labels, config=self.config)

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["VJEPA2Model", "VJEPA2PreTrainedModel", "VJEPA2ForVideoClassification"]
