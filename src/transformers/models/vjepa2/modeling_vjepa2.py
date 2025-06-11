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
from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import nn

from ...activations import ACT2FN
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, dataclass
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...utils import ModelOutput, auto_docstring, can_return_tuple, logging
from .configuration_vjepa2 import VJEPA2Config


logger = logging.get_logger(__name__)


@dataclass
class VJEPA2WithMaskedInputPredictorOutput(ModelOutput):
    """
    VJEPA Predictor outputs that also contains the masked encoder outputs

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        masked_hidden_state (`torch.FloatTensor`), *optional*, returned when `context_mask` is provided which is applied on VJEPA2Encoder outputs
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        target_hidden_state (`torch.FloatTensor`), *optional*):
            Returned when `target_mask` is provided which is applied on VJEPA2Encoder outputs.
    """

    last_hidden_state: torch.FloatTensor
    masked_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    target_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class VJEPA2WithMaskedInputModelOutput(ModelOutput):
    """
    VJEPA outputs that also contains the masked encoder outputs
    Optionally contains the predictor outputs

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        masked_hidden_state (`torch.FloatTensor`), *optional*):
            Returned when `context_mask` is provided which is applied on VJEPA2Encoder outputs.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        predictor_output (`VJEPA2WithMaskedInputPredictorOutput`, *optional*):
            Returns the output from the Predictor module
    """

    last_hidden_state: torch.FloatTensor
    masked_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
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
    freq = torch.einsum("..., f -> ... f", pos, omega)  # (..., N, D/2), outer product

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

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

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
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        pos_ids = self.get_position_ids(hidden_states, masks=position_mask)
        key_layer = self.apply_rotary_embeddings(key_layer, pos_ids)
        query_layer = self.apply_rotary_embeddings(query_layer, pos_ids)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and output_attentions:
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
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
        return "p={}".format(self.drop_prob)


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
    ) -> Tuple[torch.Tensor, ...]:
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


def apply_masks(x, masks) -> torch.Tensor:
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors of shape [B, K] containing indices of K patches in [N] to keep
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]

    return torch.cat(all_x, dim=0)


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
        context_mask: List[torch.Tensor],
        target_mask: List[torch.Tensor],
        mask_index: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        position_masks = torch.gather(position_masks, dim=1, index=argsort)
        hidden_states = torch.gather(
            hidden_states,
            dim=1,
            index=argsort.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1)),
        )
        if head_mask is not None and head_mask[0] is not None:
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
        reverse_argsort = torch.argsort(argsort, dim=1)
        hidden_states = torch.gather(
            hidden_states,
            dim=1,
            index=reverse_argsort.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1)),
        )
        return hidden_states

    @can_return_tuple
    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        context_mask: List[torch.Tensor],
        target_mask: List[torch.Tensor],
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
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


@auto_docstring
class VJEPA2PreTrainedModel(PreTrainedModel):
    config_class = VJEPA2Config
    base_model_prefix = "vjepa2"
    main_input_name = "pixel_values_videos"
    supports_gradient_checkpointing = True
    _no_split_modules = ["VJEPA2Layer"]
    _supports_sdpa = True
    _supports_flash_attn_2 = True

    def _init_weights(
        self,
        module: Union[
            nn.Linear,
            nn.Conv2d,
            nn.LayerNorm,
            VJEPA2Embeddings,
            VJEPA2PredictorEmbeddings,
        ],
    ):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, VJEPA2PredictorEmbeddings):
            if not module.zero_init_mask_tokens:
                module.mask_token = nn.init.trunc_normal_(
                    module.mask_token.to(torch.float32),
                    mean=0.0,
                    std=self.config.initializer_range,
                ).to(module.mask_token.dtype)
            else:
                module.mask_tokens.data.zero_()


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
        context_mask: Optional[List[torch.Tensor]] = None,
        target_head_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[List[torch.Tensor]] = None,
        skip_predictor: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> VJEPA2WithMaskedInputModelOutput:
        r"""
        pixel_values_videos (`torch.Tensor` with shape `[batch size x num_frames x num_channels x height x width]`, required):
            The input video pixels which is processed by VJEPA2VideoProcessor.
        context_head_mask (`torch.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
            The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard) for the context.
        target_head_mask (`torch.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
            The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard) for the target.
        context_mask (`torch.Tensor` with shape `[batch_size, patch_size, 1]`, *optional*):
            The mask position ids indicating which encoder output patches are going to be exposed to the predictor.
            By default, this mask is created as torch.arange(N).unsqueeze(0).repeat(B,1), indicating full context
            available to the predictor.
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
        encoder_output = self.forward(pixel_values_videos)
        return encoder_output.last_hidden_state


__all__ = ["VJEPA2Model", "VJEPA2PreTrainedModel"]
