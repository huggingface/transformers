# coding=utf-8
# Copyright 2023 Meta AI and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch DepthPro model."""

from icecream import ic

import collections.abc
import math
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
from torch import nn
from dataclasses import dataclass

from ...utils import ModelOutput
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput, DepthEstimatorOutput
)
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
    torch_int,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from .configuration_depth_pro import DepthProConfig


logger = logging.get_logger(__name__)


# Copied from transformers.models.dinov2.modeling_dinov2.Dinov2PatchEmbeddings with Dinov2->DepthProViT
class DepthProViTPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        num_channels = pixel_values.shape[1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings


# Copied from transformers.models.dinov2.modeling_dinov2.DepthProViTEmbeddings
# with DepthProViT->DepthProViT and antialias=True in interpolation
class DepthProViTEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.
    """

    def __init__(self, config: DepthProConfig) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.patch_embeddings = DepthProViTPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.patch_size = config.patch_size
        self.config = config

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing and interpolation at torch.float32 precision.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """

        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embeddings

        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = torch_int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        target_dtype = patch_pos_embed.dtype

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.to(torch.float32),
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
            antialias=True, # except for this, the class is same as transformers.models.dinov2.modeling_dinov2.DepthProPatchEmbeddings
        ).to(dtype=target_dtype)

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = pixel_values.shape
        target_dtype = self.patch_embeddings.projection.weight.dtype
        embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)

        embeddings = self.dropout(embeddings)

        return embeddings


# Copied from transformers.models.vit.modeling_vit.ViTSelfAttention with ViT->DepthProViT
class DepthProViTSelfAttention(nn.Module):
    def __init__(self, config: DepthProConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


# Copied from transformers.models.dinov2.modeling_dinov2.Dinov2SelfAttention with Dinov2->DepthProViT
class DepthProViTSdpaSelfAttention(DepthProViTSelfAttention):
    def __init__(self, config: DepthProConfig) -> None:
        super().__init__(config)
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "DepthProModel is using DepthProViTSdpaSelfAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states, head_mask=head_mask, output_attentions=output_attentions
            )

        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        context_layer = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            head_mask,
            self.attention_probs_dropout_prob if self.training else 0.0,
            is_causal=False,
            scale=None,
        )

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer, None


# Copied from transformers.models.vit.modeling_vit.ViTSelfOutput with ViT->DepthProViT
class DepthProViTSelfOutput(nn.Module):
    """
    The residual connection is defined in DepthProViTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: DepthProConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTAttention with ViT->DepthProViT
class DepthProViTAttention(nn.Module):
    def __init__(self, config: DepthProConfig) -> None:
        super().__init__()
        self.attention = DepthProViTSelfAttention(config)
        self.output = DepthProViTSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTSdpaAttention with ViT->DepthProViT
class DepthProViTSdpaAttention(DepthProViTAttention):
    def __init__(self, config: DepthProConfig) -> None:
        super().__init__(config)
        self.attention = DepthProViTSdpaSelfAttention(config)


# Copied from transformers.models.dinov2.modeling_dinov2.Dinov2SdpaAttention with Dinov2->DepthProViT
class DepthProViTLayerScale(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.lambda1 = nn.Parameter(config.layerscale_value * torch.ones(config.hidden_size))

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return hidden_state * self.lambda1


# Copied from transformers.models.beit.modeling_beit.drop_path
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


# Copied from transformers.models.beit.modeling_beit.BeitDropPath
class DepthProViTDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


# Copied from transformers.models.dinov2.modeling_dinov2.Dinov2MLP with Dinov2->DepthProViT
class DepthProViTMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        in_features = out_features = config.hidden_size
        hidden_features = int(config.hidden_size * config.mlp_ratio)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        if isinstance(config.hidden_act, str):
            self.activation = ACT2FN[config.hidden_act]
        else:
            self.activation = config.hidden_act
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.fc1(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.fc2(hidden_state)
        return hidden_state


# Copied from transformers.models.dinov2.modeling_dinov2.Dinov2SwiGLUFFN with Dinov2->DepthProViT
class DepthProViTSwiGLUFFN(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        in_features = out_features = config.hidden_size
        hidden_features = int(config.hidden_size * config.mlp_ratio)
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8

        self.weights_in = nn.Linear(in_features, 2 * hidden_features, bias=True)
        self.weights_out = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.weights_in(hidden_state)
        x1, x2 = hidden_state.chunk(2, dim=-1)
        hidden = nn.functional.silu(x1) * x2
        return self.weights_out(hidden)


DEPTHPROVIT_ATTENTION_CLASSES = {
    "eager": DepthProViTAttention,
    "sdpa": DepthProViTSdpaAttention,
}


# Copied from transformers.models.dinov2.modeling_dinov2.Dinov2Layer with Dinov2->DepthProViT
class DepthProViTLayer(nn.Module):
    """This corresponds to the Block class in the original implementation."""

    def __init__(self, config: DepthProConfig) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = DEPTHPROVIT_ATTENTION_CLASSES[config._attn_implementation](config)
        self.layer_scale1 = DepthProViTLayerScale(config)
        self.drop_path = DepthProViTDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        if config.use_swiglu_ffn:
            self.mlp = DepthProViTSwiGLUFFN(config)
        else:
            self.mlp = DepthProViTMLP(config)
        self.layer_scale2 = DepthProViTLayerScale(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.norm1(hidden_states),  # in DepthProViT, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]

        attention_output = self.layer_scale1(attention_output)
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = self.drop_path(attention_output) + hidden_states

        # in DepthProViT, layernorm is also applied after self-attention
        layer_output = self.norm2(hidden_states)
        layer_output = self.mlp(layer_output)
        layer_output = self.layer_scale2(layer_output)

        # second residual connection
        layer_output = self.drop_path(layer_output) + hidden_states

        outputs = (layer_output,) + outputs

        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTEncoder with ViT->DepthProViT
class DepthProViTEncoder(nn.Module):
    def __init__(self, config: DepthProConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([DepthProViTLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class DepthProViT(nn.Module):
    def __init__(self, config: DepthProConfig):
        super().__init__()
        self.config = config

        self.embeddings = DepthProViTEmbeddings(config)
        self.encoder = DepthProViTEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        embedding_output = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        if not return_dict:
            head_outputs = (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class DepthProUpsampleBlock(nn.Module):
    def __init__(
            self,
            input_dims,
            intermediate_dims,
            output_dims,
            n_upsample_layers,
            use_proj=True,
            bias=False,
        ) -> None:
        super().__init__()

        # create first projection block
        if use_proj:
            self.proj = nn.Conv2d(
                in_channels=input_dims,
                out_channels=intermediate_dims,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            )
        else:
            self.proj = nn.Identity()

        # create following upsample blocks
        self.upsample_blocks = nn.Sequential()
        for i in range(n_upsample_layers):
            in_channels = intermediate_dims if i == 0 else output_dims
            layer = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=output_dims,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=bias,
            )
            self.upsample_blocks.append(layer)

    def forward(self, features):
        projected = self.proj(features)
        return self.upsample_blocks(projected)

class DepthProEncoder(nn.Module):
    def __init__(self, config: DepthProConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.decoder_hidden_size = config.decoder_hidden_size
        self.patch_encoder_hook_ids = config.patch_encoder_hook_ids
        self.intermediate_feature_dims = config.intermediate_feature_dims
        self.intermediate_upsample_layers = config.intermediate_upsample_layers
 
        self.out_size = 24 # TODO: image_size // patch_size

        # patch encoder
        self.patch_encoder = DepthProViT(config)

        # image encoder
        self.image_encoder = DepthProViT(config)

        # upsampling intermediate features - (1-2) in diagram
        self.upsample_intermediate = nn.ModuleList()
        for i, (feature_dims, upsample_layers) in enumerate(zip(
            self.intermediate_feature_dims,
            self.intermediate_upsample_layers,
        )):
            intermediate_dims = self.decoder_hidden_size if i == 0 else feature_dims
            upsample_block = DepthProUpsampleBlock(
                input_dims=config.hidden_size,
                intermediate_dims=intermediate_dims,
                output_dims=feature_dims,
                n_upsample_layers=upsample_layers,
            )
            self.upsample_intermediate.append(upsample_block)

        # upsampling patch features (high_res, med_res, low_res) - (3-5) in diagram
        self.upsample_high_res = DepthProUpsampleBlock(
            input_dims=config.hidden_size,
            intermediate_dims=config.high_res_feature_dims,
            output_dims=config.high_res_feature_dims,
            n_upsample_layers=1,
        )
        self.upsample_med_res = DepthProUpsampleBlock(
            input_dims=config.hidden_size,
            intermediate_dims=config.med_res_feature_dims,
            output_dims=config.med_res_feature_dims,
            n_upsample_layers=1,
        )
        self.upsample_low_res = DepthProUpsampleBlock(
            input_dims=config.hidden_size,
            intermediate_dims=config.low_res_feature_dims,
            output_dims=config.low_res_feature_dims,
            n_upsample_layers=1,
        )

        # upsampling image features - (6) in diagram
        self.upsample_image = DepthProUpsampleBlock(
            input_dims=config.hidden_size,
            intermediate_dims=config.hidden_size,
            output_dims=config.image_feature_dims,
            n_upsample_layers=1,
            use_proj=False,
            bias=True,
        )

    def _interpolate(self, pixel_values, scale_factor):
        return nn.functional.interpolate(
            pixel_values,
            size=None,
            scale_factor=scale_factor,
            mode="bilinear",
            align_corners=False,
        )

    def _patch(self, pixel_values, overlap_ratio):
        patch_size = 384 # TODO: this should be infered
        patch_stride = int(patch_size * (1 - overlap_ratio))

        image_size = pixel_values.shape[-1]
        steps = int(math.ceil((image_size - patch_size) / patch_stride)) + 1

        x_patch_list = []
        for j in range(steps):
            j0 = j * patch_stride
            j1 = j0 + patch_size

            for i in range(steps):
                i0 = i * patch_stride
                i1 = i0 + patch_size
                x_patch_list.append(pixel_values[..., j0:j1, i0:i1])

        return torch.cat(x_patch_list, dim=0)

    def _reshape_feature(
        self, hidden_states: torch.Tensor, width, height, cls_token_offset=1
    ):
        """Discard class token and reshape 1D feature map to a 2D grid."""
        b, hw, c = hidden_states.shape

        # Remove class token.
        if cls_token_offset > 0:
            hidden_states = hidden_states[:, cls_token_offset:, :]

        # Shape: (batch, height, width, dim) -> (batch, dim, height, width)
        hidden_states = hidden_states.reshape(b, height, width, c).permute(0, 3, 1, 2)
        return hidden_states

    def _merge(self, x: torch.Tensor, batch_size: int, padding: int = 3) -> torch.Tensor:
        """Merge the patched input into a image with sliding window."""
        steps = int(math.sqrt(x.shape[0] // batch_size))

        idx = 0

        output_list = []
        for j in range(steps):
            output_row_list = []
            for i in range(steps):
                output = x[batch_size * idx : batch_size * (idx + 1)]

                if j != 0:
                    output = output[..., padding:, :]
                if i != 0:
                    output = output[..., :, padding:]
                if j != steps - 1:
                    output = output[..., :-padding, :]
                if i != steps - 1:
                    output = output[..., :, :-padding]

                output_row_list.append(output)
                idx += 1

            output_row = torch.cat(output_row_list, dim=-1)
            output_list.append(output_row)
        output = torch.cat(output_list, dim=-2)
        return output

    def forward(
        self,
        pixel_values: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = pixel_values.shape[0]

        # STEP 1: create 3-level image

        high_res = pixel_values
        med_res = self._interpolate(pixel_values, 0.5)
        low_res = self._interpolate(pixel_values, 0.25)

        # STEP 2: create patches

        high_res_patches = self._patch(high_res, 0.25)
        med_res_patches = self._patch(med_res, 0.5)
        low_res_patches = low_res

        patches = torch.cat(
            (high_res_patches, med_res_patches, low_res_patches),
            dim=0,
        )

        # STEP 3: apply patch and image encoder

        patch_encodings = self.patch_encoder(
            patches,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=True, # required for intermediate features
            return_dict=True,
        )
        image_encodings = self.image_encoder(
            pixel_values=low_res_patches,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        # STEP 4: get patch features (high_res, med_res, low_res) - (3-5) in diagram

        # a. extract hidden_state
        hidden_state = patch_encodings.last_hidden_state

        # b. reshape back to image like
        features = self._reshape_feature(
            hidden_state, self.out_size, self.out_size
        )
        high_res_features, med_res_features, low_res_features = torch.split(
            features,
            [len(high_res_patches), len(med_res_patches), len(low_res_patches)],
            dim=0,
        )

        # c. merge patches back together
        high_res_features = self._merge(high_res_features, batch_size=batch_size, padding=3)
        med_res_features = self._merge(med_res_features, batch_size=batch_size, padding=6)
        low_res_features = low_res_features # no merge required with low res image

        # d. upsample
        high_res_features = self.upsample_high_res(high_res_features)
        med_res_features = self.upsample_med_res(med_res_features)
        low_res_features = self.upsample_low_res(low_res_features)

        # STEP 5: get intermediate features - (1-2) in diagram

        intermediate_features = []
        for layer_id in self.patch_encoder_hook_ids:
            
            # a. extract hidden_state
            hidden_state = patch_encodings.hidden_states[layer_id+1] # +1 to correct index position as hidden_states contain embedding output as well

            # b. reshape back to image like
            features = self._reshape_feature(
                hidden_state,
                self.out_size,
                self.out_size,
            )

            # c. merge patches back together
            features = self._merge(
                features[: batch_size * 5 * 5], batch_size=batch_size, padding=3
            )

            # d. upsample
            features = self.upsample_intermediate[layer_id](features)

            intermediate_features.append(features)

        # STEP 6: get image features - (6) in diagram

        # a. extract hidden_state
        hidden_state = image_encodings.last_hidden_state

        # b. reshape back to image like
        image_features = self._reshape_feature(
            hidden_state, self.out_size, self.out_size
        )

        # c. merge patches back together
        # skipped, no merge required with low res image

        # d. upsample
        image_features = self.upsample_image(image_features)

        # STEP 7: return these features
        last_hidden_state =  [
            *intermediate_features,
            high_res_features,
            med_res_features,
            low_res_features,
            image_features,
        ]

        hidden_states = patch_encodings.hidden_states + image_encodings.hidden_states if output_hidden_states else None
        attentions = patch_encodings.attentions + image_encodings.attentions if output_attentions else None

        if not return_dict:
            return tuple(v for v in [last_hidden_state, hidden_states, attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=hidden_states,
            attentions=attentions,
        )


class DepthProFOVModel(nn.Module):
    def __init__(self, config: DepthProConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.decoder_hidden_size = config.decoder_hidden_size

        self.encoder = DepthProViT(config)
        self.encoder_neck = nn.Linear(self.hidden_size, self.decoder_hidden_size // 2)
        self.global_neck = nn.Sequential(
            nn.Conv2d(self.decoder_hidden_size, self.decoder_hidden_size // 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True)
        )
        self.head = nn.Sequential(
            nn.Conv2d(self.decoder_hidden_size // 2, self.decoder_hidden_size // 4, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(True),
            nn.Conv2d(self.decoder_hidden_size // 4, self.decoder_hidden_size // 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(self.decoder_hidden_size // 8, 1, kernel_size=6, stride=1, padding=0),
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        global_features: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        pixel_values = nn.functional.interpolate(
            pixel_values,
            size=None,
            scale_factor=0.25,
            mode="bilinear",
            align_corners=False,
        )
        encoder_outputs = self.encoder(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = encoder_outputs[0]

        last_hidden_state = self.encoder_neck(last_hidden_state)

        # TODO: add some comments
        last_hidden_state = last_hidden_state[:, 1:]
        last_hidden_state = last_hidden_state.permute(0, 2, 1)

        global_features = self.global_neck(global_features)

        last_hidden_state = last_hidden_state.reshape_as(global_features)
        last_hidden_state = last_hidden_state + global_features
        fov_output = self.head(last_hidden_state)
        fov_output = fov_output.reshape(1)

        if not return_dict:
            head_outputs = (fov_output,)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=fov_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# Copied from transformers.models.dpt.modeling_dpt.DPTPreActResidualLayer DPTPreAct->DepthPro
class DepthProResidualLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.use_batch_norm = config.use_batch_norm_in_decoder
        self.hidden_size = config.decoder_hidden_size

        self.activation1 = nn.ReLU()
        self.convolution1 = nn.Conv2d(
            self.hidden_size,
            self.hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=(not self.use_batch_norm),
        )

        self.activation2 = nn.ReLU()
        self.convolution2 = nn.Conv2d(
            self.hidden_size,
            self.hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=(not self.use_batch_norm),
        )

        if self.use_batch_norm:
            self.batch_norm1 = nn.BatchNorm2d(self.hidden_size)
            self.batch_norm2 = nn.BatchNorm2d(self.hidden_size)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        residual = hidden_state
        hidden_state = self.activation1(hidden_state)

        hidden_state = self.convolution1(hidden_state)

        if self.use_batch_norm:
            hidden_state = self.batch_norm1(hidden_state)

        hidden_state = self.activation2(hidden_state)
        hidden_state = self.convolution2(hidden_state)

        if self.use_batch_norm:
            hidden_state = self.batch_norm2(hidden_state)

        return hidden_state + residual


# Implementation resembles transformers.models.dpt.modeling_dpt.DPTFeatureFusionLayer
class DepthProFeatureFusionLayer(nn.Module):
    def __init__(self, config: DepthProConfig, use_deconv:bool=True) -> None:
        super().__init__()
        self.config = config
        self.use_deconv = use_deconv

        self.residual_layer1 = DepthProResidualLayer(config)
        self.residual_layer2 = DepthProResidualLayer(config)

        if self.use_deconv:
            self.deconv = nn.ConvTranspose2d(
                in_channels=config.decoder_hidden_size,
                out_channels=config.decoder_hidden_size,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=False,
            )

        self.projection = nn.Conv2d(config.decoder_hidden_size, config.decoder_hidden_size, kernel_size=1, bias=True)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, hidden_state, residual=None):
        if residual is not None:
            hidden_state = self.skip_add.add(hidden_state, self.residual_layer1(residual))

        hidden_state = self.residual_layer2(hidden_state)
        if self.use_deconv:
            hidden_state = self.deconv(hidden_state)
        hidden_state = self.projection(hidden_state)

        return hidden_state


# Implementation resembles transformers.models.dpt.modeling_dpt.DPTFeatureFusionStage
class DepthProDecoder(nn.Module):
    def __init__(self, config: DepthProConfig) -> None:
        super().__init__()
        self.config = config

        # for STEP 2: fuse low_res and image features
        self.fuse_image_with_low_res = nn.Conv2d(
            in_channels=config.low_res_feature_dims+config.image_feature_dims,
            out_channels=config.global_feature_dims,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # for STEP 3: apply decoder block for global features
        self.global_proj = nn.Conv2d(
            in_channels=config.global_feature_dims,
            out_channels=config.decoder_hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.global_fusion = DepthProFeatureFusionLayer(config)

        # for STEP 4: apply decoder block for med features
        self.med_res_proj = nn.Conv2d(
            in_channels=config.med_res_feature_dims,
            out_channels=config.decoder_hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.med_res_fusion = DepthProFeatureFusionLayer(config)

        # for STEP 5: apply decoder block for high features
        self.high_res_proj = nn.Conv2d(
            in_channels=config.high_res_feature_dims,
            out_channels=config.decoder_hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.high_res_fusion = DepthProFeatureFusionLayer(config)

        # for STEP 6: apply decoder block for intermediate features
        self.intermediate_proj = nn.Sequential()
        self.intermediate_fusion = nn.Sequential()
        for i, feature_dim in enumerate(config.intermediate_feature_dims):
            if i == 0:
                # no projection for final intermediate layer
                proj = nn.Identity()
                fusion = DepthProFeatureFusionLayer(config, use_deconv=False)
            else:
                proj = nn.Conv2d(
                    in_channels=feature_dim,
                    out_channels=config.decoder_hidden_size,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
                fusion = DepthProFeatureFusionLayer(config)

            self.intermediate_proj.append(proj)
            self.intermediate_fusion.append(fusion)

    def forward(self, hidden_states):

        # STEP 1: extract features

        intermediate_features = hidden_states[:-4]
        # intermediate_features_i.shape: [batch_size, config.intermediate_feature_dims_i, 768, 768], [1, 256, 384, 384]
        high_res_features = hidden_states[-4]
        # high_res_features.shape: [batch_size, config.high_res_feature_dims, 192, 192]
        med_res_features = hidden_states[-3]
        # med_res_features.shape: [batch_size, config.med_res_feature_dims, 96, 96]
        low_res_features = hidden_states[-2]
        # low_res_features.shape: [batch_size, config.low_res_feature_dims, 48, 48]
        image_features = hidden_states[-1]
        # image_features.shape: [batch_size, config.image_feature_dims, 48, 48]

        # STEP 2: fuse low_res and image features

        global_features = torch.cat((low_res_features, image_features), dim=1)
        global_features = self.fuse_image_with_low_res(global_features)
        # global_features.shape: [batch_size, config.global_feature_dims, 48, 48]

        # STEP 3: apply decoder block for global features

        # apply projection: used by fusion now and then fov later
        global_projected = self.global_proj(global_features)
        # apply fusion: used by next projections and fusions
        last_features = self.global_fusion(global_projected)
        # last_features.shape: [batch_size, config.decoder_hidden_size, 96, 96]

        # STEP 4: apply decoder block for med features

        projected = self.med_res_proj(med_res_features)
        last_features = self.med_res_fusion(last_features, projected)
        # last_features.shape: [batch_size, config.decoder_hidden_size, 192, 192]

        # STEP 5: apply decoder block for high features

        projected = self.high_res_proj(high_res_features)
        last_features = self.high_res_fusion(last_features, projected)
        # last_features.shape: [batch_size, config.decoder_hidden_size, 384, 384]

        # STEP 6: apply decoder block for intermediate features

        for (features, proj_layer, fusion_layer) in zip(
            # reversed becuase decoding is applied from last features to first features
            intermediate_features[::-1],
            self.intermediate_proj[::-1],
            self.intermediate_fusion[::-1],
        ):
            projected = proj_layer(features)
            last_features = fusion_layer(last_features, projected)
            # last_features.shape: [batch_size, config.decoder_hidden_size, 768, 768]
            # last_features.shape: [batch_size, config.decoder_hidden_size, 768, 768]

        return last_features, global_projected


class DepthProPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DepthProConfig
    base_model_prefix = "depth_pro"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DepthProViTSwiGLUFFN"]
    _supports_sdpa = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


DEPTH_PRO_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`DepthProConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

DEPTH_PRO_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`DPTImageProcessor.__call__`]
            for details.

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""


@dataclass
class DepthProModelOutput(BaseModelOutput):
    """
    Base class for model's outputs, with potential fov, hidden states and attentions.

    Args:
        fov (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `use_fov_model` is provided):
            Field of View Scaler.
    """
    fov: Optional[torch.FloatTensor] = None


@add_start_docstrings(
    "The bare DepthPro Model transformer outputting raw hidden-states without any specific head on top.",
    DEPTH_PRO_START_DOCSTRING,
)
class DepthProModel(DepthProPreTrainedModel):
    def __init__(self, config, use_fov_model=None):
        super().__init__(config)
        self.config = config
        self.use_fov_model = use_fov_model if use_fov_model is not None else self.config.use_fov_model

        # dinov2 (vit) like encoder
        self.encoder = DepthProEncoder(config)
        # dpt (vit) like decoder
        self.decoder = DepthProDecoder(config)
        # dinov2 (vit) like encoder
        self.fov_model = DepthProFOVModel(config) if self.use_fov_model else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        embeddings = {
            "patch_embeddings": self.encoder.patch_encoder.embeddings.patch_embeddings,
            "image_embeddings": self.encoder.image_encoder.embeddings.patch_embeddings,
        }
        if self.use_fov:
            embeddings['fov_embeddings'] = self.fov_model.embeddings.patch_embeddings
        return embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.patch_encoder.encoder.layer[layer].attention.prune_heads(heads)
            self.encoder.image_encoder.encoder.layer[layer].attention.prune_heads(heads)
            self.fov_model.encoder.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(DEPTH_PRO_INPUTS_DOCSTRING)
    # TODO
    # @add_code_sample_docstrings(
    #     checkpoint=_CHECKPOINT_FOR_DOC,
    #     output_type=BaseModelOutputWithPoolingAndIntermediateActivations,
    #     config_class=_CONFIG_FOR_DOC,
    #     modality="vision",
    #     expected_output=_EXPECTED_OUTPUT_SHAPE,
    # )
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        encodings = self.encoder(
            pixel_values,
            head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        last_hidden_state = encodings.last_hidden_state
        last_hidden_state, global_features = self.decoder(last_hidden_state)

        if self.use_fov_model:
            fov_encodings = self.fov_model(
                pixel_values=pixel_values,
                global_features=global_features.detach(),
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            fov = fov_encodings.last_hidden_state
            attentions = encodings.attentions + fov_encodings.attentions if output_attentions else None
            hidden_states = encodings.hidden_states + fov_encodings.hidden_states if output_hidden_states else None
        else:
            fov = None
            attentions = encodings.attentions
            hidden_states = encodings.hidden_states

        if not return_dict:
            outputs = (last_hidden_state, fov, hidden_states, attentions)
            outputs = (i for i in outputs if i is not None)
            return outputs

        return DepthProModelOutput(
            last_hidden_state=last_hidden_state,
            fov=fov,
            hidden_states=hidden_states,
            attentions=attentions,
        )


class DepthProDepthEstimationHead(nn.Module):
    """
    # TODO
    Output head consisting of 3 convolutional layers. It progressively halves the feature dimension and upsamples
    the predictions to the input resolution after the first convolutional layer (details can be found in the paper's
    supplementary material).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        features = config.decoder_hidden_size
        self.head = nn.Sequential(
            nn.Conv2d(features, features//2, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(
                in_channels=features//2, out_channels=features//2,
                kernel_size=2, stride=2, padding=0, bias=True
            ),
            nn.Conv2d(features//2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )


    def forward(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        predicted_depth = self.head(hidden_states)
        predicted_depth = predicted_depth.squeeze(dim=1)
        return predicted_depth


@dataclass
class DepthProDepthEstimatorOutput(DepthEstimatorOutput):
    """
    Base class for outputs of DepthProDepthEstimator.

    Args:
        fov (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `use_fov_model` is provided):
            Field of View Scaler.
    """
    fov: Optional[torch.FloatTensor] = None


@add_start_docstrings(
    """
    DepthPro Model with a depth estimation head on top (consisting of 3 convolutional layers).
    """,
    DEPTH_PRO_START_DOCSTRING,
)
class DepthProForDepthEstimation(DepthProPreTrainedModel):
    def __init__(self, config, use_fov_model=None):
        super().__init__(config)
        self.use_fov_model = use_fov_model if use_fov_model is not None else self.config.use_fov_model

        self.depth_pro = DepthProModel(config, use_fov_model=self.use_fov_model)
        self.head = DepthProDepthEstimationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(DEPTH_PRO_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=DepthEstimatorOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        head_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth depth estimation maps for computing the loss.

        Returns:

        Examples:
        ```python
        >>> from transformers import AutoImageProcessor, DPTForDepthEstimation
        >>> import torch
        >>> import numpy as np
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("Intel/dpt-large")
        >>> model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

        >>> # prepare image for the model
        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)
        ...     predicted_depth = outputs.predicted_depth

        >>> # interpolate to original size
        >>> prediction = torch.nn.functional.interpolate(
        ...     predicted_depth.unsqueeze(1),
        ...     size=image.size[::-1],
        ...     mode="bicubic",
        ...     align_corners=False,
        ... )

        >>> # visualize the prediction
        >>> output = prediction.squeeze().cpu().numpy()
        >>> formatted = (output * 255 / np.max(output)).astype("uint8")
        >>> depth = Image.fromarray(formatted)
        ```"""
        loss = None
        if labels is not None:
            raise NotImplementedError("Training is not implemented yet")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        depth_pro_outputs = self.depth_pro(
            pixel_values=pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        last_hidden_state = depth_pro_outputs[0]
        predicted_depth = self.head(last_hidden_state)

        if not return_dict:
            if loss is None:
                return (predicted_depth,) + depth_pro_outputs[1:]
            else:
                return (loss, predicted_depth) + depth_pro_outputs[1:]

        return DepthProDepthEstimatorOutput(
            loss=loss,
            predicted_depth=predicted_depth,
            fov=depth_pro_outputs.fov,
            hidden_states=depth_pro_outputs.hidden_states,
            attentions=depth_pro_outputs.attentions,
        )
