# coding=utf-8
# Copyright 2025 Apple Inc., The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch AIMv2 model."""

import collections.abc
import math
from typing import Dict, List, Optional, Set, Tuple, Union, Callable, Any

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import functools

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
    MaskedImageModelingOutput,
)

from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
    torch_int,
)
from .configuration_aimv2 import AIMv2Config



logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "AIMv2Config"

# Base docstring
_CHECKPOINT_FOR_DOC = "apple/aimv2-large-patch14-224"
_EXPECTED_OUTPUT_SHAPE = [1, 256, 1024]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "apple/aimv2-large-patch14-224"
_IMAGE_CLASS_EXPECTED_OUTPUT = "..."

class AIMv2Embeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.
    """

    def __init__(self, config: AIMv2Config) -> None:
        super().__init__()

        self.patch_embeddings = AIMv2PatchEmbeddings(config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if config.use_cls_token else None

        if config.pos_embed_type == "sincos":
            self.pos_embed = AIMv2SinCosPosEmbed(config)
        else:
            num_patches = (config.image_size // config.patch_size) ** 2
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + (1 if config.use_cls_token else 0), config.hidden_size))

        self.initialize_weights(config)

    def initialize_weights(self, config: AIMv2Config) -> None:
        if not config.pos_embed_type == "sincos":
            torch.nn.init.normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=0.02)
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d) following MAE
        w = self.patch_embeddings.projection.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def forward(self, pixel_values: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values)

        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        if callable(self.pos_embed):
            p_h, p_w = self.patch_embeddings.patch_size
            pos_embed = self.pos_embed(height // p_h, width // p_w, embeddings.shape[-1]).unsqueeze(0)
        else:
            pos_embed = self.pos_embed
        
        embeddings = embeddings + pos_embed

        return embeddings

class AIMv2PatchEmbeddings(nn.Module):
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
        self.norm = config.norm_layer(hidden_size) if config.norm_layer is not None else nn.Identity()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        num_channels = pixel_values.shape[1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        embeddings = self.norm(embeddings)
        return embeddings

class AIMv2SinCosPosEmbed(nn.Module):
    def __init__(self, config: AIMv2Config):
        super().__init__()
        self.cls_token = config.use_cls_token

    def forward(self, height: int, width: int, embed_dim: int) -> torch.Tensor:
        assert embed_dim % 2 == 0, embed_dim

        grid_h = torch.arange(height, dtype=torch.float32)
        grid_w = torch.arange(width, dtype=torch.float32)
        grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
        grid = torch.stack(grid, dim=0)
        grid = grid.reshape([2, 1, height, width])

        emb_h = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
        emb_w = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
        pos_embed = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D)
        if self.cls_token:
            pos_embed = torch.cat([torch.zeros([1, embed_dim], device=pos_embed.device), pos_embed], dim=0)
        return pos_embed

    @staticmethod
    def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
        omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
        omega /= embed_dim / 2.0
        omega = 1.0 / (10000**omega)  # (D/2,)

        pos = pos.reshape(-1)  # (M,)
        out = torch.einsum("m, d -> md", pos, omega)

        emb_sin = torch.sin(out)  # (M, D/2)
        emb_cos = torch.cos(out)  # (M, D/2)

        emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
        return emb

class AIMv2Attention(nn.Module):
    def __init__(self, config: AIMv2Config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.qkv = nn.Linear(config.hidden_size, self.all_head_size * 3, bias=config.qkv_bias)
        self.attn_drop = nn.Dropout(config.attention_probs_dropout_prob)
        self.out_proj = nn.Linear(self.all_head_size, config.hidden_size, bias=False)  # Set bias=False
        self.proj_drop = nn.Dropout(config.hidden_dropout_prob)

        self.is_causal = config.is_causal

    def forward(
        self,
        hidden_states,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        print("AIMv2Attention - hidden_states.shape:", hidden_states.shape)

        qkv = self.qkv(hidden_states)
        print("AIMv2Attention - qkv.shape (before view):", qkv.shape)
        print("AIMv2Attention - qkv size :", qkv.size())

        qkv = qkv.view(-1, hidden_states.shape[1], 3, self.num_attention_heads, self.attention_head_size)
        print("AIMv2Attention - qkv.shape (after view):", qkv.shape)

        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_length, head_size)
        print("AIMv2Attention - qkv.shape (after permute):", qkv.shape)

        query_layer, key_layer, value_layer = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attention_output = F.scaled_dot_product_attention(
            query_layer, key_layer, value_layer, attn_mask=attention_mask, is_causal=self.is_causal
        )

        attention_output = attention_output.transpose(1, 2).contiguous().view(-1, hidden_states.shape[1], self.all_head_size)
        attention_output = self.out_proj(attention_output)
        return self.proj_drop(attention_output)



class AIMv2MLP(nn.Module):
    def __init__(self, config: AIMv2Config) -> None:
        super().__init__()
        in_features = out_features = config.hidden_size
        hidden_features = config.intermediate_size

        # Define layers
        self.fc1 = nn.Linear(in_features, hidden_features, bias=config.use_bias)
        if isinstance(config.hidden_act, str):
            self.activation = ACT2FN[config.hidden_act]
        else:
            self.activation = config.hidden_act
        self.fc2 = nn.Linear(hidden_features, out_features, bias=config.use_bias)

        # Dropout layer
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # Forward pass
        hidden_state = self.fc1(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.dropout(hidden_state)  # Apply dropout after activation
        hidden_state = self.fc2(hidden_state)
        hidden_state = self.dropout(hidden_state)  # Apply dropout after output layer
        return hidden_state


class AIMv2SwiGLUFFN(nn.Module):
    def __init__(self, config: AIMv2Config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.fc3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        hidden_gelu = F.gelu(self.fc1(x), approximate="tanh")
        hidden_linear = self.fc3(x)
        hidden_swiglu = hidden_gelu * hidden_linear
        output = self.fc2(hidden_swiglu)
        return self.dropout(output)

class AIMv2Layer(nn.Module):
    """This corresponds to the Block class in the original implementation."""

    def __init__(self, config: AIMv2Config, attn_target: Callable, ffn_target: Callable = AIMv2SwiGLUFFN) -> None:
        super().__init__()
        self.config = config
        self.attention = attn_target(config)

        self.ffn = ffn_target(config)

        if config.norm_layer == nn.LayerNorm:
            self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        print("AIMv2Layer - hidden_states.shape:", hidden_states.shape)
        # Apply pre-norm before attention
        norm_hidden_states = self.norm1(hidden_states)
        attention_output = self.attention(
            norm_hidden_states,
            attention_mask=attention_mask,
        )
        # First residual connection
        hidden_states = hidden_states + attention_output

        # Apply pre-norm before feed-forward
        norm_hidden_states = self.norm2(hidden_states)
        ffn_output = self.ffn(norm_hidden_states)

        # Second residual connection
        hidden_states = hidden_states + ffn_output

        return hidden_states


class AIMv2AverageLayers(nn.Module):
    def __init__(self, config: AIMv2Config):
        super().__init__()
        self.layers = config.probe_layers
        self.reduce = config.reduce if hasattr(config, "reduce") else False
        self.hidden_size = config.hidden_size

    def forward(self, layer_features: List[torch.Tensor]) -> torch.Tensor:
        # layer_features shape: (num_layers, B, N, D)
        feats = torch.stack([layer_features[layer_id] for layer_id in self.layers], dim=0) # (num_layers, B, N, D)
        feats = feats.mean(dim=0) # (B, N, D)
        if self.reduce:
            return feats.mean(dim=1) # (B, D)
        else:
            # Reshape to (B, H, W, D)
            batch_size, num_patches, hidden_dim = feats.shape
            height = width = int(math.sqrt(num_patches))
            feats = feats.reshape(batch_size, height, width, hidden_dim)
            return feats

    @property
    def max_block_id(self) -> int:
        return max(self.layers)

class AIMv2Encoder(nn.Module):
    def __init__(self, config: AIMv2Config):
        super().__init__()
        self.config = config
        # AIM-v2 specific
        self.post_transformer_layer = AIMv2AverageLayers(config)
        if config.norm_layer==nn.RMSNorm:
            norm_layer = functools.partial(
                nn.RMSNorm, eps=config.layer_norm_eps
            )
        else:
            norm_layer = functools.partial(nn.LayerNorm, eps=config.layer_norm_eps)

        def attn_target(config: AIMv2Config) -> nn.Module:
            return AIMv2Attention(config)

        if config.ffn_target_type == "mlp":
            ffn_target = AIMv2MLP
        elif config.ffn_target_type == "swiglu":
            ffn_target = AIMv2SwiGLUFFN
        else:
            raise ValueError(f"Invalid ffn_target_type: {config.ffn_target_type}.")

        self.layer = nn.ModuleList(
            [AIMv2Layer(config, attn_target, ffn_target) for _ in range(config.num_hidden_layers)]
        )

        self.gradient_checkpointing = False

        self.norm = norm_layer(config.hidden_size) if config.post_trunk_norm else None

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

        # only evaluate up to the max block id
        max_block_id = self.post_transformer_layer.max_block_id

        features = []
        print("AIMv2Encoder - hidden_states.shape:", hidden_states.shape)
        for blk_id, blk in enumerate(self.layer):
            print("***for blk_id, blk in enumerate(self.layer):***")
            print("blk_id:" + str(blk_id) + ", blk:" + str(blk))
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(blk),
                    hidden_states,
                    head_mask[blk_id],
                )
            else:
                print("Inside else: ", "  hidden_states.shape -", hidden_states.shape)
                layer_outputs = blk(
                    hidden_states, head_mask[blk_id], output_attentions=output_attentions
                )

            hidden_states = layer_outputs
            print("Inside else: ", "  hidden_states.shape -", hidden_states.shape)

            features.append(hidden_states)
            if blk_id == max_block_id:
                break

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        #hidden_states = self.post_transformer_layer(features)  # passing only features
        hidden_states = self.post_transformer_layer(features)  # B, N, D

        if self.norm is not None:
            hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

class AIMv2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = AIMv2Config
    base_model_prefix = "aimv2"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["AIMv2Layer"]
    _supports_sdpa = True

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, AIMv2Embeddings):
            module.initialize_weights(self.config)

AIMV2_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`AIMv2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

AIMV2_BASE_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`AIMv2ImageProcessor.preprocess`] for details.

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
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

AIMv2_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`AIMv2ImageProcessor.preprocess`] for details.

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
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

@add_start_docstrings(
    "The bare AIMv2 Model transformer outputting raw hidden-states without any specific head on top.",
    AIMV2_START_DOCSTRING,
)
class AIMv2Model(AIMv2PreTrainedModel):
    def __init__(self, config: AIMv2Config):
        super().__init__(config)
        self.config = config

        self.embeddings = AIMv2Embeddings(config)
        self.encoder = AIMv2Encoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> AIMv2PatchEmbeddings:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(AIMV2_BASE_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(pixel_values, mask=mask)

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            head_outputs = (sequence_output)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class AIMv2AttentionPoolingClassifier(nn.Module):
    def __init__(
        self,
        config: AIMv2Config,
    ):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_queries = config.num_queries
        self.average_pool = config.average_pool if hasattr(config, "average_pool") else True

        self.k = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self.v = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self.cls_token = nn.Parameter(torch.randn(1, self.num_queries, config.hidden_size) * 0.02)
        self.linear = nn.Linear(config.hidden_size, config.num_labels, bias=config.proj_bias)
        self.bn = (
            nn.BatchNorm1d(config.hidden_size, affine=False, eps=1e-6)
            if config.use_batch_norm
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        x = self.bn(x.transpose(-2, -1)).transpose(-2, -1)
        cls_token = self.cls_token.expand(B, -1, -1)

        q = cls_token.reshape(
            B, self.num_queries, self.num_heads, C // self.num_heads
        ).permute(0, 2, 1, 3)
        k = (
            self.k(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        x_cls = F.scaled_dot_product_attention(q, k, v)
        x_cls = x_cls.transpose(1, 2).reshape(B, self.num_queries, C)
        x_cls = x_cls.mean(dim=1) if self.average_pool else x_cls

        out = self.linear(x_cls)
        return out

@add_start_docstrings(
    """
    AIMv2 Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
    the [CLS] token) e.g. for ImageNet.
    """,
    AIMV2_START_DOCSTRING,
)
class AIMv2ForImageClassification(AIMv2PreTrainedModel):
    def __init__(self, config: AIMv2Config) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.aimv2 = AIMv2Model(config)

        # Classifier head
        self.classifier = (
            AIMv2AttentionPoolingClassifier(config) if config.num_labels > 0 else nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()
    
    @add_start_docstrings_to_model_forward(AIMv2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.aimv2(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )