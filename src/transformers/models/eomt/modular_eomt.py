# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file ehidden_statescept in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either ehidden_statespress or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch EoMT model."""

import collections
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...utils import (
    can_return_tuple,
    logging,
)
from ..dinov2.modeling_dinov2 import (
    Dinov2DropPath,
    Dinov2Layer,
    Dinov2LayerScale,
    Dinov2MLP,
    Dinov2PatchEmbeddings,
)
from ..llama.modeling_llama import eager_attention_forward
from .configuration_eomt import EoMTConfig


logger = logging.get_logger(__name__)


class EoMTPatchEmbeddings(Dinov2PatchEmbeddings, nn.Module):
    def __init__(self, config: EoMTConfig):
        nn.Module().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.grid_size = (self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]


class EoMTEmbeddings(nn.Module):
    """
    Construct the CLS token, mask token, register tokens, position and patch embeddings.
    """

    def __init__(self, config: EoMTConfig) -> None:
        super().__init__()

        self.config = config
        self.patch_size = config.patch_size

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.register_tokens = nn.Parameter(torch.zeros(1, config.num_register_tokens, config.hidden_size))

        self.patch_embeddings = EoMTPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.num_prefix_tokens = 1 + config.num_register_tokens  # 1 for [CLS]
        self.position_embeddings = nn.Embedding(num_patches, config.hidden_size)
        self.register_buffer("position_ids", torch.arange(num_patches).expand((1, -1)), persistent=False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, _, _, _ = pixel_values.shape
        target_dtype = self.patch_embeddings.projection.weight.dtype
        embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        register_tokens = self.register_tokens.expand(batch_size, -1, -1)

        embeddings = embeddings + self.position_embeddings(self.position_ids)
        embeddings = torch.cat([cls_tokens, register_tokens, embeddings], dim=1)

        embeddings = self.dropout(embeddings)

        return embeddings


class EoMTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.attention_dropout = config.attention_dropout
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.num_key_value_groups = 1
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.qkv_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.qkv_bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.qkv_bias)
        self.proj_out = nn.Linear(self.embed_dim, self.embed_dim, bias=config.qkv_bias)
        self.proj_drop = nn.Dropout(config.projection_dropout)

        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""
        batch_size, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and output_attentions:
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            is_causal=self.is_causal,
            **kwargs,
        )

        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

        attn_output = self.proj_out(attn_output)
        attn_output = self.proj_drop(attn_output)

        output = (attn_output, attn_weights) if output_attentions else (attn_output, None)

        return output


class EoMTMLP(Dinov2MLP):
    pass


class EoMTLayerScale(Dinov2LayerScale):
    pass


class EoMTDropPath(Dinov2DropPath):
    pass


class EoMTLayer(Dinov2Layer, nn.Module):
    """This corresponds to the Block class in the original implementation."""

    def __init__(self, config: EoMTConfig) -> None:
        nn.Module().__init__()

        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = EoMTAttention(config)
        self.layer_scale1 = EoMTLayerScale(config)
        self.drop_path = EoMTDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = EoMTMLP(config)
        self.layer_scale2 = EoMTLayerScale(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.norm1(hidden_states),  # In EoMT, layernorm is applied before self-attention
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        attention_output = self.layer_scale1(attention_output)
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = self.drop_path(attention_output) + hidden_states

        # In EoMT, layernorm is also applied after self-attention
        layer_output = self.norm2(hidden_states)
        layer_output = self.mlp(layer_output)
        layer_output = self.layer_scale2(layer_output)

        # second residual connection
        layer_output = self.drop_path(layer_output) + hidden_states

        outputs = (layer_output,) + outputs

        return outputs


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = hidden_state.permute(0, 2, 3, 1)
        hidden_state = F.layer_norm(hidden_state, self.normalized_shape, self.weight, self.bias, self.eps)
        hidden_state = hidden_state.permute(0, 3, 1, 2)
        return hidden_state


class EoMTScaleLayer(nn.Module):
    def __init__(self, config: EoMTConfig):
        super().__init__()
        hidden_size = config.hidden_size
        self.conv1 = nn.ConvTranspose2d(hidden_size, hidden_size, kernel_size=2, stride=2)
        self.activation = ACT2FN[config.hidden_act]
        self.conv2 = nn.Conv2d(
            hidden_size,
            hidden_size,
            kernel_size=3,
            padding=1,
            groups=hidden_size,
            bias=False,
        )

        self.layernorm2d = LayerNorm2d(hidden_size)

    def forward(self, hidden_states: torch.tensor) -> torch.Tensor:
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.layernorm2d(hidden_states)
        return hidden_states


class EoMTScaleBlock(nn.Module):
    def __init__(self, config: EoMTConfig):
        super().__init__()
        self.num_blocks = config.num_upscale_blocks
        self.block = nn.ModuleList([EoMTScaleLayer(config) for _ in range(self.num_blocks)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for block in self.block:
            hidden_states = block(hidden_states)
        return hidden_states


class EoMTEncoder(nn.Module):
    def __init__(self, config: EoMTConfig) -> None:
        super().__init__()
        self.config = config
        self.query = nn.Embedding(config.num_queries, config.hidden_size)
        self.layers = nn.ModuleList([EoMTLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    @can_return_tuple
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layers):
            if i == len(self.layers) - self.config.num_blocks:
                query = self.query.weight[None, :, :].expand(hidden_states.shape[0], -1, -1)
                hidden_states = torch.cat((query, hidden_states), dim=1)

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

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class MaskHead(nn.Module):
    def __init__(self, config: EoMTConfig):
        super().__init__()

        hidden_size = config.hidden_size
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.activation(self.fc1(hidden_states))
        hidden_states = self.activation(self.fc2(hidden_states))
        hidden_states = self.fc3(hidden_states)
        return hidden_states


class EoMTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = EoMTConfig
    base_model_prefix = "eomt"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["EoMTSwiGLUFFN"]
    _supports_sdpa = True
    _supports_flash_attn_2 = True

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
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, EoMTLayerScale):
            module.lambda1.data.fill_(self.config.layerscale_value)


class EoMTModel(EoMTPreTrainedModel):
    def __init__(self, config: EoMTConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = EoMTEmbeddings(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.encoder = EoMTEncoder(config)
        self.upscale_block = EoMTScaleBlock(config)
        self.mask_head = MaskHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        embedding_output = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class EoMTForUniversalSegmentation(EoMTPreTrainedModel):
    main_input_name = "pixel_values"

    def __init__(self, config: EoMTConfig):
        super().__init__(config)
        self.config = config
        self.model = EoMTModel(config)
        self.class_predictor = nn.Linear(config.hidden_size, config.num_labels + 1)

        # Initialize model weights randomly.
        self.post_init()

    # A better place to add this func
    def _predict(self, logits: torch.Tensor):
        query_tokens = logits[:, : self.config.num_queries, :]
        class_logits = self.class_predictor(query_tokens)

        prefix_tokens = logits[:, self.config.num_queries + self.model.embeddings.num_prefix_tokens :, :]
        prefix_tokens = prefix_tokens.transpose(1, 2)

        grid_size = self.model.embeddings.patch_embeddings.grid_size
        prefix_tokens = prefix_tokens.reshape(prefix_tokens.shape[0], -1, *grid_size)

        query_tokens = self.model.mask_head(query_tokens)
        prefix_tokens = self.model.upscale_block(prefix_tokens)

        mask_logits = torch.einsum("bqc, bchw -> bqhw", query_tokens, prefix_tokens)

        return mask_logits, class_logits

    def forward(
        self,
        pixel_values: Tensor,
        mask_labels: Optional[List[Tensor]] = None,
        class_labels: Optional[List[Tensor]] = None,
        pixel_mask: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_auxiliary_logits: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.model(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        print(outputs.last_hidden_state.shape)

        masks_queries_logits, class_queries_logits = self._predict(outputs.last_hidden_state)

        return masks_queries_logits, class_queries_logits


__all__ = ["EoMTModel", "EoMTForUniversalSegmentation"]
