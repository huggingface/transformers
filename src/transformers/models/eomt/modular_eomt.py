# coding=utf-8
# Copyright 2025 Meta AI and The HuggingFace Inc. team. All rights reserved.
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
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...utils import (
    can_return_tuple,
)
from ..dinov2.modeling_dinov2 import (
    Dinov2Attention,
    Dinov2DropPath,
    Dinov2Layer,
    Dinov2LayerScale,
    Dinov2MLP,
    Dinov2PatchEmbeddings,
    Dinov2PreTrainedModel,
)
from ..dinov2_with_registers.modeling_dinov2_with_registers import Dinov2WithRegistersEmbeddings
from .configuration_eomt import EoMTConfig


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

class EoMTEmbeddings(Dinov2WithRegistersEmbeddings, nn.Module):
    """
    Construct the CLS token, mask token, register tokens, position and patch embeddings.
    """
    def __init__(self, config: EoMTConfig) -> None:
        super().__init__()

        num_patches = self.patch_embeddings.num_patches
        self.num_prefix_tokens = 1 + num_patches + config.num_register_tokens # 1 for [CLS]
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches, config.hidden_size))


class EoMTAttention(Dinov2Attention):
    pass


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


# ToDo: Check if layernorm2d == groupnorm with num_groups=1
class EoMTScaleBlock(nn.Module):
    def __init__(self, config: EoMTConfig):
        super().__init__()
        hidden_size = config.hidden_size
        self.deconv1 = nn.ConvTranspose2d(hidden_size, hidden_size, kernel_size=2, stride=2)
        self.activation = ACT2FN[config.hidden_act]
        self.deconv2 = nn.Conv2d(
            hidden_size,
            hidden_size,
            kernel_size=3,
            padding=1,
            groups=hidden_size,
            bias=False,
        )
        # Refer this: https://discuss.pytorch.org/t/groupnorm-num-groups-1-and-layernorm-are-not-equivalent/145468/2
        self.layernorm2d = nn.GroupNorm(num_groups=1, num_channels=hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.tensor) -> torch.Tensor:
        hidden_states = self.deconv1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.deconv2(hidden_states)
        hidden_states = self.layernorm2d(hidden_states)
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
                query = self.query.unsqueeze(0).expand(hidden_states.shape[0], -1, -1)
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
    def __init__(self, config:EoMTConfig):
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


class EoMTPreTrainedModel(Dinov2PreTrainedModel):
    pass


class EoMTModel(EoMTPreTrainedModel):
    def __init__(self, config: EoMTConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = EoMTEmbeddings(config)
        self.encoder = EoMTEncoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.upscale_block = nn.ModuleList([EoMTScaleBlock(config) for _ in range(config.num_upscale_blocks)])
        self.mask_head = MaskHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.Tensor] = None,
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

        embedding_output = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)

        encoder_outputs = self.encoder(
            embedding_output,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = sequence_output[:, 0, :]

        if not return_dict:
            head_outputs = (sequence_output, pooled_output)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class EoMTForUniversalSegmentation(nn.Module):
    main_input_name = "pixel_values"

    def __init__(self, config: EoMTConfig):
        super().__init__(config)
        self.config = config
        self.model = EoMTModel(config)
        self.class_predictor = nn.Linear(config.hidden_size, config.num_labels + 1)

        # Initialize model weights randomly.
        self.post_init()

    # A better place to add this func
    def _predict(self,logits:torch.Tensor):
        query_tokens = logits[:,:self.config.num_queries,:]
        class_logits = self.class_predictor(query_tokens)

        prefix_tokens = logits[:, self.config.num_queries+self.model.embeddings.num_prefix_tokens:, :]
        prefix_tokens = prefix_tokens.transpose(1,2)

        grid_size = self.model.embeddings.patch_embeddings.grid_size
        prefix_tokens = prefix_tokens.reshape(prefix_tokens.shape[0], -1, *grid_size)

        mask_logits = torch.einsum(
            "bqc, bchw -> bqhw", self.model.mask_head(query_tokens), self.model.upscale_block(prefix_tokens)
        )

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
            pixel_mask=pixel_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        masks_queries_logits, class_queries_logits = self._predict(outputs.pooler_output)


        return masks_queries_logits, class_queries_logits


__all__ = ["EoMTModel", "EoMTForUniversalSegmentation"]
