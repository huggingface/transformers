# coding=utf-8
# Copyright 2025 Meta AI and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Pixo model."""

from typing import Optional

import torch
from torch import nn

from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BackboneOutput, BaseModelOutput, BaseModelOutputWithPooling
from ...utils import auto_docstring, logging, torch_int
from ...utils.generic import check_model_inputs
from ..dinov2.modeling_dinov2 import Dinov2Backbone, Dinov2DropPath, Dinov2MLP
from ..vit.modeling_vit import ViTAttention, ViTPatchEmbeddings, ViTPreTrainedModel
from .configuration_pixo import PixoConfig


logger = logging.get_logger(__name__)


class PixoPatchEmbeddings(ViTPatchEmbeddings):
    pass


class PixoEmbeddings(nn.Module):
    """Construct the CLS tokens, position and patch embeddings while reusing ViT's initialization utilities."""

    def __init__(self, config: PixoConfig) -> None:
        super().__init__()
        self.patch_embeddings = PixoPatchEmbeddings(config)
        self.cls_token = nn.Parameter(torch.randn(1, config.n_cls_tokens, config.hidden_size))
        self.mask_token = None
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + config.n_cls_tokens, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.n_cls_tokens = config.n_cls_tokens
        self.patch_size = config.patch_size
        self.config = config

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        num_patches = embeddings.shape[1] - self.n_cls_tokens
        num_positions = self.position_embeddings.shape[1] - self.n_cls_tokens

        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embeddings

        class_pos_embed = self.position_embeddings[:, : self.n_cls_tokens]
        patch_pos_embed = self.position_embeddings[:, self.n_cls_tokens :]

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
        ).to(dtype=target_dtype)

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = pixel_values.shape
        target_dtype = self.patch_embeddings.projection.weight.dtype
        embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)

        embeddings = self.dropout(embeddings)

        return embeddings


class PixoAttention(ViTAttention):
    pass


class PixoDropPath(Dinov2DropPath):
    pass


class PixoMLP(Dinov2MLP):
    pass


class PixoLayer(GradientCheckpointingLayer):
    def __init__(self, config: PixoConfig) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = PixoAttention(config)
        self.drop_path = PixoDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = PixoMLP(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states_norm = self.norm1(hidden_states)
        self_attention_output = self.attention(hidden_states_norm)

        hidden_states = self.drop_path(self_attention_output) + hidden_states

        layer_output = self.norm2(hidden_states)
        layer_output = self.mlp(layer_output)

        layer_output = self.drop_path(layer_output) + hidden_states

        return layer_output


class PixoEncoder(nn.Module):
    def __init__(self, config: PixoConfig):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([PixoLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor, output_hidden_states: bool = False) -> BaseModelOutput:
        all_hidden_states = [hidden_states] if output_hidden_states else None
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states)
            if all_hidden_states:
                all_hidden_states.append(hidden_states)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=tuple(all_hidden_states) if all_hidden_states else None,
        )


class PixoPreTrainedModel(ViTPreTrainedModel):
    pass


@auto_docstring
class PixoModel(PixoPreTrainedModel):
    def __init__(self, config: PixoConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = PixoEmbeddings(config)
        self.encoder = PixoEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.post_init()

    def get_input_embeddings(self) -> PixoPatchEmbeddings:
        return self.embeddings.patch_embeddings

    @check_model_inputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPooling:
        if output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        embedding_output = self.embeddings(pixel_values)

        encoder_outputs: BaseModelOutput = self.encoder(embedding_output, output_hidden_states=output_hidden_states)
        sequence_output = encoder_outputs.last_hidden_state
        sequence_output = self.layernorm(sequence_output)
        pooled_output = sequence_output[:, : self.embeddings.n_cls_tokens, :].mean(dim=1)

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )


@auto_docstring(
    custom_intro="""
    Pixo backbone, to be used with frameworks like DETR and MaskFormer.
    """
)
class PixoBackbone(Dinov2Backbone):
    @check_model_inputs
    @auto_docstring
    def forward(
        self, pixel_values: torch.Tensor, output_hidden_states: Optional[bool] = None, **kwargs
    ) -> BackboneOutput:
        r"""
        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AutoBackbone
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> processor = AutoImageProcessor.from_pretrained("facebook/pixo-huge")
        >>> model = AutoBackbone.from_pretrained(
        ...     "facebook/pixo-huge", out_features=["stage7", "stage15", "stage23", "stage31"]
        ... )

        >>> inputs = processor(image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> feature_maps = outputs.feature_maps
        >>> list(feature_maps[-1].shape)
        [1, 1280, 16, 16]
        ```"""
        if output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states

        embedding_output = self.embeddings(pixel_values)
        output: BaseModelOutput = self.encoder(embedding_output, output_hidden_states=True)
        hidden_states = output.hidden_states

        feature_maps = []
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                if self.config.apply_layernorm:
                    hidden_state = self.layernorm(hidden_state)
                if self.config.reshape_hidden_states:
                    hidden_state = hidden_state[:, self.embeddings.n_cls_tokens :]
                    batch_size, _, height, width = pixel_values.shape
                    patch_size = self.config.patch_size
                    hidden_state = hidden_state.reshape(batch_size, height // patch_size, width // patch_size, -1)
                    hidden_state = hidden_state.permute(0, 3, 1, 2).contiguous()
                feature_maps.append(hidden_state)

        return BackboneOutput(
            feature_maps=tuple(feature_maps),
            hidden_states=hidden_states if output_hidden_states else None,
        )


__all__ = ["PixoModel", "PixoPreTrainedModel", "PixoBackbone"]
