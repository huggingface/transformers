# coding=utf-8
# Copyright 2022 Snapchat Research and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch EfficientFormer model."""

import itertools
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from .configuration_efficientformer import EfficientFormerConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "EfficientFormerConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "efficientformer-l1-300"
_EXPECTED_OUTPUT_SHAPE = [1, 197, 768]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "snap-research/efficientformer-l1-300"
_IMAGE_CLASS_EXPECTED_OUTPUT = "Egyptian cat"


EFFICIENTFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "huggingface/efficientformer-l1-300",
    # See all EfficientFormer models at https://huggingface.co/models?filter=efficientformer
]


class EfficientFormerPatchEmbeddings(nn.Module):
    """
    This class performs downsampling between two stages. For the input tensor with the shape [batch_size, num_channels,
    height, width] it produces output tensor with the shape [batch_size, num_channels, height/stride, width/stride]
    """

    def __init__(self, config: EfficientFormerConfig, num_channels: int, embed_dim: int, apply_norm: bool = True):
        super().__init__()
        self.num_channels = num_channels

        self.projection = nn.Conv2d(
            num_channels,
            embed_dim,
            kernel_size=config.downsample_patch_size,
            stride=config.downsample_stride,
            padding=config.downsample_pad,
        )
        self.norm = nn.BatchNorm2d(embed_dim) if apply_norm else nn.Identity()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )

        embeddings = self.projection(pixel_values)
        embeddings = self.norm(embeddings)

        return embeddings


class EfficientFormerSelfAttention(nn.Module):
    def __init__(self, dim: int, key_dim: int, num_heads: int, attention_ratio: int, resolution: int):
        super().__init__()

        self.num_heads = num_heads
        self.key_dim = key_dim
        self.attention_ratio = attention_ratio
        self.scale = key_dim**-0.5
        self.total_key_dim = key_dim * num_heads
        self.expanded_key_dim = int(attention_ratio * key_dim)
        self.total_expanded_key_dim = int(self.expanded_key_dim * num_heads)
        hidden_size = self.total_expanded_key_dim + self.total_key_dim * 2
        self.qkv = nn.Linear(dim, hidden_size)
        self.projection = nn.Linear(self.total_expanded_key_dim, dim)
        points = list(itertools.product(range(resolution), range(resolution)))
        num_points = len(points)
        attention_offsets = {}
        idxs = []
        for point_1 in points:
            for point_2 in points:
                offset = (abs(point_1[0] - point_2[0]), abs(point_1[1] - point_2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer("attention_bias_idxs", torch.LongTensor(idxs).view(num_points, num_points))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, "ab"):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, hidden_states: torch.Tensor, output_attentions: bool = False) -> Tuple[torch.Tensor]:
        batch_size, sequence_length, num_channels = hidden_states.shape
        qkv = self.qkv(hidden_states)
        query_layer, key_layer, value_layer = qkv.reshape(batch_size, sequence_length, self.num_heads, -1).split(
            [self.key_dim, self.key_dim, self.expanded_key_dim], dim=3
        )
        query_layer = query_layer.permute(0, 2, 1, 3)
        key_layer = key_layer.permute(0, 2, 1, 3)
        value_layer = value_layer.permute(0, 2, 1, 3)

        attention_probs = (torch.matmul(query_layer, key_layer.transpose(-2, -1))) * self.scale + (
            self.attention_biases[:, self.attention_bias_idxs] if self.training else self.ab
        )

        attention_probs = attention_probs.softmax(dim=-1)

        context_layer = torch.matmul(attention_probs, value_layer).transpose(1, 2)
        context_layer = context_layer.reshape(batch_size, sequence_length, self.total_expanded_key_dim)
        context_layer = self.projection(context_layer)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class EfficientFormerConvStem(nn.Module):
    def __init__(self, config: EfficientFormerConfig, out_channels: int):
        super().__init__()

        self.convolution1 = nn.Conv2d(config.num_channels, out_channels // 2, kernel_size=3, stride=2, padding=1)
        self.batchnorm_before = nn.BatchNorm2d(out_channels // 2)

        self.convolution2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1)
        self.batchnorm_after = nn.BatchNorm2d(out_channels)

        self.activation = nn.ReLU()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.batchnorm_before(self.convolution1(pixel_values))
        features = self.activation(features)
        features = self.batchnorm_after(self.convolution2(features))
        features = self.activation(features)

        return features


class EfficientFormerPooling(nn.Module):
    def __init__(self, pool_size: int):
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = self.pool(hidden_states) - hidden_states
        return output


class EfficientFormerDenseMlp(nn.Module):
    def __init__(
        self,
        config: EfficientFormerConfig,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.linear_in = nn.Linear(in_features, hidden_features)
        self.activation = ACT2FN[config.hidden_act]
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear_out = nn.Linear(hidden_features, out_features)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_in(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.linear_out(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class EfficientFormerConvMlp(nn.Module):
    def __init__(
        self,
        config: EfficientFormerConfig,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.convolution1 = nn.Conv2d(in_features, hidden_features, 1)
        self.actvation = ACT2FN[config.hidden_act]
        self.convolution2 = nn.Conv2d(hidden_features, out_features, 1)
        self.dropout = nn.Dropout(drop)

        self.batchnorm_before = nn.BatchNorm2d(hidden_features)
        self.batchnorm_after = nn.BatchNorm2d(out_features)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.convolution1(hidden_state)
        hidden_state = self.batchnorm_before(hidden_state)

        hidden_state = self.actvation(hidden_state)
        hidden_state = self.dropout(hidden_state)
        hidden_state = self.convolution2(hidden_state)

        hidden_state = self.batchnorm_after(hidden_state)

        hidden_state = self.dropout(hidden_state)
        return hidden_state


# Copied from transformers.models.convnext.modeling_convnext.drop_path
def drop_path(input, drop_prob: float = 0.0, training: bool = False):
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


# Copied from transformers.models.beit.modeling_beit.BeitDropPath with Beit->Bit
class EfficientFormerDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class EfficientFormerFlat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor]:
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        return hidden_states


class EfficientFormerMeta3D(nn.Module):
    def __init__(self, config: EfficientFormerConfig, dim: int, drop_path: float = 0.0):
        super().__init__()

        self.token_mixer = EfficientFormerSelfAttention(
            dim=config.dim,
            key_dim=config.key_dim,
            num_heads=config.num_attention_heads,
            attention_ratio=config.attention_ratio,
            resolution=config.resolution,
        )
        self.layernorm1 = nn.LayerNorm(dim)
        self.layernorm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * config.mlp_expansion_ratio)
        self.mlp = EfficientFormerDenseMlp(config, in_features=dim, hidden_features=mlp_hidden_dim)

        self.drop_path = EfficientFormerDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.use_layer_scale = config.use_layer_scale
        if config.use_layer_scale:
            self.layer_scale_1 = nn.Parameter(config.layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(config.layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, hidden_states: torch.Tensor, output_attentions: bool = False) -> Tuple[torch.Tensor]:
        self_attention_outputs = self.token_mixer(self.layernorm1(hidden_states), output_attentions)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.use_layer_scale:
            layer_output = hidden_states + self.drop_path(
                self.layer_scale_1.unsqueeze(0).unsqueeze(0) * attention_output
            )
            layer_output = layer_output + self.drop_path(
                self.layer_scale_2.unsqueeze(0).unsqueeze(0) * self.mlp(self.layernorm2(layer_output))
            )
        else:
            layer_output = hidden_states + self.drop_path(attention_output)
            layer_output = layer_output + self.drop_path(self.mlp(self.layernorm2(layer_output)))

        outputs = (layer_output,) + outputs

        return outputs


class EfficientFormerMeta3DLayers(nn.Module):
    def __init__(self, config: EfficientFormerConfig):
        super().__init__()
        drop_paths = [
            config.drop_path_rate * (block_idx + sum(config.depths[:-1]))
            for block_idx in range(config.num_meta3d_blocks)
        ]
        self.blocks = nn.ModuleList(
            [EfficientFormerMeta3D(config, config.hidden_sizes[-1], drop_path=drop_path) for drop_path in drop_paths]
        )

    def forward(self, hidden_states: torch.Tensor, output_attentions: bool = False) -> Tuple[torch.Tensor]:
        all_attention_outputs = () if output_attentions else None
        for layer_module in self.blocks:
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
            hidden_states = layer_module(hidden_states, output_attentions)
            if output_attentions:
                all_attention_outputs = all_attention_outputs + (hidden_states[1],)
        if output_attentions:
            outputs = (hidden_states[0],) + all_attention_outputs
            return outputs
        return hidden_states


class EfficientFormerMeta4D(nn.Module):
    def __init__(self, config: EfficientFormerConfig, dim: int, drop_path: float = 0.0):
        super().__init__()
        pool_size = config.pool_size if config.pool_size is not None else 3
        self.token_mixer = EfficientFormerPooling(pool_size=pool_size)
        mlp_hidden_dim = int(dim * config.mlp_expansion_ratio)
        self.mlp = EfficientFormerConvMlp(
            config, in_features=dim, hidden_features=mlp_hidden_dim, drop=config.hidden_dropout_prob
        )

        self.drop_path = EfficientFormerDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.use_layer_scale = config.use_layer_scale
        if config.use_layer_scale:
            self.layer_scale_1 = nn.Parameter(config.layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(config.layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor]:
        outputs = self.token_mixer(hidden_states)

        if self.use_layer_scale:
            layer_output = hidden_states + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * outputs)
            layer_output = layer_output + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(layer_output)
            )
        else:
            layer_output = hidden_states + self.drop_path(outputs)
            layer_output = layer_output + self.drop_path(self.mlp(layer_output))

        return layer_output


class EfficientFormerMeta4DLayers(nn.Module):
    def __init__(self, config: EfficientFormerConfig, stage_idx: int):
        super().__init__()
        num_layers = (
            config.depths[stage_idx] if stage_idx != -1 else config.depths[stage_idx] - config.num_meta3d_blocks
        )
        drop_paths = [
            config.drop_path_rate * (block_idx + sum(config.depths[:stage_idx])) for block_idx in range(num_layers)
        ]
        self.blocks = nn.ModuleList(
            [
                EfficientFormerMeta4D(config, config.hidden_sizes[stage_idx], drop_path=drop_path)
                for drop_path in drop_paths
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor]:
        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states)
        return hidden_states


class EfficientFormerIntermediateStage(nn.Module):
    def __init__(self, config: EfficientFormerConfig, index: int):
        super().__init__()
        self.meta4D_layers = EfficientFormerMeta4DLayers(config, index)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor]:
        hidden_states = self.meta4D_layers(hidden_states)
        return hidden_states


class EfficientFormerLastStage(nn.Module):
    def __init__(self, config: EfficientFormerConfig):
        super().__init__()
        self.meta4D_layers = EfficientFormerMeta4DLayers(config, -1)
        self.flat = EfficientFormerFlat()
        self.meta3D_layers = EfficientFormerMeta3DLayers(config)

    def forward(self, hidden_states: torch.Tensor, output_attentions: bool = False) -> Tuple[torch.Tensor]:
        hidden_states = self.meta4D_layers(hidden_states)
        hidden_states = self.flat(hidden_states)
        hidden_states = self.meta3D_layers(hidden_states, output_attentions)

        return hidden_states


class EfficientFormerEncoder(nn.Module):
    def __init__(self, config: EfficientFormerConfig):
        super().__init__()
        self.config = config
        num_intermediate_stages = len(config.depths) - 1
        downsamples = [
            config.downsamples[i] or config.hidden_sizes[i] != config.hidden_sizes[i + 1]
            for i in range(num_intermediate_stages)
        ]
        intermediate_stages = []
        for i in range(num_intermediate_stages):
            intermediate_stages.append(EfficientFormerIntermediateStage(config, i))
            if downsamples[i]:
                intermediate_stages.append(
                    EfficientFormerPatchEmbeddings(config, config.hidden_sizes[i], config.hidden_sizes[i + 1])
                )

        self.intermediate_stages = nn.ModuleList(intermediate_stages)
        self.last_stage = EfficientFormerLastStage(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
    ) -> BaseModelOutput:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        for layer_module in self.intermediate_stages:
            hidden_states = layer_module(hidden_states)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        layer_output = self.last_stage(hidden_states, output_attentions=output_attentions)
        if output_attentions:
            all_self_attentions = all_self_attentions + layer_output[1:]

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (layer_output[0],)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=layer_output[0],
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class EfficientFormerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = EfficientFormerConfig
    base_model_prefix = "efficientformer"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = False

    def _init_weights(self, module: nn.Module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


EFFICIENTFORMER_START_DOCSTRING = r"""
    This model is a PyTorch [nn.Module](https://pytorch.org/docs/stable/nn.html#nn.Module) subclass. Use it as a
    regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

    Parameters:
        config ([`EfficientFormerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

EFFICIENTFORMER_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`ViTFeatureExtractor`]. See
            [`ViTFeatureExtractor.__call__`] for details.
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
    "The bare EfficientFormer Model transformer outputting raw hidden-states without any specific head on top.",
    EFFICIENTFORMER_START_DOCSTRING,
)
class EfficientFormerModel(EfficientFormerPreTrainedModel):
    def __init__(self, config: EfficientFormerConfig):
        super().__init__(config)
        self.config = config

        self.patch_embed = EfficientFormerConvStem(config, config.hidden_sizes[0])
        self.encoder = EfficientFormerEncoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_sizes[-1], eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(EFFICIENTFORMER_INPUTS_DOCSTRING)
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
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        embedding_output = self.patch_embed(pixel_values)
        encoder_outputs = self.encoder(
            embedding_output, output_attentions=output_attentions, output_hidden_states=output_hidden_states
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


@add_start_docstrings(
    """
    EfficientFormer Model transformer with an image classification head on top (a linear layer on top of the final
    hidden state of the [CLS] token) e.g. for ImageNet.
    """,
    EFFICIENTFORMER_START_DOCSTRING,
)
class EfficientFormerForImageClassification(EfficientFormerPreTrainedModel):
    def __init__(self, config: EfficientFormerConfig):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.efficientformer = EfficientFormerModel(config)

        # Classifier head
        self.classifier = (
            nn.Linear(config.hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(EFFICIENTFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
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

        outputs = self.efficientformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.classifier(sequence_output.mean(-2))

        loss = None
        if labels is not None:
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
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@dataclass
class EfficientFormerForImageClassificationWithTeacherOutput(ModelOutput):
    """
    Output type of [`EfficientFormerForImageClassificationWithTeacher`].

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Prediction scores as the average of the cls_logits and distillation logits.
        cls_logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Prediction scores of the classification head (i.e. the linear layer on top of the final hidden state of the
            class token).
        distillation_logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Prediction scores of the distillation head (i.e. the linear layer on top of the final hidden state of the
            distillation token).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    logits: torch.FloatTensor = None
    cls_logits: torch.FloatTensor = None
    distillation_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@add_start_docstrings(
    """
    EfficientFormer Model transformer with image classification heads on top (a linear layer on top of the final hidden
    state of the [CLS] token and a linear layer on top of the final hidden state of the distillation token) e.g. for
    ImageNet.

    <Tip warning={true}>

           This model supports inference-only. Fine-tuning with distillation (i.e. with a teacher) is not yet
           supported.

    </Tip>
    """,
    EFFICIENTFORMER_START_DOCSTRING,
)
class EfficientFormerForImageClassificationWithTeacher(EfficientFormerPreTrainedModel):
    def __init__(self, config: EfficientFormerConfig):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.efficientformer = EfficientFormerModel(config)

        # Classifier head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        # Distillation head
        self.distillation_classifier = (
            nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(EFFICIENTFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=EfficientFormerForImageClassificationWithTeacherOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, EfficientFormerForImageClassificationWithTeacherOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.efficientformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        cls_logits = self.classifier(sequence_output.mean(-2))
        distillation_logits = self.distillation_classifier(sequence_output.mean(-2))

        # during inference, return the average of both classifier predictions
        logits = (cls_logits + distillation_logits) / 2

        if not return_dict:
            output = (logits, cls_logits, distillation_logits) + outputs[1:]
            return output

        return EfficientFormerForImageClassificationWithTeacherOutput(
            logits=logits,
            cls_logits=cls_logits,
            distillation_logits=distillation_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
