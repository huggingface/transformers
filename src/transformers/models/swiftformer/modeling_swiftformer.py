# coding=utf-8
# Copyright 2023 MBZUAI and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch SwiftFormer model."""

import collections.abc
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from ...utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .configuration_swiftformer import SwiftFormerConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "SwiftFormerConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "MBZUAI/swiftformer-xs"
_EXPECTED_OUTPUT_SHAPE = [1, 384, 7, 7]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "MBZUAI/swiftformer-xs"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"


class SwiftFormerPatchEmbedding(nn.Module):
    """
    Patch Embedding Layer constructed of two 2D convolutional layers.

    Input: tensor of shape `[batch_size, in_channels, height, width]`

    Output: tensor of shape `[batch_size, out_channels, height/4, width/4]`
    """

    def __init__(self, config: SwiftFormerConfig):
        super().__init__()

        in_chs = config.num_channels
        out_chs = config.embed_dims[0]
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_chs, out_chs // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_chs // 2, eps=config.batch_norm_eps),
            nn.ReLU(),
            nn.Conv2d(out_chs // 2, out_chs, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_chs, eps=config.batch_norm_eps),
            nn.ReLU(),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.patch_embedding(pixel_values)


def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment: From the official implementation:
    https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


class SwiftFormerDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, config: SwiftFormerConfig) -> None:
        super().__init__()
        self.drop_prob = config.drop_path_rate

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class SwiftFormerEmbeddings(nn.Module):
    """
    Embeddings layer consisting of a single 2D convolutional and batch normalization layer.

    Input: tensor of shape `[batch_size, channels, height, width]`

    Output: tensor of shape `[batch_size, channels, height/stride, width/stride]`
    """

    def __init__(self, config: SwiftFormerConfig, in_chs: int, out_chs: int, patch_size: int, stride: int, padding: int):
        super().__init__()
        self.embeddings = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, kernel_size=patch_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_chs, eps=config.batch_norm_eps),
            nn.ReLU(),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.embeddings(hidden_states)


class SwiftFormerConvEncoder(nn.Module):
    """
    Convolutional encoder layer for SwiftFormer.

    Input: tensor of shape `[batch_size, channels, height, width]`

    Output: tensor of shape `[batch_size, channels, height, width]`
    """

    def __init__(self, config: SwiftFormerConfig, dim: int = 512):
        super().__init__()
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
            nn.BatchNorm2d(dim, eps=config.batch_norm_eps),
            nn.ReLU(),
        )
        self.drop_conv_encoder = nn.Dropout(config.drop_conv_encoder_rate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_encoder(hidden_states)
        hidden_states = self.drop_conv_encoder(hidden_states)
        return hidden_states


class SwiftFormerMlp(nn.Module):
    """
    MLP layer for SwiftFormer.

    Input: tensor of shape `[batch_size, channels, height, width]`

    Output: tensor of shape `[batch_size, channels, height, width]`
    """

    def __init__(self, config: SwiftFormerConfig, in_features: int, hidden_features: Optional[int] = None, out_features: Optional[int] = None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm1 = nn.BatchNorm2d(in_features, eps=config.batch_norm_eps)
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.act = nn.ReLU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(config.drop_mlp_rate)

    def forward(self, x):
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiftFormerEfficientAdditiveAttention(nn.Module):
    """
    Efficient Additive Attention module for SwiftFormer.

    Input: tensor of shape `[batch_size, channels, height, width]`

    Output: tensor of shape `[batch_size, channels, height, width]`
    """

    def __init__(self, config: SwiftFormerConfig, dim: int = 512):
        super().__init__()

        self.to_query = nn.Linear(dim, dim)
        self.to_key = nn.Linear(dim, dim)

        # Initialize w_g with zeros if layer_scale_init_value is very small (zero-init case)
        # Otherwise initialize with random values
        if hasattr(config, "layer_scale_init_value") and config.layer_scale_init_value <= 1e-9:
            self.w_g = nn.Parameter(torch.zeros(dim, 1))
        else:
            self.w_g = nn.Parameter(torch.randn(dim, 1))
        self.scale_factor = dim**-0.5
        self.proj = nn.Linear(dim, dim)
        self.final = nn.Linear(dim, dim)

    def forward(self, x):
        query = self.to_query(x)
        key = self.to_key(x)

        query = torch.nn.functional.normalize(query, dim=-1)
        key = torch.nn.functional.normalize(key, dim=-1)

        query_weight = query @ self.w_g
        scaled_query_weight = query_weight * self.scale_factor
        scaled_query_weight = scaled_query_weight.softmax(dim=-1)

        global_queries = torch.sum(scaled_query_weight * query, dim=1)
        global_queries = global_queries.unsqueeze(1).repeat(1, key.shape[1], 1)

        out = self.proj(global_queries * key) + query
        out = self.final(out)

        return out


class SwiftFormerLocalRepresentation(nn.Module):
    """
    Local Representation module for SwiftFormer.

    Input: tensor of shape `[batch_size, channels, height, width]`

    Output: tensor of shape `[batch_size, channels, height, width]`
    """

    def __init__(self, config: SwiftFormerConfig, dim: int = 512):
        super().__init__()
        self.conv_encoder = SwiftFormerConvEncoder(config, dim)
        self.layer_scale = nn.Parameter(torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
        self.layer_scale.data.fill_(config.layer_scale_init_value)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.layer_scale * self.conv_encoder(hidden_states)
        return hidden_states


class SwiftFormerGlobalRepresentation(nn.Module):
    """
    Global Representation module for SwiftFormer.

    Input: tensor of shape `[batch_size, channels, height, width]`

    Output: tensor of shape `[batch_size, channels, height, width]`
    """

    def __init__(self, config: SwiftFormerConfig, dim: int = 512):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim, eps=config.batch_norm_eps)
        self.attn = SwiftFormerEfficientAdditiveAttention(config, dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = hidden_states.shape
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.attn(hidden_states)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, channels, height, width)
        return hidden_states


class SwiftFormerBlock(nn.Module):
    """
    SwiftFormer block.

    Input: tensor of shape `[batch_size, channels, height, width]`

    Output: tensor of shape `[batch_size, channels, height, width]`
    """

    def __init__(self, config: SwiftFormerConfig, dim: int = 512):
        super().__init__()
        self.local_representation = SwiftFormerLocalRepresentation(config, dim)
        self.global_representation = SwiftFormerGlobalRepresentation(config, dim)
        self.mlp = SwiftFormerMlp(config, dim, int(dim * config.mlp_ratio))

        if config.use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=config.use_layer_scale
            )
            self.layer_scale_1.data.fill_(config.layer_scale_init_value)
            self.layer_scale_2 = nn.Parameter(
                torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=config.use_layer_scale
            )
            self.layer_scale_2.data.fill_(config.layer_scale_init_value)
        else:
            self.layer_scale_1 = 1.0
            self.layer_scale_2 = 1.0

        self.drop_path = SwiftFormerDropPath(config) if config.drop_path_rate > 0.0 else nn.Identity()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Local representation
        hidden_states = self.local_representation(hidden_states)

        # Global representation
        batch_size, channels, height, width = hidden_states.shape
        global_hidden_states = self.global_representation(hidden_states)
        hidden_states = hidden_states + self.drop_path(self.layer_scale_1 * global_hidden_states)

        # MLP
        mlp_hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + self.drop_path(self.layer_scale_2 * mlp_hidden_states)

        return hidden_states


class SwiftFormerStage(nn.Module):
    """
    SwiftFormer stage.

    Input: tensor of shape `[batch_size, in_channels, height, width]`

    Output: tensor of shape `[batch_size, out_channels, height/stride, width/stride]`
    """

    def __init__(
        self,
        config: SwiftFormerConfig,
        in_chs: int,
        out_chs: int,
        depth: int,
        downsample: bool = True,
    ):
        super().__init__()
        self.downsample = downsample
        if self.downsample:
            self.downsample_layers = SwiftFormerEmbeddings(
                config, in_chs, out_chs, config.down_patch_size, config.down_stride, config.down_pad
            )
        self.blocks = nn.ModuleList([SwiftFormerBlock(config, out_chs) for _ in range(depth)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.downsample:
            hidden_states = self.downsample_layers(hidden_states)

        for block in self.blocks:
            hidden_states = block(hidden_states)

        return hidden_states


class SwiftFormerEncoder(nn.Module):
    """
    SwiftFormer encoder.

    Input: tensor of shape `[batch_size, in_channels, height, width]`

    Output: tensor of shape `[batch_size, out_channels, height/32, width/32]`
    """

    def __init__(self, config: SwiftFormerConfig):
        super().__init__()
        self.patch_embed = SwiftFormerPatchEmbedding(config)
        self.network = nn.ModuleList(
            [
                SwiftFormerStage(
                    config,
                    config.embed_dims[i],
                    config.embed_dims[i + 1],
                    config.depths[i],
                    config.downsamples[i],
                )
                for i in range(len(config.depths))
            ]
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(pixel_values)

        for stage in self.network:
            hidden_states = stage(hidden_states)

        return hidden_states


class SwiftFormerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SwiftFormerConfig
    base_model_prefix = "swiftformer"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


@dataclass
class SwiftFormerEncoderOutput(ModelOutput):
    """
    SwiftFormer encoder's outputs, with potential hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class SwiftFormerModel(SwiftFormerPreTrainedModel):
    def __init__(self, config: SwiftFormerConfig):
        super().__init__(config)
        self.config = config

        self.encoder = SwiftFormerEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithNoAttention]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AutoModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("MBZUAI/swiftformer-xs")
        >>> model = AutoModel.from_pretrained("MBZUAI/swiftformer-xs")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        embedding_output = pixel_values

        encoder_outputs = self.encoder(embedding_output)

        if not return_dict:
            return (encoder_outputs,)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=encoder_outputs,
            hidden_states=None,
        )


class SwiftFormerPooler(nn.Module):
    def __init__(self, config: SwiftFormerConfig):
        super().__init__()
        self.dense = nn.Linear(config.embed_dims[-1], config.embed_dims[-1])
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output


@dataclass
class SwiftFormerModelWithPoolingOutput(ModelOutput):
    """
    SwiftFormer model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state after a pooling operation on the spatial dimensions.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, num_channels, height, width)`.
    """

    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class SwiftFormerForImageClassification(SwiftFormerPreTrainedModel):
    def __init__(self, config: SwiftFormerConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.swiftformer = SwiftFormerModel(config)

        # Classifier head
        self.classifier = (
            nn.Linear(config.embed_dims[-1], config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(
        """
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
                [`AutoImageProcessor.__call__`] for details.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutputWithNoAttention]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AutoModelForImageClassification
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("MBZUAI/swiftformer-xs")
        >>> model = AutoModelForImageClassification.from_pretrained("MBZUAI/swiftformer-xs")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> logits = outputs.logits
        >>> # model predicts one of the 1000 ImageNet classes
        >>> predicted_class_idx = logits.argmax(-1).item()
        >>> print("Predicted class:", model.config.id2label[predicted_class_idx])
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.swiftformer(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if isinstance(outputs, tuple):
            feature_map = outputs[0]
        else:
            feature_map = outputs.last_hidden_state

        # Global average pooling
        pooled_output = torch.mean(feature_map, dim=[-2, -1])

        logits = self.classifier(pooled_output)

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

        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
        )
