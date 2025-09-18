# coding=utf-8
# Copyright 2022 Meta Platforms, Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch ConvNext model."""

from typing import Optional

import torch
from torch import nn

from ...activations import ACT2FN
from ...modeling_outputs import (
    BackboneOutput,
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
)
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring, logging
from ...utils.backbone_utils import BackboneMixin
from ...utils.generic import can_return_tuple
from .configuration_convnext import ConvNextConfig


logger = logging.get_logger(__name__)


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


# Copied from transformers.models.beit.modeling_beit.BeitDropPath with Beit->ConvNext
class ConvNextDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"p={self.drop_prob}"


class ConvNextLayerNorm(nn.LayerNorm):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, height,
    width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, *, eps=1e-6, data_format="channels_last", **kwargs):
        super().__init__(normalized_shape, eps=eps, **kwargs)
        if data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(f"Unsupported data format: {data_format}")
        self.data_format = data_format

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Tensor of shape (batch_size, channels, height, width) OR (batch_size, height, width, channels)
        """
        if self.data_format == "channels_first":
            features = features.permute(0, 2, 3, 1)
            features = super().forward(features)
            features = features.permute(0, 3, 1, 2)
        else:
            features = super().forward(features)
        return features


class ConvNextEmbeddings(nn.Module):
    """This class is comparable to (and inspired by) the SwinEmbeddings class
    found in src/transformers/models/swin/modeling_swin.py.
    """

    def __init__(self, config):
        super().__init__()
        self.patch_embeddings = nn.Conv2d(
            config.num_channels, config.hidden_sizes[0], kernel_size=config.patch_size, stride=config.patch_size
        )
        self.layernorm = ConvNextLayerNorm(config.hidden_sizes[0], eps=1e-6, data_format="channels_first")
        self.num_channels = config.num_channels

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        num_channels = pixel_values.shape[1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        embeddings = self.patch_embeddings(pixel_values)
        embeddings = self.layernorm(embeddings)
        return embeddings


class ConvNextLayer(nn.Module):
    """This corresponds to the `Block` class in the original implementation.

    There are two equivalent implementations: [DwConv, LayerNorm (channels_first), Conv, GELU,1x1 Conv]; all in (N, C,
    H, W) (2) [DwConv, Permute to (N, H, W, C), LayerNorm (channels_last), Linear, GELU, Linear]; Permute back

    The authors used (2) as they find it slightly faster in PyTorch.

    Args:
        config ([`ConvNextConfig`]): Model configuration class.
        dim (`int`): Number of input channels.
        drop_path (`float`): Stochastic depth rate. Default: 0.0.
    """

    def __init__(self, config, dim, drop_path=0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.layernorm = ConvNextLayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = ACT2FN[config.hidden_act]
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.layer_scale_parameter = (
            nn.Parameter(config.layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if config.layer_scale_init_value > 0
            else None
        )
        self.drop_path = ConvNextDropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = features
        features = self.dwconv(features)
        features = features.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        features = self.layernorm(features)
        features = self.pwconv1(features)
        features = self.act(features)
        features = self.pwconv2(features)
        if self.layer_scale_parameter is not None:
            features = self.layer_scale_parameter * features
        features = features.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        features = residual + self.drop_path(features)
        return features


class ConvNextStage(nn.Module):
    """ConvNeXT stage, consisting of an optional downsampling layer + multiple residual blocks.

    Args:
        config ([`ConvNextConfig`]): Model configuration class.
        in_channels (`int`): Number of input channels.
        out_channels (`int`): Number of output channels.
        depth (`int`): Number of residual blocks.
        drop_path_rates(`list[float]`): Stochastic depth rates for each layer.
    """

    def __init__(self, config, in_channels, out_channels, kernel_size=2, stride=2, depth=2, drop_path_rates=None):
        super().__init__()

        if in_channels != out_channels or stride > 1:
            self.downsampling_layer = nn.ModuleList(
                [
                    ConvNextLayerNorm(in_channels, eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
                ]
            )
        else:
            self.downsampling_layer = nn.ModuleList()
        drop_path_rates = drop_path_rates or [0.0] * depth
        self.layers = nn.ModuleList(
            [ConvNextLayer(config, dim=out_channels, drop_path=drop_path_rates[j]) for j in range(depth)]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        for layer in self.downsampling_layer:
            features = layer(features)
        for layer in self.layers:
            features = layer(features)
        return features


class ConvNextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.stages = nn.ModuleList()
        drop_path_rates = [
            x.tolist()
            for x in torch.linspace(0, config.drop_path_rate, sum(config.depths), device="cpu").split(config.depths)
        ]
        prev_chs = config.hidden_sizes[0]
        for i in range(config.num_stages):
            out_chs = config.hidden_sizes[i]
            stage = ConvNextStage(
                config,
                in_channels=prev_chs,
                out_channels=out_chs,
                stride=2 if i > 0 else 1,
                depth=config.depths[i],
                drop_path_rates=drop_path_rates[i],
            )
            self.stages.append(stage)
            prev_chs = out_chs

    def forward(
        self, hidden_states: torch.Tensor, output_hidden_states: Optional[bool] = False
    ) -> BaseModelOutputWithNoAttention:
        all_hidden_states = [hidden_states] if output_hidden_states else None

        for layer_module in self.stages:
            hidden_states = layer_module(hidden_states)
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)

        return BaseModelOutputWithNoAttention(last_hidden_state=hidden_states, hidden_states=all_hidden_states)


@auto_docstring
class ConvNextPreTrainedModel(PreTrainedModel):
    config: ConvNextConfig
    base_model_prefix = "convnext"
    main_input_name = "pixel_values"
    _no_split_modules = ["ConvNextLayer"]
    _can_record_outputs = {}  # hidden states are collected explicitly

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, ConvNextLayerNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, ConvNextLayer):
            if module.layer_scale_parameter is not None:
                module.layer_scale_parameter.data.fill_(self.config.layer_scale_init_value)


@auto_docstring
class ConvNextModel(ConvNextPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = ConvNextEmbeddings(config)
        self.encoder = ConvNextEncoder(config)

        # final layernorm layer
        self.layernorm = nn.LayerNorm(config.hidden_sizes[-1], eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self, pixel_values: Optional[torch.FloatTensor] = None, output_hidden_states: Optional[bool] = None
    ) -> BaseModelOutputWithPoolingAndNoAttention:
        if output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        embedding_output = self.embeddings(pixel_values)
        encoder_outputs: BaseModelOutputWithNoAttention = self.encoder(
            embedding_output, output_hidden_states=output_hidden_states
        )
        last_hidden_state = encoder_outputs.last_hidden_state

        # global average pooling, (N, C, H, W) -> (N, C)
        pooled_output = self.layernorm(last_hidden_state.mean([-2, -1]))

        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )


@auto_docstring(
    custom_intro="""
    ConvNext Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """
)
class ConvNextForImageClassification(ConvNextPreTrainedModel):
    accepts_loss_kwargs = False

    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.convnext = ConvNextModel(config)

        # Classifier head
        if config.num_labels > 0:
            self.classifier = nn.Linear(config.hidden_sizes[-1], config.num_labels)
        else:
            self.classifier = nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self, pixel_values: Optional[torch.FloatTensor] = None, labels: Optional[torch.LongTensor] = None, **kwargs
    ) -> ImageClassifierOutputWithNoAttention:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs: BaseModelOutputWithPoolingAndNoAttention = self.convnext(pixel_values, **kwargs)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(labels=labels, pooled_logits=logits, config=self.config)

        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )


@auto_docstring(
    custom_intro="""
    ConvNeXt backbone, to be used with frameworks like DETR and MaskFormer.
    """
)
class ConvNextBackbone(ConvNextPreTrainedModel, BackboneMixin):
    has_attentions = False

    def __init__(self, config):
        super().__init__(config)
        super()._init_backbone(config)

        self.embeddings = ConvNextEmbeddings(config)
        self.encoder = ConvNextEncoder(config)
        self.num_features = [config.hidden_sizes[0]] + config.hidden_sizes

        # Add layer norms to hidden states of out_features
        hidden_states_norms = {}
        for stage, num_channels in zip(self._out_features, self.channels):
            hidden_states_norms[stage] = ConvNextLayerNorm(num_channels, data_format="channels_first")
        self.hidden_states_norms = nn.ModuleDict(hidden_states_norms)

        # initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
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

        >>> processor = AutoImageProcessor.from_pretrained("facebook/convnext-tiny-224")
        >>> model = AutoBackbone.from_pretrained("facebook/convnext-tiny-224")

        >>> inputs = processor(image, return_tensors="pt")
        >>> outputs = model(**inputs)
        ```"""
        if output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states

        embedding_output = self.embeddings(pixel_values)
        outputs: BaseModelOutputWithPoolingAndNoAttention = self.encoder(embedding_output, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        feature_maps = []
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                hidden_state = self.hidden_states_norms[stage](hidden_state)
                feature_maps.append(hidden_state)

        return BackboneOutput(
            feature_maps=tuple(feature_maps),
            hidden_states=hidden_states if output_hidden_states else None,
        )


__all__ = ["ConvNextForImageClassification", "ConvNextModel", "ConvNextPreTrainedModel", "ConvNextBackbone"]
