# coding=utf-8
# Copyright 2023 HUST-VL and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch ViTMatte model."""

from typing import Optional

import torch
from torch import nn

from ... import AutoBackbone
from ...modeling_utils import PreTrainedModel
from ...utils.backbone_utils import BackboneMixin
from .configuration_vitmatte import VitMatteConfig


VITMATTE_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "hustvl/vitmatte-small-composition-1k",
    # See all VitMatte models at https://huggingface.co/models?filter=vitmatte
]


class VitMattePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = VitMatteConfig
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, VitMattePreTrainedModel):
            module.backbone.init_weights()

    def init_weights(self):
        """Initialize the weights"""
        self.backbone.init_weights()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BackboneMixin):
            module.gradient_checkpointing = value


class VitMatteBasicConv3x3(nn.Module):
    """
    Basic convolution layers including: Conv3x3, BatchNorm2d, ReLU layers.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        padding=1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class VitMatteConvStream(nn.Module):
    """
    Simple ConvStream containing a series of basic conv3x3 layers to extract detail features.
    """

    def __init__(
        self,
        in_channels=4,
        out_channels=[48, 96, 192],
    ):
        super().__init__()
        self.convs = nn.ModuleList()

        self.conv_chans = out_channels
        self.conv_chans.insert(0, in_channels)

        for i in range(len(self.conv_chans) - 1):
            in_chan_ = self.conv_chans[i]
            out_chan_ = self.conv_chans[i + 1]
            self.convs.append(VitMatteBasicConv3x3(in_chan_, out_chan_))

    def forward(self, x):
        out_dict = {"D0": x}
        for i in range(len(self.convs)):
            x = self.convs[i](x)
            name_ = "D" + str(i + 1)
            out_dict[name_] = x

        return out_dict


class VitMatteFusionBlock(nn.Module):
    """
    Simple fusion block to fuse features from ConvStream and Plain Vision Transformer.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = VitMatteBasicConv3x3(in_channels, out_channels, stride=1, padding=1)

    def forward(self, x, D):
        F_up = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        out = torch.cat([D, F_up], dim=1)
        out = self.conv(out)

        return out


class VitMatteHead(nn.Module):
    """
    Simple Matting Head, containing only conv3x3 and conv1x1 layers.
    """

    def __init__(self, in_channels=32, mid_channels=16):
        super().__init__()
        self.matting_convs = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(True),
            nn.Conv2d(mid_channels, 1, 1, 1, 0),
        )

    def forward(self, x):
        x = self.matting_convs(x)

        return x


class VitMatteDetailCaptureModule(nn.Module):
    """
    Simple and lightweight Detail Capture Module for ViT Matting.
    """

    def __init__(
        self,
        config,
        image_channels=4,
        convstream_out=[48, 96, 192],
        fusion_out=[256, 128, 64, 32],
    ):
        super().__init__()
        if len(fusion_out) != len(convstream_out) + 1:
            raise ValueError("The length of fusion_out should be equal to the length of convstream_out + 1.")

        self.convstream = VitMatteConvStream(in_channels=image_channels)
        self.conv_chans = self.convstream.conv_chans

        self.fusion_blocks = nn.ModuleList()
        self.fusion_channels = fusion_out
        in_channels = config.hidden_size
        self.fusion_channels.insert(0, in_channels)
        for i in range(len(self.fusion_channels) - 1):
            self.fusion_blocks.append(
                VitMatteFusionBlock(
                    in_channels=self.fusion_channels[i] + self.conv_chans[-(i + 1)],
                    out_channels=self.fusion_channels[i + 1],
                )
            )

        self.matting_head = VitMatteHead(in_channels=fusion_out[-1])

    def forward(self, features, images):
        detail_features = self.convstream(images)
        for i in range(len(self.fusion_blocks)):
            d_name_ = "D" + str(len(self.fusion_blocks) - i - 1)
            features = self.fusion_blocks[i](features, detail_features[d_name_])

        phas = torch.sigmoid(self.matting_head(features))

        return phas


class VitMatteForImageMatting(VitMattePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.backbone = AutoBackbone.from_config(config.backbone_config)
        self.decoder = VitMatteDetailCaptureModule(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, VitMatteForImageMatting
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> processor = AutoImageProcessor.from_pretrained("hustvl/vitmatte-small-composition-1k")
        >>> model = VitMatteForImageMatting.from_pretrained("hustvl/vitmatte-small-composition-1k")

        >>> inputs = processor(image, return_tensors="pt")
        >>> outputs = model(**inputs)
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        outputs = self.backbone.forward_with_filtered_kwargs(
            pixel_values, output_hidden_states=output_hidden_states, output_attentions=output_attentions
        )

        features = outputs.feature_maps[-1]

        # TODO: already permute in backbone?
        features = features.permute(0, 3, 1, 2)
        print("Shape of backbone features:", features.shape)
        outputs = self.decoder(features, pixel_values)

        if labels is not None:
            raise NotImplementedError("Training is not yet supported")

        return outputs
