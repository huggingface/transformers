# coding=utf-8
# Copyright 2025 the Fast authors and HuggingFace Inc. team.  All rights reserved.
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
"""PyTorch FAST model."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils import is_timm_available


if is_timm_available():
    from timm import create_model


from transformers import (
    AutoBackbone,
    FastConfig,
    PreTrainedModel,
    add_start_docstrings,
    is_timm_available,
    requires_backends,
)
from transformers.utils import ModelOutput, add_start_docstrings_to_model_forward, replace_return_docstrings


_CONFIG_FOR_DOC = "FastConfig"

FAST_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`FastConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

FAST_FOR_CAPTIONING_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`FastImageProcessor.__call__`] for details.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

@dataclass
class FastForSceneTextRecognitionOutput(ModelOutput):
    """
    Adapted from the base class for vision model's outputs that also contains image embeddings of the pooling of the
    last hidden states. This class also adds the loss term from the text decoder as well as the image-text similarity
    scores.

    Args:
        loss (`torch.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Languge modeling loss from the text decoder.
        text_hidden (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional*):
            The image hidden states.
    """

    loss: Optional[torch.Tensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None

def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        padding1 = get_same_padding(kernel_size[0])
        padding2 = get_same_padding(kernel_size[1])
        return padding1, padding2
    return kernel_size // 2


class FastConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        bias=False,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride

        padding = get_same_padding(self.kernel_size)

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        return hidden_states

class FastRepConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: int = 1,
        bias: bool = False
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        padding = (int((kernel_size[0] - 1) / 2), int((kernel_size[1] - 1) / 2))

        self.activation = nn.ReLU(inplace=True)

        self.main_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.main_batch_norm = nn.BatchNorm2d(num_features=out_channels)

        vertical_padding = (int((kernel_size[0] - 1) / 2), 0)
        horizontal_padding = (0, int((kernel_size[1] - 1) / 2))

        if kernel_size[1] != 1:
            self.vertical_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size[0], 1),
                stride=stride,
                padding=vertical_padding,
                bias=bias,
            )
            self.vertical_batch_norm = nn.BatchNorm2d(num_features=out_channels)
        else:
            self.vertical_conv, self.vertical_batch_norm = None, None

        if kernel_size[0] != 1:
            self.horizontal_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, kernel_size[1]),
                stride=stride,
                padding=horizontal_padding,
                bias=bias,
            )
            self.horizontal_batch_norm = nn.BatchNorm2d(num_features=out_channels)
        else:
            self.horizontal_conv, self.horizontal_batch_norm = None, None

        if out_channels == in_channels and stride == 1:
            self.identity_batch_norm = nn.BatchNorm2d(num_features=in_channels)
        else:
            self.identity_batch_norm = None
        
        #TODO: check if needed
        self.rbr_identity = (
            nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
        )

    def forward(self, hidden_states):
        # if self.training:

        main_outputs = self.main_conv(hidden_states)
        main_outputs = self.main_batch_norm(main_outputs)
        if self.vertical_conv is not None:
            vertical_outputs = self.vertical_conv(hidden_states)
            vertical_outputs = self.vertical_batch_norm(vertical_outputs)
        else:
            vertical_outputs = 0

        if self.horizontal_conv is not None:
            horizontal_outputs = self.horizontal_conv(hidden_states)
            horizontal_outputs = self.horizontal_batch_norm(horizontal_outputs)
        else:
            horizontal_outputs = 0

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(hidden_states)

        return self.activation(main_outputs + vertical_outputs + horizontal_outputs + id_out)


class FastPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = FastConfig
    base_model_prefix = "fast"
    main_input_name = "pixel_values"

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight.data, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()


class FASTNeck(nn.Module):
    def __init__(self, config):
        super().__init__()
        reduce_layer_configs = list(
            zip(
                config.neck_in_channels,
                config.neck_out_channels,
                config.neck_kernel_size,
                config.neck_stride,
            )
        )
        self.num_layers = len(reduce_layer_configs)
        self.reduce_layers = nn.ModuleList([
            FastRepConvLayer(*config) for config in reduce_layer_configs
        ])

    def forward(self, hidden_states):
        first_layer_hidden = hidden_states[0]
        first_layer_hidden = self.reduce_layers[0](first_layer_hidden)
        output_stages = [first_layer_hidden]

        for layer_ix in range(1, self.num_layers):
            layer_out = self.reduce_layers[layer_ix](hidden_states[layer_ix])
            _, _, height, width = first_layer_hidden.size()
            layer_out = F.interpolate(layer_out, size=(height, width), mode="bilinear")
            output_stages.append(layer_out)

        combined_hidden_states = torch.cat(output_stages, 1)
        return combined_hidden_states


class FASTHead(nn.Module):
    def __init__(self, config: FastConfig) -> None:
        super().__init__()
        self.conv = FastRepConvLayer(
            config.head_conv_in_channels,
            config.head_conv_out_channels,
            config.head_conv_kernel_size,
            config.head_conv_stride,
        )

        self.final = FastConvLayer(
            config.head_final_in_channels,
            config.head_final_out_channels,
            config.head_final_kernel_size,
            config.head_final_stride,
            config.head_final_bias,
        )

        self.pooling_size = config.head_pooling_size

        self.pooling_1s = nn.MaxPool2d(kernel_size=self.pooling_size, stride=1, padding=(self.pooling_size - 1) // 2)
        self.pooling_2s = nn.MaxPool2d(
            kernel_size=self.pooling_size // 2 + 1, stride=1, padding=(self.pooling_size // 2) // 2
        )

        if config.head_dropout_ratio > 0:
            self.dropout = nn.Dropout2d(config.head_dropout_ratio)
        else:
            self.dropout = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv(hidden_states)
        if self.dropout is not None:
            hidden_states = self.dropout(hidden_states)
        hidden_states = self.final(hidden_states)
        return hidden_states



@add_start_docstrings(
    """FAST (faster arbitararily-shaped text detector) proposes an accurate and efficient scene text detection
    framework, termed FAST (i.e., faster arbitrarily-shaped text detector).FAST has two new designs. (1) They design a
    minimalist kernel representation (only has 1-channel output) to model text with arbitrary shape, as well as a
    GPU-parallel post-processing to efficiently assemble text lines with a negligible time overhead. (2) We search the
    network architecture tailored for text detection, leading to more powerful features than most networks that are
    searched for image classification.""",
    FAST_START_DOCSTRING,
)
class FastForSceneTextRecognition(FastPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        if config.use_timm_backbone:
            requires_backends(self, ["timm"])
            kwargs = {}
            if config.dilation:
                kwargs["output_stride"] = 16
            backbone = create_model(
                config.backbone,
                pretrained=config.use_pretrained_backbone,
                features_only=True,
                out_indices=(1, 2, 3, 4),
                in_chans=config.num_channels,
                **kwargs,
            )
        else:
            backbone = AutoBackbone.from_config(config.backbone_config)
        self.backbone = backbone
        self.neck = FASTNeck(config=config)
        self.text_detection_head = FASTHead(config=config)

        self.pooling_1s = nn.MaxPool2d(
            kernel_size=config.head_pooling_size, stride=1, padding=(config.head_pooling_size - 1) // 2
        )
        self.pooling_2s = nn.MaxPool2d(
            kernel_size=config.head_pooling_size // 2 + 1, stride=1, padding=(config.head_pooling_size // 2) // 2
        )
        self.post_init()

    def _upsample(self, x, size, scale=1):
        _, _, H, W = size
        return F.interpolate(x, size=(H // scale, W // scale), mode="bilinear")

    def _max_pooling(self, x, scale=1):
        if scale == 1:
            x = self.pooling_1s(x)
        elif scale == 2:
            x = self.pooling_2s(x)
        return x


    @add_start_docstrings_to_model_forward(FAST_FOR_CAPTIONING_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FastForSceneTextRecognitionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = None,
        labels: Dict = None,
    ):
        r"""
        labels (`Dict[str, torch.Tensor]`, *optional*):
            Should contain 3 keys: gt_texts,gt_kernels,gt_instances

        Returns:

                Examples:

        ```python
        >>> from transformers import FastImageProcessor, FastForSceneTextRecognition
        >>> from PIL import Image
        >>> import requests

        >>> url = "https://huggingface.co/datasets/Raghavan/fast_model_samples/resolve/main/img657.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        >>> processor = FastImageProcessor.from_pretrained("jadechoghari/fast-tiny")
        >>> model = FastForSceneTextRecognition.from_pretrained("jadechoghari/fast-tiny")
        >>> inputs = processor(image, return_tensors="pt")
        >>> # forward pass
        >>> outputs = model(pixel_values=inputs["pixel_values"])
        >>> target_sizes = [(image.height, image.width)]
        >>> threshold = 0.88
        >>> text_locations = processor.post_process_text_detection(outputs, target_sizes, threshold, bbox_type="rect")
        >>> print(text_locations[0]["bboxes"][0][:10])
        [151, 151, 160, 56, 355, 74, 346, 169]
        ```
        """
        # outputs = {}
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        features = (
            self.backbone(pixel_values) if self.config.use_timm_backbone else self.backbone(pixel_values).feature_maps
        )

        hidden_states = self.neck(features)

        text_detection_output = self.text_detection_head(hidden_states)

        all_hidden_states = (features, hidden_states)

        loss = None
        if labels:
            upsampled_output = F.interpolate(
                text_detection_output, 
                size=(pixel_values.size(2), pixel_values.size(3)), 
                mode="bilinear"
            )
            # TODO: refactor
            loss = self.loss_function(upsampled_output, labels)
            kernels = upsampled_output[:, 0, :, :]  # 4*640*640
            texts = self._max_pooling(kernels, scale=1)  # 4*640*640
            loss = loss(upsampled_output, labels, texts)
            
        text_detection_output = F.interpolate(
            text_detection_output,
            size=(pixel_values.size(2) // 4, pixel_values.size(3) // 4),
            mode="bilinear"
        )

        if not return_dict:
            output = (loss, text_detection_output) if loss is not None else (text_detection_output,)
            return output + (all_hidden_states,) if output_hidden_states else output

        return FastForSceneTextRecognitionOutput(
            loss=loss,
            last_hidden_state=text_detection_output,
            hidden_states=all_hidden_states if output_hidden_states else None,
        )

__all__ = ["FastForSceneTextRecognition"]