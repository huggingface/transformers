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

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modeling_outputs import SemanticSegmenterOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    can_return_tuple,
)
from ...utils.backbone_utils import load_backbone
from .configuration_fast import FastConfig


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

FAST_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Use [`FastImageProcessor`] to preprocess input images. See [`FastImageProcessor.__call__`] for details.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers.
        labels (`Dict[str, torch.Tensor]`, *optional*):
            Dictionary of ground truth tensors used for computing loss. Should contain:
            - `"gt_texts"`: ground truth text regions
            - `"gt_kernels"`: kernel regions for training
            - `"gt_instances"`: instance segmentation labels
"""


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        padding1 = get_same_padding(kernel_size[0])
        padding2 = get_same_padding(kernel_size[1])
        return padding1, padding2
    return kernel_size // 2


class FastRepConvLayer(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: tuple[int, int], stride: int = 1, bias: bool = False
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        padding = (int((kernel_size[0] - 1) / 2), int((kernel_size[1] - 1) / 2))

        self.activation = nn.ReLU(inplace=True)
        self.identity = nn.Identity()
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
            self.vertical_conv = nn.Identity()
            self.vertical_batch_norm = nn.Identity()

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
            self.horizontal_conv = nn.Identity()
            self.horizontal_batch_norm = nn.Identity()

        if out_channels == in_channels and stride == 1:
            self.use_identity = True
            self.identity = nn.BatchNorm2d(out_channels)
        else:
            self.use_identity = False
            self.identity = None

    def forward(self, hidden_states: torch.Tensor):
        main = self.main_conv(hidden_states)
        main = self.main_batch_norm(main)

        vertical = self.vertical_conv(hidden_states)
        vertical = self.vertical_batch_norm(vertical)

        horizontal = self.horizontal_conv(hidden_states)
        horizontal = self.horizontal_batch_norm(horizontal)

        identity = self.identity(hidden_states) if self.use_identity else 0

        return self.activation(main + vertical + horizontal + identity)


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
            module.weight.data.zero_()
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()


class FastNeck(nn.Module):
    def __init__(self, config: FastConfig):
        super().__init__()
        self.reduce_layers = nn.ModuleList()
        for in_channels, out_channels, kernel_size, stride in zip(
            config.neck_in_channels,
            config.neck_out_channels,
            config.neck_kernel_size,
            config.neck_stride,
        ):
            layer = FastRepConvLayer(in_channels, out_channels, kernel_size, stride)
            self.reduce_layers.append(layer)
        self.num_layers = len(self.reduce_layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
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


class FastHead(nn.Module):
    def __init__(self, config: FastConfig) -> None:
        super().__init__()
        self.conv = FastRepConvLayer(
            config.head_conv_in_channels,
            config.head_conv_out_channels,
            config.head_conv_kernel_size,
            config.head_conv_stride,
        )

        padding = get_same_padding(config.head_final_kernel_size)

        self.final_conv = nn.Conv2d(
            config.head_final_in_channels,
            config.head_final_out_channels,
            kernel_size=config.head_final_kernel_size,
            stride=config.head_final_stride,
            padding=padding,
            bias=config.head_final_bias,
        )

        self.pooling_size = config.head_pooling_size
        self.pooling_1s = nn.MaxPool2d(kernel_size=self.pooling_size, stride=1, padding=(self.pooling_size - 1) // 2)
        self.pooling_2s = nn.MaxPool2d(
            kernel_size=self.pooling_size // 2 + 1, stride=1, padding=(self.pooling_size // 2) // 2
        )

        if config.head_dropout_ratio > 0:
            self.dropout = nn.Dropout2d(config.head_dropout_ratio)
        else:
            self.dropout = nn.Identity()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.final_conv(hidden_states)
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

        self.backbone = load_backbone(config.backbone_config)
        self.neck = FastNeck(config=config)
        self.text_detection_head = FastHead(config=config)

        self.pooling_1s = nn.MaxPool2d(
            kernel_size=config.head_pooling_size, stride=1, padding=(config.head_pooling_size - 1) // 2
        )
        self.pooling_2s = nn.MaxPool2d(
            kernel_size=config.head_pooling_size // 2 + 1, stride=1, padding=(config.head_pooling_size // 2) // 2
        )
        self.post_init()

    @can_return_tuple
    @add_start_docstrings_to_model_forward(FAST_INPUTS_DOCSTRING)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_hidden_states: Optional[bool] = True,
        labels: Optional[dict] = None,
    ):
        r"""
        labels (`Dict[str, torch.Tensor]`, *optional*):
            Should contain 3 keys: gt_texts, gt_kernels, gt_instances

        Returns:

                Examples:

        ```python
        >>> from transformers import FastImageProcessor, FastForSceneTextRecognition
        >>> from PIL import Image
        >>> import requests
        >>> import torch
        >>> url = "https://huggingface.co/datasets/Raghavan/fast_model_samples/resolve/main/img657.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        >>> processor = FastImageProcessor.from_pretrained("jadechoghari/fast-tiny")
        >>> model = FastForSceneTextRecognition.from_pretrained("jadechoghari/fast-tiny")
        >>> inputs = processor(image, return_tensors="pt")
        >>> with torch.no_grad():
        ...     outputs = model(**inputs)
        >>> target_sizes = [(image.height, image.width)]
        >>> text_locations = processor.post_process_text_detection(
        ...     outputs,
        ...     target_sizes=target_sizes,
        ...     threshold=0.88,
        ...     output_type="boxes"  # or "polygons"
        ... )
        >>> print(text_locations[0]["boxes"][0])
        [151, 151, 160, 56, 355, 74, 346, 169]
        ```
        """
        features = self.backbone(pixel_values).feature_maps

        hidden_states = self.neck(features)

        text_detection_output = self.text_detection_head(hidden_states)

        all_hidden_states = (features, hidden_states)

        loss = None
        if labels:
            upsampled_output = F.interpolate(
                text_detection_output, size=(pixel_values.size(2), pixel_values.size(3)), mode="bilinear"
            )
            kernels = upsampled_output[:, 0, :, :]  # 4*640*640
            texts = self.pooling_1s(kernels)  # 4*640*640
            loss = self.loss_function(upsampled_output, labels, texts)

        text_detection_output = F.interpolate(
            text_detection_output, size=(pixel_values.size(2) // 4, pixel_values.size(3) // 4), mode="bilinear"
        )

        return SemanticSegmenterOutput(
            loss=loss,
            logits=text_detection_output,
            hidden_states=all_hidden_states if output_hidden_states else None,
        )


__all__ = ["FastPreTrainedModel", "FastForSceneTextRecognition"]
