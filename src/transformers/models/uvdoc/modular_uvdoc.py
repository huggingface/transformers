# Copyright 2026 The PaddlePaddle Team and The HuggingFace Inc. team. All rights reserved.
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

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2.functional as tvF
from huggingface_hub.dataclasses import strict

from ...activations import ACT2FN
from ...configuration_utils import PreTrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...image_processing_utils_fast import BaseImageProcessorFast
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import SizeDict, PILImageResampling
from ...modeling_outputs import BaseModelOutputWithNoAttention
from ...modeling_utils import PreTrainedModel
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TransformersKwargs, auto_docstring, requires_backends
from ...utils.generic import TensorType
from ...utils.output_capturing import capture_outputs

from ..pp_lcnet.modeling_pp_lcnet import PPLCNetConvLayer


@auto_docstring(checkpoint="PaddlePaddle/UVDoc_safetensors")
@strict(accept_kwargs=True)
class UVDocConfig(PreTrainedConfig):
    r"""
    num_filter (`int`, *optional*, defaults to 32):
        Number of convolutional filters in the initial convolutional layers.
    in_channels (`int`, *optional*, defaults to 3):
        Number of input channels. Defaults to 3 for RGB images; set to 1 for grayscale images.
    kernel_size (`int`, *optional*, defaults to 5):
        Kernel size for convolutional layers in the backbone network.
    block_stride_values (`list[int]` or `tuple[int, ...]`, *optional*, defaults to `(1, 2, 2)`):
        Strides for downsampling operations in the backbone network.
    feature_map_multipliers (`list[int]` or `tuple[int, ...]`, *optional*, defaults to `(1, 2, 4)`):
        Scaling factors for feature map dimensions in multi-scale feature fusion modules.
    block_counts_per_stage (`list[int]` or `tuple[int, ...]`, *optional*, defaults to `(3, 4, 6)`):
        Number of residual blocks in each stage of the model backbone.
    dilation_values (`tuple[list[int]]` or `tuple[tuple[int, ...], ...]`, *optional*, defaults to `((1,), (2,), (5,), (8, 3, 2), (12, 7, 4), (18, 12, 6))`):
        Dilation rates for dilated convolutional layers in bridge modules. Each inner tuple/list contains dilation
        rates for a single bridge block.
    padding_mode (`str`, *optional*, defaults to `"reflect"`):
        Padding mode for convolutional layers. Supported modes are `"reflect"`, `"constant"`, and `"replicate"`.
    """

    model_type = "uvdoc"

    num_filter: int = 32
    in_channels: int = 3
    kernel_size: int = 5
    block_stride_values: list[int] | tuple[int, ...] = (1, 2, 2)
    feature_map_multipliers: list[int] | tuple[int, ...] = (1, 2, 4)
    block_counts_per_stage: list[int] | tuple[int, ...] = (3, 4, 6)
    dilation_values: list[list[int]] | tuple[tuple[int, ...], ...] = (
        (1,),
        (2,),
        (5,),
        (8, 3, 2),
        (12, 7, 4),
        (18, 12, 6),
    )
    padding_mode: str = "reflect"


@auto_docstring
class UVDocImageProcessorFast(BaseImageProcessorFast):
    do_rescale = True
    do_resize = True
    size = {"height": 712, "width": 488}
    resample = PILImageResampling.BILINEAR

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["tvF.InterpolationMode"],
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        requires_backends(self, "torch")
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            # RGB to BGR conversion
            stacked_images = stacked_images[:, [2, 1, 0], :, :]
            processed_images_grouped[shape] = stacked_images

        rescale_and_normalize_images = reorder_images(processed_images_grouped, grouped_images_index)

        original_images = rescale_and_normalize_images.copy()

        grouped_images, grouped_images_index = group_images_by_shape(
            rescale_and_normalize_images, disable_grouping=disable_grouping
        )
        interpolated_images_grouped = {}
        # Upsample images and extract originals for post-processing
        for shape, stacked_images in grouped_images.items():
            # Interpolate to target size (use interpolate with align_corners=True to match original implementation)
            if do_resize:
                stacked_images = F.interpolate(
                    stacked_images, size=(size.height, size.width), mode=interpolation.value, align_corners=True
                )
            interpolated_images_grouped[shape] = stacked_images

        pixel_values = reorder_images(interpolated_images_grouped, grouped_images_index)

        return BatchFeature(
            data={"pixel_values": pixel_values, "original_images": original_images},
            tensor_type=return_tensors,
            skip_tensor_conversion=["original_images"],
        )

    def post_process_document_rectification(
        self,
        prediction: torch.Tensor,
        original_images: list[torch.Tensor],
        scale: float = 255.0,
    ) -> list[dict[str, torch.Tensor]]:
        """
        Post-process document rectification predictions to convert them into rectified images.

        Args:
            prediction: Predicted 2D Bezier mesh coordinates, shape (B, 2, H, W)
            original_images: List of original input tensors, each of shape (C, H_i, W_i). Images may have different sizes.
            scale: Scaling factor for output images (default: 255.0)

        Returns:
            List of dictionaries containing rectified images. Each dictionary has:
                - "images": Rectified image tensor of shape (H, W, 3) with dtype torch.uint8
                          and BGR channel order (suitable for OpenCV visualization)
        """
        requires_backends(self, "torch")
        image_list = list(original_images)
        scale = torch.tensor(float(scale), device=prediction.device)
        results = []

        for i, original_image in enumerate(image_list):
            # Ensure (1, C, H, W) for grid_sample
            if original_image.ndim == 3:
                original_image = original_image.unsqueeze(0)
            original_image = original_image.to(prediction.device)
            original_height, original_width = original_image.shape[2:]

            # Upsample predicted mesh for this image to its original size
            upsampled_mesh = F.interpolate(
                prediction[i : i + 1],
                size=(original_height, original_width),
                mode="bilinear",
                align_corners=True,
            )
            # Permute mesh for grid_sample: (1, H, W, 2)
            rearranged_mesh = upsampled_mesh.permute(0, 2, 3, 1)

            # Apply spatial transformation to rectify the document
            rectified = F.grid_sample(original_image, rearranged_mesh, align_corners=True)

            # Remove batch dimension and rearrange channels: (H, W, C)
            image = rectified.squeeze(0).permute(1, 2, 0)

            # Scale and convert to uint8 with BGR channel
            image = image * scale

            image = image.flip(dims=[-1]).to(dtype=torch.uint8, non_blocking=True, copy=False)

            results.append({"images": image})

        return results


class UVDocConvLayer(PPLCNetConvLayer):
    """Convolutional layer with batch normalization and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        padding_mode: str = "zeros",
        bias: bool = False,
        dilation: int = 1,
        activation: str = "relu",
    ):
        super().__init__()

        self.convolution = nn.Conv2d(
            in_channels,
            out_channels,
            bias=bias,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            dilation=dilation,
        )


class UVDocResidualBlockWithDilation(nn.Module):
    """Residual block with optional downsampling and dilation support."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        downsample: bool | None = None,
        block_index: bool = False,
        activation: str = "relu",
    ):
        super().__init__()

        self.conv_down = None
        if downsample:
            self.conv_down = UVDocConvLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=True,
                activation=None,
            )

        if stride != 1 or block_index == 0:
            stride, padding, dilation = stride, kernel_size // 2, 1
        else:
            stride, padding, dilation = 1, 3 * (kernel_size // 2), 3

        self.conv_start = UVDocConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        )

        self.conv_final = UVDocConvLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=True,
            dilation=dilation,
            activation=None,
        )

        self.act_fn = ACT2FN[activation] if activation is not None else nn.Identity()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        if self.conv_down is not None:
            residual = self.conv_down(hidden_states)
        hidden_states = self.conv_start(hidden_states)
        hidden_states = self.conv_final(hidden_states)
        hidden_states = hidden_states + residual

        hidden_states = self.act_fn(hidden_states)
        return hidden_states


class UVDocResNetStage(nn.Module):
    """A ResNet stage containing multiple residual blocks."""

    def __init__(self, config, in_channels, feature_map_multipliers, block_count, block_stride):
        super().__init__()
        out_channels = config.num_filter * feature_map_multipliers

        downsample = None
        if block_stride != 1 or in_channels != out_channels:
            downsample = True

        self.layers = nn.ModuleList([])
        for index in range(block_count):
            layer = UVDocResidualBlockWithDilation(
                in_channels=in_channels if index == 0 else out_channels,
                out_channels=out_channels,
                kernel_size=config.kernel_size,
                stride=block_stride if index == 0 else 1,
                downsample=downsample if index == 0 else None,
                block_index=index,
            )
            self.layers.append(layer)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class UVDocResNet(nn.Module):
    """ResNet backbone with multiple stages for feature extraction."""

    def __init__(self, config):
        super().__init__()
        self.in_channels = config.num_filter * config.feature_map_multipliers[0]

        self.stages = nn.ModuleList([])
        for feature_map, block_count, block_stride in zip(
            config.feature_map_multipliers, config.block_counts_per_stage, config.block_stride_values
        ):
            stage = UVDocResNetStage(
                config=config,
                in_channels=self.in_channels,
                feature_map_multipliers=feature_map,
                block_count=block_count,
                block_stride=block_stride,
            )
            self.stages.append(stage)
            self.in_channels = config.num_filter * feature_map

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            hidden_states = stage(hidden_states)
        return hidden_states


class UVDocResNetHead(nn.Module):
    """Initial processing head with downsample and upsample convolutions."""

    def __init__(self, config):
        super().__init__()
        in_channels = config.in_channels
        num_filter = config.num_filter
        map_number = config.feature_map_multipliers[0]
        kernel_size = config.kernel_size
        out_channels = num_filter * map_number

        self.conv_down = UVDocConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
        )

        self.conv_up = UVDocConvLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_down(hidden_states)
        hidden_states = self.conv_up(hidden_states)
        return hidden_states


class UVDocBridgeBlock(nn.Module):
    """Bridge module with dilated convolutions for long-range dependencies."""

    def __init__(self, config, dilation_values):
        super().__init__()
        in_channels = config.num_filter * config.feature_map_multipliers[2]

        self.blocks = nn.ModuleList([])
        for dilation in dilation_values:
            self.blocks.append(UVDocConvLayer(in_channels, in_channels, padding=dilation, dilation=dilation))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return hidden_states


class UVDocPointPositions2D(nn.Module):
    """Module for predicting 2D point positions for document rectification."""

    def __init__(self, config):
        super().__init__()

        self.conv_down = UVDocConvLayer(
            in_channels=config.num_filter * config.feature_map_multipliers[2],
            out_channels=config.num_filter * config.feature_map_multipliers[0],
            kernel_size=config.kernel_size,
            stride=1,
            padding=config.kernel_size // 2,
            padding_mode=config.padding_mode,
            activation="prelu",
        )

        self.conv_up = nn.Conv2d(
            config.num_filter * config.feature_map_multipliers[0],
            2,
            kernel_size=config.kernel_size,
            stride=1,
            padding=config.kernel_size // 2,
            padding_mode=config.padding_mode,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_down(hidden_states)
        hidden_states = self.conv_up(hidden_states)
        return hidden_states


@auto_docstring
class UVDocPreTrainedModel(PreTrainedModel):
    config: UVDocConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"
    input_modalities = ("image",)
    _can_compile_fullgraph = True
    _can_record_outputs = {
        "hidden_states": UVDocBridgeBlock,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        """Initialize the weights."""
        if isinstance(module, nn.PReLU):
            module.reset_parameters()


@auto_docstring(
    custom_intro=r"""
    The model takes raw document images (pixel values) as input, processes them through the UVDoc backbone to predict spatial transformation parameters,
    and outputs the rectified (corrected) document image tensor.
    """
)
class UVDocModel(UVDocPreTrainedModel):

    def __init__(self, config: UVDocConfig):
        super().__init__(config)

        self.resnet_head = UVDocResNetHead(config)
        self.resnet_down = UVDocResNet(config)

        self.bridge = nn.ModuleList([])
        for dilation_value in config.dilation_values:
            self.bridge.append(UVDocBridgeBlock(config, dilation_value))

        self.num_bridge_layers = len(self.bridge)

        self.bridgeconnector = UVDocConvLayer(
            in_channels=config.num_filter * config.feature_map_multipliers[2] * self.num_bridge_layers,
            out_channels=config.num_filter * config.feature_map_multipliers[2],
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
        )

        self.out_point_positions2D = UVDocPointPositions2D(config)
        self.post_init()

    @capture_outputs
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor] | BaseModelOutputWithNoAttention:
        hidden_states = self.resnet_head(pixel_values)
        resnet_down = self.resnet_down(hidden_states)

        bridge_outputs = []
        for bridge_layer in self.bridge:
            bridge_output = bridge_layer(resnet_down)
            bridge_outputs.append(bridge_output)

        bridge_concat = torch.cat(bridge_outputs, dim=1)
        bridge = self.bridgeconnector(bridge_concat)

        out_point_positions2D = self.out_point_positions2D(bridge)

        return BaseModelOutputWithNoAttention(last_hidden_state=out_point_positions2D)


__all__ = [
    "UVDocImageProcessorFast",
    "UVDocConfig",
    "UVDocModel",
    "UVDocPreTrainedModel",
]
