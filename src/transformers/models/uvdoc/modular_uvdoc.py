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

from ...activations import ACT2FN
from ...configuration_utils import PreTrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...image_processing_utils_fast import BaseImageProcessorFast
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import SizeDict
from ...modeling_outputs import BaseModelOutputWithNoAttention
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import TensorType
from ...utils.output_capturing import capture_outputs


@auto_docstring(
    checkpoint="PaddlePaddle/UVDoc_safetensors",
    custom_args=r"""
    num_filter (`int`, *optional*, defaults to 32):
        The number of convolutional filters (output channels) in the initial convolutional layers of the model,
        controlling the depth of feature maps extracted from input document images. Larger values increase
        model capacity but also computational cost.
    in_channels (`int`, *optional*, defaults to 3):
        The number of input channels of the model. Defaults to 3 for RGB document images; set to 1 for grayscale
        document images.
    kernel_size (`int`, *optional*, defaults to 5):
        The size of convolutional kernels used in the backbone network, typically an odd integer to ensure
        symmetric padding and preserve spatial dimensions of feature maps.
    block_stride_values (`List[int]`, *optional*, defaults to `[1, 2, 2]`):
        The strides for downsampling operations in the backbone network, corresponding to the scale factor between
        consecutive stages of the model. Smaller strides reduce the spatial dimension of feature maps while retaining
    feature_map_multipliers (`List[int]`, *optional*, defaults to `[1, 2, 4]`):
        The scaling factors for feature map dimensions in multi-scale feature fusion modules, used to align
        feature maps of different resolutions for document structure restoration.
    block_counts_per_stage (`List[int]`, *optional*, defaults to `[3, 4, 6]`):
        The number of residual blocks in each stage of the model backbone, determining the depth of the network.
        More blocks enhance feature extraction capability but increase inference time.
    dilation_values (`List[List[int]]`, *optional*, defaults to `None`):
        A nested list of dilation rates for dilated convolutional layers in bridge modules.
        Each inner list contains dilation rates for a single bridge block.
        Dilated convolution expands the receptive field without increasing kernel size,
        critical for capturing long-range geometric dependencies in distorted documents.
    padding_mode (`str`, *optional*, defaults to `"reflect"`):
        The padding mode for convolutional layers, used to handle boundary pixels of document images. Supported
        modes include `"reflect"` (recommended for document rectification to avoid edge artifacts), `"constant"`,
        and `"replicate"`.
    upsample_size (`List[int]`, *optional*, defaults to `[712, 488]`):
        The target spatial size (width, height) of the upsampled output image, matching the desired resolution
        of the rectified document. Adjust based on your input document size and task requirements.
    upsample_mode (`str`, *optional*, defaults to `"bilinear"`):
        The interpolation mode for upsampling layers to restore the original image resolution. Supported modes
        include `"bilinear"` (smooth upsampling, recommended for document images), `"nearest"`, and `"bicubic"`.
    """,
)
class UVDocConfig(PreTrainedConfig):
    model_type = "uvdoc"

    def __init__(
        self,
        num_filter: int = 32,
        in_channels: int = 3,
        kernel_size: int = 5,
        block_stride_values: list | None = None,
        feature_map_multipliers: list | None = None,
        block_counts_per_stage: list | None = None,
        dilation_values: list | None = None,
        padding_mode: str = "reflect",
        upsample_size: list | None = None,
        upsample_mode: str = "bilinear",
        **kwargs,
    ):
        self.num_filter = num_filter
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.block_stride_values = block_stride_values if block_stride_values is not None else [1, 2, 2]
        self.feature_map_multipliers = feature_map_multipliers if feature_map_multipliers is not None else [1, 2, 4]
        self.block_counts_per_stage = block_counts_per_stage if block_counts_per_stage is not None else [3, 4, 6]
        self.dilation_values = dilation_values if dilation_values is not None else [[1], [2], [5], [8, 3, 2], [12, 7, 4], [18, 12, 6]]
        self.padding_mode = padding_mode
        self.upsample_size = upsample_size
        self.upsample_mode = upsample_mode

        super().__init__(**kwargs)


@auto_docstring
class UVDocImageProcessorFast(BaseImageProcessorFast):
    do_rescale = True

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
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(image=stacked_images, size=size, interpolation=interpolation)
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            # RGB to BGR conversion
            stacked_images = stacked_images[:, [2, 1, 0], :, :]
            processed_images_grouped[shape] = stacked_images

        pixel_values = reorder_images(processed_images_grouped, grouped_images_index)

        return BatchFeature(
            data={"pixel_values": pixel_values},
            tensor_type=return_tensors,
        )

    def post_process_document_rectification(self, images, scale=None):
        if isinstance(scale, (str, float, int)):
            scale = torch.tensor(float(scale), device=images.device)
        else:
            scale = torch.tensor(255.0, device=images.device)

        results = []
        for image in images:
            image = image[0] if isinstance(image, tuple) else image
            image = image.squeeze().permute(1, 2, 0)
            image = image * scale
            image = image.flip(dims=[-1]).to(dtype=torch.uint8, non_blocking=True, copy=False)

            results.append({"images": image,})

        return results


class UVDocResidualBlockWithDilation(nn.Module):
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

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        residual = hidden_state
        if self.conv_down is not None:
            residual = self.conv_down(hidden_state)
        hidden_state = self.conv_start(hidden_state)
        hidden_state = self.conv_final(hidden_state)
        hidden_state += residual

        hidden_state = self.act_fn(hidden_state)
        return hidden_state


class UVDocResNetStage(nn.Module):
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

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class UVDocResNetStraight(nn.Module):
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

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            hidden_state = stage(hidden_state)
        return hidden_state


class UVDocResNetHead(nn.Module):
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

    def forward(self, hidden_state):
        hidden_state = self.conv_down(hidden_state)
        hidden_state = self.conv_up(hidden_state)
        return hidden_state


class UVDocConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        padding_mode: str | None = None,
        bias: bool = False,
        dilation: int | None = None,
        activation: str = "relu",
    ):
        super().__init__()

        if dilation is None and padding_mode is None:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                bias=bias,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        elif padding_mode is not None:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                bias=bias,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode=padding_mode,
            )
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                bias=bias,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act_fn = ACT2FN[activation] if activation is not None else nn.Identity()

    def forward(self, hidden_state):
        hidden_state = self.conv(hidden_state)
        hidden_state = self.norm(hidden_state)
        hidden_state = self.act_fn(hidden_state)
        return hidden_state


class UVDocBridgeBlock(nn.Module):
    def __init__(self, config, dilation_value):
        super().__init__()
        in_channels = config.num_filter * config.feature_map_multipliers[2]

        self.blocks = nn.ModuleList([])
        for dilation in dilation_value:
            self.blocks.append(UVDocConvLayer(in_channels, in_channels, padding=dilation, dilation=dilation))

    def forward(self, hidden_state):
        for block in self.blocks:
            hidden_state = block(hidden_state)
        return hidden_state


class UVDocPointPositions2D(nn.Module):
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

    def forward(self, hidden_state):
        hidden_state = self.conv_down(hidden_state)
        hidden_state = self.conv_up(hidden_state)
        return hidden_state


@auto_docstring
class UVDocPreTrainedModel(PreTrainedModel):
    config: UVDocConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"
    input_modalities = ("image",)
    _can_compile_fullgraph = True

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

        self.upsample_size = config.upsample_size
        self.upsample_mode = config.upsample_mode

        self.resnet_head = UVDocResNetHead(config)
        self.resnet_down = UVDocResNetStraight(config)

        self.bridge = nn.ModuleList([])
        for dilation_value in config.dilation_values:
            self.bridge.append(UVDocBridgeBlock(config, dilation_value))

        self.num_bridge_layers = len(self.bridge)

        self.bridge_concat = UVDocConvLayer(
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
        residual = pixel_values
        original_height, original_width = pixel_values.shape[2:]
        hidden_state = F.interpolate(
            pixel_values,
            size=(self.upsample_size[0], self.upsample_size[1]),
            mode=self.upsample_mode,
            align_corners=True,
        )
        hidden_state = self.resnet_head(hidden_state)
        resnet_down = self.resnet_down(hidden_state)

        bridge_outputs = []
        for bridge_layer in self.bridge:
            bridge_output = bridge_layer(resnet_down)
            bridge_outputs.append(bridge_output)

        bridge_concat = torch.cat(bridge_outputs, dim=1)
        bridge = self.bridge_concat(bridge_concat)

        out_point_positions2D = self.out_point_positions2D(bridge)

        upsampled_2d_bezier_mesh = F.interpolate(
            out_point_positions2D,
            size=(original_height, original_width),
            mode=self.upsample_mode,
            align_corners=True,
        )

        rearranged_bezier_mesh = upsampled_2d_bezier_mesh.permute(0, 2, 3, 1)
        rectified_image_output = F.grid_sample(residual, rearranged_bezier_mesh, align_corners=True)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=rectified_image_output,
        )


@auto_docstring(
    custom_intro=r"""
    The model takes raw document images (pixel values) as input, processes them through the UVDoc backbone to predict spatial transformation parameters,
    and outputs the rectified (corrected) document image tensor.
    """
)
class UVDocForDocumentRectification(UVDocPreTrainedModel):
    _keys_to_ignore_on_load_missing = ["num_batches_tracked"]

    def __init__(self, config: UVDocConfig):
        super().__init__(config)
        self.model = UVDocModel(config)
        self.post_init()

    @capture_outputs
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor] | BaseModelOutputWithNoAttention:
        return self.model(pixel_values)


__all__ = [
    "UVDocForDocumentRectification",
    "UVDocImageProcessorFast",
    "UVDocConfig",
    "UVDocModel",
    "UVDocPreTrainedModel",
]
