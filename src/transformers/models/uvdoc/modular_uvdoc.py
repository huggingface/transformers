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

from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...activations import ACT2FN
from ...configuration_utils import PreTrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...image_processing_utils_fast import BaseImageProcessorFast
from ...modeling_outputs import BaseModelOutputWithNoAttention
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring, can_return_tuple

from .image_utils import SizeDict

from ...utils.output_capturing import capture_outputs
from ...utils.generic import TensorType

from .image_transforms import group_images_by_shape, reorder_images


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
    map_num (`List[int]`, *optional*, defaults to `[1, 2, 4, 8, 16]`):
        The scaling factors for feature map dimensions in multi-scale feature fusion modules, used to align
        feature maps of different resolutions for document structure restoration.
    block_nums (`List[int]`, *optional*, defaults to `[3, 4, 6, 3]`):
        The number of residual blocks in each stage of the model backbone, determining the depth of the network.
        More blocks enhance feature extraction capability but increase inference time.
    dilation_values (`Dict[str, Union[int, List[int]]]`, *optional*, defaults to `None`):
        A dictionary of dilation rates for dilated convolutional layers in bridge modules (e.g., "bridge_1": 1,
        "bridge_4": [8, 3, 2]). Dilated convolution expands the receptive field without increasing kernel size,
        critical for capturing long-range geometric dependencies in distorted documents. If `None`, default values
        will be used:{
            "bridge_1": 1,
            "bridge_2": 2,
            "bridge_3": 5,
            "bridge_4": [8, 3, 2],
            "bridge_5": [12, 7, 4],
            "bridge_6": [18, 12, 6]
        }
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
        stride: list | None = None,
        map_num: list | None = None,
        block_nums: list | None = None,
        dilation_values: dict | None = None,
        padding_mode: str = "reflect",
        upsample_size: list | None = None,
        upsample_mode: str = "bilinear",
        **kwargs,
    ):
        self.num_filter = num_filter
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.map_num = map_num
        self.block_nums = block_nums
        self.dilation_values = dilation_values
        self.padding_mode = padding_mode
        self.upsample_size = upsample_size
        self.upsample_mode = upsample_mode

        super().__init__(**kwargs)


@auto_docstring
class UVDocImageProcessorFast(BaseImageProcessorFast):

    image_mean = [0, 0, 0]
    image_std = [1, 1, 1]
    do_rescale = True
    do_normalize = True

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

        post_process_images = []
        for image in images:
            image = image[0] if isinstance(image, tuple) else image
            image = image.squeeze().permute(1, 2, 0)
            image = image * scale
            image = image.flip(dims=[-1]).to(dtype=torch.uint8, non_blocking=True, copy=False)

            post_process_images.append(image)
        
        return post_process_images

    def doctr(self, pred: Union[torch.Tensor, tuple[torch.Tensor, ...]], scale: torch.Tensor) -> torch.Tensor:
        
        

        return image


class UVDocResidualBlockWithDilation(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        downsample: Optional[bool] = None,
        is_top: bool = False,
    ):
        
        super().__init__()

        self.downsample = downsample
        if downsample:
            self.downsample_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            )
            self.downsample_norm = nn.BatchNorm2d(out_channels)

        if stride != 1 or is_top:
            stride1, padding, dilation = stride, kernel_size // 2, 1
        else:
            stride1, padding, dilation = 1, 3 * (kernel_size // 2), 3

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride1, padding, dilation=dilation)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding, dilation=dilation)

        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.norm2 = nn.BatchNorm2d(out_channels)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        identity = hidden_state
        if self.downsample:
            identity = self.downsample_conv(hidden_state)
            identity = self.downsample_norm(identity)

        hidden_state = self.conv1(hidden_state)
        hidden_state = self.norm1(hidden_state)
        hidden_state = self.relu(hidden_state)
        hidden_state = self.conv2(hidden_state)
        hidden_state = self.norm2(hidden_state)
        hidden_state += identity
        hidden_state = self.relu(hidden_state)
        return hidden_state


class UVDocResNetStraight(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.in_channels = config.num_filter * config.map_num[0]
        self.relu = nn.ReLU()

        self.layers = nn.ModuleList([])
        for map_num, block_num, stride in zip(config.map_num[:3], config.block_nums[:3], config.stride[:3]):
            layers = nn.ModuleList([])
            out_channels = config.num_filter * map_num

            downsample = None
            if stride != 1 or self.in_channels != out_channels:
                downsample = True

            for i in range(block_num):
                layer = UVDocResidualBlockWithDilation(
                    in_channels=self.in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=config.kernel_size,
                    stride=stride if i == 0 else 1,
                    downsample=downsample if i == 0 else None,
                    is_top=i == 0,
                )
                layers.append(layer)
            self.layers.append(layers)
            self.in_channels = out_channels

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        for layers in self.layers:
            for layer in layers:
                hidden_state = layer(hidden_state)

        return hidden_state


class UVDocResNetHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_channels = config.in_channels
        num_filter = config.num_filter
        map_num_0 = config.map_num[0]
        kernel_size = config.kernel_size
        out_channels = num_filter * map_num_0

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=False,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
        )
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            bias=False,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
        )
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, hidden_state):
        hidden_state = self.conv1(hidden_state)
        hidden_state = self.norm1(hidden_state)
        hidden_state = self.relu1(hidden_state)

        hidden_state = self.conv2(hidden_state)
        hidden_state = self.norm2(hidden_state)
        hidden_state = self.relu2(hidden_state)
        return hidden_state


@auto_docstring(
    custom_intro="""
    """
)
class UVDocPreTrainedModel(PreTrainedModel):

    config: UVDocConfig
    base_model_prefix = "uvdoc"
    main_input_name = "pixel_values"
    input_modalities = ("image",)


class UVDocConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 3,
        activation: str = "relu",
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            bias=False,
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
    def __init__(self, config, block_index):
        super().__init__()
        dilation_values = config.dilation_values[block_index]
        in_channels = config.num_filter * config.map_num[2]

        self.blocks = nn.ModuleList([])

        if isinstance(dilation_values, int):
            self.blocks.append(
                UVDocConvLayer(in_channels, in_channels, padding=dilation_values, dilation=dilation_values)
            )
        else:
            for dilation in dilation_values:
                self.blocks.append(UVDocConvLayer(in_channels, in_channels, padding=dilation, dilation=dilation))

    def forward(self, hidden_state):
        for block in self.blocks:
            hidden_state = block(hidden_state)
        return hidden_state


class UVDocPointPositions2D(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.conv1 = nn.Conv2d(
            config.num_filter * config.map_num[2],
            config.num_filter * config.map_num[0],
            bias=False,
            kernel_size=config.kernel_size,
            stride=1,
            padding=config.kernel_size // 2,
            padding_mode=config.padding_mode,
        )
        self.norm1 = nn.BatchNorm2d(config.num_filter * config.map_num[0])
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(
            config.num_filter * config.map_num[0],
            2,
            kernel_size=config.kernel_size,
            stride=1,
            padding=config.kernel_size // 2,
            padding_mode=config.padding_mode,
        )

    def forward(self, hidden_state):
        hidden_state = self.conv1(hidden_state)
        hidden_state = self.norm1(hidden_state)
        hidden_state = self.prelu(hidden_state)
        hidden_state = self.conv2(hidden_state)
        return hidden_state


@auto_docstring
class UVDocModel(UVDocPreTrainedModel):

    def __init__(self, config: UVDocConfig) -> None:
        super().__init__(config)

        self.upsample_size = config.upsample_size
        self.upsample_mode = config.upsample_mode

        self.resnet_head = UVDocResNetHead(config)
        self.resnet_down = UVDocResNetStraight(config)

        self.bridge = nn.ModuleList([])
        for block_index in config.dilation_values.keys():
            self.bridge.append(UVDocBridgeBlock(config, block_index))

        self.num_bridge_layers = len(self.bridge)

        self.bridge_concat = UVDocConvLayer(
            in_channels=config.num_filter * config.map_num[2] * self.num_bridge_layers,
            out_channels=config.num_filter * config.map_num[2],
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
        )

        self.out_point_positions2D = UVDocPointPositions2D(config)

        self.post_init()

    @capture_outputs
    @can_return_tuple
    def forward(
        self,
        hidden_state: torch.FloatTensor,
        **kwargs: Any,
    ) -> Union[tuple[torch.FloatTensor, ...], BaseModelOutputWithNoAttention]:
        
        identity = hidden_state
        original_height, original_width = hidden_state.shape[2:]
        hidden_state = F.interpolate(
            hidden_state,
            size=(self.upsample_size[0], self.upsample_size[1]),
            mode=self.upsample_mode,
            align_corners=True,
        )
        hidden_state = self.resnet_head(hidden_state)
        resnet_down = self.resnet_down(hidden_state)

        bridge_outputs = []
        for bridge_layer in self.bridge:
            bridge_out = bridge_layer(resnet_down)
            bridge_outputs.append(bridge_out)

        last_hidden_state = bridge_outputs[-1] if bridge_outputs else None

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
        rectified_image_output = F.grid_sample(identity, rearranged_bezier_mesh, align_corners=True)


        return BaseModelOutputWithNoAttention(
            last_hidden_state=rectified_image_output,
        )


@dataclass
class UVDocForDocumentRectificationOutput(BaseModelOutputWithNoAttention):
    logits: Optional[torch.FloatTensor] = None


@auto_docstring(
    custom_intro=r"""
    """
)
class UVDocForDocumentRectification(UVDocPreTrainedModel):
    _keys_to_ignore_on_load_missing = ["num_batches_tracked"]

    def __init__(self, config: UVDocConfig) -> None:
        super().__init__(config)
        self.model = UVDocModel(config)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        **kwargs: Any,
    ) -> Union[tuple[torch.FloatTensor, ...], BaseModelOutputWithNoAttention]:

        outputs = self.model(pixel_values)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=outputs.last_hidden_state,
        )


__all__ = [
    "UVDocForDocumentRectification",
    "UVDocImageProcessorFast",
    "UVDocConfig",
    "UVDocModel",
    "UVDocPreTrainedModel",
]
