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


from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ...activations import ACT2FN
from ...backbone_utils import BackboneConfigMixin
from ...configuration_utils import PreTrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...image_processing_backends import TorchvisionBackend
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import PILImageResampling, SizeDict
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithNoAttention
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import TensorType
from ...utils.import_utils import requires
from ...utils.output_capturing import capture_outputs
from ..pp_ocrv5_server_det.modeling_pp_ocrv5_server_det import PPOCRV5ServerDetPreTrainedModel
from ..pp_lcnet.modeling_pp_lcnet import PPLCNetConvLayer

@auto_docstring(checkpoint="PaddlePaddle/UVDoc_safetensors")
@strict(accept_kwargs=True)
class UVDocConfig(BackboneConfigMixin, PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`UVDocModel`]. It is used to instantiate
    a UVDoc model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the UVDoc
    [PaddlePaddle/UVDoc_safetensors](https://huggingface.co/PaddlePaddle/UVDoc_safetensors) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        kernel_size (`int`, *optional*, defaults to 5):
            Kernel size for convolutional layers in the backbone network.
        stage_layer_num (`list[int]` or `tuple[int, ...]`, *optional*, defaults to `(3, 4, 6)`):
            Number of layers in each ResNet stage.
        resnet_head (`Sequence[list[int] | tuple[int, ...]]`, *optional*, defaults to `((3, 32), (32, 32))`):
            Configuration for the ResNet head layers in format [in_channels, out_channels].
        resnet_down (`Sequence[list[int] | tuple[int, ...]]`, *optional*, defaults to `((32, 32), (32, 64), (64, 128))`):
            Configuration for the ResNet downsampling stages in format [in_channels, out_channels].
        stage_configs (`Sequence[Sequence[tuple[int, int, int, bool] | list[int | bool]]]`, *optional*, defaults to `(((32, 32, 1, False), 
            (32, 32, 3, False), (32, 32, 3, False)), ((32, 64, 1, True), (64, 64, 3, False), (64, 64, 3, False), (64, 64, 3, False)), ((64, 128, 1, True), (128, 128, 3, False), (128, 128, 3, False), 
            (128, 128, 3, False), (128, 128, 3, False), (128, 128, 3, False)))`):
            Configuration for the ResNet stages in format [in_channels, out_channels, dilation_value, downsample].
        bridge_connector (`list[int] | tuple[int, ...]`, *optional*, defaults to `(128, 128)`):
            Configuration for the bridge connector in format [in_channels, out_channels].
        out_point_positions2D (`Sequence[list[int] | tuple[int, ...]]`, *optional*, defaults to `((128, 32), (32, 2))`):
            Configuration for the output point positions 2D layer in format [in_channels, out_channels].
        dilation_values (`list[list[int]] | tuple[tuple[int, ...], ...]`, *optional*, defaults to `((1,), (2,), (5,), (8, 3, 2), (12, 7, 4), (18, 12, 6))`):
            Dilation rates for dilated convolutional layers in bridge modules. Each inner tuple/list contains dilation
            rates for a single bridge block.
        padding_mode (`str`, *optional*, defaults to `"reflect"`):
            Padding mode for convolutional layers. Supported modes are `"reflect"`, `"constant"`, and `"replicate"`.
        hidden_act (`str`, *optional*, defaults to `"prelu"`):
            Activation function for hidden layers.
    """

    model_type = "uvdoc"

    _out_features: list[str] | None = None
    _out_indices: list[int] | None = None

    hidden_act: str = "prelu"
    bridge_in_channels = 128
    kernel_size: int = 5
    stage_layer_num: list[int] | tuple[int, ...] = (3, 4, 6)
    resnet_head: Sequence[list[int] | tuple[int, ...]] = (
        (3, 32),
        (32, 32),
    )

    stage_configs: Sequence[Sequence[tuple[int, int, int, bool] | list[int | bool]]] = (
        (
            (32, 32, 1, False),
            (32, 32, 3, False),
            (32, 32, 3, False),
        ),
        (
            (32, 64, 1, True),
            (64, 64, 3, False),
            (64, 64, 3, False),
            (64, 64, 3, False),
        ),
        (
            (64, 128, 1, True),
            (128, 128, 3, False),
            (128, 128, 3, False),
            (128, 128, 3, False),
            (128, 128, 3, False),
            (128, 128, 3, False),
        ),
    )

    bridge_connector: list[int] | tuple[int, ...] = (128, 128)
    out_point_positions2D: Sequence[list[int] | tuple[int, ...]] = ((128, 32), (32, 2))
    dilation_values: list[list[int]] | tuple[tuple[int, ...], ...] = (
        (1,),
        (2,),
        (5,),
        (8, 3, 2),
        (12, 7, 4),
        (18, 12, 6),
    )
    padding_mode: str = "reflect"

    def __post_init__(self, **kwargs):

        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(self.stage_configs) + 1)]
        self.set_output_features_output_indices(
            out_indices=kwargs.pop("out_indices", None), out_features=kwargs.pop("out_features", None)
        )
        super().__post_init__(**kwargs)


@auto_docstring
@requires(backends=("torch",))
class UVDocImageProcessor(TorchvisionBackend):
    do_rescale = True
    do_resize = True
    size = {"height": 712, "width": 488}
    resample = PILImageResampling.BILINEAR

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
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
                    stacked_images, size=(size.height, size.width), mode="bilinear", align_corners=True
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


class UVDocResidualBlock(nn.Module):
    """Base residual block with dilation support."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        downsample: bool = False,
        activation: str = "relu",
    ):
        super().__init__()

        self.conv_down = (
            UVDocConvLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=True,
                activation=None,
            )
            if downsample
            else nn.Identity()
        )

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
        residual = self.conv_down(hidden_states)
        hidden_states = self.conv_start(hidden_states)
        hidden_states = self.conv_final(hidden_states)
        hidden_states = hidden_states + residual
        hidden_states = self.act_fn(hidden_states)
        return hidden_states


class UVDocResNetStage(nn.Module):
    """A ResNet stage containing multiple residual blocks."""

    def __init__(self, config, stage_index):
        super().__init__()

        stages = config.stage_configs[stage_index]
        self.layers = nn.ModuleList([])
        for in_channels, out_channels, dilation, downsample in stages:
            layer = UVDocResidualBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=2 if downsample else 1,
                padding=dilation * 2,
                dilation=dilation,
                downsample=downsample,
                kernel_size=config.kernel_size,
            )
            self.layers.append(layer)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class UVDocResNet(nn.Module):
    """Initial resnet_head and resnet_down."""

    def __init__(self, config):
        super().__init__()
        self.resnet_head = nn.ModuleList([])
        for i in range(len(config.resnet_head)):
            self.resnet_head.append(
                UVDocConvLayer(
                    in_channels=config.resnet_head[i][0],
                    out_channels=config.resnet_head[i][1],
                    kernel_size=config.kernel_size,
                    stride=2,
                    padding=config.kernel_size // 2,
                )
            )

        self.resnet_down = nn.ModuleList([])
        for stage_index in range(len(config.stage_configs)):
            stage = UVDocResNetStage(config, stage_index)
            self.resnet_down.append(stage)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for head in self.resnet_head:
            hidden_states = head(hidden_states)
        for stage in self.resnet_down:
            hidden_states = stage(hidden_states)
        return hidden_states


class UVDocBridgeBlock(GradientCheckpointingLayer):
    """Bridge module with dilated convolutions for long-range dependencies."""

    def __init__(self, in_channels, dilation_values):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for dilation in dilation_values:
            self.blocks.append(UVDocConvLayer(in_channels, in_channels, padding=dilation, dilation=dilation))

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return hidden_states


class UVDocPointPositions2D(nn.Module):
    """Module for predicting 2D point positions for document rectification."""

    def __init__(self, config):
        super().__init__()

        self.conv_down = UVDocConvLayer(
            in_channels=config.out_point_positions2D[0][0],
            out_channels=config.out_point_positions2D[0][1],
            kernel_size=config.kernel_size,
            stride=1,
            padding=config.kernel_size // 2,
            padding_mode=config.padding_mode,
            activation=config.hidden_act,
        )

        self.conv_up = nn.Conv2d(
            in_channels=config.out_point_positions2D[1][0],
            out_channels=config.out_point_positions2D[1][1],
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
class UVDocPreTrainedModel(PPOCRV5ServerDetPreTrainedModel):

    supports_gradient_checkpointing = True
    _can_record_outputs = {
        "hidden_states": UVDocBridgeBlock,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(module)
        """Initialize the weights."""
        if isinstance(module, nn.PReLU):
            module.reset_parameters()


class UVDocBridge(UVDocPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bridge = nn.ModuleList([])
        for dilation_value in config.dilation_values:
            self.bridge.append(UVDocBridgeBlock(config.bridge_in_channels, dilation_value))
        self.post_init()

    @capture_outputs
    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        outputs = []
        for layer in self.bridge:
            output = layer(hidden_states)
            outputs.append(output)
        hidden_states = torch.cat(outputs, dim=1)

        return BaseModelOutputWithNoAttention(last_hidden_state=hidden_states)


@auto_docstring(
    custom_intro="""
    UVDoc backbone model for feature extraction.
    """
)
class UVDocBackbone(BackboneMixin, UVDocPreTrainedModel):
    has_attentions = False

    def __init__(self, config: UVDocConfig):
        super().__init__(config)

        self.resnet = UVDocResNet(config)
        self.bridge = UVDocBridge(config)

        self.post_init()

    @can_return_tuple
    @filter_output_hidden_states
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        **kwargs,
    ) -> BaseModelOutputWithNoAttention:
        hidden_states = self.resnet(pixel_values)
        outputs = self.bridge(hidden_states, **kwargs)
        return BaseModelOutputWithNoAttention(
            hidden_states=outputs.hidden_states,
            last_hidden_state=outputs.last_hidden_state,
        )


class UVDocHead(UVDocPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_bridge_layers = len(config.dilation_values)

        self.bridge_connector = UVDocConvLayer(
            in_channels=config.bridge_connector[0] * self.num_bridge_layers,
            out_channels=config.bridge_connector[1],
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
        )

        self.out_point_positions2D = UVDocPointPositions2D(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.torch.Tensor:
        hidden_states = self.bridge_connector(hidden_states)
        hidden_states = self.out_point_positions2D(hidden_states)
        return hidden_states


@auto_docstring(
    custom_intro=r"""
    The model takes raw document images (pixel values) as input, processes them through the UVDoc backbone to predict spatial transformation parameters,
    and outputs the rectified (corrected) document image tensor.
    """
)
class UVDocModel(UVDocPreTrainedModel):
    def __init__(self, config: UVDocConfig):
        super().__init__(config)

        self.backbone = UVDocBackbone(config)
        self.head = UVDocHead(config)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor] | BaseModelOutputWithNoAttention:
        outputs = self.backbone(pixel_values, **kwargs)
        head_outputs = self.head(outputs.last_hidden_state, **kwargs)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=head_outputs,
            hidden_states=outputs.hidden_states,
        )



__all__ = [
    "UVDocImageProcessor",
    "UVDocConfig",
    "UVDocModel",
    "UVDocPreTrainedModel",
]
