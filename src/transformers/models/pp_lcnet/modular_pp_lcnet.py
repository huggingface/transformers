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

import math

import torch
import torch.nn as nn
import torchvision.transforms.v2.functional as tvF
from huggingface_hub.dataclasses import strict

from ...activations import ACT2FN
from ...backbone_utils import BackboneConfigMixin, BackboneMixin, filter_output_hidden_states
from ...configuration_utils import PreTrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...image_processing_backends import TorchvisionBackend
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import PILImageResampling, SizeDict
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BackboneOutput,
    BaseModelOutputWithNoAttention,
)
from ...modeling_utils import PreTrainedModel
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
)
from ...utils.generic import TensorType, merge_with_config_defaults
from ...utils.import_utils import requires
from ...utils.output_capturing import capture_outputs
from ..mobilenet_v2.modeling_mobilenet_v2 import make_divisible
from ..resnet.modeling_resnet import ResNetConvLayer


@auto_docstring(checkpoint="PaddlePaddle/PP-LCNet_x1_0_doc_ori_safetensors")
@strict
class PPLCNetConfig(BackboneConfigMixin, PreTrainedConfig):
    r"""
    scale (`float`, *optional*, defaults to 1.0):
        The scaling factor for the model's channel dimensions, used to adjust the model size and computational cost
        without changing the overall architecture (e.g., 0.25, 0.5, 1.0, 1.5).
    block_configs (`list[list[tuple]]`, *optional*, defaults to `None`):
        Configuration for each block in each stage. Each tuple contains:
        (kernel_size, in_channels, out_channels, stride, use_squeeze_excitation).
        If `None`, uses the default PP-LCNet configuration.
    stem_channels (`int`, *optional*, defaults to 16):
        The number of output channels for the stem layer.
    stem_stride (`int`, *optional*, defaults to 2):
        The stride for the stem convolution layer.
    reduction (`int`, *optional*, defaults to 4):
        The reduction factor for feature channel dimensions in the squeeze-and-excitation (SE) blocks, used to
        reduce the number of model parameters and computational complexity while maintaining feature representability.
    class_expand (`int`, *optional*, defaults to 1280):
        The number of hidden units in the expansion layer of the classification head, used to enhance the model's
        feature representation capability before the final classification layer.
    divisor (`int`, *optional*, defaults to 8):
        The divisor used to ensure that various model parameters (e.g., channel dimensions, kernel sizes) are
        multiples of this value, promoting efficient model implementation and resource utilization.
    """

    model_type = "pp_lcnet"

    scale: float | int = 1.0
    block_configs: list | None = None
    stem_channels: int = 16
    stem_stride: int = 2
    reduction: int = 4
    class_expand: int = 1280
    divisor: int = 8
    hidden_act: str = "hardswish"
    _out_features: list[str] | None = None
    _out_indices: list[int] | None = None
    hidden_dropout_prob: float = 0.2

    def __post_init__(self, **kwargs):
        # Default block configs for PP-LCNet
        # Each tuple: (kernel_size, in_channels, out_channels, stride, use_squeeze_excitation)
        self.block_configs = (
            [
                # Stage 1 (blocks2)
                [[3, 16, 32, 1, False]],
                # Stage 2 (blocks3)
                [[3, 32, 64, 2, False], [3, 64, 64, 1, False]],
                # Stage 3 (blocks4)
                [[3, 64, 128, 2, False], [3, 128, 128, 1, False]],
                # Stage 4 (blocks5)
                [
                    [3, 128, 256, 2, False],
                    [5, 256, 256, 1, False],
                    [5, 256, 256, 1, False],
                    [5, 256, 256, 1, False],
                    [5, 256, 256, 1, False],
                    [5, 256, 256, 1, False],
                ],
                # Stage 5 (blocks6)
                [[5, 256, 512, 2, True], [5, 512, 512, 1, True]],
            ]
            if self.block_configs is None
            else self.block_configs
        )

        self.depths = [len(blocks) for blocks in self.block_configs]
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(self.block_configs) + 1)]
        self.set_output_features_output_indices(
            out_indices=kwargs.pop("out_indices", None), out_features=kwargs.pop("out_features", None)
        )
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if len(self.block_configs) != 5:
            raise ValueError(f"block_configs must have 5 stages, but got {len(self.block_configs)}")


class PPLCNetImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    resize_short (`int`, *optional*, defaults to 256):
        target_short_edge (Union[int, None]): Desired length for the shorter edge of the image.
    size_divisor (`int`, *optional*, defaults to 1):
        Divisor to align image dimensions.
    """

    resize_short: int
    size_divisor: int


@auto_docstring
@requires(backends=("torch",))
class PPLCNetImageProcessor(TorchvisionBackend):
    resample = 2
    image_mean = [0.406, 0.456, 0.485]
    image_std = [0.225, 0.224, 0.229]
    size = {"height": 256, "width": 256}
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_center_crop = True
    crop_size = 224
    resize_short = 256
    size_divisor = 1
    valid_kwargs = PPLCNetImageProcessorKwargs

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        resize_short: int,
        size_divisor: int,
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        return_tensors: str | TensorType | None,
        disable_grouping: bool | None = False,
        **kwargs,
    ) -> BatchFeature:
        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                # Unlike TorchvisionBackend, which resizes to a fixed target,
                # this implementation first calculates the target size dynamically to preserve
                # the aspect ratio, using the shorter edge as a reference.
                resize_size = size
                if self.resize_short is not None:
                    resize_size = self.get_image_size(
                        stacked_images[0], target_short_edge=resize_short, size_divisor=size_divisor
                    )
                stacked_images = self.resize(stacked_images, size=resize_size, resample=resample)
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group images by size for further processing
        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_center_crop:
                stacked_images = self.center_crop(stacked_images, crop_size)
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images
        processed_images = reorder_images(processed_images_grouped, grouped_images_index)

        # RGB -> BGR
        images = [image[[2, 1, 0], :, :] for image in processed_images]
        return BatchFeature(data={"pixel_values": images}, tensor_type=return_tensors)

    def get_image_size(
        self,
        image: "torch.Tensor",
        target_short_edge: int | None,
        size_divisor: int | None,
    ) -> tuple[SizeDict, torch.Tensor]:
        _, height, width = image.shape
        resize_scale = target_short_edge / min(height, width)
        resized_height = round(height * resize_scale)
        resized_width = round(width * resize_scale)
        if size_divisor is not None:
            resized_height = math.ceil(resized_height / size_divisor) * size_divisor
            resized_width = math.ceil(resized_width / size_divisor) * size_divisor

        return SizeDict(height=resized_height, width=resized_width)


class PPLCNetConvLayer(ResNetConvLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        activation: str = "hardswish",
        groups: int = 1,
    ):
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
            groups=groups,
        )


class PPLCNetDepthwiseSeparableConvLayer(GradientCheckpointingLayer):
    """
    Depthwise Separable Convolution Layer: Depthwise Conv -> SE Module (optional) -> Pointwise Conv
    Core component of lightweight models (e.g., MobileNet, PP-LCNet) that significantly reduces
    the number of parameters and computational cost.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        kernel_size,
        use_squeeze_excitation,
        config,
    ):
        super().__init__()
        self.depthwise_convolution = PPLCNetConvLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
            activation=config.hidden_act,
        )
        self.squeeze_excitation_module = (
            PPLCNetSqueezeExcitationModule(in_channels, config.reduction) if use_squeeze_excitation else nn.Identity()
        )
        self.pointwise_convolution = PPLCNetConvLayer(
            in_channels=in_channels,
            kernel_size=1,
            out_channels=out_channels,
            stride=1,
            activation=config.hidden_act,
        )

    def forward(self, hidden_state):
        hidden_state = self.depthwise_convolution(hidden_state)
        hidden_state = self.squeeze_excitation_module(hidden_state)
        hidden_state = self.pointwise_convolution(hidden_state)

        return hidden_state


class PPLCNetSqueezeExcitationModule(nn.Module):
    """
    Squeeze-and-Excitation (SE) Module: Adaptive feature recalibration
    Enhances the model's ability to focus on important channels by learning channel-wise attention weights.
    """

    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.convolutions = nn.ModuleList()
        for in_channels, out_channels, activation in [
            [channel, channel // reduction, nn.ReLU()],
            [channel // reduction, channel, nn.Hardsigmoid()],
        ]:
            self.convolutions.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                )
            )
            self.convolutions.append(activation)

    def forward(self, hidden_state):
        residual = hidden_state
        hidden_state = self.avg_pool(hidden_state)
        for layer in self.convolutions:
            hidden_state = layer(hidden_state)
        hidden_state = residual * hidden_state

        return hidden_state


class PPLCNetBlock(nn.Module):
    def __init__(self, config, stage_index):
        super().__init__()
        self.config = config

        blocks = config.block_configs[stage_index]

        self.layers = nn.ModuleList()
        for kernel_size, in_channels, out_channels, stride, use_squeeze_excitation in blocks:
            scaled_in_channels = make_divisible(in_channels * config.scale, config.divisor)
            scaled_out_channels = make_divisible(out_channels * config.scale, config.divisor)

            depthwise_block = PPLCNetDepthwiseSeparableConvLayer(
                in_channels=scaled_in_channels,
                out_channels=scaled_out_channels,
                kernel_size=kernel_size,
                stride=stride,
                use_squeeze_excitation=use_squeeze_excitation,
                config=config,
            )
            self.layers.append(depthwise_block)

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


@auto_docstring
class PPLCNetPreTrainedModel(PreTrainedModel):
    """
    An abstract base class for PP-LCNet models that inherits from Hugging Face PreTrainedModel.
    Provides common functionality for weight initialization and loading.
    """

    config: PPLCNetConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"
    input_modalities = ("image",)
    _can_compile_fullgraph = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["PPLCNetDepthwiseSeparableConvLayer"]
    _can_record_outputs = {
        "hidden_states": PPLCNetBlock,
    }


class PPLCNetEncoder(PPLCNetPreTrainedModel):
    def __init__(self, config: PPLCNetConfig):
        super().__init__(config)
        self.config = config

        # stem
        self.convolution = PPLCNetConvLayer(
            in_channels=3,
            kernel_size=3,
            out_channels=make_divisible(config.stem_channels * config.scale, config.divisor),
            stride=config.stem_stride,
            activation=config.hidden_act,
        )
        # stages
        self.blocks = nn.ModuleList([])
        for stage_index in range(len(config.block_configs)):
            block = PPLCNetBlock(config, stage_index)
            self.blocks.append(block)

        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(self, pixel_values: torch.Tensor, **kwargs) -> BaseModelOutputWithNoAttention:
        hidden_state = self.convolution(pixel_values)
        for block in self.blocks:
            hidden_state = block(hidden_state)

        return BaseModelOutputWithNoAttention(last_hidden_state=hidden_state)


@auto_docstring(
    custom_intro="""
    PPLCNet backbone model for feature extraction.
    """
)
class PPLCNetBackbone(BackboneMixin, PPLCNetPreTrainedModel):
    has_attentions = False

    def __init__(self, config: PPLCNetConfig):
        super().__init__(config)
        num_features = [config.stem_channels]
        for block in config.block_configs:
            num_features.append(int(block[-1][2] * config.scale))
        self.num_features = num_features
        self.encoder = PPLCNetEncoder(config)

        self.post_init()

    @can_return_tuple
    @filter_output_hidden_states
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BackboneOutput:
        r"""
        Examples:

        ```python
        >>> from transformers import PPLCNetConfig, PPLCNetBackbone
        >>> import torch

        >>> config = PPLCNetConfig()
        >>> model = PPLCNetBackbone(config)

        >>> pixel_values = torch.randn(1, 3, 224, 224)

        >>> with torch.no_grad():
        ...     outputs = model(pixel_values)

        >>> feature_maps = outputs.feature_maps
        >>> list(feature_maps[-1].shape)
        ```"""
        kwargs["output_hidden_states"] = True  # required to extract layers for the stages
        hidden_states = self.encoder(pixel_values, **kwargs).hidden_states

        feature_maps = ()
        for idx, stage in enumerate(self.stage_names):
            if stage in self.out_features:
                feature_maps += (hidden_states[idx],)

        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=hidden_states,
        )


@auto_docstring
class PPLCNetForImageClassification(PPLCNetPreTrainedModel):
    _keys_to_ignore_on_load_missing = ["num_batches_tracked"]

    def __init__(self, config: PPLCNetConfig):
        super().__init__(config)
        self.encoder = PPLCNetEncoder(config)

        self.config = config
        self.num_labels = config.num_labels
        last_block_out_channels = config.block_configs[-1][-1][2]
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.last_convolution = nn.Conv2d(
            in_channels=make_divisible(last_block_out_channels * config.scale, config.divisor),
            out_channels=config.class_expand,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.act_fn = ACT2FN[config.hidden_act]
        self.hidden_dropout_prob = config.hidden_dropout_prob

        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.head = nn.Linear(config.class_expand, config.num_labels) if config.num_labels > 0 else nn.Identity()

        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithNoAttention:
        r"""
        Examples:

        ```python
        >>> import requests
        >>> from PIL import Image
        >>> from transformers import AutoModelForImageClassification, AutoImageProcessor

        >>> model_path = "PaddlePaddle/PP-LCNet_x1_0_table_cls_safetensors"
        >>> model = AutoModelForImageClassification.from_pretrained(model_path)
        >>> image_processor = AutoImageProcessor.from_pretrained(model_path)

        >>> url = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/img_rot180_demo.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> predicted_label = outputs.last_hidden_state.argmax(-1).item()
        >>> print(model.config.id2label[predicted_label])
        wireless_table
        ```"""
        outputs = self.encoder(pixel_values, **kwargs)

        last_hidden_state = self.avg_pool(outputs.last_hidden_state)

        last_hidden_state = self.last_convolution(last_hidden_state)
        last_hidden_state = self.act_fn(last_hidden_state)
        last_hidden_state = last_hidden_state * (1 - self.hidden_dropout_prob)

        last_hidden_state = self.flatten(last_hidden_state)
        last_hidden_state = self.head(last_hidden_state)

        return BaseModelOutputWithNoAttention(last_hidden_state=last_hidden_state, hidden_states=outputs.hidden_states)


__all__ = [
    "PPLCNetBackbone",
    "PPLCNetForImageClassification",
    "PPLCNetImageProcessor",
    "PPLCNetConfig",
    "PPLCNetPreTrainedModel",
]
