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

import torch
import torch.nn as nn
import torchvision.transforms.v2.functional as tvF
from huggingface_hub.dataclasses import strict

from ...backbone_utils import (
    consolidate_backbone_kwargs_to_config,
)
from ...configuration_utils import PreTrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import PILImageResampling, SizeDict
from ...modeling_outputs import BaseModelOutputWithNoAttention
from ...processing_utils import Unpack
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    logging,
)
from ...utils.generic import TensorType
from ..pp_ocrv5_server_rec.configuration_pp_ocrv5_server_rec import PPOCRV5ServerRecConfig
from ..pp_ocrv5_server_rec.image_processing_pp_ocrv5_server_rec import PPOCRV5ServerRecImageProcessor
from ..pp_ocrv5_server_rec.modeling_pp_ocrv5_server_rec import (
    PPOCRV5ServerRecConvLayer,
    PPOCRV5ServerRecEncoderWithSVTR,
    PPOCRV5ServerRecForTextRecognition,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="PaddlePaddle/PP-OCRv6_small_rec_safetensors")
@strict
class PPOCRV6SmallRecConfig(PPOCRV5ServerRecConfig):
    r"""
    head_out_channels (`int`, *optional*, defaults to 18714):
        The number of output channels from the PPOCRV6SmallRecHead, responsible for final classification.
    """

    head_out_channels: int = 18714

    def __post_init__(self, **kwargs):
        if self.conv_kernel_size is None:
            self.conv_kernel_size = [1, 7]
        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_config_type="pp_lcnet_v4",
            **kwargs,
        )
        PreTrainedConfig.__post_init__(**kwargs)


class PPOCRV6SmallRecImageProcessor(PPOCRV5ServerRecImageProcessor):
    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        do_pad: bool | None,
        pad_size: SizeDict | None,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}

        # [Key Change] Use get_target_size to calculate target_size for resizing.
        shape_list = list(grouped_images.keys())
        target_size = self.get_target_size(shape_list)

        for shape, stacked_images in grouped_images.items():
            if do_resize:
                # [Key Change] Use antialias=False to align with cv2.resize
                stacked_images = self.resize(
                    image=stacked_images, size=target_size, resample=resample, antialias=False
                )
            # [Key Change] RGB to BGR conversion
            stacked_images = stacked_images[:, [2, 1, 0], :, :]
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_center_crop:
                stacked_images = self.center_crop(stacked_images, crop_size)
            # Fused rescale and normalize
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images
        processed_images = reorder_images(processed_images_grouped, grouped_images_index)

        if do_pad and target_size.width < pad_size.width:
            processed_images = self.pad(processed_images, pad_size=pad_size, disable_grouping=disable_grouping)

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)


class PPOCRV6SmallRecConvLayer(PPOCRV5ServerRecConvLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] = (3, 3),
        stride: int = 1,
        activation: str = "silu",
        groups: int = 1,
    ):
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2),
            bias=False,
            groups=groups,
        )


class PPOCRV6SmallRecEncoderWithSVTR(PPOCRV5ServerRecEncoderWithSVTR):
    def __init__(
        self,
        config,
    ):
        super().__init__(config)
        in_channels = config.backbone_config.block_configs[-1][-1][2]
        hidden_size = config.hidden_size
        self.conv_block = nn.ModuleList(
            [
                # skip_conv
                PPOCRV6SmallRecConvLayer(
                    in_channels=in_channels, out_channels=hidden_size, kernel_size=(1, 1), activation=config.hidden_act
                ),
                # conv_reduce
                PPOCRV6SmallRecConvLayer(
                    in_channels=in_channels, out_channels=hidden_size, kernel_size=(1, 1), activation=config.hidden_act
                ),
                # local_conv
                PPOCRV6SmallRecConvLayer(
                    in_channels=hidden_size,
                    out_channels=hidden_size,
                    kernel_size=config.conv_kernel_size,
                    activation=config.hidden_act,
                    groups=hidden_size,
                ),
            ]
        )

    def forward(self, hidden_states: torch.FloatTensor, **kwargs: Unpack[TransformersKwargs]):
        # PP-OCRv6_small_rec uses the output of the first conv block as the residual.
        residual = self.conv_block[0](hidden_states)

        hidden_states = self.conv_block[1](hidden_states)
        hidden_states = hidden_states + self.conv_block[2](hidden_states)

        batch_size, channels, height, width = hidden_states.shape
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        for block in self.svtr_block:
            hidden_states = block(hidden_states)

        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.view(batch_size, height, width, channels).permute(0, 3, 1, 2)
        # PP-OCRv6_small_rec uses fewer conv blocks and residual fusion instead of concat fusion.
        hidden_states = hidden_states + residual
        hidden_states = hidden_states.squeeze(2).transpose(1, 2)

        return BaseModelOutputWithNoAttention(last_hidden_state=hidden_states)


@auto_docstring(custom_intro="PPOCR6SmallRec model for text recognition tasks.")
class PPOCRV6SmallRecForTextRecognition(PPOCRV5ServerRecForTextRecognition):
    pass


__all__ = [
    "PPOCRV6SmallRecForTextRecognition",
    "PPOCRV6SmallRecConfig",
    "PPOCRV6SmallRecImageProcessor",
    "PPOCRV6SmallRecModel",  # noqa: F822
    "PPOCRV6SmallRecEncoderWithSVTR",
    "PPOCRV6SmallRecPreTrainedModel",  # noqa: F822
]
