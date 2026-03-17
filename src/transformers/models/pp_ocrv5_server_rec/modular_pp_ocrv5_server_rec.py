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
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2.functional as tvF

from ...backbone_utils import consolidate_backbone_kwargs_to_config, load_backbone
from ...configuration_utils import PreTrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...image_processing_utils_fast import BaseImageProcessorFast, group_images_by_shape, reorder_images
from ...image_utils import (
    PILImageResampling,
    SizeDict,
)
from ...modeling_outputs import BaseModelOutputWithNoAttention
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    logging,
)
from ...utils.constants import (  # noqa: F401
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
)
from ...utils.generic import TensorType
from ..auto import AutoConfig
from ..blip_2.modeling_blip_2 import Blip2Attention
from ..focalnet.modeling_focalnet import FocalNetMlp
from ..pp_ocrv5_server_det.modeling_pp_ocrv5_server_det import PPOCRV5ServerDetPreTrainedModel
from ..resnet.modeling_resnet import ResNetConvLayer


logger = logging.get_logger(__name__)


@auto_docstring(
    checkpoint="PaddlePaddle/PP-OCRv5_server_rec_safetensors",
    custom_args=r"""
    head_out_channels (`int`, *optional*, defaults to 18385):
        The number of output channels from the PPOCRV5ServerRecHead, responsible for final classification.
    """,
)
class PPOCRV5ServerRecConfig(PreTrainedConfig):
    model_type = "pp_ocrv5_server_rec"
    sub_configs = {"backbone_config": AutoConfig}

    def __init__(
        self,
        backbone_config=None,
        hidden_act: str = "silu",
        hidden_size: int = 120,
        mlp_ratio: float = 2.0,
        depth: int = 2,
        head_out_channels: int = 18385,
        conv_kernel_size: list = [1, 3],
        qkv_bias: bool = True,
        num_attention_heads: int = 8,
        attention_dropout: float = 0.0,
        **kwargs,
    ):
        backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=backbone_config,
            default_config_type="hgnet_v2",
            default_config_kwargs={
                "arch": "L",
                "return_idx": [0, 1, 2, 3],
                "freeze_stem_only": True,
                "freeze_at": 0,
                "freeze_norm": True,
                "lr_mult_list": [1.0, 1.0, 1.0, 1.0, 1.0],
                "out_features": ["stage1", "stage2", "stage3", "stage4"],
                "stage_downsample": [True, True, True, True],
                "stage_downsample_strides": [[2, 1], [1, 2], [2, 1], [2, 1]],
            },
            **kwargs,
        )
        self.backbone_config = backbone_config

        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.mlp_ratio = mlp_ratio
        self.depth = depth
        self.head_out_channels = head_out_channels
        self.conv_kernel_size = conv_kernel_size
        self.qkv_bias = qkv_bias
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout

        super().__init__(**kwargs)


class PPOCRV5ServerRecImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    max_image_width (`int`, *optional*, defaults to `3200`):
        Maximum image width used during resizing.
    character_list (`list`, *optional*, defaults to `[]`):
        Vocabulary list used for text recognition.
    """

    max_image_width: int
    character_list: str


@auto_docstring
class PPOCRV5ServerRecImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 48, "width": 320}
    pad_size = {"height": 48, "width": 320}
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_pad = True
    max_image_width = 3200
    character_list = []
    valid_kwargs = PPOCRV5ServerRecImageProcessorKwargs

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["tvF.InterpolationMode"],
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
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                # [Key Change] Use get_target_size to calculate target_size for resizing.
                target_size = self.get_target_size(stacked_images)
                stacked_images = self.resize(
                    image=stacked_images.float(), size=target_size, interpolation=interpolation
                )
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

    def get_target_size(self, images):
        max_width = -1
        max_height = -1
        for image in images:
            _, height, width = image.shape
            if width > max_width:
                max_width = width
                max_height = height

        default_height, default_width = self.size["height"], self.size["width"]
        ratio = max(max_width / max_height, default_width / default_height)

        target_width = int(default_height * ratio)
        target_height = default_height

        if target_width > self.max_image_width:
            target_width = self.max_image_width
        else:
            ratio = max_width / float(max_height)
            if target_width >= math.ceil(default_height * ratio):
                target_width = int(math.ceil(default_height * ratio))

        return SizeDict(height=target_height, width=target_width)

    def post_process_text_recognition(
        self,
        outputs,
    ) -> tuple[list[str], list[float]]:
        logits = outputs.last_hidden_state
        batch_size = logits.shape[0]

        preds_prob, preds_idx = logits.max(dim=-1)
        results = []
        for idx in range(batch_size):
            selection = torch.ones(len(preds_idx[idx]), dtype=torch.bool, device=preds_idx.device)

            # remove_duplicate
            selection[1:] = preds_idx[idx][1:] != preds_idx[idx][:-1]
            # ignore blank token
            selection &= preds_idx[idx] != 0

            char_list = []
            for text_id in preds_idx[idx][selection]:
                char_list.append(self.character_list[text_id])

            results.append(
                {
                    "text": "".join(char_list),
                    "score": preds_prob[idx][selection].mean().item(),
                }
            )

        return results


class PPOCRV5ServerRecBlock(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        hidden_size = config.hidden_size
        mlp_ratio = config.mlp_ratio

        self.mixer = PPOCRV5ServerRecAttention(config)
        self.mlp = PPOCRV5ServerRecMlp(
            config=config,
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
        )
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-5)

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.mixer(self.norm1(hidden_states))[0]
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))

        return hidden_states


class PPOCRV5ServerRecAttention(Blip2Attention):
    pass


class PPOCRV5ServerRecConvLayer(ResNetConvLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] = (3, 3),
        stride: int = 1,
        activation: str = "silu",
    ):
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2),
            bias=False,
        )


class PPOCRV5ServerRecHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        in_channels = config.backbone_config.stage_out_channels[-1]
        self.ctc_encoder = PPOCRV5ServerRecEncoderWithSVTR(in_channels, config)
        self.ctc_head = nn.Linear(config.hidden_size, config.head_out_channels)

    def forward(self, hidden_states):
        hidden_states = self.ctc_encoder(hidden_states)
        hidden_states = hidden_states.squeeze(2).permute(0, 2, 1)
        hidden_states = self.ctc_head(hidden_states)
        hidden_states = F.softmax(hidden_states, dim=2)

        return hidden_states


class PPOCRV5ServerRecMlp(FocalNetMlp):
    pass


class PPOCRV5ServerRecEncoderWithSVTR(nn.Module):
    def __init__(
        self,
        in_channels,
        config,
    ):
        super().__init__()
        hidden_size = config.hidden_size

        self.conv1 = PPOCRV5ServerRecConvLayer(
            in_channels=in_channels,
            out_channels=in_channels // 8,
            kernel_size=config.conv_kernel_size,
        )
        self.conv2 = PPOCRV5ServerRecConvLayer(
            in_channels=in_channels // 8, out_channels=hidden_size, kernel_size=(1, 1)
        )

        self.svtr_block = nn.ModuleList()
        for _ in range(config.depth):
            self.svtr_block.append(PPOCRV5ServerRecBlock(config=config))

        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.conv3 = PPOCRV5ServerRecConvLayer(in_channels=hidden_size, out_channels=in_channels, kernel_size=(1, 1))
        self.conv4 = PPOCRV5ServerRecConvLayer(
            in_channels=2 * in_channels,
            out_channels=in_channels // 8,
            kernel_size=config.conv_kernel_size,
        )

        self.conv5 = PPOCRV5ServerRecConvLayer(
            in_channels=in_channels // 8, out_channels=hidden_size, kernel_size=(1, 1)
        )

    def forward(self, hidden_states):
        residual = hidden_states

        hidden_states = self.conv1(hidden_states)
        hidden_states = self.conv2(hidden_states)

        batch_size, channels, height, width = hidden_states.shape
        hidden_states = hidden_states.flatten(2).permute(0, 2, 1)
        for blk in self.svtr_block:
            hidden_states = blk(hidden_states)

        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.view(batch_size, height, width, channels).permute(0, 3, 1, 2)
        hidden_states = self.conv3(hidden_states)
        hidden_states = self.conv4(torch.cat((residual, hidden_states), dim=1))
        hidden_states = self.conv5(hidden_states)

        return hidden_states


class PPOCRV5ServerRecPreTrainedModel(PPOCRV5ServerDetPreTrainedModel):
    pass


@auto_docstring(custom_intro="PPOCRV5ServerRec model, consisting of Backbone and Head networks.")
class PPOCRV5ServerRecModel(PPOCRV5ServerRecPreTrainedModel):
    def __init__(self, config: PPOCRV5ServerRecConfig):
        super().__init__(config)
        self.backbone = load_backbone(config)
        # PP-OCRv5_server_rec needs to modify the stride for the HGNetV2 layers
        self.backbone.embedder.stem3.convolution.stride = (1, 1)
        for idx, stride in enumerate(self.backbone.config.stage_downsample_strides):
            self.backbone.encoder.stages[idx].downsample.convolution.stride = stride

        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor] | BaseModelOutputWithNoAttention:
        outputs = self.backbone(pixel_values, **kwargs)
        hidden_state = outputs.feature_maps[-1]
        hidden_state = F.avg_pool2d(hidden_state, (3, 2))

        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_state,
            hidden_states=outputs.hidden_states,
        )


@auto_docstring(custom_intro="PPOCRV5ServerRec model for text recognition tasks.")
class PPOCRV5ServerRecForTextRecognition(PPOCRV5ServerRecPreTrainedModel):
    _keys_to_ignore_on_load_missing = ["num_batches_tracked"]

    def __init__(self, config: PPOCRV5ServerRecConfig):
        super().__init__(config)
        self.model = PPOCRV5ServerRecModel(config)
        self.head = PPOCRV5ServerRecHead(config)

        self.post_init()

    @can_return_tuple
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor] | BaseModelOutputWithNoAttention:
        outputs = self.model(pixel_values, **kwargs)
        logits = self.head(outputs.last_hidden_state)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=logits,
            hidden_states=outputs.hidden_states,
        )


__all__ = [
    "PPOCRV5ServerRecForTextRecognition",
    "PPOCRV5ServerRecImageProcessorFast",
    "PPOCRV5ServerRecConfig",
    "PPOCRV5ServerRecModel",
    "PPOCRV5ServerRecPreTrainedModel",
]
