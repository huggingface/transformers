# coding=utf-8
# Copyright 2024 HuggingFace Inc. team. All rights reserved.
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
import torch.utils.checkpoint
from torch import nn

from transformers.models.llava_next_video.configuration_llava_next_video import (
    LlavaNextVideoConfig,
)
from transformers.models.llava_next_video.modeling_llava_next_video import (
    LlavaNextVideoForConditionalGeneration,
    LlavaNextVideoPooler,
)

from ...utils import (
    logging,
)


logger = logging.get_logger(__name__)


class LlavaNextVideo2Config(LlavaNextVideoConfig):
    pass


class LlavaNextVideo2Pooler(LlavaNextVideoPooler):
    def __init__(self, config):
        super().__init__()
        mode = config.spatial_pool_mode
        stride = config.spatial_pool_stride
        out_channels = getattr(config, "spatial_pool_out_channels", config.vision_config.hidden_size)
        self.image_size = config.vision_config.image_size // config.vision_config.patch_size**2
        if mode == "average":
            self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride)
        elif mode == "max":
            self.pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
        elif mode == "bilinear":
            self.pool = nn.Upsample(scale_factor=(1 / stride), mode="bilinear")
        elif mode == "conv":
            self.pool = nn.Conv2d(
                in_channels=config.vision_config.hidden_size,
                out_channels=out_channels,
                kernel_size=stride,
                stride=stride,
            )
        else:
            raise ValueError(f"Unknown pooling mode: {mode}. Has to be one of [`average`, `max`, `conv`, `bilinear`]")


class LlavaNextVideo2ForConditionalGeneration(LlavaNextVideoForConditionalGeneration):
    def _get_video_features(self, pixel_values):
        batch_size, frames, channels, height, width = pixel_values.shape
        pixel_values = pixel_values.reshape(batch_size * frames, channels, height, width)
        image_features = self.vision_tower(pixel_values, output_hidden_states=True)
        selected_image_feature = image_features.hidden_states[self.vision_feature_layer]
        if self.vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif self.vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature

        # Same as image features except that video has pooling layer
        image_features = self.vision_resampler(selected_image_feature)
        image_features = self.multi_modal_projector(image_features)
        image_features = torch.split(image_features, frames, dim=0)

        features_processed = []
        for feature in image_features:
            resize_height = int(math.sqrt(feature.shape[1]))
            feature = feature.view(frames, 1, resize_height, resize_height, -1)
            feature = feature.permute(4, 0, 2, 1, 3).contiguous()
            feature = feature.flatten(1, 2).flatten(2, 3)
            feature = torch.cat(
                (feature, self.image_newline[:, None, None].expand(*feature.shape[:-1], 1).to(feature.device)),
                dim=-1,
            )
            feature = feature.permute(1, 2, 0)
            features_processed.append(feature)
        return features_processed
