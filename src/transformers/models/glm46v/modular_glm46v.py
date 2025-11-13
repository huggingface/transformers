# coding=utf-8
# Copyright 2025 the HuggingFace Team. All rights reserved.
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

from typing import Optional, Union

import numpy as np

from ...configuration_utils import PreTrainedConfig
from ...video_utils import VideoMetadata
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel
from ..glm4v.image_processing_glm4v import Glm4vImageProcessor
from ..glm4v.image_processing_glm4v_fast import Glm4vImageProcessorFast
from ..glm4v.modeling_glm4v import Glm4vForConditionalGeneration, Glm4vModel, Glm4vPreTrainedModel
from ..glm4v.processing_glm4v import Glm4vProcessor
from ..glm4v.video_processing_glm4v import Glm4vVideoProcessor


class Glm46VConfig(PreTrainedConfig):
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig}

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=151343,
        video_token_id=151344,
        image_start_token_id=151339,
        image_end_token_id=151340,
        video_start_token_id=151341,
        video_end_token_id=151342,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            vision_config["model_type"] = vision_config.get("model_type", "glm4v")
            self.vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        elif vision_config is None:
            self.vision_config = CONFIG_MAPPING["glm4v"]()

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "glm4v_text")
            self.text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            self.text_config = CONFIG_MAPPING["glm4v_text"]()

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.video_start_token_id = video_start_token_id
        self.video_end_token_id = video_end_token_id
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id

        super().__init__(**kwargs)


class Glm46VPreTrainedModel(Glm4vPreTrainedModel):
    _can_record_outputs = None
    _no_split_modules = None


class Glm46VModel(Glm4vModel):
    _no_split_modules = None

    def __init__(self, config):
        super().__init__(config)
        self.visual = AutoModel.from_config(config.vision_config)
        self.language_model = AutoModel.from_config(config.text_config)


class Glm46VForConditionalGeneration(Glm4vForConditionalGeneration):
    pass


class Glm46VProcessor(Glm4vProcessor):
    def replace_frame_token_id(self, timestamp_sec):
        return f"<|begin_of_image|>{self.image_token}<|end_of_image|>{timestamp_sec:.1f} seconds"


class Glm46VImageProcessor(Glm4vImageProcessor):
    pass


class Glm46VImageProcessorFast(Glm4vImageProcessorFast):
    pass


class Glm46VVideoProcessor(Glm4vVideoProcessor):
    def sample_frames(
        self,
        metadata: VideoMetadata,
        fps: Optional[Union[int, float]] = None,
        **kwargs,
    ):
        if metadata is None or getattr(metadata, "fps", None) is None:
            raise ValueError(
                "Asked to sample frames per second but no video metadata was provided which is required when sampling in Glm46V. "
                "Please pass in `VideoMetadata` object or set `do_sample_frames=False`"
            )

        total_frames = metadata.total_num_frames
        max_frame_idx = total_frames - 1
        duration = metadata.duration or round(max_frame_idx / metadata.fps) + 1

        DYNAMIC_FPS_THRES = {30: 3, 300: 1, 2400: 0.5}
        MAX_FRAME_COUNT_DYNAMIC = 640
        MAX_DURATION = 2400
        effective_duration = min(duration, MAX_DURATION)
        if effective_duration <= 30:
            target_fps = DYNAMIC_FPS_THRES[30]
        elif effective_duration <= 300:
            target_fps = DYNAMIC_FPS_THRES[300]
        else:
            target_fps = DYNAMIC_FPS_THRES[2400]
        extract_t = int(effective_duration * target_fps * self.temporal_patch_size)
        extract_t = min(extract_t, MAX_FRAME_COUNT_DYNAMIC)

        duration_per_frame = 1 / metadata.fps
        timestamps = [i * duration_per_frame for i in range(total_frames)]
        max_second = int(duration)

        if total_frames < extract_t:
            frame_indices = np.linspace(0, total_frames - 1, extract_t, dtype=int).tolist()
        else:
            frame_indices = []
            current_second = 0
            inv_fps = 1 / (self.temporal_patch_size * target_fps)
            for frame_index in range(total_frames):
                if timestamps[frame_index] >= current_second:
                    current_second += inv_fps
                    frame_indices.append(frame_index)
                    if current_second >= max_second:
                        break

        if len(frame_indices) < extract_t:
            if len(frame_indices) == 0:
                start, end = 0, max(total_frames - 1, 0)
            else:
                start, end = frame_indices[0], frame_indices[-1]
            frame_indices = np.linspace(start, end, extract_t, dtype=int).tolist()
        elif len(frame_indices) > extract_t:
            frame_indices = np.linspace(0, total_frames - 1, extract_t, dtype=int).tolist()

        seen, uniq = set(), []
        for idx in frame_indices:
            if idx not in seen:
                seen.add(idx)
                uniq.append(idx)

        if len(uniq) & 1:
            uniq.append(uniq[-1])

        return np.array(uniq)


__all__ = [
    "Glm46VConfig",
    "Glm46VModel",
    "Glm46VPreTrainedModel",
    "Glm46VForConditionalGeneration",
    "Glm46VProcessor",
    "Glm46VImageProcessor",
    "Glm46VImageProcessorFast",
    "Glm46VVideoProcessor",
]
