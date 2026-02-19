# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OmniVinci configuration (HF-style canonical config file)."""

from copy import deepcopy

from transformers import PretrainedConfig


# Core token/config constants migrated from constants.py.
IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_SOUND_TOKEN = "<sound>"
SENTINEL_TOKEN = "<vila/sentinel>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

MEDIA_TOKENS = {
    "image": "<image>",
    "video": "<vila/video>",
    "sound": "<sound>",
}

MM_BOS_EOS_TOKENS = {
    "image": ["<|image_bos|>", "<|image_eos|>"],
    "video": ["<|video_bos|>", "<|video_eos|>"],
    "sound": ["<|sound_bos|>", "<|sound_eos|>"],
}


class OmniVinciConfig(PretrainedConfig):
    """Configuration class for OmniVinci models.

    Migration note:
    We intentionally keep `model_type = "vila"` at this stage to preserve
    compatibility with existing checkpoints and current loading behavior.
    """

    model_type = "vila"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        llm_cfg=None,
        vision_tower_cfg=None,
        mm_projector_cfg=None,
        sound_tower_cfg=None,
        sound_mm_projector_cfg=None,
        architectures=None,
        resume_path=None,
        hidden_size=None,
        mm_hidden_size=None,
        image_aspect_ratio=None,
        num_video_frames=None,
        fps=None,
        mm_vision_select_layer=None,
        mm_vision_select_feature=None,
        mm_use_im_start_end=False,
        mm_use_im_patch_token=False,
        mm_projector_lr=None,
        vision_tower_lr=None,
        vision_resolution=None,
        interpolate_mode=None,
        s2=None,
        dynamic_s2=None,
        s2_scales=None,
        s2_max_split_size=None,
        s2_resize_output_to_scale_idx=0,
        min_tiles: int | None = 1,
        max_tiles: int | None = 12,
        num_time_tokens=None,
        time_token_format=None,
        image_encoder: str = '{"_target_": "llava.model.encoders.BasicImageEncoder"}',
        video_encoder: str = '{"_target_": "llava.model.encoders.TSPVideoEncoder"}',
        sound_encoder: str = '{"_target_": "llava.model.encoders.BasicSoundEncoder"}',
        ignore_index: int = IGNORE_INDEX,
        default_image_token: str = DEFAULT_IMAGE_TOKEN,
        default_sound_token: str = DEFAULT_SOUND_TOKEN,
        sentinel_token: str = SENTINEL_TOKEN,
        default_im_start_token: str = DEFAULT_IM_START_TOKEN,
        default_im_end_token: str = DEFAULT_IM_END_TOKEN,
        media_tokens=None,
        mm_bos_eos_tokens=None,
        **kwargs,
    ):
        self.architectures = architectures
        self.llm_cfg = llm_cfg
        self.vision_tower_cfg = vision_tower_cfg
        self.mm_projector_cfg = mm_projector_cfg
        self.sound_tower_cfg = sound_tower_cfg
        self.sound_mm_projector_cfg = sound_mm_projector_cfg
        self.resume_path = resume_path

        self.hidden_size = hidden_size
        self.mm_hidden_size = mm_hidden_size
        self.image_aspect_ratio = image_aspect_ratio
        self.num_video_frames = num_video_frames
        self.fps = fps
        self.mm_vision_select_layer = mm_vision_select_layer
        self.mm_vision_select_feature = mm_vision_select_feature
        self.mm_use_im_start_end = mm_use_im_start_end
        self.mm_use_im_patch_token = mm_use_im_patch_token
        self.mm_projector_lr = mm_projector_lr
        self.vision_tower_lr = vision_tower_lr
        self.vision_resolution = vision_resolution
        self.interpolate_mode = interpolate_mode
        self.s2 = s2
        self.dynamic_s2 = dynamic_s2
        self.s2_scales = s2_scales
        self.s2_max_split_size = s2_max_split_size
        self.s2_resize_output_to_scale_idx = s2_resize_output_to_scale_idx
        self.min_tiles = min_tiles
        self.max_tiles = max_tiles
        self.num_time_tokens = num_time_tokens
        self.time_token_format = time_token_format

        self.image_encoder = image_encoder
        self.video_encoder = video_encoder
        self.sound_encoder = sound_encoder
        self.audio_sampling_rate = 16000
        self.audio_chunk_length = 120
        self.interleaved_vis_aud_in_video = True
        self.interleaved_video_segment_duration = 30
        self.audio_hop_length = 60

        self.ignore_index = ignore_index
        self.default_image_token = default_image_token
        self.default_sound_token = default_sound_token
        self.sentinel_token = sentinel_token
        self.default_im_start_token = default_im_start_token
        self.default_im_end_token = default_im_end_token
        self.media_tokens = deepcopy(MEDIA_TOKENS if media_tokens is None else media_tokens)
        self.mm_bos_eos_tokens = deepcopy(MM_BOS_EOS_TOKENS if mm_bos_eos_tokens is None else mm_bos_eos_tokens)

        super().__init__(**kwargs)


__all__ = [
    "OmniVinciConfig",
    "IGNORE_INDEX",
    "DEFAULT_IMAGE_TOKEN",
    "DEFAULT_SOUND_TOKEN",
    "SENTINEL_TOKEN",
    "DEFAULT_IM_START_TOKEN",
    "DEFAULT_IM_END_TOKEN",
    "MEDIA_TOKENS",
    "MM_BOS_EOS_TOKENS",
]
