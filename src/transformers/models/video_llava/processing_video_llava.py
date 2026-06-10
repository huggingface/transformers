# Copyright 2023 The HuggingFace Inc. team.
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
"""
Processor class for VideoLlava.
"""

import numpy as np

from ...image_utils import get_image_size, to_numpy_array
from ...processing_utils import ProcessingKwargs, ProcessorMixin
from ...utils import auto_docstring


class VideoLlavaProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "common_kwargs": {
            "return_tensors": "pt",
        },
    }


@auto_docstring
class VideoLlavaProcessor(ProcessorMixin):
    valid_processor_kwargs = VideoLlavaProcessorKwargs

    def __init__(
        self,
        image_processor=None,
        video_processor=None,
        tokenizer=None,
        patch_size=14,
        vision_feature_select_strategy="default",
        image_token="<image>",  # set the default and let users change if they have peculiar special tokens in rare cases
        video_token="<video>",
        chat_template=None,
        num_additional_image_tokens=1,
        **kwargs,
    ):
        r"""
        patch_size (`int`, *optional*, defaults to 14):
            Patch size from the vision tower.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Should be same as in model's config
        image_token (`str`, *optional*, defaults to `"<image>"`):
            Special token used to denote image location.
        video_token (`str`, *optional*, defaults to `"<video>"`):
            Special token used to denote video location.
        num_additional_image_tokens (`int`, *optional*, defaults to 1):
            Number of additional tokens added to the image embeddings, such as CLS (+1). If the backbone has no CLS or other
            extra tokens appended, no need to set this arg.
        """
        self.patch_size = patch_size
        self.num_additional_image_tokens = num_additional_image_tokens
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.image_token = tokenizer.image_token if hasattr(tokenizer, "image_token") else image_token
        self.video_token = tokenizer.video_token if hasattr(tokenizer, "video_token") else video_token
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.video_token_id = tokenizer.convert_tokens_to_ids(self.video_token)
        super().__init__(image_processor, video_processor, tokenizer, chat_template=chat_template)

    def replace_image_token(self, image_inputs: dict, image_idx: int) -> str:
        image = image_inputs["pixel_values_images"][image_idx]
        height, width = get_image_size(to_numpy_array(image))
        num_image_tokens = (height // self.patch_size) * (width // self.patch_size)
        num_image_tokens += self.num_additional_image_tokens
        if self.vision_feature_select_strategy == "default":
            num_image_tokens -= 1
        return self.image_token * num_image_tokens

    def replace_video_token(self, video_inputs: dict, video_idx: int) -> str:
        one_video = video_inputs["pixel_values_videos"][video_idx]
        if isinstance(one_video, (list, tuple)):
            one_video = np.array(one_video)
        else:
            one_video = to_numpy_array(one_video)
        height, width = get_image_size(one_video[0])
        num_frames = one_video.shape[0]
        num_image_tokens = (height // self.patch_size) * (width // self.patch_size)
        num_image_tokens += self.num_additional_image_tokens
        return self.video_token * (num_image_tokens * num_frames)


__all__ = ["VideoLlavaProcessor"]
