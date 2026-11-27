# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Processor class for Ovis2.5."""

from ...processing_utils import MultiModalData, ProcessorMixin
from ...utils import auto_docstring


@auto_docstring
class Ovis2_5Processor(ProcessorMixin):
    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        chat_template=None,
        **kwargs,
    ):
        self.image_token = tokenizer.image_token
        self.video_token = tokenizer.video_token
        self.image_start_token = tokenizer.image_start_token
        self.image_end_token = tokenizer.image_end_token
        self.video_start_token = tokenizer.video_start_token
        self.video_end_token = tokenizer.video_end_token

        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.video_token_id = tokenizer.convert_tokens_to_ids(self.video_token)
        self.image_start_token_id = tokenizer.convert_tokens_to_ids(self.image_start_token)
        self.image_end_token_id = tokenizer.convert_tokens_to_ids(self.image_end_token)
        self.video_start_token_id = tokenizer.convert_tokens_to_ids(self.video_start_token)
        self.video_end_token_id = tokenizer.convert_tokens_to_ids(self.video_end_token)

        super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template)

    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        merge_length = self.image_processor.merge_size**2
        num_visual_tokens = image_inputs["image_grid_thw"][image_idx].prod() // merge_length
        return self.image_start_token + self.image_token * num_visual_tokens + self.image_end_token

    def replace_video_token(self, video_inputs: dict, video_idx: int, **kwargs) -> str:
        merge_length = self.video_processor.merge_size**2
        num_visual_tokens = video_inputs["video_grid_thw"][video_idx].prod() // merge_length
        return self.video_start_token + self.video_token * num_visual_tokens + self.video_end_token

    def _get_num_multimodal_tokens(self, image_sizes=None, video_sizes=None, **kwargs):
        """Compute multimodal token counts without materializing pixel values."""
        vision_data = {}

        if image_sizes is not None:
            merge_size = kwargs.get("merge_size") or self.image_processor.merge_size
            num_image_patches = [
                self.image_processor.get_number_of_image_patches(height, width, kwargs)
                for height, width in image_sizes
            ]
            vision_data["num_image_patches"] = num_image_patches
            vision_data["num_image_tokens"] = [num_patches // merge_size**2 for num_patches in num_image_patches]

        if video_sizes is not None:
            merge_size = kwargs.get("merge_size") or self.video_processor.merge_size
            num_video_patches = [
                self.video_processor.get_number_of_video_patches(num_frames, height, width, kwargs)
                for num_frames, height, width in video_sizes
            ]
            vision_data["num_video_tokens"] = [num_patches // merge_size**2 for num_patches in num_video_patches]

        return MultiModalData(**vision_data)


__all__ = ["Ovis2_5Processor"]
