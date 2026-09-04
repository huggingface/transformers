# Copyright 2025 Baidu and HuggingFace Inc. team. All rights reserved.
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
import os.path
from pathlib import Path
from shutil import SameFileError, copyfile

import numpy as np

from ...image_utils import ImageInput
from ...processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import TextInput
from ...utils import auto_docstring
from ...video_utils import VideoInput


class Ernie4_5_VLMoeProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_token_type_ids": False,
            "return_mm_token_type_ids": True,
        },
    }


@auto_docstring
class Ernie4_5_VLMoeProcessor(ProcessorMixin):
    valid_processor_kwargs = Ernie4_5_VLMoeProcessorKwargs

    def __init__(self, image_processor=None, tokenizer=None, video_processor=None, chat_template=None, **kwargs):
        self.image_token = tokenizer.image_token
        self.image_end_token = tokenizer.image_end_token
        self.image_start_token = tokenizer.image_start_token
        self.video_token = tokenizer.video_token
        self.video_end_token = tokenizer.video_end_token
        self.video_start_token = tokenizer.video_start_token

        self.image_token_id = tokenizer.image_token_id
        self.image_end_token_id = tokenizer.image_end_token_id
        self.image_start_token_id = tokenizer.image_start_token_id
        self.video_token_id = tokenizer.video_token_id
        self.video_end_token_id = tokenizer.video_end_token_id
        self.video_start_token_id = tokenizer.video_start_token_id

        super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template)

    def save_pretrained(self, save_directory, push_to_hub: bool = False, **kwargs):
        """We additionally save a copy of the font to the `save_directory` (if we found a file there)"""
        os.makedirs(save_directory, exist_ok=True)

        if os.path.isfile(self.video_processor.font):
            try:
                copyfile(self.video_processor.font, Path(save_directory, Path(self.video_processor.font).name))
            except SameFileError:  # already exists which we allow (copy if needed)
                pass

        return super().save_pretrained(save_directory, push_to_hub, **kwargs)

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | list[TextInput] | None = None,
        videos: VideoInput | None = None,
        **kwargs: Unpack[Ernie4_5_VLMoeProcessorKwargs],
    ):
        model_inputs = super().__call__(images=images, text=text, videos=videos, **kwargs)
        model_inputs["moe_mm_token_type_ids"] = self.create_moe_mm_token_type_ids(model_inputs["input_ids"])
        return model_inputs

    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        merge_length = self.image_processor.merge_size**2
        num_image_tokens = image_inputs["image_grid_thw"][image_idx].prod() // merge_length
        return self.image_token * num_image_tokens

    def replace_video_token(self, video_inputs: dict, video_idx: int, **kwargs) -> str:
        merge_length = self.video_processor.merge_size**2 * self.video_processor.temporal_patch_size
        num_video_tokens = video_inputs["video_grid_thw"][video_idx].prod() // merge_length
        return self.video_token * num_video_tokens

    def create_moe_mm_token_type_ids(self, input_ids: list) -> list[list[int]]:
        """
        Build per-token modality type IDs for MoE blocks.

        Returns:
            `list[list[int]]`: A list of the same structure as ``input_ids``, where each
            integer is the modality type ID for the corresponding token. Note that this is different
            from `mm_token_type_ids` since MoE encodes start/end of vision tokens as well
        """
        array_ids = np.array(input_ids)
        moe_mm_token_type_ids = np.zeros_like(input_ids)  # text
        moe_mm_token_type_ids[array_ids == self.image_token_id] = 1  # img
        moe_mm_token_type_ids[array_ids == self.video_token_id] = 2  # vid

        # moe additionally adds start/end tokens
        for token_id in [
            self.image_start_token_id,
            self.image_end_token_id,
        ]:
            moe_mm_token_type_ids[array_ids == token_id] = 1
        for token_id in [
            self.video_start_token_id,
            self.video_end_token_id,
        ]:
            moe_mm_token_type_ids[array_ids == token_id] = 2

        moe_mm_token_type_ids = moe_mm_token_type_ids.astype(int)

        # Cast MoE token types to the same input type as input IDs
        if isinstance(input_ids, np.ndarray):
            return moe_mm_token_type_ids
        elif hasattr(input_ids, "device") and hasattr(input_ids, "dtype"):
            # torch.Tensor (or tensor-like) without importing torch
            return type(input_ids)(moe_mm_token_type_ids).to(device=input_ids.device, dtype=input_ids.dtype)
        else:
            return moe_mm_token_type_ids.tolist()

    @property
    def model_input_names(self):
        """Additional `mm_token_type_ids` used for modality isolated MoE"""
        return super().model_input_names + ["mm_token_type_ids", "moe_mm_token_type_ids"]

    def _get_num_multimodal_tokens(self, image_sizes=None, video_sizes=None, **kwargs):
        """
        Computes the number of placeholder tokens needed for multimodal inputs with the given sizes.
        Args:
            image_sizes (`list[list[int]]`, *optional*):
                The input sizes formatted as (height, width) per each image.
            video_sizes (`list[list[int]]`, *optional*):
                The input sizes formatted as (num_frames, height, width) per each video.
        Returns:
            `MultiModalData`: A `MultiModalData` object holding number of tokens per each of the provided
            input modalities, along with other useful data.
        """

        vision_data = {}
        if image_sizes is not None:
            images_kwargs = Ernie4_5_VLMoeProcessorKwargs._defaults.get("images_kwargs", {})
            images_kwargs.update(kwargs)
            merge_size = images_kwargs.get("merge_size", None) or self.image_processor.merge_size

            num_image_patches = [
                self.image_processor.get_number_of_image_patches(*image_size, images_kwargs)
                for image_size in image_sizes
            ]
            num_image_tokens = [(num_patches // merge_size**2) for num_patches in num_image_patches]
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})

        if video_sizes is not None:
            videos_kwargs = Ernie4_5_VLMoeProcessorKwargs._defaults.get("videos_kwargs", {})
            videos_kwargs.update(kwargs)
            temporal_merge_size = (
                videos_kwargs.get("temporal_patch_size", None) or self.video_processor.temporal_patch_size
            )

            num_video_patches = [
                self.video_processor.get_number_of_video_patches(*video_size, videos_kwargs)
                for video_size in video_sizes
            ]
            num_video_tokens = [
                (num_patches // merge_size**2 // temporal_merge_size) for num_patches in num_video_patches
            ]
            vision_data["num_video_tokens"] = num_video_tokens

        return MultiModalData(**vision_data)


__all__ = ["Ernie4_5_VLMoeProcessor"]
