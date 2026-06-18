# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

import re

import numpy as np

from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput, concatenate_list
from ...processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring
from ...video_utils import VideoInput


class InternVLProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding_side": "left",
            "return_mm_token_type_ids": False,
        },
        "images_kwargs": {
            "crop_to_patches": True,
        },
        "videos_kwargs": {
            "return_tensors": "pt",
        },
    }


@auto_docstring
class InternVLProcessor(ProcessorMixin):
    valid_processor_kwargs = InternVLProcessorKwargs

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        image_seq_length: int = 256,
        chat_template=None,
        **kwargs,
    ):
        r"""
        image_seq_length (`int`, *optional*, defaults to 256):
            The number of image token to use per image patch. it should be set so that:
            image_seq_length = (config.image_size // config.patch_size) ** 2 * (config.scale_factor**2)
        """
        super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template, **kwargs)

        self.image_seq_length = image_seq_length
        self.start_image_token = tokenizer.start_image_token
        self.end_image_token = tokenizer.end_image_token
        self.start_image_token_id = tokenizer.start_image_token_id
        self.end_image_token_id = tokenizer.end_image_token_id
        self.image_token = tokenizer.context_image_token
        self.video_token = tokenizer.video_token
        self.image_token_id = tokenizer.context_image_token_id

    @property
    def image_token_ids(self) -> list[int]:
        return [self.image_token_id, self.start_image_token_id, self.end_image_token_id]

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        videos: VideoInput | None = None,
        **kwargs: Unpack[InternVLProcessorKwargs],
    ) -> BatchFeature:
        r"""
        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """
        output_kwargs = self._merge_kwargs(
            InternVLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        return_tensors = output_kwargs["text_kwargs"].get("return_tensors")

        # Keep track of how many image/videos per sample we have, and in which order
        text = [text] if isinstance(text, str) else text
        pattern = re.compile(f"(?P<image>{re.escape(self.image_token)})|(?P<video>{re.escape(self.video_token)})")
        visuals_order = [match.lastgroup for sample in text for match in re.finditer(pattern, sample)]
        model_inputs = super().__call__(images=images, text=text, videos=videos, **output_kwargs)

        # Merge image and video pixel into a single array, as model expects only `pixel_values` as arg
        if images is not None:
            image_num_patches_indices = np.cumsum(model_inputs.pop("num_patches"))
        if videos is not None:
            video_pixel_values = model_inputs.pop("pixel_values_videos")
            batch_size, num_frames, *_ = video_pixel_values.shape
            video_pixel_values = video_pixel_values.flatten(0, 1)
            video_patch_indices = np.arange(num_frames * batch_size + 1, step=num_frames)

        image_index = video_index = 0
        image_video_patches = []
        for vision_type in visuals_order:
            if vision_type == "image":
                start_index = image_num_patches_indices[image_index] if image_index > 0 else 0
                end_index = image_num_patches_indices[image_index]
                image_video_patches.append(model_inputs["pixel_values"][start_index:end_index])
                image_index += 1
            else:
                start_index = video_patch_indices[video_index]
                end_index = video_patch_indices[video_index + 1]
                image_video_patches.append(video_pixel_values[start_index:end_index])
                video_index += 1

        if image_video_patches:
            model_inputs["pixel_values"] = concatenate_list(image_video_patches)
        return BatchFeature(data=model_inputs, tensor_type=return_tensors)

    def validate_inputs(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        videos: VideoInput | None = None,
        **kwargs: Unpack[InternVLProcessorKwargs],
    ):
        super().validate_inputs(images=images, text=text, videos=videos, **kwargs)
        if text is None:
            raise ValueError("You have to specify text.")

    def replace_image_token(self, image_inputs: dict, image_idx: int) -> str:
        image_num_patches = image_inputs["num_patches"]
        return f"{self.start_image_token}{self.image_token * self.image_seq_length * image_num_patches[image_idx]}{self.end_image_token}"

    def replace_video_token(self, video_inputs: dict, video_idx: int) -> str:
        num_frames = video_inputs["pixel_values_videos"][video_idx].shape[0]
        return "\n".join(
            f"Frame{i + 1}: {self.start_image_token}{self.image_token * self.image_seq_length}{self.end_image_token}"
            for i in range(num_frames)
        )

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        """
        Computes the number of placeholder tokens needed for multimodal inputs with the given sizes.

        Args:
            image_sizes (`list[list[int]]`, *optional*):
                The input sizes formatted as (height, width) per each image.

        Returns:
            `MultiModalData`: A `MultiModalData` object holding number of tokens per each of the provided
            input modalities, along with other useful data.
        """

        vision_data = {}
        if image_sizes is not None:
            images_kwargs = InternVLProcessorKwargs._defaults.get("images_kwargs", {})
            images_kwargs.update(kwargs)

            num_image_patches = [
                self.image_processor.get_number_of_image_patches(*image_size, images_kwargs)
                for image_size in image_sizes
            ]
            # Add 2 for BOI and EOI tokens
            num_image_tokens = [2 + (self.image_seq_length * num_patches) for num_patches in num_image_patches]
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})

        return MultiModalData(**vision_data)

    @property
    def model_input_names(self):
        # Overwritten because InternVL renames video inputs to `pixel_values` before returning
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return tokenizer_input_names + image_processor_input_names


__all__ = ["InternVLProcessor"]
