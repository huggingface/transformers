# coding=utf-8
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
from typing import Optional, Union

import numpy as np

from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...video_utils import VideoInput


class Ernie4_5_VL_MoeProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_token_type_ids": False,
            "return_mm_token_type_ids": True,
        },
    }


class Ernie4_5_VL_MoeProcessor(ProcessorMixin):
    r"""
    Constructs a Ernie 4.5 VL processor which wraps a Ernie 4.5 VL image processor and a Llama tokenizer into a single processor.
    [`Ernie4_5_VL_MoeProcessor`] offers all the functionalities of [`Ernie4_5_VL_MoeImageProcessor`] and [`LlamaTokenizerFast`]. See the
    [`~Ernie4_5_VL_MoeProcessor.__call__`] and [`~Ernie4_5_VL_MoeProcessor.decode`] for more information.
    Args:
        image_processor ([`Ernie4_5_VL_MoeImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`], *optional*):
            The tokenizer is a required input.
        video_processor ([`Ernie4_5_VL_MoeVideoProcessor`], *optional*):
            The video processor is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

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

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        videos: Optional[VideoInput] = None,
        **kwargs: Unpack[Ernie4_5_VL_MoeProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the vision inputs, this method forwards the `vision_infos` and `kwargs` arguments to
        Ernie4_5_VL_MoeImageProcessor's [`~Ernie4_5_VL_MoeImageProcessor.__call__`] if `vision_infos` is not `None`.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `list[PIL.Image.Image]`, `list[np.ndarray]`, `list[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `list[str]`, `list[list[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            videos (`np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`):
                The image or batch of videos to be prepared. Each video can be a 4D NumPy array or PyTorch
                tensor, or a nested list of 3D frames. Both channels-first and channels-last formats are supported.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **pixel_values_videos** -- Pixel values of videos to be fed to a model. Returned when `videos` is not `None`.
            - **image_grid_thw** -- List of image 3D grid in LLM. Returned when `images` is not `None`.
            - **video_grid_thw** -- List of video 3D grid in LLM. Returned when `videos` is not `None`.
            - **mm_token_type_ids** -- List of token type ids differentiating between image, video and text input.
              Returned when `text` is not `None`.
        """
        output_kwargs = self._merge_kwargs(
            Ernie4_5_VL_MoeProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        image_inputs = videos_inputs = {}
        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]

        if videos is not None:
            videos_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            video_grid_thw = videos_inputs["video_grid_thw"]

        if not isinstance(text, list):
            text = [text]

        text = text.copy()  # below lines change text in-place

        if images is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    num_image_tokens = image_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if videos is not None:
            merge_length = self.video_processor.merge_size**2 * self.video_processor.temporal_patch_size
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    num_video_tokens = video_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(self.video_token, "<|placeholder|>" * num_video_tokens, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"], return_tensors=None)
        self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])

            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])  # text
            mm_token_type_ids[array_ids == self.image_token_id] = 1  # img
            mm_token_type_ids[array_ids == self.video_token_id] = 2  # vid

            # moe additionally adds start/end tokens
            moe_mm_token_type_ids = np.copy(mm_token_type_ids)
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

            # convert to base type
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.astype(int).tolist()
            text_inputs["moe_mm_token_type_ids"] = moe_mm_token_type_ids.astype(int).tolist()

        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs}, tensor_type=return_tensors)

    @property
    def model_input_names(self):
        """Additional `mm_token_type_ids` used for modality isolated MoE"""
        model_input_names = super().model_input_names
        model_input_names.append("mm_token_type_ids")
        model_input_names.append("moe_mm_token_type_ids")
        return model_input_names

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
            images_kwargs = Ernie4_5_VL_MoeProcessorKwargs._defaults.get("images_kwargs", {})
            images_kwargs.update(kwargs)
            merge_size = images_kwargs.get("merge_size", None) or self.image_processor.merge_size

            num_image_patches = [
                self.image_processor.get_number_of_image_patches(*image_size, images_kwargs)
                for image_size in image_sizes
            ]
            num_image_tokens = [(num_patches // merge_size**2) for num_patches in num_image_patches]
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})

        if video_sizes is not None:
            videos_kwargs = Ernie4_5_VL_MoeProcessorKwargs._defaults.get("videos_kwargs", {})
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


__all__ = ["Ernie4_5_VL_MoeProcessor"]
