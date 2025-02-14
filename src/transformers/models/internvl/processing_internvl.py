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


from typing import List, Optional, Union

import numpy as np

from transformers.processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput, concatenate_list, make_batched_videos, make_flat_list_of_images


class InternVLImagesKwargs(ImagesKwargs, total=False):
    crop_to_patches: Optional[bool]
    min_patches: Optional[int]
    max_patches: Optional[int]


class InternVLProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: InternVLImagesKwargs
    _defaults = {
        "text_kwargs": {
            "padding_side": "left",
        },
        "images_kwargs": {
            "crop_to_patches": True,
            "min_patches": 1,
            "max_patches": 12,
        },
    }


class InternVLProcessor(ProcessorMixin):
    r"""
    Constructs a InternVL processor which wraps a [`GotOcr2ImageProcessor`] and
    [`PretrainedTokenizerFast`] tokenizer into a single processor that inherits both the image processor and
    tokenizer functionalities. See the [`~InternVLProcessor.__call__`] and [`~InternVLProcessor.decode`] for more information.
    Args:
        image_processor ([`GotOcr2ImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`PreTrainedTokenizer`, `PreTrainedTokenizerFast`], *optional*):
            The tokenizer is a required input.
        image_seq_length (`int`, *optional*, defaults to 256):
            The number of image token to use per image patch.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template", "image_seq_length"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self, image_processor=None, tokenizer=None, image_seq_length: int = 256, chat_template=None, **kwargs
    ):
        self.image_seq_length = image_seq_length

        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        audio=None,
        videos=None,
        **kwargs: Unpack[InternVLProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to PreTrainedTokenizerFast's [`~PreTrainedTokenizerFast.__call__`] to encode the text if `text`
        is not `None`, otherwise encode default OCR queries which depends on the `format`, `box`, `color`, `multi_page` and
        `crop_to_patches` arguments. To prepare the vision inputs, this method forwards the `images` and `kwrags` arguments to
        GotOcr2ImageProcessor's [`~GotOcr2ImageProcessor.__call__`] if `images` is not `None`.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).

            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """
        if text is None:
            raise ValueError("You have to specify text.")

        output_kwargs = self._merge_kwargs(
            InternVLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if not isinstance(text, (list, tuple)):
            text = [text]

        # Process images and videos separately, as videos don't support crop_to_patches
        image_num_patches = []
        video_num_patches = []
        image_videos_inputs = {}
        if images is not None:
            images = make_flat_list_of_images(images)
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_num_patches = image_inputs.pop("num_patches")
            image_pixel_values = image_inputs.pop("pixel_values")
            image_num_patches_indices = np.cumsum(image_num_patches)
        if videos is not None:
            videos = make_batched_videos(videos)
            num_frames_per_video = [len(video) for video in videos]
            patch_indices = np.cumsum(num_frames_per_video)
            output_kwargs["images_kwargs"]["crop_to_patches"] = False
            video_inputs = self.image_processor(images=videos, **output_kwargs["images_kwargs"])
            video_num_patches = video_inputs.pop("num_patches")
            video_pixel_values = video_inputs.pop("pixel_values")
            video_num_patches_indices = np.cumsum(video_num_patches)

        if images is not None or videos is not None:
            image_index = 0
            video_index = 0
            processed_text = []
            image_video_patches = []  # List to store processed image/video patches
            # Support interlaced image and video in prompts:
            # Processed patches of images and videos are inserted in `image_video_patches` in the order they appear in the prompts
            for prompt in text:
                new_prompt = prompt
                while "<image>" in new_prompt or "<video>" in new_prompt:
                    if "<image>" in new_prompt and (
                        "<video>" not in new_prompt or new_prompt.index("<image>") < new_prompt.index("<video>")
                    ):
                        # Get the slice of patches corresponding to the current image
                        start_index = image_num_patches_indices[image_index - 1] if image_index > 0 else 0
                        end_index = image_num_patches_indices[image_index]
                        image_video_patches.append(image_pixel_values[start_index:end_index])
                        # Replace the corresponding image placeholder with the correct number of image tokens
                        new_prompt = new_prompt.replace(
                            "<image>",
                            f"<img>{'<IMG_CONTEXT>' * self.image_seq_length * image_num_patches[image_index]}</img>",
                            1,
                        )
                        image_index += 1
                    else:
                        # Get the slice of patches corresponding to the current video
                        # Here we need to account for both the multiple video frames and the potential multiple patches per frame
                        # As of now, InternVL only supports one patch per frame, but we keep the code flexible for future updates
                        current_patch_index = patch_indices[video_index - 1] if video_index > 0 else 0
                        end_patch_index = patch_indices[video_index]
                        start_index = video_num_patches_indices[current_patch_index] if video_index > 0 else 0
                        end_index = video_num_patches_indices[end_patch_index - 1]
                        image_video_patches.append(video_pixel_values[start_index:end_index])
                        # Get the number of patches per frame and replace the video placeholder with the correct number of image tokens
                        num_patches = list(video_num_patches[current_patch_index:end_patch_index])
                        video_prompt = "\n".join(
                            f"Frame{i+1}: <img>{'<IMG_CONTEXT>'*self.image_seq_length* num_patches[i]}</img>"
                            for i in range(len(num_patches))
                        )
                        new_prompt = new_prompt.replace("<video>", video_prompt, 1)
                        video_index += 1
                processed_text.append(new_prompt)
            # Concatenate the interlaced image and video patches (function agnostic to the patches type (list, numpy array, torch tensor))
            image_videos_inputs = {"pixel_values": concatenate_list(image_video_patches)}
            text = processed_text

        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        return BatchFeature(data={**text_inputs, **image_videos_inputs})

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(tokenizer_input_names) + list(image_processor_input_names)


__all__ = ["InternVLProcessor"]
