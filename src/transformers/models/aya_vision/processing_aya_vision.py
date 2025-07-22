# coding=utf-8
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

from typing import Optional, Union

import numpy as np

from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput, make_flat_list_of_images
from ...processing_utils import ImagesKwargs, MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput


class AyaVisionImagesKwargs(ImagesKwargs, total=False):
    crop_to_patches: Optional[bool]
    min_patches: Optional[int]
    max_patches: Optional[int]


class AyaVisionProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: AyaVisionImagesKwargs
    _defaults = {
        "text_kwargs": {
            "padding_side": "left",
            "padding": True,
            "return_mm_token_type_ids": False,
        },
        "images_kwargs": {
            "crop_to_patches": True,
        },
    }


class AyaVisionProcessor(ProcessorMixin):
    r"""
    Constructs a AyaVision processor which wraps a [`AutoImageProcessor`] and
    [`PretrainedTokenizerFast`] tokenizer into a single processor that inherits both the image processor and
    tokenizer functionalities. See the [`~AyaVisionProcessor.__call__`] and [`~AyaVisionProcessor.decode`] for more information.
    Args:
        image_processor ([`AutoImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`PreTrainedTokenizer`, `PreTrainedTokenizerFast`], *optional*):
            The tokenizer is a required input.
        patch_size (`int`, *optional*, defaults to 28):
            The size of image patches for tokenization.
        img_size (`int`, *optional*, defaults to 364):
            The size of the image to be tokenized. This should correspond to the size given to the image processor.
        image_token (`str`, *optional*, defaults to `"<image>"`):
            The token to be used to represent an image in the text.
        downsample_factor (`int`, *optional*, defaults to 1):
            The factor by which to scale the patch size.
        start_of_img_token (`str`, *optional*, defaults to `"<|START_OF_IMG|>"`):
            The token to be used to represent the start of an image in the text.
        end_of_img_token (`str`, *optional*, defaults to `"<|END_OF_IMG|>"`):
            The token to be used to represent the end of an image in the text.
        img_patch_token (`str`, *optional*, defaults to `"<|IMG_PATCH|>"`):
            The token to be used to represent an image patch in the text.
        img_line_break_token (`str`, *optional*, defaults to `"<|IMG_LINE_BREAK|>"`):
            The token to be used to represent a line break in the text.
        tile_token (`str`, *optional*, defaults to `"TILE"`):
            The token to be used to represent an image patch in the text.
        tile_global_token (`str`, *optional*, defaults to `"TILE_GLOBAL"`):
            The token to be used to represent the cover image in the text.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        patch_size: int = 28,
        img_size: int = 364,
        image_token="<image>",  # set the default and let users change if they have peculiar special tokens in rare cases
        downsample_factor: int = 1,
        start_of_img_token="<|START_OF_IMG|>",
        end_of_img_token="<|END_OF_IMG|>",
        img_patch_token="<|IMG_PATCH|>",
        img_line_break_token="<|IMG_LINE_BREAK|>",
        tile_token="TILE",
        tile_global_token="TILE_GLOBAL",
        chat_template=None,
        **kwargs,
    ):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

        self.image_token = image_token
        self.patch_size = patch_size * downsample_factor
        self.img_size = img_size

        self.start_of_img_token = start_of_img_token
        self.end_of_img_token = end_of_img_token
        self.img_patch_token = img_patch_token
        self.img_line_break_token = img_line_break_token
        self.tile_token = tile_token
        self.tile_global_token = tile_global_token
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.img_patch_token)
        self.image_ids = tokenizer.convert_tokens_to_ids(
            [img_patch_token, tile_token, tile_global_token, start_of_img_token, end_of_img_token]
        )

    def _prompt_split_image(self, num_patches):
        """
        Create a structured string representation of image tokens

        Args:
           num_patches: Number of patches in the image

        Returns:
            String with appropriate image tokens
        """

        img_patches_per_tile = (self.img_size // self.patch_size) ** 2
        img_string = f"{self.start_of_img_token}"
        if num_patches > 1:
            for idx in range(1, num_patches):
                img_string += f"{self.tile_token}_{idx}" + f"{self.img_patch_token}" * img_patches_per_tile

        img_string += f"{self.tile_global_token}" + f"{self.img_patch_token}" * img_patches_per_tile
        img_string += f"{self.end_of_img_token}"
        return img_string

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Optional[Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]] = None,
        audio=None,
        videos=None,
        **kwargs: Unpack[AyaVisionProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to PreTrainedTokenizerFast's [`~PreTrainedTokenizerFast.__call__`] to encode the text.
        To prepare the vision inputs, this method forwards the `images` and `kwargs` arguments to
        GotOcr2ImageProcessor's [`~GotOcr2ImageProcessor.__call__`] if `images` is not `None`.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `list[PIL.Image.Image]`, `list[np.ndarray]`, `list[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `list[str]`, `list[list[str]]`):
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
            AyaVisionProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if not isinstance(text, (list, tuple)):
            text = [text]

        # Process images
        image_inputs = {}
        if images is not None:
            images = make_flat_list_of_images(images)
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            num_patches = image_inputs.pop("num_patches")
            image_index = 0
            processed_text = []
            for prompt in text:
                new_prompt = prompt
                while "<image>" in new_prompt:
                    # Replace the image placeholder with structured image tokens
                    image_tokens = self._prompt_split_image(num_patches[image_index])
                    new_prompt = new_prompt.replace("<image>", image_tokens, 1)
                    image_index += 1
                processed_text.append(new_prompt)

            if image_index != len(images):
                raise ValueError("Number of image placeholders in the prompt does not match the number of images.")

            text = processed_text

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"], return_tensors=None)

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[np.isin(array_ids, self.image_ids)] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)

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
            images_kwargs = AyaVisionProcessorKwargs._defaults.get("images_kwargs", {})
            images_kwargs.update(kwargs)

            num_image_patches = [
                self.image_processor.get_number_of_image_patches(*image_size, images_kwargs)
                for image_size in image_sizes
            ]

            token_per_patch = (self.img_size // self.patch_size) ** 2
            num_image_tokens = [
                token_per_patch + 3 + sum(token_per_patch + 1 for _ in range(1, num_patches))
                for num_patches in num_image_patches
            ]  # Add +3 and +1 for BOI/EOI and image tile tokens
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})

        return MultiModalData(**vision_data)

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


__all__ = ["AyaVisionProcessor"]
