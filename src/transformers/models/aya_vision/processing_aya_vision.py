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


from functools import partial
from typing import Dict, List, Optional, Union

import numpy as np

from transformers.processing_utils import (
    AllKwargsForChatTemplate,
    ImagesKwargs,
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
)
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from ...image_processing_utils import BatchFeature
from ...image_utils import (
    ImageInput,
    make_flat_list_of_images,
    get_image_size,
)

# Add constants for image token handling
START_OF_IMG = "<|START_OF_IMG|>"
END_OF_IMG = "<|END_OF_IMG|>"
IMG_PATCH = "<|IMG_PATCH|>"
IMG_LINE_BREAK = "<|IMG_LINE_BREAK|>"

TILE = "TILE"
TILE_GLOBAL = "TILE_GLOBAL"


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
        patch_size (`int`, *optional*, defaults to 14):
            The size of image patches for tokenization.
        downsample_factor (`int`, *optional*, defaults to 1):
            The factor by which to scale the patch size.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = [
        "chat_template", 
        "image_token", 
        "patch_size", 
        "downsample_factor",
        "vision_feature_select_strategy",
        "max_splits_per_img", 
        "size"
    ]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self, 
        image_processor=None, 
        tokenizer=None, 
        patch_size: int = 14,
        downsample_factor: int = 1,
        chat_template=None, 
        **kwargs
    ):
        self.patch_size = patch_size * downsample_factor
        self.img_size = kwargs.get("size", 364)
        self.max_splits_per_img = kwargs.get("max_splits_per_img", 12)
        self.vision_feature_select_strategy = kwargs.get("vision_feature_select_strategy", "full")
        self.image_token = kwargs.get("image_token", "<image>")

        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        
        # Configure the image processor with our parameters
        if self.image_processor is not None:
            self.image_processor.img_size = self.img_size
            self.image_processor.max_splits_per_image = self.max_splits_per_img

    def img_tokens_from_size(self, width: int, height: int) -> str:
        """
        Convert image dimensions to appropriate token representation
        
        Args:
            width: Image width
            height: Image height
            
        Returns:
            String representation of image tokens
        """
        w_patch = width / self.patch_size
        h_patch = height / self.patch_size

        # Number of crops/tiles after resizing to optimal aspect ratio
        w_tiles = width // self.img_size
        h_tiles = height // self.img_size

        assert w_patch % 1 == 0 and h_patch % 1 == 0, "height and width doesn't match the patch size"
        return self.create_image_str(w_tiles, h_tiles)

    def create_image_str(self, w_tiles, h_tiles):
        """
        Create a structured string representation of image tokens
        
        Args:
            w_tiles: Number of tiles horizontally
            h_tiles: Number of tiles vertically
            
        Returns:
            String with appropriate image tokens
        """
        idx = 1
        img_patches_per_tile = (self.img_size // self.patch_size) ** 2

        img_string = f"{START_OF_IMG}"
        if h_tiles * w_tiles > 1:
            for h_tile in range(h_tiles):
                for w_tile in range(w_tiles):
                    img_string += f"{TILE}_{idx}" + f"{IMG_PATCH}" * img_patches_per_tile
                    idx += 1

        img_string += f"{TILE_GLOBAL}" + f"{IMG_PATCH}" * img_patches_per_tile
        img_string += f"{END_OF_IMG}"
        return img_string

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        audio=None,
        **kwargs: Unpack[AyaVisionProcessorKwargs],
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
            image_num_patches = image_inputs.pop("num_patches")
            image_pixel_values = image_inputs.pop("pixel_values")
            
            # Get resized dimensions for each image
            size = output_kwargs["images_kwargs"].get("size", None)
            resized_dimensions = [
                self.image_processor.get_resized_dimensions(img, size) 
                for img in images
            ]
            
            image_index = 0
            processed_text = []
            for prompt in text:
                new_prompt = prompt
                while "<image>" in new_prompt:
                    # Replace the image placeholder with structured image tokens
                    height, width = resized_dimensions[image_index]
                    image_tokens = self.img_tokens_from_size(width, height)
                    new_prompt = new_prompt.replace("<image>", image_tokens, 1)
                    image_index += 1
                processed_text.append(new_prompt)

            if image_index != len(images):
                raise ValueError("Number of image placeholders in the prompt does not match the number of images.")

            image_inputs = {"pixel_values": image_pixel_values}
            text = processed_text

        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        return BatchFeature(data={**text_inputs, **image_inputs})

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

    def apply_chat_template(
        self,
        conversation: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
        chat_template: Optional[str] = None,
        **kwargs: Unpack[AllKwargsForChatTemplate],
    ):
        """
        Similar to the `apply_chat_template` method on tokenizers, this method applies a Jinja template to input
        conversations to turn them into a single tokenizable string.

        The input is expected to be in the following format, where each message content is a list consisting of text and
        optionally image inputs. One can also provide an image, URL or local path which will be used to form
        `pixel_values` when `return_dict=True`. If not provided, one will get only the formatted text, optionally tokenized text.

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "https://www.ilankelman.org/stopsigns/australia.jpg"},
                    {"type": "text", "text": "Please describe this image in detail."},
                ],
            },
        ]

        Args:
            conversation (`Union[List[Dict, [str, str]], List[List[Dict[str, str]]]]`):
                The conversation to format.
            chat_template (`Optional[str]`, *optional*):
                The Jinja template to use for formatting the conversation. If not provided, the tokenizer's
                chat template is used.
        """
        return super().apply_chat_template(
            conversation, chat_template, **kwargs
        )



__all__ = ["AyaVisionProcessor"]