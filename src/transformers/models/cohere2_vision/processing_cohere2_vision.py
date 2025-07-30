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
from ...image_utils import ImageInput
from ...processing_utils import ImagesKwargs, MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput


class Cohere2VisionImagesKwargs(ImagesKwargs, total=False):
    max_patches: Optional[int]


class Cohere2VisionProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Cohere2VisionImagesKwargs
    _defaults = {
        "text_kwargs": {
            "padding_side": "left",
            "padding": True,
            "return_mm_token_type_ids": False,
        },
    }


class Cohere2VisionProcessor(ProcessorMixin):
    r"""
    Constructs a Cohere2Vision processor which wraps a [`AutoImageProcessor`] and
    [`PretrainedTokenizerFast`] tokenizer into a single processor that inherits both the image processor and
    tokenizer functionalities. See the [`~Cohere2VisionProcessor.__call__`] and [`~Cohere2VisionProcessor.decode`] for more information.
    Args:
        image_processor ([`AutoImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`PreTrainedTokenizer`, `PreTrainedTokenizerFast`], *optional*):
            The tokenizer is a required input.
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
        chat_template=None,
        **kwargs,
    ):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

        self.patch_size = self.image_processor.patch_size
        self.boi_token = tokenizer.boi_token
        self.eoi_token = tokenizer.eoi_token
        self.image_token = tokenizer.image_token
        self.img_line_break_token = tokenizer.img_line_break_token
        self.image_token_id = tokenizer.image_token_id

        self.image_ids = tokenizer.convert_tokens_to_ids(
            [
                self.image_token,
                self.boi_token,
                self.eoi_token,
                self.img_line_break_token,
            ]
        )

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Optional[Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]] = None,
        **kwargs: Unpack[Cohere2VisionProcessorKwargs],
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
        elif not isinstance(text, (list, tuple)):
            text = [text]

        output_kwargs = self._merge_kwargs(
            Cohere2VisionProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        # Process images
        image_inputs = {}
        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            batch_num_patches = iter(image_inputs.pop("num_patches"))
            processed_text = []
            for sample in text:
                while self.image_token in sample:
                    num_patches = next(batch_num_patches)
                    img_patches_per_tile = int(self.patch_size**2)

                    img_string = f"{self.boi_token}"
                    for idx in range(1, num_patches):
                        img_string += "<placeholder>" * img_patches_per_tile + self.img_line_break_token
                    img_string += "<placeholder>" * img_patches_per_tile + self.img_line_break_token
                    img_string += f"{self.eoi_token}"

                    sample = sample.replace(self.image_token, img_string, 1)
                processed_text.append(sample)
            text = [sample.replace("<placeholder>", self.image_token) for sample in processed_text]

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
            images_kwargs = Cohere2VisionProcessorKwargs._defaults.get("images_kwargs", {})
            images_kwargs.update(kwargs)

            num_image_patches = [
                self.image_processor.get_number_of_image_patches(*image_size, images_kwargs)
                for image_size in image_sizes
            ]

            token_per_patch = int(self.patch_size**2)
            num_image_tokens = [
                2 + sum(token_per_patch + 1 for _ in range(num_patches)) for num_patches in num_image_patches
            ]  # Add +2 and +1 for BOI/EOI and image break tokens
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


__all__ = ["Cohere2VisionProcessor"]
