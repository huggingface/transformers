# coding=utf-8
# Copyright 2024 HuggingFace Inc. team. All rights reserved.
#
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
from ...processing_utils import ImagesKwargs, MultiModalData, ProcessingKwargs, ProcessorMixin, TextKwargs, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import is_vision_available


if is_vision_available():
    from .image_processing_emu3 import smart_resize


class Emu3TextKwargs(TextKwargs, total=False):
    return_for_image_generation: bool


class Emu3ImagesKwargs(ImagesKwargs, total=False):
    ratio: str
    image_area: int


class Emu3ProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: Emu3TextKwargs
    images_kwargs: Emu3ImagesKwargs
    _defaults = {
        "text_kwargs": {
            "return_for_image_generation": False,
            "return_mm_token_type_ids": False,
        },
        "images_kwargs": {
            "ratio": "1:1",
            "image_area": 518400,
        },
    }


class Emu3Processor(ProcessorMixin):
    r"""
    Constructs a Emu3 processor which wraps a Emu3 image processor and a GPT2 tokenizer into a single
    processor.

    [`Emu3Processor`] offers all the functionalities of [`Emu3ImageProcessor`] and [`GPT2TokenizerFast`].
    See the [`~Emu3Processor.__call__`] and [`~Emu3Processor.decode`] for more information.

    Args:
        image_processor ([`Emu3ImageProcessor`]):
            The image processor is a required input.
        tokenizer ([`Emu3TokenizerFast`]):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ["image_processor", "tokenizer"]
    tokenizer_class = ("GPT2Tokenizer", "GPT2TokenizerFast")
    image_processor_class = "Emu3ImageProcessor"

    def __init__(
        self,
        image_processor,
        tokenizer,
        chat_template=None,
        **kwargs,
    ):
        self.image_token = tokenizer.image_token  # image_token as placeholder to be replaced by vq-vae tokens
        self.image_token_id = tokenizer.image_token_id
        self.image_start_token = tokenizer.boi_token  # "<|image start|>" fixed tokens for start and end of image
        self.image_end_token = tokenizer.eoi_token  # "<|image end|>"
        self.fake_token_around_image = tokenizer.image_wrapper_token  # "<|image token|>"  every image starts with it
        self.eof_token = tokenizer.eof_token  # "<|extra_201|>"
        self.bos_token = tokenizer.bos_token
        self.downsample_ratio = 8
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Optional[Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]] = None,
        audio=None,
        videos=None,
        **kwargs: Unpack[Emu3ProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to Emu3TokenizerFast's [`~Emu3TokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwargs` arguments to
        CLIPImageProcessor's [`~CLIPImageProcessor.__call__`] if `images` is not `None`. Please refer to the docstring
        of the above two methods for more information.

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
        # check if images and text inputs are reversed for BC

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise TypeError("Invalid input text. Please provide a string, or a list of strings")

        output_kwargs = self._merge_kwargs(
            Emu3ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        return_for_image_generation = output_kwargs["text_kwargs"].pop("return_for_image_generation", False)
        ratio = output_kwargs["images_kwargs"].pop("ratio", None)
        image_area = output_kwargs["images_kwargs"].pop("image_area", None)

        if return_for_image_generation and images is not None:
            raise ValueError("You should not provide `images` when `return_for_image_generation=True`")

        if not return_for_image_generation and text is None and images is None:
            raise ValueError("You must provide either text or images when `return_for_image_generation=False`")

        image_features = {}
        image_start_tokens = f"{self.image_start_token}"
        image_end_tokens = f"{self.eof_token}{self.image_end_token}"

        # generate text from image + text input, so we add placeholders for image tokens
        if not return_for_image_generation and images is not None:
            image_features = self.image_processor(images, **output_kwargs["images_kwargs"])
            image_sizes = iter(image_features.image_sizes)

            prompt_strings = []
            for sample in text:
                while self.image_token in sample:
                    image_size = next(image_sizes)
                    height, width = image_size
                    height = height // self.downsample_ratio
                    width = width // self.downsample_ratio
                    image_seq_length = height * (width + 1)  # +1 for extra row when converting to BPE in modeling code

                    image_placeholder = f"{image_start_tokens}{height}*{width}{self.fake_token_around_image}{'<placeholder>' * image_seq_length}{image_end_tokens}"
                    sample = sample.replace(self.image_token, image_placeholder, 1)
                    sample = f"{self.bos_token}{sample}"  # add BOS because GPT tokenizer doesn't add it
                prompt_strings.append(sample)
            text = [sample.replace("<placeholder>", self.image_token) for sample in prompt_strings]

        # generate image from text input, so we add begin-of-image tokens from where image generation starts
        elif return_for_image_generation:
            height, width = self.calculate_generate_size(ratio, image_area, self.downsample_ratio)
            image_prompt = f"{image_start_tokens}{height}*{width}{self.fake_token_around_image}"
            text = [f"{self.bos_token}{sample}{image_prompt}" for sample in text]
            image_features["image_sizes"] = [[height, width]] * len(text)

        # else just generate from text-only input, and we do no special treatment for text
        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"], return_tensors=None)
        self._check_special_mm_tokens(text, text_inputs, modalities=["image"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(data={**text_inputs, **image_features}, tensor_type=return_tensors)

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
            num_image_tokens = []
            for height, width in image_sizes:
                height, width = smart_resize(
                    height,
                    width,
                    self.image_processor.spatial_factor,
                    self.image_processor.min_pixels,
                    self.image_processor.max_pixels,
                )
                height = height // self.downsample_ratio
                width = width // self.downsample_ratio
                image_seq_length = height * (width + 1)  # +1 for extra row when converting to BPE in modeling code
                num_image_tokens.append(image_seq_length)

            num_image_patches = [1] * len(image_sizes)
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})

        return MultiModalData(**vision_data)

    def calculate_generate_size(self, ratio, image_area, spatial_factor):
        width, height = map(int, ratio.split(":"))
        current_area = width * height
        target_ratio = (image_area / current_area) ** 0.5

        token_height = int(round(height * target_ratio / spatial_factor))
        token_width = int(round(width * target_ratio / spatial_factor))
        return token_height, token_width

    def postprocess(self, images: ImageInput, **kwargs):
        return self.image_processor.postprocess(images, **kwargs)


__all__ = ["Emu3Processor"]
