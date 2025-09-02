# coding=utf-8
# Copyright 2023 the HuggingFace Inc. team. All rights reserved.
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

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, make_nested_list_of_images
from ...processing_utils import (
    ImagesKwargs,
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
)
from ...tokenization_utils_base import BatchEncoding, TextInput
from ...utils import logging


logger = logging.get_logger(__name__)


class Lfm2VlImagesKwargs(ImagesKwargs, total=False):
    downsample_factor: int
    do_image_splitting: bool
    min_tiles: int
    max_tiles: int
    use_thumbnail: bool
    min_image_tokens: int
    max_image_tokens: int
    encoder_patch_size: int
    tile_size: int
    max_pixels_tolerance: float


class Lfm2VlProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Lfm2VlImagesKwargs

    _defaults = {
        "text_kwargs": {
            "add_special_tokens": False,
            "padding": False,
            "is_split_into_words": False,
        },
    }


class Lfm2VlProcessor(ProcessorMixin):
    r"""
    Constructs a Lfm2Vl processor which wraps a Lfm2Tokenizer tokenizer and Lfm2VlImageProcessor into a single processor.

    [`Lfm2VlProcessor`] offers all the functionalities of [`Lfm2ImageProcessor`] and [`Lfm2Tokenizer`].

    Args:
            image_processor (`Lfm2VlImageProcessor`):
                An instance of [`Lfm2VlImageProcessor`]. The image processor is a required input.
            tokenizer (`PreTrainedTokenizerBase`):
                An instance of [`PreTrainedTokenizerBase`]. This should correspond with the model's text model. The tokenizer is a required input.
            chat_template (`str`): <fill_docstring>
            use_image_special_tokens (`bool`): <fill_docstring>
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "Lfm2VlImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor,
        tokenizer,
        chat_template: str,
        use_image_special_tokens: bool,
        **kwargs,
    ):
        self.image_token = getattr(tokenizer, "image_token", "<image>")
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.use_image_special_tokens = use_image_special_tokens
        self.image_start_token = getattr(tokenizer, "image_start_token", "<|image_start|>")
        self.image_end_token = getattr(tokenizer, "image_end_token", "<|image_end|>")
        self.image_thumbnail_token = getattr(tokenizer, "image_thumbnail", "<|img_thumbnail|>")
        super().__init__(image_processor, tokenizer, chat_template=chat_template, **kwargs)

    def process_vision(
        self,
        text: list[str],
        images: list[list[ImageInput]],
        use_image_special_tokens: bool,
        output_kwargs: dict,
    ):
        if text is not None:
            n_images_in_text = [sample.count(self.image_token) for sample in text]

        n_images_in_images = [len(sublist) for sublist in images]

        if n_images_in_images != n_images_in_text:
            raise ValueError(
                f"The number of images in the text {n_images_in_text} and images {n_images_in_images} should be the same."
            )

        prompt_strings = []

        for sample_text, sample_images in zip(text, images, strict=False):
            split_sample = sample_text.split(self.image_token)
            sample_text_with_image_tokens = ""
            for i, image in enumerate(sample_images):
                sample_text_with_image_tokens += split_sample[i]
                if use_image_special_tokens:
                    sample_text_with_image_tokens += self.image_start_token
                (
                    num_tokens_per_tile,
                    num_rows,
                    num_cols,
                    num_thumbnail_tokens,
                ) = self.image_processor.get_tile_grid_and_sizes(
                    image,
                    output_kwargs["images_kwargs"],
                )
                if num_rows > 1 or num_cols > 1:
                    for row in range(num_rows):
                        for col in range(num_cols):
                            if use_image_special_tokens:
                                sample_text_with_image_tokens += f"<|img_row_{row + 1}_col_{col + 1}|>"
                            sample_text_with_image_tokens += self.image_token * num_tokens_per_tile

                    if num_thumbnail_tokens > 0:
                        if use_image_special_tokens:
                            sample_text_with_image_tokens += self.image_thumbnail_token
                        sample_text_with_image_tokens += self.image_token * num_thumbnail_tokens
                else:
                    sample_text_with_image_tokens += self.image_token * num_tokens_per_tile

                if use_image_special_tokens:
                    sample_text_with_image_tokens += self.image_end_token

                sample_text_with_image_tokens += split_sample[i + 1]

            prompt_strings.append(sample_text_with_image_tokens)

        return prompt_strings

    def __call__(
        self,
        images: Optional[Union[ImageInput, list[ImageInput], list[list[ImageInput]]]] = None,
        text: Optional[Union[TextInput, list[TextInput]]] = None,
        use_image_special_tokens: bool = True,
        **kwargs: Unpack[Lfm2VlProcessorKwargs],
    ) -> BatchEncoding:
        """
        Processes the input prompts and returns a BatchFeature.

        Example:

        ```python
        >>> import requests
        >>> from transformers import AutoProcessor
        >>> from transformers.image_utils import load_image
        >>> processor = AutoProcessor.from_pretrained("", trust_remote_code=True)

        >>> url1 = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        >>> url2 = "https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg"

        >>> image1, image2 = load_image(url1), load_image(url2)
        >>> images = [image1, image2]

        >>> conversation = [
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {"type": "image", "url": image1},
        ...             {"type": "image", "url": image2},
        ...             {"type": "text", "text": "Compare the two images."},
        ...         ],
        ...     },
        ... ]
        >>> chat_inputs = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        >>> outputs = processor(images=images, text=chat_inputs, return_tensors="pt")
        >>> input_ids = outputs.input_ids
        >>> input_tokens = processor.tokenizer.batch_decode(input_ids)
        >>> print(input_tokens)
        '['user\nCompare the two images.\nassistant\n']'
        ```

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `list[PIL.Image.Image]`, `list[np.ndarray]`, `list[torch.Tensor]`, *optional*):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. If is of type `list[ImageInput]`, it's assumed that this is for a single prompt i.e. of batch size 1.
            text (`TextInput`, *optional*):
                The sequence or batch of sequences to be encoded.
                Wherever an image token, `<image>` is encountered it is expanded to a proper sequence of image tokens.
            return_tensors (`Optional[str, TensorType]`, *optional*):
                If set, will return tensors of a particular framework. See [`PreTrainedTokenizerFast.__call__`] for more
                information.
        """
        if text is None and images is None:
            raise ValueError("You must provide one of `text` or `images`.")

        if images is not None and text is None:
            raise ValueError(
                "You must provide `text` when `images` is provided. Minimal text consists of a single image token."
            )

        output_kwargs = self._merge_kwargs(
            Lfm2VlProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if text is not None:
            if isinstance(text, str):
                text = [text]
            elif not isinstance(text, list) and not isinstance(text[0], str):
                raise ValueError("Invalid input text. Please provide a string, or a list of strings")
            n_images_in_text = sum([sample.count(self.image_token) for sample in text])
            if n_images_in_text > 0 and (images is None):
                raise ValueError(f"We detected {n_images_in_text} tokens in the text but no images were passed")

        inputs = {}

        use_image_special_tokens = (
            self.use_image_special_tokens if use_image_special_tokens is None else use_image_special_tokens
        )

        if images is not None:
            images = make_nested_list_of_images(images)
            vision_inputs = self.image_processor(
                images,
                **output_kwargs["images_kwargs"],
            )
            inputs.update(vision_inputs)

            text = self.process_vision(text, images, use_image_special_tokens, output_kwargs)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)

        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, text_inputs, modalities=["image"])
        inputs.update(text_inputs)

        return BatchFeature(inputs, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LFM2Tokeniser's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        batched_decode_output = self.tokenizer.batch_decode(*args, **kwargs)
        return batched_decode_output

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LFM2Tokeniser's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        decode_output = self.tokenizer.decode(*args, **kwargs)
        return decode_output

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(image_processor_input_names + tokenizer_input_names))


__all__ = ["Lfm2VlProcessor"]
