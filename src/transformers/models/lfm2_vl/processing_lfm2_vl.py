# coding=utf-8
# Copyright 2025 the HuggingFace Inc. team. All rights reserved.
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
import math
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
    downsample_factor: Optional[int]
    do_image_splitting: Optional[bool]
    min_tiles: Optional[int]
    max_tiles: Optional[int]
    use_thumbnail: Optional[bool]
    min_image_tokens: Optional[int]
    max_image_tokens: Optional[int]
    encoder_patch_size: Optional[int]
    tile_size: Optional[int]
    max_pixels_tolerance: Optional[float]
    patch_size: Optional[int]
    do_pad: Optional[bool]
    return_row_col_info: Optional[bool]


class Lfm2VlProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Lfm2VlImagesKwargs

    _defaults = {
        "images_kwargs": {
            "return_row_col_info": True,
        },
        "text_kwargs": {
            "use_image_special_tokens": True,
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
        chat_template (`str`, *optional*):
            A Jinja template which will be used to convert lists of messages in a chat into a tokenizable string.
        use_image_special_tokens (`bool`, *optional*, defaults to `True`):
            Whether to use image special tokens or not when processing.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "Lfm2VlImageProcessorFast"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor,
        tokenizer,
        chat_template: Optional[str] = None,
        use_image_special_tokens: Optional[bool] = True,
        **kwargs,
    ):
        self.image_token = tokenizer.image_token
        self.image_token_id = tokenizer.image_token_id
        self.use_image_special_tokens = use_image_special_tokens
        self.image_start_token = tokenizer.image_start_token
        self.image_end_token = tokenizer.image_end_token
        self.image_thumbnail_token = tokenizer.image_thumbnail
        super().__init__(image_processor, tokenizer, chat_template=chat_template, **kwargs)

    def __call__(
        self,
        images: Optional[Union[ImageInput, list[ImageInput], list[list[ImageInput]]]] = None,
        text: Optional[Union[TextInput, list[TextInput]]] = None,
        **kwargs: Unpack[Lfm2VlProcessorKwargs],
    ) -> BatchEncoding:
        """
        Processes the input prompts and returns a BatchFeature.
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

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        n_images_in_text = [sample.count(self.image_token) for sample in text]
        if sum(n_images_in_text) > 0 and images is None:
            raise ValueError(f"We detected {sum(n_images_in_text)} tokens in the text but no images were passed")

        inputs = {}
        use_image_special_tokens = output_kwargs["text_kwargs"].pop("use_image_special_tokens")

        if images is not None:
            images = self.image_processor.fetch_images(images)
            batched_images = make_nested_list_of_images(images)
            vision_inputs = self.image_processor(batched_images, **output_kwargs["images_kwargs"])

            n_images_in_images = [len(sublist) for sublist in batched_images]
            if n_images_in_images != n_images_in_text:
                raise ValueError(
                    f"The number of images in the text {n_images_in_text} and images {n_images_in_images} should be the same."
                )

            text = self.expand_text_with_placeholders(
                text,
                batched_images,
                image_rows=vision_inputs.pop("image_rows"),
                image_cols=vision_inputs.pop("image_cols"),
                image_sizes=vision_inputs.pop("image_sizes"),
                use_image_special_tokens=use_image_special_tokens,
                **output_kwargs["images_kwargs"],
            )
            inputs.update(vision_inputs)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)

        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        inputs.update(text_inputs)

        return BatchFeature(inputs, tensor_type=return_tensors)

    def expand_text_with_placeholders(
        self,
        text: list[str],
        images: list[list[ImageInput]],
        image_rows: list[list[int]],
        image_cols: list[list[int]],
        image_sizes: list[list[int]],
        use_image_special_tokens: bool,
        **images_kwargs,
    ):
        prompt_strings = []

        image_data = iter(zip(*[image_rows, image_cols, image_sizes]))
        for sample_text, sample_images in zip(text, images):
            split_sample = sample_text.split(self.image_token)
            sample_text_with_image_tokens = ""
            for i, image in enumerate(sample_images):
                sample_text_with_image_tokens += split_sample[i]
                if use_image_special_tokens:
                    sample_text_with_image_tokens += self.image_start_token

                rows, cols, image_size = next(image_data)
                num_thumbnail_tokens, num_tokens_per_tile = self._get_image_num_tokens(image_size, **images_kwargs)

                if rows > 1 or cols > 1:
                    for row in range(rows):
                        for col in range(cols):
                            if use_image_special_tokens:
                                sample_text_with_image_tokens += f"<|img_row_{row + 1}_col_{col + 1}|>"
                            sample_text_with_image_tokens += self.image_token * num_tokens_per_tile

                    if num_thumbnail_tokens > 0:
                        if use_image_special_tokens:
                            sample_text_with_image_tokens += self.image_thumbnail_token
                        sample_text_with_image_tokens += self.image_token * num_thumbnail_tokens
                else:
                    sample_text_with_image_tokens += self.image_token * num_thumbnail_tokens

                if use_image_special_tokens:
                    sample_text_with_image_tokens += self.image_end_token

                sample_text_with_image_tokens += split_sample[i + 1]
            prompt_strings.append(sample_text_with_image_tokens)

        return prompt_strings

    def _get_image_num_tokens(self, image_size: list[int], **images_kwargs) -> tuple[int, int]:
        tile_size = images_kwargs.get("tile_size", self.image_processor.tile_size)
        downsample_factor = images_kwargs.get("downsample_factor", self.image_processor.downsample_factor)
        encoder_patch_size = images_kwargs.get("encoder_patch_size", self.image_processor.encoder_patch_size)
        use_thumbnail = images_kwargs.get("use_thumbnail", self.image_processor.use_thumbnail)

        thumbnail_tokens = 0
        if use_thumbnail:
            image_height, image_width = image_size
            num_patches_height = image_height // encoder_patch_size
            num_patches_width = image_width // encoder_patch_size
            dwn_num_patches_height = math.ceil(num_patches_height / downsample_factor)
            dwn_num_patches_width = math.ceil(num_patches_width / downsample_factor)
            thumbnail_tokens = dwn_num_patches_height * dwn_num_patches_width

        num_patches_tile = tile_size // encoder_patch_size
        dwn_num_patches_tile = math.ceil(num_patches_tile / downsample_factor)
        tile_tokens = dwn_num_patches_tile * dwn_num_patches_tile

        return thumbnail_tokens, tile_tokens

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

        # LFM2-VL has no dedicated tokenizer class and uses the Base class with default model input names
        tokenizer_input_names = [name for name in tokenizer_input_names if name != "token_type_ids"]
        return list(tokenizer_input_names + image_processor_input_names)


__all__ = ["Lfm2VlProcessor"]
