# Copyright 2024 The HuggingFace Inc. team.
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
"""
Processor class for Idefics3.
"""

import re
from itertools import accumulate
from typing import TYPE_CHECKING, Union

import numpy as np

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, is_valid_image
from ...processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import AddedToken, BatchEncoding, TextInput
from ...utils import auto_docstring, logging


if TYPE_CHECKING:
    from ...tokenization_utils_base import PreTokenizedInput

logger = logging.get_logger(__name__)


def is_url(val) -> bool:
    return isinstance(val, str) and val.startswith("http")


def is_image_or_image_url(elem):
    return is_url(elem) or is_valid_image(elem)


class Idefics3ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "add_special_tokens": True,
            "padding": False,
            "is_split_into_words": False,
            "return_mm_token_type_ids": False,
        },
        "images_kwargs": {
            "return_row_col_info": True,
        },
    }


@auto_docstring
class Idefics3Processor(ProcessorMixin):
    def __init__(
        self, image_processor, tokenizer=None, image_seq_len: int = 169, chat_template: str | None = None, **kwargs
    ):
        r"""
        image_seq_len (`int`, *optional*, defaults to 169):
            The length of the image sequence i.e. the number of <image> tokens per image in the input.
            This parameter is used to build the string from the input prompt and image tokens and should match the
            value the model used. It is computed as: image_seq_len = int(((image_size // patch_size) ** 2) / (scale_factor**2))
        """
        self.fake_image_token = AddedToken("<fake_token_around_image>", normalized=False, special=True).content
        self.image_token = AddedToken("<image>", normalized=False, special=True).content
        self.end_of_utterance_token = AddedToken("<end_of_utterance>", normalized=False, special=True).content
        self.global_image_tag = "<global-img>"  # https://github.com/huggingface/transformers/pull/32473/files/8063e5e17362571b693f1db95167f5443a3be1b2#r1734825341
        self.image_seq_len = image_seq_len
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.fake_image_token_id = tokenizer.convert_tokens_to_ids(self.fake_image_token)
        self.global_image_token_id = tokenizer.convert_tokens_to_ids(self.global_image_tag)
        self.row_col_ids = [
            tokenizer.convert_tokens_to_ids(f"<row_{i + 1}_col_{j + 1}>") for i in range(6) for j in range(6)
        ]

        # This regex matches one or more occurrences of <global-img> tags (optionally surrounded by newline characters)
        # or <row_x_col_y> tags (where x and y are digits, also optionally surrounded by newline characters).
        self._regex_to_remove_extra_special_tokens = re.compile(r"(\n?<global-img>\n?|<row_\d+_col_\d+>\n?)+")

        tokens_to_add = {
            "additional_special_tokens": [
                self.fake_image_token,
                self.image_token,
                self.end_of_utterance_token,
            ]
        }
        tokenizer.add_special_tokens(tokens_to_add)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)

        super().__init__(image_processor, tokenizer, chat_template=chat_template, **kwargs)

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | list[ImageInput] | list[list[ImageInput]] = None,
        text: Union[TextInput, "PreTokenizedInput", list[TextInput], list["PreTokenizedInput"]] = None,
        image_seq_len: int | None = None,
        **kwargs: Unpack[Idefics3ProcessorKwargs],
    ) -> BatchEncoding:
        r"""
        image_seq_len (`int`, *optional*):
            The length of the image sequence. If not provided, the default value of self.image_seq_len is used.
            image_seq_len should be equal to int(((image_size // patch_size) ** 2) / (scale_factor**2))
        """
        if text is None and images is None:
            raise ValueError("You must provide either `text` or `images`.")

        output_kwargs = self._merge_kwargs(
            Idefics3ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)

        n_images_in_text = []
        n_images_in_images = []
        inputs = {}

        if text is not None:
            if isinstance(text, str):
                text = [text]
            elif not isinstance(text, list) and not isinstance(text[0], str):
                raise ValueError("Invalid input text. Please provide a string, or a list of strings")
            n_images_in_text = [sample.count(self.image_token) for sample in text]

        if images is not None:
            if is_image_or_image_url(images):
                images = [[images]]
            elif isinstance(images, (list, tuple)) and is_image_or_image_url(images[0]):
                if text is not None:
                    if sum(n_images_in_text) != len(images):
                        raise ValueError(
                            f"The total number of {self.image_token} tokens in the prompts should be the same as the number of images passed."
                            f" Found {sum(n_images_in_text)} {self.image_token} tokens and {len(images)} images."
                        )
                    # Reorganize the images to match the prompts
                    cumsum_images_in_text = [0] + list(accumulate(n_images_in_text))
                    images = [
                        images[cumsum_images_in_text[i] : cumsum_images_in_text[i + 1]]
                        for i in range(len(n_images_in_text))
                    ]
                else:
                    images = [images]
            elif (
                not isinstance(images, (list, tuple))
                and not isinstance(images[0], (list, tuple))
                and not is_image_or_image_url(images[0][0])
            ):
                raise ValueError(
                    "Invalid input images. Please provide a single image or a list of images or a list of list of images."
                )
            n_images_in_images = [len(sample) for sample in images]

            # Load images if they are URLs
            images = self.image_processor.fetch_images(images)

            output_kwargs["images_kwargs"]["return_row_col_info"] = True
            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
            inputs.update(image_inputs)

            if text is not None:
                if n_images_in_images != n_images_in_text:
                    raise ValueError(
                        f"The number of images in the text {n_images_in_text} and images {n_images_in_images} should be the same."
                    )

                text, text_replacement_offsets = self.get_text_replacement(text, image_inputs=image_inputs)
                text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
                self._check_special_mm_tokens(text, text_inputs, modalities=["image"])
                inputs.update(text_inputs)

        elif text is not None:
            if any(n_images_in_text):
                raise ValueError(
                    f"Found {sum(n_images_in_text)} {self.image_token} tokens in the text but no images were passed."
                )
            text_inputs = self.tokenizer(text=text, **output_kwargs["text_kwargs"])
            inputs.update(text_inputs)

        # FIXME: `batch_image_seq_lengths` is lost
        batch_image_seq_lengths = []
        if return_mm_token_type_ids:
            array_ids = np.array(inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(array_ids)
            for i, seq_lengths in enumerate(batch_image_seq_lengths):
                image_start_positions = np.where(array_ids[i] == self.fake_image_token_id)[0]
                j = 0
                for seq_len in seq_lengths:
                    if j >= len(image_start_positions):
                        break
                    start = image_start_positions[j]
                    end = start + seq_len
                    mm_token_type_ids[i, start:end] = 1
                    j = np.searchsorted(image_start_positions, end)

            inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(data=inputs, tensor_type=return_tensors)

    def replace_image_token(self, text: str, image_inputs: dict, batch_idx: int, image_index: int) -> str:
        image_rows = image_inputs["rows"][batch_idx][image_index]
        image_cols = image_inputs["cols"][batch_idx][image_index]
        if image_rows == 0 and image_cols == 0:
            return (
                f"{self.fake_token_around_image}"
                + f"{self.global_img_token}"
                + f"{self.image_token}" * self.image_seq_len
                + f"{self.fake_token_around_image}"
            )
        else:
            text_split_images = ""
            for n_h in range(image_rows):
                for n_w in range(image_cols):
                    text_split_images += (
                        f"{self.fake_token_around_image}"
                        + f"<row_{n_h + 1}_col_{n_w + 1}>"
                        + f"{self.image_token}" * self.image_seq_len
                    )
                text_split_images += "\n"

            text_split_images += (
                f"\n{self.fake_token_around_image}"
                + f"{self.global_img_token}"
                + f"{self.image_token}" * self.image_seq_len
                + f"{self.fake_token_around_image}"
            )
            return text_split_images

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
            images_kwargs = Idefics3ProcessorKwargs._defaults.get("images_kwargs", {})
            images_kwargs.update(kwargs)

            num_image_row_cols = [
                self.image_processor.get_number_of_image_patches(*image_size, images_kwargs)
                for image_size in image_sizes
            ]

            base_image_length = self.image_seq_len + 3
            col_length = self.image_seq_len + 2
            num_image_tokens = []
            num_image_patches = []

            for num_patches, num_rows, num_cols in num_image_row_cols:
                row_length = col_length * num_cols + 1
                num_image_tokens.append(base_image_length + (row_length * num_rows))
                num_image_patches.append(num_patches)

            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})

        return MultiModalData(**vision_data)


__all__ = ["Idefics3Processor"]
