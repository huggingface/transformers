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
    valid_processor_kwargs = Idefics3ProcessorKwargs

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
        images, text = self.prepare_inputs_layout(images=images, text=text, **kwargs)
        self.validate_inputs(images=images, text=text, **kwargs)

        output_kwargs = self._merge_kwargs(
            Idefics3ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        image_seq_len = image_seq_len if image_seq_len is not None else self.image_seq_len
        return_text_replacement_offsets = output_kwargs["text_kwargs"].pop("return_text_replacement_offsets", False)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)

        image_inputs = text_inputs = {}
        if images is not None:
            image_inputs, images_replacements = self._process_images(images, **output_kwargs["images_kwargs"])

            # Pop inputs unused by the model
            image_inputs.pop("rows", None)
            image_inputs.pop("cols", None)

            if text is not None:
                text, text_replacement_offsets = self.get_text_with_replacements(
                    text, images_replacements=images_replacements
                )
                text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
                if return_text_replacement_offsets:
                    text_inputs["text_replacement_offsets"] = text_replacement_offsets

                batch_image_seq_lengths = []
                for batch_id, text_replacement_offset in enumerate(text_replacement_offsets):
                    image_seq_lens = []
                    for data in text_replacement_offset:
                        start, end = data["new_span"]
                        start_id_pos = text_inputs.char_to_token(batch_id, start)
                        end_id_pos = text_inputs.char_to_token(batch_id, end - 1)
                        # Add one to go from zero-indexing to actual length
                        image_seq_lens.append(end_id_pos - start_id_pos + 1)
                    batch_image_seq_lengths.append(image_seq_lens)

                if return_mm_token_type_ids:
                    text_inputs["mm_token_type_ids"] = self.create_mm_token_type_ids(
                        text_inputs["input_ids"], batch_image_seq_lengths
                    )
                self._check_special_mm_tokens(text, text_inputs, modalities=["image"])

        elif text is not None:
            text_inputs = self.tokenizer(text=text, **output_kwargs["text_kwargs"])

        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)

    def prepare_inputs_layout(
        self,
        images: ImageInput | None = None,
        text: Union[TextInput, "PreTokenizedInput", list[TextInput], list["PreTokenizedInput"]] = None,
        **kwargs: Unpack[Idefics3ProcessorKwargs],
    ):
        if text is not None:
            if isinstance(text, str):
                text = [text]
            text = text.copy()

        if images is not None:
            images = self.image_processor.fetch_images(images)
            if is_valid_image(images):
                images = [[images]]
            elif isinstance(images, (list, tuple)) and is_valid_image(images[0]):
                if text is not None:
                    # Reorganize the images to match the prompts
                    n_images_in_text = [sample.count(self.image_token) for sample in text]
                    cumsum_images_in_text = [0] + list(accumulate(n_images_in_text))
                    split_images = [
                        images[cumsum_images_in_text[i] : cumsum_images_in_text[i + 1]]
                        for i in range(len(n_images_in_text))
                    ]
                    # Append the rest if any, we will error out when validating if they don't match with text
                    if len(images) > cumsum_images_in_text[-1]:
                        images = split_images + [images[cumsum_images_in_text[-1] :]]
                    else:
                        images = split_images
                else:
                    images = [images]

        return images, text

    def validate_inputs(
        self,
        images: ImageInput | None = None,
        text: Union[TextInput, "PreTokenizedInput", list[TextInput], list["PreTokenizedInput"]] = None,
        **kwargs: Unpack[ProcessingKwargs],
    ):
        super().validate_inputs(images, text, **kwargs)

        if text is None and images is None:
            raise ValueError("You must provide either `text` or `images`.")

        if text is not None:
            n_images_in_text = [sample.count(self.image_token) for sample in text]
            if images is not None:
                n_images_in_images = [len(sublist) for sublist in images]
                if n_images_in_text != n_images_in_images:
                    raise ValueError(
                        f"The total number of {self.image_token} tokens in the prompts should be the same as the number of images passed."
                        f" Found {n_images_in_text} {self.image_token} tokens and {n_images_in_images} images per sample."
                    )
            elif images is None and any(n_images_in_text):
                raise ValueError(
                    f"Found {sum(n_images_in_text)} {self.image_token} tokens in the text but no images were passed."
                )

    def replace_image_token(self, image_inputs: dict, image_idx: int) -> str:
        image_rows = [row for row_list in image_inputs["rows"] for row in row_list][image_idx]
        image_cols = [col for col_list in image_inputs["cols"] for col in col_list][image_idx]
        if image_rows == 0 and image_cols == 0:
            return (
                f"{self.fake_image_token}"
                + f"{self.global_image_tag}"
                + f"{self.image_token}" * self.image_seq_len
                + f"{self.fake_image_token}"
            )
        else:
            text_split_images = ""
            for n_h in range(image_rows):
                for n_w in range(image_cols):
                    text_split_images += (
                        f"{self.fake_image_token}"
                        + f"<row_{n_h + 1}_col_{n_w + 1}>"
                        + f"{self.image_token}" * self.image_seq_len
                    )
                text_split_images += "\n"

            text_split_images += (
                f"\n{self.fake_image_token}"
                + f"{self.global_image_tag}"
                + f"{self.image_token}" * self.image_seq_len
                + f"{self.fake_image_token}"
            )
            return text_split_images

    def create_mm_token_type_ids(self, input_ids: list, batch_image_seq_lengths: list[int]) -> list[list[int]]:
        # We have to iterate for each list separately because inputs
        # might be non-padded lists and we can't cast numpy on that!
        # Then cast numpy as each input for faster indexing
        mm_token_type_ids = []
        for i, seq_lengths in enumerate(batch_image_seq_lengths):
            array_ids = np.array(input_ids[i])
            mm_token_types = np.zeros_like(array_ids)
            image_start_positions = np.where(array_ids == self.fake_image_token_id)[0]
            j = 0
            for seq_len in seq_lengths:
                if j >= len(image_start_positions):
                    break
                start = image_start_positions[j]
                end = start + seq_len
                mm_token_types[start:end] = 1
                j = np.searchsorted(image_start_positions, end)
            mm_token_type_ids.append(mm_token_types.tolist())

        return mm_token_type_ids

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
