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
Processor class for IDEFICS2.
"""

import re
from itertools import accumulate
from typing import TYPE_CHECKING, Optional, Union

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, is_valid_image, load_image
from ...processing_utils import (
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
)
from ...tokenization_utils_base import AddedToken, TextInput
from ...utils import auto_docstring, logging


if TYPE_CHECKING:
    from ...tokenization_utils_base import PreTokenizedInput


logger = logging.get_logger(__name__)


def is_url(val) -> bool:
    return isinstance(val, str) and val.startswith("http")


def is_image_or_image_url(elem):
    return is_url(elem) or is_valid_image(elem)


class Idefics2ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "add_special_tokens": True,
            "padding": False,
            "is_split_into_words": False,
        },
    }


@auto_docstring
class Idefics2Processor(ProcessorMixin):
    def __init__(
        self, image_processor, tokenizer=None, image_seq_len: int = 64, chat_template: Optional[str] = None, **kwargs
    ):
        r"""
        image_seq_len (`int`, *optional*, defaults to 64):
            The length of the image sequence i.e. the number of <image> tokens per image in the input.
            This parameter is used to build the string from the input prompt and image tokens and should match the
            config.perceiver_config.resampler_n_latents value for the model used.
        """
        if not hasattr(tokenizer, "image_token"):
            self.fake_image_token = AddedToken("<fake_token_around_image>", normalized=False, special=True).content
            self.image_token = AddedToken("<image>", normalized=False, special=True).content
            tokens_to_add = {"additional_special_tokens": [self.fake_image_token, self.image_token]}
            tokenizer.add_special_tokens(tokens_to_add)
            self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        else:
            self.fake_image_token = tokenizer.image_boundary_token
            self.image_token = tokenizer.image_token
            self.image_token_id = tokenizer.image_token_id

        self.end_of_utterance_token = AddedToken("<end_of_utterance>", normalized=False, special=True)
        tokenizer.add_special_tokens({"additional_special_tokens": [self.end_of_utterance_token]})
        self.image_seq_len = image_seq_len

        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def _extract_images_from_prompts(self, prompts):
        prompt_images = []
        for prompt in prompts:
            images = []
            for elem in prompt:
                if is_valid_image(elem):
                    images.append(elem)
                elif is_url(elem):
                    images.append(load_image(elem))
            prompt_images.append(images)
        return prompt_images

    @auto_docstring
    def __call__(
        self,
        images: Union[ImageInput, list[ImageInput], list[list[ImageInput]]] = None,
        text: Union[TextInput, "PreTokenizedInput", list[TextInput], list["PreTokenizedInput"]] = None,
        **kwargs: Unpack[Idefics2ProcessorKwargs],
    ) -> BatchFeature:
        if text is None and images is None:
            raise ValueError("You must provide either `text` or `images`.")

        output_kwargs = self._merge_kwargs(
            Idefics2ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)

        n_images_in_text = []
        inputs = {}

        if text is not None:
            if isinstance(text, str):
                text = [text]
            elif not isinstance(text, list) and not isinstance(text[0], str):
                raise ValueError("Invalid input text. Please provide a string, or a list of strings")

            # Replace the image token with fake tokens around the expanded image token sequence of length `image_seq_len`
            fake_image_token = self.fake_image_token
            image_token = self.image_token
            image_str = f"{fake_image_token}{image_token * self.image_seq_len}{fake_image_token}"

            if self.image_processor.do_image_splitting:
                # A single image token is split into 4 patches + 1 original image
                image_str = image_str * 5

            prompt_strings = []
            closing_fake_pattern = re.compile(rf"{re.escape(fake_image_token)}(?=[^\s<])")
            for sample in text:
                n_images_in_text.append(sample.count(image_token))
                sample = sample.replace(image_token, image_str)
                # Remove any double fake tokens if images are adjacent
                sample = sample.replace(f"{fake_image_token}{fake_image_token}", f"{fake_image_token}")
                # Ensure words attached directly after the closing fake token remain word-boundary aligned
                sample = closing_fake_pattern.sub(f"{fake_image_token} ", sample)
                prompt_strings.append(sample)

            text_inputs = self.tokenizer(prompt_strings, **output_kwargs["text_kwargs"])
            self._check_special_mm_tokens(prompt_strings, text_inputs, modalities=["image"])
            inputs.update(text_inputs)

        if images is not None:
            if is_image_or_image_url(images):
                images = [[images]]
            elif isinstance(images, (list, tuple)) and is_image_or_image_url(images[0]):
                if text is not None:
                    if sum(n_images_in_text) != len(images):
                        raise ValueError(
                            f"The total number of {image_token} tokens in the prompts should be the same as the number of images passed."
                            f" Found {sum(n_images_in_text)} {image_token} tokens and {len(images)} images."
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
            if text is not None and not n_images_in_images == n_images_in_text:
                raise ValueError(
                    f"The number of images in the text {n_images_in_text} and images  {n_images_in_images} should be the same."
                )

            # Load images if they are URLs
            images = [[load_image(im) for im in sample] for sample in images]
            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
            inputs.update(image_inputs)

        return BatchFeature(inputs, tensor_type=return_tensors)


__all__ = ["Idefics2Processor"]
