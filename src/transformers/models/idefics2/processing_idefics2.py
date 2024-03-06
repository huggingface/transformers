# coding=utf-8
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

from typing import List, Optional, Union

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, is_valid_image
from ...processing_utils import ProcessorMixin
from ...tokenization_utils import AddedToken
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, TextInput, TruncationStrategy
from ...utils import TensorType


def build_string_from_input(prompt, image_seq_len, bos_token, image_token, fake_image_token):
    """
    Builds a string from the input prompt and image tokens.

    For example, for the call:

    build_string_from_input(
        prompt=["Initial str", img1, img2, "mid str", img3],
        image_seq_len=2,
        bos_token="<s>",
        image_token="<im>",
        fake_image_token="<fake>"
    )

    The output will be:

    "<s>Initial str<fake><im><im><fake><im><im><fake>mid str<fake><im><im><fake>"

    Args:
        prompt (`List[Union[str, ImageInput]]`): The input prompt.
        image_seq_len (`int`): The length of the image sequence.
        bos_token (`str`): The beginning of sentence token.
        image_token (`str`): The image token.
        fake_image_token (`str`): The fake image token.
    """
    s = f"{bos_token}"
    open_image_tag = False
    for elem in prompt:
        if is_valid_image(elem):
            s += f"{fake_image_token}{image_token * image_seq_len}"
            open_image_tag = True
        else:
            if open_image_tag:
                s += f"{fake_image_token}"
                open_image_tag = False
            s += elem
    if open_image_tag:
        s += f"{fake_image_token}"
    return s



class Idefics2Processor(ProcessorMixin):
    r"""
    Constructs a IDEFICS2 processor which wraps a LLama tokenizer and IDEFICS2 image processor into a single processor.

    [`IdeficsProcessor`] offers all the functionalities of [`Idefics2ImageProcessor`] and [`LlamaTokenizerFast`]. See
    the docstring of [`~IdeficsProcessor.__call__`] and [`~IdeficsProcessor.decode`] for more information.

    Args:
        image_processor (`Idefics2ImageProcessor`):
            An instance of [`Idefics2ImageProcessor`]. The image processor is a required input.
        tokenizer (`LlamaTokenizerFast`):
            An instance of [`LlamaTokenizerFast`]. The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "Idefics2ImageProcessor"
    tokenizer_class = "LlamaTokenizerFast"

    def __init__(self, image_processor, tokenizer=None, image_seq_len: int = 64, **kwargs):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        self.fake_image_token = "<fake_token_around_image>"
        self.image_token = "<image>"
        self.image_seq_len = image_seq_len

        tokens_to_add = [
            AddedToken(self.fake_image_token, lstrip=True, rstrip=False, normalized=False),
            AddedToken(self.image_token, lstrip=True, rstrip=False, normalized=False)
        ]
        tokenizer.add_tokens(tokens_to_add)

        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        prompts: Union[List[TextInput], List[List[TextInput]]],
        image_seq_len: Optional[int] = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> BatchEncoding:
        """ """
        image_seq_len = image_seq_len if image_seq_len is not None else self.image_seq_len

        if isinstance(prompts, list) and not isinstance(prompts[0], list):
            prompts = [prompts]

        # Build the string from the input prompt and image tokens
        prompt_strings = [
            build_string_from_input(
                prompt=prompt,
                image_seq_len=image_seq_len,
                bos_token=self.tokenizer.bos_token,
                image_token=self.image_token,
                fake_image_token=self.fake_image_token
            ) for prompt in prompts
        ]

        inputs = BatchFeature()
        text_inputs = self.tokenizer(
            text=prompt_strings,
            add_special_tokens=False,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
        )
        inputs.update(text_inputs)

        # Extract the images from the prompts
        images = [
            [elem for elem in prompt if is_valid_image(elem)] for prompt in prompts
        ]
        image_inputs = self.image_processor(images, return_tensors=return_tensors)
        inputs.update(image_inputs)

        return inputs

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
