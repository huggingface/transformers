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
Processor class for PaliGemma.
"""

from typing import List, Optional, Union

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, is_valid_image, make_flat_list_of_images
from ...processing_utils import (
    ImagesKwargs,
    ProcessingKwargs,
    ProcessorMixin,
    TextKwargs,
    Unpack,
    _validate_images_text_input_order,
)
from ...tokenization_utils_base import (
    AddedToken,
    PreTokenizedInput,
    TextInput,
)
from ...utils import logging


logger = logging.get_logger(__name__)

IMAGE_TOKEN = "<image_placeholder>" #576 image placeholder tokens

# Copied from transformers.models.idefics2.processing_idefics2.is_url
def is_url(val) -> bool:
    return isinstance(val, str) and val.startswith("http")


# Copied from transformers.models.idefics2.processing_idefics2.is_image_or_image_url
def is_image_or_image_url(elem):
    return is_url(elem) or is_valid_image(elem)


def _is_str_or_image(elem):
    return isinstance(elem, (str)) or is_image_or_image_url(elem)

class JanusProcessor(ProcessorMixin):
    r"""
    Constructs a PaliGemma processor which wraps a PaliGemma image processor and a PaliGemma tokenizer into a single processor.

    [`JanusProcessor`] offers all the functionalities of [`SiglipImageProcessor`] and [`LlamaTokenizerFast`]. See the
    [`~JanusProcessor.__call__`] and [`~JanusProcessor.decode`] for more information.

    Args:
        image_processor ([`SiglipImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`], *optional*):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ['image_processor','tokenizer']
    valid_kwargs = ['chat_template']
    image_processing_class = ["SiglipImageProcessor"]
    tokenizer_class = ["LLamaTokenizer","LlamaTokenizerFast"]

    def __init__(self, image_processor, tokenizer, chat_template, **kwargs):

        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")
        if not hasattr(image_processor, "image_seq_length"):
            raise ValueError("Image processor is missing an `image_seq_length` attribute.")

        self.image_seq_length = image_processor.image_seq_length

        if not hasattr(tokenizer, "image_token"):
            image_token = AddedToken(IMAGE_TOKEN, normalized=False, special=True)
            tokens_to_add = {"additional_special_tokens": [image_token]}
            tokenizer.add_special_tokens(tokens_to_add)
            self.image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        else:
            self.image_token_id = tokenizer.image_token_id

        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        super().__init__(image_processor, tokenizer, chat_template=chat_template)