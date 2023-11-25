# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
Processor class for CogVLM.
"""

from typing import List, Optional, Union

from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import TensorType


LANGUAGE_TOKEN_TYPE = 0
VISION_TOKEN_TYPE = 1


class CogVLMProcessor(ProcessorMixin):
    r"""
    Constructs a CogVLM processor which wraps a CLIP image processor and a LLaMa tokenizer into a single processor.

    [`CogVLMProcessor`] offers all the functionalities of [`CLIPImageProcessor`] and [`LlamaTokenizer`]. See the docstring
    of [`~CogVLMProcessor.__call__`] and [`~CogVLMProcessor.decode`] for more information.

    Args:
        image_processor (`CLIPImageProcessor`):
            An instance of [`CLIPImageProcessor`]. The image processor is a required input.
        tokenizer (`AutoTokenizer`):
            An instance of ['LlamaTokenizer`]. The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "CLIPImageProcessor"
    tokenizer_class = "LlamaTokenizer"

    def __init__(self, image_processor, tokenizer, image_size, patch_size):
        super().__init__(image_processor, tokenizer)
        self.image_size = image_size
        self.patch_size = patch_size

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        add_special_tokens: bool = False,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = True,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_token_type_ids: bool = True,
        return_length: bool = False,
        verbose: bool = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        This method uses [`CLIPImageProcessor.__call__`] method to prepare image(s) for the model, and
        [`BertTokenizerFast.__call__`] to prepare text for the model.

        Please refer to the docstring of the above two methods for more information.
        """

        input_ids = [self.tokenizer.bos_token_id]
        token_type_ids = [LANGUAGE_TOKEN_TYPE]
        pixel_values = None

        if images is not None:
            num_vision_tokens = (self.image_size // self.patch_size) * (self.image_size // self.patch_size) + 2
            input_ids += [self.tokenizer.pad_token_id] * num_vision_tokens
            token_type_ids += [VISION_TOKEN_TYPE] * num_vision_tokens
            pixel_values = self.image_processor(images, return_tensors=return_tensors).pixel_values

        if text is not None:
            text_encoding = self.tokenizer(
                text=text,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
                # TODO support the following 3 flags
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_token_type_ids=None,
                return_length=return_length,
                verbose=verbose,
                return_tensors=None,
                **kwargs,
            )
            text_ids = text_encoding.input_ids
            input_ids += text_ids
            token_type_ids += [LANGUAGE_TOKEN_TYPE] * len(text_ids)

        data = {}
        data["input_ids"] = [input_ids]
        if return_token_type_ids:
            data["token_type_ids"] = [token_type_ids]
        if return_attention_mask:
            attention_mask = [1] * len(input_ids)
            data["attention_mask"] = [attention_mask]

        result = BatchFeature(data=data, tensor_type=return_tensors)

        if pixel_values is not None:
            result["pixel_values"] = pixel_values

        return result

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
