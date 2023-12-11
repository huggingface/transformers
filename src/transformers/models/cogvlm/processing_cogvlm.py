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

    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        # TODO support all these kwargs?
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

        if images is not None:
            pixel_values = self.image_processor(images, return_tensors=return_tensors)["pixel_values"]
        else:
            pixel_values = None
        text_inputs = self.tokenizer(
            text, return_tensors=return_tensors, return_token_type_ids=return_token_type_ids, padding=padding, truncation=truncation, max_length=max_length
        )

        return BatchFeature(data={**text_inputs, "pixel_values": pixel_values})

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
