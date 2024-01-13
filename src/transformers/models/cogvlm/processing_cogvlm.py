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

    [`CogVLMProcessor`] offers all the functionalities of [`CLIPImageProcessor`] and [`LlamaTokenizerFast`]. See the docstring
    of [`~CogVLMProcessor.__call__`] and [`~CogVLMProcessor.decode`] for more information.

    Args:
        image_processor (`CLIPImageProcessor`):
            An instance of [`CLIPImageProcessor`]. The image processor is a required input.
        tokenizer (`LlamaTokenizer` or `LlamaTokenizerFast`):
            An instance of ['LlamaTokenizer`] or ['LlamaTokenizerFast`]. The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "CLIPImageProcessor"
    tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast")

    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> BatchFeature:
        """
        This method uses [`CLIPImageProcessor.__call__`] method to prepare image(s) for the model, and
        [`LlamaTokenizer.__call__`] to prepare text for the model.

        Please refer to the docstring of the above two methods for more information.
        """

        if text is None and images is None:
            raise ValueError("You have to specify either text or images. Both cannot be none.")

        if text is not None:
            text_features = self.tokenizer(
                text, return_tensors=return_tensors, padding=padding, truncation=truncation, max_length=max_length
            )

        if images is not None:
            image_features = self.image_processor(images, return_tensors=return_tensors)

        if text is not None and images is not None:
            return BatchFeature(data=dict(**text_features, **image_features), tensor_type=return_tensors)
        elif text is not None:
            return text_features
        else:
            return BatchFeature(data=dict(**image_features), tensor_type=return_tensors)

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
