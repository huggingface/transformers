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
Processor class for Kosmos2_5.
bf16 now
"""

from typing import List, Optional, Union

from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import TensorType
import torch


class Kosmos2_5Processor(ProcessorMixin):
    r"""
    Constructs a Kosmos2_5 processor which wraps a BERT tokenizer and Kosmos2_5 image processor into a single
    processor.

    [`Kosmos2_5Processor`] offers all the functionalities of [`Kosmos2_5ImageProcessor`] and [`T5TokenizerFast`]. See
    the docstring of [`~Kosmos2_5Processor.__call__`] and [`~Kosmos2_5Processor.decode`] for more information.

    Args:
        image_processor (`Kosmos2_5ImageProcessor`):
            An instance of [`Kosmos2_5ImageProcessor`]. The image processor is a required input.
        tokenizer (Union[`T5TokenizerFast`, `T5Tokenizer`]):
            An instance of ['T5TokenizerFast`] or ['T5Tokenizer`]. The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "Kosmos2_5ImageProcessor"
    tokenizer_class = "PreTrainedTokenizerFast"

    def __init__(self, image_processor, tokenizer):
        tokenizer.return_token_type_ids = False
        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        images=None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        max_patches: Optional[int] = 4096,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_token_type_ids: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchEncoding:
        """
        This method uses [`Kosmos2_5ImageProcessor.preprocess`] method to prepare image(s) for the model, and
        [`T5TokenizerFast.__call__`] to prepare text for the model.

        Please refer to the docstring of the above two methods for more information.
        """
        if images is None and text is None:
            raise ValueError("You have to specify either images or text.")

        encoding_image_processor = self.image_processor(
            images, return_tensors=return_tensors, max_patches=max_patches, **kwargs
        )

        # use updates or pop
        promt_ids = self.tokenizer(text, return_tensors="pt").input_ids[0].tolist()
        input_ids = [0, 100283] + [0] * 2048 + [100284] + promt_ids
        image_embeds_position_mask = [0]*2 + [1]*2048 + [0]*(1+len(promt_ids))
        attention_mask = [0, 1] + [1]*2048 + [1]*(1+len(promt_ids))
        
        input_ids = torch.LongTensor(input_ids).unsqueeze(0)
        attention_mask = torch.Tensor(attention_mask).unsqueeze(0)
        image_embeds_position_mask = torch.LongTensor(image_embeds_position_mask).unsqueeze(0)

        return {
            "flattened_patches": encoding_image_processor.flattened_patches.to(torch.bfloat16),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_attention_mask" : encoding_image_processor.attention_mask,
            "image_embeds_position_mask": image_embeds_position_mask,
            "image_embeds": None,
            "height": encoding_image_processor.height,
            "width": encoding_image_processor.width,
        }


    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Kosmos2_5TokenizerFast's [`~PreTrainedTokenizer.batch_decode`].
        Please refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Kosmos2_5TokenizerFast's [`~PreTrainedTokenizer.decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
