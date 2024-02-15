# coding=utf-8
# Copyright 2024 Meta Inc. and The HuggingFace Inc. team. All rights reserved.
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
Processor class for Chameleon.
"""


from typing import List, Optional, Union

from ...tokenization_utils_base import BatchEncoding
from ...image_utils import ImageInput
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import PaddingStrategy, TensorType

import PIL
import numpy as np
import torch


class ChameleonProcessor(ProcessorMixin):
    attributes = ["tokenizer", "image_processor"]
    tokenizer_class = ("ChameleonTokenizer", "ChameleonTokenizerFast")
    image_processor_class = "ChameleonImageProcessor"

    def __init__(self, tokenizer=None, image_processor=None):
        super().__init__(tokenizer, image_processor)

        image_processor.set_vocab(tokenizer.vocab)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _as_pil_image(self, image: ImageInput):
        # Convert image to PIL.Image.Image if not already.
        if isinstance(image, PIL.Image.Image):
            return image
        elif isinstance(image, np.ndarray):
            return PIL.Image.fromarray(image)
        elif isinstance(image, torch.Tensor):
            return PIL.Image.fromarray(image.numpy())
        raise ValueError(f"Unsupported input type: {type(image)}")

    def __call__(
        self,
        inputs: List[List[Union[TextInput, ImageInput, PreTokenizedInput]]] = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> BatchEncoding:
        encoding = BatchEncoding({"input_ids": []})
        for batch in inputs:
            encoding.input_ids.append([self.tokenizer.bos_token_id])
            for elem in batch:
                if isinstance(elem, (PIL.Image.Image, np.ndarray, torch.Tensor)):
                    elem_toks = self.image_processor(self._as_pil_image(elem)).tokens
                else:
                    elem_toks = self.tokenizer(elem).input_ids

                if elem_toks[0] == self.tokenizer.bos_token_id:
                    elem_toks = elem_toks[1:]

                encoding.input_ids[-1].extend(elem_toks)

        encoding = self.tokenizer.pad(
            encoding,
            padding=padding,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )
        return encoding.convert_to_tensors(return_tensors)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)
