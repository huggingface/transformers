# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Processor for LocateAnything."""

import re

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput


class LocateAnythingProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {"padding": False},
    }


class LocateAnythingProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template", "image_token", "image_start_token", "image_end_token"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        image_token="<IMG_CONTEXT>",
        image_start_token="<img>",
        image_end_token="</img>",
        **kwargs,
    ):
        self.image_token = image_token
        self.image_start_token = image_start_token
        self.image_end_token = image_end_token
        self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images: ImageInput = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        **kwargs: Unpack[LocateAnythingProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            LocateAnythingProcessorKwargs, tokenizer_init_kwargs=self.tokenizer.init_kwargs, **kwargs
        )
        if not isinstance(text, list):
            text = [text]
        text = text.copy()

        image_inputs = {}
        if images is not None:
            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
            grid = image_inputs["image_grid_hws"]
            merge_area = self.image_processor.merge_kernel_size[0] * self.image_processor.merge_kernel_size[1]

            index = 0

            def replace(match):
                nonlocal index
                number = match.group(1)
                height, width = grid[index].tolist()
                num_tokens = int(height * width) // merge_area
                index += 1
                return f"<image {number}>{self.image_start_token}{self.image_token * num_tokens}{self.image_end_token}"

            for i in range(len(text)):
                text[i] = re.sub(r"<image-(\d+)>", replace, text[i])

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


__all__ = ["LocateAnythingProcessor"]
