# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
Processor class for TrOCR.
"""

from typing import Optional, Union

from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput


class TrOCRProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {}


class TrOCRProcessor(ProcessorMixin):
    r"""
    Constructs a TrOCR processor which wraps a vision image processor and a TrOCR tokenizer into a single processor.

    [`TrOCRProcessor`] offers all the functionalities of [`ViTImageProcessor`/`DeiTImageProcessor`] and
    [`RobertaTokenizer`/`XLMRobertaTokenizer`]. See the [`~TrOCRProcessor.__call__`] and [`~TrOCRProcessor.decode`] for
    more information.

    Args:
        image_processor ([`ViTImageProcessor`/`DeiTImageProcessor`], *optional*):
            An instance of [`ViTImageProcessor`/`DeiTImageProcessor`]. The image processor is a required input.
        tokenizer ([`RobertaTokenizer`/`XLMRobertaTokenizer`], *optional*):
            An instance of [`RobertaTokenizer`/`XLMRobertaTokenizer`]. The tokenizer is a required input.
    """

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Optional[Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]] = None,
        **kwargs: Unpack[TrOCRProcessorKwargs],
    ) -> BatchFeature:
        """
        When used in normal mode, this method forwards all its arguments to AutoImageProcessor's
        [`~AutoImageProcessor.__call__`] and returns its output. If used in the context
        [`~TrOCRProcessor.as_target_processor`] this method forwards all its arguments to TrOCRTokenizer's
        [`~TrOCRTokenizer.__call__`]. Please refer to the docstring of the above two methods for more information.
        """
        if images is None and text is None:
            raise ValueError("You need to specify either an `images` or `text` input to process.")

        output_kwargs = self._merge_kwargs(
            TrOCRProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if images is not None:
            inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
        if text is not None:
            encodings = self.tokenizer(text, **output_kwargs["text_kwargs"])

        if text is None:
            return inputs
        elif images is None:
            return encodings
        else:
            inputs["labels"] = encodings["input_ids"]
            return inputs

    @property
    def model_input_names(self):
        image_processor_input_names = self.image_processor.model_input_names
        return image_processor_input_names + ["labels"]


__all__ = ["TrOCRProcessor"]
