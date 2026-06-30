# coding=utf-8
# Copyright 2026 NVIDIA and The HuggingFace Inc. team. All rights reserved.
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
"""Processor class for LocateAnything."""

import re
from typing import Union

import numpy as np

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import logging


logger = logging.get_logger(__name__)


class LocateAnythingProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {"padding": False},
        "images_kwargs": {},
    }


class LocateAnythingProcessor(ProcessorMixin):
    r"""
    Constructs a LocateAnything processor which wraps a [`LocateAnythingImageProcessor`] and a [`Qwen2Tokenizer`]
    into a single processor.

    [`LocateAnythingProcessor`] offers all the functionalities of [`LocateAnythingImageProcessor`] and
    [`Qwen2Tokenizer`]. See the [`~LocateAnythingProcessor.__call__`] and [`~LocateAnythingProcessor.decode`] for
    more information.

    Args:
        image_processor ([`LocateAnythingImageProcessor`], *optional*):
            The image processor.
        tokenizer ([`Qwen2Tokenizer`], *optional*):
            The tokenizer.
        chat_template (`str`, *optional*):
            A Jinja template which will be used to convert lists of messages in a chat into a tokenizable string.
        image_token (`str`, *optional*, defaults to `"<IMG_CONTEXT>"`):
            The placeholder token used to represent a single visual token.
        image_start_token (`str`, *optional*, defaults to `"<img>"`):
            Token inserted before the visual tokens of an image.
        image_end_token (`str`, *optional*, defaults to `"</img>"`):
            Token inserted after the visual tokens of an image.
    """

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
        self.image_placeholder = "image"
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        audio=None,
        videos=None,
        **kwargs: Unpack[LocateAnythingProcessorKwargs],
    ) -> BatchFeature:
        if text is None:
            raise ValueError("You have to specify `text`.")
        if isinstance(text, str):
            text = [text]
        elif not (isinstance(text, (list, tuple)) and isinstance(text[0], str)):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings.")

        output_kwargs = self._merge_kwargs(
            LocateAnythingProcessorKwargs, tokenizer_init_kwargs=self.tokenizer.init_kwargs, **kwargs
        )

        image_inputs = {}
        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            grid_hws = image_inputs["image_grid_hws"]
            merge = self.image_processor.merge_kernel_size[0] * self.image_processor.merge_kernel_size[1]
            num_tokens = [int(h * w) // merge for h, w in grid_hws]

            pattern = re.compile(rf"<{self.image_placeholder}-(\d+)>")
            new_text = []
            for sample in text:
                index = [0]

                def _replace(match):
                    number = match.group(1)
                    idx = int(number) - 1 if number else index[0]
                    index[0] += 1
                    placeholder = self.image_token * num_tokens[idx]
                    visible_id = number if number else str(idx + 1)
                    return f"<image {visible_id}>{self.image_start_token}{placeholder}{self.image_end_token}"

                new_text.append(pattern.sub(_replace, sample))
            text = new_text

        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        return BatchFeature(data={**text_inputs, **image_inputs})

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    @staticmethod
    def parse_boxes(answer: str, image_width: int, image_height: int) -> list[dict]:
        """Parse model output into pixel-coordinate bounding boxes (coordinates are normalized integers in [0, 1000])."""
        boxes = []
        for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
            x1, y1, x2, y2 = (int(g) for g in m.groups())
            boxes.append(
                {
                    "x1": x1 / 1000 * image_width,
                    "y1": y1 / 1000 * image_height,
                    "x2": x2 / 1000 * image_width,
                    "y2": y2 / 1000 * image_height,
                }
            )
        return boxes

    @staticmethod
    def parse_points(answer: str, image_width: int, image_height: int) -> list[dict]:
        """Parse model output into pixel-coordinate points."""
        points = []
        for m in re.finditer(r"<box><(\d+)><(\d+)></box>", answer):
            x, y = int(m.group(1)), int(m.group(2))
            points.append({"x": x / 1000 * image_width, "y": y / 1000 * image_height})
        return points


__all__ = ["LocateAnythingProcessor"]
