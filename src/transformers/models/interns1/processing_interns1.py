# coding=utf-8
# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

from typing import Optional, Union

from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ImagesKwargs, ProcessingKwargs, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...video_utils import VideoInput
from ..internvl.processing_internvl import InternVLProcessor


class InternS1ImagesKwargs(ImagesKwargs, total=False):
    crop_to_patches: Optional[bool]
    min_patches: Optional[int]
    max_patches: Optional[int]


class InternS1ProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: InternS1ImagesKwargs
    _defaults = {
        "text_kwargs": {
            "padding_side": "left",
            "return_mm_token_type_ids": False,
        },
        "images_kwargs": {
            "crop_to_patches": True,
        },
        "videos_kwargs": {},
    }


# TODO: It will support temporal information processing in the future.
class InternS1Processor(InternVLProcessor):
    r"""
    Constructs a InternS1 processor which wraps a [`AutoImageProcessor`] and
    [`PretrainedTokenizerFast`] tokenizer into a single processor that inherits both the image processor and
    tokenizer functionalities. See the [`~InternS1Processor.__call__`] and [`~InternS1Processor.decode`] for
    more information.
    """
    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Optional[Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]] = None,
        audio=None,
        videos: Optional[VideoInput] = None,
        **kwargs: Unpack[InternS1ProcessorKwargs],
    ) -> BatchFeature:
        return super().__call__(
            images=images,
            text=text,
            audio=audio,
            videos=videos,
            **kwargs,
        )


__all__ = ["InternS1Processor"]

