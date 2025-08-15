# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Fast Video processor class for InternS1."""

from typing import Union

from ...processing_utils import Unpack, VideosKwargs
from ...utils.import_utils import requires
from ..internvl.video_processing_internvl import InternVLVideoProcessor


class InternS1VideoProcessorInitKwargs(VideosKwargs):
    initial_shift: Union[bool, float, int]


@requires(backends=("torchvision",))
class InternS1VideoProcessor(InternVLVideoProcessor):
    valid_kwargs = InternS1VideoProcessorInitKwargs

    def __init__(self, **kwargs: Unpack[InternS1VideoProcessorInitKwargs]):
        super().__init__(**kwargs)


__all__ = ["InternS1VideoProcessor"]
