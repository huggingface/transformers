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
"""Fast Video processor class for InternVL."""

from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
)
from ...processing_utils import Unpack, VideosKwargs
from ...utils import (
    is_vision_available,
)
from ...utils.import_utils import requires
from ...video_processing_utils import (
    BaseVideoProcessor,
)


if is_vision_available():
    from ...image_utils import PILImageResampling


class InternVLVideoProcessorInitKwargs(VideosKwargs): ...


@requires(backends=("torchvision",))
class InternVLVideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BICUBIC
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = {"height": 384, "width": 384}
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    valid_kwargs = InternVLVideoProcessorInitKwargs
    model_input_names = ["pixel_values_videos"]

    def __init__(self, **kwargs: Unpack[InternVLVideoProcessorInitKwargs]):
        super().__init__(**kwargs)


__all__ = ["InternVLVideoProcessor"]
