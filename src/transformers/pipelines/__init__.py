# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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

from typing import TYPE_CHECKING

from ..utils.import_utils import _LazyModule, define_import_structure


if TYPE_CHECKING:
    from ._pipeline import *
    from .audio_classification import *
    from .automatic_speech_recognition import *
    from .depth_estimation import *
    from .document_question_answering import *
    from .feature_extraction import *
    from .fill_mask import *
    from .image_classification import *
    from .image_feature_extraction import *
    from .image_segmentation import *
    from .image_text_to_text import *
    from .image_to_image import *
    from .image_to_text import *
    from .mask_generation import *
    from .object_detection import *
    from .question_answering import *
    from .table_question_answering import *
    from .text2text_generation import *
    from .text_classification import *
    from .text_generation import *
    from .text_to_audio import *
    from .token_classification import *
    from .video_classification import *
    from .visual_question_answering import *
    from .zero_shot_audio_classification import *
    from .zero_shot_classification import *
    from .zero_shot_image_classification import *
    from .zero_shot_object_detection import *
else:
    import sys

    _file = globals()["__file__"]

    sys.modules[__name__] = _LazyModule(
        __name__,
        _file,
        define_import_structure(_file),
        module_spec=__spec__,
    )
