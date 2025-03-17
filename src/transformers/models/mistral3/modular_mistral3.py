# coding=utf-8
# Copyright 2025 HuggingFace Inc. team. All rights reserved.
#
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

from ...utils import logging
from ..llava.modeling_llava import LlavaForConditionalGeneration, LlavaMultiModalProjector
from ..llava.configuration_llava import LlavaConfig
from ..pixtral.image_processing_pixtral import PixtralImageProcessor
from ..pixtral.image_processing_pixtral_fast import PixtralImageProcessorFast
from ..pixtral.processing_pixtral import PixtralProcessor


logger = logging.get_logger(__name__)


class Mistral3Config(LlavaConfig):
    pass

class Mistral3ImageProcessor(PixtralImageProcessor):
    pass

class Mistral3ImageProcessorFast(PixtralImageProcessorFast):
    pass

class Mistral3Processor(PixtralProcessor):
    pass

class Mistral3MultiModalProjector(LlavaMultiModalProjector):
    pass

class Mistral3ForConditionalGeneration(LlavaForConditionalGeneration):
    pass

__all__ = [
    "Mistral3PreTrainedModel",  # noqa
    "Mistral3ForConditionalGeneration",
    "Mistral3Config",
    "Mistral3ImageProcessor",
    "Mistral3ImageProcessorFast",
    "Mistral3Processor",
]