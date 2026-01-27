# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import unittest

from transformers.testing_utils import require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import Blip2Processor


@require_vision
class Blip2ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Blip2Processor

    @classmethod
    def _setup_tokenizer(cls):
        tokenizer_class = cls._get_component_class_from_processor("tokenizer")
        return tokenizer_class.from_pretrained("hf-internal-testing/tiny-random-GPT2Model")

    @classmethod
    def _setup_image_processor(cls):
        image_processor_class = cls._get_component_class_from_processor("image_processor")
        return image_processor_class.from_pretrained("hf-internal-testing/tiny-random-ViTModel")

    @staticmethod
    def prepare_processor_dict():
        return {"num_query_tokens": 1}
