# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torchvision_available, is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import (
        InstructBlipVideoProcessor,
    )

    if is_torchvision_available():
        pass


@require_vision
@require_torch
class InstructBlipVideoProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = InstructBlipVideoProcessor

    @classmethod
    def _setup_tokenizer(cls):
        tokenizer_class = cls._get_component_class_from_processor("tokenizer")
        return tokenizer_class.from_pretrained("hf-internal-testing/tiny-random-GPT2Model")

    @classmethod
    def _setup_qformer_tokenizer(cls):
        qformer_tokenizer_class = cls._get_component_class_from_processor("qformer_tokenizer")
        return qformer_tokenizer_class.from_pretrained("hf-internal-testing/tiny-random-bert")

    @staticmethod
    def prepare_processor_dict():
        return {"num_query_tokens": 1}

    @unittest.skip("InstructBlipVideoProcessor takes in 'images' instead of 'videos' (legacy)")
    def test_processor_with_multiple_inputs(self):
        pass
