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

from transformers import Llama4Processor
from transformers.testing_utils import require_vision

from ...test_processing_common import ProcessorTesterMixin


@require_vision
class Llama4ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Llama4Processor
    # Tiny processor created with make_tiny_processor.py from "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    tiny_model_id = "hf-internal-testing/tiny-processor-llama4"

    @classmethod
    def _setup_image_processor(cls):
        # max_patches=1 ensures each image produces exactly 1 tile, so len(pixel_values)==batch_size.
        # Small size (20×20) keeps tensor allocations minimal.
        image_processor_class = cls._get_component_class_from_processor("image_processor")
        return image_processor_class(max_patches=1, size={"height": 20, "width": 20})

    @classmethod
    def _setup_test_attributes(cls, processor):
        cls.image_token = processor.image_token
