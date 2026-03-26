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

import unittest

from transformers.testing_utils import require_torch, require_torchvision, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import Molmo2Processor

if is_torch_available():
    pass


@require_vision
@require_torch
@require_torchvision
class Molmo2ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Molmo2Processor
    model_id = "allenai/Molmo2-8B"

    @classmethod
    def _setup_from_pretrained(cls, model_id, **kwargs):
        return super()._setup_from_pretrained(model_id, **kwargs)

    def test_model_input_names(self):
        processor = self.get_processor()

        text = self.prepare_text_inputs(modalities=["image"])
        image_input = self.prepare_image_inputs()
        inputs_dict = {"text": text, "images": image_input}
        inputs = processor(**inputs_dict, return_tensors="pt")

        self.assertSetEqual(set(inputs.keys()), set(processor.model_input_names))
