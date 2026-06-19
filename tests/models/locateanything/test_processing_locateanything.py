# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from transformers import AutoTokenizer, LocateAnythingProcessor
from transformers.testing_utils import require_torch, require_vision

from ...test_processing_common import ProcessorTesterMixin


@require_torch
@require_vision
class LocateAnythingProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = LocateAnythingProcessor

    @classmethod
    def _setup_image_processor(cls):
        image_processor_class = cls._get_component_class_from_processor("image_processor")
        return image_processor_class(patch_size=14)

    @classmethod
    def _setup_tokenizer(cls):
        return AutoTokenizer.from_pretrained("nvidia/LocateAnything-3B")

    @classmethod
    def _setup_test_attributes(cls, processor):
        cls.image_token = processor.image_token

    def test_image_placeholder_expansion(self):
        processor = self.get_processor()
        image = self.prepare_image_inputs()
        inputs = processor(images=image, text=["<image-1>Locate the person."], return_tensors="pt")
        num_patches = inputs["image_grid_thw"][0].prod().item()
        merge_area = processor.image_processor.merge_kernel_size[0] * processor.image_processor.merge_kernel_size[1]
        expected_tokens = num_patches // merge_area
        n_image_tokens = int((inputs["input_ids"][0] == processor.image_token_id).sum())
        self.assertEqual(n_image_tokens, expected_tokens)
        decoded = processor.decode(inputs["input_ids"][0])
        self.assertIn("<image 1>", decoded)
        self.assertIn("<img>", decoded)
        self.assertIn("</img>", decoded)
