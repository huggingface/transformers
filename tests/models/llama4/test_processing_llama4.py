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

import numpy as np

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

    def test_get_num_multimodal_tokens_matches_processor_call(self):
        # `mm_token_type_ids` only marks `<|image|>`, so count the `<|patch|>` tokens in each replacement
        processor = self.get_processor()
        image_sizes = [(100, 100), (300, 100), (500, 30), (213, 167)]
        images = [np.random.randint(255, size=(h, w, 3), dtype=np.uint8) for h, w in image_sizes]
        text = [f"This is an image {processor.image_token}"] * len(images)
        tiling_kwargs = dict(max_patches=4, size={"height": 56, "width": 56})
        inputs = processor(
            text=text,
            images=images,
            **tiling_kwargs,
            padding=True,
            return_tensors="pt",
            return_text_replacement_offsets=True,
        )

        replacements = [r["replacement"] for offsets in inputs["text_replacement_offsets"] for r in offsets]
        num_image_tokens_from_call = [r.count(processor.img_patch_token) for r in replacements]
        num_image_tokens_from_helper = processor._get_num_multimodal_tokens(image_sizes=image_sizes, **tiling_kwargs)
        self.assertListEqual(num_image_tokens_from_call, num_image_tokens_from_helper["num_image_tokens"])
