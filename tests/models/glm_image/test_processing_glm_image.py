# Copyright 2025 The HuggingFace Team. All rights reserved.
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
from parameterized import parameterized
from PIL import Image

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import GlmImageProcessor


@require_vision
@require_torch
class GlmImageProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = GlmImageProcessor
    # Tiny processor created with make_tiny_processor.py from "zai-org/GLM-Image"
    tiny_model_id = "hf-internal-testing/tiny-processor-glm_image"

    @classmethod
    def _setup_test_attributes(cls, processor):
        cls.image_token = processor.image_token

    def prepare_images_inputs(self, batch_size: int | None = None, nested: bool = False):
        """Override to create images with valid aspect ratio (< 4) for GLM-Image."""
        # GLM-Image requires aspect ratio < 4, so use near-square images
        image_inputs = [Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))]
        if batch_size is None:
            return image_inputs
        if nested:
            return [image_inputs] * batch_size
        return image_inputs * batch_size

    def test_model_input_names(self):
        processor = self.get_processor()

        text = self.prepare_text_inputs(modalities=["image"])
        image_input = self.prepare_images_inputs()
        inputs_dict = {"text": text, "images": image_input}
        inputs = processor(**inputs_dict, return_tensors="pt")

        self.assertSetEqual(set(inputs.keys()), set(processor.model_input_names))

    @unittest.skip("tiny model has too little tokens and collapses everything to UNK which is not defined")
    def test_replacement_offsets(self):
        pass

    @parameterized.expand(
        [
            ("text",),
            ("images",),
            ("videos",),
            ("audio",),
        ]
    )
    @unittest.skip("Model changes input content as it is used by diffusers and thus is special")
    def test_subprocessor_defaults(self, modality):
        pass
