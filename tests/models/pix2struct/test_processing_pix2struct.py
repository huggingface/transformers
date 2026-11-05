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

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import (
        Pix2StructProcessor,
    )


@require_vision
@require_torch
class Pix2StructProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Pix2StructProcessor
    text_input_name = "decoder_input_ids"
    images_input_name = "flattened_patches"

    @classmethod
    def _setup_tokenizer(cls):
        tokenizer_class = cls._get_component_class_from_processor("tokenizer")
        return tokenizer_class.from_pretrained("google-t5/t5-small")

    def test_processor_max_patches(self):
        processor = self.get_processor()

        input_str = self.prepare_text_inputs()
        image_input = self.prepare_images_inputs()

        inputs = processor(text=input_str, images=image_input)

        max_patches = [512, 1024, 2048, 4096]
        expected_hidden_size = [770, 770, 770, 770]
        # with text
        for i, max_patch in enumerate(max_patches):
            inputs = processor(text=input_str, images=image_input, max_patches=max_patch)
            self.assertEqual(inputs["flattened_patches"][0].shape[0], max_patch)
            self.assertEqual(inputs["flattened_patches"][0].shape[1], expected_hidden_size[i])

        # without text input
        for i, max_patch in enumerate(max_patches):
            inputs = processor(images=image_input, max_patches=max_patch)
            self.assertEqual(inputs["flattened_patches"][0].shape[0], max_patch)
            self.assertEqual(inputs["flattened_patches"][0].shape[1], expected_hidden_size[i])

    # Rewrite as Pix2Strict processor applies custom normalization, we can't check `out.mean()`
    def _check_modality_outputs(self, inputs: dict, modality: str):
        input_key = getattr(self, f"{modality}_input_name")
        if modality in ["image"]:
            self.assertEqual(len(inputs[input_key][0]), 2048)
