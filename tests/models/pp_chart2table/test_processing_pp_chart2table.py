# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from transformers import PPChart2TableProcessor
from transformers.testing_utils import require_vision

from ...test_processing_common import ProcessorTesterMixin


@require_vision
class PPChart2TableProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = PPChart2TableProcessor

    @classmethod
    def _setup_tokenizer(cls):
        tokenizer_class = cls._get_component_class_from_processor("tokenizer")
        tokenizer = tokenizer_class.from_pretrained("PaddlePaddle/PP-Chart2Table_safetensors")
        return tokenizer

    @unittest.skip("PPChart2TableProcessor pop the image processor output 'num_patches'")
    def test_image_processor_defaults(self):
        pass

    def test_ocr_queries(self):
        processor = self.get_processor()
        image_input = self.prepare_image_inputs()
        conversation = [{"role": "system", "content": []}, {"role": "user", "content": []}]
        inputs = processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(images=image_input, text=inputs, return_tensors="pt")
        self.assertEqual(inputs["input_ids"].shape, (1, 287))
        self.assertEqual(inputs["pixel_values"].shape, (1, 3, 1024, 1024))

    def test_unstructured_kwargs_batched(self):
        if "image_processor" not in self.processor_class.get_attributes():
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_kwargs = self.prepare_processor_dict()
        processor = self.processor_class(**processor_components, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs(batch_size=2, modalities="image")
        image_input = self.prepare_image_inputs(batch_size=2)
        inputs = processor(
            text=input_str,
            images=image_input,
            return_tensors="pt",
            do_rescale=True,
            rescale_factor=-1.0,
            padding="longest",
            max_length=self.image_unstructured_max_length,
        )

        self.assertLessEqual(inputs[self.images_input_name][0][0].mean(), 0)

    @unittest.skip(
        reason="PPChart2Table relies on a predetermined input format; chat template usage is not intended, and image input cannot be None."
    )
    def test_apply_chat_template_assistant_mask(self):
        pass

    @unittest.skip(
        reason="PPChart2Table relies on a predetermined input format; chat template usage is not intended, and image input cannot be None."
    )
    def test_apply_chat_template_image_0(self):
        pass

    @unittest.skip(
        reason="PPChart2Table relies on a predetermined input format; chat template usage is not intended, and image input cannot be None."
    )
    def test_apply_chat_template_image_1(self):
        pass
