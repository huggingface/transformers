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

import shutil
import tempfile
import unittest

from transformers import AutoProcessor, GotOcr2Processor, PreTrainedTokenizerFast
from transformers.testing_utils import require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import GotOcr2ImageProcessor


@require_vision
class GotOcr2ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = GotOcr2Processor

    @classmethod
    def setUpClass(cls):
        cls.tmpdirname = tempfile.mkdtemp()

        image_processor = GotOcr2ImageProcessor()
        tokenizer = PreTrainedTokenizerFast.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf")
        processor_kwargs = {}
        processor = GotOcr2Processor(image_processor, tokenizer, **processor_kwargs)
        processor.save_pretrained(cls.tmpdirname)
        cls.image_token = processor.img_pad_token

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    def test_ocr_queries(self):
        processor = self.get_processor()
        image_input = self.prepare_image_inputs()
        inputs = processor(image_input, return_tensors="pt")
        self.assertEqual(inputs["input_ids"].shape, (1, 286))
        self.assertEqual(inputs["pixel_values"].shape, (1, 3, 384, 384))

        inputs = processor(image_input, return_tensors="pt", format=True)
        self.assertEqual(inputs["input_ids"].shape, (1, 288))
        self.assertEqual(inputs["pixel_values"].shape, (1, 3, 384, 384))

        inputs = processor(image_input, return_tensors="pt", color="red")
        self.assertEqual(inputs["input_ids"].shape, (1, 290))
        self.assertEqual(inputs["pixel_values"].shape, (1, 3, 384, 384))

        inputs = processor(image_input, return_tensors="pt", box=[0, 0, 100, 100])
        self.assertEqual(inputs["input_ids"].shape, (1, 303))
        self.assertEqual(inputs["pixel_values"].shape, (1, 3, 384, 384))

        inputs = processor([image_input, image_input], return_tensors="pt", multi_page=True, format=True)
        self.assertEqual(inputs["input_ids"].shape, (1, 547))
        self.assertEqual(inputs["pixel_values"].shape, (2, 3, 384, 384))

        inputs = processor(image_input, return_tensors="pt", crop_to_patches=True, max_patches=6)
        self.assertEqual(inputs["input_ids"].shape, (1, 1826))
        self.assertEqual(inputs["pixel_values"].shape, (7, 3, 384, 384))

    def test_processor_text_has_no_visual(self):
        # Overwritten: requires `multi_page` kwarg to process nested vision inputs
        processor = self.get_processor()

        text = self.prepare_text_inputs(batch_size=3, modalities="image")
        image_inputs = self.prepare_image_inputs(batch_size=3)
        processing_kwargs = {"return_tensors": "pt", "padding": True, "multi_page": True}

        # Call with nested list of vision inputs
        image_inputs_nested = [[image] if not isinstance(image, list) else image for image in image_inputs]
        inputs_dict_nested = {"text": text, "images": image_inputs_nested}
        inputs = processor(**inputs_dict_nested, **processing_kwargs)
        self.assertTrue(self.text_input_name in inputs)

        # Call with one of the samples with no associated vision input
        plain_text = "lower newer"
        image_inputs_nested[0] = []
        text[0] = plain_text
        inputs_dict_no_vision = {"text": text, "images": image_inputs_nested}
        inputs_nested = processor(**inputs_dict_no_vision, **processing_kwargs)
        self.assertListEqual(
            inputs[self.text_input_name][1:].tolist(), inputs_nested[self.text_input_name][1:].tolist()
        )
