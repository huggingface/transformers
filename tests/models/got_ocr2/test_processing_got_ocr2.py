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

from transformers import GotOcr2Processor
from transformers.testing_utils import is_torch_available, require_vision

from ...test_processing_common import MODALITY_TEST_SPECS, ProcessorTesterMixin


if is_torch_available():
    import torch


@require_vision
class GotOcr2ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = GotOcr2Processor
    # `num_image_tokens` is controlled via kwargs, tho overriding each testcase is overkill
    # just use higher `max_length` in tests
    images_unstructured_max_length = 300
    images_text_kwargs_max_length = 300
    images_text_kwargs_override_max_length = 300

    # Tiny processor created with make_tiny_processor.py from "stepfun-ai/GOT-OCR-2.0-hf"
    tiny_model_id = "hf-internal-testing/tiny-processor-got_ocr2"

    @classmethod
    def _setup_image_processor(cls):
        # Instantiate directly to avoid loading the full 384×384 image processor from Hub.
        image_processor_class = cls._get_component_class_from_processor("image_processor")
        return image_processor_class()

    def test_ocr_queries(self):
        processor = self.get_processor()
        image_input = self.prepare_images_inputs()
        inputs = processor(image_input, return_tensors="pt")
        self.assertEqual(inputs["input_ids"].shape, (1, 324))
        self.assertEqual(inputs["pixel_values"].shape, (1, 3, 384, 384))

        inputs = processor(image_input, return_tensors="pt", format=True)
        self.assertEqual(inputs["input_ids"].shape, (1, 328))
        self.assertEqual(inputs["pixel_values"].shape, (1, 3, 384, 384))

        inputs = processor(image_input, return_tensors="pt", color="red")
        self.assertEqual(inputs["input_ids"].shape, (1, 329))
        self.assertEqual(inputs["pixel_values"].shape, (1, 3, 384, 384))

        inputs = processor(image_input, return_tensors="pt", box=[0, 0, 100, 100])
        self.assertEqual(inputs["input_ids"].shape, (1, 341))
        self.assertEqual(inputs["pixel_values"].shape, (1, 3, 384, 384))

        inputs = processor([image_input, image_input], return_tensors="pt", multi_page=True, format=True)
        self.assertEqual(inputs["input_ids"].shape, (1, 595))
        self.assertEqual(inputs["pixel_values"].shape, (2, 3, 384, 384))

        inputs = processor(image_input, return_tensors="pt", crop_to_patches=True, max_patches=6)
        self.assertEqual(inputs["input_ids"].shape, (1, 1872))
        self.assertEqual(inputs["pixel_values"].shape, (7, 3, 384, 384))

    def test_processor_text_has_no_visual(self):
        # Overwritten: requires `multi_page` kwarg to process nested vision inputs
        processor = self.get_processor()

        text = self.prepare_text_inputs(batch_size=3, modalities="image")
        image_inputs = self.prepare_images_inputs(batch_size=3)
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

    def test_subprocessor_defaults_1_images(self):
        # overriden - pop certina keys from `merged_kwargs` which are used only by processor
        parameterized_config = MODALITY_TEST_SPECS["images"]
        subprocessor = self.get_component(parameterized_config["component_key"])

        # Get all other required components for processor
        components = {}
        for attribute in self.processor_class.get_attributes():
            components[attribute] = self.get_component(attribute)

        processor = self.processor_class(**components, **self.prepare_processor_dict())
        modality_input = self._prepare_modality_input("images")

        # merge processor defaults when calling a subprocessor
        kwargs = parameterized_config["call_time_kwargs"]
        kwargs["return_tensors"] = "pt"
        merged_kwargs = processor._merge_kwargs(
            processor.valid_processor_kwargs,
            tokenizer_init_kwargs=None,
            **kwargs,
        )
        kwargs = merged_kwargs["images_kwargs"]
        kwargs.pop("num_image_tokens")
        kwargs.pop("multi_page")

        input_subproc = subprocessor(modality_input, **kwargs)
        try:
            input_processor = processor(images=modality_input, **kwargs)
        except Exception:
            input_processor = {}

        # Verify outputs match
        for key in input_subproc:
            if input_processor and key in processor.model_input_names:
                torch.testing.assert_close(input_subproc[key], input_processor[key])
