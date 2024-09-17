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

import pytest

from transformers import AutoProcessor, Qwen2Tokenizer
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import Qwen2VLImageProcessor, Qwen2VLProcessor


@require_vision
@require_torch
class Qwen2VLProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Qwen2VLProcessor

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", patch_size=4)
        processor.save_pretrained(self.tmpdirname)

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def test_save_load_pretrained_default(self):
        tokenizer = self.get_tokenizer()
        image_processor = self.get_image_processor()

        processor = Qwen2VLProcessor(tokenizer=tokenizer, image_processor=image_processor)
        processor.save_pretrained(self.tmpdirname)
        processor = Qwen2VLProcessor.from_pretrained(self.tmpdirname, use_fast=False)

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertEqual(processor.image_processor.to_json_string(), image_processor.to_json_string())
        self.assertIsInstance(processor.tokenizer, Qwen2Tokenizer)
        self.assertIsInstance(processor.image_processor, Qwen2VLImageProcessor)

    def test_image_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = Qwen2VLProcessor(tokenizer=tokenizer, image_processor=image_processor)

        image_input = self.prepare_image_inputs()

        input_image_proc = image_processor(image_input, return_tensors="np")
        input_processor = processor(images=image_input, text="dummy", return_tensors="np")

        for key in input_image_proc.keys():
            self.assertAlmostEqual(input_image_proc[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = Qwen2VLProcessor(tokenizer=tokenizer, image_processor=image_processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()
        inputs = processor(text=input_str, images=image_input)

        self.assertListEqual(list(inputs.keys()), ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"])

        # test if it raises when no input is passed
        with pytest.raises(ValueError):
            processor()

        # test if it raises when no text is passed
        with pytest.raises(TypeError):
            processor(images=image_input)

    def test_model_input_names(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = Qwen2VLProcessor(tokenizer=tokenizer, image_processor=image_processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()
        video_inputs = self.prepare_video_inputs()

        inputs = processor(text=input_str, images=image_input, videos=video_inputs)

        self.assertListEqual(list(inputs.keys()), processor.model_input_names)

    # Qwen2-VL doesn't accept `size` and resized to an optimal size using image_processor attrbutes
    # defined at `init`. Therefore, all tests are overwritten and don't actually test if kwargs are passed
    # to image processors
    def test_image_processor_defaults_preserved_by_image_kwargs(self):
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer", max_length=117, padding="max_length")

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input)
        self.assertEqual(inputs["pixel_values"].shape[0], 800)

    def test_kwargs_overrides_default_image_processor_kwargs(self):
        image_processor = self.get_component(
            "image_processor",
        )
        tokenizer = self.get_component("tokenizer", max_length=117, padding="max_length")

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input)
        self.assertEqual(inputs["pixel_values"].shape[0], 800)

    def test_unstructured_kwargs(self):
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer")

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()
        inputs = processor(
            text=input_str,
            images=image_input,
            return_tensors="pt",
            padding="max_length",
            max_length=76,
        )

        self.assertEqual(inputs["pixel_values"].shape[0], 800)
        self.assertEqual(len(inputs["input_ids"][0]), 76)

    def test_unstructured_kwargs_batched(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer")

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = ["lower newer", "upper older longer string"]
        image_input = self.prepare_image_inputs() * 2
        inputs = processor(
            text=input_str,
            images=image_input,
            return_tensors="pt",
            padding="longest",
            max_length=76,
        )

        self.assertEqual(inputs["pixel_values"].shape[0], 1600)
        self.assertEqual(len(inputs["input_ids"][0]), 4)

    def test_structured_kwargs_nested(self):
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer")

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        # Define the kwargs for each modality
        all_kwargs = {
            "common_kwargs": {"return_tensors": "pt"},
            "text_kwargs": {"padding": "max_length", "max_length": 76},
        }

        inputs = processor(text=input_str, images=image_input, **all_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        self.assertEqual(inputs["pixel_values"].shape[0], 800)
        self.assertEqual(len(inputs["input_ids"][0]), 76)

    def test_structured_kwargs_nested_from_dict(self):
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer")

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)
        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        # Define the kwargs for each modality
        all_kwargs = {
            "common_kwargs": {"return_tensors": "pt"},
            "text_kwargs": {"padding": "max_length", "max_length": 76},
        }

        inputs = processor(text=input_str, images=image_input, **all_kwargs)
        self.assertEqual(inputs["pixel_values"].shape[0], 800)
        self.assertEqual(len(inputs["input_ids"][0]), 76)

    def test_image_processor_defaults_preserved_by_video_kwargs(self):
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer", max_length=117, padding="max_length")

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = "lower newer"
        video_input = self.prepare_video_inputs()

        inputs = processor(text=input_str, videos=video_input)
        self.assertEqual(inputs["pixel_values_videos"].shape[0], 9600)
