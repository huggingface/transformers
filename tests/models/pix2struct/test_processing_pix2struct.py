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
        image_input = self.prepare_image_inputs()

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

    @require_torch
    @require_vision
    def test_image_processor_defaults_preserved_by_image_kwargs(self):
        # Rewrite as pix2struct processor return "flattened_patches" and not "pixel_values"
        if "image_processor" not in self.processor_class.get_attributes():
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor", max_patches=1024, patch_size={"height": 8, "width": 8})
        print("image_processor", image_processor)
        tokenizer = self.get_component("tokenizer", max_length=117, padding="max_length")

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs()
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input)
        self.assertEqual(len(inputs["flattened_patches"][0][0]), 194)

    @require_torch
    @require_vision
    def test_kwargs_overrides_default_image_processor_kwargs(self):
        # Rewrite as pix2struct processor return "flattened_patches" and not "pixel_values"
        if "image_processor" not in self.processor_class.get_attributes():
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor", max_patches=4096)
        tokenizer = self.get_component("tokenizer", max_length=117, padding="max_length")

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs()
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input, max_patches=1024)
        self.assertEqual(len(inputs["flattened_patches"][0]), 1024)

    @require_torch
    @require_vision
    def test_unstructured_kwargs(self):
        # Rewrite as pix2struct processor return "decoder_input_ids" and not "input_ids"
        if "image_processor" not in self.processor_class.get_attributes():
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer")

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs()
        image_input = self.prepare_image_inputs()
        inputs = processor(
            text=input_str,
            images=image_input,
            return_tensors="pt",
            max_patches=1024,
            padding="max_length",
            max_length=76,
        )

        self.assertEqual(inputs["flattened_patches"].shape[1], 1024)
        self.assertEqual(len(inputs["decoder_input_ids"][0]), 76)

    @require_torch
    @require_vision
    def test_unstructured_kwargs_batched(self):
        # Rewrite as pix2struct processor return "decoder_input_ids" and not "input_ids"
        if "image_processor" not in self.processor_class.get_attributes():
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer")

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs(batch_size=2)
        image_input = self.prepare_image_inputs(batch_size=2)
        inputs = processor(
            text=input_str,
            images=image_input,
            return_tensors="pt",
            max_patches=1024,
            padding="longest",
            max_length=76,
        )

        self.assertEqual(inputs["flattened_patches"].shape[1], 1024)

        self.assertEqual(len(inputs["decoder_input_ids"][0]), 5)

    @require_torch
    @require_vision
    def test_structured_kwargs_nested(self):
        # Rewrite as pix2struct processor return "decoder_input_ids" and not "input_ids"
        if "image_processor" not in self.processor_class.get_attributes():
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer")

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs()
        image_input = self.prepare_image_inputs()

        # Define the kwargs for each modality
        all_kwargs = {
            "common_kwargs": {"return_tensors": "pt"},
            "images_kwargs": {"max_patches": 1024},
            "text_kwargs": {"padding": "max_length", "max_length": 76},
        }

        inputs = processor(text=input_str, images=image_input, **all_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        self.assertEqual(inputs["flattened_patches"].shape[1], 1024)

        self.assertEqual(len(inputs["decoder_input_ids"][0]), 76)

    @require_torch
    @require_vision
    def test_structured_kwargs_nested_from_dict(self):
        # Rewrite as pix2struct processor return "decoder_input_ids" and not "input_ids"
        if "image_processor" not in self.processor_class.get_attributes():
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")

        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer")

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)
        input_str = self.prepare_text_inputs()
        image_input = self.prepare_image_inputs()

        # Define the kwargs for each modality
        all_kwargs = {
            "common_kwargs": {"return_tensors": "pt"},
            "images_kwargs": {"max_patches": 1024},
            "text_kwargs": {"padding": "max_length", "max_length": 76},
        }

        inputs = processor(text=input_str, images=image_input, **all_kwargs)
        self.assertEqual(inputs["flattened_patches"].shape[1], 1024)

        self.assertEqual(len(inputs["decoder_input_ids"][0]), 76)
