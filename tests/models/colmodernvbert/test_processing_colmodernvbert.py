# Copyright 2024 HuggingFace Inc.
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
"""Testing suite for the ColModernVBert processor."""

import shutil
import tempfile
import unittest

import torch

from transformers.models.colmodernvbert.processing_colmodernvbert import ColModernVBertProcessor
from transformers.testing_utils import get_tests_dir, require_torch, require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import (
        ColModernVBertProcessor,
    )

SAMPLE_VOCAB = get_tests_dir("fixtures/vocab.txt")


@require_vision
class ColModernVBertProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = ColModernVBertProcessor

    @classmethod
    def setUpClass(cls):
        cls.tmpdirname = tempfile.mkdtemp()
        processor = ColModernVBertProcessor.from_pretrained("ModernVBERT/colmodernvbert")
        processor.save_pretrained(cls.tmpdirname)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    @require_torch
    @require_vision
    def test_process_images(self):
        # Processor configuration
        image_input = self.prepare_image_inputs()
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer", max_length=112, padding="max_length")

        # Get the processor
        processor = self.processor_class(
            tokenizer=tokenizer,
            image_processor=image_processor,
        )

        # Process the image
        batch_feature = processor.process_images(images=image_input, return_tensors="pt")

        # Assertions
        self.assertIn("pixel_values", batch_feature)
        # ModernVBert/Idefics3 usually resizes to something specific or keeps aspect ratio.
        # Let's check if pixel_values are present and have correct type.
        self.assertIsInstance(batch_feature["pixel_values"], torch.Tensor)
        # Shape depends on image processor config, so we might not want to hardcode it unless we know defaults.
        # Idefics3 default size is often dynamic or specific.

    @require_torch
    @require_vision
    def test_process_queries(self):
        # Inputs
        queries = [
            "Is attention really all you need?",
            "Are Benjamin, Antoine, Merve, and Jo best friends?",
        ]

        # Processor configuration
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer", max_length=112, padding="max_length")

        # Get the processor
        processor = self.processor_class(
            tokenizer=tokenizer,
            image_processor=image_processor,
        )

        # Process the queries
        batch_feature = processor.process_queries(text=queries, return_tensors="pt")

        # Assertions
        self.assertIn("input_ids", batch_feature)
        self.assertIsInstance(batch_feature["input_ids"], torch.Tensor)
        self.assertEqual(batch_feature["input_ids"].shape[0], len(queries))

    # The following tests override the parent tests because ColModernVBertProcessor can only take one of images or text as input at a time.

    def test_tokenizer_defaults_preserved_by_kwargs(self):
        if "image_processor" not in self.processor_class.get_attributes():
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_components["tokenizer"] = self.get_component("tokenizer", max_length=117, padding="max_length")

        processor = self.processor_class(**processor_components)
        self.skip_processor_without_typed_kwargs(processor)
        input_str = self.prepare_text_inputs()
        inputs = processor(text=input_str, return_tensors="pt")
        self.assertEqual(inputs[self.text_input_name].shape[-1], 117)

    def test_image_processor_defaults_preserved_by_image_kwargs(self):
        """
        We use do_rescale=True, rescale_factor=-1.0 to ensure that image_processor kwargs are preserved in the processor.
        We then check that the mean of the pixel_values is less than or equal to 0 after processing.
        Since the original pixel_values are in [0, 255], this is a good indicator that the rescale_factor is indeed applied.
        """
        if "image_processor" not in self.processor_class.get_attributes():
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_components["image_processor"] = self.get_component(
            "image_processor", do_rescale=True, rescale_factor=-1.0
        )
        processor_components["tokenizer"] = self.get_component("tokenizer", max_length=117, padding="max_length")

        processor = self.processor_class(**processor_components)
        self.skip_processor_without_typed_kwargs(processor)

        image_input = self.prepare_image_inputs()

        inputs = processor(images=image_input, return_tensors="pt")
        self.assertLessEqual(inputs[self.images_input_name][0][0].mean(), 0)

    def test_kwargs_overrides_default_tokenizer_kwargs(self):
        if "image_processor" not in self.processor_class.get_attributes():
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_components["tokenizer"] = self.get_component("tokenizer", padding="longest")

        processor = self.processor_class(**processor_components)
        self.skip_processor_without_typed_kwargs(processor)
        input_str = self.prepare_text_inputs()
        inputs = processor(text=input_str, return_tensors="pt", max_length=112, padding="max_length")
        self.assertEqual(inputs[self.text_input_name].shape[-1], 112)

    def test_kwargs_overrides_default_image_processor_kwargs(self):
        if "image_processor" not in self.processor_class.get_attributes():
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_components["image_processor"] = self.get_component(
            "image_processor", do_rescale=True, rescale_factor=1
        )
        processor_components["tokenizer"] = self.get_component("tokenizer", max_length=117, padding="max_length")

        processor = self.processor_class(**processor_components)
        self.skip_processor_without_typed_kwargs(processor)

        image_input = self.prepare_image_inputs()

        inputs = processor(images=image_input, do_rescale=True, rescale_factor=-1.0, return_tensors="pt")
        self.assertLessEqual(inputs[self.images_input_name][0][0].mean(), 0)

    def test_unstructured_kwargs(self):
        if "image_processor" not in self.processor_class.get_attributes():
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor = self.processor_class(**processor_components)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs()
        inputs = processor(
            text=input_str,
            return_tensors="pt",
            do_rescale=True,
            rescale_factor=-1.0,
            padding="max_length",
            max_length=76,
        )

        self.assertEqual(inputs[self.text_input_name].shape[-1], 76)

    def test_unstructured_kwargs_batched(self):
        if "image_processor" not in self.processor_class.get_attributes():
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor = self.processor_class(**processor_components)
        self.skip_processor_without_typed_kwargs(processor)

        image_input = self.prepare_image_inputs(batch_size=2)
        inputs = processor(
            images=image_input,
            return_tensors="pt",
            do_rescale=True,
            rescale_factor=-1.0,
            padding="longest",
            max_length=76,
        )

        self.assertLessEqual(inputs[self.images_input_name][0][0].mean(), 0)

    def test_doubly_passed_kwargs(self):
        if "image_processor" not in self.processor_class.get_attributes():
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor = self.processor_class(**processor_components)
        self.skip_processor_without_typed_kwargs(processor)

        image_input = self.prepare_image_inputs()
        with self.assertRaises(ValueError):
            _ = processor(
                images=image_input,
                images_kwargs={"do_rescale": True, "rescale_factor": -1.0},
                do_rescale=True,
                return_tensors="pt",
            )

    def test_structured_kwargs_nested(self):
        if "image_processor" not in self.processor_class.get_attributes():
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor = self.processor_class(**processor_components)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs()

        # Define the kwargs for each modality
        all_kwargs = {
            "common_kwargs": {"return_tensors": "pt"},
            "images_kwargs": {"do_rescale": True, "rescale_factor": -1.0},
            "text_kwargs": {"padding": "max_length", "max_length": 15, "truncation": True},
        }

        inputs = processor(text=input_str, **all_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        self.assertEqual(inputs[self.text_input_name].shape[-1], 15)

    def test_structured_kwargs_nested_from_dict(self):
        if "image_processor" not in self.processor_class.get_attributes():
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor = self.processor_class(**processor_components)
        self.skip_processor_without_typed_kwargs(processor)
        image_input = self.prepare_image_inputs()

        # Define the kwargs for each modality
        all_kwargs = {
            "common_kwargs": {"return_tensors": "pt"},
            "images_kwargs": {"do_rescale": True, "rescale_factor": -1.0},
            "text_kwargs": {"padding": "max_length", "max_length": 400},
        }

        inputs = processor(images=image_input, **all_kwargs)
        self.assertEqual(inputs[self.text_input_name].shape[-1], 400)

    # Can process only text or images at a time
    def test_model_input_names(self):
        processor = self.get_processor()
        image_input = self.prepare_image_inputs()
        inputs = processor(images=image_input)

        self.assertSetEqual(set(inputs.keys()), set(processor.model_input_names))

    def test_tokenizer_defaults(self):
        """
        Tests that tokenizer is called correctly when passing text to the processor.
        This test verifies that processor(text=X) produces the same output as tokenizer(self.query_prefix + X + suffix).
        """
        # Skip if processor doesn't have tokenizer
        if "tokenizer" not in self.processor_class.get_attributes():
            self.skipTest(f"tokenizer attribute not present in {self.processor_class}")

        # Get all required components for processor
        components = {}
        for attribute in self.processor_class.get_attributes():
            components[attribute] = self.get_component(attribute)

        processor = self.processor_class(**components)
        tokenizer = components["tokenizer"]

        input_str = ["lower newer"]

        # Process with both tokenizer and processor (disable padding to ensure same output)
        try:
            encoded_processor = processor(text=input_str, padding=False, return_tensors="pt")
        except Exception:
            # The processor does not accept text only input, so we can skip this test
            self.skipTest("Processor does not accept text-only input.")
        tok_inputs = [processor.query_prefix + s + processor.query_augmentation_token * 10 for s in input_str]
        encoded_tok = tokenizer(tok_inputs, padding=False, return_tensors="pt")

        # Verify outputs match (handle processors that might not return token_type_ids)
        for key in encoded_tok:
            if key in encoded_processor:
                self.assertListEqual(encoded_tok[key].tolist(), encoded_processor[key].tolist())

    @unittest.skip("ColModernVBert can't process text+image inputs at the same time")
    def test_processor_text_has_no_visual(self):
        pass

    @unittest.skip("ColModernVBert can't process text+image inputs at the same time")
    def test_processor_with_multiple_inputs(self):
        pass

    @unittest.skip("ColModernVBert does not have a chat template")
    def test_chat_template_save_loading(self):
        pass

    @unittest.skip("ColModernVBert does not have a chat template")
    def test_apply_chat_template_audio(self):
        pass

    @unittest.skip("ColModernVBert does not have a chat template")
    def test_apply_chat_template_decoded_video(self):
        pass

    @unittest.skip("ColModernVBert does not have a chat template")
    def test_apply_chat_template_video(self):
        pass

    @unittest.skip("ColModernVBert does not have a chat template")
    def test_apply_chat_template_image(self):
        pass

    @unittest.skip("ColModernVBert does not have a chat template")
    def test_apply_chat_template_video_frame_sampling(self):
        pass

    @unittest.skip("ColModernVBert does not have a chat template")
    def test_chat_template_audio_from_video(self):
        pass

    @unittest.skip("ColModernVBert does not have a chat template")
    def test_chat_template_jinja_kwargs(self):
        pass

    @unittest.skip("ColModernVBert does not have a chat template")
    def test_apply_chat_template_assistant_mask(self):
        pass
