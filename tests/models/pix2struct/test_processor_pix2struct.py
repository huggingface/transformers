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
import shutil
import tempfile
import unittest

import pytest

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import (
        AutoProcessor,
        Pix2StructImageProcessor,
        Pix2StructProcessor,
        PreTrainedTokenizerFast,
        T5Tokenizer,
    )


@require_vision
@require_torch
class Pix2StructProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Pix2StructProcessor
    text_input_name = "decoder_input_ids"
    images_input_name = "flattened_patches"

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()

        image_processor = Pix2StructImageProcessor()
        tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")

        processor = Pix2StructProcessor(image_processor, tokenizer)

        processor.save_pretrained(self.tmpdirname)

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def test_save_load_pretrained_additional_features(self):
        processor = Pix2StructProcessor(tokenizer=self.get_tokenizer(), image_processor=self.get_image_processor())
        processor.save_pretrained(self.tmpdirname)

        tokenizer_add_kwargs = self.get_tokenizer(bos_token="(BOS)", eos_token="(EOS)")
        image_processor_add_kwargs = self.get_image_processor(do_normalize=False, padding_value=1.0)

        processor = Pix2StructProcessor.from_pretrained(
            self.tmpdirname, bos_token="(BOS)", eos_token="(EOS)", do_normalize=False, padding_value=1.0
        )

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())
        self.assertIsInstance(processor.tokenizer, PreTrainedTokenizerFast)

        self.assertEqual(processor.image_processor.to_json_string(), image_processor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.image_processor, Pix2StructImageProcessor)

    def test_image_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = Pix2StructProcessor(tokenizer=tokenizer, image_processor=image_processor)

        image_input = self.prepare_image_inputs()

        input_feat_extract = image_processor(image_input, return_tensors="np")
        input_processor = processor(images=image_input, return_tensors="np")

        for key in input_feat_extract.keys():
            self.assertAlmostEqual(input_feat_extract[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_tokenizer(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = Pix2StructProcessor(tokenizer=tokenizer, image_processor=image_processor)

        input_str = self.prepare_text_inputs()

        encoded_processor = processor(text=input_str)

        encoded_tok = tokenizer(input_str, return_token_type_ids=False, add_special_tokens=True)

        for key in encoded_tok.keys():
            self.assertListEqual(encoded_tok[key], encoded_processor[key])

    def test_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = Pix2StructProcessor(tokenizer=tokenizer, image_processor=image_processor)

        input_str = self.prepare_text_inputs()
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input)

        self.assertListEqual(
            list(inputs.keys()), ["flattened_patches", "attention_mask", "decoder_attention_mask", "decoder_input_ids"]
        )

        # test if it raises when no input is passed
        with pytest.raises(ValueError):
            processor()

    def test_processor_max_patches(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = Pix2StructProcessor(tokenizer=tokenizer, image_processor=image_processor)

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

    def test_tokenizer_decode(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = Pix2StructProcessor(tokenizer=tokenizer, image_processor=image_processor)

        predicted_ids = [[1, 4, 5, 8, 1, 0, 8], [3, 4, 3, 1, 1, 8, 9]]

        decoded_processor = processor.batch_decode(predicted_ids)
        decoded_tok = tokenizer.batch_decode(predicted_ids)

        self.assertListEqual(decoded_tok, decoded_processor)

    def test_model_input_names(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = Pix2StructProcessor(tokenizer=tokenizer, image_processor=image_processor)

        input_str = self.prepare_text_inputs()
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input)

        # For now the processor supports only ["flattened_patches", "input_ids", "attention_mask", "decoder_attention_mask"]
        self.assertListEqual(
            list(inputs.keys()), ["flattened_patches", "attention_mask", "decoder_attention_mask", "decoder_input_ids"]
        )

        inputs = processor(text=input_str)

        # For now the processor supports only ["flattened_patches", "input_ids", "attention_mask", "decoder_attention_mask"]
        self.assertListEqual(list(inputs.keys()), ["input_ids", "attention_mask"])

    @require_torch
    @require_vision
    def test_image_processor_defaults_preserved_by_image_kwargs(self):
        # Rewrite as pix2struct processor return "flattened_patches" and not "pixel_values"
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor", max_patches=1024, patch_size={"height": 8, "width": 8})
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
        if "image_processor" not in self.processor_class.attributes:
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
        if "image_processor" not in self.processor_class.attributes:
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
        if "image_processor" not in self.processor_class.attributes:
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
        if "image_processor" not in self.processor_class.attributes:
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
        if "image_processor" not in self.processor_class.attributes:
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
