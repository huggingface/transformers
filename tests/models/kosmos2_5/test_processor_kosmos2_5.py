# Copyright 2024 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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

import os
import unittest
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from transformers.image_utils import load_image
from transformers.testing_utils import (
    require_torch,
    require_vision,
)
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin, url_to_local_path


if is_vision_available():
    from PIL import Image

    from transformers import (
        AutoProcessor,
        AutoTokenizer,
        Kosmos2_5ImageProcessor,
        Kosmos2_5Processor,
    )


@require_vision
class Kosmos2_5ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Kosmos2_5Processor
    images_input_name = "flattened_patches"
    model_id = "microsoft/kosmos-2.5"

    @unittest.skip("Kosmos2_5Processor removes 'rows' and 'cols' from the output")
    def test_image_processor_defaults(self):
        pass

    def test_image_procesor_load_save_reload(self):
        # make sure load from Hub repo. -> save -> reload locally work
        image_processor = Kosmos2_5ImageProcessor.from_pretrained("microsoft/kosmos-2.5")
        with TemporaryDirectory() as tmp_dir:
            image_processor.save_pretrained(tmp_dir)
            reloaded_image_processor = Kosmos2_5ImageProcessor.from_pretrained(tmp_dir)
            assert image_processor.to_dict() == reloaded_image_processor.to_dict()
            assert image_processor.to_json_string() == reloaded_image_processor.to_json_string()

    def test_can_load_various_tokenizers(self):
        for checkpoint in ["microsoft/kosmos-2.5"]:
            processor = AutoProcessor.from_pretrained(checkpoint)
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            self.assertEqual(processor.tokenizer.__class__, tokenizer.__class__)

    @require_torch
    def test_model_input_names(self):
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer")

        processor = Kosmos2_5Processor(tokenizer=tokenizer, image_processor=image_processor)

        input_str = "This is a test"
        image_input = self.prepare_image_inputs()

        # both image and text
        inputs = processor(text=input_str, images=image_input)
        self.assertListEqual(
            list(inputs.keys()),
            [
                "flattened_patches",
                "attention_mask",
                "width",
                "height",
                "input_ids",
                "image_embeds_position_mask",
            ],
        )
        # test if it raises when no input is passed
        with pytest.raises(ValueError):
            processor()

    @require_torch
    @require_vision
    def test_image_processor_defaults_preserved_by_image_kwargs(self):
        # Rewrite as KOSMOS-2.5 processor return "flattened_patches" and not "pixel_values"
        if "image_processor" not in self.processor_class.get_attributes():
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
        # Rewrite as KOSMOS-2.5 processor return "flattened_patches" and not "pixel_values"
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
        # Rewrite as KOSMOS-2.5 processor doesn't use `rescale_factor`
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
        self.assertEqual(len(inputs["input_ids"][0]), 76)

    @require_torch
    @require_vision
    def test_unstructured_kwargs_batched(self):
        # Rewrite as KOSMOS-2.5 processor doesn't use `rescale_factor`
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

        self.assertEqual(len(inputs["input_ids"][0]), 76)

    @require_torch
    @require_vision
    def test_structured_kwargs_nested(self):
        # Rewrite as KOSMOS-2.5 processor doesn't use `rescale_factor`
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

        self.assertEqual(len(inputs["input_ids"][0]), 76)

    @require_torch
    @require_vision
    def test_structured_kwargs_nested_from_dict(self):
        # Rewrite as KOSMOS-2.5 processor doesn't use `rescale_factor`
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

        self.assertEqual(len(inputs["input_ids"][0]), 76)

    @require_torch
    def test_full_processor(self):
        url = url_to_local_path("https://huggingface.co/microsoft/kosmos-2.5/resolve/main/receipt_00008.png")
        processor = AutoProcessor.from_pretrained("microsoft/kosmos-2.5")
        texts = ["<md>", "<ocr>"]
        expected_input_ids = [
            [100288],
            [100282],
        ]
        expected_attention_mask = [[1], [1]]

        image = load_image(url)
        # To match the official (microsoft) Kosmos-2 demo from which the expected values here are grabbed
        image_path = os.path.join(self.tmpdirname, "image.png")
        image.save(image_path)
        image = Image.open(image_path)

        # test single image
        outputs = processor(images=image, text=texts[0])
        self.assertListEqual(
            outputs.input_ids[0].numpy().tolist(),
            [0, 100283] + [0] * 2048 + [100284] + expected_input_ids[0],
        )
        self.assertListEqual(
            outputs.image_embeds_position_mask[0].numpy().tolist(),
            [0, -1] + [1] * 2048 + [-1] + [0] * (len(expected_input_ids[0])),
        )
        self.assertListEqual(
            outputs.attention_mask[0].numpy().tolist(),
            [1, 1] + [1] * 2048 + [1] + expected_attention_mask[0],
        )
        EXPECTED_FP_1 = [
            1.0,
            2.0,
            -2.9527735710144043,
            -2.672085762023926,
            -2.9933173656463623,
            -2.905944585800171,
            -2.5891761779785156,
            -2.8751866817474365,
            -2.962153434753418,
            -2.588062047958374,
        ]
        EXPECTED_FP_200 = [
            4.0,
            45.0,
            1.5713728666305542,
            1.584628939628601,
            1.3589054346084595,
            1.6515952348709106,
            1.7014952898025513,
            1.3731343746185303,
            1.6010395288467407,
            1.6607422828674316,
        ]
        self.assertTupleEqual(outputs.flattened_patches.shape, (1, 4096, 770))
        np.testing.assert_allclose(
            outputs.flattened_patches[0][1][:10].numpy().tolist(),
            EXPECTED_FP_1,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            outputs.flattened_patches[0][200][:10].numpy().tolist(),
            EXPECTED_FP_200,
            atol=1e-4,
        )

        # test a batch of images and texts, right padding
        outputs = processor(images=[image, image], text=texts)
        self.assertListEqual(
            outputs.input_ids[1].numpy().tolist(),
            [0, 100283] + [0] * 2048 + [100284] + expected_input_ids[1],
        )
        self.assertListEqual(
            outputs.image_embeds_position_mask[1].numpy().tolist(),
            [0, -1] + [1] * 2048 + [-1] + [0] * (len(expected_input_ids[1])),
        )
        self.assertListEqual(
            outputs.attention_mask[1].numpy().tolist(),
            [1, 1] + [1] * 2048 + [1] + expected_attention_mask[1],
        )
        self.assertTupleEqual(outputs.flattened_patches.shape, (2, 4096, 770))
        np.testing.assert_allclose(
            outputs.flattened_patches[1][1][:10].numpy().tolist(),
            EXPECTED_FP_1,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            outputs.flattened_patches[1][200][:10].numpy().tolist(),
            EXPECTED_FP_200,
            atol=1e-4,
        )
