# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

from transformers import (
    FuyuImageProcessor,
    FuyuProcessor,
    is_torch_available,
)
from transformers.image_utils import load_image
from transformers.testing_utils import require_torch, require_vision

from ...test_processing_common import MODALITY_TEST_SPECS, ProcessorTesterMixin, url_to_local_path


if is_torch_available():
    import torch


@require_torch
@require_vision
class FuyuProcessingTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = FuyuProcessor
    model_id = "adept/fuyu-8b"
    # Fuyu uses a tokenizer with a very large vocabulary (~262K tokens), making tests slow and
    # memory-intensive. tiny_model_id points to a trimmed tokenizer repo to keep tests lightweight.
    tiny_model_id = "hf-internal-testing/tiny-processor-fuyu"
    images_input_name = "image_patches"

    images_text_kwargs_max_length = 22
    images_text_kwargs_override_max_length = 22
    images_unstructured_max_length = 22

    @classmethod
    def _setup_test_attributes(cls, processor):
        cls.text_prompt = "Generate a coco-style caption.\\n"
        bus_image_url = url_to_local_path(
            "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/bus.png"
        )
        cls.bus_image_pil = load_image(bus_image_url)

    @unittest.skip("FuyuProcessor doesn't return typical pixel values for images")
    def test_processor_with_multiple_inputs(self):
        pass

    def test_get_num_vision_tokens(self):
        "Tests general functionality of the helper used internally in vLLM"

        processor = self.get_processor()

        output = processor._get_num_multimodal_tokens(image_sizes=[(100, 100), (300, 100), (500, 30)])
        self.assertTrue("num_image_tokens" in output)
        self.assertEqual(len(output["num_image_tokens"]), 3)

        self.assertTrue("num_image_patches" in output)
        self.assertEqual(len(output["num_image_patches"]), 3)

    def test_fuyu_processing(self):
        """
        Test to ensure that the standard processing on a gold example matches adept's code.
        """
        # fmt: off
        EXPECTED_PADDED_UNPACKED_TOKEN_INPUTS = torch.Tensor([[71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71019, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71019, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71019, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71019, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71019, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71019, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71019, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71019, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71019, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71019, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71019, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71019, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71019, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71019, 1, 128340, 71374, 71389, 120412, 71377, 71835, 71374, 73615, 71375, 71399, 71435, 71122,]]).to(torch.int64)

        processor = self.get_processor(use_tiny_ckpt=False)
        one_image_bus_model_inputs = processor(text=self.text_prompt, images=self.bus_image_pil)

        # fmt: on
        torch.testing.assert_close(one_image_bus_model_inputs["input_ids"], EXPECTED_PADDED_UNPACKED_TOKEN_INPUTS)

    def test_fuyu_processing_no_image(self):
        """
        Test to check processor works with just text input
        """
        processor_outputs = self.get_processor()(text=self.text_prompt)
        tokenizer_outputs = self.get_component("tokenizer")(self.text_prompt)
        self.assertEqual(processor_outputs["input_ids"], tokenizer_outputs["input_ids"])

    def test_fuyu_processing_multiple_image_sample(self):
        """
        Test to check processor works with multiple image inputs for a single text input
        """
        # fmt: off
        SINGLE_PADDED_UNPACKED_TOKEN_INPUTS = torch.Tensor([[71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71019, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71019, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71019, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71019, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71019, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71019, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71019, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71019, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71019, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71019, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71019, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71019, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71019, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71019, 1, 128340, 71374, 71389, 120412, 71377, 71835, 71374, 73615, 71375, 71399, 71435, 71122,]]).to(torch.int64)
        SINGLE_RESIZED_PADDED_UNPACKED_TOKEN_INPUTS = torch.Tensor([[ 71011,  71011,  71011,  71019,  71011,  71011,  71011,  71019,  71011, 71011,  71011,  71019,  71011,  71011,  71011,  71019,  71011,  71011, 71011,  71019,  71011,  71011,  71011,  71019,  71011,  71011,  71011, 71019,  71011,  71011,  71011,  71019,  71011,  71011,  71011,  71019, 71011,  71011,  71011,  71019,      1, 128340,  71374,  71389, 120412, 71377,  71835,  71374,  73615,  71375,  71399,  71435,  71122]])
        # fmt: on

        # Load once and reuse across all assertions in this test to avoid repeatedly loading the
        # full processor (which carries the large 262K-vocab tokenizer).
        processor = self.get_processor(use_tiny_ckpt=False)

        # Batch of two images - equally sized
        images = [self.bus_image_pil, self.bus_image_pil]
        processor_outputs = processor(
            text=[self.text_prompt, self.text_prompt],
            images=images,
            return_tensors="pt",
        )

        # Processes single images with different sizes as expected
        images = [self.bus_image_pil]
        processor_outputs = processor(text=self.text_prompt, images=images)
        self.assertTrue((processor_outputs["input_ids"] == SINGLE_PADDED_UNPACKED_TOKEN_INPUTS).all())

        images = [self.bus_image_pil.resize((64, 300))]
        processor_outputs = processor(text=self.text_prompt, images=images)
        self.assertTrue((processor_outputs["input_ids"] == SINGLE_RESIZED_PADDED_UNPACKED_TOKEN_INPUTS).all())

        # Batch of two images - different sizes. Left-pads the smaller image inputs
        images = [self.bus_image_pil, self.bus_image_pil.resize((64, 300))]
        processor_outputs = processor(text=[self.text_prompt, self.text_prompt], images=images)

        padding_len_token = (
            SINGLE_PADDED_UNPACKED_TOKEN_INPUTS.shape[1] - SINGLE_RESIZED_PADDED_UNPACKED_TOKEN_INPUTS.shape[1]
        )
        padded_single_resized_padded_unpacked_token_inputs = torch.cat(
            [torch.zeros([1, padding_len_token]), SINGLE_RESIZED_PADDED_UNPACKED_TOKEN_INPUTS], dim=1
        )
        expected_padded_unpacked_token_inputs = torch.cat(
            [SINGLE_PADDED_UNPACKED_TOKEN_INPUTS, padded_single_resized_padded_unpacked_token_inputs], dim=0
        )
        self.assertTrue((processor_outputs["input_ids"] == expected_padded_unpacked_token_inputs).all())

    # Rewrite as Fuyu supports tokenizer kwargs only when image is None.
    def _test_unstructured_kwargs_batched(self, modality):
        attributes = self.processor_class.get_attributes()
        processor = self.get_processor()
        self.maybe_skip_typed_test_for_modality(modality, attributes, processor)

        input_str = self.prepare_text_inputs(batch_size=2, modalities="image")
        modal_input = self._prepare_modality_input(modality, batch_size=2)
        max_length = 76  # just hardcode
        init_time_kwargs = MODALITY_TEST_SPECS[modality]["init_time_kwargs"]
        call_kwargs = MODALITY_TEST_SPECS[modality]["call_time_kwargs"]

        inputs = processor(
            text=input_str,
            max_length=max_length,
            padding="longest",
            images=modal_input,
            **call_kwargs,
            **init_time_kwargs,
        )

        self._check_modality_outputs(inputs, modality)
        self.assertTrue(
            len(inputs[self.text_input_name][0]) == len(inputs[self.text_input_name][1])
            and len(inputs[self.text_input_name][1]) < max_length
        )

    def test_processor_text_has_no_visual(self):
        # Overwritten: Fuyu has a complicated processing so we don't check id values
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
        self.assertTrue(self.text_input_name in inputs_nested)

    def test_get_num_multimodal_tokens_matches_processor_call(self):
        "Tests that the helper used internally in vLLM works correctly"

        # Override -> model siltently ignores multiimage and processes one image per sample
        processor = self.get_processor()

        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

        image_sizes = [(100, 100), (300, 100), (500, 30), (213, 167)]
        image_inputs = []
        for h, w in image_sizes:
            image_inputs.append(np.random.randint(255, size=(h, w, 3), dtype=np.uint8))

        image_token = getattr(self, "image_token", "")
        text = [f"This is an image {image_token}"] * len(image_inputs)
        inputs = processor(
            text=text, images=image_inputs, padding=True, return_mm_token_type_ids=True, return_tensors="pt"
        )

        num_image_tokens_from_call = inputs.mm_token_type_ids.sum(-1).tolist()
        num_image_tokens_from_helper = processor._get_num_multimodal_tokens(image_sizes=image_sizes)
        self.assertListEqual(num_image_tokens_from_call, num_image_tokens_from_helper["num_image_tokens"])


@require_torch
class TestProcessImagesForModelInput(unittest.TestCase):
    def setUp(self):
        """
        Adding a mix of present and absent images.
        """

        self.image_input = torch.randn([1, 1, 3, 64, 64])
        self.image_present = torch.tensor([[1]])
        self.image_unpadded_h = torch.tensor([[45]])  # Adjusted for subsequence of 1
        self.image_unpadded_w = torch.tensor([[50]])  # Adjusted for subsequence of 1
        self.image_patch_dim_h = 16
        self.image_patch_dim_w = 16
        self.image_placeholder_id = 999
        self.image_newline_id = 888
        self.variable_sized = True
        self.image_processor = FuyuImageProcessor(
            patch_size={"height": self.image_patch_dim_h, "width": self.image_patch_dim_w}
        )

    def test_process_images_for_model_input_fixed_sized(self):
        self.variable_sized = False
        result = self.image_processor.preprocess_with_tokenizer_info(
            image_input=self.image_input,
            image_present=self.image_present,
            image_unpadded_h=self.image_unpadded_h,
            image_unpadded_w=self.image_unpadded_w,
            image_placeholder_id=self.image_placeholder_id,
            image_newline_id=self.image_newline_id,
            variable_sized=self.variable_sized,
        )
        self.assertEqual(result["images"][0][0].shape, torch.Size([3, 64, 64]))
