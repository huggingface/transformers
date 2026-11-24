# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Phi-3-vision model."""

import unittest

import pytest

from transformers import (
    AutoProcessor,
    Phi3VConfig,
    Phi3VForConditionalGeneration,
    Phi3VModel,
    is_torch_available,
)
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_torch,
    slow,
    torch_device,
)

from ...test_image_processing_common import load_test_image
from ...test_modeling_common import floats_tensor
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch


class Phi3VVisionText2TextModelTester(VLMModelTester):
    base_model_class = Phi3VModel
    config_class = Phi3VConfig
    conditional_generation_class = Phi3VForConditionalGeneration

    # One 16x32 image: a global crop and 1x2 crops of 16x16, padded to 4 crops.
    # Each crop has 4x4 patches, merged to 2x2 tokens: 1 * 2 * (2 * 2 + 1) + 1 + 2 * (2 + 1) = 17 tokens
    small_config_overrides = {"image_size": 16, "num_crops": 4, "num_image_tokens": 17}

    def create_pixel_values(self, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        return floats_tensor([batch_size, self.num_crops, self.num_channels, self.image_size, self.image_size])

    def get_additional_inputs(self, config, input_ids, pixel_values, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        return {"image_sizes": torch.tensor([[self.image_size, 2 * self.image_size]] * batch_size)}


@require_torch
class Phi3VForConditionalGenerationModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = Phi3VVisionText2TextModelTester
    skip_test_image_features_output_shape = True

    @pytest.mark.xfail(reason="The vision tower's post_layernorm is not used.")
    def test_training_gradient_checkpointing(self):
        super().test_training_gradient_checkpointing()

    @pytest.mark.xfail(reason="The vision tower's post_layernorm is not used.")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        super().test_training_gradient_checkpointing_use_reentrant_false()

    @pytest.mark.xfail(reason="The vision tower's post_layernorm is not used.")
    def test_training_gradient_checkpointing_use_reentrant_true(self):
        super().test_training_gradient_checkpointing_use_reentrant_true()

    def test_reverse_loading_mapping(self):
        # The Hub checkpoints are saved from the model with head, the mapping is written for that layout
        super().test_reverse_loading_mapping(skip_base_model=True)


@require_torch
class Phi3VForConditionalGenerationIntegrationTest(unittest.TestCase):
    model_id = "microsoft/Phi-3.5-vision-instruct"

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_small_model_integration_test(self):
        processor = AutoProcessor.from_pretrained(self.model_id)
        model = Phi3VForConditionalGeneration.from_pretrained(
            self.model_id, dtype=torch.bfloat16, device_map=torch_device
        )
        url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-coco/resolve/main/val2017/000000039769.jpg"
        image = load_test_image(url)
        prompt = "<|user|>\n<|image|>\nWhat is shown in this image?<|end|>\n<|assistant|>\n"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        decoded = processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        EXPECTED_DECODED_TEXT = Expectations(
            {(None, None): "The image displays two cats lying on a pink couch with remote controls beside them."}
        ).get_expectation()
        self.assertEqual(decoded, EXPECTED_DECODED_TEXT)
