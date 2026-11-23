# Copyright 2026 NAVER Corp. and The HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch HyperCLOVAX Vision model."""

import tempfile
import unittest

import pytest

from transformers import (
    AutoProcessor,
    HyperCLOVAXVisionConfig,
    HyperCLOVAXVisionForConditionalGeneration,
    HyperCLOVAXVisionModel,
    is_torch_available,
)
from transformers.testing_utils import Expectations, cleanup, require_torch, slow, torch_device

from ...test_image_processing_common import load_test_image
from ...test_modeling_common import floats_tensor
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch
    from safetensors.torch import load_file


class HyperCLOVAXVisionVisionText2TextModelTester(VLMModelTester):
    base_model_class = HyperCLOVAXVisionModel
    config_class = HyperCLOVAXVisionConfig
    conditional_generation_class = HyperCLOVAXVisionForConditionalGeneration

    # One global view and one local grid of 16x16 pixels, each reduced to 2x2 visual tokens by the projector:
    # 4 (global view) + 2 * (2 + 1) (local grid, one newline per row) = 10 image tokens
    small_config_overrides = {
        "image_size": 16,
        "patch_size": 4,
        "projector_num_queries": 4,
        "num_patches_per_image": 2,
        "num_image_tokens": 10,
        "image_token_id": 3,
    }

    def create_pixel_values(self, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        return floats_tensor(
            [batch_size, self.num_patches_per_image, self.num_channels, self.image_size, self.image_size]
        )

    def get_additional_inputs(self, config, input_ids, pixel_values, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        return {"image_sizes": torch.tensor([[self.image_size, self.image_size]] * batch_size)}

    def get_config(self):
        config = super().get_config()
        config.image_grid_pinpoints = [[self.image_size, self.image_size]]
        return config


@require_torch
class HyperCLOVAXVisionForConditionalGenerationModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = HyperCLOVAXVisionVisionText2TextModelTester
    skip_test_image_features_output_shape = True

    @pytest.mark.xfail(reason="This architecture seems to not compute gradients for some layer.")
    def test_training_gradient_checkpointing(self):
        super().test_training_gradient_checkpointing()

    @pytest.mark.xfail(reason="This architecture seems to not compute gradients for some layer.")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        super().test_training_gradient_checkpointing_use_reentrant_false()

    @pytest.mark.xfail(reason="This architecture seems to not compute gradients for some layer.")
    def test_training_gradient_checkpointing_use_reentrant_true(self):
        super().test_training_gradient_checkpointing_use_reentrant_true()

    def test_reverse_loading_mapping(self, check_keys_were_modified=True):
        # The original checkpoints have the weights of the model with head, so the mapping only applies to it
        super().test_reverse_loading_mapping(check_keys_were_modified, skip_base_model=True)

    def test_load_original_checkpoint_format(self):
        """The original checkpoints use `model_type="hyperclovax_vlm"` and their own weight names."""
        config = self.model_tester.get_config()
        model = HyperCLOVAXVisionForConditionalGeneration(config).eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            # Saving reverts the weight renaming, so the checkpoint is written with the original names
            state_dict = load_file(f"{tmpdir}/model.safetensors")
            self.assertIn("image_newline", state_dict)
            self.assertIn("mm_projector.pos_emb", state_dict)
            self.assertIn("mm_projector.net.0.b1.conv1.conv.weight", state_dict)
            self.assertIn("mm_projector.net.2.b3.conv3.bn.weight", state_dict)
            self.assertIn("mm_projector.readout.2.weight", state_dict)
            self.assertTrue(any(key.startswith("language_model.model.layers.") for key in state_dict))
            self.assertTrue(any(key.startswith("vision_model.") for key in state_dict))

            config_dict = config.to_dict()
            config_dict["model_type"] = "hyperclovax_vlm"
            config_dict["use_nth_layer"] = config_dict.pop("vision_feature_layer")
            config_dict["num_queries_vis_abstractor_image"] = config_dict.pop("projector_num_queries")
            original_config = HyperCLOVAXVisionConfig.from_dict(config_dict)
            self.assertEqual(original_config.model_type, "hyperclovax_vision")
            self.assertEqual(original_config.vision_feature_layer, config.vision_feature_layer)
            self.assertEqual(original_config.projector_num_queries, config.projector_num_queries)

            reloaded = HyperCLOVAXVisionForConditionalGeneration.from_pretrained(tmpdir, config=original_config)

        for (name, param), (_, reloaded_param) in zip(model.state_dict().items(), reloaded.state_dict().items()):
            torch.testing.assert_close(param, reloaded_param, msg=f"Mismatch for {name}")


@slow
@require_torch
class HyperCLOVAXVisionForConditionalGenerationIntegrationTest(unittest.TestCase):
    model_id = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"

    def setUp(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-coco/resolve/main/val2017/000000039769.jpg"
        self.image = load_test_image(url)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_small_model_integration_test(self):
        model = HyperCLOVAXVisionForConditionalGeneration.from_pretrained(
            self.model_id, dtype=torch.bfloat16, device_map=torch_device
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image in one sentence."},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(images=self.image, text=prompt, return_tensors="pt").to(torch_device, torch.bfloat16)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        decoded = self.processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        EXPECTED_DECODED_TEXT = Expectations(
            {
                ("cpu", None): "Two cats are sleeping on a pink blanket with two remote controls next to them.",
            }
        ).get_expectation()
        self.assertEqual(decoded, EXPECTED_DECODED_TEXT)
