# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch PI0.5 model."""

import unittest

from datasets import load_dataset

from transformers import PI05Config, PI05Processor, is_torch_available
from transformers.testing_utils import (
    Expectations,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...test_configuration_common import ConfigTester
from ..pi0 import test_modeling_pi0
from ..pi0.test_modeling_pi0 import PI0ModelTester


if is_torch_available():
    import torch

    from transformers import PI05ForConditionalGeneration


class PI05ModelTester(PI0ModelTester):
    def __init__(self, parent):
        super().__init__(parent)
        self.encoder_seq_length = self.chunk_size
        self.key_length = self.encoder_seq_length + self.seq_length

    def get_config(self):
        return PI05Config(
            dit_config=self.dit_config,
            vlm_config={
                "vision_config": self.vision_config,
                "text_config": self.text_config,
                "image_token_index": self.image_token_index,
                "projection_dim": self.hidden_size,
            },
            chunk_size=self.chunk_size,
            max_action_dim=self.max_action_dim,
            num_inference_steps=self.num_inference_steps,
        )

    def prepare_config_and_inputs_for_common(self):
        _, inputs_dict = super().prepare_config_and_inputs_for_common()
        del inputs_dict["state"]
        return self.get_config(), inputs_dict


@require_torch
class PI05ForConditionalGenerationModelTest(test_modeling_pi0.PI0ForConditionalGenerationModelTest):
    all_model_classes = (PI05ForConditionalGeneration,) if is_torch_available() else ()
    additional_model_inputs = ["attention_mask", "actions", "noise", "timestep"]

    def setUp(self):
        self.model_tester = PI05ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=PI05Config, has_text_modality=False)


LIBERO_TASK = "put the white mug on the left plate and put the yellow and white mug on the right plate"
LIBERO_MEAN_STD = {
    "state_mean": [-0.046518784, 0.03440907, 0.7645525, 2.9722095, -0.22046979, -0.1255794, 0.026914254, -0.027190784],
    "state_std": [0.10494395, 0.1517662, 0.37851673, 0.34427372, 0.90694684, 0.32539192, 0.014175903, 0.0140588945],
    "actions_mean": [0.062781565, 0.08684081, -0.09037306, 0.00054074306, 0.00564338, -0.0052290987, -0.049640723],
    "actions_std": [0.33552372, 0.378447, 0.4447286, 0.03924354, 0.06339297, 0.077970274, 0.99876714],
}
LIBERO_QUANTILES = {
    "state_q01": [-0.18941794, -0.13594238, 0.63601464, 2.7590623, -0.7064822, -0.41491947, 0.010922795, -0.039982475],
    "state_q99": [0.047856543, 0.2223633, 0.88348854, 3.192119, 0.35816693, 0.17675057, 0.03997588, -0.011350016],
    "actions_q01": [-0.5351996, -0.603894, -0.77835274, -0.074094005, -0.116252445, -0.12600179, -1.0],
    "actions_q99": [0.7094644, 0.7320648, 0.8014754, 0.08395454, 0.13927571, 0.10880501, 0.919477],
}


@slow
@require_torch_accelerator
class PI05ModelIntegrationTest(unittest.TestCase):
    def prepare_inputs(self, statistics):
        processor = PI05Processor.from_pretrained("google/paligemma-3b-pt-224", **statistics)
        small_data = load_dataset("RaushanTurganbay/libero-small-testing", split="train")
        first_sample = small_data[0]
        inputs = processor(
            images=[[first_sample["observation.images.image"], first_sample["observation.images.wrist_image"]]],
            text=LIBERO_TASK,
            state=first_sample["observation.state"],
            actions=small_data["action"][:50],
            padding="longest",
            return_tensors="pt",
        )
        actions = inputs.pop("actions")
        torch.manual_seed(0)
        noise = torch.randn(1, 50, 32)
        return inputs, actions, noise

    def run_model(self, checkpoint, statistics, dtype):
        inputs, actions, noise = self.prepare_inputs(statistics)
        model, loading_info = PI05ForConditionalGeneration.from_pretrained(
            checkpoint, dtype=dtype, output_loading_info=True
        )
        self.assertEqual(loading_info["missing_keys"], set())
        self.assertEqual(loading_info["unexpected_keys"], set())
        model = model.to(torch_device).eval()
        model.config.loss_reduction = "none"
        inputs = inputs.to(torch_device).to(dtype)
        noise = noise.to(torch_device, dtype)
        with torch.no_grad():
            loss = model(
                **inputs,
                actions=actions.to(torch_device, dtype),
                noise=noise,
                timestep=torch.tensor([0.4], device=torch_device),
            ).loss
            sampled = model.sample_actions(**inputs, noise=noise, num_steps=10)
        self.assertEqual(sampled.shape, (1, 50, 32))
        return loss.float().mean().item(), sampled[0, :2, :7].float().cpu()

    def test_pi05_libero_finetuned_float32(self):
        loss, sampled = self.run_model("lerobot/pi05_libero_finetuned_v044", LIBERO_MEAN_STD, torch.float32)
        expected_slices = Expectations(
            {
                ("cuda", 8): [
                    [-0.1558, -0.4436, 0.1888, -0.2337, -0.0350, 0.0978, -0.9552],
                    [-0.1584, -0.4318, 0.1927, -0.0817, -0.0537, 0.0782, -0.9482],
                ],
            }
        )  # fmt: skip
        self.assertAlmostEqual(loss, 0.086348, places=3)
        torch.testing.assert_close(sampled, torch.tensor(expected_slices.get_expectation()), atol=1e-3, rtol=1e-3)

    def test_pi05_libero_finetuned_bfloat16(self):
        loss, sampled = self.run_model("lerobot/pi05_libero_finetuned_v044", LIBERO_MEAN_STD, torch.bfloat16)
        expected_slices = Expectations(
            {
                ("cuda", (8, 0)): [
                    [-0.1543, -0.4434, 0.1855, -0.2354, -0.0352, 0.0967, -0.9570],
                    [-0.1562, -0.4336, 0.1914, -0.0869, -0.0552, 0.0786, -0.9492],
                ],
                ("cuda", (8, 6)): [
                    [-0.1523, -0.4453, 0.1865, -0.2441, -0.0332, 0.0977, -0.9570],
                    [-0.1553, -0.4355, 0.1934, -0.0942, -0.0552, 0.0791, -0.9492],
                ],
            }
        )  # fmt: skip
        self.assertAlmostEqual(loss, 0.086115, places=2)
        torch.testing.assert_close(sampled, torch.tensor(expected_slices.get_expectation()), atol=5e-3, rtol=5e-3)

    def test_pi05_base_float32(self):
        loss, sampled = self.run_model("lerobot/pi05_base", LIBERO_QUANTILES, torch.float32)
        expected_slices = Expectations(
            {
                ("cuda", 8): [
                    [-0.1276, 0.0243, -0.3314, 0.5690, 0.2550, 0.0592, -0.0338],
                    [-0.1332, 0.0320, -0.3256, 0.5542, 0.2623, 0.0591, -0.0317],
                ],
            }
        )  # fmt: skip
        self.assertAlmostEqual(loss, 0.096026, places=3)
        torch.testing.assert_close(sampled, torch.tensor(expected_slices.get_expectation()), atol=1e-3, rtol=1e-3)

    def test_pi05_base_bfloat16(self):
        loss, sampled = self.run_model("lerobot/pi05_base", LIBERO_QUANTILES, torch.bfloat16)
        expected_slices = Expectations(
            {
                ("cuda", 8): [
                    [-0.1289, 0.0273, -0.3340, 0.5742, 0.2520, 0.0576, -0.0376],
                    [-0.1328, 0.0364, -0.3262, 0.5586, 0.2617, 0.0581, -0.0356],
                ],
            }
        )  # fmt: skip
        self.assertAlmostEqual(loss, 0.096172, places=2)
        torch.testing.assert_close(sampled, torch.tensor(expected_slices.get_expectation()), atol=5e-3, rtol=5e-3)
