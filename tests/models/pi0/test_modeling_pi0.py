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
"""Testing suite for the PyTorch PI0 model."""

import unittest

from transformers import AutoTokenizer, PI0Config, PI0ForConditionalGeneration, is_torch_available
from transformers.testing_utils import require_torch, slow

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch


class PI0ModelTester:
    def __init__(self, parent):
        self.parent = parent
        self.is_training = True
        self.batch_size = 2
        self.seq_length = 5
        self.num_cameras = 2
        self.image_size = 8
        self.patch_size = 4
        self.num_channels = 3
        self.vocab_size = 128
        self.hidden_size = 16
        self.num_hidden_layers = 2
        self.num_attention_heads = 2
        self.chunk_size = 4
        self.max_state_dim = 8
        self.max_action_dim = 8

    def get_config(self):
        return PI0Config(
            image_token_index=127,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            projection_dim=self.hidden_size,
            chunk_size=self.chunk_size,
            max_state_dim=self.max_state_dim,
            max_action_dim=self.max_action_dim,
            num_inference_steps=3,
            vision_config={
                "model_type": "siglip_vision_model",
                "hidden_size": self.hidden_size,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "patch_size": self.patch_size,
                "image_size": self.image_size,
                "vision_use_head": False,
                "num_channels": self.num_channels,
            },
            text_config={
                "model_type": "gemma",
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "intermediate_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 8,
            },
            expert_config={
                "model_type": "gemma",
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "intermediate_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 8,
            },
        )

    def prepare_config_and_inputs_for_common(self):
        config = self.get_config()
        pixel_values = floats_tensor(
            [self.batch_size, self.num_cameras, self.num_channels, self.image_size, self.image_size]
        )
        image_masks = torch.tensor([[True, True], [True, False]], dtype=torch.bool)
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size - 1)
        attention_mask = torch.ones(self.batch_size, self.seq_length, dtype=torch.long)
        state = floats_tensor([self.batch_size, self.max_state_dim])
        actions = floats_tensor([self.batch_size, self.chunk_size, self.max_action_dim])
        noise = floats_tensor([self.batch_size, self.chunk_size, self.max_action_dim])
        timestep = torch.tensor([0.5, 0.8], dtype=torch.float32)

        inputs_dict = {
            "pixel_values": pixel_values,
            "image_masks": image_masks,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "state": state,
            "actions": actions,
            "noise": noise,
            "timestep": timestep,
        }
        return config, inputs_dict


@require_torch
class PI0ForConditionalGenerationModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (PI0ForConditionalGeneration,) if is_torch_available() else ()
    test_pruning = False
    test_head_masking = False
    test_torchscript = False
    has_attentions = True
    _is_composite = True

    def setUp(self):
        self.model_tester = PI0ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=PI0Config, has_text_modality=False)

    def test_config(self):
        config = self.model_tester.get_config()
        self.assertIsInstance(config, PI0Config)
        self.assertEqual(config.model_type, "pi0")
        self.assertEqual(config.expert_config.hidden_size, self.model_tester.hidden_size)

    def test_model_forward(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = PI0ForConditionalGeneration(config).eval()
        with torch.no_grad():
            outputs = model(**inputs_dict)
        self.assertEqual(outputs.loss_per_sample.shape, (2, config.chunk_size, config.max_action_dim))


@require_torch
class PI0ModelSmokeTest(unittest.TestCase):
    def test_full_run_smoke(self):
        torch.manual_seed(0)
        tester = PI0ModelTester(self)
        config, inputs_dict = tester.prepare_config_and_inputs_for_common()
        model = PI0ForConditionalGeneration(config).eval()

        with torch.no_grad():
            outputs = model(**inputs_dict)
        self.assertIsNotNone(outputs.loss)
        self.assertEqual(outputs.loss.ndim, 0)

        sample_inputs = {k: v for k, v in inputs_dict.items() if k != "actions"}
        with torch.no_grad():
            sampled_actions = model.sample_actions(
                pixel_values=sample_inputs["pixel_values"],
                image_masks=sample_inputs["image_masks"],
                input_ids=sample_inputs["input_ids"],
                attention_mask=sample_inputs["attention_mask"],
                state=sample_inputs["state"],
                noise=sample_inputs["noise"].clone(),
                num_steps=2,
            )
        self.assertEqual(sampled_actions.shape, (2, config.chunk_size, config.max_action_dim))
        self.assertTrue(torch.isfinite(sampled_actions).all())


@require_torch
@slow
class PI0ModelIntegrationTest(unittest.TestCase):
    def test_pi0_base_reference_values(self):
        model = PI0ForConditionalGeneration.from_pretrained("lerobot/pi0_base", torch_dtype=torch.float32).eval()
        tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

        torch.manual_seed(42)
        state = torch.randn(1, 32)
        noisy_actions = torch.randn(1, 50, 32)
        timestep = torch.tensor([0.5], dtype=torch.float32)

        with torch.no_grad():
            suffix_embs, suffix_masks = model.embed_suffix(state, noisy_actions, timestep)
        self.assertEqual(suffix_embs.shape, (1, 51, 1024))
        self.assertEqual(suffix_masks.shape, (1, 51))
        # Aggregate mean shows small (~1e-3) drift across envs; keep token-level checks stricter below.
        self.assertAlmostEqual(suffix_embs.mean().item(), -0.10177, delta=0.002)
        torch.testing.assert_close(
            suffix_embs[0, 0, :4], torch.tensor([-0.7092, -0.5197, -0.7360, -2.2933]), atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            suffix_embs[0, -1, :4], torch.tensor([1.3611, -1.9470, 1.2340, -1.8429]), atol=1e-4, rtol=1e-4
        )

        tokenized = tokenizer(
            ["Pick up the object\n"], padding="max_length", truncation=True, max_length=48, return_tensors="pt"
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        pixel_values = torch.randn(1, 1, 3, 224, 224)
        image_masks = torch.tensor([[True]])

        with torch.no_grad():
            prefix_embs, prefix_masks = model.embed_prefix(pixel_values, input_ids, attention_mask, image_masks)
        self.assertEqual(prefix_embs.shape, (1, 304, 2048))
        self.assertEqual(prefix_masks.shape, (1, 304))
        self.assertAlmostEqual(prefix_embs.mean().item(), 0.02125, places=4)
        torch.testing.assert_close(
            prefix_embs[0, 0, :4], torch.tensor([2.6215, -0.2010, -0.0071, -0.0147]), atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            prefix_embs[0, -1, :4], torch.tensor([-8.9272, -0.7623, 0.4806, -1.4695]), atol=1e-4, rtol=1e-4
        )

        actions = torch.randn(1, 50, 32)
        noise = torch.randn(1, 50, 32)
        with torch.no_grad():
            outputs = model(
                pixel_values=pixel_values,
                image_masks=image_masks,
                input_ids=input_ids,
                attention_mask=attention_mask,
                state=state,
                actions=actions,
                noise=noise,
                timestep=timestep,
            )
        self.assertEqual(outputs.loss_per_sample.shape, (1, 50, 32))
        self.assertAlmostEqual(outputs.loss.item(), 3.8787, places=3)

        with torch.no_grad():
            sampled = model.sample_actions(
                pixel_values=pixel_values,
                image_masks=image_masks,
                input_ids=input_ids,
                attention_mask=attention_mask,
                state=state,
                num_steps=3,
            )
        self.assertEqual(sampled.shape, (1, 50, 32))
        self.assertAlmostEqual(sampled.mean().item(), -0.0617, places=3)
        self.assertAlmostEqual(sampled.std().item(), 0.2745, places=3)
        torch.testing.assert_close(
            sampled[0, 0, :4], torch.tensor([-0.1905, -0.5732, -0.5487, 0.6403]), atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            sampled[0, -1, :4], torch.tensor([-0.0038, 0.0003, -0.0060, -0.0001]), atol=1e-4, rtol=1e-4
        )
