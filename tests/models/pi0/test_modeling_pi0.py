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

import unittest

import torch

from transformers import AutoTokenizer, PI0Config, PI0ForConditionalGeneration
from transformers.testing_utils import require_torch, slow


@require_torch
class PI0ModelSmokeTest(unittest.TestCase):
    def _get_tiny_config(self):
        return PI0Config(
            image_token_index=127,
            vocab_size=128,
            hidden_size=16,
            projection_dim=16,
            chunk_size=4,
            max_state_dim=8,
            max_action_dim=8,
            num_inference_steps=3,
            vision_config={
                "model_type": "siglip_vision_model",
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "patch_size": 4,
                "image_size": 8,
                "vision_use_head": False,
            },
            text_config={
                "model_type": "gemma",
                "vocab_size": 128,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 8,
            },
            expert_config={
                "model_type": "gemma",
                "vocab_size": 128,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 8,
            },
        )

    def test_full_run_smoke(self):
        torch.manual_seed(0)

        config = self._get_tiny_config()
        model = PI0ForConditionalGeneration(config).eval()

        batch_size = 2
        seq_len = 6
        num_cameras = 2

        pixel_values = torch.randn(batch_size, num_cameras, 3, 8, 8)
        image_masks = torch.tensor([[True, True], [True, False]])
        input_ids = torch.randint(low=0, high=config.text_config.vocab_size, size=(batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        state = torch.randn(batch_size, config.max_state_dim)
        actions = torch.randn(batch_size, config.chunk_size, config.max_action_dim)
        noise = torch.randn_like(actions)
        timestep = torch.tensor([0.5, 0.8], dtype=torch.float32)

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

        self.assertIsNotNone(outputs.loss)
        self.assertEqual(outputs.loss.ndim, 0)
        self.assertEqual(outputs.loss_per_sample.shape, (batch_size, config.chunk_size, config.max_action_dim))

        with torch.no_grad():
            sampled_actions = model.sample_actions(
                pixel_values=pixel_values,
                image_masks=image_masks,
                input_ids=input_ids,
                attention_mask=attention_mask,
                state=state,
                noise=noise.clone(),
                num_steps=2,
            )

        self.assertEqual(sampled_actions.shape, (batch_size, config.chunk_size, config.max_action_dim))
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
        # small aggregate drift (~1e-3) can happen across environments; token slices are checked below.
        self.assertAlmostEqual(suffix_embs.mean().item(), -0.10177, places=3)
        torch.testing.assert_close(
            suffix_embs[0, 0, :4], torch.tensor([-0.7092, -0.5197, -0.7360, -2.2933]), atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            suffix_embs[0, -1, :4], torch.tensor([1.3611, -1.9470, 1.2340, -1.8429]), atol=1e-4, rtol=1e-4
        )

        tokenized = tokenizer(
            ["Pick up the object\n"],
            padding="max_length",
            truncation=True,
            max_length=48,
            return_tensors="pt",
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
