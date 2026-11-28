# Copyright 2026 Dustin Loring and the HuggingFace Inc. team. All rights reserved.
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

import tempfile
import unittest
from unittest.mock import patch

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, BlueberryConfig
from transformers import initialization as init
from transformers.testing_utils import require_torch, slow
from transformers.utils import is_torch_available

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import BlueberryForCausalLM, BlueberryModel, BlueberryPreTrainedModel
    from transformers.models.blueberry.modeling_blueberry import BlueberryAttention


class BlueberryModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = BlueberryModel

    small_config_overrides = {"num_unique_layers": 2, "loop_repeats": 1, "loop_gate": False, "hidden_act": "relu2"}


@require_torch
class BlueberryModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = BlueberryModelTester

    def setUp(self):
        super().setUp()
        if self._testMethodName.startswith("test_model_rope_scaling_from_config_"):
            # The reference initializes residual projections to zero, so activate attention for these comparisons.
            def init_attention_weights(model, module):
                BlueberryPreTrainedModel._init_weights(model, module)
                if isinstance(module, BlueberryAttention):
                    init.normal_(module.o_proj.weight, std=0.02)

            self.enterContext(patch.object(BlueberryModel, "_init_weights", init_attention_weights))

    def _get_conv_state_shape(self, batch_size, config):
        return (batch_size, config.hidden_size, 1)


@require_torch
class BlueberryLoopTest(unittest.TestCase):
    def setUp(self):
        self.config = BlueberryConfig(
            vocab_size=65,
            hidden_size=32,
            num_unique_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=192,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )

    def test_shared_blocks_and_auto_round_trip(self):
        model = AutoModelForCausalLM.from_config(self.config).eval()
        self.assertIsInstance(model, BlueberryForCausalLM)
        self.assertIsInstance(AutoModel.from_config(self.config), BlueberryModel)
        self.assertEqual(len(model.model.layers), 4)
        self.assertEqual(model.model.layer_indices, (0, 1, 2, 1, 2, 1, 2, 3))

        ids = torch.tensor([[1, 3, 5, 7]])
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            self.assertIsInstance(AutoConfig.from_pretrained(directory), BlueberryConfig)
            restored = AutoModelForCausalLM.from_pretrained(directory)
            with torch.no_grad():
                torch.testing.assert_close(restored(ids).logits, model(ids).logits)

    def test_chunked_decoding_matches_full_forward(self):
        model = BlueberryForCausalLM(self.config).eval()
        ids = torch.randint(3, self.config.vocab_size, (2, 145), generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            model.model.smear_lambda.fill_(0.5)
            model.model.loop_gate.weight.normal_(std=0.1)
            for layer in model.model.layers:
                layer.self_attn.o_proj.weight.normal_(std=0.1)
                layer.mlp.fc2.weight.normal_(std=0.1)

            expected = model(ids, use_cache=False).logits
            prefill = model(ids[:, :132], use_cache=True)
            decoded = model(ids[:, 132:], cache=prefill.cache, use_cache=True)

        torch.testing.assert_close(decoded.logits, expected[:, 132:], atol=1e-6, rtol=1e-5)
        self.assertEqual(len(decoded.cache), self.config.num_hidden_layers)
        self.assertEqual(decoded.cache.get_seq_length(), ids.shape[1])

    def test_resize_updates_value_embeddings(self):
        model = BlueberryForCausalLM(self.config).eval()
        model.resize_token_embeddings(67, mean_resizing=False)
        with torch.no_grad():
            logits = model(torch.tensor([[65, 66]])).logits
        self.assertEqual(logits.shape[-1], 67)
        self.assertTrue(all(embedding.num_embeddings == 67 for embedding in model.model.value_embeds.values()))

    def test_generation_with_left_padding(self):
        model = BlueberryForCausalLM(self.config).eval()
        ids = torch.tensor([[0, 0, 1, 3, 5], [1, 4, 7, 9, 6]])
        mask = ids.ne(0)
        generation_args = {"min_new_tokens": 4, "max_new_tokens": 4}
        with torch.no_grad():
            model.model.smear_lambda.fill_(0.5)
            for layer in model.model.layers:
                layer.self_attn.o_proj.weight.normal_(std=0.1)
            for implementation in ("dynamic", "static"):
                with self.subTest(implementation=implementation):
                    actual = model.generate(
                        ids, attention_mask=mask, cache_implementation=implementation, **generation_args
                    )
                    expected = model.generate(ids[0:1, 2:], use_cache=False, **generation_args)
                    torch.testing.assert_close(actual[0, 5:], expected[0, 3:])


@require_torch
class BlueberryIntegrationTest(unittest.TestCase):
    @slow
    def test_published_checkpoint_loads_natively(self):
        repo_id = "CrowdMind/Blueberry-base"
        revision = "12c9d5cb5e2ee1b612d8c3448891a9868f774fec"
        tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(repo_id, revision=revision, trust_remote_code=False).eval()
        self.assertIsInstance(model, BlueberryForCausalLM)

        inputs = tokenizer("The first blueberries of the season", return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs, use_cache=False).logits[0, -1, :8]
        expected = torch.tensor([-13.6788, -13.6735, -13.6710, -13.6770, -13.6711, -13.6747, -13.6741, -13.6758])
        torch.testing.assert_close(logits, expected, atol=1e-3, rtol=0)
