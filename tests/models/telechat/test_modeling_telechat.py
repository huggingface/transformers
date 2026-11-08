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
"""Testing suite for the PyTorch TeleChat model."""

import math
import tempfile
import unittest

from huggingface_hub.errors import StrictDataclassClassValidationError
from parameterized import parameterized

from transformers import AutoTokenizer, TeleChatConfig, is_torch_available
from transformers.testing_utils import cleanup, require_torch, require_torch_accelerator, slow, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch
    from safetensors.torch import load_file

    from transformers import TeleChatForCausalLM, TeleChatModel
    from transformers.models.telechat.modeling_telechat import TeleChatRotaryEmbedding


def original_ntk_parameters(config, seq_len):
    """The stepwise NTK scaling of the original code."""
    seq_len = max(seq_len, config.training_seqlen)
    context_value = math.log2(seq_len / config.base_seqlen) + 1
    ntk_alpha = max(2 ** math.ceil(context_value) - 1, 1)
    scale = seq_len / config.training_seqlen
    mscale = 1.0 if scale <= 1 else 0.1 * math.log(scale) + 1.0
    dim = config.head_dim
    base = config.rope_parameters["rope_theta"] * ntk_alpha ** (dim / (dim - 2))
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    return inv_freq, mscale


class TeleChatModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = TeleChatModel


@require_torch
class TeleChatModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = TeleChatModelTester

    def test_reverse_loading_mapping(self):
        # The `transformer.` prefix of the original checkpoints only exists on models with a head
        super().test_reverse_loading_mapping(skip_base_model=True)

    @unittest.skip("TeleChat ignores `rope_scaling` and always applies its own stepwise NTK scaling")
    def test_model_rope_scaling_frequencies(self):
        pass

    @parameterized.expand([("linear",), ("dynamic",), ("yarn",)])
    @unittest.skip("TeleChat ignores `rope_scaling` and always applies its own stepwise NTK scaling")
    def test_model_rope_scaling_from_config(self, scaling_type):
        pass

    @parameterized.expand([(4096, 8192), (8192, 8192), (32768, 32768)])
    def test_ntk_scaling_matches_original(self, base_seqlen, training_seqlen):
        config = TeleChatConfig(base_seqlen=base_seqlen, training_seqlen=training_seqlen)
        for seq_len in [
            1,
            training_seqlen - 1,
            training_seqlen,
            training_seqlen + 1,
            2 * training_seqlen,
            3 * training_seqlen + 7,
        ]:
            expected_inv_freq, expected_mscale = original_ntk_parameters(config, seq_len)
            inv_freq, mscale = TeleChatRotaryEmbedding.compute_ntk_parameters(config, torch.tensor(seq_len))
            torch.testing.assert_close(inv_freq, expected_inv_freq)
            self.assertAlmostEqual(mscale.item(), expected_mscale, places=5)

    def test_original_checkpoint_round_trip(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = TeleChatForCausalLM(config).to(torch_device).eval()
        # Randomise everything, norms included, so a weight that fails to load can't go unnoticed
        for param in model.parameters():
            torch.nn.init.normal_(param)
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            saved = set(load_file(f"{tmpdir}/model.safetensors"))
            # Saved in the layout of the original checkpoints
            for key in [
                "transformer.word_embeddings.weight",
                "transformer.h.0.input_layernorm.weight",
                "transformer.h.0.self_attention.query.weight",
                "transformer.h.0.self_attention.key_value.weight",
                "transformer.h.0.self_attention.dense.bias",
                "transformer.h.0.post_attention_layernorm.weight",
                "transformer.h.0.mlp.down_proj.bias",
                "transformer.ln_f.weight",
            ]:
                self.assertIn(key, saved)
            reloaded = TeleChatForCausalLM.from_pretrained(tmpdir).to(torch_device)
        for (name, param), (_, reloaded_param) in zip(model.named_parameters(), reloaded.named_parameters()):
            torch.testing.assert_close(param, reloaded_param, msg=name)

    def test_original_config_keys(self):
        config = TeleChatConfig(
            n_layer=3, n_head=8, hidden_size=64, ffn_hidden_size=96, layer_norm_epsilon=1e-4, seq_length=1024
        )
        self.assertEqual(config.num_hidden_layers, 3)
        self.assertEqual(config.num_attention_heads, 8)
        self.assertEqual(config.num_key_value_heads, 8)
        self.assertEqual(config.head_dim, 8)
        self.assertEqual(config.intermediate_size, 96)
        self.assertEqual(config.rms_norm_eps, 1e-4)
        self.assertEqual(config.max_position_embeddings, 1024)

    def test_rope_scaling_is_ignored(self):
        # The original code never reads `rope_scaling`, so neither do we
        config = TeleChatConfig(rope_theta=1e6, rope_scaling={"factor": 1.0, "rope_type": "dynamic"})
        self.assertEqual(config.rope_parameters, {"rope_type": "default", "rope_theta": 1e6})

    def test_unsupported_configs_are_rejected(self):
        with self.assertRaises(StrictDataclassClassValidationError):
            TeleChatConfig(embed_layernorm=True)
        with self.assertRaises(StrictDataclassClassValidationError):
            TeleChatConfig(apply_residual_connection_post_layernorm=True)
        with self.assertRaises(StrictDataclassClassValidationError):
            # TeleChat-52B
            TeleChatConfig(n_embd=8192, use_mup=True)


@slow
@require_torch_accelerator
class TeleChatIntegrationTest(unittest.TestCase):
    # The original code was written for fp16
    dtype = torch.float16 if is_torch_available() else None

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_model_3b_logits(self):
        model = TeleChatForCausalLM.from_pretrained("Tele-AI/TeleChat2-3B", device_map="auto", dtype=self.dtype)
        input_ids = torch.tensor([[1, 1404, 1435, 1272, 1363, 1446, 334, 1279, 1306, 395]], device=model.device)
        with torch.no_grad():
            logits = model(input_ids).logits.float().cpu()

        EXPECTED_MEAN = torch.tensor(
            [-4.919, -4.2666, -1.901, -2.9477, -1.5264, -2.575, -0.5755, -1.9635, -2.781, -1.7785]
        )
        EXPECTED_SLICE = torch.tensor(
            [-9.1016, -9.8984, -1.8105, -9.1875, -9.1797, -9.1797, -9.1797, -9.1797, -9.1797, -6.3086]
        )
        torch.testing.assert_close(logits.mean(-1)[0], EXPECTED_MEAN, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(logits[0, 0, :10], EXPECTED_SLICE, rtol=1e-2, atol=1e-2)

    def test_model_3b_generation(self):
        tokenizer = AutoTokenizer.from_pretrained("Tele-AI/TeleChat2-3B")
        model = TeleChatForCausalLM.from_pretrained("Tele-AI/TeleChat2-3B", device_map="auto", dtype=self.dtype)
        inputs = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        EXPECTED_TEXT = "The capital of France is known for its rich history, culture, and stunning architecture. The capital city,"
        self.assertEqual(text, EXPECTED_TEXT)
