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
import unittest

from parameterized import parameterized

from transformers import AutoTokenizer, is_torch_available
from transformers.testing_utils import Expectations, cleanup, require_torch, require_torch_accelerator, slow

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester, torch_device


if is_torch_available():
    import torch

    from transformers import MiMoV2FlashForCausalLM, MiMoV2FlashModel


class MiMoV2FlashModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = MiMoV2FlashModel

    def __init__(self, parent):
        super().__init__(
            parent=parent,
            batch_size=2,
            is_training=True,
            num_attention_heads=4,
            intermediate_size=64,
            hidden_act="silu",
            max_position_embeddings=64,
        )
        # MiMo-V2-Flash specific test config (other defaults match Glm4MoeConfig).
        self.v_head_dim = 16
        self.layer_types = ["full_attention", "sliding_attention"]
        self.rope_parameters = {
            "full_attention": {"rope_type": "default", "rope_theta": 5_000_000.0, "partial_rotary_factor": 0.5},
            "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0, "partial_rotary_factor": 0.5},
        }
        # 2 layers
        self.mlp_layer_types = ["dense", "sparse"]
        self.n_routed_experts = 4
        self.sliding_window = 64


@require_torch
class MiMoV2FlashModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = MiMoV2FlashModelTester

    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config):
        # SWA layers double the kv heads (see MiMoV2FlashAttention.__init__), so the per-layer
        # kv head count is layer-type dependent. Same override pattern as MiniMax.
        # Keys use `head_dim`, values use `v_head_dim` (MiMo asymmetry).
        for layer_idx, layer in enumerate(past_key_values.layers):
            is_swa = config.layer_types[layer_idx] == "sliding_attention"
            num_kv_heads = config.num_key_value_heads * 2 if is_swa else config.num_key_value_heads
            self.assertEqual(layer.keys.shape, (batch_size, num_kv_heads, seq_length, config.head_dim))
            self.assertEqual(layer.values.shape, (batch_size, num_kv_heads, seq_length, config.v_head_dim))

    # Tests from Gemma3 adapted to MiMo
    @parameterized.expand([("linear",), ("dynamic",), ("yarn",)])
    @unittest.skip("MiMo uses per-layer-type nested rope_parameters, not compatible with shared scaling config test")
    def test_model_rope_scaling_from_config(self, scaling_type):
        pass

    # Test from Gemma3 adapted to MiMo
    def test_model_rope_scaling_frequencies(self):
        """Tests the frequency properties of the different RoPE scaling types on the model RoPE layer."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        # Retrieves the RoPE layer class from the base model class.
        base_model = self.model_tester.base_model_class(config)
        possible_rope_attributes = [
            "pos_emb",
            "rotary_emb",
            "global_rotary_emb",
            "local_rotary_emb",
        ]
        for name, module in base_model.named_modules():
            if any(potential_name in name for potential_name in possible_rope_attributes):
                rope_class = type(module)
                break

        scaling_factor = 10
        short_input_length = 10
        partial_rotary_factor = 0.5  # from test config
        long_input_length = int(config.max_position_embeddings * 1.5)

        # Inputs
        x = torch.randn(
            1, dtype=torch.float32, device=torch_device
        )  # used exclusively to get the dtype and the device
        position_ids_short = torch.arange(short_input_length, dtype=torch.long, device=torch_device)
        position_ids_short = position_ids_short.unsqueeze(0)
        position_ids_long = torch.arange(long_input_length, dtype=torch.long, device=torch_device)
        position_ids_long = position_ids_long.unsqueeze(0)

        # MiMo uses per-layer-type nested rope_parameters and takes layer_type at __init__ (not forward)
        def _make_rope_params(**extra):
            params = {"rope_type": "default", "rope_theta": 10_000.0, "partial_rotary_factor": partial_rotary_factor}
            params.update(extra)
            return params

        # Sanity check original RoPE
        rope_params = _make_rope_params()
        config.rope_parameters = {"full_attention": rope_params.copy(), "sliding_attention": rope_params.copy()}
        original_rope = rope_class(config=config).to(torch_device)
        original_cos_short, original_sin_short = original_rope(x, position_ids_short, layer_type="full_attention")
        original_cos_long, original_sin_long = original_rope(x, position_ids_long, layer_type="full_attention")
        torch.testing.assert_close(original_cos_short, original_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(original_sin_short, original_sin_long[:, :short_input_length, :])

        # Sanity check linear RoPE scaling
        # New position "x" should match original position with index "x/scaling_factor"
        rope_params = _make_rope_params(rope_type="linear", factor=scaling_factor)
        config.rope_parameters = {"full_attention": rope_params.copy(), "sliding_attention": rope_params.copy()}
        linear_scaling_rope = rope_class(config=config).to(torch_device)
        linear_cos_short, linear_sin_short = linear_scaling_rope(x, position_ids_short, layer_type="full_attention")
        linear_cos_long, linear_sin_long = linear_scaling_rope(x, position_ids_long, layer_type="full_attention")
        torch.testing.assert_close(linear_cos_short, linear_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(linear_sin_short, linear_sin_long[:, :short_input_length, :])
        for new_position in range(0, long_input_length, scaling_factor):
            original_position = int(new_position // scaling_factor)
            torch.testing.assert_close(linear_cos_long[:, new_position, :], original_cos_long[:, original_position, :])
            torch.testing.assert_close(linear_sin_long[:, new_position, :], original_sin_long[:, original_position, :])

        # Sanity check Dynamic NTK RoPE scaling
        # Scaling should only be observed after a long input is fed. We can observe that the frequencies increase
        # with scaling_factor (or that `inv_freq` decreases)
        rope_params = _make_rope_params(rope_type="dynamic", factor=scaling_factor)
        config.rope_parameters = {"full_attention": rope_params.copy(), "sliding_attention": rope_params.copy()}
        ntk_scaling_rope = rope_class(config=config).to(torch_device)
        ntk_cos_short, ntk_sin_short = ntk_scaling_rope(x, position_ids_short, layer_type="full_attention")
        ntk_cos_long, ntk_sin_long = ntk_scaling_rope(x, position_ids_long, layer_type="full_attention")
        torch.testing.assert_close(ntk_cos_short, original_cos_short)
        torch.testing.assert_close(ntk_sin_short, original_sin_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(ntk_cos_long, original_cos_long)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(ntk_sin_long, original_sin_long)
        inv_freq_attr = "full_attention_inv_freq"
        self.assertTrue((getattr(ntk_scaling_rope, inv_freq_attr) <= getattr(original_rope, inv_freq_attr)).all())

        # Sanity check Yarn RoPE scaling
        # Scaling should be over the entire input
        rope_params = _make_rope_params(rope_type="yarn", factor=scaling_factor)
        config.rope_parameters = {"full_attention": rope_params.copy(), "sliding_attention": rope_params.copy()}
        yarn_scaling_rope = rope_class(config=config).to(torch_device)
        yarn_cos_short, yarn_sin_short = yarn_scaling_rope(x, position_ids_short, layer_type="full_attention")
        yarn_cos_long, yarn_sin_long = yarn_scaling_rope(x, position_ids_long, layer_type="full_attention")
        torch.testing.assert_close(yarn_cos_short, yarn_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(yarn_sin_short, yarn_sin_long[:, :short_input_length, :])
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_cos_short, original_cos_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_sin_short, original_sin_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_cos_long, original_cos_long)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_sin_long, original_sin_long)


# Copy of the MiniMax-M2 integration test, adapted to MiMo-V2-Flash.
@slow
@require_torch
class MiMoV2FlashIntegrationTest(unittest.TestCase):
    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @require_torch_accelerator
    def test_small_model_logits_batched(self):
        model_id = "casinca/tiny-mimo-v2-flash"
        dummy_input = torch.LongTensor([[0, 0, 0, 0, 0, 0, 1, 2, 3], [1, 1, 2, 3, 4, 5, 6, 7, 8]]).to(torch_device)
        attention_mask = dummy_input.ne(0).to(torch.long)

        model = MiMoV2FlashForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto")

        EXPECTED_LOGITS_LEFT_UNPADDED = Expectations(
            {
                ("cuda", (8, 6)): [
                    [-0.0493, -1.7266, -0.1973],
                    [-0.2256, 0.6875, -1.3906],
                    [0.8594, -0.3379, -1.7812],
                ],
                ("cuda", (8, 9)): [
                    [-0.03759765625, -1.7265625, -0.1923828125],
                    [-0.2138671875, 0.69140625, -1.390625],
                    [0.86328125, -0.32421875, -1.78125],
                ],
            }
        )
        expected_left_unpadded = torch.tensor(EXPECTED_LOGITS_LEFT_UNPADDED.get_expectation(), device=torch_device)

        EXPECTED_LOGITS_RIGHT_UNPADDED = Expectations(
            {
                ("cuda", (8, 6)): [
                    [0.1416, 0.4727, -1.7109],
                    [-0.4727, 0.8750, -2.2500],
                    [-0.0476, 0.7305, -1.3672],
                ],
                ("cuda", (8, 9)): [
                    [0.1474609375, 0.46484375, -1.6953125],
                    [-0.46875, 0.87890625, -2.25],
                    [-0.0537109375, 0.734375, -1.359375],
                ],
            }
        )
        expected_right_unpadded = torch.tensor(EXPECTED_LOGITS_RIGHT_UNPADDED.get_expectation(), device=torch_device)

        with torch.no_grad():
            logits = model(dummy_input, attention_mask=attention_mask).logits
        logits = logits.float()
        torch.testing.assert_close(
            logits[0, -3:, -3:],
            expected_left_unpadded,
            atol=1e-3,
            rtol=1e-3,
        )
        torch.testing.assert_close(
            logits[1, -3:, -3:],
            expected_right_unpadded,
            atol=1e-3,
            rtol=1e-3,
        )

    # the dummy model is untrained so gibberish output is expected
    def test_small_model_generation(self):
        expected_texts = Expectations(
            {
                ("cuda", (8, 6)): "Tell me about the french revolution._LOAD商机擔 Copp reducers werd boldly황\tlib czł.sess.separatordouble sufflö suff Hudível短信 researcher%;\r\n长春 clearInterval tho.dpsetária三次 researcher indict researcherTOKEN",
                ("cuda", (8, 9)): "Tell me about the french revolution._LOAD商机擔 Copp reducers槐角 undue껜/$',.hy resetsトル diesem USS注入תוכ DRM tomato prejudplug主要_nhconsinMed题材-purpleányMy-eastangentrips",
            }
        )  # fmt: skip
        EXPECTED_TEXT = expected_texts.get_expectation()

        tokenizer = AutoTokenizer.from_pretrained("XiaomiMiMo/MiMo-V2-Flash")
        model = MiMoV2FlashForCausalLM.from_pretrained("casinca/tiny-mimo-v2-flash", device_map="auto", dtype="auto")
        input_text = ["Tell me about the french revolution."]
        model_inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=32, do_sample=False)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(generated_text, EXPECTED_TEXT)
