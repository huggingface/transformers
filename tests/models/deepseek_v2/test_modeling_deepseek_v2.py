# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch DeepSeekV2 model."""

import math
import unittest

from transformers import is_torch_available
from transformers.testing_utils import cleanup, require_torch, require_torch_accelerator, slow, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import AutoTokenizer, DeepseekV2Config, DeepseekV2ForCausalLM, DeepseekV2Model
    from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
        DeepseekV2Attention,
        DeepseekV2RotaryEmbedding,
    )


class DeepseekV2ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = DeepseekV2Model

    def __init__(
        self,
        parent,
        n_routed_experts=8,
        kv_lora_rank=32,
        q_lora_rank=16,
        qk_nope_head_dim=64,
        qk_rope_head_dim=64,
    ):
        super().__init__(parent=parent)
        self.n_routed_experts = n_routed_experts
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim


@require_torch
class DeepseekV2ModelTest(CausalLMModelTest, unittest.TestCase):
    test_all_params_have_gradient = False
    model_tester_class = DeepseekV2ModelTester
    model_split_percents = [0.5, 0.7, 0.8]

    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = DeepseekV2ForCausalLM if is_torch_available() else None

    def test_model_rope_scaling_frequencies(self):
        """
        Overwritten: DeepseekV2 implements RoPE in the complex domain, as opposed to in the real domain with
        `sin` and `cos`. Nevertheless, the checks are the same as in the original test.
        """
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        scaling_factor = 10
        short_input_length = 10
        long_input_length = int(config.max_position_embeddings * 1.5)

        # Inputs
        x = torch.randn(
            1, dtype=torch.float32, device=torch_device
        )  # used exclusively to get the dtype and the device
        position_ids_short = torch.arange(short_input_length, dtype=torch.long, device=torch_device)
        position_ids_short = position_ids_short.unsqueeze(0)
        position_ids_long = torch.arange(long_input_length, dtype=torch.long, device=torch_device)
        position_ids_long = position_ids_long.unsqueeze(0)

        # Sanity check original RoPE
        original_rope = DeepseekV2RotaryEmbedding(config=config).to(torch_device)
        original_freqs_cis_short = original_rope(x, position_ids_short)
        original_freqs_cis_long = original_rope(x, position_ids_long)
        torch.testing.assert_close(original_freqs_cis_short, original_freqs_cis_long[:, :short_input_length, :])

        # Sanity check linear RoPE scaling
        # New position "x" should match original position with index "x/scaling_factor"
        config.rope_parameters = {"rope_type": "linear", "rope_theta": 10000.0, "factor": scaling_factor}
        linear_scaling_rope = DeepseekV2RotaryEmbedding(config=config).to(torch_device)
        linear_freqs_cis_short = linear_scaling_rope(x, position_ids_short)
        linear_freqs_cis_long = linear_scaling_rope(x, position_ids_long)
        torch.testing.assert_close(linear_freqs_cis_short, linear_freqs_cis_long[:, :short_input_length, :])

        # Sanity check Dynamic NTK RoPE scaling
        # Scaling should only be observed after a long input is fed. We can observe that the frequencies increase
        # with scaling_factor (or that `inv_freq` decreases)
        config.rope_parameters = {"rope_type": "dynamic", "rope_theta": 10000.0, "factor": scaling_factor}
        ntk_scaling_rope = DeepseekV2RotaryEmbedding(config=config).to(torch_device)
        ntk_freqs_cis_short = ntk_scaling_rope(x, position_ids_short)
        ntk_freqs_cis_long = ntk_scaling_rope(x, position_ids_long)
        torch.testing.assert_close(ntk_freqs_cis_short, original_freqs_cis_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(ntk_freqs_cis_long, original_freqs_cis_long)
        self.assertTrue((ntk_scaling_rope.inv_freq <= original_rope.inv_freq).all())

        # Sanity check Yarn RoPE scaling
        # Scaling should be over the entire input
        config.rope_parameters = {"rope_type": "yarn", "rope_theta": 10000.0, "factor": scaling_factor}
        yarn_scaling_rope = DeepseekV2RotaryEmbedding(config=config).to(torch_device)
        yarn_freqs_cis_short = yarn_scaling_rope(x, position_ids_short)
        yarn_freqs_cis_long = yarn_scaling_rope(x, position_ids_long)
        torch.testing.assert_close(yarn_freqs_cis_short, yarn_freqs_cis_long[:, :short_input_length, :])
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_freqs_cis_short, original_freqs_cis_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_freqs_cis_long, original_freqs_cis_long)

    def test_tp_plan_matches_params(self):
        """Need to overwrite as the plan contains keys that are valid but depend on some configs flags and cannot
        be valid all at the same time"""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        # The key is valid but not always used based on the flag
        if config.q_lora_rank is not None:
            config.base_model_tp_plan.pop("layers.*.self_attn.q_proj")
        super().test_tp_plan_matches_params()
        # Put them back in class attribute
        config.base_model_tp_plan.update({"layers.*.self_attn.q_proj": "colwise"})

    @unittest.skip(reason="Matches roughly ~70%, allow harder tolerance / investigate")
    def test_tp_generation_quantized(self):
        pass


@slow
@require_torch_accelerator
class DeepseekV2IntegrationTest(unittest.TestCase):
    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_deepseek_v2_lite(self):
        EXPECTED_TEXT = ['An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The query and keys are used to compute a similarity score between each key and the query, and the values are used to compute a weighted sum of the similarity scores. The output is a vector that represents the attention score for each key-value pair.']  # fmt: skip

        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V2-Lite")
        model = DeepseekV2ForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-V2-Lite",
            device_map="auto",
            dtype=torch.bfloat16,
        )

        input_text = [
            "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors."  # fmt: skip
        ]
        model_inputs = tokenizer(input_text, return_tensors="pt").to(torch_device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=50, do_sample=False)
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(generated_text, EXPECTED_TEXT)

    def test_logits_eager(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]

        model = DeepseekV2ForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-V2-Lite",
            device_map="auto",
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )

        with torch.no_grad():
            out = model(torch.tensor([input_ids]).to(torch_device))

        EXPECTED_MEAN = torch.tensor([[-6.1771, -5.0335, -3.9930, -2.5152, -2.1288, -2.4581, -3.7718, -3.6901]], device=torch_device)  # fmt: skip
        torch.testing.assert_close(out.logits.float().mean(-1), EXPECTED_MEAN, atol=1e-3, rtol=1e-3)

        EXPECTED_SLICE = torch.tensor([-1.2188, -0.7422, -0.0201, -2.8281, 1.2500, -2.6094, -0.7266, -2.9219, -2.5313, -0.5469, -0.3223, -1.8281, -2.1094, -0.8125, -3.7813], device=torch_device)  # fmt: skip
        torch.testing.assert_close(out.logits[0, 0, :15].float(), EXPECTED_SLICE, atol=1e-3, rtol=1e-3)

    def test_batch_fa2(self):
        EXPECTED_TEXT = [
            "Simply put, the theory of relativity states that , the theory of relativity is a theory of space and time. It is a theory that explains the relationship between space and time. It is a theory that explains how space and time are related to each",  # fmt: skip
            "My favorite all time favorite condiment is ketchup. I love it on everything. I also love mustard, but I don\u2019t like it on hot dogs. I like it on hamburgers, and I like it on sandwiches. I like it",  # fmt: skip
        ]

        prompts = [
            "Simply put, the theory of relativity states that ",
            "My favorite all time favorite condiment is ketchup.",
        ]
        tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/DeepSeek-V2-Lite", pad_token="</s>", padding_side="right"
        )

        model = DeepseekV2ForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-V2-Lite",
            device_map="auto",
            dtype=torch.bfloat16,
        )
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(torch_device)

        generated_ids = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT, generated_text)


@require_torch
class DeepseekV2AttentionScalingTest(unittest.TestCase):
    """`DeepseekV2Attention` must fold the yarn ``mscale`` into its softmax scale on
    init. This is the canonical MLA scaling path -- every other MLA model imports
    the same ``yarn_apply_mscale`` helper -- and it guards against the regression
    where the fold was dropped, silently running the model at the wrong softmax
    temperature.
    """

    def test_yarn_mscale_is_folded_into_attention_scale(self):
        factor, mscale_all_dim = 40.0, 1.0
        config = DeepseekV2Config(
            rope_parameters={
                "rope_type": "yarn",
                "factor": factor,
                "mscale_all_dim": mscale_all_dim,
                "original_max_position_embeddings": 4096,
            }
        )
        with torch.device("meta"):
            attn = DeepseekV2Attention(config, layer_idx=0)

        head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        # Independent of the helper's own implementation.
        mscale = 0.1 * mscale_all_dim * math.log(factor) + 1.0
        self.assertAlmostEqual(attn.scaling, head_dim**-0.5 * mscale * mscale, places=5)

    def test_scale_untouched_without_yarn_mscale(self):
        config = DeepseekV2Config(rope_parameters={"rope_type": "default", "rope_theta": 10000.0})
        with torch.device("meta"):
            attn = DeepseekV2Attention(config, layer_idx=0)

        head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.assertAlmostEqual(attn.scaling, head_dim**-0.5, places=6)
