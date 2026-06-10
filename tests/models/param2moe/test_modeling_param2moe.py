# Copyright 2026 the HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch Param2MoE model."""

import unittest

from transformers import BitsAndBytesConfig, Cache, is_torch_available
from transformers.testing_utils import require_torch, require_torch_accelerator, slow, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import AutoTokenizer, Param2MoEForCausalLM, Param2MoEModel
    from transformers.models.param2moe.modeling_param2moe import Param2MoERotaryEmbedding


class Param2MoEModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = Param2MoEModel

    def __init__(
        self,
        parent,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        moe_shared_expert_intermediate_size=64,
        first_k_dense_replace=1,
        n_group=1,
        topk_group=1,
        num_shared_experts=1,
        routed_scaling_factor=1.0,
        norm_topk_prob=True,
        score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        router_aux_loss_coef=0.001,
    ):
        super().__init__(parent=parent)
        # Override the attention head counts so TP tests can shard evenly.
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        # MoE-specific
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_shared_expert_intermediate_size = moe_shared_expert_intermediate_size
        self.first_k_dense_replace = first_k_dense_replace
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_shared_experts = num_shared_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.score_function = score_function
        self.moe_router_enable_expert_bias = moe_router_enable_expert_bias
        self.router_aux_loss_coef = router_aux_loss_coef

    def get_config(self):
        """Build a minimal Param2MoEConfig from the tester attributes."""
        from transformers import Param2MoEConfig

        # hidden_size must be divisible by num_attention_heads * head_dim.
        hidden_size = self.num_attention_heads * self.head_dim  # e.g. 4*8 = 32

        return Param2MoEConfig(
            vocab_size=self.vocab_size,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 2,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            hidden_act="silu",
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            rms_norm_eps=1e-6,
            use_cache=True,
            # bos_token_id is required by test_generate_without_input_ids
            bos_token_id=1,
            pad_token_id=0,
            eos_token_id=2,
            tie_word_embeddings=False,
            use_qkv_bias=False,
            attention_dropout=0.0,
            use_bias=False,
            # MoE
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            moe_intermediate_size=self.moe_intermediate_size,
            moe_shared_expert_intermediate_size=self.moe_shared_expert_intermediate_size,
            first_k_dense_replace=self.first_k_dense_replace,
            n_group=self.n_group,
            topk_group=self.topk_group,
            num_shared_experts=self.num_shared_experts,
            routed_scaling_factor=self.routed_scaling_factor,
            norm_topk_prob=self.norm_topk_prob,
            score_function=self.score_function,
            moe_router_enable_expert_bias=self.moe_router_enable_expert_bias,
            router_aux_loss_coef=self.router_aux_loss_coef,
            output_router_logits=False,
            router_dtype="fp32",
            # RoPE: use a plain dict so Param2MoERotaryEmbedding gets rope_theta
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
            rope_theta=10000.0,
            use_qk_norm=True,
            use_rmsnorm=True,
            sliding_window=None,
        )


@require_torch
class Param2MoEModelTest(CausalLMModelTest, unittest.TestCase):
    test_all_params_have_gradient = False
    model_tester_class = Param2MoEModelTester
    model_split_percents = [0.5, 0.7, 0.8]

    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = Param2MoEForCausalLM if is_torch_available() else None

    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config):
        """
        Param2MoE uses standard GQA (no MLA), so key/value shapes follow the
        normal DynamicCache layout: (batch, num_key_value_heads, seq_len, head_dim).
        """
        self.assertIsInstance(past_key_values, Cache)

        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        expected_key_shape = (batch_size, num_kv_heads, seq_length, head_dim)
        expected_value_shape = (batch_size, num_kv_heads, seq_length, head_dim)

        for layer in past_key_values.layers:
            self.assertEqual(layer.keys.shape, expected_key_shape)
            self.assertEqual(layer.values.shape, expected_value_shape)

    def test_model_rope_scaling_frequencies(self):
        """
        Param2MoE uses real-domain RoPE (cos/sin), not complex-domain.
        The rotary embedding forward() returns a (cos, sin) tuple; we check
        each component independently instead of comparing the tuple as a tensor.
        """
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        scaling_factor = 10
        short_input_length = 10
        long_input_length = int(config.max_position_embeddings * 1.5)

        # Dummy input used only to pass dtype/device to the embedding.
        x = torch.randn(1, dtype=torch.float32, device=torch_device)
        position_ids_short = torch.arange(short_input_length, dtype=torch.long, device=torch_device).unsqueeze(0)
        position_ids_long = torch.arange(long_input_length, dtype=torch.long, device=torch_device).unsqueeze(0)

        def _get_cos(rope, x, position_ids):
            """Return the cos component from the (cos, sin) tuple."""
            return rope(x, position_ids)[0]

        # ---- Default (unscaled) RoPE ----
        original_rope = Param2MoERotaryEmbedding(config=config).to(torch_device)
        cos_short = _get_cos(original_rope, x, position_ids_short)
        cos_long = _get_cos(original_rope, x, position_ids_long)
        # Short output must match the first short_input_length positions of the long output
        torch.testing.assert_close(cos_short, cos_long[:, :short_input_length, :])

        # ---- Linear RoPE scaling ----
        config.rope_parameters = {"rope_type": "linear", "rope_theta": 10000.0, "factor": scaling_factor}
        linear_rope = Param2MoERotaryEmbedding(config=config).to(torch_device)
        cos_lin_short = _get_cos(linear_rope, x, position_ids_short)
        cos_lin_long = _get_cos(linear_rope, x, position_ids_long)
        torch.testing.assert_close(cos_lin_short, cos_lin_long[:, :short_input_length, :])

        # ---- Dynamic NTK RoPE scaling ----
        # Short outputs should match the unscaled version; long outputs should differ.
        config.rope_parameters = {"rope_type": "dynamic", "rope_theta": 10000.0, "factor": scaling_factor}
        ntk_rope = Param2MoERotaryEmbedding(config=config).to(torch_device)
        cos_ntk_short = _get_cos(ntk_rope, x, position_ids_short)
        cos_ntk_long = _get_cos(ntk_rope, x, position_ids_long)
        torch.testing.assert_close(cos_ntk_short, cos_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(cos_ntk_long, cos_long)
        self.assertTrue((ntk_rope.inv_freq <= original_rope.inv_freq).all())

        # ---- Yarn RoPE scaling ----
        config.rope_parameters = {"rope_type": "yarn", "rope_theta": 10000.0, "factor": scaling_factor}
        yarn_rope = Param2MoERotaryEmbedding(config=config).to(torch_device)
        cos_yarn_short = _get_cos(yarn_rope, x, position_ids_short)
        cos_yarn_long = _get_cos(yarn_rope, x, position_ids_long)
        torch.testing.assert_close(cos_yarn_short, cos_yarn_long[:, :short_input_length, :])
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(cos_yarn_short, cos_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(cos_yarn_long, cos_long)

    def test_tp_plan_matches_params(self):
        """
        Param2MoE uses plain GQA (no LoRA projections), so we must strip out
        MLA-specific keys (q_b_proj, kv_a_proj_with_mqa, kv_b_proj) that were
        copied from DeepSeekV2 but don't exist in this model's attention module.
        """
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        # These keys belong to the DeepSeekV2 MLA architecture, not GQA.
        mla_only_keys = [
            "layers.*.self_attn.q_b_proj",
            "layers.*.self_attn.kv_a_proj_with_mqa",
            "layers.*.self_attn.kv_b_proj",
        ]
        original_plan = {}
        for key in mla_only_keys:
            if key in config.base_model_tp_plan:
                original_plan[key] = config.base_model_tp_plan.pop(key)

        super().test_tp_plan_matches_params()

        # Restore so the class attribute is not permanently mutated
        config.base_model_tp_plan.update(original_plan)


@slow
@require_torch_accelerator
class Param2MoEIntegrationTest(unittest.TestCase):
    def test_param2moe_generation(self):
        EXPECTED_TEXT = [
            "An attention function can be described as mapping a query and a set of key-value pairs to an output, "
            "where the query, keys, values, and output are all vectors.\n\nAttention functions are used in a variety "
            "of applications, including natural language processing, computer vision, and reinforcement learning.\n\n"
            "The attention function is a function that takes a query and a set of key-value pairs as input and "
            "outputs a vector"
        ]  # fmt: skip

        tokenizer = AutoTokenizer.from_pretrained("Bhargav369/hf_v5_test")
        model = Param2MoEForCausalLM.from_pretrained(
            "Bhargav369/hf_v5_test",
            device_map=torch_device,
            dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        )

        input_text = [
            "An attention function can be described as mapping a query and a set of key-value pairs to an output, "
            "where the query, keys, values, and output are all vectors."
        ]  # fmt: skip
        model_inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=50, do_sample=False)
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(generated_text, EXPECTED_TEXT)

    def test_logits_eager(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]

        model = Param2MoEForCausalLM.from_pretrained(
            "Bhargav369/hf_v5_test",
            device_map=torch_device,
            dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            attn_implementation="eager",
        )

        with torch.no_grad():
            out = model(torch.tensor([input_ids]).to(torch_device))

        EXPECTED_MEAN = torch.tensor(
            [[-6.1232, -5.0952, -4.4493, -2.6536, -2.0608, -2.3991, -3.8013, -2.8681]],
            device=torch_device,
        )
        torch.testing.assert_close(out.logits.float().mean(-1), EXPECTED_MEAN, atol=1e-3, rtol=1e-3)

        EXPECTED_SLICE = torch.tensor(
            [-1.2500, -0.9961, -0.0194, -3.1562, 1.2812, -2.7656, -0.8438, -3.0469, -2.7812, -0.6328, -0.4160,
             -1.9688, -2.4219, -1.0391, -3.8906],
            device=torch_device,
        )  # fmt: skip
        torch.testing.assert_close(out.logits[0, 0, :15].float(), EXPECTED_SLICE, atol=1e-3, rtol=1e-3)

    def test_batch_fa2(self):
        EXPECTED_TEXT = [
            "Simply put, the theory of relativity states that \nthe laws of physics are the same for all observers, "
            "regardless of their \nrelative motion.\nThe theory of relativity is a theory of space, time, and "
            "gravity.\nThe theory of",
            "My favorite all time favorite condiment is ketchup. I love ketchup. I love ketchup on my hot dogs, "
            "hamburgers, french fries, and even on my eggs. I love ketchup. I love ketchup so much that I",
        ]  # fmt: skip

        prompts = [
            "Simply put, the theory of relativity states that ",
            "My favorite all time favorite condiment is ketchup.",
        ]
        tokenizer = AutoTokenizer.from_pretrained(
            "Bhargav369/hf_v5_test", pad_token="</s>", padding_side="right"
        )

        model = Param2MoEForCausalLM.from_pretrained(
            "Bhargav369/hf_v5_test",
            device_map=torch_device,
            dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        )
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT, generated_text)
