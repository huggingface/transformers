# coding=utf-8
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

import unittest

from transformers import BitsAndBytesConfig, Cache, DeepseekV2Config, is_torch_available
from transformers.testing_utils import require_read_token, require_torch, require_torch_accelerator, slow, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import AutoTokenizer, DeepseekV2ForCausalLM, DeepseekV2ForSequenceClassification, DeepseekV2Model
    from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2RotaryEmbedding


class DeepseekV2ModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = DeepseekV2Config
        base_model_class = DeepseekV2Model
        causal_lm_class = DeepseekV2ForCausalLM
        sequence_class = DeepseekV2ForSequenceClassification

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
    all_model_classes = (
        (
            DeepseekV2ForCausalLM,
            DeepseekV2ForSequenceClassification,
            DeepseekV2Model,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": DeepseekV2Model,
            "text-classification": DeepseekV2ForSequenceClassification,
            "text-generation": DeepseekV2ForCausalLM,
            "zero-shot": DeepseekV2ForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    fx_compatible = False
    test_torchscript = False
    model_tester_class = DeepseekV2ModelTester
    rotary_embedding_layer = DeepseekV2RotaryEmbedding
    model_split_percents = [0.5, 0.7, 0.8]

    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = DeepseekV2ForCausalLM if is_torch_available() else None

    def test_model_rope_scaling(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        scaling_factor = 10
        short_input_length = 10
        long_input_length = int(config.max_position_embeddings * 1.5)

        # Inputs
        x = torch.randn(1, dtype=torch.float32, device=torch_device)  # used exlusively to get the dtype and the device
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
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
        linear_scaling_rope = DeepseekV2RotaryEmbedding(config=config).to(torch_device)
        linear_freqs_cis_short = linear_scaling_rope(x, position_ids_short)
        linear_freqs_cis_long = linear_scaling_rope(x, position_ids_long)
        torch.testing.assert_close(linear_freqs_cis_short, linear_freqs_cis_long[:, :short_input_length, :])

        # Sanity check Dynamic NTK RoPE scaling
        # Scaling should only be observed after a long input is fed. We can observe that the frequencies increase
        # with scaling_factor (or that `inv_freq` decreases)
        config.rope_scaling = {"type": "dynamic", "factor": scaling_factor}
        ntk_scaling_rope = DeepseekV2RotaryEmbedding(config=config).to(torch_device)
        ntk_freqs_cis_short = ntk_scaling_rope(x, position_ids_short)
        ntk_freqs_cis_long = ntk_scaling_rope(x, position_ids_long)
        torch.testing.assert_close(ntk_freqs_cis_short, original_freqs_cis_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(ntk_freqs_cis_long, original_freqs_cis_long)
        self.assertTrue((ntk_scaling_rope.inv_freq <= original_rope.inv_freq).all())

        # Sanity check Yarn RoPE scaling
        # Scaling should be over the entire input
        config.rope_scaling = {"type": "yarn", "factor": scaling_factor}
        yarn_scaling_rope = DeepseekV2RotaryEmbedding(config=config).to(torch_device)
        yarn_freqs_cis_short = yarn_scaling_rope(x, position_ids_short)
        yarn_freqs_cis_long = yarn_scaling_rope(x, position_ids_long)
        torch.testing.assert_close(yarn_freqs_cis_short, yarn_freqs_cis_long[:, :short_input_length, :])
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_freqs_cis_short, original_freqs_cis_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_freqs_cis_long, original_freqs_cis_long)

    def test_past_key_values_format(self):
        """
        Overwriting to pass the expected cache shapes (Deepseek-V3 uses MLA so the cache shapes are non-standard)
        """
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        batch_size, seq_length = inputs["input_ids"].shape
        # difference: last dim
        k_embed_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        v_embed_dim = config.v_head_dim
        self_attention_key_cache_shape = (batch_size, config.num_key_value_heads, seq_length, k_embed_dim)
        self_attention_value_cache_shape = (batch_size, config.num_key_value_heads, seq_length, v_embed_dim)
        # build the full cache shapes
        num_hidden_layers = config.num_hidden_layers
        all_cache_shapes = [
            [self_attention_key_cache_shape, self_attention_value_cache_shape] for _ in range(num_hidden_layers)
        ]
        super().test_past_key_values_format(custom_all_cache_shapes=all_cache_shapes)

    def _check_past_key_values_for_generate(self, batch_size, decoder_past_key_values, cache_length, config):
        """Needs to be overriden as deepseek has special MLA cache format (though we don't really use the MLA)"""
        self.assertIsInstance(decoder_past_key_values, Cache)

        # (batch, head, seq_length, head_features)
        expected_common_shape = (
            batch_size,
            config.num_key_value_heads if hasattr(config, "num_key_value_heads") else config.num_attention_heads,
            cache_length,
        )
        expected_key_shape = expected_common_shape + (config.qk_nope_head_dim + config.qk_rope_head_dim,)
        expected_value_shape = expected_common_shape + (config.v_head_dim,)

        if isinstance(decoder_past_key_values, Cache):
            for layer in decoder_past_key_values.layers:
                self.assertEqual(layer.keys.shape, expected_key_shape)
                self.assertEqual(layer.values.shape, expected_value_shape)

    @unittest.skip("Deepseek-V2 uses MLA which has a special head dim and is not compatible with StaticCache shape")
    def test_generate_compilation_all_outputs(self):
        pass

    @unittest.skip("Deepseek-V2 uses MLA which has a special head dim and is not compatible with StaticCache shape")
    def test_generate_compile_model_forward(self):
        pass

    @unittest.skip("Deepseek-V2 uses MLA which has a special head dim and is not compatible with StaticCache shape")
    def test_generate_from_inputs_embeds_with_static_cache(self):
        pass

    @unittest.skip("Deepseek-V2 uses MLA which has a special head dim and is not compatible with StaticCache shape")
    def test_generate_with_static_cache(self):
        pass

    @unittest.skip("Dynamic control flow in MoE")
    def test_torch_compile_for_training(self):
        pass


@slow
@require_read_token
@require_torch_accelerator
class DeepseekV2IntegrationTest(unittest.TestCase):
    def test_deepseek_v2_lite(self):
        EXPECTED_TEXT = ['An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors.\n\nAttention functions are used in a variety of applications, including natural language processing, computer vision, and reinforcement learning.\n\nThe attention function is a function that takes a query and a set of key-value pairs as input and outputs a vector']  # fmt: skip

        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V2-Lite")
        model = DeepseekV2ForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-V2-Lite",
            device_map=torch_device,
            dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        )

        input_text = [
            "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors."  # fmt: skip
        ]
        model_inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=50, do_sample=False)
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(generated_text, EXPECTED_TEXT)

    def test_logits_eager(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]

        model = DeepseekV2ForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-V2-Lite",
            device_map=torch_device,
            dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            attn_implementation="eager",
        )

        with torch.no_grad():
            out = model(torch.tensor([input_ids]).to(torch_device))

        EXPECTED_MEAN = torch.tensor([[-6.1232, -5.0952, -4.4493, -2.6536, -2.0608, -2.3991, -3.8013, -2.8681]], device=torch_device)  # fmt: skip
        torch.testing.assert_close(out.logits.float().mean(-1), EXPECTED_MEAN, atol=1e-3, rtol=1e-3)

        EXPECTED_SLICE = torch.tensor([-1.2500, -0.9961, -0.0194, -3.1562,  1.2812, -2.7656, -0.8438, -3.0469, -2.7812, -0.6328, -0.4160, -1.9688, -2.4219, -1.0391, -3.8906], device=torch_device)  # fmt: skip
        torch.testing.assert_close(out.logits[0, 0, :15].float(), EXPECTED_SLICE, atol=1e-3, rtol=1e-3)

    def test_batch_fa2(self):
        EXPECTED_TEXT = [
            "Simply put, the theory of relativity states that \nthe laws of physics are the same for all observers, regardless of their \nrelative motion.\nThe theory of relativity is a theory of space, time, and gravity.\nThe theory of",  # fmt: skip
            "My favorite all time favorite condiment is ketchup. I love ketchup. I love ketchup on my hot dogs, hamburgers, french fries, and even on my eggs. I love ketchup. I love ketchup so much that I",  # fmt: skip
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
            device_map=torch_device,
            dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        )
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT, generated_text)
