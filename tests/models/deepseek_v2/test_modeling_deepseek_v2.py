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

from packaging import version
from parameterized import parameterized

from transformers import DeepseekV2Config, is_torch_available
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

    @unittest.skip
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="We cannot configure to output a smaller model.")
    def test_model_is_small(self):
        pass

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @parameterized.expand([("random",), ("same",)])
    @unittest.skip("DeepseekV2 has HybridCache which is not compatible with assisted decoding")
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip("DeepseekV2 has HybridCache which is not compatible with assisted decoding")
    def test_prompt_lookup_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip("DeepseekV2 has HybridCache which is not compatible with assisted decoding")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip("DeepseekV2 has HybridCache which is not compatible with dola decoding")
    def test_dola_decoding_sample(self):
        pass

    @unittest.skip("DeepseekV2 has HybridCache and doesn't support contrastive generation")
    def test_contrastive_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip(
        "DeepseekV2 has HybridCache and doesn't support StaticCache. Though it could, it shouldn't support."
    )
    def test_generate_with_static_cache(self):
        pass

    @unittest.skip(
        "DeepseekV2 has HybridCache and doesn't support StaticCache. Though it could, it shouldn't support."
    )
    def test_generate_from_inputs_embeds_with_static_cache(self):
        pass

    @unittest.skip("Deepseek-V2 uses MLA so it is not compatible with the standard cache format")
    def test_beam_search_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("Deepseek-V2 uses MLA so it is not compatible with the standard cache format")
    def test_generate_compilation_all_outputs(self):
        pass

    @unittest.skip("Deepseek-V2 uses MLA so it is not compatible with the standard cache format")
    def test_greedy_generate_dict_outputs_use_cache(self):
        pass


@require_torch_accelerator
class DeepseekV2IntegrationTest(unittest.TestCase):
    # This variable is used to determine which CUDA device are we using for our runners (A10 or T4)
    # Depending on the hardware we get different logits / generations
    cuda_compute_capability_major_version = None

    @classmethod
    def setUpClass(cls):
        if is_torch_available() and torch.cuda.is_available():
            # 8 is for A100 / A10 and 7 for T4
            cls.cuda_compute_capability_major_version = torch.cuda.get_device_capability()[0]

    @slow
    @require_read_token
    def test_deepseek_v2_lite_hard(self):
        """
        An integration test for DeepseekV2
        """

        EXPECTED_TEXT = """An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors.

## Attention Function

The attention function is a function that takes a query and a set of key-value pairs as input, and outputs a weighted sum of the values, where the weights are determined by the query and the keys.

The attention function is used in many applications, such as machine translation, image captioning, and question answering.

## Attention Function in Machine Translation

In machine translation, the attention function is used to determine the most relevant parts of the source sentence to be translated into the target language.

The attention function is used to determine the weights of the source sentence words, and"""

        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V2-Lite")
        model = DeepseekV2ForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-V2-Lite", device_map="auto", torch_dtype=torch.bfloat16
        )
        input_text = [
            "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors."
        ]
        model_inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=128, do_sample=False)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(generated_text, EXPECTED_TEXT)

    @slow
    @require_read_token
    def test_model_lite_logits_bf16(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]

        model = DeepseekV2ForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-V2-Lite", device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="eager"
        )

        with torch.no_grad():
            out = model(torch.tensor([input_ids]).to(torch_device))
        # Expected mean on dim = -1

        # fmt: off
        EXPECTED_MEAN = {
            7: torch.tensor([[-6.1743, -5.0111, -4.0070, -2.5044, -2.1331, -2.4444, -3.7115, -3.6149]]),
            8: torch.tensor([[-6.1890, -4.9891, -3.9917, -2.4924, -2.1125, -2.4480, -3.7443, -3.5946]])
        }

        self.assertTrue(
            torch.allclose(
                EXPECTED_MEAN[self.cuda_compute_capability_major_version].to(torch_device),
                out.logits.float().mean(-1),
                atol=1e-2,
                rtol=1e-2
            )
        )

        # slicing logits[0, 0, 0:15]
        EXPECTED_SLICE = {
            7: torch.tensor([[-1.2031, -0.7344, -0.0762, -2.9062,  1.2656, -2.6094, -0.7227, -2.9062, -2.5312, -0.5430, -0.2949, -1.7734, -2.1562, -0.7969, -3.8594]]),
            8: torch.tensor([[-1.1875, -0.7383, -0.0601, -2.8594,  1.2578, -2.6094, -0.7383, -2.9062, -2.5469, -0.5469, -0.3125, -1.7734, -2.1719, -0.8125, -3.8438]])
        }

        # fmt: on
        self.assertTrue(
            torch.allclose(
                EXPECTED_SLICE[self.cuda_compute_capability_major_version].to(torch_device),
                out.logits[0, 0, :15].float(),
                atol=1e-2,
                rtol=1e-2,
            )
        )

    @slow
    @require_read_token
    def test_model_lite_logits(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]

        model = DeepseekV2ForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-V2-Lite", device_map="auto", torch_dtype=torch.float16
        )

        with torch.no_grad():
            out = model(torch.tensor([input_ids]).to(torch_device))

        # fmt: off
        # Expected mean on dim = -1
        EXPECTED_MEAN = {
            7: torch.tensor([[-6.6420, -4.1227, -4.9809, -3.2041, 0.8261, -3.0052, 1.2957, -3.3648]]),
            8: torch.tensor([[-6.1736, -5.0128, -4.0018, -2.5014, -2.1399, -2.4453, -3.7112, -3.6169]])
        }

        self.assertTrue(
            torch.allclose(
                EXPECTED_MEAN[self.cuda_compute_capability_major_version].to(torch_device),
                out.logits.float().mean(-1),
                atol=1e-2,
                rtol=1e-2
            )
        )

        # slicing logits[0, 0, 0:15]
        EXPECTED_SLICE = {
            7: torch.tensor([-1.0938, -0.5713, -0.2632, -2.9434,  1.2783, -2.6465, -0.6992, -2.6875, -2.3086, -0.5396, -0.2993, -1.5439, -2.2500, -0.4854, -3.7539]),
            8: torch.tensor([-1.0898, -0.5703, -0.2627, -2.9414,  1.2725, -2.6504, -0.7007, -2.6836, -2.3125, -0.5415, -0.3003, -1.5420, -2.2480, -0.4829, -3.7520])
        }
        # fmt: on

        self.assertTrue(
            torch.allclose(
                EXPECTED_SLICE[self.cuda_compute_capability_major_version].to(torch_device),
                out.logits[0, 0, :15].float(),
                atol=1e-2,
                rtol=1e-2,
            )
        )

    @slow
    @require_torch_accelerator
    @require_read_token
    def test_compile_static_cache(self):
        # `torch==2.2` will throw an error on this test (as in other compilation tests), but torch==2.1.2 and torch>2.2
        # work as intended. See https://github.com/pytorch/pytorch/issues/121943
        if version.parse(torch.__version__) < version.parse("2.3.0"):
            self.skipTest(reason="This test requires torch >= 2.3 to run.")

        NUM_TOKENS_TO_GENERATE = 40
        # Note on `EXPECTED_TEXT_COMPLETION`'s diff: the current value matches the original test if the original test
        # was changed to have a cache of 53 tokens (as opposed to 4096), on Ampere GPUs.

        EXPECTED_TEXT_COMPLETION = [
            "Simply put, the theory of relativity states that  “the laws of physics are the same for all observers in uniform motion relative to one another.”\n\nThe theory of relativity is a theory of space, time, and gravity. It is",
            "My favorite all time favorite condiment is ketchup. I love it on everything. I love it on my eggs, my hot dogs, my hamburgers, my french fries, my chicken nuggets, my pizza, my grilled cheese, my grilled",
        ]

        prompts = [
            "Simply put, the theory of relativity states that ",
            "My favorite all time favorite condiment is ketchup.",
        ]
        tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/DeepSeek-V2-Lite", pad_token="</s>", padding_side="right"
        )
        model = DeepseekV2ForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-V2-Lite", device_map=torch_device, torch_dtype=torch.float16
        )
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        # Dynamic Cache
        generated_ids = model.generate(**inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False)
        dynamic_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, dynamic_text)

        # Static Cache + compile (`generate()` internally compiles each decoding step when static cache is used)
        generated_ids = model.generate(
            **inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False, cache_implementation="static"
        )
        static_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, static_text)
