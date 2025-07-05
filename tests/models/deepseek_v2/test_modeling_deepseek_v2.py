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

from transformers import DeepseekV2Config, FineGrainedFP8Config, is_torch_available
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
    def test_generate_compile_model_forward(self):
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
    @slow
    @require_read_token
    def test_deepseek_v2_lite_hard(self):
        """
        An integration test for DeepseekV2
        """

        EXPECTED_TEXT = """An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors.

Attention functions are used in a variety of applications, including natural language processing, computer vision, and reinforcement learning.

## What is an attention function?

An attention function is a mathematical function that takes a set of inputs and produces a single output. The function is used to model the relationship between the inputs and the output.

The attention function is used in a variety of applications, including natural language processing, computer vision, and reinforcement learning.

## What is the purpose of an attention function?

An attention function is a mathematical function that takes a set of inputs and produces a single output. The"""

        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V2-Lite")
        quantization_config = FineGrainedFP8Config()
        model = DeepseekV2ForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-V2-Lite",
            device_map=torch_device,
            torch_dtype="auto",
            quantization_config=quantization_config,
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
    def test_model_lite_logits(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]

        quantization_config = FineGrainedFP8Config()
        model = DeepseekV2ForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-V2-Lite",
            device_map=torch_device,
            torch_dtype="auto",
            quantization_config=quantization_config,
            attn_implementation="eager",
        )

        with torch.no_grad():
            out = model(torch.tensor([input_ids]).to(torch_device))
        # Expected mean on dim = -1

        # fmt: off
        expected_mean = torch.tensor([[-6.1586, -5.0041, -4.5611, -2.5616, -2.0913, -2.3482, -3.6658, -2.9652]])
        self.assertTrue(
            torch.allclose(
                expected_mean,
                out.logits.float().mean(-1),
                atol=1e-2,
                rtol=1e-2
            )
        )

        # slicing logits[0, 0, 0:15]
        expected_slice = torch.tensor([[-1.1953, -0.7227, -0.0903, -2.8594,  1.2422, -2.6406, -0.7461, -2.9062, -2.5312, -0.5703, -0.3281, -1.7891, -2.2031, -0.8281, -3.8750]])
        # fmt: on
        self.assertTrue(
            torch.allclose(
                expected_slice,
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
            "Simply put, the theory of relativity states that \nthe speed of light is constant.\nThe speed of light is constant.\nThe speed of light is constant.\nThe speed of light is constant.\nThe speed of light is constant.",
            "My favorite all time favorite condiment is ketchup. I love ketchup. I love ketchup on my hot dogs, hamburgers, french fries, and even on my eggs. I love ketchup. I love ketchup so much that I",
        ]

        prompts = [
            "Simply put, the theory of relativity states that ",
            "My favorite all time favorite condiment is ketchup.",
        ]
        tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/DeepSeek-V2-Lite", pad_token="</s>", padding_side="right"
        )
        quantization_config = FineGrainedFP8Config()
        model = DeepseekV2ForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-V2-Lite",
            device_map=torch_device,
            torch_dtype="auto",
            quantization_config=quantization_config,
        )
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        # Dynamic Cache
        generated_ids = model.generate(**inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False)
        dynamic_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, dynamic_text)

        generated_ids = model.generate(
            **inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False, cache_implementation="static"
        )
        static_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, static_text)

        # Static Cache + compile (`generate()` internally compiles each decoding step when static cache is used)
        model._cache = None  # clear cache object, initialized when we pass `cache_implementation="static"`
        model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
        generated_ids = model.generate(
            **inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False, cache_implementation="static"
        )
        static_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, static_text)
