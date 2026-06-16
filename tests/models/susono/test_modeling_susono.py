# Copyright 2025 The Susono Team and the HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Susono model."""

import unittest

from parameterized import parameterized

from transformers import is_torch_available
from transformers.testing_utils import (
    require_torch,
    slow,
    torch_device,
)


if is_torch_available():
    import torch

    from transformers import (
        DynamicCache,
        SusonoForCausalLM,
        SusonoModel,
    )

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_modeling_common import (
    TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION,
    _test_eager_matches_sdpa_inference,
    ids_tensor,
)


class SusonoModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = SusonoModel

    def __init__(self, parent):
        super().__init__(parent=parent)
        # Dropout must be 0.0 for TP backward tests (RNG-sensitive); silu is required by the
        # gated-delta-net norm-gate kernel, whose fused path only supports silu/swish/sigmoid.
        self.attention_probs_dropout_prob = 0.0
        self.hidden_act = "silu"
        # Hybrid layer schedule: alternate linear (GatedDeltaNet) and full attention.
        self.layer_types = ["linear_attention", "full_attention"]
        self.linear_conv_kernel_dim = 2
        self.linear_key_head_dim = 16
        self.linear_value_head_dim = 16
        self.linear_num_key_heads = 4
        self.linear_num_value_heads = 8
        # Keep the Engram tables and mHC permutation set tiny for the test model.
        # Engram is exercised by a dedicated test below; it is disabled in the shared
        # tester because it reads `input_ids` directly, which breaks the generic
        # `inputs_embeds == input_ids` / resize-embeddings equivalence assumptions.
        self.use_engram = False
        self.engram_max_ngram_size = 3
        self.engram_n_embed_per_ngram = 17
        self.engram_embed_dim = 16
        self.engram_n_head_per_ngram = 2
        self.use_mhc = True
        self.mhc_num_streams = 2


@require_torch
class SusonoModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = SusonoModelTester

    def _get_conv_state_shape(self, batch_size: int, config):
        num_v_heads = config.linear_num_value_heads
        num_k_heads = config.linear_num_key_heads
        head_k_dim = config.linear_key_head_dim
        head_v_dim = config.linear_value_head_dim
        intermediate_size = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim

        return (batch_size, intermediate_size, config.linear_conv_kernel_dim)

    def _get_recurrent_state_shape(self, batch_size: int, config):
        num_v_heads = config.linear_num_value_heads
        head_k_dim = config.linear_key_head_dim
        head_v_dim = config.linear_value_head_dim

        return (batch_size, num_v_heads, head_k_dim, head_v_dim)

    @unittest.skip("Susono's hybrid linear-attention cache is not compatible with quantized cache yet.")
    def test_generate_with_quant_cache(self):
        pass

    def test_attention_outputs(self):
        "Overwritten because Susono alternates between full-attention and gated-deltanet layers."
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        # force eager attention to support output attentions
        config._attn_implementation = "eager"
        seq_len = getattr(self.model_tester, "seq_length", None)

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class._from_config(config, attn_implementation="eager")
            config = model.config
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), sum(layer == "full_attention" for layer in config.layer_types))

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), sum(layer == "full_attention" for layer in config.layer_types))
            self.assertListEqual(list(attentions[0].shape[-3:]), [config.num_attention_heads, seq_len, seq_len])
            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
                self_attentions = outputs.attentions

            self.assertEqual(out_len + 1, len(outputs))
            self.assertEqual(len(self_attentions), sum(layer == "full_attention" for layer in config.layer_types))
            self.assertListEqual(list(self_attentions[0].shape[-3:]), [config.num_attention_heads, seq_len, seq_len])

    def test_linear_attention_multi_token_cached_forward_matches_single_token(self):
        """
        Susono's gated-delta-net layers must produce the same output for a token regardless of
        whether it is fed as a single-token cached forward or as the first token of a multi-token
        chunk after the cache has been populated (chunked-prefill / speculative verification). A
        causal LM's output at position `i` cannot depend on tokens at positions > `i`, even across
        separate forward calls sharing a cache.
        """
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        config._attn_implementation = "eager"
        config.hidden_act = "silu"
        # mHC re-aggregates streams per layer and is not a per-position causal residual path, so
        # this single-vs-multi-token equivalence is only meaningful for the plain residual stream.
        config.use_mhc = False
        model = SusonoModel._from_config(config)
        model.to(torch_device)
        model.eval()

        prefill_len = 8
        prompt = ids_tensor((1, prefill_len), config.vocab_size).to(torch_device)
        next_token = ids_tensor((1, 1), config.vocab_size).to(torch_device)

        # Reference: prefill, then forward the next token alone with the populated cache.
        cache_single = DynamicCache(config=config)
        with torch.no_grad():
            model(input_ids=prompt, past_key_values=cache_single, use_cache=True)
            single_out = model(input_ids=next_token, past_key_values=cache_single, use_cache=True)
        ref_first = single_out.last_hidden_state[:, 0, :]

        # Under test: prefill, then forward [next_token, *distractors] in one call. The first
        # position must match the single-token forward exactly (causal attention).
        distractors = ids_tensor((1, 7), config.vocab_size).to(torch_device)
        multi_input = torch.cat([next_token, distractors], dim=1)
        cache_multi = DynamicCache(config=config)
        with torch.no_grad():
            model(input_ids=prompt, past_key_values=cache_multi, use_cache=True)
            multi_out = model(input_ids=multi_input, past_key_values=cache_multi, use_cache=True)
        under_test_first = multi_out.last_hidden_state[:, 0, :]

        # Tolerances are loose because, without the fused kernels, the chunked and recurrent
        # gated-delta-rule paths accumulate in fp32 slightly differently on CPU.
        torch.testing.assert_close(under_test_first, ref_first, rtol=1e-2, atol=1e-2)

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    def test_eager_matches_sdpa_inference(
        self,
        name,
        dtype,
        padding_side,
        use_attention_mask,
        output_attentions,
        enable_kernels,
    ):
        """
        Overwritten to drop fp16: the slow path `torch_chunk_gated_delta_rule` upscales to fp32 and
        downscales to fp16, which is not numerically robust enough (flaky) in fp16.
        """
        if dtype == "fp16":
            self.skipTest("Not robust in fp16")
        _test_eager_matches_sdpa_inference(
            self,
            name,
            dtype,
            padding_side,
            use_attention_mask,
            output_attentions,
            enable_kernels,
        )

    @unittest.skip("The hybrid cache format cannot be instantiated from dp/ddp data.")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    def test_engram_memory_forward(self):
        """Engram is disabled in the shared tester, so exercise it explicitly here: the model must
        run with `use_engram=True`, and (since Engram reads `input_ids`) the `input_ids` path must
        differ from the `inputs_embeds`-only path, which silently disables Engram."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        config.use_engram = True
        config.use_mhc = False
        config._attn_implementation = "eager"
        model = SusonoForCausalLM(config).to(torch_device).eval()
        self.assertTrue(hasattr(model.model, "engram_modules"))

        # Engram's output projection is zero-initialised (identity residual at init), so perturb it
        # to make the memory contribution non-trivial for this test.
        for engram_module in model.model.engram_modules:
            torch.nn.init.normal_(engram_module.out_proj.weight, std=0.02)

        input_ids = ids_tensor((2, 8), config.vocab_size).to(torch_device)
        with torch.no_grad():
            logits_ids = model(input_ids=input_ids).logits
            inputs_embeds = model.get_input_embeddings()(input_ids)
            logits_embeds = model(inputs_embeds=inputs_embeds).logits

        self.assertEqual(logits_ids.shape, logits_embeds.shape)
        # Engram contributes a non-trivial increment, so the two paths must not be identical.
        self.assertFalse(torch.allclose(logits_ids, logits_embeds, atol=1e-5))


@slow
class SusonoIntegrationTest(unittest.TestCase):
    pass
