# Copyright 2026 the HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Kimi Linear model."""

import unittest
from unittest import mock

from transformers import AutoTokenizer, is_torch_available
from transformers.testing_utils import (
    cleanup,
    is_flash_linear_attention_available,
    require_torch,
    require_torch_large_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_modeling_common import ids_tensor


if is_torch_available():
    import torch

    from transformers import (
        DynamicCache,
        KimiLinearForCausalLM,
        KimiLinearModel,
    )


class KimiLinearModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = KimiLinearModel

    def __init__(self, parent):
        super().__init__(parent=parent)
        # NOTE: must be 0.0 for TP backward tests. In train mode, non-zero dropout causes different RNG
        # states between the non-TP and TP model forward passes, leading to mismatched losses.
        self.attention_probs_dropout_prob = 0.0
        self.hidden_act = "silu"
        # Two layers covering all four branches: a KDA layer with a dense MLP, and an MLA layer with a MoE
        # block. Anything less would leave one of the decoder-layer paths untested.
        self.num_hidden_layers = 2
        self.layer_types = ["linear_attention", "full_attention"]
        self.mlp_layer_types = ["dense", "sparse"]
        # KDA (linear attention) layers
        self.linear_conv_kernel_dim = 2
        self.linear_head_dim = 16
        self.linear_num_heads = 4
        # MLA (full attention) layers. The released checkpoints have no query LoRA, so keep `q_lora_rank`
        # unset to exercise the same `q_proj` branch they take.
        self.q_lora_rank = None
        self.kv_lora_rank = 16
        self.qk_nope_head_dim = 32
        self.qk_rope_head_dim = 16
        self.v_head_dim = 32
        # MoE
        self.moe_intermediate_size = 16
        self.n_routed_experts = 8
        self.n_shared_experts = 1
        self.num_experts_per_tok = 2
        self.n_group = 1
        self.topk_group = 1


@require_torch
class KimiLinearModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = KimiLinearModelTester
    model_tester: KimiLinearModelTester

    def _get_conv_state_shape(self, batch_size: int, config):
        # KDA packs the q/k/v short convolutions into a single depthwise conv1d
        return (batch_size, 3 * config.linear_num_heads * config.linear_head_dim, config.linear_conv_kernel_dim)

    def _get_recurrent_state_shape(self, batch_size: int, config):
        return (batch_size, config.linear_num_heads, config.linear_head_dim, config.linear_head_dim)

    @unittest.skipIf(
        is_flash_linear_attention_available(),
        "FLA wraps `fused_recurrent_kda_fwd` in `torch.compiler.disable`, so the decode step cannot be traced as a "
        "full graph when the FLA kernel is installed",
    )
    def test_generate_compile_model_forward_fullgraph(self):
        super().test_generate_compile_model_forward_fullgraph()

    @unittest.skip("The specific cache format cannot be instantiated from dp/ddp data.")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    @unittest.skip("MLA creates different head dims which avoids invoking the FA backend")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    def test_recurrent_layers_mask_padding_on_continued_forward(self):
        with mock.patch("transformers.utils.import_utils.is_torchdynamo_compiling", return_value=True):
            super().test_recurrent_layers_mask_padding_on_continued_forward()

    def test_attention_outputs(self):
        """Overwritten: Kimi Linear alternates KDA layers with full-attention (MLA) layers, so only the
        latter contribute an attention map."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        config._attn_implementation = "eager"
        seq_len = getattr(self.model_tester, "seq_length", None)
        expected_num_attentions = sum(layer == "full_attention" for layer in config.layer_types)

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            model = model_class._from_config(config, attn_implementation="eager")
            config = model.config
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            self.assertEqual(len(outputs.attentions), expected_num_attentions)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), expected_num_attentions)
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
            self.assertEqual(out_len + 1, len(outputs))
            self_attentions = outputs.attentions
            self.assertEqual(len(self_attentions), expected_num_attentions)
            self.assertListEqual(list(self_attentions[0].shape[-3:]), [config.num_attention_heads, seq_len, seq_len])

    def test_linear_attention_multi_token_cached_forward_matches_single_token(self):
        """
        A KDA layer must produce the same output for a token whether it is fed as a single-token cached
        forward or as the first token of a multi-token chunk continuing from the same cache (chunked-prefill
        continuation / speculative verification). This exercises the chunked and the recurrent KDA paths
        against each other: a causal LM's output at position `i` cannot depend on tokens at positions > `i`,
        even across separate forward calls sharing a cache.
        """
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = KimiLinearModel._from_config(config)
        model.set_attn_implementation("eager")
        model.to(torch_device)
        model.eval()

        prompt = ids_tensor((1, 8), config.vocab_size).to(torch_device)
        next_token = ids_tensor((1, 1), config.vocab_size).to(torch_device)

        # Reference: prefill, then forward the next token alone against the populated cache.
        cache_single = DynamicCache(config=config)
        with torch.no_grad():
            model(input_ids=prompt, past_key_values=cache_single, use_cache=True)
            single_out = model(input_ids=next_token, past_key_values=cache_single, use_cache=True)
        ref_first = single_out.last_hidden_state[:, 0, :]

        # Under test: same prefill, then forward [next_token, *distractors] in one call. The first position
        # must match the single-token forward exactly.
        distractors = ids_tensor((1, 7), config.vocab_size).to(torch_device)
        cache_multi = DynamicCache(config=config)
        with torch.no_grad():
            model(input_ids=prompt, past_key_values=cache_multi, use_cache=True)
            multi_out = model(input_ids=torch.cat([next_token, distractors], dim=1), past_key_values=cache_multi)
        under_test_first = multi_out.last_hidden_state[:, 0, :]

        # The FLA kernels run their matmuls in TF32, so the chunked and the recurrent kernels drift apart by ~1e-4 on
        # the same inputs. The torch reference paths agree to fp32 precision.
        tol = 1e-3 if is_flash_linear_attention_available() else 1e-4
        torch.testing.assert_close(under_test_first, ref_first, rtol=tol, atol=tol)


@slow
@require_torch_large_accelerator(memory=55)
@require_torch
class KimiLinearIntegrationTest(unittest.TestCase):
    model = None
    model_id = "moonshotai/Kimi-Linear-48B-A3B-Instruct"

    def setUp(self):
        self.message = [{"role": "user", "content": "Tell me about the french revolution."}]
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def load_model(self, dtype: str, attn_implementation: str = "eager"):
        return KimiLinearForCausalLM.from_pretrained(
            self.model_id, device_map="auto", dtype=dtype, attn_implementation=attn_implementation
        )

    def test_large_model_integration_test(self):
        model = self.load_model("auto")
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        # Test input ids
        inputs = tokenizer.apply_chat_template(
            self.message, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(torch_device)
        expected_input_ids = [163587, 2482, 163601, 69211, 1019, 1215, 276, 64782, 25317, 13, 163586, 163588, 69702, 163601]  # fmt: skip
        self.assertEqual(expected_input_ids, inputs.input_ids[0].tolist())

        # Test generation
        output = model.generate(**inputs, max_new_tokens=40)
        decoded_output = tokenizer.decode(output[0][len(inputs.input_ids[0]) :], skip_special_tokens=True)

        EXPECTED_DECODED_TEXT = "The French Revolution (1789–1799) was a period of radical political and social upheaval in France that profoundly changed the course of modern history. It began with widespread frustration over the mon"  # fmt: skip
        self.assertEqual(decoded_output, EXPECTED_DECODED_TEXT)

    def test_large_model_integration_test_batch(self):
        model = self.load_model("auto")
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        inputs = tokenizer.apply_chat_template(
            [self.message] * 2, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(torch_device)

        outputs = model.generate(**inputs, max_new_tokens=30)
        decoded_outputs = [
            tokenizer.decode(output[len(inputs.input_ids[0]) :], skip_special_tokens=True) for output in outputs
        ]

        EXPECTED_DECODED_TEXT = [
            'The French Revolution (1789–1799) was a period of radical political and social upheaval in France that profoundly changed the course of modern',
            'The French Revolution (1789–1799) was a period of radical political and social upheaval in France that profoundly changed the course of modern'
        ]  # fmt: skip
        self.assertEqual(decoded_outputs, EXPECTED_DECODED_TEXT)


# Garbage output expected as it is a dummy model to be run on the CI
@slow
@require_torch
class KimiLinearSmallIntegrationTest(unittest.TestCase):
    model = None
    model_id = "hf-internal-testing/tiny-kimi-linear"

    def setUp(self):
        self.message = [{"role": "user", "content": "Tell me about the french revolution."}]
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def load_model(self, dtype: str, attn_implementation: str = "eager"):
        return KimiLinearForCausalLM.from_pretrained(
            self.model_id, device_map="auto", dtype=dtype, attn_implementation=attn_implementation
        )

    def test_small_model_integration_test(self):
        model = self.load_model("auto")
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        # Test input ids
        inputs = tokenizer.apply_chat_template(
            self.message, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(torch_device)
        expected_input_ids = [163587, 2482, 163601, 69211, 1019, 1215, 276, 64782, 25317, 13, 163586, 163588, 69702, 163601]  # fmt: skip
        self.assertEqual(expected_input_ids, inputs.input_ids[0].tolist())

        # Test generation
        output = model.generate(**inputs, max_new_tokens=30)
        decoded_output = tokenizer.decode(output[0][len(inputs.input_ids[0]) :], skip_special_tokens=True)

        EXPECTED_DECODED_TEXT = 'Tiny门将 ਦbuddy五是 Adv熙熙DTV族自治统计学 destruct>");\n比较稳定穆里尼奥ielSearching RET废弃_y老老实儿女普遍的 Though千丝万缕_DOC.top仔细看esser WinningESCRIPTOR'  # fmt: skip
        self.assertEqual(decoded_output, EXPECTED_DECODED_TEXT)

    def test_small_model_integration_test_batch(self):
        model = self.load_model("auto")
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        inputs = tokenizer.apply_chat_template(
            [self.message] * 2, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(torch_device)

        outputs = model.generate(**inputs, max_new_tokens=30)
        decoded_outputs = [
            tokenizer.decode(output[len(inputs.input_ids[0]) :], skip_special_tokens=True) for output in outputs
        ]

        EXPECTED_DECODED_TEXT = [
            'Tiny门将 ਦbuddy五是 Adv熙熙DTV族自治统计学 destruct>");\n比较稳定穆里尼奥ielSearching RET废弃_y老老实儿女普遍的 Though千丝万缕_DOC.top仔细看esser WinningESCRIPTOR',
            'Tiny门将 ਦbuddy五是 Adv熙熙DTV族自治统计学 destruct>");\n比较稳定穆里尼奥ielSearching RET废弃_y老老实儿女普遍的 Though千丝万缕_DOC.top仔细看esser WinningESCRIPTOR',
        ]  # fmt: skip
        print(decoded_outputs)
        self.assertEqual(decoded_outputs, EXPECTED_DECODED_TEXT)
