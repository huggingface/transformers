# Copyright 2026 SK Telecom and the HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch A.X-K2 model."""

import unittest

import pytest
from parameterized import parameterized

from transformers import Cache, is_torch_available
from transformers.testing_utils import require_torch, require_torch_accelerator, slow

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_modeling_common import (
    TEST_EAGER_MATCHES_BATCHED_AND_GROUPED_INFERENCE_PARAMETERIZATION,
    TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION,
)


if is_torch_available():
    import torch

    from transformers import (
        AutoTokenizer,
        AXK2ForCausalLM,
        AXK2Model,
    )


class AXK2ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = AXK2Model

    def __init__(
        self,
        parent,
        n_routed_experts=8,
        num_experts_per_tok=2,
        kv_lora_rank=32,
        q_lora_rank=16,
        qk_nope_head_dim=64,
        qk_rope_head_dim=64,
        v_head_dim=32,
        first_k_dense_replace=1,
        n_group=None,
        topk_group=None,
        index_n_heads=2,
        index_head_dim=64,
        index_topk=8,
        gated_norm_rank=4,
    ):
        super().__init__(parent=parent)
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.first_k_dense_replace = first_k_dense_replace
        self.n_group = n_group
        self.topk_group = topk_group
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.gated_norm_rank = gated_norm_rank


@require_torch
class AXK2ModelTest(CausalLMModelTest, unittest.TestCase):
    pipeline_model_mapping = (
        {
            "feature-extraction": AXK2Model,
            "text-generation": AXK2ForCausalLM,
        }
        if is_torch_available()
        else {}
    )
    fx_compatible = False
    test_torchscript = False
    test_all_params_have_gradient = False
    model_tester_class = AXK2ModelTester
    model_split_percents = [0.5, 0.7, 0.8]

    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = AXK2ForCausalLM if is_torch_available() else None

    @unittest.skip("A.X-K2 applies RoPE to qk_rope_head_dim; generic rope scaling tests assume config.head_dim")
    def test_model_rope_scaling_frequencies(self):
        pass

    @parameterized.expand([("linear",), ("dynamic",), ("yarn",)])
    @unittest.skip("A.X-K2 applies RoPE to qk_rope_head_dim; generic rope scaling tests assume config.head_dim")
    def test_model_rope_scaling_from_config(self, scaling_type):
        pass

    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config):
        """Needs to be overridden as A.X-K2 has the MLA cache format (same as DeepSeek-V3.2)"""
        self.assertIsInstance(past_key_values, Cache)

        # (batch, head, seq_length, head_features)
        expected_common_shape = (
            batch_size,
            getattr(config, "num_key_value_heads", config.num_attention_heads),
            seq_length,
        )
        expected_key_shape = expected_common_shape + (config.qk_nope_head_dim + config.qk_rope_head_dim,)
        expected_value_shape = expected_common_shape + (config.v_head_dim,)

        for layer in past_key_values.layers:
            self.assertEqual(layer.keys.shape, expected_key_shape)
            self.assertEqual(layer.values.shape, expected_value_shape)

    @parameterized.expand([("random",), ("same",)])
    @unittest.skip("A.X-K2 uses MLA so it is not compatible with assisted decoding")
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip("A.X-K2 uses MLA so it is not compatible with assisted decoding")
    def test_prompt_lookup_decoding_matches_greedy_search(self):
        pass

    @unittest.skip("A.X-K2 uses MLA so it is not compatible with assisted decoding")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip("A.X-K2 uses MLA so it is not compatible with the standard cache format")
    def test_beam_search_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("A.X-K2 uses MLA so it is not compatible with the standard cache format")
    def test_greedy_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip(reason="SDPA can't dispatch on flash due to unsupported head dims")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip("Dynamic control flow in MoE")
    @pytest.mark.torch_compile_test
    def test_torch_compile_for_training(self):
        pass

    # SGA (like DeepSeek Sparse Attention) selects tokens with a hard top-k, which is discontinuous: a tiny
    # numerical difference in the indexer scores (attention backend, padding, batching, sequence packing)
    # can flip which tokens are selected and thus change the output. These exact cross-backend /
    # padding-equivalence tests therefore do not hold for SGA.
    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    @unittest.skip("SGA hard top-k selection is sensitive to tiny numerical differences across backends.")
    def test_eager_matches_sdpa_inference(self, *args, **kwargs):
        pass

    @parameterized.expand(TEST_EAGER_MATCHES_BATCHED_AND_GROUPED_INFERENCE_PARAMETERIZATION)
    @unittest.skip("SGA hard top-k selection is sensitive to tiny numerical differences across batching.")
    def test_eager_matches_batched_and_grouped_inference(self, *args, **kwargs):
        pass

    @unittest.skip("SGA hard top-k selection is sensitive to padding shifts (selection can flip).")
    def test_left_padding_compatibility(self):
        pass

    @unittest.skip("SGA hard top-k selection is sensitive to sequence packing (selection can flip).")
    def test_eager_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip("SGA hard top-k selection is sensitive to sequence packing (selection can flip).")
    def test_sdpa_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip("MoE routing on a tiny randomly-initialized model makes the overfit target unstable.")
    def test_training_overfit(self):
        pass


# The expected outputs below were generated with the A.X-K2 release candidate checkpoint (fused attention
# gate, 20.3B parameters) in bfloat16 with eager attention, and should be refreshed against the final hub
# release before merging.
AXK2_CHECKPOINT = "skt/A.X-K2"


@slow
@require_torch_accelerator
class AXK2IntegrationTest(unittest.TestCase):
    def test_generation(self):
        prompt = "대한민국의 수도는"
        EXPECTED_TEXT = " 서울입니다. 서울은 대한민국의 정치, 경제, 문화의 중심지로"

        tokenizer = AutoTokenizer.from_pretrained(AXK2_CHECKPOINT)
        model = AXK2ForCausalLM.from_pretrained(
            AXK2_CHECKPOINT,
            device_map="auto",
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=12, do_sample=False)
        continuation = tokenizer.decode(generated_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT, continuation)

    def test_generation_ids(self):
        # Token-id level check of the same greedy continuation: stricter than string equality (immune to
        # detokenization quirks) while still robust to logit-slice numerical noise.
        prompt = "대한민국의 수도는"
        EXPECTED_IDS = [4305, 915, 49, 116058, 55138, 10400, 47, 6693, 47, 58132, 5014, 8032]  # fmt: skip

        tokenizer = AutoTokenizer.from_pretrained(AXK2_CHECKPOINT)
        model = AXK2ForCausalLM.from_pretrained(
            AXK2_CHECKPOINT,
            device_map="auto",
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=12, do_sample=False)
        self.assertEqual(EXPECTED_IDS, generated_ids[0, inputs["input_ids"].shape[1] :].tolist())

    def test_sdpa_matches_eager_generation(self):
        # SGA folds the indexer top-k into an additive mask consumed by both backends. Greedy short-form
        # generation is expected to agree between them on the same device (unlike the fast equivalence
        # tests, there is no batching/padding variation here).
        prompt = "대한민국의 수도는"

        tokenizer = AutoTokenizer.from_pretrained(AXK2_CHECKPOINT)
        inputs = tokenizer(prompt, return_tensors="pt")

        generations = {}
        for attn in ("eager", "sdpa"):
            model = AXK2ForCausalLM.from_pretrained(
                AXK2_CHECKPOINT,
                device_map="auto",
                dtype=torch.bfloat16,
                attn_implementation=attn,
            )
            device_inputs = inputs.to(model.device)
            generated_ids = model.generate(**device_inputs, max_new_tokens=12, do_sample=False)
            generations[attn] = generated_ids[0, device_inputs["input_ids"].shape[1] :].tolist()
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        self.assertEqual(generations["eager"], generations["sdpa"])
