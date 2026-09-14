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
"""Testing suite for the PyTorch GlmMoeDsa model."""

import os
import unittest
from unittest.mock import patch

import torch
from parameterized import parameterized

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    FineGrainedFP8Config,
    GlmMoeDsaConfig,
    is_torch_available,
    set_seed,
)
from transformers.distributed import DistributedConfig
from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa
from transformers.testing_utils import (
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_modeling_common import (
    TEST_EAGER_MATCHES_BATCHED_AND_GROUPED_INFERENCE_PARAMETERIZATION,
    TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION,
    ids_tensor,
)


if is_torch_available():
    from transformers import GlmMoeDsaForCausalLM, GlmMoeDsaModel


class GlmMoeDsaModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = GlmMoeDsaModel
        causal_lm_class = GlmMoeDsaForCausalLM

    def __init__(
        self,
        parent,
        n_routed_experts=8,
        kv_lora_rank=32,
        q_lora_rank=16,
        qk_nope_head_dim=64,
        qk_rope_head_dim=64,
        v_head_dim=128,
        num_hidden_layers=2,
        mlp_layer_types=["sparse", "dense"],
        index_topk=8,
    ):
        super().__init__(parent=parent, num_hidden_layers=num_hidden_layers)
        self.n_routed_experts = n_routed_experts
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.mlp_layer_types = mlp_layer_types
        self.index_topk = index_topk


@require_torch
class GlmMoeDsaModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = GlmMoeDsaModelTester
    test_all_params_have_gradient = False
    model_split_percents = [0.5, 0.7, 0.8]

    def _get_attention_shape(self, batch_size, seq_length, config):
        # This model caches the compressed MLA latents (`k_pass` as keys, `k_rot` as values) like DeepSeek-V3;
        # the shared DSA branch in `test_utils` still expects expanded K/V (the case for HY-V4, AXK2, GLM5-Next).
        return (batch_size, 1, seq_length, config.kv_lora_rank), (batch_size, 1, seq_length, config.qk_rope_head_dim)

    def _get_attention_kv_length(self, config, kv_length):
        # DSA returns the probabilities over the keys its indexer selected, `[B, H, S, min(index_topk, kv_length)]`
        return min(config.index_topk, kv_length)

    @unittest.skip("Float8 quantization + TP numerical noise exceeds match threshold")
    def test_tp_generation_quantized(self):
        pass

    def test_default_mlp_layer_types(self):
        config = GlmMoeDsaConfig(num_hidden_layers=8)
        self.assertEqual(
            config.mlp_layer_types, ["dense", "dense", "dense", "sparse", "sparse", "sparse", "sparse", "sparse"]
        )

    def test_indexer_types_respect_skip_topk_offset(self):
        config = GlmMoeDsaConfig(num_hidden_layers=8, index_topk_freq=4, index_skip_topk_offset=3)
        self.assertEqual(
            config.indexer_types,
            ["full", "full", "full", "shared", "shared", "shared", "full", "shared"],
        )

    def test_chunked_indexer_and_attention_match_unchunked(self):
        # The indexer scores and the sparse attention run in query chunks under a memory budget; chunking must not
        # change the selected keys or the outputs. The prompt exceeds `index_topk` so real selection happens.
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = GlmMoeDsaForCausalLM(config).to(torch_device).eval()
        input_ids = ids_tensor((2, config.index_topk + 5), config.vocab_size)
        with torch.no_grad():
            expected = model(input_ids).logits
            with (
                patch.object(modeling_glm_moe_dsa, "_INDEXER_SCORE_BUDGET", 1),
                patch.object(modeling_glm_moe_dsa, "_SPARSE_ATTENTION_BUDGET", 1),
            ):
                chunked = model(input_ids).logits
        torch.testing.assert_close(chunked, expected, rtol=1e-5, atol=1e-5)

    # DSA selects tokens with a hard top-k, which is discontinuous: a tiny numerical difference in the
    # indexer scores (attention backend, padding, batching, sequence packing) can flip which tokens are
    # selected and thus change the output, so these exact-equivalence tests do not hold for DSA.
    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    @unittest.skip("DSA hard top-k selection is sensitive to tiny numerical differences across backends.")
    def test_eager_matches_sdpa_inference(self, *args):
        pass

    @parameterized.expand(TEST_EAGER_MATCHES_BATCHED_AND_GROUPED_INFERENCE_PARAMETERIZATION)
    @unittest.skip("DSA hard top-k selection is sensitive to tiny numerical differences across batching.")
    def test_eager_matches_batched_and_grouped_inference(self, *args):
        pass

    @unittest.skip("DSA hard top-k selection is sensitive to sequence packing (selection can flip).")
    def test_eager_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip("DSA hard top-k selection is sensitive to sequence packing (selection can flip).")
    def test_sdpa_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip("Not sure MoE can pass this + indexer outputs are not deterministic wrt padding")
    def test_training_overfit(
        self,
    ):
        pass

    @require_torch_accelerator
    @slow
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        self.skipTest(reason="Qwen2Moe flash attention does not support right padding")

    @parameterized.expand([("random",), ("same",)])
    @unittest.skip("DSA indexer mask shape mismatch with assisted decoding")
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip("DSA indexer mask shape mismatch with assisted decoding")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip("DSA indexer mask shape mismatch with static cache")
    def test_generate_from_inputs_embeds_with_static_cache(self):
        pass

    @unittest.skip("DSA indexer mask shape mismatch with compiled forward")
    def test_generate_compile_model_forward_fullgraph(self):
        pass

    @unittest.skip("DSA indexer mask shape mismatch with compilation")
    def test_generate_compilation_all_outputs(self):
        pass

    @unittest.skip("DSA indexer mask shape mismatch with static cache")
    def test_generate_with_static_cache(self):
        pass


@require_torch_accelerator
@slow
class GlmMoeDsaIntegrationTest(unittest.TestCase):
    @unittest.skip("Test requires 2 nodes")
    def test_glm_moe_dsa_fp8_inference(self):
        # TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.run --nnodes=2 --nproc_per_node=8 --node_rank=0 --master_addr=ip-26-0-169-86 --master_port=29500
        set_seed(0)  # different ranks need the same seed
        model_id = "zai-org/GLM-5-FP8"

        quantization_config = FineGrainedFP8Config(
            modules_to_not_convert=[
                "model.layers.*.mlp.gate$",
                "model.layers.*.self_attn.indexer.weights_proj$",
                "lm_head",
            ],
            weight_block_size=(128, 128),
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            distributed_config=DistributedConfig(tp_size=int(os.environ["WORLD_SIZE"])),
            attn_implementation="eager",
        )

        prompt = ["Hi, introduce yourself", "The capital of France is known for"]
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=16,
            )

        output = tokenizer.batch_decode(outputs, skip_special_tokens=False)
        self.assertEqual(
            output,
            [
                "<|endoftext|><|endoftext|><|endoftext|>Hi, introduce yourself!\nI'm a 18 years old boy from Italy and I'm a student",
                "The capital of France is known for its rich history, culture, and the city of the of the of the of",
            ],
        )
