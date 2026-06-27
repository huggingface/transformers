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
"""Testing suite for the PyTorch Glm5Next model."""

import unittest

import torch
from parameterized import parameterized

from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    Cache,
    FineGrainedFP8Config,
    Glm5NextConfig,
    is_torch_available,
    set_seed,
)
from transformers.testing_utils import (
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_modeling_common import (
    TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION,
)


if is_torch_available():
    from transformers import Glm5NextForCausalLM, Glm5NextModel
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextIndexer


class Glm5NextModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = Glm5NextModel
        causal_lm_class = Glm5NextForCausalLM

    def __init__(
        self,
        parent,
        n_routed_experts=8,
        kv_lora_rank=32,
        q_lora_rank=16,
        qk_nope_head_dim=256,
        qk_rope_head_dim=0,
        v_head_dim=128,
        num_hidden_layers=2,
        mlp_layer_types=None,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=8,
        linear_attn_config=None,
    ):
        super().__init__(parent=parent, num_hidden_layers=num_hidden_layers)
        self.n_routed_experts = n_routed_experts
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.mlp_layer_types = mlp_layer_types or ["dense", "sparse"]
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.linear_attn_config = linear_attn_config or {
            "full_attn_layers": list(range(num_hidden_layers)),
            "head_dim": 128,
            "kda_layers": [],
            "num_heads": self.num_attention_heads,
            "short_conv_kernel_size": 4,
        }


@require_torch
class Glm5NextModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = Glm5NextModelTester
    test_all_params_have_gradient = False
    model_split_percents = [0.5, 0.7, 0.8]

    @unittest.skip("Float8 quantization + TP numerical noise exceeds match threshold")
    def test_tp_generation_quantized(self):
        pass

    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config):
        """Needs to be overridden as GLM-4.7-Flash has special MLA cache format (though we don't really use the MLA)"""
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

    def get_kpool_config(self, **kwargs):
        values = {
            "vocab_size": 32,
            "pad_token_id": 0,
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "q_lora_rank": 2,
            "kv_lora_rank": 2,
            "qk_nope_head_dim": 2,
            "qk_rope_head_dim": 0,
            "v_head_dim": 2,
            "index_head_dim": 2,
            "index_n_heads": 1,
            "index_topk": 2,
            "index_kpool": 2,
            "index_kpool_compress": True,
            "index_kpool_always_select_tail": True,
            "index_dsa_use_layernorm": False,
            "indexer_types": ["full"],
            "linear_attn_config": {
                "full_attn_layers": [0],
                "kda_layers": [],
                "head_dim": 2,
                "num_heads": 1,
                "short_conv_kernel_size": 2,
            },
            "mlp_layer_types": ["dense"],
        }
        values.update(kwargs)
        return Glm5NextConfig(**values)

    def test_kpool_parameters_affect_forward_selection(self):
        indexer = Glm5NextIndexer(self.get_kpool_config(), layer_idx=0).eval()
        hidden_states = torch.tensor(
            [[[10.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [20.0, 0.0, 0.0, 0.0]]]
        )
        q_resid = torch.tensor([[[1.0, 0.0]]]).expand(1, 4, 2)
        cos = torch.empty(1, 4, 0)
        sin = torch.empty(1, 4, 0)
        causal_mask = torch.triu(torch.full((1, 1, 4, 4), float("-inf")), diagonal=1)

        with torch.no_grad():
            indexer.wk.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]))
            indexer.wq_b.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 0.0]]))
            indexer.weights_proj.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
            indexer.index_kpool_compress_gate.zero_()
            indexer.index_kpool_compress_ape.copy_(torch.tensor([[8.0, 0.0], [-8.0, 0.0]]))
            select_slot_zero, _ = indexer(
                hidden_states,
                q_resid,
                (cos, sin),
                causal_mask,
            )
            indexer.index_kpool_compress_ape.copy_(torch.tensor([[-8.0, 0.0], [8.0, 0.0]]))
            select_slot_one, _ = indexer(
                hidden_states,
                q_resid,
                (cos, sin),
                causal_mask,
            )

        self.assertFalse(torch.equal(select_slot_zero[:, -1], select_slot_one[:, -1]))

    def test_kpool_cache_matches_full_context(self):
        config = self.get_kpool_config()
        config._attn_implementation = "eager"
        model = Glm5NextForCausalLM(config).to(torch_device).eval()
        input_ids = torch.tensor([[2, 3, 5, 7, 11, 13]], device=torch_device)

        with torch.no_grad():
            full_outputs = model(input_ids, use_cache=True)
            prefix_outputs = model(input_ids[:, :4], use_cache=True)
            cached_outputs = model(
                input_ids[:, 4:],
                past_key_values=prefix_outputs.past_key_values,
                use_cache=True,
            )

        torch.testing.assert_close(
            cached_outputs.logits,
            full_outputs.logits[:, 4:],
            rtol=1e-4,
            atol=1e-4,
        )
        cache_layer = cached_outputs.past_key_values.layers[0]
        self.assertEqual(cache_layer.indexer_keys.shape[1], input_ids.shape[1])
        self.assertEqual(cache_layer.indexer_gate_scores.shape[1], input_ids.shape[1])

    def test_kpool_cache_reorders_indexer_state(self):
        model = Glm5NextForCausalLM(self.get_kpool_config()).to(torch_device).eval()
        input_ids = torch.tensor([[2, 3, 5, 7], [11, 13, 17, 19]], device=torch_device)

        with torch.no_grad():
            cache = model(input_ids, use_cache=True).past_key_values

        layer = cache.layers[0]
        expected_keys = layer.indexer_keys[[1, 0]].clone()
        expected_gate_scores = layer.indexer_gate_scores[[1, 0]].clone()
        cache.reorder_cache(torch.tensor([1, 0], device=torch_device))

        torch.testing.assert_close(layer.indexer_keys, expected_keys)
        torch.testing.assert_close(layer.indexer_gate_scores, expected_gate_scores)

    def test_default_mlp_layer_types(self):
        config = Glm5NextConfig(
            num_hidden_layers=8,
            mlp_layer_types=["dense", "dense", "dense", "sparse", "sparse", "sparse", "sparse", "sparse"],
        )
        self.assertEqual(
            config.mlp_layer_types, ["dense", "dense", "dense", "sparse", "sparse", "sparse", "sparse", "sparse"]
        )

    def test_default_linear_attn_config(self):
        config = Glm5NextConfig(num_hidden_layers=8)
        self.assertFalse(config.mhc)
        self.assertEqual(config.linear_attn_config["kda_layers"], [0, 1, 2, 4, 5, 6])
        self.assertEqual(config.linear_attn_config["full_attn_layers"], [3, 7])
        self.assertEqual(config.linear_attn_config["head_dim"], 128)
        self.assertEqual(config.linear_attn_config["num_heads"], 64)
        self.assertEqual(config.linear_attn_config["short_conv_kernel_size"], 4)
        self.assertEqual(config.linear_attn_config["lower_bound"], None)
        self.assertFalse(config.linear_attn_config["safe_gate"])
        self.assertEqual(
            config.layer_types,
            [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
        )

    def test_kpool_config_roundtrip_and_defaults(self):
        default_config = Glm5NextConfig()
        self.assertFalse(default_config.index_kpool_always_select_tail)

        config = Glm5NextConfig(
            index_kpool=16,
            index_kpool_compress=True,
            index_kpool_always_select_tail=True,
        )
        restored = Glm5NextConfig.from_dict(config.to_dict())

        self.assertEqual(restored.index_kpool, 16)
        self.assertTrue(restored.index_kpool_compress)
        self.assertTrue(restored.index_kpool_always_select_tail)

    @parameterized.expand(["linear", "dynamic", "yarn"])
    def test_model_rope_scaling_from_config(self, scaling_type):
        self.skipTest("GLM-5-Next full-attention checkpoints use no-RoPE MLA, so RoPE scaling is not exercised")

    def test_model_rope_scaling_frequencies(self):
        self.skipTest("GLM-5-Next full-attention checkpoints use no-RoPE MLA, so RoPE scaling is not exercised")

    def test_reverse_loading_mapping(self):
        self.skipTest("GLM-5-Next keeps external checkpoint-only HC key renames covered by a dedicated mapping test")

    def test_auto_model_registration(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        self.assertIsInstance(AutoModel.from_config(config), Glm5NextModel)
        self.assertIsInstance(AutoModelForCausalLM.from_config(config), Glm5NextForCausalLM)

    def test_tiny_causal_lm_forward_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config._attn_implementation = "eager"
        model = Glm5NextForCausalLM(config).to(torch_device).eval()

        with torch.no_grad():
            outputs = model(**inputs_dict, output_hidden_states=True, output_attentions=True)

        self.assertEqual(
            outputs.logits.shape,
            (self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.vocab_size),
        )
        self.assertEqual(len(outputs.hidden_states), self.model_tester.num_hidden_layers + 1)
        self.assertEqual(len(outputs.attentions), self.model_tester.num_hidden_layers)

    def test_tiny_generation_path(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = Glm5NextForCausalLM(config).to(torch_device).eval()

        with torch.no_grad():
            generated_ids = model.generate(
                inputs_dict["input_ids"][:1, :3],
                max_new_tokens=1,
                do_sample=False,
            )

        self.assertEqual(generated_ids.shape, (1, 4))

    def test_cache_for_full_attention_tiny_model(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = Glm5NextForCausalLM(config).to(torch_device).eval()

        with torch.no_grad():
            outputs = model(**inputs_dict, use_cache=True)

        self._check_past_key_values_for_generate(
            self.model_tester.batch_size,
            outputs.past_key_values,
            self.model_tester.seq_length,
            config,
        )

    def test_kda_cache_matches_full_context(self):
        config = Glm5NextConfig(
            vocab_size=99,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            q_lora_rank=4,
            kv_lora_rank=4,
            qk_nope_head_dim=4,
            qk_rope_head_dim=0,
            v_head_dim=4,
            linear_attn_config={
                "full_attn_layers": [1],
                "head_dim": 4,
                "kda_layers": [0],
                "num_heads": 2,
                "short_conv_kernel_size": 4,
                "v_head_dim": 4,
                "lower_bound": -5.0,
            },
            mlp_layer_types=["dense", "dense"],
            pad_token_id=0,
        )
        model = Glm5NextForCausalLM(config).to(torch_device).eval()
        input_ids = torch.tensor([[5, 17, 23, 31, 43, 59]], device=torch_device)

        with torch.no_grad():
            full_outputs = model(input_ids, use_cache=True)
            prefix_outputs = model(input_ids[:, :4], use_cache=True)
            cached_outputs = model(
                input_ids[:, 4:],
                past_key_values=prefix_outputs.past_key_values,
                use_cache=True,
            )

        self.assertTrue(cached_outputs.past_key_values.has_previous_state(0))
        self.assertEqual(config.layer_types, ["linear_attention", "full_attention"])
        torch.testing.assert_close(
            cached_outputs.logits,
            full_outputs.logits[:, 4:],
            rtol=1e-4,
            atol=1e-4,
        )

    def test_prepare_inputs_keeps_kda_cache_for_generation(self):
        config = Glm5NextConfig(
            vocab_size=99,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            q_lora_rank=4,
            kv_lora_rank=4,
            qk_nope_head_dim=4,
            qk_rope_head_dim=0,
            v_head_dim=4,
            linear_attn_config={
                "full_attn_layers": [1],
                "head_dim": 4,
                "kda_layers": [0],
                "num_heads": 2,
                "short_conv_kernel_size": 4,
                "lower_bound": -5.0,
            },
            mlp_layer_types=["dense", "dense"],
            pad_token_id=0,
        )
        model = Glm5NextForCausalLM(config).to(torch_device).eval()
        input_ids = torch.tensor([[5, 17, 23, 31]], device=torch_device)

        with torch.no_grad():
            prefix_outputs = model(input_ids[:, :3], use_cache=True)

        prepared_inputs = model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=prefix_outputs.past_key_values,
            attention_mask=torch.ones_like(input_ids),
            use_cache=True,
        )

        self.assertIs(prepared_inputs["past_key_values"], prefix_outputs.past_key_values)
        self.assertTrue(prepared_inputs["use_cache"])

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    @unittest.skip("GLM-5-Next has custom MLA/KDA attention paths that need a dedicated SDPA equivalence test")
    def test_eager_matches_sdpa_inference(self, *args):
        pass

    @unittest.skip("Not sure MoE can pass this + indexer outputs are not deterministic wrt padding")
    def test_left_padding_compatibility(
        self,
    ):
        pass

    @unittest.skip("Not sure MoE can pass this + indexer outputs are not deterministic wrt padding")
    def test_sdpa_padding_matches_padding_free_with_position_ids(
        self,
    ):
        pass

    @unittest.skip("Not sure MoE can pass this + indexer outputs are not deterministic wrt padding")
    def test_training_overfit(
        self,
    ):
        pass

    @require_torch_accelerator
    @slow
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        self.skipTest(reason="GLM-5-Next does not enable Flash Attention 2 in this implementation")

    @unittest.skip("DSA indexer mask shape mismatch with assisted decoding")
    @parameterized.expand([("random",), ("same",)])
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip("DSA indexer mask shape mismatch with assisted decoding")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip("DSA indexer mask shape mismatch with inputs_embeds generation")
    def test_generate_from_inputs_embeds(self):
        pass

    @unittest.skip("DSA indexer mask shape mismatch with prompt lookup decoding")
    def test_prompt_lookup_decoding_matches_greedy_search(self):
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
class Glm5NextIntegrationTest(unittest.TestCase):
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
            tp_plan="auto",
            attn_implementation="eager",
        )

        prompt = ["Hi, introduce yourself", "The capital of France is known for"]
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=16,
            )

        output = tokenizer.decode(outputs, skip_special_tokens=False)
        self.assertEqual(
            output,
            [
                "<|endoftext|><|endoftext|><|endoftext|>Hi, introduce yourself!\nI'm a 18 years old boy from Italy and I'm a student",
                "The capital of France is known for its rich history, culture, and the city of the of the of the of",
            ],
        )
