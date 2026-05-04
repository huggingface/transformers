import unittest

import torch

from transformers import DeepseekV3Config, DeepseekV3ForCausalLM, Glm4MoeConfig, Glm4MoeForCausalLM
from transformers.generation.candidate_generators import MTPCandidateGenerator
from transformers.generation.configuration_utils import GenerationMode
from transformers.testing_utils import require_torch


DEEPSEEK_V3_TINY_KW = {
    "hidden_size": 64,
    "intermediate_size": 64,
    "moe_intermediate_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "vocab_size": 100,
    "kv_lora_rank": 16,
    "q_lora_rank": 16,
    "qk_rope_head_dim": 8,
    "v_head_dim": 16,
    "qk_nope_head_dim": 16,
    "n_routed_experts": 4,
    "first_k_dense_replace": 1,
    "num_experts_per_tok": 2,
    "n_group": 1,
    "topk_group": 1,
    "max_position_embeddings": 64,
    "rope_parameters": {"rope_theta": 10000.0},
}

GLM4_MOE_TINY_KW = {
    "hidden_size": 64,
    "intermediate_size": 64,
    "moe_intermediate_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "vocab_size": 100,
    "n_routed_experts": 4,
    "first_k_dense_replace": 1,
    "num_experts_per_tok": 2,
    "n_group": 1,
    "topk_group": 1,
    "max_position_embeddings": 64,
    "rope_parameters": {"rope_theta": 10000.0},
}


@require_torch
class MTPGenerationModeTest(unittest.TestCase):
    def _attach_random_mtp(self, model):
        model.mtp_candidate_generator = MTPCandidateGenerator(model).eval()
        return model

    def test_use_mtp_routes_to_mtp_mode(self):
        cfg = DeepseekV3Config(num_nextn_predict_layers=1, **DEEPSEEK_V3_TINY_KW)
        model = DeepseekV3ForCausalLM(cfg)
        gc = model.generation_config
        gc.use_mtp = True
        gc.do_sample = False
        self.assertEqual(gc.get_generation_mode(), GenerationMode.MTP_DECODING)

    def test_use_mtp_on_greedy_matches_plain_greedy(self):
        """With a random-init MTP generator, rejection is frequent; MTP should fall back to bonus tokens
        from the base model and reproduce plain greedy decoding token-for-token."""
        for K in (1, 2, 3):
            torch.manual_seed(0)
            cfg = DeepseekV3Config(num_nextn_predict_layers=K, **DEEPSEEK_V3_TINY_KW)
            model = DeepseekV3ForCausalLM(cfg).eval()
            self._attach_random_mtp(model)
            ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
            torch.manual_seed(0)
            baseline = model.generate(ids, max_new_tokens=10, do_sample=False)
            torch.manual_seed(0)
            with_mtp = model.generate(ids, max_new_tokens=10, do_sample=False, use_mtp=True)
            self.assertTrue(torch.equal(baseline, with_mtp), f"mismatch for K={K}: {baseline} vs {with_mtp}")

    def test_use_mtp_without_generator_raises(self):
        cfg = DeepseekV3Config(num_nextn_predict_layers=1, **DEEPSEEK_V3_TINY_KW)
        model = DeepseekV3ForCausalLM(cfg).eval()
        ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "MTPCandidateGenerator"):
            model.generate(ids, max_new_tokens=3, do_sample=False, use_mtp=True)

    def test_generator_requires_num_mtp(self):
        cfg = DeepseekV3Config(num_nextn_predict_layers=0, **DEEPSEEK_V3_TINY_KW)
        model = DeepseekV3ForCausalLM(cfg).eval()
        with self.assertRaisesRegex(ValueError, "num_nextn_predict_layers"):
            MTPCandidateGenerator(model)

    def test_glm4_moe_greedy_match(self):
        torch.manual_seed(0)
        cfg = Glm4MoeConfig(num_nextn_predict_layers=2, **GLM4_MOE_TINY_KW)
        model = Glm4MoeForCausalLM(cfg).eval()
        self._attach_random_mtp(model)
        ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        torch.manual_seed(0)
        baseline = model.generate(ids, max_new_tokens=8, do_sample=False)
        torch.manual_seed(0)
        with_mtp = model.generate(ids, max_new_tokens=8, do_sample=False, use_mtp=True)
        self.assertTrue(torch.equal(baseline, with_mtp))


@require_torch
class MTPCandidateGeneratorTest(unittest.TestCase):
    def test_constructs_matching_decoder_class(self):
        cfg = DeepseekV3Config(num_nextn_predict_layers=2, **DEEPSEEK_V3_TINY_KW)
        model = DeepseekV3ForCausalLM(cfg)
        mtp = MTPCandidateGenerator(model)
        self.assertEqual(mtp.num_mtp, 2)
        self.assertEqual(len(mtp.layers), 2)
        sample_base_layer = model.model.layers[0]
        self.assertIsInstance(mtp.layers[0].mtp_block, type(sample_base_layer))

    def test_glm4_moe_decoder_class(self):
        cfg = Glm4MoeConfig(num_nextn_predict_layers=1, **GLM4_MOE_TINY_KW)
        model = Glm4MoeForCausalLM(cfg)
        mtp = MTPCandidateGenerator(model)
        sample_base_layer = model.model.layers[0]
        self.assertIsInstance(mtp.layers[0].mtp_block, type(sample_base_layer))

    def test_model_base_unchanged_by_num_nextn_predict_layers(self):
        """Setting `num_nextn_predict_layers > 0` must not modify the base model.
        MTP lives entirely in the companion generator."""
        cfg_a = DeepseekV3Config(num_nextn_predict_layers=0, **DEEPSEEK_V3_TINY_KW)
        cfg_b = DeepseekV3Config(num_nextn_predict_layers=3, **DEEPSEEK_V3_TINY_KW)
        torch.manual_seed(0)
        model_a = DeepseekV3ForCausalLM(cfg_a)
        torch.manual_seed(0)
        model_b = DeepseekV3ForCausalLM(cfg_b)
        self.assertEqual(len(model_a.model.layers), len(model_b.model.layers))
        self.assertFalse(hasattr(model_a.model, "forward_mtp"))
        self.assertFalse(hasattr(model_b.model, "forward_mtp"))


@require_torch
class MTPContinuousBatchingTest(unittest.TestCase):
    def test_generate_batch_with_use_mtp_raises_not_implemented(self):
        from transformers import GenerationConfig
        from transformers.generation.configuration_utils import ContinuousBatchingConfig
        from transformers.generation.continuous_batching import ContinuousBatchingManager

        cfg = DeepseekV3Config(num_nextn_predict_layers=1, **DEEPSEEK_V3_TINY_KW)
        model = DeepseekV3ForCausalLM(cfg).eval()
        gc = GenerationConfig(use_mtp=True, max_new_tokens=4)
        with self.assertRaisesRegex(NotImplementedError, "use_mtp=True"):
            ContinuousBatchingManager(model, gc, ContinuousBatchingConfig())


if __name__ == "__main__":
    unittest.main()
