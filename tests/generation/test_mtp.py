import unittest

import torch

from transformers import DeepseekV3Config, DeepseekV3ForCausalLM, Glm4MoeConfig, Glm4MoeForCausalLM
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
    def test_use_mtp_routes_to_mtp_mode(self):
        cfg = DeepseekV3Config(num_nextn_predict_layers=1, **DEEPSEEK_V3_TINY_KW)
        model = DeepseekV3ForCausalLM(cfg)
        gc = model.generation_config
        gc.use_mtp = True
        gc.do_sample = False
        self.assertEqual(gc.get_generation_mode(), GenerationMode.MTP_DECODING)

    def test_use_mtp_on_greedy_matches_plain_greedy(self):
        """With random-init weights, rejection rate is high; MTP should fall back to bonus tokens from the base
        model and reproduce plain greedy decoding token-for-token."""
        for K in (1, 2, 3):
            torch.manual_seed(0)
            cfg = DeepseekV3Config(num_nextn_predict_layers=K, **DEEPSEEK_V3_TINY_KW)
            model = DeepseekV3ForCausalLM(cfg).eval()
            ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
            torch.manual_seed(0)
            baseline = model.generate(ids, max_new_tokens=10, do_sample=False)
            torch.manual_seed(0)
            with_mtp = model.generate(ids, max_new_tokens=10, do_sample=False, use_mtp=True)
            self.assertTrue(torch.equal(baseline, with_mtp), f"mismatch for K={K}: {baseline} vs {with_mtp}")

    def test_use_mtp_without_mtp_layers_raises(self):
        cfg = DeepseekV3Config(num_nextn_predict_layers=0, **DEEPSEEK_V3_TINY_KW)
        model = DeepseekV3ForCausalLM(cfg).eval()
        ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "num_nextn_predict_layers"):
            model.generate(ids, max_new_tokens=3, do_sample=False, use_mtp=True)

    def test_glm4_moe_greedy_match(self):
        torch.manual_seed(0)
        cfg = Glm4MoeConfig(num_nextn_predict_layers=2, **GLM4_MOE_TINY_KW)
        model = Glm4MoeForCausalLM(cfg).eval()
        ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        torch.manual_seed(0)
        baseline = model.generate(ids, max_new_tokens=8, do_sample=False)
        torch.manual_seed(0)
        with_mtp = model.generate(ids, max_new_tokens=8, do_sample=False, use_mtp=True)
        self.assertTrue(torch.equal(baseline, with_mtp))


@require_torch
class MTPModelLoadingTest(unittest.TestCase):
    def test_deepseek_v3_extends_layers(self):
        cfg = DeepseekV3Config(num_nextn_predict_layers=2, **DEEPSEEK_V3_TINY_KW)
        model = DeepseekV3ForCausalLM(cfg)
        total = cfg.num_hidden_layers + cfg.num_nextn_predict_layers
        self.assertEqual(len(model.model.layers), total)
        self.assertEqual(type(model.model.layers[-1]).__name__, "DeepseekV3MTPLayer")
        self.assertEqual(type(model.model.layers[cfg.num_hidden_layers - 1]).__name__, "DeepseekV3DecoderLayer")

    def test_glm4_moe_extends_layers(self):
        cfg = Glm4MoeConfig(num_nextn_predict_layers=1, **GLM4_MOE_TINY_KW)
        model = Glm4MoeForCausalLM(cfg)
        total = cfg.num_hidden_layers + cfg.num_nextn_predict_layers
        self.assertEqual(len(model.model.layers), total)
        self.assertEqual(type(model.model.layers[-1]).__name__, "Glm4MoeMTPLayer")

    def test_base_forward_ignores_mtp_layers(self):
        """Extending self.layers with MTP modules must not change the base forward output."""
        torch.manual_seed(0)
        cfg_no_mtp = DeepseekV3Config(num_nextn_predict_layers=0, **DEEPSEEK_V3_TINY_KW)
        model_no_mtp = DeepseekV3ForCausalLM(cfg_no_mtp).eval()
        base_state = model_no_mtp.state_dict()

        torch.manual_seed(0)
        cfg_mtp = DeepseekV3Config(num_nextn_predict_layers=1, **DEEPSEEK_V3_TINY_KW)
        model_mtp = DeepseekV3ForCausalLM(cfg_mtp).eval()
        # Copy the shared parameters so base-forward paths compare like-for-like.
        mtp_state = model_mtp.state_dict()
        for k, v in base_state.items():
            if k in mtp_state:
                mtp_state[k].copy_(v)

        ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        with torch.no_grad():
            out_a = model_no_mtp(ids).logits
            out_b = model_mtp(ids).logits
        torch.testing.assert_close(out_a, out_b, atol=1e-5, rtol=1e-5)

    def test_forward_mtp_shapes(self):
        cfg = DeepseekV3Config(num_nextn_predict_layers=2, **DEEPSEEK_V3_TINY_KW)
        model = DeepseekV3ForCausalLM(cfg).eval()
        ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        with torch.no_grad():
            base_out = model.model(ids, use_cache=True)
            h = base_out.last_hidden_state[:, -1:, :]
            cache = base_out.past_key_values
            for depth in range(cfg.num_nextn_predict_layers):
                tok = torch.tensor([[5 + depth]], dtype=torch.long)
                pos = torch.tensor([[ids.shape[1] + depth]], dtype=torch.long)
                h, logits = model.model.forward_mtp(
                    input_ids=tok,
                    previous_hidden_state=h,
                    past_key_values=cache,
                    position_ids=pos,
                    mtp_depth=depth,
                )
                self.assertEqual(h.shape, (1, 1, cfg.hidden_size))
                self.assertEqual(logits.shape, (1, 1, cfg.vocab_size))


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
