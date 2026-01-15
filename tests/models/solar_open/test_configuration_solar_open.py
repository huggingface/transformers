import unittest

from transformers import SolarOpenConfig


class SolarOpenConfigTest(unittest.TestCase):
    def test_default_config(self):
        """
        Simple test for SolarOpenConfig default values
        """
        config = SolarOpenConfig()
        self.assertEqual(config.model_type, "solar_open")
        self.assertEqual(config.n_routed_experts, 128)
        self.assertEqual(config.n_shared_experts, 1)
        self.assertEqual(config.n_group, 1)
        self.assertEqual(config.topk_group, 1)
        self.assertEqual(config.num_experts_per_tok, 8)
        self.assertEqual(config.moe_intermediate_size, 1280)
        self.assertEqual(config.routed_scaling_factor, 1.0)
        self.assertTrue(config.norm_topk_prob)
        self.assertFalse(config.use_qk_norm)
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.num_hidden_layers, 48)
        self.assertEqual(config.num_attention_heads, 64)
        self.assertEqual(config.head_dim, 128)
        self.assertEqual(config.num_key_value_heads, 8)
        self.assertEqual(config.vocab_size, 196608)
        self.assertEqual(config.rms_norm_eps, 1e-05)
        self.assertEqual(config.max_position_embeddings, 131072)
        self.assertEqual(config.partial_rotary_factor, 1.0)
        self.assertEqual(config.rope_parameters["rope_theta"], 1_000_000)
        self.assertEqual(config.rope_parameters["rope_type"], "yarn")
        self.assertEqual(config.rope_parameters["partial_rotary_factor"], 1.0)
        self.assertEqual(config.rope_parameters["factor"], 2.0)
        self.assertEqual(config.rope_parameters["original_max_position_embeddings"], 65_536)
        self.assertEqual(config.tie_word_embeddings, False)

    def test_rope_parameters_partially_initialized(self):
        """
        Test for SolarOpenConfig when rope_parameters is partially initialized
        """
        config = SolarOpenConfig(
            rope_parameters={
                "rope_type": "yarn",
                "factor": 2.0,
                "original_max_position_embeddings": 65536,
            }
        )
        self.assertEqual(config.rope_parameters["partial_rotary_factor"], 1.0)
        self.assertEqual(config.rope_parameters["rope_theta"], 1_000_000)

    def test_partial_rotary_factor_bc(self):
        """
        Test for SolarOpenConfig when partial_rotary_factor is set in rope_parameters
        """
        config = SolarOpenConfig(
            rope_parameters={
                "rope_type": "yarn",
                "factor": 2.0,
                "original_max_position_embeddings": 65536,
            },
            partial_rotary_factor=0.7,
        )
        self.assertEqual(config.rope_parameters["partial_rotary_factor"], 0.7)

    def test_disable_rope_scaling(self):
        """
        Test for SolarOpenConfig when rope_scaling is disabled
        """
        config = SolarOpenConfig(
            rope_parameters={
                "rope_type": "default",
            }
        )
        self.assertEqual(config.rope_parameters["rope_type"], "default")
        self.assertNotIn("factor", config.rope_parameters)
        self.assertNotIn("original_max_position_embeddings", config.rope_parameters)
        self.assertEqual(config.rope_parameters["rope_theta"], 1_000_000)
        self.assertEqual(config.rope_parameters["partial_rotary_factor"], 1.0)

    def test_rope_backward_compatibility(self):
        """
        Test for SolarOpenConfig backward compatibility for rope_parameters
        """
        config = SolarOpenConfig(
            rope_scaling={"type": "yarn", "factor": 2.0, "original_max_position_embeddings": 65536},
            partial_rotary_factor=1.0,
            rope_theta=1_000_000,
        )
        self.assertEqual(config.rope_parameters["rope_theta"], 1_000_000)
        self.assertEqual(config.rope_parameters["partial_rotary_factor"], 1.0)
        self.assertEqual(config.rope_parameters["rope_type"], "yarn")
        self.assertEqual(config.rope_parameters["factor"], 2.0)
        self.assertEqual(config.rope_parameters["original_max_position_embeddings"], 65_536)
