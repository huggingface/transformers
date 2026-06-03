import contextlib
import tempfile
import unittest
from functools import partial
from unittest.mock import patch

from parameterized import parameterized

from transformers import LlamaConfig
from transformers.heterogeneity import apply_heterogeneous_config


apply_heterogeneous_config_explicit = partial(apply_heterogeneous_config, explicit=True)


# ──────────────────────────────────────────────────────────────────────
# Tiny config factories
# ──────────────────────────────────────────────────────────────────────


def _tiny_llama_config(per_layer_config=None, **overrides):
    defaults = {
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 16,
        "vocab_size": 32,
        "max_position_embeddings": 64,
        **overrides,
    }
    return LlamaConfig(per_layer_config=per_layer_config, **defaults)


# ──────────────────────────────────────────────────────────────────────
# Tests: Config
# ──────────────────────────────────────────────────────────────────────


class TestHeterogeneousConfig(unittest.TestCase):
    def test_per_layer_config_skip_normalization(self):
        config = _tiny_llama_config(per_layer_config={1: {"skip": ["mlp", "attention"]}, 2: {"skip": []}})

        self.assertEqual(config.to_dict()["per_layer_config"], {"1": {"skip": ["attention", "mlp"]}})
        self.assertEqual(config.per_layer_config[0].skip, [])
        self.assertEqual(config.per_layer_config[1].skip, ["attention", "mlp"])
        self.assertEqual(config.per_layer_config[2].skip, [])

        set_config = _tiny_llama_config(per_layer_config={1: {"skip": {"attention"}}})
        self.assertEqual(set_config.to_dict()["per_layer_config"], {"1": {"skip": ["attention"]}})
        self.assertEqual(set_config.per_layer_config[1].skip, ["attention"])

        no_skip_config = _tiny_llama_config(per_layer_config={1: {"intermediate_size": 96}})
        self.assertNotIn("skip", no_skip_config.to_dict()["per_layer_config"]["1"])
        self.assertEqual(no_skip_config.per_layer_config[1].skip, [])

    @parameterized.expand(
        [
            ("string", "attention"),
            ("non_string_item", ["attention", 1]),
        ]
    )
    def test_per_layer_config_invalid_skip_raises(self, _name, invalid_skip):
        with self.assertRaises(TypeError):
            _tiny_llama_config(per_layer_config={0: {"skip": invalid_skip}})

    def test_per_layer_config_string_indices_are_normalized(self):
        config = _tiny_llama_config(per_layer_config={"01": {"num_key_value_heads": 2}})

        self.assertEqual(config.to_dict()["per_layer_config"], {"1": {"num_key_value_heads": 2}})
        self.assertEqual(config.per_layer_config[1].num_key_value_heads, 2)

    def test_per_layer_config_and_fallback(self):
        """Per-layer values should override, and non-overridden layers should fall back to global."""
        config = _tiny_llama_config(per_layer_config={1: {"num_key_value_heads": 2}, 3: {"num_key_value_heads": 1}})
        self.assertTrue(config.is_heterogeneous)
        self.assertEqual(config.per_layer_attributes, {"num_key_value_heads"})
        # Per-layer configs
        self.assertEqual(config.per_layer_config[1].num_key_value_heads, 2)
        self.assertEqual(config.per_layer_config[3].num_key_value_heads, 1)
        # Fallback to original global value
        self.assertEqual(config.per_layer_config[0].num_key_value_heads, 4)
        # Other attributes are unaffected
        self.assertEqual(config.per_layer_config[0].hidden_size, 64)

        # A single override should also preserve fallback for all other layers
        config2 = _tiny_llama_config(per_layer_config={1: {"num_key_value_heads": 2}})
        self.assertEqual(config2.per_layer_config[1].num_key_value_heads, 2)
        self.assertEqual(config2.per_layer_config[0].num_key_value_heads, 4)

    def test_per_layer_config_reflects_current_root_config_state(self):
        config = _tiny_llama_config(per_layer_config={0: {"intermediate_size": 64}})

        # PreTrainedModel.__init__ updates this after config construction.
        config._attn_implementation_internal = "sdpa"
        config.hidden_size = 96

        self.assertIs(type(config.per_layer_config[0]), type(config))
        self.assertFalse(config.per_layer_config[0].is_heterogeneous)
        self.assertIsNone(config.per_layer_config[0].per_layer_config)
        self.assertEqual(config.per_layer_config[0]._attn_implementation, "sdpa")
        self.assertEqual(config.per_layer_config[1]._attn_implementation, "sdpa")
        self.assertEqual(config.per_layer_config[0].hidden_size, 96)
        self.assertEqual(config.per_layer_config[1].hidden_size, 96)
        self.assertEqual(config.per_layer_config[0].intermediate_size, 64)
        self.assertEqual(config.per_layer_config[1].intermediate_size, 128)

        layer_dict = config.per_layer_config[0].to_dict()
        self.assertNotIn("per_layer_config", layer_dict)
        self.assertEqual(layer_dict["hidden_size"], 96)
        self.assertEqual(layer_dict["intermediate_size"], 64)

    def test_uniform_values_promoted_to_global(self):
        per_layer = {i: {"num_key_value_heads": 2} for i in range(4)}
        config = _tiny_llama_config(per_layer_config=per_layer)
        self.assertEqual(config.num_key_value_heads, 2)
        self.assertNotIn("num_key_value_heads", config.per_layer_attributes)

    def test_accessing_per_layer_attr_raises(self):
        config = _tiny_llama_config(per_layer_config={0: {"num_key_value_heads": 2}, 1: {"num_key_value_heads": 1}})
        with self.assertRaises(AttributeError):
            _ = config.num_key_value_heads

    def test_validation_missing_global_attr(self):
        # "fake_attr" in layer 0 but not in layer 1, and not global → should fail
        with self.assertRaises(ValueError):
            _tiny_llama_config(
                per_layer_config={
                    0: {"fake_attr": 42, "intermediate_size": 64},
                    1: {"intermediate_size": 96},
                }
            )

    @parameterized.expand(
        [
            ("negative", -1),
            ("too_large", 4),
        ]
    )
    def test_validation_layer_idx_out_of_range(self, _name, layer_idx):
        with self.assertRaises(ValueError):
            _tiny_llama_config(per_layer_config={layer_idx: {"num_key_value_heads": 2}})

    def test_save_pretrained_config_round_trip(self):
        """Config should survive save_pretrained → from_pretrained on disk."""
        per_layer = {i: {"intermediate_size": 64 + i} for i in range(0, 12, 2)}
        config = _tiny_llama_config(per_layer_config=per_layer, num_hidden_layers=12)

        # Keys are zero-padded so they sort numerically in JSON (0,1,...,10 not 0,1,10,2,...)
        d = config.to_dict()
        self.assertEqual(list(d["per_layer_config"].keys()), sorted(d["per_layer_config"].keys()))

        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_pretrained(tmpdir)
            loaded = LlamaConfig.from_pretrained(tmpdir)

        self.assertTrue(loaded.is_heterogeneous)
        for i in range(config.num_hidden_layers):
            self.assertEqual(
                config.per_layer_config[i].intermediate_size,
                loaded.per_layer_config[i].intermediate_size,
            )

    @parameterized.expand(
        [
            (
                "global_sw_global_acs",
                {"sliding_window": 4096, "attention_chunk_size": 2048},
                {0: {"intermediate_size": 64}},
                True,
            ),
            ("global_sw_per_layer_acs", {"sliding_window": 4096}, {0: {"attention_chunk_size": 2048}}, True),
            (
                "per_layer_sw_per_layer_acs_same_layer",
                {},
                {0: {"sliding_window": 4096, "attention_chunk_size": 2048}},
                True,
            ),
            (
                "per_layer_sw_per_layer_acs_different_layers",
                {"sliding_window": None, "attention_chunk_size": None},
                {0: {"sliding_window": 4096}, 1: {"attention_chunk_size": 2048}},
                False,
            ),
            (
                "global_conflict_resolved_by_per_layer_override",
                {"sliding_window": 4096, "attention_chunk_size": 2048},
                {
                    0: {"sliding_window": None},
                    1: {"sliding_window": None},
                    2: {"attention_chunk_size": None},
                    3: {"attention_chunk_size": None},
                },
                False,
            ),
        ],
    )
    def test_validation_sliding_window_and_attention_chunk_size(
        self, _name, overrides, per_layer_config, should_raise
    ):
        ctx = self.assertRaises(ValueError) if should_raise else contextlib.nullcontext()
        with ctx:
            _tiny_llama_config(per_layer_config=per_layer_config, **overrides)

    def test_all_layers_overridden_no_global_default(self):
        """Custom attribute on every layer without a global default should be accessible per layer."""
        config = _tiny_llama_config(
            per_layer_config={
                0: {"custom_attr": 10},
                1: {"custom_attr": 20},
                2: {"custom_attr": 30},
                3: {"custom_attr": 40},
            },
        )
        self.assertTrue(config.is_heterogeneous)
        self.assertEqual(config.per_layer_config[0].custom_attr, 10)
        self.assertEqual(config.per_layer_config[1].custom_attr, 20)
        self.assertEqual(config.per_layer_config[2].custom_attr, 30)
        self.assertEqual(config.per_layer_config[3].custom_attr, 40)

    @patch("transformers.configuration_utils.apply_heterogeneous_config", apply_heterogeneous_config_explicit)
    def test_explicit_fills_missing_layers_and_attributes(self):
        """explicit=True creates per-layer overrides for missing layers and fills missing attrs from global."""
        config = _tiny_llama_config(per_layer_config={0: {"num_key_value_heads": 1}})

        self.assertEqual(
            config.to_dict()["per_layer_config"],
            {
                "0": {"num_key_value_heads": 1},
                "1": {"num_key_value_heads": 4},
                "2": {"num_key_value_heads": 4},
                "3": {"num_key_value_heads": 4},
            },
        )

    @patch("transformers.configuration_utils.apply_heterogeneous_config", apply_heterogeneous_config_explicit)
    def test_explicit_does_not_promote_uniform_values(self):
        """explicit=True keeps uniform values per-layer instead of promoting to global."""
        per_layer = {i: {"num_key_value_heads": 2} for i in range(4)}
        # Without explicit: promoted to global (tested in test_uniform_values_promoted_to_global)
        # With explicit: stays per-layer
        config = _tiny_llama_config(per_layer_config=per_layer)
        self.assertIn("num_key_value_heads", config.per_layer_attributes)
        self.assertEqual(
            config.to_dict()["per_layer_config"],
            {str(i): {"num_key_value_heads": 2} for i in range(4)},
        )
