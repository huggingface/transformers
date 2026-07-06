# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import contextlib
import tempfile
import unittest

from parameterized import parameterized

from transformers import LlamaConfig
from transformers.integrations.heterogeneity import AmbiguousGlobalPerLayerAttributeError
from transformers.utils import logging as transformers_logging


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

    def test_per_layer_config_reassignment_uses_existing_global_fallback(self):
        config = _tiny_llama_config(per_layer_config={0: {"num_key_value_heads": 2}})

        config.per_layer_config = {1: {"num_key_value_heads": 1}}

        self.assertEqual(config.to_dict()["per_layer_config"], {"1": {"num_key_value_heads": 1}})
        self.assertEqual(config.per_layer_config[0].num_key_value_heads, 4)
        self.assertEqual(config.per_layer_config[1].num_key_value_heads, 1)

    def test_per_layer_values_matching_global_are_removed_from_sparse_config(self):
        config = _tiny_llama_config(
            per_layer_config={
                0: {"num_key_value_heads": 4},
                1: {"num_key_value_heads": 2},
                2: {"num_key_value_heads": 4},
            }
        )

        self.assertEqual(config.to_dict()["per_layer_config"], {"1": {"num_key_value_heads": 2}})
        self.assertEqual(config.per_layer_attributes, {"num_key_value_heads"})
        self.assertEqual(config.per_layer_config[0].num_key_value_heads, 4)
        self.assertEqual(config.per_layer_config[1].num_key_value_heads, 2)
        self.assertEqual(config.per_layer_config[2].num_key_value_heads, 4)

    def test_uniform_per_layer_values_do_not_overwrite_global(self):
        per_layer = {layer_idx: {"num_key_value_heads": 2} for layer_idx in range(4)}
        config = _tiny_llama_config(per_layer_config=per_layer)

        self.assertEqual(object.__getattribute__(config, "num_key_value_heads"), 4)
        self.assertEqual(
            config.to_dict()["per_layer_config"],
            {str(i): {"num_key_value_heads": 2} for i in range(4)},
        )
        for layer_idx in range(4):
            self.assertEqual(config.per_layer_config[layer_idx].num_key_value_heads, 2)

    def test_explicit_serialization_restores_pruned_global_values(self):
        per_layer = {layer_idx: {"num_key_value_heads": 4} for layer_idx in range(4)}
        sparse_config = _tiny_llama_config(per_layer_config=per_layer)
        explicit_config = _tiny_llama_config(
            per_layer_config=per_layer,
            serialize_explicit_per_layer_config=True,
        )

        self.assertEqual(sparse_config.to_dict()["per_layer_config"], {})
        self.assertEqual(sparse_config.per_layer_attributes, set())
        self.assertEqual(
            explicit_config.to_dict()["per_layer_config"],
            {str(i): {"num_key_value_heads": 4} for i in range(4)},
        )

    def test_per_layer_config_reflects_current_global_config_state(self):
        config = _tiny_llama_config(per_layer_config={0: {"intermediate_size": 64}})

        # PreTrainedModel.__init__ updates this after config construction.
        config._attn_implementation_internal = "sdpa"
        config.hidden_size = 96
        config.intermediate_size = 192

        self.assertIs(type(config.per_layer_config[0]), type(config))
        self.assertFalse(config.per_layer_config[0].is_heterogeneous)
        self.assertIsNone(config.per_layer_config[0].per_layer_config)
        self.assertEqual(config.per_layer_config[0]._attn_implementation, "sdpa")
        self.assertEqual(config.per_layer_config[1]._attn_implementation, "sdpa")
        self.assertEqual(config.per_layer_config[0].hidden_size, 96)
        self.assertEqual(config.per_layer_config[1].hidden_size, 96)
        self.assertEqual(config.per_layer_config[0].intermediate_size, 64)
        self.assertEqual(config.per_layer_config[1].intermediate_size, 192)

        layer_dict = config.per_layer_config[0].to_dict()
        self.assertNotIn("per_layer_config", layer_dict)
        self.assertEqual(layer_dict["hidden_size"], 96)
        self.assertEqual(layer_dict["intermediate_size"], 64)

    def test_accessing_per_layer_attr_raises(self):
        config = _tiny_llama_config(per_layer_config={0: {"num_key_value_heads": 2}, 1: {"num_key_value_heads": 1}})
        with self.assertRaisesRegex(
            AmbiguousGlobalPerLayerAttributeError, "allow_global_per_layer_attribute_access.*global value incorrectly"
        ):
            _ = config.num_key_value_heads

    def test_ambiguous_global_attr_access_is_not_treated_as_missing(self):
        config = _tiny_llama_config(per_layer_config={0: {"num_key_value_heads": 2}, 1: {"num_key_value_heads": 1}})

        self.assertIn("num_key_value_heads", config.__dict__)
        with self.assertRaises(AmbiguousGlobalPerLayerAttributeError):
            getattr(config, "num_key_value_heads", "default")
        with self.assertRaises(AmbiguousGlobalPerLayerAttributeError):
            hasattr(config, "num_key_value_heads")

    def test_allow_global_per_layer_attribute_access(self):
        config = _tiny_llama_config(
            per_layer_config={0: {"num_key_value_heads": 2}, 1: {"num_key_value_heads": 1}},
            allow_global_per_layer_attribute_access=True,
        )
        logger = transformers_logging.get_logger("transformers.integrations.heterogeneity.configuration_utils")
        logger.warning_once.cache_clear()

        with self.assertLogs(logger=logger, level="WARNING") as logs:
            self.assertEqual(config.num_key_value_heads, 4)

        self.assertIn("Reading global config value for per-layer attribute `num_key_value_heads`", logs.output[0])
        self.assertEqual(config.per_layer_config[0].num_key_value_heads, 2)
        self.assertEqual(config.per_layer_config[1].num_key_value_heads, 1)
        self.assertEqual(config.per_layer_config[2].num_key_value_heads, 4)

    def test_flags_are_applied_from_pretrained_kwargs(self):
        """The flags are properties, so `from_dict`'s kwargs handling applies them like `per_layer_config`."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _tiny_llama_config().save_pretrained(tmpdir)
            config = LlamaConfig.from_pretrained(
                tmpdir,
                per_layer_config={1: {"num_key_value_heads": 2}},
                allow_global_per_layer_attribute_access=True,
                serialize_explicit_per_layer_config=True,
            )

        self.assertEqual(config.num_key_value_heads, 4)
        self.assertEqual(
            config.to_dict()["per_layer_config"],
            {str(i): {"num_key_value_heads": 2 if i == 1 else 4} for i in range(4)},
        )
        # The property defaults must not leak into the serialization of configs that never set the flags.
        self.assertNotIn("allow_global_per_layer_attribute_access", _tiny_llama_config().to_dict())

    def test_non_per_layer_attributes_do_not_warn(self):
        config = _tiny_llama_config(per_layer_config={0: {"num_key_value_heads": 2}, 1: {"num_key_value_heads": 1}})
        logger = transformers_logging.get_logger("transformers.integrations.heterogeneity.configuration_utils")
        logger.warning_once.cache_clear()

        with self.assertNoLogs(logger=logger, level="WARNING"):
            self.assertEqual(config.hidden_size, 64)

    def test_iter_skips_per_layer_attributes_by_default(self):
        config = _tiny_llama_config(per_layer_config={0: {"num_key_value_heads": 2}, 1: {"num_key_value_heads": 1}})

        keys = list(config)

        self.assertNotIn("num_key_value_heads", keys)
        self.assertIn("hidden_size", keys)

    def test_iter_includes_per_layer_attributes_when_global_access_allowed(self):
        config = _tiny_llama_config(
            per_layer_config={0: {"num_key_value_heads": 2}, 1: {"num_key_value_heads": 1}},
            allow_global_per_layer_attribute_access=True,
        )

        self.assertIn("num_key_value_heads", list(config))

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

    def test_per_layer_config_can_serialize_explicit_layer_overrides(self):
        sparse_config = _tiny_llama_config(per_layer_config={0: {"num_key_value_heads": 1}})
        explicit_config = _tiny_llama_config(
            per_layer_config={0: {"num_key_value_heads": 1}},
            serialize_explicit_per_layer_config=True,
        )
        explicit_config.num_key_value_heads = 8
        explicit_config_dict = explicit_config.to_dict()

        self.assertEqual(sparse_config.to_dict()["per_layer_config"], {"0": {"num_key_value_heads": 1}})
        self.assertEqual(
            explicit_config_dict["per_layer_config"],
            {
                "0": {"num_key_value_heads": 1},
                "1": {"num_key_value_heads": 8},
                "2": {"num_key_value_heads": 8},
                "3": {"num_key_value_heads": 8},
            },
        )
