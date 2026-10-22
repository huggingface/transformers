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

import tempfile
import unittest

from parameterized import parameterized

from tests.heterogeneity.testing_utils import tiny_llama_config
from transformers import LlamaConfig
from transformers.integrations.heterogeneity import AmbiguousGlobalPerLayerAttributeError
from transformers.utils import logging as transformers_logging


# ──────────────────────────────────────────────────────────────────────
# Tests: Config
# ──────────────────────────────────────────────────────────────────────


class TestHeterogeneousConfig(unittest.TestCase):
    def test_per_layer_config_skip_normalization(self):
        config = tiny_llama_config(per_layer_config={1: {"skip": ["mlp", "attention"]}, 2: {"skip": []}})

        self.assertEqual(config.to_dict()["per_layer_config"], {"1": {"skip": ["attention", "mlp"]}})
        self.assertEqual(config.per_layer_config[0].skip, [])
        self.assertEqual(config.per_layer_config[1].skip, ["attention", "mlp"])
        self.assertEqual(config.per_layer_config[2].skip, [])

        set_config = tiny_llama_config(per_layer_config={1: {"skip": {"attention"}}})
        self.assertEqual(set_config.to_dict()["per_layer_config"], {"1": {"skip": ["attention"]}})
        self.assertEqual(set_config.per_layer_config[1].skip, ["attention"])

        no_skip_config = tiny_llama_config(per_layer_config={1: {"intermediate_size": 96}})
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
            tiny_llama_config(per_layer_config={0: {"skip": invalid_skip}})

    def test_per_layer_config_string_indices_are_normalized(self):
        config = tiny_llama_config(per_layer_config={"01": {"num_key_value_heads": 2}})

        self.assertEqual(config.to_dict()["per_layer_config"], {"1": {"num_key_value_heads": 2}})
        self.assertEqual(config.per_layer_config[1].num_key_value_heads, 2)

    def test_per_layer_config_cannot_be_nested(self):
        with self.assertRaisesRegex(ValueError, "cannot be nested within itself"):
            tiny_llama_config(per_layer_config={0: {"per_layer_config": {0: {"intermediate_size": 32}}}})

    @parameterized.expand(
        [
            ("_attn_implementation", "sdpa"),
            ("_attn_implementation_internal", "sdpa"),
            ("_experts_implementation", "batched_mm"),
            ("_experts_implementation_internal", "batched_mm"),
            ("is_causal", False),
        ]
    )
    def test_global_attributes_cannot_be_overridden_per_layer(self, attribute, value):
        with self.assertRaisesRegex(ValueError, f"layer 1.*{attribute}"):
            tiny_llama_config(per_layer_config={1: {attribute: value}})

    def test_per_layer_config_and_fallback(self):
        """Per-layer values should override, and non-overridden layers should fall back to global."""
        config = tiny_llama_config(per_layer_config={1: {"num_key_value_heads": 2}, 3: {"num_key_value_heads": 1}})
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
        config = tiny_llama_config(per_layer_config={0: {"num_key_value_heads": 2}})

        config.per_layer_config = {1: {"num_key_value_heads": 1}}

        self.assertEqual(config.to_dict()["per_layer_config"], {"1": {"num_key_value_heads": 1}})
        self.assertEqual(config.per_layer_config[0].num_key_value_heads, 4)
        self.assertEqual(config.per_layer_config[1].num_key_value_heads, 1)

    def test_per_layer_values_matching_global_are_removed_from_sparse_config(self):
        config = tiny_llama_config(
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
        config = tiny_llama_config(per_layer_config=per_layer)

        self.assertEqual(object.__getattribute__(config, "num_key_value_heads"), 4)
        self.assertEqual(
            config.to_dict()["per_layer_config"],
            {str(i): {"num_key_value_heads": 2} for i in range(4)},
        )
        for layer_idx in range(4):
            self.assertEqual(config.per_layer_config[layer_idx].num_key_value_heads, 2)

    def test_indexing_by_layer_type(self):
        config = tiny_llama_config(
            per_layer_config={1: {"num_key_value_heads": 2}, 3: {"num_key_value_heads": 2}},
            layer_types=["full_attention", "sliding_attention"] * 2,
        )

        self.assertEqual(config.per_layer_config["full_attention"].num_key_value_heads, 4)
        self.assertEqual(config.per_layer_config["sliding_attention"].num_key_value_heads, 2)

        # A layer type's config is the config of its first layer
        self.assertIs(config.per_layer_config["full_attention"], config.per_layer_config[0])
        self.assertIs(config.per_layer_config["sliding_attention"], config.per_layer_config[1])

    def test_indexing_homogeneous_config_by_layer_type_returns_global_config(self):
        config = tiny_llama_config(layer_types=["full_attention", "sliding_attention"] * 2)

        self.assertIs(config.per_layer_config["sliding_attention"], config)

    def test_indexing_by_layer_type_ignores_other_layer_types(self):
        """Layers of a different type may differ, only the requested type has to be homogeneous."""
        config = tiny_llama_config(
            per_layer_config={0: {"intermediate_size": 32}, 2: {"intermediate_size": 256}},
            layer_types=["full_attention", "sliding_attention"] * 2,
        )

        self.assertEqual(config.per_layer_config["sliding_attention"].intermediate_size, 128)

    def test_indexing_by_heterogeneous_layer_type_raises(self):
        config = tiny_llama_config(
            per_layer_config={0: {"intermediate_size": 32}},
            layer_types=["full_attention", "sliding_attention"] * 2,
        )

        with self.assertRaisesRegex(ValueError, "'full_attention' is not homogeneous across layers"):
            config.per_layer_config["full_attention"]

    def test_indexing_by_unknown_layer_type_raises(self):
        config = tiny_llama_config(layer_types=["full_attention"] * 4)

        with self.assertRaisesRegex(ValueError, "'sliding_attention' not found in config.layer_types"):
            config.per_layer_config["sliding_attention"]

    def test_indexing_by_layer_type_without_layer_types_raises(self):
        config = tiny_llama_config(per_layer_config={0: {"intermediate_size": 32}})

        with self.assertRaisesRegex(ValueError, "config.layer_types is not defined"):
            config.per_layer_config["full_attention"]

    def test_layer_configs_are_created_once_and_follow_global_changes(self):
        config = tiny_llama_config(per_layer_config={0: {"intermediate_size": 64}})
        layer_configs = list(config.per_layer_config)
        layer_configs[3].rope_parameters = {"rope_type": "default", "rope_theta": 1234.0}

        config.hidden_size = 96
        config.intermediate_size = 192
        # `rope_scaling` is a property that sets `rope_parameters` under the hood
        config.rope_scaling = {"rope_type": "default", "rope_theta": 9999.0}
        # An attribute the config didn't have when the layer configs were created
        config.attribute_added_later = "value"

        for layer_idx, layer_config in enumerate(layer_configs):
            self.assertIs(config.per_layer_config[layer_idx], layer_config)
        self.assertEqual([c.hidden_size for c in layer_configs], [96, 96, 96, 96])
        self.assertEqual([c.attribute_added_later for c in layer_configs], ["value"] * 4)
        # Layer 0 overrides `intermediate_size`, so it keeps its own value
        self.assertEqual([c.intermediate_size for c in layer_configs], [64, 192, 192, 192])
        # Layer 3's `rope_parameters` were changed directly, so it keeps them
        self.assertEqual([c.rope_parameters["rope_theta"] for c in layer_configs], [9999.0, 9999.0, 9999.0, 1234.0])

        # Reassigning `per_layer_config` creates new layer configs
        config.per_layer_config = {1: {"intermediate_size": 32}}
        self.assertIsNot(config.per_layer_config[1], layer_configs[1])
        self.assertEqual(config.per_layer_config[1].intermediate_size, 32)

    def test_deleting_a_global_attribute_passes_on_to_layer_configs(self):
        config = tiny_llama_config(attention_dropout=0.5, per_layer_config={1: {"attention_dropout": 0.0}})
        layer_configs = list(config.per_layer_config)
        # Layer 2's value was changed directly, so it keeps it
        layer_configs[2].attention_dropout = 0.3

        del config.attention_dropout

        # `attention_dropout` falls back to its default, 0.0, which equals layer 1's override, so it's removed
        self.assertEqual(config.to_dict()["per_layer_config"], {})
        self.assertEqual([c.attention_dropout for c in layer_configs], [0.0, 0.0, 0.3, 0.0])

    def test_deleting_a_global_attribute_without_a_default_removes_it_from_layer_configs(self):
        config = tiny_llama_config(per_layer_config={0: {"intermediate_size": 64}})
        layer_configs = list(config.per_layer_config)
        # Like `merge_with_config_defaults`, which sets `is_causal` for one forward call and then deletes it
        config.is_causal = False

        del config.is_causal

        self.assertEqual([hasattr(c, "is_causal") for c in layer_configs], [False, False, False, False])

    @parameterized.expand(
        [
            ("sparse", False),
            ("explicit", True),
        ]
    )
    def test_override_that_comes_to_equal_the_global_value_behaves_as_after_a_round_trip(
        self, _name, serialize_explicit_per_layer_config
    ):
        config = tiny_llama_config(per_layer_config={1: {"attention_dropout": 0.1}})
        config.serialize_explicit_per_layer_config = serialize_explicit_per_layer_config
        layer_configs = list(config.per_layer_config)

        # Layer 1's override now equals the global value, so it's removed
        config.attention_dropout = 0.1
        loaded_config = LlamaConfig.from_dict(config.to_dict())

        self.assertEqual(config.per_layer_attributes, loaded_config.per_layer_attributes)
        config.attention_dropout = 0.2
        loaded_config.attention_dropout = 0.2
        self.assertEqual([c.attention_dropout for c in layer_configs], [0.2, 0.2, 0.2, 0.2])
        self.assertEqual([c.attention_dropout for c in loaded_config.per_layer_config], [0.2, 0.2, 0.2, 0.2])

    def test_accessing_per_layer_attr_raises(self):
        config = tiny_llama_config(per_layer_config={0: {"num_key_value_heads": 2}, 1: {"num_key_value_heads": 1}})
        with self.assertRaisesRegex(
            AmbiguousGlobalPerLayerAttributeError, "allow_global_per_layer_attribute_access.*global value incorrectly"
        ):
            _ = config.num_key_value_heads

        # The attribute exists, so the error isn't treated as a missing attribute
        self.assertIn("num_key_value_heads", config.__dict__)
        with self.assertRaises(AmbiguousGlobalPerLayerAttributeError):
            getattr(config, "num_key_value_heads", "default")
        with self.assertRaises(AmbiguousGlobalPerLayerAttributeError):
            hasattr(config, "num_key_value_heads")

    def test_allow_global_per_layer_attribute_access(self):
        config = tiny_llama_config(
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
            tiny_llama_config().save_pretrained(tmpdir)
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
        self.assertNotIn("allow_global_per_layer_attribute_access", tiny_llama_config().to_dict())

    def test_non_per_layer_attributes_do_not_warn(self):
        config = tiny_llama_config(per_layer_config={0: {"num_key_value_heads": 2}, 1: {"num_key_value_heads": 1}})
        logger = transformers_logging.get_logger("transformers.integrations.heterogeneity.configuration_utils")
        logger.warning_once.cache_clear()

        with self.assertNoLogs(logger=logger, level="WARNING"):
            self.assertEqual(config.hidden_size, 64)

    def test_iter_skips_per_layer_attributes_by_default(self):
        config = tiny_llama_config(per_layer_config={0: {"num_key_value_heads": 2}, 1: {"num_key_value_heads": 1}})

        keys = list(config)

        self.assertNotIn("num_key_value_heads", keys)
        self.assertIn("hidden_size", keys)

    def test_iter_includes_per_layer_attributes_when_global_access_allowed(self):
        config = tiny_llama_config(
            per_layer_config={0: {"num_key_value_heads": 2}, 1: {"num_key_value_heads": 1}},
            allow_global_per_layer_attribute_access=True,
        )

        self.assertIn("num_key_value_heads", list(config))

    def test_validation_missing_global_attr(self):
        # "fake_attr" in layer 0 but not in layer 1, and not global → should fail
        with self.assertRaises(ValueError):
            tiny_llama_config(
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
            tiny_llama_config(per_layer_config={layer_idx: {"num_key_value_heads": 2}})

    def test_save_pretrained_config_round_trip(self):
        """Config should survive save_pretrained → from_pretrained on disk."""
        per_layer = {i: {"intermediate_size": 64 + i} for i in range(0, 12, 2)}
        config = tiny_llama_config(per_layer_config=per_layer, num_hidden_layers=12)

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
            ),
            ("global_sw_per_layer_acs", {"sliding_window": 4096}, {0: {"attention_chunk_size": 2048}}),
            (
                "per_layer_sw_per_layer_acs_same_layer",
                {},
                {0: {"sliding_window": 4096, "attention_chunk_size": 2048}},
            ),
        ],
    )
    def test_validation_sliding_window_and_attention_chunk_size_conflict_raises(
        self, _name, overrides, per_layer_config
    ):
        with self.assertRaises(ValueError):
            tiny_llama_config(per_layer_config=per_layer_config, **overrides)

    @parameterized.expand(
        [
            (
                "per_layer_sw_per_layer_acs_different_layers",
                {"sliding_window": None, "attention_chunk_size": None},
                {0: {"sliding_window": 4096}, 1: {"attention_chunk_size": 2048}},
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
            ),
        ],
    )
    def test_validation_sliding_window_and_attention_chunk_size_accepts_non_conflicting_layers(
        self, _name, overrides, per_layer_config
    ):
        config = tiny_llama_config(per_layer_config=per_layer_config, **overrides)

        self.assertTrue(config.is_heterogeneous)

    def test_all_layers_overridden_no_global_default(self):
        """Custom attribute on every layer without a global default should be accessible per layer."""
        config = tiny_llama_config(
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

    def test_explicit_serialization_lists_every_layer(self):
        # Layer 1's overrides equal the global values, so they're removed, but explicit serialization still lists
        # their attributes for every layer
        per_layer = {0: {"num_key_value_heads": 1}, 1: {"num_key_value_heads": 4, "intermediate_size": 128}}
        sparse_config = tiny_llama_config(per_layer_config=per_layer)
        explicit_config = tiny_llama_config(per_layer_config=per_layer, serialize_explicit_per_layer_config=True)
        # Layers without an override list the current global value
        explicit_config.num_key_value_heads = 8

        self.assertEqual(sparse_config.to_dict()["per_layer_config"], {"0": {"num_key_value_heads": 1}})
        self.assertEqual(
            explicit_config.to_dict()["per_layer_config"],
            {
                "0": {"num_key_value_heads": 1, "intermediate_size": 128},
                "1": {"num_key_value_heads": 8, "intermediate_size": 128},
                "2": {"num_key_value_heads": 8, "intermediate_size": 128},
                "3": {"num_key_value_heads": 8, "intermediate_size": 128},
            },
        )


class TestMtpHeterogeneousConfig(unittest.TestCase):
    def test_get_mtp_config_drops_main_model_per_layer_config_by_default(self):
        config = tiny_llama_config(per_layer_config={3: {"intermediate_size": 64}})
        config.num_mtp_layers = 2

        mtp_config = config.get_mtp_config()

        self.assertEqual(mtp_config.num_hidden_layers, 2)
        self.assertFalse(mtp_config.is_heterogeneous)
        self.assertEqual(mtp_config.intermediate_size, object.__getattribute__(config, "intermediate_size"))
        self.assertTrue(config.is_heterogeneous)

    def test_get_mtp_config_uses_independent_mtp_per_layer_config(self):
        config = tiny_llama_config(per_layer_config={3: {"intermediate_size": 64}})
        config.num_mtp_layers = 2
        config.mtp_per_layer_config = {
            0: {"intermediate_size": 80},
            1: {"num_key_value_heads": 2},
        }

        mtp_config = config.get_mtp_config()

        self.assertTrue(mtp_config.is_heterogeneous)
        self.assertEqual(mtp_config.num_hidden_layers, 2)
        self.assertEqual(mtp_config.per_layer_config[0].intermediate_size, 80)
        self.assertEqual(
            mtp_config.per_layer_config[1].intermediate_size,
            object.__getattribute__(config, "intermediate_size"),
        )
        self.assertEqual(mtp_config.per_layer_config[1].num_key_value_heads, 2)

    def test_get_mtp_config_validates_overrides_against_mtp_layer_count(self):
        config = tiny_llama_config()
        config.num_mtp_layers = 2
        config.mtp_per_layer_config = {2: {"intermediate_size": 80}}

        with self.assertRaisesRegex(ValueError, r"range \[0, 2\)"):
            config.get_mtp_config()
