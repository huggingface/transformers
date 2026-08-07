# Copyright 2026 Mistral AI and The HuggingFace Inc. team. All rights reserved.
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

from transformers import Mistral3Config, Mistral4Config, MistralConfig
from transformers.configuration_utils import PreTrainedConfig
from transformers.integrations.mistral.native_config import mistral_native_config_from_params
from transformers.integrations.mistral.params_conversion import (
    _MISTRAL_EXTRAS_KEY,
    mistral_native_config_from_hf_config,
    mistral_native_config_to_hf_config,
)
from transformers.models.pixtral.configuration_pixtral import PixtralVisionConfig
from transformers.utils.quantization_config import QuantizationConfigMixin

from .mistral_fixture_data import (
    RAW_MISTRAL_PARAMS,
    base_native_config,
    make_non_reversible_quant_config,
    mistral3_native_config,
    mistral4_native_config,
    perturbed_ministral3_native_config,
    perturbed_mistral3_moe_native_config,
    perturbed_mistral3_native_config,
    perturbed_mistral4_native_config,
    perturbed_mistral_native_config,
)


class TestPublicAPI(unittest.TestCase):
    """Coverage for the module's public exports."""

    def test_public_roundtrip(self) -> None:
        native = base_native_config()
        restored = mistral_native_config_from_hf_config(mistral_native_config_to_hf_config(native))
        self.assertEqual(restored, native)

    def test_mistral_native_config_from_hf_config_unsupported_type_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported HF config type"):
            mistral_native_config_from_hf_config(PreTrainedConfig())


class TestExhaustiveRoundtrip(unittest.TestCase):
    """Every native field the reverse converter reads back must survive a native -> HF -> native
    round-trip unchanged.

    Seven forward outputs are write-only, and this class cannot prove anything about them: the
    reverse converter hardcodes `mm_projector_id` and `add_pre_mm_projector_layer_norm` rather
    than reading them back, and never reads back `norm_topk_prob`, `vision_feature_layer`,
    `hidden_act`, `mscale`, or `partial_rotary_factor`. For those fields, a round-trip here
    compares the reverse converter's own constants to the forward converter's inputs, which
    cannot fail regardless of whether the forward mapping is correct. See
    `TestReverseFromPublishedConfig` in test_published_config_parity.py for the oracle that
    covers the reverse direction from data this class cannot touch.
    """

    def test_raw_quantization_config_dict_survives_round_trip(self) -> None:
        """A `quantization_config` shipped as a raw dict in `params.json` must hold the same
        runtime type (`QuantizationConfigMixin`) as one produced by a native -> HF -> native
        round-trip, or the round-trip equality check silently lies about what was preserved."""
        quant_config = {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "modules_to_not_convert": ["lm_head"],
            "weight_block_size": None,
        }
        raw = {**RAW_MISTRAL_PARAMS, "quantization_config": quant_config}

        native = mistral_native_config_from_params(raw)
        restored = mistral_native_config_from_hf_config(mistral_native_config_to_hf_config(native))

        self.assertIsInstance(native.quantization_config, QuantizationConfigMixin)
        self.assertEqual(restored, native)

    def test_quantization_config_passthrough_roundtrip(self) -> None:
        """A non-fp8 HF quantization config has no native equivalent, so it rides through
        both directions untouched in `quantization_config`."""
        for test_id, factory in [("mistral", base_native_config), ("mistral3", mistral3_native_config)]:
            with self.subTest(test_id):
                native = factory()
                native.quantization = None
                native.quantization_config = make_non_reversible_quant_config()

                restored = mistral_native_config_from_hf_config(mistral_native_config_to_hf_config(native))

                self.assertIsNone(restored.quantization)
                self.assertIsNotNone(restored.quantization_config)
                self.assertEqual(restored.quantization_config.to_dict()["quant_method"], "gptq")
                self.assertEqual(restored, native)

    def test_roundtrip_is_lossless(self) -> None:
        factories = [
            ("mistral", perturbed_mistral_native_config),
            ("ministral3", perturbed_ministral3_native_config),
            ("mistral4", perturbed_mistral4_native_config),
            ("mistral3", perturbed_mistral3_native_config),
            ("mistral3_moe_text_config", perturbed_mistral3_moe_native_config),
        ]
        for test_id, factory in factories:
            with self.subTest(test_id):
                native = factory()
                restored = mistral_native_config_from_hf_config(mistral_native_config_to_hf_config(native))
                self.assertEqual(restored, native)


class TestMistralExtras(unittest.TestCase):
    """Behaviour of the `mistral_extras` config entry that carries the native fields
    with no HF equivalent."""

    def test_extras_survive_save_and_load(self) -> None:
        native = perturbed_mistral4_native_config()
        hf = mistral_native_config_to_hf_config(native)
        extras_before = getattr(hf, _MISTRAL_EXTRAS_KEY)

        with tempfile.TemporaryDirectory() as tmp_dir:
            hf.save_pretrained(tmp_dir)
            reloaded = Mistral4Config.from_pretrained(tmp_dir)

        extras_after = getattr(reloaded, _MISTRAL_EXTRAS_KEY, None)
        self.assertEqual(extras_after, extras_before)

        restored = mistral_native_config_from_hf_config(reloaded)
        self.assertEqual(restored, native)

    def test_extras_absent_when_nothing_to_preserve(self) -> None:
        native = base_native_config()
        hf = mistral_native_config_to_hf_config(native)
        self.assertFalse(hasattr(hf, _MISTRAL_EXTRAS_KEY))

        with tempfile.TemporaryDirectory() as tmp_dir:
            hf.save_pretrained(tmp_dir)
            reloaded = MistralConfig.from_pretrained(tmp_dir)
        self.assertFalse(hasattr(reloaded, _MISTRAL_EXTRAS_KEY))

    def test_vlm_with_moe_text_config_extras_survive_save_and_load(self) -> None:
        """For a VLM with a MoE text config, the combined extras land once on the outer
        config and never on the nested text config."""
        native = perturbed_mistral3_moe_native_config()
        hf = mistral_native_config_to_hf_config(native)
        self.assertTrue(hasattr(hf, _MISTRAL_EXTRAS_KEY))
        self.assertFalse(hasattr(hf.text_config, _MISTRAL_EXTRAS_KEY))

        restored_in_memory = mistral_native_config_from_hf_config(hf)
        self.assertEqual(restored_in_memory, native)

        with tempfile.TemporaryDirectory() as tmp_dir:
            hf.save_pretrained(tmp_dir)
            reloaded = Mistral3Config.from_pretrained(tmp_dir)

        self.assertEqual(getattr(reloaded, _MISTRAL_EXTRAS_KEY), getattr(hf, _MISTRAL_EXTRAS_KEY))
        restored_from_disk = mistral_native_config_from_hf_config(reloaded)
        self.assertEqual(restored_from_disk, native)

    def test_outer_without_extras_preserves_nested_moe_residual(self) -> None:
        """An outer config only overrides the MoE fields its own extras carry, so values
        recovered from a nested text config's extras are not clobbered."""
        text_native = mistral4_native_config()
        text_native.moe.expert_parallel = 8
        text_native.moe.route_every_n = 3
        text_hf = mistral_native_config_to_hf_config(text_native)
        self.assertTrue(hasattr(text_hf, _MISTRAL_EXTRAS_KEY))

        vision_hf = mistral_native_config_to_hf_config(mistral3_native_config()).vision_config

        outer = Mistral3Config(
            text_config=text_hf,
            vision_config=vision_hf,
            image_token_id=10,
            spatial_merge_size=2,
        )
        self.assertFalse(hasattr(outer, _MISTRAL_EXTRAS_KEY))

        restored = mistral_native_config_from_hf_config(outer)
        self.assertEqual(restored.moe.expert_parallel, 8)
        self.assertEqual(restored.moe.route_every_n, 3)

    def test_mistral4_defaults_when_extras_absent(self) -> None:
        """An HF config without `mistral_extras` still reverses, using default values."""
        hf = Mistral4Config(
            hidden_size=4096,
            num_hidden_layers=32,
            intermediate_size=14336,
            num_attention_heads=32,
            num_key_value_heads=32,
            rms_norm_eps=1e-5,
            vocab_size=32000,
            max_position_embeddings=1048576,
            q_lora_rank=1024,
            qk_rope_head_dim=64,
            qk_nope_head_dim=64,
            kv_lora_rank=256,
            v_head_dim=128,
            n_routed_experts=128,
            num_experts_per_tok=4,
            moe_intermediate_size=2048,
            first_k_dense_replace=0,
            n_shared_experts=1,
            routed_scaling_factor=1.0,
            n_group=1,
            topk_group=1,
        )
        self.assertFalse(hasattr(hf, _MISTRAL_EXTRAS_KEY))
        native = mistral_native_config_from_hf_config(hf)
        self.assertEqual(native.moe.expert_parallel, 1)
        self.assertEqual(native.moe.expert_model_parallel, 1)
        self.assertEqual(native.moe.route_every_n, 1)

    def test_mistral3_defaults_when_extras_absent(self) -> None:
        text_config = MistralConfig(
            hidden_size=4096,
            num_hidden_layers=32,
            intermediate_size=14336,
            num_attention_heads=32,
            num_key_value_heads=8,
            rms_norm_eps=1e-5,
            head_dim=128,
            vocab_size=32000,
            max_position_embeddings=131072,
        )
        vision_config = PixtralVisionConfig(
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            patch_size=14,
            image_size=1540,
            intermediate_size=4096,
            hidden_act="silu",
            rope_theta=10000.0,
        )
        hf = Mistral3Config(
            text_config=text_config,
            vision_config=vision_config,
            multimodal_projector_bias=False,
            image_token_id=10,
            spatial_merge_size=2,
            vision_feature_layer=-1,
        )
        self.assertFalse(hasattr(hf, _MISTRAL_EXTRAS_KEY))
        native = mistral_native_config_from_hf_config(hf)
        self.assertEqual(native.vision_encoder.max_image_size, vision_config.image_size)
        self.assertEqual(native.vision_encoder.image_break_token_id, 12)
        self.assertEqual(native.vision_encoder.image_end_token_id, 13)


if __name__ == "__main__":
    unittest.main()
