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

import unittest
from types import SimpleNamespace

from transformers import Ministral3Config, Mistral3Config, Mistral4Config, MistralConfig
from transformers.integrations.mistral.native_config import (
    Llama4Scaling,
    MistralNativeConfig,
    QFormat,
    QuantizationArgs,
    YarnArgs,
)
from transformers.integrations.mistral.params_conversion import (
    _extract_rope_theta,
    _hf_mistral_to_native,
    _hf_quant_config_to_native,
    mistral_native_config_from_hf_config,
    mistral_native_config_to_hf_config,
)
from transformers.models.pixtral.configuration_pixtral import PixtralVisionConfig

from .mistral_fixture_data import (
    base_native_config,
    llama4_scaling,
    make_hf_fp8_quant_config,
    make_non_reversible_quant_config,
    ministral3_native_config,
    mistral3_native_config,
    mistral4_native_config,
    moe_args,
    yarn_args,
)


class TestHFToNativeMistral(unittest.TestCase):
    def test_unknown_fp8_activation_scheme_returns_none(self) -> None:
        """An unrecognised FP8 activation_scheme yields None rather than mapping to TENSOR."""
        self.assertIsNone(_hf_quant_config_to_native(make_hf_fp8_quant_config("dynamic")))

    def test_basic_reverse(self) -> None:
        config = base_native_config()
        hf = MistralConfig(
            hidden_size=4096,
            num_hidden_layers=32,
            intermediate_size=14336,
            num_attention_heads=32,
            num_key_value_heads=8,
            rms_norm_eps=1e-5,
            head_dim=128,
            vocab_size=32000,
            max_position_embeddings=32768,
            sliding_window=None,
        )
        native = mistral_native_config_from_hf_config(hf)
        self.assertEqual(native, config)

    def test_roundtrip(self) -> None:
        config = base_native_config()
        restored = mistral_native_config_from_hf_config(mistral_native_config_to_hf_config(config))
        self.assertEqual(restored, config)

    def test_reverse_with_quantization_config(self) -> None:
        hf = MistralConfig(
            hidden_size=4096,
            num_hidden_layers=32,
            intermediate_size=14336,
            num_attention_heads=32,
            num_key_value_heads=8,
            rms_norm_eps=1e-5,
            head_dim=128,
            vocab_size=32000,
            max_position_embeddings=32768,
            sliding_window=None,
            quantization_config=make_hf_fp8_quant_config(),
        )
        native = mistral_native_config_from_hf_config(hf)
        self.assertIsNotNone(native.quantization)
        self.assertIsNone(native.quantization_config)
        self.assertEqual(native.quantization, QuantizationArgs(QFormat.FP8_E4M3, "TENSOR"))
        expected = MistralNativeConfig(
            dim=4096,
            n_layers=32,
            head_dim=128,
            hidden_dim=14336,
            n_heads=32,
            n_kv_heads=8,
            rope_theta=10000.0,
            norm_eps=1e-5,
            vocab_size=32000,
            max_position_embeddings=32768,
            quantization=native.quantization,
        )
        self.assertEqual(native, expected)

    def test_reverse_with_non_reversible_quant(self) -> None:
        hf_quant = make_non_reversible_quant_config()
        hf = MistralConfig(
            hidden_size=4096,
            num_hidden_layers=32,
            intermediate_size=14336,
            num_attention_heads=32,
            num_key_value_heads=8,
            rms_norm_eps=1e-5,
            head_dim=128,
            vocab_size=32000,
            max_position_embeddings=32768,
            quantization_config=hf_quant,
        )
        native = mistral_native_config_from_hf_config(hf)
        self.assertIsNone(native.quantization)
        self.assertIsNotNone(native.quantization_config)
        self.assertEqual(native.quantization_config.to_dict()["quant_method"], "gptq")


class TestHFToNativeMinistral3(unittest.TestCase):
    def test_basic_reverse(self) -> None:
        config = ministral3_native_config()
        hf = Ministral3Config(
            hidden_size=4096,
            num_hidden_layers=32,
            intermediate_size=14336,
            num_attention_heads=32,
            num_key_value_heads=8,
            rms_norm_eps=1e-5,
            head_dim=128,
            vocab_size=32000,
            max_position_embeddings=262144,
            tie_word_embeddings=True,
            rope_parameters={
                "type": "yarn",
                "rope_theta": 1000000.0,
                "factor": 16.0,
                "original_max_position_embeddings": 16384,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "mscale_all_dim": 1.0,
                "mscale": 1.0,
                "llama_4_scaling_beta": 0.1,
            },
        )
        native = mistral_native_config_from_hf_config(hf)
        self.assertEqual(native, config)

    def test_roundtrip(self) -> None:
        """mscale is always 1.0; mscale_all_dim is 0.0 when apply_scale is True, else 1.0."""
        for apply_scale, expected_mscale_all_dim in [(False, 1.0), (True, 0.0)]:
            with self.subTest(apply_scale=apply_scale):
                ya = YarnArgs(
                    factor=16.0, original_max_position_embeddings=16384, beta=32, alpha=1, apply_scale=apply_scale
                )
                l4s = Llama4Scaling(original_max_position_embeddings=16384, beta=0.1)
                native = MistralNativeConfig(
                    dim=4096,
                    n_layers=32,
                    head_dim=128,
                    hidden_dim=14336,
                    n_heads=32,
                    n_kv_heads=8,
                    rope_theta=1000000.0,
                    norm_eps=1e-5,
                    vocab_size=32000,
                    max_position_embeddings=262144,
                    tied_embeddings=True,
                    yarn=ya,
                    llama_4_scaling=l4s,
                )
                hf = mistral_native_config_to_hf_config(native)
                rope = hf.rope_parameters
                assert "mscale" in rope, f"mscale missing from rope_parameters when {apply_scale=}"
                assert rope["mscale"] == 1.0, f"Expected mscale=1.0 for {apply_scale=}, got {rope['mscale']}"
                assert rope["mscale_all_dim"] == expected_mscale_all_dim, (
                    f"Expected mscale_all_dim={expected_mscale_all_dim} for {apply_scale=}, "
                    f"got {rope['mscale_all_dim']}"
                )
                restored = mistral_native_config_from_hf_config(hf)
                self.assertEqual(restored, native)

    def test_llama_4_scaling_defaults_to_zero_and_roundtrips_to_none(self) -> None:
        """A yarn config with no `llama_4_scaling` emits beta 0 and round-trips back to None."""
        native = MistralNativeConfig(
            dim=4096,
            n_layers=32,
            head_dim=128,
            hidden_dim=14336,
            n_heads=32,
            n_kv_heads=8,
            rope_theta=1000000.0,
            norm_eps=1e-5,
            vocab_size=32000,
            max_position_embeddings=262144,
            tied_embeddings=True,
            yarn=yarn_args(),
        )
        hf = mistral_native_config_to_hf_config(native)
        self.assertEqual(hf.rope_parameters["llama_4_scaling_beta"], 0)
        restored = mistral_native_config_from_hf_config(hf)
        self.assertIsNone(restored.llama_4_scaling)
        self.assertEqual(restored, native)

    def test_roundtrip_with_quantization(self) -> None:
        ya = yarn_args()
        l4s = llama4_scaling()
        native = MistralNativeConfig(
            dim=4096,
            n_layers=32,
            head_dim=128,
            hidden_dim=14336,
            n_heads=32,
            n_kv_heads=8,
            rope_theta=1000000.0,
            norm_eps=1e-5,
            vocab_size=32000,
            max_position_embeddings=262144,
            tied_embeddings=True,
            yarn=ya,
            llama_4_scaling=l4s,
            quantization=QuantizationArgs(QFormat.FP8_E4M3, "TENSOR"),
        )
        hf = mistral_native_config_to_hf_config(native)
        restored = mistral_native_config_from_hf_config(hf)
        self.assertIsNotNone(restored.quantization)
        self.assertIsNone(restored.quantization_config)
        self.assertEqual(restored.quantization, QuantizationArgs(QFormat.FP8_E4M3, "TENSOR"))
        self.assertEqual(restored, native)

    def test_reverse_with_non_reversible_quant(self) -> None:
        hf_quant = make_non_reversible_quant_config()
        hf = Ministral3Config(
            hidden_size=4096,
            num_hidden_layers=32,
            intermediate_size=14336,
            num_attention_heads=32,
            num_key_value_heads=8,
            rms_norm_eps=1e-5,
            head_dim=128,
            vocab_size=32000,
            max_position_embeddings=262144,
            tie_word_embeddings=True,
            rope_parameters={
                "type": "yarn",
                "rope_theta": 1000000.0,
                "factor": 16.0,
                "original_max_position_embeddings": 16384,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "mscale_all_dim": 1.0,
                "llama_4_scaling_beta": 0.1,
            },
            quantization_config=hf_quant,
        )
        native = mistral_native_config_from_hf_config(hf)
        self.assertIsNone(native.quantization)
        self.assertIsNotNone(native.quantization_config)
        self.assertEqual(native.quantization_config.to_dict()["quant_method"], "gptq")


class TestHFToNativeMistral3(unittest.TestCase):
    def test_basic_reverse(self) -> None:
        config = mistral3_native_config()
        text_config = MistralConfig(
            hidden_size=4096,
            num_hidden_layers=32,
            intermediate_size=14336,
            num_attention_heads=32,
            num_key_value_heads=8,
            rms_norm_eps=1e-5,
            head_dim=128,
            vocab_size=32000,
            rope_theta=1000000000.0,
            sliding_window=None,
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
        native = mistral_native_config_from_hf_config(hf)
        self.assertEqual(native, config)

    def test_reverse_with_non_reversible_quant(self) -> None:
        hf_quant = make_non_reversible_quant_config()
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
            quantization_config=hf_quant,
        )
        native = mistral_native_config_from_hf_config(hf)
        self.assertIsNone(native.quantization)
        self.assertIsNotNone(native.quantization_config)
        self.assertEqual(native.quantization_config.to_dict()["quant_method"], "gptq")


class TestHFToNativeMistral4(unittest.TestCase):
    def test_basic_reverse(self) -> None:
        config = mistral4_native_config()
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
        native = mistral_native_config_from_hf_config(hf)
        self.assertEqual(native, config)

    def test_roundtrip(self) -> None:
        config = mistral4_native_config()
        restored = mistral_native_config_from_hf_config(mistral_native_config_to_hf_config(config))
        self.assertEqual(restored, config)

    def test_roundtrip_with_quantization(self) -> None:
        _moe_args = moe_args()
        native = MistralNativeConfig(
            dim=4096,
            n_layers=32,
            head_dim=128,
            hidden_dim=14336,
            n_heads=32,
            n_kv_heads=32,
            rope_theta=10000.0,
            norm_eps=1e-5,
            vocab_size=32000,
            max_position_embeddings=1048576,
            q_lora_rank=1024,
            qk_rope_head_dim=64,
            qk_nope_head_dim=64,
            kv_lora_rank=256,
            v_head_dim=128,
            yarn=YarnArgs(
                factor=128.0, original_max_position_embeddings=8192, beta=32.0, alpha=1.0, apply_scale=False
            ),
            llama_4_scaling=Llama4Scaling(original_max_position_embeddings=8192, beta=0.1),
            moe=_moe_args,
            quantization=QuantizationArgs(QFormat.FP8_E4M3, "TENSOR"),
        )
        hf = mistral_native_config_to_hf_config(native)
        restored = mistral_native_config_from_hf_config(hf)
        self.assertIsNotNone(restored.quantization)
        self.assertIsNone(restored.quantization_config)
        self.assertEqual(restored.quantization, QuantizationArgs(QFormat.FP8_E4M3, "TENSOR"))
        self.assertEqual(restored, native)

    def test_reverse_with_non_reversible_quant(self) -> None:
        hf_quant = make_non_reversible_quant_config()
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
            quantization_config=hf_quant,
        )
        native = mistral_native_config_from_hf_config(hf)
        self.assertIsNone(native.quantization)
        self.assertIsNotNone(native.quantization_config)
        self.assertEqual(native.quantization_config.to_dict()["quant_method"], "gptq")


class TestErrorPaths(unittest.TestCase):
    def test_hf_mistral_missing_head_dim_raises(self) -> None:
        hf = MistralConfig(
            hidden_size=4096,
            num_hidden_layers=32,
            intermediate_size=14336,
            num_attention_heads=32,
            num_key_value_heads=8,
            rms_norm_eps=1e-5,
            vocab_size=32000,
            max_position_embeddings=32768,
        )
        # MistralConfig auto-computes head_dim; force it to None to test the guard
        hf.head_dim = None
        with self.assertRaisesRegex(ValueError, "head_dim must be set"):
            _hf_mistral_to_native(hf)

    def test_extract_rope_theta_missing_raises(self) -> None:
        config = SimpleNamespace(rope_scaling=None)
        with self.assertRaisesRegex(ValueError, "rope_theta"):
            _extract_rope_theta(config)

    def test_unrecognized_quantization_config_type_raises(self) -> None:
        """A `quantization_config` neither `None`, a `dict`, nor a `QuantizationConfigMixin`
        must be reported loudly rather than silently treated as absent."""
        hf = MistralConfig(
            hidden_size=4096,
            num_hidden_layers=32,
            intermediate_size=14336,
            num_attention_heads=32,
            num_key_value_heads=8,
            rms_norm_eps=1e-5,
            head_dim=128,
            vocab_size=32000,
            max_position_embeddings=32768,
        )
        hf.quantization_config = 42
        with self.assertRaisesRegex(TypeError, "Unsupported `quantization_config` type"):
            mistral_native_config_from_hf_config(hf)


if __name__ == "__main__":
    unittest.main()
