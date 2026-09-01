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

from parameterized import parameterized

from transformers import Ministral3Config, Mistral3Config, Mistral4Config, MistralConfig
from transformers.integrations.mistral.native_config import (
    Llama4Scaling,
    MistralNativeConfig,
    QFormat,
    QuantizationArgs,
    YarnArgs,
    mistral_native_config_from_params,
)
from transformers.integrations.mistral.params_conversion import (
    _get_maybe_quant_config,
    _get_rope_parameters,
    mistral_native_config_to_hf_config,
)
from transformers.quantizers.auto import AutoQuantizationConfig
from transformers.testing_utils import require_compressed_tensors
from transformers.utils.quantization_config import QuantizationConfigMixin

from .mistral_fixture_data import (
    RAW_MISTRAL_PARAMS,
    base_native_config,
    expected_ministral3_hf_config,
    expected_mistral3_hf_config,
    expected_mistral4_hf_config,
    expected_mistral_hf_config,
    make_hf_fp8_quant_config,
    ministral3_native_config,
    mistral3_native_config,
    mistral4_asymmetric_mla_native_config,
    mistral4_native_config,
    moe_args,
    perturbed_mistral3_native_config,
    vision_encoder_args,
)


# Shapes of real published `params.json` files. Most of these are invented shapes (matching the
# real file's schema, not its values); `_RAW_PARAMS_MISTRAL_7B_INSTRUCT_V0_3` below is the
# exception, and is real ground truth.

# Mirrors mistralai/Mistral-7B-Instruct-v0.3 verbatim: `params.json`.
_RAW_PARAMS_MISTRAL_7B_INSTRUCT_V0_3 = {
    "dim": 4096,
    "n_layers": 32,
    "head_dim": 128,
    "hidden_dim": 14336,
    "n_heads": 32,
    "n_kv_heads": 8,
    "norm_eps": 1e-05,
    "vocab_size": 32768,
    "rope_theta": 1000000.0,
}

# Mirrors `params.json` files that serialize `sliding_window` as a numeric string rather than
# an int, as called out in the reference native-weights converter's `convert_config`.
_RAW_PARAMS_STRING_SLIDING_WINDOW = {
    "dim": 4096,
    "n_layers": 32,
    "head_dim": 128,
    "hidden_dim": 14336,
    "n_heads": 32,
    "n_kv_heads": 8,
    "norm_eps": 1e-05,
    "vocab_size": 32000,
    "rope_theta": 10000.0,
    "sliding_window": "4096",
    "max_position_embeddings": 32768,
}

# Mirrors mistralai/Ministral-8B-Instruct-2410: `sliding_window` is an interleaved per-layer
# list (e.g. `[null, 32768, 32768, 32768]`), which cannot be represented by `MistralConfig` yet.
_RAW_PARAMS_MINISTRAL_8B_INTERLEAVED_SLIDING_WINDOW = {
    "dim": 4096,
    "n_layers": 4,
    "head_dim": 128,
    "hidden_dim": 12288,
    "n_heads": 32,
    "n_kv_heads": 8,
    "norm_eps": 1e-05,
    "vocab_size": 131072,
    "rope_theta": 100000000.0,
    "sliding_window": [None, 32768, 32768, 32768],
    "max_position_embeddings": 32768,
}

# Mirrors mistralai/Mistral-Small-4-119B-2603-NVFP4's `params.json` `quantization_config` block
# verbatim. That repo publishes no `config.json` (it cannot join the parity list in
# test_published_config_parity.py), so this is the only ground truth for a full
# compressed-tensors block, with nested `config_groups`/`input_activations`/`weights`, which is
# exactly what `AutoQuantizationConfig.from_dict` round-tripping needs to be stressed against.
_QUANTIZATION_CONFIG_NVFP4 = {
    "quant_method": "compressed-tensors",
    "format": "nvfp4-pack-quantized",
    "quantization_status": "compressed",
    "kv_cache_scheme": None,
    "global_compression_ratio": None,
    "sparsity_config": {},
    "transform_config": {},
    "version": "0.13.0",
    "ignore": [
        "model.embed_tokens",
        "re:patch_merger.*",
        "re:vision_encoder.*",
        "re:vision_language_adapter.*",
        "re:.*kv_a_proj_with_mqa$",
        "re:.*q_a_proj$",
        "re:.*gate$",
        "re:.*self_attn.*",
        "re:.*attention.*",
        "lm_head",
    ],
    "config_groups": {
        "NVFP4A16": {
            "format": "nvfp4-pack-quantized",
            "targets": ["Linear"],
            "output_activations": None,
            "input_activations": {
                "num_bits": 4,
                "type": "float",
                "symmetric": True,
                "strategy": "tensor_group",
                "group_size": 16,
                "dynamic": "local",
                "actorder": None,
                "observer": "static_minmax",
                "observer_kwargs": {},
                "block_structure": None,
            },
            "weights": {
                "num_bits": 4,
                "type": "float",
                "symmetric": True,
                "strategy": "tensor_group",
                "group_size": 16,
                "dynamic": False,
                "actorder": None,
                "observer": "static_minmax",
                "observer_kwargs": {},
                "block_structure": None,
                "scale_dtype": "torch.float8_e4m3fn",
                "zp_dtype": None,
            },
        }
    },
}

# Mirrors mistralai/Devstral-2-123B-Instruct-2512 verbatim: `params.json`. A text-only model,
# despite its published `config.json` listing vision/projector paths in
# `quantization_config.modules_to_not_convert` (`_KNOWN_DEVIATIONS["devstral2"]` in
# test_published_config_parity.py excludes that as a known-wrong artifact of the legacy
# conversion script; see `test_devstral2_text_only_modules_to_not_convert_excludes_vision_paths`
# below for the correct value this converter actually produces).
_RAW_PARAMS_DEVSTRAL_2_123B_INSTRUCT_2512 = {
    "dim": 12288,
    "n_layers": 88,
    "head_dim": 128,
    "hidden_dim": 28672,
    "n_heads": 96,
    "n_kv_heads": 8,
    "rope_theta": 1000000.0,
    "norm_eps": 1e-05,
    "vocab_size": 131072,
    "tied_embeddings": False,
    "max_position_embeddings": 262144,
    "q_lora_rank": None,
    "qk_rope_head_dim": None,
    "qk_nope_head_dim": None,
    "kv_lora_rank": None,
    "v_head_dim": None,
    "quantization": {"qformat_weight": "fp8_e4m3", "qscheme_act": "TENSOR"},
    "yarn": {
        "original_max_position_embeddings": 4096,
        "factor": 64,
        "apply_scale": True,
        "beta": 4,
        "alpha": 1,
    },
}

# Mirrors mistralai/Voxtral-Mini-3B-2507: `params.json` carries a `multimodal` section with a
# `whisper_model_args` audio encoder that this module has no converter for.
_RAW_PARAMS_VOXTRAL_MULTIMODAL = {
    "dim": 3072,
    "n_layers": 26,
    "head_dim": 128,
    "hidden_dim": 8192,
    "n_heads": 24,
    "n_kv_heads": 8,
    "norm_eps": 1e-05,
    "vocab_size": 131072,
    "rope_theta": 100000000.0,
    "sliding_window": None,
    "max_position_embeddings": 131072,
    "multimodal": {
        "whisper_model_args": {
            "encoder_args": {"dim": 1280, "n_layers": 32},
            "downsample_factor": 4,
        }
    },
}


class TestQuantizationArgs(unittest.TestCase):
    def test_valid_tensor_scheme(self) -> None:
        config = QuantizationArgs(QFormat.FP8_E4M3, "TENSOR")
        self.assertEqual(config, QuantizationArgs(QFormat.FP8_E4M3, "TENSOR"))

    def test_qformat_weight_is_coerced_to_qformat(self) -> None:
        """`qformat_weight` is a `QFormat` at runtime regardless of whether the raw string
        (as read from `params.json`) or the enum member was passed in."""
        config = QuantizationArgs("fp8_e4m3", "TENSOR")
        self.assertIs(config.qformat_weight, QFormat.FP8_E4M3)

    def test_invalid_scheme_raises(self) -> None:
        for scheme in ["DYNAMIC", "UNSUPPORTED", "static"]:
            with self.subTest(scheme=scheme):
                with self.assertRaisesRegex(ValueError, scheme):
                    QuantizationArgs(QFormat.FP8_E4M3, scheme)

    def test_unsupported_format_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "fp8_e4m3"):
            QuantizationArgs("int8", "TENSOR")


class TestGetMaybeQuantConfig(unittest.TestCase):
    def test_none_returns_none(self) -> None:
        self.assertIsNone(_get_maybe_quant_config(is_vision_model=False, quantization_args=None))

    def test_tensor_produces_static(self) -> None:
        qc = _get_maybe_quant_config(
            is_vision_model=False,
            quantization_args=QuantizationArgs(QFormat.FP8_E4M3, "TENSOR"),
        )
        qc_dict = qc.to_dict()
        self.assertEqual(qc_dict["quant_method"], "fp8")
        self.assertEqual(qc_dict["activation_scheme"], "static")

    def test_vision_model_adds_modules_to_skip(self) -> None:
        qc = _get_maybe_quant_config(
            is_vision_model=True,
            quantization_args=QuantizationArgs(QFormat.FP8_E4M3, "TENSOR"),
        )
        qc_dict = qc.to_dict()
        self.assertIn("model.vision_tower", qc_dict["modules_to_not_convert"])
        self.assertIn("model.multi_modal_projector", qc_dict["modules_to_not_convert"])
        self.assertIn("lm_head", qc_dict["modules_to_not_convert"])

    def test_unrecognized_qformat_weight_raises(self) -> None:
        """`QuantizationArgs.__post_init__` rejects any `qformat_weight` other than a `QFormat`
        member, so this state can only be reached by mutating an already-constructed instance.
        `_get_maybe_quant_config` must raise rather than silently returning `None` or a
        mislabeled config."""
        quantization_args = QuantizationArgs(QFormat.FP8_E4M3, "TENSOR")
        quantization_args.qformat_weight = "not-a-real-qformat"
        with self.assertRaisesRegex(ValueError, "invalid quantization config"):
            _get_maybe_quant_config(is_vision_model=False, quantization_args=quantization_args)


class TestNativeConfigFromParams(unittest.TestCase):
    """Parsing a raw `params.json` dict into a `MistralNativeConfig`."""

    def test_parses_flat_fields_and_applies_defaults(self) -> None:
        native = mistral_native_config_from_params(RAW_MISTRAL_PARAMS)
        self.assertEqual(native.dim, 4096)
        self.assertEqual(native.n_layers, 32)
        self.assertEqual(native.head_dim, 128)
        self.assertEqual(native.n_kv_heads, 8)
        self.assertEqual(native.sliding_window, 4096)
        # Absent from the raw dict, so the dataclass defaults apply.
        self.assertFalse(native.tied_embeddings)
        self.assertIsNone(native.yarn)
        self.assertIsNone(native.moe)
        self.assertIsNone(native.vision_encoder)
        self.assertIsNone(native.quantization)

    def test_parses_nested_sections(self) -> None:
        raw = {
            **RAW_MISTRAL_PARAMS,
            "tied_embeddings": True,
            "yarn": {
                "factor": 16.0,
                "original_max_position_embeddings": 16384,
                "beta": 32.0,
                "alpha": 1.0,
                "apply_scale": True,
            },
            "llama_4_scaling": {"original_max_position_embeddings": 16384, "beta": 0.1},
            "quantization": {"qformat_weight": "fp8_e4m3", "qscheme_act": "TENSOR"},
        }
        native = mistral_native_config_from_params(raw)
        self.assertTrue(native.tied_embeddings)
        self.assertEqual(native.yarn, YarnArgs(16.0, 16384, 32.0, 1.0, True))
        self.assertEqual(native.llama_4_scaling, Llama4Scaling(16384, 0.1))
        self.assertEqual(native.quantization, QuantizationArgs(QFormat.FP8_E4M3, "TENSOR"))

    def test_parsed_config_converts_to_hf(self) -> None:
        native = mistral_native_config_from_params(RAW_MISTRAL_PARAMS)
        self.assertIsInstance(mistral_native_config_to_hf_config(native), MistralConfig)

    def test_sliding_window_forwards_numeric_value_to_hf_config(self) -> None:
        """A non-null numeric `sliding_window` from `params.json` must reach the HF config
        unchanged. Hardcoding `None` in the forward mapping would only be caught by the
        round-trip loop, which a symmetric mistake in both directions could survive."""
        native = mistral_native_config_from_params(RAW_MISTRAL_PARAMS)
        hf = mistral_native_config_to_hf_config(native)
        self.assertEqual(hf.sliding_window, 4096)

    def test_missing_required_field_raises(self) -> None:
        raw = {k: v for k, v in RAW_MISTRAL_PARAMS.items() if k != "dim"}
        with self.assertRaises(KeyError):
            mistral_native_config_from_params(raw)


class TestNativeConfigFromParamsRealisticShapes(unittest.TestCase):
    """Parsing published `params.json` shapes, each omitting fields `RAW_MISTRAL_PARAMS` supplies."""

    def test_missing_max_position_embeddings_defaults(self) -> None:
        """Mistral-7B-Instruct-v0.3 shape: no `max_position_embeddings`, so the default applies."""
        native = mistral_native_config_from_params(_RAW_PARAMS_MISTRAL_7B_INSTRUCT_V0_3)
        self.assertEqual(native.max_position_embeddings, 32768)
        self.assertIsInstance(mistral_native_config_to_hf_config(native), MistralConfig)

    @parameterized.expand(["head_dim", "n_kv_heads", "rope_theta"])
    def test_missing_non_defaultable_field_raises(self, field: str) -> None:
        """Every published `params.json` supplies these, so omitting one is an error, not a default."""
        raw = {k: v for k, v in RAW_MISTRAL_PARAMS.items() if k != field}
        with self.assertRaises(KeyError):
            mistral_native_config_from_params(raw)

    def test_string_sliding_window_is_coerced_to_int(self) -> None:
        native = mistral_native_config_from_params(_RAW_PARAMS_STRING_SLIDING_WINDOW)
        self.assertEqual(native.sliding_window, 4096)
        self.assertIsInstance(native.sliding_window, int)

    def test_interleaved_sliding_window_raises(self) -> None:
        """Ministral-8B-Instruct-2410 shape: per-layer interleaved `sliding_window` list."""
        with self.assertRaisesRegex(TypeError, "sliding_window"):
            mistral_native_config_from_params(_RAW_PARAMS_MINISTRAL_8B_INTERLEAVED_SLIDING_WINDOW)

    def test_voxtral_multimodal_section_raises(self) -> None:
        """Voxtral-Mini-3B-2507 shape: unsupported `multimodal`/`whisper_model_args` section."""
        with self.assertRaisesRegex(ValueError, "multimodal"):
            mistral_native_config_from_params(_RAW_PARAMS_VOXTRAL_MULTIMODAL)

    @require_compressed_tensors
    def test_ready_made_quantization_config_is_carried_through_verbatim(self) -> None:
        """Mistral-Small-4-119B-2603-NVFP4 shape: `params.json` ships an HF `quantization_config`,
        here the real compressed-tensors block (not installed locally, hence the skip)."""
        raw = {**RAW_MISTRAL_PARAMS, "quantization_config": _QUANTIZATION_CONFIG_NVFP4}
        expected = AutoQuantizationConfig.from_dict(_QUANTIZATION_CONFIG_NVFP4)

        native = mistral_native_config_from_params(raw)

        self.assertIsInstance(native.quantization_config, QuantizationConfigMixin)
        self.assertEqual(native.quantization_config, expected)
        self.assertIsNone(native.quantization)
        self.assertEqual(mistral_native_config_to_hf_config(native).quantization_config, expected)

    def test_devstral2_text_only_modules_to_not_convert_excludes_vision_paths(self) -> None:
        """Devstral-2-123B-Instruct-2512 is text-only, so `is_vision_model=False` must yield
        exactly `["lm_head"]`. Its published `config.json` wrongly carries vision/projector
        paths too (excluded from parity as a known deviation), which otherwise leaves this
        branch with no published ground truth at all."""
        native = mistral_native_config_from_params(_RAW_PARAMS_DEVSTRAL_2_123B_INSTRUCT_2512)
        hf = mistral_native_config_to_hf_config(native)
        self.assertEqual(hf.quantization_config.to_dict()["modules_to_not_convert"], ["lm_head"])

    def test_vision_encoder_missing_required_fields_raises_domain_error(self) -> None:
        """Pixtral-12B-2409 shape: `vision_encoder` missing several required fields must raise
        a `ValueError` naming the section and the missing keys, not a bare `KeyError`."""
        raw = {**RAW_MISTRAL_PARAMS, "vision_encoder": {"hidden_size": 1024, "num_hidden_layers": 24}}
        with self.assertRaisesRegex(ValueError, r"vision_encoder.*max_image_size"):
            mistral_native_config_from_params(raw)

    def test_moe_missing_required_fields_raises_domain_error(self) -> None:
        raw = {**RAW_MISTRAL_PARAMS, "moe": {"num_experts": 8}}
        with self.assertRaisesRegex(ValueError, r"moe.*num_experts_per_tok"):
            mistral_native_config_from_params(raw)


class TestMistralNativeConfig(unittest.TestCase):
    def test_mutual_exclusivity_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Cannot set both"):
            MistralNativeConfig(
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
                quantization=QuantizationArgs(QFormat.FP8_E4M3, "TENSOR"),
                quantization_config=make_hf_fp8_quant_config(),
            )

    def test_single_or_no_quantization_ok(self) -> None:
        test_cases = [
            ("native_only", {"quantization": QuantizationArgs(QFormat.FP8_E4M3, "TENSOR")}),
            ("hf_only", {"quantization_config": make_hf_fp8_quant_config()}),
            ("neither", {}),
        ]
        for test_id, quant_kwargs in test_cases:
            with self.subTest(test_id):
                native = MistralNativeConfig(
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
                    **quant_kwargs,
                )
                has_native = "quantization" in quant_kwargs
                has_hf = "quantization_config" in quant_kwargs
                self.assertEqual(native.quantization is not None, has_native)
                self.assertEqual(native.quantization_config is not None, has_hf)


class TestNativeToHF(unittest.TestCase):
    def test_native_to_hf(self) -> None:
        test_cases = [
            ("mistral", base_native_config, MistralConfig, expected_mistral_hf_config),
            ("ministral3", ministral3_native_config, Ministral3Config, expected_ministral3_hf_config),
            ("mistral4", mistral4_native_config, Mistral4Config, expected_mistral4_hf_config),
            ("mistral3", mistral3_native_config, Mistral3Config, expected_mistral3_hf_config),
        ]
        for test_id, native_factory, expected_type, expected_factory in test_cases:
            with self.subTest(test_id):
                hf = mistral_native_config_to_hf_config(native_factory())
                self.assertIsInstance(hf, expected_type)
                self.assertEqual(hf, expected_factory())

    def test_partial_rotary_factor_uses_qk_rope_over_total(self) -> None:
        """`qk_rope_head_dim=32`, `qk_nope_head_dim=96` is asymmetric, so only the correct
        `qk_rope / (qk_nope + qk_rope)` formula can produce 0.25; the swapped formula
        (`qk_nope / (qk_nope + qk_rope)`) would produce 0.75 instead. Every other MLA fixture
        in this module has an equal rope/nope split, which makes the two formulas
        indistinguishable."""
        hf = mistral_native_config_to_hf_config(mistral4_asymmetric_mla_native_config())
        self.assertEqual(hf.rope_parameters["partial_rotary_factor"], 0.25)

    def test_non_yarn_omits_llama_4_scaling_beta(self) -> None:
        hf = mistral_native_config_to_hf_config(base_native_config())
        self.assertNotIn("llama_4_scaling_beta", hf.rope_parameters)

    def test_forward_moe_without_mla_raises(self) -> None:
        moe = moe_args()
        native = MistralNativeConfig(
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
            moe=moe,
        )
        with self.assertRaisesRegex(ValueError, "MOE and MLA"):
            mistral_native_config_to_hf_config(native)

    def test_qk_rope_nope_mismatch_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "qk_rope and qk_nope must both be None or both set"):
            _get_rope_parameters(
                rope_theta=10000.0,
                yarn_args=None,
                llama4_scaling=None,
                qk_rope=64,
                qk_nope=None,
            )

    def test_llama_4_scaling_without_yarn_raises(self) -> None:
        native = mistral4_native_config()
        native.yarn = None
        native.llama_4_scaling = Llama4Scaling(original_max_position_embeddings=16384, beta=0.2)
        with self.assertRaisesRegex(ValueError, "llama_4_scaling"):
            mistral_native_config_to_hf_config(native)

    def test_mm_projector_id_other_than_patch_merge_raises(self) -> None:
        native = mistral3_native_config()
        native.vision_encoder.mm_projector_id = "something_else"
        with self.assertRaisesRegex(ValueError, "mm_projector_id.*something_else.*patch_merge"):
            mistral_native_config_to_hf_config(native)

    def test_outer_tie_word_embeddings_matches_native(self) -> None:
        """The outer `Mistral3Config.tie_word_embeddings` must reflect `native.tied_embeddings`
        directly: the reverse converter reads tying off `text_config` instead, so a wrong
        outer value here is otherwise invisible to every round-trip test."""
        native = perturbed_mistral3_native_config()
        self.assertTrue(native.tied_embeddings)
        hf = mistral_native_config_to_hf_config(native)
        self.assertEqual(hf.tie_word_embeddings, native.tied_embeddings)

    def test_add_pre_mm_projector_layer_norm_false_raises(self) -> None:
        native = mistral3_native_config()
        native.vision_encoder.add_pre_mm_projector_layer_norm = False
        with self.assertRaisesRegex(ValueError, "add_pre_mm_projector_layer_norm"):
            mistral_native_config_to_hf_config(native)

    def test_mistral4_head_dim_inconsistent_with_mla_split_raises(self) -> None:
        native = mistral4_native_config()
        native.head_dim = native.qk_nope_head_dim + native.qk_rope_head_dim + 1
        with self.assertRaisesRegex(ValueError, "head_dim"):
            mistral_native_config_to_hf_config(native)

    def test_vision_encoder_unsupported_type_raises(self) -> None:
        """`vision_encoder` is typed `VisionEncoderArgs | None` but not enforced at runtime;
        no public path can build a `MistralNativeConfig` with any other value, but the dispatcher
        must still refuse rather than silently returning `None`."""
        native = base_native_config()
        native.vision_encoder = "not-a-vision-encoder"
        with self.assertRaisesRegex(ValueError, "Unsupported vision_encoder type"):
            mistral_native_config_to_hf_config(native)


class TestNativeToHFQuantization(unittest.TestCase):
    def test_tensor_fp8_propagates_to_mistral3(self) -> None:
        native = MistralNativeConfig(
            dim=4096,
            n_layers=32,
            head_dim=128,
            hidden_dim=14336,
            n_heads=32,
            n_kv_heads=8,
            rope_theta=1000000000.0,
            norm_eps=1e-5,
            vocab_size=32000,
            max_position_embeddings=131072,
            quantization=QuantizationArgs(QFormat.FP8_E4M3, "TENSOR"),
            vision_encoder=vision_encoder_args(),
        )
        hf = mistral_native_config_to_hf_config(native)
        qc = hf.quantization_config
        if hasattr(qc, "to_dict"):
            qc = qc.to_dict()
        self.assertEqual(qc["activation_scheme"], "static")

    def test_quantization_config_passthrough_mistral(self) -> None:
        native = MistralNativeConfig(
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
            quantization_config=make_hf_fp8_quant_config("dynamic"),
        )
        hf = mistral_native_config_to_hf_config(native)
        qc_dict = hf.quantization_config.to_dict()
        self.assertEqual(qc_dict["activation_scheme"], "dynamic")
        self.assertEqual(qc_dict["quant_method"], "fp8")

    def test_native_quantization_forward_still_works(self) -> None:
        native = MistralNativeConfig(
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
            quantization=QuantizationArgs(QFormat.FP8_E4M3, "TENSOR"),
        )
        hf = mistral_native_config_to_hf_config(native)
        qc_dict = hf.quantization_config.to_dict()
        self.assertEqual(qc_dict["activation_scheme"], "static")


if __name__ == "__main__":
    unittest.main()
