# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import ast
import importlib
import unittest
from pathlib import Path

from transformers.testing_utils import require_torch

from ..alm_tester import ALMModelTester
from ..causal_lm_tester import CausalLMModelTester
from ..model_tester_config import ModelTesterConfigMixin
from ..models.mistral_large4.test_modeling_mistral_large4 import MistralLarge4VisionText2TextModelTester
from ..multimodal_tester import MultiModalModelTester
from ..vlm_tester import VLMModelTester


class SmallConfigDefaultsTest(unittest.TestCase):
    def _tester_class(self, base):
        class Tester(base):
            small_config_overrides = {
                "hidden_size": 64,
                "image_size": 16,
                "patch_size": 4,
                "num_mel_bins": 24,
                "no_rope_layers": [1, 0],
            }

            def _check_required_attributes(self):
                # These tests exercise constructor defaults without choosing model classes.
                pass

        return Tester

    def test_model_overrides_shared_defaults_and_caller_wins(self):
        for base in (MultiModalModelTester, VLMModelTester, ALMModelTester):
            with self.subTest(base=base.__name__):
                tester = self._tester_class(base)
                self.assertEqual(tester(self).hidden_size, 64)
                self.assertEqual(tester(self, hidden_size=128).hidden_size, 128)
                self.assertEqual(tester(self).intermediate_size, 32)

    def test_vision_derived_defaults_use_model_and_caller_overrides(self):
        tester = self._tester_class(VLMModelTester)
        model_defaults = tester(self)
        self.assertEqual(model_defaults.num_image_tokens, 16)
        self.assertEqual(model_defaults.seq_length, 23)
        caller_defaults = tester(self, image_size=8)
        self.assertEqual(caller_defaults.num_image_tokens, 4)
        self.assertEqual(caller_defaults.seq_length, 11)
        self.assertEqual(tester(self, num_image_tokens=2).seq_length, 9)

    def test_audio_model_overrides_modality_defaults(self):
        tester = self._tester_class(ALMModelTester)
        self.assertEqual(tester(self).num_mel_bins, 24)
        self.assertEqual(tester(self, num_mel_bins=16).num_mel_bins, 16)

    def test_inherited_hook_keeps_existing_vision_defaults(self):
        class LegacyTester(VLMModelTester):
            def _check_required_attributes(self):
                pass

        tester = LegacyTester(self)
        self.assertEqual(tester.hidden_size, 32)
        self.assertEqual(tester.image_size, 8)
        self.assertEqual(tester.num_image_tokens, 4)
        self.assertEqual(tester.seq_length, 11)


class CausalLMSmallConfigDefaultsTest(unittest.TestCase):
    def _tester_class(self, defaults):
        class Tester(CausalLMModelTester):
            small_config_overrides = defaults

            @classmethod
            def _verify_and_infer_model_attributes(cls):
                # These tests exercise constructor defaults without choosing model classes.
                pass

        return Tester

    def test_model_overrides_shared_and_causal_defaults_and_caller_wins(self):
        tester = self._tester_class({"hidden_size": 64, "hidden_dropout_prob": 0.0, "mamba_n_heads": 4})
        defaults = tester(self)
        self.assertEqual(defaults.hidden_size, 64)
        self.assertEqual(defaults.hidden_dropout_prob, 0.0)
        self.assertEqual(defaults.mamba_n_heads, 4)
        self.assertEqual(defaults.intermediate_size, 32)
        self.assertEqual(defaults.attention_probs_dropout_prob, 0.1)
        caller = tester(self, hidden_size=128, hidden_dropout_prob=0.2)
        self.assertEqual(caller.hidden_size, 128)
        self.assertEqual(caller.hidden_dropout_prob, 0.2)

    def test_head_dim_derived_from_final_widths(self):
        tester = self._tester_class({"hidden_size": 64, "num_attention_heads": 4})
        self.assertEqual(tester(self).head_dim, 16)
        self.assertEqual(tester(self, hidden_size=32).head_dim, 8)
        self.assertEqual(self._tester_class({"head_dim": 32})(self).head_dim, 32)

    def test_inherited_hook_keeps_existing_defaults(self):
        tester = self._tester_class({})(self)
        self.assertEqual(tester.hidden_size, 32)
        self.assertEqual(tester.head_dim, 16)
        self.assertEqual(tester.batch_size, 13)
        self.assertEqual(tester.mamba_chunk_size, 16)
        self.assertFalse(tester.tie_word_embeddings)
        self.assertIsNone(tester.scope)


class MergedSmallConfigDefaultsTest(unittest.TestCase):
    def test_merged_values_do_not_require_a_tester_instance(self):
        # These bases cannot be instantiated without model classes, but their
        # merged defaults are useful independently of any model or test case.
        for base in (CausalLMModelTester, MultiModalModelTester, VLMModelTester, ALMModelTester):
            with self.subTest(base=base.__name__):
                values = base.get_small_config_defaults(hidden_size=64, num_attention_heads=4)
                self.assertEqual(values["hidden_size"], 64)
                self.assertEqual(values["vocab_size"], 99)
                if base in (CausalLMModelTester, VLMModelTester):
                    self.assertEqual(values["head_dim"], 16)
        self.assertEqual(VLMModelTester.get_small_config_defaults()["seq_length"], 11)
        self.assertEqual(ALMModelTester.get_small_config_defaults()["seq_length"], 32)

    def test_inherited_model_overrides_and_explicit_derived_values(self):
        class Parent(VLMModelTester):
            small_config_overrides = {"image_size": 16, "hidden_size": 64, "seq_length": 29}

        class Child(Parent):
            small_config_overrides = {"patch_size": 8, "head_dim": 24}

        values = Child.get_small_config_defaults()
        self.assertEqual(values["image_size"], 16)
        self.assertEqual(values["num_image_tokens"], 4)
        self.assertEqual(values["seq_length"], 29)
        self.assertEqual(values["head_dim"], 24)
        caller = Child.get_small_config_defaults(seq_length=41, head_dim=8, num_image_tokens=2)
        self.assertEqual((caller["seq_length"], caller["head_dim"], caller["num_image_tokens"]), (41, 8, 2))

    def test_mutable_defaults_and_overrides_are_isolated(self):
        class Tester(VLMModelTester):
            small_config_overrides = {"no_rope_layers": [1, 0]}

            def _check_required_attributes(self):
                pass

        values = Tester.get_small_config_defaults()
        values["vision_rope_parameters"]["rope_theta"] = 1
        values["no_rope_layers"].append(1)
        first, second = Tester(self), Tester(self)
        first.no_rope_layers.append(0)
        fresh = Tester.get_small_config_defaults()
        self.assertEqual(fresh["vision_rope_parameters"]["rope_theta"], 10_000)
        self.assertEqual(fresh["no_rope_layers"], [1, 0])
        self.assertEqual(second.no_rope_layers, [1, 0])
        self.assertEqual(Tester.small_config_overrides["no_rope_layers"], [1, 0])

    def test_mistral_overrides_are_a_literal_and_preserve_test_dimensions(self):
        path = Path(__file__).parents[1] / "models/mistral_large4/test_modeling_mistral_large4.py"
        tree = ast.parse(path.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name.endswith("ModelTester"))
        overrides = next(
            node.value
            for node in cls.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "small_config_overrides" for target in node.targets)
        )
        self.assertEqual(ast.literal_eval(overrides), MistralLarge4VisionText2TextModelTester.small_config_overrides)
        values = MistralLarge4VisionText2TextModelTester.get_small_config_defaults()
        self.assertEqual(values["hidden_size"], 32)
        self.assertEqual(values["qk_rope_head_dim"], 8)
        self.assertEqual(values["n_routed_experts"], 8)
        tester = MistralLarge4VisionText2TextModelTester(self)
        for name, value in values.items():
            self.assertEqual(getattr(tester, name), value)


# Testers that still set values in their own `__init__`, so `get_small_config_defaults()` doesn't
# return what they build. Move those values into `small_config_overrides` (or a derived value in
# `get_small_config_defaults`) and remove the tester from this list. Don't add new entries.
TESTERS_NOT_YET_STATIC = {
    "afmoe::AfmoeModelTester",
    "apertus::ApertusModelTester",
    "aria::AriaVisionText2TextModelTester",
    "audioflamingo3::AudioFlamingo3ModelTester",
    "axk1::AXK1ModelTester",
    "axk2::AXK2ModelTester",
    "bailing_hybrid::BailingHybridModelTester",
    "beam::BeamModelTester",
    "blt::BltModelTester",
    "cohere2_moe::Cohere2MoeModelTester",
    "cohere_compass::CohereCompassModelTester",
    "cohere_compass::CohereCompassTextModelTester",
    "cosmos3_edge::Cosmos3EdgeVisionText2TextModelTester",
    "cosmos3_omni::Cosmos3OmniVisionText2TextModelTester",
    "dbrx::DbrxModelTester",
    "deepseek_ocr2::DeepseekOcr2VisionText2TextModelTester",
    "deepseek_v2::DeepseekV2ModelTester",
    "deepseek_v32::DeepseekV32ModelTester",
    "deepseek_v4::DeepseekV4ModelTester",
    "deepseek_vl::DeepseekVLVisionText2TextModelTester",
    "dots1::Dots1ModelTester",
    "ernie4_5_vl_moe::Ernie4_5_VLMoeVisionText2TextModelTester",
    "exaone4::Exaone4ModelTester",
    "exaone4_5::Exaone4_5_ModelTester",
    "falcon::FalconModelTester",
    "flex_olmo::FlexOlmoModelTester",
    "flint::FlintModelTester",
    "fun_asr_nano::FunAsrNanoModelTester",
    "gemma3::Gemma3TextModelTester",
    "gemma3::Gemma3Vision2TextModelTester",
    "gemma3n::Gemma3nTextModelTester",
    "gemma4::Gemma4Audio2TextModelTester",
    "gemma4::Gemma4TextModelTester",
    "gemma4::Gemma4Vision2TextModelTester",
    "gemma4_unified::Gemma4UnifiedTextModelTester",
    "glm4::Glm4ModelTester",
    "glm4_moe::Glm4MoeModelTester",
    "glm4_moe_lite::Glm4MoeLiteModelTester",
    "glm5_next::Glm5NextVisionText2TextModelTester",
    "glm::GlmModelTester",
    "glm_moe_dsa::GlmMoeDsaModelTester",
    "glmasr::GlmAsrModelTester",
    "gpt2::GPT2ModelTester",
    "granite4_vision::Granite4VisionModelTester",
    "granite_speech::GraniteSpeechModelTester",
    "granite_speech_plus::GraniteSpeechPlusForConditionalGenerationModelTester",
    "granitemoe::GraniteMoeModelTester",
    "granitemoeshared::GraniteMoeSharedModelTester",
    "hrm_text::HrmTextModelTester",
    "hunyuan_vl::HunYuanVLVisionText2TextModelTester",
    "hy_v4::HYV4ModelTester",
    "hyperclovax_vision_v2::HCXVisionV2VisionText2TextModelTester",
    "inkling::InklingAudio2TextModelTester",
    "inkling::InklingTextModelTester",
    "inkling::InklingVision2TextModelTester",
    "jetmoe::JetMoeModelTester",
    "kimi_k25::Kimi_K25VisionText2TextModelTester",
    "kimi_linear::KimiLinearModelTester",
    "kolibri1::Kolibri1ModelTester",
    "laguna::LagunaModelTester",
    "lfm2::Lfm2ModelTester",
    "lfm2_moe::Lfm2MoeModelTester",
    "lfm2_vl::Lfm2VlModelTester",
    "lfm3::Lfm3ModelTester",
    "lfm3_vl::Lfm3VlVisionText2TextModelTester",
    "llava_next::LlavaNextVisionText2TextModelTester",
    "longcat_flash::LongcatFlashModelTester",
    "mellum::MellumModelTester",
    "mimo_v2_flash::MiMoV2FlashModelTester",
    "minicpm3::MiniCPM3ModelTester",
    "minicpm_ocr::MiniCPMOcrVisionText2TextModelTester",
    "minicpmv4_6::MiniCPMV4_6VisionText2TextModelTester",
    "minicpmv4_7::MiniCPMV4_7VisionText2TextModelTester",
    "minimax::MiniMaxModelTester",
    "minimax_m3_vl::MiniMaxM3VLVisionText2TextModelTester",
    "molmo2::Molmo2VisionText2TextModelTester",
    "moss_transcribe_diarize::MossTranscribeDiarizeModelTester",
    "muse_glimmer::MuseGlimmerVision2TextModelTester",
    "muse_spark::MuseSparkVision2TextModelTester",
    "musicflamingo::MusicFlamingoModelTester",
    "nemotron3_5_transcribe::Nemotron3_5TranscribeModelTester",
    "nemotron_h_omni::NemotronHOmniAudio2TextModelTester",
    "nemotron_h_omni::NemotronHOmniVision2TextModelTester",
    "olmo3::Olmo3ModelTester",
    "olmo_hybrid::OlmoHybridModelTester",
    "phi3::Phi3ModelTester",
    "pp_formulanet::PPFormulaNetModelTester",
    "qianfan_ocr::QianfanOCRVisionText2TextModelTester",
    "qwen2_audio::Qwen2AudioModelTester",
    "qwen3::Qwen3ModelTester",
    "qwen3_5::Qwen3_5TextModelTester",
    "qwen3_5_moe::Qwen3_5MoeTextModelTester",
    "qwen3_5_moe::Qwen3_5MoeVisionText2TextModelTester",
    "qwen3_asr::Qwen3ASRModelTester",
    "qwen3_next::Qwen3NextModelTester",
    "qwen3_vl::Qwen3VLVisionText2TextModelTester",
    "qwen3_vl_moe::Qwen3VLMoeVisionText2TextModelTester",
    "qwen4_exp::Qwen4ExpTextModelTester",
    "qwen4_exp::Qwen4ExpVisionText2TextModelTester",
    "recurrent_gemma::RecurrentGemmaModelTester",
    "seed_oss::SeedOssModelTester",
    "shieldgemma2::ShieldGemma2ModelTester",
    "solar_open::SolarOpenModelTester",
    "step3p7::Step3p7VisionText2TextModelTester",
    "ultravbert::UltraVBertModelTester",
    "unlimited_ocr::UnlimitedOcrVisionText2TextModelTester",
    "voxtral::VoxtralModelTester",
    "voxtral_realtime::VoxtralRealtimeModelTester",
    "youtu::YoutuModelTester",
    "zaya::ZayaModelTester",
}

_TESTER_BASE_NAMES = ("CausalLMModelTester", "VLMModelTester", "ALMModelTester", "MultiModalModelTester")


def _static_runtime_mismatches(tester):
    """Fields where a built tester differs from its class-level `get_small_config_defaults()`."""
    static = type(tester).get_small_config_defaults()
    runtime = {
        name: value
        for name, value in vars(tester).items()
        if name != "parent" and isinstance(value, (int, float, str, bool, type(None), list, tuple, dict))
    }
    return sorted(name for name, value in runtime.items() if name not in static or static[name] != value)


@require_torch
class StaticRuntimeParityTest(unittest.TestCase):
    def test_static_defaults_match_built_testers(self):
        models_dir = Path(__file__).parents[1] / "models"
        mismatches, seen = {}, set()
        for path in sorted(models_dir.glob("*/test_modeling_*.py")):
            if not any(name in path.read_text() for name in _TESTER_BASE_NAMES):
                continue
            module = importlib.import_module(f"tests.models.{path.parent.name}.{path.stem}")
            for case in vars(module).values():
                if not (isinstance(case, type) and issubclass(case, unittest.TestCase)):
                    continue
                tester_class = getattr(case, "model_tester_class", None)
                if case.__module__ != module.__name__ or not (
                    isinstance(tester_class, type) and issubclass(tester_class, ModelTesterConfigMixin)
                ):
                    continue
                test_case = case(next(name for name in dir(case) if name.startswith("test_")))
                test_case.setUp()
                key = f"{path.parent.name}::{type(test_case.model_tester).__name__}"
                seen.add(key)
                fields = _static_runtime_mismatches(test_case.model_tester)
                if fields:
                    mismatches.setdefault(key, set()).update(fields)

        new = {key: sorted(fields) for key, fields in mismatches.items() if key not in TESTERS_NOT_YET_STATIC}
        self.assertFalse(
            new,
            "These testers build values that `get_small_config_defaults()` doesn't return. Declare them in "
            f"`small_config_overrides` instead of setting them in `__init__`: {new}",
        )
        fixed = sorted(TESTERS_NOT_YET_STATIC - mismatches.keys())
        self.assertFalse(fixed, f"These testers now match or no longer exist; remove them from the list: {fixed}")
