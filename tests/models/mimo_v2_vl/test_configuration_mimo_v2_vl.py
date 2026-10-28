# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, sentinel

from parameterized import parameterized

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    MiMoV2FlashConfig,
    MiMoV2FlashForCausalLM,
    MiMoV2Processor,
    MiMoV2VLAudioConfig,
    MiMoV2VLAudioTokenizerConfig,
    MiMoV2VLConfig,
    MiMoV2VLForConditionalGeneration,
    MiMoV2VLTextConfig,
    MiMoV2VLVisionConfig,
)
from transformers.testing_utils import require_torch, require_torchvision, require_vision

from .test_modeling_mimo_v2_vl import get_config
from .test_processing_mimo_v2 import make_processor


class MiMoV2VLConfigValidationTest(unittest.TestCase):
    @parameterized.expand([(None,), ("flat",), ("per_layer",)])
    def test_updated_native_settings_survive_serialization(self, rope_format):
        parameters = None
        if rope_format == "flat":
            parameters = {"rope_type": "default", "rope_theta": 10000.0}
        elif rope_format == "per_layer":
            parameters = {
                "full_attention": {"rope_type": "default", "rope_theta": 10000.0},
                "sliding_attention": {"rope_type": "default", "rope_theta": 1000.0},
            }
        config = MiMoV2VLTextConfig(
            num_hidden_layers=2, hybrid_layer_pattern=[0, 1], sliding_window=4, rope_parameters=parameters
        )
        config.sliding_window = 9
        config.layer_types = ["sliding_attention", "full_attention"]
        config.rope_theta = 20000.0
        config.swa_rope_theta = 30000.0
        config.partial_rotary_factor = 0.5
        original_parameters = copy.deepcopy(config.rope_parameters)
        serialized = config.to_dict()
        self.assertEqual(config.rope_parameters, original_parameters)
        restored = MiMoV2VLTextConfig.from_dict(serialized)
        self.assertEqual(restored.sliding_window, 9)
        self.assertEqual(restored.sliding_window_size, 8)
        self.assertEqual(restored.layer_types, config.layer_types)
        self.assertEqual(restored.hybrid_layer_pattern, [1, 0])
        self.assertEqual(restored.rope_theta, 20000.0)
        self.assertEqual(restored.swa_rope_theta, 30000.0)
        self.assertEqual(restored.partial_rotary_factor, 0.5)
        self.assertEqual(restored.to_dict(), serialized)

    def test_conflicting_window_fields_raise(self):
        with self.assertRaisesRegex(ValueError, "sliding_window_size"):
            MiMoV2VLTextConfig(sliding_window=9, sliding_window_size=3)
        config = MiMoV2VLTextConfig(sliding_window=1, sliding_window_size=0)
        self.assertEqual(config.sliding_window, 1)

    def test_legacy_flat_window_fields_are_normalized(self):
        config = AutoConfig.for_model(
            "mimo_v2", vision_config={"depth": 1}, sliding_window=128, sliding_window_size=128
        )
        self.assertIsInstance(config, MiMoV2VLConfig)
        self.assertEqual(config.text_config.sliding_window, 129)
        self.assertEqual(config.text_config.sliding_window_size, 128)
        restored = MiMoV2VLConfig.from_dict(config.to_dict())
        self.assertEqual(restored.text_config.sliding_window, 129)

    @parameterized.expand(
        [
            ("text_zero_kv_heads", MiMoV2VLTextConfig, {"num_key_value_heads": 0}),
            ("text_zero_sliding_kv_heads", MiMoV2VLTextConfig, {"swa_num_key_value_heads": 0}),
            ("text_zero_head_dim", MiMoV2VLTextConfig, {"head_dim": 0}),
            ("text_zero_value_dim", MiMoV2VLTextConfig, {"v_head_dim": 0}),
            ("text_zero_sliding_value_dim", MiMoV2VLTextConfig, {"swa_v_head_dim": 0}),
            ("text_zero_expert_groups", MiMoV2VLTextConfig, {"n_group": 0}),
            ("text_zero_experts", MiMoV2VLTextConfig, {"n_routed_experts": 0}),
            ("text_invalid_rotary_factor", MiMoV2VLTextConfig, {"partial_rotary_factor": 1.5}),
            ("text_nonfinite_rotary_factor", MiMoV2VLTextConfig, {"partial_rotary_factor": float("nan")}),
            ("text_zero_rope_base", MiMoV2VLTextConfig, {"rope_theta": 0}),
            ("text_negative_sliding_rope_base", MiMoV2VLTextConfig, {"swa_rope_theta": -1}),
            ("text_nonfinite_rope_base", MiMoV2VLTextConfig, {"rope_theta": float("inf")}),
            ("vision_zero_heads", MiMoV2VLVisionConfig, {"num_heads": 0}),
            ("vision_zero_kv_heads", MiMoV2VLVisionConfig, {"num_key_value_heads": 0}),
            ("vision_zero_head_dim", MiMoV2VLVisionConfig, {"qk_channels": 0}),
            ("vision_zero_patch_size", MiMoV2VLVisionConfig, {"patch_size": 0}),
            ("vision_zero_merge_size", MiMoV2VLVisionConfig, {"spatial_merge_size": 0}),
            ("audio_zero_channels", MiMoV2VLAudioConfig, {"audio_channels": 0}),
            ("audio_zero_group_size", MiMoV2VLAudioConfig, {"group_size": 0}),
            ("audio_zero_segment_size", MiMoV2VLAudioConfig, {"audio_segment_size": 0}),
            ("audio_zero_heads", MiMoV2VLAudioConfig, {"input_local_attn_heads": 0}),
            ("audio_odd_head_dim", MiMoV2VLAudioConfig, {"input_local_head_dim": 3}),
            ("codec_zero_heads", MiMoV2VLAudioTokenizerConfig, {"encoder_attention_heads": 0}),
            ("codec_zero_hidden_size", MiMoV2VLAudioTokenizerConfig, {"hidden_size": 0}),
            ("codec_zero_quantizers", MiMoV2VLAudioTokenizerConfig, {"num_quantizers": 0}),
            ("codec_empty_codebooks", MiMoV2VLAudioTokenizerConfig, {"codebook_size": []}),
            ("codec_zero_codebook_size", MiMoV2VLAudioTokenizerConfig, {"codebook_size": [0]}),
        ]
    )
    def test_invalid_architecture_settings_raise_value_error(self, _, config_class, kwargs):
        with self.assertRaises(ValueError):
            config_class(**kwargs)


@require_torch
@require_torchvision
@require_vision
class MiMoV2AutoSelectionTest(unittest.TestCase):
    def test_explicit_config_registration_takes_precedence(self):
        from transformers import PreTrainedConfig
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        class UserConfig(PreTrainedConfig):
            model_type = "mimo_v2"

        with patch.dict(CONFIG_MAPPING._extra_content, {"mimo_v2": UserConfig}):
            self.assertIsInstance(AutoConfig.for_model("mimo_v2", vision_config={"depth": 28}), UserConfig)
            with tempfile.TemporaryDirectory() as directory:
                Path(directory, "config.json").write_text(
                    json.dumps({"model_type": "mimo_v2", "vision_config": {"depth": 28}}), encoding="utf-8"
                )
                self.assertIsInstance(AutoConfig.from_pretrained(directory, trust_remote_code=False), UserConfig)

    @parameterized.expand(
        [
            ("XiaomiMiMo/MiMo-V2.5", 4096, 48, 256, 64, 4, 1e-5, None),
            ("XiaomiMiMo/MiMo-V2.6-Flash-RL", 4096, 48, 256, 64, 4, 1e-6, "bfloat16"),
            ("XiaomiMiMo/MiMo-V2.6-Pro-RL", 6144, 70, 384, 128, 8, 1e-5, "bfloat16"),
        ]
    )
    def test_multimodal_family_configurations(
        self, model_id, hidden_size, layers, experts, heads, kv_heads, epsilon, router_dtype
    ):
        metadata = {
            "hidden_size": hidden_size,
            "num_hidden_layers": layers,
            "n_routed_experts": experts,
            "num_attention_heads": heads,
            "num_key_value_heads": kv_heads,
            "swa_num_attention_heads": heads,
            "swa_num_key_value_heads": 8,
            "layernorm_epsilon": epsilon,
            "rope_theta": 10000000,
            "vision_config": {"out_hidden_size": hidden_size},
            "audio_config": {"out_hidden_size": hidden_size},
            "processor_config": {"audio_token_id": 151669},
            "architectures": ["MiMoV2ForCausalLM"],
        }
        if router_dtype is not None:
            metadata["moe_router_dtype"] = router_dtype
        config = AutoConfig.for_model("mimo_v2", **metadata)
        self.assertIsInstance(config, MiMoV2VLConfig, model_id)
        self.assertEqual(config.text_config.hidden_size, hidden_size)
        self.assertEqual(config.text_config.num_hidden_layers, layers)
        self.assertEqual(config.text_config.rms_norm_eps, epsilon)
        self.assertEqual(config.text_config.moe_router_dtype, router_dtype or "float32")
        self.assertEqual(config.audio_token_id, 151669)

    def test_processor_token_ids_are_defaults_not_overrides(self):
        config = MiMoV2VLConfig(processor_config={"image_token_id": 123, "audio_token_id": 125}, image_token_id=124)
        self.assertEqual(config.image_token_id, 124)
        self.assertEqual(config.audio_token_id, 125)

    @parameterized.expand(
        [(False, "MiMoV2ForCausalLM"), (True, "MiMoV2ForCausalLM"), (False, "UnrelatedName"), (True, "UnrelatedName")]
    )
    def test_auto_config_uses_encoder_settings(self, multimodal, architecture):
        tiny = get_config()
        metadata = tiny.text_config.to_dict()
        metadata.update(
            model_type="mimo_v2",
            architectures=[architecture],
            auto_map={"AutoConfig": "configuration_mimo_v2.MiMoV2Config"},
        )
        if multimodal:
            metadata.update(
                vision_config=tiny.vision_config.to_dict(),
                audio_config=tiny.audio_config.to_dict(),
                audio_tokenizer_config=tiny.audio_tokenizer_config.to_dict(),
            )
        else:
            metadata.update(vision_config=None, audio_config={})
        with tempfile.TemporaryDirectory(prefix="unrelated-repository-name-") as directory:
            Path(directory, "config.json").write_text(json.dumps(metadata), encoding="utf-8")
            config = AutoConfig.from_pretrained(directory, trust_remote_code=False, local_files_only=True)
            self.assertIsInstance(config, MiMoV2VLConfig if multimodal else MiMoV2FlashConfig)
            model = AutoModelForCausalLM.from_config(config)
            self.assertIsInstance(model, MiMoV2VLForConditionalGeneration if multimodal else MiMoV2FlashForCausalLM)

    def test_text_pro_parameters_are_preserved(self):
        config = AutoConfig.for_model(
            "mimo_v2",
            hidden_size=6144,
            num_hidden_layers=3,
            num_attention_heads=128,
            num_key_value_heads=8,
            swa_num_attention_heads=128,
            swa_num_key_value_heads=8,
            layernorm_epsilon=1e-5,
            attention_value_scale=0.612,
            hybrid_layer_pattern=[0, 1, 0],
            moe_layer_freq=[0, 1, 1],
            rope_theta=10000000,
            swa_rope_theta=10000,
            sliding_window_size=128,
        )
        self.assertIsInstance(config, MiMoV2FlashConfig)
        self.assertEqual(config.swa_num_key_value_heads, 8)
        self.assertEqual(config.rms_norm_eps, 1e-5)
        self.assertEqual(config.layer_types, ["full_attention", "sliding_attention", "full_attention"])
        self.assertEqual(config.mlp_layer_types, ["dense", "sparse", "sparse"])
        self.assertEqual(config.rope_parameters["full_attention"]["rope_theta"], 10000000)
        self.assertEqual(config.sliding_window, 129)
        self.assertEqual(config.attention_value_scale, 0.612)

    def test_auto_processor_overrides_only_legacy_qwen_declaration(self):
        tiny = get_config()
        metadata = tiny.text_config.to_dict()
        metadata.update(
            model_type="mimo_v2",
            architectures=["MiMoV2ForCausalLM"],
            vision_config=tiny.vision_config.to_dict(),
            audio_config=tiny.audio_config.to_dict(),
        )
        with tempfile.TemporaryDirectory() as directory:
            make_processor().tokenizer.save_pretrained(directory)
            Path(directory, "config.json").write_text(json.dumps(metadata), encoding="utf-8")
            Path(directory, "preprocessor_config.json").write_text(
                json.dumps(
                    {"processor_class": "Qwen2_5_VLProcessor", "image_processor_type": "Qwen2VLImageProcessor"}
                ),
                encoding="utf-8",
            )
            processor = AutoProcessor.from_pretrained(directory, trust_remote_code=False, local_files_only=True)
            self.assertIsInstance(processor, MiMoV2Processor)
            self.assertEqual(processor.image_processor.patch_size, 2)
            Path(directory, "processor_config.json").write_text(
                json.dumps({"processor_class": "Qwen2_5_VLProcessor"}), encoding="utf-8"
            )
            with patch("transformers.Qwen2_5_VLProcessor.from_pretrained", return_value=sentinel.processor):
                self.assertIs(
                    AutoProcessor.from_pretrained(directory, trust_remote_code=False, local_files_only=True),
                    sentinel.processor,
                )
