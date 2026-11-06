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
"""Tests for loading the original `OpenGVLab/InternVL2-*` checkpoints."""

import tempfile
import unittest

from huggingface_hub import get_safetensors_metadata

from transformers import (
    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING,
    AutoConfig,
    AutoTokenizer,
    InternVL2Config,
    InternVLForConditionalGeneration,
    LlamaConfig,
    Qwen2Config,
)
from transformers.testing_utils import require_torch, slow
from transformers.utils import is_torch_available


if is_torch_available():
    import torch

    from transformers.modeling_utils import str_to_torch_dtype


def get_original_config_dict(architecture="Qwen2ForCausalLM", model_type="qwen2"):
    return {
        "architectures": ["InternVLChatModel"],
        "model_type": "internvl_chat",
        "image_token_id": 151648,
        "select_layer": -1,
        "template": "Hermes-2",
        "llm_config": {
            "architectures": [architecture],
            "model_type": model_type,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "tie_word_embeddings": False,
        },
        "vision_config": {
            "architectures": ["InternVisionModel"],
            "model_type": "intern_vit_6b",
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "qk_normalization": True,
            "qkv_bias": False,
            "use_flash_attn": True,
        },
    }


class InternVL2ConfigTest(unittest.TestCase):
    def test_original_layout(self):
        config = InternVL2Config.from_dict(get_original_config_dict())

        self.assertIsInstance(config.text_config, Qwen2Config)
        self.assertEqual(config.text_config.hidden_size, 64)
        self.assertEqual(config.vision_config.model_type, "internvl_vision")
        self.assertEqual(config.vision_config.hidden_size, 32)
        self.assertTrue(config.vision_config.use_qk_norm)
        self.assertFalse(config.vision_config.attention_bias)
        self.assertFalse(config.tie_word_embeddings)
        self.assertEqual(config.image_token_id, 151648)

    def test_llama_language_model(self):
        config = InternVL2Config.from_dict(get_original_config_dict("LlamaForCausalLM", "llama"))
        self.assertIsInstance(config.text_config, LlamaConfig)

    def test_unsupported_language_model(self):
        with self.assertRaisesRegex(ValueError, "InternLM2ForCausalLM"):
            InternVL2Config.from_dict(get_original_config_dict("InternLM2ForCausalLM", "internlm2"))

    @require_torch
    def test_auto_model_mapping(self):
        self.assertIs(MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING[InternVL2Config], InternVLForConditionalGeneration)

    def test_save_and_reload(self):
        config = InternVL2Config.from_dict(get_original_config_dict())
        with tempfile.TemporaryDirectory() as tmp_dir:
            config.save_pretrained(tmp_dir)
            reloaded = InternVL2Config.from_pretrained(tmp_dir)

        self.assertEqual(reloaded.text_config.to_dict(), config.text_config.to_dict())
        self.assertEqual(reloaded.vision_config.to_dict(), config.vision_config.to_dict())
        self.assertFalse(reloaded.tie_word_embeddings)
        self.assertEqual(reloaded.image_token_id, 151648)


@slow
@require_torch
class InternVL2IntegrationTest(unittest.TestCase):
    checkpoints = [
        "OpenGVLab/InternVL2-1B",
        "OpenGVLab/InternVL2-40B",
        "OpenGVLab/InternVL2-Llama3-76B",
    ]

    def test_all_weights_load(self):
        # Meta tensors shaped from the safetensors headers, so no weights are downloaded
        for checkpoint in self.checkpoints:
            with self.subTest(checkpoint=checkpoint):
                state_dict = {
                    name: torch.empty(info.shape, dtype=str_to_torch_dtype[info.dtype], device="meta")
                    for file_metadata in get_safetensors_metadata(checkpoint).files_metadata.values()
                    for name, info in file_metadata.tensors.items()
                }
                _, loading_info = InternVLForConditionalGeneration.from_pretrained(
                    None,
                    config=AutoConfig.from_pretrained(checkpoint),
                    state_dict=state_dict,
                    device_map="meta",
                    output_loading_info=True,
                )

                self.assertEqual(sorted(loading_info["missing_keys"]), [])
                self.assertEqual(sorted(loading_info["unexpected_keys"]), [])
                self.assertEqual(sorted(loading_info["mismatched_keys"]), [])

    def test_image_token_id_matches_tokenizer(self):
        for checkpoint in self.checkpoints:
            with self.subTest(checkpoint=checkpoint):
                config = AutoConfig.from_pretrained(checkpoint)
                tokenizer = AutoTokenizer.from_pretrained(checkpoint)
                self.assertEqual(config.image_token_id, tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>"))
