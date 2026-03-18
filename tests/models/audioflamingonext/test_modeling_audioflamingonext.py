# Copyright 2026 NVIDIA CORPORATION and the HuggingFace Inc. team. All rights
# reserved.
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

from transformers import AudioFlamingoNextConfig, AudioFlamingoNextForConditionalGeneration, AudioFlamingoNextProcessor
from transformers.models.auto import AutoConfig, AutoModel
from transformers.testing_utils import require_torch


@require_torch
class AudioFlamingoNextSmokeTest(unittest.TestCase):
    def test_top_level_exports(self):
        self.assertEqual(AudioFlamingoNextProcessor.__name__, "AudioFlamingoNextProcessor")
        self.assertEqual(AudioFlamingoNextConfig.model_type, "audioflamingonext")

    def test_auto_model_from_config(self):
        config = AudioFlamingoNextConfig(
            audio_config={
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_mel_bins": 128,
                "max_source_positions": 1500,
                "scale_embedding": False,
                "activation_function": "gelu",
                "dropout": 0.0,
                "attention_dropout": 0.0,
                "activation_dropout": 0.0,
                "layerdrop": 0.0,
            },
            text_config={
                "model_type": "qwen2",
                "vocab_size": 64,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "max_position_embeddings": 128,
                "tie_word_embeddings": False,
                "use_cache": False,
            },
        )

        self.assertEqual(AutoConfig.for_model("audioflamingonext").model_type, "audioflamingonext")
        self.assertEqual(config.audio_config.model_type, "audioflamingo3_encoder")
        self.assertIsInstance(AutoModel.from_config(config), AudioFlamingoNextForConditionalGeneration)
