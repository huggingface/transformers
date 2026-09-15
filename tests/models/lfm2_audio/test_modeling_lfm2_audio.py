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
"""Tests for LFM2-Audio."""

import math
import tempfile
import unittest
from unittest.mock import patch

from transformers import (
    Lfm2AudioConfig,
    Lfm2AudioDetokenizer,
    Lfm2AudioForConditionalGeneration,
    Lfm2AudioModel,
    Lfm2Config,
)
from transformers.testing_utils import require_torch, torch_device
from transformers.utils import is_torch_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import floats_tensor, ids_tensor


if is_torch_available():
    import torch


class Lfm2AudioModelTester:
    def __init__(self, parent, batch_size=2, sequence_length=8):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = sequence_length
        self.hidden_size = 32
        self.num_hidden_layers = 2
        self.num_attention_heads = 4
        self.vocab_size = 32
        self.is_training = True

    def get_config(self):
        return Lfm2AudioConfig(
            codebooks=2,
            audio_vocab_size=17,
            audio_eos_token_id=16,
            audio_token_id=6,
            audio_start_token_id=4,
            text_end_token_id=5,
            bos_token_id=1,
            eos_token_id=3,
            pad_token_id=0,
            preprocessor={"features": 8},
            encoder={
                "feat_in": 8,
                "n_layers": 1,
                "d_model": 16,
                "n_heads": 4,
                "subsampling_conv_channels": 8,
                "dropout": 0.0,
                "dropout_pre_encoder": 0.0,
                "dropout_emb": 0.0,
                "dropout_att": 0.0,
            },
            lfm={
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "intermediate_size": 64,
                "num_hidden_layers": self.num_hidden_layers,
                "num_attention_heads": self.num_attention_heads,
                "num_key_value_heads": 2,
                "layer_types": ["full_attention", "conv"],
                "max_position_embeddings": 64,
            },
            depthformer={
                "layers": 2,
                "dim": self.hidden_size,
                "num_attention_heads": self.num_attention_heads,
                "num_key_value_heads": 2,
                "intermediate_size": 64,
            },
        )

    def prepare_config_and_inputs_for_common(self):
        config = self.get_config()
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        input_ids[input_ids == config.audio_token_id] = 2
        attention_mask = input_ids.new_ones(input_ids.shape)
        return config, {"input_ids": input_ids, "attention_mask": attention_mask}


@require_torch
class Lfm2AudioModelTest(unittest.TestCase):
    all_model_classes = (Lfm2AudioModel, Lfm2AudioForConditionalGeneration)

    def setUp(self):
        self.model_tester = Lfm2AudioModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Lfm2AudioConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_text_forward(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).eval()
            outputs = model(**inputs)
            expected_shape = (self.model_tester.batch_size, self.model_tester.seq_length, config.hidden_size)
            if model_class is Lfm2AudioForConditionalGeneration:
                expected_shape = (self.model_tester.batch_size, self.model_tester.seq_length, config.vocab_size)
            self.assertEqual(outputs[0].shape, expected_shape)

    def test_save_and_load(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).eval()
            with torch.no_grad():
                expected = model(**inputs)[0]
            with tempfile.TemporaryDirectory() as directory:
                model.save_pretrained(directory)
                reloaded = model_class.from_pretrained(directory).to(torch_device).eval()
            with torch.no_grad():
                actual = reloaded(**inputs)[0]
            torch.testing.assert_close(actual, expected)

    def test_audio_input(self):
        config = self.model_tester.get_config()
        model = Lfm2AudioForConditionalGeneration(config).to(torch_device).eval()

        input_ids = ids_tensor([self.model_tester.batch_size, self.model_tester.seq_length], config.vocab_size)
        input_ids[:, 1:5] = config.audio_token_id
        modality_ids = input_ids.new_ones(input_ids.shape)
        modality_ids[:, 1:5] = 2
        input_features = floats_tensor([self.model_tester.batch_size, 32, 8])
        feature_mask = input_ids.new_ones((self.model_tester.batch_size, 32))

        outputs = model(
            input_ids=input_ids,
            input_features=input_features,
            input_features_attention_mask=feature_mask,
            modality_ids=modality_ids,
        )

        self.assertEqual(outputs.logits.shape, (self.model_tester.batch_size, self.model_tester.seq_length, 32))
        self.assertEqual(outputs.audio_hidden_states.shape, (self.model_tester.batch_size * 4, 32))
        self.assertEqual(model.model.conformer.config._attn_implementation, "eager")
        expected_inv_freq = torch.exp(
            torch.arange(0, config.encoder.hidden_size, 2, device=torch_device, dtype=torch.float32)
            * -(math.log(10_000.0) / config.encoder.hidden_size)
        )
        torch.testing.assert_close(
            model.model.conformer.encode_positions.inv_freq, expected_inv_freq, atol=0.0, rtol=0.0
        )

    def test_audio_loss_is_shifted(self):
        config = self.model_tester.get_config()
        model = Lfm2AudioForConditionalGeneration(config).to(torch_device).eval()
        input_ids = ids_tensor([self.model_tester.batch_size, self.model_tester.seq_length], config.vocab_size)
        input_ids[input_ids == config.audio_token_id] = 2
        audio_labels = input_ids.new_full((*input_ids.shape, config.codebooks), -100)
        audio_labels[:, 2:4] = ids_tensor([self.model_tester.batch_size, 2, config.codebooks], config.audio_vocab_size)

        outputs = model(input_ids=input_ids, audio_labels=audio_labels)

        self.assertEqual(outputs.audio_logits.shape, (self.model_tester.batch_size * 2, 2, 17))
        self.assertTrue(outputs.audio_loss.isfinite())

    def test_combined_loss_is_weighted_by_supervised_tokens(self):
        config = self.model_tester.get_config()
        model = Lfm2AudioForConditionalGeneration(config).to(torch_device).eval()
        input_ids = ids_tensor([self.model_tester.batch_size, self.model_tester.seq_length], config.vocab_size)
        input_ids[input_ids == config.audio_token_id] = 2
        labels = input_ids.clone()
        labels[:, 0] = -100
        audio_labels = input_ids.new_full((*input_ids.shape, config.codebooks), -100)
        audio_labels[:, 2:4] = ids_tensor([self.model_tester.batch_size, 2, config.codebooks], config.audio_vocab_size)

        outputs = model(input_ids=input_ids, labels=labels, audio_labels=audio_labels)

        text_tokens = (labels[:, 1:] != -100).sum()
        audio_tokens = self.model_tester.batch_size * 2
        expected = (outputs.text_loss * text_tokens + outputs.audio_loss * audio_tokens) / (text_tokens + audio_tokens)
        torch.testing.assert_close(outputs.loss, expected)

    def test_depthformer_cache_matches_full_forward(self):
        config = self.model_tester.get_config()
        depthformer = Lfm2AudioForConditionalGeneration(config).model.depthformer.to(torch_device).eval()
        hidden_states = floats_tensor([self.model_tester.batch_size, config.codebooks, config.depthformer.dim])

        full_output, _ = depthformer(hidden_states)
        cached_output = []
        cache = None
        for position in range(config.codebooks):
            output, cache = depthformer(
                hidden_states[:, position : position + 1], past_key_values=cache, use_cache=True
            )
            cached_output.append(output)

        torch.testing.assert_close(full_output, torch.cat(cached_output, dim=1), atol=1e-5, rtol=1e-5)

    def test_sequential_generation_switches_to_audio(self):
        config = self.model_tester.get_config()
        model = Lfm2AudioForConditionalGeneration(config).to(torch_device).eval()
        input_ids = ids_tensor([1, 3], config.vocab_size)
        sampled_tokens = [
            input_ids.new_tensor([config.audio_start_token_id]),
            input_ids.new_tensor([1]),
            input_ids.new_tensor([2]),
        ]

        with patch.object(model, "_sample", side_effect=sampled_tokens):
            output = model.generate(input_ids=input_ids, max_new_tokens=2)

        self.assertEqual(output.sequences.tolist(), [[config.audio_start_token_id]])
        self.assertEqual(output.audio_codes.shape, (1, config.codebooks, 1))
        self.assertEqual(output.modalities.tolist(), [[1, 3]])

    def test_interleaved_generation_does_not_return_terminal_eos(self):
        config = self.model_tester.get_config()
        model = Lfm2AudioForConditionalGeneration(config).to(torch_device).eval()
        input_ids = ids_tensor([1, 3], config.vocab_size)
        input_ids[input_ids == config.audio_token_id] = 2

        with patch.object(model, "_sample", return_value=input_ids.new_tensor([config.eos_token_id])):
            output = model.generate(input_ids=input_ids, max_new_tokens=1, generation_mode="interleaved")

        self.assertEqual(output.sequences.shape, (1, 0))
        self.assertEqual(output.audio_codes.shape, (1, config.codebooks, 0))
        self.assertEqual(output.modalities.shape, (1, 0))


@require_torch
class Lfm2AudioDetokenizerTest(unittest.TestCase):
    all_model_classes = (Lfm2AudioDetokenizer,)

    def test_forward(self):
        config = Lfm2Config(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            layer_types=["full_attention"],
            max_position_embeddings=64,
            output_size=1282,
        )
        audio_codes = ids_tensor([1, 8, 2], 2048).to(torch_device)

        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).eval()
            waveform = model(audio_codes)

            self.assertEqual(model.lfm.config._attn_implementation, "sdpa")
            self.assertEqual(waveform.shape, (1, 3840))
            self.assertTrue(waveform.isfinite().all())


if __name__ == "__main__":
    unittest.main()
