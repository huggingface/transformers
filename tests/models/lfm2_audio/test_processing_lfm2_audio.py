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
"""Tests for the LFM2-Audio processor."""

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers

from transformers import (
    AutoProcessor,
    Lfm2AudioDetokenizer,
    Lfm2AudioProcessor,
    ParakeetFeatureExtractor,
    PreTrainedTokenizerFast,
)
from transformers.testing_utils import require_librosa, require_torch, require_torch_gpu


@require_torch
@require_librosa
class Lfm2AudioProcessorTest(unittest.TestCase):
    def setUp(self):
        vocabulary = {
            "<unk>": 0,
            "<|pad|>": 1,
            "<|startoftext|>": 2,
            "<|im_start|>": 3,
            "<|im_end|>": 4,
            "<|reserved_123|>": 5,
            "system": 6,
            "user": 7,
            "assistant": 8,
            "Perform": 9,
            "ASR.": 10,
            "hello": 11,
        }
        tokenizer_backend = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
        tokenizer_backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_backend,
            unk_token="<unk>",
            pad_token="<|pad|>",
            bos_token="<|startoftext|>",
            additional_special_tokens=["<|im_start|>", "<|im_end|>", "<|reserved_123|>"],
        )
        feature_extractor = ParakeetFeatureExtractor(
            feature_size=8,
            sampling_rate=16_000,
            hop_length=160,
            n_fft=512,
            win_length=400,
        )
        self.processor = Lfm2AudioProcessor(feature_extractor, tokenizer)

    def test_audio_placeholder_expansion(self):
        audio = np.zeros(1600, dtype=np.float32)
        outputs = self.processor(text=[self.processor.audio_token], audio=[audio])

        encoded_frames = int(outputs.input_features_attention_mask.sum())
        expected_tokens = (encoded_frames + 7) // 8
        self.assertEqual(int((outputs.input_ids == self.processor.audio_token_id).sum()), expected_tokens)
        self.assertTrue((outputs.modality_ids == 2).all())

    def test_audio_placeholder_expansion_at_subsampling_boundary(self):
        # 12,800 samples produce 80 reported frames plus the terminal centered-STFT frame. Liquid Audio includes
        # that terminal frame, so FastConformer's 8x subsampling requires 11 placeholders rather than 10.
        audio = np.zeros(12_800, dtype=np.float32)
        outputs = self.processor(text=[self.processor.audio_token], audio=[audio])

        self.assertEqual(outputs.input_features.shape[1], 81)
        self.assertEqual(int(outputs.input_features_attention_mask.sum()), 81)
        self.assertEqual(int((outputs.input_ids == self.processor.audio_token_id).sum()), 11)
        self.assertEqual(int((outputs.modality_ids == 2).sum()), 11)

    @require_torch_gpu
    def test_transcription_request_on_cuda(self):
        import torch

        audio = np.zeros(1600, dtype=np.float32)
        outputs = self.processor.apply_transcription_request(audio, device=torch.device("cuda"))

        self.assertEqual(outputs.input_features.device.type, "cuda")
        self.assertEqual(outputs.input_features.dtype, torch.float32)
        self.assertEqual(outputs.input_features_attention_mask.device.type, "cuda")

    def test_text_to_speech_template(self):
        outputs = self.processor.apply_text_to_speech_request("hello")
        prompt = self.processor.tokenizer.decode(outputs.input_ids[0], skip_special_tokens=False)

        self.assertIn("system", prompt)
        self.assertIn("hello", prompt)
        self.assertNotIn(self.processor.audio_token, prompt)

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as directory:
            self.processor.save_pretrained(directory)
            reloaded = AutoProcessor.from_pretrained(directory)

        self.assertIsInstance(reloaded, Lfm2AudioProcessor)
        self.assertEqual(reloaded.audio_token, self.processor.audio_token)
        self.assertEqual(reloaded.feature_extractor.feature_size, 8)
        self.assertEqual(reloaded.audio_codec_model_id, "kyutai/mimi")

    def test_decode_audio_uses_native_mimi_interface(self):
        import torch

        class DummyMimi(torch.nn.Module):
            def decode(self, audio_codes, return_dict=True):
                self.audio_codes = audio_codes
                return SimpleNamespace(audio_values=torch.zeros((audio_codes.shape[0], 1, 32)))

        audio_codes = torch.randint(0, 2048, (8, 2))
        audio_codes = torch.cat([audio_codes, torch.full((8, 1), 2048)], dim=-1)
        codec = DummyMimi()

        audio = self.processor.decode_audio(audio_codes, audio_codec=codec)

        self.assertEqual(codec.audio_codes.shape, (1, 8, 2))
        self.assertEqual(audio.shape, (1, 32))

    def test_decode_audio_prefers_bundled_detokenizer(self):
        import torch

        class DummyDetokenizer(torch.nn.Module):
            def forward(self, audio_codes):
                self.audio_codes = audio_codes
                return torch.zeros((audio_codes.shape[0], 32))

        audio_codes = torch.randint(0, 2048, (8, 2))
        audio_codes = torch.cat([audio_codes, torch.full((8, 1), 2048)], dim=-1)
        decoder = DummyDetokenizer()
        self.processor.decoder_model_id = "dummy/model"

        with patch.object(Lfm2AudioDetokenizer, "from_pretrained", return_value=decoder) as from_pretrained:
            audio = self.processor.decode_audio(audio_codes)

        from_pretrained.assert_called_once_with(
            "dummy/model",
            subfolder="audio_detokenizer",
            dtype=torch.float32,
        )
        self.assertEqual(decoder.audio_codes.shape, (1, 8, 2))
        self.assertEqual(audio.shape, (1, 32))


if __name__ == "__main__":
    unittest.main()
