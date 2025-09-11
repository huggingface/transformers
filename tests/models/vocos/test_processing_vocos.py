# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

import shutil
import tempfile
import unittest

import numpy as np
import torch
from parameterized import parameterized

from transformers import EncodecModel, VocosFeatureExtractor, VocosProcessor
from transformers.testing_utils import require_torch


# copied from Dia tests
def check_models_equal(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).any():
            return False
    return True


@require_torch
class VocosProcessorTest(unittest.TestCase):
    def setUp(self):
        self.checkpoint = "Manel/vocos-mel-24khz"
        self.tmpdir = tempfile.mkdtemp()

        self.processor = VocosProcessor(
            feature_extractor=self.get_feature_extractor(), audio_tokenizer=self.get_audio_tokenizer()
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def get_feature_extractor(self, **kwargs):
        return VocosFeatureExtractor.from_pretrained(self.checkpoint, **kwargs)

    def get_audio_tokenizer(self, **kwargs):
        return EncodecModel.from_pretrained("facebook/encodec_24khz", **kwargs)

    def test_save_load_pretrained_default(self):
        feature_extractor = self.get_feature_extractor()
        audio_tokenizer = self.get_audio_tokenizer()

        processor = VocosProcessor(feature_extractor=feature_extractor, audio_tokenizer=audio_tokenizer)
        processor.save_pretrained(self.tmpdir)

        # load back
        processor = VocosProcessor.from_pretrained(self.tmpdir)
        self.assertEqual(processor.feature_extractor.to_json_string(), feature_extractor.to_json_string())
        self.assertIsInstance(processor.feature_extractor, VocosFeatureExtractor)

        self.assertEqual(processor.audio_tokenizer.__class__.__name__, audio_tokenizer.__class__.__name__)
        self.assertIsInstance(processor.audio_tokenizer, EncodecModel)
        self.assertEqual(processor.audio_tokenizer.name_or_path, audio_tokenizer.name_or_path)
        self.assertTrue(check_models_equal(processor.audio_tokenizer, audio_tokenizer))

    def test_save_load_pretrained_additional_features(self):
        processor = VocosProcessor(
            feature_extractor=self.get_feature_extractor(), audio_tokenizer=self.get_audio_tokenizer()
        )

        processor.save_pretrained(self.tmpdir)

        feature_extractor_add_kwargs = self.get_feature_extractor(sampling_rate=16000)
        audio_tokenizer_add_kwargs = self.get_audio_tokenizer()

        processor = VocosProcessor.from_pretrained(self.tmpdir, sampling_rate=16000)

        self.assertEqual(processor.feature_extractor.to_json_string(), feature_extractor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.feature_extractor, VocosFeatureExtractor)

        self.assertEqual(processor.audio_tokenizer.__class__.__name__, audio_tokenizer_add_kwargs.__class__.__name__)
        self.assertEqual(processor.audio_tokenizer.name_or_path, audio_tokenizer_add_kwargs.name_or_path)

        self.assertTrue(check_models_equal(processor.audio_tokenizer, audio_tokenizer_add_kwargs))
        self.assertIsInstance(processor.audio_tokenizer, EncodecModel)

    def test_feature_extractor_mel_path(self):
        # making sure processor and feature extractor give same results for mel path
        audio = np.random.randn(2400).astype(np.float32)

        feature_extractor = self.get_feature_extractor()

        out_feature_extractor = feature_extractor(
            raw_speech=audio, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt"
        )["input_features"]

        out_processor = self.processor(audio=audio, return_tensors="pt")["features"]

        self.assertEqual(out_processor.shape, out_feature_extractor.shape)
        np.testing.assert_allclose(out_processor, out_feature_extractor, atol=1e-5)

    @parameterized.expand([[1.5], [3.0], [6.0], [12.0]])
    def test_encodec_audio_vs_codes_consistency(self, bandwidth):
        # checking that audio to codes and direct codes input give same outputs
        audio = torch.randn(1, 1024, dtype=torch.float32)

        output_processor = self.processor(audio=audio, bandwidth=bandwidth, return_tensors="pt")["features"]

        audio_tokenizer = self.processor.audio_tokenizer

        with torch.no_grad():
            encoded_frames = audio_tokenizer.encoder(audio.unsqueeze(1))
            codes = audio_tokenizer.quantizer.encode(encoded_frames, bandwidth=bandwidth)

        output_codes = self.processor(codes=codes, bandwidth=bandwidth, return_tensors="pt")["features"]

        torch.testing.assert_close(output_processor, output_codes)

    @parameterized.expand([[1.5], [3.0], [6.0], [12.0]])
    def test_encodec_codes_inputs(self, bandwidth):
        num_quantizers = self.processor.audio_tokenizer.quantizer.get_num_quantizers_for_bandwidth(bandwidth)

        seq_len = 200
        codes = torch.randint(
            0, self.processor.audio_tokenizer.quantizer.codebook_size, (num_quantizers, seq_len), dtype=torch.long
        )
        outputs = self.processor(codes=codes, bandwidth=bandwidth, return_tensors="pt")

        self.assertIn("features", outputs)
        self.assertIn("bandwidth", outputs)
        self.assertEqual(outputs["features"].shape[-1], seq_len)
        self.assertIsInstance(outputs["bandwidth"], (float, torch.Tensor))

    def test_neither_audio_nor_codes_raises(self):
        with self.assertRaises(ValueError):
            self.processor()
