# Copyright 2024 The HuggingFace Team. All rights reserved.
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
"""Tests for the MusicGen processor."""

import random
import shutil
import tempfile
import unittest

import numpy as np

from transformers import T5Tokenizer, T5TokenizerFast
from transformers.testing_utils import require_sentencepiece, require_torch, require_torchaudio
from transformers.utils.import_utils import is_torchaudio_available


if is_torchaudio_available():
    from transformers import MusicgenMelodyFeatureExtractor, MusicgenMelodyProcessor


global_rng = random.Random()


# Copied from tests.models.whisper.test_feature_extraction_whisper.floats_list
def floats_list(shape, scale=1.0, rng=None, name=None):
    """Creates a random float32 tensor"""
    if rng is None:
        rng = global_rng

    values = []
    for batch_idx in range(shape[0]):
        values.append([])
        for _ in range(shape[1]):
            values[-1].append(rng.random() * scale)

    return values


@require_torch
@require_sentencepiece
@require_torchaudio
# Copied from tests.models.musicgen.test_processing_musicgen.MusicgenProcessorTest with Musicgen->MusicgenMelody, Encodec->MusicgenMelody, padding_mask->attention_mask, input_values->input_features
class MusicgenMelodyProcessorTest(unittest.TestCase):
    def setUp(self):
        # Ignore copy
        self.checkpoint = "facebook/musicgen-melody"
        self.tmpdirname = tempfile.mkdtemp()

    def get_tokenizer(self, **kwargs):
        return T5Tokenizer.from_pretrained(self.checkpoint, **kwargs)

    def get_feature_extractor(self, **kwargs):
        return MusicgenMelodyFeatureExtractor.from_pretrained(self.checkpoint, **kwargs)

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def test_save_load_pretrained_default(self):
        tokenizer = self.get_tokenizer()
        feature_extractor = self.get_feature_extractor()

        processor = MusicgenMelodyProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        processor.save_pretrained(self.tmpdirname)
        processor = MusicgenMelodyProcessor.from_pretrained(self.tmpdirname)

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertIsInstance(processor.tokenizer, T5TokenizerFast)

        self.assertEqual(processor.feature_extractor.to_json_string(), feature_extractor.to_json_string())
        self.assertIsInstance(processor.feature_extractor, MusicgenMelodyFeatureExtractor)

    def test_save_load_pretrained_additional_features(self):
        processor = MusicgenMelodyProcessor(
            tokenizer=self.get_tokenizer(), feature_extractor=self.get_feature_extractor()
        )
        processor.save_pretrained(self.tmpdirname)

        tokenizer_add_kwargs = self.get_tokenizer(bos_token="(BOS)", eos_token="(EOS)")
        feature_extractor_add_kwargs = self.get_feature_extractor(do_normalize=False, padding_value=1.0)

        processor = MusicgenMelodyProcessor.from_pretrained(
            self.tmpdirname, bos_token="(BOS)", eos_token="(EOS)", do_normalize=False, padding_value=1.0
        )

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())
        self.assertIsInstance(processor.tokenizer, T5TokenizerFast)

        self.assertEqual(processor.feature_extractor.to_json_string(), feature_extractor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.feature_extractor, MusicgenMelodyFeatureExtractor)

    def test_feature_extractor(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = MusicgenMelodyProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        raw_speech = floats_list((3, 1000))

        input_feat_extract = feature_extractor(raw_speech, return_tensors="np")
        input_processor = processor(raw_speech, return_tensors="np")

        for key in input_feat_extract:
            self.assertAlmostEqual(input_feat_extract[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_tokenizer(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = MusicgenMelodyProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        input_str = "This is a test string"

        encoded_processor = processor(text=input_str)

        encoded_tok = tokenizer(input_str)

        for key in encoded_tok:
            self.assertListEqual(encoded_tok[key], encoded_processor[key])

    def test_tokenizer_decode(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = MusicgenMelodyProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        predicted_ids = [[1, 4, 5, 8, 1, 0, 8], [3, 4, 3, 1, 1, 8, 9]]

        decoded_processor = processor.batch_decode(sequences=predicted_ids)
        decoded_tok = tokenizer.batch_decode(predicted_ids)

        self.assertListEqual(decoded_tok, decoded_processor)

    # Ignore copy
    def test_decode_audio(self):
        feature_extractor = self.get_feature_extractor(padding_side="left")
        tokenizer = self.get_tokenizer()

        processor = MusicgenMelodyProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        attention_mask = np.zeros((3, 20))
        attention_mask[0, -5:] = 1
        attention_mask[1, -20:] = 1
        attention_mask[2, -10:] = 1

        generated_speech = np.asarray(floats_list((3, 20)))[:, None, :]
        decoded_audios = processor.batch_decode(generated_speech, attention_mask=attention_mask)

        self.assertIsInstance(decoded_audios, list)

        for audio in decoded_audios:
            self.assertIsInstance(audio, np.ndarray)

        self.assertTrue(decoded_audios[0].shape == (1, 5))
        self.assertTrue(decoded_audios[1].shape == (1, 20))
        self.assertTrue(decoded_audios[2].shape == (1, 10))
