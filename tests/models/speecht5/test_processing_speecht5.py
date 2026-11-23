# Copyright 2022 The HuggingFace Team. All rights reserved.
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
"""Tests for the SpeechT5 processors."""

import shutil
import tempfile
import unittest

from transformers import is_speech_available, is_torch_available
from transformers.models.speecht5 import SpeechT5Tokenizer
from transformers.testing_utils import get_tests_dir, require_speech, require_torch


if is_speech_available() and is_torch_available():
    from transformers import SpeechT5AudioProcessor, SpeechT5Processor

    from ...test_processing_common import floats_list


SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece_bpe_char.model")


@require_torch
@require_speech
class SpeechT5ProcessorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdirname = tempfile.mkdtemp()

        tokenizer = SpeechT5Tokenizer(SAMPLE_VOCAB)
        tokenizer.save_pretrained(cls.tmpdirname)

        audio_processor_map = {
            "feature_size": 1,
            "padding_value": 0.0,
            "sampling_rate": 16000,
            "do_normalize": False,
            "num_mel_bins": 80,
            "hop_length": 16,
            "win_length": 64,
            "win_function": "hann_window",
            "fmin": 80,
            "fmax": 7600,
            "mel_floor": 1e-10,
            "reduction_factor": 2,
            "return_attention_mask": True,
        }

        audio_processor = SpeechT5AudioProcessor(**audio_processor_map)
        tokenizer = SpeechT5Tokenizer.from_pretrained(cls.tmpdirname)
        processor = SpeechT5Processor(tokenizer=tokenizer, audio_processor=audio_processor)
        processor.save_pretrained(cls.tmpdirname)

    def get_tokenizer(self, **kwargs):
        return SpeechT5Tokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_audio_processor(self, **kwargs):
        return SpeechT5AudioProcessor.from_pretrained(self.tmpdirname, **kwargs)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    def test_save_load_pretrained_default(self):
        tokenizer = self.get_tokenizer()
        audio_processor = self.get_audio_processor()

        processor = SpeechT5Processor(tokenizer=tokenizer, audio_processor=audio_processor)

        processor.save_pretrained(self.tmpdirname)
        processor = SpeechT5Processor.from_pretrained(self.tmpdirname)

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertIsInstance(processor.tokenizer, SpeechT5Tokenizer)

        self.assertEqual(processor.audio_processor.to_json_string(), audio_processor.to_json_string())
        self.assertIsInstance(processor.audio_processor, SpeechT5AudioProcessor)

    def test_save_load_pretrained_additional_features(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = SpeechT5Processor(tokenizer=self.get_tokenizer(), audio_processor=self.get_audio_processor())
            processor.save_pretrained(tmpdir)

            tokenizer_add_kwargs = SpeechT5Tokenizer.from_pretrained(tmpdir, bos_token="(BOS)", eos_token="(EOS)")
            audio_processor_add_kwargs = SpeechT5AudioProcessor.from_pretrained(
                tmpdir, do_normalize=False, padding_value=1.0
            )

            processor = SpeechT5Processor.from_pretrained(
                tmpdir, bos_token="(BOS)", eos_token="(EOS)", do_normalize=False, padding_value=1.0
            )

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())
        self.assertIsInstance(processor.tokenizer, SpeechT5Tokenizer)

        self.assertEqual(processor.audio_processor.to_json_string(), audio_processor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.audio_processor, SpeechT5AudioProcessor)

    def test_audio_processor(self):
        audio_processor = self.get_audio_processor()
        tokenizer = self.get_tokenizer()

        processor = SpeechT5Processor(tokenizer=tokenizer, audio_processor=audio_processor)

        raw_speech = floats_list((3, 1000))

        input_feat_extract = audio_processor(audio=raw_speech, return_tensors="np")
        input_processor = processor(audio=raw_speech, return_tensors="np")

        for key in input_feat_extract:
            self.assertAlmostEqual(input_feat_extract[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_audio_processor_target(self):
        audio_processor = self.get_audio_processor()
        tokenizer = self.get_tokenizer()

        processor = SpeechT5Processor(tokenizer=tokenizer, audio_processor=audio_processor)

        raw_speech = floats_list((3, 1000))

        input_feat_extract = audio_processor(audio_target=raw_speech, return_tensors="np")
        input_processor = processor(audio_target=raw_speech, return_tensors="np")

        for key in input_feat_extract:
            self.assertAlmostEqual(input_feat_extract[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_tokenizer(self):
        audio_processor = self.get_audio_processor()
        tokenizer = self.get_tokenizer()

        processor = SpeechT5Processor(tokenizer=tokenizer, audio_processor=audio_processor)

        input_str = "This is a test string"

        encoded_processor = processor(text=input_str)
        encoded_tok = tokenizer(input_str)

        for key in encoded_tok:
            self.assertListEqual(encoded_tok[key], encoded_processor[key])

    def test_tokenizer_target(self):
        audio_processor = self.get_audio_processor()
        tokenizer = self.get_tokenizer()

        processor = SpeechT5Processor(tokenizer=tokenizer, audio_processor=audio_processor)

        input_str = "This is a test string"

        encoded_processor = processor(text_target=input_str)
        encoded_tok = tokenizer(input_str)

        for key in encoded_tok:
            self.assertListEqual(encoded_tok[key], encoded_processor[key])

    def test_tokenizer_decode(self):
        audio_processor = self.get_audio_processor()
        tokenizer = self.get_tokenizer()

        processor = SpeechT5Processor(tokenizer=tokenizer, audio_processor=audio_processor)

        predicted_ids = [[1, 4, 5, 8, 1, 0, 8], [3, 4, 3, 1, 1, 8, 9]]

        decoded_processor = processor.batch_decode(predicted_ids)
        decoded_tok = tokenizer.batch_decode(predicted_ids)

        self.assertListEqual(decoded_tok, decoded_processor)
