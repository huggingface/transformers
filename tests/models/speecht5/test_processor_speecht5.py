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

import json
import os
import shutil
import tempfile
import unittest

from transformers import is_speech_available, is_torch_available
from transformers.models.speecht5 import SpeechT5Tokenizer
from transformers.testing_utils import get_tests_dir, require_torch, require_torchaudio
from transformers.utils import FEATURE_EXTRACTOR_NAME


if is_speech_available() and is_torch_available():
    from transformers import SpeechT5FeatureExtractor, SpeechT5Processor

    from .test_feature_extraction_speecht5 import floats_list


SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece_bpe_char.model")


@require_torch
@require_torchaudio
class SpeechT5ProcessorTest(unittest.TestCase):
    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()

        tokenizer = SpeechT5Tokenizer(SAMPLE_VOCAB)
        tokenizer.save_pretrained(self.tmpdirname)

        feature_extractor_map = {
            "feature_size": 1,
            "padding_value": 0.0,
            "sampling_rate": 16000,
            "do_normalize": False,
            "num_mel_bins": 80,
            "hop_length": 16,
            "win_length": 64,
            "win_function": "hann_window",
            "frame_signal_scale": 1.0,
            "fmin": 80,
            "fmax": 7600,
            "mel_floor": 1e-10,
            "reduction_factor": 2,
            "return_attention_mask": True,
        }

        self.feature_extraction_file = os.path.join(self.tmpdirname, FEATURE_EXTRACTOR_NAME)
        with open(self.feature_extraction_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(feature_extractor_map) + "\n")

    def get_tokenizer(self, **kwargs):
        return SpeechT5Tokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_feature_extractor(self, **kwargs):
        return SpeechT5FeatureExtractor.from_pretrained(self.tmpdirname, **kwargs)

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def test_save_load_pretrained_default(self):
        tokenizer = self.get_tokenizer()
        feature_extractor = self.get_feature_extractor()

        processor = SpeechT5Processor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        processor.save_pretrained(self.tmpdirname)
        processor = SpeechT5Processor.from_pretrained(self.tmpdirname)

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertIsInstance(processor.tokenizer, SpeechT5Tokenizer)

        self.assertEqual(processor.feature_extractor.to_json_string(), feature_extractor.to_json_string())
        self.assertIsInstance(processor.feature_extractor, SpeechT5FeatureExtractor)

    def test_save_load_pretrained_additional_features(self):
        processor = SpeechT5Processor(tokenizer=self.get_tokenizer(), feature_extractor=self.get_feature_extractor())
        processor.save_pretrained(self.tmpdirname)

        tokenizer_add_kwargs = self.get_tokenizer(bos_token="(BOS)", eos_token="(EOS)")
        feature_extractor_add_kwargs = self.get_feature_extractor(do_normalize=False, padding_value=1.0)

        processor = SpeechT5Processor.from_pretrained(
            self.tmpdirname, bos_token="(BOS)", eos_token="(EOS)", do_normalize=False, padding_value=1.0
        )

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())
        self.assertIsInstance(processor.tokenizer, SpeechT5Tokenizer)

        self.assertEqual(processor.feature_extractor.to_json_string(), feature_extractor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.feature_extractor, SpeechT5FeatureExtractor)

    def test_feature_extractor(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = SpeechT5Processor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        raw_speech = floats_list((3, 1000))

        input_feat_extract = feature_extractor(audio=raw_speech, return_tensors="np")
        input_processor = processor(audio=raw_speech, return_tensors="np")

        for key in input_feat_extract.keys():
            self.assertAlmostEqual(input_feat_extract[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_feature_extractor_target(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = SpeechT5Processor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        raw_speech = floats_list((3, 1000))

        input_feat_extract = feature_extractor(audio_target=raw_speech, return_tensors="np")
        input_processor = processor(audio_target=raw_speech, return_tensors="np")

        for key in input_feat_extract.keys():
            self.assertAlmostEqual(input_feat_extract[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_tokenizer(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = SpeechT5Processor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        input_str = "This is a test string"

        encoded_processor = processor(text=input_str)
        encoded_tok = tokenizer(input_str)

        for key in encoded_tok.keys():
            self.assertListEqual(encoded_tok[key], encoded_processor[key])

    def test_tokenizer_target(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = SpeechT5Processor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        input_str = "This is a test string"

        encoded_processor = processor(text_target=input_str)
        encoded_tok = tokenizer(input_str)

        for key in encoded_tok.keys():
            self.assertListEqual(encoded_tok[key], encoded_processor[key])

    def test_tokenizer_decode(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = SpeechT5Processor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        predicted_ids = [[1, 4, 5, 8, 1, 0, 8], [3, 4, 3, 1, 1, 8, 9]]

        decoded_processor = processor.batch_decode(predicted_ids)
        decoded_tok = tokenizer.batch_decode(predicted_ids)

        self.assertListEqual(decoded_tok, decoded_processor)

    def test_model_input_names(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = SpeechT5Processor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        self.assertListEqual(
            processor.model_input_names,
            feature_extractor.model_input_names,
            msg="`processor` and `feature_extractor` model input names do not match",
        )
