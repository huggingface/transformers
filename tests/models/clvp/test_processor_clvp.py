# Copyright 2023 The HuggingFace Team. All rights reserved.
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


import gc
import shutil
import tempfile
import unittest

from transformers import ClvpFeatureExtractor, ClvpProcessor, ClvpTokenizer
from transformers.testing_utils import require_torch

from .test_feature_extraction_clvp import floats_list


@require_torch
class ClvpProcessorTest(unittest.TestCase):
    def setUp(self):
        self.checkpoint = "susnato/clvp_dev"
        self.tmpdirname = tempfile.mkdtemp()

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmpdirname)
        gc.collect()

    # Copied from transformers.tests.models.whisper.test_processor_whisper.WhisperProcessorTest.get_tokenizer with Whisper->Clvp
    def get_tokenizer(self, **kwargs):
        return ClvpTokenizer.from_pretrained(self.checkpoint, **kwargs)

    # Copied from transformers.tests.models.whisper.test_processor_whisper.WhisperProcessorTest.get_feature_extractor with Whisper->Clvp
    def get_feature_extractor(self, **kwargs):
        return ClvpFeatureExtractor.from_pretrained(self.checkpoint, **kwargs)

    # Copied from transformers.tests.models.whisper.test_processor_whisper.WhisperProcessorTest.test_save_load_pretrained_default with Whisper->Clvp
    def test_save_load_pretrained_default(self):
        tokenizer = self.get_tokenizer()
        feature_extractor = self.get_feature_extractor()

        processor = ClvpProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        processor.save_pretrained(self.tmpdirname)
        processor = ClvpProcessor.from_pretrained(self.tmpdirname)

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertIsInstance(processor.tokenizer, ClvpTokenizer)

        self.assertEqual(processor.feature_extractor.to_json_string(), feature_extractor.to_json_string())
        self.assertIsInstance(processor.feature_extractor, ClvpFeatureExtractor)

    # Copied from transformers.tests.models.whisper.test_processor_whisper.WhisperProcessorTest.test_feature_extractor with Whisper->Clvp,processor(raw_speech->processor(raw_speech=raw_speech
    def test_feature_extractor(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = ClvpProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        raw_speech = floats_list((3, 1000))

        input_feat_extract = feature_extractor(raw_speech, return_tensors="np")
        input_processor = processor(raw_speech=raw_speech, return_tensors="np")

        for key in input_feat_extract.keys():
            self.assertAlmostEqual(input_feat_extract[key].sum(), input_processor[key].sum(), delta=1e-2)

    # Copied from transformers.tests.models.whisper.test_processor_whisper.WhisperProcessorTest.test_tokenizer with Whisper->Clvp
    def test_tokenizer(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = ClvpProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        input_str = "This is a test string"

        encoded_processor = processor(text=input_str)

        encoded_tok = tokenizer(input_str)

        for key in encoded_tok.keys():
            self.assertListEqual(encoded_tok[key], encoded_processor[key])

    # Copied from transformers.tests.models.whisper.test_processor_whisper.WhisperProcessorTest.test_tokenizer_decode with Whisper->Clvp
    def test_tokenizer_decode(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = ClvpProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        predicted_ids = [[1, 4, 5, 8, 1, 0, 8], [3, 4, 3, 1, 1, 8, 9]]

        decoded_processor = processor.batch_decode(predicted_ids)
        decoded_tok = tokenizer.batch_decode(predicted_ids)

        self.assertListEqual(decoded_tok, decoded_processor)

    def test_save_load_pretrained_additional_features(self):
        processor = ClvpProcessor(tokenizer=self.get_tokenizer(), feature_extractor=self.get_feature_extractor())
        processor.save_pretrained(self.tmpdirname)

        tokenizer_add_kwargs = self.get_tokenizer(pad_token="(PAD)")
        feature_extractor_add_kwargs = self.get_feature_extractor(sampling_rate=16000)

        processor = ClvpProcessor.from_pretrained(
            self.tmpdirname,
            pad_token="(PAD)",
            sampling_rate=16000,
        )

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())
        self.assertIsInstance(processor.tokenizer, ClvpTokenizer)

        self.assertEqual(processor.feature_extractor.to_json_string(), feature_extractor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.feature_extractor, ClvpFeatureExtractor)

    def test_model_input_names(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = ClvpProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        self.assertListEqual(
            sorted(processor.model_input_names),
            sorted(set(feature_extractor.model_input_names + tokenizer.model_input_names)),
            msg="`processor` and `feature_extractor` model input names do not match",
        )
