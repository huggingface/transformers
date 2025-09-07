# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from transformers import ParakeetCTCTokenizer, ParakeetFeatureExtractor, ParakeetProcessor
from transformers.testing_utils import require_torch, require_torchaudio

from ...test_processing_common import ProcessorTesterMixin
from .test_feature_extraction_parakeet import floats_list


@require_torch
@require_torchaudio
class ParakeetProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = ParakeetProcessor
    audio_input_name = "input_values"
    text_input_name = "labels"

    @classmethod
    def setUpClass(cls):
        cls.checkpoint = "bezzam/parakeet-ctc-1.1b-hf"
        cls.tmpdirname = tempfile.mkdtemp()
        processor = ParakeetProcessor.from_pretrained(cls.checkpoint)
        processor.save_pretrained(cls.tmpdirname)

    @classmethod
    def get_tokenizer(cls, **kwargs):
        return ParakeetCTCTokenizer.from_pretrained(cls.checkpoint, **kwargs)

    @classmethod
    def get_feature_extractor(cls, **kwargs):
        return ParakeetFeatureExtractor.from_pretrained(cls.checkpoint, **kwargs)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    def test_save_load_pretrained_default(self):
        tokenizer = self.get_tokenizer()
        feature_extractor = self.get_feature_extractor()

        processor = ParakeetProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        processor.save_pretrained(self.tmpdirname)
        processor = ParakeetProcessor.from_pretrained(self.tmpdirname)

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer.get_vocab())

        self.assertEqual(processor.feature_extractor.to_json_string(), feature_extractor.to_json_string())
        self.assertIsInstance(processor.feature_extractor, ParakeetFeatureExtractor)

    def test_save_load_pretrained_additional_features(self):
        processor = ParakeetProcessor(tokenizer=self.get_tokenizer(), feature_extractor=self.get_feature_extractor())
        processor.save_pretrained(self.tmpdirname)

        tokenizer_add_kwargs = self.get_tokenizer(bos_token="(BOS)", eos_token="(EOS)")
        feature_extractor_add_kwargs = self.get_feature_extractor(do_normalize=False, padding_value=1.0)

        processor = ParakeetProcessor.from_pretrained(
            self.tmpdirname, bos_token="(BOS)", eos_token="(EOS)", do_normalize=False, padding_value=1.0
        )

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())

        self.assertEqual(processor.feature_extractor.to_json_string(), feature_extractor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.feature_extractor, ParakeetFeatureExtractor)

    def test_feature_extractor(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = ParakeetProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        raw_speech = floats_list((3, 1000))

        input_feat_extract = feature_extractor(raw_speech, return_tensors="np")
        input_processor = processor(raw_speech, return_tensors="np")

        for key in input_feat_extract:
            self.assertAlmostEqual(input_feat_extract[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_tokenizer(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = ParakeetProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        input_str = "This is a test string"

        encoded_processor = processor(text=input_str)

        encoded_tok = tokenizer(input_str)

        for key in encoded_tok:
            self.assertListEqual(encoded_tok[key], encoded_processor[key])

    def test_tokenizer_decode(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = ParakeetProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        predicted_ids = [[1, 4, 5, 8, 1, 0, 8], [3, 4, 3, 1, 1, 8, 9]]

        # without grouping
        decoded_processor = processor.batch_decode(predicted_ids, group_tokens=False)
        decoded_tok = tokenizer.batch_decode(predicted_ids, group_tokens=False)
        self.assertListEqual(decoded_tok, decoded_processor)

        # with grouping
        decoded_processor = processor.batch_decode(predicted_ids, group_tokens=True)
        decoded_tok = tokenizer.batch_decode(predicted_ids, group_tokens=True)
        self.assertListEqual(decoded_tok, decoded_processor)

    def test_model_input_names(self):
        processor = self.get_processor()

        text = "lower newer"
        audio_inputs = self.prepare_audio_inputs()

        inputs = processor(text=text, audio=audio_inputs, return_attention_mask=True, return_tensors="pt")
        self.assertSetEqual(set(inputs.keys()), set(processor.model_input_names))
