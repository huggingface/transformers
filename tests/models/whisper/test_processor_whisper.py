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

import shutil
import tempfile
import unittest

import pytest

from transformers import WhisperTokenizer, is_speech_available
from transformers.testing_utils import require_sentencepiece, require_torch, require_torchaudio

from .test_feature_extraction_whisper import floats_list


if is_speech_available():
    from transformers import WhisperFeatureExtractor, WhisperProcessor


TRANSCRIBE = 50358
NOTIMESTAMPS = 50362


@require_torch
@require_torchaudio
@require_sentencepiece
class WhisperProcessorTest(unittest.TestCase):
    def setUp(self):
        self.checkpoint = "openai/whisper-small.en"
        self.tmpdirname = tempfile.mkdtemp()

    def get_tokenizer(self, **kwargs):
        return WhisperTokenizer.from_pretrained(self.checkpoint, **kwargs)

    def get_feature_extractor(self, **kwargs):
        return WhisperFeatureExtractor.from_pretrained(self.checkpoint, **kwargs)

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def test_save_load_pretrained_default(self):
        tokenizer = self.get_tokenizer()
        feature_extractor = self.get_feature_extractor()

        processor = WhisperProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        processor.save_pretrained(self.tmpdirname)
        processor = WhisperProcessor.from_pretrained(self.tmpdirname)

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertIsInstance(processor.tokenizer, WhisperTokenizer)

        self.assertEqual(processor.feature_extractor.to_json_string(), feature_extractor.to_json_string())
        self.assertIsInstance(processor.feature_extractor, WhisperFeatureExtractor)

    def test_save_load_pretrained_additional_features(self):
        processor = WhisperProcessor(tokenizer=self.get_tokenizer(), feature_extractor=self.get_feature_extractor())
        processor.save_pretrained(self.tmpdirname)

        tokenizer_add_kwargs = self.get_tokenizer(bos_token="(BOS)", eos_token="(EOS)")
        feature_extractor_add_kwargs = self.get_feature_extractor(do_normalize=False, padding_value=1.0)

        processor = WhisperProcessor.from_pretrained(
            self.tmpdirname, bos_token="(BOS)", eos_token="(EOS)", do_normalize=False, padding_value=1.0
        )

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())
        self.assertIsInstance(processor.tokenizer, WhisperTokenizer)

        self.assertEqual(processor.feature_extractor.to_json_string(), feature_extractor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.feature_extractor, WhisperFeatureExtractor)

    def test_feature_extractor(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = WhisperProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        raw_speech = floats_list((3, 1000))

        input_feat_extract = feature_extractor(raw_speech, return_tensors="np")
        input_processor = processor(raw_speech, return_tensors="np")

        for key in input_feat_extract.keys():
            self.assertAlmostEqual(input_feat_extract[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_tokenizer(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = WhisperProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        input_str = "This is a test string"

        encoded_processor = processor(text=input_str)

        encoded_tok = tokenizer(input_str)

        for key in encoded_tok.keys():
            self.assertListEqual(encoded_tok[key], encoded_processor[key])

    def test_tokenizer_decode(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = WhisperProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        predicted_ids = [[1, 4, 5, 8, 1, 0, 8], [3, 4, 3, 1, 1, 8, 9]]

        decoded_processor = processor.batch_decode(predicted_ids)
        decoded_tok = tokenizer.batch_decode(predicted_ids)

        self.assertListEqual(decoded_tok, decoded_processor)

    def test_model_input_names(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = WhisperProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        self.assertListEqual(
            processor.model_input_names,
            feature_extractor.model_input_names,
            msg="`processor` and `feature_extractor` model input names do not match",
        )

    def test_get_decoder_prompt_ids(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = WhisperProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)
        forced_decoder_ids = processor.get_decoder_prompt_ids(task="transcribe", no_timestamps=True)

        self.assertIsInstance(forced_decoder_ids, list)
        for ids in forced_decoder_ids:
            self.assertIsInstance(ids, (list, tuple))

        expected_ids = [TRANSCRIBE, NOTIMESTAMPS]
        self.assertListEqual([ids[-1] for ids in forced_decoder_ids], expected_ids)

    def test_get_prompt_ids(self):
        processor = WhisperProcessor(tokenizer=self.get_tokenizer(), feature_extractor=self.get_feature_extractor())
        prompt_ids = processor.get_prompt_ids("Mr. Quilter")
        decoded_prompt = processor.tokenizer.decode(prompt_ids)

        self.assertListEqual(prompt_ids.tolist(), [50360, 1770, 13, 2264, 346, 353])
        self.assertEqual(decoded_prompt, "<|startofprev|> Mr. Quilter")

    def test_empty_get_prompt_ids(self):
        processor = WhisperProcessor(tokenizer=self.get_tokenizer(), feature_extractor=self.get_feature_extractor())
        prompt_ids = processor.get_prompt_ids("")
        decoded_prompt = processor.tokenizer.decode(prompt_ids)

        self.assertListEqual(prompt_ids.tolist(), [50360, 220])
        self.assertEqual(decoded_prompt, "<|startofprev|> ")

    def test_get_prompt_ids_with_special_tokens(self):
        processor = WhisperProcessor(tokenizer=self.get_tokenizer(), feature_extractor=self.get_feature_extractor())

        def _test_prompt_error_raised_helper(prompt, special_token):
            with pytest.raises(ValueError) as excinfo:
                processor.get_prompt_ids(prompt)
            expected = f"Encountered text in the prompt corresponding to disallowed special token: {special_token}."
            self.assertEqual(expected, str(excinfo.value))

        _test_prompt_error_raised_helper("<|startofprev|> test", "<|startofprev|>")
        _test_prompt_error_raised_helper("test <|notimestamps|>", "<|notimestamps|>")
        _test_prompt_error_raised_helper("test <|zh|> test <|transcribe|>", "<|zh|>")
