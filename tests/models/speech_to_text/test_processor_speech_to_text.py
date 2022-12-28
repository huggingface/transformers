# Copyright 2021 The HuggingFace Team. All rights reserved.
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
from pathlib import Path
from shutil import copyfile

from transformers import Speech2TextTokenizer, is_speech_available
from transformers.models.speech_to_text.tokenization_speech_to_text import VOCAB_FILES_NAMES, save_json
from transformers.testing_utils import get_tests_dir, require_sentencepiece, require_torch, require_torchaudio
from transformers.utils import FEATURE_EXTRACTOR_NAME

from .test_feature_extraction_speech_to_text import floats_list


if is_speech_available():
    from transformers import Speech2TextFeatureExtractor, Speech2TextProcessor


SAMPLE_SP = get_tests_dir("fixtures/test_sentencepiece.model")


@require_torch
@require_torchaudio
@require_sentencepiece
class Speech2TextProcessorTest(unittest.TestCase):
    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()

        vocab = ["<s>", "<pad>", "</s>", "<unk>", "▁This", "▁is", "▁a", "▁t", "est"]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        save_dir = Path(self.tmpdirname)
        save_json(vocab_tokens, save_dir / VOCAB_FILES_NAMES["vocab_file"])
        if not (save_dir / VOCAB_FILES_NAMES["spm_file"]).exists():
            copyfile(SAMPLE_SP, save_dir / VOCAB_FILES_NAMES["spm_file"])

        tokenizer = Speech2TextTokenizer.from_pretrained(self.tmpdirname)
        tokenizer.save_pretrained(self.tmpdirname)

        feature_extractor_map = {
            "feature_size": 24,
            "num_mel_bins": 24,
            "padding_value": 0.0,
            "sampling_rate": 16000,
            "return_attention_mask": False,
            "do_normalize": True,
        }
        save_json(feature_extractor_map, save_dir / FEATURE_EXTRACTOR_NAME)

    def get_tokenizer(self, **kwargs):
        return Speech2TextTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_feature_extractor(self, **kwargs):
        return Speech2TextFeatureExtractor.from_pretrained(self.tmpdirname, **kwargs)

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def test_save_load_pretrained_default(self):
        tokenizer = self.get_tokenizer()
        feature_extractor = self.get_feature_extractor()

        processor = Speech2TextProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        processor.save_pretrained(self.tmpdirname)
        processor = Speech2TextProcessor.from_pretrained(self.tmpdirname)

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertIsInstance(processor.tokenizer, Speech2TextTokenizer)

        self.assertEqual(processor.feature_extractor.to_json_string(), feature_extractor.to_json_string())
        self.assertIsInstance(processor.feature_extractor, Speech2TextFeatureExtractor)

    def test_save_load_pretrained_additional_features(self):
        processor = Speech2TextProcessor(
            tokenizer=self.get_tokenizer(), feature_extractor=self.get_feature_extractor()
        )
        processor.save_pretrained(self.tmpdirname)

        tokenizer_add_kwargs = self.get_tokenizer(bos_token="(BOS)", eos_token="(EOS)")
        feature_extractor_add_kwargs = self.get_feature_extractor(do_normalize=False, padding_value=1.0)

        processor = Speech2TextProcessor.from_pretrained(
            self.tmpdirname, bos_token="(BOS)", eos_token="(EOS)", do_normalize=False, padding_value=1.0
        )

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())
        self.assertIsInstance(processor.tokenizer, Speech2TextTokenizer)

        self.assertEqual(processor.feature_extractor.to_json_string(), feature_extractor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.feature_extractor, Speech2TextFeatureExtractor)

    def test_feature_extractor(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = Speech2TextProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        raw_speech = floats_list((3, 1000))

        input_feat_extract = feature_extractor(raw_speech, return_tensors="np")
        input_processor = processor(raw_speech, return_tensors="np")

        for key in input_feat_extract.keys():
            self.assertAlmostEqual(input_feat_extract[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_tokenizer(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = Speech2TextProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        input_str = "This is a test string"

        encoded_processor = processor(text=input_str)

        encoded_tok = tokenizer(input_str)

        for key in encoded_tok.keys():
            self.assertListEqual(encoded_tok[key], encoded_processor[key])

    def test_tokenizer_decode(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = Speech2TextProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        predicted_ids = [[1, 4, 5, 8, 1, 0, 8], [3, 4, 3, 1, 1, 8, 9]]

        decoded_processor = processor.batch_decode(predicted_ids)
        decoded_tok = tokenizer.batch_decode(predicted_ids)

        self.assertListEqual(decoded_tok, decoded_processor)

    def test_model_input_names(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = Speech2TextProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        self.assertListEqual(
            processor.model_input_names,
            feature_extractor.model_input_names,
            msg="`processor` and `feature_extractor` model input names do not match",
        )
