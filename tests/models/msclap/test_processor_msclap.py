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

import random
import shutil
import tempfile
import unittest

from transformers import ClapFeatureExtractor, GPT2Tokenizer, GPT2TokenizerFast, MSClapProcessor
from transformers.testing_utils import require_sentencepiece, require_torchaudio


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


@require_torchaudio
@require_sentencepiece
# Copied from tests.models.clap.test_processor_clap.ClapProcessorTest with ClapProcessor->MSClapProcessor, laion/clap-htsat-fused->kamilakesbi/ms_clap, Roberta->GPT2
class MSClapProcessorTest(unittest.TestCase):
    def setUp(self):
        self.checkpoint = "laion/clap-htsat-unfused"
        self.tmpdirname = tempfile.mkdtemp()

    def get_tokenizer(self, **kwargs):
        return GPT2Tokenizer.from_pretrained(self.checkpoint, **kwargs)

    def get_feature_extractor(self, **kwargs):
        return ClapFeatureExtractor.from_pretrained(self.checkpoint, **kwargs)

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def test_save_load_pretrained_default(self):
        tokenizer = self.get_tokenizer()
        feature_extractor = self.get_feature_extractor()

        processor = MSClapProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        processor.save_pretrained(self.tmpdirname)
        processor = MSClapProcessor.from_pretrained(self.tmpdirname)

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertIsInstance(processor.tokenizer, GPT2TokenizerFast)

        self.assertEqual(processor.feature_extractor.to_json_string(), feature_extractor.to_json_string())
        self.assertIsInstance(processor.feature_extractor, ClapFeatureExtractor)

    def test_save_load_pretrained_additional_features(self):
        processor = MSClapProcessor(tokenizer=self.get_tokenizer(), feature_extractor=self.get_feature_extractor())
        processor.save_pretrained(self.tmpdirname)

        tokenizer_add_kwargs = self.get_tokenizer(bos_token="(BOS)", eos_token="(EOS)")
        feature_extractor_add_kwargs = self.get_feature_extractor(do_normalize=False, padding_value=1.0)

        processor = MSClapProcessor.from_pretrained(
            self.tmpdirname, bos_token="(BOS)", eos_token="(EOS)", do_normalize=False, padding_value=1.0
        )

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())
        self.assertIsInstance(processor.tokenizer, GPT2TokenizerFast)

        self.assertEqual(processor.feature_extractor.to_json_string(), feature_extractor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.feature_extractor, ClapFeatureExtractor)

    def test_feature_extractor(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = MSClapProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        raw_speech = floats_list((3, 1000))

        input_feat_extract = feature_extractor(raw_speech, return_tensors="np")
        input_processor = processor(audios=raw_speech, return_tensors="np")

        for key in input_feat_extract.keys():
            self.assertAlmostEqual(input_feat_extract[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_tokenizer(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = MSClapProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        input_str = "This is a test string"

        encoded_processor = processor(text=input_str)

        encoded_tok = tokenizer(input_str)

        for key in encoded_tok.keys():
            self.assertListEqual(encoded_tok[key], encoded_processor[key])

    # Ignore copy
    def test_tokenizer_decode(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = MSClapProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        predicted_ids = [[40, 423, 257, 3797, 220, 0, 220], [40, 588, 616, 3290, 764, 220]]

        decoded_processor = processor.batch_decode(predicted_ids)
        decoded_tok = tokenizer.batch_decode(predicted_ids)

        self.assertListEqual(decoded_tok, decoded_processor)

    def test_model_input_names(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = MSClapProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        self.assertListEqual(
            processor.model_input_names[2:],
            feature_extractor.model_input_names,
            msg="`processor` and `feature_extractor` model input names do not match",
        )
