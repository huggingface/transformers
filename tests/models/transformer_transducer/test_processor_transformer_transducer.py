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

from transformers.testing_utils import require_tokenizers, require_torch
from transformers.utils import is_torchaudio_available


if is_torchaudio_available():
    from transformers import (
        TransformerTransducerFeatureExtractor,
        TransformerTransducerProcessor,
        TransformerTransducerTokenizer,
    )


TRANSCRIBE = 50358
NOTIMESTAMPS = 50362


@require_torch
@require_tokenizers
class TransformerTransducerProcessorTest(unittest.TestCase):
    def setup(self):
        self.tmpdirname = tempfile.mkdtemp()

    def get_tokenizer(self, **kwargs_init):
        kwargs = self.add_kwargs_tokens_map.copy()
        kwargs.update(kwargs_init)
        return TransformerTransducerTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_feature_extractor(self, **kwargs):
        return TransformerTransducerFeatureExtractor.from_pretrained(self.tmpdirname, **kwargs)

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def test_save_load_pretrained_default(self):
        tokenizer = self.get_tokenizer()
        extractor = self.get_feature_extractor()

        processor = TransformerTransducerProcessor(tokenizer=tokenizer, feature_extractor=extractor)

        # [NOTE]: test for save processor
        processor.save_pretrained(self.tmpdirname)

        # [NOTE]: test for load processor
        processor = TransformerTransducerProcessor.from_pretrained(self.tmpdirname)

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertIsInstance(processor.tokenizer, TransformerTransducerTokenizer)

        self.assertEqual(processor.feature_extractor.to_json_string(), extractor.to_json_string())
        self.assertIsInstance(processor.feature_extractor, TransformerTransducerFeatureExtractor)

    def test_save_load_pretrained_additional_features(self):
        # ??? 이 부분은 더 확인할 필요가 있음.
        tokenizer = self.get_tokenizer()
        extractor = self.get_feature_extractor()

        processor = TransformerTransducerProcessor(tokenizer=tokenizer, feature_extractor=extractor)
        processor.save_pretrained(self.tmpdirname)

        tokenizer_add_kwargs = self.get_tokenizer(bos_token="<bos>", eos_token="<eos>")
        feature_extractor_add_kwargs = self.get_feature_extractor(do_normalize=False, padding_value=1.0)

        processor = TransformerTransducerProcessor.from_pretrained(
            self.tmpdirname,
            bos_token="<boe>",
            eos_token="<eos>",
            do_normalize=False,
            padding_value=1.0,
        )

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())
        self.assertIsInstance(processor.tokenizer, TransformerTransducerTokenizer)

        self.assertEqual(processor.feature_extractor.to_json_string(), feature_extractor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.feature_extractor, TransformerTransducerFeatureExtractor)

    def test_save_load_auto_pretrained_default(self):
        pass

    def test_save_load_auto_pretrained_additional_features(self):
        pass

    def test_tokenizer_encode(self):
        pass

    def test_tokenizer_decode(self):
        pass

    def test_extractor(self):
        pass

    def test_model_input_names(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = TransformerTransducerProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        self.assertListEqual(
            processor.model_input_names,
            feature_extractor.model_input_names,
            msg="`processor` and `feature_extractor` model input names do not match",
        )
