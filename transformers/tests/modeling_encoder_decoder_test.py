# coding=utf-8
# Copyright 2018 The Hugging Face Inc. Team
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

import logging
import unittest
import pytest

from transformers import is_torch_available

if is_torch_available():
    from transformers import BertModel, BertForMaskedLM, Model2Model
    from transformers.modeling_bert import BERT_PRETRAINED_MODEL_ARCHIVE_MAP
else:
    pytestmark = pytest.mark.skip("Require Torch")


class EncoderDecoderModelTest(unittest.TestCase):
    @pytest.mark.slow
    def test_model2model_from_pretrained(self):
        logging.basicConfig(level=logging.INFO)
        for model_name in list(BERT_PRETRAINED_MODEL_ARCHIVE_MAP.keys())[:1]:
            model = Model2Model.from_pretrained(model_name)
            self.assertIsInstance(model.encoder, BertModel)
            self.assertIsInstance(model.decoder, BertForMaskedLM)
            self.assertEqual(model.decoder.config.is_decoder, True)
            self.assertEqual(model.encoder.config.is_decoder, False)

    def test_model2model_from_pretrained_not_bert(self):
        logging.basicConfig(level=logging.INFO)
        with self.assertRaises(ValueError):
            _ = Model2Model.from_pretrained('roberta')

        with self.assertRaises(ValueError):
            _ = Model2Model.from_pretrained('distilbert')

        with self.assertRaises(ValueError):
            _ = Model2Model.from_pretrained('does-not-exist')


if __name__ == "__main__":
    unittest.main()
