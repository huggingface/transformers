# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import shutil
import pytest
import logging

from transformers import is_torch_available

if is_torch_available():
    from transformers import (AutoConfig, BertConfig,
                                    AutoModel, BertModel,
                                    AutoModelWithLMHead, BertForMaskedLM,
                                    AutoModelForSequenceClassification, BertForSequenceClassification,
                                    AutoModelForQuestionAnswering, BertForQuestionAnswering)
    from transformers.modeling_bert import BERT_PRETRAINED_MODEL_ARCHIVE_MAP

    from .modeling_common_test import (CommonTestCases, ids_tensor)
    from .configuration_common_test import ConfigTester
else:
    pytestmark = pytest.mark.skip("Require Torch")


class AutoModelTest(unittest.TestCase):
    def test_model_from_pretrained(self):
        logging.basicConfig(level=logging.INFO)
        for model_name in list(BERT_PRETRAINED_MODEL_ARCHIVE_MAP.keys())[:1]:
            config = AutoConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, BertConfig)

            model = AutoModel.from_pretrained(model_name)
            model, loading_info = AutoModel.from_pretrained(model_name, output_loading_info=True)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, BertModel)
            for value in loading_info.values():
                self.assertEqual(len(value), 0)

    def test_lmhead_model_from_pretrained(self):
        logging.basicConfig(level=logging.INFO)
        for model_name in list(BERT_PRETRAINED_MODEL_ARCHIVE_MAP.keys())[:1]:
            config = AutoConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, BertConfig)

            model = AutoModelWithLMHead.from_pretrained(model_name)
            model, loading_info = AutoModelWithLMHead.from_pretrained(model_name, output_loading_info=True)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, BertForMaskedLM)

    def test_sequence_classification_model_from_pretrained(self):
        logging.basicConfig(level=logging.INFO)
        for model_name in list(BERT_PRETRAINED_MODEL_ARCHIVE_MAP.keys())[:1]:
            config = AutoConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, BertConfig)

            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            model, loading_info = AutoModelForSequenceClassification.from_pretrained(model_name, output_loading_info=True)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, BertForSequenceClassification)

    def test_question_answering_model_from_pretrained(self):
        logging.basicConfig(level=logging.INFO)
        for model_name in list(BERT_PRETRAINED_MODEL_ARCHIVE_MAP.keys())[:1]:
            config = AutoConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, BertConfig)

            model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            model, loading_info = AutoModelForQuestionAnswering.from_pretrained(model_name, output_loading_info=True)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, BertForQuestionAnswering)


if __name__ == "__main__":
    unittest.main()
