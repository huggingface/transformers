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


import logging
import unittest

from transformers import is_tf_available

from .utils import DUMMY_UNKWOWN_IDENTIFIER, SMALL_MODEL_IDENTIFIER, require_tf, slow


if is_tf_available():
    from transformers import (
        AutoConfig,
        BertConfig,
        TFAutoModel,
        TFBertModel,
        TFAutoModelForPreTraining,
        TFBertForPreTraining,
        TFAutoModelWithLMHead,
        TFBertForMaskedLM,
        TFRobertaForMaskedLM,
        TFAutoModelForSequenceClassification,
        TFBertForSequenceClassification,
        TFAutoModelForQuestionAnswering,
        TFBertForQuestionAnswering,
    )


@require_tf
class TFAutoModelTest(unittest.TestCase):
    @slow
    def test_model_from_pretrained(self):
        import h5py

        self.assertTrue(h5py.version.hdf5_version.startswith("1.10"))

        logging.basicConfig(level=logging.INFO)
        # for model_name in list(TF_BERT_PRETRAINED_MODEL_ARCHIVE_MAP.keys())[:1]:
        for model_name in ["bert-base-uncased"]:
            config = AutoConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, BertConfig)

            model = TFAutoModel.from_pretrained(model_name)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, TFBertModel)

    @slow
    def test_model_for_pretraining_from_pretrained(self):
        import h5py

        self.assertTrue(h5py.version.hdf5_version.startswith("1.10"))

        logging.basicConfig(level=logging.INFO)
        # for model_name in list(TF_BERT_PRETRAINED_MODEL_ARCHIVE_MAP.keys())[:1]:
        for model_name in ["bert-base-uncased"]:
            config = AutoConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, BertConfig)

            model = TFAutoModelForPreTraining.from_pretrained(model_name)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, TFBertForPreTraining)

    @slow
    def test_lmhead_model_from_pretrained(self):
        logging.basicConfig(level=logging.INFO)
        # for model_name in list(TF_BERT_PRETRAINED_MODEL_ARCHIVE_MAP.keys())[:1]:
        for model_name in ["bert-base-uncased"]:
            config = AutoConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, BertConfig)

            model = TFAutoModelWithLMHead.from_pretrained(model_name)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, TFBertForMaskedLM)

    @slow
    def test_sequence_classification_model_from_pretrained(self):
        logging.basicConfig(level=logging.INFO)
        # for model_name in list(TF_BERT_PRETRAINED_MODEL_ARCHIVE_MAP.keys())[:1]:
        for model_name in ["bert-base-uncased"]:
            config = AutoConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, BertConfig)

            model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, TFBertForSequenceClassification)

    @slow
    def test_question_answering_model_from_pretrained(self):
        logging.basicConfig(level=logging.INFO)
        # for model_name in list(TF_BERT_PRETRAINED_MODEL_ARCHIVE_MAP.keys())[:1]:
        for model_name in ["bert-base-uncased"]:
            config = AutoConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, BertConfig)

            model = TFAutoModelForQuestionAnswering.from_pretrained(model_name)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, TFBertForQuestionAnswering)

    def test_from_pretrained_identifier(self):
        logging.basicConfig(level=logging.INFO)
        model = TFAutoModelWithLMHead.from_pretrained(SMALL_MODEL_IDENTIFIER)
        self.assertIsInstance(model, TFBertForMaskedLM)
        self.assertEqual(model.num_parameters(), 14830)
        self.assertEqual(model.num_parameters(only_trainable=True), 14830)

    def test_from_identifier_from_model_type(self):
        logging.basicConfig(level=logging.INFO)
        model = TFAutoModelWithLMHead.from_pretrained(DUMMY_UNKWOWN_IDENTIFIER)
        self.assertIsInstance(model, TFRobertaForMaskedLM)
        self.assertEqual(model.num_parameters(), 14830)
        self.assertEqual(model.num_parameters(only_trainable=True), 14830)
