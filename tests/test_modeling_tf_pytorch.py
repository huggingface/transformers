# coding=utf-8
# Copyright 2020 The HuggingFace Team. All rights reserved.
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


import unittest

from transformers import is_tf_available, is_torch_available
from transformers.testing_utils import DUMMY_UNKWOWN_IDENTIFIER, SMALL_MODEL_IDENTIFIER, is_pt_tf_cross_test, slow


if is_tf_available():
    from transformers import (
        AutoConfig,
        BertConfig,
        GPT2Config,
        T5Config,
        TFAutoModel,
        TFAutoModelForCausalLM,
        TFAutoModelForMaskedLM,
        TFAutoModelForPreTraining,
        TFAutoModelForQuestionAnswering,
        TFAutoModelForSeq2SeqLM,
        TFAutoModelForSequenceClassification,
        TFAutoModelWithLMHead,
        TFBertForMaskedLM,
        TFBertForPreTraining,
        TFBertForQuestionAnswering,
        TFBertForSequenceClassification,
        TFBertModel,
        TFGPT2LMHeadModel,
        TFRobertaForMaskedLM,
        TFT5ForConditionalGeneration,
    )
    from transformers.models.bert.modeling_tf_bert import TF_BERT_PRETRAINED_MODEL_ARCHIVE_LIST
    from transformers.models.gpt2.modeling_tf_gpt2 import TF_GPT2_PRETRAINED_MODEL_ARCHIVE_LIST
    from transformers.models.t5.modeling_tf_t5 import TF_T5_PRETRAINED_MODEL_ARCHIVE_LIST

if is_torch_available():
    from transformers import (
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForMaskedLM,
        AutoModelForPreTraining,
        AutoModelForQuestionAnswering,
        AutoModelForSeq2SeqLM,
        AutoModelForSequenceClassification,
        AutoModelWithLMHead,
        BertForMaskedLM,
        BertForPreTraining,
        BertForQuestionAnswering,
        BertForSequenceClassification,
        BertModel,
        GPT2LMHeadModel,
        RobertaForMaskedLM,
        T5ForConditionalGeneration,
    )


@is_pt_tf_cross_test
class TFPTAutoModelTest(unittest.TestCase):
    @slow
    def test_model_from_pretrained(self):
        import h5py

        self.assertTrue(h5py.version.hdf5_version.startswith("1.10"))

        # for model_name in TF_BERT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
        for model_name in ["bert-base-uncased"]:
            config = AutoConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, BertConfig)

            model = TFAutoModel.from_pretrained(model_name, from_pt=True)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, TFBertModel)

            model = AutoModel.from_pretrained(model_name, from_tf=True)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, BertModel)

    @slow
    def test_model_for_pretraining_from_pretrained(self):
        import h5py

        self.assertTrue(h5py.version.hdf5_version.startswith("1.10"))

        # for model_name in TF_BERT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
        for model_name in ["bert-base-uncased"]:
            config = AutoConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, BertConfig)

            model = TFAutoModelForPreTraining.from_pretrained(model_name, from_pt=True)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, TFBertForPreTraining)

            model = AutoModelForPreTraining.from_pretrained(model_name, from_tf=True)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, BertForPreTraining)

    @slow
    def test_model_for_causal_lm(self):
        for model_name in TF_GPT2_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            config = AutoConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, GPT2Config)

            model = TFAutoModelForCausalLM.from_pretrained(model_name, from_pt=True)
            model, loading_info = TFAutoModelForCausalLM.from_pretrained(
                model_name, output_loading_info=True, from_pt=True
            )
            self.assertIsNotNone(model)
            self.assertIsInstance(model, TFGPT2LMHeadModel)

            model = AutoModelForCausalLM.from_pretrained(model_name, from_tf=True)
            model, loading_info = AutoModelForCausalLM.from_pretrained(
                model_name, output_loading_info=True, from_tf=True
            )
            self.assertIsNotNone(model)
            self.assertIsInstance(model, GPT2LMHeadModel)

    @slow
    def test_lmhead_model_from_pretrained(self):
        for model_name in TF_BERT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            config = AutoConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, BertConfig)

            model = TFAutoModelWithLMHead.from_pretrained(model_name, from_pt=True)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, TFBertForMaskedLM)

            model = AutoModelWithLMHead.from_pretrained(model_name, from_tf=True)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, BertForMaskedLM)

    @slow
    def test_model_for_masked_lm(self):
        for model_name in TF_BERT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            config = AutoConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, BertConfig)

            model = TFAutoModelForMaskedLM.from_pretrained(model_name, from_pt=True)
            model, loading_info = TFAutoModelForMaskedLM.from_pretrained(
                model_name, output_loading_info=True, from_pt=True
            )
            self.assertIsNotNone(model)
            self.assertIsInstance(model, TFBertForMaskedLM)

            model = AutoModelForMaskedLM.from_pretrained(model_name, from_tf=True)
            model, loading_info = AutoModelForMaskedLM.from_pretrained(
                model_name, output_loading_info=True, from_tf=True
            )
            self.assertIsNotNone(model)
            self.assertIsInstance(model, BertForMaskedLM)

    @slow
    def test_model_for_encoder_decoder_lm(self):
        for model_name in TF_T5_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            config = AutoConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, T5Config)

            model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name, from_pt=True)
            model, loading_info = TFAutoModelForSeq2SeqLM.from_pretrained(
                model_name, output_loading_info=True, from_pt=True
            )
            self.assertIsNotNone(model)
            self.assertIsInstance(model, TFT5ForConditionalGeneration)

            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, from_tf=True)
            model, loading_info = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, output_loading_info=True, from_tf=True
            )
            self.assertIsNotNone(model)
            self.assertIsInstance(model, T5ForConditionalGeneration)

    @slow
    def test_sequence_classification_model_from_pretrained(self):
        # for model_name in TF_BERT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
        for model_name in ["bert-base-uncased"]:
            config = AutoConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, BertConfig)

            model = TFAutoModelForSequenceClassification.from_pretrained(model_name, from_pt=True)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, TFBertForSequenceClassification)

            model = AutoModelForSequenceClassification.from_pretrained(model_name, from_tf=True)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, BertForSequenceClassification)

    @slow
    def test_question_answering_model_from_pretrained(self):
        # for model_name in TF_BERT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
        for model_name in ["bert-base-uncased"]:
            config = AutoConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, BertConfig)

            model = TFAutoModelForQuestionAnswering.from_pretrained(model_name, from_pt=True)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, TFBertForQuestionAnswering)

            model = AutoModelForQuestionAnswering.from_pretrained(model_name, from_tf=True)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, BertForQuestionAnswering)

    def test_from_pretrained_identifier(self):
        model = TFAutoModelWithLMHead.from_pretrained(SMALL_MODEL_IDENTIFIER, from_pt=True)
        self.assertIsInstance(model, TFBertForMaskedLM)
        self.assertEqual(model.num_parameters(), 14410)
        self.assertEqual(model.num_parameters(only_trainable=True), 14410)

        model = AutoModelWithLMHead.from_pretrained(SMALL_MODEL_IDENTIFIER, from_tf=True)
        self.assertIsInstance(model, BertForMaskedLM)
        self.assertEqual(model.num_parameters(), 14410)
        self.assertEqual(model.num_parameters(only_trainable=True), 14410)

    def test_from_identifier_from_model_type(self):
        model = TFAutoModelWithLMHead.from_pretrained(DUMMY_UNKWOWN_IDENTIFIER, from_pt=True)
        self.assertIsInstance(model, TFRobertaForMaskedLM)
        self.assertEqual(model.num_parameters(), 14410)
        self.assertEqual(model.num_parameters(only_trainable=True), 14410)

        model = AutoModelWithLMHead.from_pretrained(DUMMY_UNKWOWN_IDENTIFIER, from_tf=True)
        self.assertIsInstance(model, RobertaForMaskedLM)
        self.assertEqual(model.num_parameters(), 14410)
        self.assertEqual(model.num_parameters(only_trainable=True), 14410)
