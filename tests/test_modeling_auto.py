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


import unittest

from transformers import is_torch_available
from transformers.testing_utils import DUMMY_UNKWOWN_IDENTIFIER, SMALL_MODEL_IDENTIFIER, require_torch, slow


if is_torch_available():
    from transformers import (
        AutoConfig,
        BertConfig,
        GPT2Config,
        T5Config,
        AutoModel,
        BertModel,
        AutoModelForPreTraining,
        BertForPreTraining,
        AutoModelForCausalLM,
        GPT2LMHeadModel,
        AutoModelWithLMHead,
        AutoModelForMaskedLM,
        BertForMaskedLM,
        RobertaForMaskedLM,
        AutoModelForSeq2SeqLM,
        T5ForConditionalGeneration,
        AutoModelForSequenceClassification,
        BertForSequenceClassification,
        AutoModelForQuestionAnswering,
        BertForQuestionAnswering,
        AutoModelForTokenClassification,
        BertForTokenClassification,
    )
    from transformers.modeling_bert import BERT_PRETRAINED_MODEL_ARCHIVE_LIST
    from transformers.modeling_gpt2 import GPT2_PRETRAINED_MODEL_ARCHIVE_LIST
    from transformers.modeling_t5 import T5_PRETRAINED_MODEL_ARCHIVE_LIST
    from transformers.modeling_auto import (
        MODEL_MAPPING,
        MODEL_FOR_PRETRAINING_MAPPING,
        MODEL_FOR_QUESTION_ANSWERING_MAPPING,
        MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
        MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
        MODEL_WITH_LM_HEAD_MAPPING,
        MODEL_FOR_CAUSAL_LM_MAPPING,
        MODEL_FOR_MASKED_LM_MAPPING,
        MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    )


@require_torch
class AutoModelTest(unittest.TestCase):
    @slow
    def test_model_from_pretrained(self):
        for model_name in BERT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            config = AutoConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, BertConfig)

            model = AutoModel.from_pretrained(model_name)
            model, loading_info = AutoModel.from_pretrained(model_name, output_loading_info=True)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, BertModel)
            for value in loading_info.values():
                self.assertEqual(len(value), 0)

    @slow
    def test_model_for_pretraining_from_pretrained(self):
        for model_name in BERT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            config = AutoConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, BertConfig)

            model = AutoModelForPreTraining.from_pretrained(model_name)
            model, loading_info = AutoModelForPreTraining.from_pretrained(model_name, output_loading_info=True)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, BertForPreTraining)
            for key, value in loading_info.items():
                # Only one value should not be initialized and in the missing keys.
                self.assertEqual(len(value), 1 if key == "missing_keys" else 0)

    @slow
    def test_lmhead_model_from_pretrained(self):
        for model_name in BERT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            config = AutoConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, BertConfig)

            model = AutoModelWithLMHead.from_pretrained(model_name)
            model, loading_info = AutoModelWithLMHead.from_pretrained(model_name, output_loading_info=True)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, BertForMaskedLM)

    @slow
    def test_model_for_causal_lm(self):
        for model_name in GPT2_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            config = AutoConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, GPT2Config)

            model = AutoModelForCausalLM.from_pretrained(model_name)
            model, loading_info = AutoModelForCausalLM.from_pretrained(model_name, output_loading_info=True)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, GPT2LMHeadModel)

    @slow
    def test_model_for_masked_lm(self):
        for model_name in BERT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            config = AutoConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, BertConfig)

            model = AutoModelForMaskedLM.from_pretrained(model_name)
            model, loading_info = AutoModelForMaskedLM.from_pretrained(model_name, output_loading_info=True)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, BertForMaskedLM)

    @slow
    def test_model_for_encoder_decoder_lm(self):
        for model_name in T5_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            config = AutoConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, T5Config)

            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            model, loading_info = AutoModelForSeq2SeqLM.from_pretrained(model_name, output_loading_info=True)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, T5ForConditionalGeneration)

    @slow
    def test_sequence_classification_model_from_pretrained(self):
        for model_name in BERT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            config = AutoConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, BertConfig)

            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            model, loading_info = AutoModelForSequenceClassification.from_pretrained(
                model_name, output_loading_info=True
            )
            self.assertIsNotNone(model)
            self.assertIsInstance(model, BertForSequenceClassification)

    @slow
    def test_question_answering_model_from_pretrained(self):
        for model_name in BERT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            config = AutoConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, BertConfig)

            model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            model, loading_info = AutoModelForQuestionAnswering.from_pretrained(model_name, output_loading_info=True)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, BertForQuestionAnswering)

    @slow
    def test_token_classification_model_from_pretrained(self):
        for model_name in BERT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            config = AutoConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, BertConfig)

            model = AutoModelForTokenClassification.from_pretrained(model_name)
            model, loading_info = AutoModelForTokenClassification.from_pretrained(model_name, output_loading_info=True)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, BertForTokenClassification)

    def test_from_pretrained_identifier(self):
        model = AutoModelWithLMHead.from_pretrained(SMALL_MODEL_IDENTIFIER)
        self.assertIsInstance(model, BertForMaskedLM)
        self.assertEqual(model.num_parameters(), 14830)
        self.assertEqual(model.num_parameters(only_trainable=True), 14830)

    def test_from_identifier_from_model_type(self):
        model = AutoModelWithLMHead.from_pretrained(DUMMY_UNKWOWN_IDENTIFIER)
        self.assertIsInstance(model, RobertaForMaskedLM)
        self.assertEqual(model.num_parameters(), 14830)
        self.assertEqual(model.num_parameters(only_trainable=True), 14830)

    def test_parents_and_children_in_mappings(self):
        # Test that the children are placed before the parents in the mappings, as the `instanceof` will be triggered
        # by the parents and will return the wrong configuration type when using auto models

        mappings = (
            MODEL_MAPPING,
            MODEL_FOR_PRETRAINING_MAPPING,
            MODEL_FOR_QUESTION_ANSWERING_MAPPING,
            MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
            MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
            MODEL_WITH_LM_HEAD_MAPPING,
            MODEL_FOR_CAUSAL_LM_MAPPING,
            MODEL_FOR_MASKED_LM_MAPPING,
            MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
        )

        for mapping in mappings:
            mapping = tuple(mapping.items())
            for index, (child_config, child_model) in enumerate(mapping[1:]):
                for parent_config, parent_model in mapping[: index + 1]:
                    with self.subTest(
                        msg="Testing if {} is child of {}".format(child_config.__name__, parent_config.__name__)
                    ):
                        self.assertFalse(issubclass(child_config, parent_config))
                        self.assertFalse(issubclass(child_model, parent_model))
