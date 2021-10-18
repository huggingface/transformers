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

import copy
import tempfile
import unittest

from transformers import CONFIG_MAPPING, AutoConfig, BertConfig, GPT2Config, T5Config, is_tf_available
from transformers.testing_utils import DUMMY_UNKNOWN_IDENTIFIER, SMALL_MODEL_IDENTIFIER, require_tf, slow

from .test_modeling_bert import BertModelTester


if is_tf_available():
    from transformers import (
        TFAutoModel,
        TFAutoModelForCausalLM,
        TFAutoModelForMaskedLM,
        TFAutoModelForPreTraining,
        TFAutoModelForQuestionAnswering,
        TFAutoModelForSeq2SeqLM,
        TFAutoModelForSequenceClassification,
        TFAutoModelForTokenClassification,
        TFAutoModelWithLMHead,
        TFBertForMaskedLM,
        TFBertForPreTraining,
        TFBertForQuestionAnswering,
        TFBertForSequenceClassification,
        TFBertModel,
        TFFunnelBaseModel,
        TFFunnelModel,
        TFGPT2LMHeadModel,
        TFRobertaForMaskedLM,
        TFT5ForConditionalGeneration,
    )
    from transformers.models.auto.modeling_tf_auto import (
        TF_MODEL_FOR_CAUSAL_LM_MAPPING,
        TF_MODEL_FOR_MASKED_LM_MAPPING,
        TF_MODEL_FOR_PRETRAINING_MAPPING,
        TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING,
        TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
        TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
        TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
        TF_MODEL_MAPPING,
        TF_MODEL_WITH_LM_HEAD_MAPPING,
    )
    from transformers.models.bert.modeling_tf_bert import TF_BERT_PRETRAINED_MODEL_ARCHIVE_LIST
    from transformers.models.gpt2.modeling_tf_gpt2 import TF_GPT2_PRETRAINED_MODEL_ARCHIVE_LIST
    from transformers.models.t5.modeling_tf_t5 import TF_T5_PRETRAINED_MODEL_ARCHIVE_LIST


class NewModelConfig(BertConfig):
    model_type = "new-model"


if is_tf_available():

    class TFNewModel(TFBertModel):
        config_class = NewModelConfig


@require_tf
class TFAutoModelTest(unittest.TestCase):
    @slow
    def test_model_from_pretrained(self):
        import h5py

        self.assertTrue(h5py.version.hdf5_version.startswith("1.10"))

        # for model_name in TF_BERT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
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

        # for model_name in TF_BERT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
        for model_name in ["bert-base-uncased"]:
            config = AutoConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, BertConfig)

            model = TFAutoModelForPreTraining.from_pretrained(model_name)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, TFBertForPreTraining)

    @slow
    def test_model_for_causal_lm(self):
        for model_name in TF_GPT2_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            config = AutoConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, GPT2Config)

            model = TFAutoModelForCausalLM.from_pretrained(model_name)
            model, loading_info = TFAutoModelForCausalLM.from_pretrained(model_name, output_loading_info=True)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, TFGPT2LMHeadModel)

    @slow
    def test_lmhead_model_from_pretrained(self):
        for model_name in TF_BERT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            config = AutoConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, BertConfig)

            model = TFAutoModelWithLMHead.from_pretrained(model_name)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, TFBertForMaskedLM)

    @slow
    def test_model_for_masked_lm(self):
        for model_name in TF_BERT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            config = AutoConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, BertConfig)

            model = TFAutoModelForMaskedLM.from_pretrained(model_name)
            model, loading_info = TFAutoModelForMaskedLM.from_pretrained(model_name, output_loading_info=True)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, TFBertForMaskedLM)

    @slow
    def test_model_for_encoder_decoder_lm(self):
        for model_name in TF_T5_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            config = AutoConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, T5Config)

            model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
            model, loading_info = TFAutoModelForSeq2SeqLM.from_pretrained(model_name, output_loading_info=True)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, TFT5ForConditionalGeneration)

    @slow
    def test_sequence_classification_model_from_pretrained(self):
        # for model_name in TF_BERT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
        for model_name in ["bert-base-uncased"]:
            config = AutoConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, BertConfig)

            model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, TFBertForSequenceClassification)

    @slow
    def test_question_answering_model_from_pretrained(self):
        # for model_name in TF_BERT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
        for model_name in ["bert-base-uncased"]:
            config = AutoConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, BertConfig)

            model = TFAutoModelForQuestionAnswering.from_pretrained(model_name)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, TFBertForQuestionAnswering)

    def test_from_pretrained_identifier(self):
        model = TFAutoModelWithLMHead.from_pretrained(SMALL_MODEL_IDENTIFIER)
        self.assertIsInstance(model, TFBertForMaskedLM)
        self.assertEqual(model.num_parameters(), 14410)
        self.assertEqual(model.num_parameters(only_trainable=True), 14410)

    def test_from_identifier_from_model_type(self):
        model = TFAutoModelWithLMHead.from_pretrained(DUMMY_UNKNOWN_IDENTIFIER)
        self.assertIsInstance(model, TFRobertaForMaskedLM)
        self.assertEqual(model.num_parameters(), 14410)
        self.assertEqual(model.num_parameters(only_trainable=True), 14410)

    def test_from_pretrained_with_tuple_values(self):
        # For the auto model mapping, FunnelConfig has two models: FunnelModel and FunnelBaseModel
        model = TFAutoModel.from_pretrained("sgugger/funnel-random-tiny")
        self.assertIsInstance(model, TFFunnelModel)

        config = copy.deepcopy(model.config)
        config.architectures = ["FunnelBaseModel"]
        model = TFAutoModel.from_config(config)
        self.assertIsInstance(model, TFFunnelBaseModel)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            model = TFAutoModel.from_pretrained(tmp_dir)
            self.assertIsInstance(model, TFFunnelBaseModel)

    def test_parents_and_children_in_mappings(self):
        # Test that the children are placed before the parents in the mappings, as the `instanceof` will be triggered
        # by the parents and will return the wrong configuration type when using auto models
        mappings = (
            TF_MODEL_MAPPING,
            TF_MODEL_FOR_PRETRAINING_MAPPING,
            TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING,
            TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
            TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
            TF_MODEL_WITH_LM_HEAD_MAPPING,
            TF_MODEL_FOR_CAUSAL_LM_MAPPING,
            TF_MODEL_FOR_MASKED_LM_MAPPING,
            TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
        )

        for mapping in mappings:
            mapping = tuple(mapping.items())
            for index, (child_config, child_model) in enumerate(mapping[1:]):
                for parent_config, parent_model in mapping[: index + 1]:
                    with self.subTest(msg=f"Testing if {child_config.__name__} is child of {parent_config.__name__}"):
                        self.assertFalse(issubclass(child_config, parent_config))

                    # Tuplify child_model and parent_model since some of them could be tuples.
                    if not isinstance(child_model, (list, tuple)):
                        child_model = (child_model,)
                    if not isinstance(parent_model, (list, tuple)):
                        parent_model = (parent_model,)

                    for child, parent in [(a, b) for a in child_model for b in parent_model]:
                        assert not issubclass(child, parent), f"{child.__name__} is child of {parent.__name__}"

    def test_new_model_registration(self):
        try:
            AutoConfig.register("new-model", NewModelConfig)

            auto_classes = [
                TFAutoModel,
                TFAutoModelForCausalLM,
                TFAutoModelForMaskedLM,
                TFAutoModelForPreTraining,
                TFAutoModelForQuestionAnswering,
                TFAutoModelForSequenceClassification,
                TFAutoModelForTokenClassification,
            ]

            for auto_class in auto_classes:
                with self.subTest(auto_class.__name__):
                    # Wrong config class will raise an error
                    with self.assertRaises(ValueError):
                        auto_class.register(BertConfig, TFNewModel)
                    auto_class.register(NewModelConfig, TFNewModel)
                    # Trying to register something existing in the Transformers library will raise an error
                    with self.assertRaises(ValueError):
                        auto_class.register(BertConfig, TFBertModel)

                    # Now that the config is registered, it can be used as any other config with the auto-API
                    tiny_config = BertModelTester(self).get_config()
                    config = NewModelConfig(**tiny_config.to_dict())
                    model = auto_class.from_config(config)
                    self.assertIsInstance(model, TFNewModel)

                    with tempfile.TemporaryDirectory() as tmp_dir:
                        model.save_pretrained(tmp_dir)
                        new_model = auto_class.from_pretrained(tmp_dir)
                        self.assertIsInstance(new_model, TFNewModel)

        finally:
            if "new-model" in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content["new-model"]
            for mapping in (
                TF_MODEL_MAPPING,
                TF_MODEL_FOR_PRETRAINING_MAPPING,
                TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING,
                TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
                TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
                TF_MODEL_FOR_CAUSAL_LM_MAPPING,
                TF_MODEL_FOR_MASKED_LM_MAPPING,
            ):
                if NewModelConfig in mapping._extra_content:
                    del mapping._extra_content[NewModelConfig]
