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

from transformers import CONFIG_MAPPING, AutoConfig, BertConfig, GPT2Config, T5Config, TapasConfig, is_tf_available
from transformers.testing_utils import (
    DUMMY_UNKNOWN_IDENTIFIER,
    SMALL_MODEL_IDENTIFIER,
    RequestCounter,
    require_tensorflow_probability,
    require_tf,
    slow,
)

from ..bert.test_modeling_bert import BertModelTester


if is_tf_available():
    from transformers import (
        TFAutoModel,
        TFAutoModelForCausalLM,
        TFAutoModelForMaskedLM,
        TFAutoModelForPreTraining,
        TFAutoModelForQuestionAnswering,
        TFAutoModelForSeq2SeqLM,
        TFAutoModelForSequenceClassification,
        TFAutoModelForTableQuestionAnswering,
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
        TFTapasForQuestionAnswering,
    )
    from transformers.models.auto.modeling_tf_auto import (
        TF_MODEL_FOR_CAUSAL_LM_MAPPING,
        TF_MODEL_FOR_MASKED_LM_MAPPING,
        TF_MODEL_FOR_PRETRAINING_MAPPING,
        TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING,
        TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
        TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
        TF_MODEL_MAPPING,
    )
    from transformers.models.bert.modeling_tf_bert import TF_BERT_PRETRAINED_MODEL_ARCHIVE_LIST
    from transformers.models.gpt2.modeling_tf_gpt2 import TF_GPT2_PRETRAINED_MODEL_ARCHIVE_LIST
    from transformers.models.t5.modeling_tf_t5 import TF_T5_PRETRAINED_MODEL_ARCHIVE_LIST
    from transformers.models.tapas.modeling_tf_tapas import TF_TAPAS_PRETRAINED_MODEL_ARCHIVE_LIST


class NewModelConfig(BertConfig):
    model_type = "new-model"


if is_tf_available():

    class TFNewModel(TFBertModel):
        config_class = NewModelConfig


@require_tf
class TFAutoModelTest(unittest.TestCase):
    @slow
    def test_model_from_pretrained(self):
        model_name = "bert-base-cased"
        config = AutoConfig.from_pretrained(model_name)
        self.assertIsNotNone(config)
        self.assertIsInstance(config, BertConfig)

        model = TFAutoModel.from_pretrained(model_name)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, TFBertModel)

    @slow
    def test_model_for_pretraining_from_pretrained(self):
        model_name = "bert-base-cased"
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

    @slow
    @require_tensorflow_probability
    def test_table_question_answering_model_from_pretrained(self):
        for model_name in TF_TAPAS_PRETRAINED_MODEL_ARCHIVE_LIST[5:6]:
            config = AutoConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, TapasConfig)

            model = TFAutoModelForTableQuestionAnswering.from_pretrained(model_name)
            model, loading_info = TFAutoModelForTableQuestionAnswering.from_pretrained(
                model_name, output_loading_info=True
            )
            self.assertIsNotNone(model)
            self.assertIsInstance(model, TFTapasForQuestionAnswering)

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

    def test_repo_not_found(self):
        with self.assertRaisesRegex(
            EnvironmentError, "bert-base is not a local folder and is not a valid model identifier"
        ):
            _ = TFAutoModel.from_pretrained("bert-base")

    def test_revision_not_found(self):
        with self.assertRaisesRegex(
            EnvironmentError, r"aaaaaa is not a valid git identifier \(branch name, tag name or commit id\)"
        ):
            _ = TFAutoModel.from_pretrained(DUMMY_UNKNOWN_IDENTIFIER, revision="aaaaaa")

    def test_model_file_not_found(self):
        with self.assertRaisesRegex(
            EnvironmentError,
            "hf-internal-testing/config-no-model does not appear to have a file named tf_model.h5",
        ):
            _ = TFAutoModel.from_pretrained("hf-internal-testing/config-no-model")

    def test_model_from_pt_suggestion(self):
        with self.assertRaisesRegex(EnvironmentError, "Use `from_pt=True` to load this model"):
            _ = TFAutoModel.from_pretrained("hf-internal-testing/tiny-bert-pt-only")

    def test_cached_model_has_minimum_calls_to_head(self):
        # Make sure we have cached the model.
        _ = TFAutoModel.from_pretrained("hf-internal-testing/tiny-random-bert")
        with RequestCounter() as counter:
            _ = TFAutoModel.from_pretrained("hf-internal-testing/tiny-random-bert")
            self.assertEqual(counter.get_request_count, 0)
            self.assertEqual(counter.head_request_count, 1)
            self.assertEqual(counter.other_request_count, 0)

        # With a sharded checkpoint
        _ = TFAutoModel.from_pretrained("ArthurZ/tiny-random-bert-sharded")
        with RequestCounter() as counter:
            _ = TFAutoModel.from_pretrained("ArthurZ/tiny-random-bert-sharded")
            self.assertEqual(counter.get_request_count, 0)
            # There is no pytorch_model.bin so we still get one call for this one.
            self.assertEqual(counter.head_request_count, 2)
            self.assertEqual(counter.other_request_count, 0)
