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
import os
import tempfile
import unittest

from transformers import is_torch_available
from transformers.testing_utils import (
    DUMMY_UNKNOWN_IDENTIFIER,
    SMALL_MODEL_IDENTIFIER,
    require_scatter,
    require_torch,
    slow,
)


if is_torch_available():
    import torch

    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForMaskedLM,
        AutoModelForPreTraining,
        AutoModelForQuestionAnswering,
        AutoModelForSeq2SeqLM,
        AutoModelForSequenceClassification,
        AutoModelForTableQuestionAnswering,
        AutoModelForTokenClassification,
        AutoModelWithLMHead,
        BertConfig,
        BertForMaskedLM,
        BertForPreTraining,
        BertForQuestionAnswering,
        BertForSequenceClassification,
        BertForTokenClassification,
        BertModel,
        FunnelBaseModel,
        FunnelModel,
        GPT2Config,
        GPT2LMHeadModel,
        PreTrainedModel,
        RobertaForMaskedLM,
        T5Config,
        T5ForConditionalGeneration,
        TapasConfig,
        TapasForQuestionAnswering,
    )
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_CAUSAL_LM_MAPPING,
        MODEL_FOR_MASKED_LM_MAPPING,
        MODEL_FOR_PRETRAINING_MAPPING,
        MODEL_FOR_QUESTION_ANSWERING_MAPPING,
        MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
        MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
        MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING,
        MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
        MODEL_MAPPING,
        MODEL_WITH_LM_HEAD_MAPPING,
    )
    from transformers.models.bert.modeling_bert import BERT_PRETRAINED_MODEL_ARCHIVE_LIST
    from transformers.models.gpt2.modeling_gpt2 import GPT2_PRETRAINED_MODEL_ARCHIVE_LIST
    from transformers.models.t5.modeling_t5 import T5_PRETRAINED_MODEL_ARCHIVE_LIST
    from transformers.models.tapas.modeling_tapas import TAPAS_PRETRAINED_MODEL_ARCHIVE_LIST


if is_torch_available():

    class FakeModel(PreTrainedModel):
        config_class = BertConfig
        base_model_prefix = "fake"

        def __init__(self, config):
            super().__init__(config)
            self.linear = torch.nn.Linear(config.hidden_size, config.hidden_size)

        def forward(self, x):
            return self.linear(x)

        def _init_weights(self, module):
            pass


# Make sure this is synchronized with the model above.
FAKE_MODEL_CODE = """
import torch
from transformers import BertConfig, PreTrainedModel

class FakeModel(PreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "fake"

    def __init__(self, config):
        super().__init__(config)
        self.linear = torch.nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x):
        return self.linear(x)

    def _init_weights(self, module):
        pass
"""


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

            self.assertEqual(len(loading_info["missing_keys"]), 0)
            self.assertEqual(len(loading_info["unexpected_keys"]), 8)
            self.assertEqual(len(loading_info["mismatched_keys"]), 0)
            self.assertEqual(len(loading_info["error_msgs"]), 0)

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
            # Only one value should not be initialized and in the missing keys.
            missing_keys = loading_info.pop("missing_keys")
            self.assertListEqual(["cls.predictions.decoder.bias"], missing_keys)
            for key, value in loading_info.items():
                self.assertEqual(len(value), 0)

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
    @require_scatter
    def test_table_question_answering_model_from_pretrained(self):
        for model_name in TAPAS_PRETRAINED_MODEL_ARCHIVE_LIST[5:6]:
            config = AutoConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, TapasConfig)

            model = AutoModelForTableQuestionAnswering.from_pretrained(model_name)
            model, loading_info = AutoModelForTableQuestionAnswering.from_pretrained(
                model_name, output_loading_info=True
            )
            self.assertIsNotNone(model)
            self.assertIsInstance(model, TapasForQuestionAnswering)

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
        self.assertEqual(model.num_parameters(), 14410)
        self.assertEqual(model.num_parameters(only_trainable=True), 14410)

    def test_from_identifier_from_model_type(self):
        model = AutoModelWithLMHead.from_pretrained(DUMMY_UNKNOWN_IDENTIFIER)
        self.assertIsInstance(model, RobertaForMaskedLM)
        self.assertEqual(model.num_parameters(), 14410)
        self.assertEqual(model.num_parameters(only_trainable=True), 14410)

    def test_from_pretrained_with_tuple_values(self):
        # For the auto model mapping, FunnelConfig has two models: FunnelModel and FunnelBaseModel
        model = AutoModel.from_pretrained("sgugger/funnel-random-tiny")
        self.assertIsInstance(model, FunnelModel)

        config = copy.deepcopy(model.config)
        config.architectures = ["FunnelBaseModel"]
        model = AutoModel.from_config(config)
        self.assertIsInstance(model, FunnelBaseModel)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            model = AutoModel.from_pretrained(tmp_dir)
            self.assertIsInstance(model, FunnelBaseModel)

    def test_parents_and_children_in_mappings(self):
        # Test that the children are placed before the parents in the mappings, as the `instanceof` will be triggered
        # by the parents and will return the wrong configuration type when using auto models

        mappings = (
            MODEL_MAPPING,
            MODEL_FOR_PRETRAINING_MAPPING,
            MODEL_FOR_QUESTION_ANSWERING_MAPPING,
            MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING,
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
                    assert not issubclass(
                        child_config, parent_config
                    ), f"{child_config.__name__} is child of {parent_config.__name__}"

                    # Tuplify child_model and parent_model since some of them could be tuples.
                    if not isinstance(child_model, (list, tuple)):
                        child_model = (child_model,)
                    if not isinstance(parent_model, (list, tuple)):
                        parent_model = (parent_model,)

                    for child, parent in [(a, b) for a in child_model for b in parent_model]:
                        assert not issubclass(child, parent), f"{child.__name__} is child of {parent.__name__}"

    def test_from_pretrained_dynamic_model(self):
        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
        )
        config.auto_map = {"AutoModel": "modeling.FakeModel"}
        model = FakeModel(config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            with open(os.path.join(tmp_dir, "modeling.py"), "w") as f:
                f.write(FAKE_MODEL_CODE)

            new_model = AutoModel.from_pretrained(tmp_dir, trust_remote_code=True)
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                self.assertTrue(torch.equal(p1, p2))
