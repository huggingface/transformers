# Copyright 2018 Salesforce and HuggingFace Inc. team.
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

from transformers import CTRLConfig, is_torch_available
from transformers.testing_utils import cleanup, require_torch, slow, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...model_tester_config import ModelTesterConfigMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        CTRLForSequenceClassification,
        CTRLLMHeadModel,
        CTRLModel,
    )


class CTRLModelTester(ModelTesterConfigMixin):
    defaults = {}

    small_config_overrides = {
        "batch_size": 14,
        "seq_length": 7,
        "is_training": True,
        "use_token_type_ids": True,
        "use_input_mask": True,
        "use_labels": True,
        "use_mc_token_ids": True,
        "vocab_size": 99,
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "intermediate_size": 37,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 16,
        "type_sequence_label_size": 2,
        "initializer_range": 0.02,
        "num_labels": 3,
        "num_choices": 4,
        "scope": None,
    }

    @classmethod
    def get_small_config_defaults(cls, **kwargs):
        values = super().get_small_config_defaults(**kwargs)
        values["pad_token_id"] = values["vocab_size"] - 1
        return values

    def __init__(self, parent, **kwargs):
        self.parent = parent
        for key, value in self.get_small_config_defaults(**kwargs).items():
            setattr(self, key, value)

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        mc_token_ids = None
        if self.use_mc_token_ids:
            mc_token_ids = ids_tensor([self.batch_size, self.num_choices], self.seq_length)

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        return (
            config,
            input_ids,
            input_mask,
            token_type_ids,
            mc_token_ids,
            sequence_labels,
            token_labels,
            choice_labels,
        )

    def get_config(self):
        return CTRLConfig(
            vocab_size=self.vocab_size,
            n_embd=self.hidden_size,
            n_layer=self.num_hidden_layers,
            n_head=self.num_attention_heads,
            dff=self.intermediate_size,
            # hidden_act=self.hidden_act,
            # hidden_dropout_prob=self.hidden_dropout_prob,
            # attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            n_positions=self.max_position_embeddings,
            # type_vocab_size=self.type_vocab_size,
            # initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
        )

    def create_and_check_ctrl_model(self, config, input_ids, input_mask, token_type_ids, *args):
        model = CTRLModel(config=config)
        model.to(torch_device)
        model.eval()

        model(input_ids, token_type_ids=token_type_ids)
        model(input_ids, token_type_ids=token_type_ids)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(len(result.cache), config.n_layer)

    def create_and_check_lm_head_model(self, config, input_ids, input_mask, token_type_ids, *args):
        model = CTRLLMHeadModel(config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids, token_type_ids=token_type_ids, labels=input_ids)
        self.parent.assertEqual(result.loss.shape, ())
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()

        (
            config,
            input_ids,
            input_mask,
            token_type_ids,
            mc_token_ids,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs

        inputs_dict = {"input_ids": input_ids, "token_type_ids": token_type_ids}

        return config, inputs_dict


@require_torch
class CTRLModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    model_tester_class = CTRLModelTester

    all_model_classes = (CTRLModel, CTRLLMHeadModel, CTRLForSequenceClassification) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": CTRLModel,
            "text-classification": CTRLForSequenceClassification,
            "text-generation": CTRLLMHeadModel,
            "zero-shot": CTRLForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    test_resize_embeddings = False

    # TODO: Fix the failed tests
    def is_pipeline_test_to_skip(
        self,
        pipeline_test_case_name,
        config_class,
        model_architecture,
        tokenizer_name,
        image_processor_name,
        feature_extractor_name,
        processor_name,
    ):
        if pipeline_test_case_name == "ZeroShotClassificationPipelineTests":
            # Get `tokenizer does not have a padding token` error for both fast/slow tokenizers.
            # `CTRLConfig` was never used in pipeline tests, either because of a missing checkpoint or because a tiny
            # config could not be created.
            return True

        return False

    def setUp(self):
        self.model_tester = CTRLModelTester(self)
        self.config_tester = ConfigTester(self, config_class=CTRLConfig, n_embd=37)

    def tearDown(self):
        super().tearDown()
        # clean-up as much as possible GPU memory occupied by PyTorch
        cleanup(torch_device)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_ctrl_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_ctrl_model(*config_and_inputs)

    def test_ctrl_lm_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lm_head_model(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        model_name = "Salesforce/ctrl"
        model = CTRLModel.from_pretrained(model_name)
        self.assertIsNotNone(model)


@require_torch
class CTRLModelLanguageGenerationTest(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        # clean-up as much as possible GPU memory occupied by PyTorch
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_lm_generate_ctrl(self):
        model = CTRLLMHeadModel.from_pretrained("Salesforce/ctrl")
        model.to(torch_device)
        input_ids = torch.tensor(
            [[11859, 0, 1611, 8]], dtype=torch.long, device=torch_device
        )  # Legal the president is
        expected_output_ids = [
            11859,
            0,
            1611,
            8,
            5,
            150,
            26449,
            2,
            19,
            348,
            469,
            3,
            2595,
            48,
            20740,
            246533,
            246533,
            19,
            30,
            5,
        ]  # Legal the president is a good guy and I don't want to lose my job. \n \n I have a

        # We limit the generation output to (max_length - input_length) while by default 20 new tokens will be generated.
        output_ids = model.generate(input_ids, do_sample=False, max_length=20)
        self.assertListEqual(output_ids[0].tolist(), expected_output_ids)
