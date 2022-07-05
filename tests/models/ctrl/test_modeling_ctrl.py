# coding=utf-8
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


import gc
import unittest

from transformers import CTRLConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...generation.test_generation_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask


if is_torch_available():
    import torch

    from transformers import (
        CTRL_PRETRAINED_MODEL_ARCHIVE_LIST,
        CTRLForSequenceClassification,
        CTRLLMHeadModel,
        CTRLModel,
    )


class CTRLModelTester:
    def __init__(
        self,
        parent,
    ):
        self.parent = parent
        self.batch_size = 14
        self.seq_length = 7
        self.is_training = True
        self.use_token_type_ids = True
        self.use_input_mask = True
        self.use_labels = True
        self.use_mc_token_ids = True
        self.vocab_size = 99
        self.hidden_size = 32
        self.num_hidden_layers = 5
        self.num_attention_heads = 4
        self.intermediate_size = 37
        self.hidden_act = "gelu"
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.max_position_embeddings = 512
        self.type_vocab_size = 16
        self.type_sequence_label_size = 2
        self.initializer_range = 0.02
        self.num_labels = 3
        self.num_choices = 4
        self.scope = None
        self.pad_token_id = self.vocab_size - 1

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

        head_mask = ids_tensor([self.num_hidden_layers, self.num_attention_heads], 2)

        return (
            config,
            input_ids,
            input_mask,
            head_mask,
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
            # intermediate_size=self.intermediate_size,
            # hidden_act=self.hidden_act,
            # hidden_dropout_prob=self.hidden_dropout_prob,
            # attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            n_positions=self.max_position_embeddings,
            # type_vocab_size=self.type_vocab_size,
            # initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
        )

    def create_and_check_ctrl_model(self, config, input_ids, input_mask, head_mask, token_type_ids, *args):
        model = CTRLModel(config=config)
        model.to(torch_device)
        model.eval()

        model(input_ids, token_type_ids=token_type_ids, head_mask=head_mask)
        model(input_ids, token_type_ids=token_type_ids)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(len(result.past_key_values), config.n_layer)

    def create_and_check_lm_head_model(self, config, input_ids, input_mask, head_mask, token_type_ids, *args):
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
            head_mask,
            token_type_ids,
            mc_token_ids,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs

        inputs_dict = {"input_ids": input_ids, "token_type_ids": token_type_ids, "head_mask": head_mask}

        return config, inputs_dict

    def create_and_check_ctrl_for_sequence_classification(self, config, input_ids, head_mask, token_type_ids, *args):
        config.num_labels = self.num_labels
        model = CTRLForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
        result = model(input_ids, token_type_ids=token_type_ids, labels=sequence_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))


@require_torch
class CTRLModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):

    all_model_classes = (CTRLModel, CTRLLMHeadModel, CTRLForSequenceClassification) if is_torch_available() else ()
    all_generative_model_classes = (CTRLLMHeadModel,) if is_torch_available() else ()
    test_pruning = True
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = CTRLModelTester(self)
        self.config_tester = ConfigTester(self, config_class=CTRLConfig, n_embd=37)

    def tearDown(self):
        super().tearDown()
        # clean-up as much as possible GPU memory occupied by PyTorch
        gc.collect()
        torch.cuda.empty_cache()

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
        for model_name in CTRL_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = CTRLModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


@require_torch
class CTRLModelLanguageGenerationTest(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        # clean-up as much as possible GPU memory occupied by PyTorch
        gc.collect()
        torch.cuda.empty_cache()

    @slow
    def test_lm_generate_ctrl(self):
        model = CTRLLMHeadModel.from_pretrained("ctrl")
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

        output_ids = model.generate(input_ids, do_sample=False)
        self.assertListEqual(output_ids[0].tolist(), expected_output_ids)
