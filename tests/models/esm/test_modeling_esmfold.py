# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch ESM model. """


import unittest

from transformers import EsmConfig, is_torch_available
from transformers.testing_utils import TestCasePlus, require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask


if is_torch_available():
    import torch

    from transformers.models.esm.modeling_esmfold import EsmForProteinFolding


class EsmFoldModelTester:
    def __init__(
        self,
        parent,
    ):
        self.parent = parent
        self.batch_size = 13
        self.seq_length = 7
        self.is_training = False
        self.use_input_mask = True
        self.use_token_type_ids = False
        self.use_labels = False
        self.vocab_size = 19
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

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        return config, input_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        config = EsmConfig(
            vocab_size=33,
            hidden_size=self.hidden_size,
            pad_token_id=1,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
            is_folding_model=True,
            esmfold_config={"trunk": {"num_blocks": 2}, "fp16_esm": False},
        )
        return config

    def create_and_check_model(self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels):
        model = EsmForProteinFolding(config=config).float()
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        result = model(input_ids)

        self.parent.assertEqual(result.positions.shape, (8, self.batch_size, self.seq_length, 14, 3))
        self.parent.assertEqual(result.angles.shape, (8, self.batch_size, self.seq_length, 7, 2))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class EsmFoldModelTest(ModelTesterMixin, unittest.TestCase):

    test_mismatched_shapes = False

    all_model_classes = (EsmForProteinFolding,) if is_torch_available() else ()
    all_generative_model_classes = ()
    test_sequence_classification_problem_types = False

    def setUp(self):
        self.model_tester = EsmFoldModelTester(self)
        self.config_tester = ConfigTester(self, config_class=EsmConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip("Does not support attention outputs")
    def test_attention_outputs(self):
        pass

    @unittest.skip
    def test_correct_missing_keys(self):
        pass

    @unittest.skip("Esm does not support embedding resizing")
    def test_resize_embeddings_untied(self):
        pass

    @unittest.skip("Esm does not support embedding resizing")
    def test_resize_tokens_embeddings(self):
        pass

    @unittest.skip("ESMFold does not support passing input embeds!")
    def test_inputs_embeds(self):
        pass

    @unittest.skip("ESMFold does not support head pruning.")
    def test_head_pruning(self):
        pass

    @unittest.skip("ESMFold does not support head pruning.")
    def test_head_pruning_integration(self):
        pass

    @unittest.skip("ESMFold does not support head pruning.")
    def test_head_pruning_save_load_from_config_init(self):
        pass

    @unittest.skip("ESMFold does not support head pruning.")
    def test_head_pruning_save_load_from_pretrained(self):
        pass

    @unittest.skip("ESMFold does not support head pruning.")
    def test_headmasking(self):
        pass

    @unittest.skip("ESMFold does not output hidden states in the normal way.")
    def test_hidden_states_output(self):
        pass

    @unittest.skip("ESMfold does not output hidden states in the normal way.")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip("ESMFold only has one output format.")
    def test_model_outputs_equivalence(self):
        pass

    @unittest.skip("This test doesn't work for ESMFold and doesn't test core functionality")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip("ESMFold does not support input chunking.")
    def test_feed_forward_chunking(self):
        pass

    @unittest.skip("ESMFold doesn't respect you and it certainly doesn't respect your initialization arguments.")
    def test_initialization(self):
        pass

    @unittest.skip("ESMFold doesn't support torchscript compilation.")
    def test_torchscript_output_attentions(self):
        pass

    @unittest.skip("ESMFold doesn't support torchscript compilation.")
    def test_torchscript_output_hidden_state(self):
        pass

    @unittest.skip("ESMFold doesn't support torchscript compilation.")
    def test_torchscript_simple(self):
        pass

    @unittest.skip("ESMFold doesn't support data parallel.")
    def test_multi_gpu_data_parallel_forward(self):
        pass


@require_torch
class EsmModelIntegrationTest(TestCasePlus):
    @slow
    def test_inference_protein_folding(self):
        model = EsmForProteinFolding.from_pretrained("Rocketknight1/esmfold_v1").float()
        model.eval()
        input_ids = torch.tensor([[0, 6, 4, 13, 5, 4, 16, 12, 11, 7, 2]])
        position_outputs = model(input_ids)["positions"]
        expected_slice = torch.tensor([2.5828, 0.7993, -10.9334], dtype=torch.float32)
        self.assertTrue(torch.allclose(position_outputs[0, 0, 0, 0], expected_slice, atol=1e-4))
