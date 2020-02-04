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

from .test_configuration_common import ConfigTester
from .test_modeling_common import ModelTesterMixin, ids_tensor
from .utils import CACHE_DIR, require_torch, slow, torch_device


if is_torch_available():
    import torch
    from transformers import (
        BARTModel,
        BARTConfig,
    )
    from transformers.modeling_bart import BART_PRETRAINED_MODEL_ARCHIVE_MAP


@require_torch
class BARTModelTest(ModelTesterMixin, unittest.TestCase):

    all_model_classes = (BARTModel,) if is_torch_available() else ()

    test_pruning = False
    test_torchscript = False
    test_resize_embeddings = False
    test_head_masking = False  # TODO(SS): may want to fix this
    is_encoder_decoder = True


    class ModelTester(object):
        def __init__(
            self,
            parent,
            batch_size=13,
            seq_length=7,
            is_training=True,
            use_input_mask=True,
            use_token_type_ids=True,
            use_labels=True,
            vocab_size=99,
            hidden_size=32,
            num_hidden_layers=5,
            num_attention_heads=4,
            intermediate_size=37,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=16,

            initializer_range=0.02,
            num_labels=3,
            num_choices=4,
            scope=None,
        ):
            self.parent = parent
            self.batch_size = 13
            self.seq_length = 7
            self.is_training = True
            self.use_input_mask = True
            self.use_token_type_ids = False
            self.use_labels = False
            self.vocab_size = 99
            self.hidden_size = 32
            self.num_hidden_layers = 5
            self.num_attention_heads = 4
            self.intermediate_size = 37
            self.hidden_act = 'gelu'
            self.hidden_dropout_prob = 0.1
            self.attention_probs_dropout_prob = 0.1
            self.max_position_embeddings = 1024

            #self.e

            #self.type_sequence_label_size = 16
            #self.initializer_range = 0.02
            #self.num_labels = 3
            #self.num_choices = 4
            #self.scope = None

        def prepare_config_and_inputs(self):
            input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

            input_mask = None
            if self.use_input_mask:
                input_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

            token_type_ids = None
            if self.use_token_type_ids:
                token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)


            config = BARTConfig(
                vocab_size=self.vocab_size,
                d_model=self.hidden_size,
                encoder_layers=self.num_hidden_layers,
                decoder_layers=self.num_hidden_layers,
                encoder_attention_heads=self.num_attention_heads,
                decoder_attention_heads=self.num_attention_heads,
                encoder_ffn_dim=self.intermediate_size,
                decoder_ffn_dim=self.intermediate_size,
                hidden_act=self.hidden_act,
                dropout=self.hidden_dropout_prob,
                attention_dropout=self.attention_probs_dropout_prob,
                max_position_embeddings=self.max_position_embeddings,
                #type_vocab_size=self.type_vocab_size,
                #initializer_range=self.initializer_range,
            )
            sequence_labels = None
            token_labels = None
            choice_labels = None
            return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

        def prepare_config_and_inputs_for_common(self):
            config_and_inputs = self.prepare_config_and_inputs()
            (config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels,) = config_and_inputs
            return config, {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": input_mask}

        def check_loss_output(self, result):
            self.parent.assertListEqual(list(result["loss"].size()), [])

        def create_and_check_bart_forward(
            self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
        ):
            model = BARTModel(config=config)
            model.to(torch_device)
            model.eval()
            _ = model(input_ids, attention_mask=input_mask)  # check that attention_mask doesnt break or something
            decoder_features, = model(input_ids)
            self.assertTrue(isinstance(decoder_features, torch.Tensor)) # no hidden states or attentions
            self.parent.assertEqual(
                decoder_features.size(), (self.batch_size, self.seq_length, self.hidden_size,)
            )




    def setUp(self):
        self.model_tester = BARTModelTest.ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=BARTConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_bart_forward(*config_and_inputs)

    # def test_for_masked_lm(self):
    #     config_and_inputs = self.model_tester.prepare_config_and_inputs()
    #     self.model_tester.create_and_check_roberta_for_masked_lm(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in list(BART_PRETRAINED_MODEL_ARCHIVE_MAP.keys())[:1]:
            model = BARTModel.from_pretrained(model_name, cache_dir=CACHE_DIR)
            self.assertIsNotNone(model)


class BartModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_no_head(self):
        model = BARTModel.from_pretrained("bart-large")
        input_ids = torch.tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        output = model(input_ids)[0]
        expected_shape = torch.Size((1, 11, 50265))
        self.assertEqual(output.shape, expected_shape)
        # compare the actual values for a slice.
        expected_slice = torch.Tensor(
            [[[33.8843, -4.3107, 22.7779], [4.6533, -2.8099, 13.6252], [1.8222, -3.6898, 8.8600]]]
        )
        self.assertTrue(torch.allclose(output[:, :3, :3], expected_slice, atol=1e-3))

    # @slow
    # def test_inference_classification_head(self):
    #     model = RobertaForSequenceClassification.from_pretrained("roberta-large-mnli")
    #
    #     input_ids = torch.tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
    #     output = model(input_ids)[0]
    #     expected_shape = torch.Size((1, 3))
    #     self.assertEqual(output.shape, expected_shape)
    #     expected_tensor = torch.Tensor([[-0.9469, 0.3913, 0.5118]])
    #     self.assertTrue(torch.allclose(output, expected_tensor, atol=1e-3))
