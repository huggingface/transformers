# coding=utf-8
# Copyright 2020 Huggingface
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


import os
import tempfile
import unittest

from transformers import is_torch_available

from .test_configuration_common import ConfigTester
from .test_modeling_common import ModelTesterMixin
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
    test_torchscript = False  # TODO(SS): may want to fix this
    test_resize_embeddings = False  # TODO(SS): may want to fix this
    test_head_masking = False  # TODO(SS): may want to fix this
    is_encoder_decoder = True

    class ModelTester(object):
        def __init__(
            self, parent,
        ):
            self.parent = parent
            self.batch_size = 13
            self.seq_length = 7
            self.is_training = True
            # self.use_token_type_ids = False
            self.use_labels = False
            self.vocab_size = 99
            self.hidden_size = 32
            self.num_hidden_layers = 5
            self.num_attention_heads = 4
            self.intermediate_size = 37
            self.hidden_act = "gelu"
            self.hidden_dropout_prob = 0.1
            self.attention_probs_dropout_prob = 0.1
            self.max_position_embeddings = 12
            torch.manual_seed(0)

            # self.e

            # self.type_sequence_label_size = 16
            # self.initializer_range = 0.02
            # self.num_labels = 3
            # self.num_choices = 4
            # self.scope = None

        def prepare_config_and_inputs(self):
            input_ids = torch.Tensor(
                [
                    [41, 82, 10, 2, 83, 74, 45],
                    [15, 83, 19, 13, 44, 62, 18],
                    [61, 65, 92, 14, 18, 65, 13],
                    [6, 57, 89, 14, 54, 55, 54],
                    [95, 0, 47, 28, 71, 77, 86],
                    [52, 29, 51, 91, 12, 52, 57],
                    [23, 88, 28, 21, 47, 80, 0],
                    [96, 59, 49, 54, 0, 47, 8],
                    [59, 9, 90, 0, 93, 44, 92],
                    [42, 51, 24, 94, 40, 7, 6],
                    [37, 72, 79, 45, 5, 12, 17],
                    [2, 9, 64, 29, 28, 62, 31],
                    [44, 7, 32, 39, 1, 43, 29],
                ]
            ).long()
            self.parent.assertEqual(input_ids.size(), (self.batch_size, self.seq_length))
            self.parent.assertGreaterEqual(self.vocab_size, input_ids.max().item())

            # input_ids = INPUT_IDS# ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
            input_mask = torch.Tensor(
                [
                    [0, 0, 0, 0, 0, 0, 1],
                    [1, 1, 0, 1, 1, 0, 1],
                    [0, 1, 0, 0, 0, 1, 1],
                    [1, 1, 1, 0, 0, 1, 0],
                    [1, 1, 1, 0, 1, 0, 0],
                    [0, 1, 1, 0, 0, 1, 1],
                    [1, 0, 1, 0, 1, 1, 0],
                    [1, 0, 1, 0, 1, 0, 0],
                    [0, 0, 1, 1, 1, 0, 1],
                    [0, 0, 1, 0, 1, 0, 0],
                    [1, 0, 1, 1, 1, 0, 1],
                    [1, 0, 0, 0, 0, 1, 1],
                    [0, 0, 1, 0, 1, 0, 0],
                ]
            ).long()
            # input_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

            config = BARTConfig(
                vocab_size=self.vocab_size,
                d_model=self.hidden_size,
                encoder_layers=self.num_hidden_layers,
                decoder_layers=self.num_hidden_layers,
                encoder_attention_heads=self.num_attention_heads,
                decoder_attention_heads=self.num_attention_heads,
                encoder_ffn_dim=self.intermediate_size,
                decoder_ffn_dim=self.intermediate_size,
                dropout=self.hidden_dropout_prob,
                attention_dropout=self.attention_probs_dropout_prob,
                max_position_embeddings=self.max_position_embeddings,
            )
            sequence_labels = None
            token_labels = None
            choice_labels = None
            token_type_ids = None
            return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

        def prepare_config_and_inputs_for_common(self):
            config_and_inputs = self.prepare_config_and_inputs()
            (
                config,
                input_ids,
                token_type_ids,
                input_mask,
                sequence_labels,
                token_labels,
                choice_labels,
            ) = config_and_inputs
            return (
                config,
                {
                    "input_ids": input_ids,
                    "token_type_ids": token_type_ids,
                    "attention_mask": input_mask,
                    "encoder_input_ids": input_ids,
                    "decoder_input_ids": input_ids,  # HACK(SS): not clear which I'm supposed to do
                },
            )

        def check_loss_output(self, result):
            self.parent.assertListEqual(list(result["loss"].size()), [])

    def setUp(self):
        self.model_tester = BARTModelTest.ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=BARTConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    # def test_initialization(self):

    def test_model(self):
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = self.model_tester.prepare_config_and_inputs()

        model = BARTModel(config=config)
        model.to(torch_device)
        model.eval()
        # test init
        self.assertTrue((model.encoder.embed_tokens.weight == model.shared.weight).all().item())

        def _check_var(module):
            self.assertAlmostEqual(torch.std(module.weight).item(), config.init_std, 2)

        _check_var(model.encoder.embed_tokens)
        _check_var(model.encoder.layers[0].self_attn.k_proj)
        _check_var(model.encoder.layers[0].fc1)
        _check_var(model.encoder.embed_positions)

        decoder_features_with_mask, _ = model(
            input_ids, attention_mask=input_mask
        )  # check that attention_mask doesnt break or something
        decoder_features, enc_features = model(input_ids)
        self.assertTrue(isinstance(decoder_features, torch.Tensor))  # no hidden states or attentions
        self.assertEqual(
            decoder_features.size(), (self.model_tester.batch_size, self.model_tester.seq_length, config.d_model)
        )
        self.assertTrue((decoder_features_with_mask == decoder_features).all().item())

        # import numpy as np
        # last_few_features = decoder_features_with_mask.detach().contiguous().view(-1,)[-3:].numpy()
        # expected_result = np.array([0.688, 0.533, -0.663])
        # np.testing.assert_almost_equal(last_few_features, expected_result, 3)

    def test_save_load_strict(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model2, info = model_class.from_pretrained(tmpdirname, output_loading_info=True)
            self.assertEqual(info["missing_keys"], [])

    # def test_for_masked_lm(self):
    #     config_and_inputs = self.model_tester.prepare_config_and_inputs()
    #     self.model_tester.create_and_check_roberta_for_masked_lm(*config_and_inputs)

    @slow
    @unittest.skipUnless(os.path.exists("/Users/shleifer"), "Placeholder for pretrained check")
    def test_forward_pass_same(self):
        # TODO(SS): delete this
        import numpy as np

        cfg = BARTConfig()
        model = BARTModel(config=cfg)

        model.load_state_dict(torch.load("/Users/shleifer/upgraded_bart_model.pt"))
        model.eval()
        tokens = torch.Tensor([0, 30086, 38, 437, 13049, 2]).long()
        decoder_features = model(tokens)[0]
        # expected_result = [0.688, 0.533, -0.663])
        last_few_features = decoder_features.detach().contiguous().view(-1,)[:5].numpy()
        expected_result = np.array([0.3997, 0.8051, -1.5407, -0.0942, 0.2665])
        np.testing.assert_almost_equal(last_few_features, expected_result, 3)


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

    @slow
    def test_model_from_pretrained(self):
        for model_name in list(BART_PRETRAINED_MODEL_ARCHIVE_MAP.keys())[:1]:
            model = BARTModel.from_pretrained(model_name, cache_dir=CACHE_DIR)
            self.assertIsNotNone(model)

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
