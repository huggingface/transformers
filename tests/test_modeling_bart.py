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


import tempfile
import unittest

from transformers import is_torch_available

from .test_configuration_common import ConfigTester
from .test_modeling_common import ModelTesterMixin, ids_tensor
from .utils import CACHE_DIR, require_torch, slow, torch_device


if is_torch_available():
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        BartModel,
        BartForMaskedLM,
        BartForSequenceClassification,
        BartConfig,
    )
    from transformers.modeling_bart import BART_PRETRAINED_MODEL_ARCHIVE_MAP


@require_torch
class BARTModelTest(ModelTesterMixin, unittest.TestCase):

    all_model_classes = (BartModel, BartForMaskedLM) if is_torch_available() else ()
    is_encoder_decoder = True
    test_resize_embeddings = True
    # TODO(SS): fix the below in a separate PR
    test_pruning = False
    test_torchscript = False
    test_head_masking = False

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

        def prepare_config_and_inputs(self):
            input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
            input_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

            config = BartConfig(
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
            decoder_lm_labels = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
            return (
                config,
                input_ids,
                token_type_ids,
                input_mask,
                sequence_labels,
                token_labels,
                choice_labels,
                decoder_lm_labels,
            )

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
                decoder_lm_labels,
            ) = config_and_inputs
            return (
                config,
                {
                    #"input_ids": input_ids,
                    "token_type_ids": token_type_ids,
                    "attention_mask": input_mask,
                    "encoder_input_ids": input_ids,
                    "decoder_input_ids": input_ids,
                    "decoder_lm_labels": decoder_lm_labels,
                },
            )

        def check_loss_output(self, result):
            self.parent.assertListEqual(list(result["loss"].size()), [])

    def setUp(self):
        self.model_tester = BARTModelTest.ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=BartConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        (config, input_ids, token_type_ids, input_mask, *unused) = self.model_tester.prepare_config_and_inputs()

        model = BartModel(config)
        model.to(torch_device)
        model.eval()
        # test init
        self.assertTrue((model.encoder.embed_tokens.weight == model.shared.weight).all().item())

        def _check_var(module):
            """Check that we initialized various parameters from N(0, config.init_std)."""
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
        # self.assertTrue((decoder_features_with_mask == decoder_features).all().item())  # TODO(SS): BUG?

    def test_save_load_strict(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model2, info = model_class.from_pretrained(tmpdirname, output_loading_info=True)
            self.assertEqual(info["missing_keys"], [])

    def test_incremental_state(self):
        pass
        # TODO(SS), separate PR: try to generate with model using incremental state


@require_torch
class BartSequenceClassifTest(unittest.TestCase):
    batch_size = 13

    def test_forward(self):
        input_ids = torch.Tensor(
            [
                [71, 82, 18, 33, 46, 91, 2],
                [68, 34, 26, 58, 30, 82, 2],
                [5, 97, 17, 39, 94, 40, 2],
                [76, 83, 94, 25, 70, 78, 2],
                [87, 59, 41, 35, 48, 66, 2],
                [55, 13, 16, 58, 5, 2, 1],  # note padding
                [64, 27, 31, 51, 12, 75, 2],
                [52, 64, 86, 17, 83, 39, 2],
                [48, 61, 9, 24, 71, 82, 2],
                [26, 1, 60, 48, 22, 13, 2],
                [21, 5, 62, 28, 14, 76, 2],
                [45, 98, 37, 86, 59, 48, 2],
                [70, 70, 50, 9, 28, 0, 2],
            ]
        ).long()

        config = BartConfig(
            vocab_size=99,
            d_model=24,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=32,
            decoder_ffn_dim=32,
            max_position_embeddings=48,
        )
        model = BartForSequenceClassification(config)
        outputs = model(input_ids=input_ids)
        logits = outputs[0]
        expected_shape = torch.Size((self.batch_size, config.num_labels))
        self.assertEqual(logits.shape, expected_shape)

        lm_model = BartForMaskedLM(config)
        output = lm_model(input_ids=input_ids)[0]
        expected_shape = (self.batch_size, input_ids.shape[1], config.vocab_size)
        self.assertEqual(output.shape, expected_shape)


class BartModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_no_head(self):
        model = BartModel.from_pretrained("bart-large")
        input_ids = torch.tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        with torch.no_grad():
            output = model(input_ids=input_ids)[0]
        expected_shape = torch.Size((1, 11, 1024))
        self.assertEqual(output.shape, expected_shape)
        expected_slice = torch.Tensor(
            [[0.7144, 0.8143, -1.2813], [0.7144, 0.8143, -1.2813], [-0.0467, 2.5911, -2.1845]]
        )
        self.assertTrue(torch.allclose(output[:, :3, :3], expected_slice, atol=1e-3))

    @slow
    def test_mnli_inference(self):
        model = AutoModelForSequenceClassification.from_pretrained("bart-large-mnli")
        input_ids = torch.Tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]]).long()
        with torch.no_grad():
            logits = model(input_ids)[0]
        expected_shape = torch.Size((1, 3))
        self.assertEqual(logits.shape, expected_shape)
        expected_slice = torch.Tensor([[0.1907, 1.4342, -1.0289]])
        self.assertTrue(torch.allclose(logits, expected_slice, atol=1e-3))

    @slow
    def test_model_from_pretrained(self):
        for model_name in list(BART_PRETRAINED_MODEL_ARCHIVE_MAP.keys())[:1]:
            model = BartModel.from_pretrained(model_name, cache_dir=CACHE_DIR)
            self.assertIsNotNone(model)
