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
""" Testing suite for the PyTorch TimeSeriesTransformer model. """

import copy
import inspect
import tempfile
import unittest

from transformers import MODEL_MAPPING, is_torch_available
from transformers.models.auto import get_values
from transformers.testing_utils import require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch

    from transformers import (
        TimeSeriesTransformerConfig,
        TimeSeriesTransformerForPrediction,
        TimeSeriesTransformerModel,
    )
    from transformers.models.time_series_transformer.modeling_time_series_transformer import (
        TimeSeriesTransformerDecoder,
        TimeSeriesTransformerEncoder,
    )


@require_torch
class TimeSeriesTransformerModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        prediction_length=7,
        context_length=14,
        cardinality=19,
        embedding_dimension=5,
        num_time_features=4,
        is_training=True,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        lags_seq=[1, 2, 3, 4, 5],
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.cardinality = cardinality
        self.num_time_features = num_time_features
        self.lags_seq = lags_seq
        self.embedding_dimension = embedding_dimension
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob

        self.encoder_seq_length = context_length
        self.key_length = context_length
        self.decoder_seq_length = prediction_length

    def prepare_config_and_inputs(self):
        _past_length = self.context_length + max(self.lags_seq)

        feat_static_cat = ids_tensor([self.batch_size, 1], self.cardinality)
        feat_static_real = floats_tensor([self.batch_size, 1])

        past_time_feat = floats_tensor([self.batch_size, _past_length, self.num_time_features])
        past_target = floats_tensor([self.batch_size, _past_length])
        past_observed_values = floats_tensor([self.batch_size, _past_length])

        # decoder inputs
        future_time_feat = floats_tensor([self.batch_size, self.prediction_length, self.num_time_features])
        future_target = floats_tensor([self.batch_size, self.prediction_length])

        config = TimeSeriesTransformerConfig(
            encoder_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            encoder_attention_heads=self.num_attention_heads,
            decoder_attention_heads=self.num_attention_heads,
            encoder_ffn_dim=self.intermediate_size,
            decoder_ffn_dim=self.intermediate_size,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            lags_seq=self.lags_seq,
            num_time_features=self.num_time_features,
            num_feat_static_cat=1,
            cardinality=[self.cardinality],
            embedding_dimension=[self.embedding_dimension],
        )

        inputs_dict = {
            "past_target": past_target,
            "feat_static_cat": feat_static_cat,
            "feat_static_real": feat_static_real,
            "past_time_feat": past_time_feat,
            "future_time_feat": future_time_feat,
            "past_observed_values": past_observed_values,
            "future_target": future_target,
        }
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def check_encoder_decoder_model_standalone(self, config, inputs_dict):
        model = TimeSeriesTransformerModel(config=config).to(torch_device).eval()
        outputs = model(**inputs_dict)

        encoder_last_hidden_state = outputs.encoder_last_hidden_state
        last_hidden_state = outputs.last_hidden_state

        with tempfile.TemporaryDirectory() as tmpdirname:
            encoder = model.get_encoder()
            encoder.save_pretrained(tmpdirname)
            encoder = TimeSeriesTransformerEncoder.from_pretrained(tmpdirname).to(torch_device)

        transformer_inputs, _, _ = model.create_network_inputs(**inputs_dict)
        enc_input = transformer_inputs[:, : config.context_length, ...]
        dec_input = transformer_inputs[:, config.context_length :, ...]

        encoder_last_hidden_state_2 = encoder(inputs_embeds=enc_input)[0]

        self.parent.assertTrue((encoder_last_hidden_state_2 - encoder_last_hidden_state).abs().max().item() < 1e-3)

        with tempfile.TemporaryDirectory() as tmpdirname:
            decoder = model.get_decoder()
            decoder.save_pretrained(tmpdirname)
            decoder = TimeSeriesTransformerDecoder.from_pretrained(tmpdirname).to(torch_device)

        last_hidden_state_2 = decoder(
            inputs_embeds=dec_input,
            encoder_hidden_states=encoder_last_hidden_state,
        )[0]

        self.parent.assertTrue((last_hidden_state_2 - last_hidden_state).abs().max().item() < 1e-3)


@require_torch
class TimeSeriesTransformerModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (
        (TimeSeriesTransformerModel, TimeSeriesTransformerForPrediction) if is_torch_available() else ()
    )
    all_generative_model_classes = (TimeSeriesTransformerForPrediction,) if is_torch_available() else ()
    is_encoder_decoder = True
    test_pruning = False
    test_head_masking = False
    test_missing_keys = False
    test_torchscript = False
    test_inputs_embeds = False
    test_model_common_attributes = False

    def setUp(self):
        self.model_tester = TimeSeriesTransformerModelTester(self)
        self.config_tester = ConfigTester(self, config_class=TimeSeriesTransformerConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_save_load_strict(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        for model_class in self.all_model_classes:
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model2, info = model_class.from_pretrained(tmpdirname, output_loading_info=True)
            self.assertEqual(info["missing_keys"], [])

    def test_encoder_decoder_model_standalone(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.check_encoder_decoder_model_standalone(*config_and_inputs)

    # Ignore since we have no tokens embeddings
    def test_resize_tokens_embeddings(self):
        pass

    # # Input is 'feat_static_cat' not 'input_ids'
    def test_model_main_input_name(self):
        model_signature = inspect.signature(getattr(TimeSeriesTransformerModel, "forward"))
        # The main input is the name of the argument after `self`
        observed_main_input_name = list(model_signature.parameters.keys())[1]
        self.assertEqual(TimeSeriesTransformerModel.main_input_name, observed_main_input_name)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = [
                "past_target",
                "feat_static_cat",
                "feat_static_real",
                "past_time_feat",
                "past_observed_values",
                "future_time_feat",
                "future_target",
            ]

            expected_arg_names.extend(
                [
                    "future_observed_values",
                    "encoder_outputs",
                    "use_cache",
                    "output_attentions",
                    "output_hidden_states",
                    "return_dict",
                ]
                if "future_observed_values" in arg_names
                else [
                    "attention_mask",
                    "decoder_attention_mask",
                    "head_mask",
                    "decoder_head_mask",
                    "cross_attn_head_mask",
                    "encoder_outputs",
                    "past_key_values",
                    "output_hidden_states",
                    "use_cache",
                    "output_attentions",
                    "return_dict",
                ]
            )

            self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        seq_len = getattr(self.model_tester, "seq_length", None)
        decoder_seq_length = getattr(self.model_tester, "decoder_seq_length", seq_len)
        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", seq_len)
        decoder_key_length = getattr(self.model_tester, "decoder_key_length", decoder_seq_length)
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            self.assertListEqual(
                list(attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
            )
            out_len = len(outputs)

            correct_outlen = 6

            if "last_hidden_state" in outputs:
                correct_outlen += 1

            if "past_key_values" in outputs:
                correct_outlen += 1  # past_key_values have been returned

            if "loss" in outputs:
                correct_outlen += 1

            if "params" in outputs:
                correct_outlen += 1

            self.assertEqual(out_len, correct_outlen)

            # decoder attentions
            decoder_attentions = outputs.decoder_attentions
            self.assertIsInstance(decoder_attentions, (list, tuple))
            self.assertEqual(len(decoder_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(decoder_attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, decoder_seq_length, decoder_key_length],
            )

            # cross attentions
            cross_attentions = outputs.cross_attentions
            self.assertIsInstance(cross_attentions, (list, tuple))
            self.assertEqual(len(cross_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(cross_attentions[0].shape[-3:]),
                [
                    self.model_tester.num_attention_heads,
                    decoder_seq_length,
                    encoder_key_length,
                ],
            )

        # Check attention is always last and order is fine
        inputs_dict["output_attentions"] = True
        inputs_dict["output_hidden_states"] = True
        model = model_class(config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            outputs = model(**self._prepare_for_class(inputs_dict, model_class))

        self.assertEqual(out_len + 2, len(outputs))

        self_attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions

        self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
        self.assertListEqual(
            list(self_attentions[0].shape[-3:]),
            [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
        )


@require_torch
@slow
class TimeSeriesTransformerModelIntegrationTests(unittest.TestCase):
    def test_inference_no_head(self):
        model = TimeSeriesTransformerModel.from_pretrained("huggingface/time-series-transformer-tourism-monthly").to(
            torch_device
        )

        raise NotImplementedError("To do")

    def test_inference_head(self):
        # model = TimeSeriesTransformerForPrediction.from_pretrained("huggingface/time-series-transformer-tourism-monthly").to(torch_device)

        raise NotImplementedError("To do")

    def test_seq_to_seq_generation(self):
        raise NotImplementedError("Generation not implemented yet")
