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
import tempfile
import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device
from transformers.utils import cached_property

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
        future_observed_values = floats_tensor([self.batch_size, self.prediction_length])

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
            "feat_static_cat": feat_static_cat,
            "feat_static_real": feat_static_real,
            "past_time_feat": past_time_feat,
            "past_target": past_target,
            "future_time_feat": future_time_feat,
            "past_observed_values": past_observed_values,
            "future_target": future_target,
        }
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def create_and_check_decoder_model_past_large_inputs(self, config, inputs_dict):
        model = TimeSeriesTransformerModel(config=config).get_decoder().to(torch_device).eval()
        input_ids = inputs_dict["input_ids"]
        attention_mask = inputs_dict["attention_mask"]

        # first forward pass
        outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)

        output, past_key_values = outputs.to_tuple()

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_attn_mask = ids_tensor((self.batch_size, 3), 2)

        # append to next input_ids and
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        next_attention_mask = torch.cat([attention_mask, next_attn_mask], dim=-1)

        output_from_no_past = model(next_input_ids, attention_mask=next_attention_mask)["last_hidden_state"]
        output_from_past = model(next_tokens, attention_mask=next_attention_mask, past_key_values=past_key_values)[
            "last_hidden_state"
        ]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-2))

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

        encoder_last_hidden_state_2 = encoder(input_ids=enc_input)[0]

        self.parent.assertTrue((encoder_last_hidden_state_2 - encoder_last_hidden_state).abs().max().item() < 1e-3)

        with tempfile.TemporaryDirectory() as tmpdirname:
            decoder = model.get_decoder()
            decoder.save_pretrained(tmpdirname)
            decoder = TimeSeriesTransformerDecoder.from_pretrained(tmpdirname).to(torch_device)

        last_hidden_state_2 = decoder(
            input_ids=dec_input,
            encoder_hidden_states=encoder_last_hidden_state,
        )[0]

        self.parent.assertTrue((last_hidden_state_2 - last_hidden_state).abs().max().item() < 1e-3)


@require_torch
class TimeSeriesTransformerModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            TimeSeriesTransformerModel,
            TimeSeriesTransformerForPrediction,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (TimeSeriesTransformerForPrediction,) if is_torch_available() else ()
    is_encoder_decoder = True
    test_pruning = False
    test_head_masking = False
    test_missing_keys = False
    test_torchscript = False

    def setUp(self):
        self.model_tester = TimeSeriesTransformerModelTester(self)
        self.config_tester = ConfigTester(self, config_class=TimeSeriesTransformerConfig)

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

    # def test_generate_fp16(self):
    #     config, input_dict = self.model_tester.prepare_config_and_inputs()
    #     input_ids = input_dict["input_ids"]
    #     attention_mask = input_ids.ne(1).to(torch_device)
    #     model = TimeSeriesTransformerForPrediction(config).eval().to(torch_device)
    #     if torch_device == "cuda":
    #         model.half()
    #     model.generate(input_ids, attention_mask=attention_mask)
    #     model.generate(num_beams=4, do_sample=True, early_stopping=False, num_return_sequences=3)


def assert_tensors_close(a, b, atol=1e-12, prefix=""):
    """If tensors have different shapes, different values or a and b are not both tensors, raise a nice Assertion error."""
    if a is None and b is None:
        return True
    try:
        if torch.allclose(a, b, atol=atol):
            return True
        raise
    except Exception:
        pct_different = (torch.gt((a - b).abs(), atol)).float().mean().item()
        if a.numel() > 100:
            msg = f"tensor values are {pct_different:.1%} percent different."
        else:
            msg = f"{a} != {b}"
        if prefix:
            msg = prefix + ": " + msg
        raise AssertionError(msg)


def _long_tensor(tok_lst):
    return torch.tensor(tok_lst, dtype=torch.long, device=torch_device)


TOLERANCE = 1e-4


@require_torch
@slow
class TimeSeriesTransformerModelIntegrationTests(unittest.TestCase):
    # @cached_property
    # def default_tokenizer(self):
    #     return TimeSeriesTransformerTokenizer.from_pretrained("huggingface/tst-ett")

    def test_inference_no_head(self):
        model = TimeSeriesTransformerModel.from_pretrained("huggingface/tst-ett").to(torch_device)
        input_ids = _long_tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        decoder_input_ids = _long_tensor([[2, 0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588]])
        inputs_dict = prepare_time_series_transformer_inputs_dict(model.config, input_ids, decoder_input_ids)
        with torch.no_grad():
            output = model(**inputs_dict)[0]
        expected_shape = torch.Size((1, 11, 1024))
        self.assertEqual(output.shape, expected_shape)
        # change to expected output here
        expected_slice = torch.tensor(
            [[0.7144, 0.8143, -1.2813], [0.7144, 0.8143, -1.2813], [-0.0467, 2.5911, -2.1845]], device=torch_device
        )
        self.assertTrue(torch.allclose(output[:, :3, :3], expected_slice, atol=TOLERANCE))

    def test_inference_head(self):
        model = TimeSeriesTransformerForPrediction.from_pretrained("huggingface/tst-ett").to(torch_device)

        # change to intended input
        input_ids = _long_tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        decoder_input_ids = _long_tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        inputs_dict = prepare_time_series_transformer_inputs_dict(model.config, input_ids, decoder_input_ids)
        with torch.no_grad():
            output = model(**inputs_dict)[0]
        expected_shape = torch.Size((1, 11, model.config.vocab_size))
        self.assertEqual(output.shape, expected_shape)
        # change to expected output here
        expected_slice = torch.tensor(
            [[0.7144, 0.8143, -1.2813], [0.7144, 0.8143, -1.2813], [-0.0467, 2.5911, -2.1845]], device=torch_device
        )
        self.assertTrue(torch.allclose(output[:, :3, :3], expected_slice, atol=TOLERANCE))

    def test_seq_to_seq_generation(self):
        raise NotImplementedError("Generation not implemented yet")
