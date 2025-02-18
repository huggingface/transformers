# coding=utf-8
# Copyright 2020 HuggingFace Inc. team.
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


from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from transformers import is_tf_available, is_torch_available
from transformers.testing_utils import require_tf, slow

from ...test_modeling_tf_common import ids_tensor
from ..bert.test_modeling_tf_bert import TFBertModelTester
from ..gpt2.test_modeling_tf_gpt2 import TFGPT2ModelTester
from ..rembert.test_modeling_tf_rembert import TFRemBertModelTester
from ..roberta.test_modeling_tf_roberta import TFRobertaModelTester


if is_tf_available():
    from transformers import (
        AutoConfig,
        AutoTokenizer,
        EncoderDecoderConfig,
        TFAutoModel,
        TFAutoModelForCausalLM,
        TFBertLMHeadModel,
        TFBertModel,
        TFEncoderDecoderModel,
        TFGPT2LMHeadModel,
        TFRemBertForCausalLM,
        TFRemBertModel,
        TFRobertaForCausalLM,
        TFRobertaModel,
    )
    from transformers.modeling_tf_outputs import TFBaseModelOutput

if is_torch_available():
    pass


@require_tf
class TFEncoderDecoderMixin:
    def get_encoder_decoder_model(self, config, decoder_config):
        raise NotImplementedError

    def prepare_config_and_inputs(self):
        raise NotImplementedError

    def get_pretrained_model(self):
        raise NotImplementedError

    def check_encoder_decoder_model_from_pretrained_configs(
        self,
        config,
        input_ids,
        attention_mask,
        encoder_hidden_states,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        **kwargs,
    ):
        encoder_decoder_config = EncoderDecoderConfig.from_encoder_decoder_configs(config, decoder_config)
        self.assertTrue(encoder_decoder_config.decoder.is_decoder)

        enc_dec_model = TFEncoderDecoderModel(encoder_decoder_config)

        self.assertTrue(enc_dec_model.config.is_encoder_decoder)

        outputs_encoder_decoder = enc_dec_model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            kwargs=kwargs,
        )

        self.assertEqual(
            outputs_encoder_decoder["logits"].shape, (decoder_input_ids.shape + (decoder_config.vocab_size,))
        )
        self.assertEqual(
            outputs_encoder_decoder["encoder_last_hidden_state"].shape, (input_ids.shape + (config.hidden_size,))
        )

    def check_encoder_decoder_model(
        self,
        config,
        input_ids,
        attention_mask,
        encoder_hidden_states,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        **kwargs,
    ):
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = TFEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        self.assertTrue(enc_dec_model.config.decoder.is_decoder)
        self.assertTrue(enc_dec_model.config.decoder.add_cross_attention)
        self.assertTrue(enc_dec_model.config.is_encoder_decoder)

        outputs_encoder_decoder = enc_dec_model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            kwargs=kwargs,
        )
        self.assertEqual(
            outputs_encoder_decoder["logits"].shape, (decoder_input_ids.shape + (decoder_config.vocab_size,))
        )
        self.assertEqual(
            outputs_encoder_decoder["encoder_last_hidden_state"].shape, (input_ids.shape + (config.hidden_size,))
        )

        encoder_outputs = TFBaseModelOutput(last_hidden_state=encoder_hidden_states)
        outputs_encoder_decoder = enc_dec_model(
            input_ids=None,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            kwargs=kwargs,
        )

        self.assertEqual(
            outputs_encoder_decoder["logits"].shape, (decoder_input_ids.shape + (decoder_config.vocab_size,))
        )
        self.assertEqual(
            outputs_encoder_decoder["encoder_last_hidden_state"].shape, (input_ids.shape + (config.hidden_size,))
        )

    def check_encoder_decoder_model_from_pretrained(
        self,
        config,
        input_ids,
        attention_mask,
        encoder_hidden_states,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        return_dict,
        **kwargs,
    ):
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        kwargs = {"encoder_model": encoder_model, "decoder_model": decoder_model, "return_dict": return_dict}
        enc_dec_model = TFEncoderDecoderModel.from_encoder_decoder_pretrained(**kwargs)
        outputs_encoder_decoder = enc_dec_model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            return_dict=True,
            kwargs=kwargs,
        )

        self.assertEqual(
            outputs_encoder_decoder["logits"].shape, (decoder_input_ids.shape + (decoder_config.vocab_size,))
        )
        self.assertEqual(
            outputs_encoder_decoder["encoder_last_hidden_state"].shape, (input_ids.shape + (config.hidden_size,))
        )

    def check_save_and_load(
        self,
        config,
        input_ids,
        attention_mask,
        encoder_hidden_states,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        **kwargs,
    ):
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = TFEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)

        outputs = enc_dec_model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            kwargs=kwargs,
        )
        out_2 = np.array(outputs[0])
        out_2[np.isnan(out_2)] = 0

        with tempfile.TemporaryDirectory() as tmpdirname:
            enc_dec_model.save_pretrained(tmpdirname)
            enc_dec_model = TFEncoderDecoderModel.from_pretrained(tmpdirname)

            after_outputs = enc_dec_model(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                kwargs=kwargs,
            )
            out_1 = np.array(after_outputs[0])
            out_1[np.isnan(out_1)] = 0
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 1e-5)

    def check_encoder_decoder_model_labels(
        self,
        config,
        input_ids,
        attention_mask,
        encoder_hidden_states,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        labels,
        **kwargs,
    ):
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = TFEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)

        outputs_encoder_decoder = enc_dec_model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            kwargs=kwargs,
        )

        # Make sure `loss` exist
        self.assertIn("loss", outputs_encoder_decoder)

        batch_size, seq_len = decoder_input_ids.shape
        expected_shape = (batch_size, seq_len, decoder_config.vocab_size)
        self.assertEqual(outputs_encoder_decoder["logits"].shape, expected_shape)
        self.assertEqual(
            outputs_encoder_decoder["encoder_last_hidden_state"].shape, (input_ids.shape + (config.hidden_size,))
        )

    def _check_output_with_attentions(
        self, outputs_encoder_decoder, config, input_ids, decoder_config, decoder_input_ids
    ):
        encoder_attentions = outputs_encoder_decoder["encoder_attentions"]
        self.assertEqual(len(encoder_attentions), config.num_hidden_layers)

        self.assertEqual(
            encoder_attentions[0].shape[-3:], (config.num_attention_heads, input_ids.shape[-1], input_ids.shape[-1])
        )

        decoder_attentions = outputs_encoder_decoder["decoder_attentions"]
        num_decoder_layers = (
            decoder_config.num_decoder_layers
            if hasattr(decoder_config, "num_decoder_layers")
            else decoder_config.num_hidden_layers
        )
        self.assertEqual(len(decoder_attentions), num_decoder_layers)

        self.assertEqual(
            decoder_attentions[0].shape[-3:],
            (decoder_config.num_attention_heads, decoder_input_ids.shape[-1], decoder_input_ids.shape[-1]),
        )

        cross_attentions = outputs_encoder_decoder["cross_attentions"]
        self.assertEqual(len(cross_attentions), num_decoder_layers)

        cross_attention_input_seq_len = decoder_input_ids.shape[-1] * (
            1 + (decoder_config.ngram if hasattr(decoder_config, "ngram") else 0)
        )
        self.assertEqual(
            cross_attentions[0].shape[-3:],
            (decoder_config.num_attention_heads, cross_attention_input_seq_len, input_ids.shape[-1]),
        )

    def check_encoder_decoder_model_output_attentions(
        self,
        config,
        input_ids,
        attention_mask,
        encoder_hidden_states,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        **kwargs,
    ):
        # make the decoder inputs a different shape from the encoder inputs to harden the test
        decoder_input_ids = decoder_input_ids[:, :-1]
        decoder_attention_mask = decoder_attention_mask[:, :-1]
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = TFEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        outputs_encoder_decoder = enc_dec_model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            output_attentions=True,
            kwargs=kwargs,
        )
        self._check_output_with_attentions(
            outputs_encoder_decoder, config, input_ids, decoder_config, decoder_input_ids
        )

    def check_encoder_decoder_model_output_attentions_from_config(
        self,
        config,
        input_ids,
        attention_mask,
        encoder_hidden_states,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        **kwargs,
    ):
        # Similar to `check_encoder_decoder_model_output_attentions`, but with `output_attentions` triggered from the
        # config file. Contrarily to most models, changing the model's config won't work -- the defaults are loaded
        # from the inner models' configurations.

        decoder_input_ids = decoder_input_ids[:, :-1]
        decoder_attention_mask = decoder_attention_mask[:, :-1]
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = TFEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        enc_dec_model.config.output_attentions = True  # model config -> won't work
        outputs_encoder_decoder = enc_dec_model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            kwargs=kwargs,
        )
        self.assertTrue(
            all(
                key not in outputs_encoder_decoder
                for key in ["encoder_attentions", "decoder_attentions", "cross_attentions"]
            )
        )

        config.output_attentions = True  # inner model config -> will work
        decoder_config.output_attentions = True
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = TFEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        outputs_encoder_decoder = enc_dec_model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            kwargs=kwargs,
        )
        self._check_output_with_attentions(
            outputs_encoder_decoder, config, input_ids, decoder_config, decoder_input_ids
        )

    def check_encoder_decoder_model_generate(self, input_ids, config, decoder_config, **kwargs):
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = TFEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)

        # Generate until max length
        if hasattr(enc_dec_model.config, "eos_token_id"):
            enc_dec_model.config.eos_token_id = None
        if hasattr(enc_dec_model.config, "decoder") and hasattr(enc_dec_model.config.decoder, "eos_token_id"):
            enc_dec_model.config.decoder.eos_token_id = None
        if hasattr(enc_dec_model.generation_config, "eos_token_id"):
            enc_dec_model.generation_config.eos_token_id = None

        # Bert does not have a bos token id, so use pad_token_id instead
        generated_output = enc_dec_model.generate(
            input_ids, decoder_start_token_id=enc_dec_model.config.decoder.pad_token_id
        )
        self.assertEqual(tuple(generated_output.shape.as_list()), (input_ids.shape[0],) + (decoder_config.max_length,))

    def test_encoder_decoder_model(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.check_encoder_decoder_model(**input_ids_dict)

    def test_encoder_decoder_model_from_pretrained_configs(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.check_encoder_decoder_model_from_pretrained_configs(**input_ids_dict)

    def test_encoder_decoder_model_from_pretrained(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.check_encoder_decoder_model_from_pretrained(**input_ids_dict, return_dict=False)

    def test_encoder_decoder_model_from_pretrained_return_dict(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.check_encoder_decoder_model_from_pretrained(**input_ids_dict, return_dict=True)

    def test_save_and_load_from_pretrained(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.check_save_and_load(**input_ids_dict)

    def test_encoder_decoder_model_labels(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.check_encoder_decoder_model_labels(**input_ids_dict)

    def test_encoder_decoder_model_output_attentions(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.check_encoder_decoder_model_output_attentions(**input_ids_dict)

    def test_encoder_decoder_model_output_attentions_from_config(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.check_encoder_decoder_model_output_attentions_from_config(**input_ids_dict)

    def test_encoder_decoder_model_generate(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.check_encoder_decoder_model_generate(**input_ids_dict)

    def assert_almost_equals(self, a: np.ndarray, b: np.ndarray, tol: float):
        diff = np.abs((a - b)).max()
        self.assertLessEqual(diff, tol, f"Difference between torch and tf is {diff} (>= {tol}).")

    def test_model_save_load_from_pretrained(self):
        model_2 = self.get_pretrained_model()
        input_ids = ids_tensor([13, 5], model_2.config.encoder.vocab_size)
        decoder_input_ids = ids_tensor([13, 1], model_2.config.decoder.vocab_size)
        attention_mask = ids_tensor([13, 5], vocab_size=2)

        outputs = model_2(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
        )
        out_2 = np.array(outputs[0])
        out_2[np.isnan(out_2)] = 0

        with tempfile.TemporaryDirectory() as tmp_dirname:
            model_2.save_pretrained(tmp_dirname)
            model_1 = TFEncoderDecoderModel.from_pretrained(tmp_dirname)

            after_outputs = model_1(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
            )
            out_1 = np.array(after_outputs[0])
            out_1[np.isnan(out_1)] = 0
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 1e-5)


@require_tf
class TFBertEncoderDecoderModelTest(TFEncoderDecoderMixin, unittest.TestCase):
    def setUp(self):
        self.encoder_model_tester = TFBertModelTester(self, batch_size=13)
        self.decoder_model_tester = TFBertModelTester(self, batch_size=13)

    def get_pretrained_model(self):
        return TFEncoderDecoderModel.from_encoder_decoder_pretrained(
            "hf-internal-testing/tiny-random-bert",
            "hf-internal-testing/tiny-random-bert",
        )

    def get_encoder_decoder_model(self, config, decoder_config):
        encoder_model = TFBertModel(config, name="encoder")
        decoder_model = TFBertLMHeadModel(decoder_config, name="decoder")
        return encoder_model, decoder_model

    def prepare_config_and_inputs(self):
        encoder_config_and_inputs = self.encoder_model_tester.prepare_config_and_inputs()
        decoder_config_and_inputs = self.decoder_model_tester.prepare_config_and_inputs_for_decoder()
        (
            config,
            input_ids,
            token_type_ids,
            attention_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = encoder_config_and_inputs
        (
            decoder_config,
            decoder_input_ids,
            decoder_token_type_ids,
            decoder_attention_mask,
            decoder_sequence_labels,
            decoder_token_labels,
            decoder_choice_labels,
            encoder_hidden_states,
            encoder_attention_mask,
        ) = decoder_config_and_inputs

        # make sure that cross attention layers are added
        decoder_config.add_cross_attention = True
        #  disable cache for now
        decoder_config.use_cache = False
        return {
            "config": config,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_config": decoder_config,
            "decoder_input_ids": decoder_input_ids,
            "decoder_token_type_ids": decoder_token_type_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_sequence_labels": decoder_sequence_labels,
            "decoder_token_labels": decoder_token_labels,
            "decoder_choice_labels": decoder_choice_labels,
            "encoder_hidden_states": encoder_hidden_states,
            "labels": decoder_token_labels,
        }


@require_tf
class TFGPT2EncoderDecoderModelTest(TFEncoderDecoderMixin, unittest.TestCase):
    def setUp(self):
        self.encoder_model_tester = TFBertModelTester(self, batch_size=13)
        self.decoder_model_tester = TFGPT2ModelTester(self)

    def get_pretrained_model(self):
        return TFEncoderDecoderModel.from_encoder_decoder_pretrained(
            "hf-internal-testing/tiny-random-bert",
            "hf-internal-testing/tiny-random-gpt2",
        )

    def get_encoder_decoder_model(self, config, decoder_config):
        encoder_model = TFBertModel(config, name="encoder")
        decoder_model = TFGPT2LMHeadModel(decoder_config, name="decoder")
        return encoder_model, decoder_model

    def prepare_config_and_inputs(self):
        encoder_config_and_inputs = self.encoder_model_tester.prepare_config_and_inputs()
        decoder_config_and_inputs = self.decoder_model_tester.prepare_config_and_inputs_for_decoder()
        (
            config,
            input_ids,
            token_type_ids,
            attention_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = encoder_config_and_inputs
        (
            decoder_config,
            decoder_input_ids,
            decoder_attention_mask,
            decoder_head_mask,
            decoder_token_type_ids,
            decoder_sequence_labels,
            decoder_token_labels,
            decoder_choice_labels,
            encoder_hidden_states,
            encoder_attention_mask,
        ) = decoder_config_and_inputs

        # make sure that cross attention layers are added
        decoder_config.add_cross_attention = True
        # disable cache for now
        decoder_config.use_cache = False
        return {
            "config": config,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_config": decoder_config,
            "decoder_input_ids": decoder_input_ids,
            "decoder_token_type_ids": decoder_token_type_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_sequence_labels": decoder_sequence_labels,
            "decoder_token_labels": decoder_token_labels,
            "decoder_choice_labels": decoder_choice_labels,
            "encoder_hidden_states": encoder_hidden_states,
            "labels": decoder_token_labels,
        }


@require_tf
class TFRoBertaEncoderDecoderModelTest(TFEncoderDecoderMixin, unittest.TestCase):
    def setUp(self):
        self.encoder_model_tester = TFRobertaModelTester(self)
        self.decoder_model_tester = TFRobertaModelTester(self)

    def get_pretrained_model(self):
        return TFEncoderDecoderModel.from_encoder_decoder_pretrained(
            "hf-internal-testing/tiny-random-roberta",
            "hf-internal-testing/tiny-random-roberta",
        )

    def get_encoder_decoder_model(self, config, decoder_config):
        encoder_model = TFRobertaModel(config, name="encoder")
        decoder_model = TFRobertaForCausalLM(decoder_config, name="decoder")
        return encoder_model, decoder_model

    def prepare_config_and_inputs(self):
        encoder_config_and_inputs = self.encoder_model_tester.prepare_config_and_inputs()
        decoder_config_and_inputs = self.decoder_model_tester.prepare_config_and_inputs_for_decoder()
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = encoder_config_and_inputs
        (
            decoder_config,
            decoder_input_ids,
            decoder_token_type_ids,
            decoder_input_mask,
            decoder_sequence_labels,
            decoder_token_labels,
            decoder_choice_labels,
            encoder_hidden_states,
            encoder_attention_mask,
        ) = decoder_config_and_inputs

        # make sure that cross attention layers are added
        decoder_config.add_cross_attention = True
        #  disable cache for now
        decoder_config.use_cache = False
        return {
            "config": config,
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "decoder_config": decoder_config,
            "decoder_input_ids": decoder_input_ids,
            "decoder_token_type_ids": decoder_token_type_ids,
            "decoder_attention_mask": decoder_input_mask,
            "decoder_sequence_labels": decoder_sequence_labels,
            "decoder_token_labels": decoder_token_labels,
            "decoder_choice_labels": decoder_choice_labels,
            "encoder_hidden_states": encoder_hidden_states,
            "labels": decoder_token_labels,
        }


@require_tf
class TFRembertEncoderDecoderModelTest(TFEncoderDecoderMixin, unittest.TestCase):
    def setUp(self):
        self.encoder_model_tester = TFRemBertModelTester(self)
        self.decoder_model_tester = TFRemBertModelTester(self)

    def get_pretrained_model(self):
        return TFEncoderDecoderModel.from_encoder_decoder_pretrained(
            "hf-internal-testing/tiny-random-rembert",
            "hf-internal-testing/tiny-random-rembert",
        )

    def get_encoder_decoder_model(self, config, decoder_config):
        encoder_model = TFRemBertModel(config, name="encoder")
        decoder_model = TFRemBertForCausalLM(decoder_config, name="decoder")
        return encoder_model, decoder_model

    def prepare_config_and_inputs(self):
        encoder_config_and_inputs = self.encoder_model_tester.prepare_config_and_inputs()
        decoder_config_and_inputs = self.decoder_model_tester.prepare_config_and_inputs_for_decoder()
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = encoder_config_and_inputs
        (
            decoder_config,
            decoder_input_ids,
            decoder_token_type_ids,
            decoder_input_mask,
            decoder_sequence_labels,
            decoder_token_labels,
            decoder_choice_labels,
            encoder_hidden_states,
            encoder_attention_mask,
        ) = decoder_config_and_inputs

        # make sure that cross attention layers are added
        decoder_config.add_cross_attention = True
        #  disable cache for now
        decoder_config.use_cache = False
        return {
            "config": config,
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "decoder_config": decoder_config,
            "decoder_input_ids": decoder_input_ids,
            "decoder_token_type_ids": decoder_token_type_ids,
            "decoder_attention_mask": decoder_input_mask,
            "decoder_sequence_labels": decoder_sequence_labels,
            "decoder_token_labels": decoder_token_labels,
            "decoder_choice_labels": decoder_choice_labels,
            "encoder_hidden_states": encoder_hidden_states,
            "labels": decoder_token_labels,
        }


@require_tf
class TFEncoderDecoderModelTest(unittest.TestCase):
    def get_from_encoderdecoder_pretrained_model(self):
        return TFEncoderDecoderModel.from_encoder_decoder_pretrained(
            "google-bert/bert-base-cased", "google-bert/bert-base-cased"
        )

    def get_decoder_config(self):
        config = AutoConfig.from_pretrained("google-bert/bert-base-cased")
        config.is_decoder = True
        config.add_cross_attention = True
        return config

    def get_encoderdecoder_model(self):
        return TFEncoderDecoderModel.from_pretrained("patrickvonplaten/bert2bert-cnn_dailymail-fp16")

    def get_encoder_decoder_models(self):
        encoder_model = TFBertModel.from_pretrained("google-bert/bert-base-cased", name="encoder")
        decoder_model = TFBertLMHeadModel.from_pretrained(
            "google-bert/bert-base-cased", config=self.get_decoder_config(), name="decoder"
        )
        return {"encoder": encoder_model, "decoder": decoder_model}

    def _check_configuration_tie(self, model):
        assert id(model.decoder.config) == id(model.config.decoder)
        assert id(model.encoder.config) == id(model.config.encoder)

    @slow
    def test_configuration_tie(self):
        model = self.get_from_encoderdecoder_pretrained_model()
        self._check_configuration_tie(model)

        model = TFEncoderDecoderModel(**self.get_encoder_decoder_models())
        self._check_configuration_tie(model)

        # # This should be enabled once we upload the TF version of
        # # "patrickvonplaten/bert2bert-cnn_dailymail-fp16" to the Hub.
        # model = self.get_encoderdecoder_model()
        # self._check_configuration_tie(model)


@require_tf
class TFEncoderDecoderModelSaveLoadTests(unittest.TestCase):
    def get_encoder_decoder_config(self):
        encoder_config = AutoConfig.from_pretrained("google-bert/bert-base-uncased")
        decoder_config = AutoConfig.from_pretrained(
            "google-bert/bert-base-uncased", is_decoder=True, add_cross_attention=True
        )
        return EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)

    def get_encoder_decoder_config_small(self):
        encoder_config = AutoConfig.from_pretrained("hf-internal-testing/tiny-bert")
        decoder_config = AutoConfig.from_pretrained(
            "hf-internal-testing/tiny-bert", is_decoder=True, add_cross_attention=True
        )
        return EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)

    def test_encoder_decoder_save_load_from_encoder_decoder(self):
        config = self.get_encoder_decoder_config_small()

        # create two random BERT models for bert2bert & initialize weights (+cross_attention weights)
        encoder = TFBertModel(config.encoder)
        encoder.build_in_name_scope()
        decoder = TFBertLMHeadModel(config.decoder)
        decoder.build_in_name_scope()

        encoder_decoder_orig = TFEncoderDecoderModel(encoder=encoder, decoder=decoder)

        input_ids = ids_tensor([13, 5], encoder.config.vocab_size)
        decoder_input_ids = ids_tensor([13, 1], decoder.config.vocab_size)

        logits_orig = encoder_decoder_orig(input_ids=input_ids, decoder_input_ids=decoder_input_ids).logits

        with tempfile.TemporaryDirectory() as tmp_dirname:
            encoder_path = os.path.join(tmp_dirname, "encoder")
            decoder_path = os.path.join(tmp_dirname, "decoder")

            encoder.save_pretrained(encoder_path)
            decoder.save_pretrained(decoder_path)

            encoder_decoder = TFEncoderDecoderModel.from_encoder_decoder_pretrained(encoder_path, decoder_path)

        logits_1 = encoder_decoder(input_ids=input_ids, decoder_input_ids=decoder_input_ids).logits

        self.assertTrue(logits_orig.numpy().sum() - logits_1.numpy().sum() < 1e-3)

        max_diff = np.max(np.abs(logits_1.numpy() - logits_orig.numpy()))
        self.assertAlmostEqual(max_diff, 0.0, places=4)

        with tempfile.TemporaryDirectory() as tmp_dirname:
            encoder_decoder.save_pretrained(tmp_dirname)
            encoder_decoder = TFEncoderDecoderModel.from_pretrained(tmp_dirname)

        logits_2 = encoder_decoder(input_ids=input_ids, decoder_input_ids=decoder_input_ids).logits

        max_diff = np.max(np.abs(logits_2.numpy() - logits_orig.numpy()))
        self.assertAlmostEqual(max_diff, 0.0, places=4)

    @slow
    def test_encoder_decoder_from_pretrained(self):
        load_weight_prefix = TFEncoderDecoderModel.load_weight_prefix

        config = self.get_encoder_decoder_config()
        encoder_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        decoder_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

        input_ids = encoder_tokenizer("who sings does he love me with reba", return_tensors="tf").input_ids
        decoder_input_ids = decoder_tokenizer("Linda Davis", return_tensors="tf").input_ids

        with tempfile.TemporaryDirectory() as tmp_dirname:
            # Since most of HF's models don't have pretrained cross-attention layers, they are randomly
            # initialized even if we create models using `from_pretrained` method.
            # For the tests, the decoder need to be a model with pretrained cross-attention layers.
            # So we create pretrained models (without `load_weight_prefix`), save them, and later,
            # we load them using `from_pretrained`.
            # (we don't need to do this for encoder, but let's make the code more similar between encoder/decoder)
            encoder = TFAutoModel.from_pretrained("google-bert/bert-base-uncased", name="encoder")
            # It's necessary to specify `add_cross_attention=True` here.
            decoder = TFAutoModelForCausalLM.from_pretrained(
                "google-bert/bert-base-uncased", is_decoder=True, add_cross_attention=True, name="decoder"
            )
            pretrained_encoder_dir = os.path.join(tmp_dirname, "pretrained_encoder")
            pretrained_decoder_dir = os.path.join(tmp_dirname, "pretrained_decoder")
            encoder.save_pretrained(pretrained_encoder_dir)
            decoder.save_pretrained(pretrained_decoder_dir)
            del encoder
            del decoder

            enc_dec_model = TFEncoderDecoderModel.from_encoder_decoder_pretrained(
                pretrained_encoder_dir,
                pretrained_decoder_dir,
            )
            # check that the from pretrained methods work
            enc_dec_model.save_pretrained(tmp_dirname)
            enc_dec_model = TFEncoderDecoderModel.from_pretrained(tmp_dirname)

            output = enc_dec_model(input_ids, decoder_input_ids=decoder_input_ids, labels=decoder_input_ids)

            loss_pretrained = output.loss
            del enc_dec_model

            # Create the model using `__init__` with loaded ``pretrained`` encoder / decoder
            encoder = TFAutoModel.from_pretrained(
                pretrained_encoder_dir, load_weight_prefix=load_weight_prefix, name="encoder"
            )
            decoder = TFAutoModelForCausalLM.from_pretrained(
                pretrained_decoder_dir, load_weight_prefix=load_weight_prefix, name="decoder"
            )
            enc_dec_model = TFEncoderDecoderModel(config=config, encoder=encoder, decoder=decoder)

        output = enc_dec_model(input_ids, decoder_input_ids=decoder_input_ids, labels=decoder_input_ids)

        loss_init = output.loss

        max_diff = np.max(np.abs(loss_pretrained - loss_init))
        expected_diff = 0.0

        self.assertAlmostEqual(max_diff, expected_diff, places=4)
