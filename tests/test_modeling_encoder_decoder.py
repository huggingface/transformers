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


import tempfile
import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

# TODO(PVP): this line reruns all the tests in BertModelTest; not sure whether this can be prevented
# for now only run module with pytest tests/test_modeling_encoder_decoder.py::EncoderDecoderModelTest
from .test_modeling_bert import BertModelTester
from .test_modeling_common import ids_tensor


if is_torch_available():
    from transformers import BertModel, EncoderDecoderModel, EncoderDecoderConfig
    from transformers.modeling_bert import BertLMHeadModel
    import numpy as np
    import torch


@require_torch
class EncoderDecoderModelTest(unittest.TestCase):
    def prepare_config_and_inputs_bert(self):
        bert_model_tester = BertModelTester(self)
        encoder_config_and_inputs = bert_model_tester.prepare_config_and_inputs()
        decoder_config_and_inputs = bert_model_tester.prepare_config_and_inputs_for_decoder()
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

    def create_and_check_bert_encoder_decoder_model_from_pretrained_configs(
        self,
        config,
        input_ids,
        attention_mask,
        encoder_hidden_states,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        **kwargs
    ):
        encoder_decoder_config = EncoderDecoderConfig.from_encoder_decoder_configs(config, decoder_config)
        self.assertTrue(encoder_decoder_config.decoder.is_decoder)

        enc_dec_model = EncoderDecoderModel(encoder_decoder_config)
        enc_dec_model.to(torch_device)
        enc_dec_model.eval()

        self.assertTrue(enc_dec_model.config.is_encoder_decoder)

        outputs_encoder_decoder = enc_dec_model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )

        self.assertEqual(outputs_encoder_decoder[0].shape, (decoder_input_ids.shape + (decoder_config.vocab_size,)))
        self.assertEqual(outputs_encoder_decoder[1].shape, (input_ids.shape + (config.hidden_size,)))

    def create_and_check_bert_encoder_decoder_model(
        self,
        config,
        input_ids,
        attention_mask,
        encoder_hidden_states,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        **kwargs
    ):
        encoder_model = BertModel(config)
        decoder_model = BertLMHeadModel(decoder_config)
        enc_dec_model = EncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        self.assertTrue(enc_dec_model.config.decoder.is_decoder)
        self.assertTrue(enc_dec_model.config.is_encoder_decoder)
        enc_dec_model.to(torch_device)
        outputs_encoder_decoder = enc_dec_model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )

        self.assertEqual(outputs_encoder_decoder[0].shape, (decoder_input_ids.shape + (decoder_config.vocab_size,)))
        self.assertEqual(outputs_encoder_decoder[1].shape, (input_ids.shape + (config.hidden_size,)))
        encoder_outputs = (encoder_hidden_states,)
        outputs_encoder_decoder = enc_dec_model(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )

        self.assertEqual(outputs_encoder_decoder[0].shape, (decoder_input_ids.shape + (decoder_config.vocab_size,)))
        self.assertEqual(outputs_encoder_decoder[1].shape, (input_ids.shape + (config.hidden_size,)))

    def create_and_check_bert_encoder_decoder_model_from_pretrained(
        self,
        config,
        input_ids,
        attention_mask,
        encoder_hidden_states,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        **kwargs
    ):
        encoder_model = BertModel(config)
        decoder_model = BertLMHeadModel(decoder_config)
        kwargs = {"encoder_model": encoder_model, "decoder_model": decoder_model}
        enc_dec_model = EncoderDecoderModel.from_encoder_decoder_pretrained(**kwargs)
        enc_dec_model.to(torch_device)
        outputs_encoder_decoder = enc_dec_model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )

        self.assertEqual(outputs_encoder_decoder[0].shape, (decoder_input_ids.shape + (decoder_config.vocab_size,)))
        self.assertEqual(outputs_encoder_decoder[1].shape, (input_ids.shape + (config.hidden_size,)))

    def create_and_check_save_and_load(
        self,
        config,
        input_ids,
        attention_mask,
        encoder_hidden_states,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        **kwargs
    ):
        encoder_model = BertModel(config)
        decoder_model = BertLMHeadModel(decoder_config)
        enc_dec_model = EncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        enc_dec_model.to(torch_device)
        enc_dec_model.eval()
        with torch.no_grad():
            outputs = enc_dec_model(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
            )
            out_2 = outputs[0].cpu().numpy()
            out_2[np.isnan(out_2)] = 0

            with tempfile.TemporaryDirectory() as tmpdirname:
                enc_dec_model.save_pretrained(tmpdirname)
                EncoderDecoderModel.from_pretrained(tmpdirname)

                after_outputs = enc_dec_model(
                    input_ids=input_ids,
                    decoder_input_ids=decoder_input_ids,
                    attention_mask=attention_mask,
                    decoder_attention_mask=decoder_attention_mask,
                )
                out_1 = after_outputs[0].cpu().numpy()
                out_1[np.isnan(out_1)] = 0
                max_diff = np.amax(np.abs(out_1 - out_2))
                self.assertLessEqual(max_diff, 1e-5)

    def create_and_check_save_and_load_encoder_decoder_model(
        self,
        config,
        input_ids,
        attention_mask,
        encoder_hidden_states,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        **kwargs
    ):
        encoder_model = BertModel(config)
        decoder_model = BertLMHeadModel(decoder_config)
        enc_dec_model = EncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        enc_dec_model.to(torch_device)
        enc_dec_model.eval()
        with torch.no_grad():
            outputs = enc_dec_model(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
            )
            out_2 = outputs[0].cpu().numpy()
            out_2[np.isnan(out_2)] = 0

            with tempfile.TemporaryDirectory() as encoder_tmp_dirname, tempfile.TemporaryDirectory() as decoder_tmp_dirname:
                enc_dec_model.encoder.save_pretrained(encoder_tmp_dirname)
                enc_dec_model.decoder.save_pretrained(decoder_tmp_dirname)
                EncoderDecoderModel.from_encoder_decoder_pretrained(
                    encoder_pretrained_model_name_or_path=encoder_tmp_dirname,
                    decoder_pretrained_model_name_or_path=decoder_tmp_dirname,
                )

                after_outputs = enc_dec_model(
                    input_ids=input_ids,
                    decoder_input_ids=decoder_input_ids,
                    attention_mask=attention_mask,
                    decoder_attention_mask=decoder_attention_mask,
                )
                out_1 = after_outputs[0].cpu().numpy()
                out_1[np.isnan(out_1)] = 0
                max_diff = np.amax(np.abs(out_1 - out_2))
                self.assertLessEqual(max_diff, 1e-5)

    def check_loss_output(self, loss):
        self.assertEqual(loss.size(), ())

    def create_and_check_bert_encoder_decoder_model_labels(
        self,
        config,
        input_ids,
        attention_mask,
        encoder_hidden_states,
        decoder_config,
        decoder_input_ids,
        decoder_attention_mask,
        labels,
        **kwargs
    ):
        encoder_model = BertModel(config)
        decoder_model = BertLMHeadModel(decoder_config)
        enc_dec_model = EncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        enc_dec_model.to(torch_device)
        outputs_encoder_decoder = enc_dec_model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

        mlm_loss = outputs_encoder_decoder[0]
        self.check_loss_output(mlm_loss)
        # check that backprop works
        mlm_loss.backward()

        self.assertEqual(outputs_encoder_decoder[1].shape, (decoder_input_ids.shape + (decoder_config.vocab_size,)))
        self.assertEqual(outputs_encoder_decoder[2].shape, (input_ids.shape + (config.hidden_size,)))

    def create_and_check_bert_encoder_decoder_model_generate(self, input_ids, config, decoder_config, **kwargs):
        encoder_model = BertModel(config)
        decoder_model = BertLMHeadModel(decoder_config)
        enc_dec_model = EncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        enc_dec_model.to(torch_device)

        # Bert does not have a bos token id, so use pad_token_id instead
        generated_output = enc_dec_model.generate(
            input_ids, decoder_start_token_id=enc_dec_model.config.decoder.pad_token_id
        )
        self.assertEqual(generated_output.shape, (input_ids.shape[0],) + (decoder_config.max_length,))

    def test_bert_encoder_decoder_model(self):
        input_ids_dict = self.prepare_config_and_inputs_bert()
        self.create_and_check_bert_encoder_decoder_model(**input_ids_dict)

    def test_bert_encoder_decoder_model_from_pretrained_configs(self):
        input_ids_dict = self.prepare_config_and_inputs_bert()
        self.create_and_check_bert_encoder_decoder_model_from_pretrained_configs(**input_ids_dict)

    def test_bert_encoder_decoder_model_from_pretrained(self):
        input_ids_dict = self.prepare_config_and_inputs_bert()
        self.create_and_check_bert_encoder_decoder_model_from_pretrained(**input_ids_dict)

    def test_save_and_load_from_pretrained(self):
        input_ids_dict = self.prepare_config_and_inputs_bert()
        self.create_and_check_save_and_load(**input_ids_dict)

    def test_save_and_load_from_encoder_decoder_pretrained(self):
        input_ids_dict = self.prepare_config_and_inputs_bert()
        self.create_and_check_save_and_load_encoder_decoder_model(**input_ids_dict)

    def test_bert_encoder_decoder_model_labels(self):
        input_ids_dict = self.prepare_config_and_inputs_bert()
        self.create_and_check_bert_encoder_decoder_model_labels(**input_ids_dict)

    def test_bert_encoder_decoder_model_generate(self):
        input_ids_dict = self.prepare_config_and_inputs_bert()
        self.create_and_check_bert_encoder_decoder_model_generate(**input_ids_dict)

    @slow
    def test_real_bert_model_from_pretrained(self):
        model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")
        self.assertIsNotNone(model)

    @slow
    def test_real_bert_model_from_pretrained_has_cross_attention(self):
        model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")
        self.assertTrue(hasattr(model.decoder.bert.encoder.layer[0], "crossattention"))

    @slow
    def test_real_bert_model_save_load_from_pretrained(self):
        model_2 = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")
        model_2.to(torch_device)
        input_ids = ids_tensor([13, 5], model_2.config.encoder.vocab_size)
        decoder_input_ids = ids_tensor([13, 1], model_2.config.encoder.vocab_size)
        attention_mask = ids_tensor([13, 5], vocab_size=2)
        with torch.no_grad():
            outputs = model_2(input_ids=input_ids, decoder_input_ids=decoder_input_ids, attention_mask=attention_mask,)
            out_2 = outputs[0].cpu().numpy()
            out_2[np.isnan(out_2)] = 0

            with tempfile.TemporaryDirectory() as tmp_dirname:
                model_2.save_pretrained(tmp_dirname)
                model_1 = EncoderDecoderModel.from_pretrained(tmp_dirname)
                model_1.to(torch_device)

                after_outputs = model_1(
                    input_ids=input_ids, decoder_input_ids=decoder_input_ids, attention_mask=attention_mask,
                )
                out_1 = after_outputs[0].cpu().numpy()
                out_1[np.isnan(out_1)] = 0
                max_diff = np.amax(np.abs(out_1 - out_2))
                self.assertLessEqual(max_diff, 1e-5)
