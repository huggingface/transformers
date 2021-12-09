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


import os
import tempfile
import unittest

import numpy as np

from transformers import is_tf_available, is_torch_available
from transformers.testing_utils import is_pt_tf_cross_test, require_tf, require_torch, slow, torch_device

from .test_modeling_tf_bert import TFBertModelTester
from .test_modeling_tf_common import ids_tensor
from .test_modeling_tf_gpt2 import TFGPT2ModelTester
from .test_modeling_tf_rembert import TFRemBertModelTester
from .test_modeling_tf_roberta import TFRobertaModelTester


if is_tf_available():
    import tensorflow as tf

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
    import torch

    from transformers import BertLMHeadModel, BertModel, EncoderDecoderModel


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
        **kwargs
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
        **kwargs
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
        **kwargs
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
        **kwargs
    ):
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = TFEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)

        outputs = enc_dec_model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
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
        **kwargs
    ):
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = TFEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)

        outputs_encoder_decoder = enc_dec_model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

        # Make sure `loss` exist
        assert "loss" in outputs_encoder_decoder

        batch_size, seq_len = decoder_input_ids.shape
        expected_shape = (batch_size, seq_len - 1, decoder_config.vocab_size)
        self.assertEqual(outputs_encoder_decoder["logits"].shape, expected_shape)
        self.assertEqual(
            outputs_encoder_decoder["encoder_last_hidden_state"].shape, (input_ids.shape + (config.hidden_size,))
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
        **kwargs
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
        )

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

    def check_encoder_decoder_model_generate(self, input_ids, config, decoder_config, **kwargs):
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = TFEncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)

        # Bert does not have a bos token id, so use pad_token_id instead
        generated_output = enc_dec_model.generate(
            input_ids, decoder_start_token_id=enc_dec_model.config.decoder.pad_token_id
        )
        self.assertEqual(tuple(generated_output.shape.as_list()), (input_ids.shape[0],) + (decoder_config.max_length,))

    def check_pt_tf_equivalence(self, pt_model, tf_model, inputs_dict):

        pt_model.to(torch_device)
        pt_model.eval()

        # prepare inputs
        tf_inputs = inputs_dict
        pt_inputs = {k: torch.tensor(v.numpy()) for k, v in tf_inputs.items()}

        with torch.no_grad():
            pt_outputs = pt_model(**pt_inputs).to_tuple()

        tf_outputs = tf_model(**inputs_dict).to_tuple()
        self.assertEqual(len(tf_outputs), len(pt_outputs), "Output lengths differ between TF and PyTorch")
        for tf_output, pt_output in zip(tf_outputs, pt_outputs):
            self.assert_almost_equals(tf_output.numpy(), pt_output.numpy(), 1e-3)

        # PT -> TF
        with tempfile.TemporaryDirectory() as encoder_tmp_dirname, tempfile.TemporaryDirectory() as decoder_tmp_dirname:

            pt_model.encoder.save_pretrained(encoder_tmp_dirname)
            pt_model.decoder.save_pretrained(decoder_tmp_dirname)
            tf_model_loaded = TFEncoderDecoderModel.from_encoder_decoder_pretrained(
                encoder_tmp_dirname, decoder_tmp_dirname, encoder_from_pt=True, decoder_from_pt=True
            )
            # This is only for copying some specific attributes of this particular model.
            tf_model_loaded.config = pt_model.config

        tf_outputs_loaded = tf_model_loaded(**inputs_dict).to_tuple()
        self.assertEqual(len(tf_outputs_loaded), len(pt_outputs), "Output lengths differ between TF and PyTorch")
        for tf_output_loaded, pt_output in zip(tf_outputs_loaded, pt_outputs):
            self.assert_almost_equals(tf_output_loaded.numpy(), pt_output.numpy(), 1e-3)

    def check_equivalence_pt_to_tf(self, config, decoder_config, inputs_dict):

        encoder_decoder_config = EncoderDecoderConfig.from_encoder_decoder_configs(config, decoder_config)

        pt_model = EncoderDecoderModel(encoder_decoder_config)

        with tempfile.TemporaryDirectory() as encoder_tmp_dirname, tempfile.TemporaryDirectory() as decoder_tmp_dirname:

            pt_model.encoder.save_pretrained(encoder_tmp_dirname)
            pt_model.decoder.save_pretrained(decoder_tmp_dirname)
            tf_model = TFEncoderDecoderModel.from_encoder_decoder_pretrained(
                encoder_tmp_dirname, decoder_tmp_dirname, encoder_from_pt=True, decoder_from_pt=True
            )
            # This is only for copying some specific attributes of this particular model.
            tf_model.config = pt_model.config

        self.check_pt_tf_equivalence(pt_model, tf_model, inputs_dict)

    def check_equivalence_tf_to_pt(self, config, decoder_config, inputs_dict):

        encoder_decoder_config = EncoderDecoderConfig.from_encoder_decoder_configs(config, decoder_config)

        # Using `_tf_model`, the test will fail, because the weights of `_tf_model` get extended before saving
        # the encoder/decoder models.
        # There was a (very) ugly potential fix, which wasn't integrated to `transformers`: see
        #   https://github.com/huggingface/transformers/pull/13222/commits/dbb3c9de76eee235791d2064094654637c99f36d#r697304245
        #   (the change in `src/transformers/modeling_tf_utils.py`)
        _tf_model = TFEncoderDecoderModel(encoder_decoder_config)
        # Make sure model is built
        _tf_model(**inputs_dict)

        # Using `tf_model` to pass the test.
        encoder = _tf_model.encoder.__class__(encoder_decoder_config.encoder)
        decoder = _tf_model.decoder.__class__(encoder_decoder_config.decoder)
        # Make sure models are built
        encoder(encoder.dummy_inputs)
        decoder(decoder.dummy_inputs)
        tf_model = TFEncoderDecoderModel(encoder=encoder, decoder=decoder)

        with tempfile.TemporaryDirectory() as encoder_tmp_dirname, tempfile.TemporaryDirectory() as decoder_tmp_dirname:

            tf_model.encoder.save_pretrained(encoder_tmp_dirname)
            tf_model.decoder.save_pretrained(decoder_tmp_dirname)
            pt_model = EncoderDecoderModel.from_encoder_decoder_pretrained(
                encoder_tmp_dirname, decoder_tmp_dirname, encoder_from_tf=True, decoder_from_tf=True
            )
            # This is only for copying some specific attributes of this particular model.
            pt_model.config = tf_model.config

        self.check_pt_tf_equivalence(pt_model, tf_model, inputs_dict)

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

    def test_encoder_decoder_model_generate(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.check_encoder_decoder_model_generate(**input_ids_dict)

    def assert_almost_equals(self, a: np.ndarray, b: np.ndarray, tol: float):
        diff = np.abs((a - b)).max()
        self.assertLessEqual(diff, tol, f"Difference between torch and tf is {diff} (>= {tol}).")

    @is_pt_tf_cross_test
    def test_pt_tf_equivalence(self):

        config_inputs_dict = self.prepare_config_and_inputs()
        # Keep only common arguments
        arg_names = [
            "config",
            "input_ids",
            "attention_mask",
            "decoder_config",
            "decoder_input_ids",
            "decoder_attention_mask",
            "encoder_hidden_states",
        ]
        config_inputs_dict = {k: v for k, v in config_inputs_dict.items() if k in arg_names}

        config = config_inputs_dict.pop("config")
        decoder_config = config_inputs_dict.pop("decoder_config")

        inputs_dict = config_inputs_dict
        # `encoder_hidden_states` is not used in model call/forward
        del inputs_dict["encoder_hidden_states"]

        # Avoid the case where a sequence has no place to attend (after combined with the causal attention mask)
        batch_size = inputs_dict["decoder_attention_mask"].shape[0]
        inputs_dict["decoder_attention_mask"] = tf.constant(
            np.concatenate([np.ones(shape=(batch_size, 1)), inputs_dict["decoder_attention_mask"][:, 1:]], axis=1)
        )

        # TF models don't use the `use_cache` option and cache is not returned as a default.
        # So we disable `use_cache` here for PyTorch model.
        decoder_config.use_cache = False

        self.assertTrue(decoder_config.cross_attention_hidden_size is None)

        # check without `enc_to_dec_proj` projection
        self.assertTrue(config.hidden_size == decoder_config.hidden_size)
        self.check_equivalence_pt_to_tf(config, decoder_config, inputs_dict)
        self.check_equivalence_tf_to_pt(config, decoder_config, inputs_dict)

        # This is not working, because pt/tf equivalence test for encoder-decoder use `from_encoder_decoder_pretrained`,
        # which randomly initialize `enc_to_dec_proj`.
        # # check `enc_to_dec_proj` work as expected
        # decoder_config.hidden_size = decoder_config.hidden_size * 2
        # self.assertTrue(config.hidden_size != decoder_config.hidden_size)
        # self.check_equivalence_pt_to_tf(config, decoder_config, inputs_dict)
        # self.check_equivalence_tf_to_pt(config, decoder_config, inputs_dict)

        # Let's just check `enc_to_dec_proj` can run for now
        decoder_config.hidden_size = decoder_config.hidden_size * 2
        self.assertTrue(config.hidden_size != decoder_config.hidden_size)
        encoder_decoder_config = EncoderDecoderConfig.from_encoder_decoder_configs(config, decoder_config)
        model = TFEncoderDecoderModel(encoder_decoder_config)
        model(**inputs_dict)

    @slow
    def test_real_model_save_load_from_pretrained(self):
        model_2 = self.get_pretrained_model()
        input_ids = ids_tensor([13, 5], model_2.config.encoder.vocab_size)
        decoder_input_ids = ids_tensor([13, 1], model_2.config.encoder.vocab_size)
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
    def get_pretrained_model(self):
        return TFEncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")

    def get_encoder_decoder_model(self, config, decoder_config):
        encoder_model = TFBertModel(config, name="encoder")
        decoder_model = TFBertLMHeadModel(decoder_config, name="decoder")
        return encoder_model, decoder_model

    def prepare_config_and_inputs(self):
        model_tester_encoder = TFBertModelTester(self, batch_size=13)
        model_tester_decoder = TFBertModelTester(self, batch_size=13)
        encoder_config_and_inputs = model_tester_encoder.prepare_config_and_inputs()
        decoder_config_and_inputs = model_tester_decoder.prepare_config_and_inputs_for_decoder()
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

    @slow
    @is_pt_tf_cross_test
    def test_bert2bert_summarization(self):

        from transformers import EncoderDecoderModel

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        """Not working, because pt checkpoint has `encoder.encoder.layer...` while tf model has `encoder.bert.encoder.layer...`.
        (For Bert decoder, there is no issue, because `BertModel` is wrapped into `decoder` as `bert`)
        model = TFEncoderDecoderModel.from_pretrained("patrickvonplaten/bert2bert-cnn_dailymail-fp16", from_pt=True)
        """

        # workaround to load from pt
        _model = EncoderDecoderModel.from_pretrained("patrickvonplaten/bert2bert-cnn_dailymail-fp16")
        _model.encoder.save_pretrained("./encoder")
        _model.decoder.save_pretrained("./decoder")
        model = TFEncoderDecoderModel.from_encoder_decoder_pretrained(
            "./encoder", "./decoder", encoder_from_pt=True, decoder_from_pt=True
        )
        model.config = _model.config

        ARTICLE_STUDENTS = """(CNN)Sigma Alpha Epsilon is under fire for a video showing party-bound fraternity members singing a racist chant. SAE's national chapter suspended the students, but University of Oklahoma President David Boren took it a step further, saying the university's affiliation with the fraternity is permanently done. The news is shocking, but it's not the first time SAE has faced controversy. SAE was founded March 9, 1856, at the University of Alabama, five years before the American Civil War, according to the fraternity website. When the war began, the group had fewer than 400 members, of which "369 went to war for the Confederate States and seven for the Union Army," the website says. The fraternity now boasts more than 200,000 living alumni, along with about 15,000 undergraduates populating 219 chapters and 20 "colonies" seeking full membership at universities. SAE has had to work hard to change recently after a string of member deaths, many blamed on the hazing of new recruits, SAE national President Bradley Cohen wrote in a message on the fraternity's website. The fraternity's website lists more than 130 chapters cited or suspended for "health and safety incidents" since 2010. At least 30 of the incidents involved hazing, and dozens more involved alcohol. However, the list is missing numerous incidents from recent months. Among them, according to various media outlets: Yale University banned the SAEs from campus activities last month after members allegedly tried to interfere with a sexual misconduct investigation connected to an initiation rite. Stanford University in December suspended SAE housing privileges after finding sorority members attending a fraternity function were subjected to graphic sexual content. And Johns Hopkins University in November suspended the fraternity for underage drinking. "The media has labeled us as the 'nation's deadliest fraternity,' " Cohen said. In 2011, for example, a student died while being coerced into excessive alcohol consumption, according to a lawsuit. SAE's previous insurer dumped the fraternity. "As a result, we are paying Lloyd's of London the highest insurance rates in the Greek-letter world," Cohen said. Universities have turned down SAE's attempts to open new chapters, and the fraternity had to close 12 in 18 months over hazing incidents."""
        EXPECTED_SUMMARY_STUDENTS = """sae was founded in 1856, five years before the civil war. the fraternity has had to work hard to change recently. the university of oklahoma president says the university's affiliation with the fraternity is permanently done. the sae has had a string of members in recent months."""

        input_dict = tokenizer(ARTICLE_STUDENTS, return_tensors="tf")
        output_ids = model.generate(input_ids=input_dict["input_ids"], max_length=None).numpy().tolist()
        summary = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        self.assertEqual(summary, [EXPECTED_SUMMARY_STUDENTS])

        # Test with the TF checkpoint
        model = TFEncoderDecoderModel.from_pretrained("ydshieh/bert2bert-cnn_dailymail-fp16")

        output_ids = model.generate(input_ids=input_dict["input_ids"], max_length=None).numpy().tolist()
        summary = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        self.assertEqual(summary, [EXPECTED_SUMMARY_STUDENTS])


@require_tf
class TFGPT2EncoderDecoderModelTest(TFEncoderDecoderMixin, unittest.TestCase):
    def get_pretrained_model(self):
        return TFEncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-cased", "gpt2")

    def get_encoder_decoder_model(self, config, decoder_config):
        encoder_model = TFBertModel(config, name="encoder")
        decoder_model = TFGPT2LMHeadModel(decoder_config, name="decoder")
        return encoder_model, decoder_model

    def prepare_config_and_inputs(self):
        model_tester_encoder = TFBertModelTester(self, batch_size=13)
        model_tester_decoder = TFGPT2ModelTester(self)
        encoder_config_and_inputs = model_tester_encoder.prepare_config_and_inputs()
        decoder_config_and_inputs = model_tester_decoder.prepare_config_and_inputs_for_decoder()
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

    @slow
    @is_pt_tf_cross_test
    def test_bert2gpt2_summarization(self):

        from transformers import EncoderDecoderModel

        tokenizer_in = AutoTokenizer.from_pretrained("bert-base-cased")
        tokenizer_out = AutoTokenizer.from_pretrained("gpt2")

        """Not working, because pt checkpoint has `encoder.encoder.layer...` while tf model has `encoder.bert.encoder.layer...`.
        (For GPT2 decoder, there is no issue)
        model = TFEncoderDecoderModel.from_pretrained("patrickvonplaten/bert2gpt2-cnn_dailymail-fp16", from_pt=True)
        """

        # workaround to load from pt
        _model = EncoderDecoderModel.from_pretrained("patrickvonplaten/bert2gpt2-cnn_dailymail-fp16")
        _model.encoder.save_pretrained("./encoder")
        _model.decoder.save_pretrained("./decoder")
        model = TFEncoderDecoderModel.from_encoder_decoder_pretrained(
            "./encoder", "./decoder", encoder_from_pt=True, decoder_from_pt=True
        )
        model.config = _model.config

        ARTICLE_STUDENTS = """(CNN)Sigma Alpha Epsilon is under fire for a video showing party-bound fraternity members singing a racist chant. SAE's national chapter suspended the students, but University of Oklahoma President David Boren took it a step further, saying the university's affiliation with the fraternity is permanently done. The news is shocking, but it's not the first time SAE has faced controversy. SAE was founded March 9, 1856, at the University of Alabama, five years before the American Civil War, according to the fraternity website. When the war began, the group had fewer than 400 members, of which "369 went to war for the Confederate States and seven for the Union Army," the website says. The fraternity now boasts more than 200,000 living alumni, along with about 15,000 undergraduates populating 219 chapters and 20 "colonies" seeking full membership at universities. SAE has had to work hard to change recently after a string of member deaths, many blamed on the hazing of new recruits, SAE national President Bradley Cohen wrote in a message on the fraternity's website. The fraternity's website lists more than 130 chapters cited or suspended for "health and safety incidents" since 2010. At least 30 of the incidents involved hazing, and dozens more involved alcohol. However, the list is missing numerous incidents from recent months. Among them, according to various media outlets: Yale University banned the SAEs from campus activities last month after members allegedly tried to interfere with a sexual misconduct investigation connected to an initiation rite. Stanford University in December suspended SAE housing privileges after finding sorority members attending a fraternity function were subjected to graphic sexual content. And Johns Hopkins University in November suspended the fraternity for underage drinking. "The media has labeled us as the 'nation's deadliest fraternity,' " Cohen said. In 2011, for example, a student died while being coerced into excessive alcohol consumption, according to a lawsuit. SAE's previous insurer dumped the fraternity. "As a result, we are paying Lloyd's of London the highest insurance rates in the Greek-letter world," Cohen said. Universities have turned down SAE's attempts to open new chapters, and the fraternity had to close 12 in 18 months over hazing incidents."""
        EXPECTED_SUMMARY_STUDENTS = """SAS Alpha Epsilon suspended the students, but university president says it's permanent.\nThe fraternity has had to deal with a string of student deaths since 2010.\nSAS has more than 200,000 members, many of whom are students.\nA student died while being forced into excessive alcohol consumption."""

        input_dict = tokenizer_in(ARTICLE_STUDENTS, return_tensors="tf")
        output_ids = model.generate(input_ids=input_dict["input_ids"], max_length=None).numpy().tolist()
        summary = tokenizer_out.batch_decode(output_ids, skip_special_tokens=True)

        self.assertEqual(summary, [EXPECTED_SUMMARY_STUDENTS])


@require_tf
class TFRoBertaEncoderDecoderModelTest(TFEncoderDecoderMixin, unittest.TestCase):
    def get_pretrained_model(self):
        return TFEncoderDecoderModel.from_encoder_decoder_pretrained("roberta-base", "roberta-base")

    def get_encoder_decoder_model(self, config, decoder_config):
        encoder_model = TFRobertaModel(config, name="encoder")
        decoder_model = TFRobertaForCausalLM(decoder_config, name="decoder")
        return encoder_model, decoder_model

    def prepare_config_and_inputs(self):
        model_tester_encoder = TFRobertaModelTester(self)
        model_tester_decoder = TFRobertaModelTester(self)
        encoder_config_and_inputs = model_tester_encoder.prepare_config_and_inputs()
        decoder_config_and_inputs = model_tester_decoder.prepare_config_and_inputs_for_decoder()
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
    def get_pretrained_model(self):
        return TFEncoderDecoderModel.from_encoder_decoder_pretrained("google/rembert", "google/rembert")

    def get_encoder_decoder_model(self, config, decoder_config):
        encoder_model = TFRemBertModel(config, name="encoder")
        decoder_model = TFRemBertForCausalLM(decoder_config, name="decoder")
        return encoder_model, decoder_model

    def prepare_config_and_inputs(self):
        model_tester_encoder = TFRemBertModelTester(self)
        model_tester_decoder = TFRemBertModelTester(self)
        encoder_config_and_inputs = model_tester_encoder.prepare_config_and_inputs()
        decoder_config_and_inputs = model_tester_decoder.prepare_config_and_inputs_for_decoder()
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
        return TFEncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-cased", "bert-base-cased")

    def get_decoder_config(self):
        config = AutoConfig.from_pretrained("bert-base-cased")
        config.is_decoder = True
        config.add_cross_attention = True
        return config

    def get_encoderdecoder_model(self):
        return TFEncoderDecoderModel.from_pretrained("patrickvonplaten/bert2bert-cnn_dailymail-fp16")

    def get_encoder_decoder_models(self):
        encoder_model = TFBertModel.from_pretrained("bert-base-cased", name="encoder")
        decoder_model = TFBertLMHeadModel.from_pretrained(
            "bert-base-cased", config=self.get_decoder_config(), name="decoder"
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
        encoder_config = AutoConfig.from_pretrained("bert-base-uncased")
        decoder_config = AutoConfig.from_pretrained("bert-base-uncased", is_decoder=True, add_cross_attention=True)
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
        encoder(encoder.dummy_inputs)
        decoder = TFBertLMHeadModel(config.decoder)
        decoder(decoder.dummy_inputs)

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

    @require_torch
    @is_pt_tf_cross_test
    def test_encoder_decoder_save_load_from_encoder_decoder_from_pt(self):
        config = self.get_encoder_decoder_config_small()

        # create two random BERT models for bert2bert & initialize weights (+cross_attention weights)
        encoder_pt = BertModel(config.encoder).to(torch_device).eval()
        decoder_pt = BertLMHeadModel(config.decoder).to(torch_device).eval()

        encoder_decoder_pt = EncoderDecoderModel(encoder=encoder_pt, decoder=decoder_pt).to(torch_device).eval()

        input_ids = ids_tensor([13, 5], encoder_pt.config.vocab_size)
        decoder_input_ids = ids_tensor([13, 1], decoder_pt.config.vocab_size)

        pt_input_ids = torch.tensor(input_ids.numpy(), device=torch_device, dtype=torch.long)
        pt_decoder_input_ids = torch.tensor(decoder_input_ids.numpy(), device=torch_device, dtype=torch.long)

        logits_pt = encoder_decoder_pt(input_ids=pt_input_ids, decoder_input_ids=pt_decoder_input_ids).logits

        # PyTorch => TensorFlow
        with tempfile.TemporaryDirectory() as tmp_dirname_1, tempfile.TemporaryDirectory() as tmp_dirname_2:
            encoder_decoder_pt.encoder.save_pretrained(tmp_dirname_1)
            encoder_decoder_pt.decoder.save_pretrained(tmp_dirname_2)
            encoder_decoder_tf = TFEncoderDecoderModel.from_encoder_decoder_pretrained(
                tmp_dirname_1, tmp_dirname_2, encoder_from_pt=True, decoder_from_pt=True
            )

        logits_tf = encoder_decoder_tf(input_ids=input_ids, decoder_input_ids=decoder_input_ids).logits

        max_diff = np.max(np.abs(logits_pt.detach().cpu().numpy() - logits_tf.numpy()))
        self.assertAlmostEqual(max_diff, 0.0, places=3)

        # Make sure `from_pretrained` following `save_pretrained` work and give the same result
        with tempfile.TemporaryDirectory() as tmp_dirname:
            encoder_decoder_tf.save_pretrained(tmp_dirname)
            encoder_decoder_tf = TFEncoderDecoderModel.from_pretrained(tmp_dirname)

            logits_tf_2 = encoder_decoder_tf(input_ids=input_ids, decoder_input_ids=decoder_input_ids).logits

            max_diff = np.max(np.abs(logits_tf_2.numpy() - logits_tf.numpy()))
            self.assertAlmostEqual(max_diff, 0.0, places=3)

        # TensorFlow => PyTorch
        with tempfile.TemporaryDirectory() as tmp_dirname:
            encoder_decoder_tf.save_pretrained(tmp_dirname)
            encoder_decoder_pt = EncoderDecoderModel.from_pretrained(tmp_dirname, from_tf=True)

        max_diff = np.max(np.abs(logits_pt.detach().cpu().numpy() - logits_tf.numpy()))
        self.assertAlmostEqual(max_diff, 0.0, places=3)

    @slow
    def test_encoder_decoder_from_pretrained(self):
        load_weight_prefix = "tf_encoder_decoder_model_1"

        config = self.get_encoder_decoder_config()
        encoder_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        decoder_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        input_ids = encoder_tokenizer("who sings does he love me with reba", return_tensors="tf").input_ids
        decoder_input_ids = decoder_tokenizer("Linda Davis", return_tensors="tf").input_ids

        with tempfile.TemporaryDirectory() as tmp_dirname:

            # Since most of HF's models don't have pretrained cross-attention layers, they are randomly
            # initialized even if we create models using `from_pretrained` method.
            # For the tests, the decoder need to be a model with pretrained cross-attention layers.
            # So we create pretrained models (without `load_weight_prefix`), save them, and later,
            # we load them using `from_pretrained`.
            # (we don't need to do this for encoder, but let's make the code more similar between encoder/decoder)
            encoder = TFAutoModel.from_pretrained("bert-base-uncased", name="encoder")
            # It's necessary to specify `add_cross_attention=True` here.
            decoder = TFAutoModelForCausalLM.from_pretrained(
                "bert-base-uncased", is_decoder=True, add_cross_attention=True, name="decoder"
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
