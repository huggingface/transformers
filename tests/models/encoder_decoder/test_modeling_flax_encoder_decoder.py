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

import numpy as np

from transformers import is_flax_available, is_torch_available
from transformers.testing_utils import is_pt_flax_cross_test, require_flax, slow, torch_device

from ...test_modeling_flax_common import ids_tensor
from ..bart.test_modeling_flax_bart import FlaxBartStandaloneDecoderModelTester
from ..bert.test_modeling_flax_bert import FlaxBertModelTester
from ..gpt2.test_modeling_flax_gpt2 import FlaxGPT2ModelTester


if is_flax_available():
    from transformers import (
        AutoTokenizer,
        EncoderDecoderConfig,
        FlaxBartForCausalLM,
        FlaxBertForCausalLM,
        FlaxBertModel,
        FlaxEncoderDecoderModel,
        FlaxGPT2LMHeadModel,
    )
    from transformers.modeling_flax_pytorch_utils import (
        convert_pytorch_state_dict_to_flax,
        load_flax_weights_in_pytorch_model,
    )

if is_torch_available():
    import torch

    from transformers import EncoderDecoderModel


@require_flax
class FlaxEncoderDecoderMixin:
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

        enc_dec_model = FlaxEncoderDecoderModel(encoder_decoder_config)

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
        enc_dec_model = FlaxEncoderDecoderModel.from_encoder_decoder_pretrained(**kwargs)
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
        **kwargs,
    ):
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        kwargs = {"encoder_model": encoder_model, "decoder_model": decoder_model}
        enc_dec_model = FlaxEncoderDecoderModel.from_encoder_decoder_pretrained(**kwargs)

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
            FlaxEncoderDecoderModel.from_pretrained(tmpdirname)

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

    def check_encoder_decoder_model_from_encoder_decoder_pretrained(
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
        # assert that model attributes match those of configs
        self.assertEqual(config.use_cache, encoder_model.config.use_cache)
        self.assertEqual(decoder_config.use_cache, decoder_model.config.use_cache)

        with tempfile.TemporaryDirectory() as enc_tmpdir:
            with tempfile.TemporaryDirectory() as dec_tmpdir:
                encoder_model.save_pretrained(enc_tmpdir)
                decoder_model.save_pretrained(dec_tmpdir)
                # load a model from pretrained encoder and decoder checkpoints, setting one encoder and one decoder kwarg opposite to that specified in their respective configs
                enc_dec_model = FlaxEncoderDecoderModel.from_encoder_decoder_pretrained(
                    encoder_pretrained_model_name_or_path=enc_tmpdir,
                    decoder_pretrained_model_name_or_path=dec_tmpdir,
                    encoder_use_cache=not config.use_cache,
                    decoder_use_cache=not decoder_config.use_cache,
                )

        # assert that setting encoder and decoder kwargs opposite to those in the configs has correctly been applied
        self.assertNotEqual(config.use_cache, enc_dec_model.config.encoder.use_cache)
        self.assertNotEqual(decoder_config.use_cache, enc_dec_model.config.decoder.use_cache)

        outputs_encoder_decoder = enc_dec_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        self.assertEqual(
            outputs_encoder_decoder["logits"].shape, (decoder_input_ids.shape + (decoder_config.vocab_size,))
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
        kwargs = {"encoder_model": encoder_model, "decoder_model": decoder_model}
        enc_dec_model = FlaxEncoderDecoderModel.from_encoder_decoder_pretrained(**kwargs)
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
        kwargs = {"encoder_model": encoder_model, "decoder_model": decoder_model}
        enc_dec_model = FlaxEncoderDecoderModel.from_encoder_decoder_pretrained(**kwargs)

        pad_token_id = enc_dec_model.config.decoder.pad_token_id
        eos_token_id = enc_dec_model.config.decoder.eos_token_id
        decoder_start_token_id = enc_dec_model.config.decoder.decoder_start_token_id

        # Copied from generation.utils (GPT2 doesn't have `pad_token_id`)
        if pad_token_id is None and eos_token_id is not None:
            pad_token_id = eos_token_id
        if decoder_start_token_id is None:
            decoder_start_token_id = enc_dec_model.config.decoder.bos_token_id

        # Bert does not have a bos token id, so use pad_token_id instead
        # Copied from `test_modeling_encoder_decoder.py`
        if decoder_start_token_id is None:
            decoder_start_token_id = pad_token_id

        generated_output = enc_dec_model.generate(
            input_ids,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
        )
        generated_sequences = generated_output.sequences
        self.assertEqual(generated_sequences.shape, (input_ids.shape[0],) + (decoder_config.max_length,))

    def check_pt_flax_equivalence(self, pt_model, fx_model, inputs_dict):
        pt_model.to(torch_device)
        pt_model.eval()

        # prepare inputs
        flax_inputs = inputs_dict
        pt_inputs = {k: torch.tensor(v.tolist()) for k, v in flax_inputs.items()}

        with torch.no_grad():
            pt_outputs = pt_model(**pt_inputs).to_tuple()

        fx_outputs = fx_model(**inputs_dict).to_tuple()
        self.assertEqual(len(fx_outputs), len(pt_outputs), "Output lengths differ between Flax and PyTorch")
        for fx_output, pt_output in zip(fx_outputs, pt_outputs):
            self.assert_almost_equals(fx_output, pt_output.numpy(), 1e-5)

        # PT -> Flax
        with tempfile.TemporaryDirectory() as tmpdirname:
            pt_model.save_pretrained(tmpdirname)
            fx_model_loaded = FlaxEncoderDecoderModel.from_pretrained(tmpdirname, from_pt=True)

        fx_outputs_loaded = fx_model_loaded(**inputs_dict).to_tuple()
        self.assertEqual(len(fx_outputs_loaded), len(pt_outputs), "Output lengths differ between Flax and PyTorch")
        for fx_output_loaded, pt_output in zip(fx_outputs_loaded, pt_outputs):
            self.assert_almost_equals(fx_output_loaded, pt_output.numpy(), 1e-5)

        # Flax -> PT
        with tempfile.TemporaryDirectory() as tmpdirname:
            fx_model.save_pretrained(tmpdirname)
            pt_model_loaded = EncoderDecoderModel.from_pretrained(tmpdirname, from_flax=True)

        pt_model_loaded.to(torch_device)
        pt_model_loaded.eval()

        with torch.no_grad():
            pt_outputs_loaded = pt_model_loaded(**pt_inputs).to_tuple()

        self.assertEqual(len(fx_outputs), len(pt_outputs_loaded), "Output lengths differ between Flax and PyTorch")
        for fx_output, pt_output_loaded in zip(fx_outputs, pt_outputs_loaded):
            self.assert_almost_equals(fx_output, pt_output_loaded.numpy(), 1e-5)

    def check_equivalence_pt_to_flax(self, config, decoder_config, inputs_dict):
        encoder_decoder_config = EncoderDecoderConfig.from_encoder_decoder_configs(config, decoder_config)

        pt_model = EncoderDecoderModel(encoder_decoder_config)
        fx_model = FlaxEncoderDecoderModel(encoder_decoder_config)

        fx_state = convert_pytorch_state_dict_to_flax(pt_model.state_dict(), fx_model)
        fx_model.params = fx_state

        self.check_pt_flax_equivalence(pt_model, fx_model, inputs_dict)

    def check_equivalence_flax_to_pt(self, config, decoder_config, inputs_dict):
        encoder_decoder_config = EncoderDecoderConfig.from_encoder_decoder_configs(config, decoder_config)

        pt_model = EncoderDecoderModel(encoder_decoder_config)
        fx_model = FlaxEncoderDecoderModel(encoder_decoder_config)

        pt_model = load_flax_weights_in_pytorch_model(pt_model, fx_model.params)

        self.check_pt_flax_equivalence(pt_model, fx_model, inputs_dict)

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

    def test_encoder_decoder_model_from_encoder_decoder_pretrained(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.check_encoder_decoder_model_from_encoder_decoder_pretrained(**input_ids_dict)

    def test_encoder_decoder_model_output_attentions(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.check_encoder_decoder_model_output_attentions(**input_ids_dict)

    def test_encoder_decoder_model_generate(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.check_encoder_decoder_model_generate(**input_ids_dict)

    def assert_almost_equals(self, a: np.ndarray, b: np.ndarray, tol: float):
        diff = np.abs((a - b)).max()
        self.assertLessEqual(diff, tol, f"Difference between torch and flax is {diff} (>= {tol}).")

    @is_pt_flax_cross_test
    def test_pt_flax_equivalence(self):
        config_inputs_dict = self.prepare_config_and_inputs()
        config = config_inputs_dict.pop("config")
        decoder_config = config_inputs_dict.pop("decoder_config")

        inputs_dict = config_inputs_dict
        # `encoder_hidden_states` is not used in model call/forward
        del inputs_dict["encoder_hidden_states"]

        # Avoid the case where a sequence has no place to attend (after combined with the causal attention mask)
        batch_size = inputs_dict["decoder_attention_mask"].shape[0]
        inputs_dict["decoder_attention_mask"] = np.concatenate(
            [np.ones(shape=(batch_size, 1)), inputs_dict["decoder_attention_mask"][:, 1:]], axis=1
        )

        # Flax models don't use the `use_cache` option and cache is not returned as a default.
        # So we disable `use_cache` here for PyTorch model.
        decoder_config.use_cache = False

        self.assertTrue(decoder_config.cross_attention_hidden_size is None)

        # check without `enc_to_dec_proj` projection
        decoder_config.hidden_size = config.hidden_size
        self.assertTrue(config.hidden_size == decoder_config.hidden_size)
        self.check_equivalence_pt_to_flax(config, decoder_config, inputs_dict)
        self.check_equivalence_flax_to_pt(config, decoder_config, inputs_dict)

        # check `enc_to_dec_proj` work as expected
        decoder_config.hidden_size = decoder_config.hidden_size * 2
        self.assertTrue(config.hidden_size != decoder_config.hidden_size)
        self.check_equivalence_pt_to_flax(config, decoder_config, inputs_dict)
        self.check_equivalence_flax_to_pt(config, decoder_config, inputs_dict)

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
            model_1 = FlaxEncoderDecoderModel.from_pretrained(tmp_dirname)

            after_outputs = model_1(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
            )
            out_1 = np.array(after_outputs[0])
            out_1[np.isnan(out_1)] = 0
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 1e-5)


@require_flax
class FlaxGPT2EncoderDecoderModelTest(FlaxEncoderDecoderMixin, unittest.TestCase):
    def get_encoder_decoder_model(self, config, decoder_config):
        encoder_model = FlaxBertModel(config)
        decoder_model = FlaxGPT2LMHeadModel(decoder_config)
        return encoder_model, decoder_model

    def prepare_config_and_inputs(self):
        model_tester_encoder = FlaxBertModelTester(self, batch_size=13)
        model_tester_decoder = FlaxGPT2ModelTester(self, batch_size=13)
        encoder_config_and_inputs = model_tester_encoder.prepare_config_and_inputs()
        decoder_config_and_inputs = model_tester_decoder.prepare_config_and_inputs_for_decoder()
        (config, input_ids, token_type_ids, attention_mask) = encoder_config_and_inputs
        (
            decoder_config,
            decoder_input_ids,
            decoder_attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
        ) = decoder_config_and_inputs

        # make sure that cross attention layers are added
        decoder_config.add_cross_attention = True
        return {
            "config": config,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_config": decoder_config,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "encoder_hidden_states": encoder_hidden_states,
        }

    def get_pretrained_model(self):
        return FlaxEncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-cased", "gpt2")

    @slow
    def test_bert2gpt2_summarization(self):
        tokenizer_in = AutoTokenizer.from_pretrained("bert-base-cased")
        tokenizer_out = AutoTokenizer.from_pretrained("gpt2")

        model = FlaxEncoderDecoderModel.from_pretrained(
            "patrickvonplaten/bert2gpt2-cnn_dailymail-fp16", pad_token_id=tokenizer_out.eos_token_id
        )

        ARTICLE_STUDENTS = """(CNN)Sigma Alpha Epsilon is under fire for a video showing party-bound fraternity members singing a racist chant. SAE's national chapter suspended the students, but University of Oklahoma President David Boren took it a step further, saying the university's affiliation with the fraternity is permanently done. The news is shocking, but it's not the first time SAE has faced controversy. SAE was founded March 9, 1856, at the University of Alabama, five years before the American Civil War, according to the fraternity website. When the war began, the group had fewer than 400 members, of which "369 went to war for the Confederate States and seven for the Union Army," the website says. The fraternity now boasts more than 200,000 living alumni, along with about 15,000 undergraduates populating 219 chapters and 20 "colonies" seeking full membership at universities. SAE has had to work hard to change recently after a string of member deaths, many blamed on the hazing of new recruits, SAE national President Bradley Cohen wrote in a message on the fraternity's website. The fraternity's website lists more than 130 chapters cited or suspended for "health and safety incidents" since 2010. At least 30 of the incidents involved hazing, and dozens more involved alcohol. However, the list is missing numerous incidents from recent months. Among them, according to various media outlets: Yale University banned the SAEs from campus activities last month after members allegedly tried to interfere with a sexual misconduct investigation connected to an initiation rite. Stanford University in December suspended SAE housing privileges after finding sorority members attending a fraternity function were subjected to graphic sexual content. And Johns Hopkins University in November suspended the fraternity for underage drinking. "The media has labeled us as the 'nation's deadliest fraternity,' " Cohen said. In 2011, for example, a student died while being coerced into excessive alcohol consumption, according to a lawsuit. SAE's previous insurer dumped the fraternity. "As a result, we are paying Lloyd's of London the highest insurance rates in the Greek-letter world," Cohen said. Universities have turned down SAE's attempts to open new chapters, and the fraternity had to close 12 in 18 months over hazing incidents."""

        EXPECTED_SUMMARY_STUDENTS = """SAE's national chapter suspended the students, but university president says it's permanent.\nSAE's national chapter has had to work hard to change recently.\nSAE's chapter has more than 200,000 members.\nSAE's chapter has been criticized for its hazing of new recruits."""

        input_dict = tokenizer_in(ARTICLE_STUDENTS, return_tensors="np")
        output_ids = model.generate(input_dict["input_ids"]).sequences
        summary = tokenizer_out.batch_decode(output_ids, skip_special_tokens=True)

        self.assertEqual(summary, [EXPECTED_SUMMARY_STUDENTS])


@require_flax
class FlaxBartEncoderDecoderModelTest(FlaxEncoderDecoderMixin, unittest.TestCase):
    def get_encoder_decoder_model(self, config, decoder_config):
        encoder_model = FlaxBertModel(config)
        decoder_model = FlaxBartForCausalLM(decoder_config)
        return encoder_model, decoder_model

    def prepare_config_and_inputs(self):
        model_tester_encoder = FlaxBertModelTester(self, batch_size=13)
        model_tester_decoder = FlaxBartStandaloneDecoderModelTester(self, batch_size=13)
        encoder_config_and_inputs = model_tester_encoder.prepare_config_and_inputs()
        decoder_config_and_inputs = model_tester_decoder.prepare_config_and_inputs_for_decoder()
        (config, input_ids, token_type_ids, attention_mask) = encoder_config_and_inputs
        (
            decoder_config,
            decoder_input_ids,
            decoder_attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
        ) = decoder_config_and_inputs

        # make sure that cross attention layers are added
        decoder_config.add_cross_attention = True
        return {
            "config": config,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_config": decoder_config,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "encoder_hidden_states": encoder_hidden_states,
        }

    def get_pretrained_model(self):
        return FlaxEncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-cased", "facebook/bart-base")


@require_flax
class FlaxBertEncoderDecoderModelTest(FlaxEncoderDecoderMixin, unittest.TestCase):
    def get_encoder_decoder_model(self, config, decoder_config):
        encoder_model = FlaxBertModel(config)
        decoder_model = FlaxBertForCausalLM(decoder_config)
        return encoder_model, decoder_model

    def prepare_config_and_inputs(self):
        model_tester_encoder = FlaxBertModelTester(self, batch_size=13)
        model_tester_decoder = FlaxBertModelTester(self, batch_size=13)
        encoder_config_and_inputs = model_tester_encoder.prepare_config_and_inputs()
        decoder_config_and_inputs = model_tester_decoder.prepare_config_and_inputs_for_decoder()
        (config, input_ids, token_type_ids, attention_mask) = encoder_config_and_inputs
        (
            decoder_config,
            decoder_input_ids,
            decoder_attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
        ) = decoder_config_and_inputs

        # make sure that cross attention layers are added
        decoder_config.add_cross_attention = True
        return {
            "config": config,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_config": decoder_config,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "encoder_hidden_states": encoder_hidden_states,
        }

    def get_pretrained_model(self):
        return FlaxEncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-cased", "bert-base-cased")


@require_flax
class FlaxEncoderDecoderModelTest(unittest.TestCase):
    def get_from_encoderdecoder_pretrained_model(self):
        return FlaxEncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-cased", "gpt2")

    def _check_configuration_tie(self, model):
        module = model.module.bind(model.params)

        assert id(module.decoder.config) == id(model.config.decoder)
        assert id(module.encoder.config) == id(model.config.encoder)

    @slow
    def test_configuration_tie(self):
        model = self.get_from_encoderdecoder_pretrained_model()
        self._check_configuration_tie(model)
