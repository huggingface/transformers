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

from .test_modeling_bert import BertModelTester
from .test_modeling_bert_generation import BertGenerationEncoderTester
from .test_modeling_common import ids_tensor
from .test_modeling_gpt2 import GPT2ModelTester
from .test_modeling_prophetnet import ProphetNetStandaloneDecoderModelTester
from .test_modeling_roberta import RobertaModelTester


if is_torch_available():
    import numpy as np
    import torch

    from transformers import (
        AutoTokenizer,
        BertGenerationDecoder,
        BertGenerationEncoder,
        BertLMHeadModel,
        BertModel,
        BertTokenizer,
        EncoderDecoderConfig,
        EncoderDecoderModel,
        GPT2LMHeadModel,
        ProphetNetForCausalLM,
        RobertaForCausalLM,
        RobertaModel,
    )
    from transformers.modeling_outputs import BaseModelOutput


@require_torch
class EncoderDecoderMixin:
    def get_encoder_decoder_model(self, config, decoder_config):
        pass

    def prepare_config_and_inputs(self):
        pass

    def get_pretrained_model(self):
        pass

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
        enc_dec_model = EncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        self.assertTrue(enc_dec_model.config.decoder.is_decoder)
        self.assertTrue(enc_dec_model.config.decoder.add_cross_attention)
        self.assertTrue(enc_dec_model.config.is_encoder_decoder)
        enc_dec_model.to(torch_device)
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

        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)
        outputs_encoder_decoder = enc_dec_model(
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
        enc_dec_model = EncoderDecoderModel.from_encoder_decoder_pretrained(**kwargs)
        enc_dec_model.to(torch_device)
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

    def check_save_and_load_encoder_decoder_model(
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
        enc_dec_model = EncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        enc_dec_model.to(torch_device)
        outputs_encoder_decoder = enc_dec_model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

        loss = outputs_encoder_decoder["loss"]
        # check that backprop works
        loss.backward()

        self.assertEqual(
            outputs_encoder_decoder["logits"].shape, (decoder_input_ids.shape + (decoder_config.vocab_size,))
        )
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
        labels,
        **kwargs
    ):
        # make the decoder inputs a different shape from the encoder inputs to harden the test
        decoder_input_ids = decoder_input_ids[:, :-1]
        decoder_attention_mask = decoder_attention_mask[:, :-1]
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        enc_dec_model = EncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        enc_dec_model.to(torch_device)
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
        enc_dec_model = EncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        enc_dec_model.to(torch_device)

        # Bert does not have a bos token id, so use pad_token_id instead
        generated_output = enc_dec_model.generate(
            input_ids, decoder_start_token_id=enc_dec_model.config.decoder.pad_token_id
        )
        self.assertEqual(generated_output.shape, (input_ids.shape[0],) + (decoder_config.max_length,))

    def create_and_check_encoder_decoder_shared_weights(
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
        torch.manual_seed(0)
        encoder_model, decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        model = EncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
        model.to(torch_device)
        model.eval()
        # load state dict copies weights but does not tie them
        decoder_state_dict = model.decoder._modules[model.decoder.base_model_prefix].state_dict()
        model.encoder.load_state_dict(decoder_state_dict, strict=False)

        torch.manual_seed(0)
        tied_encoder_model, tied_decoder_model = self.get_encoder_decoder_model(config, decoder_config)
        config = EncoderDecoderConfig.from_encoder_decoder_configs(
            tied_encoder_model.config, tied_decoder_model.config, tie_encoder_decoder=True
        )
        tied_model = EncoderDecoderModel(encoder=tied_encoder_model, decoder=tied_decoder_model, config=config)
        tied_model.to(torch_device)
        tied_model.eval()

        model_result = model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )

        tied_model_result = tied_model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )

        # check that models has less parameters
        self.assertLess(sum(p.numel() for p in tied_model.parameters()), sum(p.numel() for p in model.parameters()))
        random_slice_idx = ids_tensor((1,), model_result[0].shape[-1]).item()

        # check that outputs are equal
        self.assertTrue(
            torch.allclose(
                model_result[0][0, :, random_slice_idx], tied_model_result[0][0, :, random_slice_idx], atol=1e-4
            )
        )

        # check that outputs after saving and loading are equal
        with tempfile.TemporaryDirectory() as tmpdirname:
            tied_model.save_pretrained(tmpdirname)
            tied_model = EncoderDecoderModel.from_pretrained(tmpdirname)
            tied_model.to(torch_device)
            tied_model.eval()

            # check that models has less parameters
            self.assertLess(
                sum(p.numel() for p in tied_model.parameters()), sum(p.numel() for p in model.parameters())
            )
            random_slice_idx = ids_tensor((1,), model_result[0].shape[-1]).item()

            tied_model_result = tied_model(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
            )

            # check that outputs are equal
            self.assertTrue(
                torch.allclose(
                    model_result[0][0, :, random_slice_idx], tied_model_result[0][0, :, random_slice_idx], atol=1e-4
                )
            )

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

    def test_save_and_load_from_encoder_decoder_pretrained(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.check_save_and_load_encoder_decoder_model(**input_ids_dict)

    def test_encoder_decoder_model_labels(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.check_encoder_decoder_model_labels(**input_ids_dict)

    def test_encoder_decoder_model_output_attentions(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.check_encoder_decoder_model_output_attentions(**input_ids_dict)

    def test_encoder_decoder_model_generate(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.check_encoder_decoder_model_generate(**input_ids_dict)

    def test_encoder_decoder_model_shared_weights(self):
        input_ids_dict = self.prepare_config_and_inputs()
        self.create_and_check_encoder_decoder_shared_weights(**input_ids_dict)

    @slow
    def test_real_model_save_load_from_pretrained(self):
        model_2 = self.get_pretrained_model()
        model_2.to(torch_device)
        input_ids = ids_tensor([13, 5], model_2.config.encoder.vocab_size)
        decoder_input_ids = ids_tensor([13, 1], model_2.config.encoder.vocab_size)
        attention_mask = ids_tensor([13, 5], vocab_size=2)
        with torch.no_grad():
            outputs = model_2(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
            )
            out_2 = outputs[0].cpu().numpy()
            out_2[np.isnan(out_2)] = 0

            with tempfile.TemporaryDirectory() as tmp_dirname:
                model_2.save_pretrained(tmp_dirname)
                model_1 = EncoderDecoderModel.from_pretrained(tmp_dirname)
                model_1.to(torch_device)

                after_outputs = model_1(
                    input_ids=input_ids,
                    decoder_input_ids=decoder_input_ids,
                    attention_mask=attention_mask,
                )
                out_1 = after_outputs[0].cpu().numpy()
                out_1[np.isnan(out_1)] = 0
                max_diff = np.amax(np.abs(out_1 - out_2))
                self.assertLessEqual(max_diff, 1e-5)


@require_torch
class BertEncoderDecoderModelTest(EncoderDecoderMixin, unittest.TestCase):
    def get_pretrained_model(self):
        return EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-cased", "bert-base-cased")

    def get_encoder_decoder_model(self, config, decoder_config):
        encoder_model = BertModel(config)
        decoder_model = BertLMHeadModel(decoder_config)
        return encoder_model, decoder_model

    def prepare_config_and_inputs(self):
        model_tester = BertModelTester(self)
        encoder_config_and_inputs = model_tester.prepare_config_and_inputs()
        decoder_config_and_inputs = model_tester.prepare_config_and_inputs_for_decoder()
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

    @slow
    def test_bert2bert_summarization(self):
        model = EncoderDecoderModel.from_pretrained("patrickvonplaten/bert2bert-cnn_dailymail-fp16")
        model.to(torch_device)
        tokenizer = BertTokenizer.from_pretrained("patrickvonplaten/bert2bert-cnn_dailymail-fp16")

        ARTICLE = """(CNN)Sigma Alpha Epsilon is under fire for a video showing party-bound fraternity members singing a racist chant. SAE's national chapter suspended the students, but University of Oklahoma President David Boren took it a step further, saying the university's affiliation with the fraternity is permanently done. The news is shocking, but it's not the first time SAE has faced controversy. SAE was founded March 9, 1856, at the University of Alabama, five years before the American Civil War, according to the fraternity website. When the war began, the group had fewer than 400 members, of which "369 went to war for the Confederate States and seven for the Union Army," the website says. The fraternity now boasts more than 200,000 living alumni, along with about 15,000 undergraduates populating 219 chapters and 20 "colonies" seeking full membership at universities. SAE has had to work hard to change recently after a string of member deaths, many blamed on the hazing of new recruits, SAE national President Bradley Cohen wrote in a message on the fraternity's website. The fraternity's website lists more than 130 chapters cited or suspended for "health and safety incidents" since 2010. At least 30 of the incidents involved hazing, and dozens more involved alcohol. However, the list is missing numerous incidents from recent months. Among them, according to various media outlets: Yale University banned the SAEs from campus activities last month after members allegedly tried to interfere with a sexual misconduct investigation connected to an initiation rite. Stanford University in December suspended SAE housing privileges after finding sorority members attending a fraternity function were subjected to graphic sexual content. And Johns Hopkins University in November suspended the fraternity for underage drinking. "The media has labeled us as the 'nation's deadliest fraternity,' " Cohen said. In 2011, for example, a student died while being coerced into excessive alcohol consumption, according to a lawsuit. SAE's previous insurer dumped the fraternity. "As a result, we are paying Lloyd's of London the highest insurance rates in the Greek-letter world," Cohen said. Universities have turned down SAE's attempts to open new chapters, and the fraternity had to close 12 in 18 months over hazing incidents."""

        EXPECTED_SUMMARY = """sae was founded in 1856, five years before the civil war. the fraternity has had to work hard to change recently. the university of oklahoma president says the university's affiliation with the fraternity is permanently done. the sae has had a string of members in recent months."""

        input_ids = tokenizer(ARTICLE, return_tensors="pt").input_ids.to(torch_device)
        output_ids = model.generate(input_ids)
        summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        self.assertEqual(summary, EXPECTED_SUMMARY)


@require_torch
class BertGenerationEncoderDecoderModelTest(EncoderDecoderMixin, unittest.TestCase):
    def get_pretrained_model(self):
        return EncoderDecoderModel.from_encoder_decoder_pretrained(
            "google/bert_for_seq_generation_L-24_bbc_encoder", "google/bert_for_seq_generation_L-24_bbc_encoder"
        )

    def get_encoder_decoder_model(self, config, decoder_config):
        encoder_model = BertGenerationEncoder(config)
        decoder_model = BertGenerationDecoder(decoder_config)
        return encoder_model, decoder_model

    def prepare_config_and_inputs(self):
        model_tester = BertGenerationEncoderTester(self)
        encoder_config_and_inputs = model_tester.prepare_config_and_inputs()
        decoder_config_and_inputs = model_tester.prepare_config_and_inputs_for_decoder()
        (
            config,
            input_ids,
            input_mask,
            token_labels,
        ) = encoder_config_and_inputs
        (
            decoder_config,
            decoder_input_ids,
            decoder_input_mask,
            decoder_token_labels,
            encoder_hidden_states,
            encoder_attention_mask,
        ) = decoder_config_and_inputs

        # make sure that cross attention layers are added
        decoder_config.add_cross_attention = True
        return {
            "config": config,
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "decoder_config": decoder_config,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_input_mask,
            "decoder_token_labels": decoder_token_labels,
            "encoder_hidden_states": encoder_hidden_states,
            "labels": decoder_token_labels,
        }

    @slow
    def test_roberta2roberta_summarization(self):
        model = EncoderDecoderModel.from_pretrained("google/roberta2roberta_L-24_bbc")
        model.to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained("google/roberta2roberta_L-24_bbc")

        ARTICLE = """The problem is affecting people using the older versions of the PlayStation 3, called the "Fat" model.The problem isn't affecting the newer PS3 Slim systems that have been on sale since September last year.Sony have also said they are aiming to have the problem fixed shortly but is advising some users to avoid using their console for the time being."We hope to resolve this problem within the next 24 hours," a statement reads. "In the meantime, if you have a model other than the new slim PS3, we advise that you do not use your PS3 system, as doing so may result in errors in some functionality, such as recording obtained trophies, and not being able to restore certain data."We believe we have identified that this problem is being caused by a bug in the clock functionality incorporated in the system."The PlayStation Network is used by millions of people around the world.It allows users to play their friends at games like Fifa over the internet and also do things like download software or visit online stores."""

        EXPECTED_SUMMARY = """Sony has said that a bug in its PlayStation 3 console is preventing them from using the machine as a computer."""

        input_ids = tokenizer(ARTICLE, return_tensors="pt").input_ids.to(torch_device)
        output_ids = model.generate(input_ids)
        summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        self.assertEqual(summary, EXPECTED_SUMMARY)


@require_torch
class RoBertaEncoderDecoderModelTest(EncoderDecoderMixin, unittest.TestCase):
    def get_encoder_decoder_model(self, config, decoder_config):
        encoder_model = RobertaModel(config)
        decoder_model = RobertaForCausalLM(decoder_config)
        return encoder_model, decoder_model

    def prepare_config_and_inputs(self):
        model_tester = RobertaModelTester(self)
        encoder_config_and_inputs = model_tester.prepare_config_and_inputs()
        decoder_config_and_inputs = model_tester.prepare_config_and_inputs_for_decoder()
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

    def get_pretrained_model(self):
        return EncoderDecoderModel.from_encoder_decoder_pretrained("roberta-base", "roberta-base")


@require_torch
class GPT2EncoderDecoderModelTest(EncoderDecoderMixin, unittest.TestCase):
    def get_encoder_decoder_model(self, config, decoder_config):
        encoder_model = BertModel(config)
        decoder_model = GPT2LMHeadModel(decoder_config)
        return encoder_model, decoder_model

    def prepare_config_and_inputs(self):
        model_tester_encoder = BertModelTester(self, batch_size=13)
        model_tester_decoder = GPT2ModelTester(self, batch_size=13)
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
            decoder_input_mask,
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

    def get_pretrained_model(self):
        return EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-cased", "gpt2")

    def test_encoder_decoder_model_shared_weights(self):
        pass


@require_torch
class ProphetNetEncoderDecoderModelTest(EncoderDecoderMixin, unittest.TestCase):
    def get_encoder_decoder_model(self, config, decoder_config):
        encoder_model = BertModel(config)
        decoder_model = ProphetNetForCausalLM(decoder_config)
        return encoder_model, decoder_model

    def prepare_config_and_inputs(self):
        model_tester_encoder = BertModelTester(self, batch_size=13)
        model_tester_decoder = ProphetNetStandaloneDecoderModelTester(
            self, batch_size=13, hidden_size=32, max_position_embeddings=512
        )
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
            decoder_attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            lm_labels,
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
            "decoder_attention_mask": decoder_attention_mask,
            "encoder_hidden_states": encoder_hidden_states,
            "labels": lm_labels,
        }

    def get_pretrained_model(self):
        return EncoderDecoderModel.from_encoder_decoder_pretrained("bert-large-uncased", "prophetnet-large-uncased")

    def test_encoder_decoder_model_shared_weights(self):
        pass
