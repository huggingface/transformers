# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Prism model."""

import copy
import tempfile
import unittest

import pytest

from transformers import PrismConfig, is_torch_available
from transformers.testing_utils import (
    require_flash_attn,
    require_sentencepiece,
    require_tokenizers,
    require_torch,
    require_torch_fp16,
    require_torch_gpu,
    slow,
    torch_device,
)
from transformers.utils import cached_property

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor


if is_torch_available():
    import torch

    from transformers import PrismForConditionalGeneration, PrismModel, PrismTokenizer
    from transformers.models.prism.modeling_prism import PrismDecoder, PrismEncoder


def prepare_prism_inputs_dict(
    config,
    input_ids,
    decoder_input_ids,
    attention_mask=None,
    decoder_attention_mask=None,
    head_mask=None,
    decoder_head_mask=None,
    cross_attn_head_mask=None,
):
    if attention_mask is None:
        attention_mask = input_ids.ne(config.pad_token_id)
    if decoder_attention_mask is None:
        decoder_attention_mask = decoder_input_ids.ne(config.pad_token_id)
    if head_mask is None:
        head_mask = torch.ones(config.encoder_layers, config.encoder_attention_heads, device=torch_device)
    if decoder_head_mask is None:
        decoder_head_mask = torch.ones(config.decoder_layers, config.decoder_attention_heads, device=torch_device)
    if cross_attn_head_mask is None:
        cross_attn_head_mask = torch.ones(config.decoder_layers, config.decoder_attention_heads, device=torch_device)
    return {
        "input_ids": input_ids,
        "decoder_input_ids": decoder_input_ids,
        "attention_mask": attention_mask,
        "decoder_attention_mask": attention_mask,
        "head_mask": head_mask,
        "decoder_head_mask": decoder_head_mask,
        "cross_attn_head_mask": cross_attn_head_mask,
    }


class PrismModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_labels=False,
        vocab_size=99,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=4,
        hidden_act="relu", 
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        max_position_embeddings=20,
        eos_token_id=2,
        pad_token_id=1,
        bos_token_id=0,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.max_position_embeddings = max_position_embeddings
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        input_ids[:, -1] = self.eos_token_id  # Eos Token
        decoder_input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        # we need to clamp the input ids here to avoid having pad token in between
        # this is because for Prism the position_ids are prepared such that
        # all pad tokens have pos id = 2 and rest are between 2..seq_length
        # and the seq_length here is seq_length - num_pad_tokens
        # but when using past, there is no way of knowing if the past input ids had
        # pad tokens in them, which results in incorrect seq_lenth and which in turn results in
        # position_ids being off by num_pad_tokens in past input
        input_ids = input_ids.clamp(self.pad_token_id + 1)
        decoder_input_ids = decoder_input_ids.clamp(self.pad_token_id + 1)

        config = self.get_config()
        inputs_dict = prepare_prism_inputs_dict(config, input_ids, decoder_input_ids)
        return config, inputs_dict

    def get_config(self):
        return PrismConfig(
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
            encoder_layerdrop=self.encoder_layerdrop,
            decoder_layerdrop=self.decoder_layerdrop,
            max_position_embeddings=self.max_position_embeddings,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
        )

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def create_and_check_decoder_model_past_large_inputs(self, config, inputs_dict):
        model = PrismModel(config=config).get_decoder().to(torch_device).eval()
        input_ids = inputs_dict["input_ids"]
        attention_mask = inputs_dict["attention_mask"]
        head_mask = inputs_dict["head_mask"]

        # first forward pass
        outputs = model(input_ids, attention_mask=attention_mask, head_mask=head_mask, use_cache=True)

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
        model = PrismModel(config=config).to(torch_device).eval()
        outputs = model(**inputs_dict)

        encoder_last_hidden_state = outputs.encoder_last_hidden_state
        last_hidden_state = outputs.last_hidden_state

        with tempfile.TemporaryDirectory() as tmpdirname:
            encoder = model.get_encoder()
            encoder.save_pretrained(tmpdirname)
            encoder = PrismEncoder.from_pretrained(tmpdirname).to(torch_device)

        encoder_last_hidden_state_2 = encoder(inputs_dict["input_ids"], attention_mask=inputs_dict["attention_mask"])[
            0
        ]

        self.parent.assertTrue((encoder_last_hidden_state_2 - encoder_last_hidden_state).abs().max().item() < 1e-3)

        with tempfile.TemporaryDirectory() as tmpdirname:
            decoder = model.get_decoder()
            decoder.save_pretrained(tmpdirname)
            decoder = PrismDecoder.from_pretrained(tmpdirname).to(torch_device)

        last_hidden_state_2 = decoder(
            input_ids=inputs_dict["decoder_input_ids"],
            attention_mask=inputs_dict["decoder_attention_mask"],
            encoder_hidden_states=encoder_last_hidden_state,
            encoder_attention_mask=inputs_dict["attention_mask"],
        )[0]

        self.parent.assertTrue((last_hidden_state_2 - last_hidden_state).abs().max().item() < 1e-3)


@require_torch
class PrismModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            PrismModel,
            PrismForConditionalGeneration,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (PrismForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "translation": PrismForConditionalGeneration,
        }
        if is_torch_available()
        else {}
    )
    is_encoder_decoder = True
    fx_compatible = False
    test_pruning = False
    test_missing_keys = False
    has_tied_weights = True

    def setUp(self):
        self.model_tester = PrismModelTester(self)
        self.config_tester = ConfigTester(self, config_class=PrismConfig)

    def test_load_save_without_tied_weights(self):
        self.skipTest("Model requires tied weights, skipping test_load_save_without_tied_weights.")

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

    def test_decoder_model_past_with_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

    def test_encoder_decoder_model_standalone(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.check_encoder_decoder_model_standalone(*config_and_inputs)

    def test_inputs_embeds(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in (PrismModel, PrismForConditionalGeneration):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = copy.deepcopy(self._prepare_for_class(inputs_dict, model_class))

            if not self.is_encoder_decoder:
                input_ids = inputs["input_ids"]
                del inputs["input_ids"]
            else:
                encoder_input_ids = inputs["input_ids"]
                decoder_input_ids = inputs.get("decoder_input_ids", encoder_input_ids)
                del inputs["input_ids"]
                inputs.pop("decoder_input_ids", None)

            wte = model.get_input_embeddings()
            if not self.is_encoder_decoder:
                inputs["inputs_embeds"] = wte(input_ids)
            else:
                inputs["inputs_embeds"] = wte(encoder_input_ids)
                inputs["decoder_inputs_embeds"] = wte(decoder_input_ids)

            with torch.no_grad():
                model(**inputs)[0]

    @require_torch_fp16
    def test_generate_fp16(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs()
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        model = PrismForConditionalGeneration(config).eval().to(torch_device)
        model.half()
        model.generate(input_ids, attention_mask=attention_mask)
        model.generate(num_beams=4, do_sample=True, early_stopping=False, num_return_sequences=3)

    def test_eager_matches_sdpa_inference_0_float16(self):
        if not torch.cuda.is_available():
            self.skipTest("Skipping float16 tests on CPU. Requires GPU.")


def _long_tensor(tok_lst):
    return torch.tensor(tok_lst, dtype=torch.long, device=torch_device)


TOLERANCE = 1e-4

CHECKPOINT_NAME = "dariast/prism"


@require_torch
@require_sentencepiece
@require_tokenizers
@slow
class PrismModelIntegrationTests(unittest.TestCase):
    @cached_property
    def default_tokenizer(self):
        return PrismTokenizer.from_pretrained(CHECKPOINT_NAME)

    def test_inference_no_head(self):
        model = PrismModel.from_pretrained(CHECKPOINT_NAME).to(torch_device)
        input_ids = _long_tensor(
            [[85, 16370, 15443, 5, 160, 50, 11, 1814, 21392, 8, 14787, 8, 14, 747, 125, 3522, 15352, 6, 2]]
        )
        decoder_input_ids = _long_tensor(
            [[2, 85, 43185, 25, 38476, 165, 5330, 1185, 269, 784, 9474, 21, 7142, 220, 13, 3051, 3079, 6, 2]]
        )
        inputs_dict = prepare_prism_inputs_dict(model.config, input_ids, decoder_input_ids)
        with torch.no_grad():
            output = model(**inputs_dict)[0]
        expected_shape = torch.Size((1, 19, 1280))
        self.assertEqual(output.shape, expected_shape)
        # change to expected output here
        expected_slice = torch.tensor(
            [[-0.1113, -0.0307, -0.5641], [-0.4884, -0.2435, -0.3268], [-0.3196, -0.2342, -1.3295]],
            device=torch_device,
        )
        self.assertTrue(torch.allclose(output[:, :3, :3], expected_slice, atol=TOLERANCE))

    def test_inference_head(self):
        model = PrismForConditionalGeneration.from_pretrained(CHECKPOINT_NAME).to(torch_device)
        # change to intended input
        input_ids = _long_tensor(
            [[85, 16370, 15443, 5, 160, 50, 11, 1814, 21392, 8, 14787, 8, 14, 747, 125, 3522, 15352, 6, 2]]
        )
        decoder_input_ids = _long_tensor(
            [[2, 85, 43185, 25, 38476, 165, 5330, 1185, 269, 784, 9474, 21, 7142, 220, 13, 3051, 3079, 6, 2]]
        )
        inputs_dict = prepare_prism_inputs_dict(model.config, input_ids, decoder_input_ids)
        with torch.no_grad():
            output = model(**inputs_dict)[0]
        expected_shape = torch.Size((1, 19, model.config.vocab_size))
        self.assertEqual(output.shape, expected_shape)
        # change to expected output here
        expected_slice = torch.tensor(
            [[-0.2506, -0.2493, 3.5705], [-0.1940, -0.2163, 3.7116], [0.2059, 0.1973, 5.9530]], device=torch_device
        )
        self.assertTrue(torch.allclose(output[:, :3, :3], expected_slice, atol=TOLERANCE))

    def test_seq_to_seq_generation(self):
        model = PrismForConditionalGeneration.from_pretrained(CHECKPOINT_NAME).to(torch_device)
        tokenizer = PrismTokenizer.from_pretrained(CHECKPOINT_NAME, src_lang="fr", tgt_lang="en")

        src_fr = [
            "L'affaire NSA souligne l'absence totale de débat sur le renseignement",
            "Selon moi, il y a deux niveaux de réponse de la part du gouvernement français.",
            "Lorsque François Hollande téléphone à Barack Obama ou quand le ministre des affaires étrangères Laurent"
            "Fabius convoque l'ambassadeur des Etats-Unis, ils réagissent à une vraie découverte, qui est celle de"
            "l'ampleur de la surveillance américaine sur l'ensemble des communications en France.",
        ]

        # The below article tests that we don't add any hypotheses outside of the top n_beams
        dct = tokenizer(src_fr, padding=True, return_tensors="pt")

        hypotheses_batch = model.generate(
            input_ids=dct["input_ids"].to(torch_device),
            attention_mask=dct["attention_mask"].to(torch_device),
            num_beams=5,
            max_new_tokens=100,
            forced_bos_token_id=tokenizer.get_lang_id("en"),
        )

        expected_en = [
            "but the NSA case highlights the total lack of debate on intelligence",
            "In my view, there are two levels of response on the part of the French Government.",
            "When François Hollande calls Barack Obama or when Foreign Minister Laurent Fabius calls the US ambassador, they react to a real discovery, which is the extent of US surveillance of all communications in France.",
        ]

        generated = tokenizer.batch_decode(
            hypotheses_batch.tolist(), clean_up_tokenization_spaces=True, skip_special_tokens=True
        )
        print(generated)
        assert generated == expected_en

    @require_flash_attn
    @require_torch_gpu
    @pytest.mark.flash_attn_test
    @slow
    def test_flash_attn_2_seq_to_seq_generation(self):
        """
        Overwritting the common test as the test is flaky on tiny models
        """
        model = PrismForConditionalGeneration.from_pretrained(
            CHECKPOINT_NAME, attn_implementation="flash_attention_2"
        ).to(torch_device)

        tokenizer = PrismTokenizer.from_pretrained(CHECKPOINT_NAME, src_lang="fr", tgt_lang="en")

        src_fr = [
            "L'affaire NSA souligne l'absence totale de débat sur le renseignement",
            "Selon moi, il y a deux niveaux de réponse de la part du gouvernement français.",
            "Lorsque François Hollande téléphone à Barack Obama ou quand le ministre des affaires étrangères Laurent"
            "Fabius convoque l'ambassadeur des Etats-Unis, ils réagissent à une vraie découverte, qui est celle de"
            "l'ampleur de la surveillance américaine sur l'ensemble des communications en France.",
        ]

        # The below article tests that we don't add any hypotheses outside of the top n_beams
        dct = tokenizer(src_fr, padding=True, return_tensors="pt")

        hypotheses_batch = model.generate(
            input_ids=dct["input_ids"].to(torch_device),
            attention_mask=dct["attention_mask"].to(torch_device),
            num_beams=5,
            max_new_tokens=100,
            forced_bos_token_id=tokenizer.get_lang_id("en"),
        )

        expected_en = [
            "but the NSA case highlights the total lack of debate on intelligence",
            "In my view, there are two levels of response on the part of the French Government.",
            "When François Hollande calls Barack Obama or when Foreign Minister Laurent Fabius calls the US ambassador, they react to a real discovery, which is the extent of US surveillance of all communications in France.",
        ]

        generated = tokenizer.batch_decode(
            hypotheses_batch.tolist(), clean_up_tokenization_spaces=True, skip_special_tokens=True
        )
        assert generated == expected_en
