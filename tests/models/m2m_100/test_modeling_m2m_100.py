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
"""Testing suite for the PyTorch M2M100 model."""

import copy
import tempfile
import unittest

import pytest

from transformers import M2M100Config, is_torch_available
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
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        M2M100DecoderModel,
        M2M100EncoderModel,
        M2M100ForConditionalGeneration,
        M2M100Model,
        M2M100Tokenizer,
    )
    from transformers.modeling_outputs import BaseModelOutput
    from transformers.models.m2m_100.modeling_m2m_100 import M2M100Decoder, M2M100Encoder


def prepare_m2m_100_inputs_dict(
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


class M2M100ModelTester:
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
        # this is because for M2M100 the position_ids are prepared such that
        # all pad tokens have pos id = 2 and rest are between 2..seq_length
        # and the seq_length here is seq_length - num_pad_tokens
        # but when using past, there is no way of knowing if the past input ids had
        # pad tokens in them, which results in incorrect seq_lenth and which in turn results in
        # position_ids being off by num_pad_tokens in past input
        input_ids = input_ids.clamp(self.pad_token_id + 1)
        decoder_input_ids = decoder_input_ids.clamp(self.pad_token_id + 1)

        config = self.get_config()
        inputs_dict = prepare_m2m_100_inputs_dict(config, input_ids, decoder_input_ids)
        return config, inputs_dict

    def get_config(self):
        return M2M100Config(
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
        model = M2M100Model(config=config).get_decoder().to(torch_device).eval()
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
        model = M2M100Model(config=config).to(torch_device).eval()
        outputs = model(**inputs_dict)

        encoder_last_hidden_state = outputs.encoder_last_hidden_state
        last_hidden_state = outputs.last_hidden_state

        with tempfile.TemporaryDirectory() as tmpdirname:
            encoder = model.get_encoder()
            encoder.save_pretrained(tmpdirname)
            encoder = M2M100Encoder.from_pretrained(tmpdirname).to(torch_device)

        encoder_last_hidden_state_2 = encoder(inputs_dict["input_ids"], attention_mask=inputs_dict["attention_mask"])[
            0
        ]

        self.parent.assertTrue((encoder_last_hidden_state_2 - encoder_last_hidden_state).abs().max().item() < 1e-3)

        with tempfile.TemporaryDirectory() as tmpdirname:
            decoder = model.get_decoder()
            decoder.save_pretrained(tmpdirname)
            decoder = M2M100Decoder.from_pretrained(tmpdirname).to(torch_device)

        last_hidden_state_2 = decoder(
            input_ids=inputs_dict["decoder_input_ids"],
            attention_mask=inputs_dict["decoder_attention_mask"],
            encoder_hidden_states=encoder_last_hidden_state,
            encoder_attention_mask=inputs_dict["attention_mask"],
        )[0]

        self.parent.assertTrue((last_hidden_state_2 - last_hidden_state).abs().max().item() < 1e-3)


@require_torch
class M2M100ModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            M2M100Model,
            M2M100ForConditionalGeneration,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (M2M100ForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": M2M100Model,
            "summarization": M2M100ForConditionalGeneration,
            "text2text-generation": M2M100ForConditionalGeneration,
            "translation": M2M100ForConditionalGeneration,
        }
        if is_torch_available()
        else {}
    )
    is_encoder_decoder = True
    fx_compatible = True
    test_pruning = False
    test_missing_keys = False

    # TODO: Fix the failed tests
    def is_pipeline_test_to_skip(
        self, pipeline_test_casse_name, config_class, model_architecture, tokenizer_name, processor_name
    ):
        if pipeline_test_casse_name == "TranslationPipelineTests":
            # Get `ValueError: Translation requires a `src_lang` and a `tgt_lang` for this model`.
            # `M2M100Config` was never used in pipeline tests: cannot create a simple tokenizer.
            return True

        return False

    def setUp(self):
        self.model_tester = M2M100ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=M2M100Config)

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

        for model_class in (M2M100Model, M2M100ForConditionalGeneration):
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
        model = M2M100ForConditionalGeneration(config).eval().to(torch_device)
        model.half()
        model.generate(input_ids, attention_mask=attention_mask)
        model.generate(num_beams=4, do_sample=True, early_stopping=False, num_return_sequences=3)

    @unittest.skip(
        reason="This architecure has tied weights by default and there is no way to remove it, check: https://github.com/huggingface/transformers/pull/31771#issuecomment-2210915245"
    )
    def test_load_save_without_tied_weights(self):
        pass


def _long_tensor(tok_lst):
    return torch.tensor(tok_lst, dtype=torch.long, device=torch_device)


TOLERANCE = 1e-4


@require_torch
@require_sentencepiece
@require_tokenizers
@slow
class M2M100ModelIntegrationTests(unittest.TestCase):
    @cached_property
    def default_tokenizer(self):
        return M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

    def test_inference_no_head(self):
        model = M2M100Model.from_pretrained("facebook/m2m100_418M").to(torch_device)
        input_ids = _long_tensor([[128028, 98, 12, 30527, 2732, 159, 7755, 61904, 39144, 38, 2]])
        decoder_input_ids = _long_tensor([[2, 128028, 98, 12, 30527, 2732, 159, 7755, 61904, 39144, 38]])
        inputs_dict = prepare_m2m_100_inputs_dict(model.config, input_ids, decoder_input_ids)
        with torch.no_grad():
            output = model(**inputs_dict)[0]
        expected_shape = torch.Size((1, 11, 1024))
        self.assertEqual(output.shape, expected_shape)
        # change to expected output here
        expected_slice = torch.tensor(
            [[-0.7780, -0.1676, 0.1038], [-6.7556, -1.3992, 0.0567], [-7.5383, -0.5920, -0.2779]], device=torch_device
        )
        self.assertTrue(torch.allclose(output[:, :3, :3], expected_slice, atol=TOLERANCE))

    def test_inference_head(self):
        model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M").to(torch_device)

        # change to intended input
        input_ids = _long_tensor([[128028, 98, 12, 30527, 2732, 159, 7755, 61904, 39144, 38, 2]])
        decoder_input_ids = _long_tensor([[2, 128028, 98, 12, 30527, 2732, 159, 7755, 61904, 39144, 38]])
        inputs_dict = prepare_m2m_100_inputs_dict(model.config, input_ids, decoder_input_ids)
        with torch.no_grad():
            output = model(**inputs_dict)[0]
        expected_shape = torch.Size((1, 11, model.config.vocab_size))
        self.assertEqual(output.shape, expected_shape)
        # change to expected output here
        expected_slice = torch.tensor(
            [[-1.0448, -1.0411, 3.7992], [-3.2191, -3.2386, -1.3451], [-3.6210, -3.5993, 0.4925]], device=torch_device
        )
        self.assertTrue(torch.allclose(output[:, :3, :3], expected_slice, atol=TOLERANCE))

    def test_seq_to_seq_generation(self):
        model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M").to(torch_device)
        tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", src_lang="fr", tgt_lang="en")

        src_fr = [
            "L'affaire NSA souligne l'absence totale de débat sur le renseignement",
            "Selon moi, il y a deux niveaux de réponse de la part du gouvernement français.",
            "Lorsque François Hollande téléphone à Barack Obama ou quand le ministre des affaires étrangères Laurent"
            " Fabius convoque l'ambassadeur des Etats-Unis, ils réagissent à une vraie découverte, qui est celle de"
            " l'ampleur de la surveillance américaine sur l'ensemble des communications en France.",
        ]

        # The below article tests that we don't add any hypotheses outside of the top n_beams
        dct = tokenizer(src_fr, padding=True, return_tensors="pt")

        hypotheses_batch = model.generate(
            input_ids=dct["input_ids"].to(torch_device),
            attention_mask=dct["attention_mask"].to(torch_device),
            num_beams=5,
            forced_bos_token_id=tokenizer.get_lang_id("en"),
        )

        expected_en = [
            "The NSA case highlights the total absence of intelligence debate",
            "I think there are two levels of response from the French government.",
            "When François Hollande calls Barack Obama or when Foreign Minister Laurent Fabius calls the U.S."
            " Ambassador, they respond to a real discovery, which is that of the scale of U.S. surveillance on all"
            " communications in France.",
        ]

        generated = tokenizer.batch_decode(
            hypotheses_batch.tolist(), clean_up_tokenization_spaces=True, skip_special_tokens=True
        )
        assert generated == expected_en

    @require_flash_attn
    @require_torch_gpu
    @pytest.mark.flash_attn_test
    @slow
    def test_flash_attn_2_seq_to_seq_generation(self):
        """
        Overwritting the common test as the test is flaky on tiny models
        """
        model = M2M100ForConditionalGeneration.from_pretrained(
            "facebook/m2m100_418M", attn_implementation="flash_attention_2"
        ).to(torch_device)

        tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", src_lang="fr", tgt_lang="en")

        src_fr = [
            "L'affaire NSA souligne l'absence totale de débat sur le renseignement",
            "Selon moi, il y a deux niveaux de réponse de la part du gouvernement français.",
            "Lorsque François Hollande téléphone à Barack Obama ou quand le ministre des affaires étrangères Laurent"
            " Fabius convoque l'ambassadeur des Etats-Unis, ils réagissent à une vraie découverte, qui est celle de"
            " l'ampleur de la surveillance américaine sur l'ensemble des communications en France.",
        ]

        # The below article tests that we don't add any hypotheses outside of the top n_beams
        dct = tokenizer(src_fr, padding=True, return_tensors="pt")

        hypotheses_batch = model.generate(
            input_ids=dct["input_ids"].to(torch_device),
            attention_mask=dct["attention_mask"].to(torch_device),
            num_beams=5,
            forced_bos_token_id=tokenizer.get_lang_id("en"),
        )

        expected_en = [
            "The NSA case highlights the total absence of intelligence debate",
            "I think there are two levels of response from the French government.",
            "When François Hollande calls Barack Obama or when Foreign Minister Laurent Fabius calls the U.S."
            " Ambassador, they respond to a real discovery, which is that of the scale of U.S. surveillance on all"
            " communications in France.",
        ]

        generated = tokenizer.batch_decode(
            hypotheses_batch.tolist(), clean_up_tokenization_spaces=True, skip_special_tokens=True
        )
        assert generated == expected_en


class M2M100EncoderModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=False,
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
        use_attention_mask: bool = True,
        is_encoder_decoder=False,
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
        self.use_attention_mask = use_attention_mask
        self.is_encoder_decoder = is_encoder_decoder

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        attention_mask = None
        if self.use_attention_mask:
            attention_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

        config = M2M100Config(
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
            is_encoder_decoder=self.is_encoder_decoder,
        )

        return (
            config,
            input_ids,
            attention_mask,
        )

    def create_and_check_model(
        self,
        config,
        input_ids,
        attention_mask,
    ):
        model = M2M100EncoderModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        result = model(input_ids=input_ids)
        encoder_output = result.last_hidden_state

        self.parent.assertEqual(encoder_output.size(), (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_model_fp16_forward(
        self,
        config,
        input_ids,
        attention_mask,
    ):
        model = M2M100EncoderModel(config=config).to(torch_device).half().eval()
        output = model(input_ids, attention_mask=attention_mask)["last_hidden_state"]
        self.parent.assertFalse(torch.isnan(output).any().item())

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            attention_mask,
        ) = config_and_inputs

        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


class M2M100EncoderModelModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (M2M100EncoderModel,) if is_torch_available() else ()
    test_pruning = False

    def setUp(self):
        self.model_tester = M2M100EncoderModelTester(self)
        self.config_tester = ConfigTester(self, config_class=M2M100Config, d_model=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skipIf(torch_device == "cpu", "Cant do half precision")
    def test_model_fp16_forward(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_fp16_forward(*config_and_inputs)

    @unittest.skip(
        reason="This architecure has tied weights by default and there is no way to remove it, check: https://github.com/huggingface/transformers/pull/31771#issuecomment-2210915245"
    )
    def test_load_save_without_tied_weights(self):
        pass


class M2M100DecoderModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=False,
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
        use_attention_mask: bool = True,
        is_encoder_decoder=True,
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
        self.use_attention_mask = use_attention_mask
        self.is_encoder_decoder = is_encoder_decoder

        # some tests ask for encoder-related parameters (e.g. test_attention_outputs)
        self.encoder_seq_length = 1

    def prepare_config_and_inputs(self):
        # encoder_input_ids should usually be ignored, but adding them for compatibility with some tests like test_resize_tokens_embeddings
        encoder_input_ids = ids_tensor([self.batch_size, 1], self.vocab_size)
        decoder_input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        attention_mask = None
        if self.use_attention_mask:
            attention_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

        embeddings = torch.randn([self.batch_size, 1, self.hidden_size], device=torch_device)
        encoder_outputs = BaseModelOutput(last_hidden_state=embeddings)

        inputs_dict = {
            "input_ids": encoder_input_ids,  # the input ids are always ignored anyway
            "attention_mask": None,  # the inputs are pooled embeddins, so they are never masked
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": attention_mask,
            "encoder_outputs": encoder_outputs,
        }

        config = M2M100Config(
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
            is_encoder_decoder=self.is_encoder_decoder,
        )

        return (config, inputs_dict)

    def create_and_check_model(
        self,
        config,
        **inputs_dict,
    ):
        model = M2M100DecoderModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            **inputs_dict,
        )
        logits = result.logits

        self.parent.assertEqual(logits.size(), (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_model_fp16_forward(
        self,
        config,
        encoder_outputs,
        **inputs_dict,
    ):
        model = M2M100DecoderModel(config=config).to(torch_device).half().eval()
        encoder_outputs_half = BaseModelOutput(last_hidden_state=encoder_outputs.last_hidden_state.half())
        output = model(encoder_outputs=encoder_outputs_half, **inputs_dict)
        self.parent.assertFalse(torch.isnan(output["logits"]).any().item())

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, inputs_dict = config_and_inputs
        return config, inputs_dict


class M2M100DecoderModelTest(ModelTesterMixin, unittest.TestCase):  # TODO: add GenerationTesterMixin
    all_model_classes = (M2M100DecoderModel,) if is_torch_available() else ()
    test_pruning = False
    test_head_masking = (
        False  # this would require also masking the attention heads of the fake encoder, which is cumbersome
    )

    def setUp(self):
        self.model_tester = M2M100DecoderModelTester(self)
        self.config_tester = ConfigTester(self, config_class=M2M100Config, d_model=37)
        self.is_encoder_decoder = True

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(config, **inputs_dict)

    @unittest.skipIf(torch_device == "cpu", "Cant do half precision")
    def test_model_fp16_forward(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_fp16_forward(config, **inputs_dict)

    @unittest.skip(
        reason="This architecure has tied weights by default and there is no way to remove it, check: https://github.com/huggingface/transformers/pull/31771#issuecomment-2210915245"
    )
    def test_load_save_without_tied_weights(self):
        pass

    @unittest.skip(
        reason="The encoder hidden states are an input to M2M100DecoderModelTest, so they do not have gradients, as the test expects."
    )
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(
        reason="The batching test does not split the precomputed encoder outputs into batches, so it cannot really batch the inputs."
    )
    def test_batching_equivalence(self):
        pass


@require_torch
@require_sentencepiece
@require_tokenizers
@slow
class SonarIntegrationTests(unittest.TestCase):
    @cached_property
    def default_tokenizer(self):
        from transformers import NllbTokenizer

        return NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

    def default_encoder(self):
        return M2M100EncoderModel.from_pretrained("cointegrated/SONAR_200_text_encoder_hf").to(torch_device)

    def default_decoder(self):
        return M2M100DecoderModel.from_pretrained("cointegrated/SONAR_200_text_decoder_hf").to(torch_device)

    def test_encoding_and_decoding(self):
        tokenizer = self.default_tokenizer
        encoder = self.default_encoder()
        tokenizer.src_lang = "eng_Latn"

        sentences = ["My name is SONAR.", "I can embed the sentences into vectorial space."]
        batch = tokenizer(sentences, padding=True, return_tensors="pt").to(torch_device)

        with torch.inference_mode():
            enc_out = encoder(**batch, pool_last_hidden_state=True)
        assert enc_out.last_hidden_state.shape == (2, 1, 1024)
        embeddings = enc_out.last_hidden_state.squeeze(1)

        ref_embeddings = torch.tensor(
            [
                [-0.005286, 0.002008, -0.000562, 0.006344, 0.006329],
                [-0.000330, -0.007055, 0.007644, 0.001841, 0.003727],
            ],
            device=torch_device,
        )
        assert torch.allclose(embeddings[:, :5], ref_embeddings, rtol=1e-3)

        decoder = self.default_decoder()

        gen_out = decoder.generate(
            # passing encoder_outputs is not recommended, because beam search decoding modifies them in place, which is ugly
            # encoder_outputs=enc_out,
            encoder_outputs=BaseModelOutput(last_hidden_state=enc_out.last_hidden_state.clone()),
            num_beams=5,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
        )
        text_out = tokenizer.batch_decode(gen_out, skip_special_tokens=True)
        assert text_out == ["My name is SONAR.", "I can embed the sentences into vector space."]
