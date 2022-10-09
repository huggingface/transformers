# coding=utf-8
# Copyright 2021, The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch DALLE_MEGA model. """


import copy
import tempfile
import unittest

import timeout_decorator  # noqa

from transformers import DalleMegaConfig, is_torch_available
from transformers.testing_utils import require_sentencepiece, require_tokenizers, require_torch, slow, torch_device
from transformers.utils import cached_property

from ...generation.test_generation_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch

    from transformers import (
        AutoModelForSequenceClassification,
        BartTokenizer,
        DalleMegaForConditionalGeneration,
        DalleMegaModel,
        pipeline,
    )
    from transformers.models.dallemega.modeling_dallemega import DalleMegaDecoder, DalleMegaEncoder, shift_tokens_right


def prepare_dallemega_inputs_dict(
    config,
    input_ids,
    decoder_input_ids=None,
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


class DalleMegaModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_labels=False,
        encoder_vocab_size=99,
        decoder_vocab_size=99,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=4,
        activation_function="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        encoder_max_positions=20,
        decoder_max_positions=20,
        eos_token_id=2,
        pad_token_id=1,
        bos_token_id=0,
        decoder_start_token_id=3,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_labels = use_labels
        self.encoder_vocab_size = encoder_vocab_size
        self.decoder_vocab_size = decoder_vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.activation_function = activation_function
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.encoder_max_positions = encoder_max_positions
        self.decoder_max_positions = decoder_max_positions
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.decoder_start_token_id = decoder_start_token_id

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.encoder_vocab_size)
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.encoder_vocab_size).clamp(
            3,
        )
        input_ids[:, -1] = self.eos_token_id  # Eos Token

        decoder_input_ids = ids_tensor([self.batch_size, self.seq_length], self.decoder_vocab_size)

        config = self.get_config()
        inputs_dict = prepare_dallemega_inputs_dict(config, input_ids, decoder_input_ids)
        return config, inputs_dict

    def get_config(self):
        return DalleMegaConfig(
            encoder_vocab_size=self.encoder_vocab_size,
            decoder_vocab_size=self.decoder_vocab_size,
            d_model=self.hidden_size,
            encoder_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            encoder_attention_heads=self.num_attention_heads,
            decoder_attention_heads=self.num_attention_heads,
            encoder_ffn_dim=self.intermediate_size,
            decoder_ffn_dim=self.intermediate_size,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            encoder_max_positions=self.encoder_max_positions,
            decoder_max_positions=self.decoder_max_positions,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
            decoder_start_token_id=self.decoder_start_token_id,
        )

    def get_pipeline_config(self):
        config = self.get_config()
        config.encoder_max_positions = 100
        config.decoder_max_positions = 100
        config.encoder_vocab_size = 300
        config.decoder_vocab_size = 300
        return config

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def create_and_check_decoder_model_past_large_inputs(self, config, inputs_dict):
        model = DalleMegaModel(config=config).get_decoder().to(torch_device).eval()
        input_ids = inputs_dict["input_ids"]
        attention_mask = inputs_dict["attention_mask"]
        head_mask = inputs_dict["head_mask"]

        # first forward pass
        outputs = model(input_ids, attention_mask=attention_mask, head_mask=head_mask, use_cache=True)

        output, past_key_values = outputs.to_tuple()

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.decoder_vocab_size)
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
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def check_encoder_decoder_model_standalone(self, config, inputs_dict):
        model = DalleMegaModel(config=config).to(torch_device).eval()
        outputs = model(**inputs_dict)

        encoder_last_hidden_state = outputs.encoder_last_hidden_state
        last_hidden_state = outputs.last_hidden_state

        with tempfile.TemporaryDirectory() as tmpdirname:
            encoder = model.get_encoder()
            encoder.save_pretrained(tmpdirname)
            encoder = DalleMegaEncoder.from_pretrained(tmpdirname).to(torch_device)

        encoder_last_hidden_state_2 = encoder(inputs_dict["input_ids"], attention_mask=inputs_dict["attention_mask"])[
            0
        ]

        self.parent.assertTrue((encoder_last_hidden_state_2 - encoder_last_hidden_state).abs().max().item() < 1e-3)

        with tempfile.TemporaryDirectory() as tmpdirname:
            decoder = model.get_decoder()
            decoder.save_pretrained(tmpdirname)
            decoder = DalleMegaDecoder.from_pretrained(tmpdirname).to(torch_device)

        last_hidden_state_2 = decoder(
            input_ids=inputs_dict["decoder_input_ids"],
            attention_mask=inputs_dict["decoder_attention_mask"],
            encoder_hidden_states=encoder_last_hidden_state,
            encoder_attention_mask=inputs_dict["attention_mask"],
        )[0]

        self.parent.assertTrue((last_hidden_state_2 - last_hidden_state).abs().max().item() < 1e-3)


@require_torch
class DalleMegaHeadTests(unittest.TestCase):
    vocab_size = 99

    def _get_config_and_data(self):
        input_ids = torch.tensor(
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
            ],
            dtype=torch.long,
            device=torch_device,
        )

        batch_size = input_ids.shape[0]
        config = DalleMegaConfig(
            encoder_vocab_size=self.vocab_size,
            decoder_vocab_size=self.vocab_size,
            d_model=24,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=32,
            decoder_ffn_dim=32,
            encoder_max_positions=48,
            decoder_max_positions=48,
            eos_token_id=2,
            pad_token_id=1,
            bos_token_id=0,
            decoder_start_token_id=2,
        )
        return config, input_ids, batch_size

    @timeout_decorator.timeout(1)
    def test_lm_forward(self):
        config, input_ids, batch_size = self._get_config_and_data()
        lm_labels = ids_tensor([batch_size, input_ids.shape[1]], config.decoder_vocab_size).to(torch_device)

        lm_model = DalleMegaForConditionalGeneration(config).to(torch_device)

        outputs = lm_model(input_ids=input_ids, labels=lm_labels)

        expected_shape = (batch_size, input_ids.shape[1], config.decoder_vocab_size)
        self.assertEqual(outputs["logits"].shape, expected_shape)
        self.assertIsInstance(outputs["loss"].item(), float)

    def test_lm_uneven_forward(self):
        config = DalleMegaConfig(
            encoder_vocab_size=self.vocab_size,
            decoder_vocab_size=self.vocab_size,
            d_model=14,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=8,
            decoder_ffn_dim=8,
            encoder_max_positions=48,
            decoder_max_positions=48,
            eos_token_id=2,
            pad_token_id=1,
            bos_token_id=0,
            decoder_start_token_id=2,
        )
        lm_model = DalleMegaForConditionalGeneration(config).to(torch_device)
        context = torch.tensor(
            [[71, 82, 18, 33, 46, 91, 2], [68, 34, 26, 58, 30, 2, 1]], device=torch_device, dtype=torch.long
        )
        summary = torch.tensor([[82, 71, 82, 18, 2], [58, 68, 2, 1, 1]], device=torch_device, dtype=torch.long)
        outputs = lm_model(input_ids=context, decoder_input_ids=summary, labels=summary)
        expected_shape = (*summary.shape, config.decoder_vocab_size)
        self.assertEqual(outputs["logits"].shape, expected_shape)

    def test_generate_beam_search(self):
        input_ids = torch.tensor([[71, 82, 2], [68, 34, 2]], device=torch_device, dtype=torch.long)
        config = DalleMegaConfig(
            encoder_vocab_size=self.vocab_size,
            decoder_vocab_size=self.vocab_size,
            d_model=24,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=32,
            decoder_ffn_dim=32,
            encoder_max_positions=48,
            decoder_max_positions=48,
            eos_token_id=2,
            pad_token_id=1,
            bos_token_id=0,
            decoder_start_token_id=2,
        )
        lm_model = DalleMegaForConditionalGeneration(config).to(torch_device)
        lm_model.eval()

        max_length = 5
        generated_ids = lm_model.generate(
            input_ids.clone(),
            do_sample=True,
            num_return_sequences=1,
            num_beams=2,
            no_repeat_ngram_size=3,
            max_length=max_length,
        )
        self.assertEqual(generated_ids.shape, (input_ids.shape[0], max_length))

    def test_shift_tokens_right(self):
        input_ids = torch.tensor([[71, 82, 18, 33, 2, 1, 1], [68, 34, 26, 58, 30, 82, 2]], dtype=torch.long)
        shifted = shift_tokens_right(input_ids, 1, 2)
        n_pad_before = input_ids.eq(1).float().sum()
        n_pad_after = shifted.eq(1).float().sum()
        self.assertEqual(shifted.shape, input_ids.shape)
        self.assertEqual(n_pad_after, n_pad_before - 1)
        self.assertTrue(torch.eq(shifted[:, 0], 2).all())

    @slow
    def test_tokenization(self):
        tokenizer = BartTokenizer.from_pretrained("facebook/dallemega-large")
        examples = [" Hello world", " DomDramg"]  # need leading spaces for equality
        fairseq_results = [
            torch.tensor([0, 20920, 232, 2]),
            torch.tensor([0, 11349, 495, 4040, 571, 2]),
        ]
        for ex, desired_result in zip(examples, fairseq_results):
            dallemega_toks = tokenizer.encode(ex, return_tensors="pt").squeeze()
            assert_tensors_close(desired_result.long(), dallemega_toks, prefix=ex)

    def test_generate_fp16(self):
        config, input_ids, batch_size = self._get_config_and_data()
        attention_mask = input_ids.ne(1).to(torch_device)
        model = DalleMegaForConditionalGeneration(config).eval().to(torch_device)
        if torch_device == "cuda":
            model.half()
        model.generate(input_ids, attention_mask=attention_mask)
        model.generate(num_beams=4, do_sample=True, early_stopping=False, num_return_sequences=3)

    def test_resize_tokens_embeddings_more(self):
        config, input_ids, _ = self._get_config_and_data()

        def _get_embs(m):
            return (m.get_input_embeddings().weight.data.clone(), m.get_output_embeddings().weight.data.clone())

        model = DalleMegaForConditionalGeneration(config).eval().to(torch_device)

        new_encoder_vocab_size = 45
        new_decoder_vocab_size = 46

        model.resize_token_embeddings(new_encoder_vocab_size)
        model.resize_decoder_token_embeddings(new_decoder_vocab_size)

        input_new, output_new = _get_embs(model)
        self.assertEqual(input_new.shape, (new_encoder_vocab_size, config.d_model))
        self.assertEqual(output_new.shape, (new_decoder_vocab_size, config.d_model))


@require_torch
class DalleMegaModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            DalleMegaModel,
            DalleMegaForConditionalGeneration,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (DalleMegaForConditionalGeneration,) if is_torch_available() else ()
    is_encoder_decoder = True
    fx_compatible = False
    test_pruning = False
    test_missing_keys = False

    def setUp(self):
        self.model_tester = DalleMegaModelTester(self)
        # we set has_text_modality to False since the model has two vocab_size size attributes
        # encoder_vocab_size and decoder_vocab_size, and we don't use the attrubute vocab_size
        # to avoid confusion.Setting has_text_modality=False won't check the vocab_size attribute.
        self.config_tester = ConfigTester(self, config_class=DalleMegaConfig, has_text_modality=False)

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

    def test_resize_tokens_embeddings(self):
        (
            original_config,
            inputs_dict,
        ) = self.model_tester.prepare_config_and_inputs_for_common()
        if not self.test_resize_embeddings:
            return

        for model_class in self.all_model_classes:
            config = copy.deepcopy(original_config)
            model = model_class(config)
            model.to(torch_device)

            if self.model_tester.is_training is False:
                model.eval()

            model_vocab_size = config.encoder_vocab_size
            # Retrieve the embeddings and clone theme
            model_embed = model.resize_token_embeddings(model_vocab_size)
            cloned_embeddings = model_embed.weight.clone()

            # Check that resizing the token embeddings with a larger vocab size increases the model's vocab size
            model_embed = model.resize_token_embeddings(model_vocab_size + 10)
            self.assertEqual(model.config.encoder_vocab_size, model_vocab_size + 10)
            # Check that it actually resizes the embeddings matrix
            self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] + 10)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that resizing the token embeddings with a smaller vocab size decreases the model's vocab size
            model_embed = model.resize_token_embeddings(model_vocab_size - 15)
            self.assertEqual(model.config.encoder_vocab_size, model_vocab_size - 15)
            # Check that it actually resizes the embeddings matrix
            self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] - 15)

            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            # Input ids should be clamped to the maximum size of the vocabulary
            inputs_dict["input_ids"].clamp_(max=model_vocab_size - 15 - 1)

            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that adding and removing tokens has not modified the first part of the embedding matrix.
            models_equal = True
            for p1, p2 in zip(cloned_embeddings, model_embed.weight):
                if p1.data.ne(p2.data).sum() > 0:
                    models_equal = False

            self.assertTrue(models_equal)

    @unittest.skip(
        "DalleMega does not share the embeddingsbetween encoder and decoder. This test is implemented for decoder"
        " below."
    )
    def test_resize_embeddings_untied(self):
        pass

    def test_resize_decoder_embeddings_tied(self):
        (
            original_config,
            inputs_dict,
        ) = self.model_tester.prepare_config_and_inputs_for_common()
        if not self.test_resize_embeddings:
            return

        original_config.tie_word_embeddings = True

        for model_class in self.all_model_classes:
            config = copy.deepcopy(original_config)
            model = model_class(config)
            model.to(torch_device)

            if self.model_tester.is_training is False:
                model.eval()

            model_vocab_size = config.decoder_vocab_size
            # Retrieve the embeddings and clone theme
            model_embed = model.resize_decoder_token_embeddings(model_vocab_size)
            cloned_embeddings = model_embed.weight.clone()

            # Check that resizing the token embeddings with a larger vocab size increases the model's vocab size
            model_embed = model.resize_decoder_token_embeddings(model_vocab_size + 10)
            self.assertEqual(model.config.decoder_vocab_size, model_vocab_size + 10)
            # Check that it actually resizes the embeddings matrix
            self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] + 10)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that resizing the token embeddings with a smaller vocab size decreases the model's vocab size
            model_embed = model.resize_decoder_token_embeddings(model_vocab_size - 15)
            self.assertEqual(model.config.decoder_vocab_size, model_vocab_size - 15)
            # Check that it actually resizes the embeddings matrix
            self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] - 15)

            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            # Input ids should be clamped to the maximum size of the vocabulary
            inputs_dict["decoder_input_ids"].clamp_(max=model_vocab_size - 15 - 1)

            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that adding and removing tokens has not modified the first part of the embedding matrix.
            models_equal = True
            for p1, p2 in zip(cloned_embeddings, model_embed.weight):
                if p1.data.ne(p2.data).sum() > 0:
                    models_equal = False

            self.assertTrue(models_equal)

    def test_resize_decoder_embeddings_untied(self):
        (
            original_config,
            inputs_dict,
        ) = self.model_tester.prepare_config_and_inputs_for_common()
        if not self.test_resize_embeddings:
            return

        original_config.tie_word_embeddings = False

        # if model cannot untied embeddings -> leave test
        if original_config.tie_word_embeddings:
            return

        for model_class in self.all_model_classes:
            config = copy.deepcopy(original_config)
            model = model_class(config).to(torch_device)

            # if no output embeddings -> leave test
            if model.get_output_embeddings() is None:
                continue

            # Check that resizing the token embeddings with a larger vocab size increases the model's vocab size
            model_vocab_size = config.decoder_vocab_size
            model.resize_decoder_token_embeddings(model_vocab_size + 10)
            self.assertEqual(model.config.decoder_vocab_size, model_vocab_size + 10)
            output_embeds = model.get_output_embeddings()
            self.assertEqual(output_embeds.weight.shape[0], model_vocab_size + 10)
            # Check bias if present
            if output_embeds.bias is not None:
                self.assertEqual(output_embeds.bias.shape[0], model_vocab_size + 10)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that resizing the token embeddings with a smaller vocab size decreases the model's vocab size
            model.resize_decoder_token_embeddings(model_vocab_size - 15)
            self.assertEqual(model.config.decoder_vocab_size, model_vocab_size - 15)
            # Check that it actually resizes the embeddings matrix
            output_embeds = model.get_output_embeddings()
            self.assertEqual(output_embeds.weight.shape[0], model_vocab_size - 15)
            # Check bias if present
            if output_embeds.bias is not None:
                self.assertEqual(output_embeds.bias.shape[0], model_vocab_size - 15)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            # Input ids should be clamped to the maximum size of the vocabulary
            if "decoder_input_ids" in inputs_dict:
                inputs_dict["decoder_input_ids"].clamp_(max=model_vocab_size - 15 - 1)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

    def test_tie_model_weights(self):
        if not self.test_torchscript:
            return

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_same_values(layer_1, layer_2):
            equal = True
            for p1, p2 in zip(layer_1.weight, layer_2.weight):
                if p1.data.ne(p2.data).sum() > 0:
                    equal = False
            return equal

        for model_class in self.all_model_classes:
            config.torchscript = True
            model_not_tied = model_class(config)
            if model_not_tied.get_output_embeddings() is None:
                continue

            config_tied = copy.deepcopy(config)
            config_tied.torchscript = False
            model_tied = model_class(config_tied)
            params_tied = list(model_tied.parameters())
            # Check that the embedding layer and decoding layer are the same in size and in value
            # self.assertTrue(check_same_values(embeddings, decoding))

            # # Check that after modification, they remain the same.
            # embeddings.weight.data.div_(2)
            # # Check that the embedding layer and decoding layer are the same in size and in value
            # self.assertTrue(embeddings.weight.shape, decoding.weight.shape)
            # self.assertTrue(check_same_values(embeddings, decoding))

            # # Check that after modification, they remain the same.
            # decoding.weight.data.div_(4)
            # # Check that the embedding layer and decoding layer are the same in size and in value
            # self.assertTrue(embeddings.weight.shape, decoding.weight.shape)
            # self.assertTrue(check_same_values(embeddings, decoding))

            # Check that after resize they remain tied.
            model_tied.resize_token_embeddings(config.encoder_vocab_size + 10)
            model_tied.resize_decoder_token_embeddings(config.decoder_vocab_size + 10)
            params_tied_2 = list(model_tied.parameters())
            self.assertEqual(len(params_tied_2), len(params_tied))

            # decoding.weight.data.mul_(20)
            # # Check that the embedding layer and decoding layer are the same in size and in value
            # self.assertTrue(model.transformer.wte.weight.shape, model.lm_head.weight.shape)
            # self.assertTrue(check_same_values(model.transformer.wte, model.lm_head))

    def _check_scores(self, batch_size, scores, length, config):
        expected_shape = (batch_size, config.decoder_vocab_size)
        self.assertIsInstance(scores, tuple)
        self.assertEqual(len(scores), length)
        self.assertListEqual([iter_scores.shape for iter_scores in scores], [expected_shape] * len(scores))

    def test_decoder_model_past_with_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

    def test_encoder_decoder_model_standalone(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.check_encoder_decoder_model_standalone(*config_and_inputs)

    # DalleMegaForSequenceClassification does not support inputs_embeds
    def test_inputs_embeds(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in (DalleMegaModel, DalleMegaForConditionalGeneration):
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

    def test_generate_fp16(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs()
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        model = DalleMegaForConditionalGeneration(config).eval().to(torch_device)
        if torch_device == "cuda":
            model.half()
        model.generate(input_ids, attention_mask=attention_mask)
        model.generate(num_beams=4, do_sample=True, early_stopping=False, num_return_sequences=3)


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


@require_torch
@require_sentencepiece
@require_tokenizers
class DalleMegaModelIntegrationTests(unittest.TestCase):
    @cached_property
    def default_tokenizer(self):
        return BartTokenizer.from_pretrained("facebook/dallemega-large")

    @slow
    def test_logits(self):
        pass

    @slow
    def test_generate(self):
        pass
