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
""" Testing suite for the PyTorch Whisper model. """

import copy
import inspect
import os
import tempfile
import unittest

import numpy as np

import transformers
from transformers import WhisperConfig
from transformers.testing_utils import is_pt_flax_cross_test, require_torch, require_torchaudio, slow, torch_device
from transformers.utils import cached_property, is_flax_available, is_torch_available
from transformers.utils.import_utils import is_datasets_available

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_datasets_available():
    import datasets
    from datasets import load_dataset

if is_torch_available():
    import torch

    from transformers import (
        WhisperFeatureExtractor,
        WhisperForAudioClassification,
        WhisperForConditionalGeneration,
        WhisperModel,
        WhisperProcessor,
        set_seed,
    )
    from transformers.models.whisper.modeling_whisper import WhisperDecoder, WhisperEncoder

if is_flax_available():
    import jax.numpy as jnp

    from transformers.modeling_flax_pytorch_utils import (
        convert_pytorch_state_dict_to_flax,
        load_flax_weights_in_pytorch_model,
    )


def prepare_whisper_inputs_dict(
    config,
    input_features,
    decoder_input_ids,
    attention_mask=None,
    decoder_attention_mask=None,
    head_mask=None,
    decoder_head_mask=None,
    cross_attn_head_mask=None,
):
    if decoder_attention_mask is None:
        decoder_attention_mask = decoder_input_ids.ne(config.pad_token_id)
    if head_mask is None:
        head_mask = torch.ones(config.encoder_layers, config.encoder_attention_heads, device=torch_device)
    if decoder_head_mask is None:
        decoder_head_mask = torch.ones(config.decoder_layers, config.decoder_attention_heads, device=torch_device)
    if cross_attn_head_mask is None:
        cross_attn_head_mask = torch.ones(config.decoder_layers, config.decoder_attention_heads, device=torch_device)
    return {
        # "input_ids": input_features,
        "input_features": input_features,
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": decoder_attention_mask,
        "head_mask": head_mask,
        "decoder_head_mask": decoder_head_mask,
        "cross_attn_head_mask": cross_attn_head_mask,
    }


@require_torch
class WhisperModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=60,
        is_training=True,
        use_labels=False,
        vocab_size=200,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        input_channels=1,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=20,
        max_source_positions=30,
        max_target_positions=40,
        bos_token_id=98,
        eos_token_id=98,
        pad_token_id=0,
        num_mel_bins=80,
        decoder_start_token_id=85,
        num_conv_layers=1,
        suppress_tokens=None,
        begin_suppress_tokens=None,
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
        self.input_channels = input_channels
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.num_mel_bins = num_mel_bins
        self.max_position_embeddings = max_position_embeddings
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.num_conv_layers = num_conv_layers
        self.suppress_tokens = suppress_tokens
        self.begin_suppress_tokens = begin_suppress_tokens

    def prepare_config_and_inputs(self):
        input_features = floats_tensor([self.batch_size, self.num_mel_bins, self.seq_length], self.vocab_size)

        decoder_input_ids = torch.tensor(self.batch_size * [[self.decoder_start_token_id]], device=torch_device)

        config = self.get_config()
        inputs_dict = prepare_whisper_inputs_dict(
            config,
            attention_mask=None,
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
        )
        return config, inputs_dict

    def get_config(self):
        return WhisperConfig(
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            encoder_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            encoder_attention_heads=self.num_attention_heads,
            decoder_attention_heads=self.num_attention_heads,
            input_channels=self.input_channels,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            max_source_positions=self.max_source_positions,
            max_target_positions=self.max_target_positions,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
            decoder_ffn_dim=self.hidden_size,
            encoder_ffn_dim=self.hidden_size,
            decoder_start_token_id=self.decoder_start_token_id,
            suppress_tokens=self.suppress_tokens,
            begin_suppress_tokens=self.begin_suppress_tokens,
        )

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def get_subsampled_output_lengths(self, input_lengths):
        """
        Computes the output length of the convolutional layers
        """

        for i in range(self.num_conv_layers):
            input_lengths = (input_lengths - 1) // 2 + 1

        return input_lengths

    def create_and_check_model_forward(self, config, inputs_dict, freeze_encoder=False):
        model = WhisperModel(config=config).to(torch_device).eval()

        if freeze_encoder:
            model.freeze_encoder()

        input_features = inputs_dict["input_features"]
        decoder_input_ids = inputs_dict["decoder_input_ids"]

        # first forward pass
        last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state

        self.parent.assertTrue(last_hidden_state.shape, (13, 7, 16))

    def create_and_check_decoder_model_past_large_inputs(self, config, inputs_dict):
        model = WhisperModel(config=config).get_decoder().to(torch_device).eval()
        input_ids = inputs_dict["decoder_input_ids"]
        attention_mask = inputs_dict["decoder_attention_mask"]

        # first forward pass
        outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)

        output, past_key_values = outputs.to_tuple()

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size).clamp(2)
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
        model = WhisperModel(config=config).to(torch_device).eval()
        outputs = model(**inputs_dict)

        encoder_last_hidden_state = outputs.encoder_last_hidden_state
        last_hidden_state = outputs.last_hidden_state

        with tempfile.TemporaryDirectory() as tmpdirname:
            encoder = model.get_encoder()
            encoder.save_pretrained(tmpdirname)
            encoder = WhisperEncoder.from_pretrained(tmpdirname).to(torch_device)

        encoder_last_hidden_state_2 = encoder(inputs_dict["input_features"])[0]

        self.parent.assertTrue((encoder_last_hidden_state_2 - encoder_last_hidden_state).abs().max().item() < 1e-3)

        with tempfile.TemporaryDirectory() as tmpdirname:
            decoder = model.get_decoder()
            decoder.save_pretrained(tmpdirname)
            decoder = WhisperDecoder.from_pretrained(tmpdirname).to(torch_device)

        last_hidden_state_2 = decoder(
            input_ids=inputs_dict["decoder_input_ids"],
            attention_mask=inputs_dict["decoder_attention_mask"],
            encoder_hidden_states=encoder_last_hidden_state,
        )[0]

        self.parent.assertTrue((last_hidden_state_2 - last_hidden_state).abs().max().item() < 1e-3)


@require_torch
class WhisperModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (WhisperModel, WhisperForConditionalGeneration) if is_torch_available() else ()
    all_generative_model_classes = (WhisperForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "audio-classification": WhisperForAudioClassification,
            "automatic-speech-recognition": WhisperForConditionalGeneration,
            "feature-extraction": WhisperModel,
        }
        if is_torch_available()
        else {}
    )
    is_encoder_decoder = True
    fx_compatible = False
    test_pruning = False
    test_missing_keys = False
    # Needs higher percentages after model tester's vocab_size is changed to 200 (PR #21222)
    # `0.5` is for `test_disk_offload` (which also works for `test_model_parallelism`)
    model_split_percents = [0.5, 0.8, 0.9]

    input_name = "input_features"

    # TODO: Fix the failed tests
    def is_pipeline_test_to_skip(
        self, pipeline_test_casse_name, config_class, model_architecture, tokenizer_name, processor_name
    ):
        if pipeline_test_casse_name in [
            "AutomaticSpeechRecognitionPipelineTests",
            "AudioClassificationPipelineTests",
        ]:
            # RuntimeError: The size of tensor a (1500) must match the size of tensor b (30) at non-singleton
            # dimension 1
            return True

        return False

    def setUp(self):
        self.model_tester = WhisperModelTester(self)
        self.config_tester = ConfigTester(self, config_class=WhisperConfig)
        self.maxDiff = 3000

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

    def test_model_forward(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_forward(*config_and_inputs)

    def test_model_forward_with_frozen_encoder(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_forward(*config_and_inputs, freeze_encoder=True)

    def test_requires_grad_with_frozen_encoder(self):
        config = self.model_tester.get_config()
        for model_class in self.all_model_classes:
            model = model_class(config)
            model.freeze_encoder()

            try:
                encoder_grads = [param.requires_grad for param in model.encoder.parameters()]
                decoder_grads = [param.requires_grad for param in model.decoder.parameters()]
            except AttributeError:
                encoder_grads = [param.requires_grad for param in model.model.encoder.parameters()]
                decoder_grads = [param.requires_grad for param in model.model.decoder.parameters()]

            self.assertFalse(all(encoder_grads))
            self.assertTrue(all(decoder_grads))

    def test_decoder_model_past_with_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

    def test_encoder_decoder_model_standalone(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.check_encoder_decoder_model_standalone(*config_and_inputs)

    def _get_input_ids_and_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        input_ids = inputs_dict[self.input_name]

        # cut to half length & take max batch_size 3
        max_batch_size = 3
        input_ids = input_ids[:max_batch_size, :, :]

        # generate max 3 tokens
        max_length = input_ids.shape[-1] + 3
        if config.eos_token_id is not None and config.pad_token_id is None:
            # hack to allow generate for models such as GPT2 as is done in `generate()`
            config.pad_token_id = config.eos_token_id

        return config, input_ids, None, max_length

    def test_inputs_embeds(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = copy.deepcopy(self._prepare_for_class(inputs_dict, model_class))

            decoder_input_ids = inputs.pop("decoder_input_ids", None)
            inputs.pop("decoder_attention_mask", None)

            wte = model.get_input_embeddings()
            inputs["decoder_inputs_embeds"] = wte(decoder_input_ids)

            with torch.no_grad():
                model(**inputs)[0]

    # training is not supported yet
    def test_training(self):
        pass

    def test_training_gradient_checkpointing(self):
        pass

    def test_generate_with_head_masking(self):
        pass

    def test_generate_fp16(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs()
        config.max_target_positions = 400
        input_features = input_dict["input_features"]
        model = WhisperForConditionalGeneration(config).eval().to(torch_device)
        if torch_device == "cuda":
            input_features = input_features.half()
            model.half()
        model.generate(input_features)
        model.generate(input_features, num_beams=4, do_sample=True, early_stopping=False, num_return_sequences=3)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = [
                "input_features",
                "attention_mask",
                "decoder_input_ids",
                "decoder_attention_mask",
            ]
            expected_arg_names.extend(
                ["head_mask", "decoder_head_mask", "cross_attn_head_mask", "encoder_outputs"]
                if "head_mask" and "decoder_head_mask" and "cross_attn_head_mask" in arg_names
                else ["encoder_outputs"]
            )
            self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states

            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
            )
            self.assertEqual(len(hidden_states), expected_num_layers)

            if hasattr(self.model_tester, "encoder_seq_length"):
                seq_length = self.model_tester.encoder_seq_length
            else:
                seq_length = self.model_tester.seq_length

            subsampled_seq_length = model._get_feat_extract_output_lengths(seq_length)

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [subsampled_seq_length, self.model_tester.hidden_size],
            )

            if config.is_encoder_decoder:
                hidden_states = outputs.decoder_hidden_states

                self.assertIsInstance(hidden_states, (list, tuple))
                self.assertEqual(len(hidden_states), expected_num_layers)

                decoder_seq_length = getattr(self.model_tester, "decoder_seq_length", 1)

                self.assertListEqual(
                    list(hidden_states[0].shape[-2:]),
                    [decoder_seq_length, self.model_tester.hidden_size],
                )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        seq_len = getattr(self.model_tester, "seq_length", None)
        decoder_seq_length = getattr(self.model_tester, "decoder_seq_length", 1)
        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", seq_len)
        decoder_key_length = getattr(self.model_tester, "decoder_key_length", 1)
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            subsampled_encoder_seq_length = model._get_feat_extract_output_lengths(encoder_seq_length)
            subsampled_encoder_key_length = model._get_feat_extract_output_lengths(encoder_key_length)

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
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            self.assertListEqual(
                list(attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, subsampled_encoder_seq_length, subsampled_encoder_key_length],
            )
            out_len = len(outputs)

            correct_outlen = 5

            # loss is at first position
            if "labels" in inputs_dict:
                correct_outlen += 1  # loss is added to beginning
            if "past_key_values" in outputs:
                correct_outlen += 1  # past_key_values have been returned

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
                    subsampled_encoder_key_length,
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

            added_hidden_states = 2
            self.assertEqual(out_len + added_hidden_states, len(outputs))

            self_attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions

            self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(self_attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, subsampled_encoder_seq_length, subsampled_encoder_key_length],
            )

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

            model_vocab_size = config.vocab_size
            # Retrieve the embeddings and clone theme
            model_embed = model.resize_token_embeddings(model_vocab_size)
            cloned_embeddings = model_embed.weight.clone()

            # Check that resizing the token embeddings with a larger vocab size increases the model's vocab size
            model_embed = model.resize_token_embeddings(model_vocab_size + 10)
            self.assertEqual(model.config.vocab_size, model_vocab_size + 10)
            # Check that it actually resizes the embeddings matrix
            self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] + 10)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that resizing the token embeddings with a smaller vocab size decreases the model's vocab size
            model_embed = model.resize_token_embeddings(model_vocab_size - 15)
            self.assertEqual(model.config.vocab_size, model_vocab_size - 15)
            # Check that it actually resizes the embeddings matrix
            self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] - 15)

            # make sure that decoder_input_ids are resized
            if "decoder_input_ids" in inputs_dict:
                inputs_dict["decoder_input_ids"].clamp_(max=model_vocab_size - 15 - 1)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that adding and removing tokens has not modified the first part of the embedding matrix.
            models_equal = True
            for p1, p2 in zip(cloned_embeddings, model_embed.weight):
                if p1.data.ne(p2.data).sum() > 0:
                    models_equal = False

            self.assertTrue(models_equal)

    def test_resize_embeddings_untied(self):
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
            model_vocab_size = config.vocab_size
            model.resize_token_embeddings(model_vocab_size + 10)
            self.assertEqual(model.config.vocab_size, model_vocab_size + 10)
            output_embeds = model.get_output_embeddings()
            self.assertEqual(output_embeds.weight.shape[0], model_vocab_size + 10)
            # Check bias if present
            if output_embeds.bias is not None:
                self.assertEqual(output_embeds.bias.shape[0], model_vocab_size + 10)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that resizing the token embeddings with a smaller vocab size decreases the model's vocab size
            model.resize_token_embeddings(model_vocab_size - 15)
            self.assertEqual(model.config.vocab_size, model_vocab_size - 15)
            # Check that it actually resizes the embeddings matrix
            output_embeds = model.get_output_embeddings()
            self.assertEqual(output_embeds.weight.shape[0], model_vocab_size - 15)
            # Check bias if present
            if output_embeds.bias is not None:
                self.assertEqual(output_embeds.bias.shape[0], model_vocab_size - 15)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            if "decoder_input_ids" in inputs_dict:
                inputs_dict["decoder_input_ids"].clamp_(max=model_vocab_size - 15 - 1)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

    def test_generate_without_input_ids(self):
        pass

    @staticmethod
    def _get_encoder_outputs(
        model, input_ids, attention_mask, output_attentions=None, output_hidden_states=None, num_interleave=1
    ):
        encoder = model.get_encoder()
        encoder_outputs = encoder(
            input_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.repeat_interleave(
            num_interleave, dim=0
        )
        input_ids = input_ids[:, :, 0]
        input_ids = torch.zeros_like(input_ids[:, :1], dtype=torch.long) + torch.tensor(
            [model._get_decoder_start_token_id()], device=input_ids.device
        )
        attention_mask = None
        return encoder_outputs, input_ids, attention_mask

    def _check_outputs(self, output, input_ids, config, use_cache=False, num_return_sequences=1):
        batch_size, mel, seq_length = input_ids.shape
        subsampled_seq_length = self.model_tester.get_subsampled_output_lengths(seq_length)
        num_sequences_in_output = batch_size * num_return_sequences
        gen_len = (
            output.sequences.shape[-1] - 1 if config.is_encoder_decoder else output.sequences.shape[-1] - seq_length
        )

        # scores
        self._check_scores(num_sequences_in_output, output.scores, length=gen_len, config=config)

        # Attentions
        # encoder
        self._check_encoder_attention_for_generate(
            output.encoder_attentions, batch_size, config, subsampled_seq_length
        )
        # decoder
        self._check_attentions_for_generate(
            num_sequences_in_output,
            output.decoder_attentions,
            min_length=1,
            max_length=output.sequences.shape[-1],
            config=config,
            use_cache=use_cache,
        )

        # Hidden States
        # encoder
        self._check_encoder_hidden_states_for_generate(
            output.encoder_hidden_states, batch_size, config, subsampled_seq_length
        )

        # decoder
        self._check_hidden_states_for_generate(
            num_sequences_in_output,
            output.decoder_hidden_states,
            min_length=1,
            max_length=output.sequences.shape[-1],
            config=config,
            use_cache=use_cache,
        )

    def _create_and_check_torchscript(self, config, inputs_dict):
        if not self.test_torchscript:
            return

        configs_no_init = _config_zero_init(config)  # To be sure we have no Nan
        configs_no_init.torchscript = True
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            model.to(torch_device)
            model.eval()
            inputs = self._prepare_for_class(inputs_dict, model_class)

            try:
                model.config.use_cache = False  # FSTM still requires this hack -> FSTM should probably be refactored similar to BART afterward
                input_features = inputs["input_features"]
                decoder_input_ids = inputs["decoder_input_ids"]
                decoder_attention_mask = inputs["decoder_attention_mask"]
                # prepare `attention_mask` with shape (batch_size, sequence_length)
                attention_mask = torch.ones(
                    input_features.shape[0],
                    input_features.shape[-1],
                    device=input_features.device,
                    dtype=input_features.dtype,
                )
                traced_model = torch.jit.trace(
                    model, (input_features, attention_mask, decoder_input_ids, decoder_attention_mask)
                )

            except RuntimeError:
                self.fail("Couldn't trace module.")

            with tempfile.TemporaryDirectory() as tmp_dir_name:
                pt_file_name = os.path.join(tmp_dir_name, "traced_model.pt")

                try:
                    torch.jit.save(traced_model, pt_file_name)
                except Exception:
                    self.fail("Couldn't save module.")

                try:
                    loaded_model = torch.jit.load(pt_file_name)
                except Exception:
                    self.fail("Couldn't load module.")

            model.to(torch_device)
            model.eval()

            loaded_model.to(torch_device)
            loaded_model.eval()

            model_state_dict = model.state_dict()
            loaded_model_state_dict = loaded_model.state_dict()

            self.assertEqual(set(model_state_dict.keys()), set(loaded_model_state_dict.keys()))

            models_equal = True
            for layer_name, p1 in model_state_dict.items():
                p2 = loaded_model_state_dict[layer_name]
                if p1.data.ne(p2.data).sum() > 0:
                    models_equal = False

            self.assertTrue(models_equal)

    @is_pt_flax_cross_test
    def test_equivalence_pt_to_flax(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        init_shape = (1,) + inputs_dict["input_features"].shape[1:]

        for model_class in self.all_model_classes:
            with self.subTest(model_class.__name__):
                fx_model_class_name = "Flax" + model_class.__name__

                if not hasattr(transformers, fx_model_class_name):
                    # no flax model exists for this class
                    return

                # Output all for aggressive testing
                config.output_hidden_states = True
                config.output_attentions = self.has_attentions

                fx_model_class = getattr(transformers, fx_model_class_name)

                # load PyTorch class
                pt_model = model_class(config).eval()
                # Flax models don't use the `use_cache` option and cache is not returned as a default.
                # So we disable `use_cache` here for PyTorch model.
                pt_model.config.use_cache = False

                # load Flax class
                fx_model = fx_model_class(config, input_shape=init_shape, dtype=jnp.float32)

                # make sure only flax inputs are forward that actually exist in function args
                fx_input_keys = inspect.signature(fx_model.__call__).parameters.keys()

                # prepare inputs
                pt_inputs = self._prepare_for_class(inputs_dict, model_class)

                # remove function args that don't exist in Flax
                pt_inputs = {k: v for k, v in pt_inputs.items() if k in fx_input_keys}

                # send pytorch inputs to the correct device
                pt_inputs = {
                    k: v.to(device=torch_device) if isinstance(v, torch.Tensor) else v for k, v in pt_inputs.items()
                }

                # convert inputs to Flax
                fx_inputs = {k: np.array(v) for k, v in pt_inputs.items() if torch.is_tensor(v)}

                fx_state = convert_pytorch_state_dict_to_flax(pt_model.state_dict(), fx_model)
                fx_model.params = fx_state

                # send pytorch model to the correct device
                pt_model.to(torch_device)

                with torch.no_grad():
                    pt_outputs = pt_model(**pt_inputs)
                fx_outputs = fx_model(**fx_inputs)

                fx_keys = tuple([k for k, v in fx_outputs.items() if v is not None])
                pt_keys = tuple([k for k, v in pt_outputs.items() if v is not None])

                self.assertEqual(fx_keys, pt_keys)
                self.check_pt_flax_outputs(fx_outputs, pt_outputs, model_class)

                with tempfile.TemporaryDirectory() as tmpdirname:
                    pt_model.save_pretrained(tmpdirname)
                    fx_model_loaded = fx_model_class.from_pretrained(tmpdirname, input_shape=init_shape, from_pt=True)

                fx_outputs_loaded = fx_model_loaded(**fx_inputs)

                fx_keys = tuple([k for k, v in fx_outputs_loaded.items() if v is not None])
                pt_keys = tuple([k for k, v in pt_outputs.items() if v is not None])

                self.assertEqual(fx_keys, pt_keys)
                self.check_pt_flax_outputs(fx_outputs_loaded, pt_outputs, model_class)

    @is_pt_flax_cross_test
    def test_equivalence_flax_to_pt(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        init_shape = (1,) + inputs_dict["input_features"].shape[1:]

        for model_class in self.all_model_classes:
            with self.subTest(model_class.__name__):
                fx_model_class_name = "Flax" + model_class.__name__

                if not hasattr(transformers, fx_model_class_name):
                    # no flax model exists for this class
                    return

                # Output all for aggressive testing
                config.output_hidden_states = True
                config.output_attentions = self.has_attentions

                fx_model_class = getattr(transformers, fx_model_class_name)

                # load PyTorch class
                pt_model = model_class(config).eval()
                # Flax models don't use the `use_cache` option and cache is not returned as a default.
                # So we disable `use_cache` here for PyTorch model.
                pt_model.config.use_cache = False

                # load Flax class
                fx_model = fx_model_class(config, input_shape=init_shape, dtype=jnp.float32)

                # make sure only flax inputs are forward that actually exist in function args
                fx_input_keys = inspect.signature(fx_model.__call__).parameters.keys()

                # prepare inputs
                pt_inputs = self._prepare_for_class(inputs_dict, model_class)

                # remove function args that don't exist in Flax
                pt_inputs = {k: v for k, v in pt_inputs.items() if k in fx_input_keys}

                # send pytorch inputs to the correct device
                pt_inputs = {
                    k: v.to(device=torch_device) if isinstance(v, torch.Tensor) else v for k, v in pt_inputs.items()
                }

                # convert inputs to Flax
                fx_inputs = {k: np.array(v) for k, v in pt_inputs.items() if torch.is_tensor(v)}

                pt_model = load_flax_weights_in_pytorch_model(pt_model, fx_model.params)

                # make sure weights are tied in PyTorch
                pt_model.tie_weights()

                # send pytorch model to the correct device
                pt_model.to(torch_device)

                with torch.no_grad():
                    pt_outputs = pt_model(**pt_inputs)
                fx_outputs = fx_model(**fx_inputs)

                fx_keys = tuple([k for k, v in fx_outputs.items() if v is not None])
                pt_keys = tuple([k for k, v in pt_outputs.items() if v is not None])

                self.assertEqual(fx_keys, pt_keys)
                self.check_pt_flax_outputs(fx_outputs, pt_outputs, model_class)

                with tempfile.TemporaryDirectory() as tmpdirname:
                    fx_model.save_pretrained(tmpdirname)
                    pt_model_loaded = model_class.from_pretrained(tmpdirname, from_flax=True)

                # send pytorch model to the correct device
                pt_model_loaded.to(torch_device)
                pt_model_loaded.eval()

                with torch.no_grad():
                    pt_outputs_loaded = pt_model_loaded(**pt_inputs)

                fx_keys = tuple([k for k, v in fx_outputs.items() if v is not None])
                pt_keys = tuple([k for k, v in pt_outputs_loaded.items() if v is not None])

                self.assertEqual(fx_keys, pt_keys)
                self.check_pt_flax_outputs(fx_outputs, pt_outputs_loaded, model_class)

    def test_mask_feature_prob(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.mask_feature_prob = 0.2
        config.mask_feature_length = 2

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.train()

            # forward pass
            encoder_last_hidden_state = model(**input_dict).encoder_last_hidden_state
            self.assertTrue(encoder_last_hidden_state.shape, (13, 30, 16))

    def test_mask_time_prob(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.mask_time_prob = 0.2
        config.mask_time_length = 2

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.train()

            # forward pass
            encoder_last_hidden_state = model(**input_dict).encoder_last_hidden_state
            self.assertTrue(encoder_last_hidden_state.shape, (13, 30, 16))


@require_torch
@require_torchaudio
class WhisperModelIntegrationTests(unittest.TestCase):
    @cached_property
    def default_processor(self):
        return WhisperProcessor.from_pretrained("openai/whisper-base")

    def _load_datasamples(self, num_samples):
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        # automatic decoding with librispeech
        speech_samples = ds.sort("id").select(range(num_samples))[:num_samples]["audio"]

        return [x["array"] for x in speech_samples]

    @slow
    def test_tiny_logits_librispeech(self):
        torch_device = "cpu"
        set_seed(0)
        model = WhisperModel.from_pretrained("openai/whisper-tiny")
        model.to(torch_device)
        input_speech = self._load_datasamples(1)
        feature_extractor = WhisperFeatureExtractor()
        input_features = feature_extractor(input_speech, return_tensors="pt").input_features

        with torch.no_grad():
            logits = model(
                input_features,
                decoder_input_ids=torch.tensor([[50258, 50259, 50359]]),
                output_hidden_states=False,
                output_attentions=False,
                return_dict=False,
                use_cache=False,
            )

        # fmt: off
        EXPECTED_LOGITS = torch.tensor(
            [
                2.9892, -6.7607, 5.7348, 3.6096, 0.2152, -5.7321, 4.8855, -1.6407,
                0.2823, -1.5718, 10.4269, 3.4427, 0.0219, -8.0612, 3.4784, 8.4246,
                4.0575, -2.2864, 11.1084, 0.9963, 0.9884, -8.5154, -3.5469, -9.3713,
                0.9786, 3.5435, 7.4850, -5.2579, -1.4366, 10.4841
            ]
        )
        # fmt: on
        self.assertTrue(torch.allclose(logits[0][0, 0, :30].cpu(), EXPECTED_LOGITS, atol=1e-4))

        # fmt: off
        EXPECTED_GENERATION = torch.tensor(
            [
                -1.4651, -2.6944, 2.7821, 2.3793, 4.0738, 0.0188, -3.3203, 1.9836,
                0.0520, 0.7095, 1.1063, 0.2952, -3.6786, -0.5249, 0.3105, 4.7691,
                1.1562, 1.3046, 0.5810, -0.3624, 1.7006, 1.3424, 0.9817, 2.1958,
                1.8775, -5.7046, -0.7679, 4.0113, 2.6848, 2.8609
            ]
        )
        # fmt: on

        head_logits = logits[0] @ model.decoder.embed_tokens.weight.T
        self.assertTrue(torch.allclose(head_logits[0, 0, :30].cpu(), EXPECTED_GENERATION, atol=1e-4))

    @slow
    def test_small_en_logits_librispeech(self):
        set_seed(0)
        torch_device = "cpu"
        model = WhisperModel.from_pretrained("openai/whisper-small.en")
        model.to(torch_device)

        input_speech = self._load_datasamples(1)

        feaure_extractor = WhisperFeatureExtractor()
        input_features = feaure_extractor(input_speech, return_tensors="pt").input_features.to(torch_device)

        logits = model(
            input_features,
            decoder_input_ids=torch.tensor([[model.config.decoder_start_token_id]]),
            output_hidden_states=False,
            output_attentions=False,
            use_cache=False,
        )

        logits = logits.last_hidden_state @ model.decoder.embed_tokens.weight.T

        # fmt: off
        EXPECTED_LOGITS = torch.tensor(
            [
                -3.6784, -7.7211, -9.5070, -11.9286, -7.6489, -9.7026, -5.6188,
                -8.0104, -4.6238, -5.1833, -9.0485, -3.4079, -5.4874, -2.6935,
                -6.3479, -7.3398, -6.9558, -7.6867, -7.4748, -8.3463, -9.9781,
                -10.8389, -10.3105, -11.7201, -9.7261, -7.1590, -5.9272, -12.4509,
                -11.1146, -8.1918
            ]
        )
        # fmt: on
        self.assertTrue(torch.allclose(logits[0, 0, :30].cpu(), EXPECTED_LOGITS, atol=1e-4))

    @slow
    def test_large_logits_librispeech(self):
        set_seed(0)

        torch_device = "cpu"
        model = WhisperModel.from_pretrained("openai/whisper-large")
        model.to(torch_device)

        input_speech = self._load_datasamples(1)

        processor = WhisperProcessor.from_pretrained("openai/whisper-large")
        processed_inputs = processor(
            audio=input_speech, text="This part of the speech", add_special_tokens=False, return_tensors="pt"
        )
        input_features = processed_inputs.input_features.to(torch_device)
        decoder_input_ids = processed_inputs.labels.to(torch_device)

        logits = model(
            input_features,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=False,
            output_attentions=False,
            use_cache=False,
        )

        logits = logits.last_hidden_state @ model.decoder.embed_tokens.weight.T

        # fmt: off
        EXPECTED_LOGITS = torch.tensor(
            [
                2.1382, 0.9381, 4.4671, 3.5589, 2.4022, 3.8576, -0.6521, 2.5472,
                1.8301, 1.9957, 2.3432, 1.4678, 0.5459, 2.2597, 1.5179, 2.5357,
                1.1624, 0.6194, 1.0757, 1.8259, 2.4076, 1.6601, 2.3503, 1.3376,
                1.9891, 1.8635, 3.8931, 5.3699, 4.4772, 3.9184
            ]
        )
        # fmt: on

        self.assertTrue(torch.allclose(logits[0, 0, :30].cpu(), EXPECTED_LOGITS, atol=1e-4))

    @slow
    def test_tiny_en_generation(self):
        torch_device = "cpu"
        set_seed(0)
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
        model.to(torch_device)
        model.config.decoder_start_token_id = 50257

        input_speech = self._load_datasamples(1)
        input_features = processor.feature_extractor(raw_speech=input_speech, return_tensors="pt").input_features.to(
            torch_device
        )

        generated_ids = model.generate(input_features, num_beams=5, max_length=20)
        transcript = processor.tokenizer.batch_decode(generated_ids)[0]

        EXPECTED_TRANSCRIPT = (
            "<|startoftranscript|><|notimestamps|> Mr. Quilter is the apostle of the middle"
            " classes, and we are glad to"
        )
        self.assertEqual(transcript, EXPECTED_TRANSCRIPT)

    @slow
    def test_tiny_generation(self):
        torch_device = "cpu"
        set_seed(0)
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
        model.to(torch_device)

        input_speech = self._load_datasamples(1)
        input_features = processor.feature_extractor(raw_speech=input_speech, return_tensors="pt").input_features.to(
            torch_device
        )

        generated_ids = model.generate(input_features, num_beams=5, max_length=20)
        transcript = processor.tokenizer.decode(generated_ids[0])

        EXPECTED_TRANSCRIPT = (
            "<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Mr. Quilter is the apostle of the middle"
            " classes and we are glad"
        )
        self.assertEqual(transcript, EXPECTED_TRANSCRIPT)

    @slow
    def test_large_generation(self):
        torch_device = "cpu"
        set_seed(0)
        processor = WhisperProcessor.from_pretrained("openai/whisper-large")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")
        model.to(torch_device)

        input_speech = self._load_datasamples(1)
        input_features = processor.feature_extractor(raw_speech=input_speech, return_tensors="pt").input_features.to(
            torch_device
        )

        generated_ids = model.generate(
            input_features, do_sample=False, max_length=20, language="<|en|>", task="transcribe"
        )
        transcript = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        EXPECTED_TRANSCRIPT = " Mr. Quilter is the apostle of the middle classes and we are glad"
        self.assertEqual(transcript, EXPECTED_TRANSCRIPT)

    @slow
    def test_large_generation_multilingual(self):
        torch_device = "cpu"
        set_seed(0)
        processor = WhisperProcessor.from_pretrained("openai/whisper-large")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")
        model.to(torch_device)

        ds = load_dataset("common_voice", "ja", split="test", streaming=True)
        ds = ds.cast_column("audio", datasets.Audio(sampling_rate=16_000))
        input_speech = next(iter(ds))["audio"]["array"]
        input_features = processor.feature_extractor(raw_speech=input_speech, return_tensors="pt").input_features.to(
            torch_device
        )

        generated_ids = model.generate(
            input_features, do_sample=False, max_length=20, language="<|ja|>", task="transcribe"
        )
        transcript = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        EXPECTED_TRANSCRIPT = "木村さんに電話を貸してもらいました"
        self.assertEqual(transcript, EXPECTED_TRANSCRIPT)

        generated_ids = model.generate(
            input_features, do_sample=False, max_length=20, language="<|en|>", task="transcribe"
        )
        transcript = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        EXPECTED_TRANSCRIPT = " Kimura-san called me."
        self.assertEqual(transcript, EXPECTED_TRANSCRIPT)

        generated_ids = model.generate(
            input_features, do_sample=False, max_length=20, language="<|ja|>", task="translate"
        )
        transcript = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        EXPECTED_TRANSCRIPT = " I borrowed a phone from Kimura san"
        self.assertEqual(transcript, EXPECTED_TRANSCRIPT)

    @slow
    def test_large_batched_generation(self):
        set_seed(0)
        processor = WhisperProcessor.from_pretrained("openai/whisper-large")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")

        input_speech = self._load_datasamples(4)
        input_features = processor.feature_extractor(raw_speech=input_speech, return_tensors="pt").input_features
        generated_ids = model.generate(input_features, max_length=20, task="translate")

        # fmt: off
        EXPECTED_LOGITS = torch.tensor(
            [
                [50258, 50259, 50358, 50363, 2221, 13, 2326, 388, 391, 307, 264, 50244, 295, 264, 2808, 5359, 293, 321, 366, 5404],
                [50258, 50259, 50358, 50363, 6966, 307, 2221, 13, 2326, 388, 391, 311, 9060, 1570, 1880, 813, 702, 1871, 13, 50257],
                [50258, 50259, 50358, 50363, 634, 5112, 505, 300, 412, 341, 42729, 3196, 295, 264, 1064, 11, 365, 5272, 293, 12904],
                [50258, 50259, 50358, 50363, 634, 575, 12525, 22618, 1968, 6144, 35617, 20084, 1756, 311, 589, 307, 534, 10281, 934, 439]
            ]
        )
        # fmt: on

        self.assertTrue(torch.allclose(generated_ids, EXPECTED_LOGITS))

        # fmt: off
        EXPECTED_TRANSCRIPT = [
            " Mr. Quilter is the apostle of the middle classes and we are glad",
            " Nor is Mr. Quilter's manner less interesting than his matter.",
            " He tells us that at this festive season of the year, with Christmas and roast",
            " He has grave doubts whether Sir Frederick Layton's work is really Greek after all",
        ]
        # fmt: on

        transcript = processor.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertListEqual(transcript, EXPECTED_TRANSCRIPT)

    @slow
    def test_tiny_en_batched_generation(self):
        set_seed(0)
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
        model.to(torch_device)

        input_speech = self._load_datasamples(4)
        input_features = processor.feature_extractor(raw_speech=input_speech, return_tensors="pt").input_features.to(
            torch_device
        )
        generated_ids = model.generate(input_features, max_length=20).to("cpu")

        # fmt: off
        EXPECTED_LOGITS = torch.tensor(
            [
                [50257, 50362, 1770, 13, 2264, 346, 353, 318, 262, 46329, 286, 262, 3504, 6097, 11, 290, 356, 389, 9675, 284],
                [50257, 50362, 5414, 318, 1770, 13, 2264, 346, 353, 338, 5642, 1342, 3499, 621, 465, 2300, 13, 50256, 50256, 50256],
                [50257, 50362, 679, 4952, 514, 326, 379, 428, 43856, 1622, 286, 262, 614, 11, 351, 6786, 290, 32595, 12023, 28236],
                [50257, 50362, 679, 468, 12296, 17188, 1771, 7361, 26113, 18881, 1122, 338, 670, 318, 1107, 8312, 706, 477, 290, 460]
            ]

        )
        # fmt: on

        self.assertTrue(torch.allclose(generated_ids, EXPECTED_LOGITS))

        # fmt: off
        EXPECTED_TRANSCRIPT = [
            " Mr. Quilter is the apostle of the middle classes, and we are glad to",
            " Nor is Mr. Quilter's manner less interesting than his matter.",
            " He tells us that at this festive season of the year, with Christmas and roast beef looming",
            " He has grave doubts whether Sir Frederick Layton's work is really Greek after all and can",
        ]
        # fmt: on

        transcript = processor.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertListEqual(transcript, EXPECTED_TRANSCRIPT)

    @slow
    def test_tiny_timestamp_generation(self):
        set_seed(0)
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
        model.to(torch_device)

        input_speech = np.concatenate(self._load_datasamples(4))
        input_features = processor.feature_extractor(raw_speech=input_speech, return_tensors="pt").input_features.to(
            torch_device
        )

        generated_ids = model.generate(input_features, max_length=448, return_timestamps=True).to("cpu")

        # fmt: off
        EXPECTED_OUTPUT = torch.tensor([50258, 50259, 50359, 50364, 2221, 13, 2326, 388, 391, 307, 264, 50244, 295, 264, 2808, 5359, 11, 293, 321, 366, 5404, 281, 2928, 702, 14943, 13, 50692, 50692, 6966, 307, 2221, 13, 2326, 388, 391, 311, 9060, 1570, 1880, 813, 702, 1871, 13, 50926, 50926, 634, 5112, 505, 300, 412, 341, 42729, 3196, 295, 264, 1064, 11, 365, 5272, 293, 12904, 9256, 450, 10539, 51208, 51208, 949, 505, 11, 14138, 10117, 490, 3936, 293, 1080, 3542, 5160, 881, 26336, 281, 264, 1575, 13, 51552, 51552, 634, 575, 12525, 22618, 1968, 6144, 35617, 7354, 1292, 6, 589, 307, 534, 10281, 934, 439, 11, 293, 51836, 51836, 50257])
        # fmt: on

        self.assertTrue(torch.allclose(generated_ids, EXPECTED_OUTPUT))

        EXPECTED_TRANSCRIPT = [
            {
                "text": (
                    " Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel. Nor is"
                    " Mr. Quilter's manner less interesting than his matter. He tells us that at this festive season"
                    " of the year, with Christmas and roast beef looming before us, similarly drawn from eating and"
                    " its results occur most readily to the mind. He has grave doubts whether Sir Frederick Latins'"
                    " work is really Greek after all, and"
                ),
                "offsets": [
                    {
                        "text": (
                            " Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."
                        ),
                        "timestamp": (0.0, 6.5600000000000005),
                    },
                    {
                        "text": " Nor is Mr. Quilter's manner less interesting than his matter.",
                        "timestamp": (6.5600000000000005, 11.24),
                    },
                    {
                        "text": (
                            " He tells us that at this festive season of the year, with Christmas and roast beef"
                            " looming"
                        ),
                        "timestamp": (11.24, 16.88),
                    },
                    {
                        "text": (
                            " before us, similarly drawn from eating and its results occur most readily to the mind."
                        ),
                        "timestamp": (16.88, 23.76),
                    },
                    {
                        "text": (
                            " He has grave doubts whether Sir Frederick Latins' work is really Greek after all, and"
                        ),
                        "timestamp": (23.76, 29.44),
                    },
                ],
            }
        ]

        transcript = processor.batch_decode(generated_ids, skip_special_tokens=True, output_offsets=True)
        self.assertEqual(transcript, EXPECTED_TRANSCRIPT)

    @slow
    def test_tiny_specaugment_librispeech(self):
        torch_device = "cpu"
        set_seed(0)
        # Apply SpecAugment
        model = WhisperModel.from_pretrained("openai/whisper-tiny", apply_spec_augment=True)
        # Set model to training mode to enable SpecAugment
        model.train()
        model.to(torch_device)
        input_speech = self._load_datasamples(1)
        feature_extractor = WhisperFeatureExtractor()
        input_features = feature_extractor(input_speech, return_tensors="pt").input_features

        with torch.no_grad():
            logits = model(
                input_features,
                decoder_input_ids=torch.tensor([[50258, 50259, 50359]]),
                output_hidden_states=False,
                output_attentions=False,
                return_dict=False,
                use_cache=False,
            )

        # fmt: off
        EXPECTED_LOGITS = torch.tensor(
            [
                0.9362, -4.7105, 5.0879, 3.9642, 1.0013, -6.0096, 4.7285, -3.1847,
                -0.8648, 1.9631, 6.2653, 3.6936, 0.3575, -4.5818, 3.0564, 7.8712,
                2.9951, 0.6848, 9.9497, -2.6638, 1.1571, -6.8546, -1.4333, -7.7584,
                1.1200, 3.9030, 4.4655, -4.4919, -1.1703, 9.6241
            ]
        )
        # fmt: on
        self.assertTrue(torch.allclose(logits[0][0, 0, :30].cpu(), EXPECTED_LOGITS, atol=1e-4))


def prepare_whisper_encoder_inputs_dict(config, input_features, head_mask=None):
    if head_mask is None:
        head_mask = torch.ones(config.encoder_layers, config.encoder_attention_heads, device=torch_device)
    return {"input_features": input_features, "head_mask": head_mask}


@require_torch
class WhisperEncoderModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=60,
        is_training=True,
        use_labels=True,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        input_channels=1,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=20,
        max_source_positions=30,
        num_mel_bins=80,
        num_conv_layers=1,
        suppress_tokens=None,
        begin_suppress_tokens=None,
        classifier_proj_size=4,
        num_labels=2,
        is_encoder_decoder=False,
        is_decoder=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.input_channels = input_channels
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.num_mel_bins = num_mel_bins
        self.max_position_embeddings = max_position_embeddings
        self.max_source_positions = max_source_positions
        self.num_conv_layers = num_conv_layers
        self.suppress_tokens = suppress_tokens
        self.begin_suppress_tokens = begin_suppress_tokens
        self.classifier_proj_size = classifier_proj_size
        self.num_labels = num_labels
        self.is_encoder_decoder = is_encoder_decoder
        self.is_decoder = is_decoder

    def get_config(self):
        return WhisperConfig(
            d_model=self.hidden_size,
            encoder_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            encoder_attention_heads=self.num_attention_heads,
            decoder_attention_heads=self.num_attention_heads,
            input_channels=self.input_channels,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            max_source_positions=self.max_source_positions,
            decoder_ffn_dim=self.hidden_size,
            encoder_ffn_dim=self.hidden_size,
            suppress_tokens=self.suppress_tokens,
            begin_suppress_tokens=self.begin_suppress_tokens,
            classifier_proj_size=self.classifier_proj_size,
            num_labels=self.num_labels,
            is_encoder_decoder=self.is_encoder_decoder,
            is_decoder=self.is_decoder,
        )

    def prepare_config_and_inputs(self):
        input_features = floats_tensor([self.batch_size, self.num_mel_bins, self.seq_length])

        config = self.get_config()
        inputs_dict = prepare_whisper_encoder_inputs_dict(
            config,
            input_features=input_features,
        )
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def get_subsampled_output_lengths(self, input_lengths):
        """
        Computes the output length of the convolutional layers
        """

        for i in range(self.num_conv_layers):
            input_lengths = (input_lengths - 1) // 2 + 1

        return input_lengths

    @property
    def encoder_seq_length(self):
        return self.get_subsampled_output_lengths(self.seq_length)

    def create_and_check_model_forward(self, config, inputs_dict, freeze_encoder=False):
        model = WhisperForAudioClassification(config=config).to(torch_device).eval()

        if freeze_encoder:
            model.freeze_encoder()

        input_features = inputs_dict["input_features"]

        # first forward pass
        last_hidden_state = model(input_features).logits

        self.parent.assertTrue(last_hidden_state.shape, (13, 2))


@require_torch
class WhisperEncoderModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (WhisperForAudioClassification,) if is_torch_available() else ()
    is_encoder_decoder = False
    fx_compatible = False
    test_pruning = False
    test_missing_keys = False

    input_name = "input_features"

    def setUp(self):
        self.model_tester = WhisperEncoderModelTester(self)
        self.config_tester = ConfigTester(self, config_class=WhisperConfig)
        self.maxDiff = 3000

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["input_features", "head_mask", "encoder_outputs"]
            self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)

    @unittest.skip(reason="Some undefined behavior encountered with tiny versions of this model. Skip for now.")
    def test_cpu_offload(self):
        pass

    @unittest.skip(reason="Some undefined behavior encountered with tiny versions of this model. Skip for now.")
    def test_disk_offload(self):
        pass

    @unittest.skip(reason="Some undefined behavior encountered with tiny versions of this model. Skip for now.")
    def test_model_parallelism(self):
        pass

    # input embeds is meaningless for an encoder-only acoustic model
    def test_inputs_embeds(self):
        pass

    # the equivalent test is passing the encoder outputs directly to the model
    def test_encoder_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = copy.deepcopy(self._prepare_for_class(inputs_dict, model_class))

            with torch.no_grad():
                outputs = model(**inputs)[0]

            input_ids = inputs["input_features"]
            del inputs["input_features"]

            encoder = model.encoder

            with torch.no_grad():
                inputs["encoder_outputs"] = encoder(input_ids)
                outputs_embeds = model(**inputs)[0]

            self.assertTrue((outputs_embeds == outputs).all())

    # Needs to override as the encoder input embedding is a Conv1d
    def test_model_common_attributes(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (torch.nn.Conv1d))
            model.set_input_embeddings(torch.nn.Conv1d(10, 10, 3))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, torch.nn.Conv1d))

    # WhisperEncoder cannot resize token embeddings since it has no tokens embeddings
    def test_resize_tokens_embeddings(self):
        pass
