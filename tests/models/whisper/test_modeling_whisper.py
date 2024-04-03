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
import random
import re
import tempfile
import time
import unittest

import numpy as np
import pytest
from huggingface_hub import hf_hub_download

import transformers
from transformers import WhisperConfig
from transformers.testing_utils import (
    is_pt_flax_cross_test,
    require_flash_attn,
    require_torch,
    require_torch_fp16,
    require_torch_gpu,
    require_torchaudio,
    slow,
    torch_device,
)
from transformers.utils import cached_property, is_flax_available, is_torch_available, is_torchaudio_available
from transformers.utils.import_utils import is_datasets_available

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_datasets_available():
    import datasets
    from datasets import Audio, load_dataset

if is_torch_available():
    import torch

    from transformers import (
        WhisperFeatureExtractor,
        WhisperForAudioClassification,
        WhisperForCausalLM,
        WhisperForConditionalGeneration,
        WhisperModel,
        WhisperProcessor,
        set_seed,
    )
    from transformers.generation.logits_process import LogitsProcessor
    from transformers.models.whisper.modeling_whisper import WhisperDecoder, WhisperEncoder, sinusoids

    class DummyTimestampLogitProcessor(LogitsProcessor):
        """This processor fakes the correct timestamps tokens pattern [TOK_1] [TOK_2] ... [TOK_N] [TIME_STAMP_TOK_1] [TIME_STAMP_TOK_2] [TOK_N+1] ..."""

        def __init__(
            self, timestamp_begin, vocab_size, batch_size, max_length, min_space=3, seed=0, is_length_ascending=True
        ):
            self.timestamp_begin = timestamp_begin
            self.vocab_size = vocab_size

            self.min_space_between_timestamps = min_space
            self.timestamp_tokens = torch.arange(self.timestamp_begin, self.vocab_size)
            self.timestamp_tokens.to(torch_device)
            self.is_length_ascending = is_length_ascending

            self.no_time_stamp_counter = batch_size * [0]
            self.prev_highest_timestamp = batch_size * [0]
            self.batch_size = batch_size
            self.max_length = max_length
            self.count = 0
            self.begin_index = 0

            self.let_pass = [[] for _ in range(batch_size)]
            for k in range(batch_size):
                random.seed(seed + k)
                for _ in range(10000):
                    self.let_pass[k].append(random.randint(1, 10) <= 3)

        def set_begin_index(self, begin_index: int):
            self.begin_index = begin_index

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
            # we don't want to randomely sample timestamp tokens
            if input_ids.shape[-1] != self.begin_index:
                scores[:, self.timestamp_begin :] = -float("inf")

            self.no_time_stamp_counter = [x + 1 for x in self.no_time_stamp_counter]
            for k in range(input_ids.shape[0]):
                # make sure to use correct index if a batch was removed
                if self.is_length_ascending and input_ids.shape[0] < self.batch_size:
                    prev_k = k + self.batch_size - input_ids.shape[0]
                else:
                    prev_k = k

                if input_ids[k, -1] == self.timestamp_begin:
                    self.no_time_stamp_counter[prev_k] = 0

                can_produce = self.no_time_stamp_counter[prev_k] > self.min_space_between_timestamps
                must_produce = (
                    input_ids[k][2:].le(self.timestamp_begin).all() and input_ids.shape[-1] == self.max_length - 1
                )
                # produce timestamp with 30%
                if (can_produce and self.let_pass[prev_k][self.count]) or must_produce:
                    self.no_time_stamp_counter[prev_k] = 0
                    self.prev_highest_timestamp[prev_k] = max(input_ids[k].max() + 1, self.timestamp_tokens[0].item())

                    # force a timestamp
                    scores[k, :] = -float("inf")
                    scores[k, self.prev_highest_timestamp[prev_k]] = 10.0

                if (
                    input_ids.shape[-1] > 3
                    and input_ids[k, -1].item() in self.timestamp_tokens
                    and input_ids[k, -2].item() not in self.timestamp_tokens
                ):
                    # force the same as before
                    scores[k, :] = -float("inf")
                    scores[k, input_ids[k, -1].item()] = 10.0

            self.count += 1

            if torch.isinf(scores).all():
                raise ValueError("Dummy logit processor is incorrectly set up. Scores should not be all inf.")

            return scores


if is_torchaudio_available():
    import torchaudio


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
        batch_size=3,  # need batch_size != num_hidden_layers
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
            "text-generation": WhisperForCausalLM,
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

    def test_requires_grad_encoder_embed_positions(self):
        config = self.model_tester.get_config()
        for model_class in self.all_model_classes:
            model = model_class(config)
            encoder = model.get_encoder()
            self.assertFalse(encoder.embed_positions.weight.requires_grad)

    def test_encoder_sinusoidal_embed_positions(self):
        config = self.model_tester.get_config()
        for model_class in self.all_model_classes:
            model = model_class(config)
            embeds = model.get_encoder().embed_positions.weight
            self.assertTrue(torch.allclose(embeds, sinusoids(*embeds.shape)))

    def test_decoder_model_past_with_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

    def test_encoder_decoder_model_standalone(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.check_encoder_decoder_model_standalone(*config_and_inputs)

    def _get_input_ids_and_config(self, batch_size=3):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        input_ids = inputs_dict[self.input_name]

        # cut to half length & take max batch_size=batch_size
        input_ids = input_ids[:batch_size, :, :]

        # generate max 3 tokens
        max_length = 4
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

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    def test_generate_with_head_masking(self):
        pass

    @require_torch_fp16
    def test_generate_fp16(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs()
        config.max_target_positions = 400
        input_features = input_dict["input_features"]
        model = WhisperForConditionalGeneration(config).eval().to(torch_device)
        input_features = input_features.half()
        model.half()
        model.generate(input_features)
        model.generate(input_features, num_beams=4, do_sample=True, early_stopping=False, num_return_sequences=3)

    def test_generate_language(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs()
        input_features = input_dict["input_features"]
        model = WhisperForConditionalGeneration(config).to(torch_device)
        # Hack to keep the test fast and not require downloading a model with a generation_config
        model.generation_config.__setattr__("lang_to_id", {"<|en|>": 1})
        model.generation_config.__setattr__("task_to_id", {"transcribe": 2})

        # test language code
        model.generate(input_features, language="en")
        # test tokenizer code
        model.generate(input_features, language="<|en|>")
        # test language name
        model.generate(input_features, language="English")

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

    @require_flash_attn
    @require_torch_gpu
    @pytest.mark.flash_attn_test
    @slow
    def test_flash_attn_2_inference_equivalence(self):
        import torch

        for model_class in self.all_model_classes:
            if not model_class._supports_flash_attn_2:
                return

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_fa = model_class.from_pretrained(
                    tmpdirname, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
                )
                model_fa.to(torch_device)

                model = model_class.from_pretrained(
                    tmpdirname,
                    torch_dtype=torch.bfloat16,
                )
                model.to(torch_device)

                dummy_input = inputs_dict[model.main_input_name][:1]
                if dummy_input.dtype in [torch.float32, torch.float16]:
                    dummy_input = dummy_input.to(torch.bfloat16)

                decoder_input_ids = inputs_dict.get("decoder_input_ids", dummy_input)[:1]

                outputs = model(dummy_input, decoder_input_ids=decoder_input_ids, output_hidden_states=True)
                outputs_fa = model_fa(dummy_input, decoder_input_ids=decoder_input_ids, output_hidden_states=True)

                logits = outputs.decoder_hidden_states[-1]
                logits_fa = outputs_fa.decoder_hidden_states[-1]

                # whisper FA2 needs very high tolerance
                assert torch.allclose(logits_fa, logits, atol=4e-1)

                # check with inference + dropout
                model.train()
                _ = model_fa(dummy_input, decoder_input_ids=decoder_input_ids)

    @require_flash_attn
    @require_torch_gpu
    @pytest.mark.flash_attn_test
    @slow
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        import torch

        for model_class in self.all_model_classes:
            if not model_class._supports_flash_attn_2:
                return

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_fa = model_class.from_pretrained(
                    tmpdirname, torch_dtype=torch.float16, attn_implementation="flash_attention_2"
                )
                model_fa.to(torch_device)

                model = model_class.from_pretrained(tmpdirname, torch_dtype=torch.float16)
                model.to(torch_device)

                dummy_input = inputs_dict[model.main_input_name][:1]
                dummy_input = dummy_input.to(torch.float16)

                decoder_input_ids = torch.tensor([[0, 1, 2, 3, 4, 5]], device=dummy_input.device, dtype=torch.long)
                decoder_attention_mask = torch.tensor(
                    [[0, 0, 0, 1, 1, 1]], device=dummy_input.device, dtype=torch.long
                )

                outputs = model(dummy_input, decoder_input_ids=decoder_input_ids, output_hidden_states=True)
                outputs_fa = model_fa(dummy_input, decoder_input_ids=decoder_input_ids, output_hidden_states=True)

                logits = outputs.decoder_hidden_states[-1]
                logits_fa = outputs_fa.decoder_hidden_states[-1]

                # whisper FA2 needs very high tolerance
                assert torch.allclose(logits_fa, logits, atol=4e-1)

                other_inputs = {
                    "decoder_input_ids": decoder_input_ids,
                    "decoder_attention_mask": decoder_attention_mask,
                    "output_hidden_states": True,
                }

                outputs = model(dummy_input, **other_inputs)
                outputs_fa = model_fa(dummy_input, **other_inputs)

                logits = outputs.decoder_hidden_states[-1]
                logits_fa = outputs_fa.decoder_hidden_states[-1]

                # whisper FA2 needs very high tolerance
                assert torch.allclose(logits_fa[:, -2:], logits[:, -2:], atol=4e-1)

    def _create_and_check_torchscript(self, config, inputs_dict):
        if not self.test_torchscript:
            return

        configs_no_init = _config_zero_init(config)  # To be sure we have no Nan
        configs_no_init.torchscript = True
        configs_no_init._attn_implementation = "eager"
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

            non_persistent_buffers = {}
            for key in loaded_model_state_dict.keys():
                if key not in model_state_dict.keys():
                    non_persistent_buffers[key] = loaded_model_state_dict[key]

            loaded_model_state_dict = {
                key: value for key, value in loaded_model_state_dict.items() if key not in non_persistent_buffers
            }

            self.assertEqual(set(model_state_dict.keys()), set(loaded_model_state_dict.keys()))

            model_buffers = list(model.buffers())
            for non_persistent_buffer in non_persistent_buffers.values():
                found_buffer = False
                for i, model_buffer in enumerate(model_buffers):
                    if torch.equal(non_persistent_buffer, model_buffer):
                        found_buffer = True
                        break

                self.assertTrue(found_buffer)
                model_buffers.pop(i)

            models_equal = True
            for layer_name, p1 in model_state_dict.items():
                p2 = loaded_model_state_dict[layer_name]
                if p1.data.ne(p2.data).sum() > 0:
                    models_equal = False

            self.assertTrue(models_equal)

    def check_pt_tf_outputs(self, tf_outputs, pt_outputs, model_class, tol=5e-5, name="outputs", attributes=None):
        # We override with a slightly higher tol value, as test recently became flaky
        super().check_pt_tf_outputs(tf_outputs, pt_outputs, model_class, tol, name, attributes)

    def check_pt_flax_outputs(self, fx_outputs, pt_outputs, model_class, tol=5e-5, name="outputs", attributes=None):
        # We override with a slightly higher tol value, as test recently became flaky
        super().check_pt_flax_outputs(fx_outputs, pt_outputs, model_class, tol, name, attributes)

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
                fx_inputs = {k: np.array(v.to("cpu")) for k, v in pt_inputs.items() if torch.is_tensor(v)}

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
                fx_inputs = {k: np.array(v.to("cpu")) for k, v in pt_inputs.items() if torch.is_tensor(v)}

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

    def test_generate_with_prompt_ids_and_task_and_language(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = WhisperForConditionalGeneration(config).eval().to(torch_device)
        input_features = input_dict["input_features"]
        prompt_ids = torch.arange(5).to(torch_device)
        language = "<|de|>"
        task = "translate"
        lang_id = 6
        task_id = 7
        model.generation_config.__setattr__("lang_to_id", {language: lang_id})
        model.generation_config.__setattr__("task_to_id", {task: task_id})

        output = model.generate(input_features, max_new_tokens=5, task=task, language=language, prompt_ids=prompt_ids)

        expected_output_start = [
            *prompt_ids.tolist(),
            model.generation_config.decoder_start_token_id,
            lang_id,
            task_id,
        ]
        for row in output.tolist():
            self.assertListEqual(row[: len(expected_output_start)], expected_output_start)

    def test_generate_with_prompt_ids_and_forced_decoder_ids(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = WhisperForConditionalGeneration(config).eval().to(torch_device)
        input_features = input_dict["input_features"]
        prompt_ids = torch.arange(5).to(torch_device)
        forced_decoder_ids = [(1, 6), (2, 7), (3, 8)]

        output = model.generate(
            input_features, max_new_tokens=5, forced_decoder_ids=forced_decoder_ids, prompt_ids=prompt_ids
        )

        expected_output_start = [
            *prompt_ids.tolist(),
            model.generation_config.decoder_start_token_id,
            *[token for _rank, token in forced_decoder_ids],
        ]
        for row in output.tolist():
            self.assertListEqual(row[: len(expected_output_start)], expected_output_start)

    def test_generate_with_prompt_ids_max_length(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.max_target_positions = 7

        model = WhisperForConditionalGeneration(config).eval().to(torch_device)
        input_features = input_dict["input_features"]
        decoder_input_ids = torch.arange(5).to(torch_device)
        prompt_ids = decoder_input_ids[:4]
        max_new_tokens = 8

        with self.assertRaisesRegex(
            ValueError,
            f"The length of `decoder_input_ids` equal `prompt_ids` plus special start tokens is {decoder_input_ids.shape[-1]}, and the `max_new_tokens` "
            f"is {max_new_tokens}. Thus, the combined length of "
            f"`decoder_input_ids` and `max_new_tokens` is: {max_new_tokens + decoder_input_ids.shape[-1]}. This exceeds the "
            f"`max_target_positions` of the Whisper model: {config.max_target_positions}. "
            "You should either reduce the length of your prompt, or reduce the value of `max_new_tokens`, "
            f"so that their combined length is less than {config.max_target_positions}.",
        ):
            model.generate(input_features, max_new_tokens=max_new_tokens, prompt_ids=prompt_ids)

        model.generate(input_features, max_new_tokens=1, prompt_ids=prompt_ids)

    def test_generate_longform_with_prompt_ids(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = WhisperForConditionalGeneration(config).eval().to(torch_device)

        prompt_ids = torch.arange(5).to(torch_device)
        model.generation_config.no_timestamps_token_id = 11
        model.generation_config.pad_token_id = 10

        # make sure prompt token ids [0-9] can't be generated
        model.generation_config.suppress_tokens = list(range(10))

        input_features = input_dict["input_features"]

        language = "<|de|>"
        lang_id = 6

        input_features = input_features.repeat(1, 1, 50)
        attention_mask = torch.ones_like(input_features, dtype=torch.long)[:, 0]

        for prompt_type in ["first-segment", "all-segments"]:
            for task_id, task in enumerate(["translate", "transcribe"]):
                task_id = 7 + task_id

                model.generation_config.__setattr__("lang_to_id", {language: lang_id})
                model.generation_config.__setattr__("task_to_id", {task: task_id})

                output = model.generate(
                    input_features,
                    attention_mask=attention_mask,
                    prompt_condition_type=prompt_type,
                    max_new_tokens=5,
                    task=task,
                    language=language,
                    prompt_ids=prompt_ids,
                    condition_on_prev_tokens=True,
                )
                for row in output.tolist():
                    # make sure no token below 10 is in generated output => this means for long-form prompt ids should NOT be returned
                    assert not any(i in row for i in model.generation_config.suppress_tokens)

    def _check_longform_generate_single_batch(self, condition_on_prev_tokens):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()

        model = WhisperForConditionalGeneration(config).eval().to(torch_device)
        input_features = input_dict["input_features"]

        # len = 250 with num_input_frames = 60
        long_input_features = torch.cat([input_features.repeat(1, 1, 4), input_features[:, :, :10]], dim=-1)

        # force bsz=1
        long_input_features = long_input_features[:1]
        vocab_size = model.config.vocab_size

        batch_size = 1
        num_timestamp_tokens = 20
        max_length = 16
        logits_processor = [
            DummyTimestampLogitProcessor(
                vocab_size - num_timestamp_tokens,
                vocab_size,
                batch_size=batch_size,
                max_length=max_length,
                min_space=4,
            )
        ]

        # each chunk should not be longer than 10
        model.generation_config.max_length = max_length

        # if input features are long can't set return_timestamps to False
        with self.assertRaises(ValueError):
            _ = model.generate(long_input_features, logits_processor=logits_processor, return_timestamps=False)

        # if input features are long need to set generation config
        with self.assertRaises(ValueError):
            _ = model.generate(long_input_features, logits_processor=logits_processor)

        timestamp_begin = vocab_size - num_timestamp_tokens
        model.generation_config.no_timestamps_token_id = timestamp_begin - 1
        model.generation_config.eos_token_id = None
        model.config.eos_token_id = None
        model.generation_config._detect_timestamp_from_logprob = False
        # make sure that we only have the same begin token
        model.generation_config.max_initial_timestamp_index = 0
        model.generation_config.prev_bos_token_id = timestamp_begin - 3

        gen_kwargs = {
            "logits_processor": logits_processor,
            "return_segments": True,
            "condition_on_prev_tokens": condition_on_prev_tokens,
        }

        if condition_on_prev_tokens:
            gen_kwargs["no_speech_threshold"] = 0.6
            gen_kwargs["temperature"] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
            gen_kwargs["compression_ratio_threshold"] = 2.4
            gen_kwargs["logprob_threshold"] = -1.0

        outputs = model.generate(long_input_features, **gen_kwargs)

        segments = outputs["segments"][0]

        for _, segment in enumerate(segments):
            assert segment["start"] <= segment["end"], "start has to be smaller equal end"
            assert any(
                s > timestamp_begin for s in segment["tokens"][1:]
            ), f"At least one segment token should be a timestamp token, but not first., {segment['tokens']}"
            assert (
                segment["tokens"].shape[-1] <= max_length
            ), "make sure that no segment is larger than max generation length"

    def test_longform_generate_single_batch(self):
        self._check_longform_generate_single_batch(condition_on_prev_tokens=False)

    def test_longform_generate_single_batch_cond_prev(self):
        self._check_longform_generate_single_batch(condition_on_prev_tokens=True)

    def _check_longform_generate_multi_batch(self, condition_on_prev_tokens):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()

        model = WhisperForConditionalGeneration(config).eval().to(torch_device)
        input_features = input_dict["input_features"].to(torch_device)
        input_features = input_features[:2]

        # len = 250 with num_input_frames = 60
        long_input_features = torch.cat([input_features.repeat(1, 1, 4), input_features[:, :, :10]], dim=-1)
        input_features_2 = long_input_features[1:]
        attention_mask = torch.ones(
            (2, long_input_features.shape[-1]), dtype=input_features.dtype, device=input_features.device
        )
        attention_mask[0, 200:] = 0

        # force bsz=1
        vocab_size = model.config.vocab_size

        batch_size = 1
        num_timestamp_tokens = 20
        max_new_tokens = 16
        timestamp_begin = vocab_size - num_timestamp_tokens
        model.generation_config.no_timestamps_token_id = timestamp_begin - 1
        model.generation_config.eos_token_id = None
        model.config.eos_token_id = None
        model.generation_config._detect_timestamp_from_logprob = False
        # make sure that we only have the same begin token
        model.generation_config.max_initial_timestamp_index = 0
        model.generation_config.max_new_tokens = max_new_tokens
        model.generation_config.prev_bos_token_id = timestamp_begin - 3

        logits_processor = [
            DummyTimestampLogitProcessor(
                vocab_size - num_timestamp_tokens,
                vocab_size,
                batch_size=batch_size,
                max_length=max_new_tokens,
                min_space=4,
                seed=1,
            )
        ]
        outputs_2 = model.generate(
            input_features_2,
            max_new_tokens=max_new_tokens,
            logits_processor=logits_processor,
            condition_on_prev_tokens=condition_on_prev_tokens,
            return_segments=True,
        )
        tokens_2 = outputs_2["sequences"][0]
        segments_2 = outputs_2["segments"][0]

        batch_size = 2
        logits_processor = [
            DummyTimestampLogitProcessor(
                vocab_size - num_timestamp_tokens,
                vocab_size,
                batch_size=batch_size,
                max_length=max_new_tokens,
                min_space=4,
                seed=0,
            )
        ]
        gen_kwargs = {
            "logits_processor": logits_processor,
            "return_segments": True,
            "condition_on_prev_tokens": condition_on_prev_tokens,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
        }

        outputs = model.generate(long_input_features, **gen_kwargs)
        tokens = outputs["sequences"][1]
        segments = outputs["segments"][1]

        # make sure batched and non-batched is the same
        assert tokens_2.tolist() == tokens[: tokens_2.shape[-1]].tolist()

        for seg1, seg2 in zip(segments_2, segments):
            assert seg1["start"] == seg2["start"]
            assert seg1["end"] == seg2["end"]
            assert seg1["tokens"].tolist() == seg2["tokens"].tolist()

    def test_longform_generate_multi_batch(self):
        self._check_longform_generate_multi_batch(condition_on_prev_tokens=False)

    def test_longform_generate_multi_batch_cond_prev(self):
        self._check_longform_generate_multi_batch(condition_on_prev_tokens=True)


@require_torch
@require_torchaudio
class WhisperModelIntegrationTests(unittest.TestCase):
    def setUp(self):
        self._unpatched_generation_mixin_generate = transformers.GenerationMixin.generate

    def tearDown(self):
        transformers.GenerationMixin.generate = self._unpatched_generation_mixin_generate

    @cached_property
    def default_processor(self):
        return WhisperProcessor.from_pretrained("openai/whisper-base")

    def _load_datasamples(self, num_samples):
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        # automatic decoding with librispeech
        speech_samples = ds.sort("id").select(range(num_samples))[:num_samples]["audio"]

        return [x["array"] for x in speech_samples]

    def _patch_generation_mixin_generate(self, check_args_fn=None):
        test = self

        def generate(self, *args, **kwargs):
            if check_args_fn is not None:
                check_args_fn(*args, **kwargs)
            return test._unpatched_generation_mixin_generate(self, *args, **kwargs)

        transformers.GenerationMixin.generate = generate

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

        token = os.getenv("HF_HUB_READ_TOKEN", True)
        ds = load_dataset("mozilla-foundation/common_voice_6_1", "ja", split="test", streaming=True, token=token)
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

        EXPECTED_OUTPUT = torch.tensor([50258, 50259, 50359, 50364, 2221, 13, 2326, 388, 391, 307, 264, 50244, 295, 264, 2808, 5359, 11, 293, 321, 366, 5404, 281, 2928, 702, 14943, 13, 50692, 50692, 6966, 307, 2221, 13, 2326, 388, 391, 311, 9060, 1570, 1880, 813, 702, 1871, 13, 50926, 50926, 634, 5112, 505, 300, 412, 341, 42729, 3196, 295, 264, 1064, 11, 365, 5272, 293, 12904, 9256, 450, 10539, 51208, 51208, 949, 505, 11, 14138, 10117, 490, 3936, 293, 1080, 3542, 5160, 881, 26336, 281, 264, 1575, 13, 51552, 51552, 634, 575, 12525, 22618, 1968, 6144, 35617, 7354, 1292, 6, 589, 307, 534, 10281, 934, 439, 11, 293, 51836, 51836, 50257])  # fmt: skip

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
    def test_tiny_token_timestamp_generation(self):
        set_seed(0)
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
        model.to(torch_device)
        model.generation_config.alignment_heads = [[2, 2], [3, 0], [3, 2], [3, 3], [3, 4], [3, 5]]

        input_speech = self._load_datasamples(4)
        input_features = processor.feature_extractor(raw_speech=input_speech, return_tensors="pt").input_features.to(
            torch_device
        )

        generate_outputs = model.generate(
            input_features, max_length=448, return_timestamps=True, return_token_timestamps=True
        )

        self.assertEqual(generate_outputs.sequences.shape, generate_outputs.token_timestamps.shape)

        # fmt: off
        EXPECTED_OUTPUT = torch.tensor([
            [ 0.0000, 0.0000, 0.0000, 0.0000, 0.4800, 0.8200, 0.9600, 1.1200, 1.1200, 1.2200, 1.5000, 1.7200, 2.0000, 2.3400, 2.5000, 2.6600, 3.1800, 3.5600, 3.6800, 3.8000, 4.1000, 4.3000, 4.5800, 4.9400, 5.3800, 12.4200, 12.8400, 26.9200, 26.9200, 26.9200, 26.9200, 26.9200, 26.9200, 26.9200, 26.9200, 26.9200, 26.9200, 26.9200, 26.9400, 26.9400, 26.9400, 26.9400, 29.8400 ],
            [ 0.0000, 0.0000, 0.0000, 0.0000, 0.5200, 0.9000, 1.1400, 1.4200, 1.5200, 1.6800, 1.6800, 1.8800, 2.1000, 2.2200, 2.6200, 3.1400, 3.5800, 3.9600, 4.4000, 17.3000, 17.3000, 26.7200, 26.7200, 26.7200, 26.7200, 26.7200, 26.7200, 26.7200, 26.7200, 26.7200, 26.7200, 26.7200, 26.7200, 26.7200, 26.7200, 26.7200, 26.7400, 26.7400, 26.7400, 26.7400, 26.7400, 26.7400, 28.0000 ],
            [ 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.7600, 1.0000, 1.4200, 1.8000, 1.9400, 2.1800, 2.5200, 3.0200, 3.3200, 3.5400, 3.9400, 4.5600, 4.9200, 5.2800, 5.5600, 5.9000, 6.1600, 6.3000, 6.4800, 6.4800, 6.6400, 7.8200, 7.9600, 8.2200, 8.6000, 8.9200, 9.2200, 9.5200, 9.7200, 10.0600, 10.5400, 10.8800, 11.2600, 11.5400, 11.7400, 12.0800, 15.6800, 15.6800],
            [ 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.7400, 1.0400, 1.3200, 1.6800, 2.1400, 2.4800, 2.7800, 3.0800, 3.1600, 3.4000, 3.6000, 4.0200, 4.2200, 4.8600, 5.2400, 5.7400, 6.3400, 6.6200, 6.7600, 6.7600, 6.8600, 7.2400, 7.4200, 7.6800, 7.9200, 8.4800, 8.7600, 9.2000, 9.2000, 9.4200, 15.8200, 15.8200, 29.6400, 29.6600, 29.6600, 29.6600, 29.6600, 29.7600]
        ])
        # fmt: on

        self.assertTrue(torch.allclose(generate_outputs.token_timestamps.to("cpu"), EXPECTED_OUTPUT))

    @slow
    def test_tiny_token_timestamp_batch_generation(self):
        set_seed(0)
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
        model.to(torch_device)
        model.generation_config.alignment_heads = [[2, 2], [3, 0], [3, 2], [3, 3], [3, 4], [3, 5]]

        num_samples = 4
        num_return_sequences = 2

        input_speech = self._load_datasamples(num_samples)
        input_features = processor.feature_extractor(raw_speech=input_speech, return_tensors="pt").input_features.to(
            torch_device
        )

        generate_outputs = model.generate(
            input_features,
            max_length=448,
            return_timestamps=True,
            return_token_timestamps=True,
            num_beams=3,
            num_return_sequences=num_return_sequences,
        )

        # task id and lang id prompts should not have timestamp tokens
        self.assertEqual(generate_outputs.sequences.shape[-1] - 2, generate_outputs.token_timestamps.shape[-1])

        self.assertEqual(len(generate_outputs.sequences), num_return_sequences * num_samples)

    @slow
    def test_tiny_token_timestamp_generation_longform(self):
        set_seed(0)
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
        model.to(torch_device)
        model.generation_config.alignment_heads = [[2, 2], [3, 0], [3, 2], [3, 3], [3, 4], [3, 5]]

        input_speech = self._load_datasamples(5)
        long_input_speech = np.concatenate(input_speech, dtype=np.float32)
        inputs = processor.feature_extractor(
            raw_speech=long_input_speech,
            return_tensors="pt",
            truncation=False,  # False so the audio isn't truncated and whole audio is sent to the model
            return_attention_mask=True,
            padding=True,
        )

        inputs = inputs.to(torch_device)
        generate_outputs = model.generate(**inputs, return_segments=True, return_token_timestamps=True)

        token_timestamps_shape = [
            [segment["token_timestamps"].shape for segment in segment_list]
            for segment_list in generate_outputs["segments"]
        ]
        tokens_shape = [
            [segment["tokens"].shape for segment in segment_list] for segment_list in generate_outputs["segments"]
        ]
        self.assertListEqual(tokens_shape, token_timestamps_shape)

        # fmt: off
        EXPECTED_OUTPUT = [
            torch.tensor([0.0000, 0.4200, 0.8200, 0.9400, 1.1200, 1.1200, 1.2200, 1.5000, 1.7200, 2.0400, 2.3400, 2.5200, 2.6600, 3.2000, 3.4400, 3.5600, 3.6800, 3.8200, 4.1000, 4.3000, 4.5800, 4.9400, 5.4000, 6.3600]),
            torch.tensor([ 6.5400,  6.5400,  6.7400,  6.9600,  7.2600,  7.3400,  7.5800,  7.5800, 7.6400,  7.8400,  8.1000,  8.5000,  9.0000,  9.4800,  9.7200, 10.2600, 11.1000]),
            torch.tensor([11.2200, 11.2200, 11.4200, 11.6600, 12.0800, 12.4400, 12.5800, 12.8400, 13.1800, 13.6800, 14.0000, 14.2200, 14.6200, 14.9800, 15.2200, 15.6000, 15.9400, 16.2000, 16.5600, 16.8400, 16.9800]),
            torch.tensor([16.9800, 16.9800, 17.3200, 18.1600, 18.6400, 18.8600, 19.2800, 19.5600, 19.8800, 20.1800, 20.3800, 20.7200, 21.1600, 21.5400, 21.9000, 22.2000, 22.4200, 22.8600, 23.7000]),
            torch.tensor([23.7000, 23.7000, 23.9400, 24.1800, 24.3800, 24.8400, 25.2800, 25.6600, 25.9200, 26.2600, 26.4000, 26.5800, 26.7600, 27.1400, 27.3800, 28.0400, 28.3800, 28.8200, 29.3400, 29.5200]),
            torch.tensor([29.4400, 29.4400, 29.7000, 30.0800, 30.3800, 30.5400, 30.8200, 31.0600, 31.6600, 31.9200, 32.3000, 32.4800, 32.6200, 33.6800]),
            torch.tensor([33.8000, 33.8000, 33.9800, 33.9800, 34.1800, 34.4400, 34.6200, 35.0000, 35.2200, 35.3200, 35.5600, 35.9200, 36.3800, 36.6200, 36.6600, 36.9600, 37.3400, 37.9800, 38.5800, 38.7200, 38.9800, 39.4400, 39.5800, 39.8000, 40.1200, 40.2600]),
            torch.tensor([40.5200, 40.5200, 40.6200, 41.1000, 41.5400, 41.9200, 42.1000, 42.3200, 42.3200, 43.0600, 44.6000]),
            torch.tensor([44.7000, 44.7000, 44.8600, 44.9400, 45.1400, 45.1400, 45.2800, 45.6200, 45.9000, 46.2600, 47.1600, 47.4800, 47.7400, 48.1000, 48.2800, 48.4000, 48.6200, 48.8400, 49.0400, 49.2800, 49.4800, 49.6600, 49.9400, 50.5400]),
            torch.tensor([50.5400, 50.5400, 50.6600, 50.8800, 51.2400, 51.7200, 52.8400]),
            torch.tensor([52.9600, 52.9600, 53.0400, 53.2600, 53.4200, 53.5800, 53.9200, 54.1200, 54.7200, 54.9400, 55.2600, 55.6200, 55.9800, 56.5600, 56.8000, 56.9200, 57.3600, 57.9200, 58.1800, 58.5000, 58.6400, 58.8200]),
            torch.tensor([58.6800, 58.6800, 59.1400, 59.5400, 59.9200, 60.1600, 60.3800, 60.8200, 61.6200, 62.2600, 75.2000]),
        ]
        # fmt: on

        for segment, exp_segment in zip(generate_outputs["segments"][0], EXPECTED_OUTPUT):
            self.assertTrue(torch.allclose(segment["token_timestamps"], exp_segment))

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

    @slow
    def test_generate_with_prompt_ids(self):
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
        model.to(torch_device)
        input_speech = self._load_datasamples(4)[-1:]
        input_features = processor(input_speech, return_tensors="pt").input_features.to(torch_device)

        output_without_prompt = model.generate(input_features)
        prompt_ids = processor.get_prompt_ids("Leighton", return_tensors="pt").to(torch_device)
        output_with_prompt = model.generate(input_features, prompt_ids=prompt_ids)

        expected_without_prompt = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|> He has grave doubts whether Sir Frederick Layton's work is really Greek after all and can discover in it but little of Rocky Ithaca.<|endoftext|>"
        expected_with_prompt = "<|startofprev|> Leighton<|startoftranscript|><|en|><|transcribe|><|notimestamps|> He has grave doubts whether Sir Frederick Leighton's work is really Greek after all and can discover in it but little of Rocky Ithaca.<|endoftext|>"

        output_without_prompt = processor.decode(output_without_prompt[0])
        output_with_prompt = processor.decode(output_with_prompt[0])

        self.assertEqual(output_without_prompt, expected_without_prompt)
        self.assertEqual(output_with_prompt, expected_with_prompt)

    @slow
    def test_language_detection(self):
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
        model.to(torch_device)
        input_speech = self._load_datasamples(4)[-1:]
        input_features = processor(input_speech, return_tensors="pt").input_features.to(torch_device)

        lang_id = model.detect_language(input_features)[0].item()

        ids_to_lang = {v: k for k, v in model.generation_config.lang_to_id.items()}

        assert ids_to_lang[lang_id] == "<|en|>"

        audio = hf_hub_download("Narsil/asr_dummy", filename="hindi.ogg", repo_type="dataset")

        raw_audio, sr = torchaudio.load(audio)
        input_speech = torchaudio.transforms.Resample(sr, 16_000)(raw_audio).numpy()

        input_features = processor(input_speech, return_tensors="pt").input_features.to(torch_device)

        lang_id = model.detect_language(input_features)[0].item()

        assert ids_to_lang[lang_id] == "<|hi|>"

    @slow
    def test_default_multilingual_transcription_short_form(self):
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
        model.to(torch_device)

        audio = hf_hub_download("Narsil/asr_dummy", filename="hindi.ogg", repo_type="dataset")

        raw_audio, sr = torchaudio.load(audio)
        input_speech = torchaudio.transforms.Resample(sr, 16_000)(raw_audio).numpy()

        input_features = processor(input_speech, return_tensors="pt").input_features.to(torch_device)

        # model.generation_config.forced_decoder_ids defaults to [1, null] for lang_token
        sequences = model.generate(input_features)

        transcription = processor.batch_decode(sequences, skip_special_tokens=False)[0]

        assert (
            transcription
            == "<|startoftranscript|><|hi|><|transcribe|><|notimestamps|> Mirchi mein ki tene vibinda prajatiya hai<|endoftext|>"
        )

        # set forced_decoder_ids to English
        model.generation_config.forced_decoder_ids[0][-1] = 50259

        sequences = model.generate(input_features)
        transcription = processor.batch_decode(sequences, skip_special_tokens=False)[0]

        assert (
            transcription
            == "<|startoftranscript|><|en|><|transcribe|><|notimestamps|> MIRCHI MET, which is the name of the Bible.<|endoftext|>"
        )

    @slow
    def test_default_multilingual_transcription_long_form(self):
        processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
        model.to(torch_device)

        audio = hf_hub_download("Narsil/asr_dummy", filename="hindi.ogg", repo_type="dataset")

        raw_audio, sr = torchaudio.load(audio)
        input_speech = torchaudio.transforms.Resample(sr, 16_000)(raw_audio)

        input_speech = input_speech.repeat(1, 10).numpy()
        input_features = processor(
            input_speech, return_tensors="pt", padding="longest", truncation=False
        ).input_features.to(torch_device)

        # model.generation_config.forced_decoder_ids defaults to [1, null] for lang_token
        sequences = model.generate(input_features)

        transcription = processor.batch_decode(sequences)[0]

        assert transcription == " मिर्ची में कितने विबिन्द प्रजातियां हैं? मिर्ची में कितने विबिन्द प्रजातियां हैं?"

        # set forced_decoder_ids to English
        model.generation_config.forced_decoder_ids[0][-1] = 50259

        sequences = model.generate(input_features)
        transcription = processor.batch_decode(sequences)[0]

        assert (
            transcription
            == " How many different species are there in the chilli? How many different species are there in the chili?"
        )

    @slow
    def test_generate_with_prompt_ids_and_forced_decoder_ids(self):
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
        model.to(torch_device)
        input_speech = self._load_datasamples(1)
        input_features = processor(input_speech, return_tensors="pt").input_features.to(torch_device)
        task = "translate"
        language = "de"
        expected_tokens = [f"<|{task}|>", f"<|{language}|>"]
        prompt = "test prompt"
        prompt_ids = processor.get_prompt_ids(prompt, return_tensors="pt").to(torch_device)

        output = model.generate(input_features, task=task, language=language, prompt_ids=prompt_ids)
        text = processor.decode(output[0])

        self.assertTrue(prompt in text)
        self.assertTrue(all(token in text for token in expected_tokens))

    @slow
    def test_generate_with_prompt_ids_and_no_non_prompt_forced_decoder_ids(self):
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
        model.to(torch_device)
        input_speech = self._load_datasamples(1)
        input_features = processor(input_speech, return_tensors="pt").input_features.to(torch_device)
        prompt = "test prompt"
        prompt_ids = processor.get_prompt_ids(prompt, return_tensors="pt").to(torch_device)

        model.generation_config.forced_decoder_ids = None
        model.config.forced_decoder_ids = None

        output = model.generate(input_features, prompt_ids=prompt_ids, return_timestamps=True)
        text = processor.decode(output[0])

        self.assertTrue(prompt in text)

    @slow
    @require_torch_gpu
    def test_speculative_decoding_distil(self):
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model_id = "openai/whisper-large-v2"
        model = WhisperForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(torch_device)

        processor = WhisperProcessor.from_pretrained(model_id)

        assistant_model_id = "distil-whisper/distil-large-v2"
        assistant_model = WhisperForCausalLM.from_pretrained(
            assistant_model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        assistant_model.to(torch_device)

        dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        sample = dataset[0]["audio"]

        input_features = (
            processor(sample["array"], return_tensors="pt").input_features.to(torch_device).to(torch.float16)
        )

        # warm up assisted decoding
        _ = model.generate(input_features, assistant_model=assistant_model)
        # warm up non-assisted decoding
        _ = model.generate(input_features)

        # assisted decoding
        start_time = time.time()
        tokens = model.generate(input_features, assistant_model=assistant_model)
        total_time_assist = time.time() - start_time

        transcription_ass = processor.batch_decode(tokens, skip_special_tokens=True)

        # non-assisted decoding
        start_time = time.time()
        tokens = model.generate(input_features)
        total_time_non_assist = time.time() - start_time

        transcription_non_ass = processor.batch_decode(tokens, skip_special_tokens=True)

        assert transcription_ass == transcription_non_ass
        assert transcription_ass == [
            " Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel."
        ]
        assert total_time_non_assist > total_time_assist, "Make sure that assistant decoding is faster"

    @slow
    @require_torch_gpu
    def test_speculative_decoding_non_distil(self):
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model_id = "openai/whisper-large-v2"
        model = WhisperForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(torch_device)

        processor = WhisperProcessor.from_pretrained(model_id)

        assistant_model_id = "openai/whisper-tiny"
        assistant_model = WhisperForConditionalGeneration.from_pretrained(
            assistant_model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        assistant_model.to(torch_device)

        dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        sample = dataset[0]["audio"]

        input_features = (
            processor(sample["array"], return_tensors="pt").input_features.to(torch_device).to(torch.float16)
        )

        # warm up assisted decoding
        _ = model.generate(input_features, assistant_model=assistant_model)
        # warm up non-assisted decoding
        _ = model.generate(input_features)

        # assisted decoding
        start_time = time.time()
        tokens = model.generate(input_features, assistant_model=assistant_model)
        total_time_assist = time.time() - start_time

        transcription_ass = processor.batch_decode(tokens, skip_special_tokens=True)

        # non-assisted decoding
        start_time = time.time()
        tokens = model.generate(input_features)
        total_time_non_assist = time.time() - start_time

        transcription_non_ass = processor.batch_decode(tokens, skip_special_tokens=True)

        assert transcription_ass == transcription_non_ass
        assert transcription_ass == [
            " Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel."
        ]
        assert total_time_non_assist > total_time_assist, "Make sure that assistant decoding is faster"

    @slow
    def test_whisper_longform_single_batch(self):
        # fmt: off
        EXPECTED_TEXT = [' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel. Nor is Mr. Quilter\'s manner less interesting than his matter. He tells us that at this festive season of the year, with Christmas and roast beef looming before us, similes drawn from eating and its results occur most readily to the mind. He has grave doubts whether Sir Frederick Layton\'s work is really Greek after all, and can discover in it but little of rocky Ithaca. Linnell\'s pictures are a sort of up-gards and atom paintings, and Mason\'s exquisite idles are as national as a jingo poem. Mr. Birk at Foster\'s landscapes smile at one much in the same way that Mr. Carker used to flash his teeth. Mr. John Collier gives his sitter a cheerful slap in the back, before he says, like a shampoo or a Turkish bath. Next man, it is obviously unnecessary for us to point out how luminous these criticisms are, how delicate an expression. On the general principles of art, Mr. Quilter writes with equal lucidity. he tells us is of a different quality to mathematics, and finish in art is adding more effect. As for etchings, there are two kinds, British and foreign. He laments most bitterly the divorce that has been made between decorative art and what we usually call pictures. Makes the customary appeal to the last judgment and reminds us that in the great days of art Michelangelo was the furnishing upholsterer. Near the fire, any ornaments Fred brought home from India on the mantelboard. In fact, he is quite severe on Mr. Ruskin for not recognizing that a picture should denote the frailty of man. And remarks was pleasing courtesy in Felicitis Grace that many faces are feeling. Only, unfortunately, his own work never does get good. Mr. Quilter has missed his chance, for he has failed even to make himself the Tupper of painting. By Harry Quilter M.A. A man said to the universe, Sir, I exist. Sweat-covered Breon\'s body trickling into the tight-lowing cloth that was the only german he wore. The cut on his chest still dripping blood. The ache of his overstrained eyes, even the soaring arena around him with thousands of spectators, retrovealities not worth thinking about. His instant panic was followed by a small sharp blow high on his chest. One minute, a voice said, and a time buzzer sounded. A minute is not a very large measure of time, and his body needed every fraction of it. The buzzers were triggered his muscles into complete relaxation. Oli\'s heart and lungs worked on at a strong, measured rate. He was in reverie, sliding along the borders of consciousness. The contestants in the twenties needed undisturbed rest. Therefore, nights in the dormitories were as quiet as death. Particularly so, on this last night, when only two of the little cubicles were occupied, The thousands of others standing with dark empty doors. The other voice snapped with a harsh urgency, clearly used to command. I\'m here because the matter is of utmost importance, and brand is the one I must see. Now stand aside. The twenties, he must have drawn his gun because the intruder said quickly, but that away you\'re being a fool. out, through his silence then, and still wondering, Breon was once more asleep. Ten seconds, he asked the handler who was needing his aching muscles. A red-haired mountain of a man, with an apparently inexhaustible store of energy. There could be little art in this last and final round of fencing. Just thrust and parry, and victory to the stronger. man who entered the twenties had his own training tricks. They were appeared to be an immediate association with the death trauma, as if the two were inextricably linked into one. The strength that enables someone in a trance to hold his body stiff and unsupported except at two points, the head and heels. This is physically impossible when conscious. had died before during the 20s and death during the last round was in some ways easier than defeat. Breathing deeply, Breon\'s softly spoke the auto-hypnotic phrases that triggered the process. When the buzzer sounded, he pulled his foil from his second startled grasp and ran forward. Our role looked amazed at the sudden fury of the attack, then smiled. He thought it was the last burst of energy. He knew how close they both were to exhaustion. Breon saw something close to panic on his opponent\'s face when the man finally recognized his error. A wave of despair rolled out from our rogue. Breon sensed it and knew the fifth point was his. Then the powerful twist that\'s rested aside, in and under the guard, because he was sleeping instead of conquering, the lovely rose princess has become a fiddle without a bow, while poor Shaggy sits there, accooing dove. He has gone, and gone for good," answered Polychrom, who had managed to squeeze into the room beside the dragon, and had witnessed the occurrences with much interest. I have remained a prisoner only because I wished to be one. And with says he stepped forward and burst the stout chains as easily as if they had been threads. The little girl had been asleep, but she heard the wraps and opened the door. The king has flooded disgrace, and your friends are asking for you. I begged Ruggadot long ago to send him away, but he would not do so. I also offered to help your brother to escape, but he would not go. He eats and sleeps very steadily, replied the new king. I hope he doesn\'t work too hard, said Shaggy. He doesn\'t work at all. In fact, there\'s nothing he can do in these dominions as well as our gnomes, whose numbers are so great that it worries us to keep them all busy. Not exactly, we\'ve turned Calico. Where is my brother now, inquired Shaggy. In the metal forest. Where is that? The middle forest is in the great domed cavern, the largest and all-ard dominions, replied Calico. Calico hesitated. However, if we look sharp, we may be able to discover one of these secret ways. Oh no, I\'m quite sure he didn\'t. That\'s funny, remarked Betsy thoughtfully. I don\'t believe Anne knew any magic, or she\'d have worked it before. I do not know, confess Shaggy. True, agreed Calico. Calico went to the big gong and pounded on it just as Virgato used to do, but no one answered the summons. Having returned to the Royal Cavern, Calico first pounded the gong and then sat in the throne, wearing Virgato\'s discarded ruby crown and holding in his hand to scepter which reggative head so often thrown at his head.']
        # fmt: on

        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
        model = model.to(torch_device)

        ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean")
        one_audio = np.concatenate([x["array"] for x in ds["validation"]["audio"]], dtype=np.float32)

        input_features = processor(one_audio, return_tensors="pt", truncation=False, padding="longest")[
            "input_features"
        ]
        input_features = input_features.to(device=torch_device)

        result = model.generate(input_features, return_timestamps=True)
        decoded = processor.batch_decode(result, skip_special_tokens=True)

        assert decoded == EXPECTED_TEXT

        decoded_with_timestamps = processor.batch_decode(result, skip_special_tokens=True, decode_with_timestamps=True)

        no_timestamp_matches = re.split(r"<\|[\d\.]+\|>", decoded_with_timestamps[0])

        assert ["".join(no_timestamp_matches)] == EXPECTED_TEXT

        timestamp_matches = re.findall(r"<\|[\d\.]+\|>", decoded_with_timestamps[0])

        timestamp_floats = [float(t[2:-2]) for t in timestamp_matches]

        is_increasing = all(timestamp_floats[i] <= timestamp_floats[i + 1] for i in range(len(timestamp_floats) - 1))

        assert is_increasing

    @slow
    def test_whisper_longform_prompt_ids(self):
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
        model = model.to(torch_device)

        prompt = "Mr. Kilter, Ruggedo."  # let's force Mr. Quilter -> Mr. Kilter
        prompt_ids = processor.get_prompt_ids(prompt, return_tensors="pt").to(torch_device)

        ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean")
        one_audio = np.concatenate([x["array"] for x in ds["validation"]["audio"]], dtype=np.float32)

        first_text = ds["validation"][0]["text"].lower()
        last_text = ds["validation"][-1]["text"].lower()

        input_features = processor(one_audio, return_tensors="pt", truncation=False, padding="longest")[
            "input_features"
        ]
        input_features = input_features.to(device=torch_device)

        result = model.generate(
            input_features,
            prompt_ids=prompt_ids,
            return_timestamps=True,
            prompt_condition_type="first-segment",
            condition_on_prev_tokens=True,
        )
        decoded_first_segment = processor.batch_decode(result, skip_special_tokens=True)

        result = model.generate(
            input_features,
            prompt_ids=prompt_ids,
            return_timestamps=True,
            prompt_condition_type="all-segments",
            condition_on_prev_tokens=True,
        )
        decoded_all_segments = processor.batch_decode(result, skip_special_tokens=True)

        # show that first segment has quilter and last segment has ruggedo
        assert "quilter" in first_text
        assert "ruggedo" in last_text

        # condition on first segment correctly changes to kilter in first segment, but does not transcribe "ruggedo" correctly
        assert "kilter" in decoded_first_segment[0][: len(first_text)].lower()
        assert "ruggedo" not in decoded_first_segment[0][-len(last_text) :].lower()

        # condition on all-segment correctly changes to kilter in first segment and correctly transcribes "ruggedo"
        assert "kilter" in decoded_all_segments[0][: len(first_text)].lower()
        assert "ruggedo" in decoded_all_segments[0][-len(last_text) :].lower()

    @slow
    def test_whisper_longform_single_batch_prev_cond(self):
        # fmt: off
        EXPECTED_TEXT = [""" Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel. Nor is Mr. Quilter's manner less interesting than his matter. He tells us that at this festive season of the year, with Christmas and roast beef looming before us, similes drawn from eating and its results occur most readily to the mind. He has grieved doubts whether Sir Frederick Layton's work is really Greek after all, and can discover in it but little of rocky Ithaca. Linnell's pictures are a sort of up-gards and atom paintings, and Mason's exquisite itals are as national as a jingo poem. Mr. Birk at Foster's landscapes smile at one much in the same way that Mr. Carker used to flash his teeth. When Mr. John Collier gives his sitter a cheerful slap in the back, before he says like a shampooer and a Turkish bath, next man it is obviously unnecessary for us to point out how luminous these criticisms are, how delicate an expression. On the general principles of art, Mr. Quilter writes with equal lucidity. He tells us is of a different quality to mathematics, and finish in art is adding more effect. As for etchings, there are two kinds, British and foreign. He laments most bitterly the divorce that has been made between decorative art and what we usually call pictures. Makes a customary appeal to the last judgment and reminds us that in the great days of art Michelangelo was the furnishing upholsterer. Near the fire, any ornaments Fred brought home from India on the mental board. In fact, he is quite severe on Mr. Ruskin for not recognizing that a picture should denote the frailty of man, and remarks was pleasing courtesy in felicitous grace that many faces are feeling. Unfortunately his own work never does get good. Mr. Quilter has missed his chance, for he has failed even to make himself the tupper of painting. By Harry Quilter M. A. A man said to the universe, Sir, I exist. Sweat covered Breon's body trickling into the tight-lowing cloth that was the only german he wore. The cut on his chest still dripping blood. The ache of his overstrained eyes, even the soaring arena around him with thousands of spectators, retroveilities not worth thinking about. His instant panic was followed by a small sharp blow high on his chest. One minute, a voice said, and a time buzzer sounded. A minute is not a very large measure of time, and his body needed every fraction of it. The buzzers were triggered as muscles into complete relaxation. Only his heart and lungs worked on at a strong measured rate. He was in reverie, sliding along the borders of consciousness. The contestants in the twenties needed undisturbed rest. Therefore, nights in the dormitories were as quiet as death. Particularly so, on this last night, when only two of the little cubicles were occupied, the thousands of others standing with dark empty doors. The other voice snapped with a harsh urgency, clearly used to command. I'm here because the matter is of utmost importance, and brand is the one I must see. Now stand aside. The twenties, he must have drawn his gun because the intruder said quickly, but that away you're being a fool. But there was silence then, and still wondering, Breon was once more asleep. Ten seconds, he asked the handler who was needing his aching muscles. A red-haired mountain of a man with an apparently inexhaustible store of energy. There could be little art in this last and final round of fencing. Just thrust and parry and victory to the stronger. Your man who entered the twenties had his own training tricks. They were appeared to be an immediate association with the death trauma, as if the two were inextricably linked into one. The strength that enables someone in a trance to hold his body stiff and unsupported except at two points, the head and heels. This is physically impossible when conscious. Breon's death was in some ways easier than defeat. Breon's softly spoke the auto-hypnotic phrases that triggered the process. When the buzzer sounded, he pulled his foil from his second startled grasp and ran forward. Our role looked amazed at the sudden fury of the attack, then smiled. He thought it was the last burst of energy. He knew how close they both were to exhaustion. Breon saw something close to panic on his opponent's face when the man finally recognized his error. A wave of despair rolled out from our rogue. Breon sensed it and knew the fifth point was his. Then the powerful twist that's rested aside, in and under the guard, because he was sleeping instead of conquering, the lovely rose princess has become a fiddle without a bow, while poor Shaggy sits there, accooing dove. He has gone and gone for good, answered Polychrome, who had managed to squeeze into the room beside the dragon, and had witnessed the occurrences with much interest. I have remained a prisoner only because I wished to be one. And with this, he stepped forward and burst the stout chains as easily as if they had been threads. The little girl had been asleep, but she heard the wraps and opened the door. The king has flooded disgrace, and your friends are asking for you. I begged Ruggido long ago to send him away, but he would not do so. I also offered to help your brother to escape, but he would not go. He eats and sleeps very steadily, replied the new king. I hope he doesn't work too hard, since Shaggy. He doesn't work at all. In fact, there's nothing he can do in these dominions, as well as our gnomes, whose numbers are so great that it worries us to keep them all busy. Not exactly, we've turned Calico. Where is my brother now, inquired Shaggy. In the metal forest. Where is that? The metal forest is in the great domed cavern, the largest and all-ard dominions, replied Calico. Calico hesitated. However, if we look sharp, we may be able to discover one of these secret ways. Oh no, I'm quite sure he didn't. That's funny, remarked Betsy thoughtfully. I don't believe Anne knew any magic, or she'd have worked it before. I do not know, confessed Shaggy. True, agreed Calico. Calico went to the big gong and pounded on it, just as we're good to be used to do, but no one answered the summons. Having returned to the royal cavern, Calico first pounded the gong and then sat in the throne, wearing Regidos discarded Ruby crown, and holding in his hand to scepter which Regidos had so often thrown at his head."""]
        # fmt: on

        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
        model = model.to(torch_device)

        ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean")
        one_audio = np.concatenate([x["array"] for x in ds["validation"]["audio"]], dtype=np.float32)

        input_features = processor(one_audio, return_tensors="pt", truncation=False, padding="longest")[
            "input_features"
        ]
        input_features = input_features.to(device=torch_device)

        gen_kwargs = {
            "return_timestamps": True,
            "no_speech_threshold": 0.6,
            "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            "compression_ratio_threshold": 1.35,
            "condition_on_prev_tokens": True,
            "logprob_threshold": -1.0,
        }

        torch.manual_seed(0)
        result = model.generate(input_features, **gen_kwargs)
        decoded = processor.batch_decode(result, skip_special_tokens=True)

        assert decoded == EXPECTED_TEXT

    @slow
    def test_whisper_longform_multi_batch_beam(self):
        # fmt: off
        EXPECTED_TEXT = [' A man said to the universe, Sir, I exist. Sweat-covered Brienne\'s body trickling into the titling cloth that was the only german he wore. The cut on his chest was still dripping blood. The ache of his overstrained eyes, even the soaring arena around him with thousands of spectators, rich trivialities not worth thinking about. His instant panic was followed by a small sharp blow high on his chest. One minute, a voice said, and a time buzzer sounded. A minute is not a very large measure of time, and his body needed every fraction of it. The buzzers were, triggered his muscles into complete relaxation. Oily his heart and lungs worked on at a strong, measured rate. He was in reverie, sliding along the borders of consciousness. The contestants in the 20s needed undisturbed rest. Therefore, nights in the dormitories were as quiet as death. Particularly so, on this last night, when only two of the little cubicles were occupied. The thousands of others standing with dark empty doors. The other voice snapped with a harsh urgency, clearly used to command. I\'m here because the matter is of utmost importance, and brand is the one I must see. Now stand aside. The 20s, he must have drawn his gun because the intruder said quickly, but that away, you\'re being a fool. Out, there was silence then, and still wondering, Breon was once more asleep. Ten seconds, he asked the handler who was needing his aching muscles. A red-haired mountain of a man, with an apparently inexhaustible store of energy. There could be little art in this last and final round of fencing. Just thrust and parry, and victory to the stronger. Every man who entered the 20s had his own training tricks. There appeared to be an immediate association with the death trauma, as if the two were andextricably linked into one. The strength that enables someone in a trance to hold his body stiff and unsupported, except at two points, the head and heels. This is physically impossible when conscious. Others had died before during the 20s, and death during the last round was, in some ways, easier than defeat. Breeding deeply, Breon\'s softly spoke the auto-hypnotic phrases that triggered the process. When the buzzer sounded, he pulled his foil from his second startled grasp and ran forward. Our role looked amazed at the sudden fury of the attack, then smiled. He thought it was the last burst of energy. He knew how close they both were to exhaustion. Breon saw something close to panic on his opponent\'s face when the man finally recognized his error. A wave of despair rolled out from our rogue. Breon sensed it and knew the fifth point was his. Then the powerful twist that\'s rested aside, in and under the guard, Mr. Quilter is the apostle of the middle classes, and we\'re glad to welcome his gospel. Nor is Mr. Quilter\'s manner less interesting than his matter. He tells us that at this festive season of the year, with Christmas and Rose beef looming before us, similes drawn from eating and its results occur most readily to the mind. He has grave doubts whether Sir Frederick Layton\'s work is really Greek after all, and can discover in it but little of rocky Ithaca. Linnell\'s pictures are a sort of up-gards and atom paintings, and Mason\'s exquisite idles are as national as a jingo poem. Mr. Burkett Foster\'s landscapes smile at one much in the same way that Mr. Carker used to flash his teeth. And Mr. John Collier gives his sitter a cheerful slap in the back, before he says, like a shampooer and a Turkish bath. Next man, it is obviously unnecessary for us to point out how luminous these criticisms are, how delicate in expression. From the general principles of art, Mr. Quilter writes with equal lucidity. Painting he tells us is of a different quality to mathematics, and finish in art is adding more effect. As for etchings, there are two kinds, British and foreign. He laments most bitterly the divorce that has been made between decorative art and what we usually call pictures. The customary appeal to the last judgment and reminds us that in the great days of art Michelangelo was the furnishing upholsterer, near the fire, and the ornaments Fred brought home from India on the mental board. In fact, he is quite severe on Mr. Ruskin for not recognizing that a picture should denote the frailty of man. In remarks was pleasing courtesy and fellas of this grace that many faces are feeling. Only unfortunately his own work never does get good. Mr. Quilter has missed his chance, for he has failed even to make himself the tupper of painting. By Harry Quilter M.A. Because you are sleeping instead of conquering, the lovely rose princess has become a fiddle without a bow, while poor Shaggy sits there, accoing dove. He has gone, and gone for good," answered Polychrom, who had managed to squeeze into the room beside the dragon, and had witnessed the occurrences with much interest. I have remained a prisoner only because I wished to be one. And with this, he stepped forward and burst the stout chains as easily as if they had been threads. A little girl had been asleep, but she heard the wraps and opened the door. The king has fled and disgraced, and your friends are asking for you. I begged Ruggido long ago to send him away, but he would not do so. I also offered to help your brother to escape, but he would not go. He eats and sleeps very steadily, replied the new king. I hope he doesn\'t work too hard, since Shaggy. He doesn\'t work at all. In fact, there is nothing he can do in these dominions, as well as our gnomes, whose numbers are so great that it worries us to keep them all busy. Not exactly, we\'ve turned Calico, whereas my brother now, inquired Shaggy. In the metal forest. Where is that? The metal forest is in the great domed cavern, the largest in all our dominions, replied Calico. Calico hesitated. However, if we look sharp, we may be able to discover one of these secret ways. Oh no, I\'m quite sure he didn\'t. That\'s funny, remarked Betsy thoughtfully. I don\'t believe and knew any magic, or she\'d have worked it before. I do not know, confessed Shaggy. True, a great Calico. Calico went to the big gong and pounded on it, just as we\'re good to be used to do, but no one answered the summons. Having returned to the Royal Cavern, Calico first pounded the gong and then sat in the throne, wearing Regido\'s discarded ruby crown, and holding in his hand to scepter which Regido had so often thrown at his head.']
        # fmt: on

        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
        model = model.to(torch_device)

        ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean")
        one_audio = np.concatenate([x["array"] for x in ds["validation"]["audio"]], dtype=np.float32)

        input_features = processor(one_audio, return_tensors="pt", truncation=False, padding="longest")[
            "input_features"
        ]
        input_features = input_features.to(device=torch_device)

        gen_kwargs = {
            "return_timestamps": True,
            "no_speech_threshold": 0.6,
            "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            "num_beams": 2,
            "compression_ratio_threshold": 1.35,
            "condition_on_prev_tokens": True,
            "logprob_threshold": -1.0,
        }

        def check_gen_kwargs(inputs, generation_config, *args, **kwargs):
            assert generation_config.num_beams == gen_kwargs["num_beams"]

        self._patch_generation_mixin_generate(check_args_fn=check_gen_kwargs)

        torch.manual_seed(0)
        result = model.generate(input_features, **gen_kwargs)
        decoded = processor.batch_decode(result, skip_special_tokens=True)

        assert decoded == EXPECTED_TEXT

    @slow
    def test_whisper_longform_multi_batch(self):
        # fmt: off
        EXPECTED_TEXT_1 = [" Mr. Quilter's manner less interesting than his matter. He tells us that at this festive season of the year, with Christmas and roast beef looming before us, similes drawn from eating and its results occur most readily to the mind. He has grave doubts whether Sir Frederick Layton's work is really Greek after all, and can discover in it but little of rocky Ithaca. Linnell's pictures are a sort of up-gards and atom paintings, and Mason's exquisite idles are as national as a jingo poem. Mr. Birkett Foster's landscapes smile at one much in the same way that Mr. Carker used to flash his teeth. And Mr. John Collier gives his sitter a cheerful slap in the back, before he says, like a shampooer and a Turkish bath. Next man, it is obviously unnecessary for us to point out how luminous these criticisms are, how delicate an expression. On the general principles of art, Mr. Quilter writes with equal lucidity. Painting he tells us is of a different quality to mathematics, and finish in art is adding more effect. As for etchings, there are two kinds, British and foreign. He laments most bitterly the divorce that has been made between decorative art and what we usually call pictures. Mix a customary appeal to the last judgment and reminds us that in the great days of art Michelangelo was the furnishing a poster or near the fire, and the ornaments Fred brought home from India on the mental board. In fact, he is quite severe on Mr. Ruskin for not recognizing that a picture should denote the frailty of man. And remarks was pleasing courtesy in Felicitis Grace that many faces are feeling. Only unfortunately his own work never does get good. Mr. Quilter has missed his chance, for he has failed even to make himself the Tupper of painting. a Harry Quilter M.A. A man said to the universe, Sir, I exist. Sweat-covered Breon's body trickling into the tight-wing cloth that was the only germany war. The cut on his chest still dripping blood. The ache of his overstrained eyes, even the soaring arena around him with thousands of spectators, retrovealities not worth thinking about. His instant panic was followed by a small sharp blow high on his chest. One minute, a voice said, and a time buzzer sounded. A minute is not a very large measure of time, and his body needed every fraction of it. The buzzers were, triggered his muscles into complete relaxation. Oily his heart and lungs worked on at a strong, measured rate. He was in reverie, sliding along the borders of consciousness. The contestants in the 20s needed undisturbed rest. Therefore, knights in the dormitories were as quiet as death. Particularly so, on this last night, when only two of the little cubicles were occupied, the thousands of others standing with dark empty doors. The other voice snapped with a harsh urgency, clearly used to command. I'm here because the matter is of utmost importance, and brand is the one I must see. Now stand aside. The twenty's he must have drawn his gun, because the intruder said quickly, but that away you're being a fool. Out there was silence then, and still wondering, Breon was once more asleep. Ten seconds he asked the handler who was needing his aching muscles. a red-haired mountain of a man with an apparently inexhaustible store of energy. There could be little art in this last and final round of fencing, just thrust and parry and victory to the stronger. Every man who entered the twenties had his own training tricks. There appeared to be an immediate association with the death trauma as if the two were andextricably linked into one. The strength that enables someone in a trance to hold his body stiff and unsupported except at two points, the head and heels. This is physically impossible when conscious. Others had died before during the twenties and death during the last round was, in some ways, easier than defeat. Breeding deeply, Breon's softly spoke the auto-hypnotic phrases that triggered the process. When the buzzer sounded, he pulled his foil from his second startled grasp and ran forward. I rolled the mazed at the sudden fury of the attack, then smiled. He thought it was the last burst of energy. He knew how close they both were to exhaustion. Breon saw something close to panic on his opponent's face when the man finally recognized his error. A wave of despair rolled out from our rogue, pre-inscented and new to fifth point was his. Then the powerful twist that's rest of the side, in and under the guard, because you were sleeping instead of conquering, the lovely rose princess has become a fiddle without a bow, while poor Shaggy sits there, a cooing dove. He has gone and gone for good, answered Polychrome, who had managed to squeeze into the room beside the dragon, and had witnessed the occurrences with much interest. I have remained a prisoner only because I wished to be one. And with this, he stepped forward and burst the stout chains as easily as if they had been threads. The little girl had been asleep, but she heard the wraps and opened the door. The king has flooded disgrace, and your friends are asking for you. I begged Ruggadot long ago to send him away, but he would not do so. I also offered to help your brother to escape, but he would not go. He eats and sleeps very steadily, replied the new king. I hope he doesn't work too hard, since Shaggy. He doesn't work at all. In fact, there's nothing he can do in these dominions, as well as our gnomes, whose numbers are so great that it worries us to keep them all busy. Not exactly, return Calico. Where is my brother now? choir-dshaggy, in the metal forest. Where is that? The metal forest is in the great domed cavern, the largest and all-ard dominions, replied Calico. Calico hesitated. However, if we look sharp, we may be able to discover one of these secret ways. Oh, no, I'm quite sure he didn't. That's funny, remarked Betsy thoughtfully. I don't believe and knew any magic, or she'd have worked it before. I do not know, confess shaggy. True, a great calico. Calico went to the big gong and pounded on it, just as Virgado used to do, but no one answered the summons. Having returned to the Royal Cavern, Calico first pounded the gong and then sat in the throne, wearing Virgados discarded Ruby Crown, and holding in his hand to scepter, which Virgado had so often thrown at his head. head."]
        EXPECTED_TEXT_2 = [" Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel. Nor is Mr. Quilter's manner less interesting than his matter. He tells us that at this festive season of the year, with Christmas and roast beef looming before us, similes drawn from eating and its results occur most readily to the mind. He has grave doubts whether Sir Frederick Layton's work is really Greek after all, and can discover in it but little of rocky Ithaca. Linnell's pictures are a sort of up-gards and atom paintings, and Mason's exquisite idles are as national as a jingo poem. Mr. Burkett Foster's landscapes smile at one much in the same way that Mr. Carker."]
        EXPECTED_TEXT_3 = [" possible. Nor is Mr. Quilter's manner less interesting than his matter. He tells us that at this festive season of the year, with Christmas and roast beef looming before us, similes drawn from eating and its results occur most readily to the mind. He has grieved doubts whether Sir Frederick Layton's work is really greek after all, and can discover in it but little of rocky Ithaca. Linnell's pictures are a sort of up-guards and atom paintings, and Mason's exquisite idles are as national as a jingo poem. Mr. Birk at Foster's landscapes smile at one much in the same way that Mr. Carker used to flash his teeth. And Mr. John Collier gives his sitter a cheerful slap in the back, before he says, like a shampooer and a Turkish bath, next man, it is obviously unnecessary for us to point out how luminous these criticisms are, how delicate an expression. Under general principles of art, Mr. Quilter writes with equal lucidity. Painting, he tells us, is of a different quality to mathematics and finish in art is adding more effect. As for etchings, there are two kinds, British and foreign. He laments most bitterly the divorce that has been made between decorative art and what we usually call pictures. Mix a customary appeal to the last judgment and reminds us that in the great days of art Michelangelo was the furnishing upholsterer. Near the fire. any ornaments Fred brought home from India on the mental board. In fact, he is quite severe on Mr. Ruskin for not recognizing that a picture should denote the frailty of man, and remarks was pleasing courtesy in Felicitis Grace that many faces are feeling. Only, unfortunately, his own work never does get good. Mr. Quilter has missed his chance, for he has failed even to make himself the tupper of painting. By Harry Quilter M.A. A man said to the universe, Sir, I exist. Sweat-covered Breon's body trickling into the titling cloth that was the only german he wore. The cut on his chest still dripping blood. The ache of his overstrained eyes. Even to soaring arena around him with thousands of spectators, retrovealities not worth thinking about. His instant panic was followed by a small sharp blow high on his chest. One minute, a voice said, and a time buzzer sounded. A minute is not a very large measure of time, and his body needed every fraction of it. The buzzers were triggered as muscles into complete relaxation. Oily his heart and lungs worked on at a strong measured rate. He was in In reverie, sliding along the borders of consciousness. The contestants in the 20s needed undisturbed rest. Therefore, nights in the dormitories were as quiet as death. Particularly so, on this last night, when only two of the little cubicles were occupied, the thousands of others standing with dark empty doors. The other voice snapped with a harsh urgency clearly used to command. I'm here because the matter is of utmost importance, and brand is the one I must see. Now stand aside. The twenty's he must have drawn his gun, because the intruder said quickly, but that away you're being a fool. Out there was silence then, and still wondering, Breon was once more asleep. Ten seconds he asked the handler who was needing his aching muscles. a red-haired mountain of a man with an apparently inexhaustible store of energy. There could be little art in this last and final round of fencing, just thrust and parry and victory to the stronger. Every man who entered the twenties had his own training tricks. There appeared to be an immediate association with the death trauma as if the two were andextricably linked into one. The strength that enables someone in a trance to hold his body stiff and unsupported except at two points, the head and heels. This is physically impossible when conscious. Others had died before during the twenties and death during the last round was, in some ways, easier than defeat. Breeding deeply, Breon's softly spoke the auto-hypnotic phrases that triggered the process. When the buzzer sounded, he pulled his foil from his second startled grasp and ran forward. Our role looked amazed at the sudden fury of the attack, then smiled. He thought it was the last burst of energy. He knew how close they both were to exhaustion. Breon saw something close to panic on his opponent's face when the man finally recognized his error. A wave of despair rolled out from our rogue, re-insunced it and knew the fifth point was his. Then the powerful twist that's rest of the side, in and under the guard, because you were sleeping instead of conquering, the lovely rose princess has become a fiddle without a bow, while poor Shaggy sits there, a cooing dove. He has gone and gone for good, answered Polychrome, who had managed to squeeze into the room beside the dragon, and had witnessed the occurrences with much interest. I have remained a prisoner only because I wished to be one. And with this, he stepped forward and burst the stout chains as easily as if they had been threads. The little girl had been asleep, but she heard the wraps and opened the door. The king has fled and disgraced, and your friends are asking for you. I begged Ruggadot long ago to send him away, but he would not do so. I also offered to help your brother to escape, but he would not go. He eats and sleeps very steadily, replied the new king. I hope he doesn't work too hard, since Shaggy. He doesn't work at all. In fact, there's nothing he can do in these dominions as well as our gnomes, whose numbers are so great that it worries us to keep them all busy. Not exactly, we've turned Calico. Where is my brother now? quared shaggy. In the metal forest. Where is that? The metal forest is in the great domed cavern, the largest and all-ard dominions, replied Calico. Calico hesitated. However, if we look sharp, we may be able to discover one of these secret ways. Oh no, I'm quite sure he didn't. And that's funny, remarked Betsy thoughtfully. I don't believe Anne knew any magic, or she'd have worked it before. I do not know, confess Shaggy. True, a great calico. Calico went to the big gong and pounded on it, just as we're good to have used to do, but no one answered the summons. Having returned to the Royal Cavern, Calico first pounded the gong and then sat in the thrown wearing ruggedos discarded ruby crown and holding in his hand to septor which ruggedo had so often thrown at his head."]
        EXPECTED_TEXT_4 = [' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel. Nor is Mr. Quilter\'s manner less interesting than his matter. He tells us that at this festive season of the year, with Christmas and roast beef looming before us, similes drawn from eating and its results occur most readily to the mind. He has grave doubts whether Sir Frederick Layton\'s work is really Greek after all, and can discover in it but little of rocky Ithaca. Linnell\'s pictures are a sort of up-gards and atom paintings, and Mason\'s exquisite idles are as national as a jingo poem. Mr. Birk at Foster\'s landscapes smile at one much in the same way that Mr. Carker used to flash his teeth. Mr. John Collier gives his sitter a cheerful slap in the back, before he says, like a shampoo or a Turkish bath. Next man, it is obviously unnecessary for us to point out how luminous these criticisms are, how delicate an expression. On the general principles of art, Mr. Quilter writes with equal lucidity. he tells us is of a different quality to mathematics, and finish in art is adding more effect. As for etchings, there are two kinds, British and foreign. He laments most bitterly the divorce that has been made between decorative art and what we usually call pictures. Makes the customary appeal to the last judgment and reminds us that in the great days of art Michelangelo was the furnishing upholsterer. Near the fire, any ornaments Fred brought home from India on the mantelboard. In fact, he is quite severe on Mr. Ruskin for not recognizing that a picture should denote the frailty of man. And remarks was pleasing courtesy in Felicitis Grace that many faces are feeling. Only, unfortunately, his own work never does get good. Mr. Quilter has missed his chance, for he has failed even to make himself the Tupper of painting. By Harry Quilter M.A. A man said to the universe, Sir, I exist. Sweat-covered Breon\'s body trickling into the tight-lowing cloth that was the only german he wore. The cut on his chest still dripping blood. The ache of his overstrained eyes, even the soaring arena around him with thousands of spectators, retrovealities not worth thinking about. His instant panic was followed by a small sharp blow high on his chest. One minute, a voice said, and a time buzzer sounded. A minute is not a very large measure of time, and his body needed every fraction of it. The buzzers were triggered his muscles into complete relaxation. Oli\'s heart and lungs worked on at a strong, measured rate. He was in reverie, sliding along the borders of consciousness. The contestants in the twenties needed undisturbed rest. Therefore, nights in the dormitories were as quiet as death. Particularly so, on this last night, when only two of the little cubicles were occupied, The thousands of others standing with dark empty doors. The other voice snapped with a harsh urgency, clearly used to command. I\'m here because the matter is of utmost importance, and brand is the one I must see. Now stand aside. The twenties, he must have drawn his gun because the intruder said quickly, but that away you\'re being a fool. out, through his silence then, and still wondering, Breon was once more asleep. Ten seconds, he asked the handler who was needing his aching muscles. A red-haired mountain of a man, with an apparently inexhaustible store of energy. There could be little art in this last and final round of fencing. Just thrust and parry, and victory to the stronger. man who entered the twenties had his own training tricks. They were appeared to be an immediate association with the death trauma, as if the two were inextricably linked into one. The strength that enables someone in a trance to hold his body stiff and unsupported except at two points, the head and heels. This is physically impossible when conscious. had died before during the 20s and death during the last round was in some ways easier than defeat. Breathing deeply, Breon\'s softly spoke the auto-hypnotic phrases that triggered the process. When the buzzer sounded, he pulled his foil from his second startled grasp and ran forward. Our role looked amazed at the sudden fury of the attack, then smiled. He thought it was the last burst of energy. He knew how close they both were to exhaustion. Breon saw something close to panic on his opponent\'s face when the man finally recognized his error. A wave of despair rolled out from our rogue. Breon sensed it and knew the fifth point was his. Then the powerful twist that\'s rested aside, in and under the guard, because he was sleeping instead of conquering, the lovely rose princess has become a fiddle without a bow, while poor Shaggy sits there, accooing dove. He has gone, and gone for good," answered Polychrom, who had managed to squeeze into the room beside the dragon, and had witnessed the occurrences with much interest. I have remained a prisoner only because I wished to be one. And with says he stepped forward and burst the stout chains as easily as if they had been threads. The little girl had been asleep, but she heard the wraps and opened the door. The king has flooded disgrace, and your friends are asking for you. I begged Ruggadot long ago to send him away, but he would not do so. I also offered to help your brother to escape, but he would not go. He eats and sleeps very steadily, replied the new king. I hope he doesn\'t work too hard, said Shaggy. He doesn\'t work at all. In fact, there\'s nothing he can do in these dominions as well as our gnomes, whose numbers are so great that it worries us to keep them all busy. Not exactly, we\'ve turned Calico. Where is my brother now, inquired Shaggy. In the metal forest. Where is that? The middle forest is in the great domed cavern, the largest and all-ard dominions, replied Calico. Calico hesitated. However, if we look sharp, we may be able to discover one of these secret ways. Oh no, I\'m quite sure he didn\'t. That\'s funny, remarked Betsy thoughtfully. I don\'t believe Anne knew any magic, or she\'d have worked it before. I do not know, confess Shaggy. True, agreed Calico. Calico went to the big gong and pounded on it just as Virgato used to do, but no one answered the summons. Having returned to the Royal Cavern, Calico first pounded the gong and then sat in the throne, wearing Virgato\'s discarded ruby crown and holding in his hand to scepter which reggative head so often thrown at his head.']
        # fmt: on

        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
        model = model.to(torch_device)

        ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean")
        one_audio = np.concatenate([x["array"] for x in ds["validation"]["audio"]], dtype=np.float32)
        audios = []
        audios.append(one_audio[110000:])
        audios.append(one_audio[:800000])
        audios.append(one_audio[80000:])
        audios.append(one_audio[:])

        decoded_single = []
        for audio in audios:
            inputs = processor(audio, return_tensors="pt", truncation=False)
            inputs = inputs.to(device=torch_device)

            result = model.generate(**inputs, return_timestamps=True)
            decoded_single.append(processor.batch_decode(result, skip_special_tokens=True))

        inputs = processor(
            audios, return_tensors="pt", truncation=False, padding="longest", return_attention_mask=True
        )
        inputs = inputs.to(device=torch_device)

        result = model.generate(**inputs, return_timestamps=True)
        decoded_all = processor.batch_decode(result, skip_special_tokens=True)

        # make sure single & batch is exactly the same
        assert decoded_all[0:1] == decoded_single[0]
        assert decoded_all[1:2] == decoded_single[1]
        assert decoded_all[2:3] == decoded_single[2]
        assert decoded_all[3:4] == decoded_single[3]

        # exact match
        assert decoded_all[0:1] == EXPECTED_TEXT_1
        assert decoded_all[1:2] == EXPECTED_TEXT_2
        assert decoded_all[2:3] == EXPECTED_TEXT_3
        assert decoded_all[3:4] == EXPECTED_TEXT_4

    @slow
    def test_whisper_longform_multi_batch_prev_cond(self):
        # fmt: off
        EXPECTED_TEXT_1 = [" Mr. Quilters manner less interesting than his matter. He tells us that at this festive season of the year, with Christmas and roast beef looming before us, similarly drawn from eating and its results occur most readily to the mind. He has grave doubts whether Sir Frederick Layton's work is really Greek after all and can discover in it but little of Rocky Ithaca. The Nils, pictures are sort of upguards and atom paintings and Mason's exquisite itals are as national as a jingo poem. Mr. Berkett Foster's landscapes smile at one much in the same way that Mr. Carker used to flash his teeth. And Mr. John Collier gives his sitter a cheerful slap on the back before he says like a shampooer and a Turkish bath. Next man, it is obviously unnecessary for us to point out how luminous these criticisms are, how delicate and expression. On the general principles of art, Mr. Quilters writes with equal lucidity. Painting he tells us is of a different quality to mathematics and finish in art is adding more effect. As for etchings, there are of two kinds, British and foreign. He laments most bitterly the divorce that has been made between decorative art and what we usually call pictures makes a customary appeal to the last judgment and reminds us that in the great days of art Michelangelo was the furnishing apostorer. Near the fire, any ornaments Fred brought home from India on the mental board. In fact, he is quite severe on Mr. Ruskin, for not recognizing that a picture should denote the frailty of man. And remarks with pleasing courtesy and solicitous grace that many phases of feeling only, unfortunately, his own work never does get good. Mr. Quilters has missed his chance, for he has failed even to make himself the tougher of painting. My hair equal to M.A. A man said to the universe, Sir, I exist. Sweat covered Breon's body, trickling into the tight-wing cloth that was the only garment he wore. The cut on his chest still dripping blood. The ache of his overstrain dyes. Even the soaring arena around him with thousands of spectators, retrievalidies not worth thinking about. His instant panic was followed by a small sharp blow, high on his chest. One minute, a voice said, and a time buzzer sounded. A minute is not a very large measure of time, and his body needed every fraction of it. The buzzer's were triggered as muscles into complete relaxation. Only his heart and lungs worked on at a strong, measured rate. He was in reverie, sliding along the borders of consciousness. The contestants in the 20s needed undisturbed rest. Therefore, knights and the dormitories were as quiet as death. Particularly so, on this last night, when only two of the little cubicles were occupied, the thousands of others standing with dark empty doors. The other voice snapped with a harsh urgency, clearly used to command. I'm here because the matter is of utmost importance. And brand is the one I must see. Now stand aside. To 20s, he must have drawn his gun because the intruder said quickly. But that away, he'd be no fool. Out, the resoundance then, and still wondering, Brienne was once more asleep. Ten seconds, he asked the handler who was needing his aching muscles. A red-haired mountain of a man, with an apparently inexhaustible story of energy. There could be little art in this last and final round of fencing. Just thrust and parry and victory to the stronger. Every man who entered the 20s had his own training tricks. There appeared to be an immediate association with the death trauma as if the two were inexplicably linked into one. This strength that enables someone in a trance to hold his body stiff and unsupported, except at two points, the head and heels. This is physically impossible when conscious. Others had died before during the 20s, and death during the last round was, in some ways, easier than defeat. Breathing deeply, Brienne softly spoke the other hypnotic phrases that triggered the process. In the buzzer sounded, he pulled his foil from his second startled grasp and ran forward. I rolled the maze at the sudden fury of the attack, then smiled. He thought it was the last burst of energy. He knew how close they both were to exhaustion. Brienne saw something close to panic on his opponent's face when the man finally recognized his error. A wave of despair rolled out from our role. Brienne sensed it and knew the fifth point was his. Then the powerful twist that's right to the side, in and under the guard, because he was sleeping instead of conquering, the lovely rose princess has become a fiddle with a bow, while poor shaggy sits there, a cooling dove. He has gone and gone for good, answered polychrome, who had managed to squeeze into the room beside the dragon, and had witnessed the occurrences with much interest. I have remained a prisoner only because I wished to be one. And with this, he stepped forward and burst the stoutchanges as easily as if they had been threads. The little girl had been asleep, but she heard the wraps and opened the door. The king has fled in disgrace in your friends, they're asking for you. I begged Ruggano a long ago to send him away, but he would not do so. I also offered to help you run into escape, but he would not go. He eats and sleeps very steadily, replied the new king. I hope he doesn't work too hard since shaggy. He doesn't work at all. In fact, there's nothing he can do in these dominions, as well as our nooms, whose numbers are so great that it worries us to keep them all busy. Not exactly, we've turned Calico, whereas my brother now inquired shaggy in the metal forest. Where is that? The metal forest is in the great domed cavern, the largest and all our dominions replied Calico. Calico hesitated. However, if we look sharp, we may be able to discover one of these secret ways. Oh no, I'm quite sure he didn't. That's funny, remarked to Bedsey thoughtfully. I don't believe Anne knew any magic or she'd have worked before. I do not know, confessed shaggy. True, agreed Calico. Calico went to the big gong and pounded on it just as Ruggano used to do, but no one answered the summons. Having returned to the royal cavern, Calico first pounded the gong and then sat in the throne, wearing Ruggano's discarded ruby crown. And holding in his hand the scepter which Ruggano had so often thrown at his head."]
        EXPECTED_TEXT_2 = [" Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel. Nor is Mr. Quilter's manner less interesting than his matter. He tells us that at this festive season of the year, with Christmas and roast beef looming before us, similarly drawn from eating and its results occur most readily to the mind. He has grave doubts whether Sir Frederick Latins' work is really Greek after all, and can discover in it but little of rocky Ithaca. Lennials, pictures are a sort of upguards and atom paintings, and Mason's exquisite idles are as national as a jingo poem. Mr. Berkett Foster's landscapes smile at one much in the same way that Mr. Carker"]
        EXPECTED_TEXT_3 = [" gospel. Nor is Mr. Quilter's manner less interesting than his matter. He tells us that at this festive season of the year, with Christmas and roast beef looming before us, similarly drawn from eating in its results occur most readily to the mind. He has grave doubts whether Sir Frederick Latins work is really Greek after all and can discover in it but little of rocky ithaka. Lennils, pictures, are a sort of upguards and atom paintings and Mason's exquisite itals are as national as a jingo poem. Mr. Birkut Foster's landscapes smile at one much in the same way that Mr. Carker used to flash his teeth. And Mr. John Collier gives his sitter a cheerful slap on the back before he says like a shampooer and a Turkish bath. Next man, it is obviously unnecessary for us to point out how luminous these criticisms are, how delicate and expression. Under general principles of art, Mr. Quilter writes with equal lucidity. Painting he tells us is of a different quality to mathematics and finish in art is adding more effect. As for etchings, thereof two kinds, British and foreign. He laments most bitterly the divorce that has been made between decorative art and what we usually call pictures makes a customary appeal to the last judgment and reminds us that in the great days of art Michelangelo was the furnishing apostoror. Near the fire, any ornaments spread brought home from India on the mental board. In fact, he is quite severe on Mr. Ruskin for not recognizing that a picture should denote the frailty of man. And remarks with pleasing courtesy and solicitous grace that many faces are feeling, only unfortunately his own work never does get good. Mr. Quilter has missed his chance. For he has failed even to make himself the tougher of painting. By Harry Quilter M.A. A man said to the universe, Sir, I exist. Sweat covered Brienne's body trickling into the tight-wing cloth that was the only garment you wore. The cut on his chest still dripping blood. The ache of his overstrained eyes. Even the soaring arena around him with thousands of spectators, retrievalidies not worth thinking about. His instant panic was followed by a small sharp blow, high on his chest. One minute, a voice said, and a time buzzer sounded. A minute is not a very large measure of time, and his body needed every fraction of it. The buzzer's were triggered his muscles into complete relaxation. Only his heart and lungs worked on at a strong measured rate. He was in reverie, sliding out on the borders of consciousness. The contestants in the 20s needed undisturbed rest. Therefore, knights and the dormitories were as quiet as death. Particularly so, on this last night, when only two of the little cubicles were occupied, the thousands of others standing with dark empty doors. The other voice snapped with a harsh urgency, clearly used to command. I'm here because the matter is of utmost importance, and brand is the one I must see. Now stand aside. The 20s, he must have drawn his gun because the intruder said quickly, but that away here being a fool. Out, there is silence then, and still wondering, Brienne was once more asleep. 10 seconds, he asked the handler who was needing his aching muscles. I've read here at Mountain of a Man with an apparently inexhaustible story of energy. There could be little art in this last and final round of fencing, just thrust and parry and victory to the stronger. Every man who entered the 20s had his own training tricks. There appeared to be an immediate association with the death trauma as if the two were anextricably linked into one. The strength that enables someone in a trance to hold his body stiff and unsupported, except at two points, the head and heels. This is physically impossible when conscious. Others had died before during the 20s, and death during the last round was, in some ways, easier than defeat. Breathing deeply, Brienne's softly spoke the odd hypnotic phrases that triggered the process. When the buzzer sounded, he pulled his foil from his second startled grasp and ran forward. I rolled up the maze at the sudden fury of the attack, then smiled. He said it was the last burst of energy. He knew how close they both were to exhaustion. Brienne saw something close to panic on his opponent's face when the man finally recognized his error. A wave of despair rolled out from our ol' Brienne sensed it and knew the fifth point was his. Then the powerful twist that's right to decide, in and under the guard, because he was sleeping instead of conquering, the lovely rose princess has become a fiddle with a bow, while poor shaggy sits there, a cooling dove. He has gone and gone for good, answered polychrome, who had managed to squeeze into the room beside the dragon, and had witnessed the occurrences with much interest. I have remained a prisoner only because I wished to be one. And with this, he stepped forward and burst the stout chains as easily as if they had been threads. The little girl had been asleep, but she heard the wraps and opened the door. The king has fled in disgrace in your friends, they're asking for you. I begged Brienne to long ago to send him away, but he would not do so. I also offered to help you brother to escape, but he would not go. He eats and sleeps very steadily, replied the new king. I hope he doesn't work too hard, since Shaggy. He doesn't work at all. In fact, there's nothing he can do in these dominions as well as our nooms, whose numbers are so great that it worries us to keep them all busy. Not exactly, we've turned Calico, whereas my brother now inquired Shaggy in the metal forest. Where is that? The metal forest is in the great domed cavern, the largest and all our dominions replied Calico. Calico hesitated. However, if we look sharp, we may be able to discover one of these secret ways. Oh no, I'm quite sure he didn't. That's funny, remarked to bed see you thoughtfully. I don't believe Anne knew any magic or she'd have worked it before. I do not know, confessed Shaggy. True, agreed Calico. Calico went to the big gone and pounded on it, just as we're good or used to do, but no one answered the summons. Having returned to the royal cavern, Calico first pounded the gone and then sat in the throne, wearing reggos, discarded ruby crown, and holding in his hand to scepter which reggos hand so often thrown at his head."]
        EXPECTED_TEXT_4 = [" Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel. Nor is Mr. Quilter's manner less interesting than his matter. He tells us that at this festive season of the year, with Christmas and roast beef looming before us, similarly drawn from eating and its results occur most readily to the mind. He has grave doubts whether Sir Frederick Latins' work is really Greek after all, and can discover in it but little of rocky Ithaca. Lennils, pictures, are a sort of upguards and atom paintings, and Mason's exquisite idles are as national as a jingo poem. Mr. Berkett Foster's landscapes smile at one much in the same way that Mr. Carker used to flash his teeth. And Mr. John Collier gives his sitter a cheerful slap on the back before he says, like a shampooer in a Turkish bath. Next man, it is obviously unnecessary for us to point out how luminous these criticisms are, how delicate and expression. On the general principles of art, Mr. Quilter writes with equal lucidity. Painting he tells us is of a different quality to mathematics, and finish in art is adding more effect. As for etchings, thereof two kinds, British and foreign. He laments most bitterly the divorce that has been made between decorative art and what we usually call pictures makes a customary appeal to the last judgment and reminds us that in the great days of art Michelangelo was the furnishing apostorer. Near the fire, any ornaments Fred brought home from India on the mental board. In fact, he is quite severe on Mr. Ruskin, for not recognizing that a picture should denote the frailty of man. And remarks with pleasing courtesy and solicitous grace that many phases of feeling only, unfortunately, his own work never does, get good. Mr. Quilter has missed his chance, for he has failed even to make himself the tougher of painting. By Harry Quilter, M.A. A man said to the universe, Sir, I exist. Sweat covered Breon's body, trickling into the tight-wing cloth that was the only garment you wore. The cut on his chest still dripping blood. The ache of his overstrained eyes, even the soaring arena around him with thousands of spectators were trivialities not worth thinking about. His instant panic was followed by a small sharp blow, high on his chest. One minute, a voice said, and a time buzzer sounded. A minute is not a very large measure of time, and his body needed every fraction of it. The buzzer's were triggered as muscles into complete relaxation. Only his heart and lungs worked on at a strong, measured rate. He was in reverie, sliding along the borders of consciousness. The contestants in the 20s needed undisturbed rest. Therefore, knights and the dormitories were as quiet as death. Particularly so, on this last night, when only two of the little cubicles were occupied, the thousands of others standing with dark empty doors. The other voice snapped with a harsh urgency, clearly used to command. I'm here because the matter is of utmost importance. And brand is the one I must see. Now stand aside. To 20s, he must have drawn his gun because the intruder said quickly, but that away, he could be no fool. Out, there was silence then, and still wondering, Brienne was once more asleep. Ten seconds, he asked the handler who was needing his aching muscles. I've read here at Mountain of a Man, with an apparently inexhaustible story of energy. There could be little art in this last and final round of fencing. Just thrust and parry and victory to the stronger. Every man who entered the 20s had his own training tricks. There appeared to be an immediate association with the death trauma, as if the two were inextricably linked into one. The strength that enables someone in a trance to hold his body stiff and unsupported, except at two points, the head and heels. This is physically impossible when conscious. Others had died before during the 20s, and death during the last round was, in some ways, easier than defeat. Breathing deeply, Brienne softly spoke the other hypnotic phrases that triggered the process. When the buzzer sounded, he pulled his foil from his second startled grasp and ran forward. I rolled the maze at the sudden fury of the attack, then smiled. He thought it was the last burst of energy. He knew how close they both were to exhaustion. Brienne saw something close to panic on his opponent's face when the man finally recognized his error. A wave of despair rolled out from Irohog. Brienne sensed it and knew the fifth point was his. Then the powerful twist that's for us to decide, in and under the guard, because he was sleeping instead of conquering, the lovely rose princess has become a fiddle with a bow, while poor shaggy sits there, a cooling dove. He has gone and gone for good, answered polychrome, who had managed to squeeze into the room beside the dragon, and had witnessed the occurrences with much interest. I have remained a prisoner only because I wished to be one. And with this, he stepped forward and burst the stoutchanges as easily as if they had been threads. The little girl had been asleep, but she heard the wraps and opened the door. The king has fled in disgrace in your friends, they are asking for you. I begged Ruggano a long ago to send him away, but he would not do so. I also offered to help you brother to escape, but he would not go. He eats and sleeps very steadily, replied the new king. I hope he doesn't work too hard since shaggy. He doesn't work at all. In fact, there is nothing he can do in these dominions, as well as our nooms, whose numbers are so great that it worries us to keep them all busy. And exactly we've turned Calico, where is my brother now in Quaragejji, in the metal forest? Where is that? The metal forest is in the great donned cavern, the largest and all our dominions replied Calico. Calico hesitated. However, if we look sharp, we may be able to discover one of these secret ways. Oh no, I'm quite sure he didn't. That's funny, remarked to Bedzeeth thoughtfully. I don't believe Anne knew any magic or she'd have worked before. I do not know, confessed shaggy. True, agreed Calico. Calico went to the big gong and pounded on it just as we're good to have used to do, but no one answered the summons. Having returned to the royal cavern, Calico first pounded the gong and then sat in the throne, wearing reggos, discarded ruby crown. And holding in his hand to scepter which reggos had so often thrown at his head."]
        # fmt: on

        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
        model = model.to(torch_device)

        ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean")
        one_audio = np.concatenate([x["array"] for x in ds["validation"]["audio"]], dtype=np.float32)
        audios = []
        audios.append(one_audio[110000:])
        audios.append(one_audio[:800000])
        audios.append(one_audio[80000:])
        audios.append(one_audio[:])

        gen_kwargs = {
            "return_timestamps": True,
            "no_speech_threshold": 0.6,
            "temperature": 0.0,
            "compression_ratio_threshold": 1.35,
            "condition_on_prev_tokens": True,
            "logprob_threshold": -1.0,
        }

        decoded_single = []
        for audio in audios:
            inputs = processor(audio, return_tensors="pt", truncation=False)
            inputs = inputs.to(device=torch_device)

            result = model.generate(**inputs, **gen_kwargs)
            decoded_single.append(processor.batch_decode(result, skip_special_tokens=True))

        # exact match
        assert decoded_single[0] == EXPECTED_TEXT_1
        assert decoded_single[1] == EXPECTED_TEXT_2
        assert decoded_single[2] == EXPECTED_TEXT_3
        assert decoded_single[3] == EXPECTED_TEXT_4

    @slow
    def test_whisper_longform_multi_batch_hard(self):
        # fmt: off
        EXPECTED_TEXT = [
            " Folks, if you watch the show, you know, I spent a lot of time right over there. Patiently and astutely scrutinizing the boxwood and mahogany chest set of the day's biggest stories developing the central headline pawns, definitely maneuvering an oso topical night to F6, fainting a classic Sicilian, nade door variation on the news, all the while seeing eight moves deep and patiently marshalling the latest press releases into a fisher's shows in Lip Nitsky attack that culminates in the elegant lethal slow-played, all-passant checkmate that is my nightly monologue. But sometimes, sometimes, folks, I. CHEERING AND APPLAUSE Sometimes I startle away, cubside down in the monkey bars of a condemned playground on a super fun site. Get all hept up on goofballs. Rummage that were discarded tag bag of defective toys. Yank out a fist bowl of disembodied doll limbs, toss them on a stained kid's place mat from a defunct dennies. set up a table inside a rusty cargo container down by the Wharf and challenged toothless drifters to the godless bughouse blitz of tournament that is my segment. Meanwhile.",
            " Folks, I spend a lot of time right over there, night after night after night, actually. Carefully selecting for you the day's noosiest, most aerodynamic headlines, stress testing, and those topical anti-lock breaks and power steering, painstakingly stitching, leather seating so soft, it would make JD power and her associates blush to create the luxury sedan that is my nightly monologue. But sometimes, you sometimes, folks. I lurched a consciousness in the back of an abandoned school and slap myself awake with a crusty floor mat. Before using a mouse-bitten timing belt to strap some old plywood to a couple of discarded oil drums, then by the light of a heathen moon, render a gas tank out of an empty big gulp, fill with white claw and denatured alcohol, then light a match and let her rip and the demented one man soapbox derby of news that is my segment. Me, Guadalupe! No!",
            " Ladies and gentlemen, you know, I spent a lot of time right over there Raising the finest Holstein news cattle firmly yet tenderly milking the latest headlines from their jokes swollen teats Churning the daily stories into the decadent proven-style style triple cream breed that is my nightly monologue But sometimes sometimes folks I stagger home hungry after being released by the police and Root around in the neighbor's trash can for an old milk carton scrape out the blooming dairy residue into the remains of a wet cheese rod I won from a rat in a pre-donned street fight. Put it in a discarded paint can to leave it to ferment next to a trash fire then hunker down and hallucinate while eating the listeria laden demon custard of news that is my segment. You mean one of them.",
            " Folks, if you watch this show, you know I spend most of my time right over there carefully sorting through the day's biggest stories and selecting only the most subtle and unblemished ostrich and crocodile news leather, which I then entrust to artisan graduates of the Ichol Gregoire Ferrandi, who carefully dye them in a palette of bright zesty shades and adorn them in the finest and most topical inlay work using hand tools and double magnifying glasses, then assemble them according to now classic and elegant geometry using our signature saddles stitching. In line it with bees, wax, coated linen, finely attached a mallet, hammered strap, pearled hardware, and close-shit to create for you the one-of-a-kind hoke couture, Erme's Birkin bag that is my monologue. But sometimes, sometimes folks, sometimes. Sometimes I wake up in the last car of an abandoned roller coaster at Coney Island where I'm I'm hiding from the triads. I have some engine lubricants out of a safe way bag and stagger down the shore to tear the sail off a beach schooner. Then I rip the coaxial cable out of an RV and elderly couple from Utah, Hank, and Mabel lovely folks. And use it to stitch the sail into a loose pouch like a rock sack. And I stow away in the back of a garbage truck to the junkyard where I pick through to the debris for only the broken toys that make me the saddest until I have loaded for you. The Hobo Fugitives bug out, bindle of news that is my segment. Me one!",
            " You know, folks, I spent a lot of time crafting for you a bespoke playlist of the day's biggest stories right over there. Meticulously selecting the most topical chakra affirming scented candles, and using Feng Shui to perfectly align the joke energy in the exclusive boutique yoga retreat that is my monologue. But sometimes just sometimes I go to the dumpster behind the waffle house at three in the morning, take off my shirt, cover myself, and used fry oil, wrap my hands with some double-duct tape by stole from the broken car window. Pound a six-pack of blueberry hard-seltzer and a sack of pills I stole from a parked ambulance. Then arm wrestle a raccoon in the back alley vision quest of news that is my segment. Meanwhile!",
            " You know, folks, I spend most of my time right over there. Mining the day's biggest, most important stories, collecting the finest, most topical iron or hand hammering it into joke panels. Then I craft sheets of bronze and blazing with patterns that tell an epic tale of conquest and glory. Then, using the Germanic tradition press-black process, I place thin sheets of foil against the scenes and by hammering or otherwise applying pressure from the back, I project these scenes into a pair of cheat cards in a faceplate and, finally, using fluted strips of white alloyed molding, I divide the designs into framed panels and hold it all together using bronze rivets to create the beautiful and intimidating, Anglo-Saxon battle helm that is my nightly monologue. Sometimes, sometimes folks. Sometimes, just sometimes, I come into my sense as fully naked on the deck of a pirate besieged melee container ship that picked me up floating on the detached door of a portapotty in the Indian Ocean. Then after a sunstroke-induced realization of the crew of this ship plans to sell me an exchange for a bag of oranges to fight off scurvy, I lead a mutiny using only a PVC pipe at a pool chain that accepting my new role as Captain and declaring myself king of the windarc seas. I grab a dirty mop bucket covered in barnacles and adorn it with the teeth of the vanquished to create the sopping wet pirate crown of news that is my segment. Meanwhile!",
            " Folks, if you watch this show, you know I spend most of my time right over there carefully blending for you the day's Newsiest most topical flower eggs milk and butter and Stranding into a fine batter to make delicate and informative comedy pancakes Then I glaze them in the juice and zest of the most relevant midnight Valencia oranges and douse it all and a fine Dela main de voyage cognac Before prom baying and basting them tables. I deserve for you the James Beard award worthy crepe suzzette That is my nightly monologue, but sometimes just sometimes folks. I wake up in the baggage hold of Greyhound bus. It's being hoisted by the scrap yard claw toward the burn pit. Escape to a nearby abandoned price chopper where I scrounge for old bread scraps and busted open bags of starfruit candies and expired eggs. Chuck it all on a dirty hubcap and slap it over a tire fire before using the legs of a strain, pair of sweatpants and as oven mitts to extract and serve the demented transience poundcake of news that is my segment. Me, Guadalupe!",
            " Folks, if you watched the show and I hope you do, I spent a lot of time right over there. Tiredlessly studying the lineage of the days most important thoroughbred stories and whole-stiner headlines, working with the best trainers, money can buy to rear their comedy offspring with a hand that is stern yet gentle into the triple crown winning equine specimen. That is my nightly monologue, but sometimes, sometimes, folks, I break into an unincorporated veterinary genetics lab and grab whatever test tubes I can find and then under a grow light I got from a discarded chia pet. I mixed the pilfered DNA of a horse and whatever was in a tube labeled Keith Colan extra. Slurrying the concoction with caffeine pills and a microwave red bull, I screamed, sang a prayer to Janice, initiator of human life and God of transformation as a half horse, half man, freak. Seizes to life before me and the hideous collection of loose animal parts and corrupted man tissue that is my segment. Meanwhile!"
        ]
        # fmt: on

        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
        model = model.to(torch_device)

        ds = load_dataset("distil-whisper/meanwhile", "default")["test"]
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))

        num_samples = 8

        audio = ds[:num_samples]["audio"]
        audios = [x["array"] for x in audio]

        decoded_single = []
        for audio in audios:
            inputs = processor(audio, return_tensors="pt", truncation=False, sampling_rate=16_000)
            inputs = inputs.to(device=torch_device)

            result = model.generate(**inputs, return_timestamps=True)
            decoded_single += processor.batch_decode(result, skip_special_tokens=True)

        inputs = processor(
            audios, return_tensors="pt", truncation=False, padding="longest", return_attention_mask=True
        )
        inputs = inputs.to(device=torch_device)

        result = model.generate(**inputs, return_timestamps=True)
        decoded_all = processor.batch_decode(result, skip_special_tokens=True)

        for i in range(num_samples):
            assert decoded_all[i] == decoded_single[i]
            assert decoded_all[i] == EXPECTED_TEXT[i]

    @slow
    def test_whisper_longform_multi_batch_hard_prev_cond(self):
        # fmt: off
        EXPECTED_TEXT = [
            " Folks, if you watch the show, you know I spent a lot of time right over there. Patiently and astutely scrutinizing the boxwood and mahogany chest set of the day's biggest stories, developing the central headline pawns, definitely maneuvering an oh-so-topical night to F6, faming of classic Sicilian, named or variation on the news, all the while seeing eight moves deep and patiently marshalling the latest press releases into a Fisher show's in lip-nitsky attack that culminates in the elegant lethal slow played all pass on checkmate that is my nightly monologue, but sometimes sometimes folks I sometimes I start a little wake-up side down in the monkey bars of a condemned playground on a super fun site, get all hepped up on goofballs, rummage that would discard a tag bag of defective toys, yank out a fistball of disembodied doll limbs, toss them on a stain kid's place mad from a defunked denies, set up a table inside a rusty cargo container down by the warf and challenge toothless drifters to the godless bughouse blitz of tournament that is my segment.",
            " Folks, I spent a lot of time right over there night after night, actually. Carefully selecting for you the day's newsiest, most aerodynamic headlines, stress testing on those topical anti-lock breaks and power steering, painstakingly stitching, leather seating, so soft, it would make JD power and her associates blush. To create the luxury sedan that is my nightly monologue, but sometimes I just sometimes focus. I lurched to consciousness in the back of an abandoned school bus and slapped myself awake with a crusty floor mat. Before using a mouse-bitten timing belt to strap some old plywood to a couple of discarded oil drums, then by the light of a heathen-moon render a gas tank out of an empty big gulp, filled with white claw and de-natured alcohol, then light a match, letter-ripping the dis-mented one-man soapbox derby of news that is my segment.",
            " Ladies and gentlemen, you know, I spent a lot of time right over there, raising the finest hosting news cattle firmly, yet tenderly milking the latest headlines from their jokes, swollen teats, churning the daily stories into the decadent Provincil style triple cream-breed. It is my nightly monologue, but sometimes sometimes I stagger home hungry after being released by the police and root around in the neighbors trash can for an old milk carton scrape out the blooming dairy residue into the remains of a wet cheese rind I won from a rat and a pre-drawn street fight. Put it into discarded paint can to leave it to ferment next to a trash fire than a hunker down in hallucinate while eating the lusteria latent demon custard of news that is my segment.",
            " Folks, you watched this show, you know I spend most of my time right over there, carefully sorting through the days, big stories, and selecting only the most subtle, and unblemished ostrich and crocodile news leather, which I then entrust to artisan graduates of the Ickel Greg Waferandi, who carefully died them in a pallet of bright, zesty shades, and adorn them in the finest most topical inlay work, using hand tools and double magnifying glasses, then assemble them according to now classic and elegant geometry using our signature saddle stitching, and line it with bees, wax, coated linen, and finally attach a mallet hammered strap, perled hardware, and close-shet to create for you the one of a kind hope, kutur, earn-may is burkin bag that is my monologue, but sometimes, sometimes, sometimes. Sometimes, sometimes I wake up in the last car of an abandoned roller coaster at Kony Island, where I'm hiding from the triads, I have some engine lubricants out of a safe way bag and staggered down the shore to tear the sail off a beach sooner than I ripped the coaxial cable out of an RV and elderly couple from Utah, Hank, and Mabel Lovelyfokes, and use it to stitch the sail into a loose pouch like rock sack, and I stole a bag of a garbage truck to the junkyard, where I picked through to the debris for only the broken toys that make me the saddest, until I have loaded for you. The hobo fugitives bug out Bindle of news that is my segment.",
            " You know, folks, I spent a lot of time crafting for you a bespoke playlist of the day's big stories right over there. meticulously selecting the most topical chakra affirming scented candles, using Feng Shui, to perfectly align the joke energy in the exclusive boutique yoga retreat that is my monologue, but sometimes just sometimes, I go to the dumpster behind the waffle house at three in the morning, take off my shirt, cover myself and use fry oil, wrap my hands and some old duct tape I stole from a broken car window, pound a six pack of blueberry hard-seller and a second pill, as I stole from a park damsel, and it's then arm wrestle a raccoon in the back alley vision quest of news that is my segment.",
            " You know, folks, I spend most of my time right over there. Mining the days, biggest, most important stories, collecting the finest, most topical iron or hand hammering it into joke panels, then I craft sheets of bronze and blazing with patterns that tell an epic tale of conquest and glory. Then, using the Germanic tradition press, black process, I place thin sheets of foil against the scenes and by hammering or otherwise applying pressure from the back, I project these scenes into a pair of cheat cards and a face plate, and finally using fluted strips of white alloyed molding I divide the designs into framed panels and hold it all together using bronze rivets to create the beautiful and intimidating Anglo-Saxon battle helm that is my nightly monologue. Sometimes, sometimes, folks. Sometimes, just sometimes, I come to my senses fully naked on the deck of a pirate, beceived, melee, container ship that picked me up floating on the detainees. Then after I sunstroke in juice, realization of the crew of this ship plans to sell me and exchange for a bag of oranges to fight off scurvy, I lead a mutiny using only a PVC pipe in a pool chain that accepting my new role as captain and declaring myself king of the wind arc seas. I grab a dirty muck bucket covered in barnacles and a dornet with the teeth of the vanquished to create the softening wet pirate crown of news that is my segment. I'm going to use the white paper to create the softened white paper to create the softened white paper to create the softened white pirate crown of news that is my segment. Meanwhile.",
            " Folks, if you watch this show, you know I spend most of my time right over there carefully blending for you the day's newsiest, most topical flower eggs, milk and butter. And straining into a fine batter to make delicate and informative comedy pancakes, then I glaze them in the juice and zest of the most relevant midnight valencio oranges. And doubts at all, and I find delimane de voyage cognac, before from bang and basting them tables, I deserve you the James Beard Award worthy creeps to ZET. That is my nightly monologue, but sometimes sometimes folks I wake up in the baggage hole of Greyhound bus, it's being hoisted by the scrapyard claw toward the burn pit. Escape to a nearby abandoned price chopper where I scrounge for old bread scraps, busted open bags of starfruit candies and expired eggs. Chuck it all on a dirty hubcap and slap it over a tire fire before using the legs of a strained pair of sweatpants and as ovenmets to extract and serve the demented transients pound cake of news that is my segment.",
            " Folks, if you watch the show and I hope you do, I spend a lot of time right over there. Tirelessly studying the lineage of the day's most important thoroughbred stories and whole-stiner headlines, working with the best trainers money can buy to rear their comedy offspring with a hand that is stern yet gentle into the triple crown winning equine specimen that is my nightly monologue. But sometimes sometimes folks I break into an unincorporated veterinary genetics lab. And grab whatever test tubes I can find and then under a grow light I got from it a discarded chia pet. I mixed the pill for DNA of a horse and whatever was in a tube labeled Keith Cole and extra. Sloering the concoction with caffeine pills and a microwave bread bowl, I screamed sing a prayer to Janice initiator of human life and God of transformation as a half horse, half man freak, seasons to life before me. And the hideous collection of loose animal parts and corrupted men tissue that is my segment.",
        ]
        # fmt: on

        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
        model = model.to(torch_device)

        ds = load_dataset("distil-whisper/meanwhile", "default")["test"]
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))

        num_samples = 8

        audio = ds[:num_samples]["audio"]
        audios = [x["array"] for x in audio]

        inputs = processor(
            audios, return_tensors="pt", truncation=False, padding="longest", return_attention_mask=True
        )
        inputs = inputs.to(device=torch_device)

        gen_kwargs = {
            "return_timestamps": True,
            "no_speech_threshold": 0.6,
            "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            "compression_ratio_threshold": 1.35,
            "condition_on_prev_tokens": True,
            "logprob_threshold": -1.0,
            "num_beams": 5,
        }

        torch.manual_seed(0)
        result = model.generate(**inputs, **gen_kwargs)
        decoded_all = processor.batch_decode(result, skip_special_tokens=True)

        for i in range(num_samples):
            assert decoded_all[i] == EXPECTED_TEXT[i]

    @slow
    def test_whisper_longform_no_speech_detection(self):
        # fmt: off
        EXPECTED_TEXT = [
            " Folks, if you watch the show, you know, I spent a lot of time right over there. Patiently and astutely scrutinizing the boxwood and mahogany chest set of the day's biggest stories. Developing the central headline pawns, definitely maneuvering and also topical night to F6.",
            " Folks, I spent a lot of time right over there night after night, actually. Carefully selecting for you the day's newsiest, most aerodynamic headlines, stress testing",
            ' Ladies and gentlemen, you know, I spent a lot of time right over there raising the finest Holstein news cattle firmly yet tenderly milking the latest headlines from their joke swollen teats',
            ' Folks, you watched this show, you know I spend most of my time right over there, carefully sorting through the days, big stories, and selecting only the most subtle and unblemished ostrich and crocodile news leather, which I then entrust to artisan graduates of the',
            " You know, folks, I spent a lot of time crafting for you a bespoke playlist of the day's big stories right over there. meticulously selecting the most topical chakra affirming scented candles, using Feng Shui,",
            ' You know, folks, I spend most of my time right over there. Mining the days, biggest, most important stories, collecting the finest, most topical iron or hand hammering it into joke panels, then I craft sheets of bronze and blazing with patterns that tell an epic tale of conquest.',
            " Folks, if you watch this show, you know I spend most of my time right over there, carefully blending for you the day's newsiest, most topical flower eggs, milk and butter. And straining into a fine batter to make delicate and informative comedy pancakes, then I glaze them in the juice and zest of the most...",
            " Folks, if you watch the show and I hope you do, I spent a lot of time right over there. Tirelessly studying the lineage of the day's most important thoroughbred stories and whole-stiner headlines.",
        ]
        # fmt: on

        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
        model = model.to(torch_device)

        ds = load_dataset("distil-whisper/meanwhile", "default")["test"]
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))

        num_samples = 8

        audio = ds[:num_samples]["audio"]
        audios = [x["array"] for x in audio]

        # Make sure the second chunk is silent
        for audio in audios:
            audio[15 * 16000 : 60 * 16000] = 0.0

        inputs = processor(
            audios, return_tensors="pt", truncation=False, padding="longest", return_attention_mask=True
        )
        inputs = inputs.to(device=torch_device)

        gen_kwargs = {
            "return_timestamps": True,
            "no_speech_threshold": 0.2,
            "temperature": (0.0,),
            "compression_ratio_threshold": 1.35,
            "condition_on_prev_tokens": True,
            "logprob_threshold": 0.0,  # Ignore logprob, use only no-speech prob
            "num_beams": 5,
        }

        torch.manual_seed(0)
        result = model.generate(**inputs, **gen_kwargs)
        decoded_all = processor.batch_decode(result, skip_special_tokens=True)

        for i in range(num_samples):
            assert decoded_all[i] == EXPECTED_TEXT[i]


def prepare_whisper_encoder_inputs_dict(config, input_features, head_mask=None):
    if head_mask is None:
        head_mask = torch.ones(config.encoder_layers, config.encoder_attention_heads, device=torch_device)
    return {"input_features": input_features, "head_mask": head_mask}


@require_torch
class WhisperEncoderModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,  # need batch_size != num_hidden layers
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

    def create_and_check_model_forward(self, config, inputs_dict, use_weighted_layer_sum=False):
        config.use_weighted_layer_sum = use_weighted_layer_sum
        model = WhisperForAudioClassification(config=config)
        model.to(torch_device).eval()

        input_features = inputs_dict["input_features"]

        with torch.no_grad():
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

    def test_forward_pass(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_forward(*config_and_inputs)

    def test_forward_pass_weighted_layer_sum(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_forward(*config_and_inputs, use_weighted_layer_sum=True)

    @unittest.skip(reason="Some undefined behavior encountered with tiny versions of this model. Skip for now.")
    def test_cpu_offload(self):
        pass

    @unittest.skip(reason="Some undefined behavior encountered with tiny versions of this model. Skip for now.")
    def test_disk_offload_bin(self):
        pass

    @unittest.skip(reason="Some undefined behavior encountered with tiny versions of this model. Skip for now.")
    def test_disk_offload_safetensors(self):
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

            encoder = model.encoder

            encoder_inputs = {"input_features": inputs["input_features"]}
            del inputs["input_features"]

            if "head_mask" in inputs:
                encoder_inputs["head_mask"] = inputs["head_mask"]
            if "attention_mask" in inputs:
                encoder_inputs["attention_mask"] = inputs["attention_mask"]
            if "output_attentions" in inputs:
                encoder_inputs["output_attentions"] = inputs["output_attentions"]

            with torch.no_grad():
                inputs["encoder_outputs"] = encoder(**encoder_inputs)
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
                fx_inputs = {k: np.array(v.to("cpu")) for k, v in pt_inputs.items() if torch.is_tensor(v)}

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
                fx_inputs = {k: np.array(v.to("cpu")) for k, v in pt_inputs.items() if torch.is_tensor(v)}

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


class WhisperStandaloneDecoderModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,  # need batch_size != num_hidden layers
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

        decoder_input_ids = torch.tensor(
            self.batch_size * [[self.decoder_start_token_id, 3, 3, 7, 2]], device=torch_device
        )

        config = self.get_config()
        config.is_encoder_decoder = False
        inputs_dict = prepare_whisper_inputs_dict(
            config,
            attention_mask=None,
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
        )

        inputs_dict.pop("input_features")
        inputs_dict.pop("head_mask")
        inputs_dict.pop("decoder_head_mask")
        inputs_dict.pop("cross_attn_head_mask")

        inputs_dict["attention_mask"] = inputs_dict.pop("decoder_attention_mask")
        inputs_dict["input_ids"] = inputs_dict.pop("decoder_input_ids")
        return config, inputs_dict

    @property
    def encoder_seq_length(self):
        return 5

    @property
    def seq_length(self):
        return 5

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

        inputs_dict["input_ids"][:, -1] = self.pad_token_id

        return config, inputs_dict

    def prepare_config_and_inputs_for_decoder(self):
        config, input_features = self.prepare_config_and_inputs()
        input_ids = input_features["input_ids"]
        encoder_hidden_states = floats_tensor([self.batch_size, self.decoder_seq_length, self.hidden_size])

        return (config, input_ids, encoder_hidden_states)

    def create_and_check_decoder_model_past(self, config, input_ids):
        config.use_cache = True
        model = WhisperDecoder(config=config).to(torch_device).eval()
        # first forward pass
        outputs = model(input_ids, use_cache=True)
        outputs_use_cache_conf = model(input_ids)
        outputs_no_past = model(input_ids, use_cache=False)

        self.parent.assertTrue(len(outputs) == len(outputs_use_cache_conf))
        self.parent.assertTrue(len(outputs) == len(outputs_no_past) + 1)

        past_key_values = outputs["past_key_values"]

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size)

        # append to next input_ids and
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)

        output_from_no_past = model(next_input_ids)["last_hidden_state"]
        output_from_past = model(next_tokens, past_key_values=past_key_values)["last_hidden_state"]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, next_input_ids.shape[-1] - 1, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, 0, random_slice_idx].detach()

        # test that outputs are equal for slice
        assert torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3)

    def create_and_check_decoder_model_attention_mask_past(self, config, input_ids):
        model = WhisperDecoder(config=config).to(torch_device).eval()

        # create attention mask
        attn_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)

        half_seq_length = input_ids.shape[-1] // 2
        attn_mask[:, half_seq_length:] = 0

        # first forward pass
        past_key_values = model(input_ids, attention_mask=attn_mask, use_cache=True)["past_key_values"]

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size)

        # change a random masked slice from input_ids
        random_seq_idx_to_change = ids_tensor((1,), half_seq_length).item() + 1
        random_other_next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size).squeeze(-1)
        input_ids[:, -random_seq_idx_to_change] = random_other_next_tokens

        # append to next input_ids and attn_mask
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        attn_mask = torch.cat(
            [attn_mask, torch.ones((attn_mask.shape[0], 1), dtype=torch.long, device=torch_device)],
            dim=1,
        )

        # get two different outputs
        output_from_no_past = model(next_input_ids, attention_mask=attn_mask)["last_hidden_state"]
        output_from_past = model(next_tokens, attention_mask=attn_mask, past_key_values=past_key_values)[
            "last_hidden_state"
        ]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, next_input_ids.shape[-1] - 1, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, 0, random_slice_idx].detach()

        # test that outputs are equal for slice
        assert torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3)


@require_torch
class WhisperStandaloneDecoderModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (WhisperDecoder, WhisperForCausalLM) if is_torch_available() else ()
    all_generative_model_classes = (WhisperForCausalLM,) if is_torch_available() else ()
    fx_comptatible = False
    test_pruning = False
    is_encoder_decoder = False
    test_missing_keys = False

    def setUp(self):
        self.model_tester = WhisperStandaloneDecoderModelTester(self, is_training=False)
        self.config_tester = ConfigTester(self, config_class=WhisperConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_decoder_model_past(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        config, inputs_dict = config_and_inputs

        self.model_tester.create_and_check_decoder_model_past(config=config, input_ids=inputs_dict["input_ids"])

    def test_decoder_model_attn_mask_past(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        config, inputs_dict = config_and_inputs

        self.model_tester.create_and_check_decoder_model_attention_mask_past(
            config=config, input_ids=inputs_dict["input_ids"]
        )

    @unittest.skip("Generate needs input ids")
    def test_generate_without_input_ids(self):
        # generate only works with input ids for whisper
        pass

    @unittest.skip("Decoder can't keep attention grads")
    def test_retain_grad_hidden_states_attentions(self):
        # decoder cannot keep gradients
        return

    @unittest.skip("The model doesn't support fast init from base")
    def test_save_load_fast_init_from_base(self):
        pass
