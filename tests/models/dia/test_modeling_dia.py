# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Dia model."""

import copy
import inspect
import os
import random
import tempfile
import unittest
from typing import Tuple

import pytest
from parameterized import parameterized

from transformers.masking_utils import AttentionMask
from transformers.models.dia import DiaConfig, DiaDecoderConfig, DiaEncoderConfig
from transformers.testing_utils import (
    is_flaky,
    require_flash_attn,
    require_torch,
    require_torch_fp16,
    require_torch_gpu,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available, is_torchaudio_available
from transformers.utils.import_utils import is_datasets_available

from ...generation.test_utils import GenerationTesterMixin
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_datasets_available():
    pass

if is_torch_available():
    import torch

    from transformers import (
        DiaForConditionalGeneration,
        DiaModel,
    )
    from transformers.generation import (
        GenerateEncoderDecoderOutput,
    )
    from transformers.generation.logits_process import LogitsProcessor
    from transformers.models.dia.modeling_dia import DiaDecoder, DiaEncoder

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
    pass


def prepare_dia_inputs_dict(
    encoder_input_ids,
    decoder_input_ids,
):
    return {
        "encoder_input_ids": encoder_input_ids,
        "decoder_input_ids": decoder_input_ids,
    }


@require_torch
class DiaModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,  # need batch_size != num_hidden_layers
        # Encoder specific
        encoder_max_length=60,
        encoder_seq_length=60,
        encoder_vocab_size=100,
        encoder_hidden_layers=2,
        encoder_hidden_size=16,
        encoder_num_attention_heads=4,
        encoder_head_dim=64,
        # Decoder specific
        decoder_max_length=40,
        decoder_hidden_layers=2,
        decoder_hidden_size=32,  # Typically larger than encoder
        decoder_num_attention_heads=4,
        decoder_head_dim=64,
        decoder_cross_attention_heads=4,
        decoder_cross_head_dim=64,
        # Common
        vocab_size=200,
        hidden_act="silu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        is_training=False,
        use_labels=False,
        # Special tokens
        eos_token_id=98,
        pad_token_id=99,
        bos_token_id=100,
        # Other
        seq_length=30,
        num_conv_layers=1,
        suppress_tokens=None,
        delay_pattern=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        # Set default delay pattern if not provided
        self.delay_pattern = (
            delay_pattern if delay_pattern is not None else [0, 1, 2, 3, 4, 5, 6, 7, 8]
        )  # 9 channels by default
        # Encoder
        self.encoder_max_length = encoder_max_length
        self.encoder_seq_length = encoder_seq_length
        self.encoder_vocab_size = encoder_vocab_size
        self.encoder_hidden_layers = encoder_hidden_layers
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_num_attention_heads = encoder_num_attention_heads
        self.encoder_head_dim = encoder_head_dim
        # Decoder
        self.decoder_max_length = decoder_max_length
        self.decoder_hidden_layers = decoder_hidden_layers
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_num_attention_heads = decoder_num_attention_heads
        self.decoder_head_dim = decoder_head_dim
        self.decoder_cross_attention_heads = decoder_cross_attention_heads
        self.decoder_cross_head_dim = decoder_cross_head_dim
        # Common
        self.vocab_size = vocab_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.is_training = is_training
        self.use_labels = use_labels
        # Special tokens
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        # Other
        self.seq_length = seq_length
        self.num_conv_layers = num_conv_layers
        self.suppress_tokens = suppress_tokens

        assert eos_token_id < pad_token_id and eos_token_id < bos_token_id

    def prepare_config_and_inputs(self) -> Tuple[DiaConfig, dict]:
        encoder_input_ids = ids_tensor([self.batch_size, self.encoder_seq_length], self.encoder_vocab_size)
        decoder_input_ids = torch.tensor(
            self.batch_size * [[self.bos_token_id] * len(self.delay_pattern)], device=torch_device
        )

        config = self.get_config()
        inputs_dict = prepare_dia_inputs_dict(
            encoder_input_ids=encoder_input_ids,
            decoder_input_ids=decoder_input_ids,
        )
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self) -> Tuple[DiaConfig, dict]:
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def prepare_config_and_inputs_for_generate(self) -> Tuple[DiaConfig, dict]:
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def get_config(self):
        encoder_config = DiaEncoderConfig(
            max_length=self.encoder_max_length,
            num_hidden_layers=self.encoder_hidden_layers,
            hidden_size=self.encoder_hidden_size,
            num_attention_heads=self.encoder_num_attention_heads,
            num_key_value_heads=self.encoder_num_attention_heads,  # Same as num_attention_heads for testing
            head_dim=self.encoder_head_dim,
            intermediate_size=self.encoder_hidden_size * 4,  # Standard ratio
            vocab_size=self.encoder_vocab_size,
            dropout=self.hidden_dropout_prob,
            hidden_act=self.hidden_act,
        )

        # Number of channels must match delay pattern length
        num_channels = len(self.delay_pattern)

        decoder_config = DiaDecoderConfig(
            max_length=self.decoder_max_length,
            num_hidden_layers=self.decoder_hidden_layers,
            hidden_size=self.decoder_hidden_size,
            intermediate_size=self.decoder_hidden_size * 4,  # Standard ratio
            num_attention_heads=self.decoder_num_attention_heads,
            num_key_value_heads=max(1, self.decoder_num_attention_heads // 4),  # Grouped attention
            head_dim=self.decoder_head_dim,
            cross_num_attention_heads=self.decoder_cross_attention_heads,
            cross_head_dim=self.decoder_cross_head_dim,
            cross_num_key_value_heads=max(1, self.decoder_cross_attention_heads // 2),
            cross_hidden_size=self.encoder_hidden_size,  # Match encoder hidden size
            vocab_size=self.vocab_size,
            dropout=self.hidden_dropout_prob,
            hidden_act=self.hidden_act,
            num_channels=num_channels,  # Must match delay pattern length
        )

        config = DiaConfig(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            delay_pattern=self.delay_pattern,
        )

        return config

    def get_subsampled_output_lengths(self, input_lengths):
        """
        Computes the output length of the convolutional layers
        """

        for i in range(self.num_conv_layers):
            input_lengths = (input_lengths - 1) // 2 + 1

        return input_lengths

    def create_and_check_model_forward(self, config, inputs_dict, freeze_encoder=False):
        model = DiaModel(config=config).to(torch_device).eval()

        if freeze_encoder:
            model.freeze_encoder()

        encoder_input_ids = inputs_dict["encoder_input_ids"]
        decoder_input_ids = inputs_dict["decoder_input_ids"]

        # first forward pass
        last_hidden_state = model(
            encoder_input_ids=encoder_input_ids, decoder_input_ids=decoder_input_ids
        ).last_hidden_state

        self.parent.assertTrue(last_hidden_state.shape, (13, 7, 16))

    def create_and_check_decoder_model_past_large_inputs(self, config, inputs_dict):
        model = DiaModel(config=config).get_decoder().to(torch_device).eval()
        encoder_input_ids = inputs_dict["encoder_input_ids"]
        decoder_input_ids = inputs_dict["decoder_input_ids"]

        # first forward pass
        outputs = model(encoder_input_ids=encoder_input_ids, decoder_input_ids=decoder_input_ids, use_cache=True)

        output, past_key_values = outputs.to_tuple()

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size).clamp(2)
        next_attn_mask = ids_tensor((self.batch_size, 3), 2)

        # append to next input_ids and
        next_input_ids = torch.cat([decoder_input_ids, next_tokens], dim=-1)
        next_attention_mask = torch.cat([AttentionMask, next_attn_mask], dim=-1)

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
        model = DiaModel(config=config).to(torch_device).eval()
        outputs = model(**inputs_dict)

        encoder_last_hidden_state = outputs.encoder_last_hidden_state
        last_hidden_state = outputs.last_hidden_state

        with tempfile.TemporaryDirectory() as tmpdirname:
            encoder = model.get_encoder()
            encoder.save_pretrained(tmpdirname)
            encoder = DiaEncoder.from_pretrained(tmpdirname).to(torch_device)

        encoder_last_hidden_state_2 = encoder(inputs_dict["input_features"])[0]

        self.parent.assertTrue((encoder_last_hidden_state_2 - encoder_last_hidden_state).abs().max().item() < 1e-3)

        with tempfile.TemporaryDirectory() as tmpdirname:
            decoder = model.get_decoder()
            decoder.save_pretrained(tmpdirname)
            decoder = DiaDecoder.from_pretrained(tmpdirname).to(torch_device)

        last_hidden_state_2 = decoder(
            input_ids=inputs_dict["decoder_input_ids"],
            attention_mask=inputs_dict["decoder_attention_mask"],
            encoder_hidden_states=encoder_last_hidden_state,
        )[0]

        self.parent.assertTrue((last_hidden_state_2 - last_hidden_state).abs().max().item() < 1e-3)


@require_torch
class DiaModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (DiaForConditionalGeneration,) if is_torch_available() else ()
    is_encoder_decoder = True
    fx_compatible = False
    test_pruning = False
    test_missing_keys = False
    # Needs higher percentages after model tester's vocab_size is changed to 200 (PR #21222)
    # `0.5` is for `test_disk_offload` (which also works for `test_model_parallelism`)
    model_split_percents = [0.5, 0.8, 0.9]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_tester = DiaModelTester(parent=self)

    # training is not supported yet
    @unittest.skip(reason="Training is not supported yet")
    def test_training(self):
        pass

    @unittest.skip(reason="Training is not supported yet")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="")
    def test_generate_with_head_masking(self):
        pass

    @parameterized.expand([("offloaded",)])
    @pytest.mark.generate
    @unittest.skip(reason="Dia doesnt work with offloaded cache implementation yet")
    def test_offloaded_cache_implementation(self, cache_implementation):
        pass

    @require_torch_fp16
    def test_generate_fp16(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs()
        config.max_target_positions = 400
        input_features = input_dict["input_features"]
        model = DiaForConditionalGeneration(config).eval().to(torch_device)
        input_features = input_features.half()
        model.half()
        model.generate(input_features)
        model.generate(input_features, num_beams=4, do_sample=True, early_stopping=False, num_return_sequences=3)

    def test_generate_language(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs()
        input_features = input_dict["input_features"]
        model = DiaForConditionalGeneration(config).to(torch_device)
        # Hack to keep the test fast and not require downloading a model with a generation_config
        model.generation_config.__setattr__("lang_to_id", {"<|en|>": 1})
        model.generation_config.__setattr__("task_to_id", {"transcribe": 2})

        # test language code
        model.generate(input_features, language="en")
        # test language token
        model.generate(input_features, language="<|en|>")
        # test language name
        model.generate(input_features, language="English")
        # test language code list
        model.generate(input_features, language=["en"] * input_features.shape[0])
        # test language token list
        model.generate(input_features, language=["<|en|>"] * input_features.shape[0])
        # test language name list
        model.generate(input_features, language=["English"] * input_features.shape[0])
        # test list of the wrong length
        with self.assertRaises(ValueError):
            model.generate(input_features, language=["en"] * (input_features.shape[0] + 1))

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
            self.skipTest(reason="test_resize_embeddings is False")

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
            self.skipTest(reason="test_resize_embeddings is False")

        original_config.tie_word_embeddings = False

        # if model cannot untied embeddings -> leave test
        if original_config.tie_word_embeddings:
            self.skipTest(reason="Model cannot untie embeddings")

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

    @unittest.skip(reason="")
    def test_generate_without_input_ids(self):
        pass

    @require_flash_attn
    @require_torch_gpu
    @pytest.mark.flash_attn_test
    @slow
    def test_flash_attn_2_inference_equivalence(self):
        import torch

        for model_class in self.all_model_classes:
            if not model_class._supports_flash_attn_2:
                self.skipTest(reason="Model does not support Flash Attention 2")

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

                # dia FA2 needs very high tolerance
                torch.testing.assert_close(logits_fa, logits, rtol=4e-1, atol=4e-1)

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
                self.skipTest(reason="Model does not support flash_attention_2")

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

                # dia FA2 needs very high tolerance
                torch.testing.assert_close(logits_fa, logits, rtol=4e-1, atol=4e-1)

                other_inputs = {
                    "decoder_input_ids": decoder_input_ids,
                    "decoder_attention_mask": decoder_attention_mask,
                    "output_hidden_states": True,
                }

                outputs = model(dummy_input, **other_inputs)
                outputs_fa = model_fa(dummy_input, **other_inputs)

                logits = outputs.decoder_hidden_states[-1]
                logits_fa = outputs_fa.decoder_hidden_states[-1]

                # dia FA2 needs very high tolerance
                torch.testing.assert_close(logits_fa[:, -2:], logits[:, -2:], rtol=4e-1, atol=4e-1)

    def _create_and_check_torchscript(self, config, inputs_dict):
        if not self.test_torchscript:
            self.skipTest(reason="test_torchscript is set to False")

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

    def test_generate_with_prompt_ids_max_length(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.max_target_positions = 7

        model = DiaForConditionalGeneration(config).eval().to(torch_device)
        input_features = input_dict["input_features"]
        decoder_input_ids = torch.arange(5).to(torch_device)
        prompt_ids = decoder_input_ids[:4]
        max_new_tokens = 8

        with self.assertRaisesRegex(
            ValueError,
            f"The length of `decoder_input_ids`, including special start tokens, prompt tokens, and previous tokens, is {decoder_input_ids.shape[-1]}, "
            f" and `max_new_tokens` is {max_new_tokens}. Thus, the combined length of "
            f"`decoder_input_ids` and `max_new_tokens` is: {max_new_tokens + decoder_input_ids.shape[-1]}. This exceeds the "
            f"`max_target_positions` of the Dia model: {config.max_target_positions}. "
            "You should either reduce the length of your prompt, or reduce the value of `max_new_tokens`, "
            f"so that their combined length is less than {config.max_target_positions}.",
        ):
            model.generate(input_features, max_new_tokens=max_new_tokens, prompt_ids=prompt_ids)

        model.generate(input_features, max_new_tokens=1, prompt_ids=prompt_ids)

    def test_generate_longform_with_prompt_ids(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = DiaForConditionalGeneration(config).eval().to(torch_device)

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
                    self.assertTrue(not any(i in row for i in model.generation_config.suppress_tokens))

    def _check_longform_generate_single_batch(self, condition_on_prev_tokens):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()

        model = DiaForConditionalGeneration(config).eval().to(torch_device)
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
            self.assertTrue(segment["start"] <= segment["end"], "start has to be smaller equal end")
            self.assertTrue(
                any(s > timestamp_begin for s in segment["tokens"][1:]),
                f"At least one segment token should be a timestamp token, but not first., {segment['tokens']}",
            )
            self.assertTrue(
                segment["tokens"].shape[-1] <= max_length,
                "make sure that no segment is larger than max generation length",
            )

    def test_longform_generate_single_batch(self):
        self._check_longform_generate_single_batch(condition_on_prev_tokens=False)

    def test_longform_generate_single_batch_cond_prev(self):
        self._check_longform_generate_single_batch(condition_on_prev_tokens=True)

    def _check_longform_generate_multi_batch(self, condition_on_prev_tokens):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()

        model = DiaForConditionalGeneration(config).eval().to(torch_device)
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
        self.assertEqual(tokens_2.tolist(), tokens[: tokens_2.shape[-1]].tolist())

        for seg1, seg2 in zip(segments_2, segments):
            self.assertEqual(seg1["start"], seg2["start"])
            self.assertEqual(seg1["end"], seg2["end"])
            self.assertEqual(seg1["tokens"].tolist(), seg2["tokens"].tolist())

    def test_longform_generate_multi_batch(self):
        self._check_longform_generate_multi_batch(condition_on_prev_tokens=False)

    def test_longform_generate_multi_batch_cond_prev(self):
        self._check_longform_generate_multi_batch(condition_on_prev_tokens=True)

    @is_flaky()  # TODO (joao, sanchit): fails ~9% of the times. Does the original test have the same issue?
    def test_custom_4d_attention_mask(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = DiaForConditionalGeneration(config).to(device=torch_device, dtype=torch.float32)
        model.eval()

        (
            input_ids,
            position_ids,
            input_ids_shared_prefix,
            mask_shared_prefix,
            position_ids_shared_prefix,
        ) = self._get_custom_4d_mask_test_data()

        with torch.no_grad():
            logits = model.forward(
                decoder_input_ids=input_ids,
                input_features=input_dict["input_features"],
                decoder_position_ids=position_ids,
            ).logits
            # logits.shape == torch.Size([3, 4, ...])

            logits_shared_prefix = model(
                decoder_input_ids=input_ids_shared_prefix,
                input_features=input_dict["input_features"],
                decoder_attention_mask=mask_shared_prefix,
                decoder_position_ids=position_ids_shared_prefix,
            )[0]
            # logits_shared_prefix.shape == torch.Size([1, 6, ...])

        out_last_tokens = logits[:, -1, :]  # last tokens in each batch line
        out_shared_prefix_last_tokens = logits_shared_prefix[0, -3:, :]  # last three tokens

        # comparing softmax-normalized logits:
        normalized_0 = torch.nn.functional.softmax(out_last_tokens)
        normalized_1 = torch.nn.functional.softmax(out_shared_prefix_last_tokens)
        torch.testing.assert_close(normalized_0, normalized_1, rtol=1e-3, atol=1e-4)

    @parameterized.expand([(True,), (False,)])
    def test_generate_output_type(self, return_dict_in_generate):
        expected_output_type = GenerateEncoderDecoderOutput if return_dict_in_generate else torch.Tensor
        for model_class in self.all_generative_model_classes:
            config, inputs = self.model_tester.prepare_config_and_inputs()
            model = model_class(config).to(torch_device).eval()

            # short-form generation without fallback
            pred_ids = model.generate(**inputs, return_dict_in_generate=return_dict_in_generate)
            self.assertIsInstance(pred_ids, expected_output_type)

            # short-form generation with fallback
            pred_ids = model.generate(
                **inputs,
                logprob_threshold=-1.0,
                temperature=[0.0, 0.1],
                return_dict_in_generate=return_dict_in_generate,
            )
            self.assertIsInstance(pred_ids, expected_output_type)

    def test_labels_sequence_max_length_correct(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_generative_model_classes:
            input_features = input_dict["input_features"]

            labels_length = config.max_target_positions
            labels = torch.ones(1, labels_length, dtype=torch.int64).to(torch_device)

            model = model_class(config).to(torch_device)
            model(input_features=input_features, labels=labels)

    def test_labels_sequence_max_length_correct_after_changing_config(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_generative_model_classes:
            input_features = input_dict["input_features"]

            config.max_target_positions += 100

            labels_length = config.max_target_positions
            labels = torch.ones(1, labels_length, dtype=torch.int64).to(torch_device)

            model = model_class(config).to(torch_device)
            model(input_features=input_features, labels=labels)

    def test_labels_sequence_max_length_error(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_generative_model_classes:
            input_features = input_dict["input_features"]

            labels_length = config.max_target_positions + 1
            labels = torch.ones(1, labels_length, dtype=torch.int64).to(torch_device)

            model = model_class(config).to(torch_device)
            with self.assertRaises(ValueError):
                model(input_features=input_features, labels=labels)

    def test_labels_sequence_max_length_error_after_changing_config(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_generative_model_classes:
            model = model_class(config).to(torch_device)
            input_features = input_dict["input_features"]

            labels_length = config.max_target_positions + 1
            labels = torch.ones(1, labels_length, dtype=torch.int64).to(torch_device)

            new_max_length = config.max_target_positions + 100
            model.config.max_length = new_max_length
            model.generation_config.max_length = new_max_length
            config.max_target_positions = new_max_length

            with self.assertRaises(ValueError):
                model(input_features=input_features, labels=labels)

    # TODO (joao, eustache): fix me :) The model is not returning a `Cache` by default
    @unittest.skip(reason="Dia's custom generate is not consistent regarding the cache return types")
    def test_generate_compile_model_forward(self):
        pass

    # TODO (joao, eustache): fix me :)
    @unittest.skip(reason="A CUDA exception is thrown when storing extra outputs")
    def test_generate_compilation_all_outputs(self):
        pass
