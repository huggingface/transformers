# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch T5Gemma model."""

import copy
import inspect
import unittest

import pytest
from parameterized import parameterized

from transformers import T5GemmaConfig, T5GemmaModuleConfig, is_torch_available
from transformers.testing_utils import (
    require_torch,
    require_torch_accelerator,
    require_torch_gpu,
    require_torch_sdpa,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    import torch.nn.functional as F

    from transformers import (
        T5GemmaEncoderModel,
        T5GemmaForConditionalGeneration,
        T5GemmaForSequenceClassification,
        T5GemmaForTokenClassification,
        T5GemmaModel,
    )
    from transformers.cache_utils import Cache


class T5GemmaModelTester:
    config_class = T5GemmaConfig
    module_config_class = T5GemmaModuleConfig

    if is_torch_available():
        model_class = T5GemmaModel
        for_causal_lm_class = T5GemmaForConditionalGeneration
        for_sequence_class = T5GemmaForSequenceClassification
        for_token_class = T5GemmaForTokenClassification

    def __init__(
        self,
        parent,
        batch_size=13,
        is_training=True,
        use_attention_mask=True,
        use_labels=True,
        vocab_size=99,
        # decoder-specific
        seq_length=7,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=37,
        # encoder-specific
        encoder_seq_length=7,
        encoder_hidden_size=32,
        encoder_num_hidden_layers=2,
        encoder_num_attention_heads=4,
        encoder_num_key_value_heads=2,
        encoder_intermediate_size=37,
        # common
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        scope=None,
        # special ids
        eos_token_id=1,
        pad_token_id=0,
        bos_token_id=2,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.is_training = is_training
        self.use_attention_mask = use_attention_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        # decoder
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        # encoder
        self.encoder_seq_length = encoder_seq_length
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_num_hidden_layers = encoder_num_hidden_layers
        self.encoder_num_attention_heads = encoder_num_attention_heads
        self.encoder_num_key_value_heads = encoder_num_key_value_heads
        self.encoder_intermediate_size = encoder_intermediate_size
        # common
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope
        self.head_dim = self.hidden_size // self.num_attention_heads
        # assume encoder and decoder have the same head dimension.
        assert self.head_dim == self.encoder_hidden_size // self.encoder_num_attention_heads
        # special ids
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        # assume the number of attention heads are the same across encoder and decoder
        # only used for generation testing purpose.
        assert self.num_attention_heads == self.encoder_num_attention_heads

    def get_encoder_config(self):
        return self.module_config_class(
            vocab_size=self.vocab_size,
            hidden_size=self.encoder_hidden_size,
            num_hidden_layers=self.encoder_num_hidden_layers,
            num_attention_heads=self.encoder_num_attention_heads,
            num_key_value_heads=self.encoder_num_key_value_heads,
            intermediate_size=self.encoder_intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=False,
            initializer_range=self.initializer_range,
            head_dim=self.head_dim,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
        )

    def get_decoder_config(self):
        return self.module_config_class(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            cross_attention_hidden_size=self.encoder_hidden_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=True,
            initializer_range=self.initializer_range,
            head_dim=self.head_dim,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
        )

    def get_config(self, is_encoder_decoder=True):
        return self.config_class(
            encoder=self.get_encoder_config(),
            decoder=self.get_decoder_config(),
            is_encoder_decoder=is_encoder_decoder,
            # Used for generation test.
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
        )

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.encoder_seq_length], self.vocab_size)
        decoder_input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        # Remove BOS symbols from inputs.
        input_ids = torch.where(input_ids == self.bos_token_id, 42, input_ids)
        decoder_input_ids = torch.where(decoder_input_ids == self.bos_token_id, 42, decoder_input_ids)

        attention_mask = None
        decoder_attention_mask = None
        if self.use_attention_mask:
            attention_mask = ids_tensor([self.batch_size, self.encoder_seq_length], vocab_size=2)
            decoder_attention_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

        lm_labels = None
        if self.use_labels:
            lm_labels = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        config = self.get_config()

        return (
            config,
            input_ids,
            decoder_input_ids,
            attention_mask,
            decoder_attention_mask,
            lm_labels,
        )

    # Copied from tests.models.t5.test_modeling_t5.T5ModelTester.prepare_config_and_inputs_for_common
    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            decoder_input_ids,
            attention_mask,
            decoder_attention_mask,
            lm_labels,
        ) = config_and_inputs

        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
        }
        return config, inputs_dict

    def create_and_check_model(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = self.model_class(config=config).to(torch_device).eval()

        result = model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )

        decoder_output = result.last_hidden_state
        decoder_past = result.past_key_values
        encoder_output = result.encoder_last_hidden_state

        self.parent.assertEqual(
            encoder_output.size(), (self.batch_size, self.encoder_seq_length, self.encoder_hidden_size)
        )
        self.parent.assertEqual(decoder_output.size(), (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertIsNotNone(decoder_past)
        self.parent.assertEqual(len(decoder_past.self_attention_cache), config.decoder.num_hidden_layers)
        self.parent.assertEqual(len(decoder_past.cross_attention_cache.key_cache), config.decoder.num_hidden_layers)

    def check_prepare_lm_labels_via_shift_left(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = self.model_class(config=config).to(torch_device).eval()

        # _shift_right should be called on labels
        shifted_labels = model._shift_right(lm_labels)

        # first token should be decoder_start_token_id
        self.parent.assertTrue(torch.all(shifted_labels[:, 0] == config.decoder.bos_token_id))

        # the rest should be the labels shifted by one, with -100 replaced by pad_token_id
        labels_without_ignore_index = lm_labels.masked_fill(lm_labels == -100, config.decoder.pad_token_id)
        self.parent.assertTrue(torch.all(shifted_labels[:, 1:] == labels_without_ignore_index[:, :-1]))

    def create_and_check_with_lm_head(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = self.for_causal_lm_class(config=config).to(torch_device).eval()
        outputs = model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )
        self.parent.assertEqual(len(outputs), 4)
        self.parent.assertEqual(outputs["logits"].size(), (self.batch_size, self.seq_length, self.vocab_size))
        self.parent.assertEqual(outputs["loss"].size(), ())

    def create_and_check_with_sequence_classification_head(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        labels = torch.tensor([1] * self.batch_size, dtype=torch.long, device=torch_device)
        model = self.for_sequence_class(config=config).to(torch_device).eval()
        outputs = model(
            input_ids=input_ids,
            decoder_input_ids=input_ids,
            labels=labels,
        )
        self.parent.assertEqual(outputs["logits"].size(), (self.batch_size, config.num_labels))
        self.parent.assertEqual(outputs["loss"].size(), ())

    def create_and_check_encoderonly_for_sequence_classification_head(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
        is_encoder_decoder,
    ):
        labels = torch.tensor([1] * self.batch_size, dtype=torch.long, device=torch_device)
        model = self.for_sequence_class(config=config, is_encoder_decoder=is_encoder_decoder)
        model = model.to(torch_device).eval()
        outputs = model(
            input_ids=input_ids,
            decoder_input_ids=input_ids,
            labels=labels,
        )

        self.parent.assertEqual(outputs["logits"].size(), (self.batch_size, config.num_labels))
        self.parent.assertEqual(outputs["loss"].size(), ())

    def create_and_check_encoderonly_for_token_classification_head(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
        is_encoder_decoder,
    ):
        labels = torch.tensor([1] * self.seq_length * self.batch_size, dtype=torch.long, device=torch_device)
        model = self.for_token_class(config=config, is_encoder_decoder=is_encoder_decoder)
        model = model.to(torch_device).eval()
        outputs = model(
            input_ids=input_ids,
            decoder_input_ids=input_ids,
            labels=labels,
        )

        self.parent.assertEqual(outputs["logits"].size(), (self.batch_size, self.seq_length, config.num_labels))
        self.parent.assertEqual(outputs["loss"].size(), ())

    def create_and_check_decoder_model_past(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = self.model_class(config=config).get_decoder().to(torch_device).eval()
        encoder_hidden_states = torch.ones(
            (self.batch_size, self.encoder_seq_length, self.encoder_hidden_size), dtype=torch.float32
        ).to(torch_device)

        # first forward pass
        outputs = model(input_ids, encoder_hidden_states=encoder_hidden_states, use_cache=True)
        outputs_use_cache_conf = model(input_ids, encoder_hidden_states=encoder_hidden_states)
        outputs_no_past = model(input_ids, encoder_hidden_states=encoder_hidden_states, use_cache=False)

        self.parent.assertTrue(len(outputs) == len(outputs_use_cache_conf))
        self.parent.assertTrue(len(outputs) == len(outputs_no_past) + 1)

        output, past_key_values = outputs.to_tuple()

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size)

        # append to next input_ids and
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)

        output_from_no_past = model(next_input_ids, encoder_hidden_states=encoder_hidden_states)["last_hidden_state"]
        output_from_past = model(
            next_tokens, encoder_hidden_states=encoder_hidden_states, past_key_values=past_key_values
        )["last_hidden_state"]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -1, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, 0, random_slice_idx].detach()

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def create_and_check_decoder_model_attention_mask_past(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = self.model_class(config=config).get_decoder().to(torch_device).eval()
        encoder_hidden_states = torch.ones(
            (self.batch_size, self.encoder_seq_length, self.encoder_hidden_size), dtype=torch.float32
        ).to(torch_device)

        # create attention mask
        attn_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)

        half_seq_length = input_ids.shape[-1] // 2
        attn_mask[:, half_seq_length:] = 0

        # first forward pass
        output, past_key_values = model(
            input_ids, encoder_hidden_states=encoder_hidden_states, attention_mask=attn_mask, use_cache=True
        ).to_tuple()

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
        output_from_no_past = model(
            next_input_ids, encoder_hidden_states=encoder_hidden_states, attention_mask=attn_mask
        )["last_hidden_state"]
        output_from_past = model(
            next_tokens,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
            attention_mask=attn_mask,
        )["last_hidden_state"]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -1, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, 0, random_slice_idx].detach()

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def create_and_check_decoder_model_past_large_inputs(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = self.model_class(config=config).get_decoder().to(torch_device).eval()
        encoder_hidden_states = torch.ones(
            (self.batch_size, self.encoder_seq_length, self.encoder_hidden_size), dtype=torch.float32
        ).to(torch_device)

        # first forward pass
        outputs = model(
            input_ids, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask, use_cache=True
        )

        output, past_key_values = outputs.to_tuple()

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_mask = ids_tensor((self.batch_size, 3), vocab_size=2)

        # append to next input_ids and
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        next_attention_mask = torch.cat([attention_mask, next_mask], dim=-1)

        output_from_no_past = model(
            next_input_ids, encoder_hidden_states=encoder_hidden_states, attention_mask=next_attention_mask
        )["last_hidden_state"]
        output_from_past = model(
            next_tokens,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=next_attention_mask,
            past_key_values=past_key_values,
        )["last_hidden_state"]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def create_and_check_generate_with_past_key_values(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = self.for_causal_lm_class(config=config).to(torch_device).eval()
        torch.manual_seed(0)
        output_without_past_cache = model.generate(
            input_ids[:1], num_beams=2, max_length=5, do_sample=True, use_cache=False
        )
        torch.manual_seed(0)
        output_with_past_cache = model.generate(input_ids[:1], num_beams=2, max_length=5, do_sample=True)
        self.parent.assertTrue(torch.all(output_with_past_cache == output_without_past_cache))

    def create_and_check_model_fp16_forward(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = self.model_class(config=config).to(torch_device).half().eval()
        output = model(input_ids, decoder_input_ids=input_ids, attention_mask=attention_mask)["last_hidden_state"]
        self.parent.assertFalse(torch.isnan(output).any().item())


@require_torch
class T5GemmaModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            T5GemmaModel,
            T5GemmaForConditionalGeneration,
            T5GemmaForSequenceClassification,
            T5GemmaForTokenClassification,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": T5GemmaModel,
            "summarization": T5GemmaForConditionalGeneration,
            "text-classification": T5GemmaForSequenceClassification,
            "text2text-generation": T5GemmaForConditionalGeneration,
            "translation": T5GemmaForConditionalGeneration,
            "zero-shot": T5GemmaForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )

    test_headmasking = False
    test_pruning = False
    _is_stateful = True
    is_encoder_decoder = True
    model_split_percents = [0.5, 0.6]

    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = T5GemmaForConditionalGeneration if is_torch_available() else None

    def setUp(self):
        self.model_tester = T5GemmaModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=T5GemmaConfig,
            # For faking the testing.
            hidden_size=37,
            vocab_size=self.model_tester.vocab_size,
            num_attention_heads=self.model_tester.num_attention_heads,
            num_hidden_layers=self.model_tester.num_hidden_layers,
        )

    # Copied from tests.models.t5.test_modeling_t5.T5ModelTest.is_pipeline_test_to_skip
    def is_pipeline_test_to_skip(
        self,
        pipeline_test_case_name,
        config_class,
        model_architecture,
        tokenizer_name,
        image_processor_name,
        feature_extractor_name,
        processor_name,
    ):
        if tokenizer_name is None:
            return True
        if pipeline_test_case_name == "QAPipelineTests" and not tokenizer_name.endswith("Fast"):
            return True

        return False

    # Copied from tests.models.t5.test_modeling_t5.T5ModelTest.test_config
    def test_config(self):
        self.config_tester.run_common_tests()

    # Copied from tests.models.t5.test_modeling_t5.T5ModelTest.test_shift_right
    def test_shift_right(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_prepare_lm_labels_via_shift_left(*config_and_inputs)

    # Copied from tests.models.t5.test_modeling_t5.T5ModelTest.test_model
    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    # Based on tests.models.t5.test_modeling_t5.T5ModelTest.test_inputs_embeds
    def test_inputs_embeds(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in (T5GemmaModel, T5GemmaForConditionalGeneration):
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

    # Copied from tests.models.t5.test_modeling_t5.T5ModelTest.test_config_and_model_silu_gated
    def test_config_and_model_silu_gated(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        config = config_and_inputs[0]
        config.feed_forward_proj = "gated-silu"
        self.model_tester.create_and_check_model(*config_and_inputs)

    # Copied from tests.models.t5.test_modeling_t5.T5ModelTest.test_with_lm_head
    def test_with_lm_head(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_with_lm_head(*config_and_inputs)

    # Copied from tests.models.t5.test_modeling_t5.T5ModelTest.test_with_sequence_classification_head
    def test_with_sequence_classification_head(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_with_sequence_classification_head(*config_and_inputs)

    @parameterized.expand([(True,), (False,)])
    def test_encoderonly_sequence_classification_head(self, is_encoder_decoder):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_encoderonly_for_sequence_classification_head(
            *config_and_inputs, is_encoder_decoder
        )

    @parameterized.expand([(True,), (False,)])
    def test_encoderonly_token_classification_head(self, is_encoder_decoder):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_encoderonly_for_token_classification_head(
            *config_and_inputs, is_encoder_decoder
        )

    # Copied from tests.models.t5.test_modeling_t5.T5ModelTest.test_decoder_model_past
    def test_decoder_model_past(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_past(*config_and_inputs)

    # Copied from tests.models.t5.test_modeling_t5.T5ModelTest.test_decoder_model_past_with_attn_mask
    def test_decoder_model_past_with_attn_mask(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_attention_mask_past(*config_and_inputs)

    # Based on tests.models.t5.test_modeling_t5.T5ModelTest.test_decoder_model_past_with_3d_attn_mask
    def test_decoder_model_past_with_3d_attn_mask(self):
        (
            config,
            input_ids,
            decoder_input_ids,
            attention_mask,
            decoder_attention_mask,
            lm_labels,
        ) = self.model_tester.prepare_config_and_inputs()

        attention_mask = ids_tensor(
            [self.model_tester.batch_size, self.model_tester.encoder_seq_length, self.model_tester.encoder_seq_length],
            vocab_size=2,
        )
        decoder_attention_mask = ids_tensor(
            [self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.seq_length],
            vocab_size=2,
        )

        self.model_tester.create_and_check_decoder_model_attention_mask_past(
            config,
            input_ids,
            decoder_input_ids,
            attention_mask,
            decoder_attention_mask,
            lm_labels,
        )

    # Copied from tests.models.t5.test_modeling_t5.T5ModelTest.test_decoder_model_past_with_large_inputs
    def test_decoder_model_past_with_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

    # Copied from tests.models.t5.test_modeling_t5.T5ModelTest.test_generate_with_past_key_values
    def test_generate_with_past_key_values(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_generate_with_past_key_values(*config_and_inputs)

    @unittest.skipIf(torch_device == "cpu", "Can't do half precision")
    # Copied from tests.models.t5.test_modeling_t5.T5ModelTest.test_model_fp16_forward
    def test_model_fp16_forward(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_fp16_forward(*config_and_inputs)

    # Based on tests.models.gemma.test_modeling_gemma.GemmaModelTest.test_Gemma_sequence_classification_model with Gemma -> T5Gemma (Add is_encoder_decoder option)
    def test_T5Gemma_sequence_classification_model(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)

        for is_encoder_decoder in [True, False]:
            model = (
                self.model_tester.for_sequence_class(config, is_encoder_decoder=is_encoder_decoder)
                .to(torch_device)
                .eval()
            )
            result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
            self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    # Based on tests.models.gemma.test_modeling_gemma.GemmaModelTest.test_Gemma_sequence_classification_model_for_single_label with Gemma -> T5Gemma (Add is_encoder_decoder option)
    def test_T5Gemma_sequence_classification_model_for_single_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "single_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)

        for is_encoder_decoder in [True, False]:
            model = (
                self.model_tester.for_sequence_class(config, is_encoder_decoder=is_encoder_decoder)
                .to(torch_device)
                .eval()
            )
            result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
            self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    # Based on tests.models.gemma.test_modeling_gemma.GemmaModelTest.test_Gemma_sequence_classification_model_for_multi_label with Gemma -> T5Gemma (Add is_encoder_decoder option)
    def test_T5Gemma_sequence_classification_model_for_multi_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "multi_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor(
            [self.model_tester.batch_size, config.num_labels], self.model_tester.type_sequence_label_size
        ).to(torch.float)

        for is_encoder_decoder in [True, False]:
            model = (
                self.model_tester.for_sequence_class(config, is_encoder_decoder=is_encoder_decoder)
                .to(torch_device)
                .eval()
            )
            result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
            self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    # Based on tests.models.gemma.test_modeling_gemma.GemmaModelTest.test_Gemma_token_classification_model with Gemma -> T5Gemma (Add is_encoder_decoder option)
    def test_T5Gemma_token_classification_model(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        token_labels = ids_tensor([self.model_tester.batch_size, self.model_tester.seq_length], config.num_labels)

        for is_encoder_decoder in [True, False]:
            model = (
                self.model_tester.for_token_class(config, is_encoder_decoder=is_encoder_decoder)
                .to(torch_device)
                .eval()
            )

            result = model(input_ids, attention_mask=attention_mask, labels=token_labels)
            self.assertEqual(
                result.logits.shape,
                (self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.num_labels),
            )

    # Based on tests.models.gemma.test_modeling_gemma.GemmaModelTest.test_sdpa_equivalence
    # Add decoder_input_ids and adjust hidden states.
    @require_torch_sdpa
    @require_torch_accelerator
    def test_sdpa_equivalence(self):
        for model_class in self.all_model_classes:
            if not model_class._supports_sdpa:
                self.skipTest(reason="Model does not support SDPA")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config).to(torch_device)
            dummy_input = inputs_dict[model_class.main_input_name].to(torch_device)
            decoder_dummy_input = torch.ones_like(dummy_input)

            model.config._attn_implementation = "sdpa"
            states_sdpa = model(dummy_input, decoder_input_ids=decoder_dummy_input, output_hidden_states=True)

            model.config._attn_implementation = "eager"
            states_eager = model(dummy_input, decoder_input_ids=decoder_dummy_input, output_hidden_states=True)

            if hasattr(states_sdpa, "decoder_hidden_states"):
                states_sdpa = states_sdpa.decoder_hidden_states[-1]
                states_eager = states_eager.decoder_hidden_states[-1]
            else:
                states_sdpa = states_sdpa.hidden_states[-1]
                states_eager = states_eager.hidden_states[-1]

            torch.testing.assert_close(states_sdpa, states_eager, atol=1e-5, rtol=1e-5)

    @unittest.skip("T5Gemma eager/FA2 attention outputs are expected to be different")
    def test_flash_attn_2_equivalence(self):
        pass

    # Based on tests.test_modeling_common.ModelTesterMixin.test_attention_outputs
    # Skip token classification
    def test_attention_outputs(self):
        if not self.has_attentions:
            self.skipTest(reason="Model does not output attentions")

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        # force eager attention to support output attentions
        config._attn_implementation = "eager"

        seq_len = getattr(self.model_tester, "seq_length", None)
        decoder_seq_length = getattr(self.model_tester, "decoder_seq_length", seq_len)
        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", seq_len)
        decoder_key_length = getattr(self.model_tester, "decoder_key_length", decoder_seq_length)
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)
        chunk_length = getattr(self.model_tester, "chunk_length", None)
        if chunk_length is not None and hasattr(self.model_tester, "num_hashes"):
            encoder_seq_length = encoder_seq_length * self.model_tester.num_hashes

        for model_class in self.all_model_classes:
            # Skip token and sequence classification.
            if model_class in [self.model_tester.for_token_class, self.model_tester.for_sequence_class]:
                continue

            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            model = model_class._from_config(config, attn_implementation="eager")
            config = model.config
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config._attn_implementation = "eager"
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            if chunk_length is not None:
                self.assertListEqual(
                    list(attentions[0].shape[-4:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, chunk_length, encoder_key_length],
                )
            else:
                self.assertListEqual(
                    list(attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
                )
            out_len = len(outputs)

            if self.is_encoder_decoder:
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
                        encoder_key_length,
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

            if hasattr(self.model_tester, "num_hidden_states_types"):
                added_hidden_states = self.model_tester.num_hidden_states_types
            elif self.is_encoder_decoder:
                added_hidden_states = 2
            else:
                added_hidden_states = 1
            self.assertEqual(out_len + added_hidden_states, len(outputs))

            self_attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions

            self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
            if chunk_length is not None:
                self.assertListEqual(
                    list(self_attentions[0].shape[-4:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, chunk_length, encoder_key_length],
                )
            else:
                self.assertListEqual(
                    list(self_attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
                )

    # Based on tests.generation.test_utils.GenerationTesterMixin.test_past_key_values_format
    # Adjust encoder attention number for cross-attention caching and update attention head dimension
    @pytest.mark.generate
    def test_past_key_values_format(self, custom_all_cache_shapes=None):
        """
        Test that the KV cache is formatted correctly. Exceptions need to explicitly overwrite this test, or pass the
        expected cache shapes.
        Having a standard KV cache format is important for a consistent API (and for advanced generation methods).
        """
        for model_class in self.all_generative_model_classes:
            config, inputs = self.model_tester.prepare_config_and_inputs_for_common()

            # 1. If it doesn't support cache, skip the test
            if not hasattr(config.get_text_config(), "use_cache"):
                self.skipTest(reason=f"{model_class.__name__} doesn't support caching")

            model = model_class(config).to(torch_device)
            model = model.eval()
            if "use_cache" not in inputs:
                inputs["use_cache"] = True
            outputs = model(**inputs)

            if "past_key_values" not in outputs:
                self.skipTest(reason="This model doesn't return `past_key_values`")

            # 2. retrieve the KV cache and compute its default expected shapes (if no custom shapes are provided)
            past_kv = outputs["past_key_values"]
            is_legacy_cache = not isinstance(past_kv, Cache)

            text_config = config.get_text_config().decoder
            num_decoder_layers = text_config.num_hidden_layers

            if custom_all_cache_shapes is None:
                num_query_attention_heads = getattr(
                    text_config, "decoder_attention_heads", text_config.num_attention_heads
                )
                per_head_embed_dim = text_config.head_dim
                num_key_value_heads = (
                    text_config.num_key_value_heads
                    if getattr(text_config, "num_key_value_heads", None) is not None
                    else num_query_attention_heads
                )
                if config.is_encoder_decoder:
                    encoder_num_attention_heads = num_key_value_heads
                    encoder_per_head_embed_dim = per_head_embed_dim
                    batch_size, seq_length = inputs["decoder_input_ids"].shape[:2]
                    # The sequence length for the encoder K V depends on the model. Since it is not manipulated in
                    # autoregressive generation, we're keeping the test general and not checking the 3rd dim
                    default_cross_attention_shape = (
                        batch_size,
                        encoder_num_attention_heads,
                        encoder_per_head_embed_dim,
                    )
                    default_self_attention_shape = (batch_size, num_key_value_heads, seq_length, per_head_embed_dim)
                    all_cache_shapes = [
                        [
                            default_self_attention_shape,
                            default_self_attention_shape,
                            default_cross_attention_shape,
                            default_cross_attention_shape,
                        ]
                        for _ in range(num_decoder_layers)
                    ]
                else:
                    batch_size, seq_length = inputs["input_ids"].shape[:2]
                    default_self_attention_shape = (batch_size, num_key_value_heads, seq_length, per_head_embed_dim)
                    all_cache_shapes = [
                        [default_self_attention_shape, default_self_attention_shape] for _ in range(num_decoder_layers)
                    ]

            else:
                all_cache_shapes = custom_all_cache_shapes

            # 3. Check cache shapes
            # 3.1. Encoder-Decoder checks
            if config.is_encoder_decoder:
                num_cache_decoder_layers = (
                    len(past_kv) if is_legacy_cache else len(past_kv.self_attention_cache.key_cache)
                )
                self.assertEqual(num_cache_decoder_layers, num_decoder_layers)

                for i in range(num_decoder_layers):
                    if is_legacy_cache:
                        self.assertEqual(len(past_kv[0]), 4)  # legacy check: confirm number of elements in tuple

                    # Self attention
                    self_attention_layer_key_cache = (
                        past_kv[i][0] if is_legacy_cache else past_kv.self_attention_cache.key_cache[i]
                    )
                    self_attention_layer_value_cache = (
                        past_kv[i][1] if is_legacy_cache else past_kv.self_attention_cache.value_cache[i]
                    )
                    self.assertEqual(self_attention_layer_key_cache.shape, all_cache_shapes[i][0])
                    self.assertEqual(self_attention_layer_value_cache.shape, all_cache_shapes[i][1])

                    # Cross attention (ignore 3rd dim, see default shape preparation)
                    cross_attention_layer_key_cache = (
                        past_kv[i][2] if is_legacy_cache else past_kv.cross_attention_cache.key_cache[i]
                    )
                    cross_attention_layer_value_cache = (
                        past_kv[i][3] if is_legacy_cache else past_kv.cross_attention_cache.value_cache[i]
                    )
                    cross_attention_layer_key_cache = cross_attention_layer_key_cache[:, :, 0, :]
                    cross_attention_layer_value_cache = cross_attention_layer_value_cache[:, :, 0, :]
                    self.assertEqual(cross_attention_layer_key_cache.shape, all_cache_shapes[i][2])
                    self.assertEqual(cross_attention_layer_value_cache.shape, all_cache_shapes[i][3])

            # 3.2. Decoder-only checks
            else:
                num_cache_decoder_layers = len(past_kv) if is_legacy_cache else len(past_kv.key_cache)
                self.assertEqual(num_cache_decoder_layers, num_decoder_layers)

                for i in range(num_decoder_layers):
                    if is_legacy_cache:
                        self.assertEqual(len(past_kv[0]), 2)  # legacy check: confirm number of elements in tuple

                    # Self attention
                    self_attention_layer_key_cache = past_kv[i][0] if is_legacy_cache else past_kv.key_cache[i]
                    self_attention_layer_value_cache = past_kv[i][1] if is_legacy_cache else past_kv.value_cache[i]
                    self.assertEqual(self_attention_layer_key_cache.shape, all_cache_shapes[i][0])
                    self.assertEqual(self_attention_layer_value_cache.shape, all_cache_shapes[i][1])

    @unittest.skip("Mismatch issue doesn't exist in T5Gemma.")
    def test_load_with_mismatched_shapes(self):
        pass

    # Based on tests.generation.test_utils.GenerationTesterMixin.test_generate_continue_from_past_key_values
    # Updated decoder_attention_mask to consider the appended bos token
    @pytest.mark.generate
    def test_generate_continue_from_past_key_values(self):
        # Tests that we can continue generating from past key values, returned from a previous `generate` call
        for model_class in self.all_generative_model_classes:
            if model_class == self.model_tester.for_token_class:
                continue
            if any(model_name in model_class.__name__.lower() for model_name in ["imagegpt", "mllama"]):
                self.skipTest(reason="Won't fix: old model with unique inputs/caches/other")
            if any(model_name in model_class.__name__.lower() for model_name in ["umt5"]):
                self.skipTest(reason="TODO: needs modeling or test input preparation fixes for compatibility")

            config, inputs = self.model_tester.prepare_config_and_inputs_for_common()

            if not hasattr(config.get_text_config(), "use_cache"):
                self.skipTest(reason=f"{model_class.__name__} doesn't support caching")

            # Let's make it always:
            # 1. use cache (for obvious reasons)
            # 2. generate to max length (which can be achieved by setting the eos token to an invalid value), which
            #    would make the test flaky (e.g. EOS is generated on iteration 1 on both generations, but the
            #    continuation would force it to generate beyond an EOS token)
            # 3. ignore `token_type_ids` for simplicity
            # 4. ignore `forced_eos_token_id`, which requires further manipulation of the continuation inputs and is
            #    active by default on some models
            # 5. ignore `encoder_no_repeat_ngram_size`, which is set by default in some encoder-decoder models. When
            #    we use their decoder as a stand-alone model, `encoder_no_repeat_ngram_size` actually prevents
            #    repetition exclusively from the prompt. This test relies on comparing one call vs 2 calls
            #    with cache, what is considered a prompt is different in the two cases.

            if "token_type_ids" in inputs:
                del inputs["token_type_ids"]

            model = model_class(config).to(torch_device)
            model.eval()

            # If "past_key_values" is not returned, skip the test (e.g. RWKV uses a different cache name and format)
            outputs = model(**inputs)
            if "past_key_values" not in outputs:
                self.skipTest(reason="This model doesn't return `past_key_values`")

            generate_kwargs = {
                "pad_token_id": -1,
                "eos_token_id": -1,
                "forced_eos_token_id": None,
                "encoder_no_repeat_ngram_size": 0,
                "use_cache": True,
                "do_sample": False,
                "return_dict_in_generate": True,
                "output_scores": True,
            }

            # Traditional way of generating text, with `return_dict_in_generate` to return the past key values
            outputs = model.generate(**inputs, **generate_kwargs, max_new_tokens=4)

            # Let's generate again, but passing the past key values in between (3 + 1 = 4 tokens). Note that the
            # inputs may need to be tweaked across `generate` calls (like the attention mask).
            outputs_cached = model.generate(**inputs, **generate_kwargs, max_new_tokens=3)

            # Continue from the tokens generated above, preparing the inputs accordingly
            inputs["past_key_values"] = outputs_cached.past_key_values
            new_attention_len = outputs_cached.sequences.shape[-1]

            # It must be encoder-decoder models
            self.assertTrue(config.is_encoder_decoder)

            inputs["decoder_input_ids"] = outputs_cached.sequences
            if "decoder_attention_mask" in inputs:
                decoder_attention_mask = inputs["decoder_attention_mask"]

                # Add BOS mask: the new sequence comes with a new BOS token, which is not included in the original inputs
                padding_tensor = torch.ones_like(decoder_attention_mask[:, :1])
                decoder_attention_mask = torch.cat([padding_tensor, decoder_attention_mask], dim=1)

                inputs["decoder_attention_mask"] = torch.nn.functional.pad(
                    decoder_attention_mask,
                    (0, new_attention_len - decoder_attention_mask.shape[1]),
                    mode="constant",
                    value=1,
                )

            first_caches_scores = outputs_cached.scores
            outputs_cached = model.generate(**inputs, **generate_kwargs, max_new_tokens=1)
            full_cached_scores = first_caches_scores + outputs_cached.scores
            outputs_cached.scores = full_cached_scores

            # The two sets of generated text and past kv should be equal to each other
            self._check_similar_generate_outputs(outputs, outputs_cached)
            for layer_idx in range(len(outputs_cached.past_key_values)):
                for kv_idx in range(len(outputs_cached.past_key_values[layer_idx])):
                    self.assertTrue(
                        torch.allclose(
                            outputs.past_key_values[layer_idx][kv_idx],
                            outputs_cached.past_key_values[layer_idx][kv_idx],
                        )
                    )

    # Based on tests.test_modeling_common.ModelTesterMixin.test_inputs_embeds_matches_input_ids
    # Update encoder and decoder embeddings
    def test_inputs_embeds_matches_input_ids(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model_class = self.model_tester.model_class

        model = model_class(config)
        model.to(torch_device)
        model.eval()

        model_forward_args = inspect.signature(model.forward).parameters
        if "inputs_embeds" not in model_forward_args:
            self.skipTest(reason="This model doesn't use `inputs_embeds`")

        inputs = copy.deepcopy(self._prepare_for_class(inputs_dict, model_class))
        pad_token_id = config.pad_token_id if config.pad_token_id is not None else 1

        encoder_embedding = model.get_encoder().get_input_embeddings()
        decoder_embedding = model.get_decoder().get_input_embeddings()

        encoder_input_ids = inputs["input_ids"]
        decoder_input_ids = inputs.get("decoder_input_ids", encoder_input_ids)
        encoder_input_ids[encoder_input_ids == pad_token_id] = max(0, pad_token_id + 1)
        decoder_input_ids[decoder_input_ids == pad_token_id] = max(0, pad_token_id + 1)
        del inputs["input_ids"]
        inputs.pop("decoder_input_ids", None)

        inputs_embeds = encoder_embedding(encoder_input_ids)
        decoder_inputs_embeds = decoder_embedding(decoder_input_ids)
        with torch.no_grad():
            out_ids = model(input_ids=encoder_input_ids, decoder_input_ids=decoder_input_ids, **inputs)[0]
            out_embeds = model(inputs_embeds=inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds, **inputs)[0]

        torch.testing.assert_close(out_embeds, out_ids)

    # Based on tests.test_modeling_common.ModelTesterMixin.test_inputs_embeds_matches_input_ids
    # Adjust token classiifcation
    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            if model_class in [self.model_tester.for_token_class, self.model_tester.for_sequence_class]:
                model = model_class(config, is_encoder_decoder=False)
            else:
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
                if hasattr(self.model_tester, "chunk_length") and self.model_tester.chunk_length > 1:
                    seq_length = seq_length * self.model_tester.chunk_length
            else:
                seq_length = self.model_tester.seq_length

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [seq_length, self.model_tester.hidden_size],
            )

            if config.is_encoder_decoder:
                hidden_states = outputs.decoder_hidden_states

                self.assertIsInstance(hidden_states, (list, tuple))
                self.assertEqual(len(hidden_states), expected_num_layers)
                seq_len = getattr(self.model_tester, "seq_length", None)
                decoder_seq_length = getattr(self.model_tester, "decoder_seq_length", seq_len)

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

    # Based on tests.models.t5.test_modeling_t5.T5ModelTest.test_custom_4d_attention_mask
    # Excluding the final token from input_ids
    def test_custom_4d_attention_mask(self):
        for model_class in self.all_generative_model_classes:
            config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config).to(device=torch_device, dtype=torch.float32)

            (
                input_ids,
                _,
                input_ids_shared_prefix,
                mask_shared_prefix,
                _,
            ) = self._get_custom_4d_mask_test_data()

            logits = model.forward(
                decoder_input_ids=input_ids,
                input_ids=input_ids[:, :-1],
            ).logits
            # logits.shape == torch.Size([3, 4, ...])

            logits_shared_prefix = model(
                input_ids=input_ids[:1, :-1],
                decoder_input_ids=input_ids_shared_prefix,
                decoder_attention_mask=mask_shared_prefix,
            )[0]
            # logits_shared_prefix.shape == torch.Size([1, 6, ...])

            out_last_tokens = logits[:, -1, :]  # last tokens in each batch line
            out_shared_prefix_last_tokens = logits_shared_prefix[0, -3:, :]  # last three tokens

            # comparing softmax-normalized logits:
            normalized_0 = F.softmax(out_last_tokens)
            normalized_1 = F.softmax(out_shared_prefix_last_tokens)
            torch.testing.assert_close(normalized_0, normalized_1, rtol=1e-3, atol=1e-4)

    # Based on tests.test_modeling_common.ModelTesterMixin.test_flex_attention_with_grads
    # Update hidden size for encoder and decoder
    @require_torch_gpu
    def test_flex_attention_with_grads(self):
        for model_class in self.all_model_classes:
            # TODO: raushan, fix for composite models after making VLMs support new attn API
            if not model_class._supports_flex_attn or self._is_composite:
                self.skipTest(reason="This model does not support flex attention")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config._attn_implementation = "flex_attention"
            # Flex Attention cannot use dropout
            config.encoder.attention_dropout = 0
            config.decoder.attention_dropout = 0

            # Flex attention relies on triton on compilation
            # However, triton cannot handle hidden dimensions of less than 16
            # --> forcing at least a hidden dim of 16
            config.encoder.hidden_size *= max(
                16
                // getattr(
                    config.encoder, "head_dim", config.encoder.hidden_size // config.encoder.num_attention_heads
                ),
                1,
            )
            config.decoder.hidden_size *= max(
                16
                // getattr(
                    config.decoder, "head_dim", config.decoder.hidden_size // config.decoder.num_attention_heads
                ),
                1,
            )
            config.decoder.cross_attention_hidden_size = config.encoder.hidden_size

            config.decoder.head_dim = max(16, config.decoder.head_dim)
            config.encoder.head_dim = max(16, config.encoder.head_dim)

            model = model_class(config).to(device=torch_device)
            self.assertTrue(model.config._attn_implementation == "flex_attention")

            # Elaborate workaround for encoder-decoder models as some do not specify their main input
            dummy_inputs = {model.main_input_name: inputs_dict[model.main_input_name].to(torch_device)}
            if config.is_encoder_decoder:
                dummy_inputs["decoder_input_ids"] = inputs_dict["decoder_input_ids"].to(torch_device)
                dummy_inputs["decoder_attention_mask"] = inputs_dict["decoder_attention_mask"].to(torch_device)

            # If this does not raise an error, the test passes (see https://github.com/huggingface/transformers/pull/35605)
            _ = model(**dummy_inputs)

    @unittest.skip("EncoderDecoderCache can't be gathered because it is not iterable.")
    def test_multi_gpu_data_parallel_forward(self):
        pass


class T5GemmaEncoderOnlyModelTester:
    config_class = T5GemmaConfig
    module_config_class = T5GemmaModuleConfig

    if is_torch_available():
        model_class = T5GemmaEncoderModel

    def __init__(
        self,
        parent,
        batch_size=13,
        is_training=True,
        use_attention_mask=True,
        use_labels=True,
        vocab_size=99,
        seq_length=7,
        # default to encoders
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=37,
        # common
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        scope=None,
        # special ids
        eos_token_id=1,
        pad_token_id=0,
        bos_token_id=2,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.is_training = is_training
        self.use_attention_mask = use_attention_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        # encoder
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        # common
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope
        self.head_dim = self.hidden_size // self.num_attention_heads
        # special ids
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id

    def get_encoder_config(self):
        return self.module_config_class(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=False,
            initializer_range=self.initializer_range,
            head_dim=self.head_dim,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
        )

    def get_config(self):
        return self.config_class(
            encoder=self.get_encoder_config(),
            decoder=None,
            is_encoder_decoder=False,
            # Used for generation test.
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
        )

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        # Remove BOS symbols from inputs.
        input_ids = torch.where(input_ids == self.bos_token_id, 42, input_ids)

        attention_mask = None
        if self.use_attention_mask:
            attention_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

        config = self.get_config()

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
        model = self.model_class(config=config)
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
        model = self.model_class(config=config).to(torch_device).half().eval()
        output = model(input_ids, attention_mask=attention_mask)["last_hidden_state"]
        self.parent.assertFalse(torch.isnan(output).any().item())

    def create_and_check_with_token_classification_head(
        self,
        config,
        input_ids,
        attention_mask,
    ):
        labels = torch.tensor([1] * self.seq_length * self.batch_size, dtype=torch.long, device=torch_device)
        model = T5GemmaForTokenClassification(config=config, is_encoder_decoder=False).to(torch_device).eval()
        outputs = model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )
        self.parent.assertEqual(outputs["logits"].size(), (self.batch_size, self.seq_length, config.num_labels))
        self.parent.assertEqual(outputs["loss"].size(), ())

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


@require_torch
class T5GemmaEncoderOnlyModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (T5GemmaEncoderModel, T5GemmaForTokenClassification) if is_torch_available() else ()
    test_pruning = False
    test_resize_embeddings = False
    test_headmasking = False
    _is_stateful = True
    is_encoder_decoder = False
    model_split_percents = [0.4, 0.5]

    def setUp(self):
        self.model_tester = T5GemmaEncoderOnlyModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=T5GemmaConfig,
            # For faking the testing.
            hidden_size=37,
            vocab_size=self.model_tester.vocab_size,
            num_attention_heads=self.model_tester.num_attention_heads,
            num_hidden_layers=self.model_tester.num_hidden_layers,
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skipIf(torch_device == "cpu", "Can't do half precision")
    def test_model_fp16_forward(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_fp16_forward(*config_and_inputs)

    def test_with_token_classification_head(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_with_token_classification_head(*config_and_inputs)

    @unittest.skip("No loss in the output of T5GemmaEncoderModel")
    def test_training(self):
        pass

    @unittest.skip("No loss in the output of T5GemmaEncoderModel")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip("No loss in the output of T5GemmaEncoderModel")
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip("No loss in the output of T5GemmaEncoderModel")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    # Based on tests.test_modeling_common.ModelTesterMixin.test_flex_attention_with_grads
    # Update hidden size for encoder
    @require_torch_gpu
    def test_flex_attention_with_grads(self):
        for model_class in self.all_model_classes:
            # TODO: raushan, fix for composite models after making VLMs support new attn API
            if not model_class._supports_flex_attn or self._is_composite:
                self.skipTest(reason="This model does not support flex attention")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config._attn_implementation = "flex_attention"
            # Flex Attention cannot use dropout
            config.encoder.attention_dropout = 0

            # Flex attention relies on triton on compilation
            # However, triton cannot handle hidden dimensions of less than 16
            # --> forcing at least a hidden dim of 16
            config.encoder.hidden_size *= max(
                16
                // getattr(
                    config.encoder, "head_dim", config.encoder.hidden_size // config.encoder.num_attention_heads
                ),
                1,
            )
            config.encoder.head_dim = max(16, config.encoder.head_dim)

            model = model_class(config).to(device=torch_device)
            self.assertTrue(model.config._attn_implementation == "flex_attention")

            # Elaborate workaround for encoder-decoder models as some do not specify their main input
            dummy_inputs = {model.main_input_name: inputs_dict[model.main_input_name].to(torch_device)}

            # If this does not raise an error, the test passes (see https://github.com/huggingface/transformers/pull/35605)
            _ = model(**dummy_inputs)


# Based on tests.models.t5.test_modeling_t5.TestAsymmetricT5
# Adapted for T5Gemma
@require_torch
class TestAsymmetricT5Gemma(unittest.TestCase):
    def build_model_and_check_forward_pass(self, **kwargs):
        tester = T5GemmaModelTester(self, **kwargs)
        config, *inputs = tester.prepare_config_and_inputs()
        (
            input_ids,
            decoder_input_ids,
            attention_mask,
            decoder_attention_mask,
            lm_labels,
        ) = inputs
        model = T5GemmaForConditionalGeneration(config=config).to(torch_device).eval()
        outputs = model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )
        # outputs = model(*inputs)
        assert len(outputs) == 4
        assert outputs["logits"].size() == (tester.batch_size, tester.seq_length, tester.vocab_size)
        assert outputs["loss"].size() == ()
        return model.model

    def test_small_decoder(self):
        model = self.build_model_and_check_forward_pass(num_hidden_layers=1, encoder_num_hidden_layers=2)
        assert len(model.encoder.layers) == 2
        assert len(model.decoder.layers) == 1

    def test_defaulting_to_symmetry(self):
        model = self.build_model_and_check_forward_pass(num_hidden_layers=2, encoder_num_hidden_layers=2)
        assert len(model.decoder.layers) == len(model.encoder.layers) == 2
