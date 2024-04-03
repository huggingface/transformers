# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch RecurrentGemma model. """
import tempfile
import unittest

import pytest
from parameterized import parameterized

from transformers import AutoModelForCausalLM, AutoTokenizer, RecurrentGemmaConfig, is_torch_available
from transformers.testing_utils import (
    require_bitsandbytes,
    require_flash_attn,
    require_read_token,
    require_torch,
    require_torch_gpu,
    require_torch_sdpa,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import RecurrentGemmaForCausalLM, RecurrentGemmaModel


class RecurrentGemmaModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=48,
        is_training=True,
        use_input_mask=False,
        use_token_type_ids=False,
        use_labels=True,
        num_hidden_layers=3,
        vocab_size=99,
        hidden_size=32,
        intermediate_size=3 * 32,
        num_heads=2,
        lru_width=2 * 32,
        embeddings_scale_by_sqrt_dim=True,
        attention_window_size=16,
        conv1d_width=4,
        logits_soft_cap=30.0,
        rms_norm_eps=1e-6,
        use_cache=True,
        rope_theta=10000.0,
        type_vocab_size=16,
        type_sequence_label_size=2,
        num_labels=3,
        num_choices=4,
        pad_token_id=0,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels

        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.lru_width = lru_width if lru_width is not None else hidden_size
        self.embeddings_scale_by_sqrt_dim = embeddings_scale_by_sqrt_dim
        self.attention_window_size = attention_window_size
        self.conv1d_width = conv1d_width
        self.logits_soft_cap = logits_soft_cap
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta

        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.pad_token_id = pad_token_id
        self.scope = scope

    # Copied from tests.models.mistral.test_modeling_mistral.MistralModelTester.prepare_config_and_inputs
    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = torch.tril(torch.ones(self.batch_size, self.seq_length)).to(torch_device)

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    # Ignore copy
    def get_config(self):
        return RecurrentGemmaConfig(
            num_hidden_layers=self.num_hidden_layers,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_heads=self.num_heads,
            lru_width=self.lru_width,
            embeddings_scale_by_sqrt_dim=self.embeddings_scale_by_sqrt_dim,
            attention_window_size=self.attention_window_size,
            conv1d_width=self.conv1d_width,
            logits_soft_cap=self.logits_soft_cap,
            rms_norm_eps=self.rms_norm_eps,
            use_cache=self.use_cache,
            rope_theta=self.rope_theta,
            pad_token_id=self.pad_token_id,
        )

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.create_and_check_model with Llama->Gemma
    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = RecurrentGemmaModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.create_and_check_model_as_decoder with Llama->Gemma
    def create_and_check_model_as_decoder(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        config.add_cross_attention = True
        model = RecurrentGemmaModel(config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        result = model(
            input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
        )
        result = model(input_ids, attention_mask=input_mask)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.create_and_check_for_causal_lm with Llama->Gemma
    def create_and_check_for_causal_lm(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        model = RecurrentGemmaForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.create_and_check_decoder_model_past_large_inputs with Llama->Gemma
    def create_and_check_decoder_model_past_large_inputs(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        config.is_decoder = True
        config.add_cross_attention = True
        model = RecurrentGemmaForCausalLM(config=config)
        model.to(torch_device)
        model.eval()

        # first forward pass
        outputs = model(
            input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_mask = ids_tensor((self.batch_size, 3), vocab_size=2)

        # append to next input_ids and
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        next_attention_mask = torch.cat([input_mask, next_mask], dim=-1)

        output_from_no_past = model(
            next_input_ids,
            attention_mask=next_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_hidden_states=True,
        )["hidden_states"][0]
        output_from_past = model(
            next_tokens,
            attention_mask=next_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
        )["hidden_states"][0]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.prepare_config_and_inputs_for_common with Llama->Gemma
    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class RecurrentGemmaModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (RecurrentGemmaModel, RecurrentGemmaForCausalLM) if is_torch_available() else ()
    all_generative_model_classes = (RecurrentGemmaForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": RecurrentGemmaModel,
            # "text-classification": GemmaForSequenceClassification,
            "text-generation": RecurrentGemmaForCausalLM,
            # "zero-shot": GemmaForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False

    # Need to remove 0.9 in `test_cpu_offload`
    # This is because we are hitting edge cases with the causal_mask buffer
    model_split_percents = [0.5, 0.6]

    # TODO (ydshieh): Check this. See https://app.circleci.com/pipelines/github/huggingface/transformers/79245/workflows/9490ef58-79c2-410d-8f51-e3495156cf9c/jobs/1012146
    def is_pipeline_test_to_skip(
        self, pipeline_test_casse_name, config_class, model_architecture, tokenizer_name, processor_name
    ):
        return True

    def setUp(self):
        self.model_tester = RecurrentGemmaModelTester(self)
        self.config_tester = ConfigTester(self, config_class=RecurrentGemmaConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_various_embeddings(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        for type in ["absolute", "relative_key", "relative_key_query"]:
            config_and_inputs[0].position_embedding_type = type
            self.model_tester.create_and_check_model(*config_and_inputs)

    # def test_RecurrentGemma_sequence_classification_model(self):
    #     config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
    #     print(config)
    #     config.num_labels = 3
    #     input_ids = input_dict["input_ids"]
    #     attention_mask = input_ids.ne(1).to(torch_device)
    #     sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
    #     model = GemmaForSequenceClassification(config)
    #     model.to(torch_device)
    #     model.eval()
    #     result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
    #     self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))
    #
    # def test_RecurrentGemma_sequence_classification_model_for_single_label(self):
    #     config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
    #     config.num_labels = 3
    #     config.problem_type = "single_label_classification"
    #     input_ids = input_dict["input_ids"]
    #     attention_mask = input_ids.ne(1).to(torch_device)
    #     sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
    #     model = GemmaForSequenceClassification(config)
    #     model.to(torch_device)
    #     model.eval()
    #     result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
    #     self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))
    #
    # def test_RecurrentGemma_sequence_classification_model_for_multi_label(self):
    #     config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
    #     config.num_labels = 3
    #     config.problem_type = "multi_label_classification"
    #     input_ids = input_dict["input_ids"]
    #     attention_mask = input_ids.ne(1).to(torch_device)
    #     sequence_labels = ids_tensor(
    #         [self.model_tester.batch_size, config.num_labels], self.model_tester.type_sequence_label_size
    #     ).to(torch.float)
    #     model = GemmaForSequenceClassification(config)
    #     model.to(torch_device)
    #     model.eval()
    #     result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
    #     self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    @unittest.skip("TODO @gante fix this for Llama")
    @parameterized.expand([(1, False), (1, True), (4, False)])
    def test_new_cache_format(self, num_beams, do_sample):
        pass

    @unittest.skip("RecurrentGemma buffers include complex numbers, which breaks this test")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip("RecurrentGemma uses GQA on all models so the KV cache is a non standard format")
    def test_past_key_values_format(self):
        pass


@require_torch_gpu
@slow
@require_read_token
class RecurrentGemmaIntegrationTest(unittest.TestCase):
    input_text = ["Hello I am doing", "Hi today"]

    def test_model_2b_fp32(self):
        model_id = "google/recurrentgemma-2b"
        EXPECTED_TEXTS = [
            "Hello I am doing a project on the 1990s and I need to know what the most popular music",
            "Hi today I am going to share with you a very easy and simple recipe of <strong><em>Kaju Kat",
        ]

        model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True).to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    # def test_model_2b_fp16(self):
    #     model_id = "google/gemma-2b"
    #     EXPECTED_TEXTS = [
    #         "Hello I am doing a project on the 1990s and I need to know what the most popular music",
    #         "Hi today I am going to share with you a very easy and simple recipe of <strong><em>Kaju Kat",
    #     ]
    #
    #     model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, torch_dtype=torch.float16).to(
    #         torch_device
    #     )
    #
    #     tokenizer = AutoTokenizer.from_pretrained(model_id)
    #     inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)
    #
    #     output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    #     output_text = tokenizer.batch_decode(output, skip_special_tokens=True)
    #
    #     self.assertEqual(output_text, EXPECTED_TEXTS)
    #
    # def test_model_2b_fp16_static_cache(self):
    #     model_id = "google/gemma-2b"
    #     EXPECTED_TEXTS = [
    #         "Hello I am doing a project on the 1990s and I need to know what the most popular music",
    #         "Hi today I am going to share with you a very easy and simple recipe of <strong><em>Kaju Kat",
    #     ]
    #
    #     model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, torch_dtype=torch.float16).to(
    #         torch_device
    #     )
    #
    #     model.generation_config.cache_implementation = "static"
    #
    #     tokenizer = AutoTokenizer.from_pretrained(model_id)
    #     inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)
    #
    #     output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    #     output_text = tokenizer.batch_decode(output, skip_special_tokens=True)
    #
    #     self.assertEqual(output_text, EXPECTED_TEXTS)

    def test_model_2b_bf16(self):
        model_id = "google/recurrentgemma-2b"
        EXPECTED_TEXTS = [
            "Hello I am doing a project on the 1990s and I need to know what the most popular music",
            "Hi today I am going to share with you a very easy and simple recipe of <strong><em>Khichdi",
        ]

        model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16).to(
            torch_device
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    def test_model_2b_eager(self):
        model_id = "google/recurrentgemma-2b"
        EXPECTED_TEXTS = [
            "Hello I am doing a project on the 1990s and I am looking for some information on the ",
            "Hi today I am going to share with you a very easy and simple recipe of <strong><em>Kaju Kat",
        ]

        model = AutoModelForCausalLM.from_pretrained(
            model_id, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, attn_implementation="eager"
        )
        model.to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)
