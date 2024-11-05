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
"""Testing suite for the PyTorch Gemma model."""

import tempfile
import unittest

import pytest
from packaging import version

from transformers import AutoModelForCausalLM, AutoTokenizer, GemmaConfig, is_torch_available
from transformers.generation.configuration_utils import GenerationConfig
from transformers.testing_utils import (
    is_flaky,
    require_bitsandbytes,
    require_flash_attn,
    require_read_token,
    require_torch,
    require_torch_accelerator,
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

    from transformers import (
        GemmaForCausalLM,
        GemmaForSequenceClassification,
        GemmaForTokenClassification,
        GemmaModel,
    )


@require_torch
class GemmaModelTester:
    config_class = GemmaConfig
    if is_torch_available():
        model_class = GemmaModel
        for_causal_lm_class = GemmaForCausalLM
        for_sequence_class = GemmaForSequenceClassification
        for_token_class = GemmaForTokenClassification

    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=False,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
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
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.pad_token_id = pad_token_id
        self.scope = scope
        self.head_dim = self.hidden_size // self.num_attention_heads

    # Copied from tests.models.mistral.test_modeling_mistral.MistralModelTester.prepare_config_and_inputs
    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = torch.tril(torch.ones_like(input_ids).to(torch_device))

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

    def get_config(self):
        return self.config_class(
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
            pad_token_id=self.pad_token_id,
            head_dim=self.head_dim,
        )

    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = self.model_class(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

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
        model = self.model_class(config)
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
        model = self.for_causal_lm_class(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

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
        model = self.for_causal_lm_class(config=config)
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
class GemmaModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (GemmaModel, GemmaForCausalLM, GemmaForSequenceClassification, GemmaForTokenClassification)
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (GemmaForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": GemmaModel,
            "text-classification": GemmaForSequenceClassification,
            "token-classification": GemmaForTokenClassification,
            "text-generation": GemmaForCausalLM,
            "zero-shot": GemmaForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False

    # Need to remove 0.9 in `test_cpu_offload`
    # This is because we are hitting edge cases with the causal_mask buffer
    model_split_percents = [0.5, 0.6]

    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = GemmaForCausalLM if is_torch_available() else None

    # TODO (ydshieh): Check this. See https://app.circleci.com/pipelines/github/huggingface/transformers/79245/workflows/9490ef58-79c2-410d-8f51-e3495156cf9c/jobs/1012146
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
        return True

    def setUp(self):
        self.model_tester = GemmaModelTester(self)
        self.config_tester = ConfigTester(self, config_class=GemmaConfig, hidden_size=37)

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

    def test_Gemma_sequence_classification_model(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        print(config)
        config.num_labels = 3
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = self.model_tester.for_sequence_class(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_Gemma_sequence_classification_model_for_single_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "single_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = self.model_tester.for_sequence_class(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_Gemma_sequence_classification_model_for_multi_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "multi_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor(
            [self.model_tester.batch_size, config.num_labels], self.model_tester.type_sequence_label_size
        ).to(torch.float)
        model = self.model_tester.for_sequence_class(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_Gemma_token_classification_model(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        token_labels = ids_tensor([self.model_tester.batch_size, self.model_tester.seq_length], config.num_labels)
        model = self.model_tester.for_token_class(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=token_labels)
        self.assertEqual(
            result.logits.shape,
            (self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.num_labels),
        )

    @unittest.skip(reason="Gemma buffers include complex numbers, which breaks this test")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="Gemma uses GQA on all models so the KV cache is a non standard format")
    def test_past_key_values_format(self):
        pass

    @require_flash_attn
    @require_torch_gpu
    @pytest.mark.flash_attn_test
    @slow
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        self.skipTest(reason="Gemma flash attention does not support right padding")

    @require_torch_sdpa
    @require_torch_accelerator
    @slow
    def test_sdpa_equivalence(self):
        for model_class in self.all_model_classes:
            if not model_class._supports_sdpa:
                self.skipTest(reason="Model does not support SDPA")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_sdpa = model_class.from_pretrained(
                    tmpdirname, torch_dtype=torch.float16, attn_implementation="sdpa"
                )
                model_sdpa.to(torch_device)

                model = model_class.from_pretrained(tmpdirname, torch_dtype=torch.float16, attn_implementation="eager")
                model.to(torch_device)

                dummy_input = inputs_dict[model_class.main_input_name]
                dummy_input = dummy_input.to(torch_device)
                outputs = model(dummy_input, output_hidden_states=True)
                outputs_sdpa = model_sdpa(dummy_input, output_hidden_states=True)

                logits = outputs.hidden_states[-1]
                logits_sdpa = outputs_sdpa.hidden_states[-1]

                # gemma sdpa needs a high tolerance
                assert torch.allclose(logits_sdpa, logits, atol=3e-3)

    @require_flash_attn
    @require_torch_gpu
    @pytest.mark.flash_attn_test
    @is_flaky()
    @slow
    def test_flash_attn_2_equivalence(self):
        for model_class in self.all_model_classes:
            if not model_class._supports_flash_attn_2:
                self.skipTest(reason="Model does not support Flash Attention 2")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_fa = model_class.from_pretrained(
                    tmpdirname, torch_dtype=torch.float16, attn_implementation="flash_attention_2"
                )
                model_fa.to(torch_device)

                model = model_class.from_pretrained(tmpdirname, torch_dtype=torch.float16, attn_implementation="eager")
                model.to(torch_device)

                dummy_input = inputs_dict[model_class.main_input_name]
                dummy_input = dummy_input.to(torch_device)
                outputs = model(dummy_input, output_hidden_states=True)
                outputs_fa = model_fa(dummy_input, output_hidden_states=True)

                logits = outputs.hidden_states[-1]
                logits_fa = outputs_fa.hidden_states[-1]

                # gemma flash attention 2 needs a high tolerance
                assert torch.allclose(logits_fa, logits, atol=3e-3)


@slow
@require_torch_accelerator
class GemmaIntegrationTest(unittest.TestCase):
    input_text = ["Hello I am doing", "Hi today"]
    # This variable is used to determine which CUDA device are we using for our runners (A10 or T4)
    # Depending on the hardware we get different logits / generations
    cuda_compute_capability_major_version = None

    @classmethod
    def setUpClass(cls):
        if is_torch_available() and torch.cuda.is_available():
            # 8 is for A100 / A10 and 7 for T4
            cls.cuda_compute_capability_major_version = torch.cuda.get_device_capability()[0]

    @require_read_token
    def test_model_2b_fp16(self):
        model_id = "google/gemma-2b"
        EXPECTED_TEXTS = [
            "Hello I am doing a project on the 1990s and I need to know what the most popular music",
            "Hi today I am going to share with you a very easy and simple recipe of <strong><em>Kaju Kat",
        ]

        model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, torch_dtype=torch.float16).to(
            torch_device
        )

        model.generation_config.cache_implementation = "static"

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    @require_read_token
    def test_model_2b_bf16(self):
        model_id = "google/gemma-2b"

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

    @require_read_token
    def test_model_2b_eager(self):
        model_id = "google/gemma-2b"

        EXPECTED_TEXTS = [
            "Hello I am doing a project on the 1990s and I need to know what the most popular music",
            "Hi today I am going to share with you a very easy and simple recipe of <strong><em>Khichdi",
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

    @require_torch_sdpa
    @require_read_token
    def test_model_2b_sdpa(self):
        model_id = "google/gemma-2b"

        EXPECTED_TEXTS = [
            "Hello I am doing a project on the 1990s and I need to know what the most popular music",
            "Hi today I am going to share with you a very easy and simple recipe of <strong><em>Khichdi",
        ]

        model = AutoModelForCausalLM.from_pretrained(
            model_id, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        model.to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    @require_flash_attn
    @require_read_token
    @pytest.mark.flash_attn_test
    def test_model_2b_flash_attn(self):
        model_id = "google/gemma-2b"
        EXPECTED_TEXTS = [
            "Hello I am doing a project on the 1990s and I need to know what the most popular music",
            "Hi today I am going to share with you a very easy and simple recipe of <strong><em>Kaju Kat",
        ]

        model = AutoModelForCausalLM.from_pretrained(
            model_id, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
        model.to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    @require_bitsandbytes
    @require_read_token
    def test_model_2b_4bit(self):
        model_id = "google/gemma-2b"
        EXPECTED_TEXTS = [
            "Hello I am doing a project and I need to make a 3d model of a house. I have been using",
            "Hi today I'd like to share with you my experience with the new wattpad wattpad wattpad wattpad wattpad wattpad wattpad",
        ]

        model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, load_in_4bit=True)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    @unittest.skip(reason="The test will not fit our CI runners")
    @require_read_token
    def test_model_7b_fp32(self):
        model_id = "google/gemma-7b"
        EXPECTED_TEXTS = [
            "Hello my name is ***** ***** I will be assisting you today. I am sorry to hear about your issue. I will",
            "Hi,\n\nI have a problem with my 2005 1.6 16",
        ]

        model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True).to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    @require_read_token
    def test_model_7b_fp16(self):
        if self.cuda_compute_capability_major_version == 7:
            self.skipTest("This test is failing (`torch.compile` fails) on Nvidia T4 GPU (OOM).")

        model_id = "google/gemma-7b"
        EXPECTED_TEXTS = [
            """Hello I am doing a project on a 1999 4.0L 4x4. I""",
            "Hi today I am going to show you how to make a simple and easy to make a DIY 3D",
        ]

        model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, torch_dtype=torch.float16).to(
            torch_device
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    @require_read_token
    def test_model_7b_bf16(self):
        if self.cuda_compute_capability_major_version == 7:
            self.skipTest("This test is failing (`torch.compile` fails) on Nvidia T4 GPU (OOM).")

        model_id = "google/gemma-7b"

        # Key 9 for MI300, Key 8 for A100/A10, and Key 7 for T4.
        #
        # Note: Key 9 is currently set for MI300, but may need potential future adjustments for H100s,
        # considering differences in hardware processing and potential deviations in generated text.
        EXPECTED_TEXTS = {
            7: [
                """Hello I am doing a project on a 1991 240sx and I am trying to find""",
                "Hi today I am going to show you how to make a very simple and easy to make a very simple and",
            ],
            8: [
                "Hello I am doing a project for my school and I am trying to make a program that will read a .txt file",
                "Hi today I am going to show you how to make a very simple and easy to make a very simple and",
            ],
            9: [
                "Hello I am doing a project for my school and I am trying to get a servo to move a certain amount of degrees",
                "Hi today I am going to show you how to make a very simple and easy to make DIY light up sign",
            ],
        }

        model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16).to(
            torch_device
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(output_text, EXPECTED_TEXTS[self.cuda_compute_capability_major_version])

    @require_read_token
    def test_model_7b_fp16_static_cache(self):
        if self.cuda_compute_capability_major_version == 7:
            self.skipTest("This test is failing (`torch.compile` fails) on Nvidia T4 GPU (OOM).")

        model_id = "google/gemma-7b"
        EXPECTED_TEXTS = [
            """Hello I am doing a project on a 1999 4.0L 4x4. I""",
            "Hi today I am going to show you how to make a simple and easy to make a DIY 3D",
        ]

        model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, torch_dtype=torch.float16).to(
            torch_device
        )

        model.generation_config.cache_implementation = "static"

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)
        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(output_text, EXPECTED_TEXTS)

    @require_bitsandbytes
    @require_read_token
    def test_model_7b_4bit(self):
        model_id = "google/gemma-7b"
        EXPECTED_TEXTS = [
            "Hello I am doing a project for my school and I am trying to make a program that will take a number and then",
            "Hi today I am going to talk about the best way to get rid of acne. miniaturing is a very",
        ]

        model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, load_in_4bit=True)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(output_text, EXPECTED_TEXTS)

    @slow
    @require_torch_gpu
    @require_read_token
    def test_compile_static_cache(self):
        # `torch==2.2` will throw an error on this test (as in other compilation tests), but torch==2.1.2 and torch>2.2
        # work as intended. See https://github.com/pytorch/pytorch/issues/121943
        if version.parse(torch.__version__) < version.parse("2.3.0"):
            self.skipTest(reason="This test requires torch >= 2.3 to run.")

        NUM_TOKENS_TO_GENERATE = 40
        EXPECTED_TEXT_COMPLETION = [
            "Hello I am doing a project on the 1990s and I need to know what the most popular music was in the 1990s. I have looked on the internet and I have found",
            "Hi today\nI have a problem with my 2007 1.9 tdi 105bhp.\nI have a problem with the engine management light on.\nI have checked the",
        ]

        prompts = ["Hello I am doing", "Hi today"]
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b", pad_token="</s>", padding_side="right")
        model = GemmaForCausalLM.from_pretrained("google/gemma-2b", device_map=torch_device, torch_dtype=torch.float16)
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        # Dynamic Cache
        generated_ids = model.generate(**inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False)
        dynamic_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, dynamic_text)  # Both GPU architectures have the same output

        # Static Cache
        generated_ids = model.generate(
            **inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False, cache_implementation="static"
        )
        static_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, static_text)

        # Static Cache + compile
        model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
        generated_ids = model.generate(
            **inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False, cache_implementation="static"
        )
        static_compiled_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, static_compiled_text)

    @slow
    @require_read_token
    def test_export_static_cache(self):
        if version.parse(torch.__version__) < version.parse("2.3.0"):
            self.skipTest(reason="This test requires torch >= 2.3 to run.")

        from transformers.integrations.executorch import (
            TorchExportableModuleWithStaticCache,
            convert_and_export_with_cache,
        )

        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b", pad_token="</s>", padding_side="right")
        EXPECTED_TEXT_COMPLETION = [
            "Hello I am doing a project on the 1990s and I need to know what the most popular music was in the 1990s. I have looked on the internet and I have found",
        ]
        max_generation_length = tokenizer(EXPECTED_TEXT_COMPLETION, return_tensors="pt", padding=True)[
            "input_ids"
        ].shape[-1]

        # Load model
        device = "cpu"
        dtype = torch.bfloat16
        cache_implementation = "static"
        attn_implementation = "sdpa"
        batch_size = 1
        model = GemmaForCausalLM.from_pretrained(
            "google/gemma-2b",
            device_map=device,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
            generation_config=GenerationConfig(
                use_cache=True,
                cache_implementation=cache_implementation,
                max_length=max_generation_length,
                cache_config={
                    "batch_size": batch_size,
                    "max_cache_len": max_generation_length,
                },
            ),
        )

        prompts = ["Hello I am doing"]
        prompt_tokens = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        prompt_token_ids = prompt_tokens["input_ids"]
        max_new_tokens = max_generation_length - prompt_token_ids.shape[-1]

        # Static Cache + eager
        eager_generated_ids = model.generate(
            **prompt_tokens, max_new_tokens=max_new_tokens, do_sample=False, cache_implementation=cache_implementation
        )
        eager_generated_text = tokenizer.batch_decode(eager_generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, eager_generated_text)

        # Static Cache + export
        exported_program = convert_and_export_with_cache(model)
        ep_generated_ids = TorchExportableModuleWithStaticCache.generate(
            exported_program=exported_program, prompt_token_ids=prompt_token_ids, max_new_tokens=max_new_tokens
        )
        ep_generated_text = tokenizer.batch_decode(ep_generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, ep_generated_text)

    def test_model_2b_bf16_dola(self):
        model_id = "google/gemma-2b"
        # ground truth text generated with dola_layers="low", repetition_penalty=1.2
        EXPECTED_TEXTS = [
            "Hello I am doing an experiment and need to get the mass of a block. The problem is, it has no scale",
            "Hi today we have the review for a <strong>2016/2017</strong> season of",
        ]

        model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16).to(
            torch_device
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(
            **inputs, max_new_tokens=20, do_sample=False, dola_layers="low", repetition_penalty=1.2
        )
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(output_text, EXPECTED_TEXTS)
