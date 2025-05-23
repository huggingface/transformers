# Copyright 2025 HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
"""Testing suite for the PyTorch nGPT model."""

import tempfile
import unittest

import pytest

from transformers import NGPTConfig, is_torch_available
from transformers.testing_utils import (
    Expectations,
    is_flaky,
    require_flash_attn,
    require_read_token,
    require_torch,
    require_torch_accelerator,
    require_torch_gpu,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import AutoTokenizer, NGPTForCausalLM, NGPTModel


class NGPTModelTester:
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
        return NGPTConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=False,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
        )

    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = NGPTModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

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
class NGPTModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            NGPTModel,
            NGPTForCausalLM,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": NGPTModel,
            "text-generation": NGPTForCausalLM,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    fx_compatible = False  # Broken by attention refactor cc @Cyrilvallez

    # Need to use `0.8` instead of `0.9` for `test_cpu_offload`
    # This is because we are hitting edge cases with the causal_mask buffer
    model_split_percents = [0.5, 0.7, 0.8]

    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = NGPTForCausalLM if is_torch_available() else None

    def setUp(self):
        self.model_tester = NGPTModelTester(self)
        self.config_tester = ConfigTester(self, config_class=NGPTConfig, hidden_size=37)

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

    @unittest.skip("Eager and SDPA do not produce the same outputs, thus this test fails")
    def test_model_outputs_equivalence(self, **kwargs):
        pass

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

                # ngpt flash attention 2 needs a high tolerance
                assert torch.allclose(logits_fa, logits, atol=1e-2)


@require_torch_accelerator
class NGPTIntegrationTest(unittest.TestCase):
    # This variable is used to determine which CUDA device are we using for our runners (A10 or T4)
    # Depending on the hardware we get different logits / generations
    cuda_compute_capability_major_version = None

    @classmethod
    def setUpClass(cls):
        if is_torch_available() and torch.cuda.is_available():
            # 8 is for A100 / A10 and 7 for T4
            cls.cuda_compute_capability_major_version = torch.cuda.get_device_capability()[0]

    @slow
    @require_read_token
    def test_ngpt_8b_generation_sdpa(self):
        text = ["What is the largest planet in solar system?"]
        EXPECTED_TEXT = [
            "What is the largest planet in solar system?\nAnswer: Jupiter\n\nWhat is the answer",
        ]
        model_id = "nvidia/Normalized-Nemotron-8B-Reasoning"
        model = NGPTForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto", attn_implementation="sdpa"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(text, return_tensors="pt").to(torch_device)

        output = model.generate(**inputs, do_sample=False, max_new_tokens=10)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT, output_text)

    @slow
    @require_read_token
    def test_ngpt_8b_generation_eager(self):
        text = ["What is the largest planet in solar system?"]
        EXPECTED_TEXTS = Expectations(
            {
                ("xpu", 3): [
                    "What is the largest planet in solar system?\nAnswer: Jupiter\n\nWhat is the answer: What is the name of the 19",
                ],
                ("cuda", 7): [
                    "What is the largest planet in solar system?\nAnswer: Jupiter\n\nWhat is the answer",
                ],
            }
        )
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        model_id = "nvidia/Normalized-Nemotron-8B-Reasoning"
        model = NGPTForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto", attn_implementation="eager"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(text, return_tensors="pt").to(torch_device)

        output = model.generate(**inputs, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT, output_text)

    @slow
    @require_read_token
    def test_ngpt_8b_generation_fa2(self):
        text = ["What is the largest planet in solar system?"]
        EXPECTED_TEXT = [
            "What is the largest planet in solar system?\nAnswer: Jupiter\n\nWhat is the answer",
        ]
        model_id = "nvidia/Normalized-Nemotron-8B-Reasoning"
        model = NGPTForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto", attn_implementation="flash_attention_2"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(text, return_tensors="pt").to(torch_device)

        output = model.generate(**inputs, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT, output_text)
