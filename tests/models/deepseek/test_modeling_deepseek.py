# coding=utf-8
# Copyright 2024 BigCode and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch deepseek model."""

import tempfile
import unittest

import pytest

from transformers import DeepseekConfig, is_torch_available
from transformers.testing_utils import (
    require_bitsandbytes,
    require_flash_attn,
    require_torch,
    require_torch_gpu,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        AutoTokenizer,
        DeepseekForCausalLM,
        DeepseekForQuestionAnswering,
        DeepseekForSequenceClassification,
        DeepseekForTokenClassification,
        DeepseekModel,
    )


class DeepseekModelTester:
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
        moe_intermediate_size=5,
        n_shared_experts=2,
        n_routed_experts=8,
        attention_dropout=0.1,
        max_position_embeddings=512,
        num_experts_per_tok=4,
        moe_layer_freq=1,
        first_k_dense_replace=0,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        aux_loss_alpha=0.001,
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
        self.moe_intermediate_size = moe_intermediate_size
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_layer_freq = moe_layer_freq
        self.first_k_dense_replace = first_k_dense_replace
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.aux_loss_alpha = aux_loss_alpha
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.pad_token_id = pad_token_id
        self.scope = scope

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.prepare_config_and_inputs
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
        return DeepseekConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            moe_intermediate_size=self.moe_intermediate_size,
            n_shared_experts=self.n_shared_experts,
            n_routed_experts=self.n_routed_experts,
            attention_dropout=self.attention_dropout,
            max_position_embeddings=self.max_position_embeddings,
            num_experts_per_tok=self.num_experts_per_tok,
            moe_layer_freq=self.moe_layer_freq,
            first_k_dense_replace=self.first_k_dense_replace,
            initializer_range=self.initializer_range,
            aux_loss_alpha=self.aux_loss_alpha,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.pad_token_id,
            bos_token_id=self.pad_token_id,
        )

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.create_and_check_model with Llama->Deepseek
    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = DeepseekModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.create_and_check_model_as_decoder with Llama->Deepseek
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
        model = DeepseekModel(config)
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

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.create_and_check_for_causal_lm with Llama->Deepseek
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
        model = DeepseekForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.create_and_check_decoder_model_past_large_inputs with Llama->Deepseek
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
        model = DeepseekForCausalLM(config=config)
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

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.prepare_config_and_inputs_for_common
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
# Copied from transformers.tests.models.llama.test_modeling_llama.LlamaModelTest with Llama->Deepseek, llama->deepseek
class DeepseekModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            DeepseekModel,
            DeepseekForCausalLM,
            DeepseekForSequenceClassification,
            DeepseekForQuestionAnswering,
            DeepseekForTokenClassification,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (DeepseekForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": DeepseekModel,
            "text-classification": DeepseekForSequenceClassification,
            "text-generation": DeepseekForCausalLM,
            "zero-shot": DeepseekForSequenceClassification,
            "question-answering": DeepseekForQuestionAnswering,
            "token-classification": DeepseekForTokenClassification,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False

    def setUp(self):
        self.model_tester = DeepseekModelTester(self)
        self.config_tester = ConfigTester(self, config_class=DeepseekConfig, hidden_size=37)

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

    def test_initialization(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if "mlp.gate.weight" in name:
                    self.assertLess(
                        param.data.mean().abs().item(),
                        5.0,
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )
                elif param.requires_grad:
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )

    def test_Deepseek_sequence_classification_model(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        print(config)
        config.num_labels = 3
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = DeepseekForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(
            result.logits.shape,
            (self.model_tester.batch_size, self.model_tester.num_labels),
        )

    def test_Deepseek_sequence_classification_model_for_single_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "single_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = DeepseekForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(
            result.logits.shape,
            (self.model_tester.batch_size, self.model_tester.num_labels),
        )

    def test_Deepseek_sequence_classification_model_for_multi_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "multi_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor(
            [self.model_tester.batch_size, config.num_labels],
            self.model_tester.type_sequence_label_size,
        ).to(torch.float)
        model = DeepseekForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(
            result.logits.shape,
            (self.model_tester.batch_size, self.model_tester.num_labels),
        )

    def test_deepseek_token_classification_model(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        token_labels = ids_tensor(
            [self.model_tester.batch_size, self.model_tester.seq_length],
            config.num_labels,
        )
        model = DeepseekForTokenClassification(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=token_labels)
        self.assertEqual(
            result.logits.shape,
            (
                self.model_tester.batch_size,
                self.model_tester.seq_length,
                self.model_tester.num_labels,
            ),
        )

    @require_flash_attn
    @require_torch_gpu
    @pytest.mark.flash_attn_test
    @slow
    def test_flash_attn_2_generate_padding_right(self):
        import torch

        for model_class in self.all_generative_model_classes:
            config, _ = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model = model_class.from_pretrained(tmpdirname, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(
                    torch_device
                )

                dummy_input = torch.LongTensor([[0, 2, 3, 4], [0, 2, 3, 4]]).to(torch_device)
                dummy_attention_mask = torch.LongTensor([[1, 1, 1, 1], [1, 1, 1, 0]]).to(torch_device)

                model.generate(
                    dummy_input,
                    attention_mask=dummy_attention_mask,
                    max_new_tokens=1,
                    do_sample=False,
                )

                model = model_class.from_pretrained(
                    tmpdirname,
                    torch_dtype=torch.float16,
                    attn_implementation="flash_attention_2",
                    low_cpu_mem_usage=True,
                ).to(torch_device)

                with self.assertRaises(ValueError):
                    _ = model.generate(
                        dummy_input,
                        attention_mask=dummy_attention_mask,
                        max_new_tokens=1,
                        do_sample=False,
                    )

    @require_flash_attn
    @require_torch_gpu
    @pytest.mark.flash_attn_test
    @slow
    def test_flash_attn_2_generate_use_cache(self):
        import torch

        max_new_tokens = 30

        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            dummy_input = inputs_dict[model_class.main_input_name]
            if dummy_input.dtype in [torch.float32, torch.bfloat16]:
                dummy_input = dummy_input.to(torch.float16)

            # make sure that all models have enough positions for generation
            if hasattr(config, "max_position_embeddings"):
                config.max_position_embeddings = max_new_tokens + dummy_input.shape[1] + 1

            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                dummy_attention_mask = inputs_dict.get("attention_mask", torch.ones_like(dummy_input))
                # NOTE: Deepseek apparently does not support right padding + use_cache with FA2.
                dummy_attention_mask[:, -1] = 1

                model = model_class.from_pretrained(
                    tmpdirname,
                    torch_dtype=torch.float16,
                    attn_implementation="flash_attention_2",
                    low_cpu_mem_usage=True,
                ).to(torch_device)

                # Just test that a large cache works as expected
                _ = model.generate(
                    dummy_input,
                    attention_mask=dummy_attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )


@slow
@require_torch_gpu  # NotImplementedError: The operator 'aten::scatter_reduce.two_out' is not currently implemented for the MPS device.
class DeepseekIntegrationTest(unittest.TestCase):
    @require_bitsandbytes
    def test_deepseek_batched_generation_4bit(self):
        EXPECTED_TEXT = [
            "Mixture-of-experts is “a model that uses multiple neural networks, each "
            "specialized to perform a specific task, to solve a complex problem.”\n"
            "\n"
            "In other words, it’s a way to use multiple neural networks",
            "DeepSeek is a Chinese search engine that uses AI to provide users with more "
            "accurate and relevant search results. It is designed to compete with other "
            "major search engines such as Baidu and Google.\n"
            "DeepSeek uses a",
        ]
        model_id = "deepseek-ai/deepseek-moe-16b-chat"

        model = DeepseekForCausalLM.from_pretrained(model_id, load_in_4bit=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token

        text = ["Mixture-of-experts is ", "DeepSeek is"]
        inputs = tokenizer(text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT, output_text)

    @require_bitsandbytes
    def test_deepseek_batched_generation_4bit_sdpa(self):
        EXPECTED_TEXT = [
            "Mixture-of-experts is “a model that uses multiple neural networks, each "
            "specialized to perform a specific task, to solve a complex problem.”\n"
            "\n"
            "In other words, it’s a way to use multiple neural networks",
            "DeepSeek is a Chinese search engine that uses AI to provide users with more "
            "accurate and relevant search results. It is designed to compete with other "
            "major search engines such as Baidu and Google.\n"
            "DeepSeek uses a",
        ]
        model_id = "deepseek-ai/deepseek-moe-16b-chat"

        model = DeepseekForCausalLM.from_pretrained(model_id, load_in_4bit=True, attn_implementation="sdpa")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token

        text = ["Mixture-of-experts is ", "DeepSeek is"]
        inputs = tokenizer(text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT, output_text)

    @require_bitsandbytes
    @require_flash_attn
    @pytest.mark.flash_attn_test
    def test_deepseek_batched_generation_4bit_fa2(self):
        EXPECTED_TEXT = [
            "Mixture-of-experts is “a model that uses multiple neural networks, each "
            "specialized to perform a specific task, to solve a complex problem.”\n"
            "\n"
            "In other words, it’s a way to use multiple neural networks",
            "DeepSeek is a Chinese search engine that uses AI to provide users with more "
            "accurate and relevant search results. It is designed to compete with other "
            "major search engines such as Baidu and Google.\n"
            "DeepSeek uses a",
        ]
        model_id = "deepseek-ai/deepseek-moe-16b-chat"

        model = DeepseekForCausalLM.from_pretrained(
            model_id, load_in_4bit=True, attn_implementation="flash_attention_2"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token

        text = ["Mixture-of-experts is ", "DeepSeek is"]
        inputs = tokenizer(text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT, output_text)
