# coding=utf-8
# Copyright 2025 the Fromthesky Research Labs, LLC. All rights reserved.
# Copyright 2025 the HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch PLDR-LLM model."""

import pytest
import unittest

from packaging import version

from transformers import AutoTokenizer, StaticCache, is_torch_available
from transformers.generation.configuration_utils import GenerationConfig
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_read_token,
    require_torch,
    require_torch_accelerator,
    run_test_using_subprocess,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        PldrllmConfig,
        PldrllmForCausalLM,
        PldrllmForQuestionAnswering,
        PldrllmForSequenceClassification,
        PldrllmForTokenClassification,
        PldrllmModel,
        LlamaTokenizerFast
    )
    from transformers.models.pldrllm.modeling_pldrllm import PldrllmRotaryEmbedding


class PldrllmModelTester_G_NONE(CausalLMModelTester):
    if is_torch_available():
        config_class = PldrllmConfig
        base_model_class = PldrllmModel
        causal_lm_class = PldrllmForCausalLM
        sequence_class = PldrllmForSequenceClassification
        token_class = PldrllmForTokenClassification
    
    def __init__(self, parent, batch_size=13, seq_length=7, is_training=True, 
                 use_input_mask=True, use_token_type_ids=False, use_labels=True, 
                 vocab_size=99, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, 
                 num_key_value_heads=2, intermediate_size=37, hidden_act="silu", hidden_dropout_prob=0.0, 
                 attention_probs_dropout_prob=0.0, max_position_embeddings=512, type_vocab_size=16, 
                 type_sequence_label_size=2, initializer_range=0.02, num_labels=3, num_choices=4, 
                 pad_token_id=0, bos_token_id=2, eos_token_id=3, is_decoder=True, scope=None, expert_interval=1, 
                 moe_layer_start_index=0, moe_intermediate_size=12, shared_expert_intermediate_size=36, shared_expert_gate=True, 
                 moe_num_shared_experts=2, num_experts_per_tok=2, num_experts=8, mamba_n_groups=1, mamba_n_heads=16, mamba_d_state=16, 
                 mamba_d_conv=4, mamba_expand=2, mamba_chunk_size=16,
                 reference_rope=False, custom_G_type=None, cache_first_G=False):
        super().__init__(parent, batch_size, seq_length, is_training, use_input_mask, use_token_type_ids, use_labels, vocab_size, hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads, intermediate_size, hidden_act, hidden_dropout_prob, attention_probs_dropout_prob, max_position_embeddings, type_vocab_size, type_sequence_label_size, initializer_range, num_labels, num_choices, pad_token_id, bos_token_id, eos_token_id, is_decoder, scope, expert_interval, moe_layer_start_index, moe_intermediate_size, shared_expert_intermediate_size, shared_expert_gate, moe_num_shared_experts, num_experts_per_tok, num_experts, mamba_n_groups, mamba_n_heads, mamba_d_state, mamba_d_conv, mamba_expand, mamba_chunk_size)
        self.reference_rope=reference_rope
        self.custom_G_type=custom_G_type
        self.cache_first_G=cache_first_G


@require_torch
class PldrllmModelTest_G_NONE(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (
            PldrllmModel,
            PldrllmForCausalLM,
            PldrllmForSequenceClassification,
            PldrllmForQuestionAnswering,
            PldrllmForTokenClassification,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": PldrllmModel,
            "text-classification": PldrllmForSequenceClassification,
            "text-generation": PldrllmForCausalLM,
            "zero-shot": PldrllmForSequenceClassification,
            "question-answering": PldrllmForQuestionAnswering,
            "token-classification": PldrllmForTokenClassification,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    fx_compatible = False  # Broken by attention refactor cc @Cyrilvallez
    model_tester_class = PldrllmModelTester_G_NONE
    rotary_embedding_layer = PldrllmRotaryEmbedding  # Enables RoPE tests if set

    # Need to use `0.8` instead of `0.9` for `test_cpu_offload`
    # This is because we are hitting edge cases with the causal_mask buffer
    model_split_percents = [0.5, 0.7, 0.8]

    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = PldrllmForCausalLM if is_torch_available() else None

class PldrllmModelTester_G_NONE_cache_first_G(CausalLMModelTester):
    if is_torch_available():
        config_class = PldrllmConfig
        base_model_class = PldrllmModel
        causal_lm_class = PldrllmForCausalLM
        sequence_class = PldrllmForSequenceClassification
        token_class = PldrllmForTokenClassification
    
    def __init__(self, parent, batch_size=13, seq_length=7, is_training=True, 
                 use_input_mask=True, use_token_type_ids=False, use_labels=True, 
                 vocab_size=99, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, 
                 num_key_value_heads=2, intermediate_size=37, hidden_act="silu", hidden_dropout_prob=0.0, 
                 attention_probs_dropout_prob=0.0, max_position_embeddings=512, type_vocab_size=16, 
                 type_sequence_label_size=2, initializer_range=0.02, num_labels=3, num_choices=4, 
                 pad_token_id=0, bos_token_id=2, eos_token_id=3, is_decoder=True, scope=None, expert_interval=1, 
                 moe_layer_start_index=0, moe_intermediate_size=12, shared_expert_intermediate_size=36, shared_expert_gate=True, 
                 moe_num_shared_experts=2, num_experts_per_tok=2, num_experts=8, mamba_n_groups=1, mamba_n_heads=16, mamba_d_state=16, 
                 mamba_d_conv=4, mamba_expand=2, mamba_chunk_size=16,
                 reference_rope=False, custom_G_type=None, cache_first_G=True):
        super().__init__(parent, batch_size, seq_length, is_training, use_input_mask, use_token_type_ids, use_labels, vocab_size, hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads, intermediate_size, hidden_act, hidden_dropout_prob, attention_probs_dropout_prob, max_position_embeddings, type_vocab_size, type_sequence_label_size, initializer_range, num_labels, num_choices, pad_token_id, bos_token_id, eos_token_id, is_decoder, scope, expert_interval, moe_layer_start_index, moe_intermediate_size, shared_expert_intermediate_size, shared_expert_gate, moe_num_shared_experts, num_experts_per_tok, num_experts, mamba_n_groups, mamba_n_heads, mamba_d_state, mamba_d_conv, mamba_expand, mamba_chunk_size)
        self.reference_rope=reference_rope
        self.custom_G_type=custom_G_type
        self.cache_first_G=cache_first_G


@require_torch
class PldrllmModelTest_G_NONE_cache_first_G(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (
            PldrllmModel,
            PldrllmForCausalLM,
            PldrllmForSequenceClassification,
            PldrllmForQuestionAnswering,
            PldrllmForTokenClassification,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": PldrllmModel,
            "text-classification": PldrllmForSequenceClassification,
            "text-generation": PldrllmForCausalLM,
            "zero-shot": PldrllmForSequenceClassification,
            "question-answering": PldrllmForQuestionAnswering,
            "token-classification": PldrllmForTokenClassification,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    fx_compatible = False  # Broken by attention refactor cc @Cyrilvallez
    model_tester_class = PldrllmModelTester_G_NONE_cache_first_G
    rotary_embedding_layer = PldrllmRotaryEmbedding  # Enables RoPE tests if set

    # Need to use `0.8` instead of `0.9` for `test_cpu_offload`
    # This is because we are hitting edge cases with the causal_mask buffer
    model_split_percents = [0.5, 0.7, 0.8]

    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = PldrllmForCausalLM if is_torch_available() else None


class PldrllmModelTester_G_IDENTITY(CausalLMModelTester):
    if is_torch_available():
        config_class = PldrllmConfig
        base_model_class = PldrllmModel
        causal_lm_class = PldrllmForCausalLM
        sequence_class = PldrllmForSequenceClassification
        token_class = PldrllmForTokenClassification
    
    def __init__(self, parent, batch_size=13, seq_length=7, is_training=True, 
                 use_input_mask=True, use_token_type_ids=False, use_labels=True, 
                 vocab_size=99, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, 
                 num_key_value_heads=2, intermediate_size=37, hidden_act="silu", hidden_dropout_prob=0.0, 
                 attention_probs_dropout_prob=0.0, max_position_embeddings=512, type_vocab_size=16, 
                 type_sequence_label_size=2, initializer_range=0.02, num_labels=3, num_choices=4, 
                 pad_token_id=0, bos_token_id=2, eos_token_id=3, is_decoder=True, scope=None, expert_interval=1, 
                 moe_layer_start_index=0, moe_intermediate_size=12, shared_expert_intermediate_size=36, shared_expert_gate=True, 
                 moe_num_shared_experts=2, num_experts_per_tok=2, num_experts=8, mamba_n_groups=1, mamba_n_heads=16, mamba_d_state=16, 
                 mamba_d_conv=4, mamba_expand=2, mamba_chunk_size=16,
                 reference_rope=False, custom_G_type='identity'):
        super().__init__(parent, batch_size, seq_length, is_training, use_input_mask, use_token_type_ids, use_labels, vocab_size, hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads, intermediate_size, hidden_act, hidden_dropout_prob, attention_probs_dropout_prob, max_position_embeddings, type_vocab_size, type_sequence_label_size, initializer_range, num_labels, num_choices, pad_token_id, bos_token_id, eos_token_id, is_decoder, scope, expert_interval, moe_layer_start_index, moe_intermediate_size, shared_expert_intermediate_size, shared_expert_gate, moe_num_shared_experts, num_experts_per_tok, num_experts, mamba_n_groups, mamba_n_heads, mamba_d_state, mamba_d_conv, mamba_expand, mamba_chunk_size)
        self.reference_rope=reference_rope
        self.custom_G_type=custom_G_type


@require_torch
class PldrllmModelTest_G_IDENTITY(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (
            PldrllmModel,
            PldrllmForCausalLM,
            PldrllmForSequenceClassification,
            PldrllmForQuestionAnswering,
            PldrllmForTokenClassification,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": PldrllmModel,
            "text-classification": PldrllmForSequenceClassification,
            "text-generation": PldrllmForCausalLM,
            "zero-shot": PldrllmForSequenceClassification,
            "question-answering": PldrllmForQuestionAnswering,
            "token-classification": PldrllmForTokenClassification,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    fx_compatible = False  # Broken by attention refactor cc @Cyrilvallez
    model_tester_class = PldrllmModelTester_G_IDENTITY
    rotary_embedding_layer = PldrllmRotaryEmbedding  # Enables RoPE tests if set

    # Need to use `0.8` instead of `0.9` for `test_cpu_offload`
    # This is because we are hitting edge cases with the causal_mask buffer
    model_split_percents = [0.5, 0.7, 0.8]

    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = PldrllmForCausalLM if is_torch_available() else None

class PldrllmModelTester_G_RANDOM(CausalLMModelTester):
    if is_torch_available():
        config_class = PldrllmConfig
        base_model_class = PldrllmModel
        causal_lm_class = PldrllmForCausalLM
        sequence_class = PldrllmForSequenceClassification
        token_class = PldrllmForTokenClassification
    
    def __init__(self, parent, batch_size=13, seq_length=7, is_training=True, 
                 use_input_mask=True, use_token_type_ids=False, use_labels=True, 
                 vocab_size=99, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, 
                 num_key_value_heads=2, intermediate_size=37, hidden_act="silu", hidden_dropout_prob=0.0, 
                 attention_probs_dropout_prob=0.0, max_position_embeddings=512, type_vocab_size=16, 
                 type_sequence_label_size=2, initializer_range=0.02, num_labels=3, num_choices=4, 
                 pad_token_id=0, bos_token_id=2, eos_token_id=3, is_decoder=True, scope=None, expert_interval=1, 
                 moe_layer_start_index=0, moe_intermediate_size=12, shared_expert_intermediate_size=36, shared_expert_gate=True, 
                 moe_num_shared_experts=2, num_experts_per_tok=2, num_experts=8, mamba_n_groups=1, mamba_n_heads=16, mamba_d_state=16, 
                 mamba_d_conv=4, mamba_expand=2, mamba_chunk_size=16,
                 reference_rope=False, custom_G_type='random'):
        super().__init__(parent, batch_size, seq_length, is_training, use_input_mask, use_token_type_ids, use_labels, vocab_size, hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads, intermediate_size, hidden_act, hidden_dropout_prob, attention_probs_dropout_prob, max_position_embeddings, type_vocab_size, type_sequence_label_size, initializer_range, num_labels, num_choices, pad_token_id, bos_token_id, eos_token_id, is_decoder, scope, expert_interval, moe_layer_start_index, moe_intermediate_size, shared_expert_intermediate_size, shared_expert_gate, moe_num_shared_experts, num_experts_per_tok, num_experts, mamba_n_groups, mamba_n_heads, mamba_d_state, mamba_d_conv, mamba_expand, mamba_chunk_size)
        self.reference_rope=reference_rope
        self.custom_G_type=custom_G_type


@require_torch
class PldrllmModelTest_G_RANDOM(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (
            PldrllmModel,
            PldrllmForCausalLM,
            PldrllmForSequenceClassification,
            PldrllmForQuestionAnswering,
            PldrllmForTokenClassification,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": PldrllmModel,
            "text-classification": PldrllmForSequenceClassification,
            "text-generation": PldrllmForCausalLM,
            "zero-shot": PldrllmForSequenceClassification,
            "question-answering": PldrllmForQuestionAnswering,
            "token-classification": PldrllmForTokenClassification,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    fx_compatible = False  # Broken by attention refactor cc @Cyrilvallez
    model_tester_class = PldrllmModelTester_G_RANDOM
    rotary_embedding_layer = PldrllmRotaryEmbedding  # Enables RoPE tests if set

    # Need to use `0.8` instead of `0.9` for `test_cpu_offload`
    # This is because we are hitting edge cases with the causal_mask buffer
    model_split_percents = [0.5, 0.7, 0.8]

    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = PldrllmForCausalLM if is_torch_available() else None


@require_torch_accelerator
@require_read_token
class PldrllmIntegrationTest(unittest.TestCase):
    def setup(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        # TODO (joao): automatic compilation, i.e. compilation when `cache_implementation="static"` is used, leaves
        # some memory allocated in the cache, which means some object is not being released properly. This causes some
        # unoptimal memory usage, e.g. after certain tests a 7B model in FP16 no longer fits in a 24GB GPU.
        # Investigate the root cause.
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_pldrllm_110M_hard(self):

        expected_texts = Expectations(
            {
                ("rocm", (9, 5)): "Tell me about the french revolution. I'm a french guy, and I'm a french guy. I'm a french guy. I'm a french guy. I'm a french guy. I'm a french guy. I'm a french guy. I'm a french guy. I'm a french guy. I'm a french guy. I'm a french guy. I'm a french guy. I'm a french guy. I'm a french guy. I'm a french guy. I'm a french guy. I'm a french guy. I'm a french guy. I",
                ("cuda", None): "Tell me about the french revolution. I'm a french guy, and I'm a french guy. I'm a french guy. I'm a french guy. I'm a french guy. I'm a french guy. I'm a french guy. I'm a french guy. I'm a french guy. I'm a french guy. I'm a french guy. I'm a french guy. I'm a french guy. I'm a french guy. I'm a french guy. I'm a french guy. I'm a french guy. I'm a french guy. I",
            }
        )  # fmt: skip
        EXPECTED_TEXT = expected_texts.get_expectation()
        model_path="fromthesky/PLDR-LLM-v51-110M-3"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = PldrllmForCausalLM.from_pretrained(
            model_path, device_map="auto", dtype=torch.bfloat16
        )
        input_text = ["Tell me about the french revolution."]
        model_inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=128, do_sample=False)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        self.assertEqual(generated_text, EXPECTED_TEXT)

    @slow
    def test_pldrllm_110M_logits_bf16(self):
        input_ids = [ 5273,   356,   330,   264, 13967,  6971,   262]
        model_path = "fromthesky/PLDR-LLM-v51-110M-3"

        model = PldrllmForCausalLM.from_pretrained(
            model_path, device_map="auto", dtype=torch.bfloat16, attn_implementation="eager"
        )

        with torch.no_grad():
            out = model(torch.tensor([input_ids]).to(torch_device))
        # Expected mean on dim = -1

        # fmt: off
        expected_means = Expectations(
            {
            ("xpu", 3): torch.tensor([[-13.5005, -17.2374, -18.3841, -12.8294,  -3.0489, -11.0871, -11.0553]]),
            ("cuda", 7): torch.tensor([[-13.4639, -17.1359, -18.4118, -12.7880,  -3.1017, -11.0843, -10.9833]]),
            ("cuda", 8): torch.tensor([[-13.4639, -17.1359, -18.4118, -12.7880,  -3.1017, -11.0843, -10.9833]]),
            # ("rocm", (9, 4)): torch.tensor([[-6.5094, -4.1329, -4.9754, -3.5042,  0.8082, -2.9443,  1.2830, -3.3539]]),
        })

        expected_mean = expected_means.get_expectation().to(torch_device)
        actual_mean = out.logits.float().mean(-1)
        self.assertTrue(
            torch.allclose(
                expected_mean,
                actual_mean,
                atol=1e-2,
                rtol=1e-2
            )
        )

        # slicing logits[0, 0, 0:15]
        expected_slices = Expectations(
            {
            ("xpu", 3): torch.tensor([[-14.8750, -14.8750, -14.8750,  -8.1875, -14.8750, -14.8750, -14.8750, -14.8750, -14.8750, -14.8750, -14.8750, -14.8750, -14.8750, -14.8750, -14.8750]]),
            ("cuda", 7): torch.tensor([[-14.8125, -14.8125, -14.8125,  -8.1875, -14.8125, -14.8125, -14.8125, -14.8125, -14.8125, -14.8125, -14.8125, -14.8125, -14.8125, -14.8125, -14.8125]]),
            ("cuda", 8): torch.tensor([[-14.8125, -14.8125, -14.8125,  -8.1875, -14.8125, -14.8125, -14.8125, -14.8125, -14.8125, -14.8125, -14.8125, -14.8125, -14.8125, -14.8125, -14.8125]]),
        })
        # fmt: on
        expected_slice = expected_slices.get_expectation().to(torch_device)
        actual_slice = out.logits[0, 0, :15].float()
        self.assertTrue(torch.allclose(expected_slice, actual_slice, atol=1e-2, rtol=1e-2))

    @pytest.mark.skip(reason="float16 resolution is too low for the model parameter weights.")
    @slow
    def test_pldrllm110M_logits_fp16(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]

        model = PldrllmForCausalLM.from_pretrained(
            "fromthesky/PLDR-LLM-v51-110M-test", device_map="auto", dtype=torch.float16
        )

        with torch.no_grad():
            out = model(torch.tensor([input_ids]).to(torch_device))

        # fmt: off
        # Expected mean on dim = -1
        expected_means = Expectations(
          {
            ("xpu", 3): torch.tensor([[-6.6544, -4.1259, -4.9840, -3.2456,  0.8261, -3.0124,  1.2971, -3.3641]]),
            ("cuda", 7): torch.tensor([[-6.6420, -4.1227, -4.9809, -3.2041, 0.8261, -3.0052, 1.2957, -3.3648]]),
            ("cuda", 8): torch.tensor([[-6.6544, -4.1259, -4.9840, -3.2456,  0.8261, -3.0124,  1.2971, -3.3641]]),
        })

        expected_mean = expected_means.get_expectation()
        self.assertTrue(
            torch.allclose(
                expected_mean.to(torch_device),
                out.logits.float().mean(-1),
                atol=1e-2,
                rtol=1e-2
            )
        )

        # slicing logits[0, 0, 0:15]
        expected_slices = Expectations(
            {
              ("xpu", 3): torch.tensor([-14.8750, -14.8750, -14.8750,  -8.1875, -14.8750, -14.8750, -14.8750, -14.8750, -14.8750, -14.8750, -14.8750, -14.8750, -14.8750, -14.8750, -14.8750]),
              ("cuda", 7): torch.tensor([-14.8125, -14.8125, -14.8125,  -8.1875, -14.8125, -14.8125, -14.8125, -14.8125, -14.8125, -14.8125, -14.8125, -14.8125, -14.8125, -14.8125, -14.8125]),
              ("cuda", 8): torch.tensor([-14.8125, -14.8125, -14.8125,  -8.1875, -14.8125, -14.8125, -14.8125, -14.8125, -14.8125, -14.8125, -14.8125, -14.8125, -14.8125, -14.8125, -14.8125])
        })
        # fmt: on

        expected_slice = expected_slices.get_expectation()
        self.assertTrue(
            torch.allclose(
                expected_slice.to(torch_device),
                out.logits[0, 0, :15].float(),
                atol=1e-2,
                rtol=1e-2,
            )
        )


    # TODO: check why we have the following strange situation.
    # without running in subprocess, this test causes subsequent tests failing with `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0!`
    @run_test_using_subprocess
    @slow
    def test_pldrllm_110M_dola_generation(self):
        # ground truth text generated with dola_layers="low", repetition_penalty=1.2
        EXPECTED_TEXT_COMPLETION = (
                                    "Simply put, the theory of relativity states that physicists can only be understood by the physical and mental processes "
                                    "of their being. This is because the physical and mental processes of their being are not physical, but are physical and mental "
                                    "processes of their being. This is because the physical processes of their being are physical and mental processes of their being. "
                                    "This is because the"
        )
        model_path="fromthesky/PLDR-LLM-v51-110M-3"
        prompt = "Simply put, the theory of relativity states that "
        tokenizer = LlamaTokenizerFast.from_pretrained(model_path, add_eos_token=False, legacy=False)
        model = PldrllmForCausalLM.from_pretrained(
            model_path, device_map="sequential", dtype=torch.bfloat16
        )
        model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # greedy generation outputs
        generated_ids = model.generate(
            **model_inputs, max_new_tokens=64, 
            top_p=None, temperature=1, 
            do_sample=False, dola_layers="low",
            trust_remote_code=True)
        
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)

    @slow
    @require_torch_accelerator
    def test_compile_static_cache(self):
        # `torch==2.2` will throw an error on this test (as in other compilation tests), but torch==2.1.2 and torch>2.2
        # work as intended. See https://github.com/pytorch/pytorch/issues/121943
        if version.parse(torch.__version__) < version.parse("2.3.0"):
            self.skipTest(reason="This test requires torch >= 2.3 to run.")
        self.maxDiff=None

        NUM_TOKENS_TO_GENERATE = 40
        # Note on `EXPECTED_TEXT_COMPLETION`'s diff: the current value matches the original test if the original test
        # was changed to have a cache of 53 tokens (as opposed to 4096), on Ampere GPUs.
        EXPECTED_TEXT_COMPLETION = [
            "Simply put, the theory of relativity states that 20% of the world’s population is a 20% of the world’s population. "
            "The theory of relativity states that 20% of the world’s population is",
            "My favorite all time favorite condiment is ketchup. I love ketchup. I love ketchup. I love ketchup. I love ketchup. "
            "I love ketchup. I love ketchup. I love ketchup. I love ketchup. I love ketchup. I love ketchup.",
        ]

        prompts = [
            "Simply put, the theory of relativity states that ",
            "My favorite all time favorite condiment is ketchup.",
        ]
        model_path="fromthesky/PLDR-LLM-v51G-106M-1"
        tokenizer = LlamaTokenizerFast.from_pretrained(
            model_path, padding_side="right", legacy=False, add_eos_token=False
        )
        model = PldrllmForCausalLM.from_pretrained(
            model_path, device_map=torch_device, dtype=torch.bfloat16
        )
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        # Dynamic Cache
        generated_ids = model.generate(**inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False)
        dynamic_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        self.assertEqual(EXPECTED_TEXT_COMPLETION, dynamic_text)

        # Static Cache + compile (`generate()` internally compiles each decoding step when static cache is used)
        generated_ids = model.generate(
            **inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False, cache_implementation="static"
        )
        static_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        self.assertEqual(EXPECTED_TEXT_COMPLETION, static_text)

    @slow
    def test_export_static_cache(self):
        if version.parse(torch.__version__) < version.parse("2.4.0"):
            self.skipTest(reason="This test requires torch >= 2.4 to run.")

        from transformers.integrations.executorch import (
            TorchExportableModuleWithStaticCache,
        )

        model_path="fromthesky/PLDR-LLM-v51G-106M-test"
        pldrllm_models = {
            model_path: [
                "Simply put, the theory of relativity states that 100% of the universe is the universe. "
                "The theory of relativity states that 100% of the universe is the universe. The theory of relativity states that "
            ],
        }

        for pldrllm_model_ckp, EXPECTED_TEXT_COMPLETION in pldrllm_models.items():
            # Load tokenizer
            tokenizer = LlamaTokenizerFast.from_pretrained(pldrllm_model_ckp, padding_side="left", legacy=False)
            max_generation_length = tokenizer(EXPECTED_TEXT_COMPLETION, return_tensors="pt", padding=True)[
                "input_ids"
            ].shape[-1]

            # Load model
            device = "cpu"  # TODO (joao / export experts): should be on `torch_device`, but causes GPU OOM
            dtype = torch.bfloat16
            cache_implementation = "static"
            attn_implementation = "sdpa"
            batch_size = 1
            model = PldrllmForCausalLM.from_pretrained(
                pldrllm_model_ckp,
                device_map=device,
                dtype=dtype,
                use_cache=True,
                attn_implementation=attn_implementation,
                generation_config=GenerationConfig(
                    use_cache=True,
                    cache_implementation=cache_implementation,
                    max_length=max_generation_length,
                    cache_config={
                        "batch_size": batch_size,
                        "max_cache_len": max_generation_length,
                        "device": device,
                    },
                ),
            )

            prompts = ["Simply put, the theory of relativity states that "]
            prompt_tokens = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            prompt_token_ids = prompt_tokens["input_ids"]
            max_new_tokens = max_generation_length - prompt_token_ids.shape[-1]

            # Static Cache + export
            from transformers.integrations.executorch import TorchExportableModuleForDecoderOnlyLM

            exportable_module = TorchExportableModuleForDecoderOnlyLM(model)
            exported_program = exportable_module.export(
                input_ids=prompt_token_ids,
                cache_position=torch.arange(prompt_token_ids.shape[-1], dtype=torch.long, device=model.device),
            )
            ep_generated_ids = TorchExportableModuleWithStaticCache.generate(
                exported_program=exported_program, prompt_token_ids=prompt_token_ids, max_new_tokens=max_new_tokens
            )
            ep_generated_text = tokenizer.batch_decode(ep_generated_ids, skip_special_tokens=True)

            self.assertEqual(EXPECTED_TEXT_COMPLETION, ep_generated_text)


@slow
@require_torch_accelerator
class Mask4DTestHard(unittest.TestCase):
    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def setUp(self):
        cleanup(torch_device, gc_collect=True)
        model_name="fromthesky/PLDR-LLM-v51G-106M-1"
        self.model_dtype = torch.float32
        self.tokenizer = LlamaTokenizerFast.from_pretrained(model_name)
        self.model = PldrllmForCausalLM.from_pretrained(model_name, dtype=self.model_dtype).to(torch_device)

    def get_test_data(self):
        template = "my favorite {}"
        items = ("pet is a", "artist plays a", "name is L")  # same number of tokens in each item

        batch_separate = [template.format(x) for x in items]  # 3 separate lines
        batch_shared_prefix = template.format(" ".join(items))  # 1 line with options concatenated

        input_ids = self.tokenizer(batch_separate, return_tensors="pt").input_ids.to(torch_device)
        input_ids_shared_prefix = self.tokenizer(batch_shared_prefix, return_tensors="pt").input_ids.to(torch_device)

        mask_shared_prefix = torch.tensor(
            [
                [
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                        [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                        [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                    ]
                ]
            ],
            device=torch_device,
        )

        position_ids = torch.arange(input_ids.shape[1]).tile(input_ids.shape[0], 1).to(torch_device)

        # building custom positions ids based on custom mask
        position_ids_shared_prefix = (mask_shared_prefix.sum(dim=-1) - 1).reshape(1, -1)
        # effectively: position_ids_shared_prefix = torch.tensor([[0, 1, 2, 3, 4, 5, 3, 4, 5, 3, 4, 5]]).to(device)

        # inverting the mask
        min_dtype = torch.finfo(self.model_dtype).min
        mask_shared_prefix = (mask_shared_prefix.eq(0.0)).to(dtype=self.model_dtype) * min_dtype

        return input_ids, position_ids, input_ids_shared_prefix, mask_shared_prefix, position_ids_shared_prefix

    def test_stacked_causal_mask(self):
        (
            input_ids,
            position_ids,
            input_ids_shared_prefix,
            mask_shared_prefix,
            position_ids_shared_prefix,
        ) = self.get_test_data()

        # regular batch
        logits = self.model.forward(input_ids, position_ids=position_ids).logits
        logits_last = logits[:, -1, :]  # last tokens in each batch line
        decoded = [self.tokenizer.decode(t) for t in logits_last.argmax(dim=-1)]

        # single forward run with 4D custom mask
        logits_shared_prefix = self.model.forward(
            input_ids_shared_prefix, attention_mask=mask_shared_prefix, position_ids=position_ids_shared_prefix
        ).logits
        logits_shared_prefix_last = logits_shared_prefix[
            0, torch.where(position_ids_shared_prefix == position_ids_shared_prefix.max())[1], :
        ]  # last three tokens
        decoded_shared_prefix = [self.tokenizer.decode(t) for t in logits_shared_prefix_last.argmax(dim=-1)]

        self.assertEqual(decoded, decoded_shared_prefix)

    def test_partial_stacked_causal_mask(self):
        # Same as the test above, but the input is passed in two groups. It tests that we can pass partial 4D attention masks

        (
            input_ids,
            position_ids,
            input_ids_shared_prefix,
            mask_shared_prefix,
            position_ids_shared_prefix,
        ) = self.get_test_data()

        # regular batch
        logits = self.model.forward(input_ids, position_ids=position_ids).logits
        logits_last = logits[:, -1, :]  # last tokens in each batch line
        decoded = [self.tokenizer.decode(t) for t in logits_last.argmax(dim=-1)]

        # 2 forward runs with custom 4D masks
        part_a = 3  # split point

        input_1a = input_ids_shared_prefix[:, :part_a]
        position_ids_1a = position_ids_shared_prefix[:, :part_a]
        mask_1a = mask_shared_prefix[:, :, :part_a, :part_a]

        outs_1a = self.model.forward(input_1a, attention_mask=mask_1a, position_ids=position_ids_1a)
        past_key_values_a = outs_1a["past_key_values"]

        # Case 1: we pass a 4D attention mask regarding the current sequence length (i.e. [..., seq_len, full_len])
        input_1b = input_ids_shared_prefix[:, part_a:]
        position_ids_1b = position_ids_shared_prefix[:, part_a:]
        mask_1b = mask_shared_prefix[:, :, part_a:, :]
        outs_1b = self.model.forward(
            input_1b,
            attention_mask=mask_1b,
            position_ids=position_ids_1b,
            past_key_values=past_key_values_a,
        )
        decoded_1b = [
            self.tokenizer.decode(t)
            for t in outs_1b.logits.argmax(-1)[
                0, torch.where(position_ids_shared_prefix == position_ids_shared_prefix.max())[1] - part_a
            ]
        ]
        self.assertEqual(decoded, decoded_1b)

    def test_stacked_causal_mask_static_cache(self):
        """same as above but with StaticCache"""
        (
            input_ids,
            position_ids,
            input_ids_shared_prefix,
            mask_shared_prefix,
            position_ids_shared_prefix,
        ) = self.get_test_data()

        # regular batch
        logits = self.model.forward(input_ids, position_ids=position_ids).logits
        logits_last = logits[:, -1, :]  # last tokens in each batch line
        decoded = [self.tokenizer.decode(t) for t in logits_last.argmax(dim=-1)]

        # upgrade the model with StaticCache
        max_cache_len = 16  # note that max_cache_len is greater than the attention_mask.shape[-1]
        past_key_values = StaticCache(max_cache_len=max_cache_len, config=self.model.config)

        padded_attention_mask = torch.nn.functional.pad(
            input=mask_shared_prefix,
            pad=(0, max_cache_len - mask_shared_prefix.shape[-1]),
            mode="constant",
            value=torch.finfo(self.model_dtype).min,
        )

        # single forward run with 4D custom mask
        logits_shared_prefix = self.model.forward(
            input_ids_shared_prefix,
            attention_mask=padded_attention_mask,
            position_ids=position_ids_shared_prefix,
            cache_position=torch.arange(input_ids_shared_prefix.shape[-1], device=torch_device),
            past_key_values=past_key_values,
        ).logits
        logits_shared_prefix_last = logits_shared_prefix[
            0, torch.where(position_ids_shared_prefix == position_ids_shared_prefix.max())[1], :
        ]  # last three tokens
        decoded_shared_prefix = [self.tokenizer.decode(t) for t in logits_shared_prefix_last.argmax(dim=-1)]

        self.assertEqual(decoded, decoded_shared_prefix)

    def test_partial_stacked_causal_mask_static_cache(self):
        # Same as the test above, but the input is passed in two groups. It tests that we can pass partial 4D attention masks
        # we pass a 4D attention mask shaped [..., seq_len, full_static_cache_len])
        (
            input_ids,
            position_ids,
            input_ids_shared_prefix,
            mask_shared_prefix,
            position_ids_shared_prefix,
        ) = self.get_test_data()

        # regular batch
        logits = self.model.forward(input_ids, position_ids=position_ids).logits
        logits_last = logits[:, -1, :]  # last tokens in each batch line
        decoded = [self.tokenizer.decode(t) for t in logits_last.argmax(dim=-1)]

        # upgrade the model with StaticCache
        max_cache_len = 16  # note that max_cache_len is greater than the attention_mask.shape[-1]
        past_key_values = StaticCache(max_cache_len=max_cache_len, config=self.model.config)

        # forward run for the first part of input
        part_a = 3  # split point

        input_1a = input_ids_shared_prefix[:, :part_a]
        position_ids_1a = position_ids_shared_prefix[:, :part_a]
        mask_1a = mask_shared_prefix[:, :, :part_a, :part_a]

        padded_mask_1a = torch.nn.functional.pad(
            input=mask_1a,
            pad=(0, max_cache_len - mask_1a.shape[-1]),
            mode="constant",
            value=torch.finfo(self.model_dtype).min,
        )

        _ = self.model.forward(
            input_1a,
            attention_mask=padded_mask_1a,
            position_ids=position_ids_1a,
            cache_position=torch.arange(part_a, device=torch_device),
            past_key_values=past_key_values,
        )

        # forward run for the second part of input
        input_1b = input_ids_shared_prefix[:, part_a:]
        position_ids_1b = position_ids_shared_prefix[:, part_a:]
        mask_1b = mask_shared_prefix[:, :, part_a:, :]

        padded_mask_1b = torch.nn.functional.pad(
            input=mask_1b, pad=(0, max_cache_len - mask_1b.shape[-1]), mode="constant", value=0
        )

        outs_1b = self.model.forward(
            input_1b,
            attention_mask=padded_mask_1b,
            position_ids=position_ids_1b,
            cache_position=torch.arange(
                part_a,
                input_ids_shared_prefix.shape[-1],
                device=torch_device,
            ),
            past_key_values=past_key_values,
        )
        decoded_1b = [
            self.tokenizer.decode(t)
            for t in outs_1b.logits.argmax(-1)[
                0, torch.where(position_ids_shared_prefix == position_ids_shared_prefix.max())[1] - part_a
            ]
        ]
        self.assertEqual(decoded, decoded_1b)
