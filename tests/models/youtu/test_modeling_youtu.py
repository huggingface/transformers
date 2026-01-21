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
"""Testing suite for the PyTorch Youtu-LLM model."""

import unittest

import pytest
from packaging import version
from parameterized import parameterized

from transformers import AutoTokenizer, YoutuConfig, is_torch_available
from transformers.testing_utils import (
    cleanup,
    require_torch,
    require_torch_accelerator,
    require_torch_large_accelerator,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    torch.set_float32_matmul_precision("high")

    from transformers import (
        Cache,
        YoutuForCausalLM,
        YoutuModel,
    )


class YoutuModelTester:
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
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=32,
        v_head_dim=32,
        qk_nope_head_dim=32,
        hidden_act="silu",
        max_position_embeddings=512,
        initializer_range=0.02,
        embedding_initializer_range=0.04,
        attention_probs_dropout_prob=0.1,
        type_vocab_size=16,
        type_sequence_label_size=2,
        num_labels=3,
        num_choices=4,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=1,
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
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.embedding_initializer_range = embedding_initializer_range
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
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
        return YoutuConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            kv_lora_rank=self.kv_lora_rank,
            q_lora_rank=self.q_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            qk_nope_head_dim=self.qk_nope_head_dim,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            embedding_initializer_range=self.embedding_initializer_range,
            use_cache=True,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            attention_dropout=self.attention_probs_dropout_prob,
        )

    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = YoutuModel(config=config)
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
class YoutuModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            YoutuModel,
            YoutuForCausalLM,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (YoutuForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": YoutuModel,
            "text-generation": YoutuForCausalLM,
        }
        if is_torch_available()
        else {}
    )

    # Need to use `0.8` instead of `0.9` for `test_cpu_offload`
    # This is because we are hitting edge cases with the causal_mask buffer
    model_split_percents = [0.5, 0.7, 0.8]

    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = YoutuForCausalLM if is_torch_available() else None

    def setUp(self):
        self.model_tester = YoutuModelTester(self)
        self.config_tester = ConfigTester(self, config_class=YoutuConfig, hidden_size=37)

    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config):
        """Needs to be overridden as youtu-llm has special MLA cache format (though we don't really use the MLA)"""
        self.assertIsInstance(past_key_values, Cache)

        # (batch, head, seq_length, head_features)
        expected_common_shape = (
            batch_size,
            getattr(config, "num_key_value_heads", config.num_attention_heads),
            seq_length,
        )
        expected_key_shape = expected_common_shape + (config.qk_nope_head_dim + config.qk_rope_head_dim,)
        expected_value_shape = expected_common_shape + (config.v_head_dim,)

        for layer in past_key_values.layers:
            self.assertEqual(layer.keys.shape, expected_key_shape)
            self.assertEqual(layer.values.shape, expected_value_shape)

    @parameterized.expand([("random",), ("same",)])
    @unittest.skip("Youtu-LLM is not compatible with assisted decoding")
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip("Youtu-LLM is not compatible with assisted decoding")
    def test_prompt_lookup_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip("Youtu-LLM is not compatible with assisted decoding")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip("Youtu-LLM uses MLA so it is not compatible with the standard cache format")
    def test_beam_search_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("Youtu-LLM uses MLA so it is not compatible with the standard cache format")
    def test_greedy_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip(reason="SDPA can't dispatch on flash due to unsupported head dims")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(reason="Youtu-LLM is not suitable for testing with extreme small vocabulary")
    def test_resize_tokens_embeddings(self):
        pass

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @require_torch_large_accelerator
    @slow
    def test_eager_matches_sdpa_generate(self):
        max_new_tokens = 30

        tokenizer = AutoTokenizer.from_pretrained("Junrulu/Youtu-LLM-2B-Base-hf")

        model_sdpa = YoutuForCausalLM.from_pretrained(
            "Junrulu/Youtu-LLM-2B-Base-hf",
            dtype=torch.float16,
            attn_implementation="sdpa",
        ).to(torch_device)

        self.assertTrue(model_sdpa.config._attn_implementation == "sdpa")

        model_eager = YoutuForCausalLM.from_pretrained(
            "Junrulu/Youtu-LLM-2B-Base-hf",
            dtype=torch.float16,
            attn_implementation="eager",
        ).to(torch_device)

        self.assertTrue(model_eager.config._attn_implementation == "eager")

        texts = [
            "hi here's a longer context, getting longer and",
            "Today I am in Paris and",
        ]

        for padding_side in ["left", "right"]:
            tokenizer.padding_side = padding_side
            tokenizer.pad_token = tokenizer.eos_token

            inputs = tokenizer(texts, return_tensors="pt", padding=True).to(torch_device)

            res_eager = model_eager.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            res_sdpa = model_sdpa.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

            with self.subTest(f"{padding_side}"):
                torch.testing.assert_close(
                    res_eager,
                    res_sdpa,
                    msg=f"\n{tokenizer.batch_decode(res_eager)} \nvs\n{tokenizer.batch_decode(res_sdpa)}",
                )


@slow
class YoutuIntegrationTest(unittest.TestCase):
    def tearDown(self):
        # See LlamaIntegrationTest.tearDown(). Can be removed once LlamaIntegrationTest.tearDown() is removed.
        cleanup(torch_device, gc_collect=False)

    @require_torch_accelerator
    def test_dynamic_cache(self):
        # `torch==2.2` will throw an error on this test (as in other compilation tests), but torch==2.1.2 and torch>2.2
        # work as intended. See https://github.com/pytorch/pytorch/issues/121943
        if version.parse(torch.__version__) < version.parse("2.3.0"):
            self.skipTest(reason="This test requires torch >= 2.3 to run.")

        NUM_TOKENS_TO_GENERATE = 40
        EXPECTED_TEXT_COMPLETION = [
            "Simply put, the theory of relativity states that , the speed of light is constant in all reference frames. This means that if you are traveling at the speed of light, you will never reach the speed of light. This is because the speed of",
            "My favorite all time favorite condiment is ketchup. I love it on everything. I love it on burgers, hot dogs, and even on my fries. I also love it on my french fries. I love it on my french fries. I love",
        ]

        prompts = [
            "Simply put, the theory of relativity states that ",
            "My favorite all time favorite condiment is ketchup.",
        ]
        tokenizer = AutoTokenizer.from_pretrained("Junrulu/Youtu-LLM-2B-Base-hf")
        model = YoutuForCausalLM.from_pretrained(
            "Junrulu/Youtu-LLM-2B-Base-hf", device_map=torch_device, dtype=torch.float16
        )
        if model.config.tie_word_embeddings:
            # Youtu-LLM-2B-Base contains extra repeated weights for the tied embeddings, we can tie weights here according to its config
            model.tie_weights()
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        # Dynamic Cache
        generated_ids = model.generate(**inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False)
        dynamic_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, dynamic_text)

    @require_torch_accelerator
    def test_static_cache(self):
        # `torch==2.2` will throw an error on this test (as in other compilation tests), but torch==2.1.2 and torch>2.2
        # work as intended. See https://github.com/pytorch/pytorch/issues/121943
        if version.parse(torch.__version__) < version.parse("2.3.0"):
            self.skipTest(reason="This test requires torch >= 2.3 to run.")

        NUM_TOKENS_TO_GENERATE = 40
        EXPECTED_TEXT_COMPLETION = [
            "Simply put, the theory of relativity states that , the speed of light is constant in all reference frames. This means that if you are traveling at the speed of light, you will never reach the speed of light. This is because the speed of",
            "My favorite all time favorite condiment is ketchup. I love it on everything. I love it on burgers, hot dogs, and even on my fries. I also love it on my french fries. I love it on my french fries. I love",
        ]

        prompts = [
            "Simply put, the theory of relativity states that ",
            "My favorite all time favorite condiment is ketchup.",
        ]
        tokenizer = AutoTokenizer.from_pretrained("Junrulu/Youtu-LLM-2B-Base-hf")
        model = YoutuForCausalLM.from_pretrained(
            "Junrulu/Youtu-LLM-2B-Base-hf", device_map=torch_device, dtype=torch.float16
        )
        if model.config.tie_word_embeddings:
            # Youtu-LLM-2B-Base contains extra repeated weights for the tied embeddings, we can tie weights here according to its config
            model.tie_weights()
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        # Static Cache
        generated_ids = model.generate(
            **inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False, cache_implementation="static"
        )
        static_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, static_text)

    @slow
    @require_torch_accelerator
    @pytest.mark.torch_compile_test
    def test_compile_static_cache(self):
        # `torch==2.2` will throw an error on this test (as in other compilation tests), but torch==2.1.2 and torch>2.2
        # work as intended. See https://github.com/pytorch/pytorch/issues/121943
        if version.parse(torch.__version__) < version.parse("2.3.0"):
            self.skipTest(reason="This test requires torch >= 2.3 to run.")

        NUM_TOKENS_TO_GENERATE = 40
        EXPECTED_TEXT_COMPLETION = [
            "Simply put, the theory of relativity states that , the speed of light is constant in all reference frames. This means that if you are traveling at the speed of light, you will never reach the speed of light. This is because the speed of",
            "My favorite all time favorite condiment is ketchup. I love it on everything. I love it on burgers, hot dogs, and even on my fries. I also love it on my french fries. I love it on my french fries. I love",
        ]

        prompts = [
            "Simply put, the theory of relativity states that ",
            "My favorite all time favorite condiment is ketchup.",
        ]
        tokenizer = AutoTokenizer.from_pretrained("Junrulu/Youtu-LLM-2B-Base-hf")
        model = YoutuForCausalLM.from_pretrained(
            "Junrulu/Youtu-LLM-2B-Base-hf", device_map=torch_device, dtype=torch.float16
        )
        if model.config.tie_word_embeddings:
            # Youtu-LLM-2B-Base contains extra repeated weights for the tied embeddings, we can tie weights here according to its config
            model.tie_weights()
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        # Static Cache + compile
        model._cache = None  # clear cache object, initialized when we pass `cache_implementation="static"`
        model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=False, dynamic=True)
        generated_ids = model.generate(
            **inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False, cache_implementation="static"
        )
        static_compiled_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, static_compiled_text)
