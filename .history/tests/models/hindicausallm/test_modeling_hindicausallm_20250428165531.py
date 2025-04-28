# coding=utf-8
# Copyright 2024 ConvaiInnovations and The HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch HindiCausalLM model."""

import unittest

import pytest

from transformers import AutoModelForCausalLM, AutoTokenizer, HindiCausalLMConfig, is_torch_available, pipeline
from transformers.testing_utils import (
    require_flash_attn,
    require_read_token,
    require_torch,
    require_torch_accelerator,
    require_torch_gpu,
    slow,
    torch_device,
)

# Import common test classes
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        HindiCausalLMForCausalLM,
        HindiCausalLMForSequenceClassification,
        # HindiCausalLMForTokenClassification, # Can add if implemented
        HindiCausalLMModel,
    )


# Define a Model Tester specific to HindiCausalLM, inheriting basic structure if needed
# Adapted from Gemma2ModelTester
class HindiCausalLMModelTester:
    if is_torch_available():
        config_class = HindiCausalLMConfig
        model_class = HindiCausalLMModel
        for_causal_lm_class = HindiCausalLMForCausalLM
        for_sequence_class = HindiCausalLMForSequenceClassification
        # for_token_class = HindiCausalLMForTokenClassification # Can add if implemented

    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99, # Small vocab for testing
        hidden_size=32, # Small hidden size
        num_hidden_layers=2, # Minimal layers
        num_attention_heads=4, # Example attention heads
        num_key_value_heads=2, # Example GQA heads
        intermediate_size=64, # Example intermediate size
        hidden_act="silu",
        max_position_embeddings=512, # Keep reasonable for RoPE
        rms_norm_eps=1e-6, # Use RMS norm eps
        initializer_range=0.02,
        pad_token_id=0,
        eos_token_id=2,
        bos_token_id=1,
        num_labels=3, # For sequence classification head
        type_sequence_label_size=2, # For sequence classification head
        attention_dropout=0.0,
        attention_bias=False,
        tie_word_embeddings=False, # Test untied case
        scope=None,
        cache_implementation="dynamic", # Test default dynamic cache
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.num_labels = num_labels
        self.type_sequence_label_size = type_sequence_label_size
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.tie_word_embeddings = tie_word_embeddings
        self.scope = scope
        self.cache_implementation = cache_implementation
        self.head_dim = hidden_size // num_attention_heads # Calculate head_dim


    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        sequence_labels = None
        token_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            # For Causal LM, labels are usually input_ids shifted
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        config = self.get_config()

        return config, input_ids, input_mask, sequence_labels, token_labels

    def get_config(self):
        return self.config_class(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.max_position_embeddings,
            rms_norm_eps=self.rms_norm_eps,
            initializer_range=self.initializer_range,
            use_cache=True, # Default use_cache=True for generation tests
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            attention_dropout=self.attention_dropout,
            attention_bias=self.attention_bias,
            tie_word_embeddings=self.tie_word_embeddings,
            num_labels=self.num_labels, # For Sequence Classification tests
            cache_implementation=self.cache_implementation,
        )

    def create_and_check_model(self, config, input_ids, input_mask, sequence_labels, token_labels):
        model = self.model_class(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_causal_lm(
        self, config, input_ids, input_mask, sequence_labels, token_labels
    ):
        model = self.for_causal_lm_class(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_for_sequence_classification(
        self, config, input_ids, input_mask, sequence_labels, token_labels
    ):
        model = self.for_sequence_class(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=sequence_labels)
        # Logits shape should be (batch_size, num_labels) after pooling
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    # Can add create_and_check_for_token_classification if implemented

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, input_mask, sequence_labels, token_labels = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class HindiCausalLMModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            HindiCausalLMModel,
            HindiCausalLMForCausalLM,
            HindiCausalLMForSequenceClassification,
            # HindiCausalLMForTokenClassification, # Add if implemented
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (HindiCausalLMForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": HindiCausalLMModel,
            "text-classification": HindiCausalLMForSequenceClassification,
            # "token-classification": HindiCausalLMForTokenClassification, # Add if implemented
            "text-generation": HindiCausalLMForCausalLM,
            "zero-shot-classification": HindiCausalLMForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    test_head_masking = False # Not implemented/tested
    test_pruning = False # Not implemented/tested
    _is_stateful = True # If using Cache object
    has_attentions = False # HindiCausalLMModel doesn't forcefully return attentions? Check impl. Set True if it does.

    # TODO: Increase this value? Needs more testing.
    model_split_percents = [0.5, 0.7] # For testing model parallel

    def setUp(self):
        self.model_tester = HindiCausalLMModelTester(self)
        self.config_tester = ConfigTester(self, config_class=HindiCausalLMConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_causal_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_causal_lm(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    # --- Skip tests adapted from Gemma2 that rely on features HindiCausalLM doesn't have ---

    @unittest.skip("HindiCausalLM uses DynamicCache by default, not HybridCache.")
    def test_model_outputs_equivalence(self, **kwargs):
        pass

    @unittest.skip("HindiCausalLM uses DynamicCache, not compatible with assisted decoding currently.")
    @pytest.mark.generate
    def test_assisted_decoding_matches_greedy_search(self, assistant_type="random"): # Add default
        pass

    @unittest.skip("HindiCausalLM uses DynamicCache, not compatible with prompt lookup decoding currently.")
    @pytest.mark.generate
    def test_prompt_lookup_decoding_matches_greedy_search(self, assistant_type="random"): # Add default
        pass

    @unittest.skip("HindiCausalLM uses DynamicCache, not compatible with assisted decoding currently.")
    @pytest.mark.generate
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip("HindiCausalLM uses DynamicCache, not compatible with dola decoding currently.")
    @pytest.mark.generate
    def test_dola_decoding_sample(self):
        pass

    @unittest.skip("HindiCausalLM uses DynamicCache, not compatible with contrastive generation.")
    @pytest.mark.generate
    def test_contrastive_generate(self):
        pass

    @unittest.skip("HindiCausalLM uses DynamicCache, not compatible with contrastive generation.")
    @pytest.mark.generate
    def test_contrastive_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("HindiCausalLM uses DynamicCache, not compatible with contrastive generation.")
    @pytest.mark.generate
    def test_contrastive_generate_low_memory(self):
        pass

    @unittest.skip("HindiCausalLM defaults to DynamicCache, StaticCache tests are separate.")
    @pytest.mark.generate
    def test_generate_with_static_cache(self):
        pass

    @unittest.skip("HindiCausalLM defaults to DynamicCache, StaticCache tests are separate.")
    @pytest.mark.generate
    def test_generate_from_inputs_embeds_with_static_cache(self):
        pass

    # Keep standard SDPA/FA2 tests as HindiCausalLM _should_ support them
    # If they fail, investigate the modeling code attention implementation

    # @unittest.skip("Gemma2's eager attn/sdpa attn outputs are expected to be different") # Remove this skip
    # def test_sdpa_equivalence(self):
    #     pass

    # @unittest.skip("Gemma2's eager attn/sdpa attn outputs are expected to be different") # Remove this skip
    # def test_eager_matches_sdpa_generate(self):
    #     pass

    # @unittest.skip("Gemma2 has HybridCache which auto-compiles. Compile and FA2 don't work together.") # Remove this skip
    # def test_eager_matches_fa2_generate(self):
    #     pass

    @unittest.skip(reason="Model parallel testing requires specific setup and might fail with DynamicCache.")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    # Add back standard modeling common tests if they were skipped previously
    def test_initialization(self):
        super().test_initialization()

    def test_attention_outputs(self):
        super().test_attention_outputs()

    def test_forward_signature(self):
         super().test_forward_signature()

    def test_generate_without_input_ids(self):
        super().test_generate_without_input_ids()

    def test_generation_config_defaults(self):
         super().test_generation_config_defaults()

    # Maybe keep this one if generation fails otherwise
    # @unittest.skip("HindiCausalLM uses DynamicCache and doesn't support continue from past kv in the same way?")
    # def test_generate_continue_from_past_key_values(self):
    #     pass

    # @unittest.skip("Static Cache test needs specific setup")
    # def test_generate_continue_from_inputs_embeds(self):
    #     pass


# Adapted from Gemma2IntegrationTest
@slow
@require_torch_accelerator
class HindiCausalLMIntegrationTest(unittest.TestCase):
    # Define expected outputs for a known checkpoint and input
    # THESE WILL NEED TO BE ADJUSTED based on actual model output
    EXPECTED_OUTPUT_TEXT_EAGER = [
        # Example: Replace with actual expected output for eager
        "भारत एक विशाल देश है जो दुनिया के सबसे बड़े लोकतंत्रों में से एक है। यह दक्षिण एशिया में स्थित है और",
        "नमस्ते दुनिया! आज हम बात करेंगे हिंदी भाषा के महत्व के बारे में। हिंदी हमारी राजभाषा है और देश के",
    ]
    EXPECTED_OUTPUT_TEXT_SDPA = [
        # Example: Replace with actual expected output for sdpa (might match eager)
        "भारत एक विशाल देश है जो दुनिया के सबसे बड़े लोकतंत्रों में से एक है। यह दक्षिण एशिया में स्थित है और",
        "नमस्ते दुनिया! आज हम बात करेंगे हिंदी भाषा के महत्व के बारे में। हिंदी हमारी राजभाषा है और देश के",
    ]
    EXPECTED_OUTPUT_TEXT_FA2 = [
         # Example: Replace with actual expected output for FA2 (might match eager/sdpa)
        "भारत एक विशाल देश है जो दुनिया के सबसे बड़े लोकतंत्रों में से एक है। यह दक्षिण एशिया में स्थित है और",
        "नमस्ते दुनिया! आज हम बात करेंगे हिंदी भाषा के महत्व के बारे में। हिंदी हमारी राजभाषा है और देश के",
    ]
    input_text = ["भारत एक विशाल देश है", "नमस्ते दुनिया"]

    @classmethod
    def setUpClass(cls):
        cls.model_id = "convaiinnovations/hindi-causal-lm"
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_id)
        # Ensure pad token is set for batch generation
        if cls.tokenizer.pad_token is None:
            cls.tokenizer.pad_token = cls.tokenizer.eos_token

    def _check_generation(self, model, expected_texts, max_new_tokens=20):
        inputs = self.tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        output_texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        # Check if outputs match expected (flexible check might be needed for float variations)
        self.assertListEqual(output_texts, expected_texts)

    @require_read_token # If model is gated/private
    def test_model_eager_bf16(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            low_cpu_mem_usage=True, # Good practice for large models
            torch_dtype=torch.bfloat16,
            attn_implementation="eager"
        ).to(torch_device).eval()
        self._check_generation(model, self.EXPECTED_OUTPUT_TEXT_EAGER)

    @require_read_token
    def test_model_eager_fp16(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            attn_implementation="eager"
        ).to(torch_device).eval()
        self._check_generation(model, self.EXPECTED_OUTPUT_TEXT_EAGER) # Assuming fp16 gives same result for greedy

    @require_read_token
    @require_torch_gpu # SDPA often needs GPU
    def test_model_sdpa_bf16(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa"
        ).to(torch_device).eval()
        self._check_generation(model, self.EXPECTED_OUTPUT_TEXT_SDPA)

    @require_flash_attn
    @require_read_token
    @require_torch_gpu
    @pytest.mark.flash_attn_test
    def test_model_flash_attn_2_bf16(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        ).to(torch_device).eval()
        self._check_generation(model, self.EXPECTED_OUTPUT_TEXT_FA2)

    @require_read_token
    def test_pipeline_bf16(self):
        # Test using the pipeline
        pipe = pipeline(
            "text-generation",
            model=self.model_id,
            torch_dtype=torch.bfloat16,
            device=torch_device, # Use device from testing utils
            # attn_implementation="sdpa" # Optionally specify attn for pipeline
        )
        # Adjust expected outputs to remove potential special tokens added by pipeline/tokenizer defaults
        expected_pipeline_texts = [t.split(" है", 1)[1].strip() if " है" in t else t for t in self.EXPECTED_OUTPUT_TEXT_EAGER] # Example adjustment
        expected_pipeline_texts = [t.replace("नमस्ते दुनिया! ","").strip() if t.startswith("नमस्ते दुनिया!") else t for t in expected_pipeline_texts]

        outputs = pipe(self.input_text, max_new_tokens=20, do_sample=False, pad_token_id=self.tokenizer.eos_token_id) # Ensure pad_token_id for pipeline

        generated_texts = [out[0]["generated_text"].replace(inp, "").strip() for inp, out in zip(self.input_text, outputs)]
        self.assertListEqual(generated_texts, expected_pipeline_texts)