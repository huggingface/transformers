# Copyright 2024 Convai Innovations Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch ConvaiCausalLM model."""

import tempfile
import unittest

import pytest

from transformers import AutoModelForCausalLM, AutoTokenizer, ConvaiCausalLMConfig, is_torch_available
from transformers.testing_utils import (
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
        ConvaiCausalLMForCausalLM,
        # ConvaiCausalLMForSequenceClassification, # Add if implemented
        # ConvaiCausalLMForTokenClassification, # Add if implemented
        ConvaiCausalLMModel,
    )


@require_torch
class ConvaiCausalLMModelTester:
    config_class = ConvaiCausalLMConfig
    if is_torch_available():
        model_class = ConvaiCausalLMModel
        for_causal_lm_class = ConvaiCausalLMForCausalLM
        # for_sequence_class = ConvaiCausalLMForSequenceClassification # Add if implemented
        # for_token_class = ConvaiCausalLMForTokenClassification # Add if implemented

    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=False, # Typically False for decoder-only models
        use_labels=True,
        # Use smaller, testing-friendly values, but keep vocab consistent if needed
        vocab_size=1000, # Smaller vocab for faster tests unless 16000 is strictly needed
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2, # Test GQA
        intermediate_size=37,
        hidden_act="silu", # From user config
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=64, # Smaller for tests
        type_vocab_size=16, # Keep default, though likely unused
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        pad_token_id=0, # From user config
        bos_token_id=1, # From user config
        eos_token_id=2, # From user config
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
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.scope = scope
        # Calculate head_dim based on test config
        self.head_dim = self.hidden_size // self.num_attention_heads

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size, extra_indices=[self.pad_token_id]) # Ensure pad_token_id might be generated
        input_ids[input_ids == self.pad_token_id] = self.vocab_size - 1 # Replace pad with another valid token for testing logic

        input_mask = None
        if self.use_input_mask:
            # Create a causal mask for decoder-only model testing
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

    def get_config(self):
        # Returns a ConvaiCausalLMConfig with the parameters defined in __init__
        return self.config_class(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            # hidden_dropout_prob=self.hidden_dropout_prob, # Often not in base config, used in specific heads
            attention_dropout=self.attention_probs_dropout_prob, # Use attention_dropout
            max_position_embeddings=self.max_position_embeddings,
            # type_vocab_size=self.type_vocab_size, # Not typically part of causal LM config
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            head_dim=self.head_dim,
            rms_norm_eps=1e-6, # Keep default or user value
            use_cache=True,
            # is_decoder=True, # Implicit for causal LM
        )

    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = self.model_class(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids) # Test without mask if supported
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
        # Add position_ids if the model expects them explicitly, otherwise they are usually created internally
        # inputs_dict["position_ids"] = torch.arange(0, self.seq_length).unsqueeze(0).repeat(self.batch_size, 1).to(torch_device)
        return config, inputs_dict


@require_torch
class ConvaiCausalLMModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (ConvaiCausalLMModel, ConvaiCausalLMForCausalLM) # Add other heads like ConvaiCausalLMForSequenceClassification if implemented
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": ConvaiCausalLMModel,
            "text-generation": ConvaiCausalLMForCausalLM,
            # Add other mappings if heads are implemented
            # "text-classification": ConvaiCausalLMForSequenceClassification,
            # "token-classification": ConvaiCausalLMForTokenClassification,
            # "zero-shot": ConvaiCausalLMForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False # Usually False for Causal LMs
    test_pruning = False # Usually False for Causal LMs
    # Need to remove 0.9 in `test_cpu_offload`
    # This is because we are hitting edge cases with the causal_mask buffer
    model_split_percents = [0.5, 0.6] # Keep from Gemma

    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = ConvaiCausalLMForCausalLM if is_torch_available() else None

    # TODO (ydshieh): Check this. See https://app.circleci.com/pipelines/github/huggingface/transformers/79245/workflows/9490ef58-79c2-410d-8f51-e3495156cf9c/jobs/1012146
    # Keep this skip from Gemma for now, may need adjustment based on model capabilities
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
        # Example: Skip text-classification if the head is not implemented
        if pipeline_test_case_name == "TextClassificationPipelineTests" and ConvaiCausalLMForSequenceClassification not in self.all_model_classes:
             return True
        if pipeline_test_case_name == "TokenClassificationPipelineTests" and ConvaiCausalLMForTokenClassification not in self.all_model_classes:
             return True
        # Inherited skip logic from Gemma - review if necessary
        if pipeline_test_case_name == "ZeroShotClassificationPipelineTests" and ConvaiCausalLMForSequenceClassification not in self.all_model_classes:
            return True

        return False # Default to not skipping

    def setUp(self):
        self.model_tester = ConvaiCausalLMModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ConvaiCausalLMConfig, hidden_size=37)

    def test_config(self):
        # Test common configuration patterns
        self.config_tester.run_common_tests()

    def test_model(self):
        # Test the base model forward pass
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    # Commenting out sequence/token classification tests as the heads are not in the provided modeling code
    # def test_ConvaiCausalLM_sequence_classification_model(self):
    #     config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
    #     config.num_labels = 3
    #     input_ids = input_dict["input_ids"]
    #     attention_mask = input_ids.ne(self.model_tester.pad_token_id).to(torch_device) # Use actual pad_token_id
    #     sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
    #     model = self.model_tester.for_sequence_class(config) # Ensure this class exists
    #     model.to(torch_device)
    #     model.eval()
    #     result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
    #     self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    # def test_ConvaiCausalLM_token_classification_model(self):
    #     config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
    #     config.num_labels = 3
    #     input_ids = input_dict["input_ids"]
    #     attention_mask = input_ids.ne(self.model_tester.pad_token_id).to(torch_device) # Use actual pad_token_id
    #     token_labels = ids_tensor([self.model_tester.batch_size, self.model_tester.seq_length], config.num_labels)
    #     model = self.model_tester.for_token_class(config=config) # Ensure this class exists
    #     model.to(torch_device)
    #     model.eval()
    #     result = model(input_ids, attention_mask=attention_mask, labels=token_labels)
    #     self.assertEqual(
    #         result.logits.shape,
    #         (self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.num_labels),
    #     )

    # --- SDPA / Flash Attention Tests (Adapted from Gemma) ---

    @require_torch_sdpa
    @require_torch_accelerator
    @slow
    def test_sdpa_equivalence(self):
        # Test equivalence between eager and SDPA attention implementations
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
                model_sdpa.eval()

                model_eager = model_class.from_pretrained(tmpdirname, torch_dtype=torch.float16, attn_implementation="eager")
                model_eager.to(torch_device)
                model_eager.eval()

                # Get inputs and move to device
                inputs = {k: v.to(torch_device) for k, v in inputs_dict.items()}

                # Run forward passes
                with torch.no_grad():
                    outputs_eager = model_eager(**inputs, output_hidden_states=True)
                    outputs_sdpa = model_sdpa(**inputs, output_hidden_states=True)

                # Compare last hidden states (adjust tolerance if needed)
                # Use a slightly higher tolerance as SDPA can have small differences
                self.assertTrue(
                    torch.allclose(outputs_sdpa.last_hidden_state, outputs_eager.last_hidden_state, atol=1e-3)
                )

                # Compare hidden states if output_hidden_states=True
                if config.output_hidden_states:
                    for h_eager, h_sdpa in zip(outputs_eager.hidden_states, outputs_sdpa.hidden_states):
                         self.assertTrue(torch.allclose(h_sdpa, h_eager, atol=1e-3))


    @require_flash_attn
    @require_torch_gpu
    @pytest.mark.flash_attn_test
    @slow
    def test_flash_attn_2_inference_equivalence(self):
        # Test equivalence between eager and Flash Attention 2 implementations
        for model_class in self.all_model_classes:
            if not model_class._supports_flash_attn_2:
                self.skipTest(reason="Model does not support Flash Attention 2")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            # Flash Attn needs sequence length multiple of 8
            if inputs_dict["input_ids"].shape[1] % 8 != 0:
                config.max_position_embeddings = (inputs_dict["input_ids"].shape[1] // 8 + 1) * 8
                self.model_tester.seq_length = config.max_position_embeddings
                config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()


            with tempfile.TemporaryDirectory() as tmpdirname:
                model_eager = model_class(config, attn_implementation="eager")
                model_eager.save_pretrained(tmpdirname)

                model_fa = model_class.from_pretrained(
                    tmpdirname, torch_dtype=torch.float16, attn_implementation="flash_attention_2"
                )
                model_fa.to(torch_device)
                model_fa.eval()

                model_eager = model_class.from_pretrained(tmpdirname, torch_dtype=torch.float16, attn_implementation="eager")
                model_eager.to(torch_device)
                model_eager.eval()

                inputs = {k: v.to(torch_device) for k, v in inputs_dict.items()}

                with torch.no_grad():
                    outputs_eager = model_eager(**inputs)
                    outputs_fa = model_fa(**inputs)

                # Compare last hidden states (adjust tolerance if needed)
                # Flash Attention might require a slightly higher tolerance
                self.assertTrue(
                    torch.allclose(outputs_fa.last_hidden_state, outputs_eager.last_hidden_state, atol=5e-3)
                )


@slow
@require_torch_accelerator
class ConvaiCausalLMIntegrationTest(unittest.TestCase):
    # NOTE: Replace these with actual expected outputs after running with your model
    input_text = ["भारत एक विशाल देश है", "नमस्ते दुनिया"]
    EXPECTED_TEXTS_GENERATION = [
        "भारत एक विशाल देश है, जिसकी संस्कृति", # REPLACE THIS
        "नमस्ते दुनिया, मैं एक भाषा मॉडल हूँ", # REPLACE THIS
    ]
    # Logits tests need expected values derived from running the model
    EXPECTED_LOGITS_SLICE = [1.0, 2.0, 3.0, 4.0, 5.0] # REPLACE THIS with actual slice, e.g., logits[0, -1, 100:105]
    EXPECTED_MEAN_LOGITS = [
        [1.1, 1.2, 1.3], # REPLACE THIS (shape: batch_size, seq_len)
        [2.1, 2.2, 2.3], # REPLACE THIS
    ]

    # Point to your model on the Hub
    model_id = "convaiinnovations/hindi-causal-lm"

    @require_read_token # Add if your model is private/gated
    def test_model_generation_fp16(self):
        # Test generation with FP16
        model = AutoModelForCausalLM.from_pretrained(self.model_id, low_cpu_mem_usage=True, torch_dtype=torch.float16).to(
            torch_device
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        # Adjust generation parameters as needed for your model
        output = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        # ** Replace EXPECTED_TEXTS_GENERATION with the actual output you get **
        print("Actual Generation Output:", output_text) # Print to help get the expected value
        # self.assertEqual(output_text, self.EXPECTED_TEXTS_GENERATION) # UNCOMMENT and ASSERT after replacing

    @require_read_token # Add if needed
    def test_model_logits_bf16(self):
        # Test logits with BF16
        model = AutoModelForCausalLM.from_pretrained(self.model_id, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16).to(
            torch_device
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # ** Replace EXPECTED_MEAN_LOGITS and EXPECTED_LOGITS_SLICE with actual values **
        print("Actual Logits Mean:", logits.mean(-1).cpu().numpy())
        print("Actual Logits Slice [0, -1, 100:105]:", logits[0, -1, 100:105].cpu().numpy())
        # self.assertTrue(np.allclose(logits.mean(-1).cpu().numpy(), self.EXPECTED_MEAN_LOGITS, atol=1e-2))
        # self.assertTrue(np.allclose(logits[0, -1, 100:105].cpu().numpy(), self.EXPECTED_LOGITS_SLICE, atol=1e-2)) # Adjust slice as needed
        pass # Remove pass and uncomment asserts after replacing placeholders

    @require_bitsandbytes
    @require_read_token # Add if needed
    def test_model_4bit_generation(self):
        # Test generation with 4-bit quantization
        model = AutoModelForCausalLM.from_pretrained(self.model_id, low_cpu_mem_usage=True, load_in_4bit=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        # ** Replace EXPECTED_TEXTS_4BIT with the actual output you get **
        print("Actual 4-bit Generation Output:", output_text)
        # self.assertEqual(output_text, EXPECTED_TEXTS_4BIT) # UNCOMMENT and ASSERT after replacing

    # Add more integration tests as needed (e.g., SDPA, Flash Attention 2, different dtypes)
    # following the structure of the Gemma tests, remembering to update expected outputs.
