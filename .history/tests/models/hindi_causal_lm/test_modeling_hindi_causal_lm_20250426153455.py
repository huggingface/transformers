# coding=utf-8
# Copyright 2024 The Convai Innovations Authors and The HuggingFace Team. All rights reserved.
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
""" Testing suite for the PyTorch HindiCausalLM model. """
import unittest

from transformers import HindiCausalLMConfig, is_torch_available
from transformers.testing_utils import require_sentencepiece, require_tokenizers, require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers.models.hindi_causal_lm.modeling_hindi_causal_lm import (
        HindiCausalLMForCausalLM,
        HindiCausalLMModel,
    )


# Define a Model Tester specific to HindiCausalLM
class HindiCausalLMModelTester:
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
        num_attention_heads=4, # Must divide hidden_size
        intermediate_size=37, # Can be different
        hidden_act="silu",
        max_position_embeddings=512,
        rms_norm_eps=1e-6, # Use a common value for testing stability
        initializer_range=0.02,
        pad_token_id=0,
        eos_token_id=2,
        bos_token_id=1,
        scope=None, # Keeps track of shared tensors
        # Add other relevant config parameters if needed for testing
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
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.scope = scope
        # Note: RoPE is implicitly tested via forward pass

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        lm_labels = None
        if self.use_labels:
            lm_labels = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        config = self.get_config()

        return config, input_ids, input_mask, lm_labels

    def get_config(self):
        return HindiCausalLMConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            rms_norm_eps=self.rms_norm_eps,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            # Set use_cache to False for simpler testing, gradients etc.
            use_cache=False,
        )

    def create_and_check_model(self, config, input_ids, input_mask, lm_labels):
        model = HindiCausalLMModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids) # Test without mask
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_model_as_decoder(self, config, input_ids, input_mask, lm_labels):
        config.add_cross_attention = True # Required for encoder-decoder tests, but model is decoder-only
        model = HindiCausalLMModel(config) # Treat as decoder
        model.to(torch_device)
        model.eval()
        # Simulate cross-attention inputs (zeros) if needed by test suite, though model ignores them
        encoder_hidden_states = torch.zeros(self.batch_size, self.seq_length, config.hidden_size, device=torch_device)
        encoder_attention_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2).to(torch_device)
        # Decoder-only models don't use cross-attention, so these inputs are ignored
        # Check if the base class tests require handling this gracefully
        # result = model(input_ids, attention_mask=input_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask)
        # result = model(input_ids, attention_mask=input_mask, encoder_hidden_states=encoder_hidden_states)
        result = model(input_ids, attention_mask=input_mask) # Call without cross-attention args
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))


    def create_and_check_for_causal_lm(self, config, input_ids, input_mask, lm_labels):
        model = HindiCausalLMForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=lm_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, input_mask, lm_labels = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class HindiCausalLMModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (HindiCausalLMModel, HindiCausalLMForCausalLM) if is_torch_available() else ()
    all_generative_model_classes = (HindiCausalLMForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = (
         {"feature-extraction": HindiCausalLMModel, "text-generation": HindiCausalLMForCausalLM}
         if is_torch_available()
         else {}
    )
    test_head_masking = False # Head masking not standard/easily supported with complex attention like RoPE
    test_pruning = False # Pruning not standard
    test_missing_keys = False # Might fail due to custom components / RoPE internals
    test_model_parallel = False # Requires more setup

    # TODO: Check which common tests fail and potentially skip them if they rely on unsupported features
    # e.g., tests for features not implemented like encoder-decoder behavior, certain head types etc.

    def setUp(self):
        self.model_tester = HindiCausalLMModelTester(self)
        self.config_tester = ConfigTester(self, config_class=HindiCausalLMConfig, hidden_size=37) # Use odd hidden_size for tests

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_causal_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_causal_lm(*config_and_inputs)

    # Override tests that might fail due to custom/RoPE implementation if necessary
    # For example, gradient tests might be sensitive
    def test_training(self):
         # Skip if complex components cause issues, or debug
         super().test_training()

    def test_retain_grad_hidden_states_attentions(self):
         # Skip if RoPE or other components interfere with grad checks
         super().test_retain_grad_hidden_states_attentions()


    # Add slow tests for the actual pretrained model
    @slow
    @require_sentencepiece
    @require_tokenizers
    def test_model_from_pretrained(self):
        model_name = "convaiinnovations/hindi-foundational-model-base"
        # Ensure custom code is discoverable or registered before this test
        # This might require running tests within the transformers repo context
        # where the model files are placed correctly.
        try:
            model = HindiCausalLMModel.from_pretrained(model_name)
            self.assertIsNotNone(model)
            model = HindiCausalLMForCausalLM.from_pretrained(model_name)
            self.assertIsNotNone(model)
        except ImportError:
             self.skipTest("HindiCausalLM custom code not found or SentencePiece not installed.")
        except Exception as e:
             self.fail(f"Loading pretrained model failed with {e}")


# Add Tokenizer tests if needed (requires SentencePiece installed)
# class HindiCausalLMTokenizerTest(unittest.TestCase): ...

