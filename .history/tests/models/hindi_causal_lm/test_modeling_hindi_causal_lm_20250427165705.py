# Copyright 2025 The HuggingFace Team. All rights reserved.
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

"""Testing suite for the Hindi Causal LM model."""

import unittest

from transformers import is_torch_available
from transformers.models.hindi_causal_lm.configuration_hindi_causal_lm import HindiCausalLMConfig
from transformers.testing_utils import require_sentencepiece, require_tokenizers, require_torch, slow, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask


if is_torch_available():
    import torch

    from transformers.models.hindi_causal_lm.modeling_hindi_causal_lm import (
        HindiCausalLMForCausalLM,
        HindiCausalLMModel,
        HindiCausalLMPreTrainedModel, # Import PreTrainedModel if needed for specific tests
    )
    # Make sure dummy objects are correctly handled or imported if torch isn't available
else:
    # Define dummy classes or import from dummy module if torch is not available
    # This structure assumes dummy objects are defined elsewhere if torch is missing.
    # For simplicity in this context, we'll assume torch is available for the test definitions.
    # If you have dummy_pt_objects.py, import from there.
    # from transformers.models.hindi_causal_lm.dummy_pt_objects import (...)
    pass


class HindiCausalLMModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=False, # HindiCausalLM doesn't use token_type_ids
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4, # Ensure hidden_size is divisible by this
        intermediate_size=64,
        hidden_act="silu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        # type_vocab_size=16, # Not used
        # type_sequence_label_size=2, # Not used for Causal LM
        initializer_range=0.02,
        scope=None,
        pad_token_id=0, # Define pad_token_id
        bos_token_id=1,
        eos_token_id=2,
        normalization_layer="rmsnorm",
        positional_encoding_type="rope",
        rope_theta=10000.0, # Add rope_theta for RoPE
        tie_word_embeddings=True, # Define tie_word_embeddings
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids # Keep False
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        # self.type_vocab_size = type_vocab_size
        # self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.scope = scope
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.normalization_layer = normalization_layer
        self.positional_encoding_type = positional_encoding_type
        self.rope_theta = rope_theta
        self.tie_word_embeddings = tie_word_embeddings

        # Ensure hidden_size is divisible by num_attention_heads
        if self.hidden_size % self.num_attention_heads != 0:
             # Adjust hidden_size to be divisible
             self.hidden_size = (self.hidden_size // self.num_attention_heads) * self.num_attention_heads
             if self.hidden_size == 0: # Prevent hidden_size 0 if heads > initial hidden_size
                 self.hidden_size = self.num_attention_heads
             print(f"Adjusted hidden_size to {self.hidden_size} to be divisible by {self.num_attention_heads} heads.")


    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])
        token_type_ids = None # Not used

        # Labels for Causal LM
        token_labels = None
        if self.use_labels:
            # For Causal LM, labels are usually the same as input_ids shifted
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        config = self.get_config()

        # Removed sequence_labels and choice_labels as they aren't used for Causal LM base tests
        return config, input_ids, token_type_ids, input_mask, token_labels # Return token_labels

    def get_config(self):
        # Returns a configuration object for the model
        return HindiCausalLMConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            normalization_layer=self.normalization_layer,
            positional_encoding_type=self.positional_encoding_type,
            rope_theta=self.rope_theta,
            tie_word_embeddings=self.tie_word_embeddings,
            use_cache=True, # Enable cache for generation tests
        )

    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, token_labels # Adjusted signature
    ):
        # Test the base model (HindiCausalLMModel)
        # Need to instantiate with config only if HindiCausalLMModel is available
        if not is_torch_available(): return

        model = HindiCausalLMModel(config=config)
        model.to(torch_device)
        model.eval()

        # Test forward pass with different combinations of inputs
        result = model(input_ids, attention_mask=input_mask) # Removed token_type_ids
        result = model(input_ids) # Test without attention_mask

        # Check output shape
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_causal_lm(
        self, config, input_ids, token_type_ids, input_mask, token_labels # Adjusted signature
    ):
         # Test the Causal LM head model (HindiCausalLMForCausalLM)
        if not is_torch_available(): return

        model = HindiCausalLMForCausalLM(config=config)
        model.to(torch_device)
        model.eval()

        # Test forward pass with labels
        result = model(input_ids, attention_mask=input_mask, labels=token_labels) # Removed token_type_ids
        # Check output shapes
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))
        # Check loss calculation (optional, but good)
        self.parent.assertIsNotNone(result.loss)

        # Test forward pass without labels (inference)
        result_no_labels = model(input_ids, attention_mask=input_mask)
        self.parent.assertEqual(result_no_labels.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))
        self.parent.assertIsNone(result_no_labels.loss) # Loss should be None

    def prepare_config_and_inputs_for_common(self):
        # Prepares config and a dictionary of inputs for common tests like serialization
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, _, input_mask, _ = config_and_inputs # Adjusted unpacking
        # Common tests often only need input_ids and attention_mask
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class HindiCausalLMModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    # Define the model classes to be tested
    all_model_classes = (HindiCausalLMModel, HindiCausalLMForCausalLM) if is_torch_available() else ()
    # Define the generative model classes (those with a generate method)
    all_generative_model_classes = (HindiCausalLMForCausalLM,) if is_torch_available() else ()

    # Tests to skip or modify
    test_pruning = False # Skip pruning tests if not supported/implemented
    test_head_masking = False # Skip head masking if not supported
    test_missing_keys = False # Skip if expected missing keys (like position_ids) are handled
    is_encoder_decoder = False # This is a decoder-only model

    def setUp(self):
        # Set up the model tester and config tester
        self.model_tester = HindiCausalLMModelTester(self)
        # Use a hidden_size divisible by potential head counts for ConfigTester
        self.config_tester = ConfigTester(self, config_class=HindiCausalLMConfig, hidden_size=32)

    def test_config(self):
        # Run common configuration tests
        self.config_tester.run_common_tests()

    def test_model(self):
        # Test the base model creation and forward pass
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_causal_lm(self):
        # Test the Causal LM head model creation and forward pass
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_causal_lm(*config_and_inputs)

    @slow # Mark as slow test
    def test_model_from_pretrained(self):
        # Test loading the model from a pretrained checkpoint on the Hub
        model_name = "convaiinnovations/hindi-foundational-model-base"
        try:
            model = HindiCausalLMModel.from_pretrained(model_name)
            self.assertIsNotNone(model)
            model_lm = HindiCausalLMForCausalLM.from_pretrained(model_name)
            self.assertIsNotNone(model_lm)
        except EnvironmentError as e:
             # Handle cases where the model might not be downloadable in restricted environments
             self.skipTest(f"Could not download pretrained model: {e}")
        except Exception as e:
             self.fail(f"Loading pretrained model failed with an unexpected error: {e}")


    def test_generate(self):
        """Test that the model can generate text using GenerationMixin."""
        if not is_torch_available(): return

        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        config = config_and_inputs[0]
        # --- Ensure pad_token_id is set in the test config ---
        # Use the pad_token_id from the tester or set a default if needed
        config.pad_token_id = self.model_tester.pad_token_id if hasattr(self.model_tester, 'pad_token_id') else 0
        config.eos_token_id = self.model_tester.eos_token_id if hasattr(self.model_tester, 'eos_token_id') else 2
        config.bos_token_id = self.model_tester.bos_token_id if hasattr(self.model_tester, 'bos_token_id') else 1
        config.use_cache = True # Ensure cache is enabled for generation

        # Use the Causal LM model for generation tests
        model = HindiCausalLMForCausalLM(config)
        model.to(torch_device)
        model.eval()

        # Prepare a simple input
        input_ids = ids_tensor([self.model_tester.batch_size, self.model_tester.seq_length], config.vocab_size).to(torch_device)
        attention_mask = torch.ones_like(input_ids).to(torch_device) # Assume all input tokens are attended

        # --- Test Greedy Generation ---
        # Keep max_length short for testing speed
        max_gen_length = self.model_tester.seq_length + 5
        generated_output_greedy = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask, # Pass attention mask
            max_length=max_gen_length,
            do_sample=False, # Greedy search
            pad_token_id=config.pad_token_id,
            eos_token_id=config.eos_token_id,
            bos_token_id=config.bos_token_id,
        )
        self.assertIsNotNone(generated_output_greedy)

        # --- Extract the tensor before checking shape ---
        if isinstance(generated_output_greedy, torch.Tensor):
            generated_ids_tensor_greedy = generated_output_greedy
        elif hasattr(generated_output_greedy, "sequences"):
            generated_ids_tensor_greedy = generated_output_greedy.sequences
        else:
            self.fail(f"Greedy generate() returned unexpected type: {type(generated_output_greedy)}")

        # --- Assertions for Greedy ---
        self.assertEqual(generated_ids_tensor_greedy.shape[0], self.model_tester.batch_size) # Batch size should match
        # Generated length should be <= max_length
        self.assertTrue(generated_ids_tensor_greedy.shape[1] <= max_gen_length)
        # Should generate something longer than input unless EOS is hit early
        # self.assertTrue(generated_ids_tensor_greedy.shape[1] > self.model_tester.seq_length) # This might fail if EOS is predicted immediately

        # --- Test Sampling Generation --- (Optional but good)
        generated_output_sample = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_gen_length,
            do_sample=True, # Enable sampling
            top_k=50,
            top_p=0.95,
            pad_token_id=config.pad_token_id,
            eos_token_id=config.eos_token_id,
            bos_token_id=config.bos_token_id,
        )
        self.assertIsNotNone(generated_output_sample)

        if isinstance(generated_output_sample, torch.Tensor):
            generated_ids_tensor_sample = generated_output_sample
        elif hasattr(generated_output_sample, "sequences"):
            generated_ids_tensor_sample = generated_output_sample.sequences
        else:
            self.fail(f"Sample generate() returned unexpected type: {type(generated_output_sample)}")

        self.assertEqual(generated_ids_tensor_sample.shape[0], self.model_tester.batch_size)
        self.assertTrue(generated_ids_tensor_sample.shape[1] <= max_gen_length)

    def test_weight_initialization(self):
        """Tests weight initialization, including embedding tying."""
        if not is_torch_available(): return

        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = HindiCausalLMForCausalLM(config) # Test on the LM head model

        # Check initialization std deviation (approximate)
        # This is a basic check, more rigorous checks can be added
        for name, param in model.named_parameters():
            if param.requires_grad and param.ndim > 1: # Check linear/embedding weights
                # Check std dev is close to initializer_range (allow some tolerance)
                self.assertLess(abs(param.data.std() - config.initializer_range), 0.1,
                                msg=f"Std dev of {name} is not close to initializer_range {config.initializer_range}")

        # --- Check that weights are properly tied ---
        if config.tie_word_embeddings:
            # Get pointers to the weights
            input_embedding_weight = model.get_input_embeddings().weight
            output_embedding_weight = model.get_output_embeddings().weight
            # Check if they point to the same memory location
            self.assertIs(input_embedding_weight, output_embedding_weight, "Word embeddings are not tied")
            # Optionally, check if values are identical (redundant if pointers match, but safe)
            self.assertTrue(torch.allclose(input_embedding_weight, output_embedding_weight), "Tied weights values differ")
        else:
             # If not tied, ensure they are different objects (unless initialized identically by chance)
             self.assertIsNot(model.get_input_embeddings().weight, model.get_output_embeddings().weight,
                              "Word embeddings should not be tied but are")

        # Check that lm_head weight has the correct shape
        self.assertEqual(model.lm_head.weight.shape, (config.vocab_size, config.hidden_size))


@require_sentencepiece
@require_tokenizers
@require_torch
class HindiCausalLMIntegrationTest(unittest.TestCase):

    @slow # Mark as slow test
    def test_inference_causal_lm(self):
        """Tests forward pass with a pretrained model and specific input."""
        model_name = "convaiinnovations/hindi-foundational-model-base"
        try:
            model = HindiCausalLMForCausalLM.from_pretrained(model_name)
            model.to(torch_device)
            model.eval()
        except EnvironmentError as e:
            self.skipTest(f"Cannot download model {model_name}: {e}")
        except Exception as e:
            self.fail(f"Loading integration test model failed: {e}")

        # Hindi text: "हिंदी भाषा" - Use IDs corresponding to a real tokenizer if possible
        # Example IDs (replace with actual IDs from your tokenizer for "हिंदी भाषा")
        # Assuming <s>=1, <pad>=0, </s>=2, <unk>=3
        # Tokenize "हिंदी भाषा" with your actual tokenizer to get correct IDs
        # E.g., if tokenizer.encode("हिंदी भाषा") -> [47, 5096, 4329, 3697, 2567, 956]
        # Need to add BOS token if model expects it
        # input_ids = torch.tensor([[1, 47, 5096, 4329, 3697, 2567, 956]], device=torch_device) # Example with BOS
        input_ids = torch.tensor([[47, 5096, 4329, 3697, 2567, 956]], device=torch_device) # Example without BOS
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask) # Pass attention_mask

        # Check logits shape: [batch_size, seq_length, vocab_size]
        # Get vocab size from the loaded model's config
        expected_vocab_size = model.config.vocab_size # Should be 16000 for this model
        expected_shape = torch.Size([1, input_ids.shape[1], expected_vocab_size])
        self.assertEqual(output.logits.shape, expected_shape)

        # Optional: Check if loss is None when no labels are provided
        self.assertIsNone(output.loss)