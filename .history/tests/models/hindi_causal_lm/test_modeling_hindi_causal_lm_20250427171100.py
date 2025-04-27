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
        HindiCausalLMPreTrainedModel,
    )
else:
    # Import dummy objects if torch is not available
    from transformers.models.hindi_causal_lm.dummy_pt_objects import (
        HindiCausalLMForCausalLM,
        HindiCausalLMModel,
        HindiCausalLMPreTrainedModel,
     )


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
        initializer_range=0.02,
        scope=None,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        unk_token_id=3, # Add unk token id
        normalization_layer="rmsnorm",
        positional_encoding_type="rope",
        rope_theta=10000.0,
        tie_word_embeddings=True,
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
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.scope = scope
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.unk_token_id = unk_token_id
        self.normalization_layer = normalization_layer
        self.positional_encoding_type = positional_encoding_type
        self.rope_theta = rope_theta
        self.tie_word_embeddings = tie_word_embeddings

        # Ensure hidden_size is divisible by num_attention_heads
        if self.hidden_size % self.num_attention_heads != 0:
             self.hidden_size = (self.hidden_size // self.num_attention_heads) * self.num_attention_heads
             if self.hidden_size == 0:
                 self.hidden_size = self.num_attention_heads
             print(f"Adjusted hidden_size to {self.hidden_size} to be divisible by {self.num_attention_heads} heads.")


    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])
        token_type_ids = None # Not used

        token_labels = None
        if self.use_labels:
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        config = self.get_config()
        return config, input_ids, token_type_ids, input_mask, token_labels

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
            unk_token_id=self.unk_token_id,
            normalization_layer=self.normalization_layer,
            positional_encoding_type=self.positional_encoding_type,
            rope_theta=self.rope_theta,
            tie_word_embeddings=self.tie_word_embeddings,
            use_cache=True, # Important for generation tests
        )

    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, token_labels # Use token_labels
    ):
        if not is_torch_available():
             return

        model = HindiCausalLMModel(config=config)
        model.to(torch_device)
        model.eval()

        # Test forward pass with different combinations of inputs
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)

        # Check base model output shape
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        # Base model doesn't return loss or logits
        self.parent.assertNotIn("loss", result)
        self.parent.assertNotIn("logits", result)


    def create_and_check_for_causal_lm(
        self, config, input_ids, token_type_ids, input_mask, token_labels # Use token_labels
    ):
        if not is_torch_available():
             return

        model = HindiCausalLMForCausalLM(config=config)
        model.to(torch_device)
        model.eval()

        # Test forward pass with labels
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))
        self.parent.assertIsNotNone(result.loss)

        # Test forward pass without labels (inference)
        result_no_labels = model(input_ids, attention_mask=input_mask)
        self.parent.assertEqual(result_no_labels.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))
        self.parent.assertIsNone(result_no_labels.loss)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, _, input_mask, _ = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class HindiCausalLMModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (HindiCausalLMModel, HindiCausalLMForCausalLM) if is_torch_available() else ()
    all_generative_model_classes = (HindiCausalLMForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = ( # Define mapping for pipeline tests
        {
            "feature-extraction": HindiCausalLMModel,
            "text-generation": HindiCausalLMForCausalLM,
            # Add other tasks if classification/token heads are implemented
        }
        if is_torch_available()
        else {}
    )

    # Tests to skip or modify based on model capabilities
    test_pruning = False
    test_head_masking = False
    test_missing_keys = False # Check this carefully - may need to ignore position_ids
    is_encoder_decoder = False
    # Skip tests known to fail with tuple caches or specific generation features if not supported
    # E.g., Gemma2 skips many tests due to HybridCache
    @unittest.skip("HindiCausalLM uses tuple cache, incompatible with this test.")
    def test_new_cache_format(self):
        pass
    # Add other skips based on CI/CD failures for unsupported features

    def setUp(self):
        self.model_tester = HindiCausalLMModelTester(self)
        self.config_tester = ConfigTester(self, config_class=HindiCausalLMConfig, hidden_size=32) # Use divisible hidden_size

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_causal_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_causal_lm(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        model_name = "convaiinnovations/hindi-foundational-model-base"
        try:
            # Test loading both base and LM model
            model_base = HindiCausalLMModel.from_pretrained(model_name)
            self.assertIsNotNone(model_base)
            model_lm = HindiCausalLMForCausalLM.from_pretrained(model_name)
            self.assertIsNotNone(model_lm)
        except EnvironmentError as e:
             self.skipTest(f"Could not download pretrained model {model_name}: {e}")
        except Exception as e:
             self.fail(f"Loading pretrained model {model_name} failed with an unexpected error: {e}")

    # test_generate is inherited from GenerationTesterMixin and should work
    # if prepare_inputs_for_generation and _reorder_cache are correct.
    # We add a specific simpler generate test below for clarity.

    def test_simple_generate_greedy(self):
        """Test basic greedy generation explicitly."""
        if not is_torch_available():
            return

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.pad_token_id = self.model_tester.pad_token_id
        config.eos_token_id = self.model_tester.eos_token_id
        config.bos_token_id = self.model_tester.bos_token_id
        config.use_cache = True # Essential for generation

        model = HindiCausalLMForCausalLM(config)
        model.to(torch_device)
        model.eval()

        input_ids = inputs_dict["input_ids"]
        attention_mask = inputs_dict["attention_mask"]

        # Generate using greedy decoding
        max_gen_length = self.model_tester.seq_length + 10
        generated_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_gen_length,
            do_sample=False, # Greedy
            pad_token_id=config.pad_token_id,
            eos_token_id=config.eos_token_id,
        )
        self.assertIsNotNone(generated_output)

        # Extract the tensor (handles both tensor and object output)
        if isinstance(generated_output, torch.Tensor):
            generated_ids_tensor = generated_output
        elif hasattr(generated_output, "sequences"):
            generated_ids_tensor = generated_output.sequences
        else:
            self.fail(f"generate() returned unexpected type: {type(generated_output)}")

        # Basic shape checks
        self.assertEqual(generated_ids_tensor.shape[0], self.model_tester.batch_size)
        self.assertTrue(generated_ids_tensor.shape[1] <= max_gen_length)
        # Check if it generated at least one token beyond input (unless EOS)
        self.assertTrue(generated_ids_tensor.shape[1] > self.model_tester.seq_length)


    def test_weight_initialization(self):
        """Tests weight initialization, including embedding tying."""
        if not is_torch_available():
             return

        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = HindiCausalLMForCausalLM(config)

        # Check initialization std dev
        for name, param in model.named_parameters():
            if param.requires_grad and param.ndim > 1: # Check weights, not biases/norms
                # Allow slightly larger tolerance for initialization checks
                self.assertLess(abs(param.data.std() - config.initializer_range), 0.1 * config.initializer_range + 0.01,
                                msg=f"Std dev of {name} ({param.data.std():.4f}) is not close to initializer_range {config.initializer_range}")

        # Check embedding tying
        if config.tie_word_embeddings:
            input_embedding_weight = model.get_input_embeddings().weight
            output_embedding_weight = model.get_output_embeddings().weight
            self.assertIs(input_embedding_weight, output_embedding_weight, "Word embeddings are not tied")
            self.assertTrue(torch.allclose(input_embedding_weight.data, output_embedding_weight.data), "Tied weights values differ")
        else:
             if model.get_input_embeddings() is not None and model.get_output_embeddings() is not None:
                 self.assertIsNot(model.get_input_embeddings().weight, model.get_output_embeddings().weight,
                                  "Word embeddings should not be tied but are")

        # Check lm_head shape
        self.assertEqual(model.lm_head.weight.shape, (config.vocab_size, config.hidden_size))


@require_sentencepiece
@require_tokenizers
@require_torch
class HindiCausalLMIntegrationTest(unittest.TestCase):

    @slow
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

        # Example Hindi text: "हिंदी भाषा"
        # These IDs need to come from the actual tokenizer for "hindi-foundational-model-base"
        # Example IDs (replace if needed): <s> हिंदी भाषा </s> -> [1, 47, 5096, 4329, 3697, 2567, 956, 2]
        # Input for generation often omits the final EOS. Input for forward pass usually includes it.
        input_ids = torch.tensor([[1, 47, 5096, 4329, 3697, 2567, 956]], device=torch_device) # With BOS
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)

        # Check logits shape: [batch_size, seq_length, vocab_size]
        expected_vocab_size = model.config.vocab_size # Should be 16000 for this model
        expected_shape = torch.Size([1, input_ids.shape[1], expected_vocab_size])

        self.assertIsNotNone(output.logits)
        self.assertEqual(output.logits.shape, expected_shape)

        # Check loss is None when no labels are provided
        self.assertIsNone(output.loss)

    @slow
    def test_generation_integration(self):
        """Tests generation with a pretrained model."""
        model_name = "convaiinnovations/hindi-foundational-model-base"
        try:
            model = HindiCausalLMForCausalLM.from_pretrained(model_name)
            model.to(torch_device)
            model.eval()
            # Assuming the tokenizer is available via AutoTokenizer or custom class
            # For this test, we'll focus on the model's ability to generate IDs
            # tokenizer = AutoTokenizer.from_pretrained(model_name)
        except EnvironmentError as e:
            self.skipTest(f"Cannot download model {model_name}: {e}")
        except Exception as e:
            self.fail(f"Loading integration test model failed: {e}")

        # Input prompt: "भारत की राजधानी" (Capital of India) -> Needs tokenization
        # Example IDs (replace with actual): [1, 150, 439, 1858, 4329] # Includes BOS
        input_ids = torch.tensor([[1, 150, 439, 1858, 4329]], device=torch_device)
        max_length = input_ids.shape[1] + 10 # Generate a few tokens

        with torch.no_grad():
            generated_output = model.generate(
                input_ids,
                max_length=max_length,
                do_sample=False # Use greedy for deterministic output
            )

        self.assertIsNotNone(generated_output)
        # Extract tensor
        if hasattr(generated_output, "sequences"):
            generated_ids = generated_output.sequences
        else:
            generated_ids = generated_output

        self.assertEqual(generated_ids.shape[0], 1) # Batch size 1
        self.assertTrue(generated_ids.shape[1] > input_ids.shape[1]) # Check it generated something
        self.assertTrue(generated_ids.shape[1] <= max_length) # Check max length

        # Optional: Decode and check if the output makes sense (e.g., " नई दिल्ली")
        # generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        # print(f"Generated text: {generated_text}") # Manual inspection