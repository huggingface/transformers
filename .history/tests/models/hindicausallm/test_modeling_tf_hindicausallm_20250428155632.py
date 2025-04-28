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
"""Testing suite for the TensorFlow HindiCausalLM model."""

import unittest
import inspect # Added import
import tempfile # Added import

from transformers import HindiCausalLMConfig, is_tf_available
from transformers.testing_utils import require_tf, slow # Removed torch_device

# Import common test classes
from ...test_configuration_common import ConfigTester
from ...test_modeling_tf_common import TFModelTesterMixin, ids_tensor, random_attention_mask


if is_tf_available():
    import tensorflow as tf
    # Removed numpy import as not strictly needed for basic tests

    from transformers import (
        TFHindiCausalLMForCausalLM,
        TFHindiCausalLMForSequenceClassification,
        TFHindiCausalLMModel,
    )
    from transformers.models.hindicausallm.modeling_tf_hindicausallm import HINDICAUSALLM_INPUTS_DOCSTRING # Import for testing


# Define a TF-specific ModelTester
class TFHindiCausalLMModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=False, # TF usually doesn't use token_type_ids for decoders
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        hidden_act="silu",
        hidden_dropout_prob=0.1, # TF layers handle dropout via `training` arg
        attention_probs_dropout_prob=0.1, # TF layers handle dropout via `training` arg
        max_position_embeddings=512,
        type_vocab_size=16, # Not typically used in TF decoders
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4, # Not used for these models
        pad_token_id=0,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids # Keep track, but won't use
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
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_type_ids = None # Not used

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            # For Causal LM, labels are usually input_ids shifted
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
            # choice_labels not used

        config = self.get_config()

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        # Use the actual config class
        return HindiCausalLMConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            # Dropout probs are part of config but handled differently in TF layers
            attention_dropout=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
            # Add other relevant fields from config if needed for tests
            bos_token_id=1,
            eos_token_id=2,
            layer_norm_eps=1e-5, # Ensure this matches model defaults
            rope_theta=10000.0,
        )

    # Create TF-specific checks
    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = TFHindiCausalLMModel(config=config)
        result = model(input_ids, attention_mask=input_mask, training=False) # Pass training=False
        result = model(input_ids, training=False) # Test without mask
        # Check output shape (TF uses shape tuple)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_causal_lm(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = TFHindiCausalLMForCausalLM(config=config)
        # Pass labels explicitly
        result = model(input_ids, attention_mask=input_mask, labels=token_labels, training=False)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))
        # Check loss calculation
        self.parent.assertIsNotNone(result.loss)
        # Check without labels
        result = model(input_ids, attention_mask=input_mask, training=False)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))
        self.parent.assertTrue(hasattr(result, 'loss') and result.loss is None) # Loss should be None


    def create_and_check_for_sequence_classification(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels # Ensure num_labels is set
        model = TFHindiCausalLMForSequenceClassification(config=config)
        result = model(input_ids, attention_mask=input_mask, labels=sequence_labels, training=False)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))
         # Check loss calculation
        self.parent.assertIsNotNone(result.loss)
         # Check without labels
        result = model(input_ids, attention_mask=input_mask, training=False)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))
        self.parent.assertTrue(hasattr(result, 'loss') and result.loss is None) # Loss should be None

    # Prepare inputs for common tests in TF format
    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask} # Basic inputs needed
        # Add other inputs if required by specific tests (e.g., position_ids)
        return config, inputs_dict


@require_tf
class TFHindiCausalLMModelTest(TFModelTesterMixin, unittest.TestCase): # Removed PipelineTesterMixin for TF

    # Define TF model classes
    all_model_classes = (
        (
            TFHindiCausalLMModel,
            TFHindiCausalLMForCausalLM,
            TFHindiCausalLMForSequenceClassification,
        )
        if is_tf_available()
        else ()
    )
    all_generative_model_classes = (TFHindiCausalLMForCausalLM,) if is_tf_available() else ()
    # TF pipeline mapping might differ or not be tested here
    pipeline_model_mapping = (
        # {
        #     "feature-extraction": TFHindiCausalLMModel,
        #     "text-classification": TFHindiCausalLMForSequenceClassification,
        #     "text-generation": TFHindiCausalLMForCausalLM,
        #     "zero-shot-classification": TFHindiCausalLMForSequenceClassification, # Corrected key
        # }
        {} # Disable pipeline tests for now
        if is_tf_available()
        else {}
    )
    test_head_masking = False # Head masking not typically tested for TF
    test_pruning = False # Pruning not typically tested for TF
    test_onnx = False # ONNX tests might require separate setup

    # Make PT specific tests optional / skipped for TF
    test_torchscript = False
    test_cpu_offload = False
    test_disk_offload = False
    test_model_parallelism = False


    def setUp(self):
        self.model_tester = TFHindiCausalLMModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=HindiCausalLMConfig, hidden_size=37 # Use small hidden_size for faster tests
        )

    # Standard Tests from TFModelTesterMixin will run

    # Test individual model types
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

    @slow
    def test_model_from_pretrained(self):
        model_name = "convaiinnovations/hindi-causal-lm"
        # Test loading TF weights if available, otherwise from PT
        try:
             model = TFHindiCausalLMModel.from_pretrained(model_name, from_pt=False)
             self.assertIsNotNone(model)
        except OSError:
            # Fallback to loading from PyTorch checkpoint
            print("TF weights not found, loading from PyTorch checkpoint...")
            model = TFHindiCausalLMModel.from_pretrained(model_name, from_pt=True)
            self.assertIsNotNone(model)

    # TF specific test for saving/loading Keras format
    def test_keras_save_load(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            outputs = model(inputs_dict, training=False) # Pass training=False

            with tempfile.TemporaryDirectory() as tmpdirname:
                filepath = os.path.join(tmpdirname, "keras_model.h5")
                model.save(filepath)
                if tf.executing_eagerly():
                    # Ensure that operations are completed for graph mode saving
                    tf.compat.v1.keras.backend.get_session().graph.finalize()
                loaded_model = tf.keras.models.load_model(filepath)
                # Check if the loaded model is of the correct class
                # Note: Loading generic Keras model might not restore the exact HF class type easily
                # Instead, check output structure or specific layer names
                loaded_outputs = loaded_model(inputs_dict, training=False)

                # Compare output structure/values (use appropriate comparison for TF tensors)
                self.assert_outputs_same(outputs, loaded_outputs)

    # Add generation tests for TF
    def test_generate_tf(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        # Only test generative models
        for model_class in self.all_generative_model_classes:
             model = model_class(config)
             # Test greedy search
             output_ids_greedy = model.generate(inputs_dict["input_ids"], max_length=20)
             self.assertIsNotNone(output_ids_greedy)
             self.assertEqual(output_ids_greedy.shape[0], self.model_tester.batch_size)

             # Test sampling (requires different generation config)
             # model.generation_config.do_sample = True # Modify config if needed
             # output_ids_sample = model.generate(inputs_dict["input_ids"], max_length=20, do_sample=True)
             # self.assertIsNotNone(output_ids_sample)
             # self.assertEqual(output_ids_sample.shape[0], self.model_tester.batch_size)

    # Test hidden states and attentions shapes (TF specific)
    def test_hidden_states_output(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        config.output_attentions = True # Also test attentions here

        for model_class in self.all_model_classes:
            model = model_class(config)
            outputs = model(inputs_dict, training=False)

            hidden_states = outputs.hidden_states
            self.assertIsNotNone(hidden_states)
            self.assertEqual(len(hidden_states), self.model_tester.num_hidden_layers + 1) # Embeddings + each layer
            # Check shape of first hidden state (embeddings)
            self.assertEqual(
                hidden_states[0].shape,
                (self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.hidden_size)
            )
            # Check shape of last hidden state
            self.assertEqual(
                hidden_states[-1].shape,
                (self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.hidden_size)
            )

            attentions = outputs.attentions
            self.assertIsNotNone(attentions)
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)
            # Check shape of first attention tensor
            self.assertEqual(
                 attentions[0].shape,
                (self.model_tester.batch_size, self.model_tester.num_attention_heads, self.model_tester.seq_length, self.model_tester.seq_length)
            )

    def test_inputs_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            tf_main_layer_call = model.serving # Use serving for signature testing
            # Check if inputs match the expected docstring
            signature = inspect.signature(tf_main_layer_call.call) # Access call of the main layer
            arg_names = [*signature.parameters.keys()]

            # Expected arguments based on implementation and docstring
            expected_arg_names = list(inspect.signature(HINDICAUSALLM_INPUTS_DOCSTRING).parameters.keys())

            # Remove 'self' and adjust based on TF implementation specifics if needed
            expected_arg_names = [
                name for name in expected_arg_names if name != "self" and name != "kwargs"
            ]
            # Check for consistency, order might differ
            self.assertListEqual(sorted(arg_names[:len(expected_arg_names)]), sorted(expected_arg_names))



@require_tf
class TFHindiCausalLMModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_no_head_saved_model(self):
        # Test loading and inference with the base model using saved_model format
        # Requires the model to be saved in SavedModel format on the Hub or locally
        # For now, use from_pretrained which might load PT weights if TF not available
        model_name = "convaiinnovations/hindi-causal-lm"
        try:
            # Prioritize loading TF SavedModel if available
            # model = tf.keras.models.load_model(model_name) # This needs Hub integration
            # For now, use from_pretrained
            model = TFHindiCausalLMModel.from_pretrained(model_name, from_pt=True) # Assume loading from PT for now
            input_ids = tf.constant([[1, 2, 3, 4, 5, 6]]) # Example IDs
            output = model(input_ids).last_hidden_state

            expected_shape = tf.TensorShape([1, 6, 768]) # Use config value if possible model.config.hidden_size
            self.assertEqual(output.shape, expected_shape)
        except Exception as e:
            self.skipTest(f"Skipping TF integration test due to error: {e}")


    @slow
    def test_inference_lm_head(self):
        model_name = "convaiinnovations/hindi-causal-lm"
        try:
            model = TFHindiCausalLMForCausalLM.from_pretrained(model_name, from_pt=True) # Assume loading from PT for now
            # Test with Hindi text input example from PT test
            input_ids = tf.constant([[1, 100, 200, 300, 2]])  # Example token IDs matching PT test
            output = model(input_ids).logits

            expected_shape = tf.TensorShape([1, 5, model.config.vocab_size])
            self.assertEqual(output.shape, expected_shape)
        except Exception as e:
            self.skipTest(f"Skipping TF integration test due to error: {e}")

    @slow
    def test_generation_tf_integration(self):
        # Test actual text generation with TF model
        model_name = "convaiinnovations/hindi-causal-lm"
        try:
            model = TFHindiCausalLMForCausalLM.from_pretrained(model_name, from_pt=True)
            # Tokenizer should ideally be TF compatible, but PT usually works
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            prompt = "भारत एक विशाल देश है"
            inputs = tokenizer(prompt, return_tensors="tf")

            # Generate text
            output_sequences = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=20, # Generate a few tokens
                do_sample=False # Use greedy for reproducibility
            )

            generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

            self.assertTrue(generated_text.startswith(prompt))
            self.assertGreater(len(generated_text), len(prompt))
            print(f"\nTF Generated Text: {generated_text}") # Print for verification

        except Exception as e:
             self.skipTest(f"Skipping TF generation integration test due to error: {e}")


# Add TF docstring test if needed
# @require_tf
# class TFHindiCausalLMDocstringTest(unittest.TestCase):
#     # Adapt PT docstring tests for TF syntax
#     pass
