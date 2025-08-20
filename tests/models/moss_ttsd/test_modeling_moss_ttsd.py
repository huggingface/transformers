# coding=utf-8
# Copyright 2025 OpenMOSS and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch MOSS-TTSD model."""

import copy
import unittest
from unittest.mock import MagicMock, patch

from transformers import MossTTSDConfig, MossTTSDForCausalLM, MossTTSDModel
from transformers.models.auto import get_values
from transformers.models.auto.modeling_auto import MODEL_FOR_BACKBONE_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers.testing_utils import is_torch_available, require_torch, slow, torch_device
from transformers.utils import is_torchaudio_available

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask


if is_torch_available():
    import torch

if is_torchaudio_available():
    pass


# Pretrained model list (empty as no models are publicly released yet)
MOSS_TTSD_PRETRAINED_MODEL_ARCHIVE_LIST = []


class MossTTSDModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        use_labels=True,
        vocab_size=1124,  # Need total vocab including speech tokens
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
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
        scope=None,
        channels=8,  # Updated to match processor default
        speech_vocab_size=1025,  # Updated to more realistic value
        speech_token_range=(99, 1123),  # Speech tokens within vocab range
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
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope
        self.channels = channels
        self.speech_vocab_size = speech_vocab_size
        self.speech_token_range = speech_token_range

    def prepare_config_and_inputs(self):
        # Generate input_ids with proper vocab ranges for each channel
        # Channel 0: full vocab_size, Channels 1-7: speech_vocab_size
        input_ids = torch.zeros([self.batch_size, self.seq_length, self.channels], dtype=torch.long, device=torch_device)
        input_ids[:, :, 0] = ids_tensor([self.batch_size, self.seq_length], self.vocab_size).to(torch_device)
        for i in range(1, self.channels):
            input_ids[:, :, i] = ids_tensor([self.batch_size, self.seq_length], self.speech_vocab_size).to(torch_device)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            # Labels also need proper ranges
            token_labels = torch.zeros([self.batch_size, self.seq_length, self.channels], dtype=torch.long, device=torch_device)
            token_labels[:, :, 0] = ids_tensor([self.batch_size, self.seq_length], self.vocab_size).to(torch_device)
            for i in range(1, self.channels):
                token_labels[:, :, i] = ids_tensor([self.batch_size, self.seq_length], self.speech_vocab_size).to(torch_device)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        return MossTTSDConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            attention_dropout=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            channels=self.channels,
            speech_vocab_size=self.speech_vocab_size,
            speech_token_range=self.speech_token_range,
            num_key_value_heads=self.num_attention_heads,
            head_dim=self.hidden_size // self.num_attention_heads,
        )

    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = MossTTSDModel(config=config)
        model.to(torch_device)
        model.eval()
        # MOSS-TTSD doesn't use token_type_ids
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        # MOSS-TTSD doesn't have pooler_output

    def create_and_check_for_causal_lm(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = MossTTSDForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        # MOSS-TTSD doesn't use token_type_ids
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertIsNotNone(result.loss)
        # MOSS-TTSD uses only vocab_size for logits, not vocab_size + speech_vocab_size
        self.parent.assertEqual(
            result.logits.shape, (self.batch_size, self.seq_length, config.vocab_size)
        )

    def create_and_check_decoder_model_past_large_inputs(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = MossTTSDForCausalLM(config=config)
        model.to(torch_device)
        model.eval()

        outputs = model(
            input_ids,
            attention_mask=input_mask,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = torch.zeros((self.batch_size, 3, self.channels), dtype=torch.long, device=torch_device)
        next_tokens[:, :, 0] = ids_tensor((self.batch_size, 3), config.vocab_size).to(torch_device)
        for i in range(1, self.channels):
            next_tokens[:, :, i] = ids_tensor((self.batch_size, 3), config.speech_vocab_size).to(torch_device)
        next_mask = ids_tensor((self.batch_size, 3), vocab_size=2).to(torch_device)

        # append to next input_ids and
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-2)
        next_attention_mask = torch.cat([input_mask, next_mask], dim=-1)

        output_from_no_past = model(
            next_input_ids,
            attention_mask=next_attention_mask,
        )["logits"]
        output_from_past = model(
            next_tokens,
            attention_mask=next_attention_mask,
            past_key_values=past_key_values,
        )["logits"]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

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
        # MOSS-TTSD doesn't use token_type_ids
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class MossTTSDModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (MossTTSDModel, MossTTSDForCausalLM) if is_torch_available() else ()
    all_generative_model_classes = (MossTTSDForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": MossTTSDModel,
            "text-generation": MossTTSDForCausalLM,
        }
        if is_torch_available()
        else {}
    )
    fx_compatible = False
    test_pruning = False
    test_missing_keys = False
    test_model_parallel = False
    test_head_masking = False
    test_resize_embeddings = False
    test_resize_tokens_embeddings = False  # MOSS-TTSD uses special token handling
    test_torchscript = False  # Multi-channel input may not be fully compatible

    def setUp(self):
        self.model_tester = MossTTSDModelTester(self)
        self.config_tester = ConfigTester(self, config_class=MossTTSDConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_causal_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_causal_lm(*config_and_inputs)

    def test_decoder_model_past_with_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

    def test_model_attention_outputs(self):
        """Test that model returns attention outputs when requested."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs_dict)
            attentions = outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, self.model_tester.seq_length, self.model_tester.seq_length],
            )
            out_len = len(outputs)

            # Check attention weights are coherent
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs_dict)
            added_hidden_states = 2
            self.assertEqual(out_len + added_hidden_states, len(outputs))

    def test_multi_channel_input_shapes(self):
        """Test that model handles multi-channel input correctly."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            # Test with different channel configurations
            batch_size, seq_length = 2, 5
            channels = config.channels

            # Test basic multi-channel input
            input_ids = torch.zeros((batch_size, seq_length, channels), dtype=torch.long)
            input_ids[:, :, 0] = torch.randint(0, config.vocab_size, (batch_size, seq_length))
            for i in range(1, channels):
                input_ids[:, :, i] = torch.randint(0, config.speech_vocab_size, (batch_size, seq_length))
            attention_mask = torch.ones(batch_size, seq_length)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            if hasattr(outputs, "last_hidden_state"):
                self.assertEqual(outputs.last_hidden_state.shape[:2], (batch_size, seq_length))

    def test_speech_token_handling(self):
        """Test that speech tokens are handled correctly."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        model = MossTTSDForCausalLM(config)
        model.to(torch_device)
        model.eval()

        # Create input with speech tokens
        batch_size, seq_length = 2, 10
        input_ids = torch.zeros((batch_size, seq_length, config.channels), dtype=torch.long)
        input_ids[:, :, 0] = torch.randint(0, config.vocab_size, (batch_size, seq_length))
        for i in range(1, config.channels):
            input_ids[:, :, i] = torch.randint(0, config.speech_vocab_size, (batch_size, seq_length))

        # Add some speech tokens
        if config.speech_token_range:
            speech_start, speech_end = config.speech_token_range
            # Insert speech tokens within valid vocab range in first channel
            # Make sure we don't exceed vocab_size
            valid_end = min(speech_end, config.vocab_size)
            if valid_end > speech_start:
                input_ids[:, 2:4, 0] = torch.randint(speech_start, valid_end, (batch_size, 2))
            else:
                # If speech token range is outside vocab, use regular tokens
                input_ids[:, 2:4, 0] = torch.randint(0, config.vocab_size, (batch_size, 2))

        attention_mask = torch.ones(batch_size, seq_length)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Check that logits have correct vocabulary size
        self.assertEqual(outputs.logits.shape[-1], config.vocab_size)

    def _generate_valid_input_ids(self, batch_size, seq_length, config):
        """Generate valid input_ids for the given config with proper vocab ranges."""
        input_ids = torch.zeros((batch_size, seq_length, config.channels), dtype=torch.long)
        input_ids[:, :, 0] = torch.randint(0, config.vocab_size, (batch_size, seq_length))
        for i in range(1, config.channels):
            input_ids[:, :, i] = torch.randint(0, config.speech_vocab_size, (batch_size, seq_length))
        return input_ids
    
    def test_generation_with_processor_integration(self):
        """Test generation with integrated processor (mock)."""
        config = self.model_tester.get_config()
        model = MossTTSDForCausalLM(config)
        model.to(torch_device)
        model.eval()

        # Create mock processor
        with patch("transformers.MossTTSDProcessor") as MockProcessor:
            mock_processor = MagicMock()
            mock_input_data = {
                "input_ids": self._generate_valid_input_ids(1, 5, config),
                "attention_mask": torch.ones(1, 5),
            }
            mock_processor.return_value = mock_input_data
            MockProcessor.from_pretrained.return_value = mock_processor

            # Test that model can generate with processor-formatted inputs
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=mock_input_data["input_ids"],
                    attention_mask=mock_input_data["attention_mask"],
                    max_new_tokens=3,
                    do_sample=False,
                )

            # Check output shape
            self.assertEqual(outputs.shape[0], 1)  # batch size
            self.assertEqual(outputs.shape[2], config.channels)  # channels preserved

    @slow
    def test_model_from_pretrained(self):
        for model_name in MOSS_TTSD_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = MossTTSDModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


@require_torch
class MossTTSDModelIntegrationTest(unittest.TestCase):
    """Integration tests for MOSS-TTSD model with more realistic scenarios."""

    def setUp(self):
        # Create a small but realistic config
        self.config = MossTTSDConfig(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=256,
            channels=8,
            speech_vocab_size=1025,
            speech_token_range=(50000, 51023),
            max_position_embeddings=128,
        )

    def test_forward_pass_consistency(self):
        """Test that forward pass is consistent across multiple runs."""
        model = MossTTSDModel(self.config)
        model.to(torch_device)
        model.eval()

        # Create deterministic input
        torch.manual_seed(42)
        batch_size, seq_length = 2, 10
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_length, self.config.channels))
        attention_mask = torch.ones(batch_size, seq_length)

        with torch.no_grad():
            output1 = model(input_ids=input_ids, attention_mask=attention_mask)
            output2 = model(input_ids=input_ids, attention_mask=attention_mask)

        # Outputs should be identical for same input
        self.assertTrue(torch.allclose(output1.last_hidden_state, output2.last_hidden_state, atol=1e-6))

    def test_causal_lm_loss_computation(self):
        """Test that causal LM loss is computed correctly."""
        model = MossTTSDForCausalLM(self.config)
        model.to(torch_device)
        model.train()  # Training mode for loss computation

        batch_size, seq_length = 2, 8
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_length, self.config.channels))
        attention_mask = torch.ones(batch_size, seq_length)

        # Create labels (shift by one position for causal modeling)
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]  # Shift left
        labels[:, -1] = -100  # Ignore last position

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # Loss should be computed
        self.assertIsNotNone(outputs.loss)
        self.assertTrue(outputs.loss.item() > 0)  # Loss should be positive

        # Logits should have correct shape
        expected_vocab_size = self.config.vocab_size + self.config.speech_vocab_size
        self.assertEqual(outputs.logits.shape, (batch_size, seq_length, expected_vocab_size))

    def test_attention_mask_behavior(self):
        """Test that attention mask correctly prevents attention to padded positions."""
        model = MossTTSDModel(self.config)
        model.to(torch_device)
        model.eval()

        batch_size, seq_length = 2, 6
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_length, self.config.channels))

        # Create attention mask with different lengths
        attention_mask = torch.ones(batch_size, seq_length)
        attention_mask[0, 4:] = 0  # First sequence has length 4
        attention_mask[1, 5:] = 0  # Second sequence has length 5

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)

        # Check that attention weights are zero for masked positions
        attentions = outputs.attentions[0]  # First layer attention

        # For first batch item, positions 4 and 5 should have zero attention weights
        self.assertTrue(torch.all(attentions[0, :, :, 4:] == 0))
        # For second batch item, position 5 should have zero attention weights
        self.assertTrue(torch.all(attentions[1, :, :, 5:] == 0))

    def test_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        model = MossTTSDForCausalLM(self.config)
        model.to(torch_device)
        model.train()

        batch_size, seq_length = 1, 4
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_length, self.config.channels))
        attention_mask = torch.ones(batch_size, seq_length)
        labels = input_ids.clone()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Compute gradients
        loss.backward()

        # Check that gradients exist and are non-zero for key parameters
        embedding_grads = model.moss_ttsd.embeddings.word_embeddings.weight.grad
        self.assertIsNotNone(embedding_grads)
        self.assertTrue(torch.any(embedding_grads != 0))

        # Check transformer layer gradients
        first_layer = model.moss_ttsd.encoder.layer[0]
        self.assertIsNotNone(first_layer.attention.self.query.weight.grad)
        self.assertTrue(torch.any(first_layer.attention.self.query.weight.grad != 0))

    @require_torch
    def test_model_outputs_equivalence(self):
        """Test that model outputs are equivalent when using different input formats."""
        model = MossTTSDModel(self.config)
        model.to(torch_device)
        model.eval()

        batch_size, seq_length = 1, 5
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_length, self.config.channels))
        attention_mask = torch.ones(batch_size, seq_length)

        with torch.no_grad():
            # Test with explicit attention mask
            outputs1 = model(input_ids=input_ids, attention_mask=attention_mask)

            # Test without attention mask (should default to all ones)
            outputs2 = model(input_ids=input_ids)

            # Outputs should be identical since attention mask is all ones
            self.assertTrue(torch.allclose(outputs1.last_hidden_state, outputs2.last_hidden_state, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
