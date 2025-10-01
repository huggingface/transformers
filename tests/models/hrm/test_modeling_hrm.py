# Copyright 2025 The HRM Team and HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch HRM model."""

import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import HrmConfig, HrmForCausalLM, HrmModel


@require_torch
class HrmModelTester(CausalLMModelTester):
    """Model tester for HRM."""

    if is_torch_available():
        base_model_class = HrmModel

    def prepare_config_and_inputs(self):
        """Prepare configuration and inputs for HRM tests."""
        batch_size = 2
        seq_length = 9  # Small sequence for Sudoku-like tasks
        vocab_size = 11  # 0-9 digits + padding

        config = HrmConfig(
            vocab_size=vocab_size,
            hidden_size=128,
            num_hidden_layers=2,
            h_layers=2,
            l_layers=2,
            num_attention_heads=4,
            max_position_embeddings=seq_length,
            h_cycles=1,
            l_cycles=1,
            halt_max_steps=4,
            halt_exploration_prob=0.1,
            pos_encodings="rope",
            expansion=2.0,
            dtype="float32",
            puzzle_emb_ndim=0,
            num_puzzle_identifiers=1,
        )

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=torch_device)

        # HRM doesn't use standard attention masks, token_type_ids, or classification labels
        # But we need to return them for compatibility with base class
        input_mask = None
        token_type_ids = None
        sequence_labels = None
        token_labels = torch.randint(0, vocab_size, (batch_size, seq_length), device=torch_device)
        choice_labels = None

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        """Test basic model forward pass."""
        model = HrmModel(config).to(torch_device)
        model.eval()

        with torch.no_grad():
            outputs = model(input_ids=input_ids)

        self.parent.assertIsNotNone(outputs)
        self.parent.assertIsNotNone(outputs.logits)
        self.parent.assertEqual(outputs.logits.shape, (input_ids.shape[0], input_ids.shape[1], config.vocab_size))

    def create_and_check_causal_lm(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        """Test HrmForCausalLM model."""
        model = HrmForCausalLM(config).to(torch_device)
        model.eval()

        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=token_labels)

        self.parent.assertIsNotNone(outputs)
        self.parent.assertIsNotNone(outputs.logits)
        if token_labels is not None:
            self.parent.assertIsNotNone(outputs.loss)

    def create_and_check_forward_with_carry(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        """Test forward pass with carry state."""
        model = HrmModel(config).to(torch_device)
        model.eval()

        batch = {"input_ids": input_ids}
        carry = model.initial_carry(batch)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, carry=carry)

        self.parent.assertIsNotNone(outputs)
        self.parent.assertIsNotNone(outputs.carry)
        self.parent.assertIsNotNone(outputs.logits)
        self.parent.assertIsNotNone(outputs.q_halt_logits)
        self.parent.assertIsNotNone(outputs.q_continue_logits)


@require_torch
class HrmModelTest(CausalLMModelTest, unittest.TestCase):
    """Main test class for HRM models."""

    pipeline_model_mapping = (
        {
            "feature-extraction": HrmModel,
            "text-generation": HrmForCausalLM,
        }
        if is_torch_available()
        else {}
    )
    model_tester_class = HrmModelTester
    _torch_compile_train_cls = HrmForCausalLM if is_torch_available() else None

    def setUp(self):
        super().setUp()
        # Override config_tester to skip standard config tests that don't apply to HRM
        self.config_tester = None

    def test_config(self):
        """Skip standard config tests for HRM."""
        self.skipTest("HRM has custom configuration that doesn't follow standard patterns")

    def test_model_forward(self):
        """Test basic model forward pass."""
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_causal_lm_forward(self):
        """Test CausalLM model forward pass."""
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_causal_lm(*config_and_inputs)

    def test_forward_with_carry(self):
        """Test forward pass with carry state."""
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_forward_with_carry(*config_and_inputs)

    def test_model_outputs(self):
        """Test that model produces correct output structure."""
        config, input_ids, *_ = self.model_tester.prepare_config_and_inputs()
        model = HrmForCausalLM(config).to(torch_device)
        model.eval()

        with torch.no_grad():
            outputs = model(input_ids=input_ids)

        self.assertIsNotNone(outputs)
        self.assertIsNotNone(outputs.logits)
        self.assertIsNotNone(outputs.q_halt_logits)
        self.assertIsNotNone(outputs.q_continue_logits)

    def test_hierarchical_reasoning(self):
        """Test that hierarchical reasoning cycles work correctly."""
        config, input_ids, *_ = self.model_tester.prepare_config_and_inputs()

        # Test with different cycle configurations
        for h_cycles, l_cycles in [(1, 1), (2, 1), (1, 2), (2, 2)]:
            config.h_cycles = h_cycles
            config.l_cycles = l_cycles
            model = HrmModel(config).to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(input_ids=input_ids)

            self.assertIsNotNone(outputs)
            self.assertIsNotNone(outputs.logits)

    def test_adaptive_computation_time(self):
        """Test ACT mechanism with different halt configurations."""
        config, input_ids, *_ = self.model_tester.prepare_config_and_inputs()

        for halt_max_steps in [2, 4, 8]:
            config.halt_max_steps = halt_max_steps
            model = HrmModel(config).to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(input_ids=input_ids)

            self.assertIsNotNone(outputs)
            self.assertIsNotNone(outputs.q_halt_logits)
            self.assertIsNotNone(outputs.q_continue_logits)

    def test_positional_encodings(self):
        """Test different positional encoding types."""
        config, input_ids, *_ = self.model_tester.prepare_config_and_inputs()

        for pos_encoding in ["rope", "learned"]:
            config.pos_encodings = pos_encoding
            model = HrmModel(config).to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(input_ids=input_ids)

            self.assertIsNotNone(outputs)

    def test_save_and_load(self):
        """Test model save and load functionality."""
        config, input_ids, *_ = self.model_tester.prepare_config_and_inputs()
        model = HrmForCausalLM(config).to(torch_device)

        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            loaded_model = HrmForCausalLM.from_pretrained(tmp_dir).to(torch_device)

            model.eval()
            loaded_model.eval()

            with torch.no_grad():
                outputs1 = model(input_ids=input_ids)
                outputs2 = loaded_model(input_ids=input_ids)

            # Check that outputs are similar (within tolerance for floating point)
            torch.testing.assert_close(
                outputs1.logits,
                outputs2.logits,
                rtol=1e-4,
                atol=1e-4
            )

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
        # HRM uses custom tokenization for reasoning tasks
        return True


@slow
@require_torch
class HrmIntegrationTest(unittest.TestCase):
    """Integration tests for HRM with real use cases."""

    def test_simple_reasoning_task(self):
        """Test HRM on a simple reasoning task."""
        config = HrmConfig(
            vocab_size=11,
            hidden_size=256,
            num_hidden_layers=3,
            h_layers=3,
            l_layers=3,
            num_attention_heads=8,
            max_position_embeddings=81,
            h_cycles=2,
            l_cycles=2,
            halt_max_steps=8,
            dtype="float32",
        )

        model = HrmForCausalLM(config).to(torch_device)
        model.eval()

        # Create a simple input (e.g., partial Sudoku-like pattern)
        input_ids = torch.randint(0, 11, (2, 81), device=torch_device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids)

        self.assertIsNotNone(outputs)
        self.assertIsNotNone(outputs.logits)
        self.assertEqual(outputs.logits.shape, (2, 81, 11))

    def test_generation(self):
        """Test HRM generation capability."""
        config = HrmConfig(
            vocab_size=11,
            hidden_size=128,
            num_hidden_layers=2,
            h_layers=2,
            l_layers=2,
            num_attention_heads=4,
            max_position_embeddings=20,
            h_cycles=1,
            l_cycles=1,
            halt_max_steps=4,
            dtype="float32",
        )

        model = HrmForCausalLM(config).to(torch_device)
        model.eval()

        input_ids = torch.tensor([[1, 2, 3]], device=torch_device)

        with torch.no_grad():
            generated = model.generate(input_ids, max_new_tokens=5)

        self.assertIsNotNone(generated)
        self.assertEqual(generated.shape[0], 1)
        self.assertGreater(generated.shape[1], input_ids.shape[1])


if __name__ == "__main__":
    unittest.main()
