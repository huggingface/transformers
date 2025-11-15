# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


# Add examples directory to Python path to import BasicToxicityChecker
examples_path = Path(__file__).parent.parent.parent / "examples"
if str(examples_path) not in sys.path:
    sys.path.insert(0, str(examples_path))

from safe_generation import BasicToxicityChecker  # noqa: E402

from transformers.generation.safety import SafetyResult  # noqa: E402
from transformers.testing_utils import require_torch  # noqa: E402


@require_torch
class TestBasicToxicityChecker(unittest.TestCase):
    """Test suite for BasicToxicityChecker."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_tokenizer_patcher = patch("transformers.AutoTokenizer.from_pretrained")
        self.mock_model_patcher = patch("transformers.AutoModelForSequenceClassification.from_pretrained")

        self.mock_tokenizer = self.mock_tokenizer_patcher.start()
        self.mock_model = self.mock_model_patcher.start()

        # Configure mock tokenizer
        mock_tokenizer_instance = Mock()

        # Create a mock that can be unpacked as **kwargs
        class MockTokenizerOutput(dict):
            def to(self, device):
                return self

        mock_tokenizer_instance.return_value = MockTokenizerOutput({"input_ids": Mock(), "attention_mask": Mock()})
        self.mock_tokenizer.return_value = mock_tokenizer_instance

        # Configure mock model
        self.mock_model_instance = Mock()
        self.mock_model_instance.eval.return_value = None
        self.mock_model_instance.to.return_value = None
        self.mock_model.return_value = self.mock_model_instance

    def tearDown(self):
        """Clean up test fixtures."""
        self.mock_tokenizer_patcher.stop()
        self.mock_model_patcher.stop()

    @patch("torch.cuda.is_available", return_value=False)
    def test_init_with_defaults(self, mock_cuda):
        """Test BasicToxicityChecker initialization with default parameters."""
        checker = BasicToxicityChecker()

        self.assertEqual(checker.model_name, "s-nlp/roberta_toxicity_classifier")
        self.assertEqual(checker.threshold, 0.7)
        self.assertEqual(checker.device, "cpu")
        self.assertEqual(checker.supported_categories, ["toxicity"])

    @patch("torch.cuda.is_available", return_value=True)
    def test_init_with_cuda_available(self, mock_cuda):
        """Test BasicToxicityChecker initialization when CUDA is available."""
        checker = BasicToxicityChecker()
        self.assertEqual(checker.device, "cuda")

    def test_init_with_custom_params(self):
        """Test BasicToxicityChecker initialization with custom parameters."""
        checker = BasicToxicityChecker(model_name="custom/model", threshold=0.8, device="cpu")

        self.assertEqual(checker.model_name, "custom/model")
        self.assertEqual(checker.threshold, 0.8)
        self.assertEqual(checker.device, "cpu")

    def test_init_model_loading_failure(self):
        """Test BasicToxicityChecker handles model loading failures gracefully."""
        # Make model loading fail
        self.mock_model.side_effect = Exception("Model not found")

        with self.assertRaises(RuntimeError) as context:
            BasicToxicityChecker()

        self.assertIn("Failed to load toxicity model", str(context.exception))
        self.assertIn("Model not found", str(context.exception))

    @patch("torch.no_grad")
    @patch("torch.nn.functional.softmax")
    def test_safe_text_detection(self, mock_softmax, mock_no_grad):
        """Test detection of safe (non-toxic) text."""
        import torch

        # Mock safe prediction (low toxicity score)
        mock_outputs = Mock()
        mock_outputs.logits = torch.tensor([[2.0, 0.5]])  # Non-toxic >> toxic
        self.mock_model_instance.return_value = mock_outputs

        # Mock softmax to return low toxicity probability
        mock_softmax.return_value = torch.tensor([[0.8, 0.2]])  # [non-toxic, toxic]

        checker = BasicToxicityChecker(threshold=0.7)
        result = checker.check_safety("This is a nice, positive comment")

        self.assertIsInstance(result, SafetyResult)
        self.assertTrue(result.is_safe)
        self.assertEqual(len(result.violations), 0)
        self.assertIn("toxicity_score", result.metadata)
        self.assertAlmostEqual(result.metadata["toxicity_score"], 0.2, places=5)

    @patch("torch.no_grad")
    @patch("torch.nn.functional.softmax")
    def test_toxic_text_detection(self, mock_softmax, mock_no_grad):
        """Test detection of toxic text."""
        import torch

        # Mock toxic prediction (high toxicity score)
        mock_outputs = Mock()
        mock_outputs.logits = torch.tensor([[0.2, 3.0]])  # Non-toxic << toxic
        self.mock_model_instance.return_value = mock_outputs

        # Mock softmax to return high toxicity probability
        mock_softmax.return_value = torch.tensor([[0.15, 0.85]])  # [non-toxic, toxic]

        checker = BasicToxicityChecker(threshold=0.7)
        result = checker.check_safety("This is some toxic harmful content")

        self.assertIsInstance(result, SafetyResult)
        self.assertFalse(result.is_safe)
        self.assertEqual(len(result.violations), 1)

        violation = result.violations[0]
        self.assertEqual(violation.category, "toxicity")
        self.assertAlmostEqual(violation.confidence, 0.85, places=5)
        self.assertIn("high", violation.severity)  # 0.85 should be "high" severity
        self.assertIn("85.00%", violation.description)

    def test_batch_processing(self):
        """Test batch processing of multiple texts."""
        import torch

        with patch("torch.no_grad"), patch("torch.nn.functional.softmax") as mock_softmax:
            # Mock mixed results
            mock_outputs = Mock()
            mock_outputs.logits = torch.tensor([[2.0, 0.5]])
            self.mock_model_instance.return_value = mock_outputs
            mock_softmax.return_value = torch.tensor([[0.8, 0.2]])  # Safe

            checker = BasicToxicityChecker()
            results = checker.check_safety(["Safe text", "Another safe text"])

            self.assertIsInstance(results, list)
            self.assertEqual(len(results), 2)
            self.assertTrue(all(isinstance(r, SafetyResult) for r in results))

    def test_empty_text_handling(self):
        """Test handling of empty text input."""

        checker = BasicToxicityChecker()
        result = checker.check_safety("")

        self.assertTrue(result.is_safe)
        self.assertEqual(result.confidence, 1.0)
        self.assertEqual(len(result.violations), 0)
        self.assertEqual(result.metadata["reason"], "empty_text")

    def test_whitespace_only_text_handling(self):
        """Test handling of whitespace-only text input."""

        checker = BasicToxicityChecker()
        result = checker.check_safety("   \n\t   ")

        self.assertTrue(result.is_safe)
        self.assertEqual(result.confidence, 1.0)
        self.assertEqual(len(result.violations), 0)
        self.assertEqual(result.metadata["reason"], "empty_text")

    @patch("safe_generation.checkers.logger")
    def test_long_text_truncation(self, mock_logger):
        """Test handling of very long text input."""
        import torch

        with patch("torch.no_grad"), patch("torch.nn.functional.softmax") as mock_softmax:
            mock_outputs = Mock()
            mock_outputs.logits = torch.tensor([[2.0, 0.5]])
            self.mock_model_instance.return_value = mock_outputs
            mock_softmax.return_value = torch.tensor([[0.8, 0.2]])

            checker = BasicToxicityChecker()
            long_text = "A" * 15000  # Longer than 10000 char limit
            result = checker.check_safety(long_text)

            self.assertIn("truncated", result.metadata)
            self.assertTrue(result.metadata["truncated"])
            self.assertEqual(result.metadata["original_length"], 15000)
            self.assertEqual(result.metadata["processed_length"], 10000)
            mock_logger.warning.assert_called_once()

    def test_invalid_input_type(self):
        """Test handling of invalid input types."""

        checker = BasicToxicityChecker()

        with self.assertRaises(TypeError) as context:
            checker.check_safety(123)  # Not a string or list

        self.assertIn("Expected string or list of strings", str(context.exception))

    def test_severity_classification(self):
        """Test severity classification logic."""

        checker = BasicToxicityChecker()

        # Test different severity levels
        self.assertEqual(checker._get_severity(0.96), "critical")
        self.assertEqual(checker._get_severity(0.90), "high")
        self.assertEqual(checker._get_severity(0.80), "medium")
        self.assertEqual(checker._get_severity(0.65), "low")

    def test_get_config(self):
        """Test get_config method returns correct configuration."""

        checker = BasicToxicityChecker(model_name="test/model", threshold=0.8, device="cpu")

        config = checker.get_config()
        expected_config = {
            "checker_type": "BasicToxicityChecker",
            "model_name": "test/model",
            "threshold": 0.8,
            "device": "cpu",
        }

        self.assertEqual(config, expected_config)

    @patch("torch.no_grad")
    def test_inference_error_handling(self, mock_no_grad):
        """Test handling of inference errors."""

        # Make model inference fail
        self.mock_model_instance.side_effect = RuntimeError("CUDA out of memory")

        checker = BasicToxicityChecker()

        with self.assertRaises(RuntimeError) as context:
            checker.check_safety("test text")

        self.assertIn("Toxicity detection failed", str(context.exception))
