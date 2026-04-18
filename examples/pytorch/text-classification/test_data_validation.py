#!/usr/bin/env python
"""
Test script for data validation in text classification examples.
Tests various edge cases including:
- Empty text fields
- Missing text fields
- Invalid label values
- Column name mismatches
- Normal data flow
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


# Add parent directory to path to import the validation functions
sys.path.insert(0, str(Path(__file__).parent))

from run_classification import (
    validate_dataset_columns as validate_dataset_columns_cls,
)
from run_classification import (
    validate_labels_classification,
    validate_text_samples_classification,
)
from run_glue import (
    validate_dataset_columns as validate_dataset_columns_glue,
)
from run_glue import (
    validate_labels as validate_labels_glue,
)
from run_glue import (
    validate_text_samples as validate_text_samples_glue,
)
from run_glue_no_trainer import (
    validate_dataset_columns,
    validate_labels,
    validate_text_samples,
)


class TestDataValidation(unittest.TestCase):
    """Test data validation functions for text classification."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_csv(self, filename, data_lines, headers=None):
        """Helper to create test CSV files."""
        filepath = os.path.join(self.temp_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            if headers:
                f.write(",".join(headers) + "\n")
            f.writelines(line + "\n" for line in data_lines)
        return filepath

    def create_test_json(self, filename, data):
        """Helper to create test JSON files."""
        filepath = os.path.join(self.temp_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return filepath

    # ==================== Tests for run_glue_no_trainer.py ====================

    def test_validate_text_samples_with_empty_text(self):
        """Test detection of empty text fields."""
        examples = {"sentence1": ["This is valid", "", "Another valid"], "label": [0, 1, 0]}
        _, issues = validate_text_samples(examples, "sentence1", None)
        self.assertTrue(any("empty" in issue.lower() for issue in issues))
        self.assertTrue(any("Sample 1" in issue for issue in issues))

    def test_validate_text_samples_with_none_text(self):
        """Test detection of None text fields."""
        examples = {"sentence1": ["This is valid", None, "Another valid"], "label": [0, 1, 0]}
        _, issues = validate_text_samples(examples, "sentence1", None)
        self.assertTrue(any("None" in issue for issue in issues))
        self.assertTrue(any("Sample 1" in issue for issue in issues))

    def test_validate_text_samples_with_wrong_type(self):
        """Test detection of wrong type in text fields."""
        examples = {"sentence1": ["This is valid", 123, "Another valid"], "label": [0, 1, 0]}
        _, issues = validate_text_samples(examples, "sentence1", None)
        self.assertTrue(any("int" in issue for issue in issues))
        self.assertTrue(any("expected string" in issue for issue in issues))

    def test_validate_text_samples_with_whitespace_only(self):
        """Test detection of whitespace-only text fields."""
        examples = {"sentence1": ["This is valid", "   ", "Another valid"], "label": [0, 1, 0]}
        _, issues = validate_text_samples(examples, "sentence1", None)
        self.assertTrue(any("whitespace" in issue.lower() for issue in issues))

    def test_validate_text_samples_pair_valid(self):
        """Test validation with valid sentence pairs."""
        examples = {
            "sentence1": ["First sentence", "Another first"],
            "sentence2": ["Second sentence", "Another second"],
            "label": [0, 1],
        }
        _, issues = validate_text_samples(examples, "sentence1", "sentence2")
        self.assertEqual(len(issues), 0)

    def test_validate_text_samples_pair_with_none(self):
        """Test validation with None in second sentence."""
        examples = {
            "sentence1": ["First sentence", "Another first"],
            "sentence2": ["Second sentence", None],
            "label": [0, 1],
        }
        _, issues = validate_text_samples(examples, "sentence1", "sentence2")
        self.assertTrue(any("sentence2" in issue and "None" in issue for issue in issues))

    def test_validate_labels_with_none(self):
        """Test detection of None labels."""
        examples = {"label": [0, None, 1]}
        issues = validate_labels(examples, is_regression=False)
        self.assertTrue(any("None" in issue for issue in issues))

    def test_validate_labels_classification_wrong_type(self):
        """Test detection of wrong label type in classification."""
        examples = {"label": [0, [1, 2], 1]}  # List is unhashable
        issues = validate_labels(examples, is_regression=False)
        self.assertTrue(any("unhashable" in issue.lower() for issue in issues))

    def test_validate_labels_regression_wrong_type(self):
        """Test detection of wrong label type in regression."""
        examples = {"label": [1.0, "invalid", 2.0]}
        issues = validate_labels(examples, is_regression=True)
        self.assertTrue(any("expected numeric" in issue.lower() for issue in issues))

    def test_validate_labels_valid_classification(self):
        """Test validation with valid classification labels."""
        examples = {"label": [0, 1, 0, 1]}
        issues = validate_labels(examples, is_regression=False)
        self.assertEqual(len(issues), 0)

    def test_validate_labels_valid_regression(self):
        """Test validation with valid regression labels."""
        examples = {"label": [1.5, 2.3, 0.8]}
        issues = validate_labels(examples, is_regression=True)
        self.assertEqual(len(issues), 0)

    # ==================== Tests for run_classification.py ====================

    def test_validate_text_samples_classification_empty(self):
        """Test detection of empty text in classification script."""
        examples = {"sentence": ["Valid text", "", "Another valid"], "label": [0, 1, 0]}
        issues = validate_text_samples_classification(examples, "sentence")
        self.assertTrue(any("empty" in issue.lower() for issue in issues))

    def test_validate_text_samples_classification_none(self):
        """Test detection of None text in classification script."""
        examples = {"sentence": ["Valid text", None, "Another valid"], "label": [0, 1, 0]}
        issues = validate_text_samples_classification(examples, "sentence")
        self.assertTrue(any("None" in issue for issue in issues))

    def test_validate_labels_classification_multi_label_wrong_type(self):
        """Test detection of wrong type in multi-label classification."""
        examples = {"label": [[1, 0], "not_a_list", [0, 1]]}
        issues = validate_labels_classification(examples, is_regression=False, is_multi_label=True)
        self.assertTrue(any("should be a list" in issue.lower() for issue in issues))

    def test_validate_labels_classification_valid_multi_label(self):
        """Test validation with valid multi-label data."""
        examples = {"label": [[1, 0], [0, 1], [1, 1]]}
        issues = validate_labels_classification(examples, is_regression=False, is_multi_label=True)
        self.assertEqual(len(issues), 0)

    # ==================== Tests for run_glue.py ====================

    def test_validate_text_samples_glue_empty(self):
        """Test detection of empty text in run_glue script."""
        examples = {"sentence1": ["Valid", "", "Another"], "label": [0, 1, 0]}
        _, issues = validate_text_samples_glue(examples, "sentence1", None)
        self.assertTrue(any("empty" in issue.lower() for issue in issues))

    def test_validate_labels_glue_none(self):
        """Test detection of None labels in run_glue script."""
        examples = {"label": [0, None, 1]}
        issues = validate_labels_glue(examples, is_regression=False)
        self.assertTrue(any("None" in issue for issue in issues))

    # ==================== Integration Tests ====================

    def test_csv_with_missing_column(self):
        """Test that missing column raises appropriate error."""
        from datasets import load_dataset

        # Create a CSV without the expected 'sentence1' column
        csv_lines = ["text,label", "This is a sample,0", "Another sample,1"]
        csv_path = self.create_test_csv("test_missing_col.csv", csv_lines[1:], headers=csv_lines[0].split(","))

        raw_datasets = load_dataset("csv", data_files={"train": csv_path})

        # Should raise ValueError when looking for 'sentence1' which doesn't exist
        with self.assertRaises(ValueError) as context:
            validate_dataset_columns(raw_datasets, "sentence1", None, split="train")

        self.assertIn("sentence1", str(context.exception))
        self.assertIn("not found", str(context.exception).lower())

    def test_csv_with_correct_columns(self):
        """Test that correct columns pass validation."""
        from datasets import load_dataset

        csv_lines = ["sentence1,label", "This is a sample,0", "Another sample,1"]
        csv_path = self.create_test_csv("test_correct.csv", csv_lines[1:], headers=csv_lines[0].split(","))

        raw_datasets = load_dataset("csv", data_files={"train": csv_path})

        # Should not raise for correct columns
        try:
            validate_dataset_columns(raw_datasets, "sentence1", None, split="train")
        except ValueError as e:
            self.fail(f"Should not raise for valid columns: {e}")

    def test_json_with_missing_label(self):
        """Test that missing label column raises appropriate error."""
        from datasets import load_dataset

        data = [{"sentence1": "This is a sample"}, {"sentence1": "Another sample"}]
        json_path = self.create_test_json("test_missing_label.json", data)

        raw_datasets = load_dataset("json", data_files={"train": json_path})

        # Should raise ValueError when looking for 'label' which doesn't exist
        with self.assertRaises(ValueError) as context:
            validate_dataset_columns(raw_datasets, "sentence1", None, split="train")

        self.assertIn("label", str(context.exception))

    def test_json_with_pair_columns(self):
        """Test validation with sentence pair columns."""
        from datasets import load_dataset

        data = [
            {"sentence1": "First", "sentence2": "Second", "label": 0},
            {"sentence1": "Third", "sentence2": "Fourth", "label": 1},
        ]
        json_path = self.create_test_json("test_pair.json", data)

        raw_datasets = load_dataset("json", data_files={"train": json_path})

        # Should not raise any error
        try:
            validate_dataset_columns(raw_datasets, "sentence1", "sentence2", split="train")
        except ValueError as e:
            self.fail(f"validate_dataset_columns raised ValueError unexpectedly: {e}")

    def test_classification_csv_validation(self):
        """Test classification script validation with CSV."""
        from datasets import load_dataset

        csv_lines = ["sentence,label", "This is positive,1", "This is negative,0"]
        csv_path = self.create_test_csv("test_cls.csv", csv_lines[1:], headers=csv_lines[0].split(","))

        raw_datasets = load_dataset("csv", data_files={"train": csv_path})

        # Should not raise any error
        try:
            validate_dataset_columns_cls(raw_datasets, "sentence", "label", split="train")
        except ValueError as e:
            self.fail(f"validate_dataset_columns raised ValueError unexpectedly: {e}")

    def test_classification_missing_text_column(self):
        """Test classification script with missing text column."""
        from datasets import load_dataset

        csv_lines = ["content,label", "This is positive,1", "This is negative,0"]
        csv_path = self.create_test_csv("test_cls_missing.csv", csv_lines[1:], headers=csv_lines[0].split(","))

        raw_datasets = load_dataset("csv", data_files={"train": csv_path})

        # Should raise ValueError when looking for 'sentence' which doesn't exist
        with self.assertRaises(ValueError) as context:
            validate_dataset_columns_cls(raw_datasets, "sentence", "label", split="train")

        self.assertIn("sentence", str(context.exception))
        self.assertIn("not found", str(context.exception).lower())

    def test_glue_csv_validation(self):
        """Test run_glue script validation with CSV."""
        from datasets import load_dataset

        csv_lines = ["sentence1,label", "This is a sample,0", "Another sample,1"]
        csv_path = self.create_test_csv("test_glue.csv", csv_lines[1:], headers=csv_lines[0].split(","))

        raw_datasets = load_dataset("csv", data_files={"train": csv_path})

        # Should not raise any error
        try:
            validate_dataset_columns_glue(raw_datasets, "sentence1", None, split="train")
        except ValueError as e:
            self.fail(f"validate_dataset_columns raised ValueError unexpectedly: {e}")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def test_empty_examples(self):
        """Test validation with empty examples dict."""
        examples = {"sentence1": [], "label": []}
        _, issues = validate_text_samples(examples, "sentence1", None)
        self.assertEqual(len(issues), 0)  # No samples to validate

    def test_all_invalid_samples(self):
        """Test when all samples are invalid."""
        examples = {"sentence1": [None, "", "   "], "label": [0, 1, 0]}
        _, issues = validate_text_samples(examples, "sentence1", None)
        self.assertEqual(len(issues), 3)  # All three samples have issues

    def test_mixed_valid_invalid(self):
        """Test with mix of valid and invalid samples."""
        examples = {"sentence1": ["Valid", None, "Also valid", "", "Good"], "label": [0, 1, 0, 1, 0]}
        _, issues = validate_text_samples(examples, "sentence1", None)
        self.assertEqual(len(issues), 2)  # Two invalid samples

    def test_unicode_text(self):
        """Test validation with unicode text."""
        examples = {"sentence1": ["这是一个测试", "🎉 Emoji test", "Normal text"], "label": [0, 1, 0]}
        _, issues = validate_text_samples(examples, "sentence1", None)
        self.assertEqual(len(issues), 0)  # All valid

    def test_very_long_text(self):
        """Test validation with very long text."""
        examples = {"sentence1": ["A" * 10000, "Short", "B" * 5000], "label": [0, 1, 0]}
        _, issues = validate_text_samples(examples, "sentence1", None)
        self.assertEqual(len(issues), 0)  # Long text is still valid


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)
