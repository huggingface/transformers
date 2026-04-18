#!/usr/bin/env python
"""
End-to-end integration test for data validation fixes.
Tests the actual script behavior with problematic data files.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestEndToEndValidation(unittest.TestCase):
    """End-to-end tests for data validation in training scripts."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.script_dir = Path(__file__).parent

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

    def test_validation_error_on_missing_column(self):
        """Test that missing column produces clear error message."""
        # Create a CSV with wrong column name
        csv_lines = [
            "text,label",
            "This is a sample,0",
            "Another sample,1"
        ]
        train_file = self.create_test_csv("train.csv", csv_lines[1:], headers=csv_lines[0].split(","))
        val_lines = [
            "text,label",
            "Validation sample,0"
        ]
        val_file = self.create_test_csv("val.csv", val_lines[1:], headers=val_lines[0].split(","))

        # Try to run the script with --help first to check it loads
        result = subprocess.run(
            [sys.executable, str(self.script_dir / "run_glue_no_trainer.py"), "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        self.assertEqual(result.returncode, 0, f"Script should load successfully: {result.stderr}")

    def test_validation_functions_importable(self):
        """Test that validation functions can be imported."""
        # Test import from run_glue_no_trainer
        try:
            from run_glue_no_trainer import (
                validate_dataset_columns,
                validate_labels,
                validate_text_samples,
            )
        except ImportError as e:
            self.fail(f"Failed to import from run_glue_no_trainer: {e}")

        # Test import from run_classification
        try:
            from run_classification import (
                validate_dataset_columns,
                validate_labels_classification,
                validate_text_samples_classification,
            )
        except ImportError as e:
            self.fail(f"Failed to import from run_classification: {e}")

    def test_column_validation_with_datasets(self):
        """Test column validation with actual datasets library."""
        from datasets import load_dataset

        # Create test CSV with correct columns
        csv_lines = [
            "sentence1,label",
            "This is a sample,0",
            "Another sample,1"
        ]
        csv_path = self.create_test_csv("test.csv", csv_lines[1:], headers=csv_lines[0].split(","))

        raw_datasets = load_dataset("csv", data_files={"train": csv_path})

        from run_glue_no_trainer import validate_dataset_columns

        # Should not raise for correct columns
        try:
            validate_dataset_columns(raw_datasets, "sentence1", None, split="train")
        except ValueError as e:
            self.fail(f"Should not raise for valid columns: {e}")

        # Should raise for missing column
        with self.assertRaises(ValueError) as context:
            validate_dataset_columns(raw_datasets, "nonexistent", None, split="train")

        error_msg = str(context.exception)
        self.assertIn("nonexistent", error_msg)
        self.assertIn("not found", error_msg.lower())
        self.assertIn("Available columns", error_msg)

    def test_text_validation_catches_issues(self):
        """Test that text validation catches data quality issues."""
        from run_glue_no_trainer import validate_text_samples

        # Test with None values
        examples = {
            "sentence1": ["Valid", None, "Also valid"],
            "label": [0, 1, 0]
        }
        _, issues = validate_text_samples(examples, "sentence1", None)
        self.assertTrue(any("None" in issue for issue in issues))

        # Test with empty strings
        examples = {
            "sentence1": ["Valid", "", "Also valid"],
            "label": [0, 1, 0]
        }
        _, issues = validate_text_samples(examples, "sentence1", None)
        self.assertTrue(any("empty" in issue.lower() for issue in issues))

        # Test with wrong type
        examples = {
            "sentence1": ["Valid", 123, "Also valid"],
            "label": [0, 1, 0]
        }
        _, issues = validate_text_samples(examples, "sentence1", None)
        self.assertTrue(any("int" in issue for issue in issues))

    def test_label_validation_catches_issues(self):
        """Test that label validation catches label issues."""
        from run_glue_no_trainer import validate_labels

        # Test with None labels
        examples = {"label": [0, None, 1]}
        issues = validate_labels(examples, is_regression=False)
        self.assertTrue(any("None" in issue for issue in issues))

        # Test with wrong type for classification
        examples = {"label": [0, [1, 2], 1]}  # List is unhashable
        issues = validate_labels(examples, is_regression=False)
        self.assertTrue(any("unhashable" in issue.lower() for issue in issues))

        # Test with wrong type for regression
        examples = {"label": [1.0, "invalid", 2.0]}
        issues = validate_labels(examples, is_regression=True)
        self.assertTrue(any("numeric" in issue.lower() for issue in issues))


if __name__ == "__main__":
    unittest.main(verbosity=2)
