#!/usr/bin/env python
"""
Test script to validate data validation functionality for all text classification scripts.
Tests various error scenarios: empty text, missing columns, invalid labels, etc.
Covers: run_glue.py, run_classification.py, run_glue_no_trainer.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd


def create_test_csv(data, filepath):
    """Create a test CSV file."""
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)


def create_test_json(data, filepath):
    """Create a test JSON file."""
    with open(filepath, "w") as f:
        f.writelines(json.dumps(item) + "\n" for item in data)


def test_run_glue_validation():
    """Test validation functions from run_glue.py."""
    print("\n" + "=" * 60)
    print("Testing run_glue.py validation functions")
    print("=" * 60)

    sys.path.insert(0, str(Path(__file__).parent))
    from datasets import Dataset
    from run_glue import DataValidationError, validate_dataset_for_classification

    print("\n--- Test 1: Normal valid data ---")
    data = {
        "sentence1": ["This is a test sentence.", "Another test sentence.", "Third test sentence."],
        "label": [0, 1, 0],
    }
    dataset = Dataset.from_dict(data)
    try:
        validate_dataset_for_classification(dataset, "test", "sentence1", None, is_regression=False)
        print("✓ PASS: Normal data validated successfully")
    except DataValidationError as e:
        print(f"✗ FAIL: Unexpected error: {e}")
        return False

    print("\n--- Test 2: Empty text field ---")
    data = {
        "sentence1": ["Valid text", "", "Another valid text"],
        "label": [0, 1, 0],
    }
    dataset = Dataset.from_dict(data)
    try:
        validate_dataset_for_classification(dataset, "test", "sentence1", None, is_regression=False)
        print("✗ FAIL: Should have raised DataValidationError")
        return False
    except DataValidationError as e:
        print(f"✓ PASS: Correctly caught: {e}")

    print("\n--- Test 3: None text field ---")
    data = {
        "sentence1": ["Valid text", None, "Another valid text"],
        "label": [0, 1, 0],
    }
    dataset = Dataset.from_dict(data)
    try:
        validate_dataset_for_classification(dataset, "test", "sentence1", None, is_regression=False)
        print("✗ FAIL: Should have raised DataValidationError")
        return False
    except DataValidationError as e:
        print(f"✓ PASS: Correctly caught: {e}")

    print("\n--- Test 4: Missing required column ---")
    data = {
        "wrong_column": ["Some text", "Another text"],
        "label": [0, 1],
    }
    dataset = Dataset.from_dict(data)
    try:
        validate_dataset_for_classification(dataset, "test", "sentence1", None, is_regression=False)
        print("✗ FAIL: Should have raised DataValidationError")
        return False
    except DataValidationError as e:
        print(f"✓ PASS: Correctly caught: {e}")

    print("\n--- Test 5: Missing label column ---")
    data = {
        "sentence1": ["Valid text", "Another valid text"],
    }
    dataset = Dataset.from_dict(data)
    try:
        validate_dataset_for_classification(dataset, "test", "sentence1", None, is_regression=False)
        print("✗ FAIL: Should have raised DataValidationError")
        return False
    except DataValidationError as e:
        print(f"✓ PASS: Correctly caught: {e}")

    print("\n--- Test 6: Invalid label (None) ---")
    data = {
        "sentence1": ["Valid text", "Another text", "Third text"],
        "label": [0, None, 1],
    }
    dataset = Dataset.from_dict(data)
    try:
        validate_dataset_for_classification(dataset, "test", "sentence1", None, is_regression=False)
        print("✗ FAIL: Should have raised DataValidationError")
        return False
    except DataValidationError as e:
        print(f"✓ PASS: Correctly caught: {e}")

    print("\n--- Test 7: Empty string label ---")
    data = {
        "sentence1": ["Valid text", "Another text", "Third text"],
        "label": ["positive", "", "negative"],
    }
    dataset = Dataset.from_dict(data)
    try:
        validate_dataset_for_classification(dataset, "test", "sentence1", None, is_regression=False)
        print("✗ FAIL: Should have raised DataValidationError")
        return False
    except DataValidationError as e:
        print(f"✓ PASS: Correctly caught: {e}")

    print("\n--- Test 8: Two sentence columns (normal) ---")
    data = {
        "sentence1": ["Question 1", "Question 2"],
        "sentence2": ["Answer 1", "Answer 2"],
        "label": [0, 1],
    }
    dataset = Dataset.from_dict(data)
    try:
        validate_dataset_for_classification(dataset, "test", "sentence1", "sentence2", is_regression=False)
        print("✓ PASS: Two sentence data validated")
    except DataValidationError as e:
        print(f"✗ FAIL: Unexpected error: {e}")
        return False

    print("\n--- Test 9: Regression data (normal) ---")
    data = {
        "sentence1": ["Text 1", "Text 2", "Text 3"],
        "label": [1.5, 2.3, 0.8],
    }
    dataset = Dataset.from_dict(data)
    try:
        validate_dataset_for_classification(dataset, "test", "sentence1", None, is_regression=True)
        print("✓ PASS: Regression data validated")
    except DataValidationError as e:
        print(f"✗ FAIL: Unexpected error: {e}")
        return False

    return True


def test_run_classification_validation():
    """Test validation functions from run_classification.py."""
    print("\n" + "=" * 60)
    print("Testing run_classification.py validation functions")
    print("=" * 60)

    sys.path.insert(0, str(Path(__file__).parent))
    from datasets import Dataset
    from run_classification import DataValidationError
    from run_classification import validate_dataset_for_classification as validate_classification

    print("\n--- Test 1: Multiple text columns (normal) ---")
    data = {
        "text1": ["Hello world", "Foo bar"],
        "text2": ["Another text", "More text"],
        "label": [0, 1],
    }
    dataset = Dataset.from_dict(data)
    try:
        validate_classification(dataset, "test", ["text1", "text2"], is_regression=False)
        print("✓ PASS: Multiple text columns validated")
    except DataValidationError as e:
        print(f"✗ FAIL: Unexpected error: {e}")
        return False

    print("\n--- Test 2: Single text column with list ---")
    data = {
        "sentence": ["Text 1", "Text 2", "Text 3"],
        "label": [0, 1, 0],
    }
    dataset = Dataset.from_dict(data)
    try:
        validate_classification(dataset, "test", ["sentence"], is_regression=False)
        print("✓ PASS: Single text column validated")
    except DataValidationError as e:
        print(f"✗ FAIL: Unexpected error: {e}")
        return False

    print("\n--- Test 3: Empty text in first column ---")
    data = {
        "sentence": ["Valid text", "", "Another text"],
        "label": [0, 1, 0],
    }
    dataset = Dataset.from_dict(data)
    try:
        validate_classification(dataset, "test", ["sentence"], is_regression=False)
        print("✗ FAIL: Should have raised DataValidationError")
        return False
    except DataValidationError as e:
        print(f"✓ PASS: Correctly caught: {e}")

    print("\n--- Test 4: Empty text in second column ---")
    data = {
        "text1": ["Valid text", "Valid"],
        "text2": ["", "Valid text"],
        "label": [0, 1],
    }
    dataset = Dataset.from_dict(data)
    try:
        validate_classification(dataset, "test", ["text1", "text2"], is_regression=False)
        print("✗ FAIL: Should have raised DataValidationError")
        return False
    except DataValidationError as e:
        print(f"✓ PASS: Correctly caught: {e}")

    return True


def test_run_glue_no_trainer_validation():
    """Test validation functions from run_glue_no_trainer.py."""
    print("\n" + "=" * 60)
    print("Testing run_glue_no_trainer.py validation functions")
    print("=" * 60)

    sys.path.insert(0, str(Path(__file__).parent))
    from datasets import Dataset
    from run_glue_no_trainer import (
        DataValidationError,
    )
    from run_glue_no_trainer import (
        validate_dataset_for_classification as validate_no_trainer,
    )

    print("\n--- Test 1: Normal data (same as run_glue) ---")
    data = {
        "sentence1": ["This is a test.", "Another test."],
        "label": [0, 1],
    }
    dataset = Dataset.from_dict(data)
    try:
        validate_no_trainer(dataset, "test", "sentence1", None, is_regression=False)
        print("✓ PASS: Normal data validated")
    except DataValidationError as e:
        print(f"✗ FAIL: Unexpected error: {e}")
        return False

    print("\n--- Test 2: Missing sentence2 column ---")
    data = {
        "sentence1": ["Question"],
        "label": [0],
    }
    dataset = Dataset.from_dict(data)
    try:
        validate_no_trainer(dataset, "test", "sentence1", "sentence2", is_regression=False)
        print("✗ FAIL: Should have raised DataValidationError")
        return False
    except DataValidationError as e:
        print(f"✓ PASS: Correctly caught: {e}")

    return True


def test_csv_json_file_loading():
    """Test CSV and JSON file loading with validation."""
    print("\n" + "=" * 60)
    print("Testing CSV/JSON file loading with validation")
    print("=" * 60)

    from datasets import load_dataset

    sys.path.insert(0, str(Path(__file__).parent))
    from run_glue import DataValidationError, validate_dataset_for_classification

    print("\n--- Test 1: CSV file loading ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test.csv")
        data = {"sentence1": ["Text 1", "Text 2"], "label": [0, 1]}
        create_test_csv(data, csv_path)

        dataset = load_dataset("csv", data_files={"test": csv_path})["test"]

        try:
            validate_dataset_for_classification(dataset, "test", "sentence1", None, is_regression=False)
            print("✓ PASS: CSV file validated successfully")
        except DataValidationError as e:
            print(f"✗ FAIL: Unexpected error: {e}")
            return False

    print("\n--- Test 2: JSON file loading ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "test.json")
        data = [{"sentence1": "Text 1", "label": 0}, {"sentence1": "Text 2", "label": 1}]
        create_test_json(data, json_path)

        dataset = load_dataset("json", data_files={"test": json_path})["test"]

        try:
            validate_dataset_for_classification(dataset, "test", "sentence1", None, is_regression=False)
            print("✓ PASS: JSON file validated successfully")
        except DataValidationError as e:
            print(f"✗ FAIL: Unexpected error: {e}")
            return False

    print("\n--- Test 3: CSV with empty text ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test.csv")
        data = {"sentence1": ["Valid text", "", "Another text"], "label": [0, 1, 0]}
        create_test_csv(data, csv_path)

        dataset = load_dataset("csv", data_files={"test": csv_path})["test"]

        try:
            validate_dataset_for_classification(dataset, "test", "sentence1", None, is_regression=False)
            print("✗ FAIL: Should have raised DataValidationError")
            return False
        except DataValidationError as e:
            print(f"✓ PASS: Correctly caught: {e}")

    print("\n--- Test 4: CSV with wrong column name ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test.csv")
        data = {"wrong_column": ["Text 1", "Text 2"], "label": [0, 1]}
        create_test_csv(data, csv_path)

        dataset = load_dataset("csv", data_files={"test": csv_path})["test"]

        try:
            validate_dataset_for_classification(dataset, "test", "sentence1", None, is_regression=False)
            print("✗ FAIL: Should have raised DataValidationError")
            return False
        except DataValidationError as e:
            print(f"✓ PASS: Correctly caught: {e}")

    return True


def run_all_tests():
    """Run all test cases."""
    print("\n" + "=" * 60)
    print("RUNNING ALL DATA VALIDATION TESTS")
    print("=" * 60)

    tests = [
        ("run_glue.py validation", test_run_glue_validation),
        ("run_classification.py validation", test_run_classification_validation),
        ("run_glue_no_trainer.py validation", test_run_glue_no_trainer_validation),
        ("CSV/JSON file loading", test_csv_json_file_loading),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ FAIL: Unexpected exception in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} test groups passed")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
