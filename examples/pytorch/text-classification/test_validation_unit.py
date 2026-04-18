#!/usr/bin/env python
"""
Unit tests for dataset validation function.
"""

import sys
import os
import tempfile
import csv
import pandas as pd
from datasets import Dataset, DatasetDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_classification import validate_and_clean_dataset
import logging
logging.basicConfig(level=logging.INFO)


def test_normal_data():
    print("\n=== Test 1: Normal data should pass validation ===")
    data = {
        "train": Dataset.from_dict({
            "sentence": ["This is good", "This is bad"],
            "label": [1, 0]
        }),
        "validation": Dataset.from_dict({
            "sentence": ["Great movie", "Terrible film"],
            "label": [1, 0]
        })
    }
    raw_datasets = DatasetDict(data)

    try:
        validate_and_clean_dataset(raw_datasets, text_column_names=None, label_column_name=None, is_regression=None)
        print("✓ PASSED: Normal data passed validation")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_missing_text_column():
    print("\n=== Test 2: Missing text column should raise clear error ===")
    data = {
        "train": Dataset.from_dict({
            "wrong_column": ["This is good", "This is bad"],
            "label": [1, 0]
        })
    }
    raw_datasets = DatasetDict(data)

    try:
        validate_and_clean_dataset(raw_datasets, text_column_names=None, label_column_name=None, is_regression=None)
        print("✗ FAILED: Should have raised an error")
        return False
    except ValueError as e:
        if "not found" in str(e) and "sentence" in str(e).lower():
            print(f"✓ PASSED: Clear error message: {e}")
            return True
        else:
            print(f"✗ FAILED: Unexpected error: {e}")
            return False
    except Exception as e:
        print(f"✗ FAILED: Wrong exception type: {type(e).__name__}: {e}")
        return False


def test_custom_text_column_name():
    print("\n=== Test 3: Custom text column name correctly specified ===")
    data = {
        "train": Dataset.from_dict({
            "review": ["This is good", "This is bad"],
            "label": [1, 0]
        })
    }
    raw_datasets = DatasetDict(data)

    try:
        validate_and_clean_dataset(raw_datasets, text_column_names="review", label_column_name=None, is_regression=None)
        print("✓ PASSED: Custom column name correctly handled")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_empty_text():
    print("\n=== Test 4: Empty text strings should raise clear error ===")
    data = {
        "train": Dataset.from_dict({
            "sentence": ["", "   ", "valid text"],
            "label": [1, 0, 1]
        })
    }
    raw_datasets = DatasetDict(data)

    try:
        validate_and_clean_dataset(raw_datasets, text_column_names=None, label_column_name=None, is_regression=None)
        print("✗ FAILED: Should have raised an error")
        return False
    except ValueError as e:
        if "empty" in str(e).lower():
            print(f"✓ PASSED: Clear error message: {e}")
            return True
        else:
            print(f"✗ FAILED: Unexpected error: {e}")
            return False


def test_none_text_values():
    print("\n=== Test 5: None/NaN text values should raise clear error ===")
    df = pd.DataFrame({
        "sentence": [None, "valid text"],
        "label": [1, 0]
    })
    data = {
        "train": Dataset.from_pandas(df)
    }
    raw_datasets = DatasetDict(data)

    try:
        validate_and_clean_dataset(raw_datasets, text_column_names=None, label_column_name=None, is_regression=None)
        print("✗ FAILED: Should have raised an error")
        return False
    except ValueError as e:
        if "nan" in str(e).lower() or "none" in str(e).lower():
            print(f"✓ PASSED: Clear error message: {e}")
            return True
        else:
            print(f"✗ FAILED: Unexpected error: {e}")
            return False


def test_none_label_values():
    print("\n=== Test 6: None/NaN label values should raise clear error ===")
    df = pd.DataFrame({
        "sentence": ["good text", "bad text"],
        "label": [None, 0]
    })
    data = {
        "train": Dataset.from_pandas(df)
    }
    raw_datasets = DatasetDict(data)

    try:
        validate_and_clean_dataset(raw_datasets, text_column_names=None, label_column_name=None, is_regression=None)
        print("✗ FAILED: Should have raised an error")
        return False
    except ValueError as e:
        if "label" in str(e).lower() and ("nan" in str(e).lower() or "none" in str(e).lower()):
            print(f"✓ PASSED: Clear error message: {e}")
            return True
        else:
            print(f"✗ FAILED: Unexpected error: {e}")
            return False


def test_missing_label_column():
    print("\n=== Test 7: Missing label column should raise clear error ===")
    data = {
        "train": Dataset.from_dict({
            "sentence": ["This is good", "This is bad"],
            "wrong_label": [1, 0]
        })
    }
    raw_datasets = DatasetDict(data)

    try:
        validate_and_clean_dataset(raw_datasets, text_column_names=None, label_column_name=None, is_regression=None)
        print("✗ FAILED: Should have raised an error")
        return False
    except ValueError as e:
        if "label" in str(e).lower() and "not found" in str(e).lower():
            print(f"✓ PASSED: Clear error message: {e}")
            return True
        else:
            print(f"✗ FAILED: Unexpected error: {e}")
            return False


def test_custom_label_column_name():
    print("\n=== Test 8: Custom label column name correctly specified ===")
    data = {
        "train": Dataset.from_dict({
            "sentence": ["This is good", "This is bad"],
            "rating": [1, 0]
        })
    }
    raw_datasets = DatasetDict(data)

    try:
        validate_and_clean_dataset(raw_datasets, text_column_names=None, label_column_name="rating", is_regression=None)
        print("✓ PASSED: Custom label column name correctly handled")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def main():
    tests = [
        test_normal_data,
        test_missing_text_column,
        test_custom_text_column_name,
        test_empty_text,
        test_none_text_values,
        test_none_label_values,
        test_missing_label_column,
        test_custom_label_column_name,
    ]

    print("=" * 80)
    print("Running dataset validation unit tests")
    print("=" * 80)

    results = []
    for test in tests:
        results.append(test())

    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("✓ ALL TESTS PASSED!")
        return 0
    else:
        print("✗ SOME TESTS FAILED!")
        return 1


if __name__ == "__main__":
    exit(main())
