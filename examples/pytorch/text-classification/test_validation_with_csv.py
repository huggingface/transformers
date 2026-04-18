#!/usr/bin/env python
"""
Test validation using actual CSV files through the complete data loading path.
This tests the exact code path used for local data reading.
"""

import sys
import os
import csv
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from datasets import load_dataset
from run_classification import validate_and_clean_dataset
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def create_csv(filepath, rows):
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_csv_empty_text():
    print("\n" + "="*80)
    print("TEST: CSV with empty text values (complete local reading path)")
    print("="*80)

    with tempfile.TemporaryDirectory() as tmpdir:
        data = [
            {"sentence": "", "label": 1},
            {"sentence": "   ", "label": 0},
        ]
        train_file = os.path.join(tmpdir, "train.csv")
        val_file = os.path.join(tmpdir, "val.csv")
        create_csv(train_file, data)
        create_csv(val_file, data)

        print(f"CSV files created: {train_file}, {val_file}")

        raw_datasets = load_dataset("csv", data_files={"train": train_file, "validation": val_file})
        print(f"Dataset loaded from CSV: {raw_datasets}")

        try:
            validate_and_clean_dataset(raw_datasets, None, None, None)
            print("✗ FAILED: Should have raised error for empty text")
            return False
        except ValueError as e:
            print(f"✓ PASSED: Got clear error: {e}")
            return True


def test_csv_missing_column():
    print("\n" + "="*80)
    print("TEST: CSV with wrong column names (complete local reading path)")
    print("="*80)

    with tempfile.TemporaryDirectory() as tmpdir:
        data = [
            {"my_text": "This is good", "label": 1},
            {"my_text": "This is bad", "label": 0},
        ]
        train_file = os.path.join(tmpdir, "train.csv")
        val_file = os.path.join(tmpdir, "val.csv")
        create_csv(train_file, data)
        create_csv(val_file, data)

        raw_datasets = load_dataset("csv", data_files={"train": train_file, "validation": val_file})
        print(f"Dataset columns: {raw_datasets['train'].column_names}")

        try:
            validate_and_clean_dataset(raw_datasets, None, None, None)
            print("✗ FAILED: Should have raised error for missing text column")
            return False
        except ValueError as e:
            if "not found" in str(e) and "text_column_names" in str(e):
                print(f"✓ PASSED: Got clear error with hint: {e}")
                return True
            else:
                print(f"✗ FAILED: Wrong error message: {e}")
                return False


def test_csv_correct_custom_column():
    print("\n" + "="*80)
    print("TEST: CSV with custom column names correctly specified")
    print("="*80)

    with tempfile.TemporaryDirectory() as tmpdir:
        data = [
            {"my_text": "This is good", "my_label": 1},
            {"my_text": "This is bad", "my_label": 0},
        ]
        train_file = os.path.join(tmpdir, "train.csv")
        val_file = os.path.join(tmpdir, "val.csv")
        create_csv(train_file, data)
        create_csv(val_file, data)

        raw_datasets = load_dataset("csv", data_files={"train": train_file, "validation": val_file})

        try:
            validate_and_clean_dataset(raw_datasets, "my_text", "my_label", None)
            print("✓ PASSED: Custom column names correctly handled")
            return True
        except Exception as e:
            print(f"✗ FAILED: {e}")
            return False


def test_csv_numeric_text():
    print("\n" + "="*80)
    print("TEST: CSV with numeric values in text column (should be string-convertible)")
    print("="*80)

    with tempfile.TemporaryDirectory() as tmpdir:
        data = [
            {"sentence": 12345, "label": 1},
            {"sentence": 67890, "label": 0},
        ]
        train_file = os.path.join(tmpdir, "train.csv")
        val_file = os.path.join(tmpdir, "val.csv")
        create_csv(train_file, data)
        create_csv(val_file, data)

        raw_datasets = load_dataset("csv", data_files={"train": train_file, "validation": val_file})
        print(f"Text types: {type(raw_datasets['train'][0]['sentence'])}")

        try:
            validate_and_clean_dataset(raw_datasets, None, None, None)
            print("✓ PASSED: Numeric text values accepted (convertible to string)")
            return True
        except Exception as e:
            print(f"✗ FAILED: {e}")
            return False


def test_csv_normal_data():
    print("\n" + "="*80)
    print("TEST: Normal valid CSV data (should pass)")
    print("="*80)

    with tempfile.TemporaryDirectory() as tmpdir:
        data = [
            {"sentence": "This is a great movie", "label": 1},
            {"sentence": "This is a terrible movie", "label": 0},
        ]
        train_file = os.path.join(tmpdir, "train.csv")
        val_file = os.path.join(tmpdir, "val.csv")
        create_csv(train_file, data)
        create_csv(val_file, data)

        raw_datasets = load_dataset("csv", data_files={"train": train_file, "validation": val_file})

        try:
            validate_and_clean_dataset(raw_datasets, None, None, None)
            print("✓ PASSED: Normal data passes validation")
            return True
        except Exception as e:
            print(f"✗ FAILED: {e}")
            return False


def main():
    tests = [
        test_csv_normal_data,
        test_csv_empty_text,
        test_csv_missing_column,
        test_csv_correct_custom_column,
        test_csv_numeric_text,
    ]

    print("\n" + "="*80)
    print("VALIDATION TESTS WITH ACTUAL CSV FILES (LOCAL DATA PATH)")
    print("="*80)

    results = []
    for test in tests:
        results.append(test())

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("\n✓ ALL TESTS PASSED!")
        print("\nCoverage summary:")
        print("  - Normal CSV data → passes validation")
        print("  - Empty text in CSV → caught early with clear message")
        print("  - Wrong column name → caught with helpful hint")
        print("  - Custom column names → work correctly when specified")
        print("  - Numeric text → safely accepted (convertible to string)")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
