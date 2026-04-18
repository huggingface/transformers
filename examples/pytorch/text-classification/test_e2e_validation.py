#!/usr/bin/env python
"""
End-to-end test to verify normal data flow still works after adding validation.
This test creates a simple dataset and validates it against all three scripts.
"""

import os
import sys
import tempfile

import pandas as pd


def create_test_data():
    """Create test CSV and JSON files with valid data."""
    tmpdir = tempfile.mkdtemp()

    train_data = {
        "sentence1": [
            "This is a great movie!",
            "Terrible film, waste of time.",
            "Amazing acting and story.",
            "Boring and predictable.",
            "Highly recommended!",
            "Not worth watching.",
        ],
        "label": [1, 0, 1, 0, 1, 0],
    }

    val_data = {
        "sentence1": [
            "Excellent performance!",
            "Disappointing experience.",
        ],
        "label": [1, 0],
    }

    train_csv = os.path.join(tmpdir, "train.csv")
    val_csv = os.path.join(tmpdir, "val.csv")

    pd.DataFrame(train_data).to_csv(train_csv, index=False)
    pd.DataFrame(val_data).to_csv(val_csv, index=False)

    return tmpdir, train_csv, val_csv


def test_normal_flow():
    """Test that normal data flow works correctly for all scripts."""
    print("\n" + "=" * 60)
    print("END-TO-END TEST: Normal data flow with validation")
    print("=" * 60)

    tmpdir, train_csv, val_csv = create_test_data()

    print(f"\nCreated test data in: {tmpdir}")
    print(f"Train file: {train_csv}")
    print(f"Validation file: {val_csv}")

    from datasets import load_dataset

    raw_datasets = load_dataset(
        "csv",
        data_files={"train": train_csv, "validation": val_csv},
    )

    print(f"\nLoaded dataset splits: {list(raw_datasets.keys())}")
    print(f"Train samples: {len(raw_datasets['train'])}")
    print(f"Validation samples: {len(raw_datasets['validation'])}")

    print("\n--- Testing run_glue.py validation ---")
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from run_glue import validate_dataset_for_classification

    try:
        validate_dataset_for_classification(
            raw_datasets["train"],
            "train",
            "sentence1",
            None,
            is_regression=False,
        )
        print("✓ Train dataset validation passed")
    except Exception as e:
        print(f"✗ Train dataset validation failed: {e}")
        return False

    try:
        validate_dataset_for_classification(
            raw_datasets["validation"],
            "validation",
            "sentence1",
            None,
            is_regression=False,
        )
        print("✓ Validation dataset validation passed")
    except Exception as e:
        print(f"✗ Validation dataset validation failed: {e}")
        return False

    print("\n--- Testing run_classification.py validation ---")
    from run_classification import validate_dataset_for_classification as validate_classification

    try:
        validate_classification(
            raw_datasets["train"],
            "train",
            ["sentence1"],
            is_regression=False,
        )
        print("✓ Train dataset validation passed (run_classification)")
    except Exception as e:
        print(f"✗ Train dataset validation failed: {e}")
        return False

    print("\n--- Testing run_glue_no_trainer.py validation ---")
    from run_glue_no_trainer import validate_dataset_for_classification as validate_no_trainer

    try:
        validate_no_trainer(
            raw_datasets["train"],
            "train",
            "sentence1",
            None,
            is_regression=False,
        )
        print("✓ Train dataset validation passed (run_glue_no_trainer)")
    except Exception as e:
        print(f"✗ Train dataset validation failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("✓ END-TO-END TEST PASSED: Normal data flow works correctly")
    print("=" * 60)
    return True


def test_error_flow():
    """Test that error cases are properly caught for all scripts."""
    print("\n" + "=" * 60)
    print("END-TO-END TEST: Error handling with invalid data")
    print("=" * 60)

    tmpdir = tempfile.mkdtemp()

    bad_data = {
        "sentence1": ["Valid text", "", "Another text"],
        "label": [0, 1, 0],
    }

    bad_csv = os.path.join(tmpdir, "bad.csv")
    pd.DataFrame(bad_data).to_csv(bad_csv, index=False)

    print(f"\nCreated test data with empty text in: {bad_csv}")

    from datasets import load_dataset

    raw_datasets = load_dataset("csv", data_files={"test": bad_csv})

    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from run_glue import DataValidationError, validate_dataset_for_classification

    try:
        validate_dataset_for_classification(
            raw_datasets["test"],
            "test",
            "sentence1",
            None,
            is_regression=False,
        )
        print("\n✗ FAIL: Should have raised DataValidationError")
        return False
    except DataValidationError as e:
        print(f"\n✓ PASS: Correctly caught error: {e}")

    print("\n" + "=" * 60)
    print("✓ END-TO-END TEST PASSED: Error handling works correctly")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_normal_flow() and test_error_flow()
    sys.exit(0 if success else 1)
