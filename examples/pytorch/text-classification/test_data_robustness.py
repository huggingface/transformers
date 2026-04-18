#!/usr/bin/env python
"""
Test script to verify data robustness improvements for text classification.
Covers: empty text, missing text column, label anomalies, column mismatch, and normal execution.
"""

import csv
import json
import os
import subprocess
import tempfile


def create_test_csv(filepath, data):
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)


def run_test(args, expect_success=True, description=""):
    print(f"\n{'='*80}")
    print(f"TEST: {description}")
    print(f"{'='*80}")

    cmd = ["python", "run_classification.py"] + args
    result = subprocess.run(
        cmd,
        cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=True,
        text=True,
        timeout=120,
    )

    print(f"Command: {' '.join(cmd)}")
    print(f"Return code: {result.returncode}")

    if result.stdout:
        print("\nSTDOUT:")
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.stderr:
        print("\nSTDERR:")
        print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)

    success = result.returncode == 0 if expect_success else result.returncode != 0

    if success:
        print(f"\n✓ TEST PASSED: {description}")
    else:
        print(f"\n✗ TEST FAILED: {description}")

    return success, result.stdout + result.stderr


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Working in temp directory: {tmpdir}")

        base_args = [
            "--model_name_or_path", "distilbert-base-uncased",
            "--max_seq_length", "32",
            "--max_train_samples", "4",
            "--max_eval_samples", "2",
            "--do_train",
            "--do_eval",
            "--output_dir", tmpdir,
            "--per_device_train_batch_size", "2",
            "--per_device_eval_batch_size", "2",
            "--num_train_epochs", "1",
            "--overwrite_output_dir",
            "--overwrite_cache",
        ]

        all_passed = True

        normal_data = [
            {"sentence": "This is a good movie", "label": 1},
            {"sentence": "I enjoyed watching this film", "label": 1},
            {"sentence": "Terrible acting and bad plot", "label": 0},
            {"sentence": "Worst movie ever made", "label": 0},
        ]
        train_file = os.path.join(tmpdir, "train_normal.csv")
        val_file = os.path.join(tmpdir, "val_normal.csv")
        create_test_csv(train_file, normal_data[:2])
        create_test_csv(val_file, normal_data[2:])

        args = base_args + ["--train_file", train_file, "--validation_file", val_file]
        passed, _ = run_test(args, expect_success=True, description="Normal data - should execute successfully")
        all_passed = all_passed and passed

        empty_text_data = [
            {"sentence": "", "label": 1},
            {"sentence": "   ", "label": 0},
        ]
        train_file = os.path.join(tmpdir, "train_empty.csv")
        val_file = os.path.join(tmpdir, "val_empty.csv")
        create_test_csv(train_file, empty_text_data)
        create_test_csv(val_file, normal_data[2:])

        args = base_args + ["--train_file", train_file, "--validation_file", val_file]
        passed, output = run_test(args, expect_success=False, description="Empty text - should raise clear error")
        if passed:
            passed = "empty text values" in output.lower() or "empty" in output.lower()
            print(f"  Error message verified: {passed}")
        all_passed = all_passed and passed

        wrong_col_data = [
            {"text_content": "This is good", "label": 1},
            {"text_content": "This is bad", "label": 0},
        ]
        train_file = os.path.join(tmpdir, "train_wrongcol.csv")
        val_file = os.path.join(tmpdir, "val_wrongcol.csv")
        create_test_csv(train_file, wrong_col_data)
        create_test_csv(val_file, wrong_col_data)

        args = base_args + ["--train_file", train_file, "--validation_file", val_file]
        passed, output = run_test(args, expect_success=False, description="Wrong text column name - should raise clear error")
        if passed:
            passed = "not found" in output.lower() and "sentence" in output.lower()
            print(f"  Error message verified: {passed}")
        all_passed = all_passed and passed

        args = base_args + [
            "--train_file", train_file, "--validation_file", val_file,
            "--text_column_names", "text_content"
        ]
        passed, _ = run_test(args, expect_success=True, description="Correct column name specified - should succeed")
        all_passed = all_passed and passed

        bad_label_data = [
            {"sentence": "This is good", "label": None},
            {"sentence": "This is bad", "label": 0},
        ]
        train_file = os.path.join(tmpdir, "train_badlabel.csv")
        val_file = os.path.join(tmpdir, "val_badlabel.csv")
        create_test_csv(train_file, bad_label_data[:1])
        create_test_csv(val_file, bad_label_data[1:])

        args = base_args + ["--train_file", train_file, "--validation_file", val_file]
        passed, output = run_test(args, expect_success=False, description="None/NaN label values - should raise clear error")
        if passed:
            passed = "label" in output.lower() and ("nan" in output.lower() or "none" in output.lower())
            print(f"  Error message verified: {passed}")
        all_passed = all_passed and passed

        custom_col_data = [
            {"review": "Great movie!", "rating": 1},
            {"review": "Awful experience", "rating": 0},
        ]
        train_file = os.path.join(tmpdir, "train_custom.csv")
        val_file = os.path.join(tmpdir, "val_custom.csv")
        create_test_csv(train_file, custom_col_data)
        create_test_csv(val_file, custom_col_data)

        args = base_args + [
            "--train_file", train_file, "--validation_file", val_file,
            "--text_column_names", "review",
            "--label_column_name", "rating"
        ]
        passed, _ = run_test(args, expect_success=True, description="Custom column names correctly specified - should succeed")
        all_passed = all_passed and passed

        mixed_data = [
            {"sentence": 12345, "label": 1},
            {"sentence": 3.14, "label": 0},
        ]
        train_file = os.path.join(tmpdir, "train_mixed.csv")
        val_file = os.path.join(tmpdir, "val_mixed.csv")
        create_test_csv(train_file, mixed_data[:1])
        create_test_csv(val_file, mixed_data[1:])

        args = base_args + ["--train_file", train_file, "--validation_file", val_file]
        passed, _ = run_test(args, expect_success=True, description="Numeric text converted to string - should succeed")
        all_passed = all_passed and passed

        print(f"\n{'='*80}")
        print("SUMMARY:")
        print(f"{'='*80}")
        if all_passed:
            print("✓ ALL TESTS PASSED!")
        else:
            print("✗ SOME TESTS FAILED!")
            return 1

        return 0


if __name__ == "__main__":
    exit(main())
