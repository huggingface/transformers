#!/usr/bin/env python
"""
End-to-end test for local CSV/JSON data robustness in text classification example.
This test verifies the complete data flow from local file reading to preprocessing.
"""

import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def create_csv(filepath, rows):
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def create_json(filepath, rows):
    with open(filepath, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_script(args, cwd, description):
    print(f"\n{'='*100}")
    print(f"TEST: {description}")
    print(f"{'='*100}")

    cmd = [sys.executable, "run_classification.py"] + args
    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=180,
    )

    print(f"Return code: {result.returncode}")

    output = result.stdout + result.stderr
    if len(output) > 3000:
        output = output[-3000:]
    print(f"\nOutput:\n{output}")

    return result.returncode, result.stdout + result.stderr


def main():
    script_dir = Path(__file__).parent
    all_passed = True

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\nWorking directory: {tmpdir}")

        base_args = [
            "--model_name_or_path", "distilbert-base-uncased",
            "--max_seq_length", "16",
            "--max_train_samples", "2",
            "--max_eval_samples", "2",
            "--do_train",
            "--do_eval",
            "--output_dir", os.path.join(tmpdir, "output"),
            "--per_device_train_batch_size", "2",
            "--per_device_eval_batch_size", "2",
            "--num_train_epochs", "1",
            "--overwrite_cache",
        ]

        print("\n" + "="*100)
        print("SCENARIO 1: Normal valid CSV data - should succeed")
        print("="*100)

        normal_data = [
            {"sentence": "This is a great movie", "label": 1},
            {"sentence": "I really enjoyed this film", "label": 1},
            {"sentence": "This was a terrible movie", "label": 0},
            {"sentence": "Worst acting I have ever seen", "label": 0},
        ]

        train_file = os.path.join(tmpdir, "train_normal.csv")
        val_file = os.path.join(tmpdir, "val_normal.csv")
        create_csv(train_file, normal_data[:2])
        create_csv(val_file, normal_data[2:])

        args = base_args + ["--train_file", train_file, "--validation_file", val_file]
        rc, output = run_script(args, script_dir, "Normal CSV data")
        if rc == 0:
            print("✓ PASSED: Normal data runs successfully")
        else:
            print("✗ FAILED: Normal data should succeed")
            all_passed = False

        print("\n" + "="*100)
        print("SCENARIO 2: Empty text in CSV - should raise clear error BEFORE tokenization")
        print("="*100)

        empty_text_data = [
            {"sentence": "", "label": 1},
            {"sentence": "   ", "label": 0},
        ]
        train_file = os.path.join(tmpdir, "train_empty.csv")
        val_file = os.path.join(tmpdir, "val_empty.csv")
        create_csv(train_file, empty_text_data)
        create_csv(val_file, normal_data[2:])

        args = base_args + ["--train_file", train_file, "--validation_file", val_file]
        rc, output = run_script(args, script_dir, "Empty text values")
        if rc != 0 and "empty text values" in output.lower():
            print("✓ PASSED: Empty text detected with clear error message")
        else:
            print("✗ FAILED: Empty text should raise clear error")
            all_passed = False

        print("\n" + "="*100)
        print("SCENARIO 3: Wrong text column name - should raise clear error with hint")
        print("="*100)

        wrong_col_data = [
            {"review_text": "Great movie!", "label": 1},
            {"review_text": "Bad movie", "label": 0},
        ]
        train_file = os.path.join(tmpdir, "train_wrongcol.csv")
        val_file = os.path.join(tmpdir, "val_wrongcol.csv")
        create_csv(train_file, wrong_col_data)
        create_csv(val_file, wrong_col_data)

        args = base_args + ["--train_file", train_file, "--validation_file", val_file]
        rc, output = run_script(args, script_dir, "Wrong column name (no --text_column_names)")
        if rc != 0 and "not found" in output.lower() and "text_column_names" in output:
            print("✓ PASSED: Missing text column detected with helpful hint")
        else:
            print("✗ FAILED: Should detect missing column with hint")
            all_passed = False

        print("\n" + "="*100)
        print("SCENARIO 4: Correct column name specified - should succeed")
        print("="*100)

        args = base_args + [
            "--train_file", train_file, "--validation_file", val_file,
            "--text_column_names", "review_text"
        ]
        rc, output = run_script(args, script_dir, "Correct --text_column_names specified")
        if rc == 0:
            print("✓ PASSED: Custom column name works correctly")
        else:
            print("✗ FAILED: Custom column name should succeed")
            all_passed = False

        print("\n" + "="*100)
        print("SCENARIO 5: Wrong label column name - should raise clear error")
        print("="*100)

        wrong_label_data = [
            {"sentence": "Great movie!", "rating": 1},
            {"sentence": "Bad movie", "rating": 0},
        ]
        train_file = os.path.join(tmpdir, "train_wronglabel.csv")
        val_file = os.path.join(tmpdir, "val_wronglabel.csv")
        create_csv(train_file, wrong_label_data)
        create_csv(val_file, wrong_label_data)

        args = base_args + ["--train_file", train_file, "--validation_file", val_file]
        rc, output = run_script(args, script_dir, "Wrong label column name")
        if rc != 0 and "label" in output.lower() and "not found" in output.lower():
            print("✓ PASSED: Missing label column detected with clear error")
        else:
            print("✗ FAILED: Should detect missing label column")
            all_passed = False

        print("\n" + "="*100)
        print("SCENARIO 6: Correct label column name specified - should succeed")
        print("="*100)

        args = base_args + [
            "--train_file", train_file, "--validation_file", val_file,
            "--label_column_name", "rating"
        ]
        rc, output = run_script(args, script_dir, "Correct --label_column_name specified")
        if rc == 0:
            print("✓ PASSED: Custom label column name works correctly")
        else:
            print("✗ FAILED: Custom label column name should succeed")
            all_passed = False

        print("\n" + "="*100)
        print("SCENARIO 7: Numeric values in text column - safely converted to string")
        print("="*100)

        numeric_text_data = [
            {"sentence": 12345, "label": 1},
            {"sentence": 99999, "label": 0},
        ]
        train_file = os.path.join(tmpdir, "train_numeric.csv")
        val_file = os.path.join(tmpdir, "val_numeric.csv")
        create_csv(train_file, numeric_text_data[:1])
        create_csv(val_file, numeric_text_data[1:])

        args = base_args + ["--train_file", train_file, "--validation_file", val_file]
        rc, output = run_script(args, script_dir, "Numeric text values converted to string")
        if rc == 0:
            print("✓ PASSED: Numeric values are safely converted to string")
        else:
            print("✗ FAILED: Numeric text should be handled gracefully")
            all_passed = False

        print("\n" + "="*100)
        print("SCENARIO 8: JSON format file with normal data - should succeed")
        print("="*100)

        train_file = os.path.join(tmpdir, "train_normal.json")
        val_file = os.path.join(tmpdir, "val_normal.json")
        create_json(train_file, normal_data[:2])
        create_json(val_file, normal_data[2:])

        args = base_args + ["--train_file", train_file, "--validation_file", val_file]
        rc, output = run_script(args, script_dir, "JSON format file")
        if rc == 0:
            print("✓ PASSED: JSON data runs successfully")
        else:
            print("✗ FAILED: JSON format should work")
            all_passed = False

        print("\n" + "="*100)
        print("SUMMARY OF ALL TESTS")
        print("="*100)
        if all_passed:
            print("✓ ALL TESTS PASSED!")
            print("\nThe robustness improvements are working correctly:")
            print("  - Empty text caught early with clear message")
            print("  - Column name mismatches caught with helpful hints")
            print("  - Custom column names work correctly when specified")
            print("  - Numeric text values are safely converted to string")
            print("  - Both CSV and JSON formats are supported")
            print("  - Normal data continues to work as before")
            return 0
        else:
            print("✗ SOME TESTS FAILED!")
            return 1


if __name__ == "__main__":
    exit(main())
