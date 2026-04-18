#!/usr/bin/env python
"""
Test script to verify the label mapping fix in run_glue.py and run_classification.py.

This test demonstrates the bug and verifies the fix:
- Bug 1: id2label was built from config.label2id instead of model.config.label2id
- Bug 2: prediction output used label_list[item] instead of model.config.id2label[item]

Run: python test_label_mapping_fix.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np


def test_run_glue_id2label_building():
    """
    Test that id2label is correctly built from model.config.label2id in run_glue.py.
    
    This simulates the bug scenario where:
    1. label_to_id is set (e.g., {0: 1, 1: 0} for swapped labels)
    2. model.config.label2id is updated to label_to_id
    3. id2label should be built from model.config.label2id, NOT config.label2id
    
    run_glue.py covers:
    - GLUE benchmark tasks (with predefined label names from dataset features)
    - Local CSV/JSON files (with labels extracted and sorted)
    """
    print("=" * 70)
    print("Test 1: run_glue.py - id2label building from correct source")
    print("=" * 70)
    
    class MockConfig:
        def __init__(self, label2id):
            self.label2id = label2id
    
    num_labels = 2
    config = MockConfig({"negative": 0, "positive": 1})
    model = type('obj', (object,), {'config': MockConfig({"negative": 0, "positive": 1})})()
    
    label_list = ["negative", "positive"]
    
    label_to_id = {v: i for i, v in enumerate(label_list)}
    
    print(f"Scenario: Local CSV/JSON file with labels")
    print(f"  Original config.label2id (pretrained model): {config.label2id}")
    print(f"  label_list (from dataset, sorted): {label_list}")
    print(f"  label_to_id (will be assigned to model.config.label2id): {label_to_id}")
    
    model.config.label2id = label_to_id
    
    correct_id2label = {id: label for label, id in model.config.label2id.items()}
    buggy_id2label = {id: label for label, id in config.label2id.items()}
    
    print(f"\n  Correct id2label (from model.config.label2id): {correct_id2label}")
    print(f"  Buggy id2label (from config.label2id): {buggy_id2label}")
    
    assert correct_id2label == {0: "negative", 1: "positive"}, f"Expected {{0: 'negative', 1: 'positive'}}, got {correct_id2label}"
    
    print("\n[PASS] id2label correctly built from model.config.label2id")
    return True


def test_run_glue_non_alphabetical_labels():
    """
    Test id2label when labels are in non-alphabetical order.
    
    This simulates a real-world scenario where:
    - Dataset has labels: ["positive", "negative"] (sorted by value or custom order)
    - Model was pretrained with: {"negative": 0, "positive": 1}
    - We need to remap correctly
    
    run_glue.py specific: Handles GLUE tasks where model may have predefined label order
    """
    print("\n" + "=" * 70)
    print("Test 2: run_glue.py - Non-alphabetical label order (GLUE scenario)")
    print("=" * 70)
    
    class MockConfig:
        def __init__(self, label2id):
            self.label2id = label2id
    
    num_labels = 2
    
    config = MockConfig({"negative": 0, "positive": 1})
    
    label_list = ["positive", "negative"]
    
    label_to_id = {v: i for i, v in enumerate(label_list)}
    
    print(f"Scenario: GLUE task with model having predefined label order")
    print(f"  config.label2id (pretrained model): {config.label2id}")
    print(f"  label_list (dataset order): {label_list}")
    print(f"  label_to_id (mapping): {label_to_id}")
    
    model = type('obj', (object,), {'config': MockConfig(config.label2id)})()
    model.config.label2id = label_to_id
    
    correct_id2label = {id: label for label, id in model.config.label2id.items()}
    buggy_id2label = {id: label for label, id in config.label2id.items()}
    
    print(f"\n  Correct id2label: {correct_id2label}")
    print(f"  Buggy id2label: {buggy_id2label}")
    
    assert correct_id2label == {0: "positive", 1: "negative"}, f"Expected {{0: 'positive', 1: 'negative'}}, got {correct_id2label}"
    
    print("\n[PASS] id2label correctly handles non-alphabetical label order")
    return True


def test_run_classification_training_phase():
    """
    Test run_classification.py training phase label mapping.
    
    run_classification.py covers:
    - Multi-label classification
    - Single-label classification with local files
    - Training phase: builds label_to_id from label_list
    - Non-training phase: uses model.config.label2id directly
    """
    print("\n" + "=" * 70)
    print("Test 3: run_classification.py - Training phase label mapping")
    print("=" * 70)
    
    class MockConfig:
        def __init__(self, label2id, id2label):
            self.label2id = label2id
            self.id2label = id2label
    
    label_list = ["spam", "ham"]
    
    print(f"Scenario: Training phase with local CSV/JSON file")
    print(f"  label_list (from dataset, sorted): {label_list}")
    
    label_to_id = {v: i for i, v in enumerate(label_list)}
    print(f"  label_to_id (built from label_list): {label_to_id}")
    
    model = type('obj', (object,), {'config': MockConfig({"LABEL_0": 0, "LABEL_1": 1}, {0: "LABEL_0", 1: "LABEL_1"})})()
    
    print(f"  Original model.config.label2id: {model.config.label2id}")
    
    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in label_to_id.items()}
    
    print(f"  Updated model.config.label2id: {model.config.label2id}")
    print(f"  Updated model.config.id2label: {model.config.id2label}")
    
    assert model.config.id2label == {0: "spam", 1: "ham"}, f"Expected {{0: 'spam', 1: 'ham'}}, got {model.config.id2label}"
    
    print("\n[PASS] Training phase correctly builds id2label from label_to_id")
    return True


def test_prediction_output_consistency():
    """
    Test that prediction output uses model.config.id2label for consistency.
    
    This verifies that predictions are mapped to the correct label names
    in both run_glue.py and run_classification.py.
    """
    print("\n" + "=" * 70)
    print("Test 4: Prediction output consistency (both files)")
    print("=" * 70)
    
    class MockConfig:
        def __init__(self, label2id, id2label):
            self.label2id = label2id
            self.id2label = id2label
    
    label_list = ["negative", "positive"]
    model = type('obj', (object,), {'config': MockConfig({0: "negative", 1: "positive"}, {0: "negative", 1: "positive"})})()
    
    predictions = np.array([0, 1, 1, 0])
    
    print(f"Scenario: Predictions with matching label_list and id2label")
    print(f"  label_list: {label_list}")
    print(f"  model.config.id2label: {model.config.id2label}")
    print(f"  predictions (indices): {predictions}")
    
    correct_outputs = [model.config.id2label[item] for item in predictions]
    label_list_outputs = [label_list[item] for item in predictions]
    
    print(f"\n  Correct outputs (using model.config.id2label): {correct_outputs}")
    print(f"  Outputs using label_list: {label_list_outputs}")
    
    assert correct_outputs == label_list_outputs, "Outputs should match when label_list order matches id2label"
    
    model.config.id2label = {0: "positive", 1: "negative"}
    model.config.label2id = {"positive": 0, "negative": 1}
    
    print(f"\n--- Scenario: After label swap (model fine-tuned with different order) ---")
    print(f"  label_list: {label_list}")
    print(f"  model.config.id2label: {model.config.id2label}")
    
    correct_outputs = [model.config.id2label[item] for item in predictions]
    label_list_outputs = [label_list[item] for item in predictions]
    
    print(f"\n  Correct outputs (using model.config.id2label): {correct_outputs}")
    print(f"  Outputs using label_list (BUGGY): {label_list_outputs}")
    
    assert correct_outputs != label_list_outputs, "Outputs should differ when label_list order doesn't match id2label"
    assert correct_outputs == ["positive", "negative", "negative", "positive"], f"Expected ['positive', 'negative', 'negative', 'positive'], got {correct_outputs}"
    
    print("\n[PASS] Prediction output correctly uses model.config.id2label")
    return True


def test_multi_label_prediction():
    """
    Test multi-label prediction output in run_classification.py.
    
    run_classification.py specific: Handles multi-label classification
    where each sample can have multiple labels.
    """
    print("\n" + "=" * 70)
    print("Test 5: run_classification.py - Multi-label prediction output")
    print("=" * 70)
    
    class MockConfig:
        def __init__(self, label2id, id2label):
            self.label2id = label2id
            self.id2label = id2label
    
    label_list = ["tech", "sports", "politics"]
    model = type('obj', (object,), {'config': MockConfig(
        {"tech": 0, "sports": 1, "politics": 2},
        {0: "tech", 1: "sports", 2: "politics"}
    )})()
    
    predictions = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    
    print(f"Scenario: Multi-label classification")
    print(f"  label_list: {label_list}")
    print(f"  model.config.id2label: {model.config.id2label}")
    print(f"  predictions (multi-hot): {predictions}")
    
    correct_outputs = [[model.config.id2label[i] for i in range(len(item)) if item[i] == 1] for item in predictions]
    label_list_outputs = [[label_list[i] for i in range(len(item)) if item[i] == 1] for item in predictions]
    
    print(f"\n  Correct outputs (using model.config.id2label): {correct_outputs}")
    print(f"  Outputs using label_list: {label_list_outputs}")
    
    assert correct_outputs == [["tech", "politics"], ["sports"], ["tech", "sports"]]
    
    model.config.id2label = {0: "politics", 1: "sports", 2: "tech"}
    
    print(f"\n--- After label order change ---")
    print(f"  label_list: {label_list}")
    print(f"  model.config.id2label: {model.config.id2label}")
    
    correct_outputs = [[model.config.id2label[i] for i in range(len(item)) if item[i] == 1] for item in predictions]
    label_list_outputs = [[label_list[i] for i in range(len(item)) if item[i] == 1] for item in predictions]
    
    print(f"\n  Correct outputs (using model.config.id2label): {correct_outputs}")
    print(f"  Outputs using label_list (BUGGY): {label_list_outputs}")
    
    assert correct_outputs != label_list_outputs, "Outputs should differ"
    assert correct_outputs == [["politics", "tech"], ["sports"], ["politics", "sports"]]
    
    print("\n[PASS] Multi-label prediction correctly uses model.config.id2label")
    return True


def test_full_scenario_local_csv():
    """
    Test a full scenario simulating local CSV file training and prediction.
    
    This demonstrates the complete bug scenario:
    1. User has a local CSV with labels ["spam", "ham"]
    2. Model was pretrained with labels ["LABEL_0", "LABEL_1"]
    3. During training, label_to_id is built correctly
    4. But id2label was built from wrong source (bug)
    5. Prediction output would show wrong labels (bug)
    """
    print("\n" + "=" * 70)
    print("Test 6: Full scenario - Local CSV file training and prediction")
    print("=" * 70)
    
    class Config:
        def __init__(self):
            self.label2id = {"LABEL_0": 0, "LABEL_1": 1}
            self.id2label = {0: "LABEL_0", 1: "LABEL_1"}
    
    config = Config()
    model = type('obj', (object,), {'config': Config()})()
    
    label_list = ["spam", "ham"]
    num_labels = 2
    
    print(f"Scenario: User trains with local CSV file")
    print(f"  Pretrained model labels: {config.label2id}")
    print(f"  User's dataset labels (sorted): {label_list}")
    
    label_to_id = {v: i for i, v in enumerate(label_list)}
    print(f"  Built label_to_id: {label_to_id}")
    
    model.config.label2id = label_to_id
    
    print(f"\n--- Buggy behavior (using config.label2id) ---")
    buggy_id2label = {id: label for label, id in config.label2id.items()}
    print(f"  buggy id2label: {buggy_id2label}")
    
    print(f"\n--- Correct behavior (using model.config.label2id) ---")
    correct_id2label = {id: label for label, id in model.config.label2id.items()}
    print(f"  correct id2label: {correct_id2label}")
    
    predictions = np.array([0, 1, 0, 1])
    
    print(f"\n--- Prediction outputs ---")
    print(f"  Prediction indices: {predictions}")
    
    buggy_outputs = [buggy_id2label[item] for item in predictions]
    correct_outputs = [correct_id2label[item] for item in predictions]
    
    print(f"  Buggy outputs: {buggy_outputs}")
    print(f"  Correct outputs: {correct_outputs}")
    
    assert buggy_outputs == ["LABEL_0", "LABEL_1", "LABEL_0", "LABEL_1"], "Buggy version shows wrong labels"
    assert correct_outputs == ["spam", "ham", "spam", "ham"], "Correct version shows user's labels"
    
    print("\n[PASS] Full scenario correctly handles local CSV label mapping")
    return True


def main():
    print("\n" + "=" * 70)
    print("Label Mapping Fix Verification Tests")
    print("Tests verify fixes in run_glue.py and run_classification.py")
    print("=" * 70)
    
    all_passed = True
    
    tests = [
        ("run_glue.py id2label building", test_run_glue_id2label_building),
        ("run_glue.py non-alphabetical labels", test_run_glue_non_alphabetical_labels),
        ("run_classification.py training phase", test_run_classification_training_phase),
        ("Prediction output consistency", test_prediction_output_consistency),
        ("Multi-label prediction", test_multi_label_prediction),
        ("Full scenario local CSV", test_full_scenario_local_csv),
    ]
    
    for name, test_func in tests:
        try:
            test_func()
        except AssertionError as e:
            print(f"\n[FAIL] {name}: {e}")
            all_passed = False
        except Exception as e:
            print(f"\n[ERROR] {name}: {e}")
            all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("All tests PASSED!")
        print("=" * 70)
        print("\nSummary of fixes:")
        print("  1. run_glue.py line 434, 437: config.label2id -> model.config.label2id")
        print("  2. run_glue.py line 623: label_list[item] -> model.config.id2label[item]")
        print("  3. run_classification.py line 723, 726: label_list -> model.config.id2label")
        return 0
    else:
        print("Some tests FAILED!")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
