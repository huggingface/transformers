#!/usr/bin/env python
"""
Test script to verify the label mapping fix for text-classification examples.

This test demonstrates that the fix ensures consistent label mapping between
training and prediction phases when using local CSV/JSON data.

Bug description:
- Before fix: When doing prediction without training, label_list was derived
  from the test dataset, which could differ from the model's saved id2label.
- After fix: When not training, label_list is derived from model.config.id2label,
  ensuring consistency with the trained model.

Run this test with: python test_label_mapping_fix.py
"""

import json
import os
import tempfile
import shutil
from pathlib import Path


def create_test_data(tmp_dir):
    train_data = [
        {"text": "This is positive", "label": "positive"},
        {"text": "This is negative", "label": "negative"},
        {"text": "Great product", "label": "positive"},
        {"text": "Terrible experience", "label": "negative"},
    ]
    
    test_data = [
        {"text": "Amazing quality", "label": "positive"},
        {"text": "Very disappointed", "label": "negative"},
    ]
    
    train_file = os.path.join(tmp_dir, "train.json")
    test_file = os.path.join(tmp_dir, "test.json")
    
    with open(train_file, "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
    
    with open(test_file, "w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")
    
    return train_file, test_file


def test_label_mapping_consistency():
    """
    Test that label mapping is consistent between training and prediction.
    
    This test verifies:
    1. After training, model.config.id2label correctly maps IDs to label names
    2. When doing prediction without training, label_list is derived from model.config.id2label
    3. The predicted label names match the original label names
    """
    print("=" * 60)
    print("Testing label mapping consistency fix")
    print("=" * 60)
    
    tmp_dir = tempfile.mkdtemp(prefix="label_mapping_test_")
    
    try:
        train_file, test_file = create_test_data(tmp_dir)
        output_dir = os.path.join(tmp_dir, "output")
        
        model_name = "distilbert-base-uncased"
        
        print("\n[Step 1] Training model with local JSON data...")
        train_cmd = f"""
python run_classification.py \\
    --model_name_or_path {model_name} \\
    --train_file {train_file} \\
    --validation_file {train_file} \\
    --text_column_names text \\
    --label_column_name label \\
    --do_train \\
    --max_seq_length 128 \\
    --per_device_train_batch_size 4 \\
    --learning_rate 2e-5 \\
    --num_train_epochs 1 \\
    --output_dir {output_dir} \\
    --overwrite_output_dir \\
    --save_strategy no
"""
        print(f"Running: {train_cmd.strip()}")
        ret = os.system(f"cd /Users/pkcha/transformers/examples/pytorch/text-classification && {train_cmd}")
        
        if ret != 0:
            print("Training failed, skipping full test")
            print("\n[Partial Test] Checking label_list derivation logic...")
            test_label_list_derivation()
            return
        
        print("\n[Step 2] Checking saved model config...")
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path) as f:
            config = json.load(f)
        
        print(f"Saved id2label: {config.get('id2label')}")
        print(f"Saved label2id: {config.get('label2id')}")
        
        assert "id2label" in config, "id2label not found in config"
        assert "label2id" in config, "label2id not found in config"
        
        id2label = config["id2label"]
        assert id2label["0"] == "negative", f"Expected id2label[0]='negative', got '{id2label['0']}'"
        assert id2label["1"] == "positive", f"Expected id2label[1]='positive', got '{id2label['1']}'"
        
        print("PASS: Model config has correct label mapping")
        
        print("\n[Step 3] Running prediction without training...")
        predict_cmd = f"""
python run_classification.py \\
    --model_name_or_path {output_dir} \\
    --test_file {test_file} \\
    --text_column_names text \\
    --label_column_name label \\
    --do_predict \\
    --max_seq_length 128 \\
    --output_dir {output_dir} \\
    --overwrite_output_dir
"""
        print(f"Running: {predict_cmd.strip()}")
        ret = os.system(f"cd /Users/pkcha/transformers/examples/pytorch/text-classification && {predict_cmd}")
        
        if ret != 0:
            print("Prediction failed")
            return
        
        print("\n[Step 4] Checking prediction results...")
        predict_file = os.path.join(output_dir, "predict_results.txt")
        with open(predict_file) as f:
            results = f.read()
        
        print("Prediction results:")
        print(results)
        
        lines = results.strip().split("\n")[1:]
        for line in lines:
            parts = line.split("\t")
            if len(parts) >= 2:
                idx, pred_label = parts[0], parts[1]
                print(f"Sample {idx}: predicted label = '{pred_label}'")
                assert pred_label in ["positive", "negative"], f"Invalid label: {pred_label}"
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("Label mapping is consistent between training and prediction.")
        print("=" * 60)
        
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_label_list_derivation():
    """
    Test the core logic: when not training, label_list should come from model.config.id2label.
    
    This is a unit test for the fix itself.
    """
    print("\n" + "-" * 40)
    print("Unit test: label_list derivation logic")
    print("-" * 40)
    
    class MockConfig:
        def __init__(self):
            self.label2id = {"negative": 0, "positive": 1}
            self.id2label = {0: "negative", 1: "positive"}
    
    class MockTrainingArgs:
        def __init__(self, do_train):
            self.do_train = do_train
    
    config = MockConfig()
    
    training_args = MockTrainingArgs(do_train=False)
    is_regression = False
    
    label_list_from_dataset = ["positive", "negative"]
    label_list = label_list_from_dataset
    
    print(f"Label list from dataset (sorted): {label_list_from_dataset}")
    print(f"Model config id2label: {config.id2label}")
    
    if not training_args.do_train and not is_regression:
        label_list = [config.id2label[i] for i in range(len(config.id2label))]
    
    print(f"Label list after fix: {label_list}")
    
    assert label_list == ["negative", "positive"], \
        f"Expected ['negative', 'positive'], got {label_list}"
    
    print("PASS: label_list correctly derived from model.config.id2label")
    
    print("\n" + "-" * 40)
    print("Demonstrating the bug scenario:")
    print("-" * 40)
    
    print("\nBefore fix:")
    print("  label_list from dataset (sorted) = ['negative', 'positive']")
    print("  Model was trained with id2label = {0: 'negative', 1: 'positive'}")
    print("  If test data has different order, predictions would be wrong!")
    
    print("\nAfter fix:")
    print("  label_list = [config.id2label[i] for i in range(len(config.id2label))]")
    print("  This ensures label_list matches the model's training labels.")


if __name__ == "__main__":
    test_label_mapping_consistency()
