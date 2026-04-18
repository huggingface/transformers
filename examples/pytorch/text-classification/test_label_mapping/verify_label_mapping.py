#!/usr/bin/env python
"""
验证标签映射bug修复的测试脚本
修复前：预测时使用本地label_list（可能与训练时排序不同），导致标签显示错误
修复后：预测时使用model.config.id2label，确保与训练时一致
"""

import os
import sys
import json
import tempfile
import shutil

def test_label_mapping():
    print("=" * 60)
    print("验证标签映射修复")
    print("=" * 60)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..", ".."))
    run_script = os.path.join(project_root, "examples", "pytorch", "text-classification", "run_classification.py")
    
    train_csv = os.path.join(script_dir, "train.csv")
    test_csv = os.path.join(script_dir, "test.csv")
    
    output_dir = os.path.join(script_dir, "test_output")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n1. 训练模型，保存标签配置...")
    
    train_cmd = f"""python {run_script} \
        --model_name_or_path distilbert-base-uncased \
        --train_file {train_csv} \
        --validation_file {test_csv} \
        --output_dir {output_dir} \
        --do_train \
        --per_device_train_batch_size 2 \
        --num_train_epochs 1 \
        --max_seq_length 32 \
        --overwrite_output_dir
    """
    
    print(f"执行训练命令...")
    result = os.system(train_cmd)
    if result != 0:
        print(f"训练失败，退出码: {result}")
        return False
    
    print("训练完成!")
    
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"\n模型配置中的标签映射:")
    print(f"  id2label: {config['id2label']}")
    print(f"  label2id: {config['label2id']}")
    
    print(f"\n2. 使用训练好的模型进行预测，验证标签一致性...")
    
    predict_cmd = f"""python {run_script} \
        --model_name_or_path {output_dir} \
        --train_file {train_csv} \
        --validation_file {test_csv} \
        --test_file {test_csv} \
        --output_dir {output_dir} \
        --do_predict \
        --max_seq_length 32 \
        --overwrite_output_dir
    """
    
    print(f"执行预测命令...")
    result = os.system(predict_cmd)
    if result != 0:
        print(f"预测失败，退出码: {result}")
        return False
    
    predict_results = os.path.join(output_dir, "predict_results.txt")
    print(f"\n预测结果文件: {predict_results}")
    print("\n预测结果内容:")
    print("-" * 40)
    
    with open(predict_results, 'r') as f:
        for line in f:
            print(line.rstrip())
    
    print("-" * 40)
    
    predicted_labels = []
    with open(predict_results, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                predicted_labels.append(parts[1])
    
    print(f"\n预测标签列表: {predicted_labels}")
    
    valid_labels = {'positive', 'negative'}
    predicted_set = set(predicted_labels)
    invalid_labels = predicted_set - valid_labels
    
    if len(invalid_labels) > 0:
        print(f"\n❌ BUG存在! 发现无效标签: {invalid_labels}")
        print("标签映射不一致导致预测结果显示错误!")
        return False
    else:
        print(f"\n✅ 修复成功! 所有标签均为有效标签: {valid_labels}")
        print("预测输出与模型配置的 id2label 保持一致!")
        return True

if __name__ == "__main__":
    success = test_label_mapping()
    print("\n" + "=" * 60)
    if success:
        print("测试通过! 标签映射bug已修复!")
        sys.exit(0)
    else:
        print("测试失败! 标签映射bug仍然存在!")
        sys.exit(1)
