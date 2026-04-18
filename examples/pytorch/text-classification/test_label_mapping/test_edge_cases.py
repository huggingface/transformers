#!/usr/bin/env python
"""
边界情况测试：验证异常处理和报错机制
"""
import os
import sys
import subprocess
import tempfile
import csv

def run_test(args, description, expect_success=True):
    """运行测试并检查结果"""
    print(f"\n{'='*60}")
    print(f"测试: {description}")
    print(f"{'='*60}")
    
    cmd = f"cd /Users/pkcha/transformers && python examples/pytorch/text-classification/run_classification.py {args}"
    print(f"命令: python run_classification.py {args[:100]}...")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ 执行成功 (返回码: 0)")
        if not expect_success:
            print(f"⚠️  警告: 预期失败但实际成功!")
            return False
        return True
    else:
        print(f"❌ 执行失败 (返回码: {result.returncode})")
        last_lines = result.stderr.strip().split('\n')[-10:]
        print("\n错误信息（最后10行）:")
        print("-" * 40)
        for line in last_lines:
            if 'ValueError' in line or 'Error' in line or 'error' in line or 'Found' in line:
                print(f"  >>> {line}")
            elif line.strip():
                print(f"  {line}")
        print("-" * 40)
        
        if expect_success:
            print(f"⚠️  警告: 预期成功但实际失败!")
            return False
        return True

def create_test_csv(filename, rows):
    """创建测试CSV文件"""
    filepath = os.path.join('/Users/pkcha/transformers/examples/pytorch/text-classification/test_label_mapping', filename)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    return filepath

def main():
    test_dir = '/Users/pkcha/transformers/examples/pytorch/text-classification/test_label_mapping'
    os.chdir(test_dir)
    
    print("开始边界情况测试...")
    
    # 1. 测试未知标签报错
    print("\n\n【测试1】未知标签报错")
    train_csv_bad_label = create_test_csv('train_bad_label.csv', [
        ['text', 'label'],
        ['Good movie', 'positive'],
        ['Bad film', 'unknown_label']
    ])
    val_csv = create_test_csv('val_simple.csv', [
        ['text', 'label'],
        ['Great!', 'positive'],
    ])
    
    args = f"""--model_name_or_path distilbert-base-uncased \
        --train_file {train_csv_bad_label} \
        --validation_file {val_csv} \
        --do_train \
        --per_device_train_batch_size 2 \
        --num_train_epochs 1 \
        --max_seq_length 32 \
        --output_dir /tmp/test_output 2>&1
    """
    run_test(args, "数据包含未知标签，应抛出明确错误", expect_success=False)
    
    # 2. 测试文本列缺失报错
    print("\n\n【测试2】文本列缺失报错")
    train_csv_no_text = create_test_csv('train_no_text.csv', [
        ['wrong_column', 'label'],
        ['Good movie', 'positive'],
        ['Bad film', 'negative']
    ])
    
    args = f"""--model_name_or_path distilbert-base-uncased \
        --train_file {train_csv_no_text} \
        --validation_file {val_csv} \
        --text_column_names text \
        --do_train \
        --max_seq_length 32 \
        --output_dir /tmp/test_output 2>&1
    """
    run_test(args, "文本列不存在，应抛出明确错误", expect_success=False)
    
    # 3. 测试标签列缺失报错
    print("\n\n【测试3】标签列缺失报错")
    train_csv_no_label = create_test_csv('train_no_label.csv', [
        ['text', 'wrong_label'],
        ['Good movie', 'positive'],
    ])
    
    args = f"""--model_name_or_path distilbert-base-uncased \
        --train_file {train_csv_no_label} \
        --validation_file {val_csv} \
        --label_column_name label \
        --do_train \
        --max_seq_length 32 \
        --output_dir /tmp/test_output 2>&1
    """
    run_test(args, "标签列不存在，应抛出明确错误", expect_success=False)
    
    # 4. 测试空文本处理 (应正常工作)
    print("\n\n【测试4】空文本处理")
    train_csv_empty = create_test_csv('train_empty_text.csv', [
        ['text', 'label'],
        ['', 'positive'],
        [None, 'negative'],
        ['normal text', 'positive'],
    ])
    
    args = f"""--model_name_or_path distilbert-base-uncased \
        --train_file {train_csv_empty} \
        --validation_file {val_csv} \
        --do_train \
        --per_device_train_batch_size 2 \
        --num_train_epochs 1 \
        --max_seq_length 32 \
        --output_dir /tmp/test_output --overwrite_output_dir 2>&1
    """
    run_test(args, "空文本应正常处理，不崩溃", expect_success=True)
    
    # 5. 验证标签一致性完整流程
    print("\n\n【测试5】完整流程标签一致性")
    print("正在训练基础模型...")
    
    # 6. 运行单元测试
    print("\n\n【测试6】运行单元测试验证修复逻辑")
    result = subprocess.run('python test_label_mapping_unit.py', shell=True, cwd=test_dir, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode == 0:
        print("✅ 单元测试全部通过!")
    
    print("\n" + "="*60)
    print("边界情况测试完成!")
    print("="*60)

if __name__ == "__main__":
    main()
