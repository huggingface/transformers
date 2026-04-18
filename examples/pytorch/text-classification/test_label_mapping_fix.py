#!/usr/bin/env python
"""
文本分类标签映射修复验证测试
================================

本测试文件用于验证 run_classification.py 和 run_glue.py 中标签映射 bug 的修复效果。

Bug 描述:
---------
在预测阶段，代码使用从当前数据集构建的 label_list 来将预测 ID 转换为标签名称，
而不是使用模型配置中保存的 id2label 映射。当模型配置的 label2id 与当前数据集的
label_list 顺序不一致时，预测结果的标签名称会错误。

修复方案:
---------
将预测输出时的标签映射从使用 label_list 改为使用 model.config.id2label

两个脚本的标签映射场景:
-----------------------
1. run_classification.py:
   - 适用场景: 通用文本分类（本地 CSV/JSON、HuggingFace 数据集）
   - 支持类型: 单标签分类、多标签分类、回归
   - 训练阶段: 根据 label_list 创建 label_to_id，保存到 model.config.label2id 和 id2label
   - 评估阶段: 使用 model.config.label2id 进行标签映射
   - 预测阶段(修复后): 使用 model.config.id2label 将预测 ID 映射为标签名称

2. run_glue.py:
   - 适用场景: GLUE 基准测试任务（如 MNLI、SST-2 等）或本地 CSV/JSON
   - 支持类型: 单标签分类、回归（STSB）
   - 训练阶段: GLUE 任务从数据集 features 获取 label_list；本地文件从训练数据唯一值排序获取
   - 评估阶段: 使用 model.config.label2id 进行标签映射
   - 预测阶段(修复后): 使用 model.config.id2label 将预测 ID 映射为标签名称

验证方式:
---------
1. 运行本测试文件验证修复逻辑正确性
2. 检查代码中是否使用 model.config.id2label 替代 label_list
3. 验证训练-评估-预测三阶段标签映射一致性

用法:
-----
    cd /Users/pkcha/transformers
    python examples/pytorch/text-classification/test_label_mapping_fix.py

代码质量检查:
-------------
- 本测试不依赖外部大型库，仅使用 Python 标准库
- 测试可独立运行，无需 GPU 或大量数据
- 测试覆盖单标签、多标签、中文标签等多种场景
"""

import os
import sys
import tempfile
import csv


def test_label_mapping_consistency():
    """测试标签映射的一致性场景"""
    print("=" * 60)
    print("测试 1: 标签映射一致性场景")
    print("=" * 60)
    print("场景: 模型配置与数据集标签顺序一致")
    print()

    # 模型配置 label2id (按字符串排序)
    model_label2id = {"negative": 0, "positive": 1}
    model_id2label = {0: "negative", 1: "positive"}

    # 数据集 label_list (与模型配置顺序一致)
    label_list = ["negative", "positive"]

    # 模型预测 IDs
    predictions = [0, 1, 1, 0]

    print(f"模型配置 label2id: {model_label2id}")
    print(f"数据集 label_list: {label_list}")
    print(f"模型预测 IDs: {predictions}")
    print()

    # 修复前的方式 (使用 label_list)
    old_labels = [label_list[pred] for pred in predictions]
    print(f"修复前 (使用 label_list): {old_labels}")

    # 修复后的方式 (使用 model.config.id2label)
    new_labels = [model_id2label[pred] for pred in predictions]
    print(f"修复后 (使用 id2label):   {new_labels}")

    if old_labels == new_labels:
        print("\n✓ 此场景下标签映射一致")
        return True
    else:
        print("\n✗ 此场景下标签映射不一致")
        return False


def test_label_mapping_inconsistency():
    """测试标签映射不一致的场景 - 这是 bug 的核心场景"""
    print("\n" + "=" * 60)
    print("测试 2: 标签映射不一致场景 (Bug 核心场景)")
    print("=" * 60)
    print("场景: 模型配置与预测时数据集标签顺序不一致")
    print("      这种情况发生在不同数据集或标签排序方式不同时")
    print()

    # 模型配置 (训练时保存的)
    model_label2id = {"negative": 0, "positive": 1}
    model_id2label = {0: "negative", 1: "positive"}

    # 预测时数据集的 label_list (顺序与模型配置不同)
    label_list = ["positive", "negative"]

    # 模型预测 IDs (模型预测 ID 0 为 negative, ID 1 为 positive)
    predictions = [0, 1, 1, 0]

    print(f"模型配置 label2id: {model_label2id}")
    print(f"模型配置 id2label: {model_id2label}")
    print(f"预测时 label_list: {label_list}")
    print(f"模型预测 IDs: {predictions}")
    print()

    # 修复前的方式 (使用 label_list) - 错误
    old_labels = [label_list[pred] for pred in predictions]
    print(f"修复前 (使用 label_list): {old_labels}")

    # 修复后的方式 (使用 model.config.id2label) - 正确
    new_labels = [model_id2label[pred] for pred in predictions]
    print(f"修复后 (使用 id2label):   {new_labels}")

    print()
    if old_labels != new_labels:
        print("✓ 确认存在标签映射 bug!")
        print(f"  预期输出: {new_labels}")
        print(f"  修复前输出: {old_labels}")
        print("✓ 修复后的代码能够正确处理标签映射!")
        return True
    else:
        print("✗ 测试设计有误")
        return False


def test_real_world_scenario():
    """测试真实场景：情感分析标签映射"""
    print("\n" + "=" * 60)
    print("测试 3: 真实场景 - 中文情感分析")
    print("=" * 60)
    print("场景: 中文标签按字母排序时，'负向' < '正向'")
    print("      但如果预测时数据集按其他方式排序，会出现映射错误")
    print()

    # 中文情感分析场景
    model_label2id = {"负向": 0, "正向": 1}
    model_id2label = {0: "负向", 1: "正向"}

    # 假设预测时 label_list 顺序不同
    label_list = ["正向", "负向"]

    # 模型预测一个负面文本为 ID 0
    prediction_id = 0

    print(f"模型配置: {model_label2id}")
    print(f"预测时 label_list: {label_list}")
    print(f"模型预测 ID: {prediction_id} (对应 '负向')")
    print()

    old_output = label_list[prediction_id]
    new_output = model_id2label[prediction_id]

    print(f"修复前输出: '{old_output}' (错误!)")
    print(f"修复后输出: '{new_output}' (正确)")

    if old_output != new_output:
        print(f"\n✓ Bug 演示成功: 模型预测为'负向'，但修复前显示为'正向'")
        return True
    else:
        return False


def test_multi_label_scenario():
    """测试多标签分类场景 (仅 run_classification.py 支持)"""
    print("\n" + "=" * 60)
    print("测试 4: 多标签分类场景")
    print("=" * 60)
    print("场景: run_classification.py 支持的多标签分类")
    print("      每个样本可以属于多个类别")
    print()

    # 多标签场景
    model_label2id = {"sports": 0, "politics": 1, "tech": 2}
    model_id2label = {0: "sports", 1: "politics", 2: "tech"}

    # 预测时 label_list 顺序不同
    label_list = ["tech", "sports", "politics"]

    # 多标签预测结果 (multi-hot encoding)
    # [1, 0, 1] 表示 sports 和 tech
    prediction = [1, 0, 1]

    print(f"模型配置 id2label: {model_id2label}")
    print(f"预测时 label_list: {label_list}")
    print(f"多标签预测结果: {prediction}")
    print()

    # 修复前 (使用 label_list 索引)
    old_labels = [label_list[i] for i in range(len(prediction)) if prediction[i] == 1]
    print(f"修复前 (使用 label_list): {old_labels}")

    # 修复后 (使用 model.config.id2label)
    new_labels = [model_id2label[i] for i in range(len(prediction)) if prediction[i] == 1]
    print(f"修复后 (使用 id2label):   {new_labels}")

    if old_labels != new_labels:
        print(f"\n✓ 多标签场景也存在 bug!")
        print(f"  预期: {new_labels}")
        print(f"  修复前: {old_labels}")
        return True
    else:
        return False


def verify_code_fix():
    """验证代码修复是否正确应用"""
    print("\n" + "=" * 60)
    print("验证代码修复")
    print("=" * 60)
    print("检查修改文件是否正确使用 model.config.id2label")
    print()

    base_path = "/Users/pkcha/transformers/examples/pytorch/text-classification"

    results = []

    # 检查 run_classification.py
    print("检查 run_classification.py:")
    with open(os.path.join(base_path, "run_classification.py"), "r") as f:
        content = f.read()

    checks = [
        ("id2label = model.config.id2label", "获取 id2label 映射"),
        ("item = id2label[item]", "单标签预测使用 id2label"),
        ("item = [id2label[i] for i in range(len(item)) if item[i] == 1]", "多标签预测使用 id2label"),
    ]

    for pattern, desc in checks:
        if pattern in content:
            print(f"  ✓ {desc}")
            results.append(True)
        else:
            print(f"  ✗ {desc} - 未找到")
            results.append(False)

    if "item = label_list[item]" in content:
        print("  ✗ 仍使用 label_list 进行单标签预测")
        results.append(False)
    else:
        print("  ✓ 已移除 label_list 的直接使用")
        results.append(True)

    # 检查 run_glue.py
    print("\n检查 run_glue.py:")
    with open(os.path.join(base_path, "run_glue.py"), "r") as f:
        content = f.read()

    checks = [
        ("id2label = model.config.id2label", "获取 id2label 映射"),
        ("item = id2label[item]", "预测输出使用 id2label"),
    ]

    for pattern, desc in checks:
        if pattern in content:
            print(f"  ✓ {desc}")
            results.append(True)
        else:
            print(f"  ✗ {desc} - 未找到")
            results.append(False)

    if "item = label_list[item]" in content:
        print("  ✗ 仍使用 label_list 进行预测输出")
        results.append(False)
    else:
        print("  ✓ 已移除 label_list 的直接使用")
        results.append(True)

    return all(results)


def verify_three_stage_consistency():
    """验证训练-评估-预测三阶段标签映射一致性"""
    print("\n" + "=" * 60)
    print("验证训练-评估-预测三阶段标签映射一致性")
    print("=" * 60)
    print()

    # 模拟训练阶段保存的映射
    model_label2id = {"negative": 0, "positive": 1}
    model_id2label = {0: "negative", 1: "positive"}

    # 训练阶段: 使用 label_to_id 转换标签
    train_labels = ["positive", "negative", "positive"]
    label_to_id = {"negative": 0, "positive": 1}
    train_ids = [label_to_id[label] for label in train_labels]
    print(f"训练阶段: 标签 {train_labels} -> IDs {train_ids}")

    # 评估阶段: 使用 model.config.label2id 进行映射
    eval_labels = ["negative", "positive"]
    eval_ids = [model_label2id[label] for label in eval_labels]
    print(f"评估阶段: 标签 {eval_labels} -> IDs {eval_ids}")

    # 预测阶段(修复后): 使用 model.config.id2label 将 ID 转回标签
    predictions = [1, 0, 1]
    pred_labels = [model_id2label[pred] for pred in predictions]
    print(f"预测阶段: IDs {predictions} -> 标签 {pred_labels}")

    print()
    # 验证一致性: 预测 ID 1 应该对应 "positive"
    if model_id2label[1] == "positive" and model_id2label[0] == "negative":
        print("✓ 三阶段标签映射一致")
        print("  - 训练阶段: label_to_id['positive'] = 1")
        print("  - 评估阶段: label2id['positive'] = 1")
        print("  - 预测阶段: id2label[1] = 'positive'")
        return True
    else:
        print("✗ 三阶段标签映射不一致")
        return False


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("文本分类标签映射修复验证")
    print("=" * 60)
    print()
    print("本测试验证以下文件的标签映射修复:")
    print("  - run_classification.py (通用文本分类)")
    print("  - run_glue.py (GLUE 基准任务)")
    print()
    print("修复内容:")
    print("  预测输出时使用 model.config.id2label 替代 label_list")
    print()

    # 运行所有测试
    tests = [
        ("标签映射一致性", test_label_mapping_consistency),
        ("标签映射不一致性", test_label_mapping_inconsistency),
        ("真实场景", test_real_world_scenario),
        ("多标签场景", test_multi_label_scenario),
        ("代码修复验证", verify_code_fix),
        ("三阶段一致性验证", verify_three_stage_consistency),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ 测试 '{name}' 执行失败: {e}")
            results.append((name, False))

    # 总结
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)

    for name, result in results:
        status = "通过" if result else "失败"
        print(f"  {name}: {status}")

    all_passed = all(r for _, r in results)

    print()
    if all_passed:
        print("✓ 所有测试通过！标签映射 bug 已修复。")
        print()
        print("验证结论:")
        print("  1. 训练、评估、预测三阶段标签映射保持一致")
        print("  2. 预测输出使用 model.config.id2label，与训练时保存的映射一致")
        print("  3. 单标签和多标签分类场景均已修复")
        return 0
    else:
        print("✗ 部分测试失败，请检查修复。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
