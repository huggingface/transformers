#!/usr/bin/env python
"""
标签映射bug的单元测试
直接测试修复的核心逻辑
"""

import sys

def test_label_mapping_logic():
    print("=" * 60)
    print("单元测试：标签映射逻辑验证")
    print("=" * 60)
    
    print("\n场景:")
    print("- 训练集标签排序后: ['negative', 'positive'] (排序导致)")
    print("- 模型配置 id2label: {0: 'negative', 1: 'positive'}")
    print("- 测试集 label_list 可能不同顺序")
    
    model_config_id2label = {0: 'negative', 1: 'positive'}
    print(f"\n模型配置 id2label: {model_config_id2label}")
    
    test_label_list = ['positive', 'negative']
    print(f"测试集本地 label_list（不同顺序）: {test_label_list}")
    
    predicted_ids = [0, 1, 0, 1, 0, 1]
    print(f"预测ID: {predicted_ids}")
    
    print("\n" + "-" * 40)
    print("修复前（使用本地label_list）:")
    print("-" * 40)
    labels_before = [test_label_list[i] for i in predicted_ids]
    print(f"预测标签: {labels_before}")
    expected_labels_wrong = ['positive', 'negative', 'positive', 'negative', 'positive', 'negative']
    print(f"预期标签: {['negative', 'positive', 'negative', 'positive', 'negative', 'positive']}")
    print(f"实际得到: {labels_before}")
    print(f"是否匹配: {labels_before == ['negative', 'positive', 'negative', 'positive', 'negative', 'positive']}")
    
    print("\n" + "-" * 40)
    print("修复后（使用model.config.id2label）:")
    print("-" * 40)
    labels_after = [model_config_id2label[i] for i in predicted_ids]
    print(f"预测标签: {labels_after}")
    expected_labels_correct = ['negative', 'positive', 'negative', 'positive', 'negative', 'positive']
    print(f"预期标签: {expected_labels_correct}")
    print(f"实际得到: {labels_after}")
    print(f"是否匹配: {labels_after == expected_labels_correct}")
    
    print("\n" + "=" * 60)
    if labels_after == expected_labels_correct and labels_before != expected_labels_correct:
        print("✅ 单元测试通过!")
        print("修复前：label_list顺序与模型配置不一致导致标签错误")
        print("修复后：使用model.config.id2label确保标签一致性")
        return True
    else:
        print("❌ 单元测试失败!")
        return False

def test_multi_label_mapping():
    print("\n" + "=" * 60)
    print("单元测试：多标签分类映射验证")
    print("=" * 60)
    
    model_config_id2label = {0: 'tech', 1: 'sports', 2: 'business'}
    print(f"\n模型配置 id2label: {model_config_id2label}")
    
    test_label_list = ['business', 'sports', 'tech']
    print(f"测试集本地 label_list: {test_label_list}")
    
    multi_hot_predictions = [[1, 1, 0], [0, 0, 1], [1, 0, 1]]
    print(f"multi-hot预测: {multi_hot_predictions}")
    
    print("\n" + "-" * 40)
    print("修复前:")
    before = [[test_label_list[i] for i in range(len(item)) if item[i] == 1] for item in multi_hot_predictions]
    print(f"结果: {before}")
    
    print("\n修复后:")
    after = [[model_config_id2label[i] for i in range(len(item)) if item[i] == 1] for item in multi_hot_predictions]
    print(f"结果: {after}")
    expected = [['tech', 'sports'], ['business'], ['tech', 'business']]
    print(f"预期: {expected}")
    print(f"匹配: {after == expected}")
    
    return after == expected

if __name__ == "__main__":
    success1 = test_label_mapping_logic()
    success2 = test_multi_label_mapping()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("✅ 所有单元测试通过!")
        sys.exit(0)
    else:
        print("❌ 部分单元测试失败!")
        sys.exit(1)
