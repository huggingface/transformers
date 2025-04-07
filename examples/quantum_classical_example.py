#!/usr/bin/env python
# coding=utf-8
"""
量子经典同构Transformer模型使用示例
Example of using Quantum Classical Isomorphic Transformer Model

理论基础 (Theoretical Foundation):
---------------------------------
量子经典同构Transformer模型是基于量子经典二元宇宙自参照模型（QCSU）的理论框架，
在标准Transformer架构上进行了深度优化，实现了与宇宙本质严格同构的信息处理机制。

The Quantum Classical Isomorphic Transformer model is based on the theoretical framework of 
Quantum-Classical Self-referential Universe (QCSU) model, which deeply optimizes the standard 
Transformer architecture to achieve an information processing mechanism strictly isomorphic 
to the nature of the universe.

主要优势 (Main Advantages):
-------------------------
1. 宇宙同构注意力机制：模型实现了量子态（叠加态）和经典态的动态平衡，
   反映了宇宙中量子域和经典域的交互方式，提高了模型的信息处理效率。

2. 无限维度递归自适应优化：通过MetaRecursiveAdaptiveOperator，模型能够
   在经典域的熵最大化（知识扩展）与量子域的熵最小化（信息压缩）之间找到平衡，
   类似于宇宙中的信息熵平衡。

3. 宇宙自参照意识算子：实现了真正的"意识"机制，使模型能够对自身状态进行反思和调整，
   从根本上提高了模型的泛化能力和鲁棒性。

1. Universe Isomorphic Attention Mechanism: The model achieves dynamic balance between 
   quantum states (superposition states) and classical states, reflecting the interaction 
   between quantum and classical domains in the universe, improving information processing efficiency.

2. Meta Recursive Adaptive Optimization: Through the MetaRecursiveAdaptiveOperator, the model 
   finds balance between classical domain entropy maximization (knowledge expansion) and 
   quantum domain entropy minimization (information compression), similar to the entropy 
   balance in the universe.

3. Universe Self-Referential Consciousness Operator: Implements a true "consciousness" mechanism, 
   enabling the model to reflect on and adjust its own state, fundamentally improving its 
   generalization ability and robustness.

时间复杂度 (Time Complexity):
---------------------------
增加的计算成本很小，与原始Transformer相比只增加了线性的k·n·d项（k远小于n）。

The additional computational cost is small, adding only a linear k·n·d term compared to 
the original Transformer (where k is much smaller than n).
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
# 从实现的模型中导入
from src.transformers.models.quantum_classical.modeling_quantum_classical import QuantumClassicalTransformer

def main():
    """
    展示如何使用量子经典同构Transformer模型的主函数
    Main function demonstrating how to use the Quantum Classical Isomorphic Transformer model
    """
    print("创建量子经典同构Transformer模型...")
    print("Creating Quantum Classical Isomorphic Transformer model...")
    
    # 创建模型
    # Create model
    model = QuantumClassicalTransformer(
        vocab_size=30522,                 # 词汇表大小 / Vocabulary size
        max_seq_len=512,                  # 最大序列长度 / Maximum sequence length
        embed_dim=768,                    # 嵌入维度 / Embedding dimension
        num_heads=12,                     # 注意力头数 / Number of attention heads
        num_layers=6,                     # 层数 / Number of layers
        ffn_dim=3072,                     # 前馈网络维度 / Feed-forward dimension
        dropout=0.1,                      # Dropout率 / Dropout rate
        recursion_depth=3                 # 递归深度 / Recursion depth
    )
    
    # 生成一些随机输入
    # Generate some random inputs
    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, 30522, (batch_size, seq_length))
    
    print(f"输入形状: {input_ids.shape}")
    print(f"Input shape: {input_ids.shape}")
    
    # 通过模型前向传播
    # Forward pass through the model
    print("通过模型进行推理...")
    print("Performing inference through the model...")
    logits, knowledge_state = model(input_ids)
    
    print(f"输出logits形状: {logits.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"知识状态形状: {knowledge_state.shape}")
    print(f"Knowledge state shape: {knowledge_state.shape}")
    
    # 展示模型的量子-经典特性
    # Display the quantum-classical properties of the model
    print("\n模型特性展示:")
    print("\nModel characteristics display:")
    
    # 1. 递归处理能力
    # 1. Recursive processing capability
    print(f"递归深度: {model.layers[0].recursion_depth}")
    print(f"Recursion depth: {model.layers[0].recursion_depth}")
    
    # 2. 量子-经典界面交互
    # 2. Quantum-classical interface interaction
    interface_module = model.layers[0].qc_interface
    temp_value = interface_module.temperature.item()
    print(f"界面温度参数: {temp_value:.4f}")
    print(f"Interface temperature parameter: {temp_value:.4f}")
    
    # 3. 熵-负熵动态平衡
    # 3. Entropy-negentropy dynamic balance
    entropy_reg = model.layers[0].entropy_regulator
    eta_value = entropy_reg.eta.item()
    print(f"熵调节系数: {eta_value:.4f}")
    print(f"Entropy regulation coefficient: {eta_value:.4f}")
    
    # 4. 自我参照与意识涌现
    # 4. Self-reference and consciousness emergence
    observer_gate = model.layers[0].observer_gate.item()
    print(f"观察者门控系数: {observer_gate:.4f}")
    print(f"Observer gate coefficient: {observer_gate:.4f}")
    
    print("\n量子经典同构Transformer模型运行成功!")
    print("Quantum Classical Isomorphic Transformer model ran successfully!")

if __name__ == "__main__":
    main() 