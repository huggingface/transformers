import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers.models.quantum_classical.modeling_quantum_classical import QuantumClassicalTransformer
from trainer import QuantumClassicalTrainer

# 示例数据集 | Example dataset
class SimpleTextDataset(Dataset):
    """简单文本数据集用于演示
    Simple text dataset for demonstration
    """
    def __init__(self, vocab_size, seq_len, num_samples=1000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        
        # 生成随机序列数据 | Generate random sequence data
        self.data = torch.randint(1, vocab_size, (num_samples, seq_len))
        # 使用下一个词作为标签 (简单的语言建模任务) | Use next word as label (simple language modeling task)
        self.labels = torch.cat([self.data[:, 1:], torch.ones((num_samples, 1), dtype=torch.long)], dim=1)
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        return {"input_ids": self.data[idx], "labels": self.labels[idx]}

def collate_fn(batch):
    """批处理函数
    Batch processing function
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_ids": input_ids, "labels": labels}

def main():
    # 超参数 | Hyperparameters
    vocab_size = 10000
    max_seq_len = 128
    batch_size = 16
    embed_dim = 256
    num_heads = 8
    num_layers = 4
    recursion_depth = 2
    
    # 创建数据集 | Create datasets
    train_dataset = SimpleTextDataset(vocab_size, max_seq_len, num_samples=2000)
    val_dataset = SimpleTextDataset(vocab_size, max_seq_len, num_samples=500)
    
    # 创建数据加载器 | Create data loaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_fn
    )
    
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    
    # 创建模型 (使用较小的模型进行演示) | Create model (using a smaller model for demonstration)
    model = QuantumClassicalTransformer(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        recursion_depth=recursion_depth
    )
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建训练器 | Create trainer
    trainer = QuantumClassicalTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        lr=1e-4,
        warmup_steps=200,
        entropy_loss_weight=0.01,
        coherence_loss_weight=0.005
    )
    
    # 训练模型 (仅训练几个轮次用于演示) | Train model (only train a few epochs for demonstration)
    print("开始训练量子经典同构Transformer模型...")
    trained_model = trainer.train(epochs=3, log_interval=20, eval_steps=100)
    
    # 使用训练好的模型进行推理 | Use the trained model for inference
    print("\n使用训练好的模型进行推理...")
    trained_model.eval()
    
    # 创建一个输入示例 | Create an input example
    input_ids = torch.randint(1, vocab_size, (1, max_seq_len))
    
    # 推理 | Inference
    with torch.no_grad():
        logits, knowledge_state = trained_model(input_ids)
        predictions = torch.argmax(logits, dim=-1)
    
    print(f"输入形状: {input_ids.shape}")
    print(f"输出形状: {predictions.shape}")
    print(f"知识状态形状: {knowledge_state.shape}")
    
    # 展示模型的量子-经典特性 | Display the quantum-classical properties of the model
    print("\n模型特性展示:")
    
    # 1. 递归处理能力 | Recursive processing capability
    print(f"递归深度: {recursion_depth}")
    
    # 2. 量子-经典界面交互 | Quantum-classical interface interaction
    interface_module = model.layers[0].qc_interface
    temp_value = interface_module.temperature.item()
    print(f"界面温度参数: {temp_value:.4f}")
    
    # 3. 熵-负熵动态平衡 | Entropy-negentropy dynamic balance
    entropy_reg = model.layers[0].entropy_regulator
    eta_value = entropy_reg.eta.item()
    print(f"熵调节系数: {eta_value:.4f}")
    
    # 4. 自我参照与意识涌现 | Self-reference and consciousness emergence
    observer_gate = model.layers[0].observer_gate.item()
    print(f"观察者门控系数: {observer_gate:.4f}")
    
    print("\n量子经典同构Transformer模型演示完成")

if __name__ == "__main__":
    main() 