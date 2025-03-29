import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
import time
from transformers.models.quantum_classical.modeling_quantum_classical import QuantumClassicalTransformer

class QuantumClassicalTrainer:
    """量子经典同构模型训练器
    Quantum-Classical Isomorphic Model Trainer
    """
    def __init__(
        self, 
        model, 
        train_dataloader, 
        val_dataloader=None,
        lr=5e-5, 
        warmup_steps=2000, 
        max_grad_norm=1.0,
        entropy_loss_weight=0.01,
        coherence_loss_weight=0.005,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        
        # 优化器 | Optimizer
        self.optimizer = optim.AdamW(model.parameters(), lr=lr)
        
        # 学习率调度器 | Learning rate scheduler
        self.scheduler = self.get_cosine_schedule_with_warmup(
            self.optimizer, 
            warmup_steps,
            len(train_dataloader) * 10  # 假设训练10个epoch | Assume training for 10 epochs
        )
        
        # 交叉熵损失函数 | Cross entropy loss function
        self.ce_loss_fn = nn.CrossEntropyLoss()
        
        # 额外的损失权重 | Additional loss weights
        self.entropy_loss_weight = entropy_loss_weight
        self.coherence_loss_weight = coherence_loss_weight
        self.max_grad_norm = max_grad_norm
        
    def get_cosine_schedule_with_warmup(self, optimizer, warmup_steps, total_steps, min_lr=0.0):
        """余弦衰减学习率调度器（含预热）
        Cosine decay learning rate scheduler with warmup
        """
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
            
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
    def compute_entropy_loss(self, knowledge_state):
        """计算知识状态熵损失 - 鼓励知识积累
        Calculate knowledge state entropy loss - encourage knowledge accumulation
        """
        log_probs = torch.log_softmax(knowledge_state, dim=-1)
        entropy = -torch.sum(torch.exp(log_probs) * log_probs)
        return entropy
        
    def compute_quantum_coherence_loss(self, quantum_state):
        """计算量子相干性损失 - 保持量子特性
        Calculate quantum coherence loss - maintain quantum properties
        """
        # 使用L1范数测量相干性 | Use L1 norm to measure coherence
        density_matrix = torch.matmul(quantum_state, quantum_state.transpose(-2, -1))
        off_diagonals = density_matrix - torch.diag_embed(torch.diagonal(density_matrix, dim1=-2, dim2=-1))
        coherence = torch.norm(off_diagonals, p=1)
        # 我们希望最大化相干性，因此返回负值 | We want to maximize coherence, so return negative value
        return -coherence
        
    def train_step(self, batch):
        """单步训练
        Single training step
        """
        self.model.train()
        
        # 准备输入 | Prepare input
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        # 前向传播 | Forward pass
        logits, knowledge_state = self.model(input_ids)
        
        # 计算主损失（语言模型损失）| Calculate main loss (language model loss)
        ce_loss = self.ce_loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # 计算熵损失 - 鼓励知识积累 | Calculate entropy loss - encourage knowledge accumulation
        entropy_loss = self.compute_entropy_loss(knowledge_state)
        
        # 计算量子相干性损失 | Calculate quantum coherence loss
        coherence_loss = self.compute_quantum_coherence_loss(knowledge_state)
        
        # 总损失 | Total loss
        total_loss = ce_loss + \
                    self.entropy_loss_weight * entropy_loss + \
                    self.coherence_loss_weight * coherence_loss
        
        # 反向传播 | Backward propagation
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪 | Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        # 优化步骤 | Optimization step
        self.optimizer.step()
        self.scheduler.step()
        
        return {
            "loss": total_loss.item(),
            "ce_loss": ce_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "coherence_loss": coherence_loss.item(),
            "learning_rate": self.scheduler.get_last_lr()[0]
        }
        
    @torch.no_grad()
    def evaluate(self):
        """验证模型
        Evaluate model
        """
        if self.val_dataloader is None:
            return None
            
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        for batch in self.val_dataloader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            logits, _ = self.model(input_ids)
            loss = self.ce_loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
        return {"val_loss": total_loss / total_samples}
        
    def train(self, epochs, log_interval=100, eval_steps=500):
        """完整训练循环
        Complete training loop
        """
        start_time = time.time()
        total_steps = 0
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_steps = 0
            
            for step, batch in enumerate(self.train_dataloader):
                metrics = self.train_step(batch)
                epoch_loss += metrics["loss"]
                epoch_steps += 1
                total_steps += 1
                
                # 日志记录 | Log recording
                if step % log_interval == 0:
                    elapsed = time.time() - start_time
                    print(f"Epoch: {epoch+1}/{epochs} | "
                          f"Step: {step}/{len(self.train_dataloader)} | "
                          f"Loss: {metrics['loss']:.4f} | "
                          f"CE Loss: {metrics['ce_loss']:.4f} | "
                          f"Entropy Loss: {metrics['entropy_loss']:.4f} | "
                          f"Coherence Loss: {metrics['coherence_loss']:.4f} | "
                          f"LR: {metrics['learning_rate']:.6f} | "
                          f"Time: {elapsed:.2f}s")
                
                # 验证 | Validation
                if total_steps % eval_steps == 0 and self.val_dataloader is not None:
                    val_metrics = self.evaluate()
                    val_loss = val_metrics["val_loss"]
                    print(f"Validation Loss: {val_loss:.4f}")
                    
                    # 保存最佳模型 | Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(self.model.state_dict(), "best_qc_model.pth")
                        print(f"Saved new best model with val loss: {best_val_loss:.4f}")
            
            # 每轮结束后评估 | Evaluate after each epoch
            if self.val_dataloader is not None:
                val_metrics = self.evaluate()
                print(f"Epoch {epoch+1} completed. Validation Loss: {val_metrics['val_loss']:.4f}")
            
            # 每轮结束保存检查点 | Save checkpoint after each epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'loss': epoch_loss / epoch_steps,
            }, f"qc_model_epoch_{epoch+1}.pth")
            
        print(f"Training completed in {time.time() - start_time:.2f}s")
        return self.model

# 使用示例 | Usage example
def train_quantum_classical_model(train_dataloader, val_dataloader=None, vocab_size=50000, max_seq_len=512):
    """训练量子经典同构Transformer模型
    Train Quantum-Classical Isomorphic Transformer model
    """
    # 创建模型 | Create model
    model = QuantumClassicalTransformer(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        embed_dim=768,
        num_heads=12,
        num_layers=6,
        recursion_depth=3
    )
    
    # 创建训练器 | Create trainer
    trainer = QuantumClassicalTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        lr=5e-5,
        warmup_steps=2000,
        entropy_loss_weight=0.01,
        coherence_loss_weight=0.005
    )
    
    # 训练模型 | Train model
    trained_model = trainer.train(epochs=10)
    
    return trained_model 