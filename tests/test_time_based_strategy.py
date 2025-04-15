import time
import unittest
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    IntervalStrategy,
)
from datasets import load_dataset

class TestTimeBasedStrategy(unittest.TestCase):
    def setUp(self):
        # 使用一个小型模型和数据集
        self.model_name = "distilbert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        
        # 加载一个小型数据集
        self.dataset = load_dataset("glue", "sst2", split="train[:100]")
        
        def tokenize_function(examples):
            return self.tokenizer(examples["sentence"], padding="max_length", truncation=True)
        
        self.tokenized_dataset = self.dataset.map(tokenize_function, batched=True)
        
    def test_eval_time_based(self):
        # 配置每1分钟评估一次
        training_args = TrainingArguments(
            output_dir="./test_output",
            evaluation_strategy=IntervalStrategy.TIME,
            eval_minutes=1,
            save_strategy=IntervalStrategy.NO,
            per_device_train_batch_size=8,
            num_train_epochs=1,
            logging_steps=1,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset,
            eval_dataset=self.tokenized_dataset,
        )
        
        # 记录开始时间
        start_time = time.time()
        
        # 训练一小段时间
        trainer.train()
        
        # 验证是否按时间间隔进行了评估
        # 由于评估是异步的，我们检查日志文件
        with open(f"{training_args.output_dir}/trainer_state.json", "r") as f:
            trainer_state = f.read()
            self.assertIn("eval_metrics", trainer_state)
            
    def test_save_time_based(self):
        # 配置每1分钟保存一次
        training_args = TrainingArguments(
            output_dir="./test_output",
            evaluation_strategy=IntervalStrategy.NO,
            save_strategy=IntervalStrategy.TIME,
            save_minutes=1,
            per_device_train_batch_size=8,
            num_train_epochs=1,
            logging_steps=1,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset,
        )
        
        # 记录开始时间
        start_time = time.time()
        
        # 训练一小段时间
        trainer.train()
        
        # 验证是否按时间间隔保存了检查点
        import os
        checkpoint_dirs = [d for d in os.listdir(training_args.output_dir) if d.startswith("checkpoint-")]
        self.assertGreater(len(checkpoint_dirs), 0)

if __name__ == "__main__":
    unittest.main() 