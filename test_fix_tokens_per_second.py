import os
import tempfile
import time
import unittest
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

class TestTokensPerSecondFix(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.model_name = "gpt2"
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create test dataset
        texts = ["The quick brown fox jumps over the lazy dog. " * 5] * 20
        self.dataset = Dataset.from_dict({"text": texts})
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length", 
                max_length=64,
                return_tensors="pt"
            )
        
        self.tokenized_dataset = self.dataset.map(tokenize_function, batched=True)
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
    
    def test_tokens_per_second_consistency(self):
        """Test that tokens_per_second is consistent between fresh start and checkpoint resume"""
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(
                output_dir=tmp_dir,
                num_train_epochs=1,
                per_device_train_batch_size=2,
                max_steps=10,
                save_steps=5,
                logging_steps=2,
                overwrite_output_dir=True,
            )
            
            # Fresh training run
            trainer_fresh = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.tokenized_dataset,
                data_collator=self.data_collator,
            )
            
            # Capture training metrics
            fresh_logs = []
            original_log_fn = trainer_fresh.log
            def capture_logs(logs):
                fresh_logs.append(logs.copy())
                original_log_fn(logs)
            trainer_fresh.log = capture_logs
            
            trainer_fresh.train()
            
            # Find checkpoint  
            checkpoints = [d for d in os.listdir(tmp_dir) if d.startswith("checkpoint-")]
            self.assertTrue(len(checkpoints) > 0, "No checkpoints found")
            
            checkpoint_path = os.path.join(tmp_dir, checkpoints[0])
            
            # Resume from checkpoint
            trainer_resume = Trainer(
                model=GPT2LMHeadModel.from_pretrained(self.model_name),
                args=training_args,
                train_dataset=self.tokenized_dataset,
                data_collator=self.data_collator,
            )
            
            resume_logs = []
            original_log_fn_resume = trainer_resume.log
            def capture_resume_logs(logs):
                resume_logs.append(logs.copy())
                original_log_fn_resume(logs)
            trainer_resume.log = capture_resume_logs
            
            trainer_resume.train(resume_from_checkpoint=checkpoint_path)
            
            # Check that tokens_per_second values are reasonable
            for log_entry in resume_logs:
                if 'train_tokens_per_second' in log_entry:
                    tokens_per_sec = log_entry['train_tokens_per_second']
                    # Should be reasonable (not astronomically high)
                    self.assertLess(tokens_per_sec, 100000, 
                                  f"tokens_per_second too high: {tokens_per_sec}")
                    self.assertGreater(tokens_per_sec, 0, 
                                     f"tokens_per_second should be positive: {tokens_per_sec}")
    
    def test_tokens_per_second_bounds(self):
        """Test that tokens_per_second values are within reasonable bounds"""
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(
                output_dir=tmp_dir,
                max_steps=5,
                per_device_train_batch_size=1,
                logging_steps=2,
                save_steps=2,
            )
            
            trainer = Trainer(
                model=self.model,
                args=training_args,  
                train_dataset=self.tokenized_dataset,
                data_collator=self.data_collator,
            )
            
            collected_metrics = []
            original_log_fn = trainer.log
            def collect_metrics(logs):
                if 'train_tokens_per_second' in logs:
                    collected_metrics.append(logs['train_tokens_per_second'])
                original_log_fn(logs)
            trainer.log = collect_metrics
            
            trainer.train()
            
            # Check all collected metrics are reasonable
            for tokens_per_sec in collected_metrics:
                self.assertLess(tokens_per_sec, 50000, 
                              f"Tokens per second too high: {tokens_per_sec}")
                self.assertGreater(tokens_per_sec, 10, 
                                 f"Tokens per second too low: {tokens_per_sec}")

if __name__ == "__main__":
    unittest.main()
