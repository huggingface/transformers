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
        self.model_name = "distilbert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        
        self.dataset = load_dataset("glue", "sst2", split="train[:100]")
        
        def tokenize_function(examples):
            return self.tokenizer(examples["sentence"], padding="max_length", truncation=True)
        
        self.tokenized_dataset = self.dataset.map(tokenize_function, batched=True)
        
    def test_eval_time_based(self):
        training_args = TrainingArguments(
            output_dir="./test_output",
            eval_strategy=IntervalStrategy.TIME,
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
        
        start_time = time.time()
        
        trainer.train()
        
        with open(f"{training_args.output_dir}/trainer_state.json", "r") as f:
            trainer_state = f.read()
            self.assertIn("eval_metrics", trainer_state)
            
    def test_save_time_based(self):
        import pdb; pdb.set_trace()
        training_args = TrainingArguments(
            output_dir="./test_output",
            eval_strategy=IntervalStrategy.NO,
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
        
        start_time = time.time()
        
        trainer.train()
        
        import os
        checkpoint_dirs = [d for d in os.listdir(training_args.output_dir) if d.startswith("checkpoint-")]
        self.assertGreater(len(checkpoint_dirs), 0)

if __name__ == "__main__":
    # unittest.main() 
    import pdb; pdb.set_trace()
    training_args = TrainingArguments(
        output_dir="./test_output",
        eval_strategy=IntervalStrategy.TIME,
        eval_minutes=1,
        save_strategy=IntervalStrategy.NO,
        per_device_train_batch_size=8,
        num_train_epochs=1,
        logging_steps=1,
    )