import unittest

from datasets import load_dataset

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    IntervalStrategy,
    Trainer,
    TrainingArguments,
)


class TestTimeBasedStrategy(unittest.TestCase):
    def setUp(self):
        self.model_name = "distilbert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2, device_map="auto"
        )

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
            logging_strategy=IntervalStrategy.NO,
            per_device_train_batch_size=8,
            num_train_epochs=200,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset,
            eval_dataset=self.tokenized_dataset,
        )

        trainer.train()

    def test_save_time_based(self):
        training_args = TrainingArguments(
            output_dir="./test_output",
            eval_strategy=IntervalStrategy.NO,
            save_strategy=IntervalStrategy.TIME,
            save_minutes=1,
            logging_strategy=IntervalStrategy.NO,
            per_device_train_batch_size=8,
            num_train_epochs=200,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset,
        )

        trainer.train()

    def test_logging_time_based(self):
        training_args = TrainingArguments(
            output_dir="./test_output",
            eval_strategy=IntervalStrategy.NO,
            save_strategy=IntervalStrategy.NO,
            logging_strategy=IntervalStrategy.TIME,
            logging_minutes=1,
            per_device_train_batch_size=8,
            num_train_epochs=200,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset,
        )

        trainer.train()
