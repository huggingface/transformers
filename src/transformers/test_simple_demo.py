#!/usr/bin/env python3
"""
Simple test to demonstrate the DataParallel num_items_in_batch fix.
This will show the exact difference in behavior between old and new trainers.
"""

import sys
from pathlib import Path

import torch

import shutil
import tempfile

from datasets import Dataset

from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GPTNeoXForCausalLM,
    TrainingArguments,
)

# Import both trainer versions to compare
from transformers.trainer_new import Trainer as NewTrainer
from transformers.trainer_old import Trainer as OldTrainer


def create_simple_test():

    print(f"Found {torch.cuda.device_count()} different GPUs")

    # Setup model and data
    config = AutoConfig.from_pretrained("EleutherAI/pythia-14m")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-14m")
    tokenizer.pad_token = tokenizer.eos_token

    # Create simple test data
    test_texts = ["Hello world this is a test. " * 8] * 10
    dataset = Dataset.from_dict({"text": test_texts})

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=32, padding="max_length")

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Test both trainers
    for trainer_name, TrainerClass in [("OLD", OldTrainer), ("NEW", NewTrainer)]:
        print(f"\n{'-' * 40}")
        print(f"Testing {trainer_name} Trainer")
        print(f"{'-' * 40}")

        # Create fresh model
        torch.manual_seed(42)
        model = GPTNeoXForCausalLM(config=config)

        temp_dir = tempfile.mkdtemp(prefix=f"test_{trainer_name.lower()}_")

        try:
            training_args = TrainingArguments(
                output_dir=temp_dir,
                per_device_train_batch_size=3,
                gradient_accumulation_steps=5,  # This will show the difference clearly
                max_steps=2,
                logging_steps=1,
                save_steps=1000,
                learning_rate=1e-4,
                dataloader_drop_last=True,
                report_to=[],
                disable_tqdm=True,
            )

            trainer = TrainerClass(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                processing_class=tokenizer,
                data_collator=data_collator,
            )

            print(f"Starting {trainer_name} trainer...")
            trainer.train()

        except Exception as e:
            print(f"Error in {trainer_name} trainer: {e}")

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    create_simple_test()
