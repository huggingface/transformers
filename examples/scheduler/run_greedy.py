#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Pre-train or fine-tune a causal language model using GreedyLR or Cosine scheduler.

Example usage:

    # Pre-train with GreedyLR (default):
    python run_greedy.py

    # Pre-train with cosine scheduler for comparison:
    python run_greedy.py --lr_scheduler_type cosine

    # Use a different model:
    python run_greedy.py --model_name_or_path Qwen/Qwen3-0.6B

    # Fine-tune from a pretrained checkpoint:
    python run_greedy.py --model_name_or_path meta-llama/Llama-3.2-1B --finetune
"""

import argparse
import logging

from datasets import load_dataset

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-train/fine-tune a causal LM with GreedyLR or Cosine scheduler")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Model identifier from huggingface.co/models or path to a local checkpoint",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="wikitext",
        help="The name of the dataset to use (via datasets library)",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default="wikitext-2-raw-v1",
        help="The configuration name of the dataset",
    )
    parser.add_argument("--lr_scheduler_type", type=str, default="greedy", choices=["greedy", "cosine"])
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Fine-tune from pretrained weights instead of training from scratch",
    )
    parser.add_argument("--block_size", type=int, default=512, help="Context length for tokenization")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.finetune:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    else:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        model = AutoModelForCausalLM.from_config(config)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {args.model_name_or_path} ({param_count / 1e6:.1f}M parameters)")

    raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.block_size)

    tokenized_datasets = raw_datasets.map(
        tokenize_function, batched=True, remove_columns=raw_datasets["train"].column_names
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    scheduler_kwargs = {}
    if args.lr_scheduler_type == "greedy":
        scheduler_kwargs = {"patience": 10, "factor": 0.99, "min_lr": 1e-5}

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        lr_scheduler_kwargs=scheduler_kwargs,
        max_steps=args.max_steps,
        warmup_steps=0 if args.lr_scheduler_type == "greedy" else 100,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=500,
        logging_steps=10,
        bf16=True,
        report_to="tensorboard",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets.get("validation", tokenized_datasets.get("test")),
        data_collator=data_collator,
    )

    logger.info(f"Starting training with {args.lr_scheduler_type} scheduler")
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
