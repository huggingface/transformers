#!/usr/bin/env python3

"""
Resize token embeddings under FSDP.

Adds special tokens, resizes token embeddings, and optionally runs a short sanity
training loop to verify everything works when some parameters may be on the meta
device (FSDP).
"""

import argparse

import torch
from datasets import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_utils import is_fsdp_enabled


def create_sample_dataset(tokenizer: AutoTokenizer, num_samples: int = 100) -> Dataset:
    """Create a small dataset with occurrences of the added special tokens."""
    sample_texts = [
        "This is a sample text with <COMMIT_MESSAGE>fix: resolve issue</COMMIT_MESSAGE>",
        "Another example <FILE>main.py</FILE> with special tokens.",
        "Code changes: <ADDED>new function</ADDED> and <REMOVED>old code</REMOVED>",
    ] * (num_samples // 3 + 1)
    sample_texts = sample_texts[:num_samples]

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=128,
            padding="max_length",
            return_tensors="pt",
        )

    dataset = Dataset.from_dict({"text": sample_texts})
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    return tokenized_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Resize token embeddings under FSDP")
    parser.add_argument(
        "--model_name",
        type=str,
        default="hf-internal-testing/tiny-random-LlamaForCausalLM",
        help="Model name or path",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./fsdp_token_resize_output", help="Training output directory"
    )
    parser.add_argument("--test_resize_only", action="store_true", help="Only resize tokens and exit")
    args = parser.parse_args()

    print("FSDP Enabled:", is_fsdp_enabled())

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    print("Original vocab size:", model.config.vocab_size)

    # Add a small set of special tokens (triggers resize)
    special_tokens = [
        "<COMMIT_MESSAGE>",
        "</COMMIT_MESSAGE>",
        "<FILE>",
        "</FILE>",
        "<ADDED>",
        "</ADDED>",
        "<REMOVED>",
        "</REMOVED>",
    ]
    num_added = tokenizer.add_tokens(special_tokens)
    print("Added tokens:", num_added)

    # Resize token embeddings (FSDP meta tensors are handled by the library fix)
    print("Resizing token embeddings to:", len(tokenizer))
    model.resize_token_embeddings(len(tokenizer))
    print("New embedding size:", model.get_input_embeddings().num_embeddings)

    model.config.vocab_size = len(tokenizer)

    if args.test_resize_only:
        print("Resize completed.")
        return

    # Minimal training loop to ensure resized embeddings work end-to-end
    print("Preparing small training run...")
    train_dataset = create_sample_dataset(tokenizer, num_samples=50)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        max_steps=5,
        logging_steps=1,
        save_strategy="no",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    # Quick inference check with a special token
    print("Running a quick inference check...")
    inputs = tokenizer("This is a test <COMMIT_MESSAGE>", return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    print("Logits shape:", tuple(outputs.logits.shape))
    print("Done.")


if __name__ == "__main__":
    main()
