# Copyright 2024 The HuggingFace Team. All rights reserved.
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
Worker script for loss averaging tests.

Verifies that ``average_tokens_across_devices`` produces correct loss
compared to a single-GPU baseline.

When ``--run_both_averaging_modes`` is passed, the script runs training
twice (with and without averaging) in a single process launch, saving
``<output_dir>_broken_losses.json`` and ``<output_dir>_fixed_losses.json``.

Run via torchrun or accelerate launch.
"""

import argparse
import json

import datasets
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)


class StoreLossCallback(TrainerCallback):
    """Simple callback to store the loss."""

    def __init__(self):
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if "loss" in logs:
            self.losses.append(logs["loss"])


def run_distributed_training(training_args, loss_file):
    set_seed(42)
    model_name = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
    dataset_name = "wikitext"
    dataset_config = "wikitext-2-raw-v1"
    dataset = datasets.load_dataset(dataset_name, dataset_config, split="train[:50]")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples["text"], max_length=128, padding="max_length", truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

    loss_callback = StoreLossCallback()

    training_args.logging_steps = 1
    training_args.max_steps = 10
    training_args.learning_rate = 3e-4
    training_args.disable_tqdm = True
    training_args.dataloader_drop_last = True

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_dataset,
        callbacks=[loss_callback],
        data_collator=data_collator,
    )
    trainer.train()
    with open(loss_file, "w") as f:
        json.dump(loss_callback.losses, f)


if __name__ == "__main__":
    # Parse our custom flag first, pass the rest to HfArgumentParser.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--run_both_averaging_modes", action="store_true")
    custom_args, remaining = pre_parser.parse_known_args()

    hf_parser = HfArgumentParser((TrainingArguments,))
    (training_args,) = hf_parser.parse_args_into_dataclasses(remaining)

    if custom_args.run_both_averaging_modes:
        base_dir = training_args.output_dir
        # Run without averaging ("broken")
        training_args.average_tokens_across_devices = False
        training_args.output_dir = base_dir + "/broken"
        run_distributed_training(training_args, loss_file=base_dir + "/broken_losses.json")
        # Run with averaging ("fixed")
        training_args.average_tokens_across_devices = True
        training_args.output_dir = base_dir + "/fixed"
        run_distributed_training(training_args, loss_file=base_dir + "/fixed_losses.json")
    else:
        run_distributed_training(training_args, loss_file=training_args.output_dir + "_losses.json")
