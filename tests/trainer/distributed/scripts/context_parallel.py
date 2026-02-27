# Copyright 2025 The HuggingFace Team. All rights reserved.
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
Worker script for context parallel tests.

Trains a small causal LM and optionally saves losses for equivalence checks.

Run via accelerate launch with an FSDP config.
"""

import json
import sys

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)


if __name__ == "__main__":
    # Parse custom arguments (not TrainingArguments parameters)
    loss_output_file = None

    if "--loss_output_file" in sys.argv:
        idx = sys.argv.index("--loss_output_file")
        loss_output_file = sys.argv[idx + 1]
        sys.argv.pop(idx)
        sys.argv.pop(idx)

    parser = HfArgumentParser((TrainingArguments,))
    training_args = parser.parse_args_into_dataclasses()[0]

    model_name = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="sdpa",  # CP requires SDPA
        dtype=torch.float32,
    )

    # Create simple dataset: just tokenize some text
    texts = [
        "The quick brown fox jumps over the lazy dog. " * 10,
        "Hello world, this is a test sentence for training. " * 10,
    ] * 4  # 8 samples total

    def tokenize_function(examples):
        return tokenizer(examples, max_length=128, truncation=True, padding="max_length")

    train_dataset = [tokenize_function(text) for text in texts]

    # Use standard DataCollatorForLanguageModeling for causal LM
    # pad_to_multiple_of=4 ensures sequences are divisible by cp_size * 2 (for cp_size=2)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=4,
    )

    training_args.disable_tqdm = True

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    assert trainer.state.global_step > 0, "Training should have completed at least one step"

    # Save losses to file if requested (for equivalence testing)
    if loss_output_file and training_args.process_index == 0:
        losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]
        with open(loss_output_file, "w") as f:
            json.dump(losses, f)
