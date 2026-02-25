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

"""Simple causal LM training script for FSDP distributed tests.

Uses a tiny Qwen2 model with synthetic data so tests run fast
and don't require downloading real datasets.

128 training samples are created; with per_device_train_batch_size=4
and 2 GPUs this gives 16 steps per epoch.
"""

import json
import sys

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)


def main():
    model_name = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"

    # Parse --loss_output_file (not a TrainingArguments field)
    loss_output_file = None
    if "--loss_output_file" in sys.argv:
        idx = sys.argv.index("--loss_output_file")
        loss_output_file = sys.argv[idx + 1]
        sys.argv.pop(idx)
        sys.argv.pop(idx)

    parser = HfArgumentParser((TrainingArguments,))
    (training_args,) = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Synthetic dataset — 128 samples of tokenized text
    texts = [
        "The quick brown fox jumps over the lazy dog. " * 5,
        "A journey of a thousand miles begins with a single step. " * 5,
        "To be or not to be, that is the question. " * 5,
        "All that glitters is not gold, all that wanders is not lost. " * 5,
    ] * 32

    train_dataset = [tokenizer(text, max_length=128, truncation=True, padding="max_length") for text in texts]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()

    # Save per-step losses for equivalence testing
    if loss_output_file and training_args.process_index == 0:
        losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]
        with open(loss_output_file, "w") as f:
            json.dump(losses, f)


if __name__ == "__main__":
    main()
