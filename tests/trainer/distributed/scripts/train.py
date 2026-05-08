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

"""Simple causal LM script for distributed tests (FSDP, DeepSpeed).

Uses a tiny Qwen2 model with synthetic data so tests run fast
and don't require downloading real datasets.

Supports --do_train (default) and --do_eval via TrainingArguments.

32 training samples are created; with per_device_train_batch_size=4
and 2 GPUs this gives 4 steps per epoch.
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


DTYPE_MAP = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def _pop_custom_arg(name):
    """Pop a custom --name value arg from sys.argv before HfArgumentParser sees it."""
    if name in sys.argv:
        idx = sys.argv.index(name)
        value = sys.argv[idx + 1]
        sys.argv.pop(idx)
        sys.argv.pop(idx)
        return value
    return None


def main():
    # Parse custom args (not TrainingArguments fields)
    model_name = _pop_custom_arg("--model_name") or "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
    loss_output_file = _pop_custom_arg("--loss_output_file")
    eval_output_file = _pop_custom_arg("--eval_output_file")
    model_dtype = _pop_custom_arg("--model_dtype")
    attn_impl = _pop_custom_arg("--attn_implementation")
    pad_to_multiple_of = _pop_custom_arg("--pad_to_multiple_of")

    parser = HfArgumentParser((TrainingArguments,))
    (training_args,) = parser.parse_args_into_dataclasses()

    # Default to training if neither --do_train nor --do_eval is set
    if not training_args.do_train and not training_args.do_eval:
        training_args.do_train = True

    # Auto-enable eval when an eval output file is requested
    if eval_output_file:
        training_args.do_eval = True

    torch_dtype = DTYPE_MAP[model_dtype] if model_dtype else None

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {}
    if torch_dtype:
        model_kwargs["torch_dtype"] = torch_dtype
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Synthetic dataset — 32 samples of tokenized text
    # With per_device_train_batch_size=4 and 2 GPUs this gives 4 steps per epoch.
    texts = [
        "The quick brown fox jumps over the lazy dog. " * 5,
        "A journey of a thousand miles begins with a single step. " * 5,
        "To be or not to be, that is the question. " * 5,
        "All that glitters is not gold, all that wanders is not lost. " * 5,
    ] * 8

    train_dataset = None
    eval_dataset = None
    if training_args.do_train:
        train_dataset = [tokenizer(text, max_length=128, truncation=True, padding="max_length") for text in texts]
    if training_args.do_eval:
        eval_dataset = [tokenizer(text, max_length=128, truncation=True, padding="max_length") for text in texts[:8]]

    collator_kwargs = {}
    if pad_to_multiple_of:
        collator_kwargs["pad_to_multiple_of"] = int(pad_to_multiple_of)

    training_args.disable_tqdm = True

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, **collator_kwargs),
    )

    if training_args.do_train:
        trainer.train()

    if training_args.do_eval:
        eval_metrics = trainer.evaluate()
        if eval_output_file and training_args.process_index == 0:
            with open(eval_output_file, "w") as f:
                json.dump(eval_metrics, f)

    # Save per-step losses for equivalence testing
    if training_args.do_train and loss_output_file and training_args.process_index == 0:
        losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]
        with open(loss_output_file, "w") as f:
            json.dump(losses, f)


if __name__ == "__main__":
    main()
