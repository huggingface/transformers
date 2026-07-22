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

Supports --do_train (default) and --do_eval via TrainingArguments, and two
--data_mode values: ``synthetic`` (fixed phrases, no download) and ``sft_chat``
(an OpenAI-style chat JSONL at ``--data_path``, rendered via the chat template).
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


# Fallback for tokenizers that don't ship a chat template.
CHATML_FALLBACK_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
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


def _build_sft_chat_dataset(tokenizer, data_path, max_length=256):
    """Render and tokenize an OpenAI-style chat JSONL using the tokenizer's chat template."""
    if not tokenizer.chat_template:
        tokenizer.chat_template = CHATML_FALLBACK_TEMPLATE

    samples = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            rendered = tokenizer.apply_chat_template(
                record["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            samples.append(tokenizer(rendered, max_length=max_length, truncation=True, padding="max_length"))
    return samples


def main():
    # Parse custom args (not TrainingArguments fields)
    model_name = _pop_custom_arg("--model_name") or "axolotl-ai-co/tiny-qwen2-129m"
    loss_output_file = _pop_custom_arg("--loss_output_file")
    eval_output_file = _pop_custom_arg("--eval_output_file")
    model_dtype = _pop_custom_arg("--model_dtype")
    attn_impl = _pop_custom_arg("--attn_implementation")
    pad_to_multiple_of = _pop_custom_arg("--pad_to_multiple_of")
    data_mode = _pop_custom_arg("--data_mode") or "synthetic"
    data_path = _pop_custom_arg("--data_path")

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

    train_dataset = None
    eval_dataset = None
    if data_mode == "synthetic":
        texts = [
            "The quick brown fox jumps over the lazy dog. " * 5,
            "A journey of a thousand miles begins with a single step. " * 5,
            "To be or not to be, that is the question. " * 5,
            "All that glitters is not gold, all that wanders is not lost. " * 5,
        ] * 8
        if training_args.do_train:
            train_dataset = [tokenizer(text, max_length=128, truncation=True, padding="max_length") for text in texts]
        if training_args.do_eval:
            eval_dataset = [
                tokenizer(text, max_length=128, truncation=True, padding="max_length") for text in texts[:8]
            ]
    elif data_mode == "sft_chat":
        if not data_path:
            raise ValueError("--data_path is required when --data_mode sft_chat")
        samples = _build_sft_chat_dataset(tokenizer, data_path)
        if training_args.do_train:
            train_dataset = samples
        if training_args.do_eval:
            eval_dataset = samples[: max(1, len(samples) // 4)]
    else:
        raise ValueError(f"Unknown --data_mode {data_mode!r}")

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
