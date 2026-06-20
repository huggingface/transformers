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
"""Minimal Tensor Parallelism (TP) training example using Trainer.

Tensor Parallelism shards individual weight matrices across GPUs, so every
GPU participates in every forward pass.  This is distinct from Data
Parallelism (each GPU holds a full model replica) or FSDP (each GPU holds
a shard of the full model).  TP is useful when a single model layer is too
large to fit on one GPU, or to reduce per-GPU memory at the cost of
all-reduce communication within each layer.

Requirements
------------
    pip install transformers accelerate>=1.12.0 datasets

Usage
-----
    # 2-GPU Tensor Parallelism with torchrun
    torchrun --nproc_per_node=2 tp_training.py

    # Or with accelerate (equivalent)
    accelerate launch --num_processes=2 tp_training.py

Common pitfalls
---------------
* Do NOT pass ``device_map="auto"`` when using TP with Trainer — that flag
  is for inference-time device placement and will conflict with Trainer's
  distributed setup.
* Do NOT use ``init_empty_weights()`` here — it is for inference, not
  training.
* The number of processes (``--nproc_per_node``) must divide the model's
  hidden dimensions evenly.  Most standard models (Llama, Mistral,
  SmolLM2, …) already satisfy this for TP sizes 2, 4, or 8.
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def main():
    # ------------------------------------------------------------------ #
    #  Configuration                                                       #
    # ------------------------------------------------------------------ #
    # SmolLM2-1.7B is small enough to run quickly and is publicly
    # available.  Swap in any causal LM whose architecture supports
    # tp_plan (Llama, Mistral, Gemma, Qwen, …).
    model_id = "HuggingFaceTB/SmolLM2-1.7B"
    output_dir = "./tp_training_output"
    max_seq_length = 512
    per_device_batch_size = 2
    num_train_epochs = 1

    # ------------------------------------------------------------------ #
    #  Tokenizer                                                           #
    # ------------------------------------------------------------------ #
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Many causal LMs do not define a pad token; reuse eos instead.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------ #
    #  Model — loaded with tp_plan="auto"                                 #
    # ------------------------------------------------------------------ #
    # ``tp_plan="auto"`` is the single switch that enables Tensor
    # Parallelism.  It:
    #   1. Reads WORLD_SIZE from the environment (set by torchrun /
    #      accelerate launch) to determine the TP degree.
    #   2. Shards eligible weight matrices (attention projections, MLP
    #      weights, …) across all available GPUs via DTensor.
    #   3. Sets ``model.tp_size`` so that Trainer can auto-configure the
    #      accelerate ``ParallelismConfig``.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        tp_plan="auto",
        torch_dtype=torch.bfloat16,  # recommended for modern GPUs
    )

    # ------------------------------------------------------------------ #
    #  Dataset                                                             #
    # ------------------------------------------------------------------ #
    # TinyStories is a small, freely available text corpus — good for
    # quick iteration.  Replace with your own dataset as needed.
    raw_dataset = load_dataset("roneneldan/TinyStories", split="train[:1%]")

    def tokenize(examples):
        out = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        # For causal LM the labels are the same as input_ids (shifted
        # internally by the model).
        out["labels"] = out["input_ids"].copy()
        return out

    dataset = raw_dataset.map(
        tokenize,
        batched=True,
        remove_columns=raw_dataset.column_names,
    )

    # ------------------------------------------------------------------ #
    #  Training arguments                                                  #
    # ------------------------------------------------------------------ #
    # No special TP flags are needed here.  Trainer detects
    # ``model.tp_size > 1`` and automatically creates the required
    # accelerate ``ParallelismConfig``.
    #
    # If you need explicit control (e.g. combining TP with CP or DP), you
    # can pass a ``parallelism_config`` directly:
    #
    #   from accelerate import ParallelismConfig
    #   pc = ParallelismConfig(tp_size=2, dp_replicate_size=2)
    #   args = TrainingArguments(..., parallelism_config=pc)
    #
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_batch_size,
        num_train_epochs=num_train_epochs,
        bf16=True,
        logging_steps=10,
        # ``remove_unused_columns=False`` keeps the dataset columns in the
        # format expected by the data collator.
        remove_unused_columns=False,
        # Disable DDP's unused-parameter check — with TP some parameters
        # are intentionally not used on every rank.
        ddp_find_unused_parameters=False,
        # Disable saving to keep this example self-contained.
        save_strategy="no",
        report_to=[],
    )

    # ------------------------------------------------------------------ #
    #  Trainer                                                             #
    # ------------------------------------------------------------------ #
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        # DataCollatorForLanguageModeling handles padding and creates the
        # causal-LM labels automatically.
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print(f"Starting TP training | tp_size={model.tp_size} | model={model_id}")
    trainer.train()
    print("Training complete.")


if __name__ == "__main__":
    main()
