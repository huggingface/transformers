# Copyright 2026 The HuggingFace Team. All rights reserved.
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
Worker script: `Trainer` trains a model sharded at load time with `DistributedConfig(cp_size=N)` like a single device
trains the same weights on the same data.

The loss and the gradient norm of every step, and the number of input tokens seen, are written to
``--metrics_file`` so that a run with context parallelism can be compared with a single-device run of the same samples.
Launched via ``torchrun`` from ``test_trainer_distributed_transformers.py``.
"""

import argparse
import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist
from datasets import Dataset

from transformers import AutoModelForCausalLM, LlamaConfig, Trainer, TrainerCallback, TrainingArguments, set_seed
from transformers.distributed import DistributedConfig
from transformers.distributed.utils import get_distributed_backend
from transformers.testing_utils import torch_device


NUM_STEPS = 2
SEQ_LEN = 8
NUM_SAMPLES = 32


class StepMetricsCallback(TrainerCallback):
    """The loss and the gradient norm logged at the end of every optimizer step."""

    def __init__(self):
        self.losses = []
        self.grad_norms = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            self.losses.append(logs["loss"])
            self.grad_norms.append(logs["grad_norm"])


def _tiny_config():
    """A `LlamaConfig` small enough for CPU tests, with enough key/value heads for four `cp` x `tp` ranks."""
    return LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=99,
        attn_implementation="sdpa",
    )


def _dataset(config):
    """`NUM_SAMPLES` deterministic sequences, the same for every run of this script."""
    generator = torch.Generator().manual_seed(0)
    input_ids = torch.randint(0, config.vocab_size, (NUM_SAMPLES, SEQ_LEN), generator=generator).tolist()
    return Dataset.from_dict({"input_ids": input_ids, "labels": input_ids})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--metrics_file", required=True)
    parser.add_argument("--cp_size", type=int, default=1)
    parser.add_argument("--fsdp_size", type=int, default=1)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--no_average_tokens_across_devices", action="store_true")
    script_args = parser.parse_args()

    if script_args.cp_size > 1 or script_args.fsdp_size > 1 or script_args.tp_size > 1:
        dist.init_process_group(backend=get_distributed_backend(torch_device), timeout=timedelta(seconds=120))
    rank = dist.get_rank() if dist.is_initialized() else 0

    # every run starts from the same weights, whichever parallelism it then shards them with
    config = _tiny_config()
    model_dir = os.path.join(script_args.output_dir, "model")
    set_seed(0)
    if rank == 0:
        AutoModelForCausalLM.from_config(config).save_pretrained(model_dir)
    if dist.is_initialized():
        dist.barrier()

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        distributed_config=DistributedConfig(
            cp_size=script_args.cp_size, fsdp_size=script_args.fsdp_size, tp_size=script_args.tp_size
        ),
    )
    args = TrainingArguments(
        output_dir=script_args.output_dir,
        max_steps=NUM_STEPS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=1e-3,
        logging_steps=1,
        save_strategy="no",
        train_sampling_strategy="sequential",
        dataloader_drop_last=True,
        include_num_input_tokens_seen="all",
        average_tokens_across_devices=not script_args.no_average_tokens_across_devices,
        use_cpu=torch_device == "cpu",
        report_to="none",
        disable_tqdm=True,
    )
    dataset = _dataset(config)
    callback = StepMetricsCallback()
    trainer = Trainer(model=model, args=args, train_dataset=dataset, callbacks=[callback])
    trainer.train()

    assert len(callback.losses) == NUM_STEPS, f"rank {rank}: {callback.losses}"

    # the logits of a sequence are split over the cp ranks, so only the loss can be predicted
    if script_args.cp_size > 1:
        try:
            trainer.predict(dataset)
        except ValueError as error:
            assert "only supports evaluating the loss" in str(error), error
        else:
            raise AssertionError(f"rank {rank}: predicting logits with context parallelism did not raise")

    trainer.args.prediction_loss_only = True
    metrics = {
        "losses": callback.losses,
        "grad_norms": callback.grad_norms,
        "eval_loss": trainer.evaluate(dataset)["eval_loss"],
        "num_input_tokens_seen": trainer.state.num_input_tokens_seen,
    }
    if rank == 0:
        with open(script_args.metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f)
    trainer.end()


if __name__ == "__main__":
    main()
