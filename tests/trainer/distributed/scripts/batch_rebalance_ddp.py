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
Worker script for the BatchRebalanceSampler DDP end-to-end test.

Runs under real DDP (launched by ``accelerate launch``/``torchrun``), trains a tiny model for
two epochs with ``train_sampling_strategy="batch_rebalance"``, and verifies the full stack:

  1. ``accelerator.prepare`` wraps the batch_sampler in a real ``BatchSamplerShard`` whose
     inner ``.batch_sampler`` is our ``BatchRebalanceSampler``, and the Trainer neutralises
     that wrapper to a passthrough (``num_processes == 1``) -- no double sharding.
  2. Each epoch covers every dataset sample across ranks, with no duplicates
     (``drop_last=True`` on an exactly-divisible dataset).
  3. Per-epoch reshuffling works end to end: the training loop forwards ``set_epoch`` to the
     wrapped sampler (accelerate's ``DataLoaderShard.set_epoch`` does not reach it on its own),
     so the second epoch sees a different sample order than the first.

Writes ``<output_dir>/result.json``: ``{"ok": bool, ...}``.
"""

import argparse
import json
import os

import torch
from torch import nn
from torch.utils.data import Dataset

from transformers import Trainer, TrainingArguments
from transformers.trainer_pt_utils import BatchRebalanceSampler


SEEN = []  # one entry per forward pass = one micro-batch, as a list of dataset indices


class _RecordingModel(nn.Module):
    """Minimal model: records the dataset indices of every micro-batch it trains on."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(1, 1)

    def forward(self, input_ids=None, labels=None):
        SEEN.append(input_ids.tolist())
        # meaningless but graph-connected loss so backward() works
        return {"loss": input_ids.float().sum() * self.lin.weight.sum()}


class _IndexDataset(Dataset):
    """Every token equals the dataset index (first token = marker) and the length varies with
    the index so the length-based rebalancing is meaningfully exercised. Items are precomputed
    so ``__getitem__`` raises ``IndexError`` past the end -- the length-inference loop in
    ``Trainer._get_train_sampler`` iterates via the ``__getitem__`` protocol and relies on that
    to terminate."""

    def __init__(self, n):
        self.items = [[i] * (1 + (i % 8)) for i in range(n)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return {"input_ids": self.items[i]}


def _collate(features):
    return {"input_ids": torch.tensor([f["input_ids"][0] for f in features], dtype=torch.long)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    cli = parser.parse_args()

    # Config divides exactly so drop_last=True yields clean, duplication-free coverage per
    # epoch: eff_bs = train_batch_size * grad_accum * world_size = (per_device*1) * 2 * W.
    per_device_train_batch_size = 2
    grad_accum = 2
    num_epochs = 2
    training_args = TrainingArguments(
        output_dir=cli.output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=grad_accum,
        train_sampling_strategy="batch_rebalance",
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        num_train_epochs=num_epochs,
        report_to=[],
        logging_strategy="no",
        save_strategy="no",
        disable_tqdm=True,
    )
    world_size = max(1, training_args.world_size)
    eff_bs = per_device_train_batch_size * grad_accum * world_size  # 4 * W
    n = 2 * eff_bs  # two global batches per epoch, no remainder

    trainer = Trainer(
        model=_RecordingModel(),
        args=training_args,
        train_dataset=_IndexDataset(n),
        data_collator=_collate,
    )
    trainer.train()

    # The dataloader actually used by the training loop:
    dataloader = trainer.callback_handler.train_dataloader
    wrapper = getattr(dataloader, "batch_sampler", None)
    inner = getattr(wrapper, "batch_sampler", None)

    # Per-rank sequence of micro-batches: num_epochs * (steps_per_epoch * grad_accum) forwards.
    n_fwd = len(SEEN)
    per_epoch_fwd = n_fwd // num_epochs
    epoch_sequences = [
        [idx for mb in SEEN[e * per_epoch_fwd : (e + 1) * per_epoch_fwd] for idx in mb] for e in range(num_epochs)
    ]

    local = {"n_forward": n_fwd, "epochs": epoch_sequences}
    gathered = [None] * world_size
    if world_size > 1:
        torch.distributed.all_gather_object(gathered, local)
    else:
        gathered[0] = local

    if not trainer.accelerator.is_main_process:
        return

    expected = list(range(n))
    checks = {
        # accelerate produced a real wrapper around our sampler and the Trainer made it a
        # passthrough (defeats double sharding)
        "wrapper_is_real_batchsamplershard": type(wrapper).__name__ == "BatchSamplerShard",
        "inner_is_batch_rebalance_sampler": isinstance(inner, BatchRebalanceSampler),
        "wrapper_patched_to_passthrough": getattr(wrapper, "num_processes", None) == 1,
        # every epoch covers every sample across ranks, exactly once
        "each_epoch_covers_all": all(
            sorted(idx for g in gathered for idx in g["epochs"][e]) == expected for e in range(num_epochs)
        ),
        "inner_epoch_advanced": getattr(inner, "epoch", None) == num_epochs - 1,
        "epochs_reshuffled": all(
            g["epochs"][e] != g["epochs"][e + 1] for g in gathered for e in range(num_epochs - 1)
        ),
    }
    result = {
        "ok": all(checks.values()),
        "world_size": world_size,
        "n": n,
        "effective_batch_size": eff_bs,
        "num_epochs": num_epochs,
        "inner_epoch": getattr(inner, "epoch", None),
        "checks": checks,
        "per_rank_epoch_orders": [g["epochs"] for g in gathered],
    }
    os.makedirs(cli.output_dir, exist_ok=True)
    with open(os.path.join(cli.output_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    # Visible summary line -- the distributed test harness requires the launched subprocess to
    # produce some output (it otherwise treats silence as a failure).
    print(f"batch_rebalance_ddp: {json.dumps(checks, sort_keys=True)}")


if __name__ == "__main__":
    main()
