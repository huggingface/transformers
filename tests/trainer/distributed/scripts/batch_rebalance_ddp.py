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
Worker script for the BatchRebalanceSampler double-sharding regression test.

Runs under real DDP (launched by ``accelerate launch``/``torchrun``) and exercises the
actual ``Trainer.get_train_dataloader -> accelerator.prepare(DataLoader(...)) -> patch
prepared_dataloader.batch_sampler`` path -- instead of constructing accelerate's
``BatchSamplerShard`` by hand. Verifies:

  1. ``accelerator.prepare`` wraps the batch_sampler in a real ``BatchSamplerShard`` whose
     inner ``.batch_sampler`` is our ``BatchRebalanceSampler`` (the structure the Trainer's
     detection relies on),
  2. the Trainer neutralises that wrapper to a passthrough (``num_processes == 1``),
  3. iterating the prepared dataloader and gathering sample indices across all ranks covers
     every dataset sample exactly once -- no silent double-sharding drops and, for an
     exactly-divisible dataset with ``drop_last=True``, no padding duplicates either.

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


class _DummyModel(nn.Module):
    """Minimal model so Trainer can be constructed; never actually trained here."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(1, 1)

    def forward(self, *args, **kwargs):
        return {"loss": self.lin.weight.sum() * 0.0}


class _IndexDataset(Dataset):
    """Returns variable-length ``input_ids`` (length grows with the index) so
    BatchRebalanceSampler's length-based assignment is meaningfully exercised. Items are
    precomputed so ``__getitem__`` raises ``IndexError`` past the end -- the length-inference
    loop in ``Trainer._get_train_sampler`` iterates via the ``__getitem__`` protocol and relies
    on that to terminate."""

    def __init__(self, n):
        self.items = [list(range(1 + (i % 8))) for i in range(n)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return {"input_ids": self.items[i]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    cli = parser.parse_args()

    # Use a config that divides exactly so drop_last=True yields clean, duplication-free
    # coverage: eff_bs = train_batch_size * grad_accum * world_size = (per_device*1) * 2 * W.
    per_device_train_batch_size = 2
    grad_accum = 2
    training_args = TrainingArguments(
        output_dir=cli.output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=grad_accum,
        train_sampling_strategy="batch_rebalance",
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        report_to=[],
        disable_tqdm=True,
    )
    world_size = max(1, training_args.world_size)
    eff_bs = per_device_train_batch_size * grad_accum * world_size  # 4 * W
    n = 2 * eff_bs  # two global batches, no remainder

    trainer = Trainer(model=_DummyModel(), args=training_args, train_dataset=_IndexDataset(n))
    dataloader = trainer.get_train_dataloader()

    wrapper = getattr(dataloader, "batch_sampler", None)
    inner = getattr(wrapper, "batch_sampler", None)
    wrapper_type = type(wrapper).__name__
    patched_num_processes = getattr(wrapper, "num_processes", None)

    # Iterate the *real* prepared wrapper to collect the indices each rank actually sees.
    covered = []
    for index_list in dataloader.batch_sampler:
        covered.extend(int(i) for i in index_list)

    # Gather per-rank coverage so rank 0 can check the global union.
    gathered = [None] * world_size
    if world_size > 1:
        torch.distributed.all_gather_object(gathered, covered)
    else:
        gathered[0] = covered

    if not trainer.accelerator.is_main_process:
        return

    all_covered = sorted(idx for rank_cov in gathered for idx in rank_cov)
    expected = list(range(n))

    checks = {
        # accelerate produced a real wrapper around our sampler
        "wrapper_is_real_batchsamplershard": wrapper_type == "BatchSamplerShard",
        "inner_is_batch_rebalance_sampler": isinstance(inner, BatchRebalanceSampler),
        # the Trainer neutralised the wrapper to a passthrough (defeats double-sharding)
        "wrapper_patched_to_passthrough": patched_num_processes == 1,
        # exact global coverage: every sample seen once, none dropped, none duplicated
        "coverage_full_and_unique": all_covered == expected,
    }
    result = {
        "ok": all(checks.values()),
        "world_size": world_size,
        "n": n,
        "effective_batch_size": eff_bs,
        "wrapper_type": wrapper_type,
        "patched_num_processes": patched_num_processes,
        "per_rank_counts": [len(rc) for rc in gathered],
        "total_covered": len(all_covered),
        "checks": checks,
    }
    os.makedirs(cli.output_dir, exist_ok=True)
    with open(os.path.join(cli.output_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
