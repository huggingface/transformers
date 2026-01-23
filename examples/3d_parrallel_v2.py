"""
this script is used to test training using DDP/TP/PP in the PR #29153  
"""

import os
import sys
import time
import argparse

import torch
import torch.distributed as dist
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
from torch.distributed.tensor.parallel import loss_parallel
from torch.distributed.tensor.parallel.fsdp import enable_2d_with_fsdp, enable_2d_with_fsdp_and_tp
from torch.distributed.tensor.parallel import (
    distribute_module,
    DeviceMesh,
    PairwiseParallel,
    SequenceParallel,
    prepare_module,
    tensor_parallel,
)

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from datasets import load_dataset
from accelerate.logging import get_logger
from accelerate.test_utils.training import TrainingArguments
from accelerate.utils import set_seed

import wandb

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple test training script.")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--with_tracking", action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    set_seed(args.seed)

    #safer: handle both DDP and non-DDP
    if dist.is_available() and dist.is_initialized():
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = 0

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-1M")
    tokenizer.pad_token = tokenizer.eos_token

    raw_datasets = load_dataset("roneneldan/TinyStories-1M")
    # much safer num_proc (avoid 60-proc deadlock on small machines)
    raw_datasets = raw_datasets.map(
        lambda samples: tokenizer(samples["text"]),
        batched=True,
        num_proc=min(8, os.cpu_count()),
    )

    model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M").to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_dataloader = DataLoader(
        raw_datasets["train"], batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    num_training_steps = args.num_train_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    if args.with_tracking and (not dist.is_initialized() or dist.get_rank() == 0):
        wandb.init(project="tiny-stories", config=vars(args))
        wandb.watch(model, log="all")

    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if step % 10 == 0 and (not dist.is_initialized() or dist.get_rank() == 0):
                logger.info(f"Epoch {epoch}, step {step}, loss {loss.item()}")
                if args.with_tracking:
                    wandb.log({"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]})

    # cvleanup only if distributed was initialised
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

    if args.with_tracking and (not dist.is_initialized() or dist.get_rank() == 0):
        wandb.finish()


if __name__ == "__main__":
    main()
