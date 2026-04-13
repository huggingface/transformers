# torchrun --nproc_per_node=4 train_fsdp_tp.py

import argparse
import os

import torch
from datasets import load_dataset
from torch.distributed.tensor import DTensor
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.distributed import DistributedConfig
from transformers.distributed.utils import load_optimizer, save_optimizer

def build_packed_dataset(dataset_name, tokenizer, seq_len, dp_rank, dp_world_size):
    """Stream + tokenize + greedy-pack documents into fixed-length (input, label) windows."""
    ds = load_dataset(dataset_name, name="en", split="train", streaming=True)
    ds = ds.shard(num_shards=dp_world_size, index=dp_rank)
    buf, w = [], seq_len + 1

    def pack(batch):
        for t in batch["text"]:
            buf.extend(tokenizer(t)["input_ids"])
        ids, lbls = [], []
        while len(buf) >= w:
            ids.append(buf[:seq_len]); lbls.append(buf[1:w]); del buf[:w]
        return {"input_ids": ids, "labels": lbls}

    ds = ds.map(pack, batched=True, remove_columns=ds.column_names)
    return ds.with_format("torch")

def build_fixed_batches(dp_rank):
    """Load pre-generated fixed batches for a given DP rank."""
    return torch.load(f"fixed_batches_dp{dp_rank}.pt", weights_only=True)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--tp_size", type=int, default=0, help="Tensor parallel size (0 = disabled)")
    parser.add_argument("--fsdp_size", type=int, default=0, help="FSDP size (0 = disabled)")
    parser.add_argument("--enable_sp", action="store_true", help="Enable sequence parallelism")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fixed_batches", action="store_true", help="Use pre-generated fixed batches instead of C4")
    parser.add_argument("--resume_dir", type=str, default=None, help="Resume from this checkpoint directory")
    parser.add_argument("--start_step", type=int, default=0, help="Starting step number (for logging)")
    args = parser.parse_args()

    torch.distributed.init_process_group(backend="nccl")
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.manual_seed(args.seed)

    dc_kwargs = {}
    if args.tp_size > 0:
        dc_kwargs["tp_size"] = args.tp_size
        dc_kwargs["tp_plan"] = "auto"
    if args.fsdp_size > 0:
        dc_kwargs["fsdp_size"] = args.fsdp_size
        dc_kwargs["fsdp_plan"] = "auto"
    if args.enable_sp:
        dc_kwargs["enable_sequence_parallel"] = True
    distributed_config = DistributedConfig(**dc_kwargs)

    load_path = args.resume_dir if args.resume_dir else args.model_name
    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        distributed_config=distributed_config,
        torch_dtype=torch.bfloat16,
    )

    dp_rank = model.device_mesh["fsdp"].get_local_rank() if "fsdp" in model.device_mesh.mesh_dim_names else 0
    dp_size = model.device_mesh["fsdp"].size() if "fsdp" in model.device_mesh.mesh_dim_names else 1

    if args.fixed_batches:
        fixed = build_fixed_batches(dp_rank)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        dataset = build_packed_dataset("allenai/c4", tokenizer, args.seq_len, dp_rank, dp_size)
        dataloader = iter(DataLoader(dataset, batch_size=args.batch_size))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.resume_dir:
        load_optimizer(optimizer, os.path.join(args.resume_dir, "optimizer"))
        if rank == 0:
            print(f"Resumed optimizer from {args.resume_dir}")

    model.train()
    for step in range(args.start_step, args.start_step + args.num_steps):
        if args.fixed_batches:
            input_ids = fixed[step]["input_ids"].to(f"cuda:{local_rank}")
            labels = fixed[step]["labels"].to(f"cuda:{local_rank}")
        else:
            batch = next(dataloader)
            input_ids = batch["input_ids"].to(f"cuda:{local_rank}")
            labels = batch["labels"].to(f"cuda:{local_rank}")
        loss = model(input_ids, labels=labels).loss
        loss.backward()

        # Custom grad clip: convert DTensor grads to local to avoid mixed-mesh torch.stack
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        local_grads = [g.full_tensor() if isinstance(g, DTensor) else g for g in grads]
        total_norm = torch.nn.utils.get_total_norm(local_grads, norm_type=2.0)
        torch.nn.utils.clip_grads_with_norm_(grads, max_norm=1.0, total_norm=total_norm)
        optimizer.step()
        optimizer.zero_grad()

        if rank == 0:
            print(f"Step {step:>4d} | Loss: {loss.item():.4f} | Grad norm: {total_norm.item():.4f}")

    # Save model (HF format) and optimizer (DCP)
    model.save_pretrained(args.save_dir)
    save_optimizer(optimizer, os.path.join(args.save_dir, "optimizer"))

    if rank == 0:
        print(f"Saved to {args.save_dir}")

    torch.distributed.destroy_process_group()
