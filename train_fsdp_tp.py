# torchrun --nproc_per_node=4 train_fsdp_tp.py

import os

import torch
import torch.distributed.checkpoint as dcp
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.distributed import DistributedConfig
from torchtitan.distributed import utils as dist_utils #TODO(3outeille): add this to transformers.distributed

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


if __name__ == "__main__":

    model_name = "Qwen/Qwen3-0.6B"
    dataset_name = "allenai/c4"
    seq_len = 512
    num_steps, lr = 50, 3e-4
    batch_size = 4
    save_dir = "./checkpoints"

    torch.distributed.init_process_group(backend="nccl")
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    distributed_config = DistributedConfig(tp_size=2, tp_plan="auto", fsdp_size=2, fsdp_plan="auto", enable_sequence_parallel=True)
    # distributed_config = DistributedConfig(fsdp_size=4, fsdp_plan="auto")
    # distributed_config = DistributedConfig(tp_size=4, tp_plan="auto", enable_sequence_parallel=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        distributed_config=distributed_config,
        torch_dtype=torch.bfloat16,
    )

    dp_rank = model.device_mesh["fsdp"].get_local_rank() if "fsdp" in model.device_mesh.mesh_dim_names else 0
    dp_world_size = model.device_mesh["fsdp"].size() if "fsdp" in model.device_mesh.mesh_dim_names else 1
    

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = build_packed_dataset(dataset_name, tokenizer, seq_len, dp_rank, dp_world_size)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    data_iterator = iter(dataloader)
    for step in range(num_steps):
        batch = next(data_iterator)
        input_ids = batch["input_ids"].to(f"cuda:{local_rank}")
        labels = batch["labels"].to(f"cuda:{local_rank}")

        loss = model(input_ids, labels=labels).loss
        loss.backward()
        grad_norm = dist_utils.clip_grad_norm_(list(model.parameters()), max_norm=1.0, foreach=True)
        optimizer.step()
        optimizer.zero_grad()

        if rank == 0:
            print(f"Step {step:>4d} | Loss: {loss.item():.4f} | Grad norm: {grad_norm.item():.4f}")

    # Save model (HF format) and optimizer (DCP)
    model.save_pretrained(save_dir)
    dcp.save({"optimizer": optimizer.state_dict()}, checkpoint_id=os.path.join(save_dir, "optimizer"))

    if rank == 0:
        print(f"Saved to {save_dir}")

    torch.distributed.destroy_process_group()