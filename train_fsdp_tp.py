# torchrun --nproc_per_node=4 train_fsdp_tp.py

import os
from types import SimpleNamespace

import torch
import torch.distributed.checkpoint as dcp
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.distributed import DistributedConfig
from transformers.distributed.utils import init_device_mesh
from transformers.integrations.tensor_parallel import apply_tensor_parallel
from transformers.integrations.fsdp import apply_fully_shard_data_parallel

if __name__ == "__main__":

    # model_name = "meta-llama/Llama-3.2-1B"
    model_name = "Qwen/Qwen3-0.6B"
    seq_len = 512
    num_steps, lr = 50, 3e-4
    batch_size = 4
    tp_degree = 2
    dp_degree = 2
    save_dir = "./checkpoints"

    torch.distributed.init_process_group(backend="nccl")
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # ── C4 dataset (same as torchtitan) ─────────────────────────────────
    from torchtitan.distributed import ParallelDims
    from torchtitan.distributed import utils as dist_utils
    from torchtitan.hf_datasets.text_datasets import build_text_dataloader
    from torchtitan.components.tokenizer import build_hf_tokenizer
    from torchtitan.config.job_config import JobConfig as TTJobConfig

    parallel_dims = ParallelDims(
        dp_shard=dp_degree, dp_replicate=1, tp=tp_degree,
        pp=1, ep=1, etp=1, cp=1,
        world_size=int(os.environ["WORLD_SIZE"]),
    )
    parallel_dims.build_mesh()

    tt_tokenizer = build_hf_tokenizer(
        SimpleNamespace(
            model=SimpleNamespace(
                hf_assets_path=snapshot_download(model_name),
                name="transformers_modeling_backend",
                tokenizer_path="",
            )
        )
    )
    dp_rank = parallel_dims.get_mesh("fsdp").get_local_rank()
    dp_world_size = parallel_dims.get_mesh("fsdp").size()
    tt_job_config = TTJobConfig()
    tt_job_config.training.dataset = "c4"
    tt_job_config.training.dataset_path = None
    tt_job_config.training.local_batch_size = batch_size
    tt_job_config.training.seq_len = seq_len
    dataloader = build_text_dataloader(
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        tokenizer=tt_tokenizer,
        job_config=tt_job_config,
        infinite=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        distributed_config=DistributedConfig(tp_size=2, tp_plan="auto", fsdp_size=2, fsdp_plan="auto", enable_sequence_parallel=True),
        torch_dtype=torch.bfloat16,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    data_iterator = iter(dataloader)
    for step in range(num_steps):
        # torchtitan dataloader yields ({"input": input_ids}, labels)
        # both of shape (batch, seq_len) — already shifted, no padding.
        input_dict, labels = next(data_iterator)
        input_ids = input_dict["input"].to(f"cuda:{local_rank}")
        labels = labels.to(f"cuda:{local_rank}")

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