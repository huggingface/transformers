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
""":
This script is used to test training a model using Tensor Parallelism and Data Parallelism.

Usage:
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_VISIBLE_DEVICES=5,6,7
TP_SIZE=2 DP_SIZE=2 torchrun --nproc_per_node=4 --rdzv_endpoint=localhost:29503 examples/3D_parallel.py
CP_SIZE=2 DP_SIZE=2 torchrun --nproc_per_node=4 examples/3D_parallel.py
CP_SIZE=2 TP_SIZE=2 torchrun --nproc_per_node=4 examples/3D_parallel.py
DP_SIZE=2 CP_SIZE=2 TP_SIZE=2 torchrun --nproc_per_node=8 examples/3D_parallel.py

TP_SIZE=1 CP_SIZE=4 torchrun --nproc_per_node=4 examples/3D_parallel.py
TP_SIZE=1 DP_SIZE=4 torchrun --nproc_per_node=4 examples/3D_parallel.py
TP_SIZE=4 DP_SIZE=1 torchrun --nproc_per_node=4 --rdzv_endpoint=localhost:29503 examples/3D_parallel.py
IGNORE_SANITY=1 CP_SIZE=1 TP_SIZE=1 DP_SIZE=1 torchrun --nproc_per_node=1 --rdzv_endpoint=localhost:29504 examples/3D_parallel.py
ocalhost:29504 test_train.py
"""

import logging
import os
from collections.abc import Iterable
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.optim as optim
import wandb
from datasets import load_dataset
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.experimental import context_parallel
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoModelForCausalLM, AutoTokenizer


# torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# from torch.distributed.tensor.experimental._attention import set_rotate_method

# set_rotate_method("alltoall")  # CP rotate shards using all-to-all


def main():
    tp_size = int(os.environ.get("TP_SIZE", "1"))
    dp_size = int(os.environ.get("DP_SIZE", "1"))
    cp_size = int(os.environ.get("CP_SIZE", "1"))  # Add CP size configuration
    sdpa_backend = SDPBackend.FLASH_ATTENTION  # For CP
    # sdpa_backend = SDPBackend.MATH # For CP
    global_batch_size = 8  # Desired global batch size
    seq_len = 1024  # Sequence length
    num_train_steps = 10000  # Number of training steps
    LR = 1e-5
    model_name = "HuggingFaceTB/SmolLM2-1.7B"
    # model_name = "unsloth/Llama-3.2-1B"

    CHECKPOINT_DIR = f"checkpoint_tp{tp_size}_dp{dp_size}_cp{cp_size}"

    # Initialize distributed environment
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

        assert world_size == tp_size * dp_size * cp_size, (
            f"World size ({world_size}) must equal TP size ({tp_size}) * DP size ({dp_size}) * CP size ({cp_size})"
        )

        mesh = torch.arange(world_size).reshape(dp_size, tp_size, cp_size)
        world_mesh = DeviceMesh(device_type="cuda", mesh=mesh, mesh_dim_names=("dp", "tp", "cp"))
        tp_mesh = world_mesh["tp"]
        dp_mesh = world_mesh["dp"]
        cp_mesh = world_mesh["cp"]
        world_mesh["dp", "cp"]._flatten(mesh_dim_name="dp_cp")
        logger.info(f"Created DeviceMesh: {world_mesh}")
        logger.info(
            f"Distributed setup - Rank: {rank}, World size: {world_size}, Local rank: {local_rank}, DP: {dp_mesh.get_local_rank()}, TP: {tp_mesh.get_local_rank()}, CP: {cp_mesh.get_local_rank()}"
        )

        if dist.get_rank() == 0:
            wandb.init(
                project="tp_dp_test",
                config={
                    "tp_size": tp_size,
                    "dp_size": dp_size,
                    "cp_size": cp_size,
                    "global_batch_size": global_batch_size,
                    "model_name": model_name,
                    "dataset": "roneneldan/TinyStories-1M",
                    "seq_len": seq_len,
                    "lr": LR,
                    "weight_decay": 0.1,
                },
                name=f"llama_tp{tp_size}_dp{dp_size}_cp{cp_size}"
                if model_name == "unsloth/Llama-3.2-1B"
                else f"tp{tp_size}_dp{dp_size}_cp{cp_size}",
            )
            logger.info("Wandb initialized.")
            # Log the current file to wandb
            wandb.save("test_train.py")

    # Load model and tokenizer
    logger.info(f"Loading model and tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_mesh=tp_mesh if dist.is_initialized() else None,
        tp_plan="auto",
        dtype=torch.bfloat16,
    )
    logger.info(f"Model loaded onto device mesh: {tp_mesh}")
    device = torch.device(f"cuda:{local_rank}")
    logger.info(f"Using device: {device} for non-model tensors")
    use_ddp = False
    if dist.is_initialized() and dp_mesh.size() > 1:
        model = FSDP(model, device_mesh=dp_mesh, sharding_strategy=ShardingStrategy.NO_SHARD)
        use_ddp = True
        pass

    model.train()

    logger.info("Loading TinyStories dataset...")
    raw_dataset = load_dataset("roneneldan/TinyStories", split="train[:1%]")  # Use 1% for faster testing

    def tokenize_function(examples):
        # Tokenize the text without padding
        tokenized_batch = tokenizer(
            examples["text"], padding=False, truncation=True, max_length=seq_len, return_tensors=None
        )
        # Set labels to be the same as input_ids for Causal LM
        tokenized_batch["labels"] = tokenized_batch["input_ids"].copy()
        return tokenized_batch

    tokenized_dataset = raw_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    logger.info(f"Dataset loaded and tokenized. Size: {len(tokenized_dataset)}")

    # Create packed sequences
    def create_packed_sequences(examples):
        # Flatten all sequences
        all_tokens = []
        for input_ids in examples["input_ids"]:
            all_tokens.extend(input_ids)

        # Split into sequences of seq_len + 1 (for input + label)
        num_sequences = len(all_tokens) // (seq_len + 1)
        packed_input_ids = []
        packed_labels = []

        for i in range(num_sequences):
            start_idx = i * (seq_len + 1)
            end_idx = start_idx + (seq_len + 1)
            # Get the full sequence
            full_sequence = all_tokens[start_idx:end_idx]
            # For input_ids, remove the last token
            packed_input_ids.append(full_sequence[:-1])
            # For labels, remove the first token
            packed_labels.append(full_sequence[1:])

        return {"input_ids": packed_input_ids, "labels": packed_labels}

    # Apply packing to the dataset
    packed_dataset = tokenized_dataset.map(
        create_packed_sequences,
        batched=True,
        remove_columns=tokenized_dataset.column_names,
        batch_size=1000,  # Process in batches for efficiency
        num_proc=60,
    )
    logger.info(f"Dataset packed. New size: {len(packed_dataset)}")

    # Shuffle the packed dataset
    packed_dataset = packed_dataset.shuffle(seed=42)
    logger.info("Packed dataset shuffled")

    # Calculate local batch size
    if dist.is_initialized():
        assert global_batch_size % dp_mesh.size() == 0, (
            f"Global batch size ({global_batch_size}) must be divisible by DP size ({dp_mesh.size()})"
        )
        local_batch_size = global_batch_size // dp_mesh.size()
    else:
        local_batch_size = global_batch_size

    logger.info(
        f"Global batch size: {global_batch_size}, DP size: {dp_size if dist.is_initialized() else 1}, Local batch size: {local_batch_size}"
    )

    # Simple collate function since sequences are already packed
    def collate_fn(batch):
        input_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
        labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}

    if dist.is_initialized():
        sampler = DistributedSampler(
            packed_dataset, num_replicas=dp_mesh.size(), rank=dp_mesh.get_local_rank(), shuffle=False
        )
    else:
        sampler = None

    dataloader = DataLoader(
        packed_dataset,
        batch_size=local_batch_size,
        sampler=sampler,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    logger.info(f"DataLoader created. Distributed: {dist.is_initialized()}")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)

    # Training loop
    logger.info(f"Starting training for {num_train_steps} steps...")
    model.train()
    step = 0
    while step < num_train_steps:
        for batch in dataloader:
            if step >= num_train_steps:
                break  # Exit loop if max steps reached

            # Move batch to appropriate device
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            # Add position_ids to batch before CP sharding
            batch_size = batch["input_ids"].shape[0]
            position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            batch["position_ids"] = position_ids
            from torch.distributed.tensor.experimental._attention import _cp_options

            _cp_options.enable_load_balance = False

            with sdpa_kernel(sdpa_backend):  # TODO: ideally move this to attention implementation
                cp_context = (
                    nullcontext()
                    if cp_mesh.size() == 1
                    else context_parallel(
                        cp_mesh,
                        buffers=[
                            batch["input_ids"],
                            batch["labels"],
                            batch["position_ids"],
                        ],
                        buffer_seq_dims=[1, 1, 1],
                    )
                )
                with cp_context:
                    # Pop labels from batch before model forward pass
                    labels = batch.pop("labels")
                    outputs = model(**batch)  # [mbs, seq_len/cp]
                    loss = outputs.loss
                    logits = outputs.logits

                    # Compute loss with shifted labels
                    loss = model.loss_function(
                        logits=logits, labels=None, shift_labels=labels, vocab_size=model.config.vocab_size
                    )
                    loss.backward()

                # all reduce grads across dp_cp if applicable
                all_reduce_grads(model, world_mesh, use_ddp=use_ddp)

                if hasattr(model, "clip_grad_norm_"):
                    gradnorm = model.clip_grad_norm_(max_norm=1.0, norm_type=2.0)  # TODO: fix reported gradnorm
                else:
                    # only works with FSDP's NO_SHARD otherwise we should use FSDP's clip_grad_norm_
                    assert len(list(model.parameters())) > 5, "No parameters found in model. Probably DDP bug.."
                    gradnorm = clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2.0, foreach=True)

                optimizer.step()
                # allreduce loss across cp_dp before logging
                if dist.is_initialized() and (cp_mesh.size() > 1 or dp_mesh.size() > 1):
                    dist.all_reduce(loss, group=world_mesh["dp_cp"].get_group(), op=dist.ReduceOp.AVG)
                current_loss = loss.item()

                # Log loss and gradnorm to wandb (only on rank 0 of dp group)
                if not dist.is_initialized() or dist.get_rank() == 0:
                    logger.info(
                        f"Step: {step} | GBS: {global_batch_size} | DP: {dp_mesh.size()} | TP: {tp_mesh.size()} | CP: {cp_mesh.size()} | Loss: {current_loss} | Gradnorm: {gradnorm} | lr: {LR}"
                    )
                    wandb.log(
                        {
                            "train/loss": current_loss,
                            "train/gradnorm": gradnorm,
                            "step": step,
                            "lr": LR,
                            "GBS": global_batch_size,
                        }
                    )

            step += 1  # Increment step count

    logger.info("Training loop finished.")

    # Save model using DCP (only if distributed)
    if dist.is_initialized():
        state_dict = {"app": AppState(model, optimizer)}
        dcp.save(
            state_dict=state_dict,
            checkpoint_id=CHECKPOINT_DIR,
        )
        logger.info(f"Saved checkpoint to {CHECKPOINT_DIR}")
    else:
        # Fallback to regular save for non-distributed case
        save_dir = "test_model_nondist"
        model.save_pretrained(save_dir, safe_serialization=False)
        tokenizer.save_pretrained(save_dir)  # Save tokenizer too
        logger.info(f"Saved model to {save_dir}")

    dist.destroy_process_group()
    logger.info("Cleaned up distributed process group")
    # Finish wandb run on rank 0
    if dist.get_rank() == 0:
        wandb.finish()
        logger.info("Wandb run finished.")


def all_reduce_grads(model, world_mesh, use_ddp):
    """All reduce gradients across dp_cp if applicable."""
    cp_mesh = world_mesh["cp"]
    if use_ddp:
        # DDP/FSDP takes care of syncing grads
        mesh = cp_mesh
    else:
        mesh = world_mesh["dp", "cp"]._flatten(mesh_dim_name="dp_cp")
    if dist.is_initialized() and mesh.size() > 1:
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Workaround for cross-mesh communication limitation with DTensor gradients
                if isinstance(param.grad, DTensor):
                    local_grad = param.grad.to_local()
                    # Ensure grad requires grad for inplace modification checks (might not be needed)
                    # local_grad = local_grad.detach().requires_grad_(True)
                    torch.distributed.all_reduce(local_grad, op=torch.distributed.ReduceOp.SUM, group=mesh.get_group())
                    local_grad = local_grad / mesh.size()
                    # Assign averaged grad back - need careful handling if DTensor structure is complex
                    # This simple assignment might work if the grad structure matches param structure
                    param.grad = DTensor.from_local(
                        local_grad, device_mesh=param.grad.device_mesh, placements=param.grad.placements
                    )
                else:
                    # Handle regular tensors if any exist (e.g. buffers not converted to DTensor)
                    torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.AVG, group=mesh.get_group())


class AppState(Stateful):
    """Wrapper for checkpointing the Application State including model and optimizer."""

    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        return {"model": model_state_dict, "optim": optimizer_state_dict}

    def load_state_dict(self, state_dict):
        set_state_dict(
            self.model, self.optimizer, model_state_dict=state_dict["model"], optim_state_dict=state_dict["optim"]
        )


def clip_grad_norm_(
    parameters: Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
) -> torch.Tensor:
    """
    Clip the gradient norm of an iterable of parameters.
    """
    # Filter out parameters with no gradients
    parameters = [p for p in parameters if p.grad is not None]
    assert len(parameters) > 0, "No parameters with gradients found"

    # Calculate total norm
    if norm_type == float("inf"):
        total_norm = max(p.grad.detach().abs().max() for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]), norm_type)

    # Convert DTensor to local tensor if needed
    if isinstance(total_norm, DTensor):
        total_norm = total_norm.full_tensor()

    # Clip gradients
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef)

    return total_norm


if __name__ == "__main__":
    main()
