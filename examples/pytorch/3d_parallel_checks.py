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
TP_SIZE=2 DP_SIZE=2 torchrun --nproc_per_node=4 --rdzv_endpoint=localhost:29503 test_train.py
CP_SIZE=2 DP_SIZE=2 torchrun --nproc_per_node=4 test_train.py
CP_SIZE=2 TP_SIZE=2 torchrun --nproc_per_node=4 test_train.py

TP_SIZE=1 CP_SIZE=4 torchrun --nproc_per_node=4 test_train.py
TP_SIZE=1 DP_SIZE=4 torchrun --nproc_per_node=4 test_train.py
TP_SIZE=4 DP_SIZE=1 torchrun --nproc_per_node=4 --rdzv_endpoint=localhost:29503 test_train.py
IGNORE_SANITY=1 CP_SIZE=1 TP_SIZE=1 DP_SIZE=1 torchrun --nproc_per_node=1 --rdzv_endpoint=l
ocalhost:29504 test_train.py
"""

import logging
import os
from collections.abc import Iterable
from contextlib import nullcontext
from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
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
from torch.utils.data import DataLoader, default_collate
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoModelForCausalLM, AutoTokenizer


ignore_sanity_checks = int(os.environ.get("IGNORE_SANITY", "0")) == 1
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

# set_rotate_method("alltoall")  # rotate shards using all-to-all


def main():
    tp_size = int(os.environ.get("TP_SIZE", "1"))
    dp_size = int(os.environ.get("DP_SIZE", "4"))
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
            logger.info(f"ignore_sanity_checks is set to: {ignore_sanity_checks}")
            logger.info("Wandb initialized.")
            # Log the current file to wandb
            wandb.save("test_train.py")

    else:
        logger.info("Running in non-distributed mode. DeviceMesh not applicable.")
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        wandb.init(
            project="tp_dp_test",
            config={
                "tp_size": 1,
                "dp_size": 1,
                "global_batch_size": global_batch_size,
                "model_name": model_name,
                "dataset": "roneneldan/TinyStories-1M",
                "seq_len": seq_len,
            },
            name="llama_tp1_dp1_nondist" if model_name == "unsloth/Llama-3.2-1B" else "tp1_dp1_nondist",
        )
        logger.info("Wandb initialized for non-distributed run.")

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

    if dist.is_initialized():
        assert model.config.num_key_value_heads % tp_mesh.size() == 0, (
            f"num_key_value_heads={model.config.num_key_value_heads} must be divisible by tp_size={tp_mesh.size()}"
        )
        device = torch.device(f"cuda:{local_rank}")
    else:
        model = model.to(device)

    logger.info(f"Using device: {device} for non-model tensors")
    use_ddp = False
    if dist.is_initialized() and dp_mesh.size() > 1:
        # FSDP1
        model = FSDP(model, device_mesh=dp_mesh, sharding_strategy=ShardingStrategy.NO_SHARD)
        # FSDP2
        # for transformer_block in model.model.layers:
        #     fully_shard(transformer_block, mesh=dp_mesh, reshard_after_forward=False)
        # fully_shard(model.model, mesh=dp_mesh, reshard_after_forward=False)
        # DDP
        # replicate(model, device_mesh=dp_mesh, bucket_cap_mb=100)
        # assert len(list(model.parameters()))>5, "No parameters found in model. Probably DDP/FSDP bug.." # TODO: we should be cautious abt using model.parameters()
        use_ddp = True

    model.train()
    assert len(list(model.parameters())) > 0, "No parameters found in model. Probably DDP bug.."
    assert len([p for p in model.parameters() if p.requires_grad]) > 0, (
        "No gradients found in model. Probably DDP bug.."
    )

    if dist.is_initialized() and not ignore_sanity_checks:
        # assert model is replicated across all dp
        for name, param in model.named_parameters():
            sanity_check_tensor_sync(param, dp_mesh)

        # assert model is different across tp (only for sharded params)
        for name, param in model.named_parameters():
            if isinstance(param, DTensor) and param.placements[0].is_shard():
                # Only check sharded parameters for non-sync across TP
                sanity_check_tensor_sync(param, tp_mesh, not_sync=True)
            elif isinstance(param, DTensor) and param.placements[0].is_replicate():
                # Replicated parameters should be the same across TP
                sanity_check_tensor_sync(param, tp_mesh)

        # assert model is replicated across cp
        for name, param in model.named_parameters():
            sanity_check_tensor_sync(param, cp_mesh)

    # Load and preprocess TinyStories dataset
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

            # Sanity checks for batch distribution (only if distributed)
            if dist.is_initialized() and not ignore_sanity_checks:
                # check batch is same across all tp
                sanity_check_tensor_sync(batch["input_ids"], tp_mesh)
                # check batch is different across dp
                sanity_check_tensor_sync(batch["input_ids"], dp_mesh, not_sync=True)

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
                        ],  # TODO: need to add attention mask
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

                    # Sanity checks for logits
                    if dist.is_initialized() and not ignore_sanity_checks:
                        # sanity_check_tensor_sync(logits, tp_mesh) # TODO: only true without sequence parallel
                        sanity_check_tensor_sync(logits, dp_mesh, not_sync=True)
                        sanity_check_tensor_sync(logits, cp_mesh, not_sync=True)

                    loss.backward()

                # all reduce grads across dp_cp if applicable
                all_reduce_grads(model, world_mesh, use_ddp=use_ddp)

                # Sanity checks for gradients (only if distributed)
                if dist.is_initialized() and not ignore_sanity_checks:
                    # check grads are not same across all tp (for sharded grads)
                    for name, param in model.named_parameters():
                        if param.grad is not None and isinstance(param.grad, DTensor):
                            if param.grad.placements[0].is_shard():
                                sanity_check_tensor_sync(param.grad, tp_mesh, not_sync=True)
                            elif param.grad.placements[0].is_replicate():
                                sanity_check_tensor_sync(param.grad, tp_mesh)
                    # check grads are same across dp
                    for name, param in model.named_parameters():
                        if param.grad is not None and dp_mesh.size() > 1:
                            sanity_check_tensor_sync(param.grad, dp_mesh)
                    # check grads are same across cp
                    for name, param in model.named_parameters():
                        if param.grad is not None and cp_mesh.size() > 1:
                            sanity_check_tensor_sync(param.grad, cp_mesh)

                # Calculate gradient norm and clip gradients
                if hasattr(model, "clip_grad_norm_"):
                    # when using FSDP or DDP, model.parameters() doesn't work
                    gradnorm = model.clip_grad_norm_(max_norm=1.0, norm_type=2.0)
                else:
                    assert len(list(model.parameters())) > 2, "No parameters found in model. Probably DDP bug.."
                    assert len([p for p in model.parameters() if p.requires_grad]) > 2, (
                        "No gradients found in model. Probably DDP bug.."
                    )
                    assert len([p for p in model.parameters() if p.grad is not None]) > 2, (
                        "No gradients found in model. Probably DDP bug.."
                    )
                    # only works with FSDP's NO_SHARD otherwise we should use FSDP's clip_grad_norm_
                    gradnorm = clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2.0, foreach=True)

                optimizer.step()
                # Sanity checks for updated model parameters (only if distributed)
                if dist.is_initialized() and not ignore_sanity_checks:
                    # check updated model is different across all tp (for sharded params)
                    for name, param in model.named_parameters():
                        if isinstance(param, DTensor):
                            if param.placements[0].is_shard():
                                sanity_check_tensor_sync(param, tp_mesh, not_sync=True)
                            elif param.placements[0].is_replicate():
                                sanity_check_tensor_sync(param, tp_mesh)
                    # check updated model is same across dp
                    for name, param in model.named_parameters():
                        sanity_check_tensor_sync(param, dp_mesh)
                    # check updated model is same across cp
                    for name, param in model.named_parameters():
                        sanity_check_tensor_sync(param, cp_mesh)

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

    # Example of loading the checkpoint (only if distributed)
    if dist.is_initialized():
        # Create a new model instance
        logger.info("Creating new model instance for verification")
        new_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_mesh=tp_mesh,
            dtype=torch.bfloat16,  # Use same dtype
        )
        new_optimizer = optim.AdamW(new_model.parameters(), lr=LR)

        # Load checkpoint into new model
        state_dict = {"app": AppState(new_model, new_optimizer)}
        dcp.load(
            state_dict=state_dict,
            checkpoint_id=CHECKPOINT_DIR,
        )
        logger.info("Loaded checkpoint into new model")

        # Verify model weights match
        logger.info("Verifying model weights match...")
        for (name1, param1), (name2, param2) in zip(model.named_parameters(), new_model.named_parameters()):
            torch.testing.assert_close(
                param1.to_local(),
                param2.to_local(),
                rtol=1e-3,
                atol=1e-3,
                msg=f"Weights mismatch in {name1} vs {name2}",
            )

        # Verify optimizer states match
        logger.info("Verifying optimizer states match...")
        for name1, state1 in optimizer.state_dict().items():
            state2 = new_optimizer.state_dict()[name1]
            if name1 == "state":
                # Compare state dictionaries for each parameter
                for param_id, param_state1 in state1.items():
                    param_state2 = state2[param_id]
                    # Compare each state component (step, exp_avg, exp_avg_sq)
                    for key, value1 in param_state1.items():
                        value2 = param_state2[key]
                        if isinstance(value1, DTensor):
                            # Convert DTensors to local tensors for comparison
                            torch.testing.assert_close(
                                value1.to_local(),
                                value2.to_local(),
                                rtol=1e-5,
                                atol=1e-5,
                                msg=f"Optimizer state mismatch in state[{param_id}][{key}]",
                            )
                        else:
                            torch.testing.assert_close(
                                value1,
                                value2,
                                rtol=1e-5,
                                atol=1e-5,
                                msg=f"Optimizer state mismatch in state[{param_id}][{key}]",
                            )
            elif name1 == "param_groups":
                # Compare param_groups (excluding the actual params list)
                for i, (group1, group2) in enumerate(zip(state1, state2)):
                    for key in group1:
                        if key != "params":  # Skip comparing the params list
                            assert group1[key] == group2[key], f"Param group mismatch in param_groups[{i}][{key}]"

        # Run a forward pass with both models to verify outputs match
        logger.info("Running forward pass verification...")
        with torch.no_grad():
            # Use the last batch for verification
            batch = {k: v.to(device) for k, v in batch.items()}  # Ensure batch is on correct device
            original_outputs = model(**batch)
            new_outputs = new_model(**batch)
            torch.testing.assert_close(
                original_outputs.logits.to_local(),
                new_outputs.logits.to_local(),
                rtol=1e-3,
                atol=1e-3,
                msg="Model outputs do not match!",
            )  # Increased tolerance slightly for bf16

    # Clean up distributed environment and finish wandb run
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Cleaned up distributed process group")
        # Finish wandb run on rank 0
        if dist.get_rank() == 0:
            wandb.finish()
            logger.info("Wandb run finished.")
    else:
        wandb.finish()
        logger.info("Wandb run finished.")


def all_reduce_grads(model, world_mesh, use_ddp):
    """All reduce gradients across dp_cp if applicable."""
    cp_mesh = world_mesh["cp"]
    if use_ddp:
        # DDP takes care of syncing grads
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


class ContextParallelCollator:
    """Collator for context parallel training that splits sequences into chunks."""

    def __init__(self, cp_mesh: Optional[DeviceMesh] = None):
        self.cp_mesh = cp_mesh

    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch = default_collate(batch)
        if self.cp_mesh is not None and self.cp_mesh.size() > 1:
            # Get sequence length from the input batch
            seq_len = batch["input_ids"].shape[1]
            assert seq_len % self.cp_mesh.size() == 0, (
                f"Sequence length {seq_len} must be divisible by CP size {self.cp_mesh.size()}"
            )
            chunk_size = seq_len // self.cp_mesh.size()
            cp_rank = self.cp_mesh.get_local_rank()
            start_idx = cp_rank * chunk_size
            end_idx = start_idx + chunk_size

            # Keep only the local chunk of the sequence
            batch["input_ids"] = batch["input_ids"][:, start_idx:end_idx]
            batch["attention_mask"] = batch["attention_mask"][:, start_idx:end_idx]
            batch["labels"] = batch["labels"][:, start_idx:end_idx]

        return batch


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


def sanity_check_tensor_sync(
    tensor: torch.Tensor, mesh: DeviceMesh, rtol: float = 1e-4, atol: float = 1e-4, not_sync: bool = False
) -> None:
    """
    Verify that a tensor is synchronized (or not synchronized) across all processes in the mesh's process group.
    Handles both regular tensors and DTensors.

    Args:
        tensor (torch.Tensor): The tensor to check for synchronization (can be DTensor)
        mesh (DeviceMesh): The device mesh containing the process group
        rtol (float): Relative tolerance for comparison
        atol (float): Absolute tolerance for comparison
        not_sync (bool): If True, asserts that tensors are NOT synchronized. If False, asserts they are synchronized.
    """
    if not dist.is_initialized() or mesh.size() == 1:
        return  # No need to check in non-distributed mode

    # Get the process group from the mesh
    pg = mesh.get_group()

    # Convert DTensor to local tensor if needed
    if hasattr(tensor, "to_local"):
        local_tensor = tensor.to_local()
    else:
        local_tensor = tensor

    # Gather tensors from all processes
    world_size = dist.get_world_size(pg)
    gathered_tensors = [torch.empty_like(local_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, local_tensor, group=pg)

    # Compare each tensor with the first one
    for i in range(1, world_size):
        try:
            torch.testing.assert_close(gathered_tensors[0], gathered_tensors[i], rtol=rtol, atol=atol)
        except AssertionError as e:
            if not_sync:
                continue
            # # Add detailed debugging for logit synchronization issues
            # print(f"\nLogit synchronization error between rank 0 and rank {i}:")
            # print(f"Tensor shape: {gathered_tensors[0].shape}")
            # print(f"Number of mismatched elements: {(gathered_tensors[0] != gathered_tensors[i]).sum()}")
            # print(f"Percentage of mismatched elements: {((gathered_tensors[0] != gathered_tensors[i]).sum() / gathered_tensors[0].numel() * 100):.2f}%")

            # # Find the first few mismatches
            # mismatches = torch.nonzero(gathered_tensors[0] != gathered_tensors[i])
            # print("\nFirst few mismatches:")
            # for idx in mismatches[:5]:
            #     idx = tuple(idx.tolist())
            #     print(f"Index {idx}:")
            #     print(f"Rank 0 value: {gathered_tensors[0][idx]}")
            #     print(f"Rank {i} value: {gathered_tensors[i][idx]}")
            #     print(f"Absolute difference: {abs(gathered_tensors[0][idx] - gathered_tensors[i][idx])}")
            #     print(f"Relative difference: {abs(gathered_tensors[0][idx] - gathered_tensors[i][idx]) / max(abs(gathered_tensors[0][idx]), abs(gathered_tensors[i][idx]))}")

            # # Check if differences are systematic (e.g., all positive or negative)
            # diff = gathered_tensors[0] - gathered_tensors[i]
            # print(f"\nDifference statistics:")
            # print(f"Mean difference: {diff.mean()}")
            # print(f"Std difference: {diff.std()}")
            # print(f"Max positive difference: {diff.max()}")
            # print(f"Max negative difference: {diff.min()}")
            raise e


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


def check_params_sync(model_params, original_params):
    """
    Check if original_params are being updated in sync with model parameters.

    Args:
        model_params: Iterator of model parameters after update
        original_params: List of original parameters before DDP wrapping
    """
    for mp, op in zip(model_params, original_params):
        if isinstance(mp, DTensor):
            mp = mp.to_local()
        if isinstance(op, DTensor):
            op = op.to_local()
        if not torch.allclose(mp.data, op.data, rtol=0, atol=0):
            raise RuntimeError(f"Parameters out of sync: model param {mp.data} != original param {op.data}")
    return True


def get_parameters(model: nn.Module) -> Iterable[torch.Tensor]:
    """
    Get all parameters from a model by iterating over its modules.
    This is an alternative to model.parameters() that works with DTensor models.

    Args:
        model (nn.Module): The model to get parameters from

    Returns:
        Iterable[torch.Tensor]: An iterator over all parameters in the model
    """
    for module in model._modules.values():
        # Look for parameters in module attributes
        for attr in module.__dict__.values():
            if isinstance(attr, torch.Tensor) and attr.requires_grad:
                yield attr
        # Recursively get parameters from submodules
        for param in get_parameters(module):
            yield param


def update_model_parameters(model: nn.Module) -> None:
    """
    Update model._parameters using named_modules() to ensure all parameters are properly tracked.

    Args:
        model (nn.Module): The model to update parameters for
    """
    # Clear existing parameters
    model._parameters = {}

    # Add parameters from named_modules
    for name, module in model.named_modules():
        # Skip the root module itself
        if name == "":
            continue

        # Get the parameter name by removing 'module.' prefix if it exists
        param_name = name.replace("module.", "")

        # Add weight and bias parameters if they exist
        if hasattr(module, "weight") and module.weight is not None:
            model._parameters[f"{param_name}.weight"] = module.weight
        if hasattr(module, "bias") and module.bias is not None:
            model._parameters[f"{param_name}.bias"] = module.bias


if __name__ == "__main__":
    main()
