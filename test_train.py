""":
This script is used to test training a model using Tensor Parallelism and Data Parallelism.

Usage:
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=4,5,6,7
TP_SIZE=2 DP_SIZE=2 torchrun --nproc_per_node=4 --rdzv_endpoint=localhost:29503 test_train.py
CP_SIZE=2 DP_SIZE=2 torchrun --nproc_per_node=4 test_train.py
TP_SIZE=1 DP_SIZE=4 torchrun --nproc_per_node=4 test_train.py
TP_SIZE=4 DP_SIZE=1 torchrun --nproc_per_node=4 test_train.py
TP_SIZE=1 DP_SIZE=1 torchrun --rdzv_id=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:29503 test_train.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os
import logging
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed._composable.replicate import replicate
from torch.distributed.tensor.placement_types import Replicate
from torch.distributed.tensor import DTensor
import wandb
from datasets import load_dataset
from typing import Dict, Any, Optional
from torch.distributed.tensor.experimental import context_parallel
from torch.utils.data import default_collate
from torch.nn.attention import sdpa_kernel, SDPBackend
from typing import Iterable
import math

ignore_sanity_checks = False

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    tp_size = int(os.environ.get("TP_SIZE", 1))
    dp_size = int(os.environ.get("DP_SIZE", 2))
    cp_size = int(os.environ.get("CP_SIZE", 2))  # Add CP size configuration
    # sdpa_backend = SDPBackend.FLASH_ATTENTION # For CP
    sdpa_backend = SDPBackend.MATH # For CP
    global_batch_size = 4 # Desired global batch size
    seq_len = 256 # Sequence length
    num_train_steps = 10000 # Number of training steps

    CHECKPOINT_DIR = f"checkpoint_tp{tp_size}_dp{dp_size}_cp{cp_size}"

    # Initialize distributed environment
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

        assert world_size == tp_size * dp_size * cp_size, f"World size ({world_size}) must equal TP size ({tp_size}) * DP size ({dp_size}) * CP size ({cp_size})"

        mesh = torch.arange(world_size).reshape(dp_size, tp_size, cp_size)
        world_mesh = DeviceMesh(device_type="cuda", mesh=mesh, mesh_dim_names=("dp", "tp", "cp"))
        tp_mesh = world_mesh["tp"]
        dp_mesh = world_mesh["dp"]
        cp_mesh = world_mesh["cp"]
        logger.info(f"Created DeviceMesh: {world_mesh}")
        logger.info(f"Distributed setup - Rank: {rank}, World size: {world_size}, Local rank: {local_rank}, DP: {dp_mesh.get_local_rank()}, TP: {tp_mesh.get_local_rank()}, CP: {cp_mesh.get_local_rank()}")

        if dist.get_rank() == 0:
            wandb.init(
                project="tp_dp_test",
                config={
                    "tp_size": tp_size,
                    "dp_size": dp_size,
                    "cp_size": cp_size,
                    "global_batch_size": global_batch_size,
                    "model_name": "HuggingFaceTB/SmolLM2-1.7B",
                    "dataset": "roneneldan/TinyStories-1M",
                    "seq_len": seq_len,
                },
                name=f"tp{tp_size}_dp{dp_size}_cp{cp_size}"
            )
            logger.info("Wandb initialized.")

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
                "model_name": "HuggingFaceTB/SmolLM2-1.7B",
                "dataset": "roneneldan/TinyStories-1M",
                "seq_len": seq_len,
            },
            name="tp1_dp1_nondist"
        )
        logger.info("Wandb initialized for non-distributed run.")

    # Load model and tokenizer
    model_name = "HuggingFaceTB/SmolLM2-1.7B"
    logger.info(f"Loading model and tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_mesh=tp_mesh if dist.is_initialized() else None,
        tp_plan="auto",
        torch_dtype=torch.bfloat16,
    )
    logger.info(f"Model loaded onto device mesh: {tp_mesh}")

    if dist.is_initialized():
        assert model.config.num_key_value_heads % tp_mesh.size() == 0, f"num_key_value_heads={model.config.num_key_value_heads} must be divisible by tp_size={tp_mesh.size()}"
        device = torch.device(f"cuda:{local_rank}")
    else:
        model = model.to(device)

    logger.info(f"Using device: {device} for non-model tensors")
    use_ddp = False
    if dist.is_initialized() and dp_mesh.size() > 1:
        # TODO: this only works with DDP patch
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_mesh=dp_mesh
        )
        # Warning this API is still experimental
        # model = replicate(model, device_mesh=dp_mesh, bucket_cap_mb=100)
        # logger.info("Applied DDP")
        use_ddp = True
        pass

    model.train()

    if dist.is_initialized() and not ignore_sanity_checks:
        # assert model is replicated across all dp
        for name, param in model.named_parameters():
            assert sanity_check_tensor_sync(param, dp_mesh), f"Param {name} is not replicated across all dp {param}"

        # assert model is different across tp (only for sharded params)
        for name, param in model.named_parameters():
            if isinstance(param, DTensor) and param.placements[0].is_shard():
                 # Only check sharded parameters for non-sync across TP
                 if tp_mesh.size() > 1:
                    assert not sanity_check_tensor_sync(param, tp_mesh), f"Sharded param {name} is unexpectedly the same across all tp {param}"
            elif isinstance(param, DTensor) and param.placements[0].is_replicate():
                 # Replicated parameters should be the same across TP
                 assert sanity_check_tensor_sync(param, tp_mesh), f"Replicated param {name} is not replicated across all tp {param}"

        # assert model is replicated across cp
        for name, param in model.named_parameters():
            assert sanity_check_tensor_sync(param, cp_mesh), f"Param {name} is not replicated across all cp {param}"

    # Load and preprocess TinyStories dataset
    logger.info("Loading TinyStories dataset...")
    raw_dataset = load_dataset("roneneldan/TinyStories", split="train[:1%]") # Use 1% for faster testing
    
    def tokenize_function(examples):
        # Tokenize the text
        tokenized_batch = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=seq_len, return_tensors="pt")
        # Set labels to be the same as input_ids for Causal LM
        tokenized_batch["labels"] = tokenized_batch["input_ids"].clone()
        return tokenized_batch

    tokenized_dataset = raw_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    logger.info(f"Dataset loaded and tokenized. Size: {len(tokenized_dataset)}")

    # Calculate local batch size
    if dist.is_initialized():
        assert global_batch_size % dp_mesh.size() == 0, f"Global batch size ({global_batch_size}) must be divisible by DP size ({dp_mesh.size()})"
        local_batch_size = global_batch_size // dp_mesh.size()
    else:
        local_batch_size = global_batch_size

    logger.info(f"Global batch size: {global_batch_size}, DP size: {dp_size if dist.is_initialized() else 1}, Local batch size: {local_batch_size}")

    if dist.is_initialized():
        sampler = DistributedSampler(tokenized_dataset, num_replicas=dp_mesh.size(), rank=dp_mesh.get_local_rank(), shuffle=True)
        shuffle = False # Sampler handles shuffling
    else:
        sampler = None
        shuffle = True

    # Create collator for context parallel
    # collator = ContextParallelCollator(cp_mesh=cp_mesh if dist.is_initialized() else None)
    
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=local_batch_size,
        sampler=sampler,
        shuffle=shuffle,
        # collate_fn=collator,
        # num_workers=2, # Optional: Add workers for faster data loading
        # pin_memory=True # Optional: Pin memory if using GPU
    )
    logger.info(f"DataLoader created. Distributed: {dist.is_initialized()}")

    optimizer = optim.AdamW(model.parameters(), lr=1e-5)

    # Training loop
    logger.info(f"Starting training for {num_train_steps} steps...")
    model.train()
    step = 0
    while step < num_train_steps:
        for batch in dataloader:
            if step >= num_train_steps:
                break # Exit loop if max steps reached

            # Move batch to appropriate device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Sanity checks for batch distribution (only if distributed)
            if dist.is_initialized() and not ignore_sanity_checks:
                # check batch is same across all tp
                assert sanity_check_tensor_sync(batch["input_ids"], tp_mesh), f"Batch is not same across all tp {batch['input_ids']}"
                # check batch is different across dp
                if dp_mesh.size() > 1:
                    assert not sanity_check_tensor_sync(batch["input_ids"], dp_mesh), f"Batch is same across dp {batch['input_ids']}"
                # TODO: context_parallel handles the sharding
                # check batch is different across cp (for sequence chunks)
                # if cp_mesh.size() > 1:
                #     assert not sanity_check_tensor_sync(batch["input_ids"], cp_mesh), f"Batch is same across cp {batch['input_ids']}"

            optimizer.zero_grad()

            # make attention mask all ones for now
            # batch["attention_mask"] = torch.ones_like(batch["input_ids"])

            with sdpa_kernel(sdpa_backend): # TODO: ideally move this to attention implementation
                with context_parallel(
                    cp_mesh, 
                    buffers=[batch["input_ids"], batch["labels"]], 
                    buffer_seq_dims=[1, 1],
                ):
                    outputs = model(**batch) # [mbs, seq_len/cp]
                    loss = outputs.loss
                    logits = outputs.logits # Logits might not be needed unless debugging

                    current_loss = loss.item() # Get scalar loss value

                    # Sanity checks for logits (only if distributed and no sequence parallelism)
                    # TODO: only true without sequence parallel
                    if dist.is_initialized() and not ignore_sanity_checks:
                        assert sanity_check_tensor_sync(logits, tp_mesh), f"Logits are not same across all tp when not using sequence parallel {logits}"
                        # check logits are not same across dp
                        if dp_mesh.size() > 1:
                            assert not sanity_check_tensor_sync(logits, dp_mesh), f"Logits are same across dp {logits}"
                        # check logits are not same across cp (for sequence chunks)
                        if cp_mesh.size() > 1:
                            assert not sanity_check_tensor_sync(logits, cp_mesh), f"Logits are same across cp {logits}"

                    loss.backward()

                # all reduce grads across dp_cp if applicable
                # all_reduce_grads(model, dp_mesh, cp_mesh, use_ddp=use_ddp)

                # Sanity checks for gradients (only if distributed)
                if dist.is_initialized() and not ignore_sanity_checks:
                    # check grads are not same across all tp (for sharded grads)
                    for name, param in model.named_parameters():
                         if param.grad is not None and isinstance(param.grad, DTensor):
                             if param.grad.placements[0].is_shard() and tp_mesh.size() > 1:
                                assert not sanity_check_tensor_sync(param.grad, tp_mesh), f"Sharded Grad {name} is unexpectedly same across all tp {param.grad}"
                             elif param.grad.placements[0].is_replicate():
                                assert sanity_check_tensor_sync(param.grad, tp_mesh), f"Replicated Grad {name} is not replicated across all tp {param.grad}"
                    # check grads are same across dp
                    for name, param in model.named_parameters():
                        if param.grad is not None and dp_mesh.size() > 1:
                             assert sanity_check_tensor_sync(param.grad, dp_mesh), f"Grad {name} is not same across dp {param.grad}"
                    # check grads are same across cp
                    for name, param in model.named_parameters():
                        if param.grad is not None and cp_mesh.size() > 1:
                             assert sanity_check_tensor_sync(param.grad, cp_mesh), f"Grad {name} is not same across cp {param.grad}"

                optimizer.step()
                # Sanity checks for updated model parameters (only if distributed)
                if dist.is_initialized() and not ignore_sanity_checks:
                    # check updated model is different across all tp (for sharded params)
                    for name, param in model.named_parameters():
                        if isinstance(param, DTensor):
                            if param.placements[0].is_shard() and tp_mesh.size() > 1:
                                assert not sanity_check_tensor_sync(param, tp_mesh), f"Updated sharded model {name} is unexpectedly same across all tp {param}"
                            elif param.placements[0].is_replicate():
                                assert sanity_check_tensor_sync(param, tp_mesh), f"Updated replicated model {name} is not replicated across all tp {param}"
                    # check updated model is same across dp
                    for name, param in model.named_parameters():
                        if dp_mesh.size() > 1:
                            assert sanity_check_tensor_sync(param, dp_mesh), f"Updated model {name} is not same across dp {param}"
                    # check updated model is same across cp
                    for name, param in model.named_parameters():
                        if cp_mesh.size() > 1:
                            assert sanity_check_tensor_sync(param, cp_mesh), f"Updated model {name} is not same across cp {param}"

                # Calculate gradient norm and clip gradients
                assert len(list(model.parameters()))>0, "No parameters found in model. Probably DDP bug.."
                gradnorm = clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                    norm_type=2.0,
                    foreach=True
                )

                # Log loss and gradnorm to wandb (only on rank 0 of dp group)
                if not dist.is_initialized() or dist.get_rank() == 0:
                    logger.info(f"Step: {step} | GBS: {global_batch_size} | DP: {dp_mesh.size()} | TP: {tp_mesh.size()} | CP: {cp_mesh.size()} | Loss: {current_loss} | Gradnorm: {gradnorm}")
                    wandb.log({"train/loss": current_loss, "train/gradnorm": gradnorm, "step": step})

            step += 1 # Increment step count

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
        save_dir = f"test_model_nondist"
        model.save_pretrained(save_dir, safe_serialization=False)
        tokenizer.save_pretrained(save_dir) # Save tokenizer too
        logger.info(f"Saved model to {save_dir}")

    # Example of loading the checkpoint (only if distributed)
    if dist.is_initialized():
        # Create a new model instance
        logger.info("Creating new model instance for verification")
        new_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_mesh=tp_mesh,
            torch_dtype=torch.bfloat16 # Use same dtype
        )
        new_optimizer = optim.AdamW(new_model.parameters(), lr=1e-5)
        
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
            torch.testing.assert_close(param1.to_local(), param2.to_local(), rtol=1e-3, atol=1e-3, msg=f"Weights mismatch in {name1} vs {name2}")

        # Verify optimizer states match
        logger.info("Verifying optimizer states match...")
        for name1, state1 in optimizer.state_dict().items():
            state2 = new_optimizer.state_dict()[name1]
            if name1 == 'state':
                # Compare state dictionaries for each parameter
                for param_id, param_state1 in state1.items():
                    param_state2 = state2[param_id]
                    # Compare each state component (step, exp_avg, exp_avg_sq)
                    for key, value1 in param_state1.items():
                        value2 = param_state2[key]
                        if isinstance(value1, DTensor):
                            # Convert DTensors to local tensors for comparison
                            torch.testing.assert_close(
                                value1.to_local(), value2.to_local(), 
                                rtol=1e-5, atol=1e-5, 
                                msg=f"Optimizer state mismatch in state[{param_id}][{key}]"
                            )
                        else:
                            torch.testing.assert_close(
                                value1, value2, 
                                rtol=1e-5, atol=1e-5, 
                                msg=f"Optimizer state mismatch in state[{param_id}][{key}]"
                            )
            elif name1 == 'param_groups':
                # Compare param_groups (excluding the actual params list)
                for i, (group1, group2) in enumerate(zip(state1, state2)):
                    for key in group1:
                        if key != 'params':  # Skip comparing the params list
                            assert group1[key] == group2[key], f"Param group mismatch in param_groups[{i}][{key}]"

        # Run a forward pass with both models to verify outputs match
        logger.info("Running forward pass verification...")
        with torch.no_grad():
            # Use the last batch for verification
            batch = {k: v.to(device) for k, v in batch.items()} # Ensure batch is on correct device
            original_outputs = model(**batch)
            new_outputs = new_model(**batch)
            torch.testing.assert_close(original_outputs.logits.to_local(), new_outputs.logits.to_local(), rtol=1e-3, atol=1e-3, msg="Model outputs do not match!") # Increased tolerance slightly for bf16

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

def all_reduce_grads(model, dp_mesh, cp_mesh, use_ddp):
    """All reduce gradients across dp_cp if applicable."""
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
                    torch.distributed.all_reduce(
                        local_grad,
                        op=torch.distributed.ReduceOp.AVG,
                        group=mesh.get_group()
                    )
                    # Assign averaged grad back - need careful handling if DTensor structure is complex
                    # This simple assignment might work if the grad structure matches param structure
                    param.grad = DTensor.from_local(local_grad, device_mesh=param.grad.device_mesh, placements=param.grad.placements)
                else:
                     # Handle regular tensors if any exist (e.g. buffers not converted to DTensor)
                     torch.distributed.all_reduce(
                        param.grad,
                        op=torch.distributed.ReduceOp.AVG,
                        group=mesh.get_group()
                    )

class ContextParallelCollator:
    """Collator for context parallel training that splits sequences into chunks."""
    def __init__(self, cp_mesh: Optional[DeviceMesh] = None):
        self.cp_mesh = cp_mesh
        
    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch = default_collate(batch)
        if self.cp_mesh is not None and self.cp_mesh.size() > 1:
            # Get sequence length from the input batch
            seq_len = batch["input_ids"].shape[1]
            assert seq_len % self.cp_mesh.size() == 0, f"Sequence length {seq_len} must be divisible by CP size {self.cp_mesh.size()}"
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
        return {
            "model": model_state_dict,
            "optim": optimizer_state_dict
        }

    def load_state_dict(self, state_dict):
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"]
        )



def sanity_check_tensor_sync(tensor: torch.Tensor, mesh: DeviceMesh, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """
    Verify that a tensor is synchronized across all processes in the mesh's process group.
    Handles both regular tensors and DTensors.
    
    Args:
        tensor (torch.Tensor): The tensor to check for synchronization (can be DTensor)
        mesh (DeviceMesh): The device mesh containing the process group
        rtol (float): Relative tolerance for comparison
        atol (float): Absolute tolerance for comparison
        
    Returns:
        bool: True if tensors are synchronized, False otherwise
    """
    if not dist.is_initialized() or mesh.size() == 1:
        return True  # No need to check in non-distributed mode
        
    # Get the process group from the mesh
    pg = mesh.get_group()
    
    # Convert DTensor to local tensor if needed
    if hasattr(tensor, 'to_local'):
        local_tensor = tensor.to_local()
    else:
        local_tensor = tensor
    
    # Gather tensors from all processes
    world_size = dist.get_world_size(pg)
    
    # Create a list to store gathered tensors
    gathered_tensors = [torch.empty_like(local_tensor) for _ in range(world_size)]
    
    # Gather all tensors
    dist.all_gather(gathered_tensors, local_tensor, group=pg)
    
    # Compare each tensor with the first one
    for i in range(1, world_size):
        if not torch.allclose(gathered_tensors[0], gathered_tensors[i], rtol=rtol, atol=atol):
            return False
    
    return True
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
    if len(parameters) == 0:
        return torch.tensor(0.0, device=next(parameters).device)

    # Calculate total norm
    if norm_type == float('inf'):
        total_norm = max(p.grad.detach().abs().max() for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
            norm_type
        )

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