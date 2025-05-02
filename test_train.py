""":
This script is used to test training with the SmolLM2-135M model using Tensor Parallelism.

Usage:
torchrun --nproc_per_node=<num_gpus> test_train.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

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
    rank = dist.get_rank(pg)
    
    # Create a list to store gathered tensors
    gathered_tensors = [torch.empty_like(local_tensor) for _ in range(world_size)]
    
    # Gather all tensors
    dist.all_gather(gathered_tensors, local_tensor, group=pg)
    
    # Compare each tensor with the first one
    is_synced = True
    for i in range(1, world_size):
        if not torch.allclose(gathered_tensors[0], gathered_tensors[i], rtol=rtol, atol=atol):
            is_synced = False
            logger.warning(f"Tensor mismatch between rank 0 and rank {i}")
            break
    
    return is_synced

def main():
    tp_size = 4
    dp_size = 1
    CHECKPOINT_DIR = "checkpoint"
    
    # Initialize distributed environment
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        # Create TP device mesh spanning all available ranks/GPUs
        mesh = torch.arange(world_size).reshape(dp_size, tp_size)
        device_mesh = DeviceMesh(device_type="cuda", mesh=mesh, mesh_dim_names=("dp", "tp"))
        tp_mesh = device_mesh["tp"]
        dp_mesh = device_mesh["dp"]
        logger.info(f"Created DeviceMesh: {device_mesh}")
        logger.info(f"Distributed setup - Rank: {rank}, World size: {world_size}, Local rank: {local_rank}, DP: {dp_mesh.get_local_rank()}, TP: {tp_mesh.get_local_rank()}")

    else:
        logger.info("Running in non-distributed mode. DeviceMesh not applicable.")
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    model_name = "HuggingFaceTB/SmolLM2-1.7B"
    logger.info(f"Loading model and tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        tp_plan="auto",
        device_mesh=tp_mesh if dist.is_initialized() else None
    )

    assert model.config.num_key_value_heads % tp_mesh.size() == 0, f"num_key_value_heads={model.config.num_key_value_heads} must be divisible by tp_size={tp_mesh.size()}"

    device = torch.device(f"cuda:{local_rank}")
    logger.info(f"Using device: {device} for non-model tensors")

    # Wrap model with DDP for data parallelism
    if dist.is_initialized() and dp_mesh.size() > 1:
        # TODO: DDP doesn't work with dtensors
        # model = torch.nn.parallel.DistributedDataParallel(
        #     model,
        #     # device_ids=[local_rank],
        #     # output_device=local_rank,
        #     device_mesh=dp_mesh
        # )

        # Warning this API is still experimental
        model = replicate(model, device_mesh=dp_mesh, bucket_cap_mb=100)
        logger.info("Applied DDP")

    model.train()

    # assert model is replicated across all dp
    for name, param in model.named_parameters():
        assert sanity_check_tensor_sync(param, dp_mesh), f"Param {name} is not replicated across all dp {param}"

    # assert model is different across tp
    for name, param in model.named_parameters():
        if param.placements[0].is_shard():
            assert not sanity_check_tensor_sync(param, tp_mesh), f"Param {name} is same across all tp {param}"
        if param.placements[0].is_replicate():
            assert sanity_check_tensor_sync(param, tp_mesh), f"Param {name} is not replicated across all tp {param}"
    
    # Create dummy dataset and dataloader
    dataset = DummyDataset(tokenizer, seq_len=64, size=16, seed=42)
    if dist.is_initialized():
        sampler = DistributedSampler(dataset, num_replicas=dp_mesh.size(), rank=dp_mesh.get_local_rank(), shuffle=True)
        shuffle = None
    else:
        model = model.to(device)
        sampler = None
        shuffle = True
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        sampler=sampler,
        shuffle=shuffle
    )
    logger.info(f"DataLoader created with batch size 2. Distributed: {dist.is_initialized()}")

    optimizer = optim.AdamW(model.parameters(), lr=1e-5)

    # Training loop (single step)
    logger.info("Starting single training step...")
    try:
        batch = next(iter(dataloader))
    except StopIteration:
        logger.error("DataLoader is empty. Check dataset size and batch size.")
        if dist.is_initialized():
            dist.destroy_process_group()
        return

    batch = {k: v.to(device) for k, v in batch.items()}
    # check batch is same across all tp
    assert sanity_check_tensor_sync(batch["input_ids"], tp_mesh), f"Batch is not same across all tp {batch['input_ids']}"
    # check batch is different across dp
    if dp_mesh.size() > 1:
        assert not sanity_check_tensor_sync(batch["input_ids"], dp_mesh), f"Batch is same across dp {batch['input_ids']}"

    optimizer.zero_grad()
    outputs = model(**batch)
    loss = outputs.loss
    logits = outputs.logits

    # TODO: only true without sequence parallel
    assert sanity_check_tensor_sync(logits, tp_mesh), f"Logits are not same across all tp when not using sequence parallel {logits}"
    # check logits are not same across dp
    if dp_mesh.size() > 1:
        assert not sanity_check_tensor_sync(logits, dp_mesh), f"Logits are same across dp {logits}"

    logger.info(f"Input IDs shape: {batch['input_ids'].shape}")
    logger.info(f"Logits shape: {logits.shape}")
    logger.info(f"Calculated Loss: {loss.item()}")

    loss.backward()

    # check grads are not same across all tp
    for name, param in model.named_parameters():
        if param.placements[0].is_shard():
            assert not sanity_check_tensor_sync(param.grad, tp_mesh), f"Grad {name} is same across all tp {param.grad}"
        if param.placements[0].is_replicate():
            assert sanity_check_tensor_sync(param.grad, tp_mesh), f"Grad {name} is not replicated across all tp {param.grad}"
    # check grads are same across dp
    for name, param in model.named_parameters():
        if dp_mesh.size() > 1:
            assert sanity_check_tensor_sync(param.grad, dp_mesh), f"Grad {name} is not same across dp {param.grad}"

    optimizer.step()

    # check updated model is different across all tp
    for name, param in model.named_parameters():
        if param.placements[0].is_shard():
            assert not sanity_check_tensor_sync(param, tp_mesh), f"Updated model {name} is same across all tp {param}"
        if param.placements[0].is_replicate():
            assert sanity_check_tensor_sync(param, tp_mesh), f"Updated model {name} is not replicated across all tp {param}"
    # check updated model is same across dp
    for name, param in model.named_parameters():
        if dp_mesh.size() > 1:
            assert sanity_check_tensor_sync(param, dp_mesh), f"Updated model {name} is not same across dp {param}"

    logger.info("Single training step completed.")

    # Save model using DCP
    if dist.is_initialized():
        state_dict = {"app": AppState(model, optimizer)}
        dcp.save(
            state_dict=state_dict,
            checkpoint_id=CHECKPOINT_DIR,
        )
        logger.info(f"Saved checkpoint to {CHECKPOINT_DIR}")
    else:
        # Fallback to regular save for non-distributed case
        model.save_pretrained(f"test_model_{rank}", safe_serialization=False)
        logger.info(f"Saved model to test_model_{rank}")

    # Example of loading the checkpoint
    # if dist.is_initialized():
    #     # Create a new model instance
    #     logger.info("Creating new model instance for verification")
    #     new_model = AutoModelForCausalLM.from_pretrained(
    #         model_name,
    #         tp_plan="auto",
    #         device_mesh=tp_mesh
    #     )
    #     new_optimizer = optim.AdamW(new_model.parameters(), lr=1e-5)
        
    #     # Load checkpoint into new model
    #     state_dict = {"app": AppState(new_model, new_optimizer)}
    #     dcp.load(
    #         state_dict=state_dict,
    #         checkpoint_id=CHECKPOINT_DIR,
    #     )
    #     logger.info("Loaded checkpoint into new model")

    #     # Verify model weights match
    #     logger.info("Verifying model weights match...")
    #     for (name1, param1), (name2, param2) in zip(model.named_parameters(), new_model.named_parameters()):
    #         torch.testing.assert_allclose(param1.to_local(), param2.to_local(), rtol=1e-5, atol=1e-5, msg=f"Weights mismatch in {name1} vs {name2}")

    #     # Verify optimizer states match
    #     logger.info("Verifying optimizer states match...")
    #     for (name1, state1), (name2, state2) in zip(optimizer.state_dict().items(), new_optimizer.state_dict().items()):
    #         if isinstance(state1, torch.Tensor) and isinstance(state2, torch.Tensor):
    #             torch.testing.assert_allclose(state1, state2, rtol=1e-5, atol=1e-5, msg=f"Optimizer state mismatch in {name1} vs {name2}")
    #         else:
    #             # For non-tensor states, just check equality
    #             if state1 != state2:
    #                 logger.error(f"Optimizer state mismatch in {name1} vs {name2}")

    #     # Run a forward pass with both models to verify outputs match
    #     logger.info("Running forward pass verification...")
    #     with torch.no_grad():
    #         original_outputs = model(**batch)
    #         new_outputs = new_model(**batch)
            
    #         if not torch.allclose(original_outputs.logits, new_outputs.logits, rtol=1e-5, atol=1e-5):
    #             logger.error("Model outputs do not match!")
    #             logger.error(f"Original logits: {original_outputs.logits}")
    #             logger.error(f"Loaded logits: {new_outputs.logits}")
    #         else:
    #             logger.info("Model outputs match!")

    # Clean up distributed environment
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Cleaned up distributed process group")

class DummyDataset(Dataset):
    def __init__(self, tokenizer, seq_len=512, size=100, seed=42):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.size = size
        # Create some dummy data (e.g., sequences of padding tokens)
        # In a real scenario, load actual data
        self.generator = torch.Generator().manual_seed(seed)
        self.data = [torch.randint(0, tokenizer.vocab_size, (seq_len,), generator=self.generator) for _ in range(size)]
        logger.info(f"DummyDataset created with {size} samples, sequence length {seq_len}")

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        input_ids = self.data[idx]
        # Create attention mask (all 1s for dummy data)
        attention_mask = torch.ones_like(input_ids)
        # Create labels (e.g., shifted input_ids for language modeling)
        labels = input_ids.clone()
        # For causal LM, labels are usually shifted, set padding tokens to -100
        # Simplified for this test: using input_ids as labels directly
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

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


if __name__ == "__main__":
    main() 