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

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Dummy Dataset
class DummyDataset(Dataset):
    def __init__(self, tokenizer, seq_len=512, size=100):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.size = size
        # Create some dummy data (e.g., sequences of padding tokens)
        # In a real scenario, load actual data
        self.data = [torch.randint(0, tokenizer.vocab_size, (seq_len,)) for _ in range(size)]
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

def main():
    # Initialize distributed environment
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        logger.info(f"Distributed setup - Rank: {rank}, World size: {world_size}, Local rank: {local_rank}")

        # Create TP device mesh spanning all available ranks/GPUs
        tp_device_mesh = DeviceMesh("cuda", torch.arange(world_size))
        logger.info(f"Created TP DeviceMesh: {tp_device_mesh}")

    else:
        logger.info("Running in non-distributed mode. DeviceMesh not applicable.")
        rank = 0
        world_size = 1
        local_rank = 0
        tp_device_mesh = None # No mesh for non-distributed
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    # model_name = "HuggingFaceTB/SmolLM2-135M" has 3 kv_heads
    model_name = "HuggingFaceTB/SmolLM2-1.7B"
    logger.info(f"Loading model and tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        tp_plan="auto",  # Keep auto plan for sharding strategy
        device_mesh=tp_device_mesh if dist.is_initialized() else None # Pass the mesh
    )

    assert model.config.num_key_value_heads % tp_device_mesh.size() == 0, f"num_key_value_heads={model.config.num_key_value_heads} must be divisible by tp_size={tp_device_mesh.size()}"

    # Move model to GPU - No longer needed, DeviceMesh handles placement
    device = torch.device(f"cuda:{local_rank}") # Still needed for data loader etc.
    logger.info(f"Using device: {device} for non-model tensors")
    # model = model.to(device) # REMOVED: DeviceMesh handles model placement

    # Set model to training mode
    model.train()

    # Create dummy dataset and dataloader
    dataset = DummyDataset(tokenizer, seq_len=64, size=16) # Small size for quick test
    if dist.is_initialized():
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        shuffle = False # Sampler handles shuffling
    else:
        # Non-distributed: use the determined device
        model = model.to(device) # Explicitly move model if not distributed
        sampler = None
        shuffle = True

    dataloader = DataLoader(
        dataset,
        batch_size=2, # Small batch size for test
        sampler=sampler,
        shuffle=shuffle
    )
    logger.info(f"DataLoader created with batch size 2. Distributed: {dist.is_initialized()}")

    # Define optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    # Loss is typically calculated internally by the model when labels are provided

    # Training loop (single step)
    logger.info("Starting single training step...")
    try:
        batch = next(iter(dataloader))
    except StopIteration:
        logger.error("DataLoader is empty. Check dataset size and batch size.")
        if dist.is_initialized():
            dist.destroy_process_group()
        return

    # Move batch to device (still needed for data)
    batch = {k: v.to(device) for k, v in batch.items()}

    # Clear gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(**batch)

    # Get loss and logits
    loss = outputs.loss
    logits = outputs.logits

    # Log shapes and loss
    logger.info(f"Input IDs shape: {batch['input_ids'].shape}")
    logger.info(f"Logits shape: {logits.shape}") # Expected: [batch_size, seq_len, vocab_size]
    logger.info(f"Calculated Loss: {loss.item()}")

    # Backward pass
    loss.backward()

    # Optimizer step
    optimizer.step()

    logger.info("Single training step completed.")

    # Save model
    model.save_pretrained(f"test_model_{rank}", safe_serialization=False)

    # TODO: make loading topo-aware
    # model = AutoModelForCausalLM.from_pretrained("test_model", tp_plan="auto", device_mesh=tp_device_mesh)

    # Clean up distributed environment
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Cleaned up distributed process group")

if __name__ == "__main__":
    main() 