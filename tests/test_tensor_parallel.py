import os

import torch
import torch.distributed as dist
from torch.distributed import DeviceMesh

from transformers.integrations.tensor_parallel import get_tensor_shard


def setup_distributed(rank, world_size):
    """Set up distributed environment"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup_distributed():
    """Clean up distributed environment"""
    dist.destroy_process_group()


def test_colwise_split(rank, world_size):
    """Test column-wise splitting functionality"""
    setup_distributed(rank, world_size)

    tensor = torch.randn(4, 8)
    empty_tensor = torch.empty_like(tensor)
    device_mesh = DeviceMesh("cpu", torch.arange(world_size))

    # Get shard for current rank
    rank_shard = get_tensor_shard(tensor, empty_tensor, device_mesh, rank, -1)
    if rank == 0:
        assert rank_shard.shape == (4, 4)
        assert torch.allclose(rank_shard, tensor[:, :4])
    else:
        assert rank_shard.shape == (4, 4)
        assert torch.allclose(rank_shard, tensor[:, 4:])

    cleanup_distributed()


def test_colwise_parallel_layer(rank, world_size):
    """Test ColwiseParallel layer functionality"""
    setup_distributed(rank, world_size)

    # Create a linear layer
    layer = torch.nn.Linear(8, 4)
    device_mesh = DeviceMesh("cpu", torch.arange(world_size))

    # Verify weight sharding
    weight = layer.weight
    empty_weight = torch.empty_like(weight)

    # Get shard
    rank_weight = get_tensor_shard(weight, empty_weight, device_mesh, rank, -2)

    # In ColwiseParallel, weights are sharded along the first dimension
    if rank == 0:
        assert rank_weight.shape == (2, 8)  # Modified to correct shape
        assert torch.allclose(rank_weight, weight[:2, :])  # Modified to correct shard
    else:
        assert rank_weight.shape == (2, 8)  # Modified to correct shape
        assert torch.allclose(rank_weight, weight[2:, :])  # Modified to correct shard

    cleanup_distributed()
