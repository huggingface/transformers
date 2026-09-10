import torch
import torch.distributed as dist
from torch.distributed import DeviceMesh

from transformers.integrations.tensor_parallel import get_tensor_shard


def setup_distributed():
    """Set up distributed environment"""
    if not dist.is_initialized():
        dist.init_process_group("gloo")


def test_distributed_colwise_split():
    """Test column-wise splitting functionality in distributed environment"""
    setup_distributed()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Create an example tensor
    if rank == 0:
        tensor = torch.randn(4, 8)  # 4 rows, 8 columns
        # Broadcast tensor to all processes
        dist.broadcast(tensor, src=0)
    else:
        tensor = torch.empty(4, 8)
        dist.broadcast(tensor, src=0)

    empty_tensor = torch.empty_like(tensor)
    device_mesh = DeviceMesh("cpu", torch.arange(world_size))

    # Get shard for current process
    shard = get_tensor_shard(tensor, empty_tensor, device_mesh, rank, -1)

    # Verify shard shape
    assert shard.shape == (4, 8 // world_size)

    # Verify shard content
    start_col = rank * (8 // world_size)
    end_col = (rank + 1) * (8 // world_size)
    assert torch.allclose(shard, tensor[:, start_col:end_col])


def test_distributed_colwise_parallel_layer():
    """Test ColwiseParallel layer functionality in distributed environment"""
    setup_distributed()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Create a linear layer
    if rank == 0:
        layer = torch.nn.Linear(8, 4)
        # Broadcast weights to all processes
        dist.broadcast(layer.weight, src=0)
        dist.broadcast(layer.bias, src=0)
    else:
        layer = torch.nn.Linear(8, 4)
        dist.broadcast(layer.weight, src=0)
        dist.broadcast(layer.bias, src=0)

    device_mesh = DeviceMesh("cpu", torch.arange(world_size))

    # Verify weight sharding
    weight = layer.weight
    empty_weight = torch.empty_like(weight)

    # Get shard for current process
    shard = get_tensor_shard(weight, empty_weight, device_mesh, rank, 0)  # Modified to shard along first dimension

    # Verify shard shape
    assert shard.shape == (4 // world_size, 8)

    # Verify shard content
    start_row = rank * (4 // world_size)
    end_row = (rank + 1) * (4 // world_size)
    assert torch.allclose(shard, weight[start_row:end_row, :])
