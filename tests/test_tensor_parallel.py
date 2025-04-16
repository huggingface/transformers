import os
import time
import torch
import torch.distributed as dist
from torch.distributed import DeviceMesh
from transformers.integrations.tensor_parallel import ColwiseParallel, get_tensor_shard
from tqdm import tqdm
import torch.multiprocessing as mp

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
    
    # Create ColwiseParallel instance
    colwise_parallel = ColwiseParallel()
    
    # Prepare module
    parallel_layer = colwise_parallel.prepare_module_tp(layer, device_mesh)
    
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

def run_test(test_func, test_name, world_size=2):
    """Run test function"""
    print(f"\nTest: {test_name}")
    print("-" * 30)
    
    start_time = time.time()
    with tqdm(total=100, desc="Progress", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        processes = []
        for rank in range(world_size):
            p = mp.Process(target=test_func, args=(rank, world_size))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
        
        pbar.update(100)
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Test completed! Time taken: {duration:.2f} seconds")
    print("-" * 30)

if __name__ == "__main__":
    test_functions = [test_colwise_split, test_colwise_parallel_layer]
    test_names = ["Column-wise Split Test", "ColwiseParallel Layer Test"]
    
    print("\nStarting test suite...")
    print("=" * 50)
    
    total_start_time = time.time()
    
    for i, (test_func, test_name) in enumerate(zip(test_functions, test_names), 1):
        print(f"\nTest {i}/{len(test_functions)}: {test_name}")
        run_test(test_func, test_name)
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    print("\n" + "=" * 50)
    print(f"All tests completed!")
    print(f"Total number of tests: {len(test_functions)}")
    print(f"Total time taken: {total_duration:.2f} seconds")
    print("=" * 50) 