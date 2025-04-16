#!/bin/bash

# Set environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Run single process tests
echo "Running single process tests..."
python tests/test_tensor_parallel.py

# Run multi-process tests
echo "Running distributed tests with 2 processes..."
torchrun --nproc_per_node=2 tests/test_tensor_parallel_distributed.py

echo "All tests completed!" 