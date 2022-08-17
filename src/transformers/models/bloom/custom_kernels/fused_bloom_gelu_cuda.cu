#include <ATen/Dispatch.h>
#include <THC/THCAtomics.cuh>
#include <ATen/ATen.h>
#include <torch/torch.h>
#include <vector>

#include <optional>

/**
* Friendly reminder of how multithreading works in CUDA: https://developer.nvidia.com/blog/even-easier-introduction-cuda
* Check example at https://github.com/thomasw21/LinearTransformers/blob/main/model/attention/fast_weight/fast_weight_cuda.cu
**/

// Available in pytorch main
//#define DISPATCH_CASE_FLOATING_TYPES(...) \
//  at::AT_DISPATCH_CASE(at::ScalarType::Double, __VA_ARGS__) \
//  at::AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
//  at::AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__) \
//  at::AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__) \

/*
* Forward passes
*/

/**
* compute GELU: `x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))`
**/
template<typename scalar>
__global__ void forward_masked_softmax_kernel(
    const torch::PackedTensorAccessor32<scalar, 3, torch::RestrictPtrTraits> x, // [B, N, D]
    const torch::PackedTensorAccessor32<scalar, 1, torch::RestrictPtrTraits> bias, // [D]
    torch::PackedTensorAccessor32<attention_scores_scalar, 3, torch::RestrictPtrTraits> result // [B, N, D]
) {
    const int batch_id = blockIdx.x;
    const int q_length_id = blockIdx.y;
    const int batch_time_q_length_block_size = thread.y;
    const int kv_length_id = threadIdx.x;

    const auto elt = x[batch_id][q_length_id][kv_length_id];

    // Compute gelu
    // TODO @thomasw21: Figure out where to find a tanh implementation that works for me. (I could hardcode it)
    result[batch_id][q_length_id][kv_length_id] = elt * 0.5 * (1.0 + std::tanh(0.79788456 * elt * (1 + 0.044715 * elt * elt)));
}

std::tuple<at::Tensor, std::optional<std::vector<at::Tensor>>, at::Tensor> forward(
    const at::Tensor x,
    const std::optional<std::vector<at::Tensor>> bias,
    at::Tensor result,
) {
    // TODO @thomas21: change by to this as it's cleaner when pytorch 1.13 comes out
    // DISPATCH_CASE_FLOATING_TYPES(key_layer.scalar_type(), "masked_softmax", [&] {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "gelu", [&] {
        // TODO @thomasw21 I think this is necessary if you want to support all kinds of gpus.
        // const uint64_t maxGridY = at::cuda::getCurrentDeviceProperties()->maxGridSize[1];

        /*
        * Understanding how GPUs work: https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/
        * A100 specifications: https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf
        *  - SMs: 108
        *  - TPCs: 56 (What's that?)
        *  - Memory size: 40 GB
        *  - L2 Cache size: 40960 KB (shared across all SMs)
        *  - L1/Shared memory size: 192 KB (shared across all threads within a SM)
        *  - Max Threads / SM: 2048
        *  - Max Thread Blocks / SM: 32
        */

        /*
        * We should split [batch_size_times_num_heads_block, q_length] in seperate blocks and [batch_size_times_num_heads_block_size, kv_length] a single block
        * with multiple threads as we need to `sync_threads` to run exponential sum.
        * We maximise the usage of threads within a single block
        */
        // TODO @thomasw21 figure out everything warp related:
        //  - why do they have to be power of 2
        const auto MIN_KV_LENGTH_SHARD_SIZE_PER_THREAD = 1;
        const auto MAX_THREADS_PER_SM = 1024; // TODO @thomas21 check why everyone is setting 1024 when officially it's 1024
        const auto ROWS_PER_BLOCK = (MAX_THREADS_PER_SM * MIN_KV_LENGTH_SHARD_SIZE_PER_THREAD) / kv_length;
        // TODO @thomasw21 compute `ceil`.
        const auto NUM_BLOCKS = (batch_size_times_num_heads * q_length - 1) / ROWS_PER_BLOCK + 1;

        dim3 gridDim(NUM_BLOCKS); // Number of blocks that run
        // TODO @thomas21: Maybe this needs to be converted to `MAX_THREADS_PER_SM` and let padding run nowhere.
        dim3 blockDim(ROWS_PER_BLOCK, kv_length); // Number of threads that run per block
        // TODO @thomasw21: Figure out how much I need
        //  - each thread requires `MIN_KV_LENGTH_SHARD_SIZE_PER_THREAD` in memory for each row
        //  - threads has `ROWS_PER_BLOCK` rows.
        const int shared_mem_forward = ROWS_PER_BLOCK * MIN_KV_LENGTH_SHARD_SIZE_PER_THREAD * 2 * sizeof(float);

        // 192 * 2 ** 10
        const auto MAX_L1_MEMORY = 196608;
        const auto MAX_SMs = 108;
        TORCH_CHECK(batch_size_times_num_heads * q_length < MAX_L1_MEMORY, "Shared memory exceeds 192KB limitation.");
        // TORCH_CHECK(gridDim.x * gridDim.y * gridDim.z < MAX_SMs, "A100s only have 108 SMs. Raising as require blocks is bigger.");
        TORCH_CHECK(blockDim.x * blockDim.y * blockDim.z < MAX_THREADS_PER_SM, "A100s only have 2048 threads per block. Raising as require requested threads is higher.");

        forward_masked_softmax_kernel<<<gridDim, blockDim, shared_mem_forward>>>(
            attention_scores.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            attention_mask.packed_accessor32<bool, 3, torch::RestrictPtrTraits>(),
            attention_probs.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            MIN_KV_LENGTH_SHARD_SIZE_PER_THREAD, // number of values to run

        );
    });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        &forward,
        "Bloom GELU mechanism forward (CUDA)"
    );
}