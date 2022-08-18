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
template<typename scalar_t, int64_t max_threads_per_sm>
__global__ void forward_masked_softmax_kernel(
    const scalar_t* __restrict__ x, // [B, N, D]
    scalar_t* __restrict__ result, // [B, N, D]
    int64_t num_params
) {
    const auto id = blockIdx.x * max_threads_per_sm + threadIdx.x;

    if (num_params <= id) {
        return;
    }

    // We upcast to float always
    const float elt = x[id];

    // Compute gelu
    // TODO @thomasw21: Figure out where to find a tanh implementation that works for all kinds of scalar types. (I could hardcode it)
    result[id] = static_cast<scalar_t>(elt * 0.5 * (1.0 + std::tanh(0.79788456 * elt * (1 + 0.044715 * elt * elt))));
}

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor forward(
    const at::Tensor x
) {
    CHECK_INPUT(x);

    const auto result = at::empty_like(x);

    // TODO @thomas21: change by to this as it's cleaner when pytorch 1.13 comes out
    // DISPATCH_CASE_FLOATING_TYPES(key_layer.scalar_type(), "masked_softmax", [&] {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "gelu", [&] {
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
        const auto MAX_THREADS_PER_SM = 1024; // TODO @thomas21 check why everyone is setting 1024 when officially it's 1024
        auto num_params = x.nume(); // TODO @thomasw21 get `x.size()`
        const auto NUM_BLOCKS = (num_params - 1) / MAX_THREADS_PER_SM + 1;

        dim3 gridDim(NUM_BLOCKS); // Number of blocks that run
        dim3 blockDim(MAX_THREADS_PER_SM); // Number of threads that run per block

        // 192 * 2 ** 10
        // const auto MAX_L1_MEMORY = 196608;
        // const auto MAX_SMs = 108;

        forward_masked_softmax_kernel<scalar_t, MAX_THREADS_PER_SM><<<gridDim, blockDim>>>(
            x.data<scalar_t>(),
            result.data<scalar_t>(),
            num_params,
        );
    });

    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        &forward,
        "Bloom GELU mechanism forward (CUDA)"
    );
}