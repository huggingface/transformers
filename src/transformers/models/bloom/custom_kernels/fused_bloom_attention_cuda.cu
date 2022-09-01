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
* cast to fp32 if in fp16 + mask + softmax computation in fp32 + cast back to original dtype
**/
template<typename attention_scores_scalar, int64_t min_kv_length_shard_size_per_thread>
__global__ void forward_masked_softmax_kernel(
    const torch::PackedTensorAccessor32<attention_scores_scalar, 2, torch::RestrictPtrTraits> attention_scores, // [B, KV]
    const torch::PackedTensorAccessor32<bool, 2, torch::RestrictPtrTraits> mask, // [B, KV]
    torch::PackedTensorAccessor32<attention_scores_scalar, 2, torch::RestrictPtrTraits> result, // [B, KV]
    const int64_t effective_kv_length,
    const dim3 blockDim,
    const int64_t rows_per_block,
    const int64_t kv_length,
    const int64_t batch_size
) {
    const auto row_id = threadIdx.x / effective_kv_length;
    const auto effective_kv_length_id = threadIdx.x % effective_kv_length;
    const auto kv_length_start = effective_kv_length_id * min_kv_length_shard_size_per_thread;
    auto kv_length_end_ = (effective_kv_length_id + 1) * min_kv_length_shard_size_per_thread;
    kv_length_end_ = (kv_length_end_ > kv_length) ? kv_length : kv_length_end_;
    const auto kv_length_end = kv_length_end_;

    const auto batch_id = blockIdx.x * rows_per_block + row_id;

    // We need 2 float storage for each row, one for max computation, the other for normalizing exponential
    extern __shared__ float temp_storage[];
    const auto row_id_mem_offset = row_id * 2;
    if (effective_kv_length_id == 0) {
        temp_storage[row_id_mem_offset] = -std::numeric_limits<float>::infinity();
        temp_storage[row_id_mem_offset + 1] = 0;
    }
    __syncthreads();

    // Compute mask and max
    if (batch_id < batch_size) {
        float thread_max = -std::numeric_limits<float>::infinity();
        for (int kv_length_id = kv_length_start; kv_length_id < kv_length_end; ++kv_length_id) {
            if (mask[batch_id][kv_length_id] == 0) {
                const float candidate = attention_scores[batch_id][kv_length_id];
                thread_max = (thread_max < candidate) ? candidate : thread_max;
            }
        }
        if (thread_max != -std::numeric_limits<float>::infinity()) {
            // TODO @thomasw21 with more memory we can probably compute a much faster `max-reduce` in parallel O(ln(n)) operations in each memory slot
            gpuAtomicMax(&temp_storage[row_id_mem_offset], thread_max);
        }
    }

    __syncthreads();

    // Compute exp(elt - max) masked
    float exponential[min_kv_length_shard_size_per_thread];
    if (batch_id < batch_size) {
        float thread_add = 0;
        for (int kv_length_id = kv_length_start; kv_length_id < kv_length_end; ++kv_length_id) {
            if (mask[batch_id][kv_length_id] == 0) {
                exponential[kv_length_id - kv_length_start] = std::exp(static_cast<float>(attention_scores[batch_id][kv_length_id]) - temp_storage[row_id_mem_offset]);
                thread_add = thread_add + exponential[kv_length_id - kv_length_start];
            } else {
                exponential[kv_length_id - kv_length_start] = 0.;
            }
        }
        if (thread_add > 0) {
            // TODO @thomasw21 with more memory we can probably compute a much faster `sum-reduce` in parallel O(ln(n)) operations in each memory slot
            gpuAtomicAdd(&temp_storage[row_id_mem_offset + 1], thread_add);
        }
    }

    __syncthreads();

    // Compute softmax
    if (batch_id < batch_size) {
        // If sum of all exponential is 0, we set the softmax values to 0
        if (temp_storage[row_id_mem_offset + 1] == 0.) {
            for (int kv_length_id = kv_length_start; kv_length_id < kv_length_end; ++kv_length_id) {
                result[batch_id][kv_length_id] = 0.;
            }
        } else {
            for (int kv_length_id = kv_length_start; kv_length_id < kv_length_end; ++kv_length_id) {
                result[batch_id][kv_length_id] = static_cast<attention_scores_scalar>(exponential[kv_length_id - kv_length_start] / temp_storage[row_id_mem_offset + 1]);
            }
        }
    }
}

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::tuple<at::Tensor, std::optional<std::vector<at::Tensor>>, at::Tensor> forward(
    const at::Tensor fused_qkv,
    const std::optional<std::vector<at::Tensor>> layer_past,
    const at::Tensor alibi,
    const at::Tensor attention_mask,
    const std::optional<at::Tensor> head_mask,
    const float beta,
    const float inv_norm_factor,
    const int num_heads,
    const bool use_cache
) {
    const auto batch_size = fused_qkv.size(0);
    const auto q_length = fused_qkv.size(1);
    const auto three_times_hidden_size = fused_qkv.size(2);
    const auto head_dim = three_times_hidden_size / (3 * num_heads);
    const auto batch_size_times_num_heads = batch_size * num_heads;

    // `split_heads`
    const auto fused_qkv_view = fused_qkv.view({batch_size, q_length, num_heads, 3 * head_dim});
    const auto tensor_list = fused_qkv_view.split(head_dim, -1);
    const auto query_layer = tensor_list[0].transpose(1, 2).reshape({batch_size_times_num_heads, q_length, head_dim});
    auto key_layer = tensor_list[1].permute({0, 2, 3, 1}).reshape({batch_size_times_num_heads, head_dim, q_length});
    auto value_layer = tensor_list[2].transpose(1, 2).reshape({batch_size_times_num_heads, q_length, head_dim});

    if (layer_past) {
        const auto past_key = (*layer_past).at(0);
        const auto past_value = (*layer_past).at(1);
        key_layer = at::cat({past_key, key_layer}, 2);
        value_layer = at::cat({past_value, value_layer}, 1);
    }

    std::optional<std::vector<at::Tensor>> present;
    if (use_cache) {
        present = {key_layer, value_layer};
    } else {
        present = {};
    }

    auto attention_scores = alibi.baddbmm(query_layer, key_layer, beta, inv_norm_factor);

    // Computing `optionally_cast_fp16_to_fp32 + masked_fill + softmax + cast_to_intial_dtype`
    at::Tensor attention_probs;
    if (true) {
        const auto kv_length = key_layer.size(2);

        // TODO @thomasw21: it's easier to think of attention_scores as 2D tensors
        const auto attention_scores_2d = attention_scores.view({batch_size_times_num_heads * q_length, kv_length});
        const auto attention_mask_2d = attention_mask.view({batch_size_times_num_heads * q_length, kv_length});

        // Custom kernel
        attention_probs = at::empty_like(attention_scores_2d);

        // Check that inputs and contiguous + cuda tensors
        CHECK_INPUT(attention_scores_2d);
        CHECK_INPUT(attention_mask_2d);

        // TODO @thomas21: change by to this as it's cleaner when pytorch 1.13 comes out
        // DISPATCH_CASE_FLOATING_TYPES(attention_scores.scalar_type(), "masked_softmax", [&] {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, attention_scores.scalar_type(), "masked_softmax", [&] {
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
            // TODO @thomas21 check why everyone is setting 1024 when officially it's 2048
            const auto MAX_THREADS_PER_SM = 1024;
            // TODO @thomasw21 figure out how to have longer sequences, currently the maximum is `max_kv_length = MAX_THREADS_PER_SM * MIN_KV_LENGTH_SHARD_SIZE_PER_THREAD`
            const auto MIN_KV_LENGTH_SHARD_SIZE_PER_THREAD = 4;
            // `effective_kv_length = ceil(kv_length / MIN_KV_LENGTH_SHARD_SIZE_PER_THREAD)`
            const auto effective_kv_length = (kv_length - 1)/ MIN_KV_LENGTH_SHARD_SIZE_PER_THREAD + 1;
            const auto rows_per_block = MAX_THREADS_PER_SM / effective_kv_length;
            const auto num_blocks = (batch_size_times_num_heads * q_length - 1) / rows_per_block + 1;

            const dim3 gridDim(num_blocks); // Number of blocks that run
            const dim3 blockDim(MAX_THREADS_PER_SM); // Number of threads that run per block
            const int shared_mem_forward = rows_per_block * 2 * sizeof(float);

            // 192 * 2 ** 10
            // const auto MAX_L1_MEMORY = 196608;
            // const auto MAX_SMs = 108;
            // TORCH_CHECK(batch_size_times_num_heads * q_length <= MAX_L1_MEMORY, "Shared memory exceeds 192KB limitation.");
            // TORCH_CHECK(gridDim.x * gridDim.y * gridDim.z <= MAX_SMs, "A100s only have 108 SMs. Raising as require blocks is bigger.");
            // TORCH_CHECK(blockDim.x * blockDim.y * blockDim.z <= MAX_THREADS_PER_SM, "A100s only have 2048 threads per block. Raising as require requested threads is higher.");

            forward_masked_softmax_kernel<scalar_t, MIN_KV_LENGTH_SHARD_SIZE_PER_THREAD><<<gridDim, blockDim, shared_mem_forward>>>(
                attention_scores_2d.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                attention_mask_2d.packed_accessor32<bool, 2, torch::RestrictPtrTraits>(),
                attention_probs.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                effective_kv_length,
                blockDim,
                rows_per_block,
                kv_length,
                batch_size_times_num_heads * q_length
            );
        });
        attention_probs = attention_probs.view({batch_size_times_num_heads, q_length, kv_length});
    } else {
        // Pytorch C++ API
        auto input_dtype = attention_scores.scalar_type();
        if (input_dtype == at::ScalarType::Float) {
            attention_scores = attention_scores.to(at::ScalarType::Float);
        };
        // TODO @thomasw21 Figure out how to get minimum value
        auto attn_weights = attention_scores.masked_fill_(attention_mask, -1e34);
        attention_probs = attn_weights.softmax(-1, at::ScalarType::Float).to(input_dtype);
    }

    auto context_layer = attention_probs.bmm(value_layer);

    // `_merge_heads`
    context_layer = context_layer.view({batch_size, num_heads, q_length, head_dim});
    context_layer = context_layer.permute({0, 2, 1, 3});
    context_layer = context_layer.reshape({batch_size, q_length, three_times_hidden_size / 3});

    return std::make_tuple(context_layer, present, attention_probs);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        &forward,
        "Bloom attention mechanism forward (CUDA)"
    );
}