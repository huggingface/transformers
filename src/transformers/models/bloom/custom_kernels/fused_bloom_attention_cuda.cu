#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <torch/torch.h>
#include <cub/cub.cuh>
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

/**
* cast to fp32 if in fp16 + mask + softmax computation in fp32 + cast back to original dtype
**/
template<typename attention_scores_scalar>
__global__ void forward_masked_softmax_kernel(
    const torch::PackedTensorAccessor32<attention_scores_scalar, 3, torch::RestrictPtrTraits> attention_scores, // [B, N, D]
    const torch::PackedTensorAccessor32<bool, 3, torch::RestrictPtrTraits> mask, // [B, N, D]
    torch::PackedTensorAccessor32<attention_scores_scalar, 3, torch::RestrictPtrTraits> result // [B, N, D]
) {
    const int batch_id = blockIdx.x;
    const int q_length_id = blockIdx.y;
    const int kv_length_id = threadIdx.x;

    // Specialize BlockReduce
    // 800 refers to CUDA_ARCH
    typedef cub::BlockReduce<float, static_cast<int>(blockDim.x), cub::BLOCK_REDUCE_WARP_REDUCTIONS, static_cast<int>(blockDim.y), static_cast<int>(blockDim.z), 800> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    // Compute mask
    float elt;
    if (mask[batch_id][q_length_id][kv_length_id] == 1) {
        elt = -std::numeric_limits<float>::infinity();
    } else {
        elt = attention_scores[batch_id][q_length_id][kv_length_id];
    }

    // Compute max
    const float max = BlockReduce(temp_storage).Reduce(elt, cub::Max());

    // Compute exp(elt - max) masked
    float exponential;
    if (mask[batch_id][q_length_id][kv_length_id] == 1) {
        exponential = 0;
    } else {
        exponential = std::exp(attention_scores[batch_id][q_length_id][kv_length_id] - max);
    }

    // Compute sum of exponential
    const float exponential_sum = BlockReduce(temp_storage).Sum(elt);

    // Compute softmax
    result[batch_id][q_length_id][kv_length_id] = exponential_sum;
}

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
    const auto three_times_num_heads = 3 * num_heads;
    const auto head_dim = three_times_hidden_size / three_times_num_heads;
    const auto batch_size_times_num_heads = batch_size * num_heads;

    // `split_heads`
    const auto fused_qkv_view = fused_qkv.view({batch_size, q_length, num_heads, three_times_num_heads});
    const auto tensor_list = fused_qkv_view.tensor_split(head_dim, -1);
    const auto query_layer = tensor_list[0].transpose(1, 2).reshape({batch_size_times_num_heads, q_length, three_times_num_heads});
    auto key_layer = tensor_list[1].permute({0, 2, 3, 1}).reshape({batch_size_times_num_heads, three_times_num_heads, q_length});
    auto value_layer = tensor_list[2].transpose(1, 2).reshape({batch_size_times_num_heads, q_length, three_times_num_heads});

    if (layer_past) {
        const auto past_key = (*layer_past).at(0);
        const auto past_value = (*layer_past).at(1);
        key_layer = at::cat({past_key, key_layer}, 2);
        key_layer = at::cat({past_value, value_layer}, 1);
    }

    std::optional<std::vector<at::Tensor>> present;
    if (use_cache) {
        present = {key_layer, value_layer};
    } else {
        present = {};
    }

    auto attention_scores = alibi.baddbmm(query_layer, key_layer, beta, inv_norm_factor);

    torch::Tensor attention_probs;
    if (true) {
        attention_probs = at::empty_like(attention_scores);
        const auto kv_length = key_layer.size(2);
        // TODO @thomasw21: Check that input are both in the correct device + contiguous

        // TODO @thomas21: change by to this as it's cleaner when pytorch 1.13 comes out
        // DISPATCH_CASE_FLOATING_TYPES(key_layer.scalar_type(), "masked_softmax", [&] {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, key_layer.scalar_type(), "masked_softmax", [&] {
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
            * We should split [batch_size_times_num_heads, q_length] in seperate blocks and [kv_length] a single block
            * with multiple threads as we need to `sync_threads` to run exponential sum.
            */
            // TODO @thomasw21: Figure out how much I need exactly
            dim3 gridDim(batch_size_times_num_heads, q_length); // Number of blocks that run
            dim3 blockDim(kv_length); // Number of threads that run per block
            // TODO @thomasw21: Figure out how much I need
            const int shared_mem_forward = kv_length * sizeof(float);

            // 192 * 2 ** 10
            const auto MAX_L1_MEMORY = 196608;
            const auto MAX_SMs = 108;
            const auto MAX_THREADS_PER_SM = 2048;
            TORCH_CHECK(batch_size_times_num_heads * q_length < MAX_L1_MEMORY, "Shared memory exceeds 192KB limitation.");
            TORCH_CHECK(gridDim.x * gridDim.y * gridDim.z < MAX_SMs, "A100s only have 108 SMs. Raising as require blocks is bigger.");
            TORCH_CHECK(blockDim.x * blockDim.y * blockDim.z < MAX_THREADS_PER_SM, "A100s only have 2048 threads per block. Raising as require requested threads is higher.");

            forward_masked_softmax_kernel<<<gridDim, blockDim, shared_mem_forward>>>(
                attention_scores.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                attention_mask.packed_accessor32<bool, 3, torch::RestrictPtrTraits>(),
                attention_probs.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>()
            );
        });
    } else {
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