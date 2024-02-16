#include <assert.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>

__global__ void decompress_residuals_kernel(
    const uint8_t* binary_residuals,
    const torch::PackedTensorAccessor32<at::Half, 1, torch::RestrictPtrTraits>
        bucket_weights,
    const torch::PackedTensorAccessor32<uint8_t, 1, torch::RestrictPtrTraits>
        reversed_bit_map,
    const torch::PackedTensorAccessor32<uint8_t, 2, torch::RestrictPtrTraits>
        bucket_weight_combinations,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> codes,
    const torch::PackedTensorAccessor32<at::Half, 2, torch::RestrictPtrTraits>
        centroids,
    const int n, const int dim, const int nbits, const int packed_size,
    at::Half* output) {
    const int packed_dim = (int)(dim * nbits / packed_size);
    const int i = blockIdx.x;
    const int j = threadIdx.x;

    if (i >= n) return;
    if (j >= dim * nbits / packed_size) return;

    const int code = codes[i];

    uint8_t x = binary_residuals[i * packed_dim + j];
    x = reversed_bit_map[x];
    int output_idx = (int)(j * packed_size / nbits);
    for (int k = 0; k < packed_size / nbits; k++) {
        assert(output_idx < dim);
        const int bucket_weight_idx = bucket_weight_combinations[x][k];
        output[i * dim + output_idx] = bucket_weights[bucket_weight_idx];
        output[i * dim + output_idx] += centroids[code][output_idx];
        output_idx++;
    }
}

torch::Tensor decompress_residuals_cuda(
    const torch::Tensor binary_residuals, const torch::Tensor bucket_weights,
    const torch::Tensor reversed_bit_map,
    const torch::Tensor bucket_weight_combinations, const torch::Tensor codes,
    const torch::Tensor centroids, const int dim, const int nbits) {
    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat16)
                       .device(torch::kCUDA, 0)
                       .requires_grad(false);
    torch::Tensor output =
        torch::zeros({(int)binary_residuals.size(0), (int)dim}, options);

    // TODO: Set this automatically?
    const int packed_size = 8;

    const int threads = dim / (packed_size / nbits);
    const int blocks =
        (binary_residuals.size(0) * binary_residuals.size(1)) / threads;

    decompress_residuals_kernel<<<blocks, threads>>>(
        binary_residuals.data<uint8_t>(),
        bucket_weights
            .packed_accessor32<at::Half, 1, torch::RestrictPtrTraits>(),
        reversed_bit_map
            .packed_accessor32<uint8_t, 1, torch::RestrictPtrTraits>(),
        bucket_weight_combinations
            .packed_accessor32<uint8_t, 2, torch::RestrictPtrTraits>(),
        codes.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        centroids.packed_accessor32<at::Half, 2, torch::RestrictPtrTraits>(),
        binary_residuals.size(0), dim, nbits, packed_size,
        output.data<at::Half>());

    return output;
}
