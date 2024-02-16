#include <assert.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>

#define FULL_MASK 0xffffffff

__global__ void packbits_kernel(
        const uint8_t* residuals,
        uint8_t* packed_residuals,
        const int residuals_size) {
    const int i = blockIdx.x;
    const int j = threadIdx.x;

    assert(blockDim.x == 32);

    const int residuals_idx = i * blockDim.x + j;
    if (residuals_idx >= residuals_size) {
        return;
    }

    const int packed_residuals_idx = residuals_idx / 8;


    uint32_t mask = __ballot_sync(FULL_MASK, residuals[residuals_idx]);

    mask = __brev(mask);

    if (residuals_idx % 32 == 0) {
        for (int k = 0; k < 4; k++) {
            packed_residuals[packed_residuals_idx + k] =
                (mask >> (8 * (4 - k - 1))) & 0xff;
        }
    }
}

torch::Tensor packbits_cuda(const torch::Tensor residuals) {
    auto options = torch::TensorOptions()
                        .dtype(torch::kUInt8)
                        .device(torch::kCUDA, residuals.device().index())
                        .requires_grad(false);
    assert(residuals.size(0) % 32 == 0);
    torch::Tensor packed_residuals = torch::zeros({int(residuals.size(0) / 8)}, options);

    const int threads = 32;
    const int blocks = std::ceil(residuals.size(0) / (float) threads);

    packbits_kernel<<<blocks, threads>>>(
        residuals.data<uint8_t>(),
        packed_residuals.data<uint8_t>(),
        residuals.size(0)
    );

    return packed_residuals;
}
