#include <torch/extension.h>

torch::Tensor decompress_residuals_cuda(
    const torch::Tensor binary_residuals, const torch::Tensor bucket_weights,
    const torch::Tensor reversed_bit_map,
    const torch::Tensor bucket_weight_combinations, const torch::Tensor codes,
    const torch::Tensor centroids, const int dim, const int nbits);

torch::Tensor decompress_residuals(
    const torch::Tensor binary_residuals, const torch::Tensor bucket_weights,
    const torch::Tensor reversed_bit_map,
    const torch::Tensor bucket_weight_combinations, const torch::Tensor codes,
    const torch::Tensor centroids, const int dim, const int nbits) {
    // Add input verification
    return decompress_residuals_cuda(
        binary_residuals, bucket_weights, reversed_bit_map,
        bucket_weight_combinations, codes, centroids, dim, nbits);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decompress_residuals_cpp", &decompress_residuals,
          "Decompress residuals");
}
