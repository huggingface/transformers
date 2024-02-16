#include <torch/extension.h>

torch::Tensor packbits_cuda(const torch::Tensor residuals);

torch::Tensor packbits(const torch::Tensor residuals) {
    return packbits_cuda(residuals);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("packbits_cpp", &packbits, "Pack bits");
}

