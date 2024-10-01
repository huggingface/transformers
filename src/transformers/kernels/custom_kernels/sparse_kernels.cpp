#include <torch/extension.h>
#include "cuda_launch.h"  

// binding the functions to the PyTorch module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("index_max", &index_max_kernel, "index_max (CUDA)");
    m.def("mm_to_sparse", &mm_to_sparse_kernel, "mm_to_sparse (CUDA)");
    m.def("sparse_dense_mm", &sparse_dense_mm_kernel, "sparse_dense_mm (CUDA)");
    m.def("reduce_sum", &reduce_sum_kernel, "reduce_sum (CUDA)");
    m.def("scatter", &scatter_kernel, "scatter (CUDA)");
}
