#include <torch/extension.h>
#include <ATen/ATen.h>
#include "cuda_launch.h"
#include <vector>

std::vector<at::Tensor> index_max(
  at::Tensor index_vals,
  at::Tensor indices,
  int A_num_block,
  int B_num_block
) {
  return index_max_kernel(
    index_vals,
    indices,
    A_num_block,
    B_num_block
  );
}

at::Tensor mm_to_sparse(
  at::Tensor dense_A,
  at::Tensor dense_B,
  at::Tensor indices
) {
  return mm_to_sparse_kernel(
    dense_A,
    dense_B,
    indices
  );
}

at::Tensor sparse_dense_mm(
  at::Tensor sparse_A,
  at::Tensor indices,
  at::Tensor dense_B,
  int A_num_block
) {
  return sparse_dense_mm_kernel(
    sparse_A,
    indices,
    dense_B,
    A_num_block
  );
}

at::Tensor reduce_sum(
  at::Tensor sparse_A,
  at::Tensor indices,
  int A_num_block,
  int B_num_block
) {
  return reduce_sum_kernel(
    sparse_A,
    indices,
    A_num_block,
    B_num_block
  );
}

at::Tensor scatter(
  at::Tensor dense_A,
  at::Tensor indices,
  int B_num_block
) {
  return scatter_kernel(
    dense_A,
    indices,
    B_num_block
  );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("index_max", &index_max, "index_max (CUDA)");
  m.def("mm_to_sparse", &mm_to_sparse, "mm_to_sparse (CUDA)");
  m.def("sparse_dense_mm", &sparse_dense_mm, "sparse_dense_mm (CUDA)");
  m.def("reduce_sum", &reduce_sum, "reduce_sum (CUDA)");
  m.def("scatter", &scatter, "scatter (CUDA)");
}
