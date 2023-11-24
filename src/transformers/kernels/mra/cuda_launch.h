#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>

#define min(a, b) ((a)<(b)?(a):(b))
#define max(a, b) ((a)>(b)?(a):(b))

std::vector<at::Tensor> index_max_kernel(
  at::Tensor index_vals,
  at::Tensor indices,
  int A_num_block,
  int B_num_block
);

at::Tensor mm_to_sparse_kernel(
  at::Tensor dense_A,
  at::Tensor dense_B,
  at::Tensor indices
);

at::Tensor sparse_dense_mm_kernel(
  at::Tensor sparse_A,
  at::Tensor indices,
  at::Tensor dense_B,
  int A_num_block
);

at::Tensor reduce_sum_kernel(
  at::Tensor sparse_A,
  at::Tensor indices,
  int A_num_block,
  int B_num_block
);

at::Tensor scatter_kernel(
  at::Tensor dense_A,
  at::Tensor indices,
  int B_num_block
);
