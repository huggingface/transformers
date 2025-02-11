#include <torch/extension.h>
#include <ATen/ATen.h>
#include "cuda_launch.h"
#include "cuda_kernel.h"
#include <vector>

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<at::Tensor> index_max_kernel(
  at::Tensor index_vals,  // [batch_size, 32, num_block]
  at::Tensor indices,     // [batch_size, num_block],
  int A_num_block,
  int B_num_block
) {
  int batch_size = indices.size(0);
  int num_block = indices.size(1);

  at::Tensor max_vals = at::zeros({batch_size, A_num_block * 32}, index_vals.options());
  at::Tensor max_vals_scatter = at::zeros({batch_size, 32, num_block}, index_vals.options());

  dim3 threads(256);
  dim3 blocks(batch_size);
  int shared_mem = A_num_block * 32 * sizeof(float);

  index_max_cuda_kernel<<<blocks, threads, shared_mem>>>(
    index_vals.data_ptr<float>(),
    indices.data_ptr<int>(),
    max_vals.data_ptr<float>(),
    max_vals_scatter.data_ptr<float>(),
    batch_size,
    A_num_block,
    B_num_block,
    num_block
  );

  return {max_vals, max_vals_scatter};
}

at::Tensor mm_to_sparse_kernel(
  at::Tensor dense_A,  // [batch_size, A_num_block, dim, 32]
  at::Tensor dense_B,  // [batch_size, B_num_block, dim, 32]
  at::Tensor indices   // [batch_size, num_block]
) {
  int batch_size = dense_A.size(0);
  int A_num_block = dense_A.size(1);
  int B_num_block = dense_B.size(1);
  int dim = dense_A.size(2);
  int num_block = indices.size(1);

  at::Tensor sparse_C = at::zeros({batch_size, num_block, 32, 32}, dense_A.options());

  dim3 threads(64, 4);
  dim3 blocks(num_block / 4, batch_size);

  mm_to_sparse_cuda_kernel<<<blocks, threads>>>(
    dense_A.data_ptr<float>(),
    dense_B.data_ptr<float>(),
    indices.data_ptr<int>(),
    sparse_C.data_ptr<float>(),
    batch_size,
    A_num_block,
    B_num_block,
    dim,
    num_block
  );

  return sparse_C;
}

at::Tensor sparse_dense_mm_kernel(
  at::Tensor sparse_A,  // [batch_size, num_block, 32, 32]
  at::Tensor indices,   // [batch_size, num_block]
  at::Tensor dense_B,   // [batch_size, B_num_block, dim, 32]
  int A_num_block
) {
  int batch_size = sparse_A.size(0);
  int num_block = sparse_A.size(1);
  int B_num_block = dense_B.size(1);
  int dim = dense_B.size(2);

  at::Tensor dense_C = at::zeros({batch_size, A_num_block, dim, 32}, dense_B.options());

  dim3 threads(128, 2);
  dim3 blocks(num_block / 2, batch_size);

  sparse_dense_mm_cuda_kernel<<<blocks, threads>>>(
    sparse_A.data_ptr<float>(),
    indices.data_ptr<int>(),
    dense_B.data_ptr<float>(),
    dense_C.data_ptr<float>(),
    batch_size,
    A_num_block,
    B_num_block,
    dim,
    num_block
  );

  return dense_C;
}

at::Tensor reduce_sum_kernel(
  at::Tensor sparse_A,  // [batch_size, num_block, 32, 32]
  at::Tensor indices,   // [batch_size, num_block]
  int A_num_block,
  int B_num_block
) {
  int batch_size = sparse_A.size(0);
  int num_block = sparse_A.size(1);

  at::Tensor dense_C = at::zeros({batch_size, A_num_block, 32}, sparse_A.options());

  dim3 threads(32, 4);
  dim3 blocks(num_block / 4, batch_size);

  reduce_sum_cuda_kernel<<<blocks, threads>>>(
    sparse_A.data_ptr<float>(),
    indices.data_ptr<int>(),
    dense_C.data_ptr<float>(),
    batch_size,
    A_num_block,
    B_num_block,
    num_block
  );

  return dense_C;
}

at::Tensor scatter_kernel(
  at::Tensor dense_A,   // [batch_size, A_num_block, 32]
  at::Tensor indices,   // [batch_size, num_block]
  int B_num_block
) {
  int batch_size = dense_A.size(0);
  int A_num_block = dense_A.size(1);
  int num_block = indices.size(1);

  at::Tensor sparse_C = at::zeros({batch_size, num_block, 32, 32}, dense_A.options());

  dim3 threads(32, 4);
  dim3 blocks(num_block / 4, batch_size);

  scatter_cuda_kernel<<<blocks, threads>>>(
    dense_A.data_ptr<float>(),
    indices.data_ptr<int>(),
    sparse_C.data_ptr<float>(),
    batch_size,
    A_num_block,
    B_num_block,
    num_block
  );

  return sparse_C;
}
