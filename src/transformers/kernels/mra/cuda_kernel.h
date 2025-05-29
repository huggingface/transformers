
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff
#define OPTIMAL_THREADS 256

__global__ void index_max_cuda_kernel(
  float *index_vals,       // [batch_size, 32, num_block]
  int   *indices,        // [batch_size, num_block]
  float *max_vals,        // [batch_size, A_num_block * 32]
  float *max_vals_scatter,   // [batch_size, 32, num_block]
  long batch_size,
  long A_num_block,
  long B_num_block,
  long num_block
);

__global__ void mm_to_sparse_cuda_kernel(
  float *dense_A,   // [batch_size, A_num_block, dim, 32]
  float *dense_B,   // [batch_size, B_num_block, dim, 32]
  int   *indices,   // [batch_size, num_block]
  float *sparse_C,  // [batch_size, num_block, 32, 32]
  long batch_size,
  long A_num_block,
  long B_num_block,
  long dim,
  long num_block
);

__global__ void sparse_dense_mm_cuda_kernel(
  float *sparse_A,  // [batch_size, num_block, 32, 32]
  int   *indices,   // [batch_size, num_block]
  float *dense_B,   // [batch_size, B_num_block, dim, 32]
  float *dense_C,   // [batch_size, A_num_block, dim, 32]
  long batch_size,
  long A_num_block,
  long B_num_block,
  long dim,
  long num_block
);

__global__ void reduce_sum_cuda_kernel(
  float *sparse_A,  // [batch_size, num_block, 32, 32]
  int   *indices,   // [batch_size, num_block]
  float *dense_C,   // [batch_size, A_num_block, 32]
  long batch_size,
  long A_num_block,
  long B_num_block,
  long num_block
);

__global__ void scatter_cuda_kernel(
  float *dense_A,   // [batch_size, A_num_block, 32]
  int   *indices,   // [batch_size, num_block]
  float *sparse_C,  // [batch_size, num_block, 32, 32]
  long batch_size,
  long A_num_block,
  long B_num_block,
  long num_block
);
