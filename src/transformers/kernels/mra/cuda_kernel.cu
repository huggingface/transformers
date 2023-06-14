#include "cuda_kernel.h"

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void index_max_cuda_kernel(
  float *index_vals,       // [batch_size, 32, num_block]
  int   *indices,        // [batch_size, num_block]
  float *max_vals,        // [batch_size, A_num_block * 32]
  float *max_vals_scatter,   // [batch_size, 32, num_block]
  long batch_size,
  long A_num_block,
  long B_num_block,
  long num_block
) {

  long batch_idx = blockIdx.x;

  long thread_idx = threadIdx.x;
  long num_thread = blockDim.x;

  extern __shared__ float buffer[];
  int *max_buffer = (int*)buffer;

  for (int i = 0; i < A_num_block * 32; i = i + num_thread) {
    int idx = i + thread_idx;
    if (idx < A_num_block * 32) {
      max_buffer[idx] = -1e8;
    }
  }
  __syncthreads();

  int *indices_pt = &indices[batch_idx * num_block];
  float *index_vals_pt = &index_vals[batch_idx * num_block * 32];

  for (int idx_start = 0; idx_start < 32 * num_block; idx_start = idx_start + num_thread) {
    int idx = idx_start + thread_idx;
    int A_block_idx = indices_pt[idx % num_block] / B_num_block;
    atomicMax(&max_buffer[A_block_idx * 32 + idx / num_block], (int)(index_vals_pt[idx] * 1000));
  }
  __syncthreads();
  
  float *max_vals_pt = &max_vals[batch_idx * A_num_block * 32];
  for (int i = 0; i < A_num_block * 32; i = i + num_thread) {
    int idx = i + thread_idx;
    if (idx < A_num_block * 32) {
      max_vals_pt[idx] = (float)max_buffer[idx] / 1000.;
    }
  }
  
  float *max_vals_scatter_pt = &max_vals_scatter[batch_idx * num_block * 32];
  for (int idx_start = 0; idx_start < 32 * num_block; idx_start = idx_start + num_thread) {
    int idx = idx_start + thread_idx;
    int A_block_idx = indices_pt[idx % num_block] / B_num_block;
    max_vals_scatter_pt[idx] = (float)max_buffer[A_block_idx * 32 + idx / num_block] / 1000.;
  }

}

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
) {

  long batch_idx = blockIdx.y;
  long block_idx = blockIdx.x * blockDim.y + threadIdx.y;

  long thread_idx = threadIdx.x;

  __shared__ float buffer[4096];
  float *A_buffer = &buffer[threadIdx.y * 1024]; // [2, 8, 32]
  float *B_buffer = &buffer[threadIdx.y * 1024 + 512]; // [2, 8, 32]

  long batch_idx__block_idx = batch_idx * num_block + block_idx;

  long AB_block_idx = indices[batch_idx__block_idx];
  float *dense_A_pt = &dense_A[(batch_idx * A_num_block + AB_block_idx / B_num_block) * dim * 32];
  float *dense_B_pt = &dense_B[(batch_idx * B_num_block + AB_block_idx % B_num_block) * dim * 32];

  int reg_1_idx = thread_idx / 8;    // [0000000011111111222222223333333344444444555555556666666677777777]
  int reg_2_idx = thread_idx % 8;    // [0123456701234567012345670123456701234567012345670123456701234567]

  float reg_1[8];
  float reg_2[8];

  float reg_array[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  #pragma unroll
  for (int i = 0; i < 4; i++) {
    A_buffer[i * 64 + thread_idx] = dense_A_pt[i * 64 + thread_idx];
    B_buffer[i * 64 + thread_idx] = dense_B_pt[i * 64 + thread_idx];
  }

  __syncthreads();

  #pragma unroll
  for (int i = 0; i < 4; i++) {
    reg_1[i] = A_buffer[reg_1_idx * 4 + i];
    reg_2[i] = B_buffer[reg_2_idx * 4 + i];
  }

  for (int dim_stride = 1; dim_stride < (dim / 8); dim_stride++) {

    #pragma unroll
    for (int i = 0; i < 4; i++) {
      A_buffer[(dim_stride % 2) * 256 + i * 64 + thread_idx] = dense_A_pt[dim_stride * 256 + i * 64 + thread_idx];
      B_buffer[(dim_stride % 2) * 256 + i * 64 + thread_idx] = dense_B_pt[dim_stride * 256 + i * 64 + thread_idx];
    }

    #pragma unroll
    for (int mini_dim_idx = 1; mini_dim_idx < 8; mini_dim_idx++) {
      #pragma unroll
      for (int i = 0; i < 4; i++) {
        reg_1[(mini_dim_idx % 2) * 4 + i] = A_buffer[((dim_stride - 1) % 2) * 256 + mini_dim_idx * 32 + reg_1_idx * 4 + i];
        reg_2[(mini_dim_idx % 2) * 4 + i] = B_buffer[((dim_stride - 1) % 2) * 256 + mini_dim_idx * 32 + reg_2_idx * 4 + i];
      }
      #pragma unroll
      for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
          reg_array[i * 4 + j] += reg_1[((mini_dim_idx - 1) % 2) * 4 + i] * reg_2[((mini_dim_idx - 1) % 2) * 4 + j];
        }
      }
    }

    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 4; i++) {
      reg_1[i] = A_buffer[(dim_stride % 2) * 256 + reg_1_idx * 4 + i];
      reg_2[i] = B_buffer[(dim_stride % 2) * 256 + reg_2_idx * 4 + i];
    }

    #pragma unroll
    for (int i = 0; i < 4; i++) {
      #pragma unroll
      for (int j = 0; j < 4; j++) {
        reg_array[i * 4 + j] += reg_1[4 + i] * reg_2[4 + j];
      }
    }

  }

  #pragma unroll
  for (int mini_dim_idx = 1; mini_dim_idx < 8; mini_dim_idx++) {
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      reg_1[(mini_dim_idx % 2) * 4 + i] = A_buffer[256 + mini_dim_idx * 32 + reg_1_idx * 4 + i];
      reg_2[(mini_dim_idx % 2) * 4 + i] = B_buffer[256 + mini_dim_idx * 32 + reg_2_idx * 4 + i];
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      #pragma unroll
      for (int j = 0; j < 4; j++) {
        reg_array[i * 4 + j] += reg_1[((mini_dim_idx - 1) % 2) * 4 + i] * reg_2[((mini_dim_idx - 1) % 2) * 4 + j];
      }
    }
  }
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    #pragma unroll
    for (int j = 0; j < 4; j++) {
      reg_array[i * 4 + j] += reg_1[4 + i] * reg_2[4 + j];
    }
  }
  __syncthreads();

  float *C_buffer = &buffer[threadIdx.y * 1024]; // [32, 32]

  #pragma unroll
  for (int i = 0; i < 4; i++) {
    #pragma unroll
    for (int j = 0; j < 4; j++) {
      C_buffer[(reg_2_idx * 4 + j) * 32 + reg_1_idx * 4 + i] = reg_array[i * 4 + j];
    }
  }
  __syncthreads();

  float *sparse_C_pt = &sparse_C[batch_idx__block_idx * 1024];

  #pragma unroll
  for (int i = 0; i < 16; i++) {
    sparse_C_pt[i * 64 + thread_idx] = C_buffer[i * 64 + thread_idx];
  }

}

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
) {

  long batch_idx = blockIdx.y;
  long block_idx = blockIdx.x * blockDim.y + threadIdx.y;

  long thread_idx = threadIdx.x;

  __shared__ float buffer[6144];
  float *A_buffer = &buffer[threadIdx.y * 3072]; // [32, 32]
  float *B_buffer = &buffer[threadIdx.y * 3072 + 1024]; // [32, 64]

  long batch_idx__block_idx = batch_idx * num_block + block_idx;

  float *sparse_A_pt = &sparse_A[batch_idx__block_idx * 1024];
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    A_buffer[i * 128 + thread_idx] = sparse_A_pt[i * 128 + thread_idx];
  }

  long AB_block_idx = indices[batch_idx__block_idx];
  float *dense_B_pt = &dense_B[(batch_idx * B_num_block + AB_block_idx % B_num_block) * 32 * dim];
  float *dense_C_pt = &dense_C[(batch_idx * A_num_block + AB_block_idx / B_num_block) * 32 * dim];

  // [0000000011111111222222223333333344444444555555556666666677777777]
  // [0123456701234567012345670123456701234567012345670123456701234567]
  int reg_1_idx = thread_idx / 8;
  int reg_2_idx = thread_idx % 8;

  float reg_1[8];
  float reg_2[8];

  float reg_array[16];

  for (int dim_stride = 0; dim_stride < dim; dim_stride = dim_stride + 64) {

    #pragma unroll
    for (int i = 0; i < 16; i++) {
      B_buffer[i * 128 + thread_idx] = dense_B_pt[dim_stride * 32 + i * 128 + thread_idx];
    }

    #pragma unroll
    for (int i = 0; i < 16; i++) {
      reg_array[i] = 0;
    }

    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 4; i++) {
      reg_1[i] = B_buffer[(reg_1_idx * 4 + i) * 32];
      reg_2[i] = A_buffer[reg_2_idx * 4 + i];
    }

    #pragma unroll
    for (int mini_dim_idx = 1; mini_dim_idx < 32; mini_dim_idx++) {
      #pragma unroll
      for (int i = 0; i < 4; i++) {
        reg_1[(mini_dim_idx % 2) * 4 + i] = B_buffer[(reg_1_idx * 4 + i) * 32 + mini_dim_idx];
        reg_2[(mini_dim_idx % 2) * 4 + i] = A_buffer[mini_dim_idx * 32 + reg_2_idx * 4 + i];
      }
      #pragma unroll
      for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
          reg_array[i * 4 + j] += reg_1[((mini_dim_idx - 1) % 2) * 4 + i] * reg_2[((mini_dim_idx - 1) % 2) * 4 + j];
        }
      }
    }

    #pragma unroll
    for (int i = 0; i < 4; i++) {
      #pragma unroll
      for (int j = 0; j < 4; j++) {
        reg_array[i * 4 + j] += reg_1[4 + i] * reg_2[4 + j];
      }
    }

    __syncthreads();

    float *C_buffer = &buffer[threadIdx.y * 3072 + 1024]; // [64, 32]

    #pragma unroll
    for (int i = 0; i < 4; i++) {
      #pragma unroll
      for (int j = 0; j < 4; j++) {
        C_buffer[(reg_1_idx * 4 + i) * 32 + reg_2_idx * 4 + j] = reg_array[i * 4 + j];
      }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 16; i++) {
      atomicAdd(&dense_C_pt[dim_stride * 32 + i * 128 + thread_idx], C_buffer[i * 128 + thread_idx]);
    }
    __syncthreads();

  }

}


__global__ void reduce_sum_cuda_kernel(
  float *sparse_A,  // [batch_size, num_block, 32, 32]
  int   *indices,   // [batch_size, num_block]
  float *dense_C,   // [batch_size, A_num_block, 32]
  long batch_size,
  long A_num_block,
  long B_num_block,
  long num_block
) {

  long batch_idx = blockIdx.y;
  long block_idx = blockIdx.x * blockDim.y + threadIdx.y;

  long thread_idx = threadIdx.x;

  long batch_idx__block_idx = batch_idx * num_block + block_idx;

  long AB_block_idx = indices[batch_idx__block_idx];
  float *sparse_A_pt = &sparse_A[batch_idx__block_idx * 1024];

  float reg_array[16];
  float value = 0;

  #pragma unroll
  for (int i = 0; i < 8; i++) {
    reg_array[i] = sparse_A_pt[i * 32 + thread_idx];
  }
  #pragma unroll
  for (int stride = 8; stride < 32; stride = stride + 8) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
      reg_array[(stride + i) % 16] = sparse_A_pt[(stride + i) * 32 + thread_idx];
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) {
      value = value + reg_array[(stride - 8 + i) % 16];
    }
  }
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    value = value + reg_array[8 + i];
  }

  float *dense_C_pt = &dense_C[(batch_idx * A_num_block + AB_block_idx / B_num_block) * 32];

  atomicAdd(&dense_C_pt[thread_idx], value);

}

__global__ void scatter_cuda_kernel(
  float *dense_A,   // [batch_size, A_num_block, 32]
  int   *indices,   // [batch_size, num_block]
  float *sparse_C,  // [batch_size, num_block, 32, 32]
  long batch_size,
  long A_num_block,
  long B_num_block,
  long num_block
) {

  long batch_idx = blockIdx.y;
  long block_idx = blockIdx.x * blockDim.y + threadIdx.y;

  long thread_idx = threadIdx.x;

  long batch_idx__block_idx = batch_idx * num_block + block_idx;

  long AB_block_idx = indices[batch_idx__block_idx];
  float *dense_A_pt = &dense_A[(batch_idx * A_num_block + AB_block_idx / B_num_block) * 32];
  float *sparse_C_pt = &sparse_C[(batch_idx * num_block + block_idx) * 1024];

  float value = dense_A_pt[thread_idx];

  #pragma unroll
  for (int i = 0; i < 32; i++) {
    sparse_C_pt[i * 32 + thread_idx] = value;
  }

}
