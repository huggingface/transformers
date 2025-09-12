// File from https://github.com/mlpen/YOSO/blob/main/encoders/backbones/efficient_attentions/yoso/yoso_v1/cuda/fast_lsh_cumulation_cuda.cu

#include "fast_lsh_cumulation_cuda.h"
#include "common_cuda_device.h"
#include "common_cuda.h"
#include "common.h"
#include <stdio.h>
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void fast_hadamard_transform(float *vector_buffer, int vector_dim, int dim_idx) {
  int stride = vector_dim / 2;
  while (stride > (WARP_SIZE / 2)) {
    __syncthreads();
    int sign = 1 - ((dim_idx / stride) % 2) * 2;
    float val1 = vector_buffer[dim_idx];
    float val2 = vector_buffer[dim_idx + sign * stride];
    __syncthreads();
    vector_buffer[dim_idx] = float(sign) * val1 + val2;
    stride = stride / 2;
  }

  float val = vector_buffer[dim_idx];
  #pragma unroll
  for (stride = (WARP_SIZE / 2); stride > 0; stride = stride / 2) {
    int sign = 1 - ((dim_idx / stride) % 2) * 2;
    val = float(sign) * val + __shfl_xor_sync(FULL_MASK, val, stride);
  }
  vector_buffer[dim_idx] = val;
}

__global__ void fast_hash_ver1_cuda_kernel(
  int *mask,        // [batch_size, num_vector]
  float *vector,    // [batch_size, num_vector, vector_dim]
  int *Dmat,        // [batch_size, 3, num_part, vector_dim]
  int *hash_code,   // [batch_size, num_vector, num_hash_f]
  int batch_size,
  int num_vector,
  int vector_dim,
  int num_part,
  int num_hash_f,
  int hash_code_len
) {

  int batch_idx = blockIdx.z;
  int vector_idx = blockIdx.y;
  int part_idx = blockIdx.x;

  int dim_idx = threadIdx.x;

  int batch_idx__vector_idx = batch_idx * num_vector + vector_idx;
  if (mask[batch_idx__vector_idx] == 0) {
    return;
  }

  extern __shared__ float buffer[];
  float *vector_buffer = buffer;

  vector_buffer[dim_idx] = vector[batch_idx__vector_idx * vector_dim + dim_idx];

  vector_buffer[dim_idx] = vector_buffer[dim_idx] * (float)Dmat[((batch_idx * 3 + 0) * num_part + part_idx) * vector_dim + dim_idx];
  fast_hadamard_transform(vector_buffer, vector_dim, dim_idx);
  vector_buffer[dim_idx] = vector_buffer[dim_idx] * (float)Dmat[((batch_idx * 3 + 1) * num_part + part_idx) * vector_dim + dim_idx];
  fast_hadamard_transform(vector_buffer, vector_dim, dim_idx);
  vector_buffer[dim_idx] = vector_buffer[dim_idx] * (float)Dmat[((batch_idx * 3 + 2) * num_part + part_idx) * vector_dim + dim_idx];
  fast_hadamard_transform(vector_buffer, vector_dim, dim_idx);

  int num_hash_per_part = vector_dim / hash_code_len;
  if (hash_code_len == 8 || hash_code_len == 16) {
    int code = select(vector_buffer[dim_idx] > 0, 1 << (dim_idx % hash_code_len), 0);
    for (int offset = 1; offset < hash_code_len; offset = offset * 2) {
      code += __shfl_xor_sync(FULL_MASK, code, offset);
    }
    if (dim_idx % hash_code_len == 0) {
      int hash_f_idx = part_idx * num_hash_per_part + dim_idx / hash_code_len;
      if (hash_f_idx < num_hash_f) {
        hash_code[batch_idx__vector_idx * num_hash_f + hash_f_idx] = code;
      }
    }
  } else {
    vector_buffer[dim_idx] = select(vector_buffer[dim_idx] > 0, 1 << (dim_idx % hash_code_len), 0);
    __syncthreads();
    if (dim_idx < num_hash_per_part) {
      int code = 0;
      for (int i = 0; i < hash_code_len; i++) {
        code += vector_buffer[dim_idx * hash_code_len + i];
      }
      int hash_f_idx = part_idx * num_hash_per_part + dim_idx;
      if (hash_f_idx < num_hash_f) {
        hash_code[batch_idx__vector_idx * num_hash_f + hash_f_idx] = code;
      }
    }
  }
}

__global__ void lsh_cumulation_ver1_step1_cuda_kernel(
  int *key_mask,           // [batch_size, num_key]
  int *key_hash_code,      // [batch_size, num_key, num_hash_f]
  float *value,            // [batch_size, num_key, value_dim]
  float *hashtable_value,  // [batch_size, num_hash_f, hashtable_capacity, WARP_SIZE]
  int batch_size,
  int num_hash_f,
  int hashtable_capacity,
  int num_key,
  int value_dim,
  int offset_warp
) {

  int warp_thread_idx = threadIdx.x;

  int batch_idx = blockIdx.y;
  int key_idx = blockIdx.x * blockDim.y + threadIdx.y;

  int batch_idx__key_idx = batch_idx * num_key + key_idx;
  if (key_mask[batch_idx__key_idx] == 0) {
    return;
  }

  if (num_hash_f > WARP_SIZE) {
    float warp_value = value[batch_idx__key_idx * value_dim + offset_warp + warp_thread_idx];
    for (int hash_f_start = 0; hash_f_start < num_hash_f; hash_f_start = hash_f_start + WARP_SIZE) {
      int warp_hashcode = key_hash_code[batch_idx__key_idx * num_hash_f + hash_f_start + warp_thread_idx];
      #pragma unroll
      for (int hash_f_offset = 0; hash_f_offset < WARP_SIZE; hash_f_offset++) {
        int current_hashcode = warp_hashcode;
        current_hashcode = __shfl_sync(FULL_MASK, current_hashcode, hash_f_offset);
        int hashtable_idx = (batch_idx * num_hash_f + (hash_f_start + hash_f_offset)) * hashtable_capacity + current_hashcode;
        atomicAdd(&hashtable_value[hashtable_idx * WARP_SIZE + warp_thread_idx], warp_value);
      }
    }
  } else {
    float warp_value = value[batch_idx__key_idx * value_dim + offset_warp + warp_thread_idx];
    int warp_hashcode = 0;
    if (warp_thread_idx < num_hash_f) {
      warp_hashcode = key_hash_code[batch_idx__key_idx * num_hash_f + warp_thread_idx];
    }
    for (int hash_f_idx = 0; hash_f_idx < num_hash_f; hash_f_idx++) {
      int current_hashcode = warp_hashcode;
      current_hashcode = __shfl_sync(FULL_MASK, current_hashcode, hash_f_idx);
      int hashtable_idx = (batch_idx * num_hash_f + hash_f_idx) * hashtable_capacity + current_hashcode;
      atomicAdd(&hashtable_value[hashtable_idx * WARP_SIZE + warp_thread_idx], warp_value);
    }
  }

}

__global__ void lsh_cumulation_ver1_step2_cuda_kernel(
  int *query_mask,         // [batch_size, num_query]
  int *query_hash_code,    // [batch_size, num_query, num_hash_f]
  float *hashtable_value,  // [batch_size, num_hash_f, hashtable_capacity, WARP_SIZE]
  float *cumulation_value, // [batch_size, num_query, value_dim]
  int batch_size,
  int num_hash_f,
  int hashtable_capacity,
  int num_query,
  int value_dim,
  int offset_warp
) {

  int warp_thread_idx = threadIdx.x;

  int batch_idx = blockIdx.y;
  int query_idx = blockIdx.x * blockDim.y + threadIdx.y;

  int batch_idx__query_idx = batch_idx * num_query + query_idx;
  if (query_mask[batch_idx__query_idx] == 0) {
    return;
  }

  if (num_hash_f > WARP_SIZE) {
    float warp_value = 0;
    for (int hash_f_start = 0; hash_f_start < num_hash_f; hash_f_start = hash_f_start + WARP_SIZE) {
      int warp_hashcode = query_hash_code[batch_idx__query_idx * num_hash_f + hash_f_start + warp_thread_idx];
      #pragma unroll
      for (int hash_f_offset = 0; hash_f_offset < WARP_SIZE; hash_f_offset++) {
        int current_hashcode = warp_hashcode;
        current_hashcode = __shfl_sync(FULL_MASK, current_hashcode, hash_f_offset);
        int hashtable_idx = (batch_idx * num_hash_f + (hash_f_start + hash_f_offset)) * hashtable_capacity + current_hashcode;
        warp_value = warp_value + hashtable_value[hashtable_idx * WARP_SIZE + warp_thread_idx];
      }
    }
    cumulation_value[batch_idx__query_idx * value_dim + offset_warp + warp_thread_idx] = warp_value / float(num_hash_f);
  } else {
    float warp_value = 0;
    int warp_hashcode = 0;
    if (warp_thread_idx < num_hash_f) {
      warp_hashcode = query_hash_code[batch_idx__query_idx * num_hash_f + warp_thread_idx];
    }
    for (int hash_f_idx = 0; hash_f_idx < num_hash_f; hash_f_idx++) {
      int current_hashcode = warp_hashcode;
      current_hashcode = __shfl_sync(FULL_MASK, current_hashcode, hash_f_idx);
      int hashtable_idx = (batch_idx * num_hash_f + hash_f_idx) * hashtable_capacity + current_hashcode;
      warp_value = warp_value + hashtable_value[hashtable_idx * WARP_SIZE + warp_thread_idx];
    }
    cumulation_value[batch_idx__query_idx * value_dim + offset_warp + warp_thread_idx] = warp_value / float(num_hash_f);
  }

}

__global__ void lsh_weighted_cumulation_ver1_step1_cuda_kernel(
  int *key_mask,            // [batch_size, num_key]
  int *key_hash_code,       // [batch_size, num_key, num_hash_f]
  float *key_weight,        // [batch_size, num_key, weight_dim]
  float *value,             // [batch_size, num_key, value_dim]
  float *hashtable_value,   // [batch_size, num_hash_f, hashtable_capacity, WARP_SIZE]
  int batch_size,
  int num_hash_f,
  int hashtable_capacity,
  int num_key,
  int value_dim,
  int weight_dim,
  int offset_warp,
  int weight_idx
) {

  int warp_thread_idx = threadIdx.x;

  int batch_idx = blockIdx.y;
  int key_idx = blockIdx.x * blockDim.y + threadIdx.y;

  int batch_idx__key_idx = batch_idx * num_key + key_idx;
  if (key_mask[batch_idx__key_idx] == 0) {
    return;
  }

  if (num_hash_f > WARP_SIZE) {
    float warp_value = key_weight[batch_idx__key_idx * weight_dim + weight_idx] * value[batch_idx__key_idx * value_dim + offset_warp + warp_thread_idx];
    for (int hash_f_start = 0; hash_f_start < num_hash_f; hash_f_start = hash_f_start + WARP_SIZE) {
      int warp_hashcode = key_hash_code[batch_idx__key_idx * num_hash_f + hash_f_start + warp_thread_idx];
      #pragma unroll
      for (int hash_f_offset = 0; hash_f_offset < WARP_SIZE; hash_f_offset++) {
        int current_hashcode = warp_hashcode;
        current_hashcode = __shfl_sync(FULL_MASK, current_hashcode, hash_f_offset);
        int hashtable_idx = (batch_idx * num_hash_f + (hash_f_start + hash_f_offset)) * hashtable_capacity + current_hashcode;
        atomicAdd(&hashtable_value[hashtable_idx * WARP_SIZE + warp_thread_idx], warp_value);
      }
    }
  } else {
    float warp_value = key_weight[batch_idx__key_idx * weight_dim + weight_idx] * value[batch_idx__key_idx * value_dim + offset_warp + warp_thread_idx];
    int warp_hashcode = 0;
    if (warp_thread_idx < num_hash_f) {
      warp_hashcode = key_hash_code[batch_idx__key_idx * num_hash_f + warp_thread_idx];
    }
    for (int hash_f_idx = 0; hash_f_idx < num_hash_f; hash_f_idx++) {
      int current_hashcode = warp_hashcode;
      current_hashcode = __shfl_sync(FULL_MASK, current_hashcode, hash_f_idx);
      int hashtable_idx = (batch_idx * num_hash_f + hash_f_idx) * hashtable_capacity + current_hashcode;
      atomicAdd(&hashtable_value[hashtable_idx * WARP_SIZE + warp_thread_idx], warp_value);
    }
  }

}

__global__ void lsh_weighted_cumulation_ver1_step2_cuda_kernel(
  int *query_mask,          // [batch_size, num_query]
  int *query_hash_code,     // [batch_size, num_query, num_hash_f]
  float *query_weight,      // [batch_size, num_query, weight_dim]
  float *hashtable_value,   // [batch_size, num_hash_f, hashtable_capacity, WARP_SIZE]
  float *cumulation_value,  // [batch_size, num_query, value_dim]
  int batch_size,
  int num_hash_f,
  int hashtable_capacity,
  int num_query,
  int value_dim,
  int weight_dim,
  int offset_warp,
  int weight_idx
) {

  int warp_thread_idx = threadIdx.x;

  int batch_idx = blockIdx.y;
  int query_idx = blockIdx.x * blockDim.y + threadIdx.y;

  int batch_idx__query_idx = batch_idx * num_query + query_idx;
  if (query_mask[batch_idx__query_idx] == 0) {
    return;
  }

  if (num_hash_f > WARP_SIZE) {
    float warp_value = 0;
    for (int hash_f_start = 0; hash_f_start < num_hash_f; hash_f_start = hash_f_start + WARP_SIZE) {
      int warp_hashcode = query_hash_code[batch_idx__query_idx * num_hash_f + hash_f_start + warp_thread_idx];
      #pragma unroll
      for (int hash_f_offset = 0; hash_f_offset < WARP_SIZE; hash_f_offset++) {
        int current_hashcode = warp_hashcode;
        current_hashcode = __shfl_sync(FULL_MASK, current_hashcode, hash_f_offset);
        int hashtable_idx = (batch_idx * num_hash_f + (hash_f_start + hash_f_offset)) * hashtable_capacity + current_hashcode;
        warp_value = warp_value + hashtable_value[hashtable_idx * WARP_SIZE + warp_thread_idx];
      }
    }
    float warp_weight = query_weight[batch_idx__query_idx * weight_dim + weight_idx];
    cumulation_value[batch_idx__query_idx * value_dim + offset_warp + warp_thread_idx] += warp_weight * warp_value / float(num_hash_f);
  } else {
    float warp_value = 0;
    int warp_hashcode = 0;
    if (warp_thread_idx < num_hash_f) {
      warp_hashcode = query_hash_code[batch_idx__query_idx * num_hash_f + warp_thread_idx];
    }
    for (int hash_f_idx = 0; hash_f_idx < num_hash_f; hash_f_idx++) {
      int current_hashcode = warp_hashcode;
      current_hashcode = __shfl_sync(FULL_MASK, current_hashcode, hash_f_idx);
      int hashtable_idx = (batch_idx * num_hash_f + hash_f_idx) * hashtable_capacity + current_hashcode;
      warp_value = warp_value + hashtable_value[hashtable_idx * WARP_SIZE + warp_thread_idx];
    }
    float warp_weight = query_weight[batch_idx__query_idx * weight_dim + weight_idx];
    cumulation_value[batch_idx__query_idx * value_dim + offset_warp + warp_thread_idx] += warp_weight * warp_value / float(num_hash_f);
  }

}

__global__ void count_sort_step1_cuda_kernel(
  int *key_mask,         // [batch_size, num_key]
  int *key_hash_code,    // [batch_size, num_key, num_hash_f]
  int *count_sort_table, // [batch_size, num_hash_f, hashtable_capacity]
  int batch_size,
  int num_hash_f,
  int hashtable_capacity,
  int num_key
) {

  int batch_idx = blockIdx.y;
  int key_idx = blockIdx.x * blockDim.y + threadIdx.y;
  int hash_f_idx = threadIdx.x;

  int batch_idx__key_idx = batch_idx * num_key + key_idx;
  if (key_mask[batch_idx__key_idx] == 0) {
    return;
  }

  int hash_code = key_hash_code[batch_idx__key_idx * num_hash_f + hash_f_idx];
  atomicAdd(&count_sort_table[(batch_idx * num_hash_f + hash_f_idx) * hashtable_capacity + hash_code], 1);

}

__global__ void count_sort_step2_cuda_kernel(
  int *count_sort_table,  // [batch_size, num_hash_f, hashtable_capacity]
  int batch_size,
  int num_hash_f,
  int hashtable_capacity
) {

  int batch_idx = blockIdx.y;
  int hash_f_idx = blockIdx.x;

  int num_threads = blockDim.x;
  int thread_id = threadIdx.x;

  int batch_idx__hash_f_idx = batch_idx * num_hash_f + hash_f_idx;

  extern __shared__ float buffer[];
  int *table_buffer = (int*)buffer;

  if (thread_id == 0) {
    table_buffer[0] = 0;
  }
  copy_data<int>(&count_sort_table[batch_idx__hash_f_idx * hashtable_capacity], &table_buffer[1], hashtable_capacity - 1, num_threads, thread_id);

  for (int table_idx_start = 0; table_idx_start < hashtable_capacity; table_idx_start = table_idx_start + num_threads) {
    int thread_value = table_buffer[table_idx_start + thread_id];
    int next_thread_value = 0;
    for (int offset = 1; offset < WARP_SIZE; offset = offset << 1) {
      next_thread_value = __shfl_up_sync(FULL_MASK, thread_value, offset);
      if (thread_id % WARP_SIZE >= offset) {
        thread_value = thread_value + next_thread_value;
      }
    }
    table_buffer[table_idx_start + thread_id] = thread_value;
  }
  __syncthreads();

  if (hashtable_capacity > WARP_SIZE) {
    if (thread_id < WARP_SIZE) {
      for (int table_idx_start = WARP_SIZE; table_idx_start < hashtable_capacity; table_idx_start = table_idx_start + WARP_SIZE) {
        table_buffer[table_idx_start + thread_id] += table_buffer[table_idx_start - 1];
      }
    }
  }

  copy_data<int>(table_buffer, &count_sort_table[batch_idx__hash_f_idx * hashtable_capacity], hashtable_capacity, num_threads, thread_id);

}


__global__ void count_sort_step3_cuda_kernel(
  int *key_mask,          // [batch_size, num_key]
  int *key_hash_code,     // [batch_size, num_key, num_hash_f]
  int *count_sort_table,  // [batch_size, num_hash_f, hashtable_capacity]
  int *key_sorted_idxes,  // [batch_size, num_hash_f, num_key]
  int batch_size,
  int num_hash_f,
  int hashtable_capacity,
  int num_key
) {

  int batch_idx = blockIdx.y;
  int key_idx = blockIdx.x * blockDim.y + threadIdx.y;
  int hash_f_idx = threadIdx.x;

  int batch_idx__key_idx = batch_idx * num_key + key_idx;
  if (key_mask[batch_idx__key_idx] == 0) {
    return;
  }

  int batch_idx__hash_f_idx = batch_idx * num_hash_f + hash_f_idx;

  int hash_code = key_hash_code[batch_idx__key_idx * num_hash_f + hash_f_idx];
  int sort_idx = atomicAdd(&count_sort_table[batch_idx__hash_f_idx * hashtable_capacity + hash_code], 1);
  key_sorted_idxes[batch_idx__hash_f_idx * num_key + sort_idx] = key_idx;

}

__global__ void extract_query_info_cuda_kernel(
  int *query_mask,       // [batch_size, num_query]
  int *query_hash_code,  // [batch_size, num_query, num_hash_f]
  int *count_sort_table, // [batch_size, num_hash_f, hashtable_capacity]
  int *query_info,       // [batch_size, num_query, 2, num_hash_f]
  int batch_size,
  int num_hash_f,
  int hashtable_capacity,
  int num_query
) {

  int batch_idx = blockIdx.y;
  int query_idx = blockIdx.x * blockDim.y + threadIdx.y;
  int hash_f_idx = threadIdx.x;

  int batch_idx__query_idx = batch_idx * num_query + query_idx;
  if (query_mask[batch_idx__query_idx] == 0) {
    return;
  }

  int hash_code = query_hash_code[batch_idx__query_idx * num_hash_f + hash_f_idx];
  int batch_idx__hash_f_idx__hash_code = (batch_idx * num_hash_f + hash_f_idx) * hashtable_capacity + hash_code;

  int key_offset = select(hash_code == 0, 0, count_sort_table[batch_idx__hash_f_idx__hash_code - 1]);
  int key_count = count_sort_table[batch_idx__hash_f_idx__hash_code] - key_offset;

  query_info[batch_idx__query_idx * 2 * num_hash_f + hash_f_idx] = key_offset;
  query_info[(batch_idx__query_idx * 2 + 1) * num_hash_f + hash_f_idx] = key_count;

}

__global__ void lsh_weighted_cumulation_ver2_step2_cuda_kernel(
  int *query_mask,         // [batch_size, num_query]
  int *query_info,         // [batch_size, num_query, 2, num_hash_f]
  int *key_sorted_idxes,   // [batch_size, num_hash_f, num_key]
  float *query_weight,     // [batch_size, num_query, weight_dim]
  float *key_weight,       // [batch_size, num_key, weight_dim]
  float *value,            // [batch_size, num_key, value_dim]
  float *cumulation_value, // [batch_size, num_query, value_dim]
  int batch_size,
  int num_hash_f,
  int num_query,
  int num_key,
  int value_dim,
  int weight_dim
) {

  int batch_idx = blockIdx.z;
  int hash_f_idx = blockIdx.y;
  int query_idx = blockIdx.x;

  int num_threads = blockDim.y * blockDim.x;
  int thread_id = threadIdx.y * blockDim.x + threadIdx.x;

  int num_warps = blockDim.y;
  int warp_idx = threadIdx.y;
  int warp_thread_idx = threadIdx.x;

  int batch_idx__query_idx = batch_idx * num_query + query_idx;
  if (query_mask[batch_idx__query_idx] == 0) {
    return;
  }

  int key_offset = query_info[batch_idx__query_idx * 2 * num_hash_f + hash_f_idx];
  int key_count = query_info[(batch_idx__query_idx * 2 + 1) * num_hash_f + hash_f_idx];

  if (key_count == 0) {
    return;
  }

  extern __shared__ float buffer[];

  if (key_count == 1) {
    if (warp_idx == 0) {
      int key_idx = key_sorted_idxes[(batch_idx * num_hash_f + hash_f_idx) * num_key + key_offset];
      int batch_idx__key_idx = batch_idx * num_key + key_idx;
      float weight = 0;
      for (int weight_offset = 0; weight_offset < weight_dim; weight_offset = weight_offset + WARP_SIZE) {
        int weight_dim_idx = weight_offset + warp_thread_idx;
        float val = query_weight[batch_idx__query_idx * weight_dim + weight_dim_idx] * key_weight[batch_idx__key_idx * weight_dim + weight_dim_idx];
        #pragma unroll
        for (int offset = 1; offset < WARP_SIZE; offset = offset << 1) {
          val += __shfl_xor_sync(FULL_MASK, val, offset);
        }
        weight = weight + val;
      }
      weight = weight / float(num_hash_f);
      for (int value_offset = 0; value_offset < value_dim; value_offset = value_offset + WARP_SIZE) {
        int value_dim_idx = value_offset + warp_thread_idx;
        float val = value[batch_idx__key_idx * value_dim + value_dim_idx];
        atomicAdd(&cumulation_value[batch_idx__query_idx * value_dim + value_dim_idx], weight * val);
      }
    }
  } else {
    float *weight_buffer = buffer;
    int *key_idxes_buffer = (int*)&buffer[weight_dim];

    copy_data_nonblocking<float>(&query_weight[batch_idx__query_idx * weight_dim], weight_buffer, weight_dim, num_threads, thread_id);

    while (key_count > 0) {
      int work_size = min(WARP_SIZE, key_count);
      copy_data_nonblocking<int>(&key_sorted_idxes[(batch_idx * num_hash_f + hash_f_idx) * num_key + key_offset], key_idxes_buffer, work_size, num_threads, thread_id);
      __syncthreads();
      for (int work_offset = 0; work_offset < WARP_SIZE; work_offset = work_offset + num_warps) {
        int work_idx = work_offset + warp_idx;
        if (work_idx < key_count) {
          int key_idx = key_idxes_buffer[work_idx];
          int batch_idx__key_idx = batch_idx * num_key + key_idx;
          float weight = 0;
          for (int weight_offset = 0; weight_offset < weight_dim; weight_offset = weight_offset + WARP_SIZE) {
            int weight_dim_idx = weight_offset + warp_thread_idx;
            float val = weight_buffer[weight_dim_idx] * key_weight[batch_idx__key_idx * weight_dim + weight_dim_idx];
            #pragma unroll
            for (int offset = 1; offset < WARP_SIZE; offset = offset << 1) {
              val += __shfl_xor_sync(FULL_MASK, val, offset);
            }
            weight = weight + val;
          }
          weight = weight / float(num_hash_f);
          for (int value_offset = 0; value_offset < value_dim; value_offset = value_offset + WARP_SIZE) {
            int value_dim_idx = value_offset + warp_thread_idx;
            float val = value[batch_idx__key_idx * value_dim + value_dim_idx];
            atomicAdd(&cumulation_value[batch_idx__query_idx * value_dim + value_dim_idx], weight * val);
          }
        }
      }
      key_count = key_count - work_size;
      key_offset = key_offset + work_size;
    }
  }

}

__global__ void lsh_weighted_cumulation_ver3_step2_cuda_kernel(
  int *query_sorted_idxes,   // [batch_size, num_hash_f, num_query]
  int *key_mask,             // [batch_size, num_key]
  int *key_info,             // [batch_size, num_key, 2, num_hash_f]
  float *query_weight,       // [batch_size, num_query, weight_dim]
  float *key_weight,         // [batch_size, num_key, weight_dim]
  float *value,              // [batch_size, num_key, value_dim]
  float *cumulation_value,   // [batch_size, num_query, value_dim]
  int batch_size,
  int num_hash_f,
  int num_query,
  int num_key,
  int value_dim,
  int weight_dim
) {

  int batch_idx = blockIdx.z;
  int hash_f_idx = blockIdx.y;
  int key_idx = blockIdx.x;

  int num_threads = blockDim.y * blockDim.x;
  int thread_id = threadIdx.y * blockDim.x + threadIdx.x;

  int num_warps = blockDim.y;
  int warp_idx = threadIdx.y;
  int warp_thread_idx = threadIdx.x;

  int batch_idx__key_idx = batch_idx * num_key + key_idx;
  if (key_mask[batch_idx__key_idx] == 0) {
    return;
  }

  int query_offset = key_info[batch_idx__key_idx * 2 * num_hash_f + hash_f_idx];
  int query_count = key_info[(batch_idx__key_idx * 2 + 1) * num_hash_f + hash_f_idx];

  if (query_count == 0) {
    return;
  }

  extern __shared__ float buffer[];

  if (query_count == 1) {
    if (warp_idx == 0) {
      int query_idx = query_sorted_idxes[(batch_idx * num_hash_f + hash_f_idx) * num_query + query_offset];
      int batch_idx__query_idx = batch_idx * num_query + query_idx;
      float weight = 0;
      for (int weight_offset = 0; weight_offset < weight_dim; weight_offset = weight_offset + WARP_SIZE) {
        int weight_dim_idx = weight_offset + warp_thread_idx;
        float val = key_weight[batch_idx__key_idx * weight_dim + weight_dim_idx] * query_weight[batch_idx__query_idx * weight_dim + weight_dim_idx];
        #pragma unroll
        for (int offset = 1; offset < WARP_SIZE; offset = offset << 1) {
          val += __shfl_xor_sync(FULL_MASK, val, offset);
        }
        weight = weight + val;
      }
      weight = weight / float(num_hash_f);
      for (int value_offset = 0; value_offset < value_dim; value_offset = value_offset + WARP_SIZE) {
        int value_dim_idx = value_offset + warp_thread_idx;
        float val = value[batch_idx__key_idx * value_dim + value_dim_idx];
        atomicAdd(&cumulation_value[batch_idx__query_idx * value_dim + value_dim_idx], weight * val);
      }
    }
  } else {
    float *weight_buffer = buffer;
    float *value_buffer = &buffer[weight_dim];
    int *query_idxes_buffer = (int*)&buffer[weight_dim + value_dim];

    copy_data_nonblocking<float>(&key_weight[batch_idx__key_idx * weight_dim], weight_buffer, weight_dim, num_threads, thread_id);
    copy_data_nonblocking<float>(&value[batch_idx__key_idx * value_dim], value_buffer, value_dim, num_threads, thread_id);

    while (query_count > 0) {
      int work_size = min(WARP_SIZE, query_count);
      copy_data_nonblocking<int>(&query_sorted_idxes[(batch_idx * num_hash_f + hash_f_idx) * num_query + query_offset], query_idxes_buffer, work_size, num_threads, thread_id);
      __syncthreads();
      for (int work_offset = 0; work_offset < WARP_SIZE; work_offset = work_offset + num_warps) {
        int work_idx = work_offset + warp_idx;
        if (work_idx < query_count) {
          int query_idx = query_idxes_buffer[work_idx];
          int batch_idx__query_idx = batch_idx * num_query + query_idx;
          float weight = 0;
          for (int weight_offset = 0; weight_offset < weight_dim; weight_offset = weight_offset + WARP_SIZE) {
            int weight_dim_idx = weight_offset + warp_thread_idx;
            float val = weight_buffer[weight_dim_idx] * query_weight[batch_idx__query_idx * weight_dim + weight_dim_idx];
            #pragma unroll
            for (int offset = 1; offset < WARP_SIZE; offset = offset << 1) {
              val += __shfl_xor_sync(FULL_MASK, val, offset);
            }
            weight = weight + val;
          }
          weight = weight / float(num_hash_f);
          for (int value_offset = 0; value_offset < value_dim; value_offset = value_offset + WARP_SIZE) {
            int value_dim_idx = value_offset + warp_thread_idx;
            float val = value_buffer[value_dim_idx];
            atomicAdd(&cumulation_value[batch_idx__query_idx * value_dim + value_dim_idx], weight * val);
          }
        }
      }
      query_count = query_count - work_size;
      query_offset = query_offset + work_size;
    }
  }

}

__global__ void lsh_weighted_cumulation_ver4_step2_cuda_kernel(
  int *query_sorted_idxes,   // [batch_size, num_hash_f, num_query]
  int *key_mask,             // [batch_size, num_key]
  int *key_info,             // [batch_size, num_key, 2, num_hash_f]
  float *query_weight,       // [batch_size, num_query, weight_dim]
  float *key_weight,         // [batch_size, num_key, weight_dim]
  float *value,              // [batch_size, num_key, value_dim]
  float *cumulation_value,   // [batch_size, num_query, value_dim]
  int batch_size,
  int num_hash_f,
  int num_query,
  int num_key,
  int value_dim,
  int weight_dim
) {

  int batch_idx = blockIdx.y;
  int key_idx = blockIdx.x;

  int num_threads = blockDim.y * blockDim.x;
  int thread_id = threadIdx.y * blockDim.x + threadIdx.x;

  int num_warps = blockDim.y;
  int warp_idx = threadIdx.y;
  int warp_thread_idx = threadIdx.x;

  int batch_idx__key_idx = batch_idx * num_key + key_idx;
  if (key_mask[batch_idx__key_idx] == 0) {
    return;
  }

  extern __shared__ float buffer[];
  float *weight_buffer = buffer;
  float *value_buffer = &buffer[weight_dim];
  int *key_info_buffer = (int*)&buffer[weight_dim + value_dim];

  copy_data_nonblocking<float>(&key_weight[batch_idx__key_idx * weight_dim], weight_buffer, weight_dim, num_threads, thread_id);
  copy_data_nonblocking<float>(&value[batch_idx__key_idx * value_dim], value_buffer, value_dim, num_threads, thread_id);
  copy_data_nonblocking<int>(&key_info[batch_idx__key_idx * 2 * num_hash_f], key_info_buffer, 2 * num_hash_f, num_threads, thread_id);

  int *query_offset_buffer = key_info_buffer;
  int *query_count_buffer = &key_info_buffer[num_hash_f];

  const int hashtable_size = 1024 + OPTIMAL_THREADS_PER_BLOCK;
  __shared__ int hashtable_query[hashtable_size];
  __shared__ int hashtable_count[hashtable_size];
  __shared__ int inserted_query[hashtable_size];
  __shared__ int query_counter[1];

  int hash_f_idx_base = 0;

  while (true) {

    init_buffer_nonblocking<int>(EMPTY_VALUE, hashtable_query, hashtable_size, num_threads, thread_id);
    init_buffer_nonblocking<int>(0, hashtable_count, hashtable_size, num_threads, thread_id);
    init_buffer_nonblocking<int>(EMPTY_VALUE, inserted_query, hashtable_size, num_threads, thread_id);
    init_buffer_nonblocking<int>(0, query_counter, 1, num_threads, thread_id);
    __syncthreads();

    while (hash_f_idx_base < num_hash_f) {

      int hash_f_idx = hash_f_idx_base + warp_idx;
      int batch_idx__hash_f_idx = batch_idx * num_hash_f + hash_f_idx;

      int stop_flag = 0;

      int query_offset = query_offset_buffer[hash_f_idx];
      int query_count = query_count_buffer[hash_f_idx];

      while (query_count > 0) {

        int work_size = min(query_count, WARP_SIZE);

        // try inserting query to set and check whether the query is new
        int found_new_query = 0;
        int query_idx = -1;
        if (warp_thread_idx < work_size) {
          query_idx = query_sorted_idxes[batch_idx__hash_f_idx * num_query + query_offset + warp_thread_idx];
          int slot = set_insert<int>(hashtable_query, hashtable_size, query_idx);
          if (slot >= 0) {
            found_new_query = atomicAdd(&hashtable_count[slot], 1) == 0;
          }
        }

        // compute cumulative offset
        int position_offset = found_new_query;
        int next_position_offset = 0;
        #pragma unroll
        for (int offset = 1; offset < WARP_SIZE; offset = offset << 1) {
          next_position_offset = __shfl_up_sync(FULL_MASK, position_offset, offset);
          if (thread_id % WARP_SIZE >= offset) {
            position_offset = position_offset + next_position_offset;
          }
        }

        // get the inserted query list end index
        int inserted_query_base = 0;
        if (thread_id % WARP_SIZE == WARP_SIZE - 1) {
          inserted_query_base = atomicAdd(query_counter, position_offset);
        }
        inserted_query_base = __shfl_sync(FULL_MASK, inserted_query_base, WARP_SIZE - 1);

        // insert new queries to list
        int insert_idx = inserted_query_base + position_offset - 1;
        if (found_new_query) {
          inserted_query[insert_idx] = query_idx;
        }

        // remove inserted queries from list
        query_offset_buffer[hash_f_idx] += work_size;
        query_count_buffer[hash_f_idx] -= work_size;
        query_offset += work_size;
        query_count -= work_size;

        // if list is almost full, stop inserting
        if (inserted_query_base + OPTIMAL_THREADS_PER_BLOCK > hashtable_size) {
          stop_flag = 1;
          break;
        }

      }

      if (stop_flag) {
        break;
      }

      hash_f_idx_base = hash_f_idx_base + num_warps;

    }

    __syncthreads();

    int num_distinct_query = query_counter[0];

    if (num_distinct_query > 0) {
      for (int idx_base = 0; idx_base < num_distinct_query; idx_base = idx_base + num_warps) {
        int idx = idx_base + warp_idx;
        if (idx < num_distinct_query) {
          int query_idx = inserted_query[idx];
          int batch_idx__query_idx = batch_idx * num_query + query_idx;

          int slot = set_lookup<int>(hashtable_query, hashtable_size, query_idx);
          int duplicate_count = hashtable_count[slot];

          float weight = 0;
          for (int weight_idx_base = 0; weight_idx_base < weight_dim; weight_idx_base = weight_idx_base + WARP_SIZE) {
            int weight_dim_idx = weight_idx_base + warp_thread_idx;
            float val = weight_buffer[weight_dim_idx] * query_weight[batch_idx__query_idx * weight_dim + weight_dim_idx];
            #pragma unroll
            for (int offset = 1; offset < WARP_SIZE; offset = offset << 1) {
              val += __shfl_xor_sync(FULL_MASK, val, offset);
            }
            weight = weight + val;
          }

          weight = (float)duplicate_count * weight / float(num_hash_f);

          for (int value_idx_base = 0; value_idx_base < value_dim; value_idx_base = value_idx_base + WARP_SIZE) {
            int value_dim_idx = value_idx_base + warp_thread_idx;
            float val = value_buffer[value_dim_idx];
            atomicAdd(&cumulation_value[batch_idx__query_idx * value_dim + value_dim_idx], weight * val);
          }
        }
      }
    } else {

      // all computation is completed if num_distinct_query == 0
      break;

    }

    __syncthreads();

  }

}
