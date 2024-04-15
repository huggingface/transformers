// File from https://github.com/mlpen/YOSO/blob/main/encoders/backbones/efficient_attentions/yoso/yoso_v1/cuda/fast_lsh_cumulation.cu

#include <torch/extension.h>
#include <ATen/ATen.h>
#include "fast_lsh_cumulation.h"
#include "fast_lsh_cumulation_cuda.h"
#include "common_cuda.h"
#include "common.h"
#include <vector>
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<at::Tensor> fast_hash_ver1_kernel(
  at::Tensor query_mask,
  at::Tensor query_vector,
  at::Tensor key_mask,
  at::Tensor key_vector,
  int num_hash_f,
  int hash_code_len,
  bool use_cuda
) {

  int batch_size = query_vector.size(0);
  int num_query = query_vector.size(1);
  int num_key = key_vector.size(1);
  int vector_dim = query_vector.size(2);

  int num_hash_per_part = vector_dim / hash_code_len;
  int num_part = max(1, ceil_divide(num_hash_f, num_hash_per_part));

  at::Tensor Dmat = 2 * at::randint(0, 2, {batch_size, 3, num_part, vector_dim}, query_mask.options()) - 1;
  at::Tensor query_hash_code = at::zeros({batch_size, num_query, num_hash_f}, query_mask.options());
  at::Tensor key_hash_code = at::zeros({batch_size, num_key, num_hash_f}, key_mask.options());

  int *query_mask_ptr = query_mask.data_ptr<int>();
  float *query_vector_ptr = query_vector.data_ptr<float>();
  int *key_mask_ptr = key_mask.data_ptr<int>();
  float *key_vector_ptr = key_vector.data_ptr<float>();

  int *Dmat_ptr = Dmat.data_ptr<int>();

  int *query_hash_code_ptr = query_hash_code.data_ptr<int>();
  int *key_hash_code_ptr = key_hash_code.data_ptr<int>();

  if (use_cuda) {
    {
      dim3 threads(vector_dim);
      dim3 blocks(num_part, num_query, batch_size);
      int shared_mem = vector_dim * sizeof(float);
      fast_hash_ver1_cuda_kernel<<<blocks, threads, shared_mem>>>(
        query_mask_ptr,
        query_vector_ptr,
        Dmat_ptr,
        query_hash_code_ptr,
        batch_size,
        num_query,
        vector_dim,
        num_part,
        num_hash_f,
        hash_code_len
      );
    }
    {
      dim3 threads(vector_dim);
      dim3 blocks(num_part, num_key, batch_size);
      int shared_mem = vector_dim * sizeof(float);
      fast_hash_ver1_cuda_kernel<<<blocks, threads, shared_mem>>>(
        key_mask_ptr,
        key_vector_ptr,
        Dmat_ptr,
        key_hash_code_ptr,
        batch_size,
        num_key,
        vector_dim,
        num_part,
        num_hash_f,
        hash_code_len
      );
    }
  }

  return {query_hash_code, key_hash_code};

}

at::Tensor lsh_cumulation_ver1_kernel(
  at::Tensor query_mask,
  at::Tensor query_hash_code,
  at::Tensor key_mask,
  at::Tensor key_hash_code,
  at::Tensor value,
  int hashtable_capacity,
  bool use_cuda
) {

  int batch_size = query_hash_code.size(0);
  int num_hash_f = query_hash_code.size(2);

  int num_query = query_hash_code.size(1);
  int num_key = key_hash_code.size(1);
  int value_dim = value.size(2);

  at::Tensor hashtable_value = at::empty({batch_size, num_hash_f, hashtable_capacity, WARP_SIZE}, value.options());
  at::Tensor cumulation_value = at::zeros({batch_size, num_query, value_dim}, value.options());

  if (use_cuda) {
    int threads_x = WARP_SIZE;
    int threads_y = OPTIMAL_THREADS_PER_BLOCK / WARP_SIZE;
    int block_x_step1 = num_key / threads_y;
    int block_x_step2 = num_query / threads_y;
    int block_y = batch_size;

    dim3 threads(threads_x, threads_y);
    dim3 blocks_step1(block_x_step1, block_y);
    dim3 blocks_step2(block_x_step2, block_y);

    int *query_mask_ptr = query_mask.data_ptr<int>();
    int *query_hash_code_ptr = query_hash_code.data_ptr<int>();
    int *key_mask_ptr = key_mask.data_ptr<int>();
    int *key_hash_code_ptr = key_hash_code.data_ptr<int>();
    float *value_ptr = value.data_ptr<float>();
    float *hashtable_value_ptr = hashtable_value.data_ptr<float>();
    float *cumulation_value_ptr = cumulation_value.data_ptr<float>();

    for (int value_offset = 0; value_offset < value_dim; value_offset = value_offset + WARP_SIZE) {

      cudaMemset(hashtable_value_ptr, 0, (batch_size * num_hash_f * hashtable_capacity * WARP_SIZE) * sizeof(float));

      lsh_cumulation_ver1_step1_cuda_kernel<<<blocks_step1, threads>>>(
        key_mask_ptr,
        key_hash_code_ptr,
        value_ptr,
        hashtable_value_ptr,
        batch_size,
        num_hash_f,
        hashtable_capacity,
        num_key,
        value_dim,
        value_offset
      );

      lsh_cumulation_ver1_step2_cuda_kernel<<<blocks_step2, threads>>>(
        query_mask_ptr,
        query_hash_code_ptr,
        hashtable_value_ptr,
        cumulation_value_ptr,
        batch_size,
        num_hash_f,
        hashtable_capacity,
        num_query,
        value_dim,
        value_offset
      );
    }

  }

  return cumulation_value;

}

at::Tensor lsh_weighted_cumulation_ver1_kernel(
  at::Tensor query_mask,
  at::Tensor query_hash_code,
  at::Tensor query_weight,
  at::Tensor key_mask,
  at::Tensor key_hash_code,
  at::Tensor key_weight,
  at::Tensor value,
  int hashtable_capacity,
  bool use_cuda
) {

  int batch_size = query_hash_code.size(0);
  int num_hash_f = query_hash_code.size(2);

  int num_query = query_hash_code.size(1);
  int num_key = key_hash_code.size(1);
  int value_dim = value.size(2);
  int weight_dim = query_weight.size(2);

  at::Tensor hashtable_value = at::zeros({batch_size, num_hash_f, hashtable_capacity, WARP_SIZE}, value.options());
  at::Tensor cumulation_value = at::zeros({batch_size, num_query, value_dim}, value.options());

  if (use_cuda) {
    int threads_x = WARP_SIZE;
    int threads_y = OPTIMAL_THREADS_PER_BLOCK / WARP_SIZE;
    int block_x_step1 = num_key / threads_y;
    int block_x_step2 = num_query / threads_y;
    int block_y = batch_size;

    dim3 threads(threads_x, threads_y);
    dim3 blocks_step1(block_x_step1, block_y);
    dim3 blocks_step2(block_x_step2, block_y);

    int *query_mask_ptr = query_mask.data_ptr<int>();
    int *query_hash_code_ptr = query_hash_code.data_ptr<int>();
    float *query_weight_ptr = query_weight.data_ptr<float>();
    int *key_mask_ptr = key_mask.data_ptr<int>();
    int *key_hash_code_ptr = key_hash_code.data_ptr<int>();
    float *key_weight_ptr = key_weight.data_ptr<float>();
    float *value_ptr = value.data_ptr<float>();
    float *hashtable_value_ptr = hashtable_value.data_ptr<float>();
    float *cumulation_value_ptr = cumulation_value.data_ptr<float>();

    for (int value_offset = 0; value_offset < value_dim; value_offset = value_offset + WARP_SIZE) {
      for (int weight_idx = 0; weight_idx < weight_dim; weight_idx++) {

        cudaMemset(hashtable_value_ptr, 0, (batch_size * num_hash_f * hashtable_capacity * WARP_SIZE) * sizeof(float));

        lsh_weighted_cumulation_ver1_step1_cuda_kernel<<<blocks_step1, threads>>>(
          key_mask_ptr,
          key_hash_code_ptr,
          key_weight_ptr,
          value_ptr,
          hashtable_value_ptr,
          batch_size,
          num_hash_f,
          hashtable_capacity,
          num_key,
          value_dim,
          weight_dim,
          value_offset,
          weight_idx
        );

        lsh_weighted_cumulation_ver1_step2_cuda_kernel<<<blocks_step2, threads>>>(
          query_mask_ptr,
          query_hash_code_ptr,
          query_weight_ptr,
          hashtable_value_ptr,
          cumulation_value_ptr,
          batch_size,
          num_hash_f,
          hashtable_capacity,
          num_query,
          value_dim,
          weight_dim,
          value_offset,
          weight_idx
        );
      }
    }

  }

  return cumulation_value;

}

at::Tensor lsh_weighted_cumulation_ver2_kernel(
  at::Tensor query_mask,
  at::Tensor query_hash_code,
  at::Tensor query_weight,
  at::Tensor key_mask,
  at::Tensor key_hash_code,
  at::Tensor key_weight,
  at::Tensor value,
  int hashtable_capacity,
  bool use_cuda
) {

  int batch_size = query_hash_code.size(0);
  int num_hash_f = query_hash_code.size(2);

  int num_query = query_hash_code.size(1);
  int num_key = key_hash_code.size(1);
  int value_dim = value.size(2);
  int weight_dim = query_weight.size(2);

  at::Tensor count_sort_table = at::zeros({batch_size, num_hash_f, hashtable_capacity}, query_hash_code.options());
  at::Tensor key_sorted_idxes = at::zeros({batch_size, num_hash_f, num_key}, query_hash_code.options());
  at::Tensor query_info = at::zeros({batch_size, num_query, 2, num_hash_f}, query_hash_code.options());
  at::Tensor cumulation_value = at::zeros({batch_size, num_query, value_dim}, value.options());

  if (use_cuda) {

    int *query_mask_ptr = query_mask.data_ptr<int>();
    int *query_hash_code_ptr = query_hash_code.data_ptr<int>();
    float *query_weight_ptr = query_weight.data_ptr<float>();
    int *key_mask_ptr = key_mask.data_ptr<int>();
    int *key_hash_code_ptr = key_hash_code.data_ptr<int>();
    float *key_weight_ptr = key_weight.data_ptr<float>();
    float *value_ptr = value.data_ptr<float>();

    int *count_sort_table_ptr = count_sort_table.data_ptr<int>();
    int *key_sorted_idxes_ptr = key_sorted_idxes.data_ptr<int>();
    int *query_info_ptr = query_info.data_ptr<int>();

    float *cumulation_value_ptr = cumulation_value.data_ptr<float>();

    {
      dim3 threads_step13(num_hash_f, max(1, OPTIMAL_THREADS_PER_BLOCK / num_hash_f));
      dim3 blocks_step13(num_key / max(1, OPTIMAL_THREADS_PER_BLOCK / num_hash_f), batch_size);
      dim3 threads_step2(min(hashtable_capacity, OPTIMAL_THREADS_PER_BLOCK));
      dim3 blocks_step2(num_hash_f, batch_size);
      int shared_mem = hashtable_capacity * sizeof(float);
      count_sort_step1_cuda_kernel<<<blocks_step13, threads_step13>>>(
        key_mask_ptr,
        key_hash_code_ptr,
        count_sort_table_ptr,
        batch_size,
        num_hash_f,
        hashtable_capacity,
        num_key
      );
      count_sort_step2_cuda_kernel<<<blocks_step2, threads_step2, shared_mem>>>(
        count_sort_table_ptr,
        batch_size,
        num_hash_f,
        hashtable_capacity
      );
      count_sort_step3_cuda_kernel<<<blocks_step13, threads_step13>>>(
        key_mask_ptr,
        key_hash_code_ptr,
        count_sort_table_ptr,
        key_sorted_idxes_ptr,
        batch_size,
        num_hash_f,
        hashtable_capacity,
        num_key
      );
    }
    {
      dim3 threads(num_hash_f, max(1, OPTIMAL_THREADS_PER_BLOCK / num_hash_f));
      dim3 blocks(num_query / max(1, OPTIMAL_THREADS_PER_BLOCK / num_hash_f), batch_size);
      extract_query_info_cuda_kernel<<<blocks, threads>>>(
        query_mask_ptr,
        query_hash_code_ptr,
        count_sort_table_ptr,
        query_info_ptr,
        batch_size,
        num_hash_f,
        hashtable_capacity,
        num_query
      );
    }
    {
      dim3 threads(WARP_SIZE, OPTIMAL_THREADS_PER_BLOCK / WARP_SIZE);
      dim3 blocks(num_query, num_hash_f, batch_size);
      int shared_mem = (weight_dim + WARP_SIZE) * sizeof(float);
      lsh_weighted_cumulation_ver2_step2_cuda_kernel<<<blocks, threads, shared_mem>>>(
        query_mask_ptr,
        query_info_ptr,
        key_sorted_idxes_ptr,
        query_weight_ptr,
        key_weight_ptr,
        value_ptr,
        cumulation_value_ptr,
        batch_size,
        num_hash_f,
        num_query,
        num_key,
        value_dim,
        weight_dim
      );
    }
  }

  return cumulation_value;

}

at::Tensor lsh_weighted_cumulation_ver3_kernel(
  at::Tensor query_mask,
  at::Tensor query_hash_code,
  at::Tensor query_weight,
  at::Tensor key_mask,
  at::Tensor key_hash_code,
  at::Tensor key_weight,
  at::Tensor value,
  int hashtable_capacity,
  bool use_cuda
) {

  int batch_size = query_hash_code.size(0);
  int num_hash_f = query_hash_code.size(2);

  int num_query = query_hash_code.size(1);
  int num_key = key_hash_code.size(1);
  int value_dim = value.size(2);
  int weight_dim = query_weight.size(2);

  at::Tensor count_sort_table = at::zeros({batch_size, num_hash_f, hashtable_capacity}, query_hash_code.options());
  at::Tensor query_sorted_idxes = at::zeros({batch_size, num_hash_f, num_query}, query_hash_code.options());
  at::Tensor key_info = at::zeros({batch_size, num_key, 2, num_hash_f}, query_hash_code.options());
  at::Tensor cumulation_value = at::zeros({batch_size, num_query, value_dim}, value.options());

  if (use_cuda) {

    int *query_mask_ptr = query_mask.data_ptr<int>();
    int *query_hash_code_ptr = query_hash_code.data_ptr<int>();
    float *query_weight_ptr = query_weight.data_ptr<float>();
    int *key_mask_ptr = key_mask.data_ptr<int>();
    int *key_hash_code_ptr = key_hash_code.data_ptr<int>();
    float *key_weight_ptr = key_weight.data_ptr<float>();
    float *value_ptr = value.data_ptr<float>();

    int *count_sort_table_ptr = count_sort_table.data_ptr<int>();
    int *query_sorted_idxes_ptr = query_sorted_idxes.data_ptr<int>();
    int *key_info_ptr = key_info.data_ptr<int>();

    float *cumulation_value_ptr = cumulation_value.data_ptr<float>();

    {
      dim3 threads_step13(num_hash_f, max(1, OPTIMAL_THREADS_PER_BLOCK / num_hash_f));
      dim3 blocks_step13(num_query / max(1, OPTIMAL_THREADS_PER_BLOCK / num_hash_f), batch_size);
      dim3 threads_step2(min(hashtable_capacity, OPTIMAL_THREADS_PER_BLOCK));
      dim3 blocks_step2(num_hash_f, batch_size);
      int shared_mem = hashtable_capacity * sizeof(float);
      count_sort_step1_cuda_kernel<<<blocks_step13, threads_step13>>>(
        query_mask_ptr,
        query_hash_code_ptr,
        count_sort_table_ptr,
        batch_size,
        num_hash_f,
        hashtable_capacity,
        num_query
      );
      count_sort_step2_cuda_kernel<<<blocks_step2, threads_step2, shared_mem>>>(
        count_sort_table_ptr,
        batch_size,
        num_hash_f,
        hashtable_capacity
      );
      count_sort_step3_cuda_kernel<<<blocks_step13, threads_step13>>>(
        query_mask_ptr,
        query_hash_code_ptr,
        count_sort_table_ptr,
        query_sorted_idxes_ptr,
        batch_size,
        num_hash_f,
        hashtable_capacity,
        num_query
      );
    }
    {
      dim3 threads(num_hash_f, max(1, OPTIMAL_THREADS_PER_BLOCK / num_hash_f));
      dim3 blocks(num_key / max(1, OPTIMAL_THREADS_PER_BLOCK / num_hash_f), batch_size);
      extract_query_info_cuda_kernel<<<blocks, threads>>>(
        key_mask_ptr,
        key_hash_code_ptr,
        count_sort_table_ptr,
        key_info_ptr,
        batch_size,
        num_hash_f,
        hashtable_capacity,
        num_key
      );
    }
    {
      dim3 threads(WARP_SIZE, OPTIMAL_THREADS_PER_BLOCK / WARP_SIZE);
      dim3 blocks(num_key, num_hash_f, batch_size);
      int shared_mem = (weight_dim + value_dim + WARP_SIZE) * sizeof(float);
      lsh_weighted_cumulation_ver3_step2_cuda_kernel<<<blocks, threads, shared_mem>>>(
        query_sorted_idxes_ptr,
        key_mask_ptr,
        key_info_ptr,
        query_weight_ptr,
        key_weight_ptr,
        value_ptr,
        cumulation_value_ptr,
        batch_size,
        num_hash_f,
        num_query,
        num_key,
        value_dim,
        weight_dim
      );
    }
  }

  return cumulation_value;

}

at::Tensor lsh_weighted_cumulation_ver4_kernel(
  at::Tensor query_mask,
  at::Tensor query_hash_code,
  at::Tensor query_weight,
  at::Tensor key_mask,
  at::Tensor key_hash_code,
  at::Tensor key_weight,
  at::Tensor value,
  int hashtable_capacity,
  bool use_cuda
) {

  int batch_size = query_hash_code.size(0);
  int num_hash_f = query_hash_code.size(2);

  int num_query = query_hash_code.size(1);
  int num_key = key_hash_code.size(1);
  int value_dim = value.size(2);
  int weight_dim = query_weight.size(2);

  at::Tensor count_sort_table = at::zeros({batch_size, num_hash_f, hashtable_capacity}, query_hash_code.options());
  at::Tensor query_sorted_idxes = at::zeros({batch_size, num_hash_f, num_query}, query_hash_code.options());
  at::Tensor key_info = at::zeros({batch_size, num_key, 2, num_hash_f}, query_hash_code.options());
  at::Tensor cumulation_value = at::zeros({batch_size, num_query, value_dim}, value.options());

  if (use_cuda) {

    int *query_mask_ptr = query_mask.data_ptr<int>();
    int *query_hash_code_ptr = query_hash_code.data_ptr<int>();
    float *query_weight_ptr = query_weight.data_ptr<float>();
    int *key_mask_ptr = key_mask.data_ptr<int>();
    int *key_hash_code_ptr = key_hash_code.data_ptr<int>();
    float *key_weight_ptr = key_weight.data_ptr<float>();
    float *value_ptr = value.data_ptr<float>();

    int *count_sort_table_ptr = count_sort_table.data_ptr<int>();
    int *query_sorted_idxes_ptr = query_sorted_idxes.data_ptr<int>();
    int *key_info_ptr = key_info.data_ptr<int>();

    float *cumulation_value_ptr = cumulation_value.data_ptr<float>();

    {
      dim3 threads_step13(num_hash_f, max(1, OPTIMAL_THREADS_PER_BLOCK / num_hash_f));
      dim3 blocks_step13(num_query / max(1, OPTIMAL_THREADS_PER_BLOCK / num_hash_f), batch_size);
      dim3 threads_step2(min(hashtable_capacity, OPTIMAL_THREADS_PER_BLOCK));
      dim3 blocks_step2(num_hash_f, batch_size);
      int shared_mem = hashtable_capacity * sizeof(float);
      count_sort_step1_cuda_kernel<<<blocks_step13, threads_step13>>>(
        query_mask_ptr,
        query_hash_code_ptr,
        count_sort_table_ptr,
        batch_size,
        num_hash_f,
        hashtable_capacity,
        num_query
      );
      count_sort_step2_cuda_kernel<<<blocks_step2, threads_step2, shared_mem>>>(
        count_sort_table_ptr,
        batch_size,
        num_hash_f,
        hashtable_capacity
      );
      count_sort_step3_cuda_kernel<<<blocks_step13, threads_step13>>>(
        query_mask_ptr,
        query_hash_code_ptr,
        count_sort_table_ptr,
        query_sorted_idxes_ptr,
        batch_size,
        num_hash_f,
        hashtable_capacity,
        num_query
      );
    }
    {
      dim3 threads(num_hash_f, max(1, OPTIMAL_THREADS_PER_BLOCK / num_hash_f));
      dim3 blocks(num_key / max(1, OPTIMAL_THREADS_PER_BLOCK / num_hash_f), batch_size);
      extract_query_info_cuda_kernel<<<blocks, threads>>>(
        key_mask_ptr,
        key_hash_code_ptr,
        count_sort_table_ptr,
        key_info_ptr,
        batch_size,
        num_hash_f,
        hashtable_capacity,
        num_key
      );
    }
    {
      dim3 threads(WARP_SIZE, OPTIMAL_THREADS_PER_BLOCK / WARP_SIZE);
      dim3 blocks(num_key, batch_size);
      int shared_mem = (weight_dim + value_dim + 2 * num_hash_f) * sizeof(float);
      lsh_weighted_cumulation_ver4_step2_cuda_kernel<<<blocks, threads, shared_mem>>>(
        query_sorted_idxes_ptr,
        key_mask_ptr,
        key_info_ptr,
        query_weight_ptr,
        key_weight_ptr,
        value_ptr,
        cumulation_value_ptr,
        batch_size,
        num_hash_f,
        num_query,
        num_key,
        value_dim,
        weight_dim
      );
    }
  }

  return cumulation_value;

}
