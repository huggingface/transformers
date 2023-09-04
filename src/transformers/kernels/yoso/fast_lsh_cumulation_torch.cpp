#include <torch/extension.h>
#include <ATen/ATen.h>
#include "fast_lsh_cumulation.h"
#include "common_cuda.h"
#include <vector>

std::vector<at::Tensor> fast_hash(
  at::Tensor query_mask,
  at::Tensor query_vector,
  at::Tensor key_mask,
  at::Tensor key_vector,
  int num_hash_f,
  int hash_code_len,
  bool use_cuda,
  int version
) {
  return fast_hash_ver1_kernel(
    query_mask,
    query_vector,
    key_mask,
    key_vector,
    num_hash_f,
    hash_code_len,
    use_cuda
  );
}

at::Tensor lsh_cumulation(
  at::Tensor query_mask,         // [batch_size, num_query]
  at::Tensor query_hash_code,    // [batch_size, num_query, num_hash_f]
  at::Tensor key_mask,           // [batch_size, num_key]
  at::Tensor key_hash_code,      // [batch_size, num_key, num_hash_f]
  at::Tensor value,              // [batch_size, num_key, value_dim]
  int hashtable_capacity,
  bool use_cuda,
  int version
) {
  return lsh_cumulation_ver1_kernel(
    query_mask,
    query_hash_code,
    key_mask,
    key_hash_code,
    value,
    hashtable_capacity,
    use_cuda
  );
}

at::Tensor lsh_weighted_cumulation(
  at::Tensor query_mask,         // [batch_size, num_query]
  at::Tensor query_hash_code,    // [batch_size, num_query, num_hash_f]
  at::Tensor query_weight,       // [batch_size, num_query, weight_dim]
  at::Tensor key_mask,           // [batch_size, num_key]
  at::Tensor key_hash_code,      // [batch_size, num_key, num_hash_f]
  at::Tensor key_weight,         // [batch_size, num_key, weight_dim]
  at::Tensor value,              // [batch_size, num_key, value_dim]
  int hashtable_capacity,
  bool use_cuda,
  int version
) {
  if (version == 1) {
    return lsh_weighted_cumulation_ver1_kernel(
      query_mask,
      query_hash_code,
      query_weight,
      key_mask,
      key_hash_code,
      key_weight,
      value,
      hashtable_capacity,
      use_cuda
    );
  } else if (version == 2) {
    return lsh_weighted_cumulation_ver2_kernel(
      query_mask,
      query_hash_code,
      query_weight,
      key_mask,
      key_hash_code,
      key_weight,
      value,
      hashtable_capacity,
      use_cuda
    );
  } else if (version == 3) {
    return lsh_weighted_cumulation_ver3_kernel(
      query_mask,
      query_hash_code,
      query_weight,
      key_mask,
      key_hash_code,
      key_weight,
      value,
      hashtable_capacity,
      use_cuda
    );
  } else if (version == 4) {
    return lsh_weighted_cumulation_ver4_kernel(
      query_mask,
      query_hash_code,
      query_weight,
      key_mask,
      key_hash_code,
      key_weight,
      value,
      hashtable_capacity,
      use_cuda
    );
  } else {
    return lsh_weighted_cumulation_ver3_kernel(
      query_mask,
      query_hash_code,
      query_weight,
      key_mask,
      key_hash_code,
      key_weight,
      value,
      hashtable_capacity,
      use_cuda
    );
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fast_hash", &fast_hash, "Fast Hash (CUDA)");
  m.def("lsh_cumulation", &lsh_cumulation, "LSH Cumulation (CUDA)");
  m.def("lsh_weighted_cumulation", &lsh_weighted_cumulation, "LSH Weighted Cumulation (CUDA)");
}
