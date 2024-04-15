#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>

std::vector<at::Tensor> fast_hash_ver1_kernel(
  at::Tensor query_mask,
  at::Tensor query_vector,
  at::Tensor key_mask,
  at::Tensor key_vector,
  int num_hash_f,
  int hash_code_len,
  bool use_cuda
);

at::Tensor lsh_cumulation_ver1_kernel(
  at::Tensor query_mask,
  at::Tensor query_hash_code,
  at::Tensor key_mask,
  at::Tensor key_hash_code,
  at::Tensor value,
  int hashtable_capacity,
  bool use_cuda
);

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
);

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
);

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
);

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
);
