__global__ void fast_hash_ver1_cuda_kernel(
  int *mask,        // [batch_size, num_vector]
  float *vector,    // [batch_size, num_vector, vector_dim]
  int *Dmat,        // [3, num_part, vector_dim]
  int *hash_code,   // [batch_size, num_vector, num_hash_f]
  int batch_size,
  int num_vector,
  int vector_dim,
  int num_part,
  int num_hash_f,
  int hash_code_len
);

__global__ void lsh_cumulation_ver1_step1_cuda_kernel(
  int *key_mask,           // [batch_size, num_key]
  int *key_hash_code,      // [batch_size, num_key, num_hash_f]
  float *value,            // [batch_size, num_key, value_dim]
  float *hashtable_value,  // [batch_size, num_hash_f, hashtable_capacity, value_dim]
  int batch_size,
  int num_hash_f,
  int hashtable_capacity,
  int num_key,
  int value_dim,
  int offset_warp
);

__global__ void lsh_cumulation_ver1_step2_cuda_kernel(
  int *query_mask,         // [batch_size, num_query]
  int *query_hash_code,    // [batch_size, num_query, num_hash_f]
  float *hashtable_value,  // [batch_size, num_hash_f, hashtable_capacity, value_dim]
  float *cumulation_value, // [batch_size, num_query, value_dim]
  int batch_size,
  int num_hash_f,
  int hashtable_capacity,
  int num_query,
  int value_dim,
  int offset_warp
);

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
);

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
);

__global__ void count_sort_step1_cuda_kernel(
  int *key_mask,         // [batch_size, num_key]
  int *key_hash_code,    // [batch_size, num_key, num_hash_f]
  int *count_sort_table, // [batch_size, num_hash_f, hashtable_capacity]
  int batch_size,
  int num_hash_f,
  int hashtable_capacity,
  int num_key
);

__global__ void count_sort_step2_cuda_kernel(
  int *count_sort_table,  // [batch_size, num_hash_f, hashtable_capacity]
  int batch_size,
  int num_hash_f,
  int hashtable_capacity
);

__global__ void count_sort_step3_cuda_kernel(
  int *key_mask,          // [batch_size, num_key]
  int *key_hash_code,     // [batch_size, num_key, num_hash_f]
  int *count_sort_table,  // [batch_size, num_hash_f, hashtable_capacity]
  int *key_sorted_idxes,  // [batch_size, num_hash_f, num_key]
  int batch_size,
  int num_hash_f,
  int hashtable_capacity,
  int num_key
);

__global__ void extract_query_info_cuda_kernel(
  int *query_mask,       // [batch_size, num_query]
  int *query_hash_code,  // [batch_size, num_query, num_hash_f]
  int *count_sort_table, // [batch_size, num_hash_f, hashtable_capacity]
  int *query_info,       // [batch_size, num_query, 2, num_hash_f]
  int batch_size,
  int num_hash_f,
  int hashtable_capacity,
  int num_query
);

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
);

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
);

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
);
