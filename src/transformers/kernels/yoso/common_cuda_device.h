
#include "common.h"

template<typename T>
__device__ int set_insert(T *set, int set_size, T value) {
  int slot = value % set_size;
  int start_slot = slot;
  while (true) {
    T prev = atomicCAS(&set[slot], EMPTY_VALUE, value);
    if (prev == EMPTY_VALUE || prev == value) {
      return slot;
    }
    slot = (slot + 1) % set_size;
    if (slot == start_slot) {
      return -1;
    }
  }
  return -1;
}

template<typename T>
__device__ int set_lookup(T *set, int set_size, T value) {
  int slot = value % set_size;
  int start_slot = slot;
  while (true) {
    if (set[slot] == value) {
      return slot;
    }
    slot = (slot + 1) % set_size;
    if (slot == start_slot) {
      return -1;
    }
  }
  return -1;
}

template<typename T>
__device__ void init_buffer(T init_value, T *buffer, int buffer_size, int num_threads, int thread_id) {
  __syncthreads();
  for (int i = 0; i < buffer_size; i = i + num_threads) {
    int offset_idx = i + thread_id;
    if (offset_idx < buffer_size) {
      buffer[offset_idx] = init_value;
    }
  }
  __syncthreads();
}

template<typename T>
__device__ void copy_data(T *src_pt, T *dist_pt, int data_length, int num_threads, int thread_id) {
  __syncthreads();
  for (int i = 0; i < data_length; i = i + num_threads) {
    int offset_idx = i + thread_id;
    if (offset_idx < data_length) {
      dist_pt[offset_idx] = src_pt[offset_idx];
    }
  }
  __syncthreads();
}

template<typename T>
__device__ void init_buffer_nonblocking(T init_value, T *buffer, int buffer_size, int num_threads, int thread_id) {
  for (int i = 0; i < buffer_size; i = i + num_threads) {
    int offset_idx = i + thread_id;
    if (offset_idx < buffer_size) {
      buffer[offset_idx] = init_value;
    }
  }
}

template<typename T>
__device__ void copy_data_nonblocking(T *src_pt, T *dist_pt, int data_length, int num_threads, int thread_id) {
  for (int i = 0; i < data_length; i = i + num_threads) {
    int offset_idx = i + thread_id;
    if (offset_idx < data_length) {
      dist_pt[offset_idx] = src_pt[offset_idx];
    }
  }
}
