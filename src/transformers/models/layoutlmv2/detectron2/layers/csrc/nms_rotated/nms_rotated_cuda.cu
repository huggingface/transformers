// Copyright (c) Facebook, Inc. and its affiliates.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#ifdef WITH_CUDA
#include "../box_iou_rotated/box_iou_rotated_utils.h"
#endif
// TODO avoid this when pytorch supports "same directory" hipification
#ifdef WITH_HIP
#include "box_iou_rotated/box_iou_rotated_utils.h"
#endif

using namespace detectron2;

namespace {
int const threadsPerBlock = sizeof(unsigned long long) * 8;
}

template <typename T>
__global__ void nms_rotated_cuda_kernel(
    const int n_boxes,
    const double iou_threshold,
    const T* dev_boxes,
    unsigned long long* dev_mask) {
  // nms_rotated_cuda_kernel is modified from torchvision's nms_cuda_kernel

  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // if (row_start > col_start) return;

  const int row_size =
      min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
      min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  // Compared to nms_cuda_kernel, where each box is represented with 4 values
  // (x1, y1, x2, y2), each rotated box is represented with 5 values
  // (x_center, y_center, width, height, angle_degrees) here.
  __shared__ T block_boxes[threadsPerBlock * 5];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 5 + 0] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 0];
    block_boxes[threadIdx.x * 5 + 1] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 1];
    block_boxes[threadIdx.x * 5 + 2] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 2];
    block_boxes[threadIdx.x * 5 + 3] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 3];
    block_boxes[threadIdx.x * 5 + 4] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 4];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const T* cur_box = dev_boxes + cur_box_idx * 5;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      // Instead of devIoU used by original horizontal nms, here
      // we use the single_box_iou_rotated function from box_iou_rotated_utils.h
      if (single_box_iou_rotated<T>(cur_box, block_boxes + i * 5) >
          iou_threshold) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = at::cuda::ATenCeilDiv(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

namespace detectron2 {

at::Tensor nms_rotated_cuda(
    // input must be contiguous
    const at::Tensor& dets,
    const at::Tensor& scores,
    double iou_threshold) {
  // using scalar_t = float;
  AT_ASSERTM(dets.is_cuda(), "dets must be a CUDA tensor");
  AT_ASSERTM(scores.is_cuda(), "scores must be a CUDA tensor");
  at::cuda::CUDAGuard device_guard(dets.device());

  auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));
  auto dets_sorted = dets.index_select(0, order_t);

  auto dets_num = dets.size(0);

  const int col_blocks =
      at::cuda::ATenCeilDiv(static_cast<int>(dets_num), threadsPerBlock);

  at::Tensor mask =
      at::empty({dets_num * col_blocks}, dets.options().dtype(at::kLong));

  dim3 blocks(col_blocks, col_blocks);
  dim3 threads(threadsPerBlock);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(
      dets_sorted.scalar_type(), "nms_rotated_kernel_cuda", [&] {
        nms_rotated_cuda_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            dets_num,
            iou_threshold,
            dets_sorted.data_ptr<scalar_t>(),
            (unsigned long long*)mask.data_ptr<int64_t>());
      });

  at::Tensor mask_cpu = mask.to(at::kCPU);
  unsigned long long* mask_host =
      (unsigned long long*)mask_cpu.data_ptr<int64_t>();

  std::vector<unsigned long long> remv(col_blocks);
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

  at::Tensor keep =
      at::empty({dets_num}, dets.options().dtype(at::kLong).device(at::kCPU));
  int64_t* keep_out = keep.data_ptr<int64_t>();

  int num_to_keep = 0;
  for (int i = 0; i < dets_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {
      keep_out[num_to_keep++] = i;
      unsigned long long* p = mask_host + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }

  AT_CUDA_CHECK(cudaGetLastError());
  return order_t.index(
      {keep.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep)
           .to(order_t.device(), keep.scalar_type())});
}

} // namespace detectron2
