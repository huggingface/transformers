// Copyright (c) Facebook, Inc. and its affiliates.

// modified from
// https://github.com/open-mmlab/mmdetection/blob/master/mmdet/ops/dcn/src/deform_conv_cuda_kernel.cu
// Original license: Apache 2.0
// clang-format off

// modify from
// https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/blob/mmdetection/mmdet/ops/dcn/src/deform_conv_cuda_kernel.cu

/*!
 ******************* BEGIN Caffe Copyright Notice and Disclaimer *****************
 *
 * COPYRIGHT
 *
 * All contributions by the University of California:
 * Copyright (c) 2014-2017 The Regents of the University of California (Regents)
 * All rights reserved.
 *
 * All other contributions:
 * Copyright (c) 2014-2017, the respective contributors
 * All rights reserved.
 *
 * Caffe uses a shared copyright model: each contributor holds copyright over
 * their contributions to Caffe. The project versioning records all such
 * contribution and copyright details. If a contributor wants to further mark
 * their specific copyright on a particular contribution, they should indicate
 * their copyright solely in the commit message of the change when it is
 * committed.
 *
 * LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 *FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * CONTRIBUTION AGREEMENT
 *
 * By contributing to the BVLC/caffe repository through pull-request, comment,
 * or otherwise, the contributor releases their content to the
 * license and copyright terms herein.
 *
 ***************** END Caffe Copyright Notice and Disclaimer *********************
 *
 * Copyright (c) 2018 Microsoft
 * Licensed under The MIT License [see LICENSE for details]
 * \file modulated_deformable_im2col.cuh
 * \brief Function definitions of converting an image to
 * column matrix based on kernel, padding, dilation, and offset.
 * These functions are mainly used in deformable convolution operators.
 * \ref: https://arxiv.org/abs/1703.06211
 * \author Yuwen Xiong, Haozhi Qi, Jifeng Dai, Xizhou Zhu, Han Hu, Dazhi Cheng
 */

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <THC/THCAtomics.cuh>

using namespace at;

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)


namespace {

const int CUDA_NUM_THREADS = 1024;
const int kMaxGridNum = 65535;

inline int GET_BLOCKS(const int N) {
  return std::min(kMaxGridNum, (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);
}

}

template <typename scalar_t>
__device__ scalar_t deformable_im2col_bilinear(
    const scalar_t* bottom_data,
    const int data_width,
    const int height,
    const int width,
    scalar_t h,
    scalar_t w) {
  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  scalar_t lh = h - h_low;
  scalar_t lw = w - w_low;
  scalar_t hh = 1 - lh, hw = 1 - lw;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0)
    v1 = bottom_data[h_low * data_width + w_low];
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = bottom_data[h_low * data_width + w_high];
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = bottom_data[h_high * data_width + w_low];
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = bottom_data[h_high * data_width + w_high];

  scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename scalar_t>
__device__ scalar_t get_gradient_weight(
    scalar_t argmax_h,
    scalar_t argmax_w,
    const int h,
    const int w,
    const int height,
    const int width) {
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 ||
      argmax_w >= width) {
    // empty
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  scalar_t weight = 0;
  if (h == argmax_h_low && w == argmax_w_low)
    weight = (h + 1 - argmax_h) * (w + 1 - argmax_w);
  if (h == argmax_h_low && w == argmax_w_high)
    weight = (h + 1 - argmax_h) * (argmax_w + 1 - w);
  if (h == argmax_h_high && w == argmax_w_low)
    weight = (argmax_h + 1 - h) * (w + 1 - argmax_w);
  if (h == argmax_h_high && w == argmax_w_high)
    weight = (argmax_h + 1 - h) * (argmax_w + 1 - w);
  return weight;
}

template <typename scalar_t>
__device__ scalar_t get_coordinate_weight(
    scalar_t argmax_h,
    scalar_t argmax_w,
    const int height,
    const int width,
    const scalar_t* im_data,
    const int data_width,
    const int bp_dir) {
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 ||
      argmax_w >= width) {
    // empty
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  scalar_t weight = 0;

  if (bp_dir == 0) {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_w_low + 1 - argmax_w) *
          im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += -1 * (argmax_w - argmax_w_low) *
          im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += (argmax_w_low + 1 - argmax_w) *
          im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_w - argmax_w_low) *
          im_data[argmax_h_high * data_width + argmax_w_high];
  } else if (bp_dir == 1) {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_h_low + 1 - argmax_h) *
          im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += (argmax_h_low + 1 - argmax_h) *
          im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += -1 * (argmax_h - argmax_h_low) *
          im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_h - argmax_h_low) *
          im_data[argmax_h_high * data_width + argmax_w_high];
  }

  return weight;
}

template <typename scalar_t>
__global__ void deformable_im2col_gpu_kernel(
    const int n,
    const scalar_t* data_im,
    const scalar_t* data_offset,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int channel_per_deformable_group,
    const int batch_size,
    const int num_channels,
    const int deformable_group,
    const int height_col,
    const int width_col,
    scalar_t* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    // index index of output matrix
    const int w_col = index % width_col;
    const int h_col = (index / width_col) % height_col;
    const int b_col = (index / width_col / height_col) % batch_size;
    const int c_im = (index / width_col / height_col) / batch_size;
    const int c_col = c_im * kernel_h * kernel_w;

    // compute deformable group index
    const int deformable_group_index = c_im / channel_per_deformable_group;

    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;
    scalar_t* data_col_ptr = data_col +
        ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
    // const scalar_t* data_im_ptr = data_im + ((b_col * num_channels + c_im) *
    // height + h_in) * width + w_in;
    const scalar_t* data_im_ptr =
        data_im + (b_col * num_channels + c_im) * height * width;
    const scalar_t* data_offset_ptr = data_offset +
        (b_col * deformable_group + deformable_group_index) * 2 * kernel_h *
            kernel_w * height_col * width_col;

    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        const int data_offset_h_ptr =
            ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        const int data_offset_w_ptr =
            ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col +
            w_col;
        const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
        const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
        scalar_t val = static_cast<scalar_t>(0);
        const scalar_t h_im = h_in + i * dilation_h + offset_h;
        const scalar_t w_im = w_in + j * dilation_w + offset_w;
        if (h_im > -1 && w_im > -1 && h_im < height && w_im < width) {
          // const scalar_t map_h = i * dilation_h + offset_h;
          // const scalar_t map_w = j * dilation_w + offset_w;
          // const int cur_height = height - h_in;
          // const int cur_width = width - w_in;
          // val = deformable_im2col_bilinear(data_im_ptr, width, cur_height,
          // cur_width, map_h, map_w);
          val = deformable_im2col_bilinear(
              data_im_ptr, width, height, width, h_im, w_im);
        }
        *data_col_ptr = val;
        data_col_ptr += batch_size * height_col * width_col;
      }
    }
  }
}


template <typename scalar_t>
__global__ void deformable_col2im_gpu_kernel(
    const int n,
    const scalar_t* data_col,
    const scalar_t* data_offset,
    const int channels,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int channel_per_deformable_group,
    const int batch_size,
    const int deformable_group,
    const int height_col,
    const int width_col,
    scalar_t* grad_im) {
  CUDA_KERNEL_LOOP(index, n) {
    const int j = (index / width_col / height_col / batch_size) % kernel_w;
    const int i =
        (index / width_col / height_col / batch_size / kernel_w) % kernel_h;
    const int c =
        index / width_col / height_col / batch_size / kernel_w / kernel_h;
    // compute the start and end of the output

    const int deformable_group_index = c / channel_per_deformable_group;

    int w_out = index % width_col;
    int h_out = (index / width_col) % height_col;
    int b = (index / width_col / height_col) % batch_size;
    int w_in = w_out * stride_w - pad_w;
    int h_in = h_out * stride_h - pad_h;

    const scalar_t* data_offset_ptr = data_offset +
        (b * deformable_group + deformable_group_index) * 2 * kernel_h *
            kernel_w * height_col * width_col;
    const int data_offset_h_ptr =
        ((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out;
    const int data_offset_w_ptr =
        ((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out;
    const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
    const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
    const scalar_t cur_inv_h_data = h_in + i * dilation_h + offset_h;
    const scalar_t cur_inv_w_data = w_in + j * dilation_w + offset_w;

    const scalar_t cur_top_grad = data_col[index];
    const int cur_h = (int)cur_inv_h_data;
    const int cur_w = (int)cur_inv_w_data;
    for (int dy = -2; dy <= 2; dy++) {
      for (int dx = -2; dx <= 2; dx++) {
        if (cur_h + dy >= 0 && cur_h + dy < height && cur_w + dx >= 0 &&
            cur_w + dx < width && abs(cur_inv_h_data - (cur_h + dy)) < 1 &&
            abs(cur_inv_w_data - (cur_w + dx)) < 1) {
          int cur_bottom_grad_pos =
              ((b * channels + c) * height + cur_h + dy) * width + cur_w + dx;
          scalar_t weight = get_gradient_weight(
              cur_inv_h_data,
              cur_inv_w_data,
              cur_h + dy,
              cur_w + dx,
              height,
              width);
          atomicAdd(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);
        }
      }
    }
  }
}


template <typename scalar_t>
__global__ void deformable_col2im_coord_gpu_kernel(
    const int n,
    const scalar_t* data_col,
    const scalar_t* data_im,
    const scalar_t* data_offset,
    const int channels,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int channel_per_deformable_group,
    const int batch_size,
    const int offset_channels,
    const int deformable_group,
    const int height_col,
    const int width_col,
    scalar_t* grad_offset) {
  CUDA_KERNEL_LOOP(index, n) {
    scalar_t val = 0;
    int w = index % width_col;
    int h = (index / width_col) % height_col;
    int c = (index / width_col / height_col) % offset_channels;
    int b = (index / width_col / height_col) / offset_channels;
    // compute the start and end of the output

    const int deformable_group_index = c / (2 * kernel_h * kernel_w);
    const int col_step = kernel_h * kernel_w;
    int cnt = 0;
    const scalar_t* data_col_ptr = data_col +
        deformable_group_index * channel_per_deformable_group * batch_size *
            width_col * height_col;
    const scalar_t* data_im_ptr = data_im +
        (b * deformable_group + deformable_group_index) *
            channel_per_deformable_group / kernel_h / kernel_w * height * width;
    const scalar_t* data_offset_ptr = data_offset +
        (b * deformable_group + deformable_group_index) * 2 * kernel_h *
            kernel_w * height_col * width_col;

    const int offset_c = c - deformable_group_index * 2 * kernel_h * kernel_w;

    for (int col_c = (offset_c / 2); col_c < channel_per_deformable_group;
         col_c += col_step) {
      const int col_pos =
          (((col_c * batch_size + b) * height_col) + h) * width_col + w;
      const int bp_dir = offset_c % 2;

      int j = (col_pos / width_col / height_col / batch_size) % kernel_w;
      int i =
          (col_pos / width_col / height_col / batch_size / kernel_w) % kernel_h;
      int w_out = col_pos % width_col;
      int h_out = (col_pos / width_col) % height_col;
      int w_in = w_out * stride_w - pad_w;
      int h_in = h_out * stride_h - pad_h;
      const int data_offset_h_ptr =
          (((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out);
      const int data_offset_w_ptr =
          (((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col +
           w_out);
      const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
      const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
      scalar_t inv_h = h_in + i * dilation_h + offset_h;
      scalar_t inv_w = w_in + j * dilation_w + offset_w;
      if (inv_h <= -1 || inv_w <= -1 || inv_h >= height || inv_w >= width) {
        inv_h = inv_w = -2;
      }
      const scalar_t weight = get_coordinate_weight(
          inv_h,
          inv_w,
          height,
          width,
          data_im_ptr + cnt * height * width,
          width,
          bp_dir);
      val += weight * data_col_ptr[col_pos];
      cnt += 1;
    }

    grad_offset[index] = val;
  }
}


namespace detectron2 {

void deformable_im2col(
    const at::Tensor data_im,
    const at::Tensor data_offset,
    const int channels,
    const int height,
    const int width,
    const int ksize_h,
    const int ksize_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int parallel_imgs,
    const int deformable_group,
    at::Tensor data_col) {
  // num_axes should be smaller than block size
  // todo: check parallel_imgs is correctly passed in
  int height_col =
      (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
  int width_col =
      (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * height_col * width_col * parallel_imgs;
  int channel_per_deformable_group = channels / deformable_group;

  at::cuda::CUDAGuard device_guard(data_im.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_im.scalar_type(), "deformable_im2col_gpu", ([&] {
        const scalar_t* data_im_ = data_im.data_ptr<scalar_t>();
        const scalar_t* data_offset_ = data_offset.data_ptr<scalar_t>();
        scalar_t* data_col_ = data_col.data_ptr<scalar_t>();

        deformable_im2col_gpu_kernel<<<
            GET_BLOCKS(num_kernels),
            CUDA_NUM_THREADS,
            0,
            stream>>>(
            num_kernels,
            data_im_,
            data_offset_,
            height,
            width,
            ksize_h,
            ksize_w,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            channel_per_deformable_group,
            parallel_imgs,
            channels,
            deformable_group,
            height_col,
            width_col,
            data_col_);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in deformable_im2col: %s\n", cudaGetErrorString(err));
  }
}


void deformable_col2im(
    const at::Tensor data_col,
    const at::Tensor data_offset,
    const int channels,
    const int height,
    const int width,
    const int ksize_h,
    const int ksize_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int parallel_imgs,
    const int deformable_group,
    at::Tensor grad_im) {
  // todo: make sure parallel_imgs is passed in correctly
  int height_col =
      (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
  int width_col =
      (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
  int num_kernels =
      channels * ksize_h * ksize_w * height_col * width_col * parallel_imgs;
  int channel_per_deformable_group = channels / deformable_group;

  at::cuda::CUDAGuard device_guard(data_col.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_col.scalar_type(), "deformable_col2im_gpu", ([&] {
        const scalar_t* data_col_ = data_col.data_ptr<scalar_t>();
        const scalar_t* data_offset_ = data_offset.data_ptr<scalar_t>();
        scalar_t* grad_im_ = grad_im.data_ptr<scalar_t>();

        deformable_col2im_gpu_kernel<<<
            GET_BLOCKS(num_kernels),
            CUDA_NUM_THREADS,
            0,
            stream>>>(
            num_kernels,
            data_col_,
            data_offset_,
            channels,
            height,
            width,
            ksize_h,
            ksize_w,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            channel_per_deformable_group,
            parallel_imgs,
            deformable_group,
            height_col,
            width_col,
            grad_im_);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in deformable_col2im: %s\n", cudaGetErrorString(err));
  }
}


void deformable_col2im_coord(
    const at::Tensor data_col,
    const at::Tensor data_im,
    const at::Tensor data_offset,
    const int channels,
    const int height,
    const int width,
    const int ksize_h,
    const int ksize_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int parallel_imgs,
    const int deformable_group,
    at::Tensor grad_offset) {
  int height_col =
      (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
  int width_col =
      (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
  int num_kernels = height_col * width_col * 2 * ksize_h * ksize_w *
      deformable_group * parallel_imgs;
  int channel_per_deformable_group =
      channels * ksize_h * ksize_w / deformable_group;

  at::cuda::CUDAGuard device_guard(data_col.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_col.scalar_type(), "deformable_col2im_coord_gpu", ([&] {
        const scalar_t* data_col_ = data_col.data_ptr<scalar_t>();
        const scalar_t* data_im_ = data_im.data_ptr<scalar_t>();
        const scalar_t* data_offset_ = data_offset.data_ptr<scalar_t>();
        scalar_t* grad_offset_ = grad_offset.data_ptr<scalar_t>();

        deformable_col2im_coord_gpu_kernel<<<
            GET_BLOCKS(num_kernels),
            CUDA_NUM_THREADS,
            0,
            stream>>>(
            num_kernels,
            data_col_,
            data_im_,
            data_offset_,
            channels,
            height,
            width,
            ksize_h,
            ksize_w,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            channel_per_deformable_group,
            parallel_imgs,
            2 * ksize_h * ksize_w * deformable_group,
            deformable_group,
            height_col,
            width_col,
            grad_offset_);
      }));
}

} // namespace detectron2


template <typename scalar_t>
__device__ scalar_t dmcn_im2col_bilinear(
    const scalar_t* bottom_data,
    const int data_width,
    const int height,
    const int width,
    scalar_t h,
    scalar_t w) {
  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  scalar_t lh = h - h_low;
  scalar_t lw = w - w_low;
  scalar_t hh = 1 - lh, hw = 1 - lw;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0)
    v1 = bottom_data[h_low * data_width + w_low];
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = bottom_data[h_low * data_width + w_high];
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = bottom_data[h_high * data_width + w_low];
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = bottom_data[h_high * data_width + w_high];

  scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename scalar_t>
__device__ scalar_t dmcn_get_gradient_weight(
    scalar_t argmax_h,
    scalar_t argmax_w,
    const int h,
    const int w,
    const int height,
    const int width) {
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 ||
      argmax_w >= width) {
    // empty
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  scalar_t weight = 0;
  if (h == argmax_h_low && w == argmax_w_low)
    weight = (h + 1 - argmax_h) * (w + 1 - argmax_w);
  if (h == argmax_h_low && w == argmax_w_high)
    weight = (h + 1 - argmax_h) * (argmax_w + 1 - w);
  if (h == argmax_h_high && w == argmax_w_low)
    weight = (argmax_h + 1 - h) * (w + 1 - argmax_w);
  if (h == argmax_h_high && w == argmax_w_high)
    weight = (argmax_h + 1 - h) * (argmax_w + 1 - w);
  return weight;
}

template <typename scalar_t>
__device__ scalar_t dmcn_get_coordinate_weight(
    scalar_t argmax_h,
    scalar_t argmax_w,
    const int height,
    const int width,
    const scalar_t* im_data,
    const int data_width,
    const int bp_dir) {
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 ||
      argmax_w >= width) {
    // empty
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  scalar_t weight = 0;

  if (bp_dir == 0) {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_w_low + 1 - argmax_w) *
          im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += -1 * (argmax_w - argmax_w_low) *
          im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += (argmax_w_low + 1 - argmax_w) *
          im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_w - argmax_w_low) *
          im_data[argmax_h_high * data_width + argmax_w_high];
  } else if (bp_dir == 1) {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_h_low + 1 - argmax_h) *
          im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += (argmax_h_low + 1 - argmax_h) *
          im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += -1 * (argmax_h - argmax_h_low) *
          im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_h - argmax_h_low) *
          im_data[argmax_h_high * data_width + argmax_w_high];
  }

  return weight;
}

template <typename scalar_t>
__global__ void modulated_deformable_im2col_gpu_kernel(
    const int n,
    const scalar_t* data_im,
    const scalar_t* data_offset,
    const scalar_t* data_mask,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int channel_per_deformable_group,
    const int batch_size,
    const int num_channels,
    const int deformable_group,
    const int height_col,
    const int width_col,
    scalar_t* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    // index index of output matrix
    const int w_col = index % width_col;
    const int h_col = (index / width_col) % height_col;
    const int b_col = (index / width_col / height_col) % batch_size;
    const int c_im = (index / width_col / height_col) / batch_size;
    const int c_col = c_im * kernel_h * kernel_w;

    // compute deformable group index
    const int deformable_group_index = c_im / channel_per_deformable_group;

    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;

    scalar_t* data_col_ptr = data_col +
        ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
    // const float* data_im_ptr = data_im + ((b_col * num_channels + c_im) *
    // height + h_in) * width + w_in;
    const scalar_t* data_im_ptr =
        data_im + (b_col * num_channels + c_im) * height * width;
    const scalar_t* data_offset_ptr = data_offset +
        (b_col * deformable_group + deformable_group_index) * 2 * kernel_h *
            kernel_w * height_col * width_col;

    const scalar_t* data_mask_ptr = data_mask +
        (b_col * deformable_group + deformable_group_index) * kernel_h *
            kernel_w * height_col * width_col;

    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        const int data_offset_h_ptr =
            ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        const int data_offset_w_ptr =
            ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col +
            w_col;
        const int data_mask_hw_ptr =
            ((i * kernel_w + j) * height_col + h_col) * width_col + w_col;
        const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
        const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
        const scalar_t mask = data_mask_ptr[data_mask_hw_ptr];
        scalar_t val = static_cast<scalar_t>(0);
        const scalar_t h_im = h_in + i * dilation_h + offset_h;
        const scalar_t w_im = w_in + j * dilation_w + offset_w;
        // if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {
        if (h_im > -1 && w_im > -1 && h_im < height && w_im < width) {
          // const float map_h = i * dilation_h + offset_h;
          // const float map_w = j * dilation_w + offset_w;
          // const int cur_height = height - h_in;
          // const int cur_width = width - w_in;
          // val = dmcn_im2col_bilinear(data_im_ptr, width, cur_height,
          // cur_width, map_h, map_w);
          val = dmcn_im2col_bilinear(
              data_im_ptr, width, height, width, h_im, w_im);
        }
        *data_col_ptr = val * mask;
        data_col_ptr += batch_size * height_col * width_col;
        // data_col_ptr += height_col * width_col;
      }
    }
  }
}

template <typename scalar_t>
__global__ void modulated_deformable_col2im_gpu_kernel(
    const int n,
    const scalar_t* data_col,
    const scalar_t* data_offset,
    const scalar_t* data_mask,
    const int channels,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int channel_per_deformable_group,
    const int batch_size,
    const int deformable_group,
    const int height_col,
    const int width_col,
    scalar_t* grad_im) {
  CUDA_KERNEL_LOOP(index, n) {
    const int j = (index / width_col / height_col / batch_size) % kernel_w;
    const int i =
        (index / width_col / height_col / batch_size / kernel_w) % kernel_h;
    const int c =
        index / width_col / height_col / batch_size / kernel_w / kernel_h;
    // compute the start and end of the output

    const int deformable_group_index = c / channel_per_deformable_group;

    int w_out = index % width_col;
    int h_out = (index / width_col) % height_col;
    int b = (index / width_col / height_col) % batch_size;
    int w_in = w_out * stride_w - pad_w;
    int h_in = h_out * stride_h - pad_h;

    const scalar_t* data_offset_ptr = data_offset +
        (b * deformable_group + deformable_group_index) * 2 * kernel_h *
            kernel_w * height_col * width_col;
    const scalar_t* data_mask_ptr = data_mask +
        (b * deformable_group + deformable_group_index) * kernel_h * kernel_w *
            height_col * width_col;
    const int data_offset_h_ptr =
        ((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out;
    const int data_offset_w_ptr =
        ((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out;
    const int data_mask_hw_ptr =
        ((i * kernel_w + j) * height_col + h_out) * width_col + w_out;
    const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
    const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
    const scalar_t mask = data_mask_ptr[data_mask_hw_ptr];
    const scalar_t cur_inv_h_data = h_in + i * dilation_h + offset_h;
    const scalar_t cur_inv_w_data = w_in + j * dilation_w + offset_w;

    const scalar_t cur_top_grad = data_col[index] * mask;
    const int cur_h = (int)cur_inv_h_data;
    const int cur_w = (int)cur_inv_w_data;
    for (int dy = -2; dy <= 2; dy++) {
      for (int dx = -2; dx <= 2; dx++) {
        if (cur_h + dy >= 0 && cur_h + dy < height && cur_w + dx >= 0 &&
            cur_w + dx < width && abs(cur_inv_h_data - (cur_h + dy)) < 1 &&
            abs(cur_inv_w_data - (cur_w + dx)) < 1) {
          int cur_bottom_grad_pos =
              ((b * channels + c) * height + cur_h + dy) * width + cur_w + dx;
          scalar_t weight = dmcn_get_gradient_weight(
              cur_inv_h_data,
              cur_inv_w_data,
              cur_h + dy,
              cur_w + dx,
              height,
              width);
          atomicAdd(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);
        }
      }
    }
  }
}

template <typename scalar_t>
__global__ void modulated_deformable_col2im_coord_gpu_kernel(
    const int n,
    const scalar_t* data_col,
    const scalar_t* data_im,
    const scalar_t* data_offset,
    const scalar_t* data_mask,
    const int channels,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int channel_per_deformable_group,
    const int batch_size,
    const int offset_channels,
    const int deformable_group,
    const int height_col,
    const int width_col,
    scalar_t* grad_offset,
    scalar_t* grad_mask) {
  CUDA_KERNEL_LOOP(index, n) {
    scalar_t val = 0, mval = 0;
    int w = index % width_col;
    int h = (index / width_col) % height_col;
    int c = (index / width_col / height_col) % offset_channels;
    int b = (index / width_col / height_col) / offset_channels;
    // compute the start and end of the output

    const int deformable_group_index = c / (2 * kernel_h * kernel_w);
    const int col_step = kernel_h * kernel_w;
    int cnt = 0;
    const scalar_t* data_col_ptr = data_col +
        deformable_group_index * channel_per_deformable_group * batch_size *
            width_col * height_col;
    const scalar_t* data_im_ptr = data_im +
        (b * deformable_group + deformable_group_index) *
            channel_per_deformable_group / kernel_h / kernel_w * height * width;
    const scalar_t* data_offset_ptr = data_offset +
        (b * deformable_group + deformable_group_index) * 2 * kernel_h *
            kernel_w * height_col * width_col;
    const scalar_t* data_mask_ptr = data_mask +
        (b * deformable_group + deformable_group_index) * kernel_h * kernel_w *
            height_col * width_col;

    const int offset_c = c - deformable_group_index * 2 * kernel_h * kernel_w;

    for (int col_c = (offset_c / 2); col_c < channel_per_deformable_group;
         col_c += col_step) {
      const int col_pos =
          (((col_c * batch_size + b) * height_col) + h) * width_col + w;
      const int bp_dir = offset_c % 2;

      int j = (col_pos / width_col / height_col / batch_size) % kernel_w;
      int i =
          (col_pos / width_col / height_col / batch_size / kernel_w) % kernel_h;
      int w_out = col_pos % width_col;
      int h_out = (col_pos / width_col) % height_col;
      int w_in = w_out * stride_w - pad_w;
      int h_in = h_out * stride_h - pad_h;
      const int data_offset_h_ptr =
          (((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out);
      const int data_offset_w_ptr =
          (((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col +
           w_out);
      const int data_mask_hw_ptr =
          (((i * kernel_w + j) * height_col + h_out) * width_col + w_out);
      const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
      const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
      const scalar_t mask = data_mask_ptr[data_mask_hw_ptr];
      scalar_t inv_h = h_in + i * dilation_h + offset_h;
      scalar_t inv_w = w_in + j * dilation_w + offset_w;
      if (inv_h <= -1 || inv_w <= -1 || inv_h >= height || inv_w >= width) {
        inv_h = inv_w = -2;
      } else {
        mval += data_col_ptr[col_pos] *
            dmcn_im2col_bilinear(
                    data_im_ptr + cnt * height * width,
                    width,
                    height,
                    width,
                    inv_h,
                    inv_w);
      }
      const scalar_t weight = dmcn_get_coordinate_weight(
          inv_h,
          inv_w,
          height,
          width,
          data_im_ptr + cnt * height * width,
          width,
          bp_dir);
      val += weight * data_col_ptr[col_pos] * mask;
      cnt += 1;
    }
    // KERNEL_ASSIGN(grad_offset[index], offset_req, val);
    grad_offset[index] = val;
    if (offset_c % 2 == 0)
      // KERNEL_ASSIGN(grad_mask[(((b * deformable_group +
      // deformable_group_index) * kernel_h * kernel_w + offset_c / 2) *
      // height_col + h) * width_col + w], mask_req, mval);
      grad_mask
          [(((b * deformable_group + deformable_group_index) * kernel_h *
                 kernel_w +
             offset_c / 2) *
                height_col +
            h) *
               width_col +
           w] = mval;
  }
}


namespace detectron2 {

void modulated_deformable_im2col_cuda(
    const at::Tensor data_im,
    const at::Tensor data_offset,
    const at::Tensor data_mask,
    const int batch_size,
    const int channels,
    const int height_im,
    const int width_im,
    const int height_col,
    const int width_col,
    const int kernel_h,
    const int kenerl_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int deformable_group,
    at::Tensor data_col) {
  // num_axes should be smaller than block size
  const int channel_per_deformable_group = channels / deformable_group;
  const int num_kernels = channels * batch_size * height_col * width_col;

  at::cuda::CUDAGuard device_guard(data_im.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_im.scalar_type(), "modulated_deformable_im2col_gpu", ([&] {
        const scalar_t* data_im_ = data_im.data_ptr<scalar_t>();
        const scalar_t* data_offset_ = data_offset.data_ptr<scalar_t>();
        const scalar_t* data_mask_ = data_mask.data_ptr<scalar_t>();
        scalar_t* data_col_ = data_col.data_ptr<scalar_t>();

        modulated_deformable_im2col_gpu_kernel<<<
            GET_BLOCKS(num_kernels),
            CUDA_NUM_THREADS,
            0,
            stream>>>(
            num_kernels,
            data_im_,
            data_offset_,
            data_mask_,
            height_im,
            width_im,
            kernel_h,
            kenerl_w,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            channel_per_deformable_group,
            batch_size,
            channels,
            deformable_group,
            height_col,
            width_col,
            data_col_);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf(
        "error in modulated_deformable_im2col_cuda: %s\n",
        cudaGetErrorString(err));
  }
}

void modulated_deformable_col2im_cuda(
    const at::Tensor data_col,
    const at::Tensor data_offset,
    const at::Tensor data_mask,
    const int batch_size,
    const int channels,
    const int height_im,
    const int width_im,
    const int height_col,
    const int width_col,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int deformable_group,
    at::Tensor grad_im) {
  const int channel_per_deformable_group = channels / deformable_group;
  const int num_kernels =
      channels * kernel_h * kernel_w * batch_size * height_col * width_col;

  at::cuda::CUDAGuard device_guard(data_col.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_col.scalar_type(), "modulated_deformable_col2im_gpu", ([&] {
        const scalar_t* data_col_ = data_col.data_ptr<scalar_t>();
        const scalar_t* data_offset_ = data_offset.data_ptr<scalar_t>();
        const scalar_t* data_mask_ = data_mask.data_ptr<scalar_t>();
        scalar_t* grad_im_ = grad_im.data_ptr<scalar_t>();

        modulated_deformable_col2im_gpu_kernel<<<
            GET_BLOCKS(num_kernels),
            CUDA_NUM_THREADS,
            0,
            stream>>>(
            num_kernels,
            data_col_,
            data_offset_,
            data_mask_,
            channels,
            height_im,
            width_im,
            kernel_h,
            kernel_w,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            channel_per_deformable_group,
            batch_size,
            deformable_group,
            height_col,
            width_col,
            grad_im_);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf(
        "error in modulated_deformable_col2im_cuda: %s\n",
        cudaGetErrorString(err));
  }
}

void modulated_deformable_col2im_coord_cuda(
    const at::Tensor data_col,
    const at::Tensor data_im,
    const at::Tensor data_offset,
    const at::Tensor data_mask,
    const int batch_size,
    const int channels,
    const int height_im,
    const int width_im,
    const int height_col,
    const int width_col,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int deformable_group,
    at::Tensor grad_offset,
    at::Tensor grad_mask) {
  const int num_kernels = batch_size * height_col * width_col * 2 * kernel_h *
      kernel_w * deformable_group;
  const int channel_per_deformable_group =
      channels * kernel_h * kernel_w / deformable_group;

  at::cuda::CUDAGuard device_guard(data_col.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_col.scalar_type(), "modulated_deformable_col2im_coord_gpu", ([&] {
        const scalar_t* data_col_ = data_col.data_ptr<scalar_t>();
        const scalar_t* data_im_ = data_im.data_ptr<scalar_t>();
        const scalar_t* data_offset_ = data_offset.data_ptr<scalar_t>();
        const scalar_t* data_mask_ = data_mask.data_ptr<scalar_t>();
        scalar_t* grad_offset_ = grad_offset.data_ptr<scalar_t>();
        scalar_t* grad_mask_ = grad_mask.data_ptr<scalar_t>();

        modulated_deformable_col2im_coord_gpu_kernel<<<
            GET_BLOCKS(num_kernels),
            CUDA_NUM_THREADS,
            0,
            stream>>>(
            num_kernels,
            data_col_,
            data_im_,
            data_offset_,
            data_mask_,
            channels,
            height_im,
            width_im,
            kernel_h,
            kernel_w,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            channel_per_deformable_group,
            batch_size,
            2 * kernel_h * kernel_w * deformable_group,
            deformable_group,
            height_col,
            width_col,
            grad_offset_,
            grad_mask_);
      }));
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf(
        "error in modulated_deformable_col2im_coord_cuda: %s\n",
        cudaGetErrorString(err));
  }
}

} // namespace detectron2
