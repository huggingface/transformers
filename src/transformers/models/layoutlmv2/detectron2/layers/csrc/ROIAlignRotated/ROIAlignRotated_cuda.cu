// Copyright (c) Facebook, Inc. and its affiliates.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

// TODO make it in a common file
#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

// Note: this implementation originates from the Caffe2 ROIAlignRotated Op
// and PyTorch ROIAlign (non-rotated) Op implementations.
// The key difference between this implementation and those ones is
// we don't do "legacy offset" in this version, as there aren't many previous
// works, if any, using the "legacy" ROIAlignRotated Op.
// This would make the interface a bit cleaner.

namespace detectron2 {

namespace {

template <typename T>
__device__ T bilinear_interpolate(
    const T* input,
    const int height,
    const int width,
    T y,
    T x) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    return 0;
  }

  if (y < 0) {
    y = 0;
  }

  if (x < 0) {
    x = 0;
  }

  int y_low = (int)y;
  int x_low = (int)x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;
  // do bilinear interpolation
  T v1 = input[y_low * width + x_low];
  T v2 = input[y_low * width + x_high];
  T v3 = input[y_high * width + x_low];
  T v4 = input[y_high * width + x_high];
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}

template <typename T>
__device__ void bilinear_interpolate_gradient(
    const int height,
    const int width,
    T y,
    T x,
    T& w1,
    T& w2,
    T& w3,
    T& w4,
    int& x_low,
    int& x_high,
    int& y_low,
    int& y_high) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y < 0) {
    y = 0;
  }

  if (x < 0) {
    x = 0;
  }

  y_low = (int)y;
  x_low = (int)x;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;

  // reference in forward
  // T v1 = input[y_low * width + x_low];
  // T v2 = input[y_low * width + x_high];
  // T v3 = input[y_high * width + x_low];
  // T v4 = input[y_high * width + x_high];
  // T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  return;
}

} // namespace

template <typename T>
__global__ void RoIAlignRotatedForward(
    const int nthreads,
    const T* input,
    const T spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    const T* rois,
    T* top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* current_roi = rois + n * 6;
    int roi_batch_ind = current_roi[0];

    // Do not use rounding; this implementation detail is critical
    // ROIAlignRotated supports align == true, i.e., continuous coordinate
    // by default, thus the 0.5 offset
    T offset = (T)0.5;
    T roi_center_w = current_roi[1] * spatial_scale - offset;
    T roi_center_h = current_roi[2] * spatial_scale - offset;
    T roi_width = current_roi[3] * spatial_scale;
    T roi_height = current_roi[4] * spatial_scale;
    T theta = current_roi[5] * M_PI / 180.0;
    T cos_theta = cos(theta);
    T sin_theta = sin(theta);

    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    const T* offset_input =
        input + (roi_batch_ind * channels + c) * height * width;

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // roi_start_h and roi_start_w are computed wrt the center of RoI (x, y).
    // Appropriate translation needs to be applied after.
    T roi_start_h = -roi_height / 2.0;
    T roi_start_w = -roi_width / 2.0;

    // We do average (inte  gral) pooling inside a bin
    const T count = max(roi_bin_grid_h * roi_bin_grid_w, 1); // e.g. = 4

    T output_val = 0.;
    for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
    {
      const T yy = roi_start_h + ph * bin_size_h +
          static_cast<T>(iy + .5f) * bin_size_h /
              static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T xx = roi_start_w + pw * bin_size_w +
            static_cast<T>(ix + .5f) * bin_size_w /
                static_cast<T>(roi_bin_grid_w);

        // Rotate by theta around the center and translate
        T y = yy * cos_theta - xx * sin_theta + roi_center_h;
        T x = yy * sin_theta + xx * cos_theta + roi_center_w;

        T val = bilinear_interpolate(offset_input, height, width, y, x);
        output_val += val;
      }
    }
    output_val /= count;

    top_data[index] = output_val;
  }
}

template <typename T>
__global__ void RoIAlignRotatedBackwardFeature(
    const int nthreads,
    const T* top_diff,
    const int num_rois,
    const T spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    T* bottom_diff,
    const T* rois) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* current_roi = rois + n * 6;
    int roi_batch_ind = current_roi[0];

    // Do not use rounding; this implementation detail is critical
    // ROIAlignRotated supports align == true, i.e., continuous coordinate
    // by default, thus the 0.5 offset
    T offset = (T)0.5;
    T roi_center_w = current_roi[1] * spatial_scale - offset;
    T roi_center_h = current_roi[2] * spatial_scale - offset;
    T roi_width = current_roi[3] * spatial_scale;
    T roi_height = current_roi[4] * spatial_scale;
    T theta = current_roi[5] * M_PI / 180.0;
    T cos_theta = cos(theta);
    T sin_theta = sin(theta);

    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    T* offset_bottom_diff =
        bottom_diff + (roi_batch_ind * channels + c) * height * width;

    int top_offset = (n * channels + c) * pooled_height * pooled_width;
    const T* offset_top_diff = top_diff + top_offset;
    const T top_diff_this_bin = offset_top_diff[ph * pooled_width + pw];

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // roi_start_h and roi_start_w are computed wrt the center of RoI (x, y).
    // Appropriate translation needs to be applied after.
    T roi_start_h = -roi_height / 2.0;
    T roi_start_w = -roi_width / 2.0;

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

    for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
    {
      const T yy = roi_start_h + ph * bin_size_h +
          static_cast<T>(iy + .5f) * bin_size_h /
              static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T xx = roi_start_w + pw * bin_size_w +
            static_cast<T>(ix + .5f) * bin_size_w /
                static_cast<T>(roi_bin_grid_w);

        // Rotate by theta around the center and translate
        T y = yy * cos_theta - xx * sin_theta + roi_center_h;
        T x = yy * sin_theta + xx * cos_theta + roi_center_w;

        T w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;

        bilinear_interpolate_gradient(
            height, width, y, x, w1, w2, w3, w4, x_low, x_high, y_low, y_high);

        T g1 = top_diff_this_bin * w1 / count;
        T g2 = top_diff_this_bin * w2 / count;
        T g3 = top_diff_this_bin * w3 / count;
        T g4 = top_diff_this_bin * w4 / count;

        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
          atomicAdd(
              offset_bottom_diff + y_low * width + x_low, static_cast<T>(g1));
          atomicAdd(
              offset_bottom_diff + y_low * width + x_high, static_cast<T>(g2));
          atomicAdd(
              offset_bottom_diff + y_high * width + x_low, static_cast<T>(g3));
          atomicAdd(
              offset_bottom_diff + y_high * width + x_high, static_cast<T>(g4));
        } // if
      } // ix
    } // iy
  } // CUDA_1D_KERNEL_LOOP
} // RoIAlignRotatedBackward

at::Tensor ROIAlignRotated_forward_cuda(
    const at::Tensor& input,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio) {
  AT_ASSERTM(input.device().is_cuda(), "input must be a CUDA tensor");
  AT_ASSERTM(rois.device().is_cuda(), "rois must be a CUDA tensor");
  at::TensorArg input_t{input, "input", 1}, rois_t{rois, "rois", 2};

  at::CheckedFrom c = "ROIAlignRotated_forward_cuda";
  at::checkAllSameGPU(c, {input_t, rois_t});
  at::checkAllSameType(c, {input_t, rois_t});
  at::cuda::CUDAGuard device_guard(input.device());

  auto num_rois = rois.size(0);
  auto channels = input.size(1);
  auto height = input.size(2);
  auto width = input.size(3);

  auto output = at::empty(
      {num_rois, channels, pooled_height, pooled_width}, input.options());
  auto output_size = num_rois * pooled_height * pooled_width * channels;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(
      at::cuda::ATenCeilDiv(
          static_cast<int64_t>(output_size), static_cast<int64_t>(512)),
      static_cast<int64_t>(4096)));
  dim3 block(512);

  if (output.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return output;
  }

  auto input_ = input.contiguous(), rois_ = rois.contiguous();
  AT_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "ROIAlignRotated_forward", [&] {
        RoIAlignRotatedForward<scalar_t><<<grid, block, 0, stream>>>(
            output_size,
            input_.data_ptr<scalar_t>(),
            spatial_scale,
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            sampling_ratio,
            rois_.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>());
      });
  cudaDeviceSynchronize();
  AT_CUDA_CHECK(cudaGetLastError());
  return output;
}

// TODO remove the dependency on input and use instead its sizes -> save memory
at::Tensor ROIAlignRotated_backward_cuda(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int sampling_ratio) {
  AT_ASSERTM(grad.device().is_cuda(), "grad must be a CUDA tensor");
  AT_ASSERTM(rois.device().is_cuda(), "rois must be a CUDA tensor");

  at::TensorArg grad_t{grad, "grad", 1}, rois_t{rois, "rois", 2};
  at::CheckedFrom c = "ROIAlign_backward_cuda";
  at::checkAllSameGPU(c, {grad_t, rois_t});
  at::checkAllSameType(c, {grad_t, rois_t});
  at::cuda::CUDAGuard device_guard(grad.device());

  auto num_rois = rois.size(0);
  auto grad_input =
      at::zeros({batch_size, channels, height, width}, grad.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(
      at::cuda::ATenCeilDiv(
          static_cast<int64_t>(grad.numel()), static_cast<int64_t>(512)),
      static_cast<int64_t>(4096)));
  dim3 block(512);

  // handle possibly empty gradients
  if (grad.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return grad_input;
  }

  auto grad_ = grad.contiguous(), rois_ = rois.contiguous();
  AT_DISPATCH_FLOATING_TYPES(
      grad.scalar_type(), "ROIAlignRotated_backward", [&] {
        RoIAlignRotatedBackwardFeature<scalar_t><<<grid, block, 0, stream>>>(
            grad.numel(),
            grad_.data_ptr<scalar_t>(),
            num_rois,
            spatial_scale,
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            sampling_ratio,
            grad_input.data_ptr<scalar_t>(),
            rois_.data_ptr<scalar_t>());
      });
  AT_CUDA_CHECK(cudaGetLastError());
  return grad_input;
}

} // namespace detectron2
