/*!
**************************************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************************************
* Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
*/

#include <vector>
#include "cuda/ms_deform_im2col_cuda.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#pragma once
#include <torch/extension.h>


at::Tensor ms_deform_attn_cuda_forward(
    const at::Tensor &value, 
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const int im2col_step)
{
    AT_ASSERTM(value.is_contiguous(), "value tensor has to be contiguous");
    AT_ASSERTM(spatial_shapes.is_contiguous(), "spatial_shapes tensor has to be contiguous");
    AT_ASSERTM(level_start_index.is_contiguous(), "level_start_index tensor has to be contiguous");
    AT_ASSERTM(sampling_loc.is_contiguous(), "sampling_loc tensor has to be contiguous");
    AT_ASSERTM(attn_weight.is_contiguous(), "attn_weight tensor has to be contiguous");

    AT_ASSERTM(value.type().is_cuda(), "value must be a CUDA tensor");
    AT_ASSERTM(spatial_shapes.type().is_cuda(), "spatial_shapes must be a CUDA tensor");
    AT_ASSERTM(level_start_index.type().is_cuda(), "level_start_index must be a CUDA tensor");
    AT_ASSERTM(sampling_loc.type().is_cuda(), "sampling_loc must be a CUDA tensor");
    AT_ASSERTM(attn_weight.type().is_cuda(), "attn_weight must be a CUDA tensor");

    const int batch = value.size(0);
    const int spatial_size = value.size(1);
    const int num_heads = value.size(2);
    const int channels = value.size(3);

    const int num_levels = spatial_shapes.size(0);

    const int num_query = sampling_loc.size(1);
    const int num_point = sampling_loc.size(4);

    const int im2col_step_ = std::min(batch, im2col_step);

    AT_ASSERTM(batch % im2col_step_ == 0, "batch(%d) must divide im2col_step(%d)", batch, im2col_step_);
    
    auto output = at::zeros({batch, num_query, num_heads, channels}, value.options());

    const int batch_n = im2col_step_;
    auto output_n = output.view({batch/im2col_step_, batch_n, num_query, num_heads, channels});
    auto per_value_size = spatial_size * num_heads * channels;
    auto per_sample_loc_size = num_query * num_heads * num_levels * num_point * 2;
    auto per_attn_weight_size = num_query * num_heads * num_levels * num_point;
    for (int n = 0; n < batch/im2col_step_; ++n)
    {
        auto columns = output_n.select(0, n);
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, value.type(), "ms_deform_attn_forward_cuda", ([&] {
            ms_deformable_im2col_cuda(at::cuda::getCurrentCUDAStream(),
                value.data<scalar_t>() + n * im2col_step_ * per_value_size,
                spatial_shapes.data<int64_t>(),
                level_start_index.data<int64_t>(),
                sampling_loc.data<scalar_t>() + n * im2col_step_ * per_sample_loc_size,
                attn_weight.data<scalar_t>() + n * im2col_step_ * per_attn_weight_size,
                batch_n, spatial_size, num_heads, channels, num_levels, num_query, num_point,
                columns.data<scalar_t>());

        }));
    }

    output = output.view({batch, num_query, num_heads*channels});

    return output;
}


std::vector<at::Tensor> ms_deform_attn_cuda_backward(
    const at::Tensor &value, 
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const at::Tensor &grad_output,
    const int im2col_step)
{

    AT_ASSERTM(value.is_contiguous(), "value tensor has to be contiguous");
    AT_ASSERTM(spatial_shapes.is_contiguous(), "spatial_shapes tensor has to be contiguous");
    AT_ASSERTM(level_start_index.is_contiguous(), "level_start_index tensor has to be contiguous");
    AT_ASSERTM(sampling_loc.is_contiguous(), "sampling_loc tensor has to be contiguous");
    AT_ASSERTM(attn_weight.is_contiguous(), "attn_weight tensor has to be contiguous");
    AT_ASSERTM(grad_output.is_contiguous(), "grad_output tensor has to be contiguous");

    AT_ASSERTM(value.type().is_cuda(), "value must be a CUDA tensor");
    AT_ASSERTM(spatial_shapes.type().is_cuda(), "spatial_shapes must be a CUDA tensor");
    AT_ASSERTM(level_start_index.type().is_cuda(), "level_start_index must be a CUDA tensor");
    AT_ASSERTM(sampling_loc.type().is_cuda(), "sampling_loc must be a CUDA tensor");
    AT_ASSERTM(attn_weight.type().is_cuda(), "attn_weight must be a CUDA tensor");
    AT_ASSERTM(grad_output.type().is_cuda(), "grad_output must be a CUDA tensor");

    const int batch = value.size(0);
    const int spatial_size = value.size(1);
    const int num_heads = value.size(2);
    const int channels = value.size(3);

    const int num_levels = spatial_shapes.size(0);

    const int num_query = sampling_loc.size(1);
    const int num_point = sampling_loc.size(4);

    const int im2col_step_ = std::min(batch, im2col_step);

    AT_ASSERTM(batch % im2col_step_ == 0, "batch(%d) must divide im2col_step(%d)", batch, im2col_step_);

    auto grad_value = at::zeros_like(value);
    auto grad_sampling_loc = at::zeros_like(sampling_loc);
    auto grad_attn_weight = at::zeros_like(attn_weight);

    const int batch_n = im2col_step_;
    auto per_value_size = spatial_size * num_heads * channels;
    auto per_sample_loc_size = num_query * num_heads * num_levels * num_point * 2;
    auto per_attn_weight_size = num_query * num_heads * num_levels * num_point;
    auto grad_output_n = grad_output.view({batch/im2col_step_, batch_n, num_query, num_heads, channels});
    
    for (int n = 0; n < batch/im2col_step_; ++n)
    {
        auto grad_output_g = grad_output_n.select(0, n);
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, value.type(), "ms_deform_attn_backward_cuda", ([&] {
            ms_deformable_col2im_cuda(at::cuda::getCurrentCUDAStream(),
                                    grad_output_g.data<scalar_t>(),
                                    value.data<scalar_t>() + n * im2col_step_ * per_value_size,
                                    spatial_shapes.data<int64_t>(),
                                    level_start_index.data<int64_t>(),
                                    sampling_loc.data<scalar_t>() + n * im2col_step_ * per_sample_loc_size,
                                    attn_weight.data<scalar_t>() + n * im2col_step_ * per_attn_weight_size,
                                    batch_n, spatial_size, num_heads, channels, num_levels, num_query, num_point,
                                    grad_value.data<scalar_t>() +  n * im2col_step_ * per_value_size,
                                    grad_sampling_loc.data<scalar_t>() + n * im2col_step_ * per_sample_loc_size,
                                    grad_attn_weight.data<scalar_t>() + n * im2col_step_ * per_attn_weight_size);

        }));
    }

    return {
        grad_value, grad_sampling_loc, grad_attn_weight
    };
}
