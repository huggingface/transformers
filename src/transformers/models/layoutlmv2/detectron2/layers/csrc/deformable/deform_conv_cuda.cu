// Copyright (c) Facebook, Inc. and its affiliates.

// modified from
// https://github.com/open-mmlab/mmdetection/blob/master/mmdet/ops/dcn/src/deform_conv_cuda.cpp
// Original license: Apache 2.0

// modify from
// https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/blob/mmdetection/mmdet/ops/dcn/src/deform_conv_cuda.c
// Original license: Apache 2.0

#include <torch/types.h>

#include "deform_conv.h"

#include <cmath>
#include <vector>

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
    at::Tensor data_col);

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
    at::Tensor grad_im);

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
    at::Tensor grad_offset);

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
    at::Tensor data_col);

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
    const int kenerl_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int deformable_group,
    at::Tensor grad_im);

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
    const int kenerl_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int deformable_group,
    at::Tensor grad_offset,
    at::Tensor grad_mask);

void shape_check(
    at::Tensor input,
    at::Tensor offset,
    at::Tensor* gradOutput,
    at::Tensor weight,
    int kH,
    int kW,
    int dH,
    int dW,
    int padH,
    int padW,
    int dilationH,
    int dilationW,
    int group,
    int deformable_group) {
  TORCH_CHECK(
      weight.ndimension() == 4,
      "4D weight tensor (nOutputPlane,nInputPlane,kH,kW) expected, "
      "but got: %s",
      weight.ndimension());

  TORCH_CHECK(weight.is_contiguous(), "weight tensor has to be contiguous");

  TORCH_CHECK(
      kW > 0 && kH > 0,
      "kernel size should be greater than zero, but got kH: %d kW: %d",
      kH,
      kW);

  TORCH_CHECK(
      (weight.size(2) == kH && weight.size(3) == kW),
      "kernel size should be consistent with weight, ",
      "but got kH: %d kW: %d weight.size(2): %d, weight.size(3): %d",
      kH,
      kW,
      weight.size(2),
      weight.size(3));

  TORCH_CHECK(
      dW > 0 && dH > 0,
      "stride should be greater than zero, but got dH: %d dW: %d",
      dH,
      dW);

  TORCH_CHECK(
      dilationW > 0 && dilationH > 0,
      "dilation should be greater than 0, but got dilationH: %d dilationW: %d",
      dilationH,
      dilationW);

  int ndim = input.ndimension();
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;

  if (ndim == 4) {
    dimf++;
    dimh++;
    dimw++;
  }

  TORCH_CHECK(
      ndim == 3 || ndim == 4,
      "3D or 4D input tensor expected but got: %s",
      ndim);

  long nInputPlane = weight.size(1) * group;
  long inputHeight = input.size(dimh);
  long inputWidth = input.size(dimw);
  long nOutputPlane = weight.size(0);
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;

  TORCH_CHECK(
      nInputPlane % deformable_group == 0,
      "input channels must divide deformable group size");

  if (outputWidth < 1 || outputHeight < 1)
    AT_ERROR(
        "Given input size: (%ld x %ld x %ld). "
        "Calculated output size: (%ld x %ld x %ld). Output size is too small",
        nInputPlane,
        inputHeight,
        inputWidth,
        nOutputPlane,
        outputHeight,
        outputWidth);

  TORCH_CHECK(
      input.size(1) == nInputPlane,
      "invalid number of input planes, expected: %d, but got: %d",
      nInputPlane,
      input.size(1));

  TORCH_CHECK(
      (inputHeight + 2 * padH >= kH && inputWidth + 2 * padW >= kW),
      "input image is smaller than kernel");

  TORCH_CHECK(
      (offset.size(2) == outputHeight && offset.size(3) == outputWidth),
      "invalid spatial size of offset, expected height: %d width: %d, but "
      "got height: %d width: %d",
      outputHeight,
      outputWidth,
      offset.size(2),
      offset.size(3));

  TORCH_CHECK(
      (offset.size(1) == deformable_group * 2 * kH * kW),
      "invalid number of channels of offset");

  if (gradOutput != NULL) {
    TORCH_CHECK(
        gradOutput->size(dimf) == nOutputPlane,
        "invalid number of gradOutput planes, expected: %d, but got: %d",
        nOutputPlane,
        gradOutput->size(dimf));

    TORCH_CHECK(
        (gradOutput->size(dimh) == outputHeight &&
         gradOutput->size(dimw) == outputWidth),
        "invalid size of gradOutput, expected height: %d width: %d , but "
        "got height: %d width: %d",
        outputHeight,
        outputWidth,
        gradOutput->size(dimh),
        gradOutput->size(dimw));
  }
}

int deform_conv_forward_cuda(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor offset,
    at::Tensor output,
    at::Tensor columns,
    at::Tensor ones,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH,
    int dilationW,
    int dilationH,
    int group,
    int deformable_group,
    int im2col_step) {
  // todo: resize columns to include im2col: done
  // todo: add im2col_step as input
  // todo: add new output buffer and transpose it to output (or directly
  // transpose output) todo: possibly change data indexing because of
  // parallel_imgs

  shape_check(
      input,
      offset,
      NULL,
      weight,
      kH,
      kW,
      dH,
      dW,
      padH,
      padW,
      dilationH,
      dilationW,
      group,
      deformable_group);

  input = input.contiguous();
  offset = offset.contiguous();
  weight = weight.contiguous();

  int batch = 1;
  if (input.ndimension() == 3) {
    // Force batch
    batch = 0;
    input.unsqueeze_(0);
    offset.unsqueeze_(0);
  }

  // todo: assert batchsize dividable by im2col_step

  long batchSize = input.size(0);
  long nInputPlane = input.size(1);
  long inputHeight = input.size(2);
  long inputWidth = input.size(3);

  long nOutputPlane = weight.size(0);

  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  TORCH_CHECK((offset.size(0) == batchSize), "invalid batch size of offset");

  output = output.view(
      {batchSize / im2col_step,
       im2col_step,
       nOutputPlane,
       outputHeight,
       outputWidth});
  columns = at::zeros(
      {nInputPlane * kW * kH, im2col_step * outputHeight * outputWidth},
      input.options());

  if (ones.ndimension() != 2 ||
      ones.size(0) * ones.size(1) < outputHeight * outputWidth) {
    ones = at::ones({outputHeight, outputWidth}, input.options());
  }

  input = input.view(
      {batchSize / im2col_step,
       im2col_step,
       nInputPlane,
       inputHeight,
       inputWidth});
  offset = offset.view(
      {batchSize / im2col_step,
       im2col_step,
       deformable_group * 2 * kH * kW,
       outputHeight,
       outputWidth});

  at::Tensor output_buffer = at::zeros(
      {batchSize / im2col_step,
       nOutputPlane,
       im2col_step * outputHeight,
       outputWidth},
      output.options());

  output_buffer = output_buffer.view(
      {output_buffer.size(0),
       group,
       output_buffer.size(1) / group,
       output_buffer.size(2),
       output_buffer.size(3)});

  for (int elt = 0; elt < batchSize / im2col_step; elt++) {
    deformable_im2col(
        input[elt],
        offset[elt],
        nInputPlane,
        inputHeight,
        inputWidth,
        kH,
        kW,
        padH,
        padW,
        dH,
        dW,
        dilationH,
        dilationW,
        im2col_step,
        deformable_group,
        columns);

    columns = columns.view({group, columns.size(0) / group, columns.size(1)});
    weight = weight.view(
        {group,
         weight.size(0) / group,
         weight.size(1),
         weight.size(2),
         weight.size(3)});

    for (int g = 0; g < group; g++) {
      output_buffer[elt][g] = output_buffer[elt][g]
                                  .flatten(1)
                                  .addmm_(weight[g].flatten(1), columns[g])
                                  .view_as(output_buffer[elt][g]);
    }
  }

  output_buffer = output_buffer.view(
      {output_buffer.size(0),
       output_buffer.size(1) * output_buffer.size(2),
       output_buffer.size(3),
       output_buffer.size(4)});

  output_buffer = output_buffer.view(
      {batchSize / im2col_step,
       nOutputPlane,
       im2col_step,
       outputHeight,
       outputWidth});
  output_buffer.transpose_(1, 2);
  output.copy_(output_buffer);
  output = output.view({batchSize, nOutputPlane, outputHeight, outputWidth});

  input = input.view({batchSize, nInputPlane, inputHeight, inputWidth});
  offset = offset.view(
      {batchSize, deformable_group * 2 * kH * kW, outputHeight, outputWidth});

  if (batch == 0) {
    output = output.view({nOutputPlane, outputHeight, outputWidth});
    input = input.view({nInputPlane, inputHeight, inputWidth});
    offset = offset.view({offset.size(1), offset.size(2), offset.size(3)});
  }

  return 1;
}

int deform_conv_backward_input_cuda(
    at::Tensor input,
    at::Tensor offset,
    at::Tensor gradOutput,
    at::Tensor gradInput,
    at::Tensor gradOffset,
    at::Tensor weight,
    at::Tensor columns,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH,
    int dilationW,
    int dilationH,
    int group,
    int deformable_group,
    int im2col_step) {
  shape_check(
      input,
      offset,
      &gradOutput,
      weight,
      kH,
      kW,
      dH,
      dW,
      padH,
      padW,
      dilationH,
      dilationW,
      group,
      deformable_group);

  input = input.contiguous();
  offset = offset.contiguous();
  gradOutput = gradOutput.contiguous();
  weight = weight.contiguous();

  int batch = 1;

  if (input.ndimension() == 3) {
    // Force batch
    batch = 0;
    input = input.view({1, input.size(0), input.size(1), input.size(2)});
    offset = offset.view({1, offset.size(0), offset.size(1), offset.size(2)});
    gradOutput = gradOutput.view(
        {1, gradOutput.size(0), gradOutput.size(1), gradOutput.size(2)});
  }

  long batchSize = input.size(0);
  long nInputPlane = input.size(1);
  long inputHeight = input.size(2);
  long inputWidth = input.size(3);

  long nOutputPlane = weight.size(0);

  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  TORCH_CHECK((offset.size(0) == batchSize), 3, "invalid batch size of offset");
  gradInput = gradInput.view({batchSize, nInputPlane, inputHeight, inputWidth});
  columns = at::zeros(
      {nInputPlane * kW * kH, im2col_step * outputHeight * outputWidth},
      input.options());

  // change order of grad output
  gradOutput = gradOutput.view(
      {batchSize / im2col_step,
       im2col_step,
       nOutputPlane,
       outputHeight,
       outputWidth});
  gradOutput.transpose_(1, 2);

  gradInput = gradInput.view(
      {batchSize / im2col_step,
       im2col_step,
       nInputPlane,
       inputHeight,
       inputWidth});
  input = input.view(
      {batchSize / im2col_step,
       im2col_step,
       nInputPlane,
       inputHeight,
       inputWidth});
  gradOffset = gradOffset.view(
      {batchSize / im2col_step,
       im2col_step,
       deformable_group * 2 * kH * kW,
       outputHeight,
       outputWidth});
  offset = offset.view(
      {batchSize / im2col_step,
       im2col_step,
       deformable_group * 2 * kH * kW,
       outputHeight,
       outputWidth});

  for (int elt = 0; elt < batchSize / im2col_step; elt++) {
    // divide into groups
    columns = columns.view({group, columns.size(0) / group, columns.size(1)});
    weight = weight.view(
        {group,
         weight.size(0) / group,
         weight.size(1),
         weight.size(2),
         weight.size(3)});
    gradOutput = gradOutput.view(
        {gradOutput.size(0),
         group,
         gradOutput.size(1) / group,
         gradOutput.size(2),
         gradOutput.size(3),
         gradOutput.size(4)});

    for (int g = 0; g < group; g++) {
      columns[g] = columns[g].addmm_(
          weight[g].flatten(1).transpose(0, 1),
          gradOutput[elt][g].flatten(1),
          0.0f,
          1.0f);
    }

    columns =
        columns.view({columns.size(0) * columns.size(1), columns.size(2)});
    gradOutput = gradOutput.view(
        {gradOutput.size(0),
         gradOutput.size(1) * gradOutput.size(2),
         gradOutput.size(3),
         gradOutput.size(4),
         gradOutput.size(5)});

    deformable_col2im_coord(
        columns,
        input[elt],
        offset[elt],
        nInputPlane,
        inputHeight,
        inputWidth,
        kH,
        kW,
        padH,
        padW,
        dH,
        dW,
        dilationH,
        dilationW,
        im2col_step,
        deformable_group,
        gradOffset[elt]);

    deformable_col2im(
        columns,
        offset[elt],
        nInputPlane,
        inputHeight,
        inputWidth,
        kH,
        kW,
        padH,
        padW,
        dH,
        dW,
        dilationH,
        dilationW,
        im2col_step,
        deformable_group,
        gradInput[elt]);
  }

  gradOutput.transpose_(1, 2);
  gradOutput =
      gradOutput.view({batchSize, nOutputPlane, outputHeight, outputWidth});

  gradInput = gradInput.view({batchSize, nInputPlane, inputHeight, inputWidth});
  input = input.view({batchSize, nInputPlane, inputHeight, inputWidth});
  gradOffset = gradOffset.view(
      {batchSize, deformable_group * 2 * kH * kW, outputHeight, outputWidth});
  offset = offset.view(
      {batchSize, deformable_group * 2 * kH * kW, outputHeight, outputWidth});

  if (batch == 0) {
    gradOutput = gradOutput.view({nOutputPlane, outputHeight, outputWidth});
    input = input.view({nInputPlane, inputHeight, inputWidth});
    gradInput = gradInput.view({nInputPlane, inputHeight, inputWidth});
    offset = offset.view({offset.size(1), offset.size(2), offset.size(3)});
    gradOffset =
        gradOffset.view({offset.size(1), offset.size(2), offset.size(3)});
  }

  return 1;
}

int deform_conv_backward_parameters_cuda(
    at::Tensor input,
    at::Tensor offset,
    at::Tensor gradOutput,
    at::Tensor gradWeight, // at::Tensor gradBias,
    at::Tensor columns,
    at::Tensor ones,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH,
    int dilationW,
    int dilationH,
    int group,
    int deformable_group,
    float scale,
    int im2col_step) {
  // todo: transpose and reshape outGrad
  // todo: reshape columns
  // todo: add im2col_step as input

  shape_check(
      input,
      offset,
      &gradOutput,
      gradWeight,
      kH,
      kW,
      dH,
      dW,
      padH,
      padW,
      dilationH,
      dilationW,
      group,
      deformable_group);

  input = input.contiguous();
  offset = offset.contiguous();
  gradOutput = gradOutput.contiguous();

  int batch = 1;

  if (input.ndimension() == 3) {
    // Force batch
    batch = 0;
    input = input.view(
        at::IntList({1, input.size(0), input.size(1), input.size(2)}));
    gradOutput = gradOutput.view(
        {1, gradOutput.size(0), gradOutput.size(1), gradOutput.size(2)});
  }

  long batchSize = input.size(0);
  long nInputPlane = input.size(1);
  long inputHeight = input.size(2);
  long inputWidth = input.size(3);

  long nOutputPlane = gradWeight.size(0);

  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  TORCH_CHECK((offset.size(0) == batchSize), "invalid batch size of offset");

  columns = at::zeros(
      {nInputPlane * kW * kH, im2col_step * outputHeight * outputWidth},
      input.options());

  gradOutput = gradOutput.view(
      {batchSize / im2col_step,
       im2col_step,
       nOutputPlane,
       outputHeight,
       outputWidth});
  gradOutput.transpose_(1, 2);

  at::Tensor gradOutputBuffer = at::zeros_like(gradOutput);
  gradOutputBuffer = gradOutputBuffer.view(
      {batchSize / im2col_step,
       nOutputPlane,
       im2col_step,
       outputHeight,
       outputWidth});
  gradOutputBuffer.copy_(gradOutput);
  // gradOutput is not contiguous, so we do reshape (instead of view) next
  gradOutputBuffer = gradOutputBuffer.reshape(
      {batchSize / im2col_step,
       nOutputPlane,
       im2col_step * outputHeight,
       outputWidth});

  gradOutput.transpose_(1, 2);
  gradOutput =
      gradOutput.view({batchSize, nOutputPlane, outputHeight, outputWidth});

  input = input.view(
      {batchSize / im2col_step,
       im2col_step,
       nInputPlane,
       inputHeight,
       inputWidth});
  offset = offset.view(
      {batchSize / im2col_step,
       im2col_step,
       deformable_group * 2 * kH * kW,
       outputHeight,
       outputWidth});

  for (int elt = 0; elt < batchSize / im2col_step; elt++) {
    deformable_im2col(
        input[elt],
        offset[elt],
        nInputPlane,
        inputHeight,
        inputWidth,
        kH,
        kW,
        padH,
        padW,
        dH,
        dW,
        dilationH,
        dilationW,
        im2col_step,
        deformable_group,
        columns);

    // divide into group
    gradOutputBuffer = gradOutputBuffer.view(
        {gradOutputBuffer.size(0),
         group,
         gradOutputBuffer.size(1) / group,
         gradOutputBuffer.size(2),
         gradOutputBuffer.size(3)});
    columns = columns.view({group, columns.size(0) / group, columns.size(1)});
    gradWeight = gradWeight.view(
        {group,
         gradWeight.size(0) / group,
         gradWeight.size(1),
         gradWeight.size(2),
         gradWeight.size(3)});

    for (int g = 0; g < group; g++) {
      gradWeight[g] = gradWeight[g]
                          .flatten(1)
                          .addmm_(
                              gradOutputBuffer[elt][g].flatten(1),
                              columns[g].transpose(1, 0),
                              1.0,
                              scale)
                          .view_as(gradWeight[g]);
    }
    gradOutputBuffer = gradOutputBuffer.view(
        {gradOutputBuffer.size(0),
         gradOutputBuffer.size(1) * gradOutputBuffer.size(2),
         gradOutputBuffer.size(3),
         gradOutputBuffer.size(4)});
    columns =
        columns.view({columns.size(0) * columns.size(1), columns.size(2)});
    gradWeight = gradWeight.view(
        {gradWeight.size(0) * gradWeight.size(1),
         gradWeight.size(2),
         gradWeight.size(3),
         gradWeight.size(4)});
  }

  input = input.view({batchSize, nInputPlane, inputHeight, inputWidth});
  offset = offset.view(
      {batchSize, deformable_group * 2 * kH * kW, outputHeight, outputWidth});

  if (batch == 0) {
    gradOutput = gradOutput.view({nOutputPlane, outputHeight, outputWidth});
    input = input.view({nInputPlane, inputHeight, inputWidth});
  }

  return 1;
}

void modulated_deform_conv_cuda_forward(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor ones,
    at::Tensor offset,
    at::Tensor mask,
    at::Tensor output,
    at::Tensor columns,
    int kernel_h,
    int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const int group,
    const int deformable_group,
    const bool with_bias) {
  shape_check(
      input,
      offset,
      NULL,
      weight,
      kernel_h,
      kernel_w,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      dilation_h,
      dilation_w,
      group,
      deformable_group);

  TORCH_CHECK(input.is_contiguous(), "input tensor has to be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight tensor has to be contiguous");

  const int batch = input.size(0);
  const int channels = input.size(1);
  const int height = input.size(2);
  const int width = input.size(3);

  const int channels_out = weight.size(0);
  const int channels_kernel = weight.size(1);
  const int kernel_h_ = weight.size(2);
  const int kernel_w_ = weight.size(3);

  if (kernel_h_ != kernel_h || kernel_w_ != kernel_w)
    AT_ERROR(
        "Input shape and kernel shape wont match: (%d x %d vs %d x %d).",
        kernel_h_,
        kernel_w,
        kernel_h_,
        kernel_w_);
  if (channels != channels_kernel * group)
    AT_ERROR(
        "Input shape and kernel channels wont match: (%d vs %d).",
        channels,
        channels_kernel * group);

  const int height_out =
      (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int width_out =
      (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

  // mask shape check
  TORCH_CHECK(
      (mask.size(2) == height_out && mask.size(3) == width_out),
      "invalid spatial size of mask, expected height: %d width: %d, but "
      "got height: %d width: %d",
      height_out,
      width_out,
      mask.size(2),
      mask.size(3));

  TORCH_CHECK(
      (mask.size(1) == deformable_group * kernel_h * kernel_w),
      "invalid number of channels of mask");

  if (ones.ndimension() != 2 ||
      ones.size(0) * ones.size(1) < height_out * width_out) {
    // Resize plane and fill with ones...
    ones = at::ones({height_out, width_out}, input.options());
  }

  // resize output
  output = output.view({batch, channels_out, height_out, width_out}).zero_();
  // resize temporary columns
  columns = at::zeros(
      {channels * kernel_h * kernel_w, 1 * height_out * width_out},
      input.options());

  output = output.view(
      {output.size(0),
       group,
       output.size(1) / group,
       output.size(2),
       output.size(3)});

  for (int b = 0; b < batch; b++) {
    modulated_deformable_im2col_cuda(
        input[b],
        offset[b],
        mask[b],
        1,
        channels,
        height,
        width,
        height_out,
        width_out,
        kernel_h,
        kernel_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        deformable_group,
        columns);

    // divide into group
    weight = weight.view(
        {group,
         weight.size(0) / group,
         weight.size(1),
         weight.size(2),
         weight.size(3)});
    columns = columns.view({group, columns.size(0) / group, columns.size(1)});

    for (int g = 0; g < group; g++) {
      output[b][g] = output[b][g]
                         .flatten(1)
                         .addmm_(weight[g].flatten(1), columns[g])
                         .view_as(output[b][g]);
    }

    weight = weight.view(
        {weight.size(0) * weight.size(1),
         weight.size(2),
         weight.size(3),
         weight.size(4)});
    columns =
        columns.view({columns.size(0) * columns.size(1), columns.size(2)});
  }

  output = output.view(
      {output.size(0),
       output.size(1) * output.size(2),
       output.size(3),
       output.size(4)});

  if (with_bias) {
    output += bias.view({1, bias.size(0), 1, 1});
  }
}

void modulated_deform_conv_cuda_backward(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor ones,
    at::Tensor offset,
    at::Tensor mask,
    at::Tensor columns,
    at::Tensor grad_input,
    at::Tensor grad_weight,
    at::Tensor grad_bias,
    at::Tensor grad_offset,
    at::Tensor grad_mask,
    at::Tensor grad_output,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int group,
    int deformable_group,
    const bool with_bias) {
  shape_check(
      input,
      offset,
      &grad_output,
      weight,
      kernel_h,
      kernel_w,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      dilation_h,
      dilation_w,
      group,
      deformable_group);

  TORCH_CHECK(input.is_contiguous(), "input tensor has to be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight tensor has to be contiguous");

  const int batch = input.size(0);
  const int channels = input.size(1);
  const int height = input.size(2);
  const int width = input.size(3);

  const int channels_kernel = weight.size(1);
  const int kernel_h_ = weight.size(2);
  const int kernel_w_ = weight.size(3);
  if (kernel_h_ != kernel_h || kernel_w_ != kernel_w)
    AT_ERROR(
        "Input shape and kernel shape wont match: (%d x %d vs %d x %d).",
        kernel_h_,
        kernel_w,
        kernel_h_,
        kernel_w_);
  if (channels != channels_kernel * group)
    AT_ERROR(
        "Input shape and kernel channels wont match: (%d vs %d).",
        channels,
        channels_kernel * group);

  const int height_out =
      (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int width_out =
      (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

  // mask shape check
  TORCH_CHECK(
      (mask.size(2) == height_out && mask.size(3) == width_out),
      "invalid spatial size of mask, expected height: %d width: %d, but "
      "got height: %d width: %d",
      height_out,
      width_out,
      mask.size(2),
      mask.size(3));

  TORCH_CHECK(
      (mask.size(1) == deformable_group * kernel_h * kernel_w),
      "invalid number of channels of mask");

  if (ones.ndimension() != 2 ||
      ones.size(0) * ones.size(1) < height_out * width_out) {
    // Resize plane and fill with ones...
    ones = at::ones({height_out, width_out}, input.options());
  }

  grad_input = grad_input.view({batch, channels, height, width});
  columns = at::zeros(
      {channels * kernel_h * kernel_w, height_out * width_out},
      input.options());

  grad_output = grad_output.view(
      {grad_output.size(0),
       group,
       grad_output.size(1) / group,
       grad_output.size(2),
       grad_output.size(3)});

  for (int b = 0; b < batch; b++) {
    // divide int group
    columns = columns.view({group, columns.size(0) / group, columns.size(1)});
    weight = weight.view(
        {group,
         weight.size(0) / group,
         weight.size(1),
         weight.size(2),
         weight.size(3)});

    for (int g = 0; g < group; g++) {
      columns[g].addmm_(
          weight[g].flatten(1).transpose(0, 1),
          grad_output[b][g].flatten(1),
          0.0f,
          1.0f);
    }

    columns =
        columns.view({columns.size(0) * columns.size(1), columns.size(2)});
    weight = weight.view(
        {weight.size(0) * weight.size(1),
         weight.size(2),
         weight.size(3),
         weight.size(4)});

    // gradient w.r.t. input coordinate data
    modulated_deformable_col2im_coord_cuda(
        columns,
        input[b],
        offset[b],
        mask[b],
        1,
        channels,
        height,
        width,
        height_out,
        width_out,
        kernel_h,
        kernel_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        deformable_group,
        grad_offset[b],
        grad_mask[b]);
    // gradient w.r.t. input data
    modulated_deformable_col2im_cuda(
        columns,
        offset[b],
        mask[b],
        1,
        channels,
        height,
        width,
        height_out,
        width_out,
        kernel_h,
        kernel_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        deformable_group,
        grad_input[b]);

    // gradient w.r.t. weight, dWeight should accumulate across the batch and
    // group
    modulated_deformable_im2col_cuda(
        input[b],
        offset[b],
        mask[b],
        1,
        channels,
        height,
        width,
        height_out,
        width_out,
        kernel_h,
        kernel_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        deformable_group,
        columns);

    columns = columns.view({group, columns.size(0) / group, columns.size(1)});
    grad_weight = grad_weight.view(
        {group,
         grad_weight.size(0) / group,
         grad_weight.size(1),
         grad_weight.size(2),
         grad_weight.size(3)});
    if (with_bias)
      grad_bias = grad_bias.view({group, grad_bias.size(0) / group});

    for (int g = 0; g < group; g++) {
      grad_weight[g] =
          grad_weight[g]
              .flatten(1)
              .addmm_(grad_output[b][g].flatten(1), columns[g].transpose(0, 1))
              .view_as(grad_weight[g]);
      if (with_bias) {
        grad_bias[g] =
            grad_bias[g]
                .view({-1, 1})
                .addmm_(grad_output[b][g].flatten(1), ones.view({-1, 1}))
                .view(-1);
      }
    }

    columns =
        columns.view({columns.size(0) * columns.size(1), columns.size(2)});
    grad_weight = grad_weight.view(
        {grad_weight.size(0) * grad_weight.size(1),
         grad_weight.size(2),
         grad_weight.size(3),
         grad_weight.size(4)});
    if (with_bias)
      grad_bias = grad_bias.view({grad_bias.size(0) * grad_bias.size(1)});
  }
  grad_output = grad_output.view(
      {grad_output.size(0) * grad_output.size(1),
       grad_output.size(2),
       grad_output.size(3),
       grad_output.size(4)});
}

} // namespace detectron2
