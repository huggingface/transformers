// Copyright (c) Facebook, Inc. and its affiliates.

#include <torch/extension.h>
#include "ROIAlignRotated/ROIAlignRotated.h"
#include "box_iou_rotated/box_iou_rotated.h"
#include "cocoeval/cocoeval.h"
#include "deformable/deform_conv.h"
#include "nms_rotated/nms_rotated.h"

namespace detectron2 {

#if defined(WITH_CUDA) || defined(WITH_HIP)
extern int get_cudart_version();
#endif

std::string get_cuda_version() {
#if defined(WITH_CUDA) || defined(WITH_HIP)
  std::ostringstream oss;

#if defined(WITH_CUDA)
  oss << "CUDA ";
#else
  oss << "HIP ";
#endif

  // copied from
  // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/cuda/detail/CUDAHooks.cpp#L231
  auto printCudaStyleVersion = [&](int v) {
    oss << (v / 1000) << "." << (v / 10 % 100);
    if (v % 10 != 0) {
      oss << "." << (v % 10);
    }
  };
  printCudaStyleVersion(get_cudart_version());
  return oss.str();
#else // neither CUDA nor HIP
  return std::string("not available");
#endif
}

bool has_cuda() {
#if defined(WITH_CUDA)
  return true;
#else
  return false;
#endif
}

// similar to
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Version.cpp
std::string get_compiler_version() {
  std::ostringstream ss;
#if defined(__GNUC__)
#ifndef __clang__

#if ((__GNUC__ <= 4) && (__GNUC_MINOR__ <= 8))
#error "GCC >= 4.9 is required!"
#endif

  { ss << "GCC " << __GNUC__ << "." << __GNUC_MINOR__; }
#endif
#endif

#if defined(__clang_major__)
  {
    ss << "clang " << __clang_major__ << "." << __clang_minor__ << "."
       << __clang_patchlevel__;
  }
#endif

#if defined(_MSC_VER)
  { ss << "MSVC " << _MSC_FULL_VER; }
#endif
  return ss.str();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_compiler_version", &get_compiler_version, "get_compiler_version");
  m.def("get_cuda_version", &get_cuda_version, "get_cuda_version");
  m.def("has_cuda", &has_cuda, "has_cuda");

  m.def("box_iou_rotated", &box_iou_rotated, "IoU for rotated boxes");

  m.def("deform_conv_forward", &deform_conv_forward, "deform_conv_forward");
  m.def(
      "deform_conv_backward_input",
      &deform_conv_backward_input,
      "deform_conv_backward_input");
  m.def(
      "deform_conv_backward_filter",
      &deform_conv_backward_filter,
      "deform_conv_backward_filter");
  m.def(
      "modulated_deform_conv_forward",
      &modulated_deform_conv_forward,
      "modulated_deform_conv_forward");
  m.def(
      "modulated_deform_conv_backward",
      &modulated_deform_conv_backward,
      "modulated_deform_conv_backward");

  m.def("nms_rotated", &nms_rotated, "NMS for rotated boxes");

  m.def(
      "roi_align_rotated_forward",
      &ROIAlignRotated_forward,
      "Forward pass for Rotated ROI-Align Operator");
  m.def(
      "roi_align_rotated_backward",
      &ROIAlignRotated_backward,
      "Backward pass for Rotated ROI-Align Operator");

  m.def("COCOevalAccumulate", &COCOeval::Accumulate, "COCOeval::Accumulate");
  m.def(
      "COCOevalEvaluateImages",
      &COCOeval::EvaluateImages,
      "COCOeval::EvaluateImages");
  pybind11::class_<COCOeval::InstanceAnnotation>(m, "InstanceAnnotation")
      .def(pybind11::init<uint64_t, double, double, bool, bool>());
  pybind11::class_<COCOeval::ImageEvaluation>(m, "ImageEvaluation")
      .def(pybind11::init<>());
}

#ifdef TORCH_LIBRARY
TORCH_LIBRARY(detectron2, m) {
  m.def("nms_rotated", &nms_rotated);
}
#endif
} // namespace detectron2
