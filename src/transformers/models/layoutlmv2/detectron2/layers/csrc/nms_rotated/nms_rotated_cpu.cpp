// Copyright (c) Facebook, Inc. and its affiliates.
#include "../box_iou_rotated/box_iou_rotated_utils.h"
#include "nms_rotated.h"

namespace detectron2 {

template <typename scalar_t>
at::Tensor nms_rotated_cpu_kernel(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const double iou_threshold) {
  // nms_rotated_cpu_kernel is modified from torchvision's nms_cpu_kernel,
  // however, the code in this function is much shorter because
  // we delegate the IoU computation for rotated boxes to
  // the single_box_iou_rotated function in box_iou_rotated_utils.h
  AT_ASSERTM(dets.device().is_cpu(), "dets must be a CPU tensor");
  AT_ASSERTM(scores.device().is_cpu(), "scores must be a CPU tensor");
  AT_ASSERTM(
      dets.scalar_type() == scores.scalar_type(),
      "dets should have the same type as scores");

  if (dets.numel() == 0) {
    return at::empty({0}, dets.options().dtype(at::kLong));
  }

  auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));

  auto ndets = dets.size(0);
  at::Tensor suppressed_t = at::zeros({ndets}, dets.options().dtype(at::kByte));
  at::Tensor keep_t = at::zeros({ndets}, dets.options().dtype(at::kLong));

  auto suppressed = suppressed_t.data_ptr<uint8_t>();
  auto keep = keep_t.data_ptr<int64_t>();
  auto order = order_t.data_ptr<int64_t>();

  int64_t num_to_keep = 0;

  for (int64_t _i = 0; _i < ndets; _i++) {
    auto i = order[_i];
    if (suppressed[i] == 1) {
      continue;
    }

    keep[num_to_keep++] = i;

    for (int64_t _j = _i + 1; _j < ndets; _j++) {
      auto j = order[_j];
      if (suppressed[j] == 1) {
        continue;
      }

      auto ovr = single_box_iou_rotated<scalar_t>(
          dets[i].data_ptr<scalar_t>(), dets[j].data_ptr<scalar_t>());
      if (ovr >= iou_threshold) {
        suppressed[j] = 1;
      }
    }
  }
  return keep_t.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep);
}

at::Tensor nms_rotated_cpu(
    // input must be contiguous
    const at::Tensor& dets,
    const at::Tensor& scores,
    const double iou_threshold) {
  auto result = at::empty({0}, dets.options());

  AT_DISPATCH_FLOATING_TYPES(dets.scalar_type(), "nms_rotated", [&] {
    result = nms_rotated_cpu_kernel<scalar_t>(dets, scores, iou_threshold);
  });
  return result;
}

} // namespace detectron2
