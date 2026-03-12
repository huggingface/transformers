# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fast image processor class for RF-DETR."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ...image_processing_utils_fast import BaseImageProcessorFast, SizeDict
from ...image_transforms import center_to_corners_format
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    ImageInput,
    ImageType,
    PILImageResampling,
    get_image_type,
)
from ...utils import TensorType, auto_docstring, is_torchvision_available, requires_backends
from ...utils.import_utils import requires


if is_torchvision_available():
    import torchvision.transforms.v2.functional as tvF


@auto_docstring
@requires(backends=("torchvision", "torch"))
class RfDetrImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    do_resize = True
    do_rescale = False
    do_normalize = True
    size = {"height": 512, "width": 512}
    default_to_square = True
    model_input_names = ["pixel_values"]
    num_top_queries = 300

    def _process_image(
        self,
        image: ImageInput,
        do_convert_rgb: bool | None = None,
        input_data_format: str | ChannelDimension | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        image_type = get_image_type(image)
        image = super()._process_image(
            image=image,
            do_convert_rgb=do_convert_rgb,
            input_data_format=input_data_format,
            device=device,
        )

        # Match RF-DETR preprocessing semantics:
        # PIL and uint8 NumPy inputs are scaled to [0, 1], torch inputs are not automatically scaled.
        if image_type == ImageType.PIL:
            image = image.to(dtype=torch.float32) / 255.0
        elif image_type == ImageType.NUMPY and image.dtype == torch.uint8:
            image = image.to(dtype=torch.float32) / 255.0
        else:
            image = image.to(dtype=torch.float32)

        return image

    def _preprocess(
        self,
        images: list[torch.Tensor],
        do_resize: bool,
        size: SizeDict,
        interpolation: tvF.InterpolationMode | None,
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        do_pad: bool | None,
        pad_size: SizeDict | None,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ):
        for image in images:
            if image.ndim != 3:
                raise ValueError(f"Invalid image shape. Expected shape `(C, H, W)`, but got `{tuple(image.shape)}`.")
            if image.shape[0] != 3:
                raise ValueError(f"Invalid image shape. Expected 3 channels (RGB), but got {image.shape[0]} channels.")
            if (image > 1).any():
                raise ValueError(
                    "Image has pixel values above 1. Please ensure the image is normalized (scaled to [0, 1])."
                )

        return super()._preprocess(
            images=images,
            do_resize=do_resize,
            size=size,
            interpolation=interpolation,
            do_center_crop=do_center_crop,
            crop_size=crop_size,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_pad=do_pad,
            pad_size=pad_size,
            disable_grouping=disable_grouping,
            return_tensors=return_tensors,
            **kwargs,
        )

    # Copied from transformers.models.rf_detr.image_processing_rf_detr.RfDetrImageProcessor.post_process_object_detection
    def post_process_object_detection(
        self,
        outputs: Any,
        threshold: float = 0.5,
        target_sizes: torch.Tensor | list[tuple[int, int]] | None = None,
        num_top_queries: int | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        requires_backends(self, ["torch"])

        out_logits = outputs.logits if hasattr(outputs, "logits") else outputs["pred_logits"]
        out_bbox = outputs.pred_boxes if hasattr(outputs, "pred_boxes") else outputs["pred_boxes"]

        if target_sizes is not None and len(out_logits) != len(target_sizes):
            raise ValueError("Make sure that you pass in as many target sizes as the batch dimension of the logits.")

        if num_top_queries is None:
            num_top_queries = self.num_top_queries if self.num_top_queries is not None else out_logits.shape[1]
        num_top_queries = min(num_top_queries, out_logits.shape[1] * out_logits.shape[2])
        num_classes = out_logits.shape[2]

        scores = nn.functional.sigmoid(out_logits)
        scores, topk_indexes = torch.topk(scores.flatten(1), num_top_queries, dim=1)
        labels = topk_indexes % num_classes
        topk_boxes = topk_indexes // num_classes

        boxes = center_to_corners_format(out_bbox)
        boxes = boxes.gather(dim=1, index=topk_boxes.unsqueeze(-1).repeat(1, 1, boxes.shape[-1]))

        if target_sizes is not None:
            if isinstance(target_sizes, list):
                img_h, img_w = torch.as_tensor(target_sizes).unbind(1)
            else:
                img_h, img_w = target_sizes.unbind(1)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
            boxes = boxes * scale_fct[:, None, :]

        results = []
        for score, label, box in zip(scores, labels, boxes):
            keep = score > threshold
            results.append(
                {
                    "scores": score[keep],
                    "labels": label[keep],
                    "boxes": box[keep],
                }
            )
        return results

    # Copied from transformers.models.rf_detr.image_processing_rf_detr.RfDetrImageProcessor.post_process_instance_segmentation
    def post_process_instance_segmentation(
        self,
        outputs: Any,
        threshold: float = 0.5,
        mask_threshold: float = 0.0,
        target_sizes: torch.Tensor | list[tuple[int, int]] | None = None,
        num_top_queries: int | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        requires_backends(self, ["torch"])

        out_masks = outputs.pred_masks if hasattr(outputs, "pred_masks") else outputs.get("pred_masks")
        if out_masks is None:
            raise ValueError("`outputs` must contain `pred_masks` for instance segmentation post-processing.")

        out_logits = outputs.logits if hasattr(outputs, "logits") else outputs["pred_logits"]
        out_bbox = outputs.pred_boxes if hasattr(outputs, "pred_boxes") else outputs["pred_boxes"]

        if target_sizes is not None and len(out_logits) != len(target_sizes):
            raise ValueError("Make sure that you pass in as many target sizes as the batch dimension of the logits.")

        if num_top_queries is None:
            num_top_queries = self.num_top_queries if self.num_top_queries is not None else out_logits.shape[1]
        num_top_queries = min(num_top_queries, out_logits.shape[1] * out_logits.shape[2])
        num_classes = out_logits.shape[2]

        scores = nn.functional.sigmoid(out_logits)
        scores, topk_indexes = torch.topk(scores.flatten(1), num_top_queries, dim=1)
        labels = topk_indexes % num_classes
        topk_boxes = topk_indexes // num_classes

        boxes = center_to_corners_format(out_bbox)
        boxes = boxes.gather(dim=1, index=topk_boxes.unsqueeze(-1).repeat(1, 1, boxes.shape[-1]))

        if target_sizes is not None:
            if isinstance(target_sizes, list):
                img_h, img_w = torch.as_tensor(target_sizes).unbind(1)
            else:
                img_h, img_w = target_sizes.unbind(1)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
            boxes = boxes * scale_fct[:, None, :]

        results = []
        for batch_idx in range(out_masks.shape[0]):
            box_indexes = topk_boxes[batch_idx]
            masks = out_masks[batch_idx].gather(
                dim=0,
                index=box_indexes.unsqueeze(-1).unsqueeze(-1).repeat(1, out_masks.shape[-2], out_masks.shape[-1]),
            )

            if target_sizes is not None:
                height, width = (
                    target_sizes[batch_idx].tolist() if not isinstance(target_sizes, list) else target_sizes[batch_idx]
                )
                masks = nn.functional.interpolate(
                    masks.unsqueeze(1),
                    size=(int(height), int(width)),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                masks = masks.unsqueeze(1)

            masks = masks > mask_threshold
            keep = scores[batch_idx] > threshold
            results.append(
                {
                    "scores": scores[batch_idx][keep],
                    "labels": labels[batch_idx][keep],
                    "boxes": boxes[batch_idx][keep],
                    "masks": masks[keep],
                }
            )

        return results


__all__ = ["RfDetrImageProcessorFast"]
