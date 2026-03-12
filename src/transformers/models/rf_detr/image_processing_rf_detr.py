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
"""Image processor class for RF-DETR."""

from __future__ import annotations

from typing import Any

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import center_to_corners_format, to_channel_dimension_format
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    make_flat_list_of_images,
    to_numpy_array,
    valid_images,
)
from ...utils import (
    TensorType,
    is_torch_available,
    is_torchvision_available,
    logging,
    requires_backends,
)


if is_torch_available():
    import torch
    from torch import nn

if is_torchvision_available():
    import torchvision.transforms.functional as torchvision_transforms


logger = logging.get_logger(__name__)


class RfDetrImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: dict[str, int] | int | None = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_normalize: bool = True,
        image_mean: list[float] | None = None,
        image_std: list[float] | None = None,
        num_top_queries: int | None = 300,
        **kwargs,
    ):
        super().__init__(**kwargs)
        size = {"height": 512, "width": 512} if size is None else size
        size = get_size_dict(size, default_to_square=True)

        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD
        self.num_top_queries = num_top_queries

    def _to_torch_tensor(
        self,
        image: ImageInput,
        input_data_format: ChannelDimension | str | None = None,
    ) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            image_tensor = image.to(dtype=torch.float32)
            if input_data_format is not None and ChannelDimension(input_data_format) == ChannelDimension.LAST:
                image_tensor = image_tensor.permute(2, 0, 1).contiguous()
            return image_tensor

        if input_data_format is not None:
            image = to_numpy_array(image)
            image = to_channel_dimension_format(
                image=image, channel_dim=ChannelDimension.LAST, input_channel_dim=ChannelDimension(input_data_format)
            )
        return torchvision_transforms.to_tensor(image)

    def preprocess(
        self,
        images: ImageInput | list[ImageInput],
        do_resize: bool | None = None,
        size: dict[str, int] | int | None = None,
        resample: PILImageResampling | None = None,
        do_normalize: bool | None = None,
        image_mean: list[float] | None = None,
        image_std: list[float] | None = None,
        return_tensors: str | TensorType | None = None,
        input_data_format: ChannelDimension | str | None = None,
        **kwargs,
    ) -> BatchFeature:
        requires_backends(self, ["torch", "torchvision"])
        if kwargs:
            logger.warning_once(
                f"Ignoring unsupported keyword arguments for RfDetrImageProcessor: {list(kwargs.keys())}"
            )

        images = make_flat_list_of_images(images)
        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Expected either PIL images, numpy arrays, torch tensors, or a list of those."
            )

        do_resize = self.do_resize if do_resize is None else do_resize
        size = self.size if size is None else get_size_dict(size, default_to_square=True)
        resample = self.resample if resample is None else resample
        do_normalize = self.do_normalize if do_normalize is None else do_normalize
        image_mean = self.image_mean if image_mean is None else image_mean
        image_std = self.image_std if image_std is None else image_std

        target_size = (size["height"], size["width"])

        processed_images = []
        for image in images:
            image_tensor = self._to_torch_tensor(image=image, input_data_format=input_data_format)
            if image_tensor.ndim != 3:
                raise ValueError(
                    f"Invalid image shape. Expected shape `(C, H, W)`, but got `{tuple(image_tensor.shape)}`."
                )
            if image_tensor.shape[0] != 3:
                raise ValueError(
                    f"Invalid image shape. Expected 3 channels (RGB), but got {image_tensor.shape[0]} channels."
                )

            if (image_tensor > 1).any():
                raise ValueError(
                    "Image has pixel values above 1. Please ensure the image is normalized (scaled to [0, 1])."
                )

            if do_normalize:
                image_tensor = torchvision_transforms.normalize(image_tensor, image_mean, image_std)

            if do_resize:
                image_tensor = torchvision_transforms.resize(image_tensor, target_size, interpolation=resample)

            processed_images.append(image_tensor)

        if return_tensors in [TensorType.PYTORCH, "pt"]:
            pixel_values = torch.stack(processed_images, dim=0)
        else:
            pixel_values = [image.cpu().numpy() for image in processed_images]

        return BatchFeature(data={"pixel_values": pixel_values}, tensor_type=return_tensors)

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


__all__ = ["RfDetrImageProcessor"]
