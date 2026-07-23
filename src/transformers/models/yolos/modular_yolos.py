from typing import Optional

import numpy as np
import torch
from torch import nn
from torchvision.transforms.v2 import functional as tvF

from transformers.models.detr.image_processing_detr import DetrImageProcessor
from transformers.models.detr.image_processing_pil_detr import DetrImageProcessorPil

from ...image_transforms import center_to_corners_format
from ...image_utils import PILImageResampling, SizeDict, get_image_size_for_max_height_width
from ...utils import TensorType, logging, requires_backends


logger = logging.get_logger(__name__)


def get_size_with_aspect_ratio_yolos(
    image_size: tuple[int, int], size: int, max_size: int | None = None, mod_size: int = 16
) -> tuple[int, int]:
    """
    Computes the output image size given the input image size and the desired output size, while ensuring that both
    height and width are multiples of `mod_size`.

    This mirrors the YOLOS-specific behavior used in the torch/fast backends and is required so that all YOLOS
    image processing backends (PIL, torchvision, fast) produce identical output shapes.
    """
    height, width = image_size
    raw_size = None
    if max_size is not None:
        min_original_size = float(min((height, width)))
        max_original_size = float(max((height, width)))
        if max_original_size / min_original_size * size > max_size:
            raw_size = max_size * min_original_size / max_original_size
            size = int(round(raw_size))

    if width < height:
        ow = size
        if max_size is not None and raw_size is not None:
            oh = int(raw_size * height / width)
        else:
            oh = int(size * height / width)
    elif (height <= width and height == size) or (width <= height and width == size):
        oh, ow = height, width
    else:
        oh = size
        if max_size is not None and raw_size is not None:
            ow = int(raw_size * width / height)
        else:
            ow = int(size * width / height)

    if mod_size is not None:
        ow = ow - (ow % mod_size)
        oh = oh - (oh % mod_size)

    return (oh, ow)


class YolosImageProcessor(DetrImageProcessor):
    def resize(
        self,
        image: torch.Tensor,
        size: SizeDict,
        resample: Optional["PILImageResampling | tvF.InterpolationMode | int"] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Resize the image to the given size. Size can be `min_size` (scalar) or `(height, width)` tuple. If size is an
        int, smaller edge of the image will be matched to this number.

        Args:
            image (`torch.Tensor`):
                Image to resize.
            size (`SizeDict`):
                Size of the image's `(height, width)` dimensions after resizing. Available options are:
                    - `{"height": int, "width": int}`: The image will be resized to the exact size `(height, width)`.
                        Do NOT keep the aspect ratio.
                    - `{"shortest_edge": int, "longest_edge": int}`: The image will be resized to a maximum size respecting
                        the aspect ratio and keeping the shortest edge less or equal to `shortest_edge` and the longest edge
                        less or equal to `longest_edge`.
                    - `{"max_height": int, "max_width": int}`: The image will be resized to the maximum size respecting the
                        aspect ratio and keeping the height less or equal to `max_height` and the width less or equal to
                        `max_width`.
            resample (`PILImageResampling | tvF.InterpolationMode | int`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                Resampling filter to use if resizing the image.
        """
        if size.shortest_edge and size.longest_edge:
            # Resize the image so that the shortest edge or the longest edge is of the given size
            # while maintaining the aspect ratio of the original image.
            new_size = get_size_with_aspect_ratio_yolos(image.shape[-2:], size.shortest_edge, size.longest_edge)
        elif size.max_height and size.max_width:
            new_size = get_image_size_for_max_height_width(image.shape[-2:], size.max_height, size.max_width)
        elif size.height and size.width:
            new_size = (size.height, size.width)
        else:
            raise ValueError(
                f"Size must contain 'height' and 'width' keys or 'shortest_edge' and 'longest_edge' keys. Got {size}."
            )

        image = super().resize(
            image, size=SizeDict(height=new_size[0], width=new_size[1]), resample=resample, **kwargs
        )
        return image

    def post_process_object_detection(
        self, outputs, threshold: float = 0.5, target_sizes: TensorType | list[tuple] = None
    ):
        """
        Converts the raw output of [`YolosForObjectDetection`] into final bounding boxes in (top_left_x, top_left_y,
        bottom_right_x, bottom_right_y) format. Only supports PyTorch.

        Args:
            outputs ([`YolosObjectDetectionOutput`]):
                Raw outputs of the model.
            threshold (`float`, *optional*):
                Score threshold to keep object detection predictions.
            target_sizes (`torch.Tensor` or `list[tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`tuple[int, int]`) containing the target size
                `(height, width)` of each image in the batch. If unset, predictions will not be resized.
        Returns:
            `list[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model.
        """
        out_logits, out_bbox = outputs.logits, outputs.pred_boxes

        if target_sizes is not None:
            if len(out_logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

        prob = nn.functional.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # Convert to [x0, y0, x1, y1] format
        boxes = center_to_corners_format(out_bbox)

        # Convert from relative [0, 1] to absolute [0, height] coordinates
        if target_sizes is not None:
            if isinstance(target_sizes, list):
                img_h = torch.Tensor([i[0] for i in target_sizes])
                img_w = torch.Tensor([i[1] for i in target_sizes])
            else:
                img_h, img_w = target_sizes.unbind(1)

            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
            boxes = boxes * scale_fct[:, None, :]

        results = []
        for s, l, b in zip(scores, labels, boxes):
            score = s[s > threshold]
            label = l[s > threshold]
            box = b[s > threshold]
            results.append({"scores": score, "labels": label, "boxes": box})

        return results

    def post_process_instance_segmentation(self):
        raise NotImplementedError("Segmentation post-processing is not implemented for Deformable DETR yet.")

    def post_process_semantic_segmentation(self):
        raise NotImplementedError("Semantic segmentation post-processing is not implemented for Deformable DETR yet.")

    def post_process_panoptic_segmentation(self):
        raise NotImplementedError("Panoptic segmentation post-processing is not implemented for Deformable DETR yet.")


class YolosImageProcessorPil(DetrImageProcessorPil):
    def resize(
        self,
        image: np.ndarray,
        size: SizeDict,
        resample: Optional["PILImageResampling"] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize the image to the given size. Size can be `min_size` (scalar) or `(height, width)` tuple. If size is an
        int, smaller edge of the image will be matched to this number.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`SizeDict`):
                Size of the image's `(height, width)` dimensions after resizing. Available options are:
                    - `{"height": int, "width": int}`: The image will be resized to the exact size `(height, width)`.
                        Do NOT keep the aspect ratio.
                    - `{"shortest_edge": int, "longest_edge": int}`: The image will be resized to a maximum size respecting
                        the aspect ratio and keeping the shortest edge less or equal to `shortest_edge` and the longest edge
                        less or equal to `longest_edge`.
                    - `{"max_height": int, "max_width": int}`: The image will be resized to the maximum size respecting the
                        aspect ratio and keeping the height less or equal to `max_height` and the width less or equal to
                        `max_width`.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                Resampling filter to use if resizing the image.
        """
        resample = resample if resample is not None else self.resample

        if size.shortest_edge and size.longest_edge:
            # Resize the image so that the shortest edge or the longest edge is of the given size
            # while maintaining the aspect ratio of the original image.
            new_size = get_size_with_aspect_ratio_yolos(
                image.shape[-2:],
                size.shortest_edge,
                size.longest_edge or size.shortest_edge,
            )
        elif size.max_height and size.max_width:
            new_size = get_image_size_for_max_height_width(image.shape[-2:], size.max_height, size.max_width)
        elif size.height and size.width:
            new_size = (size.height, size.width)
        else:
            raise ValueError(
                f"Size must contain 'height' and 'width' keys or 'shortest_edge' and 'longest_edge' keys. Got {size}."
            )

        image = super().resize(
            image,
            size=SizeDict(height=new_size[0], width=new_size[1]),
            resample=resample,
            **kwargs,
        )
        return image

    def post_process_object_detection(
        self, outputs, threshold: float = 0.5, target_sizes: TensorType | list[tuple] = None
    ):
        """
        Converts the raw output of [`YolosForObjectDetection`] into final bounding boxes in (top_left_x, top_left_y,
        bottom_right_x, bottom_right_y) format. Only supports PyTorch.

        Args:
            outputs ([`YolosObjectDetectionOutput`]):
                Raw outputs of the model.
            threshold (`float`, *optional*):
                Score threshold to keep object detection predictions.
            target_sizes (`torch.Tensor` or `list[tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`tuple[int, int]`) containing the target size
                `(height, width)` of each image in the batch. If unset, predictions will not be resized.
        Returns:
            `list[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model.
        """
        requires_backends(self, ["torch"])
        out_logits, out_bbox = outputs.logits, outputs.pred_boxes

        if target_sizes is not None:
            if len(out_logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

        prob = nn.functional.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # Convert to [x0, y0, x1, y1] format
        boxes = center_to_corners_format(out_bbox)

        # Convert from relative [0, 1] to absolute [0, height] coordinates
        if target_sizes is not None:
            if isinstance(target_sizes, list):
                img_h = torch.Tensor([i[0] for i in target_sizes])
                img_w = torch.Tensor([i[1] for i in target_sizes])
            else:
                img_h, img_w = target_sizes.unbind(1)

            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
            boxes = boxes * scale_fct[:, None, :]

        results = []
        for s, l, b in zip(scores, labels, boxes):
            score = s[s > threshold]
            label = l[s > threshold]
            box = b[s > threshold]
            results.append({"scores": score, "labels": label, "boxes": box})

        return results

    def post_process_instance_segmentation(self):
        raise NotImplementedError("Segmentation post-processing is not implemented for Deformable DETR yet.")

    def post_process_semantic_segmentation(self):
        raise NotImplementedError("Semantic segmentation post-processing is not implemented for Deformable DETR yet.")

    def post_process_panoptic_segmentation(self):
        raise NotImplementedError("Panoptic segmentation post-processing is not implemented for Deformable DETR yet.")


__all__ = ["YolosImageProcessor", "YolosImageProcessorPil"]
