# coding=utf-8
# Copyright 2025 The Meta AI Authors and The HuggingFace Team. All rights reserved.
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
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from ...image_processing_utils import BatchFeature, get_size_dict
from ...image_processing_utils_fast import BaseImageProcessorFast
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
    pil_torch_interpolation_mapping,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring


def _scale_boxes(boxes, target_sizes):
    """
    Scale batch of bounding boxes to the target sizes.

    Args:
        boxes (`torch.Tensor` of shape `(batch_size, num_boxes, 4)`):
            Bounding boxes to scale. Each box is expected to be in (x1, y1, x2, y2) format.
        target_sizes (`list[tuple[int, int]]` or `torch.Tensor` of shape `(batch_size, 2)`):
            Target sizes to scale the boxes to. Each target size is expected to be in (height, width) format.

    Returns:
        `torch.Tensor` of shape `(batch_size, num_boxes, 4)`: Scaled bounding boxes.
    """

    if isinstance(target_sizes, (list, tuple)):
        image_height = torch.tensor([i[0] for i in target_sizes])
        image_width = torch.tensor([i[1] for i in target_sizes])
    elif isinstance(target_sizes, torch.Tensor):
        image_height, image_width = target_sizes.unbind(1)
    else:
        raise TypeError("`target_sizes` must be a list, tuple or torch.Tensor")

    scale_factor = torch.stack([image_width, image_height, image_width, image_height], dim=1)
    scale_factor = scale_factor.unsqueeze(1).to(boxes.device)
    boxes = boxes * scale_factor
    return boxes


class Sam3FastImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    mask_size (`dict[str, int]`, *optional*):
        The size `{"height": int, "width": int}` to resize the segmentation maps to.
    """

    mask_size: dict[str, int]


@auto_docstring
class Sam3ImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 1008, "width": 1008}
    mask_size = {"height": 288, "width": 288}
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True

    valid_kwargs = Sam3FastImageProcessorKwargs

    # modular artefacts
    do_pad = None
    pad_size = None
    mask_pad_size = None

    def __init__(self, **kwargs: Unpack[Sam3FastImageProcessorKwargs]):
        super().__init__(**kwargs)

    def _further_process_kwargs(
        self,
        size: Optional[SizeDict] = None,
        mask_size: Optional[SizeDict] = None,
        default_to_square: Optional[bool] = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        data_format: Optional[ChannelDimension] = None,
        **kwargs,
    ) -> dict:
        """
        Update kwargs that need further processing before being validated
        Can be overridden by subclasses to customize the processing of kwargs.
        """
        if kwargs is None:
            kwargs = {}
        if size is not None:
            size = SizeDict(**get_size_dict(size=size, default_to_square=default_to_square))
        if mask_size is not None:
            mask_size = SizeDict(**get_size_dict(mask_size, param_name="mask_size"))
        if isinstance(image_mean, list):
            image_mean = tuple(image_mean)
        if isinstance(image_std, list):
            image_std = tuple(image_std)
        if data_format is None:
            data_format = ChannelDimension.FIRST

        kwargs["size"] = size
        kwargs["mask_size"] = mask_size
        kwargs["image_mean"] = image_mean
        kwargs["image_std"] = image_std
        kwargs["data_format"] = data_format

        # torch resize uses interpolation instead of resample
        # Check if resample is an int before checking if it's an instance of PILImageResampling
        # because if pillow < 9.1.0, resample is an int and PILImageResampling is a module.
        # Checking PILImageResampling will fail with error `TypeError: isinstance() arg 2 must be a type or tuple of types`.
        resample = kwargs.pop("resample")
        kwargs["interpolation"] = (
            pil_torch_interpolation_mapping[resample] if isinstance(resample, (PILImageResampling, int)) else resample
        )

        return kwargs

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        segmentation_maps: Optional[ImageInput] = None,
        **kwargs: Unpack[Sam3FastImageProcessorKwargs],
    ) -> BatchFeature:
        r"""
        segmentation_maps (`ImageInput`, *optional*):
            The segmentation maps to preprocess.
        """
        return super().preprocess(images, segmentation_maps, **kwargs)

    def _preprocess_image_like_inputs(
        self,
        images: ImageInput,
        segmentation_maps: Optional[ImageInput],
        do_convert_rgb: bool,
        input_data_format: ChannelDimension,
        device: Optional[Union[str, "torch.device"]] = None,
        **kwargs: Unpack[Sam3FastImageProcessorKwargs],
    ) -> BatchFeature:
        """
        Preprocess image-like inputs.
        """
        images = self._prepare_image_like_inputs(
            images=images, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format, device=device
        )
        original_sizes = [image.shape[-2:] for image in images]
        images_kwargs = kwargs.copy()
        pixel_values = self._preprocess(images, **images_kwargs)
        data = {
            "pixel_values": pixel_values,
            "original_sizes": original_sizes,
        }

        if segmentation_maps is not None:
            processed_segmentation_maps = self._prepare_image_like_inputs(
                images=segmentation_maps,
                expected_ndims=2,
                do_convert_rgb=False,
                input_data_format=ChannelDimension.FIRST,
            )

            segmentation_maps_kwargs = kwargs.copy()
            segmentation_maps_kwargs.update(
                {
                    "do_normalize": False,
                    "do_rescale": False,
                    "interpolation": pil_torch_interpolation_mapping[PILImageResampling.NEAREST],
                    "size": segmentation_maps_kwargs.pop("mask_size"),
                }
            )
            processed_segmentation_maps = self._preprocess(
                images=processed_segmentation_maps, **segmentation_maps_kwargs
            )
            data["labels"] = processed_segmentation_maps.squeeze(1).to(torch.int64)
        return BatchFeature(data=data, tensor_type=kwargs["return_tensors"])

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs,
    ) -> "torch.Tensor":
        return super()._preprocess(images, return_tensors=return_tensors, **kwargs).pixel_values

    def post_process_semantic_segmentation(
        self, outputs, target_sizes: Optional[list[tuple]] = None, threshold: float = 0.5
    ):
        """
        Converts the output of [`Sam3Model`] into semantic segmentation maps.

        Args:
            outputs ([`Sam3ImageSegmentationOutput`]):
                Raw outputs of the model containing semantic_seg.
            target_sizes (`list[tuple]` of length `batch_size`, *optional*):
                List of tuples corresponding to the requested final size (height, width) of each prediction. If unset,
                predictions will not be resized.
            threshold (`float`, *optional*, defaults to 0.5):
                Threshold for binarizing the semantic segmentation masks.

        Returns:
            semantic_segmentation: `list[torch.Tensor]` of length `batch_size`, where each item is a semantic
            segmentation map of shape (height, width) corresponding to the target_sizes entry (if `target_sizes` is
            specified). Each entry is a binary mask (0 or 1).
        """
        # Get semantic segmentation output
        # semantic_seg has shape (batch_size, 1, height, width)
        semantic_logits = outputs.semantic_seg

        if semantic_logits is None:
            raise ValueError(
                "Semantic segmentation output is not available in the model outputs. "
                "Make sure the model was run with semantic segmentation enabled."
            )

        # Apply sigmoid to convert logits to probabilities
        semantic_probs = semantic_logits.sigmoid()

        # Resize and binarize semantic segmentation maps
        if target_sizes is not None:
            if len(semantic_logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

            semantic_segmentation = []

            for idx in range(len(semantic_logits)):
                resized_probs = torch.nn.functional.interpolate(
                    semantic_probs[idx].unsqueeze(dim=0),
                    size=target_sizes[idx],
                    mode="bilinear",
                    align_corners=False,
                )
                # Binarize: values > threshold become 1, otherwise 0
                semantic_map = (resized_probs[0, 0] > threshold).to(torch.long)
                semantic_segmentation.append(semantic_map)
        else:
            # Binarize without resizing
            semantic_segmentation = (semantic_probs[:, 0] > threshold).to(torch.long)
            semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]

        return semantic_segmentation

    def post_process_object_detection(
        self, outputs, threshold: float = 0.3, target_sizes: Optional[list[tuple]] = None
    ):
        """
        Converts the raw output of [`Sam3Model`] into final bounding boxes in (top_left_x, top_left_y,
        bottom_right_x, bottom_right_y) format.

        Args:
            outputs ([`Sam3ImageSegmentationOutput`]):
                Raw outputs of the model containing pred_boxes, pred_logits, and optionally presence_logits.
            threshold (`float`, *optional*, defaults to 0.3):
                Score threshold to keep object detection predictions.
            target_sizes (`list[tuple[int, int]]`, *optional*):
                List of tuples (`tuple[int, int]`) containing the target size `(height, width)` of each image in the
                batch. If unset, predictions will not be resized.

        Returns:
            `list[dict]`: A list of dictionaries, each dictionary containing the following keys:
                - **scores** (`torch.Tensor`): The confidence scores for each predicted box on the image.
                - **boxes** (`torch.Tensor`): Image bounding boxes in (top_left_x, top_left_y, bottom_right_x,
                  bottom_right_y) format.
        """
        pred_logits = outputs.pred_logits  # (batch_size, num_queries)
        pred_boxes = outputs.pred_boxes  # (batch_size, num_queries, 4) in xyxy format
        presence_logits = outputs.presence_logits  # (batch_size, 1) or None

        batch_size = pred_logits.shape[0]

        if target_sizes is not None and len(target_sizes) != batch_size:
            raise ValueError("Make sure that you pass in as many target sizes as images")

        # Compute scores: combine pred_logits with presence_logits if available
        batch_scores = pred_logits.sigmoid()
        if presence_logits is not None:
            presence_scores = presence_logits.sigmoid()  # (batch_size, 1)
            batch_scores = batch_scores * presence_scores  # Broadcast multiplication

        # Boxes are already in xyxy format from the model
        batch_boxes = pred_boxes

        # Convert from relative [0, 1] to absolute [0, height/width] coordinates
        if target_sizes is not None:
            batch_boxes = _scale_boxes(batch_boxes, target_sizes)

        results = []
        for scores, boxes in zip(batch_scores, batch_boxes):
            keep = scores > threshold
            scores = scores[keep]
            boxes = boxes[keep]
            results.append({"scores": scores, "boxes": boxes})

        return results

    def post_process_instance_segmentation(
        self,
        outputs,
        threshold: float = 0.3,
        mask_threshold: float = 0.5,
        target_sizes: Optional[list[tuple]] = None,
    ):
        """
        Converts the raw output of [`Sam3Model`] into instance segmentation predictions with bounding boxes and masks.

        Args:
            outputs ([`Sam3ImageSegmentationOutput`]):
                Raw outputs of the model containing pred_boxes, pred_logits, pred_masks, and optionally
                presence_logits.
            threshold (`float`, *optional*, defaults to 0.3):
                Score threshold to keep instance predictions.
            mask_threshold (`float`, *optional*, defaults to 0.5):
                Threshold for binarizing the predicted masks.
            target_sizes (`list[tuple[int, int]]`, *optional*):
                List of tuples (`tuple[int, int]`) containing the target size `(height, width)` of each image in the
                batch. If unset, predictions will not be resized.

        Returns:
            `list[dict]`: A list of dictionaries, each dictionary containing the following keys:
                - **scores** (`torch.Tensor`): The confidence scores for each predicted instance on the image.
                - **boxes** (`torch.Tensor`): Image bounding boxes in (top_left_x, top_left_y, bottom_right_x,
                  bottom_right_y) format.
                - **masks** (`torch.Tensor`): Binary segmentation masks for each instance, shape (num_instances,
                  height, width).
        """
        pred_logits = outputs.pred_logits  # (batch_size, num_queries)
        pred_boxes = outputs.pred_boxes  # (batch_size, num_queries, 4) in xyxy format
        pred_masks = outputs.pred_masks  # (batch_size, num_queries, height, width)
        presence_logits = outputs.presence_logits  # (batch_size, 1) or None

        batch_size = pred_logits.shape[0]

        if target_sizes is not None and len(target_sizes) != batch_size:
            raise ValueError("Make sure that you pass in as many target sizes as images")

        # Compute scores: combine pred_logits with presence_logits if available
        batch_scores = pred_logits.sigmoid()
        if presence_logits is not None:
            presence_scores = presence_logits.sigmoid()  # (batch_size, 1)
            batch_scores = batch_scores * presence_scores  # Broadcast multiplication

        # Apply sigmoid to mask logits
        batch_masks = pred_masks.sigmoid()

        # Boxes are already in xyxy format from the model
        batch_boxes = pred_boxes

        # Scale boxes to target sizes if provided
        if target_sizes is not None:
            batch_boxes = _scale_boxes(batch_boxes, target_sizes)

        results = []
        for idx, (scores, boxes, masks) in enumerate(zip(batch_scores, batch_boxes, batch_masks)):
            # Filter by score threshold
            keep = scores > threshold
            scores = scores[keep]
            boxes = boxes[keep]
            masks = masks[keep]  # (num_keep, height, width)

            # Resize masks to target size if provided
            if target_sizes is not None:
                target_size = target_sizes[idx]
                if len(masks) > 0:
                    masks = torch.nn.functional.interpolate(
                        masks.unsqueeze(0),  # (1, num_keep, height, width)
                        size=target_size,
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)  # (num_keep, target_height, target_width)

            # Binarize masks
            masks = (masks > mask_threshold).to(torch.long)

            results.append({"scores": scores, "boxes": boxes, "masks": masks})

        return results

    def _apply_non_overlapping_constraints(self, pred_masks: torch.Tensor) -> torch.Tensor:
        """
        Apply non-overlapping constraints to the object scores in pred_masks. Here we
        keep only the highest scoring object at each spatial location in pred_masks.
        """
        batch_size = pred_masks.size(0)
        if batch_size == 1:
            return pred_masks

        device = pred_masks.device
        # "max_obj_inds": object index of the object with the highest score at each location
        max_obj_inds = torch.argmax(pred_masks, dim=0, keepdim=True)
        # "batch_obj_inds": object index of each object slice (along dim 0) in `pred_masks`
        batch_obj_inds = torch.arange(batch_size, device=device)[:, None, None, None]
        keep = max_obj_inds == batch_obj_inds
        # suppress overlapping regions' scores below -10.0 so that the foreground regions
        # don't overlap (here sigmoid(-10.0)=4.5398e-05)
        pred_masks = torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))
        return pred_masks

    def post_process_masks(
        self,
        masks,
        original_sizes,
        mask_threshold=0.0,
        binarize=True,
        max_hole_area=0.0,
        max_sprinkle_area=0.0,
        apply_non_overlapping_constraints=False,
        **kwargs,
    ):
        """
        Remove padding and upscale masks to the original image size.

        Args:
            masks (`Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]]`):
                Batched masks from the mask_decoder in (batch_size, num_channels, height, width) format.
            original_sizes (`Union[torch.Tensor, List[Tuple[int,int]]]`):
                The original sizes of each image before it was resized to the model's expected input shape, in (height,
                width) format.
            mask_threshold (`float`, *optional*, defaults to 0.0):
                Threshold for binarization and post-processing operations.
            binarize (`bool`, *optional*, defaults to `True`):
                Whether to binarize the masks.
            max_hole_area (`float`, *optional*, defaults to 0.0):
                The maximum area of a hole to fill.
            max_sprinkle_area (`float`, *optional*, defaults to 0.0):
                The maximum area of a sprinkle to fill.
            apply_non_overlapping_constraints (`bool`, *optional*, defaults to `False`):
                Whether to apply non-overlapping constraints to the masks.

        Returns:
            (`torch.Tensor`): Batched masks in batch_size, num_channels, height, width) format, where (height, width)
            is given by original_size.
        """
        if isinstance(original_sizes, (torch.Tensor, np.ndarray)):
            original_sizes = original_sizes.tolist()
        # TODO: add connected components kernel for postprocessing
        output_masks = []
        for i, original_size in enumerate(original_sizes):
            if isinstance(masks[i], np.ndarray):
                masks[i] = torch.from_numpy(masks[i])
            elif not isinstance(masks[i], torch.Tensor):
                raise TypeError("Input masks should be a list of `torch.tensors` or a list of `np.ndarray`")
            interpolated_mask = F.interpolate(masks[i], original_size, mode="bilinear", align_corners=False)
            if apply_non_overlapping_constraints:
                interpolated_mask = self._apply_non_overlapping_constraints(interpolated_mask)
            if binarize:
                interpolated_mask = interpolated_mask > mask_threshold
            output_masks.append(interpolated_mask)

        return output_masks


__all__ = ["Sam3ImageProcessorFast"]
