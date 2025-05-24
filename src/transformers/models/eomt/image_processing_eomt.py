# coding=utf-8
# Copyright 2025 Mobile Perception Systems Lab at TU/e and The HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for EoMT."""

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
    PaddingMode,
    pad,
    resize,
)
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    valid_images,
    validate_preprocess_arguments,
)
from ...utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    TensorType,
    filter_out_non_signature_kwargs,
    is_torch_available,
    logging,
)


logger = logging.get_logger(__name__)

if is_torch_available():
    import torch
    import torch.nn.functional as F


# Copied from transformers.models.detr.image_processing_detr.remove_low_and_no_objects
def remove_low_and_no_objects(masks, scores, labels, object_mask_threshold, num_labels):
    """
    Binarize the given masks using `object_mask_threshold`, it returns the associated values of `masks`, `scores` and
    `labels`.

    Args:
        masks (`torch.Tensor`):
            A tensor of shape `(num_queries, height, width)`.
        scores (`torch.Tensor`):
            A tensor of shape `(num_queries)`.
        labels (`torch.Tensor`):
            A tensor of shape `(num_queries)`.
        object_mask_threshold (`float`):
            A number between 0 and 1 used to binarize the masks.
    Raises:
        `ValueError`: Raised when the first dimension doesn't match in all input tensors.
    Returns:
        `Tuple[`torch.Tensor`, `torch.Tensor`, `torch.Tensor`]`: The `masks`, `scores` and `labels` without the region
        < `object_mask_threshold`.
    """
    if not (masks.shape[0] == scores.shape[0] == labels.shape[0]):
        raise ValueError("mask, scores and labels must have the same shape!")

    to_keep = labels.ne(num_labels) & (scores > object_mask_threshold)

    return masks[to_keep], scores[to_keep], labels[to_keep]


def check_segment_validity(mask_labels, mask_probs, k, mask_threshold=0.5, overlap_mask_area_threshold=0.8):
    # Get the mask associated with the k class
    mask_k = mask_labels == k
    mask_k_area = mask_k.sum()

    # Compute the area of all the stuff in query k
    original_mask = mask_probs[k] >= mask_threshold
    original_area = original_mask.sum()

    final_mask = mask_k & original_mask
    final_mask_area = final_mask.sum()

    mask_exists = mask_k_area > 0 and original_area > 0 and final_mask_area > 0

    if mask_exists:
        area_ratio = mask_k_area / original_area
        if not area_ratio.item() > overlap_mask_area_threshold:
            mask_exists = False

    return mask_exists, final_mask


def compute_segments(
    mask_probs,
    pred_scores,
    pred_labels,
    stuff_classes,
    mask_threshold: float = 0.5,
    overlap_mask_area_threshold: float = 0.8,
    target_size: Optional[Tuple[int, int]] = None,
):
    height = mask_probs.shape[1] if target_size is None else target_size[0]
    width = mask_probs.shape[2] if target_size is None else target_size[1]

    segmentation = torch.zeros((height, width), dtype=torch.long, device=mask_probs.device)
    segments: List[Dict] = []

    # Compute per-pixel assignment based on weighted mask scores
    mask_probs = mask_probs.sigmoid()
    mask_labels = (pred_scores[:, None, None] * mask_probs).argmax(0)

    # Keep track of instances of each class
    current_segment_id = 0
    stuff_memory_list: Dict[str, int] = {}

    for k in range(pred_labels.shape[0]):
        pred_class = pred_labels[k].item()

        # Check if mask exists and large enough to be a segment
        mask_exists, final_mask = check_segment_validity(
            mask_labels, mask_probs, k, mask_threshold, overlap_mask_area_threshold
        )

        if not mask_exists:
            continue

        if pred_class in stuff_classes:
            if pred_class in stuff_memory_list:
                segmentation[final_mask] = stuff_memory_list[pred_class]
                continue
            else:
                stuff_memory_list[pred_class] = current_segment_id

        # Fuck forgot to update seg id
        segmentation[final_mask] = current_segment_id
        segment_score = round(pred_scores[k].item(), 6)
        segments.append(
            {
                "id": current_segment_id,
                "label_id": pred_class,
                "score": segment_score,
            }
        )
    return segmentation, segments


class EoMTImageProcessor(BaseImageProcessor):
    r"""
    Constructs a EoMT image processor. The image processor can be used to prepare image(s) and optional targets
    for the model.

    This image processor inherits from [`BaseImageProcessor`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the input to a certain `size`.
        size (`int`, *optional*, defaults to 800):
            Resize the input to the given size. Only has an effect if `do_resize` is set to `True`. If size is a
            sequence like `(width, height)`, output size will be matched to this. If size is an int, smaller edge of
            the image will be matched to this number. i.e, if `height > width`, then image will be rescaled to `(size *
            height / width, size)`.
        resample (`int`, *optional*, defaults to `Resampling.BILINEAR`):
            An optional resampling filter. This can be one of `PIL.Image.Resampling.NEAREST`,
            `PIL.Image.Resampling.BOX`, `PIL.Image.Resampling.BILINEAR`, `PIL.Image.Resampling.HAMMING`,
            `PIL.Image.Resampling.BICUBIC` or `PIL.Image.Resampling.LANCZOS`. Only has an effect if `do_resize` is set
            to `True`.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the input to a certain `scale`.
        rescale_factor (`float`, *optional*, defaults to `1/ 255`):
            Rescale the input by the given factor. Only has an effect if `do_rescale` is set to `True`.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the input with mean and standard deviation.
        image_mean (`int`, *optional*, defaults to `[0.485, 0.456, 0.406]`):
            The sequence of means for each channel, to be used when normalizing images. Defaults to the ImageNet mean.
        image_std (`int`, *optional*, defaults to `[0.229, 0.224, 0.225]`):
            The sequence of standard deviations for each channel, to be used when normalizing images. Defaults to the
            ImageNet std.
        num_labels (`int`, *optional*):
            The number of labels in the segmentation map.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = 640,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        num_labels: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        size = get_size_dict(size, default_to_square=True)

        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD
        self.num_labels = num_labels

    def scale_image_size(self, image_size: Tuple[int, int], segmentation_type: str) -> Tuple[int, int]:
        """
        Scales image dimensions based on the segmentation type.

        For semantic segmentation, scales up to or exceed the target size.
        For instance or panoptic segmentation, scales down to fit within the target size.

        Args:
            image_size (`Tuple[int, int]`):
                Original image size (height, width).
            segmentation_type (`str`):
                One of "semantic", "instance", or "panoptic".

        Returns:
            `Tuple[int, int]`: Scaled image size (height, width).
        """
        target_h, target_w = self.size["height"], self.size["width"]
        orig_h, orig_w = image_size

        # For semantic segmentation: scale up so that both sides are ≥ target size
        if segmentation_type == "semantic":
            scale_factor = max(target_h / orig_h, target_w / orig_w)
        elif segmentation_type in {"instance", "panoptic"}:
            scale_factor = min(target_h / orig_h, target_w / orig_w)
        else:
            raise ValueError(f"Unknown segmentation type: {segmentation_type}")

        output_h = round(orig_h * scale_factor)
        output_w = round(orig_w * scale_factor)

        return (output_h, output_w)

    def resize(
        self,
        image: np.ndarray,
        segmentation_type: str,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format=None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image. The shortest edge of the image is resized to size["shortest_edge"], with the longest edge
        resized to keep the input aspect ratio.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the output image.
            segmentation_type (`str`):
                Type of segmentation task. One of the "semantic", "instance", or "panoptic".
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                Resampling filter to use when resiizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        image_size = get_image_size(image)

        output_size = self.scale_image_size(image_size, segmentation_type)

        image = resize(
            image=image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            return_numpy=True,
            **kwargs,
        )

        return image

    def _preprocessing_semantic_segmentation(self, image: ImageInput, image_index: int) -> Tuple[List, List]:
        """Slices an image into overlapping crops for semantic segmentation."""
        crops, crop_origins = [], []

        image_size = get_image_size(image=image)  # (H, W)
        crop_size = self.size["height"]  # or 'width', both are equal

        longer_side = max(image_size)
        num_crops = math.ceil(longer_side / crop_size)
        total_overlap = num_crops * crop_size - longer_side
        overlap_per_crop = total_overlap / (num_crops - 1) if num_crops > 1 else 0

        for i in range(num_crops):
            start = int(i * (crop_size - overlap_per_crop))
            end = start + crop_size

            if image_size[0] > image_size[1]:
                crop = image[:, start:end, :]
            else:
                crop = image[:, :, start:end]

            crops.append(crop)
            crop_origins.append([image_index, start, end])

        return crops, crop_origins

    def _pad(self, image):
        """Pads the image to the target size using zero padding."""
        height, width = get_image_size(image)
        pad_h = max(0, self.size["height"] - height)
        pad_w = max(0, self.size["width"] - width)

        padding = ((0, pad_h), (0, pad_w))

        # Channel axis is last; default padding format is compatible
        padded_image = pad(image=image, padding=padding, mode=PaddingMode.CONSTANT, constant_values=0.0)
        return padded_image

    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        segmentation_type: str,
        mask_labels: Optional[ImageInput] = None,
        class_labels: Optional[Dict[int, int]] = None,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> BatchFeature:
        """
        Preprocesses images or a batch of images.

        Args:
            images (`ImageInput`):
                Image or batch of images to preprocess.
            segmentation_type (`str`):
                Type of segmentation. Must be one of `"semantic"`, `"instance"`, or `"panoptic"`.
            mask_labels (`ImageInput`, *optional*):
                Ground truth segmentation mask corresponding to input images.
            class_labels (`Dict[int, int]`, *optional*):
                Class labels to map instance/panoptic segments to semantic classes.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the input images.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Target size as a dictionary with `"height"` and `"width"` keys.
            resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
                Resampling filter to use when resizing.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the input images by `rescale_factor`.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Factor to scale image pixel values.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the input images.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Mean for normalization. Single value or list for each channel.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Standard deviation for normalization. Single value or list for each channel.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be `"pt"`, `"tf"`, `"np"`, or `"jax"`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                Channel format of the output image. Either `"channels_first"` or `"channels_last"`.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                Channel format of the input image.
        """

        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        size = get_size_dict(size, default_to_square=False)
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        if segmentation_type not in {"semantic", "instance", "panoptic"}:
            raise ValueError(
                f"Invalid segmentation type: {segmentation_type}. Must be one of ['semantic', 'instance', 'panoptic']."
            )

        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        if do_resize:
            images = [
                self.resize(
                    image,
                    segmentation_type=segmentation_type,
                    resample=resample,
                    data_format=data_format,
                    input_data_format=input_data_format,
                )
                for image in images
            ]

        # crop_origins is only used for semantic segmentation.
        processed_images, crop_origins = [], []

        if segmentation_type == "semantic":
            all_crops, all_origins = [], []
            for idx, img in enumerate(images):
                crops, origins = self._preprocessing_semantic_segmentation(img, idx)
                all_crops.extend(crops)
                all_origins.extend(origins)
            processed_images = all_crops
            crop_origins = all_origins
        else:
            processed_images = [self._pad_image(img) for img in images]

        if do_rescale:
            images = [
                self.rescale(img, scale=rescale_factor, input_data_format=input_data_format)
                for img in processed_images
            ]

        if do_normalize:
            image_mean = np.array(image_mean).reshape(1, -1, 1, 1)
            image_std = np.array(image_std).reshape(1, -1, 1, 1)
            images = (images - image_mean) / image_std

        if mask_labels is not None:
            mask_labels = [self._pad(mask) for mask in mask_labels]

        # Normalize not working properly fix later
        # if do_normalize:
        #     images = [self.normalize(image, mean=image_mean, std=image_std, input_data_format=ChannelDimension.FIRST) for image in crops_list]

        output = {
            "pixel_values": images,
            "crop_origins": crop_origins,
            "mask_labels": mask_labels,
            "class_labels": class_labels,
        }
        return BatchFeature(output, tensor_type=return_tensors)

    def _revert_preprocessing_semantic(
        self,
        segmentation_logits: torch.Tensor,
        crop_origins: List[Tuple[int, int, int]],
        original_image_sizes: List[Tuple[int, int]],
    ) -> List[torch.Tensor]:
        """
        Reconstructs full-size semantic segmentation logits from cropped image predictions.

        Args:
            segmentation_logits (`torch.Tensor`):
                A tensor of shape `(num_crops, num_classes, crop_height, crop_width)` representing predicted logits
                for each image crop.
            crop_origins (`List[Tuple[int, int, int]]`):
                A list of tuples where each tuple contains:
                - `image_index` (int): Index of the original image this crop belongs to.
                - `start` (int): Start pixel index of the crop along the long dimension (height or width).
                - `end` (int): End pixel index of the crop along the long dimension.
            original_image_sizes (`List[Tuple[int, int]]`):
                List of original (height, width) dimensions for each image before preprocessing.
        """
        num_classes = segmentation_logits.shape[1]
        aggregated_logits = []
        crop_counts = []

        for image_size in original_image_sizes:
            height, width = self.scale_image_size(image_size, segmentation_type="semantic")
            aggregated_logits.append(torch.zeros((num_classes, height, width), device=segmentation_logits.device))
            crop_counts.append(torch.zeros((num_classes, height, width), device=segmentation_logits.device))

        # Stitch crops back into full-sized logit maps
        for crop_idx, (image_idx, crop_start, crop_end) in enumerate(crop_origins):
            if original_image_sizes[image_idx][0] > original_image_sizes[image_idx][1]:
                aggregated_logits[image_idx][:, crop_start:crop_end, :] += segmentation_logits[crop_idx]
                crop_counts[image_idx][:, crop_start:crop_end, :] += 1
            else:
                aggregated_logits[image_idx][:, :, crop_start:crop_end] += segmentation_logits[crop_idx]
                crop_counts[image_idx][:, :, crop_start:crop_end] += 1

        # Normalize and resize logits to original image size
        reconstructed_logits = []
        for idx, (logit_sum, count) in enumerate(zip(aggregated_logits, crop_counts)):
            averaged_logits = logit_sum / count.clamp(min=1)
            resized_logits = F.interpolate(
                averaged_logits[None, ...],
                size=original_image_sizes[idx],
                mode="bilinear",
                align_corners=False,
            )[0]

            print("resized_logits shape(squeeze if required):", resized_logits.shape)
            reconstructed_logits.append(resized_logits)

        return reconstructed_logits

    def _revert_preprocessing_panoptic(
        self,
        segmentation_logits: torch.Tensor,
        original_image_sizes: List[Tuple[int, int]],
    ) -> List[torch.Tensor]:
        """Restores panoptic segmentation logits to their original image resolutions."""
        resized_logits = []

        for idx, original_size in enumerate(original_image_sizes):
            target_height, target_width = self.scale_image_size(original_size, segmentation_type="panoptic")
            cropped_logits = segmentation_logits[idx][:, :target_height, :target_width]
            upsampled_logits = F.interpolate(
                cropped_logits[None, ...], size=original_size, mode="bilinear", align_corners=False
            )[0]
            resized_logits.append(upsampled_logits)
        return resized_logits

    def postprocess_semantic_segmentation(
        self,
        outputs: Tuple[np.ndarray, np.ndarray],
        crop_origins: List[Tuple[int, int, int]],
        original_image_sizes: List[Tuple[int, int]],
    ) -> np.ndarray:
        """Post-processes model outputs into final semantic segmentation prediction."""

        masks_queries_logits = torch.tensor(outputs[0])  # [batch_size, num_queries, height, width]
        class_queries_logits = torch.tensor(outputs[1])  # [batch_size, num_queries, num_classes+1]

        size = (self.size["height"], self.size["width"])
        masks_queries_logits = torch.nn.functional.interpolate(
            masks_queries_logits,
            size=size,
            mode="bilinear",
        )

        # Remove the null class `[..., :-1]`
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

        # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
        segmentation_logits = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)

        output_logits = self._revert_preprocessing_semantic(segmentation_logits, crop_origins, original_image_sizes)

        preds = output_logits[0].argmax(0).cpu().numpy()
        return preds

    def post_process_panoptic_segmentation(
        self,
        outputs,
        original_image_sizes,
        stuff_classes: List[int] = [0],
        threshold: float = 0.8,
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
    ):
        """Post-processes model outputs into final panoptic segmentation prediction."""

        masks_queries_logits = torch.tensor(outputs[0])  # [batch_size, num_queries, height, width]
        class_queries_logits = torch.tensor(outputs[1])  # [batch_size, num_queries, num_classes+1]

        batch_size = class_queries_logits.shape[0]
        num_labels = class_queries_logits.shape[-1] - 1

        size = (self.size["height"], self.size["width"])
        masks_queries_logits = torch.nn.functional.interpolate(
            masks_queries_logits,
            size=size,
            mode="bilinear",
        )

        mask_probs = self._revert_preprocessing_panoptic(masks_queries_logits, original_image_sizes)

        pred_scores, pred_labels = class_queries_logits.softmax(dim=-1).max(-1)

        results: List = []

        for i in range(batch_size):
            mask_probs, pred_scores, pred_labels = remove_low_and_no_objects(
                mask_probs[i], pred_scores[i], pred_labels[i], threshold, num_labels
            )

            # No mask found
            if mask_probs.shape[0] <= 0:
                height, width = original_image_sizes[i] if original_image_sizes is not None else mask_probs.shape[1:]
                segmentation = torch.zeros((height, width)) - 1
                results.append({"segmentation": segmentation, "segments_info": []})
                continue

            segmentation, segments = compute_segments(
                mask_probs=mask_probs,
                pred_scores=pred_scores,
                pred_labels=pred_labels,
                stuff_classes=stuff_classes,
                mask_threshold=mask_threshold,
                overlap_mask_area_threshold=overlap_mask_area_threshold,
                target_size=original_image_sizes[i] if original_image_sizes is not None else None,
            )

            results.append({"segmentation": segmentation, "segments_info": segments})
        return results


__all__ = ["EoMTImageProcessor"]
