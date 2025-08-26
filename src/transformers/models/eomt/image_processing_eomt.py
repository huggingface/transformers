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
from typing import Optional, Union

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
    infer_channel_dimension_format,
    make_flat_list_of_images,
    make_list_of_images,
    to_numpy_array,
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


# Adapted from transformers.models.maskformer.image_processing_maskformer.convert_segmentation_map_to_binary_masks
def convert_segmentation_map_to_binary_masks(
    segmentation_map: "np.ndarray",
    instance_id_to_semantic_id: Optional[dict[int, int]] = None,
    ignore_index: Optional[int] = None,
):
    if ignore_index is not None:
        segmentation_map = np.where(segmentation_map == 0, ignore_index, segmentation_map - 1)

    # Get unique ids (class or instance ids based on input)
    all_labels = np.unique(segmentation_map)

    # Drop background label if applicable
    if ignore_index is not None:
        all_labels = all_labels[all_labels != ignore_index]

    # Generate a binary mask for each object instance
    binary_masks = [(segmentation_map == i) for i in all_labels]

    # Stack the binary masks
    if binary_masks:
        binary_masks = np.stack(binary_masks, axis=0)
    else:
        binary_masks = np.zeros((0, *segmentation_map.shape))

    # Convert instance ids to class ids
    if instance_id_to_semantic_id is not None:
        labels = np.zeros(all_labels.shape[0])

        for label in all_labels:
            class_id = instance_id_to_semantic_id[label + 1 if ignore_index is not None else label]
            labels[all_labels == label] = class_id - 1 if ignore_index is not None else class_id
    else:
        labels = all_labels

    return binary_masks.astype(np.float32), labels.astype(np.int64)


def get_size_with_aspect_ratio(image_size, size, max_size=None) -> tuple[int, int]:
    """
    Computes the output image size given the input image size and the desired output size.

    Args:
        image_size (`tuple[int, int]`):
            The input image size.
        size (`int`):
            The desired output size.
        max_size (`int`, *optional*):
            The maximum allowed output size.
    """
    height, width = image_size
    raw_size = None
    if max_size is not None:
        min_original_size = float(min((height, width)))
        max_original_size = float(max((height, width)))
        if max_original_size / min_original_size * size > max_size:
            raw_size = max_size * min_original_size / max_original_size
            size = int(round(raw_size))

    if (height <= width and height == size) or (width <= height and width == size):
        oh, ow = height, width
    elif width < height:
        ow = size
        if max_size is not None and raw_size is not None:
            oh = round(raw_size * height / width)
        else:
            oh = round(size * height / width)
    else:
        oh = size
        if max_size is not None and raw_size is not None:
            ow = round(raw_size * width / height)
        else:
            ow = round(size * width / height)

    return (oh, ow)


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
        `tuple[`torch.Tensor`, `torch.Tensor`, `torch.Tensor`]`: The `masks`, `scores` and `labels` without the region
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
    target_size: Optional[tuple[int, int]] = None,
):
    height = mask_probs.shape[1] if target_size is None else target_size[0]
    width = mask_probs.shape[2] if target_size is None else target_size[1]

    segmentation = torch.zeros((height, width), dtype=torch.long, device=mask_probs.device) - 1
    segments: list[dict] = []

    # Compute per-pixel assignment based on weighted mask scores
    mask_probs = mask_probs.sigmoid()
    mask_labels = (pred_scores[:, None, None] * mask_probs).argmax(0)

    # Keep track of instances of each class
    current_segment_id = 0
    stuff_memory_list: dict[str, int] = {}

    for k in range(pred_labels.shape[0]):
        pred_class = pred_labels[k].item()

        # Check if mask exists and large enough to be a segment
        mask_exists, final_mask = check_segment_validity(
            mask_labels, mask_probs, k, mask_threshold, overlap_mask_area_threshold
        )

        if not mask_exists:
            continue

        if stuff_classes and pred_class in stuff_classes:
            if pred_class in stuff_memory_list:
                segmentation[final_mask] = stuff_memory_list[pred_class]
                continue
            else:
                stuff_memory_list[pred_class] = current_segment_id

        segmentation[final_mask] = current_segment_id
        segment_score = round(pred_scores[k].item(), 6)
        segments.append(
            {
                "id": current_segment_id,
                "label_id": pred_class,
                "score": segment_score,
            }
        )
        current_segment_id += 1
    return segmentation, segments


def get_target_size(size_dict: dict[str, int]) -> tuple[int, int]:
    """Returns the height and width from a size dict."""
    target_height = size_dict["shortest_edge"]
    target_width = size_dict.get("longest_edge") or target_height

    return target_height, target_width


class EomtImageProcessor(BaseImageProcessor):
    r"""
    Constructs a EoMT image processor. The image processor can be used to prepare image(s) and optional targets
    for the model.

    This image processor inherits from [`BaseImageProcessor`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the input to a certain `size`.
        size (`int`, *optional*, defaults to 640):
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
        do_split_image (`bool`, *optional*, defaults to `False`):
            Whether to split the input images into overlapping patches for semantic segmentation. If set to `True`, the
            input images will be split into patches of size `size["shortest_edge"]` with an overlap between patches.
            Otherwise, the input images will be padded to the target size.
        do_pad (`bool`, *optional*, defaults to `False`):
            Whether to pad the image. If `True`, will pad the patch dimension of the images in the batch to the largest
            number of patches in the batch. Padding will be applied to the bottom and right with zeros.
        image_mean (`int`, *optional*, defaults to `[0.485, 0.456, 0.406]`):
            The sequence of means for each channel, to be used when normalizing images. Defaults to the ImageNet mean.
        image_std (`int`, *optional*, defaults to `[0.229, 0.224, 0.225]`):
            The sequence of standard deviations for each channel, to be used when normalizing images. Defaults to the
            ImageNet std.
        ignore_index (`int`, *optional*):
            Label to be assigned to background pixels in segmentation maps. If provided, segmentation map pixels
            denoted with 0 (background) will be replaced with `ignore_index`.
        num_labels (`int`, *optional*):
            The number of labels in the segmentation map.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[dict[str, int]] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        do_split_image: bool = False,
        do_pad: bool = False,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        ignore_index: Optional[int] = None,
        num_labels: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        size = size if size is not None else {"shortest_edge": 640, "longest_edge": 640}
        size = get_size_dict(size, default_to_square=False)

        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.do_split_image = do_split_image
        self.do_pad = do_pad
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD
        self.ignore_index = ignore_index
        self.num_labels = num_labels

    def resize(
        self,
        image: np.ndarray,
        size: dict,
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
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                Resampling filter to use when resiizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        image_size = get_image_size(image)
        output_size = get_size_with_aspect_ratio(image_size, size["shortest_edge"], size["longest_edge"])

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

    def _split_image(self, image: ImageInput, size: dict, image_index: int) -> tuple[list, list]:
        """Slices an image into overlapping patches for semantic segmentation."""

        patches, patch_offsets = [], []

        image_size = get_image_size(image)
        patch_size = size["shortest_edge"]

        longer_side = max(image_size)
        num_patches = math.ceil(longer_side / patch_size)
        total_overlap = num_patches * patch_size - longer_side
        overlap_per_patch = total_overlap / (num_patches - 1) if num_patches > 1 else 0

        for i in range(num_patches):
            start = int(i * (patch_size - overlap_per_patch))
            end = start + patch_size

            if image_size[0] > image_size[1]:
                patch = image[:, start:end, :]
            else:
                patch = image[:, :, start:end]

            patches.append(patch)
            patch_offsets.append([image_index, start, end])

        return patches, patch_offsets

    def _pad(self, image: ImageInput, size: dict) -> np.ndarray:
        """Pads the image to the target size using zero padding."""
        height, width = get_image_size(image)

        target_height, target_width = get_target_size(size)
        pad_h = max(0, target_height - height)
        pad_w = max(0, target_width - width)

        padding = ((0, pad_h), (0, pad_w))

        # Channel axis is last; default padding format is compatible
        padded_image = pad(image=image, padding=padding, mode=PaddingMode.CONSTANT, constant_values=0.0)
        return padded_image

    def _preprocess_images(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Optional[dict[str, int]] = None,
        resample: PILImageResampling = None,
        do_split_image: Optional[bool] = None,
        do_pad: Optional[bool] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """Preprocesses a batch of images."""
        images = [to_numpy_array(image) for image in images]

        if do_resize:
            images = [
                self.resize(
                    image,
                    size=size,
                    resample=resample,
                    data_format=data_format,
                    input_data_format=input_data_format,
                )
                for image in images
            ]

        processed_images, patch_offsets = [], []

        if do_split_image:
            for idx, img in enumerate(images):
                patches, offsets = self._split_image(img, size, idx)
                processed_images.extend(patches)
                patch_offsets.extend(offsets)

            images = processed_images

        if do_pad:
            images = [self._pad(img, size) for img in images]

        if do_rescale:
            images = [self.rescale(img, scale=rescale_factor, input_data_format=input_data_format) for img in images]

        if do_normalize:
            images = [
                self.normalize(
                    image,
                    mean=image_mean,
                    std=image_std,
                    input_data_format=input_data_format,
                )
                for image in images
            ]

        return images, patch_offsets

    def _preprocess_mask(
        self,
        segmentation_map: ImageInput,
        do_resize: Optional[bool] = False,
        do_pad: Optional[bool] = False,
        size: Optional[dict[str, int]] = None,
        resample: PILImageResampling = None,
        data_format: Union[str, ChannelDimension] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """Preprocesses a single mask."""
        # Add channel dimension if missing - needed for certain transformations
        if segmentation_map.ndim == 2:
            added_channel_dim = True
            segmentation_map = segmentation_map[None, ...]
            input_data_format = ChannelDimension.FIRST
        else:
            added_channel_dim = False
            if input_data_format is None:
                input_data_format = infer_channel_dimension_format(segmentation_map)

        if do_resize:
            segmentation_map = self.resize(
                segmentation_map,
                size=size,
                resample=resample,
                data_format=data_format,
            )

        if do_pad:
            segmentation_map = self._pad(segmentation_map, size)

        # Remove extra channel dimension if added for processing
        if added_channel_dim:
            segmentation_map = segmentation_map.squeeze(0)
        return torch.from_numpy(segmentation_map)

    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        segmentation_maps: Optional[Union[list[dict[int, int]], dict[int, int]]] = None,
        instance_id_to_semantic_id: Optional[dict[int, int]] = None,
        do_split_image: Optional[bool] = None,
        do_resize: Optional[bool] = None,
        size: Optional[dict[str, int]] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        do_pad: Optional[bool] = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        ignore_index: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> BatchFeature:
        """
        Preprocesses images or a batch of images.

        Args:
            images (`ImageInput`):
                Image or batch of images to preprocess.
            segmentation_maps (`ImageInput`, *optional*):
                The corresponding semantic segmentation maps with the pixel-wise annotations.
            instance_id_to_semantic_id (`list[dict[int, int]]` or `dict[int, int]`, *optional*):
                A mapping between object instance ids and class ids.
            do_split_image (`bool`, *optional*, defaults to `self.do_split_image`):
                Whether to split the input images into overlapping patches for semantic segmentation.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the input images.
            size (`dict[str, int]`, *optional*, defaults to `self.size`):
                Target size as a dictionary with `"shortest_edge"` and `"longest_edge"` keys.
            resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
                Resampling filter to use when resizing.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the input images by `rescale_factor`.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Factor to scale image pixel values.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the input images.
            do_pad (`bool`, *optional*, defaults to `False`):
                Whether to pad the image. If `True`, will pad the patch dimension of the images in the batch to the largest
                number of patches in the batch. Padding will be applied to the bottom and right with zeros.
            image_mean (`float` or `list[float]`, *optional*, defaults to `self.image_mean`):
                Mean for normalization. Single value or list for each channel.
            image_std (`float` or `list[float]`, *optional*, defaults to `self.image_std`):
                Standard deviation for normalization. Single value or list for each channel.
            ignore_index (`int`, *optional*):
                Label to be assigned to background pixels in segmentation maps. If provided, segmentation map pixels
                denoted with 0 (background) will be replaced with `ignore_index`.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be `"pt"`, `"tf"`, `"np"`, or `"jax"`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                Channel format of the output image. Either `"channels_first"` or `"channels_last"`.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                Channel format of the input image.
        """

        do_split_image = do_split_image if do_split_image is not None else self.do_split_image
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        size = get_size_dict(size, default_to_square=False)
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        do_pad = do_pad if do_pad is not None else self.do_pad
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        ignore_index = ignore_index if ignore_index is not None else self.ignore_index

        images = self.fetch_images(images)
        images = make_flat_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
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

        pixel_values_list, patch_offsets = self._preprocess_images(
            images=images,
            do_resize=do_resize,
            size=size,
            resample=resample,
            do_split_image=do_split_image,
            do_pad=do_pad,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            data_format=data_format,
            input_data_format=input_data_format,
        )

        if segmentation_maps is not None:
            segmentation_maps = make_list_of_images(segmentation_maps, expected_ndims=2)
            segmentation_maps = [to_numpy_array(mask) for mask in segmentation_maps]

            segmentation_maps = [
                self._preprocess_mask(
                    segmentation_map,
                    do_resize=do_resize,
                    do_pad=do_pad,
                    size=size,
                    resample=PILImageResampling.NEAREST,
                    data_format=data_format,
                    input_data_format=input_data_format,
                )
                for segmentation_map in segmentation_maps
            ]

        encoded_inputs = self.encode_inputs(
            pixel_values_list,
            segmentation_maps,
            instance_id_to_semantic_id,
            ignore_index,
            return_tensors,
            input_data_format=data_format,
        )

        if do_split_image and patch_offsets:
            encoded_inputs["patch_offsets"] = [torch.tensor(offsets) for offsets in patch_offsets]

        return encoded_inputs

    def encode_inputs(
        self,
        pixel_values_list: list[ImageInput],
        segmentation_maps: ImageInput = None,
        instance_id_to_semantic_id: Optional[Union[list[dict[int, int]], dict[int, int]]] = None,
        ignore_index: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Pad images up to the largest image in a batch and create a corresponding `pixel_mask`.

        EoMT addresses semantic segmentation with a mask classification paradigm, thus input segmentation maps
        will be converted to lists of binary masks and their respective labels. Let's see an example, assuming
        `segmentation_maps = [[2,6,7,9]]`, the output will contain `mask_labels =
        [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]` (four binary masks) and `class_labels = [2,6,7,9]`, the labels for
        each mask.

        Args:
            pixel_values_list (`list[ImageInput]`):
                list of images (pixel values) to be padded. Each image should be a tensor of shape `(channels, height,
                width)`.

            segmentation_maps (`ImageInput`, *optional*):
                The corresponding semantic segmentation maps with the pixel-wise annotations.

             (`bool`, *optional*, defaults to `True`):
                Whether or not to pad images up to the largest image in a batch and create a pixel mask.

                If left to the default, will return a pixel mask that is:

                - 1 for pixels that are real (i.e. **not masked**),
                - 0 for pixels that are padding (i.e. **masked**).

            instance_id_to_semantic_id (`list[dict[int, int]]` or `dict[int, int]`, *optional*):
                A mapping between object instance ids and class ids. If passed, `segmentation_maps` is treated as an
                instance segmentation map where each pixel represents an instance id. Can be provided as a single
                dictionary with a global/dataset-level mapping or as a list of dictionaries (one per image), to map
                instance ids in each image separately.

            return_tensors (`str` or [`~file_utils.TensorType`], *optional*):
                If set, will return tensors instead of NumPy arrays. If set to `'pt'`, return PyTorch `torch.Tensor`
                objects.

            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **pixel_values** -- Pixel values to be fed to a model.
            - **mask_labels** -- Optional list of mask labels of shape `(labels, height, width)` to be fed to a model
              (when `annotations` are provided).
            - **class_labels** -- Optional list of class labels of shape `(labels)` to be fed to a model (when
              `annotations` are provided). They identify the labels of `mask_labels`, e.g. the label of
              `mask_labels[i][j]` if `class_labels[i][j]`.
        """
        ignore_index = self.ignore_index if ignore_index is None else ignore_index

        pixel_values_list = [to_numpy_array(pixel_values) for pixel_values in pixel_values_list]

        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(pixel_values_list[0])

        encoded_inputs = BatchFeature({"pixel_values": pixel_values_list}, tensor_type=return_tensors)

        if segmentation_maps is not None:
            mask_labels = []
            class_labels = []
            # Convert to list of binary masks and labels
            for idx, segmentation_map in enumerate(segmentation_maps):
                segmentation_map = to_numpy_array(segmentation_map)
                if isinstance(instance_id_to_semantic_id, list):
                    instance_id = instance_id_to_semantic_id[idx]
                else:
                    instance_id = instance_id_to_semantic_id
                # Use instance2class_id mapping per image
                masks, classes = convert_segmentation_map_to_binary_masks(
                    segmentation_map,
                    instance_id,
                    ignore_index=ignore_index,
                )

                mask_labels.append(torch.from_numpy(masks))
                class_labels.append(torch.from_numpy(classes))

            # we cannot batch them since they don't share a common class size
            encoded_inputs["mask_labels"] = mask_labels
            encoded_inputs["class_labels"] = class_labels

        return encoded_inputs

    def merge_image_patches(
        self,
        segmentation_logits: torch.Tensor,
        patch_offsets: list[tuple[int, int, int]],
        target_sizes: list[tuple[int, int]],
        size: dict[str, int],
    ) -> list[torch.Tensor]:
        """
        Reconstructs full-size semantic segmentation logits from patch predictions.

        Args:
            segmentation_logits (`torch.Tensor`):
                A tensor of shape `(num_patches, num_classes, patch_height, patch_width)` representing predicted logits
                for each image patch.
            patch_offsets (`list[tuple[int, int, int]]`):
                A list of tuples where each tuple contains:
                - `image_index` (int): Index of the original image this patch belongs to.
                - `start` (int): Start pixel index of the patch along the long dimension (height or width).
                - `end` (int): End pixel index of the patch along the long dimension.
            target_sizes (`list[tuple[int, int]]`):
                list of original (height, width) dimensions for each image before preprocessing.
            size (`dict[str, int]`):
                A size dict which was used to resize.
        """
        num_classes = segmentation_logits.shape[1]
        aggregated_logits = []
        patch_counts = []

        for image_size in target_sizes:
            height, width = get_size_with_aspect_ratio(image_size, size["shortest_edge"], size["longest_edge"])
            aggregated_logits.append(torch.zeros((num_classes, height, width), device=segmentation_logits.device))
            patch_counts.append(torch.zeros((num_classes, height, width), device=segmentation_logits.device))

        # Stitch patches back into full-sized logit maps
        for patch_idx, (image_idx, patch_start, patch_end) in enumerate(patch_offsets):
            if target_sizes[image_idx][0] > target_sizes[image_idx][1]:
                aggregated_logits[image_idx][:, patch_start:patch_end, :] += segmentation_logits[patch_idx]
                patch_counts[image_idx][:, patch_start:patch_end, :] += 1
            else:
                aggregated_logits[image_idx][:, :, patch_start:patch_end] += segmentation_logits[patch_idx]
                patch_counts[image_idx][:, :, patch_start:patch_end] += 1

        # Normalize and resize logits to original image size
        reconstructed_logits = []
        for idx, (logit_sum, count) in enumerate(zip(aggregated_logits, patch_counts)):
            averaged_logits = logit_sum / count.clamp(min=1)
            resized_logits = F.interpolate(
                averaged_logits[None, ...],
                size=target_sizes[idx],
                mode="bilinear",
                align_corners=False,
            )[0]

            reconstructed_logits.append(resized_logits)

        return reconstructed_logits

    def unpad_image(
        self,
        segmentation_logits: torch.Tensor,
        target_sizes: list[tuple[int, int]],
        size: dict[str, int],
    ) -> list[torch.Tensor]:
        """Restores panoptic segmentation logits to their original image resolutions."""

        resized_logits = []

        for idx, original_size in enumerate(target_sizes):
            target_height, target_width = get_size_with_aspect_ratio(
                original_size, size["shortest_edge"], size["longest_edge"]
            )
            cropped_logits = segmentation_logits[idx][:, :target_height, :target_width]
            upsampled_logits = F.interpolate(
                cropped_logits[None, ...], size=original_size, mode="bilinear", align_corners=False
            )[0]
            resized_logits.append(upsampled_logits)
        return resized_logits

    def post_process_semantic_segmentation(
        self,
        outputs,
        target_sizes: list[tuple[int, int]],
        size: Optional[dict[str, int]] = None,
    ) -> np.ndarray:
        """Post-processes model outputs into final semantic segmentation prediction."""

        size = size if size is not None else self.size

        masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]
        class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
        patch_offsets = outputs.patch_offsets

        output_size = get_target_size(size)
        masks_queries_logits = F.interpolate(
            masks_queries_logits,
            size=output_size,
            mode="bilinear",
        )

        # Remove the null class `[..., :-1]`
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

        segmentation_logits = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)

        output_logits = self.merge_image_patches(segmentation_logits, patch_offsets, target_sizes, size)

        preds = [logit.argmax(dim=0) for logit in output_logits]
        return preds

    def post_process_panoptic_segmentation(
        self,
        outputs,
        target_sizes: list[tuple[int, int]],
        threshold: float = 0.8,
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
        stuff_classes: Optional[list[int]] = None,
        size: Optional[dict[str, int]] = None,
    ):
        """Post-processes model outputs into final panoptic segmentation prediction."""

        size = size if size is not None else self.size

        masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]
        class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]

        batch_size = class_queries_logits.shape[0]
        num_labels = class_queries_logits.shape[-1] - 1

        output_size = get_target_size(size)
        masks_queries_logits = F.interpolate(
            masks_queries_logits,
            size=output_size,
            mode="bilinear",
        )

        mask_probs_batch = self.unpad_image(masks_queries_logits, target_sizes, size)
        pred_scores_batch, pred_labels_batch = class_queries_logits.softmax(dim=-1).max(-1)

        results: list = []

        for i in range(batch_size):
            mask_probs, pred_scores, pred_labels = remove_low_and_no_objects(
                mask_probs_batch[i], pred_scores_batch[i], pred_labels_batch[i], threshold, num_labels
            )

            # No mask found
            if mask_probs.shape[0] <= 0:
                height, width = target_sizes[i] if target_sizes is not None else mask_probs.shape[1:]
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
                target_size=target_sizes[i] if target_sizes is not None else None,
            )

            results.append({"segmentation": segmentation, "segments_info": segments})
        return results

    @filter_out_non_signature_kwargs()
    def post_process_instance_segmentation(
        self,
        outputs,
        target_sizes: list[tuple[int, int]],
        threshold: float = 0.5,
        size: Optional[dict[str, int]] = None,
    ):
        """Post-processes model outputs into Instance Segmentation Predictions."""

        size = size if size is not None else self.size

        class_queries_logits = outputs.class_queries_logits
        masks_queries_logits = outputs.masks_queries_logits

        output_size = get_target_size(size)
        masks_queries_logits = F.interpolate(
            masks_queries_logits,
            size=output_size,
            mode="bilinear",
        )

        mask_probs_batch = self.unpad_image(masks_queries_logits, target_sizes, size)

        device = masks_queries_logits.device
        batch_size = class_queries_logits.shape[0]
        num_queries = class_queries_logits.shape[-2]

        results = []

        for i in range(batch_size):
            mask_pred = mask_probs_batch[i]
            mask_class = class_queries_logits[i]

            # Remove the null class `[..., :-1]`
            scores, pred_classes = mask_class.softmax(dim=-1)[..., :-1].max(-1)
            pred_masks = (mask_pred > 0).float()

            # Calculate average mask prob
            mask_scores = (mask_pred.sigmoid().flatten(1) * pred_masks.flatten(1)).sum(1) / (
                pred_masks.flatten(1).sum(1) + 1e-6
            )
            pred_scores = scores * mask_scores

            segmentation = torch.zeros(target_sizes[i], device=device) - 1

            instance_maps, segments = [], []
            current_segment_id = 0
            for j in range(num_queries):
                score = pred_scores[j].item()

                if not torch.all(pred_masks[j] == 0) and score >= threshold:
                    segmentation[pred_masks[j] == 1] = current_segment_id
                    segments.append(
                        {
                            "id": current_segment_id,
                            "label_id": pred_classes[j].item(),
                            "score": round(score, 6),
                        }
                    )
                    current_segment_id += 1
                    instance_maps.append(pred_masks[j])

            results.append({"segmentation": segmentation, "segments_info": segments})
        return results


__all__ = ["EomtImageProcessor"]
