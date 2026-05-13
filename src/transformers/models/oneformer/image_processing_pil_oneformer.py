# Copyright 2025 SHI Labs and The HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for OneFormer."""

import json
import os

import numpy as np

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import PaddingMode
from ...image_transforms import pad as np_pad
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
    get_image_size,
    get_max_height_width,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring, is_torch_available, is_torchvision_available, logging
from ...utils.import_utils import requires


logger = logging.get_logger(__name__)

if is_torch_available():
    import torch
    from torch import nn

if is_torchvision_available():
    import torchvision.transforms.v2.functional as tvF

try:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import RepositoryNotFoundError
except ImportError:
    hf_hub_download = None
    RepositoryNotFoundError = None


def make_pixel_mask(image: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    """
    Make a pixel mask for the image, where 1 indicates a valid pixel and 0 indicates padding.

    Args:
        image (`np.ndarray`):
            Image to make the pixel mask for.
        output_size (`Tuple[int, int]`):
            Output size of the mask.
    """
    input_height, input_width = get_image_size(image, channel_dim=ChannelDimension.FIRST)
    mask = np.zeros(output_size, dtype=np.int64)
    mask[:input_height, :input_width] = 1
    return mask


# Adapted from transformers.models.oneformer.image_processing_oneformer.OneFormerImageProcessorKwargs
class OneFormerImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    repo_path (`str`, *optional*, defaults to `shi-labs/oneformer_demo`):
        Path to a local directory or HuggingFace Hub repository containing model metadata.
    class_info_file (`str`, *optional*):
        Path to the JSON file within the repository that contains class metadata.
    num_text (`int`, *optional*):
        Number of text queries for the text encoder, used as task-guiding prompts.
    num_labels (`int`, *optional*):
        Number of semantic classes for segmentation, determining the output layer's size.
    ignore_index (`int`, *optional*):
        Label to ignore in segmentation maps, often used for padding.
    do_reduce_labels (`bool`, *optional*, defaults to `False`):
        Whether to decrement all label values by 1, mapping the background class to `ignore_index`.
    """

    repo_path: str | None
    class_info_file: str | None
    num_text: int | None
    num_labels: int | None
    ignore_index: int | None
    do_reduce_labels: bool


# Adapted from transformers.models.oneformer.image_processing_oneformer.binary_mask_to_rle
def binary_mask_to_rle(mask):
    """
    Converts given binary mask of shape `(height, width)` to the run-length encoding (RLE) format.

    Args:
        mask (`torch.Tensor` or `numpy.array`):
            A binary mask tensor of shape `(height, width)` where 0 denotes background and 1 denotes the target
            segment_id or class_id.
    Returns:
        `List`: Run-length encoded list of the binary mask. Refer to COCO API for more information about the RLE
        format.
    """
    from ...utils import is_torch_tensor

    if is_torch_tensor(mask):
        mask = mask.numpy()

    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return list(runs)


# Adapted from transformers.models.oneformer.image_processing_oneformer.check_segment_validity
def check_segment_validity(mask_labels, mask_probs, k, mask_threshold=0.5, overlap_mask_area_threshold=0.8):
    # Get the mask associated with the k class
    mask_k = mask_labels == k
    mask_k_area = mask_k.sum()

    # Compute the area of all the stuff in query k
    original_area = (mask_probs[k] >= mask_threshold).sum()
    mask_exists = mask_k_area > 0 and original_area > 0

    # Eliminate disconnected tiny segments
    if mask_exists:
        area_ratio = mask_k_area / original_area
        if not area_ratio.item() > overlap_mask_area_threshold:
            mask_exists = False

    return mask_exists, mask_k


# Adapted from transformers.models.oneformer.image_processing_oneformer.compute_segments
def compute_segments(
    mask_probs,
    pred_scores,
    pred_labels,
    mask_threshold: float = 0.5,
    overlap_mask_area_threshold: float = 0.8,
    label_ids_to_fuse: set[int] | None = None,
    target_size: tuple[int, int] | None = None,
):
    height = mask_probs.shape[1] if target_size is None else target_size[0]
    width = mask_probs.shape[2] if target_size is None else target_size[1]

    segmentation = torch.zeros((height, width), dtype=torch.int32, device=mask_probs.device)
    segments: list[dict] = []

    if target_size is not None:
        mask_probs = tvF.resize(
            mask_probs.unsqueeze(0),
            size=target_size,
            interpolation=tvF.InterpolationMode.BILINEAR,
        )[0]

    current_segment_id = 0

    mask_probs *= pred_scores.view(-1, 1, 1)
    mask_labels = mask_probs.argmax(0)  # [height, width]

    stuff_memory_list: dict[str, int] = {}
    for k in range(pred_labels.shape[0]):
        pred_class = pred_labels[k].item()
        should_fuse = pred_class in label_ids_to_fuse

        mask_exists, mask_k = check_segment_validity(
            mask_labels, mask_probs, k, mask_threshold, overlap_mask_area_threshold
        )

        if mask_exists:
            if pred_class in stuff_memory_list:
                current_segment_id = stuff_memory_list[pred_class]
            else:
                current_segment_id += 1

            segmentation[mask_k] = current_segment_id
            segment_score = round(pred_scores[k].item(), 6)
            segments.append(
                {
                    "id": current_segment_id,
                    "label_id": pred_class,
                    "was_fused": should_fuse,
                    "score": segment_score,
                }
            )
            if should_fuse:
                stuff_memory_list[pred_class] = current_segment_id

    return segmentation, segments


# Adapted from transformers.models.oneformer.image_processing_oneformer.convert_segmentation_to_rle
def convert_segmentation_to_rle(segmentation):
    """
    Converts given segmentation map of shape `(height, width)` to the run-length encoding (RLE) format.

    Args:
        segmentation (`torch.Tensor` or `numpy.array`):
            A segmentation map of shape `(height, width)` where each value denotes a segment or class id.
    Returns:
        `List[List]`: A list of lists, where each list is the run-length encoding of a segment / class id.
    """
    segment_ids = torch.unique(segmentation)

    run_length_encodings = []
    for idx in segment_ids:
        mask = torch.where(segmentation == idx, 1, 0)
        rle = binary_mask_to_rle(mask)
        run_length_encodings.append(rle)

    return run_length_encodings


# Adapted from transformers.models.oneformer.image_processing_oneformer.load_metadata
def load_metadata(repo_id, class_info_file):
    fname = os.path.join("" if repo_id is None else repo_id, class_info_file)

    if not os.path.exists(fname) or not os.path.isfile(fname):
        if repo_id is None:
            raise ValueError(f"Could not file {fname} locally. repo_id must be defined if loading from the hub")
        if hf_hub_download is None:
            raise ImportError(
                "huggingface_hub is required to download metadata files. Install it with `pip install huggingface_hub`"
            )
        # We try downloading from a dataset by default for backward compatibility
        try:
            fname = hf_hub_download(repo_id, class_info_file, repo_type="dataset")
        except RepositoryNotFoundError:
            fname = hf_hub_download(repo_id, class_info_file)

    with open(fname, "r") as f:
        class_info = json.load(f)

    return class_info


# Adapted from transformers.models.oneformer.image_processing_oneformer.prepare_metadata
def prepare_metadata(class_info):
    metadata = {}
    class_names = []
    thing_ids = []
    for key, info in class_info.items():
        metadata[key] = info["name"]
        class_names.append(info["name"])
        if info["isthing"]:
            thing_ids.append(int(key))
    metadata["thing_ids"] = thing_ids
    metadata["class_names"] = class_names
    return metadata


# Adapted from transformers.models.oneformer.image_processing_oneformer.remove_low_and_no_objects
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


@auto_docstring
@requires(backends=("torch",))
class OneFormerImageProcessorPil(PilBackend):
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"shortest_edge": 800, "longest_edge": 1333}
    crop_size = None
    do_resize = True
    do_rescale = True
    do_normalize = True
    default_to_square = False
    do_center_crop = False
    do_convert_rgb = True
    rescale_factor = 1 / 255
    ignore_index = None
    do_reduce_labels = False
    repo_path = "shi-labs/oneformer_demo"
    class_info_file = None
    num_text = None
    num_labels = None
    valid_kwargs = OneFormerImageProcessorKwargs
    model_input_names = ["pixel_values", "pixel_mask", "task_inputs"]

    def __init__(self, **kwargs: Unpack[OneFormerImageProcessorKwargs]):
        super().__init__(**kwargs)
        if self.class_info_file:
            self.metadata = prepare_metadata(load_metadata(self.repo_path, self.class_info_file))

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        task_inputs: list[str] | None = None,
        segmentation_maps: ImageInput | None = None,
        instance_id_to_semantic_id: list[dict[int, int]] | dict[int, int] | None = None,
        **kwargs: Unpack[OneFormerImageProcessorKwargs],
    ) -> BatchFeature:
        r"""
        task_inputs (`list[str]`, *optional*):
            List of tasks (`"panoptic"`, `"instance"`, `"semantic"`) for each image in the batch.
        segmentation_maps (`ImageInput`, *optional*):
            The segmentation maps.
        instance_id_to_semantic_id (`Union[list[dict[int, int]], dict[int, int]]`, *optional*):
            A mapping from instance IDs to semantic IDs.
        """
        return super().preprocess(images, task_inputs, segmentation_maps, instance_id_to_semantic_id, **kwargs)

    def _preprocess_image_like_inputs(
        self,
        images: ImageInput,
        task_inputs: list[str] | None,
        segmentation_maps: ImageInput,
        instance_id_to_semantic_id: list[dict[int, int]] | dict[int, int] | None,
        do_convert_rgb: bool,
        input_data_format: ChannelDimension,
        **kwargs: Unpack[OneFormerImageProcessorKwargs],
    ) -> BatchFeature:
        """
        Preprocess image-like inputs.
        To be overridden by subclasses when image-like inputs other than images should be processed.
        It can be used for segmentation maps, depth maps, etc.
        """
        # Prepare input images
        images = self._prepare_image_like_inputs(
            images=images, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format
        )
        if segmentation_maps is not None:
            segmentation_maps = self._prepare_image_like_inputs(
                images=segmentation_maps,
                expected_ndims=2,
                do_convert_rgb=False,
                input_data_format=ChannelDimension.FIRST,
            )
        return self._preprocess(images, task_inputs, segmentation_maps, instance_id_to_semantic_id, **kwargs)

    def _preprocess(
        self,
        images: list[np.ndarray],
        task_inputs: list[str] | None,
        segmentation_maps: list[np.ndarray],
        instance_id_to_semantic_id: list[dict[int, int]] | dict[int, int] | None,
        do_resize: bool,
        size: SizeDict,
        resample: PILImageResampling | None,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        ignore_index: int | None,
        do_reduce_labels: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        # Process images one by one (no batching in PIL backend)
        processed_images = []
        processed_segmentation_maps = None
        if segmentation_maps is not None:
            processed_segmentation_maps = []

        for idx, image in enumerate(images):
            if do_resize:
                image = self.resize(image=image, size=size, resample=resample)
            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)
            processed_images.append(image)

            if segmentation_maps is not None:
                seg_map = segmentation_maps[idx]
                if do_resize:
                    seg_map = self.resize(image=seg_map, size=size, resample=PILImageResampling.NEAREST)
                processed_segmentation_maps.append(seg_map)

        encoded_inputs = self.encode_inputs(
            processed_images,
            task_inputs,
            segmentation_maps=processed_segmentation_maps,
            instance_id_to_semantic_id=instance_id_to_semantic_id,
            ignore_index=ignore_index,
            do_reduce_labels=do_reduce_labels,
            return_tensors=return_tensors,
        )

        return encoded_inputs

    def _pad_image(self, image: np.ndarray, output_size: tuple[int, int], constant_values: float = 0) -> np.ndarray:
        """
        Pad an image with zeros to the given size using numpy operations.

        Args:
            image (`np.ndarray`):
                Image array in channel-first format (C, H, W) or (H, W).
            output_size (`tuple[int, int]`):
                Target output size (height, width).
            constant_values (`float`, *optional*, defaults to 0):
                The value to use for padding.

        Returns:
            `np.ndarray`: The padded image.
        """
        input_height, input_width = get_image_size(image, channel_dim=ChannelDimension.FIRST)
        output_height, output_width = output_size

        pad_bottom = output_height - input_height
        pad_right = output_width - input_width

        # For 2D arrays (masks), use np.pad directly
        # For 3D arrays (images), use np_pad which handles channel dimension
        if image.ndim == 2:
            padding = ((0, pad_bottom), (0, pad_right))
            padded_image = np.pad(image, padding, mode="constant", constant_values=constant_values)
        else:
            padding = ((0, pad_bottom), (0, pad_right))
            padded_image = np_pad(
                image,
                padding,
                mode=PaddingMode.CONSTANT,
                constant_values=constant_values,
                data_format=ChannelDimension.FIRST,
                input_data_format=ChannelDimension.FIRST,
            )

        return padded_image

    def pad(
        self, images: list[np.ndarray], return_pixel_mask: bool = True, return_tensors: str | TensorType | None = None
    ) -> BatchFeature:
        """
        Pad a batch of images to the same size using numpy operations.

        Args:
            images (`List[np.ndarray]`):
                List of image arrays in channel-first format.
            return_pixel_mask (`bool`, *optional*, defaults to `True`):
                Whether to return pixel masks.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return.

        Returns:
            `BatchFeature`: Padded images and optional pixel masks.
        """
        pad_size = get_max_height_width(images, input_data_format=ChannelDimension.FIRST)

        padded_images = []
        pixel_masks = []
        for image in images:
            padded_image = self._pad_image(image, pad_size, constant_values=0)
            padded_images.append(padded_image)
            if return_pixel_mask:
                pixel_mask = make_pixel_mask(image, pad_size)
                pixel_masks.append(pixel_mask)

        if return_tensors == "pt":
            padded_images = [torch.from_numpy(img) for img in padded_images]
            padded_images = torch.stack(padded_images, dim=0)
            if return_pixel_mask:
                pixel_masks = [torch.from_numpy(mask) for mask in pixel_masks]
                pixel_masks = torch.stack(pixel_masks, dim=0)

        data = {"pixel_values": padded_images}
        if return_pixel_mask:
            data["pixel_mask"] = pixel_masks

        return BatchFeature(data=data, tensor_type=return_tensors)

    def convert_segmentation_map_to_binary_masks(
        self,
        segmentation_map: np.ndarray,
        instance_id_to_semantic_id: dict[int, int] | None = None,
        ignore_index: int | None = None,
        do_reduce_labels: bool = False,
    ):
        """Convert segmentation map to binary masks using NumPy operations."""
        if do_reduce_labels and ignore_index is None:
            raise ValueError("If `do_reduce_labels` is True, `ignore_index` must be provided.")

        if do_reduce_labels:
            segmentation_map = np.where(segmentation_map == 0, ignore_index, segmentation_map - 1)

        all_labels = np.unique(segmentation_map)

        if ignore_index is not None:
            all_labels = all_labels[all_labels != ignore_index]

        binary_masks = [(segmentation_map == i) for i in all_labels]
        if binary_masks:
            binary_masks = np.stack(binary_masks, axis=0)
        else:
            binary_masks = np.zeros((0, *segmentation_map.shape), dtype=np.float32)

        # Convert instance ids to class ids
        if instance_id_to_semantic_id is not None:
            labels = np.zeros(all_labels.shape[0], dtype=np.int64)

            for i, label in enumerate(all_labels):
                class_id = instance_id_to_semantic_id[(int(label) + 1 if do_reduce_labels else int(label))]
                labels[i] = class_id - 1 if do_reduce_labels else class_id
        else:
            labels = all_labels.astype(np.int64)
        return binary_masks.astype(np.float32), labels

    def get_semantic_annotations(self, label, num_class_obj):
        annotation_classes = label["classes"]
        annotation_masks = label["masks"]

        texts = ["a semantic photo"] * self.num_text
        classes = []
        masks = []

        for idx in range(len(annotation_classes)):
            class_id = annotation_classes[idx]
            mask = annotation_masks[idx]
            if not np.all(mask == 0):
                if class_id not in classes:
                    cls_name = self.metadata[str(class_id)]
                    classes.append(class_id)
                    masks.append(mask)
                    num_class_obj[cls_name] += 1
                else:
                    idx = classes.index(class_id)
                    masks[idx] += mask
                    masks[idx] = np.clip(masks[idx], 0, 1)

        num = 0
        for i, cls_name in enumerate(self.metadata["class_names"]):
            if num_class_obj[cls_name] > 0:
                for _ in range(num_class_obj[cls_name]):
                    if num >= len(texts):
                        break
                    texts[num] = f"a photo with a {cls_name}"
                    num += 1

        classes = np.array(classes) if classes else np.array([], dtype=np.int64)
        # Stack masks into a 3D array (num_masks, H, W) to match torchvision version
        if masks:
            masks = np.stack(masks, axis=0)
        else:
            # Empty masks - use shape from first annotation mask if available
            if annotation_masks and len(annotation_masks) > 0:
                mask_shape = annotation_masks[0].shape[-2:] if hasattr(annotation_masks[0], "shape") else (0, 0)
            else:
                mask_shape = (0, 0)
            masks = np.zeros((0, *mask_shape), dtype=np.float32)
        return classes, masks, texts

    def get_instance_annotations(self, label, num_class_obj):
        annotation_classes = label["classes"]
        annotation_masks = label["masks"]

        texts = ["an instance photo"] * self.num_text
        classes = []
        masks = []

        for idx in range(len(annotation_classes)):
            class_id = annotation_classes[idx]
            mask = annotation_masks[idx]

            if class_id in self.metadata["thing_ids"]:
                if not np.all(mask == 0):
                    cls_name = self.metadata[str(class_id)]
                    classes.append(class_id)
                    masks.append(mask)
                    num_class_obj[cls_name] += 1

        num = 0
        for i, cls_name in enumerate(self.metadata["class_names"]):
            if num_class_obj[cls_name] > 0:
                for _ in range(num_class_obj[cls_name]):
                    if num >= len(texts):
                        break
                    texts[num] = f"a photo with a {cls_name}"
                    num += 1

        classes = np.array(classes) if classes else np.array([], dtype=np.int64)
        # Stack masks into a 3D array (num_masks, H, W) to match torchvision version
        if masks:
            masks = np.stack(masks, axis=0)
        else:
            # Empty masks - use shape from first annotation mask if available
            if annotation_masks and len(annotation_masks) > 0:
                mask_shape = annotation_masks[0].shape[-2:] if hasattr(annotation_masks[0], "shape") else (0, 0)
            else:
                mask_shape = (0, 0)
            masks = np.zeros((0, *mask_shape), dtype=np.float32)
        return classes, masks, texts

    def get_panoptic_annotations(self, label, num_class_obj):
        annotation_classes = label["classes"]
        annotation_masks = label["masks"]

        texts = ["an panoptic photo"] * self.num_text
        classes = []
        masks = []
        for idx in range(len(annotation_classes)):
            class_id = annotation_classes[idx]
            mask = annotation_masks[idx] if hasattr(annotation_masks[idx], "data") else annotation_masks[idx]
            if not np.all(mask == 0):
                cls_name = self.metadata[str(class_id)]
                classes.append(class_id)
                masks.append(mask)
                num_class_obj[cls_name] += 1

        num = 0
        for i, cls_name in enumerate(self.metadata["class_names"]):
            if num_class_obj[cls_name] > 0:
                for _ in range(num_class_obj[cls_name]):
                    if num >= len(texts):
                        break
                    texts[num] = f"a photo with a {cls_name}"
                    num += 1

        classes = np.array(classes) if classes else np.array([], dtype=np.int64)
        # Stack masks into a 3D array (num_masks, H, W) to match torchvision version
        if masks:
            masks = np.stack(masks, axis=0)
        else:
            # Empty masks - use shape from first annotation mask if available
            if annotation_masks and len(annotation_masks) > 0:
                mask_shape = annotation_masks[0].shape[-2:] if hasattr(annotation_masks[0], "shape") else (0, 0)
            else:
                mask_shape = (0, 0)
            masks = np.zeros((0, *mask_shape), dtype=np.float32)
        return classes, masks, texts

    def encode_inputs(
        self,
        pixel_values_list: list[np.ndarray],
        task_inputs: list[str] | None = None,
        segmentation_maps: list[np.ndarray] | None = None,
        instance_id_to_semantic_id: list[dict[int, int]] | dict[int, int] | None = None,
        ignore_index: int | None = None,
        do_reduce_labels: bool = False,
        return_tensors: str | TensorType | None = None,
    ) -> BatchFeature:
        ignore_index = self.ignore_index if ignore_index is None else ignore_index
        do_reduce_labels = self.do_reduce_labels if do_reduce_labels is None else do_reduce_labels
        if task_inputs is None:
            task_inputs = ["panoptic"]
        pixel_values_list = self._prepare_image_like_inputs(
            pixel_values_list, input_data_format=ChannelDimension.FIRST
        )
        if segmentation_maps is not None:
            segmentation_maps = self._prepare_image_like_inputs(
                images=segmentation_maps,
                expected_ndims=2,
                do_convert_rgb=False,
                input_data_format=ChannelDimension.FIRST,
            )
        pad_size = get_max_height_width(pixel_values_list, input_data_format=ChannelDimension.FIRST)
        encoded_inputs = self.pad(pixel_values_list, return_tensors=return_tensors)

        annotations = None
        if segmentation_maps is not None:
            annotations = []
            for idx, segmentation_map in enumerate(segmentation_maps):
                # Use instance2class_id mapping per image
                if isinstance(instance_id_to_semantic_id, list):
                    instance_id = instance_id_to_semantic_id[idx]
                else:
                    instance_id = instance_id_to_semantic_id

                # Squeeze channel dimension if present
                if segmentation_map.ndim == 3 and segmentation_map.shape[0] == 1:
                    segmentation_map = segmentation_map.squeeze(0)

                # Convert segmentation map to binary masks using numpy operations
                masks, classes = self.convert_segmentation_map_to_binary_masks(
                    segmentation_map, instance_id, ignore_index=ignore_index, do_reduce_labels=do_reduce_labels
                )

                annotations.append({"masks": masks, "classes": classes})

        if annotations is not None:
            mask_labels = []
            class_labels = []
            text_inputs = []
            num_class_obj = dict.fromkeys(self.metadata["class_names"], 0)

            for i, label in enumerate(annotations):
                task = task_inputs[i]

                if task == "semantic":
                    classes, masks, texts = self.get_semantic_annotations(label, num_class_obj)
                elif task == "instance":
                    classes, masks, texts = self.get_instance_annotations(label, num_class_obj)
                elif task == "panoptic":
                    classes, masks, texts = self.get_panoptic_annotations(label, num_class_obj)
                else:
                    raise ValueError(f"{task} was not expected, expected `semantic`, `instance` or `panoptic`")
                # Pad masks to max size using numpy operations
                # masks is a 3D array (num_masks, H, W), iterate to get 2D slices
                padded_masks = [
                    self._pad_image(image=mask, output_size=pad_size, constant_values=ignore_index) for mask in masks
                ]
                # Stack padded masks back into 3D array (num_masks, padded_H, padded_W)
                padded_masks = (
                    np.stack(padded_masks, axis=0) if padded_masks else np.zeros((0, *pad_size), dtype=np.float32)
                )
                mask_labels.append(padded_masks)
                class_labels.append(classes)
                text_inputs.append(texts)

            encoded_inputs["mask_labels"] = [
                torch.from_numpy(mask_label) if return_tensors == "pt" else mask_label for mask_label in mask_labels
            ]
            encoded_inputs["class_labels"] = [
                torch.from_numpy(class_label) if return_tensors == "pt" else class_label
                for class_label in class_labels
            ]
            encoded_inputs["text_inputs"] = text_inputs

        encoded_inputs["task_inputs"] = [f"the task is {task_input}" for task_input in task_inputs]
        return encoded_inputs

    def post_process_semantic_segmentation(
        self, outputs, target_sizes: list[tuple[int, int]] | None = None
    ) -> "torch.Tensor":
        """
        Converts the output of [`MaskFormerForInstanceSegmentation`] into semantic segmentation maps. Only supports
        PyTorch.

        Args:
            outputs ([`MaskFormerForInstanceSegmentation`]):
                Raw outputs of the model.
            target_sizes (`List[Tuple[int, int]]`, *optional*):
                List of length (batch_size), where each list item (`Tuple[int, int]]`) corresponds to the requested
                final size (height, width) of each prediction. If left to None, predictions will not be resized.
        Returns:
            `List[torch.Tensor]`:
                A list of length `batch_size`, where each item is a semantic segmentation map of shape (height, width)
                corresponding to the target_sizes entry (if `target_sizes` is specified). Each entry of each
                `torch.Tensor` correspond to a semantic class id.
        """
        class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
        masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]

        # Remove the null class `[..., :-1]`
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

        # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
        segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
        batch_size = class_queries_logits.shape[0]

        # Resize logits and compute semantic segmentation maps
        if target_sizes is not None:
            if batch_size != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

            semantic_segmentation = []
            for idx in range(batch_size):
                resized_logits = torch.nn.functional.interpolate(
                    segmentation[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
                )
                semantic_map = resized_logits[0].argmax(dim=0)
                semantic_segmentation.append(semantic_map)
        else:
            semantic_segmentation = segmentation.argmax(dim=1)
            semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]

        return semantic_segmentation

    def post_process_instance_segmentation(
        self,
        outputs,
        task_type: str = "instance",
        is_demo: bool = True,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
        target_sizes: list[tuple[int, int]] | None = None,
        return_coco_annotation: bool | None = False,
    ):
        """
        Converts the output of [`OneFormerForUniversalSegmentationOutput`] into image instance segmentation
        predictions. Only supports PyTorch.

        Args:
            outputs ([`OneFormerForUniversalSegmentationOutput`]):
                The outputs from [`OneFormerForUniversalSegmentationOutput`].
            task_type (`str`, *optional*, defaults to "instance"):
                The post processing depends on the task token input. If the `task_type` is "panoptic", we need to
                ignore the stuff predictions.
            is_demo (`bool`, *optional)*, defaults to `True`):
                Whether the model is in demo mode. If true, use threshold to predict final masks.
            threshold (`float`, *optional*, defaults to 0.5):
                The probability score threshold to keep predicted instance masks.
            mask_threshold (`float`, *optional*, defaults to 0.5):
                Threshold to use when turning the predicted masks into binary values.
            overlap_mask_area_threshold (`float`, *optional*, defaults to 0.8):
                The overlap mask area threshold to merge or discard small disconnected parts within each binary
                instance mask.
            target_sizes (`List[Tuple]`, *optional*):
                List of length (batch_size), where each list item (`Tuple[int, int]]`) corresponds to the requested
                final size (height, width) of each prediction in batch. If left to None, predictions will not be
                resized.
            return_coco_annotation (`bool`, *optional)*, defaults to `False`):
                Whether to return predictions in COCO format.

        Returns:
            `List[Dict]`: A list of dictionaries, one per image, each dictionary containing two keys:
            - **segmentation** -- a tensor of shape `(height, width)` where each pixel represents a `segment_id`, set
              to `None` if no mask if found above `threshold`. If `target_sizes` is specified, segmentation is resized
              to the corresponding `target_sizes` entry.
            - **segments_info** -- A dictionary that contains additional information on each segment.
                - **id** -- an integer representing the `segment_id`.
                - **label_id** -- An integer representing the label / semantic class id corresponding to `segment_id`.
                - **was_fused** -- a boolean, `True` if `label_id` was in `label_ids_to_fuse`, `False` otherwise.
                  Multiple instances of the same class / label were fused and assigned a single `segment_id`.
                - **score** -- Prediction score of segment with `segment_id`.
        """
        class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
        masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]

        device = masks_queries_logits.device
        batch_size = class_queries_logits.shape[0]
        num_queries = class_queries_logits.shape[1]
        num_classes = class_queries_logits.shape[-1] - 1

        # Loop over items in batch size
        results: list[dict[str, torch.Tensor]] = []

        for i in range(batch_size):
            # [Q, K]
            scores = nn.functional.softmax(class_queries_logits[i], dim=-1)[:, :-1]
            labels = torch.arange(num_classes, device=device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)

            # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(num_queries, sorted=False)
            labels_per_image = labels[topk_indices]

            topk_indices = torch.div(topk_indices, num_classes, rounding_mode="floor")
            # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
            mask_pred = masks_queries_logits[i][topk_indices]

            # Only consider scores with confidence over [threshold] for demo
            if is_demo:
                keep = scores_per_image > threshold
                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]
                mask_pred = mask_pred[keep]

            # if this is panoptic segmentation, we only keep the "thing" classes
            if task_type == "panoptic":
                keep = torch.zeros_like(scores_per_image).bool()
                for j, lab in enumerate(labels_per_image):
                    keep[j] = lab in self.metadata["thing_ids"]

                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]
                mask_pred = mask_pred[keep]

            if mask_pred.shape[0] <= 0:
                height, width = target_sizes[i] if target_sizes is not None else mask_pred.shape[1:]
                segmentation = torch.zeros((height, width)) - 1
                results.append({"segmentation": segmentation, "segments_info": []})
                continue

            if "ade20k" in self.class_info_file and not is_demo and "instance" in task_type:
                for j in range(labels_per_image.shape[0]):
                    labels_per_image[j] = self.metadata["thing_ids"].index(labels_per_image[j].item())

            # Get segmentation map and segment information of batch item
            target_size = target_sizes[i] if target_sizes is not None else None
            segmentation, segments = compute_segments(
                mask_pred,
                scores_per_image,
                labels_per_image,
                mask_threshold,
                overlap_mask_area_threshold,
                set(),
                target_size,
            )

            # Return segmentation map in run-length encoding (RLE) format
            if return_coco_annotation:
                segmentation = convert_segmentation_to_rle(segmentation)

            results.append({"segmentation": segmentation, "segments_info": segments})
        return results

    # Adapted from transformers.models.maskformer.image_processing_maskformer.MaskFormerImageProcessor.post_process_panoptic_segmentation
    def post_process_panoptic_segmentation(
        self,
        outputs,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
        label_ids_to_fuse: set[int] | None = None,
        target_sizes: list[tuple[int, int]] | None = None,
    ) -> list[dict]:
        """
        Converts the output of [`MaskFormerForInstanceSegmentationOutput`] into image panoptic segmentation
        predictions. Only supports PyTorch.

        Args:
            outputs ([`MaskFormerForInstanceSegmentationOutput`]):
                The outputs from [`MaskFormerForInstanceSegmentation`].
            threshold (`float`, *optional*, defaults to 0.5):
                The probability score threshold to keep predicted instance masks.
            mask_threshold (`float`, *optional*, defaults to 0.5):
                Threshold to use when turning the predicted masks into binary values.
            overlap_mask_area_threshold (`float`, *optional*, defaults to 0.8):
                The overlap mask area threshold to merge or discard small disconnected parts within each binary
                instance mask.
            label_ids_to_fuse (`Set[int]`, *optional*):
                The labels in this state will have all their instances be fused together. For instance we could say
                there can only be one sky in an image, but several persons, so the label ID for sky would be in that
                set, but not the one for person.
            target_sizes (`list[Tuple]`, *optional*):
                List of length (batch_size), where each list item (`tuple[int, int]]`) corresponds to the requested
                final size (height, width) of each prediction in batch. If left to None, predictions will not be
                resized.

        Returns:
            `list[Dict]`: A list of dictionaries, one per image, each dictionary containing two keys:
            - **segmentation** -- a tensor of shape `(height, width)` where each pixel represents a `segment_id`, set
              to `None` if no mask if found above `threshold`. If `target_sizes` is specified, segmentation is resized
              to the corresponding `target_sizes` entry.
            - **segments_info** -- A dictionary that contains additional information on each segment.
                - **id** -- an integer representing the `segment_id`.
                - **label_id** -- An integer representing the label / semantic class id corresponding to `segment_id`.
                - **was_fused** -- a boolean, `True` if `label_id` was in `label_ids_to_fuse`, `False` otherwise.
                  Multiple instances of the same class / label were fused and assigned a single `segment_id`.
                - **score** -- Prediction score of segment with `segment_id`.
        """

        if label_ids_to_fuse is None:
            logger.warning("`label_ids_to_fuse` unset. No instance will be fused.")
            label_ids_to_fuse = set()

        class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
        masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]

        batch_size = class_queries_logits.shape[0]
        num_labels = class_queries_logits.shape[-1] - 1

        mask_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

        # Predicted label and score of each query (batch_size, num_queries)
        pred_scores, pred_labels = nn.functional.softmax(class_queries_logits, dim=-1).max(-1)

        # Loop over items in batch size
        results: list[dict[str, TensorType]] = []

        for i in range(batch_size):
            mask_probs_item, pred_scores_item, pred_labels_item = remove_low_and_no_objects(
                mask_probs[i], pred_scores[i], pred_labels[i], threshold, num_labels
            )

            # No mask found
            if mask_probs_item.shape[0] <= 0:
                height, width = target_sizes[i] if target_sizes is not None else mask_probs_item.shape[1:]
                segmentation = torch.zeros((height, width)) - 1
                results.append({"segmentation": segmentation, "segments_info": []})
                continue

            # Get segmentation map and segment information of batch item
            target_size = target_sizes[i] if target_sizes is not None else None
            segmentation, segments = compute_segments(
                mask_probs=mask_probs_item,
                pred_scores=pred_scores_item,
                pred_labels=pred_labels_item,
                mask_threshold=mask_threshold,
                overlap_mask_area_threshold=overlap_mask_area_threshold,
                label_ids_to_fuse=label_ids_to_fuse,
                target_size=target_size,
            )

            results.append({"segmentation": segmentation, "segments_info": segments})
        return results


__all__ = ["OneFormerImageProcessorPil"]
