# coding=utf-8
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
"""Fast Image processor class for OneFormer."""

from typing import Optional, Union

from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    BatchFeature,
    DefaultFastImageProcessorKwargs,
    get_max_height_width,
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    auto_docstring,
    is_torch_available,
    is_torchvision_available,
    is_torchvision_v2_available,
    logging,
)
from .image_processing_oneformer import load_metadata, prepare_metadata


logger = logging.get_logger(__name__)

if is_torch_available():
    import torch
    from torch import nn

if is_torchvision_available():
    if is_torchvision_v2_available():
        from torchvision.transforms.v2 import functional as F
    else:
        from torchvision.transforms import functional as F


def make_pixel_mask(image: "torch.Tensor", output_size: tuple[int, int]) -> "torch.Tensor":
    """
    Make a pixel mask for the image, where 1 indicates a valid pixel and 0 indicates padding.

    Args:
        image (`torch.Tensor`):
            Image to make the pixel mask for.
        output_size (`Tuple[int, int]`):
            Output size of the mask.
    """

    input_height, input_width = image.shape[-2], image.shape[-1]
    mask = torch.zeros(output_size, dtype=torch.int64)
    mask[:input_height, :input_width] = 1
    return mask


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
    pixels = mask.flatten()
    pixels = torch.concat([[0], pixels, [0]])
    runs = torch.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return list(runs)


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
    original_area = (mask_probs[k] >= mask_threshold).sum()
    mask_exists = mask_k_area > 0 and original_area > 0

    # Eliminate disconnected tiny segments
    if mask_exists:
        area_ratio = mask_k_area / original_area
        if not area_ratio.item() > overlap_mask_area_threshold:
            mask_exists = False

    return mask_exists, mask_k


def compute_segments(
    mask_probs,
    pred_scores,
    pred_labels,
    mask_threshold: float = 0.5,
    overlap_mask_area_threshold: float = 0.8,
    label_ids_to_fuse: Optional[set[int]] = None,
    target_size: Optional[tuple[int, int]] = None,
):
    height = mask_probs.shape[1] if target_size is None else target_size[0]
    width = mask_probs.shape[2] if target_size is None else target_size[1]

    segmentation = torch.zeros((height, width), dtype=torch.int32, device=mask_probs.device)
    segments: list[dict] = []

    if target_size is not None:
        mask_probs = F.resize(
            mask_probs.unsqueeze(0),
            size=target_size,
            interpolation=F.InterpolationMode.BILINEAR,
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


def convert_segmentation_map_to_binary_masks_fast(
    segmentation_map: "torch.Tensor",
    instance_id_to_semantic_id: Optional[dict[int, int]] = None,
    ignore_index: Optional[int] = None,
    do_reduce_labels: bool = False,
):
    if do_reduce_labels and ignore_index is None:
        raise ValueError("If `do_reduce_labels` is True, `ignore_index` must be provided.")

    if do_reduce_labels:
        segmentation_map = torch.where(segmentation_map == 0, ignore_index, segmentation_map - 1)

    all_labels = torch.unique(segmentation_map)

    if ignore_index is not None:
        all_labels = all_labels[all_labels != ignore_index]

    binary_masks = [(segmentation_map == i) for i in all_labels]

    if binary_masks:
        binary_masks = torch.stack(binary_masks, dim=0)
    else:
        binary_masks = torch.zeros((0, *segmentation_map.shape), device=segmentation_map.device)

    # Convert instance ids to class ids
    if instance_id_to_semantic_id is not None:
        labels = torch.zeros(all_labels.shape[0], device=segmentation_map.device)

        for i, label in enumerate(all_labels):
            class_id = instance_id_to_semantic_id[(label.item() + 1 if do_reduce_labels else label.item())]
            labels[i] = class_id - 1 if do_reduce_labels else class_id
    else:
        labels = all_labels

    return (
        binary_masks.float(),
        labels.long(),
    )


def get_oneformer_resize_output_image_size(
    image: "torch.Tensor",
    size: Union[int, tuple[int, int], list[int], tuple[int]],
    max_size: Optional[int] = None,
    default_to_square: bool = True,
) -> tuple:
    """
    Computes the output size given the desired size.

    Args:
        image (`torch.Tensor`):
            The input image.
        size (`int` or `Tuple[int, int]` or `List[int]` or `Tuple[int]`):
            The size of the output image.
        max_size (`int`, *optional*):
            The maximum size of the output image.
        default_to_square (`bool`, *optional*, defaults to `True`):
            Whether to default to square if no size is provided.
    Returns:
        `Tuple[int, int]`: The output size.
    """
    if isinstance(size, (tuple, list)):
        if len(size) == 2:
            return tuple(size)
        elif len(size) == 1:
            # Perform same logic as if size was an int
            size = size[0]
        else:
            raise ValueError("size must have 1 or 2 elements if it is a list or tuple")

    if default_to_square:
        return (size, size)

    height, width = image.shape[-2], image.shape[-1]
    short, long = (width, height) if width <= height else (height, width)
    requested_new_short = size

    new_short, new_long = requested_new_short, int(requested_new_short * long / short)

    if max_size is not None:
        if max_size <= requested_new_short:
            raise ValueError(
                f"max_size = {max_size} must be strictly greater than the requested "
                f"size for the smaller edge size = {size}"
            )
        if new_long > max_size:
            new_short, new_long = int(max_size * new_short / new_long), max_size

    return (new_long, new_short) if width <= height else (new_short, new_long)


class OneFormerFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    r"""
    repo_path (`str`, *optional*, defaults to `shi-labs/oneformer_demo`):
        Path to a local directory or Hugging Face Hub repository containing model metadata.
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

    repo_path: Optional[str]
    class_info_file: Optional[str]
    num_text: Optional[int]
    num_labels: Optional[int]
    ignore_index: Optional[int]
    do_reduce_labels: Optional[bool]


@auto_docstring
class OneFormerImageProcessorFast(BaseImageProcessorFast):
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
    valid_kwargs = OneFormerFastImageProcessorKwargs
    model_input_names = ["pixel_values", "pixel_mask", "task_inputs"]

    def __init__(self, **kwargs: Unpack[OneFormerFastImageProcessorKwargs]):
        super().__init__(**kwargs)
        if self.class_info_file:
            self.metadata = prepare_metadata(load_metadata(self.repo_path, self.class_info_file))

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        task_inputs: Optional[list[str]] = None,
        segmentation_maps: Optional[ImageInput] = None,
        instance_id_to_semantic_id: Optional[Union[list[dict[int, int]], dict[int, int]]] = None,
        **kwargs: Unpack[OneFormerFastImageProcessorKwargs],
    ) -> BatchFeature:
        r"""
        task_inputs (`list[str]`, *optional*):
            List of tasks (`"panoptic"`, `"instance"`, `"semantic"`) for each image in the batch.
        segmentation_maps (`ImageInput`, *optional*):
            The segmentation maps.
        instance_id_to_semantic_id (`Union[list[dict[int, int]], dict[int, int]]`, *optional*):
            A mapping from instance IDs to semantic IDs.
        """
        return super().preprocess(
            images,
            task_inputs,
            segmentation_maps,
            instance_id_to_semantic_id,
            **kwargs,
        )

    def _preprocess_image_like_inputs(
        self,
        images: ImageInput,
        task_inputs: Optional[list[str]],
        segmentation_maps: ImageInput,
        instance_id_to_semantic_id: Optional[Union[list[dict[int, int]], dict[int, int]]],
        do_convert_rgb: bool,
        input_data_format: ChannelDimension,
        device: Optional[Union[str, "torch.device"]] = None,
        **kwargs: Unpack[OneFormerFastImageProcessorKwargs],
    ) -> BatchFeature:
        """
        Preprocess image-like inputs.
        To be overriden by subclasses when image-like inputs other than images should be processed.
        It can be used for segmentation maps, depth maps, etc.
        """
        # Prepare input images
        images = self._prepare_image_like_inputs(
            images=images, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format, device=device
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
        images: list["torch.Tensor"],
        task_inputs: Optional[list[str]],
        segmentation_maps: list["torch.Tensor"],
        instance_id_to_semantic_id: Optional[Union[list[dict[int, int]], dict[int, int]]],
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"],
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        ignore_index: Optional[int],
        do_reduce_labels: Optional[bool],
        disable_grouping: Optional[bool],
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs,
    ) -> BatchFeature:
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)

        processed_images_grouped = {}

        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(image=stacked_images, size=size, interpolation=interpolation)
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images
        processed_images = reorder_images(processed_images_grouped, grouped_images_index)

        processed_segmentation_maps = None
        if segmentation_maps is not None:
            grouped_segmentation_maps, grouped_segmentation_maps_index = group_images_by_shape(
                segmentation_maps, disable_grouping=disable_grouping
            )
            processed_segmentation_maps_grouped = {}
            for shape, stacked_segmentation_maps in grouped_segmentation_maps.items():
                if do_resize:
                    stacked_segmentation_maps = self.resize(
                        stacked_segmentation_maps, size=size, interpolation=F.InterpolationMode.NEAREST_EXACT
                    )
                processed_segmentation_maps_grouped[shape] = stacked_segmentation_maps
            processed_segmentation_maps = reorder_images(
                processed_segmentation_maps_grouped, grouped_segmentation_maps_index
            )

        encoded_inputs = self._encode_inputs_fast(
            processed_images,
            task_inputs,
            segmentation_maps=processed_segmentation_maps,
            instance_id_to_semantic_id=instance_id_to_semantic_id,
            ignore_index=ignore_index,
            do_reduce_labels=do_reduce_labels,
            return_tensors=return_tensors,
        )

        return encoded_inputs

    def _pad_image_fast(
        self,
        image: "torch.Tensor",
        output_size: tuple[int, int],
        constant_values: float = 0,
    ) -> "torch.Tensor":
        """
        Pad an image with zeros to the given size using torch operations.

        Args:
            image (`torch.Tensor`):
                Image tensor in channel-first format (C, H, W).
            output_size (`tuple[int, int]`):
                Target output size (height, width).
            constant_values (`float`, *optional*, defaults to 0):
                The value to use for padding.

        Returns:
            `torch.Tensor`: The padded image.
        """
        input_height, input_width = image.shape[1], image.shape[2]
        output_height, output_width = output_size

        pad_bottom = output_height - input_height
        pad_right = output_width - input_width

        padded_image = F.pad(image, padding=[0, 0, pad_right, pad_bottom], fill=constant_values)

        return padded_image

    def pad(
        self,
        images: list["torch.Tensor"],
        return_pixel_mask: bool = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> BatchFeature:
        """
        Pad a batch of images to the same size using torch operations.

        Args:
            images (`List[torch.Tensor]`):
                List of image tensors in channel-first format.
            return_pixel_mask (`bool`, *optional*, defaults to `True`):
                Whether to return pixel masks.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return.

        Returns:
            `BatchFeature`: Padded images and optional pixel masks.
        """
        pad_size = get_max_height_width(images)

        padded_images = []
        pixel_masks = []

        for image in images:
            padded_image = self._pad_image_fast(
                image=image,
                output_size=pad_size,
                constant_values=0,
            )
            padded_images.append(padded_image)

            if return_pixel_mask:
                input_height, input_width = image.shape[1], image.shape[2]
                mask = torch.zeros(pad_size, dtype=torch.int64, device=image.device)
                mask[:input_height, :input_width] = 1
                pixel_masks.append(mask)

        if return_tensors:
            padded_images = torch.stack(padded_images, dim=0)
            if return_pixel_mask:
                pixel_masks = torch.stack(pixel_masks, dim=0)

        data = {"pixel_values": padded_images}
        if return_pixel_mask:
            data["pixel_mask"] = pixel_masks

        return BatchFeature(data=data, tensor_type=return_tensors)

    def convert_segmentation_map_to_binary_masks(
        self,
        segmentation_map: "torch.Tensor",
        instance_id_to_semantic_id: Optional[dict[int, int]] = None,
        ignore_index: Optional[int] = None,
        do_reduce_labels: bool = False,
    ):
        return convert_segmentation_map_to_binary_masks_fast(
            segmentation_map=segmentation_map,
            instance_id_to_semantic_id=instance_id_to_semantic_id,
            ignore_index=ignore_index,
            do_reduce_labels=do_reduce_labels,
        )

    def get_semantic_annotations(self, label, num_class_obj):
        annotation_classes = label["classes"]
        annotation_masks = label["masks"]

        texts = ["a semantic photo"] * self.num_text
        classes = []
        masks = []

        for idx in range(len(annotation_classes)):
            class_id = annotation_classes[idx]
            mask = annotation_masks[idx]
            if not torch.all(mask == 0):
                if class_id not in classes:
                    cls_name = self.metadata[str(class_id.cpu().item())]
                    classes.append(class_id)
                    masks.append(mask)
                    num_class_obj[cls_name] += 1
                else:
                    idx = classes.index(class_id)
                    masks[idx] += mask
                    masks[idx] = torch.clamp(masks[idx], 0, 1)

        num = 0
        for i, cls_name in enumerate(self.metadata["class_names"]):
            if num_class_obj[cls_name] > 0:
                for _ in range(num_class_obj[cls_name]):
                    if num >= len(texts):
                        break
                    texts[num] = f"a photo with a {cls_name}"
                    num += 1

        classes = torch.stack(classes)
        masks = torch.stack(masks)
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
                if not torch.all(mask == 0):
                    cls_name = self.metadata[str(class_id.cpu().item())]
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

        classes = torch.stack(classes)
        masks = torch.stack(masks)
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
            if not torch.all(mask == 0):
                cls_name = self.metadata[str(class_id.cpu().item())]
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

        classes = torch.stack(classes)
        masks = torch.stack(masks)
        return classes, masks, texts

    def _encode_inputs_fast(
        self,
        pixel_values_list: list["torch.Tensor"],
        task_inputs: Optional[list[str]] = None,
        segmentation_maps: Optional[list["torch.Tensor"]] = None,
        instance_id_to_semantic_id: Optional[Union[list[dict[int, int]], dict[int, int]]] = None,
        ignore_index: Optional[int] = None,
        do_reduce_labels: bool = False,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> BatchFeature:
        if task_inputs is None:
            task_inputs = ["panoptic"]

        pad_size = get_max_height_width(pixel_values_list)
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

                # Convert segmentation map to binary masks using torch operations
                masks, classes = self.convert_segmentation_map_to_binary_masks(
                    segmentation_map,
                    instance_id,
                    ignore_index=ignore_index,
                    do_reduce_labels=do_reduce_labels,
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
                # Pad masks to max size using torch operations
                padded_masks = [
                    self._pad_image_fast(image=mask, output_size=pad_size, constant_values=ignore_index)
                    for mask in masks
                ]
                padded_masks = torch.cat(padded_masks, dim=0)
                mask_labels.append(padded_masks)
                class_labels.append(classes)
                text_inputs.append(texts)

            encoded_inputs["mask_labels"] = mask_labels
            encoded_inputs["class_labels"] = class_labels
            encoded_inputs["text_inputs"] = text_inputs

        encoded_inputs["task_inputs"] = [f"the task is {task_input}" for task_input in task_inputs]
        return encoded_inputs

    def post_process_semantic_segmentation(
        self, outputs, target_sizes: Optional[list[tuple[int, int]]] = None
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
                resized_logits = F.resize(
                    segmentation[idx].unsqueeze(dim=0),
                    size=target_sizes[idx],
                    interpolation=F.InterpolationMode.BILINEAR,
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
        target_sizes: Optional[list[tuple[int, int]]] = None,
        return_coco_annotation: Optional[bool] = False,
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

    # Copied from transformers.models.maskformer.image_processing_maskformer.MaskFormerImageProcessor.post_process_panoptic_segmentation
    def post_process_panoptic_segmentation(
        self,
        outputs,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
        label_ids_to_fuse: Optional[set[int]] = None,
        target_sizes: Optional[list[tuple[int, int]]] = None,
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


__all__ = ["OneFormerImageProcessorFast"]
