# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""Feature extractor class for MaskFormer."""

from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from PIL import Image

from ...feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from ...image_utils import ImageFeatureExtractionMixin, ImageInput, is_torch_tensor
from ...utils import TensorType, is_torch_available, logging


if is_torch_available():
    import torch
    from torch import Tensor, nn
    from torch.nn.functional import interpolate

    if TYPE_CHECKING:
        from transformers.models.maskformer.modeling_maskformer import MaskFormerForInstanceSegmentationOutput

logger = logging.get_logger(__name__)


class MaskFormerFeatureExtractor(FeatureExtractionMixin, ImageFeatureExtractionMixin):
    r"""
    Constructs a MaskFormer feature extractor. The feature extractor can be used to prepare image(s) and optional
    targets for the model.

    This feature extractor inherits from [`FeatureExtractionMixin`] which contains most of the main methods. Users
    should refer to this superclass for more information regarding those methods.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the input to a certain `size`.
        size (`int`, *optional*, defaults to 800):
            Resize the input to the given size. Only has an effect if `do_resize` is set to `True`. If size is a
            sequence like `(width, height)`, output size will be matched to this. If size is an int, smaller edge of
            the image will be matched to this number. i.e, if `height > width`, then image will be rescaled to `(size *
            height / width, size)`.
        max_size (`int`, *optional*, defaults to 1333):
            The largest size an image dimension can have (otherwise it's capped). Only has an effect if `do_resize` is
            set to `True`.
        resample (`int`, *optional*, defaults to `PIL.Image.BILINEAR`):
            An optional resampling filter. This can be one of `PIL.Image.NEAREST`, `PIL.Image.BOX`,
            `PIL.Image.BILINEAR`, `PIL.Image.HAMMING`, `PIL.Image.BICUBIC` or `PIL.Image.LANCZOS`. Only has an effect
            if `do_resize` is set to `True`.
        size_divisibility (`int`, *optional*, defaults to 32):
            Some backbones need images divisible by a certain number. If not passed, it defaults to the value used in
            Swin Transformer.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the input with mean and standard deviation.
        image_mean (`int`, *optional*, defaults to `[0.485, 0.456, 0.406]`):
            The sequence of means for each channel, to be used when normalizing images. Defaults to the ImageNet mean.
        image_std (`int`, *optional*, defaults to `[0.229, 0.224, 0.225]`):
            The sequence of standard deviations for each channel, to be used when normalizing images. Defaults to the
            ImageNet std.
        ignore_index (`int`, *optional*):
            Value of the index (label) to ignore in the loss function.
        reduce_labels (`bool`, *optional*, defaults to `False`):
            Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0 is
            used for background, and background itself is not included in all classes of a dataset (e.g. ADE20k). The
            background label will be replaced by `ignore_index`.

    """

    model_input_names = ["pixel_values", "pixel_mask"]

    def __init__(
        self,
        do_resize=True,
        size=800,
        max_size=1333,
        resample=Image.BILINEAR,
        size_divisibility=32,
        do_normalize=True,
        image_mean=None,
        image_std=None,
        ignore_index=None,
        reduce_labels=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size = size
        self.max_size = max_size
        self.resample = resample
        self.size_divisibility = size_divisibility
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else [0.485, 0.456, 0.406]  # ImageNet mean
        self.image_std = image_std if image_std is not None else [0.229, 0.224, 0.225]  # ImageNet std
        self.ignore_index = ignore_index
        self.reduce_labels = reduce_labels

    def _resize_with_size_divisibility(self, image, size, target=None, max_size=None):
        """
        Resize the image to the given size. Size can be min_size (scalar) or (width, height) tuple. If size is an int,
        smaller edge of the image will be matched to this number.

        If given, also resize the target accordingly.
        """
        if not isinstance(image, Image.Image):
            image = self.to_pil_image(image)

        def get_size_with_aspect_ratio(image_size, size, max_size=None):
            width, height = image_size
            if max_size is not None:
                min_original_size = float(min((width, height)))
                max_original_size = float(max((width, height)))
                if max_original_size / min_original_size * size > max_size:
                    size = int(round(max_size * min_original_size / max_original_size))

            if (width <= height and width == size) or (height <= width and height == size):
                return (height, width)

            if width < height:
                output_width = size
                output_height = int(size * height / width)
            else:
                output_height = size
                output_width = int(size * width / height)

            return (output_height, output_width)

        def get_size(image_size, size, max_size=None):
            if isinstance(size, (list, tuple)):
                return size
            else:
                # size returned must be (width, height) since we use PIL to resize images
                # so we revert the tuple
                return get_size_with_aspect_ratio(image_size, size, max_size)[::-1]

        width, height = get_size(image.size, size, max_size)

        if self.size_divisibility > 0:
            height = int(np.ceil(height / self.size_divisibility)) * self.size_divisibility
            width = int(np.ceil(width / self.size_divisibility)) * self.size_divisibility

        size = (width, height)
        image = self.resize(image, size=size, resample=self.resample)

        if target is not None:
            target = self.resize(target, size=size, resample=Image.NEAREST)

        return image, target

    def __call__(
        self,
        images: ImageInput,
        segmentation_maps: ImageInput = None,
        pad_and_return_pixel_mask: Optional[bool] = True,
        instance_id_to_semantic_id: Optional[Dict[int, int]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several image(s) and optional annotations. Images are by default
        padded up to the largest image in a batch, and a pixel mask is created that indicates which pixels are
        real/which are padding.

        <Tip warning={true}>

        NumPy arrays and PyTorch tensors are converted to PIL images when resizing, so the most efficient is to pass
        PIL images.

        </Tip>

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.

            segmentation_maps (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`, *optional*):
                Optionally, the corresponding semantic segmentation maps with the pixel-wise annotations.

            pad_and_return_pixel_mask (`bool`, *optional*, defaults to `True`):
                Whether or not to pad images up to the largest image in a batch and create a pixel mask.

                If left to the default, will return a pixel mask that is:

                - 1 for pixels that are real (i.e. **not masked**),
                - 0 for pixels that are padding (i.e. **masked**).

            instance_id_to_semantic_id (`Dict[int, int]`, *optional*):
                If passed, we treat `segmentation_maps` as an instance segmentation map where each pixel represents an
                instance id. To convert it to a binary mask of shape (`batch, num_labels, height, width`) we need a
                dictionary mapping instance ids to label ids to create a semantic segmentation map.

            return_tensors (`str` or [`~file_utils.TensorType`], *optional*):
                If set, will return tensors instead of NumPy arrays. If set to `'pt'`, return PyTorch `torch.Tensor`
                objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **pixel_values** -- Pixel values to be fed to a model.
            - **pixel_mask** -- Pixel mask to be fed to a model (when `pad_and_return_pixel_mask=True` or if
              *"pixel_mask"* is in `self.model_input_names`).
            - **mask_labels** -- Optional a list of mask labels of shape `(labels, height, width)` to be fed to a model
              (when `annotations` are provided).
            - **class_labels** -- Optional a list of class labels of shape `(labels, num_labels)` to be fed to a model
              (when `annotations` are provided). They identify the labels of `mask_labels`, e.g. the label of
              `mask_labels[i][j]` if `class_labels[i][j]`.
        """
        # Input type checking for clearer error

        valid_images = False
        valid_segmentation_maps = False

        # Check that images has a valid type
        if isinstance(images, (Image.Image, np.ndarray)) or is_torch_tensor(images):
            valid_images = True
        elif isinstance(images, (list, tuple)):
            if len(images) == 0 or isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]):
                valid_images = True

        if not valid_images:
            raise ValueError(
                "Images must of type `PIL.Image.Image`, `np.ndarray` or `torch.Tensor` (single example), "
                "`List[PIL.Image.Image]`, `List[np.ndarray]` or `List[torch.Tensor]` (batch of examples)."
            )
        # Check that segmentation maps has a valid type
        if segmentation_maps is not None:
            if isinstance(segmentation_maps, (Image.Image, np.ndarray)) or is_torch_tensor(segmentation_maps):
                valid_segmentation_maps = True
            elif isinstance(segmentation_maps, (list, tuple)):
                if (
                    len(segmentation_maps) == 0
                    or isinstance(segmentation_maps[0], (Image.Image, np.ndarray))
                    or is_torch_tensor(segmentation_maps[0])
                ):
                    valid_segmentation_maps = True

            if not valid_segmentation_maps:
                raise ValueError(
                    "Segmentation maps must of type `PIL.Image.Image`, `np.ndarray` or `torch.Tensor` (single example),"
                    "`List[PIL.Image.Image]`, `List[np.ndarray]` or `List[torch.Tensor]` (batch of examples)."
                )

        is_batched = bool(
            isinstance(images, (list, tuple))
            and (isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]))
        )

        if not is_batched:
            images = [images]
            if segmentation_maps is not None:
                segmentation_maps = [segmentation_maps]

        # transformations (resizing + normalization)
        if self.do_resize and self.size is not None:
            if segmentation_maps is not None:
                for idx, (image, target) in enumerate(zip(images, segmentation_maps)):
                    image, target = self._resize_with_size_divisibility(
                        image=image, target=target, size=self.size, max_size=self.max_size
                    )
                    images[idx] = image
                    segmentation_maps[idx] = target
            else:
                for idx, image in enumerate(images):
                    images[idx] = self._resize_with_size_divisibility(
                        image=image, target=None, size=self.size, max_size=self.max_size
                    )[0]

        if self.do_normalize:
            images = [self.normalize(image=image, mean=self.image_mean, std=self.image_std) for image in images]
        # NOTE I will be always forced to pad them them since they have to be stacked in the batch dim
        encoded_inputs = self.encode_inputs(
            images,
            segmentation_maps,
            pad_and_return_pixel_mask,
            instance_id_to_semantic_id=instance_id_to_semantic_id,
            return_tensors=return_tensors,
        )

        # Convert to TensorType
        tensor_type = return_tensors
        if not isinstance(tensor_type, TensorType):
            tensor_type = TensorType(tensor_type)

        if not tensor_type == TensorType.PYTORCH:
            raise ValueError("Only PyTorch is supported for the moment.")
        else:
            if not is_torch_available():
                raise ImportError("Unable to convert output to PyTorch tensors format, PyTorch is not installed.")

        return encoded_inputs

    def _max_by_axis(self, the_list: List[List[int]]) -> List[int]:
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def convert_segmentation_map_to_binary_masks(
        self,
        segmentation_map: "np.ndarray",
        instance_id_to_semantic_id: Optional[Dict[int, int]] = None,
    ):
        if self.reduce_labels:
            segmentation_map[segmentation_map == 0] = self.ignore_index
            # instances ids start from 1!
            segmentation_map -= 1
            segmentation_map[segmentation_map == self.ignore_index - 1] = self.ignore_index

        if instance_id_to_semantic_id is not None:
            # segmentation_map will be treated as an instance segmentation map where each pixel is a instance id
            # thus it has to be converted to a semantic segmentation map
            for instance_id, label_id in instance_id_to_semantic_id.items():
                segmentation_map[segmentation_map == instance_id] = label_id
        # get all the labels in the image
        labels = np.unique(segmentation_map)
        # remove ignore index (if we have one)
        if self.ignore_index is not None:
            labels = labels[labels != self.ignore_index]
        # helping broadcast by making mask [1,W,H] and labels [C, 1, 1]
        binary_masks = segmentation_map[None] == labels[:, None, None]
        return binary_masks.astype(np.float32), labels.astype(np.int64)

    def encode_inputs(
        self,
        pixel_values_list: List["np.ndarray"],
        segmentation_maps: ImageInput = None,
        pad_and_return_pixel_mask: bool = True,
        instance_id_to_semantic_id: Optional[Dict[int, int]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ):
        """
        Pad images up to the largest image in a batch and create a corresponding `pixel_mask`.

        Args:
            pixel_values_list (`List[torch.Tensor]`):
                List of images (pixel values) to be padded. Each image should be a tensor of shape `(channels, height,
                width)`.

            segmentation_maps (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`, *optional*):
                The corresponding semantic segmentation maps with the pixel-wise annotations.

            pad_and_return_pixel_mask (`bool`, *optional*, defaults to `True`):
                Whether or not to pad images up to the largest image in a batch and create a pixel mask.

                If left to the default, will return a pixel mask that is:

                - 1 for pixels that are real (i.e. **not masked**),
                - 0 for pixels that are padding (i.e. **masked**).

            instance_id_to_semantic_id (`Dict[int, int]`, *optional*):
                If passed, we treat `segmentation_maps` as an instance segmentation map where each pixel represents an
                instance id. To convert it to a binary mask of shape (`batch, num_labels, height, width`) we need a
                dictionary mapping instance ids to label ids to create a semantic segmentation map.

            return_tensors (`str` or [`~file_utils.TensorType`], *optional*):
                If set, will return tensors instead of NumPy arrays. If set to `'pt'`, return PyTorch `torch.Tensor`
                objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **pixel_values** -- Pixel values to be fed to a model.
            - **pixel_mask** -- Pixel mask to be fed to a model (when `pad_and_return_pixel_mask=True` or if
              *"pixel_mask"* is in `self.model_input_names`).
            - **mask_labels** -- Optional a list of mask labels of shape `(labels, height, width)` to be fed to a model
              (when `annotations` are provided).
            - **class_labels** -- Optional a list of class labels of shape `(labels, num_labels)` to be fed to a model
              (when `annotations` are provided). They identify the labels of `mask_labels`, e.g. the label of
              `mask_labels[i][j]` if `class_labels[i][j]`.
        """

        max_size = self._max_by_axis([list(image.shape) for image in pixel_values_list])

        annotations = None
        if segmentation_maps is not None:
            segmentation_maps = map(np.array, segmentation_maps)
            converted_segmentation_maps = map(self.convert_segmentation_map_to_binary_masks, segmentation_maps)

            annotations = []
            for mask, classes in converted_segmentation_maps:
                annotations.append({"masks": mask, "classes": classes})

        channels, height, width = max_size
        pixel_values = []
        pixel_mask = []
        mask_labels = []
        class_labels = []
        for idx, image in enumerate(pixel_values_list):
            # create padded image
            padded_image = np.zeros((channels, height, width), dtype=np.float32)
            padded_image[: image.shape[0], : image.shape[1], : image.shape[2]] = np.copy(image)
            image = padded_image
            pixel_values.append(image)
            # if we have a target, pad it
            if annotations:
                annotation = annotations[idx]
                masks = annotation["masks"]
                # pad mask with `ignore_index`
                masks = np.pad(
                    masks,
                    ((0, 0), (0, height - masks.shape[1]), (0, width - masks.shape[2])),
                    constant_values=self.ignore_index,
                )
                annotation["masks"] = masks
            # create pixel mask
            mask = np.zeros((height, width), dtype=np.int64)
            mask[: image.shape[1], : image.shape[2]] = True
            pixel_mask.append(mask)

        # return as BatchFeature
        data = {"pixel_values": pixel_values, "pixel_mask": pixel_mask}
        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)
        # we cannot batch them since they don't share a common class size
        if annotations:
            for label in annotations:
                mask_labels.append(label["masks"])
                class_labels.append(label["classes"])

            encoded_inputs["mask_labels"] = mask_labels
            encoded_inputs["class_labels"] = class_labels

        return encoded_inputs

    def post_process_segmentation(
        self, outputs: "MaskFormerForInstanceSegmentationOutput", target_size: Tuple[int, int] = None
    ) -> "torch.Tensor":
        """
        Converts the output of [`MaskFormerForInstanceSegmentationOutput`] into image segmentation predictions. Only
        supports PyTorch.

        Args:
            outputs ([`MaskFormerForInstanceSegmentationOutput`]):
                The outputs from [`MaskFormerForInstanceSegmentation`].

            target_size (`Tuple[int, int]`, *optional*):
                If set, the `masks_queries_logits` will be resized to `target_size`.

        Returns:
            `torch.Tensor`:
                A tensor of shape (`batch_size, num_labels, height, width`).
        """
        # class_queries_logits has shape [BATCH, QUERIES, CLASSES + 1]
        class_queries_logits = outputs.class_queries_logits
        # masks_queries_logits has shape [BATCH, QUERIES, HEIGHT, WIDTH]
        masks_queries_logits = outputs.masks_queries_logits
        if target_size is not None:
            masks_queries_logits = interpolate(
                masks_queries_logits,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        # remove the null class `[..., :-1]`
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        # mask probs has shape [BATCH, QUERIES, HEIGHT, WIDTH]
        masks_probs = masks_queries_logits.sigmoid()
        # now we want to sum over the queries,
        # $ out_{c,h,w} =  \sum_q p_{q,c} * m_{q,h,w} $
        # where $ softmax(p) \in R^{q, c} $ is the mask classes
        # and $ sigmoid(m) \in R^{q, h, w}$ is the mask probabilities
        # b(atch)q(uery)c(lasses), b(atch)q(uery)h(eight)w(idth)
        segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)

        return segmentation

    def remove_low_and_no_objects(self, masks, scores, labels, object_mask_threshold, num_labels):
        """
        Binarize the given masks using `object_mask_threshold`, it returns the associated values of `masks`, `scores`
        and `labels`.

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
            `Tuple[`torch.Tensor`, `torch.Tensor`, `torch.Tensor`]`: The `masks`, `scores` and `labels` without the
            region < `object_mask_threshold`.
        """
        if not (masks.shape[0] == scores.shape[0] == labels.shape[0]):
            raise ValueError("mask, scores and labels must have the same shape!")

        to_keep = labels.ne(num_labels) & (scores > object_mask_threshold)

        return masks[to_keep], scores[to_keep], labels[to_keep]

    def post_process_semantic_segmentation(
        self, outputs: "MaskFormerForInstanceSegmentationOutput", target_size: Tuple[int, int] = None
    ) -> "torch.Tensor":
        """
        Converts the output of [`MaskFormerForInstanceSegmentationOutput`] into semantic segmentation predictions. Only
        supports PyTorch.

        Args:
            outputs ([`MaskFormerForInstanceSegmentationOutput`]):
                The outputs from [`MaskFormerForInstanceSegmentation`].

        Returns:
            `torch.Tensor`: A tensor of shape `batch_size, height, width`.
        """
        segmentation = self.post_process_segmentation(outputs, target_size)
        semantic_segmentation = segmentation.argmax(dim=1)
        return semantic_segmentation

    def post_process_panoptic_segmentation(
        self,
        outputs: "MaskFormerForInstanceSegmentationOutput",
        object_mask_threshold: float = 0.8,
        overlap_mask_area_threshold: float = 0.8,
        label_ids_to_fuse: Optional[Set[int]] = None,
    ) -> List[Dict]:
        """
        Converts the output of [`MaskFormerForInstanceSegmentationOutput`] into image panoptic segmentation
        predictions. Only supports PyTorch.

        Args:
            outputs ([`MaskFormerForInstanceSegmentationOutput`]):
                The outputs from [`MaskFormerForInstanceSegmentation`].
            object_mask_threshold (`float`, *optional*, defaults to 0.8):
                The object mask threshold.
            overlap_mask_area_threshold (`float`, *optional*, defaults to 0.8):
                The overlap mask area threshold to use.
            label_ids_to_fuse (`Set[int]`, *optional*):
                The labels in this state will have all their instances be fused together. For instance we could say
                there can only be one sky in an image, but several persons, so the label ID for sky would be in that
                set, but not the one for person.

        Returns:
            `List[Dict]`: A list of dictionaries, one per image, each dictionary containing two keys:
            - **segmentation** -- a tensor of shape `(height, width)` where each pixel represents a `segment_id`.
            - **segments** -- a dictionary with the following keys
                - **id** -- an integer representing the `segment_id`.
                - **label_id** -- an integer representing the segment's label.
                - **was_fused** -- a boolean, `True` if `label_id` was in `label_ids_to_fuse`, `False` otherwise.
        """

        if label_ids_to_fuse is None:
            logger.warning("`label_ids_to_fuse` unset. No instance will be fused.")
            label_ids_to_fuse = set()
        # class_queries_logits has shape [BATCH, QUERIES, CLASSES + 1]
        class_queries_logits = outputs.class_queries_logits
        # keep track of the number of labels, subtract -1 for null class
        num_labels = class_queries_logits.shape[-1] - 1
        # masks_queries_logits has shape [BATCH, QUERIES, HEIGHT, WIDTH]
        masks_queries_logits = outputs.masks_queries_logits
        # since all images are padded, they all have the same spatial dimensions
        _, _, height, width = masks_queries_logits.shape
        # for each query, the best scores and their indeces
        pred_scores, pred_labels = nn.functional.softmax(class_queries_logits, dim=-1).max(-1)
        # pred_scores and pred_labels shape = [BATH,NUM_QUERIES]
        mask_probs = masks_queries_logits.sigmoid()
        # mask probs has shape [BATCH, QUERIES, HEIGHT, WIDTH]
        # now, we need to iterate over the batch size to correctly process the segmentation we got from the queries using our thresholds. Even if the original predicted masks have the same shape across the batch, they won't after thresholding so batch-wise operations are impossible
        results: List[Dict[str, Tensor]] = []
        for (mask_probs, pred_scores, pred_labels) in zip(mask_probs, pred_scores, pred_labels):
            mask_probs, pred_scores, pred_labels = self.remove_low_and_no_objects(
                mask_probs, pred_scores, pred_labels, object_mask_threshold, num_labels
            )
            we_detect_something = mask_probs.shape[0] > 0

            segmentation = torch.zeros((height, width), dtype=torch.int32, device=mask_probs.device)
            segments: List[Dict] = []

            if we_detect_something:
                current_segment_id = 0
                # weight each mask by its score
                mask_probs *= pred_scores.view(-1, 1, 1)
                # find out for each pixel what is the most likely class to be there
                mask_labels = mask_probs.argmax(0)
                # mask_labels shape = [H,W] where each pixel has a class label
                stuff_memory_list: Dict[str, int] = {}
                # this is a map between stuff and segments id, the used it to keep track of the instances of one class
                for k in range(pred_labels.shape[0]):
                    pred_class = pred_labels[k].item()
                    # check if pred_class should be fused. For example, class "sky" cannot have more then one instance
                    should_fuse = pred_class in label_ids_to_fuse
                    # get the mask associated with the k class
                    mask_k = mask_labels == k
                    # create the area, since bool we just need to sum :)
                    mask_k_area = mask_k.sum()
                    # this is the area of all the stuff in query k
                    original_area = (mask_probs[k] >= 0.5).sum()

                    mask_exists = mask_k_area > 0 and original_area > 0

                    if mask_exists:
                        # find out how much of the all area mask_k is using
                        area_ratio = mask_k_area / original_area
                        mask_k_is_overlapping_enough = area_ratio.item() > overlap_mask_area_threshold

                        if mask_k_is_overlapping_enough:
                            # merge stuff regions
                            if pred_class in stuff_memory_list:
                                current_segment_id = stuff_memory_list[pred_class]
                            else:
                                current_segment_id += 1
                            # then we update out mask with the current segment
                            segmentation[mask_k] = current_segment_id
                            segments.append(
                                {
                                    "id": current_segment_id,
                                    "label_id": pred_class,
                                    "was_fused": should_fuse,
                                }
                            )
                            if should_fuse:
                                stuff_memory_list[pred_class] = current_segment_id
            results.append({"segmentation": segmentation, "segments": segments})
        return results
