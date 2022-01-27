# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""Feature extractor class for DETR."""

import io
import pathlib
import PIL
from collections import defaultdict
from typing import Dict, List, Optional, Union

import numpy as np
from PIL import Image

from ...feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from ...file_utils import TensorType, is_torch_available
from ...image_utils import ImageFeatureExtractionMixin, is_torch_tensor
from ...utils import logging


if is_torch_available():
    import torch
    from torch import nn

logger = logging.get_logger(__name__)


ImageInput = Union[Image.Image, np.ndarray, "torch.Tensor", List[Image.Image], List[np.ndarray], List["torch.Tensor"]]


class MaskFormerFeatureExtractor(FeatureExtractionMixin, ImageFeatureExtractionMixin):
    r"""
    Constructs a MaskFormer feature extractor.

    This feature extractor inherits from [`FeatureExtractionMixin`] which contains most of the main methods. Users
    should refer to this superclass for more information regarding those methods.


    Args:
        format (`str`, *optional*, defaults to `"coco_detection"`):
            Data format of the annotations. One of "coco_detection" or "coco_panoptic".
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the input to a certain `size`.
        size (`int`, *optional*, defaults to 800):
            Resize the input to the given size. Only has an effect if `do_resize` is set to `True`. If size is a
            sequence like `(width, height)`, output size will be matched to this. If size is an int, smaller edge of
            the image will be matched to this number. i.e, if `height > width`, then image will be rescaled to `(size *
            height / width, size)`.
        max_size (`int`, *optional*, defaults to `1333`):
            The largest size an image dimension can have (otherwise it's capped). Only has an effect if `do_resize` is
            set to `True`.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the input with mean and standard deviation.
        image_mean (`int`, *optional*, defaults to `[0.485, 0.456, 0.406]`):
            The sequence of means for each channel, to be used when normalizing images. Defaults to the ImageNet mean.
        image_std (`int`, *optional*, defaults to `[0.229, 0.224, 0.225]`):
            The sequence of standard deviations for each channel, to be used when normalizing images. Defaults to the
            ImageNet std.
    """

    model_input_names = ["pixel_values", "pixel_mask"]

    def __init__(
        self, do_resize=True, size=800, max_size=1333, do_normalize=True, image_mean=None, image_std=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size = size
        self.max_size = max_size
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else [0.485, 0.456, 0.406]  # ImageNet mean
        self.image_std = image_std if image_std is not None else [0.229, 0.224, 0.225]  # ImageNet std

    def _resize(self, image, size, target=None, max_size=None):
        """
        Resize the image to the given size. Size can be min_size (scalar) or (w, h) tuple. If size is an int, smaller
        edge of the image will be matched to this number.

        If given, also resize the target accordingly.
        """
        if not isinstance(image, Image.Image):
            image = self.to_pil_image(image)

        def get_size_with_aspect_ratio(image_size, size, max_size=None):
            w, h = image_size
            if max_size is not None:
                min_original_size = float(min((w, h)))
                max_original_size = float(max((w, h)))
                if max_original_size / min_original_size * size > max_size:
                    size = int(round(max_size * min_original_size / max_original_size))

            if (w <= h and w == size) or (h <= w and h == size):
                return (h, w)

            if w < h:
                ow = size
                oh = int(size * h / w)
            else:
                oh = size
                ow = int(size * w / h)

            return (oh, ow)

        def get_size(image_size, size, max_size=None):
            if isinstance(size, (list, tuple)):
                return size
            else:
                # size returned must be (w, h) since we use PIL to resize images
                # so we revert the tuple
                return get_size_with_aspect_ratio(image_size, size, max_size)[::-1]

        size = get_size(image.size, size, max_size)
        rescaled_image = self.resize(image, size=size)
        # pil image have inverted w, h
        w, h = size

        has_target = target is not None

        if has_target:
            target = target.copy()
            # store original_size
            target["original_size"] = image.size
            if "masks" in target:
                #           use PyTorch as current workaround
                # TODO replace by self.resize
                masks = torch.from_numpy(target["masks"][:, None]).float()
                interpolated_masks = nn.functional.interpolate(masks, size=(h, w), mode="nearest")[:, 0] > 0.5
                target["masks"] = interpolated_masks.numpy()

        return rescaled_image, target

    def _normalize(self, image, mean, std, target=None):
        """
        Normalize the image with a certain mean and std.

        If given, also normalize the target bounding boxes based on the size of the image.
        """

        image = self.normalize(image, mean=mean, std=std)

        return image, target

    def __call__(
        self,
        images: ImageInput,
        annotations: Union[List[Dict], List[List[Dict]]] = None,
        pad_and_return_pixel_mask: Optional[bool] = True,
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

            annotations (`Dict`, `List[Dict]`, *optional*):
                The corresponding annotations in the following format: { "masks" : the target mask, with shape [C,H,W], "labels" : the target labels, with shape [C]}

            TODO explain we pad and return a padding mask

            pad_and_return_pixel_mask (`bool`, *optional*, defaults to `True`):
                Whether or not to pad images up to the largest image in a batch and create a pixel mask.

                If left to the default, will return a pixel mask that is:

                - 1 for pixels that are real (i.e. **not masked**),
                - 0 for pixels that are padding (i.e. **masked**).

            return_tensors (`str` or [`~file_utils.TensorType`], *optional*):
                If set, will return tensors instead of NumPy arrays. If set to `'pt'`, return PyTorch `torch.Tensor`
                objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **pixel_values** -- Pixel values to be fed to a model.
            - **pixel_mask** -- Pixel mask to be fed to a model (when `pad_and_return_pixel_mask=True` or if
              *"pixel_mask"* is in `self.model_input_names`).
            - **labels** -- Optional labels to be fed to a model (when `annotations` are provided)
        """
        # Input type checking for clearer error

        valid_images = False
        valid_annotations = False

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

        is_batched = bool(
            isinstance(images, (list, tuple))
            and (isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]))
        )

        if not is_batched:
            images = [images]
            if annotations is not None:
                annotations = [annotations]

        # Check that annotations has a valid type
        if annotations is not None:
            # TODO ask best way to check!
            valid_annotations = type(annotations) is list and "masks" in annotations[0] and "labels" in annotations[0]
            if not valid_annotations:
                raise ValueError(
                    """
                    Annotations must of type `Dict` (single image) or `List[Dict]` (batch of images). In case of object
                    detection, each dictionary should contain the keys 'image_id' and 'annotations', with the latter
                    being a list of annotations in COCO format. In case of panoptic segmentation, each dictionary
                    should contain the keys 'file_name', 'image_id' and 'segments_info', with the latter being a list
                    of annotations in COCO format.
                    """
                )

        # transformations (resizing + normalization)
        if self.do_resize and self.size is not None:
            if annotations is not None:
                for idx, (image, target) in enumerate(zip(images, annotations)):
                    image, target = self._resize(image=image, target=target, size=self.size, max_size=self.max_size)
                    images[idx] = image
                    annotations[idx] = target
            else:
                for idx, image in enumerate(images):
                    images[idx] = self._resize(image=image, target=None, size=self.size, max_size=self.max_size)[0]

        if self.do_normalize:
            if annotations is not None:
                for idx, (image, target) in enumerate(zip(images, annotations)):
                    image, target = self._normalize(
                        image=image, mean=self.image_mean, std=self.image_std, target=target
                    )
                    images[idx] = image
                    annotations[idx] = target
            else:
                images = [
                    self._normalize(image=image, mean=self.image_mean, std=self.image_std)[0] for image in images
                ]
        # NOTE I will be always forced to convert them since they have to be stacked in the batch dim
        encoded_inputs = self.maybe_pad_and_create_pixel_mask(
            images, annotations, should_pad=pad_and_return_pixel_mask, return_tensors=return_tensors
        )

        if annotations is not None:
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

    def maybe_pad_and_create_pixel_mask(
        self,
        pixel_values_list: List["torch.Tensor"],
        annotations: Optional[List[Dict]] = None,
        should_pad: Optional[bool] = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ):
        """
        Pad images up to the largest image in a batch and create a corresponding `pixel_mask`.

        Args:
            pixel_values_list (`List[torch.Tensor]`):
                List of images (pixel values) to be padded. Each image should be a tensor of shape (C, H, W).
            return_tensors (`str` or [`~file_utils.TensorType`], *optional*):
                If set, will return tensors instead of NumPy arrays. If set to `'pt'`, return PyTorch `torch.Tensor`
                objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **pixel_values** -- Pixel values to be fed to a model.
            - **pixel_mask** -- Pixel mask to be fed to a model (when `pad_and_return_pixel_mask=True` or if
              *"pixel_mask"* is in `self.model_input_names`).

        """

        max_size = self._max_by_axis([list(image.shape) for image in pixel_values_list])
        c, h, w = max_size
        pixel_values = []
        pixel_mask = []
        pixel_labels = []
        class_labels = []

        for idx, image in enumerate(pixel_values_list):
            # create padded image
            if should_pad:
                padded_image = np.zeros((c, h, w), dtype=np.float32)
                padded_image[: image.shape[0], : image.shape[1], : image.shape[2]] = np.copy(image)
                image = padded_image
            pixel_values.append(image)
            # if we have a target, pad it
            if annotations:
                annotation = annotations[idx]
                masks = annotation["masks"]
                if should_pad:
                    padded_masks = np.zeros((masks.shape[0], h, w), dtype=masks.dtype)
                    padded_masks[:, : padded_masks.shape[1], : padded_masks.shape[2]] = np.copy(padded_masks)
                    masks = padded_masks
                pixel_labels.append(masks)
                class_labels.append(annotation["labels"])
            if should_pad:
                # create pixel mask
                mask = np.zeros((h, w), dtype=np.int64)
                mask[: image.shape[1], : image.shape[2]] = True
                pixel_mask.append(mask)

        # return as BatchFeature
        data = {"pixel_values": pixel_values, "pixel_mask": pixel_mask}

        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)
        if annotations:
            # BatchFeature doesn't support nested dicts, adding after it
            tensor_type = return_tensors
            if not isinstance(tensor_type, TensorType):
                tensor_type = TensorType(tensor_type)
            if not tensor_type == TensorType.PYTORCH:
                raise ValueError("Only PyTorch is supported for the moment.")
            else:
                if not is_torch_available():
                    raise ImportError("Unable to convert output to PyTorch tensors format, PyTorch is not installed.")

                encoded_inputs["labels"] = {
                    "pixel": torch.from_numpy(np.stack(pixel_labels)).float(),
                    "classes": torch.from_numpy(np.stack(class_labels)).long(),
                }
        return encoded_inputs
