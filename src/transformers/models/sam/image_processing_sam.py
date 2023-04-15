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
"""Image processor class for SAM."""
import math
from itertools import product
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import convert_to_rgb, normalize, rescale, to_channel_dimension_format
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
from ...utils import TensorType, is_torch_available, is_torch_tensor, is_vision_available, logging
from ...utils.import_utils import requires_backends


if is_vision_available():
    import PIL

if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


class SamImageProcessor(BaseImageProcessor):
    r"""
    Constructs a SAM image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by the
            `do_resize` parameter in the `preprocess` method.
        size (`dict`, *optional*, defaults to `{"height": 384, "width": 384}`):
            Size of the output image after resizing. Can be overridden by the `size` parameter in the `preprocess`
            method.
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
            Resampling filter to use if resizing the image. Only has an effect if `do_resize` is set to `True`. Can be
            overridden by the `resample` parameter in the `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Wwhether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the
            `do_rescale` parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Only has an effect if `do_rescale` is set to `True`. Can be
            overridden by the `rescale_factor` parameter in the `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method. Can be overridden by the `do_normalize` parameter in the `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
            overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        target_size: int = 1024,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        n_layers: int = 0,
        overlap_ratio: float=512 / 1500,
        points_per_crop: Optional[int] = 32,
        scale_per_layer: Optional[List[int]] = 1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"height": 384, "width": 384}
        size = get_size_dict(size, default_to_square=True)

        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD
        self.do_convert_rgb = do_convert_rgb
        self.target_size = target_size
        self.n_layers = n_layers
        self.overlap_ratio = overlap_ratio
        self.points_per_crop = points_per_crop
        self.scale_per_layer = scale_per_layer

    def get_preprocess_shape(self, oldh: int, oldw: int, long_side_length: int):
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def apply_coords(self, coords: np.ndarray, original_size, is_bounding_box=False) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(original_size[0], original_size[1], self.target_size)
        coords = deepcopy(coords).astype(float)

        if is_bounding_box:
            # reshape to .reshape(-1, 2, 2)
            coords = coords.reshape(-1, 2, 2)

        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)

        if is_bounding_box:
            # reshape back to .reshape(-1, 4)   
            coords = coords.reshape(-1, 4)

        return coords

    def pad_to_target_size(
        self,
        image: np.ndarray,
        target_size: int = None,
    ):
        requires_backends(self, "torch")

        target_size = target_size if target_size is not None else self.target_size

        import torch
        import torch.nn.functional as F

        image = torch.from_numpy(image).permute(2, 0, 1)

        height, width = image.shape[-2:]
        padh = target_size - height
        padw = target_size - width
        image = F.pad(image, (0, padw, 0, padh))

        return image.numpy()

    def resize(
        self,
        image: np.ndarray,
        target_size: int = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image to `(size["height"], size["width"])`.

        Args:
            image (`np.ndarray`):
                Image to resize.
            target_size (`int`, *optional*, defaults to `self.target_size`):
                Target size of the long side of the image.

        Returns:
            `np.ndarray`: The resized image.
        """
        requires_backends(self, "torchvision")
        target_size = target_size if target_size is not None else self.target_size

        output_size = self.get_preprocess_shape(image.shape[0], image.shape[1], target_size)

        from torchvision.transforms.functional import resize, to_pil_image

        image = to_pil_image(image)

        image = to_numpy_array(resize(image, output_size))

        return image.astype(np.float32)

    def rescale(
        self,
        image: np.ndarray,
        scale: Union[int, float],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ):
        """
        Rescale an image by a scale factor. image = image * scale.

        Args:
            image (`np.ndarray`):
                Image to rescale.
            scale (`int` or `float`):
                Scale to apply to the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
        """
        return rescale(image, scale=scale, data_format=data_format, **kwargs)

    def normalize(
        self,
        image: np.ndarray,
        mean: Union[float, List[float]],
        std: Union[float, List[float]],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Normalize an image. image = (image - image_mean) / image_std.

        Args:
            image (`np.ndarray`):
                Image to normalize.
            mean (`float` or `List[float]`):
                Image mean.
            std (`float` or `List[float]`):
                Image standard deviation.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
        """
        return normalize(image, mean=mean, std=std, data_format=data_format, **kwargs)

    def preprocess(
        self,
        images: ImageInput,
        input_points: Optional[tuple[int]] = None,
        input_labels: Optional[tuple[int]] = None,
        input_boxes: Optional[tuple[int]] = None,
        target_size: Optional[int] = None,
        do_resize: Optional[bool] = None,
        do_rescale: Optional[bool] = None,
        do_normalize: Optional[bool] = None,
        scale: Optional[Union[int, float]] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        do_convert_rgb: bool = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        **kwargs,
    ) -> PIL.Image.Image:
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Controls the size of the image after `resize`. The shortest edge of the image is resized to
                `size["shortest_edge"]` whilst preserving the aspect ratio. If the longest edge of this resized image
                is > `int(size["shortest_edge"] * (1333 / 800))`, then the image is resized again to make the longest
                edge equal to `int(size["shortest_edge"] * (1333 / 800))`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to normalize the image by if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to normalize the image by if `do_normalize` is set to `True`.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                    - Unset: Return a list of `np.ndarray`.
                    - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                    - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                    - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                    - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                    - `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                    - `ChannelDimension.LAST`: image in (height, width, num_channels) format.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        scale = scale if do_rescale and scale is not None else self.rescale_factor
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        target_size = target_size if target_size is not None else self.target_size

        images = make_list_of_images(images)

        if input_points is not None:
            if not isinstance(input_points, tuple):
                raise ValueError("Input points must be a tuple of tuple of floating integers.")
            input_points = [np.array([point]) for point in input_points]
        else:
            input_points = None

        if input_points is not None:
            if not isinstance(input_labels, tuple):
                raise ValueError("Input labels must be a tuple of integers.")
            input_labels = [np.array([label]) for label in input_labels]
        else:
            input_labels = None

        if input_boxes is not None:
            if not isinstance(input_boxes, tuple):
                raise ValueError("Input boxes must be a tuple of tuple of floating integers.")
            input_boxes = [np.array(box) for box in input_boxes]
        else:
            input_boxes = None

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        if do_normalize and (image_mean is None or image_std is None):
            raise ValueError("Image mean and std must be specified if do_normalize is True.")

        # PIL RGBA images are converted to RGB
        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        original_sizes = [image.shape[:2] for image in images]

        if input_points is not None:
            input_points = [
                self.apply_coords(point, original_size) for point, original_size in zip(input_points, original_sizes)
            ]

        if input_boxes is not None:
            input_boxes = [
                self.apply_coords(box, original_size, is_bounding_box=True) for box, original_size in zip(input_boxes, original_sizes)
            ]

        if do_resize:
            images = [self.resize(image=image) for image in images]

        if do_rescale:
            images = [self.rescale(image=image, scale=scale) for image in images]

        if do_normalize:
            images = [self.normalize(image=image, mean=image_mean, std=image_std) for image in images]

        images = [self.pad_to_target_size(image=image, target_size=target_size) for image in images]

        images = [to_channel_dimension_format(image, data_format) for image in images]

        data = {"pixel_values": images}
        if input_labels is not None:
            data["input_labels"] = input_labels
        if input_points is not None:
            data["input_points"] = input_points
        if input_boxes is not None:
            data["input_boxes"] = input_boxes

        encoded_outputs = BatchFeature(data=data, tensor_type=return_tensors)

        return encoded_outputs

    def build_point_grid(self, n_per_side: int) -> np.ndarray:
        """Generates a 2D grid of points evenly spaced in [0,1]x[0,1]."""
        offset = 1 / (2 * n_per_side)
        points_one_side = np.linspace(offset, 1 - offset, n_per_side)
        points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
        points_y = np.tile(points_one_side[:, None], (1, n_per_side))
        points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
        return points


    def build_all_layer_point_grids(
        self, points_per_crop: int = None, n_layers: int = None, scale_per_layer: int = None
    ) -> List[np.ndarray]:
        """Generates point grids for all crop layers."""
        points_per_crop = points_per_crop if points_per_crop is not None else self.points_per_crop
        n_layers = n_layers if n_layers is not None else self.n_layers
        scale_per_layer = scale_per_layer if scale_per_layer is not None else self.scale_per_layer

        points_by_layer = []
        for i in range(n_layers + 1):
            n_points = int(points_per_crop / (scale_per_layer**i))
            points_by_layer.append(self.build_point_grid(n_points))
        return points_by_layer

    def generate_crop_boxes(
        self, 
        image, 
        n_layers: int=None, 
        overlap_ratio: float=None,
        points_per_crop: int=None,
        scale_per_layer: int=None,
        return_tensors="pt",
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Generates a list of crop boxes of different sizes. Each layer
        has (2**i)**2 boxes for the ith layer.
        """
        points_per_crop = points_per_crop if points_per_crop is not None else self.points_per_crop
        scale_per_layer = scale_per_layer if scale_per_layer is not None else self.scale_per_layer
        
        if isinstance(image, list):
            raise ValueError("Only one image is allowed for crop generation.")
        image = to_numpy_array(image)

        points_grid = self.build_all_layer_point_grids(
            points_per_crop=points_per_crop, n_layers=n_layers, scale_per_layer=scale_per_layer
        )

        n_layers = n_layers if n_layers is not None else self.n_layers
        overlap_ratio = overlap_ratio if overlap_ratio is not None else self.overlap_ratio

        crop_boxes, layer_idxs = [], []
        im_h, im_w = image.shape[:2]
        short_side = min(im_h, im_w)

        # Original image
        crop_boxes.append([0, 0, im_w, im_h])
        layer_idxs.append(0)

        def crop_len(orig_len, n_crops, overlap):
            return int(math.ceil((overlap * (n_crops - 1) + orig_len) / n_crops))

        for i_layer in range(n_layers):
            n_crops_per_side = 2 ** (i_layer + 1)
            overlap = int(overlap_ratio * short_side * (2 / n_crops_per_side))

            crop_w = crop_len(im_w, n_crops_per_side, overlap)
            crop_h = crop_len(im_h, n_crops_per_side, overlap)

            crop_box_x0 = [int((crop_w - overlap) * i) for i in range(n_crops_per_side)]
            crop_box_y0 = [int((crop_h - overlap) * i) for i in range(n_crops_per_side)]

            # Crops in XYWH format
            for x0, y0 in product(crop_box_x0, crop_box_y0):
                box = [x0, y0, min(x0 + crop_w, im_w), min(y0 + crop_h, im_h)]
                crop_boxes.append(box)
                layer_idxs.append(i_layer + 1)

        # generate cropped images
        cropped_images = []
        points_per_crop = []
        for i, crop_box in enumerate(crop_boxes):
            x0, y0, x1, y1 = crop_box
            cropped_im = image[y0:y1, x0:x1, :]
            cropped_images.append(cropped_im)

            cropped_im_size = cropped_im.shape[:2]
            points_scale = np.array(cropped_im_size)[None, ::-1]

            points_per_crop.append(points_grid[layer_idxs[i]] * points_scale)

        if return_tensors == "pt":
            import torch

            crop_boxes = torch.tensor(crop_boxes, dtype=torch.float32)
            points_per_crop = torch.cat([torch.tensor(p).unsqueeze(0) for p in points_per_crop], dim=0)
        else:
            raise ValueError("Only 'pt' is supported for return_tensors.")


        return crop_boxes, points_per_crop, cropped_images

    def postprocess_masks(self, images, masks: torch.Tensor,  mask_threshold=0.0):
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        requires_backends(self, "torch")
        import torch.nn.functional as F

        if not isinstance(images, list):
            images = [images]

        images = [to_numpy_array(image) for image in images]

        original_sizes = torch.LongTensor([image.shape[:2] for image in images])

        # TODO: potentially remove this for batched decoding
        if self.do_resize:
            images = [self.resize(image=image) for image in images]

        input_sizes = images[0].shape[:2] # they all have the same shape

        image_size = (self.target_size, self.target_size)

        if len(masks.shape) == 3:
            masks = masks.unsqueeze(0)

        masks = F.interpolate(masks, image_size, mode="bilinear", align_corners=False)
        masks = masks[..., : input_sizes[0], : input_sizes[1]]

        # check if original size is not the same across batches
        output_masks = []
        if original_sizes.shape[0] > 1:
            for i in range(original_sizes.shape[0]):
                output_masks.append(
                    F.interpolate(
                        masks[i].unsqueeze(0),
                        (original_sizes[i][0].item(), original_sizes[i][1].item()),
                        mode="bilinear",
                        align_corners=False,
                    )
                )
            output_masks = torch.cat(output_masks, dim=0)
        else:
            output_masks = F.interpolate(
                masks, (original_sizes[0][0].item(), original_sizes[0][1].item()), mode="bilinear", align_corners=False
            )

        output_masks = output_masks > mask_threshold
        return output_masks
