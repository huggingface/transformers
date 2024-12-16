# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from .image_processing_utils import (
    BaseImageProcessor,
    BatchFeature,
    get_patch_output_size,
    get_size_dict,
    select_best_resolution,
)
from .image_transforms import convert_to_rgb, get_resize_output_image_size, get_size_with_aspect_ratio
from .image_utils import (
    ChannelDimension,
    ImageInput,
    ImageType,
    get_image_size,
    get_image_type,
    infer_channel_dimension_format,
    make_batched_images,
    make_list_of_images,
    validate_fast_preprocess_arguments,
    validate_kwargs,
)
from .utils import (
    TensorType,
    is_torch_available,
    is_torchvision_available,
    is_torchvision_v2_available,
    is_vision_available,
)


if is_vision_available():
    from .image_utils import PILImageResampling

if is_torch_available():
    import torch

if is_torchvision_available():
    from .image_utils import pil_torch_interpolation_mapping

    if is_torchvision_v2_available():
        from torchvision.transforms.v2 import functional as F
    else:
        from torchvision.transforms import functional as F


def get_image_size_for_max_height_width(
    image_size: Tuple[int, int],
    max_height: int,
    max_width: int,
) -> Tuple[int, int]:
    """
    Computes the output image size given the input image and the maximum allowed height and width. Keep aspect ratio.
    Important, even if image_height < max_height and image_width < max_width, the image will be resized
    to at least one of the edges be equal to max_height or max_width.

    For example:
        - input_size: (100, 200), max_height: 50, max_width: 50 -> output_size: (25, 50)
        - input_size: (100, 200), max_height: 200, max_width: 500 -> output_size: (200, 400)

    Args:
        image_size (`Tuple[int, int]`):
            The image to resize.
        max_height (`int`):
            The maximum allowed height.
        max_width (`int`):
            The maximum allowed width.
    """
    height, width = image_size
    height_scale = max_height / height
    width_scale = max_width / width
    min_scale = min(height_scale, width_scale)
    new_height = int(height * min_scale)
    new_width = int(width * min_scale)
    return new_height, new_width


def safe_squeeze(tensor: "torch.Tensor", axis: Optional[int] = None) -> "torch.Tensor":
    """
    Squeezes a tensor, but only if the axis specified has dim 1.
    """
    if axis is None:
        return tensor.squeeze()

    try:
        return tensor.squeeze(axis=axis)
    except ValueError:
        return tensor


def max_across_indices(values: Iterable[Any]) -> List[Any]:
    """
    Return the maximum value across all indices of an iterable of values.
    """
    return [max(values_i) for values_i in zip(*values)]


def get_max_height_width(images: List["torch.Tensor"]) -> Tuple[int]:
    """
    Get the maximum height and width across all images in a batch.
    """

    _, max_height, max_width = max_across_indices([img.shape for img in images])

    return (max_height, max_width)


def divide_to_patches(
    image: Union[np.array, "torch.Tensor"], patch_size: int
) -> List[Union[np.array, "torch.Tensor"]]:
    """
    Divides an image into patches of a specified size.

    Args:
        image (`Union[np.array, "torch.Tensor"]`):
            The input image.
        patch_size (`int`):
            The size of each patch.
    Returns:
        list: A list of Union[np.array, "torch.Tensor"] representing the patches.
    """
    patches = []
    height, width = get_image_size(image, channel_dim=ChannelDimension.FIRST)
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            patch = image[:, i : i + patch_size, j : j + patch_size]
            patches.append(patch)

    return patches


def group_images_by_shape(
    images: List["torch.Tensor"],
) -> Tuple[Dict[Tuple[int, int], List["torch.Tensor"]], Dict[int, Tuple[Tuple[int, int], int]]]:
    """
    Groups images by shape.
    Returns a dictionary with the shape as key and a list of images with that shape as value,
    and a dictionary with the index of the image in the original list as key and the shape and index in the grouped list as value.
    """
    grouped_images = {}
    grouped_images_index = {}
    for i, image in enumerate(images):
        shape = image.shape[1:]
        if shape not in grouped_images:
            grouped_images[shape] = []
        grouped_images[shape].append(image)
        grouped_images_index[i] = (shape, len(grouped_images[shape]) - 1)
    return grouped_images, grouped_images_index


def reorder_images(
    processed_images: Dict[Tuple[int, int], "torch.Tensor"], grouped_images_index: Dict[int, Tuple[int, int]]
) -> List["torch.Tensor"]:
    """
    Reconstructs a list of images in the original order.
    """
    return [
        processed_images[grouped_images_index[i][0]][grouped_images_index[i][1]]
        for i in range(len(grouped_images_index))
    ]


@dataclass(frozen=True)
class SizeDict:
    """
    Hashable dictionary to store image size information.
    """

    height: int = None
    width: int = None
    longest_edge: int = None
    shortest_edge: int = None
    max_height: int = None
    max_width: int = None

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Key {key} not found in SizeDict.")


class BaseImageProcessorFast(BaseImageProcessor):
    r"""
    Constructs a Fast Base image processor.

    Args:
        do_resize (`bool`, *optional*):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by the
            `do_resize` parameter in the `preprocess` method.
        size (`dict`, *optional*):
            Size of the output image after resizing. Can be overridden by the `size` parameter in the `preprocess`
            method.
        resample (`PILImageResampling`, *optional*):
            Resampling filter to use if resizing the image. Only has an effect if `do_resize` is set to `True`. Can be
            overridden by the `resample` parameter in the `preprocess` method.
        do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether to center crop the image to the specified `crop_size`. Can be overridden by `do_center_crop` in the
            `preprocess` method.
        crop_size (`Dict[str, int]` *optional*, defaults to 224):
            Size of the output image after applying `center_crop`. Can be overridden by `crop_size` in the `preprocess`
            method.
        do_rescale (`bool`, *optional*):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the
            `do_rescale` parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Only has an effect if `do_rescale` is set to `True`. Can be
            overridden by the `rescale_factor` parameter in the `preprocess` method.
        do_normalize (`bool`, *optional*):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method. Can be overridden by the `do_normalize` parameter in the `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
            overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_convert_rgb (`bool`, *optional*):
            Whether to convert the image to RGB.
    """

    resample = None
    image_mean = None
    image_std = None
    size = None
    default_to_square = None
    crop_size = None
    do_resize = None
    do_center_crop = None
    do_rescale = None
    do_normalize = None
    do_convert_rgb = None
    model_input_names = ["pixel_values"]
    valid_extra_kwargs = ["default_to_square", "device"]

    def __init__(
        self,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: Union["PILImageResampling", "F.InterpolationMode"] = None,
        do_center_crop: bool = None,
        crop_size: Dict[str, int] = None,
        do_rescale: bool = None,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = None,
        image_mean: Union[float, List[float]] = None,
        image_std: Union[float, List[float]] = None,
        do_convert_rgb: bool = None,
        **kwargs,
    ) -> None:
        size = size if size is not None else self.size
        default_to_square = kwargs.pop(
            "default_to_square", self.default_to_square if self.default_to_square is not None else True
        )
        size = get_size_dict(size, default_to_square=default_to_square) if size is not None else None
        crop_size = crop_size if crop_size is not None else self.crop_size
        crop_size = get_size_dict(crop_size, param_name="crop_size") if crop_size is not None else None

        super().__init__(**kwargs)
        self.do_resize = do_resize if do_resize is not None else self.do_resize
        self.size = size if size is not None else self.size
        self.resample = resample if resample is not None else self.resample
        self.do_center_crop = do_center_crop if do_center_crop is not None else self.do_center_crop
        self.crop_size = crop_size if crop_size is not None else self.crop_size
        self.do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        self.image_mean = image_mean if image_mean is not None else self.image_mean
        self.image_std = image_std if image_std is not None else self.image_std
        self.do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

    def resize(
        self,
        image: "torch.Tensor",
        size: SizeDict,
        interpolation: "F.InterpolationMode" = None,
        **kwargs,
    ) -> "torch.Tensor":
        """
        Resize an image to `(size["height"], size["width"])`.

        Args:
            image (`torch.Tensor`):
                Image to resize.
            size (`SizeDict`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            resample (`InterpolationMode`, *optional*, defaults to `InterpolationMode.BILINEAR`):
                `InterpolationMode` filter to use when resizing the image e.g. `InterpolationMode.BICUBIC`.

        Returns:
            `torch.Tensor`: The resized image.
        """
        interpolation = interpolation if interpolation is not None else F.InterpolationMode.BILINEAR
        if size.shortest_edge and size.longest_edge:
            # Resize the image so that the shortest edge or the longest edge is of the given size
            # while maintaining the aspect ratio of the original image.
            new_size = get_size_with_aspect_ratio(
                image.size()[-2:],
                size.shortest_edge,
                size.longest_edge,
            )
        elif size.shortest_edge:
            new_size = get_resize_output_image_size(
                image,
                size=size.shortest_edge,
                default_to_square=False,
                input_data_format=ChannelDimension.FIRST,
            )
        elif size.max_height and size.max_width:
            new_size = get_image_size_for_max_height_width(image.size()[-2:], size.max_height, size.max_width)
        elif size.height and size.width:
            new_size = (size.height, size.width)
        else:
            raise ValueError(
                "Size must contain 'height' and 'width' keys, or 'max_height' and 'max_width', or 'shortest_edge' key. Got"
                f" {size}."
            )
        return F.resize(image, new_size, interpolation=interpolation)

    def rescale(
        self,
        image: "torch.Tensor",
        scale: float,
        **kwargs,
    ) -> "torch.Tensor":
        """
        Rescale an image by a scale factor. image = image * scale.

        Args:
            image (`torch.Tensor`):
                Image to rescale.
            scale (`float`):
                The scaling factor to rescale pixel values by.

        Returns:
            `torch.Tensor`: The rescaled image.
        """
        return image * scale

    def normalize(
        self,
        image: "torch.Tensor",
        mean: Union[float, Iterable[float]],
        std: Union[float, Iterable[float]],
        **kwargs,
    ) -> "torch.Tensor":
        """
        Normalize an image. image = (image - image_mean) / image_std.

        Args:
            image (`torch.Tensor`):
                Image to normalize.
            mean (`torch.Tensor`, `float` or `Iterable[float]`):
                Image mean to use for normalization.
            std (`torch.Tensor`, `float` or `Iterable[float]`):
                Image standard deviation to use for normalization.

        Returns:
            `torch.Tensor`: The normalized image.
        """
        return F.normalize(image, mean, std)

    def center_crop(
        self,
        image: "torch.Tensor",
        size: Dict[str, int],
        **kwargs,
    ) -> "torch.Tensor":
        """
        Center crop an image to `(size["height"], size["width"])`. If the input size is smaller than `crop_size` along
        any edge, the image is padded with 0's and then center cropped.

        Args:
            image (`"torch.Tensor"`):
                Image to center crop.
            size (`Dict[str, int]`):
                Size of the output image.

        Returns:
            `torch.Tensor`: The center cropped image.
        """
        if size.height is None or size.width is None:
            raise ValueError(f"The size dictionary must have keys 'height' and 'width'. Got {size.keys()}")
        return F.center_crop(image, (size["height"], size["width"]))

    def convert_to_rgb(
        self,
        image: ImageInput,
    ) -> ImageInput:
        """
        Converts an image to RGB format. Only converts if the image is of type PIL.Image.Image, otherwise returns the image
        as is.
        Args:
            image (ImageInput):
                The image to convert.

        Returns:
            ImageInput: The converted image.
        """
        return convert_to_rgb(image)

    def _prepare_images_structure(
        self,
        images: ImageInput,
    ) -> ImageInput:
        """
        Prepare the images structure for processing.

        Args:
            images (`ImageInput`):
                The input images to process.

        Returns:
            `ImageInput`: The images with a valid nesting.
        """
        return make_list_of_images(images)

    def _prepare_input_images(
        self,
        images: ImageInput,
        do_convert_rgb: bool = None,
        device: Optional["torch.device"] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> List["torch.Tensor"]:
        """
        Prepare the input images for processing.
        """
        images = self._prepare_images_structure(images)
        image_type = get_image_type(images[0])
        if image_type not in [ImageType.PIL, ImageType.TORCH, ImageType.NUMPY]:
            raise ValueError(f"Unsupported input image type {image_type}")

        if do_convert_rgb:
            images = [self.convert_to_rgb(image) for image in images]

        if image_type == ImageType.PIL:
            images = [F.pil_to_tensor(image) for image in images]
        elif image_type == ImageType.NUMPY:
            # not using F.to_tensor as it doesn't handle (C, H, W) numpy arrays
            images = [torch.from_numpy(image).contiguous() for image in images]

        # Now that we have torch tensors, we can move them to the right device
        if device is not None:
            images = [image.to(device) for image in images]

        # We assume that all images have the same channel dimension format.
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(images[0])
        if input_data_format == ChannelDimension.LAST:
            # We force the channel dimension to be first for torch tensors as this is what torchvision expects.
            images = [image.permute(2, 0, 1).contiguous() for image in images]

        return images

    def _prepare_process_arguments(
        self,
        device: Optional["torch.device"] = None,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: Optional[Union["PILImageResampling", "F.InterpolationMode"]] = None,
        crop_size: int = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
    ) -> tuple:
        """
        Prepare the arguments for the process method.
        """
        # Make hashable for cache
        size = SizeDict(**size) if size is not None else None
        crop_size = SizeDict(**crop_size) if crop_size is not None else None
        image_mean = tuple(image_mean) if isinstance(image_mean, list) else image_mean
        image_std = tuple(image_std) if isinstance(image_std, list) else image_std

        validate_fast_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
            return_tensors=return_tensors,
            data_format=data_format,
        )

        if do_rescale and do_normalize:
            # fused rescale and normalize
            image_mean = torch.tensor(image_mean, device=device) * (1.0 / rescale_factor)
            image_std = torch.tensor(image_std, device=device) * (1.0 / rescale_factor)

        interpolation = (
            pil_torch_interpolation_mapping[resample] if isinstance(resample, (PILImageResampling, int)) else resample
        )

        return image_mean, image_std, size, crop_size, interpolation

    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: Optional[Union["PILImageResampling", "F.InterpolationMode"]] = None,
        do_center_crop: bool = None,
        crop_size: int = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Describes the maximum input dimensions to the model.
            resample (`PILImageResampling` or `InterpolationMode`, *optional*, defaults to self.resample):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
                has an effect if `do_resize` is set to `True`.
            do_center_crop (`bool`, *optional*, defaults to `self.do_center_crop`):
                Whether to center crop the image.
            crop_size (`Dict[str, int]`, *optional*, defaults to `self.crop_size`):
                Size of the output image after applying `center_crop`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
                `True`.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Default to `"pt"` for PyTorch tensors if unset.
                Fast image processors only support PyTorch tensors.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        validate_kwargs(captured_kwargs=kwargs.keys(), valid_processor_keys=self.valid_extra_kwargs)

        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        default_to_square = kwargs.pop(
            "default_to_square", self.default_to_square if self.default_to_square is not None else True
        )
        size = get_size_dict(size=size, default_to_square=default_to_square) if size is not None else None
        resample = resample if resample is not None else self.resample
        do_center_crop = do_center_crop if do_center_crop is not None else self.do_center_crop
        crop_size = crop_size if crop_size is not None else self.crop_size
        crop_size = get_size_dict(crop_size, param_name="crop_size") if crop_size is not None else None
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        device = kwargs.pop("device", None)

        images = self._prepare_input_images(
            images=images,
            do_convert_rgb=do_convert_rgb,
            device=device,
            input_data_format=input_data_format,
        )

        image_mean, image_std, size, crop_size, interpolation = self._prepare_process_arguments(
            device=images[0].device,
            do_resize=do_resize,
            size=size,
            resample=resample,
            crop_size=crop_size,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            return_tensors=return_tensors,
            data_format=data_format,
        )

        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images)
        resized_images_grouped = {}
        for shape, images in grouped_images.items():
            stacked_images = torch.stack(images, dim=0)
            if do_resize:
                stacked_images = self.resize(image=stacked_images, size=size, interpolation=interpolation)
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(resized_images)
        processed_images_grouped = {}
        for shape, images in grouped_images.items():
            stacked_images = torch.stack(images, dim=0)
            if do_center_crop:
                stacked_images = self.center_crop(stacked_images, crop_size)
            # Fused rescale and normalize
            if do_rescale and do_normalize:
                stacked_images = self.normalize(stacked_images.to(dtype=torch.float32), image_mean, image_std)
            elif do_rescale:
                stacked_images = stacked_images * rescale_factor
            elif do_normalize:
                stacked_images = self.normalize(stacked_images, image_mean, image_std)
            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)

    def to_dict(self):
        encoder_dict = super().to_dict()
        encoder_dict.pop("_valid_processor_keys", None)
        return encoder_dict


class LlavaPatchingMixin:
    def __init__(
        self,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        image_grid_pinpoints: List = None,
        resample: Optional[Union["PILImageResampling", "F.InterpolationMode"]] = None,
        do_center_crop: bool = None,
        crop_size: Dict[str, int] = None,
        do_rescale: bool = None,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: Optional[bool] = None,
        do_convert_rgb: bool = None,
        **kwargs,
    ) -> None:
        super().__init__(
            do_resize=do_resize,
            size=size,
            resample=resample,
            do_center_crop=do_center_crop,
            crop_size=crop_size,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_convert_rgb=do_convert_rgb,
            **kwargs,
        )
        self.image_grid_pinpoints = (
            image_grid_pinpoints if image_grid_pinpoints is not None else self.image_grid_pinpoints
        )
        self.do_pad = do_pad if do_pad is not None else self.do_pad

    def _prepare_images_structure(
        self,
        images: ImageInput,
    ) -> ImageInput:
        """
        Prepare the images structure for processing.

        Args:
            images (`ImageInput`):
                The input images to process.

        Returns:
            `ImageInput`: The images with a valid nesting.
        """
        return make_batched_images(images)

    def _resize_for_patching(
        self,
        image: "torch.Tensor",
        target_resolution: tuple,
        interpolation: "F.InterpolationMode",
        input_data_format: ChannelDimension,
    ) -> "torch.Tensor":
        """
        Resizes an image to a target resolution while maintaining aspect ratio.

        Args:
            image ("torch.Tensor"):
                The input image.
            target_resolution (tuple):
                The target resolution (height, width) of the image.
            interpolation (`InterpolationMode`):
                Resampling filter to use if resizing the image.
            input_data_format (`ChannelDimension` or `str`):
                The channel dimension format of the input image.

        Returns:
            "torch.Tensor": The resized and padded image.
        """
        new_height, new_width = get_patch_output_size(image, target_resolution, input_data_format)

        # Resize the image
        resized_image = F.resize(image, (new_height, new_width), interpolation=interpolation)

        return resized_image

    def _pad_for_patching(
        self, image: "torch.Tensor", target_resolution: tuple, input_data_format: ChannelDimension
    ) -> "torch.Tensor":
        """
        Pad an image to a target resolution while maintaining aspect ratio.
        """
        target_height, target_width = target_resolution
        new_height, new_width = get_patch_output_size(image, target_resolution, input_data_format)

        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2

        padded_image = F.pad(image, padding=[paste_x, paste_y, paste_x, paste_y])

        return padded_image

    def _get_image_patches(
        self,
        image: "torch.Tensor",
        grid_pinpoints,
        size: tuple,
        patch_size: int,
        interpolation: "F.InterpolationMode",
    ) -> List[np.array]:
        """
        Process an image with variable resolutions by dividing it into patches.

        Args:
            image ("torch.Tensor"):
                The input image to be processed.
            grid_pinpoints (List):
                A string representation of a list of possible resolutions.
            size (`tuple`):
                Size to resize the original image to.
            patch_size (`int`):
                Size of the patches to divide the image into.
            interpolation (`"InterpolationMode"`):
                Resampling filter to use if resizing the image.

        Returns:
            List["torch.Tensor"]: A list of NumPy arrays containing the processed image patches.
        """
        if not isinstance(grid_pinpoints, list):
            raise TypeError("grid_pinpoints must be a list of possible resolutions.")

        possible_resolutions = grid_pinpoints

        image_size = get_image_size(image, channel_dim=ChannelDimension.FIRST)
        best_resolution = select_best_resolution(image_size, possible_resolutions)
        resized_image = self._resize_for_patching(
            image, best_resolution, interpolation=interpolation, input_data_format=ChannelDimension.FIRST
        )
        padded_image = self._pad_for_patching(resized_image, best_resolution, input_data_format=ChannelDimension.FIRST)
        patches = divide_to_patches(padded_image, patch_size=patch_size)
        resized_original_image = F.resize(image, size=size, interpolation=interpolation)

        image_patches = [resized_original_image] + patches

        return image_patches

    def _pad_for_batching(
        self,
        pixel_values: List["torch.Tensor"],
    ) -> List["torch.Tensor"]:
        """
        Pads images on the `num_of_patches` dimension with zeros to form a batch of same number of patches.

        Args:
            pixel_values (`List[torch.Tensor]`):
                An array of pixel values of each images of shape (`batch_size`, `num_patches`, `image_in_3D`)

        Returns:
            List[`torch.Tensor`]: The padded images.
        """
        max_patch = max(len(x) for x in pixel_values)
        pixel_values = [
            torch.nn.functional.pad(image, pad=[0, 0, 0, 0, 0, 0, 0, max_patch - image.shape[0]])
            for image in pixel_values
        ]

        return pixel_values

    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        image_grid_pinpoints: List = None,
        resample: Optional[Union["PILImageResampling", "F.InterpolationMode"]] = None,
        do_center_crop: bool = None,
        crop_size: int = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: Optional[bool] = None,
        do_convert_rgb: bool = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Describes the maximum input dimensions to the model.
            resample (`PILImageResampling` or `InterpolationMode`, *optional*, defaults to self.resample):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
                has an effect if `do_resize` is set to `True`.
            do_center_crop (`bool`, *optional*, defaults to `self.do_center_crop`):
                Whether to center crop the image.
            crop_size (`Dict[str, int]`, *optional*, defaults to `self.crop_size`):
                Size of the output image after applying `center_crop`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
                `True`.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Default to `"pt"` for PyTorch tensors if unset.
                Fast image processors only support PyTorch tensors.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        validate_kwargs(captured_kwargs=kwargs.keys(), valid_processor_keys=self.valid_extra_kwargs)

        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        default_to_square = kwargs.pop(
            "default_to_square", self.default_to_square if self.default_to_square is not None else True
        )
        size = get_size_dict(size=size, default_to_square=default_to_square) if size is not None else None
        image_grid_pinpoints = image_grid_pinpoints if image_grid_pinpoints is not None else self.image_grid_pinpoints
        resample = resample if resample is not None else self.resample
        do_center_crop = do_center_crop if do_center_crop is not None else self.do_center_crop
        crop_size = crop_size if crop_size is not None else self.crop_size
        crop_size = get_size_dict(crop_size, param_name="crop_size") if crop_size is not None else None
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_pad = do_pad if do_pad is not None else self.do_pad
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        device = kwargs.pop("device", None)

        images = self._prepare_input_images(
            images=images,
            do_convert_rgb=do_convert_rgb,
            device=device,
            input_data_format=input_data_format,
        )

        image_mean, image_std, size, crop_size, interpolation = self._prepare_process_arguments(
            device=images[0].device,
            do_resize=do_resize,
            size=size,
            resample=resample,
            do_center_crop=do_center_crop,
            crop_size=crop_size,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_convert_rgb=do_convert_rgb,
            return_tensors=return_tensors,
            data_format=data_format,
            **kwargs,
        )

        processed_images = []
        image_sizes = []
        for image in images:
            size_tuple = (
                (size.height, size.width) if size.height and size.width else (size.shortest_edge, size.shortest_edge)
            )
            patch_size = (
                crop_size.height
                if crop_size is not None and crop_size.height
                else size.height
                if size.height
                else size.shortest_edge
            )
            image_patches = self._get_image_patches(
                image,
                image_grid_pinpoints,
                size=size_tuple,
                patch_size=patch_size,
                interpolation=interpolation,
            )

            # Group images by size for batched processing
            processed_image_patches_grouped = {}
            grouped_image_patches, grouped_image_patches_index = group_images_by_shape(image_patches)
            for shape, image_patches in grouped_image_patches.items():
                stacked_image_patches = torch.stack(image_patches, dim=0)
                if do_resize:
                    stacked_image_patches = self.resize(
                        image=stacked_image_patches,
                        size=size,
                        interpolation=interpolation,
                    )
                if do_center_crop:
                    stacked_image_patches = self.center_crop(stacked_image_patches, crop_size)
                # Fused rescale and normalize
                if do_rescale and do_normalize:
                    stacked_image_patches = self.normalize(
                        stacked_image_patches.to(dtype=torch.float32), image_mean, image_std
                    )
                elif do_rescale:
                    stacked_image_patches = stacked_image_patches * rescale_factor
                elif do_normalize:
                    stacked_image_patches = self.normalize(stacked_image_patches, image_mean, image_std)
                processed_image_patches_grouped[shape] = stacked_image_patches
            processed_image_patches = reorder_images(processed_image_patches_grouped, grouped_image_patches_index)
            processed_image_patches = (
                torch.stack(processed_image_patches, dim=0) if return_tensors else processed_image_patches
            )
            processed_images.append(processed_image_patches)
            image_sizes.append(get_image_size(image, input_data_format))

        if do_pad:
            processed_images = self._pad_for_batching(processed_images)
        processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images
        return BatchFeature(
            data={"pixel_values": processed_images, "image_sizes": image_sizes}, tensor_type=return_tensors
        )


class SemanticSegmentationMixin:
    def post_process_semantic_segmentation(self, outputs, target_sizes: List[Tuple] = None):
        """
        Converts the output of [`MobileNetV2ForSemanticSegmentation`] into semantic segmentation maps. Only supports PyTorch.

        Args:
            outputs ([`MobileNetV2ForSemanticSegmentation`]):
                Raw outputs of the model.
            target_sizes (`List[Tuple]` of length `batch_size`, *optional*):
                List of tuples corresponding to the requested final size (height, width) of each prediction. If unset,
                predictions will not be resized.

        Returns:
            semantic_segmentation: `List[torch.Tensor]` of length `batch_size`, where each item is a semantic
            segmentation map of shape (height, width) corresponding to the target_sizes entry (if `target_sizes` is
            specified). Each entry of each `torch.Tensor` correspond to a semantic class id.
        """
        logits = outputs.logits

        # Resize logits and compute semantic segmentation maps
        if target_sizes is not None:
            if len(logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

            # if is_torch_tensor(target_sizes):
            #     target_sizes = target_sizes.numpy()

            semantic_segmentation = []

            for idx in range(len(logits)):
                resized_logits = torch.nn.functional.interpolate(
                    logits[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
                )
                semantic_map = resized_logits[0].argmax(dim=0)
                semantic_segmentation.append(semantic_map)
        else:
            semantic_segmentation = logits.argmax(dim=1)
            semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]

        return semantic_segmentation
