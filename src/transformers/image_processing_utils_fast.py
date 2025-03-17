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

import inspect
from functools import lru_cache, partial
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypedDict, Union

import numpy as np

from .image_processing_utils import (
    BaseImageProcessor,
    BatchFeature,
    get_size_dict,
)
from .image_transforms import (
    convert_to_rgb,
    get_resize_output_image_size,
    get_size_with_aspect_ratio,
    group_images_by_shape,
    reorder_images,
)
from .image_utils import (
    ChannelDimension,
    ImageInput,
    ImageType,
    SizeDict,
    get_image_size,
    get_image_size_for_max_height_width,
    get_image_type,
    infer_channel_dimension_format,
    make_flat_list_of_images,
    validate_kwargs,
    validate_preprocess_arguments,
)
from .processing_utils import Unpack
from .utils import (
    TensorType,
    add_start_docstrings,
    is_torch_available,
    is_torchvision_available,
    is_torchvision_v2_available,
    is_vision_available,
    logging,
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

logger = logging.get_logger(__name__)


@lru_cache(maxsize=10)
def validate_fast_preprocess_arguments(
    do_rescale: Optional[bool] = None,
    rescale_factor: Optional[float] = None,
    do_normalize: Optional[bool] = None,
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
    do_pad: Optional[bool] = None,
    size_divisibility: Optional[int] = None,
    do_center_crop: Optional[bool] = None,
    crop_size: Optional[SizeDict] = None,
    do_resize: Optional[bool] = None,
    size: Optional[SizeDict] = None,
    resample: Optional["PILImageResampling"] = None,
    return_tensors: Optional[Union[str, TensorType]] = None,
    data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
):
    """
    Checks validity of typically used arguments in an `ImageProcessorFast` `preprocess` method.
    Raises `ValueError` if arguments incompatibility is caught.
    """
    validate_preprocess_arguments(
        do_rescale=do_rescale,
        rescale_factor=rescale_factor,
        do_normalize=do_normalize,
        image_mean=image_mean,
        image_std=image_std,
        do_pad=do_pad,
        size_divisibility=size_divisibility,
        do_center_crop=do_center_crop,
        crop_size=crop_size,
        do_resize=do_resize,
        size=size,
        resample=resample,
    )
    # Extra checks for ImageProcessorFast
    if return_tensors is not None and return_tensors != "pt":
        raise ValueError("Only returning PyTorch tensors is currently supported.")

    if data_format != ChannelDimension.FIRST:
        raise ValueError("Only channel first data format is currently supported.")


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


class DefaultFastImageProcessorKwargs(TypedDict, total=False):
    do_resize: Optional[bool]
    size: Optional[Dict[str, int]]
    default_to_square: Optional[bool]
    resample: Optional[Union["PILImageResampling", "F.InterpolationMode"]]
    do_center_crop: Optional[bool]
    crop_size: Optional[Dict[str, int]]
    do_rescale: Optional[bool]
    rescale_factor: Optional[Union[int, float]]
    do_normalize: Optional[bool]
    image_mean: Optional[Union[float, List[float]]]
    image_std: Optional[Union[float, List[float]]]
    do_convert_rgb: Optional[bool]
    return_tensors: Optional[Union[str, TensorType]]
    data_format: Optional[ChannelDimension]
    input_data_format: Optional[Union[str, ChannelDimension]]
    device: Optional["torch.device"]


BASE_IMAGE_PROCESSOR_FAST_DOCSTRING = r"""
        do_resize (`bool`, *optional*, defaults to `self.do_resize`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by the
            `do_resize` parameter in the `preprocess` method.
        size (`dict`, *optional*, defaults to `self.size`):
            Size of the output image after resizing. Can be overridden by the `size` parameter in the `preprocess`
            method.
        default_to_square (`bool`, *optional*, defaults to `self.default_to_square`):
            Whether to default to a square image when resizing, if size is an int.
        resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
            Resampling filter to use if resizing the image. Only has an effect if `do_resize` is set to `True`. Can be
            overridden by the `resample` parameter in the `preprocess` method.
        do_center_crop (`bool`, *optional*, defaults to `self.do_center_crop`):
            Whether to center crop the image to the specified `crop_size`. Can be overridden by `do_center_crop` in the
            `preprocess` method.
        crop_size (`Dict[str, int]` *optional*, defaults to `self.crop_size`):
            Size of the output image after applying `center_crop`. Can be overridden by `crop_size` in the `preprocess`
            method.
        do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the
            `do_rescale` parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `self.rescale_factor`):
            Scale factor to use if rescaling the image. Only has an effect if `do_rescale` is set to `True`. Can be
            overridden by the `rescale_factor` parameter in the `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method. Can be overridden by the `do_normalize` parameter in the `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
            overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
            Whether to convert the image to RGB.
        return_tensors (`str` or `TensorType`, *optional*, defaults to `self.return_tensors`):
            Returns stacked tensors if set to `pt, otherwise returns a list of tensors.
        data_format (`ChannelDimension` or `str`, *optional*, defaults to `self.data_format`):
            Only `ChannelDimension.FIRST` is supported. Added for compatibility with slow processors.
        input_data_format (`ChannelDimension` or `str`, *optional*, defaults to `self.input_data_format`):
            The channel dimension format for the input image. If unset, the channel dimension format is inferred
            from the input image. Can be one of:
            - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
            - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        device (`torch.device`, *optional*, defaults to `self.device`):
            The device to process the images on. If unset, the device is inferred from the input images."""

BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS = r"""
        images (`ImageInput`):
            Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
            passing in images with pixel values between 0 and 1, set `do_rescale=False`.
        do_resize (`bool`, *optional*, defaults to `self.do_resize`):
            Whether to resize the image.
        size (`Dict[str, int]`, *optional*, defaults to `self.size`):
            Describes the maximum input dimensions to the model.
        resample (`PILImageResampling` or `InterpolationMode`, *optional*, defaults to `self.resample`):
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
        return_tensors (`str` or `TensorType`, *optional*, defaults to `self.return_tensors`):
            Returns stacked tensors if set to `pt, otherwise returns a list of tensors.
        data_format (`ChannelDimension` or `str`, *optional*, defaults to `self.data_format`):
            Only `ChannelDimension.FIRST` is supported. Added for compatibility with slow processors.
        input_data_format (`ChannelDimension` or `str`, *optional*, defaults to `self.input_data_format`):
            The channel dimension format for the input image. If unset, the channel dimension format is inferred
            from the input image. Can be one of:
            - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
            - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        device (`torch.device`, *optional*, defaults to `self.device`):
            The device to process the images on. If unset, the device is inferred from the input images."""


def get_kwargs_dict_from_docstring(docstring, obj):
    lines = docstring.split("\n")
    i = 0
    indent_kwarg = " " * 8
    indent_desc = " " * 12
    current_kwarg = None
    kwargs_dict = {}
    while i < len(lines):
        line = lines[i]
        if line.startswith(indent_desc) and line.strip() != "":
            kwargs_dict[current_kwarg]["desc"].append(line.strip())
        elif line.startswith(indent_kwarg) and line.strip() != "":
            current_kwarg = line.strip().split(" ")[0]
            header = " ".join(line.strip().split(" ")[1:])

            default = header.split("defaults to ")[-1][:-2]
            default_name = default.strip("`").split("self.")[-1]
            if hasattr(obj, default_name):
                header_replaced = header.replace(default, f"`{repr(getattr(obj, default_name))}`")
            else:
                header_replaced = header

            kwargs_dict[current_kwarg] = {
                "header_replaced": header_replaced,
                "header": header,  # also keep the original header for the preprocess method
                "desc": [],
            }
        elif line.strip() != "":
            raise ValueError(f"Unexpected line in docstring: {line}. Check if the indentation is correct.")

        i += 1
    return kwargs_dict


def customize_docstrings(
    init_header: str, custom_docstring: str = "", custom_preprocess_docstring: Optional[str] = None
):
    """
    Decorator to customize the docstrings of the `__init__` and `preprocess` methods of a class.

    Args:
        init_header (`str`):
            The header to use for the `__init__` method docstring.
        custom_docstring (`str`, *optional*, defaults to `""`):
            The custom docstring to use for the `__init__` method. One can provide only the docs for the arguments that
            need to be customized. The rest of the docstring is generated from the default docstring.
        custom_preprocess_docstring (`str`, *optional*):
            The custom docstring to use for the `preprocess` method. If not provided, the `custom_docstring` is used
            for the `preprocess` method as well.
    """

    def docstring_decorator(obj):
        # Parse the base docstring, replace the defaults with the actual values from the object (for the init method)
        base_init_kwargs = get_kwargs_dict_from_docstring(BASE_IMAGE_PROCESSOR_FAST_DOCSTRING, obj)
        base_preprocess_kwargs = get_kwargs_dict_from_docstring(BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS, obj)

        # Parse the given custom docstring and replace the defaults with the actual values from the object (for the init method)
        custom_kwargs = get_kwargs_dict_from_docstring(custom_docstring, obj)
        # If a custom_preprocess_docstring is not provided, use the given custom_docstring for the preprocess method as well
        if custom_preprocess_docstring:
            custom_preprocess_kwargs = get_kwargs_dict_from_docstring(custom_preprocess_docstring, obj)
        else:
            custom_preprocess_kwargs = custom_kwargs

        # Remove the doc for unused kwargs
        unused_kwargs = getattr(obj, "unused_kwargs", [])
        for kwarg in unused_kwargs:
            base_init_kwargs.pop(kwarg, None)
            base_preprocess_kwargs.pop(kwarg, None)

        # Merge the base and custom kwargs, putting the custom ones first
        all_init_kwargs = {}
        for kwarg, values in custom_kwargs.items():
            if kwarg not in base_init_kwargs:
                all_init_kwargs[kwarg] = values
        for kwarg, values in base_init_kwargs.items():
            # if custom_kwargs overrides the doc of a base kwarg, use the custom one
            if kwarg in custom_kwargs:
                all_init_kwargs[kwarg] = custom_kwargs[kwarg]
            else:
                all_init_kwargs[kwarg] = values

        # Rebuild the docstring with the merged kwargs and the given init_header
        formatted_header = "\n    " + init_header.strip()
        docstring = formatted_header + "\n\n" + "    Args:\n"
        for kwarg, values in all_init_kwargs.items():
            docstring += f"        {kwarg} {values['header_replaced']}\n"
            for desc in values["desc"]:
                docstring += f"            {desc}\n"
        obj.__doc__ = docstring

        # Merge the base and custom kwargs, first putting the args of the preprocess method (usually just `images`)
        # then the custom ones, then the base ones
        all_preprocess_kwargs = {}
        preprocess_args = inspect.signature(obj.preprocess).parameters
        for arg_name in preprocess_args:
            if arg_name == "self" or arg_name == "kwargs":
                continue
            # if custom_kwargs overrides the doc of an arg, use the custom one
            if arg_name in custom_preprocess_kwargs:
                all_preprocess_kwargs[arg_name] = custom_preprocess_kwargs[arg_name]
            elif arg_name in base_preprocess_kwargs:
                all_preprocess_kwargs[arg_name] = base_preprocess_kwargs[arg_name]
        for kwarg, values in custom_preprocess_kwargs.items():
            if kwarg not in base_preprocess_kwargs:
                all_preprocess_kwargs[kwarg] = values
        for kwarg, values in base_preprocess_kwargs.items():
            # if custom_kwargs overrides the doc of a base kwarg, use the custom one
            if kwarg in custom_preprocess_kwargs:
                all_preprocess_kwargs[kwarg] = custom_preprocess_kwargs[kwarg]
            else:
                all_preprocess_kwargs[kwarg] = values

        # Rebuild the docstring with the merged kwargs
        preprocess_header = "\n    Preprocess an image or batch of images."
        docstring = preprocess_header + "\n\n" + "    Args:\n"
        for kwarg, values in all_preprocess_kwargs.items():
            docstring += f"        {kwarg} {values['header']}\n"
            for desc in values["desc"]:
                docstring += f"            {desc}\n"

        obj.preprocess.__doc__ = docstring

        return obj

    return docstring_decorator


@add_start_docstrings(
    "Constructs a fast base image processor.",
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
)
class BaseImageProcessorFast(BaseImageProcessor):
    resample = None
    image_mean = None
    image_std = None
    size = None
    default_to_square = True
    crop_size = None
    do_resize = None
    do_center_crop = None
    do_rescale = None
    rescale_factor = 1 / 255
    do_normalize = None
    do_convert_rgb = None
    return_tensors = None
    data_format = ChannelDimension.FIRST
    input_data_format = None
    device = None
    model_input_names = ["pixel_values"]
    valid_kwargs = DefaultFastImageProcessorKwargs
    unused_kwargs = None

    def __init__(
        self,
        **kwargs: Unpack[DefaultFastImageProcessorKwargs],
    ) -> None:
        super().__init__(**kwargs)
        kwargs = self.filter_out_unused_kwargs(kwargs)
        size = kwargs.pop("size", self.size)
        self.size = (
            get_size_dict(size=size, default_to_square=kwargs.pop("default_to_square", self.default_to_square))
            if size is not None
            else None
        )
        crop_size = kwargs.pop("crop_size", self.crop_size)
        self.crop_size = get_size_dict(crop_size, param_name="crop_size") if crop_size is not None else None
        for key in self.valid_kwargs.__annotations__.keys():
            kwarg = kwargs.pop(key, None)
            if kwarg is not None:
                setattr(self, key, kwarg)
            else:
                setattr(self, key, getattr(self, key, None))

    def resize(
        self,
        image: "torch.Tensor",
        size: SizeDict,
        interpolation: "F.InterpolationMode" = None,
        antialias: bool = True,
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
        return F.resize(image, new_size, interpolation=interpolation, antialias=antialias)

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

    @lru_cache(maxsize=10)
    def _fuse_mean_std_and_rescale_factor(
        self,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        device: Optional["torch.device"] = None,
    ) -> tuple:
        if do_rescale and do_normalize:
            # Fused rescale and normalize
            image_mean = torch.tensor(image_mean, device=device) * (1.0 / rescale_factor)
            image_std = torch.tensor(image_std, device=device) * (1.0 / rescale_factor)
            do_rescale = False
        return image_mean, image_std, do_rescale

    def rescale_and_normalize(
        self,
        images: "torch.Tensor",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Union[float, List[float]],
        image_std: Union[float, List[float]],
    ) -> "torch.Tensor":
        """
        Rescale and normalize images.
        """
        image_mean, image_std, do_rescale = self._fuse_mean_std_and_rescale_factor(
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            device=images.device,
        )
        # if/elif as we use fused rescale and normalize if both are set to True
        if do_normalize:
            images = self.normalize(images.to(dtype=torch.float32), image_mean, image_std)
        elif do_rescale:
            images = self.rescale(images, rescale_factor)

        return images

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

    def filter_out_unused_kwargs(self, kwargs: dict):
        """
        Filter out the unused kwargs from the kwargs dictionary.
        """
        if self.unused_kwargs is None:
            return kwargs

        for kwarg_name in self.unused_kwargs:
            if kwarg_name in kwargs:
                logger.warning_once(f"This processor does not use the `{kwarg_name}` parameter. It will be ignored.")
                kwargs.pop(kwarg_name)
        return kwargs

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
        return make_flat_list_of_images(images)

    def _process_image(
        self,
        image: ImageInput,
        do_convert_rgb: Optional[bool] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        device: Optional["torch.device"] = None,
    ) -> "torch.Tensor":
        image_type = get_image_type(image)
        if image_type not in [ImageType.PIL, ImageType.TORCH, ImageType.NUMPY]:
            raise ValueError(f"Unsupported input image type {image_type}")

        if do_convert_rgb:
            image = self.convert_to_rgb(image)

        if image_type == ImageType.PIL:
            image = F.pil_to_tensor(image)
        elif image_type == ImageType.NUMPY:
            # not using F.to_tensor as it doesn't handle (C, H, W) numpy arrays
            image = torch.from_numpy(image).contiguous()

        # Infer the channel dimension format if not provided
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)

        if input_data_format == ChannelDimension.LAST:
            # We force the channel dimension to be first for torch tensors as this is what torchvision expects.
            image = image.permute(2, 0, 1).contiguous()

        # Now that we have torch tensors, we can move them to the right device
        if device is not None:
            image = image.to(device)

        return image

    def _prepare_input_images(
        self,
        images: ImageInput,
        do_convert_rgb: bool = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        device: Optional["torch.device"] = None,
    ) -> List["torch.Tensor"]:
        """
        Prepare the input images for processing.
        """
        images = self._prepare_images_structure(images)
        process_image_fn = partial(
            self._process_image,
            do_convert_rgb=do_convert_rgb,
            input_data_format=input_data_format,
            device=device,
        )
        # todo: yoni - check if we can parallelize this efficiently
        processed_images = []
        for image in images:
            processed_images.append(process_image_fn(image))

        return processed_images

    def _further_process_kwargs(
        self,
        size: Optional[SizeDict] = None,
        crop_size: Optional[SizeDict] = None,
        default_to_square: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
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
        if crop_size is not None:
            crop_size = SizeDict(**get_size_dict(crop_size, param_name="crop_size"))
        if isinstance(image_mean, list):
            image_mean = tuple(image_mean)
        if isinstance(image_std, list):
            image_std = tuple(image_std)
        if data_format is None:
            data_format = ChannelDimension.FIRST

        kwargs["size"] = size
        kwargs["crop_size"] = crop_size
        kwargs["default_to_square"] = default_to_square
        kwargs["image_mean"] = image_mean
        kwargs["image_std"] = image_std
        kwargs["data_format"] = data_format

        return kwargs

    def _validate_preprocess_kwargs(
        self,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, tuple[float]]] = None,
        image_std: Optional[Union[float, tuple[float]]] = None,
        do_resize: Optional[bool] = None,
        size: Optional[SizeDict] = None,
        do_center_crop: Optional[bool] = None,
        crop_size: Optional[SizeDict] = None,
        resample: Optional[Union["PILImageResampling", "F.InterpolationMode"]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = None,
        **kwargs,
    ):
        """
        validate the kwargs for the preprocess method.
        """
        validate_fast_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            do_center_crop=do_center_crop,
            crop_size=crop_size,
            resample=resample,
            return_tensors=return_tensors,
            data_format=data_format,
        )

    @add_start_docstrings(BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS)
    def preprocess(self, images: ImageInput, **kwargs: Unpack[DefaultFastImageProcessorKwargs]) -> BatchFeature:
        validate_kwargs(captured_kwargs=kwargs.keys(), valid_processor_keys=self.valid_kwargs.__annotations__.keys())
        # Set default kwargs from self. This ensures that if a kwarg is not provided
        # by the user, it gets its default value from the instance, or is set to None.
        for kwarg_name in self.valid_kwargs.__annotations__:
            kwargs.setdefault(kwarg_name, getattr(self, kwarg_name, None))

        # Extract parameters that are only used for preparing the input images
        do_convert_rgb = kwargs.pop("do_convert_rgb")
        input_data_format = kwargs.pop("input_data_format")
        device = kwargs.pop("device")
        # Prepare input images
        images = self._prepare_input_images(
            images=images, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format, device=device
        )

        # Update kwargs that need further processing before being validated
        kwargs = self._further_process_kwargs(**kwargs)

        # Validate kwargs
        self._validate_preprocess_kwargs(**kwargs)

        # torch resize uses interpolation instead of resample
        resample = kwargs.pop("resample")
        kwargs["interpolation"] = (
            pil_torch_interpolation_mapping[resample] if isinstance(resample, (PILImageResampling, int)) else resample
        )

        # Pop kwargs that are not needed in _preprocess
        kwargs.pop("default_to_square")
        kwargs.pop("data_format")

        return self._preprocess(images=images, **kwargs)

    def _preprocess(
        self,
        images: List["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"],
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, List[float]]],
        image_std: Optional[Union[float, List[float]]],
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs,
    ) -> BatchFeature:
        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(image=stacked_images, size=size, interpolation=interpolation)
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(resized_images)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_center_crop:
                stacked_images = self.center_crop(stacked_images, crop_size)
            # Fused rescale and normalize
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)

    def to_dict(self):
        encoder_dict = super().to_dict()
        encoder_dict.pop("_valid_processor_keys", None)
        return encoder_dict


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
