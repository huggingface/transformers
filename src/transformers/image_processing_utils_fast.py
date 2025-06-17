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

from collections.abc import Iterable
from functools import lru_cache, partial
from typing import Any, Optional, TypedDict, Union

import numpy as np

from .image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
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
    auto_docstring,
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
else:
    pil_torch_interpolation_mapping = None

logger = logging.get_logger(__name__)


@lru_cache(maxsize=10)
def validate_fast_preprocess_arguments(
    do_rescale: Optional[bool] = None,
    rescale_factor: Optional[float] = None,
    do_normalize: Optional[bool] = None,
    image_mean: Optional[Union[float, list[float]]] = None,
    image_std: Optional[Union[float, list[float]]] = None,
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


def max_across_indices(values: Iterable[Any]) -> list[Any]:
    """
    Return the maximum value across all indices of an iterable of values.
    """
    return [max(values_i) for values_i in zip(*values)]


def get_max_height_width(images: list["torch.Tensor"]) -> tuple[int]:
    """
    Get the maximum height and width across all images in a batch.
    """

    _, max_height, max_width = max_across_indices([img.shape for img in images])

    return (max_height, max_width)


def divide_to_patches(
    image: Union[np.array, "torch.Tensor"], patch_size: int
) -> list[Union[np.array, "torch.Tensor"]]:
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
    size: Optional[dict[str, int]]
    default_to_square: Optional[bool]
    resample: Optional[Union["PILImageResampling", "F.InterpolationMode"]]
    do_center_crop: Optional[bool]
    crop_size: Optional[dict[str, int]]
    do_rescale: Optional[bool]
    rescale_factor: Optional[Union[int, float]]
    do_normalize: Optional[bool]
    image_mean: Optional[Union[float, list[float]]]
    image_std: Optional[Union[float, list[float]]]
    do_convert_rgb: Optional[bool]
    return_tensors: Optional[Union[str, TensorType]]
    data_format: Optional[ChannelDimension]
    input_data_format: Optional[Union[str, ChannelDimension]]
    device: Optional["torch.device"]


@auto_docstring
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

        # get valid kwargs names
        self._valid_kwargs_names = list(self.valid_kwargs.__annotations__.keys())

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
            interpolation (`InterpolationMode`, *optional*, defaults to `InterpolationMode.BILINEAR`):
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
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
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
        image_mean: Union[float, list[float]],
        image_std: Union[float, list[float]],
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
        size: dict[str, int],
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
        do_convert_rgb: Optional[bool] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        device: Optional["torch.device"] = None,
    ) -> list["torch.Tensor"]:
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

    def __call__(self, images: ImageInput, *args, **kwargs: Unpack[DefaultFastImageProcessorKwargs]) -> BatchFeature:
        return self.preprocess(images, *args, **kwargs)

    @auto_docstring
    def preprocess(self, images: ImageInput, *args, **kwargs: Unpack[DefaultFastImageProcessorKwargs]) -> BatchFeature:
        # args are not validated, but their order in the `preprocess` and `_preprocess` signatures must be the same
        validate_kwargs(captured_kwargs=kwargs.keys(), valid_processor_keys=self._valid_kwargs_names)
        # Set default kwargs from self. This ensures that if a kwarg is not provided
        # by the user, it gets its default value from the instance, or is set to None.
        for kwarg_name in self._valid_kwargs_names:
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

        # Check if resample is an int before checking if it's an instance of PILImageResampling
        # because if pillow < 9.1.0, resample is an int and PILImageResampling is a module.
        # Checking PILImageResampling will fail with error `TypeError: isinstance() arg 2 must be a type or tuple of types`.
        kwargs["interpolation"] = (
            pil_torch_interpolation_mapping[resample] if isinstance(resample, (int, PILImageResampling)) else resample
        )

        # Pop kwargs that are not needed in _preprocess
        kwargs.pop("default_to_square")
        kwargs.pop("data_format")

        return self._preprocess(images, *args, **kwargs)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"],
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
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
        encoder_dict.pop("_valid_kwargs_names", None)
        return encoder_dict


class SemanticSegmentationMixin:
    def post_process_semantic_segmentation(self, outputs, target_sizes: Optional[list[tuple]] = None):
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
