# Copyright 2022 The HuggingFace Inc. team.
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

import math
from collections.abc import Iterable
from copy import deepcopy
from functools import partial
from typing import Any, Union

from huggingface_hub.dataclasses import validate_typed_dict

from .image_processing_base import BatchFeature, ImageProcessingMixin
from .image_utils import (
    ImageInput,
    SizeDict,
    get_image_size,
    make_flat_list_of_images,
    validate_preprocess_arguments,
)
from .processing_utils import ImagesKwargs, Unpack
from .utils import (
    auto_docstring,
    is_torchvision_available,
    is_vision_available,
    logging,
)


if is_vision_available():
    from .image_utils import PILImageResampling


if is_torchvision_available():
    from torchvision.transforms.v2 import functional as tvF


logger = logging.get_logger(__name__)


INIT_SERVICE_KWARGS = [
    "processor_class",
    "image_processor_type",
]


@auto_docstring
class BaseImageProcessor(ImageProcessingMixin):
    r"""
    Base class for image processors with an inheritance-based backend architecture.

    This class defines the preprocessing pipeline: kwargs validation, input preparation, and dispatching to the
    backend's `_preprocess` method. Backend subclasses (`TorchVisionBackend`, `PilBackend`) inherit from this class
    and implement the actual image operations (resize, crop, rescale, normalize, etc.). Model-specific image
    processors then inherit from the appropriate backend class.

    Architecture Overview
    ---------------------

    The class hierarchy is:

        BaseImageProcessor (this class)
        ├── TorchVisionBackend    (GPU-accelerated, torch.Tensor)
        │   └── ModelImageProcessor (e.g. LlavaNextImageProcessor)
        └── PilBackend            (portable CPU, np.ndarray)
            └── ModelImageProcessorPil (e.g. CLIPImageProcessorPil)

    The preprocessing flow is:

        __call__() → preprocess() → _preprocess_image_like_inputs() → _prepare_image_like_inputs()
                                                                       (calls process_image per image)
                                                                     → _preprocess()
                                                                       (batch operations: resize, crop, etc.)

    - `process_image`: Implemented by backends. Converts a single raw input (PIL, NumPy, or Tensor) to the
      backend's working format (torch.Tensor or np.ndarray), handles RGB conversion and channel reordering.
    - `_preprocess`: Implemented by backends. Performs the actual batch processing (resize, center crop, rescale,
      normalize, pad) and returns a `BatchFeature`.

    Basic Implementation
    --------------------

    For processors that only need standard operations (resize, center crop, rescale, normalize), inherit from
    a backend and define class attributes:

        from transformers.image_processing_backends import PilBackend

        class MyImageProcessor(PilBackend):
            resample = PILImageResampling.BILINEAR
            image_mean = IMAGENET_DEFAULT_MEAN
            image_std = IMAGENET_DEFAULT_STD
            size = {"height": 224, "width": 224}
            do_resize = True
            do_rescale = True
            do_normalize = True

    The backend's `_preprocess` method handles the standard pipeline automatically.

    Custom Processing
    -----------------

    For processors that need custom logic (e.g., patch-based processing, multiple input types), override
    `_preprocess` in your model-specific processor. The `_preprocess` method receives already-prepared images
    (converted to the backend format with channels-first ordering) and performs the actual processing:

        class MyImageProcessor(TorchVisionBackend):
            def _preprocess(self, images, do_resize, size, **kwargs):
                # Custom processing logic
                patches = self._create_patches(images)
                # Use backend methods for standard operations
                for patch in patches:
                    patch = self.resize(patch, size=size)
                    patch = self.normalize(patch, mean=image_mean, std=image_std)
                return BatchFeature(data={"pixel_values": patches})

    For processors handling multiple input types (e.g., images + segmentation maps), override
    `_preprocess_image_like_inputs`:

        def _preprocess_image_like_inputs(
            self,
            images: ImageInput,
            segmentation_maps: ImageInput | None = None,
            **kwargs,
        ) -> BatchFeature:
            images = self._prepare_image_like_inputs(images, **kwargs)
            batch_feature = self._preprocess(images, **kwargs)

            if segmentation_maps is not None:
                maps = self._prepare_image_like_inputs(segmentation_maps, **kwargs)
                batch_feature["labels"] = self._preprocess(maps, **kwargs).pixel_values

            return batch_feature

    Extending Backend Behavior
    --------------------------

    To customize operations for a specific backend, subclass the backend and override its methods:

        from transformers.image_processing_backends import TorchVisionBackend, PilBackend

        class MyTorchVisionProcessor(TorchVisionBackend):
            def resize(self, image, size, **kwargs):
                # Custom resize logic for torchvision
                return super().resize(image, size, **kwargs)

        class MyPilProcessor(PilBackend):
            def resize(self, image, size, **kwargs):
                # Custom resize logic for PIL
                return super().resize(image, size, **kwargs)

    Custom Parameters
    -----------------

    To add parameters beyond `ImagesKwargs`, create a custom kwargs class and set it as `valid_kwargs`:

        class MyImageProcessorKwargs(ImagesKwargs):
            custom_param: int | None = None

        class MyImageProcessor(TorchVisionBackend):
            valid_kwargs = MyImageProcessorKwargs
            custom_param = 10  # default value

    Key Notes
    ---------

    - Backend selection is done at the class level: inherit from `TorchVisionBackend` or `PilBackend`
    - Backends receive images as `torch.Tensor` (TorchVision) or `np.ndarray` (PIL), always channels-first
    - All images have channel dimension first during processing, regardless of backend
    - Arguments not provided by users default to class attribute values
    - Backend classes encapsulate backend-specific logic (resize, normalize, etc.) and can be overridden
    """

    valid_kwargs = ImagesKwargs

    default_to_square = True
    rescale_factor = 1 / 255
    model_input_names = ["pixel_values"]

    def __init__(self, **kwargs: Unpack[ImagesKwargs]):
        super().__init__(**kwargs)

        kwargs = self._standardize_kwargs(**kwargs)
        for key in self.valid_kwargs.__annotations__:
            kwarg = kwargs.pop(key, None)
            if kwarg is not None:
                setattr(self, key, kwarg)
            else:
                setattr(self, key, deepcopy(getattr(self, key, None)))

        # get valid kwargs names
        self._valid_kwargs_names = list(self.valid_kwargs.__annotations__.keys())

    def __call__(self, images: ImageInput, *args, **kwargs: Unpack[ImagesKwargs]) -> BatchFeature:
        """Preprocess an image or a batch of images."""
        return self.preprocess(images, *args, **kwargs)

    def process_image(self, *args, **kwargs):
        """
        Process a single raw image into the backend's working format.

        Implemented by backend subclasses (`TorchVisionBackend`, `PilBackend`). Converts a raw input
        (PIL Image, NumPy array, or torch Tensor) to the backend's internal format (`torch.Tensor` for
        TorchVision, `np.ndarray` for PIL), handles RGB conversion and ensures channels-first ordering.
        """
        raise NotImplementedError

    def _preprocess(self, *args, **kwargs):
        """
        Perform the actual batch image preprocessing (resize, center crop, rescale, normalize, pad).

        Implemented by backend subclasses (`TorchVisionBackend`, `PilBackend`). Receives a list of
        already-prepared images (in the backend's format, channels-first) and applies the configured
        preprocessing operations. Returns a `BatchFeature` with the processed pixel values.

        Model-specific processors can override this method to implement custom preprocessing logic
        (e.g., patch-based processing in LLaVA-NeXT).
        """
        raise NotImplementedError

    def _prepare_images_structure(
        self,
        images: ImageInput,
        expected_ndims: int = 3,
    ) -> ImageInput:
        """
        Prepare the images structure for processing.

        Args:
            images (`ImageInput`):
                The input images to process.

        Returns:
            `ImageInput`: The images with a valid nesting.
        """
        images = self.fetch_images(images)
        return make_flat_list_of_images(images, expected_ndims=expected_ndims)

    def _prepare_image_like_inputs(
        self,
        images: ImageInput,
        *args,
        expected_ndims: int = 3,
        **kwargs: Unpack[ImagesKwargs],
    ) -> list[Any]:
        """
        Prepare image-like inputs for processing by converting each image via `process_image`.

        Flattens the input structure and applies `process_image` (implemented by the backend) to each
        individual image, converting raw inputs (PIL, NumPy, Tensor) into the backend's working format
        with channels-first ordering.

        Args:
            images (`ImageInput`):
                The image-like inputs to process.
            expected_ndims (`int`, *optional*, defaults to 3):
                The expected number of dimensions for the images.

        Returns:
            `list[torch.Tensor]` or `list[np.ndarray]`: The prepared images in the backend's format,
            with channels-first ordering.
        """
        images = self._prepare_images_structure(images, expected_ndims=expected_ndims)

        process_image_partial = partial(self.process_image, *args, **kwargs)

        has_nested_structure = len(images) > 0 and isinstance(images[0], list | tuple)

        if has_nested_structure:
            processed_images = [[process_image_partial(img) for img in nested_list] for nested_list in images]
        else:
            processed_images = [process_image_partial(img) for img in images]

        return processed_images

    def _preprocess_image_like_inputs(
        self,
        images: ImageInput,
        *args,
        **kwargs: Unpack[ImagesKwargs],
    ) -> BatchFeature:
        """
        Preprocess image-like inputs by preparing them and dispatching to `_preprocess`.

        This method first calls `_prepare_image_like_inputs` to convert raw inputs into the backend's
        format, then calls `_preprocess` for the actual batch processing. Override this method in
        model-specific processors that need to handle multiple image-like input types (e.g., images
        and segmentation maps) or need custom orchestration of the preprocessing pipeline.
        """
        images = self._prepare_image_like_inputs(images, *args, **kwargs)
        return self._preprocess(images, *args, **kwargs)

    def _standardize_kwargs(
        self,
        size: int | Iterable[int] | dict[str, int] | SizeDict | None = None,
        crop_size: int | Iterable[int] | dict[str, int] | SizeDict | None = None,
        pad_size: int | Iterable[int] | dict[str, int] | SizeDict | None = None,
        default_to_square: bool | None = None,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        **kwargs,
    ) -> dict:
        """
        Standardize kwargs to canonical format before validation.
        Can be overridden by subclasses to customize the processing of kwargs.
        """
        if kwargs is None:
            kwargs = {}
        if size is not None and not isinstance(size, SizeDict):
            size = SizeDict(**get_size_dict(size=size, default_to_square=default_to_square))
        if crop_size is not None and not isinstance(crop_size, SizeDict):
            crop_size = SizeDict(**get_size_dict(crop_size, param_name="crop_size"))
        if pad_size is not None and not isinstance(pad_size, SizeDict):
            pad_size = SizeDict(**get_size_dict(size=pad_size, param_name="pad_size"))
        if isinstance(image_mean, list):
            image_mean = tuple(image_mean)
        if isinstance(image_std, list):
            image_std = tuple(image_std)

        kwargs["size"] = size
        kwargs["crop_size"] = crop_size
        kwargs["pad_size"] = pad_size
        kwargs["image_mean"] = image_mean
        kwargs["image_std"] = image_std

        return kwargs

    def _validate_preprocess_kwargs(
        self,
        do_rescale: bool | None = None,
        rescale_factor: float | None = None,
        do_normalize: bool | None = None,
        image_mean: float | tuple[float] | None = None,
        image_std: float | tuple[float] | None = None,
        do_resize: bool | None = None,
        size: SizeDict | None = None,
        do_center_crop: bool | None = None,
        crop_size: SizeDict | None = None,
        resample: Union["PILImageResampling", "tvF.InterpolationMode", int] | None = None,
        **kwargs,
    ):
        """
        Validate the kwargs for the preprocess method.
        """
        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_center_crop=do_center_crop,
            crop_size=crop_size,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

    @auto_docstring
    def preprocess(self, images: ImageInput, *args, **kwargs: Unpack[ImagesKwargs]) -> BatchFeature:
        """
        Preprocess an image or a batch of images.
        """
        # Perform type validation on received kwargs
        validate_typed_dict(self.valid_kwargs, kwargs)

        # Set default kwargs from self
        for kwarg_name in self._valid_kwargs_names:
            kwargs.setdefault(kwarg_name, getattr(self, kwarg_name, None))

        # Update kwargs that need further processing before being validated
        kwargs = self._standardize_kwargs(**kwargs)

        # Validate kwargs
        self._validate_preprocess_kwargs(**kwargs)

        return self._preprocess_image_like_inputs(images, *args, **kwargs)

    def to_dict(self):
        processor_dict = super().to_dict()

        # Filter out None values that are class defaults
        filtered_dict = {}
        for key, value in processor_dict.items():
            if isinstance(value, SizeDict):
                value = dict(value)
            if value is None:
                class_default = getattr(type(self), key, "NOT_FOUND")
                # Keep None if user explicitly set it (class default is non-None)
                if class_default != "NOT_FOUND" and class_default is not None:
                    filtered_dict[key] = value
            else:
                filtered_dict[key] = value

        filtered_dict.pop("_valid_processor_keys", None)
        filtered_dict.pop("_valid_kwargs_names", None)
        return filtered_dict


VALID_SIZE_DICT_KEYS = (
    {"height", "width"},
    {"shortest_edge"},
    {"shortest_edge", "longest_edge"},
    {"longest_edge"},
    {"max_height", "max_width"},
)


def is_valid_size_dict(size_dict):
    if not isinstance(size_dict, dict):
        return False

    size_dict_keys = set(size_dict.keys())
    for allowed_keys in VALID_SIZE_DICT_KEYS:
        if size_dict_keys == allowed_keys:
            return True
    return False


def convert_to_size_dict(
    size, max_size: int | None = None, default_to_square: bool = True, height_width_order: bool = True
):
    # By default, if size is an int we assume it represents a tuple of (size, size).
    if isinstance(size, int) and default_to_square:
        if max_size is not None:
            raise ValueError("Cannot specify both size as an int, with default_to_square=True and max_size")
        return {"height": size, "width": size}
    # In other configs, if size is an int and default_to_square is False, size represents the length of
    # the shortest edge after resizing.
    elif isinstance(size, int) and not default_to_square:
        size_dict = {"shortest_edge": size}
        if max_size is not None:
            size_dict["longest_edge"] = max_size
        return size_dict
    # Otherwise, if size is a tuple it's either (height, width) or (width, height)
    elif isinstance(size, (tuple, list)) and height_width_order:
        return {"height": size[0], "width": size[1]}
    elif isinstance(size, (tuple, list)) and not height_width_order:
        return {"height": size[1], "width": size[0]}
    elif size is None and max_size is not None:
        if default_to_square:
            raise ValueError("Cannot specify both default_to_square=True and max_size")
        return {"longest_edge": max_size}

    raise ValueError(f"Could not convert size input to size dict: {size}")


def get_size_dict(
    size: int | Iterable[int] | dict[str, int] | None = None,
    max_size: int | None = None,
    height_width_order: bool = True,
    default_to_square: bool = True,
    param_name="size",
) -> dict:
    """
    Converts the old size parameter in the config into the new dict expected in the config. This is to ensure backwards
    compatibility with the old image processor configs and removes ambiguity over whether the tuple is in (height,
    width) or (width, height) format.

    - If `size` is tuple, it is converted to `{"height": size[0], "width": size[1]}` or `{"height": size[1], "width":
    size[0]}` if `height_width_order` is `False`.
    - If `size` is an int, and `default_to_square` is `True`, it is converted to `{"height": size, "width": size}`.
    - If `size` is an int and `default_to_square` is False, it is converted to `{"shortest_edge": size}`. If `max_size`
      is set, it is added to the dict as `{"longest_edge": max_size}`.

    Args:
        size (`int | Iterable[int] | dict[str, int]`, *optional*):
            The `size` parameter to be cast into a size dictionary.
        max_size (`int | None`, *optional*):
            The `max_size` parameter to be cast into a size dictionary.
        height_width_order (`bool`, *optional*, defaults to `True`):
            If `size` is a tuple, whether it's in (height, width) or (width, height) order.
        default_to_square (`bool`, *optional*, defaults to `True`):
            If `size` is an int, whether to default to a square image or not.
    """
    if not isinstance(size, dict):
        size_dict = convert_to_size_dict(size, max_size, default_to_square, height_width_order)
        logger.info(
            f"{param_name} should be a dictionary on of the following set of keys: {VALID_SIZE_DICT_KEYS}, got {size}."
            f" Converted to {size_dict}.",
        )
    else:
        size_dict = size

    if not is_valid_size_dict(size_dict):
        raise ValueError(
            f"{param_name} must have one of the following set of keys: {VALID_SIZE_DICT_KEYS}, got {size_dict.keys()}"
        )
    return size_dict


def select_best_resolution(original_size: tuple, possible_resolutions: list) -> tuple:
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    This is done by calculating the effective and wasted resolution for each possible resolution.

    The best fit resolution is the one that maximizes the effective resolution and minimizes the wasted resolution.

    Args:
        original_size (tuple):
            The original size of the image in the format (height, width).
        possible_resolutions (list):
            A list of possible resolutions in the format [(height1, width1), (height2, width2), ...].

    Returns:
        tuple: The best fit resolution in the format (height, width).
    """
    original_height, original_width = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for height, width in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (height, width)

    return best_fit


def get_patch_output_size(image, target_resolution, input_data_format):
    """
    Given an image and a target resolution, calculate the output size of the image after cropping to the target
    """
    original_height, original_width = get_image_size(image, channel_dim=input_data_format)
    target_height, target_width = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    return new_height, new_width
