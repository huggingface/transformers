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
from collections.abc import Callable, Iterable
from copy import deepcopy
from functools import partial
from typing import Optional, Union

import numpy as np
from huggingface_hub.dataclasses import validate_typed_dict

from .image_processing_backends import ImageProcessingBackend, PilBackend, TorchVisionBackend
from .image_processing_base import BatchFeature, ImageProcessingMixin
from .image_utils import (
    ChannelDimension,
    ImageInput,
    SizeDict,
    get_image_size,
    make_flat_list_of_images,
    validate_preprocess_arguments,
)
from .processing_utils import ImagesKwargs, Unpack
from .utils import (
    auto_docstring,
    is_torch_available,
    is_torchvision_available,
    is_vision_available,
    logging,
)


if is_vision_available():
    from .image_utils import PILImageResampling

if is_torch_available():
    import torch

if is_torchvision_available():
    from torchvision.transforms.v2 import functional as tvF

    from .image_utils import pil_torch_interpolation_mapping, torch_pil_interpolation_mapping
else:
    pil_torch_interpolation_mapping = None
    torch_pil_interpolation_mapping = None


logger = logging.get_logger(__name__)


INIT_SERVICE_KWARGS = [
    "processor_class",
    "image_processor_type",
]


@auto_docstring
class BaseImageProcessor(ImageProcessingMixin):
    r"""
    Base class for image processors with a pluggable backend architecture.

    This class orchestrates preprocessing by delegating to backends. It handles kwargs validation, input preparation,
    and dispatches to the selected backend's `preprocess` method. The backends implement the actual
    operations (resize, crop, rescale, normalize etc.). The architecture supports any backend (built-in, future, or
    user-defined); new backends can be added via subclassing or at runtime with `register_backend()`. Built-in
    backends include PIL (portable, CPU) and TorchVision (GPU-accelerated, faster).

    Backend Selection
    -----------------

    The processor automatically selects the best available backend, or accepts any registered backend:
    - `backend="auto"` (default): Uses torchvision if available, otherwise pil (or first available)
    - `backend="torchvision"`: TorchVision backend (GPU-accelerated, faster)
    - `backend="pil"`: PIL backend (NumPy/PIL, more portable)
    - `backend="<name>"`: Any backend registered via `register_backend()` (e.g. custom MLX, JAX backends)

        processor = MyImageProcessor(backend="torchvision")  # Fast, GPU support
        processor = MyImageProcessor(backend="pil")         # Portable, CPU-only
        MyImageProcessor.register_backend("mlx", MlxBackend)
        processor = MyImageProcessor(backend="mlx")

    Basic Implementation
    --------------------

    For processors that only need standard operations (resize, center crop, rescale, normalize), define class
    attributes:

        class MyImageProcessor(BaseImageProcessor):
            resample = PILImageResampling.BILINEAR
            image_mean = IMAGENET_DEFAULT_MEAN
            image_std = IMAGENET_DEFAULT_STD
            size = {"height": 224, "width": 224}
            do_resize = True
            do_rescale = True
            do_normalize = True

    Custom Processing
    -----------------

    Override `_preprocess_image_like_inputs` to call a custom backend, or override the backend's `preprocess` (most
    common): Custom logic usually lives in a backend subclass. The backend implements resize, rescale, normalize,
    etc. Override `_backend_classes` to use your custom backend, which overrides `preprocess` and calls
    `self.resize`, `self.center_crop`, etc. (on the backend instance). Alternatively, override
    `_preprocess_image_like_inputs` to bypass the default flow; then use `self._backend_instance.preprocess(...)`
    to delegate to the backend.

    Override `_preprocess_image_like_inputs` (for additional inputs):
        For processors handling multiple input types (e.g., images + segmentation maps), override this method:

            def _preprocess_image_like_inputs(
                self,
                images: ImageInput,
                segmentation_maps: ImageInput | None = None,
                do_convert_rgb: bool,
                input_data_format: ChannelDimension,
                device: "torch.device" | None = None,
                **kwargs,
            ) -> BatchFeature:
                images = self._prepare_image_like_inputs(images, do_convert_rgb, input_data_format, device)
                batch_feature = self._backend_instance.preprocess(images, **kwargs)

                if segmentation_maps is not None:
                    maps = self._prepare_image_like_inputs(segmentation_maps, ...)
                    batch_feature["labels"] = self._backend_instance.preprocess(maps, ...).pixel_values

                return batch_feature

    Extensible Backend Architecture
    -------------------------------

    The processor supports any backend. To customize existing backend behavior, override `_backend_classes`:

        class MyTorchVisionBackend(TorchVisionBackend):
            def resize(self, image, size, **kwargs):
                # Custom resize logic for torchvision
                return super().resize(image, size, **kwargs)

        class MyPilBackend(PilBackend):
            def resize(self, image, size, **kwargs):
                # Custom resize logic for PIL
                return super().resize(image, size, **kwargs)

        class MyImageProcessor(BaseImageProcessor):
            _backend_classes = {
                "torchvision": MyTorchVisionBackend,
                "pil": MyPilBackend,
            }

    To add a new backend (e.g. MLX, JAX, or any future/custom implementation), extend both `_backend_classes`
    and `_backend_availability_checks`. For runtime registration without subclassing, use `register_backend()`:

        class MyNewBackend(ImageProcessingBackend):
            # Implement required methods
            ...

        class MyImageProcessor(BaseImageProcessor):
            _backend_classes = {
                **BaseImageProcessor._backend_classes,
                "my_backend": MyNewBackend,
            }
            _backend_availability_checks = {
                **BaseImageProcessor._backend_availability_checks,
                "my_backend": lambda: check_my_backend_available(),
            }

    Custom Parameters
    -----------------

    To add parameters beyond `ImagesKwargs`, create a custom kwargs class and set it as `valid_kwargs`:

        class MyImageProcessorKwargs(ImagesKwargs):
            custom_param: int | None = None

        class MyImageProcessor(BaseImageProcessor):
            valid_kwargs = MyImageProcessorKwargs
            custom_param = 10  # default value

    Key Notes
    ---------

    - The architecture supports any backend: built-in (pil, torchvision), future frameworks, or user-defined
    - Backends receive images as torch.Tensor (torchvision) or np.ndarray (PIL), channel-first; custom backends
      may use other types
    - All images have channel dimension first during actual processing, regardless of backend
    - Arguments not provided by users default to class attribute values
    - Use `register_backend()` to add custom backends at runtime without modifying source code
    - Backend classes encapsulate backend-specific logic and can be overridden for customization
    """

    resample = None
    image_mean = None
    image_std = None
    size = None
    default_to_square = True
    crop_size = None
    do_resize = None
    do_center_crop = None
    do_pad = None
    pad_size = None
    do_rescale = None
    rescale_factor = 1 / 255
    do_normalize = None
    do_convert_rgb = None
    return_tensors = None
    input_data_format = None
    device = None
    model_input_names = ["pixel_values"]
    image_seq_length = None
    valid_kwargs = ImagesKwargs

    _backend_classes = {
        "torchvision": TorchVisionBackend,
        "pil": PilBackend,
    }

    _backend_availability_checks = {
        "torchvision": is_torchvision_available,
        "pil": lambda: True,  # PIL backend is always available
    }

    @classmethod
    def register_backend(
        cls,
        name: str,
        backend_class: ImageProcessingBackend,
        availability_check: Callable[[], bool] | None = None,
    ):
        """
        Register a new backend for this image processor.

        This allows users to add custom backends (e.g., MLX, JAX) without modifying the source code.
        The backend will be available for this processor class and can be selected via the `backend` parameter.

        Args:
            name (`str`):
                The name of the backend (e.g., "mlx", "jax").
            backend_class (`type[ImageProcessingBackend]`):
                The backend class that implements the backend interface. Must inherit from `ImageProcessingBackend`.
            availability_check (`callable`, *optional*):
                A function that returns `True` if the backend is available, `False` otherwise.
                This is typically a check for whether required dependencies are installed.
                If not provided, defaults to always available (returns `True`).

        Example:
            ```python
            from transformers import ImageProcessingBackend, LlavaNextImageProcessor

            class LlavaNextMlxBackend(ImageProcessingBackend):
                def resize(self, image, size, **kwargs):
                    # MLX implementation
                    pass
                # ... implement other methods

            # Register with availability check
            LlavaNextImageProcessor.register_backend(
                name="mlx",
                backend_class=LlavaNextMlxBackend,
                availability_check=lambda: is_mlx_available()
            )

            # Or register without availability check (always available)
            LlavaNextImageProcessor.register_backend(
                name="mlx",
                backend_class=LlavaNextMlxBackend,
            )

            processor = LlavaNextImageProcessor(backend="mlx")
            ```
        """
        if not issubclass(backend_class, ImageProcessingBackend):
            raise TypeError(f"Backend class must inherit from ImageProcessingBackend, got {backend_class.__name__}")

        cls._backend_classes[name] = backend_class
        cls._backend_availability_checks[name] = availability_check if availability_check is not None else lambda: True

    @classmethod
    def _get_available_backends(cls):
        """
        Get a list of available backend names based on availability checks.

        Returns:
            list[str]: List of available backend names.
        """
        available = []
        for backend_name, check_func in cls._backend_availability_checks.items():
            if check_func():
                available.append(backend_name)
        return available

    @classmethod
    def _get_default_backend(cls):
        """
        Get the default backend name (first available backend in priority order).

        Returns:
            str: Default backend name.
        """
        available = cls._get_available_backends()
        if not available:
            raise RuntimeError("No backends are available. At least 'pil' backend should be available.")
        # Priority: torchvision > pil
        if "torchvision" in available:
            return "torchvision"
        return available[0]

    def __init__(self, backend="auto", **kwargs: Unpack[ImagesKwargs]):
        super().__init__(**kwargs)

        if backend == "auto":
            backend = self._get_default_backend()
        elif backend not in self._backend_classes:
            available_backends = list(self._backend_classes.keys())
            raise ValueError(f"Invalid backend '{backend}'. Must be one of: {available_backends}, or 'auto'.")
        else:
            # Check if requested backend is available
            check_func = self._backend_availability_checks.get(backend)
            if not check_func():
                fallback_backend = self._get_default_backend()
                logger.warning_once(
                    f"You requested backend='{backend}' but it is not available. "
                    f"Falling back to backend='{fallback_backend}'."
                )
                backend = fallback_backend
        self.backend = backend
        self._backend_instance = self._backend_classes[self.backend]()

        size = kwargs.pop("size", self.size)
        self.size = (
            get_size_dict(size=size, default_to_square=kwargs.pop("default_to_square", self.default_to_square))
            if size is not None
            else None
        )
        crop_size = kwargs.pop("crop_size", self.crop_size)
        self.crop_size = get_size_dict(crop_size, param_name="crop_size") if crop_size is not None else None
        pad_size = kwargs.pop("pad_size", self.pad_size)
        self.pad_size = get_size_dict(size=pad_size, param_name="pad_size") if pad_size is not None else None

        for key in self.valid_kwargs.__annotations__:
            kwarg = kwargs.pop(key, None)
            if kwarg is not None:
                setattr(self, key, kwarg)
            else:
                setattr(self, key, deepcopy(getattr(self, key, None)))

        # get valid kwargs names
        self._valid_kwargs_names = list(self.valid_kwargs.__annotations__.keys())

    @property
    def is_fast(self) -> bool:
        """
        `bool`: Whether or not this image processor is using the fast (TorchVision) backend.
        The `is_fast` property is deprecated and will be removed in v5.3 of Transformers.
        Use the `backend` attribute instead (e.g., `processor.backend == "torchvision"`).
        """
        logger.warning_once(
            "The `is_fast` property is deprecated and will be removed in v5.3 of Transformers. "
            "Use the `backend` attribute instead (e.g., `processor.backend == 'torchvision'`)."
        )
        return self.backend == "torchvision"

    def __call__(self, images: ImageInput, *args, **kwargs: Unpack[ImagesKwargs]) -> BatchFeature:
        """Preprocess an image or a batch of images."""
        return self.preprocess(images, *args, **kwargs)

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
        do_convert_rgb: bool | None = None,
        input_data_format: str | ChannelDimension | None = None,
        device: Optional["torch.device"] = None,
        expected_ndims: int = 3,
    ) -> list["torch.Tensor"] | list[np.ndarray]:
        """
        Prepare image-like inputs for processing.

        Args:
            images (`ImageInput`):
                The image-like inputs to process.
            do_convert_rgb (`bool`, *optional*):
                Whether to convert the images to RGB.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The input data format of the images.
            device (`torch.device`, *optional*):
                The device to put the processed images on (torchvision backend only).
            expected_ndims (`int`, *optional*):
                The expected number of dimensions for the images.

        Returns:
            List[`torch.Tensor`] or List[`np.ndarray`]: The processed images.
        """
        images = self._prepare_images_structure(images, expected_ndims=expected_ndims)

        process_image_partial = partial(
            self._backend_instance.process_image,
            do_convert_rgb=do_convert_rgb,
            input_data_format=input_data_format,
            device=device if self.backend == "torchvision" else None,
        )

        has_nested_structure = len(images) > 0 and isinstance(images[0], list | tuple)

        if has_nested_structure:
            processed_images = [[process_image_partial(img) for img in nested_list] for nested_list in images]
        else:
            processed_images = [process_image_partial(img) for img in images]

        return processed_images

    def _standardize_kwargs(
        self,
        size: SizeDict | None = None,
        crop_size: SizeDict | None = None,
        pad_size: SizeDict | None = None,
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
        if size is not None:
            size = SizeDict(**get_size_dict(size=size, default_to_square=default_to_square))
        if crop_size is not None:
            crop_size = SizeDict(**get_size_dict(crop_size, param_name="crop_size"))
        if pad_size is not None:
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

    def _preprocess_image_like_inputs(
        self,
        images: ImageInput,
        *args,
        do_convert_rgb: bool,
        input_data_format: ChannelDimension,
        device: Union[str, "torch.device"] | None = None,
        **kwargs: Unpack[ImagesKwargs],
    ) -> BatchFeature:
        """
        Preprocess image-like inputs.
        To be overridden by subclasses when image-like inputs other than images should be processed.
        """
        images = self._prepare_image_like_inputs(
            images=images, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format, device=device
        )
        return self._backend_instance.preprocess(images, *args, **kwargs)

    def to_dict(self):
        encoder_dict = super().to_dict()

        # Filter out None values that are class defaults
        filtered_dict = {}
        for key, value in encoder_dict.items():
            if value is None:
                class_default = getattr(type(self), key, "NOT_FOUND")
                # Keep None if user explicitly set it (class default is non-None)
                if class_default != "NOT_FOUND" and class_default is not None:
                    filtered_dict[key] = value
            else:
                filtered_dict[key] = value

        filtered_dict.pop("_valid_processor_keys", None)
        filtered_dict.pop("_valid_kwargs_names", None)
        # Don't save backend to config - it should be selected at runtime
        filtered_dict.pop("backend", None)
        filtered_dict.pop("_backend_instance", None)
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
