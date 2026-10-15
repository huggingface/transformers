# Copyright 2026 The HuggingFace Team. All rights reserved.
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
from collections.abc import Callable
from functools import wraps

from ..feature_extraction_utils import BatchFeature
from ..utils.import_utils import is_torch_available


if is_torch_available():
    import torch

    from . import hub_kernels


_PROCESSING_KERNEL_ADAPTERS: dict[str, tuple[str, Callable]] = {}
_KERNEL_DEVICE_TYPE = "cuda"
_KERNEL_INTERPOLATIONS = {2: "bilinear", 3: "bicubic"}


def register_processing_kernel(name: str, kernel_name: str):
    """
    Register the kernel `kernel_name` of `_HUB_KERNEL_MAPPING` as an implementation of the processing op `name`.

    Processing ops have no common signature to swap, so the decorated function is an adapter: it receives the loaded
    kernel module followed by the arguments of the op, and returns the kernel result, or `None` when the kernel
    cannot handle these arguments so that the caller keeps its default implementation.
    """

    def register(adapter: Callable) -> Callable:
        _PROCESSING_KERNEL_ADAPTERS[name] = (kernel_name, adapter)
        return adapter

    return register


def run_processing_kernel(name: str, *args, **kwargs):
    """Run the kernel registered for the processing op `name`, or return `None` if the default path should be used."""
    if not is_torch_available() or name not in _PROCESSING_KERNEL_ADAPTERS or not hub_kernels._kernels_enabled:
        return None
    if not torch.cuda.is_available():
        return None
    kernel_name, adapter = _PROCESSING_KERNEL_ADAPTERS[name]
    kernel = hub_kernels.lazy_load_kernel(kernel_name)
    if kernel is None:
        return None
    return adapter(kernel, *args, **kwargs)


@register_processing_kernel("connected_component_areas", kernel_name="cv-utils")
def _connected_component_areas_kernel(kernel, regions):
    """Area of the 8-connected component of every pixel of a boolean `(batch_size, 1, height, width)` tensor."""
    if regions.device.type != _KERNEL_DEVICE_TYPE:
        return None
    return kernel.connected_component_areas(regions)


def _kernel_supports(images):
    """Whether the resize kernels can process `images` in one launch: uint8, one kernel device, one channel count."""
    return (
        bool(images)
        and images[0].device.type == _KERNEL_DEVICE_TYPE
        and all(
            image.dtype == torch.uint8 and image.device == images[0].device and image.shape[-3] == images[0].shape[-3]
            for image in images
        )
    )


@register_processing_kernel("resize_normalize", kernel_name="cv-utils")
def _resize_normalize_kernel(kernel, images, sizes, crop_size, resample, rescale_factor, image_mean, image_std):
    """Resize every image to its size in `sizes`, center crop to `crop_size` when given, then rescale and normalize."""
    interpolation = _KERNEL_INTERPOLATIONS.get(resample)
    if interpolation is None or not _kernel_supports(images):
        return None
    crop = None
    if crop_size is not None:
        crop = (crop_size.height, crop_size.width)
        if not all(crop) or any(crop[0] > height or crop[1] > width for height, width in sizes):
            return None
    return kernel.resize_normalize(images, sizes, image_mean, image_std, rescale_factor, interpolation, crop_size=crop)


@register_processing_kernel("resize_normalize_patchify", kernel_name="cv-utils")
def _resize_normalize_patchify_kernel(
    kernel,
    inputs,
    target_sizes,
    resample,
    rescale_factor,
    image_mean,
    image_std,
    patch_size,
    block_size,
    temporal_patch_size,
):
    """Resize every `(T, C, H, W)` input to its target frame size, normalize, and write its flattened patches in order."""
    interpolation = _KERNEL_INTERPOLATIONS.get(resample)
    if interpolation is None or not _kernel_supports(inputs):
        return None
    return kernel.resize_normalize_patchify(
        inputs,
        target_sizes,
        image_mean,
        image_std,
        rescale_factor,
        interpolation,
        patch_size,
        block_size,
        temporal_patch_size,
    )


def use_processing_kernel(preprocess_with_kernel: Callable):
    """
    Decorate the `_preprocess` method of a processor so that `use_kernels=True` first runs `preprocess_with_kernel`.

    `preprocess_with_kernel` receives the processor and the named arguments of `_preprocess`. It returns
    the `BatchFeature` computed by a kernel, or `None` to keep the decorated implementation.
    """

    def decorator(preprocess: Callable) -> Callable:
        signature = inspect.signature(preprocess)

        @wraps(preprocess)
        def wrapper(self, *args, **kwargs):
            if self.use_kernels:
                arguments = signature.bind(self, *args, **kwargs)
                arguments.apply_defaults()
                named_arguments = dict(arguments.arguments)
                named_arguments.pop("self")
                named_arguments.update(named_arguments.pop("kwargs", {}))
                output = preprocess_with_kernel(self, **named_arguments)
                if output is not None:
                    return output
            return preprocess(self, *args, **kwargs)

        return wrapper

    return decorator


def resize_normalize_patchify_images_with_kernel(
    processor,
    images,
    do_resize,
    size,
    resample,
    do_rescale,
    rescale_factor,
    do_normalize,
    image_mean,
    image_std,
    patch_size,
    temporal_patch_size,
    merge_size,
    return_tensors,
    **kwargs,
):
    """
    `pixel_values` and `image_grid_thw` of the Qwen2-VL image pipeline, computed by one kernel call for the batch.

    Target sizes come from the `_resized_size` method of the processor. Patches are written in blocks of
    `processor.patch_block_size`, `merge_size` when it is `None`, and returned as `(C * T, P, P)` blocks when
    `processor.flat_patches` is `False`.
    """
    if not (do_resize and do_rescale and do_normalize):
        return None
    target_sizes = [
        processor._resized_size(
            height=image.shape[-2], width=image.shape[-1], size=size, factor=patch_size * merge_size
        )
        for image in images
    ]
    kernel_output = run_processing_kernel(
        "resize_normalize_patchify",
        [image[None] for image in images],
        target_sizes,
        resample,
        rescale_factor,
        image_mean,
        image_std,
        patch_size,
        merge_size if processor.patch_block_size is None else processor.patch_block_size,
        temporal_patch_size,
    )
    if kernel_output is None:
        return None
    pixel_values, image_grid_thw = kernel_output
    if not processor.flat_patches:
        pixel_values = pixel_values.view(-1, images[0].shape[0] * temporal_patch_size, patch_size, patch_size)
    return BatchFeature(
        data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}, tensor_type=return_tensors
    )


def resize_normalize_patchify_videos_with_kernel(
    processor,
    videos,
    do_resize,
    size,
    resample,
    do_rescale,
    rescale_factor,
    do_normalize,
    image_mean,
    image_std,
    patch_size,
    temporal_patch_size,
    merge_size,
    cap_pixels_per_frame,
    return_tensors,
    **kwargs,
):
    """`pixel_values_videos` and `video_grid_thw` of the Qwen2-VL video pipeline, one kernel call for all frames."""
    if not (do_resize and do_rescale and do_normalize):
        return None
    target_sizes = [
        processor._resized_size(
            height=video.shape[-2],
            width=video.shape[-1],
            num_frames=video.shape[0],
            size=size,
            factor=patch_size * merge_size,
            temporal_factor=temporal_patch_size,
            cap_pixels_per_frame=cap_pixels_per_frame,
        )
        for video in videos
    ]
    kernel_output = run_processing_kernel(
        "resize_normalize_patchify",
        videos,
        target_sizes,
        resample,
        rescale_factor,
        image_mean,
        image_std,
        patch_size,
        merge_size,
        temporal_patch_size,
    )
    if kernel_output is None:
        return None
    pixel_values_videos, video_grid_thw = kernel_output
    return BatchFeature(
        data={"pixel_values_videos": pixel_values_videos, "video_grid_thw": video_grid_thw}, tensor_type=return_tensors
    )
