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
from ..image_utils import PILImageResampling
from ..utils.import_utils import is_torch_available


if is_torch_available():
    import torch

    from .hub_kernels import lazy_load_kernel


_PROCESSING_KERNEL_ADAPTERS: dict[str, tuple[str, Callable]] = {}
_KERNEL_DEVICE_TYPE = "cuda"
KERNEL_INTERPOLATIONS = {PILImageResampling.BILINEAR: "bilinear", PILImageResampling.BICUBIC: "bicubic"}


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
    if not is_torch_available() or name not in _PROCESSING_KERNEL_ADAPTERS:
        return None
    kernel_name, adapter = _PROCESSING_KERNEL_ADAPTERS[name]
    kernel = lazy_load_kernel(kernel_name)
    if kernel is None:
        return None
    return adapter(kernel, *args, **kwargs)


def kernel_can_resize(images, resample):
    """Whether one launch of the resize kernels can resize `images` with `resample`: uint8, one device, one channel count."""
    return (
        resample in KERNEL_INTERPOLATIONS
        and bool(images)
        and all(
            image.device.type == _KERNEL_DEVICE_TYPE
            and image.dtype == torch.uint8
            and image.device == images[0].device
            and image.shape[-3] == images[0].shape[-3]
            for image in images
        )
    )


@register_processing_kernel("connected_component_areas", kernel_name="cv-utils")
def _connected_component_areas_kernel(kernel, regions):
    """Area of the 8-connected component of every pixel of a boolean `(batch_size, 1, height, width)` tensor."""
    if regions.device.type != _KERNEL_DEVICE_TYPE:
        return None
    return kernel.connected_component_areas(regions)


@register_processing_kernel("resize_normalize", kernel_name="cv-utils")
def _resize_normalize_kernel(kernel, images, sizes, crop_size, resample, rescale_factor, image_mean, image_std):
    """Resize every image to its size in `sizes`, center crop to `crop_size` when given, then rescale and normalize."""
    if not kernel_can_resize(images, resample):
        return None
    return kernel.resize_normalize(
        images, sizes, image_mean, image_std, rescale_factor, KERNEL_INTERPOLATIONS[resample], crop_size=crop_size
    )


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
    patch_block_size,
    temporal_patch_size,
):
    """Resize every `(T, C, H, W)` input to its target frame size, normalize, and write its patches in order."""
    if not kernel_can_resize(inputs, resample):
        return None
    return kernel.resize_normalize_patchify(
        inputs,
        target_sizes,
        image_mean,
        image_std,
        rescale_factor,
        KERNEL_INTERPOLATIONS[resample],
        patch_size,
        patch_block_size,
        temporal_patch_size,
    )


@register_processing_kernel("upsample_argmax", kernel_name="cv-utils")
def _upsample_argmax_kernel(kernel, logits, target_sizes):
    """Class of every pixel of `logits` `(B, C, h, w)` upsampled to `target_sizes`, without the upsampled logits."""
    if logits.device.type != _KERNEL_DEVICE_TYPE:
        return None
    return kernel.upsample_argmax(logits, target_sizes)


def use_processing_kernel(run_with_kernel: Callable):
    """
    Decorate a processor method so that `use_kernels=True` first runs `run_with_kernel`.

    `run_with_kernel` receives the processor and the named arguments of the method. It returns the kernel result, or
    `None` to run the method as usual.
    """

    def decorator(method: Callable) -> Callable:
        signature = inspect.signature(method)

        @wraps(method)
        def wrapper(self, *args, **kwargs):
            if self.use_kernels:
                arguments = signature.bind(self, *args, **kwargs)
                arguments.apply_defaults()
                named_arguments = dict(arguments.arguments)
                named_arguments.pop("self")
                named_arguments.update(named_arguments.pop("kwargs", {}))
                output = run_with_kernel(self, **named_arguments)
                if output is not None:
                    return output
            return method(self, *args, **kwargs)

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
    """`pixel_values` and `image_grid_thw` of the Qwen2-VL image pipeline, from one kernel call for the batch."""
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
        getattr(processor, "patch_block_size", merge_size),
        temporal_patch_size,
    )
    if kernel_output is None:
        return None
    pixel_values, image_grid_thw = kernel_output
    if not getattr(processor, "flatten_patches", True):
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
    """`pixel_values_videos` and `video_grid_thw` of the Qwen2-VL video pipeline, from one kernel call for all frames."""
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
        getattr(processor, "patch_block_size", merge_size),
        temporal_patch_size,
    )
    if kernel_output is None:
        return None
    pixel_values_videos, video_grid_thw = kernel_output
    return BatchFeature(
        data={"pixel_values_videos": pixel_values_videos, "video_grid_thw": video_grid_thw}, tensor_type=return_tensors
    )


def upsample_argmax_with_kernel(processor, outputs, target_sizes, return_segmentation_scores, **kwargs):
    """Class maps of `outputs.logits` upsampled to `target_sizes`, without storing the upsampled logits."""
    if return_segmentation_scores or target_sizes is None:
        return None
    return run_processing_kernel("upsample_argmax", outputs.logits, target_sizes)
