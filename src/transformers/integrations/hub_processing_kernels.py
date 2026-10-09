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
from collections.abc import Callable

from ..utils.import_utils import is_torch_available


if is_torch_available():
    import torch

    from . import hub_kernels


_PROCESSING_KERNEL_ADAPTERS: dict[str, tuple[str, Callable]] = {}


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


_KERNEL_DEVICE_TYPE = "cuda"


@register_processing_kernel("connected_component_areas", kernel_name="cv-utils")
def _connected_component_areas_kernel(kernel, regions):
    """Area of the 8-connected component of every pixel of a boolean `(batch_size, 1, height, width)` tensor."""
    if regions.device.type != _KERNEL_DEVICE_TYPE:
        return None
    height, width = regions.shape[-2:]
    padded_regions = torch.nn.functional.pad(regions.to(torch.uint8), (0, width % 2, 0, height % 2))
    _, areas = kernel.cc_2d(padded_regions.contiguous(), get_counts=True)
    return areas[..., :height, :width]


_KERNEL_INTERPOLATIONS = {2: "bilinear", 3: "bicubic"}


def _resize_kernel_arguments(images, resample, rescale_factor, image_mean, image_std):
    """Interpolation and per-channel stats for the resize kernels, `None` when they cannot process these inputs."""
    if not images or any(
        not isinstance(image, torch.Tensor)
        or image.ndim != 3
        or image.dtype != torch.uint8
        or image.device != images[0].device
        or image.shape[0] != images[0].shape[0]
        for image in images
    ):
        return None
    interpolation = _KERNEL_INTERPOLATIONS.get(resample)
    channels = images[0].shape[0]
    image_mean = [image_mean] * channels if isinstance(image_mean, (int, float)) else list(image_mean)
    image_std = [image_std] * channels if isinstance(image_std, (int, float)) else list(image_std)
    if images[0].device.type != _KERNEL_DEVICE_TYPE or interpolation is None or len(image_mean) != channels:
        return None
    return interpolation, image_mean, image_std, rescale_factor


@register_processing_kernel("resize_normalize", kernel_name="cv-utils")
def _resize_normalize_kernel(kernel, images, size, crop_size, resample, rescale_factor, image_mean, image_std):
    """Resize every image to `size`, center crop to `crop_size` when given, then rescale and normalize."""
    arguments = _resize_kernel_arguments(images, resample, rescale_factor, image_mean, image_std)
    if arguments is None:
        return None
    interpolation, image_mean, image_std, rescale_factor = arguments
    crop = (crop_size.height, crop_size.width) if crop_size is not None else None
    if crop is not None and not all(crop):
        return None
    if size.height and size.width:
        if crop is not None and (crop[0] > size.height or crop[1] > size.width):
            return None
        resize, resize_mode = (size.height, size.width), "square"
    elif size.shortest_edge and not size.longest_edge and crop is not None and size.shortest_edge >= max(crop):
        resize, resize_mode = size.shortest_edge, "shortest_edge"
    else:
        return None
    return kernel.resize_normalize(
        images,
        resize,
        image_mean,
        image_std,
        rescale_factor=rescale_factor,
        resample=interpolation,
        antialias=True,
        crop_size=crop,
        resize_mode=resize_mode,
        round_to_uint8=True,
    )


@register_processing_kernel("resize_normalize_patchify", kernel_name="cv-utils")
def _resize_normalize_patchify_kernel(
    kernel,
    frames,
    target_sizes,
    items,
    resample,
    rescale_factor,
    image_mean,
    image_std,
    patch_size,
    merge_size,
    temporal_patch_size,
):
    """Resize every frame to its target size, normalize, and write the flattened patches of every item in order."""
    arguments = _resize_kernel_arguments(frames, resample, rescale_factor, image_mean, image_std)
    if arguments is None:
        return None
    interpolation, image_mean, image_std, rescale_factor = arguments
    return kernel.resize_normalize_patchify(
        frames,
        target_sizes,
        items,
        image_mean,
        image_std,
        rescale_factor,
        interpolation,
        True,
        patch_size,
        merge_size,
        temporal_patch_size,
        round_to_uint8=True,
    )
