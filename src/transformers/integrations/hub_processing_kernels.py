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
    return kernel.connected_component_areas(regions)


_KERNEL_INTERPOLATIONS = {2: "bilinear", 3: "bicubic"}


def _kernel_interpolation(images, resample):
    """Interpolation of the resize kernels for `resample`, `None` when they cannot process these images."""
    if not images or any(
        not isinstance(image, torch.Tensor)
        or image.ndim != 3
        or image.dtype != torch.uint8
        or image.device != images[0].device
        or image.shape[0] != images[0].shape[0]
        for image in images
    ):
        return None
    if images[0].device.type != _KERNEL_DEVICE_TYPE:
        return None
    return _KERNEL_INTERPOLATIONS.get(resample)


@register_processing_kernel("resize_normalize", kernel_name="cv-utils")
def _resize_normalize_kernel(kernel, images, size, crop_size, resample, rescale_factor, image_mean, image_std):
    """Resize every image to `size`, center crop to `crop_size` when given, then rescale and normalize."""
    interpolation = _kernel_interpolation(images, resample)
    if interpolation is None:
        return None
    crop = (crop_size.height, crop_size.width) if crop_size is not None else None
    if crop is not None and not all(crop):
        return None
    if size.height and size.width:
        if crop is not None and (crop[0] > size.height or crop[1] > size.width):
            return None
        resize = {"size": (size.height, size.width)}
    elif size.shortest_edge and not size.longest_edge and crop is not None and size.shortest_edge >= max(crop):
        resize = {"shortest_edge": size.shortest_edge}
    else:
        return None
    return kernel.resize_normalize(
        images, image_mean, image_std, rescale_factor, interpolation, crop_size=crop, **resize
    )


@register_processing_kernel("resize_normalize_patchify", kernel_name="cv-utils")
def _resize_normalize_patchify_kernel(
    kernel,
    frames,
    target_sizes,
    resample,
    rescale_factor,
    image_mean,
    image_std,
    patch_size,
    merge_size,
    temporal_patch_size,
    items=None,
):
    """Resize every frame to its target size, normalize, and write the flattened patches of every item in order."""
    interpolation = _kernel_interpolation(frames, resample)
    if interpolation is None:
        return None
    return kernel.resize_normalize_patchify(
        frames,
        target_sizes,
        image_mean,
        image_std,
        rescale_factor,
        interpolation,
        patch_size,
        merge_size,
        temporal_patch_size,
        items=items,
    )


def use_processing_kernel(preprocess_with_kernel: Callable, **options):
    """
    Decorate the `_preprocess` method of a processor so that `use_kernels=True` first runs `preprocess_with_kernel`.

    `preprocess_with_kernel` receives the processor, the named arguments of `_preprocess` and `options`. It returns
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
                output = preprocess_with_kernel(self, **named_arguments, **options)
                if output is not None:
                    return output
            return preprocess(self, *args, **kwargs)

        return wrapper

    return decorator


def resize_normalize_with_kernel(
    processor,
    images,
    do_resize,
    size,
    resample,
    do_center_crop,
    crop_size,
    do_rescale,
    rescale_factor,
    do_normalize,
    image_mean,
    image_std,
    do_pad,
    return_tensors,
    default_methods_of,
    **kwargs,
):
    """
    `pixel_values` of the default torchvision pipeline, computed by one `resize_normalize` kernel call.

    Processors that override a step of the pipeline of the class named `default_methods_of` keep their own path.
    """
    default_class = next(cls for cls in type(processor).__mro__ if cls.__name__ == default_methods_of)
    uses_default_resize_and_normalize = all(
        getattr(type(processor), name) is getattr(default_class, name)
        for name in ("resize", "center_crop", "rescale_and_normalize", "rescale", "normalize")
    )
    if not (uses_default_resize_and_normalize and do_resize and do_rescale and do_normalize and not do_pad):
        return None
    pixel_values = run_processing_kernel(
        "resize_normalize",
        images,
        size,
        crop_size if do_center_crop else None,
        resample,
        rescale_factor,
        image_mean,
        image_std,
    )
    if pixel_values is None:
        return None
    return BatchFeature(data={"pixel_values": list(pixel_values)}, tensor_type=return_tensors)


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
    compute_resized_height_and_width,
    merge_patches=True,
    flatten_patches=True,
    forced_resample=None,
    **kwargs,
):
    """
    `pixel_values` and `image_grid_thw` of the Qwen2-VL image pipeline, computed by one kernel call for the batch.

    `compute_resized_height_and_width` is the size rule of the processor, its `smart_resize`, called on every image. `merge_patches=False` writes patches row by row instead of
    in `merge_size` blocks, `flatten_patches=False` returns them as `(C * T, P, P)` blocks, and `forced_resample`
    replaces `resample` for processors whose `resize` ignores it.
    """
    if not (do_resize and do_rescale and do_normalize):
        return None
    target_sizes = [
        compute_resized_height_and_width(
            image.shape[-2],
            image.shape[-1],
            factor=patch_size * merge_size,
            min_pixels=size.shortest_edge,
            max_pixels=size.longest_edge,
        )
        for image in images
    ]
    kernel_output = run_processing_kernel(
        "resize_normalize_patchify",
        images,
        target_sizes,
        resample if forced_resample is None else forced_resample,
        rescale_factor,
        image_mean,
        image_std,
        patch_size,
        merge_size if merge_patches else 1,
        temporal_patch_size,
    )
    if kernel_output is None:
        return None
    pixel_values, image_grid_thw = kernel_output
    if not flatten_patches:
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
    frames, target_sizes, items = [], [], []
    for video in videos:
        target_size = processor._resized_size(
            *video.shape[-2:],
            video.shape[0],
            size,
            patch_size * merge_size,
            temporal_patch_size,
            bool(cap_pixels_per_frame),
        )
        items.append(list(range(len(frames), len(frames) + video.shape[0])))
        frames.extend(video)
        target_sizes.extend([target_size] * video.shape[0])
    kernel_output = run_processing_kernel(
        "resize_normalize_patchify",
        frames,
        target_sizes,
        resample,
        rescale_factor,
        image_mean,
        image_std,
        patch_size,
        merge_size,
        temporal_patch_size,
        items=items,
    )
    if kernel_output is None:
        return None
    pixel_values_videos, video_grid_thw = kernel_output
    return BatchFeature(
        data={"pixel_values_videos": pixel_values_videos, "video_grid_thw": video_grid_thw}, tensor_type=return_tensors
    )
