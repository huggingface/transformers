# Copyright 2026 OpenBMB and the HuggingFace Inc. team. All rights reserved.
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


import copy
import math

import numpy as np
import torch
from huggingface_hub.dataclasses import strict

from ...feature_extraction_utils import BatchFeature
from ...image_transforms import divide_to_patches, group_images_by_shape, reorder_images
from ...image_utils import ImageInput, SizeDict
from ...processing_utils import Unpack
from ...utils import TensorType, auto_docstring, is_cv2_available, logging, requires_backends
from ...video_utils import VideoInput
from ..minicpmv4_6.configuration_minicpmv4_6 import MiniCPMV4_6Config, MiniCPMV4_6VisionConfig
from ..minicpmv4_6.image_processing_minicpmv4_6 import (
    MiniCPMV4_6ImageProcessor,
    MiniCPMV4_6ImageProcessorKwargs,
    ensure_divide,
)
from ..minicpmv4_6.image_processing_pil_minicpmv4_6 import (
    MiniCPMV4_6ImageProcessorPil,
    MiniCPMV4_6ImageProcessorPilKwargs,
)
from ..minicpmv4_6.modeling_minicpmv4_6 import MiniCPMV4_6ForConditionalGeneration, MiniCPMV4_6Model
from ..minicpmv4_6.processing_minicpmv4_6 import MiniCPMV4_6Processor
from ..minicpmv4_6.video_processing_minicpmv4_6 import MiniCPMV4_6VideoProcessor, MiniCPMV4_6VideoProcessorKwargs


if is_cv2_available():
    import cv2


logger = logging.get_logger(__name__)


def adaptive_upsize_tier(
    image: np.ndarray,
    font_px_threshold: float,
    dense_per_kpx_threshold: float,
    tiny_pixels: int,
) -> str:
    r"""
    Pick the resize target for an image small enough that it is not sliced.

    Text that is both small and dense is worth upscaling to the full `scale_resolution`
    (`"448"`), images that are merely tiny are upscaled to a lower target (`"84"`), and
    everything else keeps its native resolution (`"native"`). Glyph height and density are
    measured with an Otsu threshold followed by 8-connected component analysis, matching the
    classifier the checkpoint was trained with, so OpenCV is required.

    Args:
        image (`np.ndarray`):
            `uint8` image to classify, shaped `(num_channels, height, width)` or `(height, width)`.
        font_px_threshold (`float`):
            Largest median glyph height, in pixels, still considered small text.
        dense_per_kpx_threshold (`float`):
            Smallest number of glyphs per 1000 pixels still considered dense text.
        tiny_pixels (`int`):
            Images with fewer pixels than this are treated as tiny.
    """
    requires_backends(adaptive_upsize_tier, ["cv2"])

    height, width = image.shape[-2:]
    if height <= 0 or width <= 0:
        return "native"

    if image.ndim == 3 and image.shape[0] >= 3:
        channels = image.astype(np.int32)
        # ITU-R 601-2 luma, matching the integer arithmetic of `PIL.Image.convert("L")`
        gray = ((channels[0] * 19595 + channels[1] * 38470 + channels[2] * 7471 + 32768) >> 16).astype(np.uint8)
    else:
        gray = np.ascontiguousarray(image.reshape(height, width)).astype(np.uint8)

    _, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    foreground = binarized == 0
    if foreground.mean() > 0.5:
        # Assume glyphs cover less than half the image, so flip light text on a dark page
        foreground = ~foreground

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8), connectivity=8)
    glyph_heights = []
    for label in range(1, num_labels):
        _, _, component_width, component_height, area = stats[label]
        if area < 3 or component_height < 2:
            continue
        if component_width >= 0.9 * width or component_height >= 0.9 * height:
            continue
        glyph_heights.append(component_height)

    font_px = float(np.median(glyph_heights)) if glyph_heights else 0.0
    density = len(glyph_heights) * 1000.0 / (width * height)
    if 0 < font_px <= font_px_threshold and density >= dense_per_kpx_threshold:
        return "448"

    if height * width < tiny_pixels:
        return "84"
    return "native"


@auto_docstring(checkpoint="openbmb/MiniCPM-OCR-1.0")
@strict
class MiniCPMOcrVisionConfig(MiniCPMV4_6VisionConfig):
    pass


@auto_docstring(checkpoint="openbmb/MiniCPM-OCR-1.0")
@strict
class MiniCPMOcrConfig(MiniCPMV4_6Config):
    r"""
    insert_layer_id (`int`, *optional*, defaults to 6):
        Vision encoder layer index after which the window-attention merger is applied.
    downsample_mode (`str`, *optional*, defaults to `"4x"`):
        Visual token downsampling ratio. `"4x"` keeps 4× more tokens than `"16x"`, which is what
        text-dense document images are read with.
    merge_kernel_size (`tuple[int, int]`, *optional*, defaults to `(2, 2)`):
        Kernel size `(h, w)` for merging adjacent visual patches in the Merger.
    merger_times (`int`, *optional*, defaults to 1):
        Number of iterative merge rounds in the Merger.
    """

    tie_word_embeddings: bool = True
    downsample_mode: str = "4x"
    drop_vision_last_layer = AttributeError()


class MiniCPMOcrImageProcessorKwargs(MiniCPMV4_6ImageProcessorKwargs, total=False):
    r"""
    max_slice_nums (`int`, *optional*, defaults to 9):
        Maximum number of slices when splitting a high-resolution image.
    scale_resolution (`int`, *optional*, defaults to 448):
        Target resolution for individual slices.
    patch_size (`int`, *optional*, defaults to 14):
        Spatial patch size of the vision encoder.
    slice_mode (`bool`, *optional*, defaults to `True`):
        Whether to split images into multiple slices for higher resolution.
    downsample_mode (`str`, *optional*, defaults to `"4x"`):
        Visual token downsampling mode. `"4x"` keeps 4x more tokens than `"16x"`,
        which applies the full merge. Overriding it per call is supported.
    use_image_id (`bool`, *optional*, defaults to `True`):
        Whether to prepend an image-id tag (``<image_id>N</image_id>``) before
        each image placeholder. Consumed by the Processor for placeholder
        generation, not by the image processing pipeline itself.
    disable_small_image_upsize (`bool`, *optional*, defaults to `False`):
        Whether to keep images smaller than `scale_resolution` squared at their native resolution
        instead of upscaling them, aligning them to the patch grid only.
    adaptive_small_image_upsize (`bool`, *optional*, defaults to `False`):
        Whether to pick the resize target of those small images from their glyph size and density
        instead of applying `disable_small_image_upsize` uniformly. Requires OpenCV
        (`opencv-python`); see [`adaptive_upsize_tier`].
    adaptive_upsize_font_px (`float`, *optional*, defaults to 12.0):
        Largest median glyph height, in pixels, still treated as small text.
    adaptive_upsize_dense_per_kpx (`float`, *optional*, defaults to 2.0):
        Smallest number of glyphs per 1000 pixels still treated as dense text.
    adaptive_upsize_tiny_pixels (`int`, *optional*, defaults to 6272):
        Images with fewer pixels than this are treated as tiny.
    adaptive_upsize_tiny_target_resolution (`int`, *optional*, defaults to 84):
        Resolution those tiny images are upscaled to.
    """

    disable_small_image_upsize: bool
    adaptive_small_image_upsize: bool
    adaptive_upsize_font_px: float
    adaptive_upsize_dense_per_kpx: float
    adaptive_upsize_tiny_pixels: int
    adaptive_upsize_tiny_target_resolution: int


class MiniCPMOcrImageProcessor(MiniCPMV4_6ImageProcessor):
    downsample_mode = "4x"
    disable_small_image_upsize = False
    adaptive_small_image_upsize = False
    adaptive_upsize_font_px = 12.0
    adaptive_upsize_dense_per_kpx = 2.0
    adaptive_upsize_tiny_pixels = 6272
    adaptive_upsize_tiny_target_resolution = 84
    valid_kwargs = MiniCPMOcrImageProcessorKwargs

    def __call__(
        self,
        images: ImageInput,
        **kwargs: Unpack[MiniCPMOcrImageProcessorKwargs],
    ) -> BatchFeature:
        # These options are read from the instance by `find_best_resize` and
        # `resolve_source_resolution` rather than received as `_preprocess` arguments. Apply the
        # overrides to a shallow copy so the processor itself is never mutated and concurrent calls
        # cannot observe each other's values.
        processor = self
        for name in (
            "downsample_mode",
            "disable_small_image_upsize",
            "adaptive_small_image_upsize",
            "adaptive_upsize_font_px",
            "adaptive_upsize_dense_per_kpx",
            "adaptive_upsize_tiny_pixels",
            "adaptive_upsize_tiny_target_resolution",
        ):
            value = kwargs.pop(name, None)
            if value is None or value == getattr(self, name):
                continue
            if processor is self:
                processor = copy.copy(self)
            setattr(processor, name, value)
        return processor.preprocess(images, **kwargs)

    def find_best_resize(
        self,
        image_size: tuple[int, int],
        scale_resolution: int,
        patch_size: int,
        allow_upscale: bool = False,
    ) -> tuple[int, int]:
        height, width = image_size
        if (height * width > scale_resolution * scale_resolution) or allow_upscale:
            aspect_ratio = width / height
            height = int(scale_resolution / math.sqrt(aspect_ratio))
            width = int(height * aspect_ratio)
        # The resize grid follows the number of 2×2 spatial merges the visual tokens go through:
        # one in `"4x"` mode (downsample MLP), two in `"16x"` mode (ViT insert merger as well).
        merge_factor = 2 if self.downsample_mode == "4x" else 4
        best_width = ensure_divide(width, patch_size * merge_factor)
        best_height = ensure_divide(height, patch_size * merge_factor)
        return best_height, best_width

    def resolve_source_resolution(self, image: "torch.Tensor", scale_resolution: int) -> tuple[int, bool]:
        r"""
        Resolve the resize target of an image that is small enough not to be sliced.

        Returns the resolution to resize towards and whether upscaling is allowed. Text that is
        small and dense is upscaled to `scale_resolution`, tiny images are upscaled to
        `adaptive_upsize_tiny_target_resolution`, and the remaining ones keep their native resolution.

        Args:
            image (`torch.Tensor`):
                Image to classify, shaped `(num_channels, height, width)`.
            scale_resolution (`int`):
                Target resolution requested for this call.
        """
        if not self.adaptive_small_image_upsize:
            return scale_resolution, not self.disable_small_image_upsize

        tier = adaptive_upsize_tier(
            image.detach().to("cpu", torch.uint8).numpy(),
            font_px_threshold=self.adaptive_upsize_font_px,
            dense_per_kpx_threshold=self.adaptive_upsize_dense_per_kpx,
            tiny_pixels=self.adaptive_upsize_tiny_pixels,
        )
        if tier == "448":
            return scale_resolution, True
        if tier == "84":
            return self.adaptive_upsize_tiny_target_resolution, True
        return scale_resolution, not self.disable_small_image_upsize

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        resample,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        max_slice_nums: int,
        scale_resolution: int,
        patch_size: int,
        slice_mode: bool,
        return_tensors: str | TensorType | None = None,
        disable_grouping: bool | None = None,
        **kwargs,
    ) -> BatchFeature:
        per_image_pixel_values: list[list[torch.Tensor]] = []
        per_image_target_sizes: list[list[list[int]]] = []
        all_grids: list[list[int]] = []

        for image in images:
            image_size = image.shape[-2:]
            best_grid = None

            if slice_mode:
                best_grid = self.get_sliced_grid(image_size, max_slice_nums, scale_resolution)

            image_patches = [image]
            if do_resize:
                # Always resize source image
                source_resolution = scale_resolution
                allow_upscale = best_grid is None
                if best_grid is None:
                    source_resolution, allow_upscale = self.resolve_source_resolution(image, scale_resolution)
                source_h, source_w = self.find_best_resize(
                    image_size, source_resolution, patch_size, allow_upscale=allow_upscale
                )
                source_img = self.resize(image, size=SizeDict(height=source_h, width=source_w), resample=resample)

                # Collect all patches for this image: [source, *slices]
                image_patches = [source_img]
                patch_height = patch_width = 0
                if best_grid is not None:
                    refine_h, refine_w = self.get_refine_size(
                        image_size, best_grid, scale_resolution, patch_size, allow_upscale=True
                    )
                    refine_img = self.resize(image, size=SizeDict(height=refine_h, width=refine_w), resample=resample)
                    refine_height, refine_width = refine_img.shape[-2:]
                    grid_y, grid_x = best_grid
                    patch_height, patch_width = refine_height // grid_y, refine_width // grid_x
                    slice_patches = divide_to_patches(refine_img, (patch_height, patch_width))
                    image_patches.extend(slice_patches)

            # Group patches by shape and batch rescale + normalize
            grouped_patches, grouped_index = group_images_by_shape(image_patches, disable_grouping=disable_grouping)
            processed_grouped = {}
            for shape, stacked in grouped_patches.items():
                stacked = self.rescale_and_normalize(
                    stacked.float(), do_rescale, rescale_factor, do_normalize, image_mean, image_std
                )
                processed_grouped[shape] = stacked
            processed_patches = reorder_images(processed_grouped, grouped_index)

            image_pv = [self.reshape_by_patch(processed_patches[0], patch_size)]
            image_ts = [[source_h // patch_size, source_w // patch_size]]
            for processed_slice in processed_patches[1:]:
                image_pv.append(self.reshape_by_patch(processed_slice, patch_size))
                image_ts.append([patch_height // patch_size, patch_width // patch_size])

            per_image_pixel_values.append(image_pv)
            per_image_target_sizes.append(image_ts)
            all_grids.append(best_grid if best_grid is not None else [0, 0])

        all_pv = [pv for sublist in per_image_pixel_values for pv in sublist]
        pixel_values = torch.cat(all_pv, dim=-1).unsqueeze(0)

        all_ts = [ts for sublist in per_image_target_sizes for ts in sublist]
        target_sizes = torch.tensor(all_ts, dtype=torch.int32)

        num_patches_per_image = [len(sublist) for sublist in per_image_pixel_values]

        return BatchFeature(
            data={
                "pixel_values": pixel_values,
                "target_sizes": target_sizes,
                "grids": all_grids,
                "num_patches_per_image": num_patches_per_image,
            },
            tensor_type=return_tensors,
            skip_tensor_conversion=["grids", "num_patches_per_image"],
        )


class MiniCPMOcrImageProcessorPilKwargs(MiniCPMV4_6ImageProcessorPilKwargs, total=False):
    r"""
    max_slice_nums (`int`, *optional*, defaults to 9):
        Maximum number of slices when splitting a high-resolution image.
    scale_resolution (`int`, *optional*, defaults to 448):
        Target resolution for individual slices.
    patch_size (`int`, *optional*, defaults to 14):
        Spatial patch size of the vision encoder.
    slice_mode (`bool`, *optional*, defaults to `True`):
        Whether to split images into multiple slices for higher resolution.
    downsample_mode (`str`, *optional*, defaults to `"4x"`):
        Visual token downsampling mode. `"4x"` keeps 4x more tokens than `"16x"`,
        which applies the full merge. Overriding it per call is supported.
    use_image_id (`bool`, *optional*, defaults to `True`):
        Whether to prepend an image-id tag (``<image_id>N</image_id>``) before
        each image placeholder. Consumed by the Processor for placeholder
        generation, not by the image processing pipeline itself.
    disable_small_image_upsize (`bool`, *optional*, defaults to `False`):
        Whether to keep images smaller than `scale_resolution` squared at their native resolution
        instead of upscaling them, aligning them to the patch grid only.
    adaptive_small_image_upsize (`bool`, *optional*, defaults to `False`):
        Whether to pick the resize target of those small images from their glyph size and density
        instead of applying `disable_small_image_upsize` uniformly. Requires OpenCV
        (`opencv-python`); see [`adaptive_upsize_tier`].
    adaptive_upsize_font_px (`float`, *optional*, defaults to 12.0):
        Largest median glyph height, in pixels, still treated as small text.
    adaptive_upsize_dense_per_kpx (`float`, *optional*, defaults to 2.0):
        Smallest number of glyphs per 1000 pixels still treated as dense text.
    adaptive_upsize_tiny_pixels (`int`, *optional*, defaults to 6272):
        Images with fewer pixels than this are treated as tiny.
    adaptive_upsize_tiny_target_resolution (`int`, *optional*, defaults to 84):
        Resolution those tiny images are upscaled to.
    """

    disable_small_image_upsize: bool
    adaptive_small_image_upsize: bool
    adaptive_upsize_font_px: float
    adaptive_upsize_dense_per_kpx: float
    adaptive_upsize_tiny_pixels: int
    adaptive_upsize_tiny_target_resolution: int


class MiniCPMOcrImageProcessorPil(MiniCPMV4_6ImageProcessorPil):
    downsample_mode = "4x"
    disable_small_image_upsize = False
    adaptive_small_image_upsize = False
    adaptive_upsize_font_px = 12.0
    adaptive_upsize_dense_per_kpx = 2.0
    adaptive_upsize_tiny_pixels = 6272
    adaptive_upsize_tiny_target_resolution = 84
    valid_kwargs = MiniCPMOcrImageProcessorPilKwargs

    def __call__(
        self,
        images: ImageInput,
        **kwargs: Unpack[MiniCPMOcrImageProcessorPilKwargs],
    ) -> BatchFeature:
        # These options are read from the instance by `find_best_resize` and
        # `resolve_source_resolution` rather than received as `_preprocess` arguments. Apply the
        # overrides to a shallow copy so the processor itself is never mutated and concurrent calls
        # cannot observe each other's values.
        processor = self
        for name in (
            "downsample_mode",
            "disable_small_image_upsize",
            "adaptive_small_image_upsize",
            "adaptive_upsize_font_px",
            "adaptive_upsize_dense_per_kpx",
            "adaptive_upsize_tiny_pixels",
            "adaptive_upsize_tiny_target_resolution",
        ):
            value = kwargs.pop(name, None)
            if value is None or value == getattr(self, name):
                continue
            if processor is self:
                processor = copy.copy(self)
            setattr(processor, name, value)
        return processor.preprocess(images, **kwargs)

    def find_best_resize(
        self,
        image_size: tuple[int, int],
        scale_resolution: int,
        patch_size: int,
        allow_upscale: bool = False,
    ) -> tuple[int, int]:
        height, width = image_size
        if (height * width > scale_resolution * scale_resolution) or allow_upscale:
            aspect_ratio = width / height
            height = int(scale_resolution / math.sqrt(aspect_ratio))
            width = int(height * aspect_ratio)
        # The resize grid follows the number of 2×2 spatial merges the visual tokens go through:
        # one in `"4x"` mode (downsample MLP), two in `"16x"` mode (ViT insert merger as well).
        merge_factor = 2 if self.downsample_mode == "4x" else 4
        best_width = ensure_divide(width, patch_size * merge_factor)
        best_height = ensure_divide(height, patch_size * merge_factor)
        return best_height, best_width

    def resolve_source_resolution(self, image: np.ndarray, scale_resolution: int) -> tuple[int, bool]:
        r"""
        Resolve the resize target of an image that is small enough not to be sliced.

        Returns the resolution to resize towards and whether upscaling is allowed. Text that is
        small and dense is upscaled to `scale_resolution`, tiny images are upscaled to
        `adaptive_upsize_tiny_target_resolution`, and the remaining ones keep their native resolution.

        Args:
            image (`np.ndarray`):
                Image to classify, shaped `(num_channels, height, width)`.
            scale_resolution (`int`):
                Target resolution requested for this call.
        """
        if not self.adaptive_small_image_upsize:
            return scale_resolution, not self.disable_small_image_upsize

        tier = adaptive_upsize_tier(
            image,
            font_px_threshold=self.adaptive_upsize_font_px,
            dense_per_kpx_threshold=self.adaptive_upsize_dense_per_kpx,
            tiny_pixels=self.adaptive_upsize_tiny_pixels,
        )
        if tier == "448":
            return scale_resolution, True
        if tier == "84":
            return self.adaptive_upsize_tiny_target_resolution, True
        return scale_resolution, not self.disable_small_image_upsize

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        resample,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        max_slice_nums: int,
        scale_resolution: int,
        patch_size: int,
        slice_mode: bool,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        per_image_pixel_values: list[list[np.ndarray]] = []
        per_image_target_sizes: list[list[list[int]]] = []
        all_grids: list[list[int]] = []

        for image in images:
            image_size = image.shape[-2:]
            best_grid = None

            if slice_mode:
                best_grid = self.get_sliced_grid(image_size, max_slice_nums, scale_resolution)

            source_img = image
            source_height, source_width = image_size
            if do_resize:
                source_resolution = scale_resolution
                allow_upscale = best_grid is None
                if best_grid is None:
                    source_resolution, allow_upscale = self.resolve_source_resolution(image, scale_resolution)
                source_height, source_width = self.find_best_resize(
                    image_size, source_resolution, patch_size, allow_upscale=allow_upscale
                )
                source_img = self.resize(
                    image, size=SizeDict(height=source_height, width=source_width), resample=resample
                )

            if do_rescale:
                source_img = self.rescale(source_img, rescale_factor)
            if do_normalize:
                source_img = self.normalize(source_img, image_mean, image_std)

            image_pv = [self.reshape_by_patch(source_img, patch_size)]
            image_ts = [[source_height // patch_size, source_width // patch_size]]

            if best_grid is not None:
                refine_img = image
                refine_h, refine_w = image_size
                if do_resize:
                    refine_h, refine_w = self.get_refine_size(
                        image_size, best_grid, scale_resolution, patch_size, allow_upscale=True
                    )
                    refine_img = self.resize(image, size=SizeDict(height=refine_h, width=refine_w), resample=resample)

                refine_height, refine_width = refine_img.shape[-2:]
                grid_y, grid_x = best_grid
                patch_height, patch_width = refine_height // grid_y, refine_width // grid_x
                slice_patches = divide_to_patches(refine_img, (refine_height // grid_y, refine_width // grid_x))

                for patch_arr in slice_patches:
                    if do_rescale:
                        patch_arr = self.rescale(patch_arr, rescale_factor)
                    if do_normalize:
                        patch_arr = self.normalize(patch_arr, image_mean, image_std)
                    image_pv.append(self.reshape_by_patch(patch_arr, patch_size))
                    image_ts.append([patch_height // patch_size, patch_width // patch_size])

            per_image_pixel_values.append(image_pv)
            per_image_target_sizes.append(image_ts)
            all_grids.append(best_grid if best_grid is not None else [0, 0])

        all_pv = [pv for sublist in per_image_pixel_values for pv in sublist]
        pixel_values = np.concatenate(all_pv, axis=-1)[np.newaxis, ...]

        all_ts = [ts for sublist in per_image_target_sizes for ts in sublist]
        target_sizes = np.array(all_ts, dtype=np.int32)

        num_patches_per_image = [len(sublist) for sublist in per_image_pixel_values]

        return BatchFeature(
            data={
                "pixel_values": pixel_values,
                "target_sizes": target_sizes,
                "grids": all_grids,
                "num_patches_per_image": num_patches_per_image,
            },
            tensor_type=return_tensors,
            skip_tensor_conversion=["grids", "num_patches_per_image"],
        )


class MiniCPMOcrVideoProcessorKwargs(MiniCPMV4_6VideoProcessorKwargs, total=False):
    r"""
    max_num_frames (`int`, *optional*, defaults to 128):
        Maximum number of main frames to sample per video.
    stack_frames (`int`, *optional*, defaults to 1):
        Sub-frames per second to stack.  ``1`` disables stacking.
    max_slice_nums (`int`, *optional*, defaults to 9):
        Maximum number of slices when splitting a high-resolution image.
    scale_resolution (`int`, *optional*, defaults to 448):
        Target resolution for individual slices.
    patch_size (`int`, *optional*, defaults to 14):
        Spatial patch size of the vision encoder.
    slice_mode (`bool`, *optional*, defaults to `True`):
        Whether to split images into multiple slices for higher resolution.
    downsample_mode (`str`, *optional*, defaults to `"4x"`):
        Visual token downsampling mode. `"4x"` keeps 4x more tokens than `"16x"`,
        which applies the full merge. Overriding it per call is supported.
    use_image_id (`bool`, *optional*, defaults to `True`):
        Whether to prepend an image-id tag (``<image_id>N</image_id>``) before
        each image placeholder. Consumed by the Processor for placeholder
        generation, not by the image processing pipeline itself.
    """


class MiniCPMOcrVideoProcessor(MiniCPMV4_6VideoProcessor):
    downsample_mode = "4x"
    valid_kwargs = MiniCPMOcrVideoProcessorKwargs

    def __call__(
        self,
        videos: VideoInput,
        **kwargs: Unpack[MiniCPMOcrVideoProcessorKwargs],
    ) -> BatchFeature:
        # `downsample_mode` also selects the patch grid `find_best_resize` aligns to, and that grid
        # is read from the instance. Run the override on a shallow copy so the processor itself is
        # never mutated and concurrent calls cannot observe each other's mode.
        downsample_mode = kwargs.pop("downsample_mode", None)
        processor = self
        if downsample_mode is not None and downsample_mode != self.downsample_mode:
            processor = copy.copy(self)
            processor.downsample_mode = downsample_mode
        return processor.preprocess(videos, **kwargs)

    def find_best_resize(
        self,
        video_size: tuple[int, int],
        scale_resolution: int,
        patch_size: int,
        allow_upscale: bool = False,
    ) -> tuple[int, int]:
        height, width = video_size
        if (height * width > scale_resolution * scale_resolution) or allow_upscale:
            aspect_ratio = width / height
            height = int(scale_resolution / math.sqrt(aspect_ratio))
            width = int(height * aspect_ratio)
        # The resize grid follows the number of 2×2 spatial merges the visual tokens go through:
        # one in `"4x"` mode (downsample MLP), two in `"16x"` mode (ViT insert merger as well).
        merge_factor = 2 if self.downsample_mode == "4x" else 4
        best_height = ensure_divide(height, patch_size * merge_factor)
        best_width = ensure_divide(width, patch_size * merge_factor)
        return best_height, best_width


class MiniCPMOcrProcessor(MiniCPMV4_6Processor):
    pass


class MiniCPMOcrModel(MiniCPMV4_6Model):
    pass


class MiniCPMOcrForConditionalGeneration(MiniCPMV4_6ForConditionalGeneration):
    pass


__all__ = [
    "MiniCPMOcrConfig",
    "MiniCPMOcrVisionConfig",
    "MiniCPMOcrPreTrainedModel",  # noqa: F822
    "MiniCPMOcrVisionPreTrainedModel",  # noqa: F822
    "MiniCPMOcrVisionModel",  # noqa: F822
    "MiniCPMOcrModel",
    "MiniCPMOcrForConditionalGeneration",
    "MiniCPMOcrImageProcessor",
    "MiniCPMOcrImageProcessorPil",
    "MiniCPMOcrVideoProcessor",
    "MiniCPMOcrProcessor",
]
