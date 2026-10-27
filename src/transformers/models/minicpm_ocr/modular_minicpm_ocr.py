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


import numpy as np
import torch
from huggingface_hub.dataclasses import strict
from PIL.Image import Resampling as PILImageResampling

from ...feature_extraction_utils import BatchFeature
from ...image_transforms import divide_to_patches, group_images_by_shape, reorder_images
from ...image_utils import SizeDict
from ...utils import TensorType, auto_docstring, is_cv2_available, logging, requires_backends
from ..minicpmv4_6.configuration_minicpmv4_6 import MiniCPMV4_6Config, MiniCPMV4_6VisionConfig
from ..minicpmv4_6.image_processing_minicpmv4_6 import MiniCPMV4_6ImageProcessor, MiniCPMV4_6ImageProcessorKwargs
from ..minicpmv4_6.modeling_minicpmv4_6 import MiniCPMV4_6ForConditionalGeneration, MiniCPMV4_6Model
from ..minicpmv4_6.processing_minicpmv4_6 import MiniCPMV4_6Processor
from ..minicpmv4_6.video_processing_minicpmv4_6 import MiniCPMV4_6VideoProcessor, MiniCPMV4_6VideoProcessorKwargs


if is_cv2_available():
    import cv2


logger = logging.get_logger(__name__)

# Calibration of the glyph classifier the checkpoint was trained with
SMALL_FONT_PX = 12.0
DENSE_GLYPHS_PER_KPX = 2.0
TINY_PIXELS = 6272
TINY_RESOLUTION = 84


def adaptive_upsize_resolution(image: "np.ndarray", scale_resolution: int) -> int | None:
    r"""
    Resolution to upscale an image (or slice) no larger than `scale_resolution` squared towards, or `None` to keep
    it native. Small dense text gets `scale_resolution`, tiny images get `TINY_RESOLUTION`. Glyph height and density
    come from an Otsu threshold and 8-connected components, matching the classifier the checkpoint was trained with,
    so OpenCV is required.

    Args:
        image (`np.ndarray`):
            `uint8` image shaped `(num_channels, height, width)` or `(height, width)`.
        scale_resolution (`int`):
            Resolution small dense text is upscaled to.
    """
    requires_backends(adaptive_upsize_resolution, ["cv2"])

    height, width = image.shape[-2:]
    if height <= 0 or width <= 0:
        return None

    if image.ndim == 3 and image.shape[0] >= 3:
        # ITU-R 601-2 luma, matching the integer arithmetic of `PIL.Image.convert("L")`
        image = image.astype(np.int32)
        gray = ((image[0] * 19595 + image[1] * 38470 + image[2] * 7471 + 32768) >> 16).astype(np.uint8)
    else:
        gray = image.reshape(height, width)

    _, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    foreground = binarized == 0
    if foreground.mean() > 0.5:
        # Assume glyphs cover less than half the image, so flip light text on a dark page
        foreground = ~foreground

    # TODO: replace cv2.connectedComponentsWithStats with a pure-torch kernel once
    # the transformers kernel library (@molbap) provides a connected-components implementation.
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
    if 0 < font_px <= SMALL_FONT_PX and density >= DENSE_GLYPHS_PER_KPX:
        return scale_resolution
    if height * width < TINY_PIXELS:
        return TINY_RESOLUTION
    return None


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
    image_size = AttributeError()


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
    small_image_upsize (`str`, *optional*, defaults to `"always"`):
        How images and slices no larger than `scale_resolution` squared are resized. `"always"` upscales them to
        `scale_resolution`, `"never"` keeps their native resolution aligned to the patch grid, and `"adaptive"`
        picks per image and per slice from glyph size and density (requires OpenCV, see
        [`adaptive_upsize_resolution`]).
    """

    small_image_upsize: str


class MiniCPMOcrImageProcessor(MiniCPMV4_6ImageProcessor):
    downsample_mode = "4x"
    small_image_upsize = "always"
    valid_kwargs = MiniCPMOcrImageProcessorKwargs

    def _resolve_tile_size(
        self,
        image: "torch.Tensor",
        grid: tuple[int, int],
        row: int,
        column: int,
        base_tile: "torch.Tensor",
        scale_resolution: int,
        patch_size: int,
        downsample_mode: str,
    ) -> "torch.Tensor":
        """Crop cell (row, column) of the original image and return it at its adaptive resolution when that costs
        fewer tokens than `base_tile`, otherwise return `base_tile`."""
        height, width = image.shape[-2:]
        grid_rows, grid_columns = grid
        cell = image[
            ...,
            round(height * row / grid_rows) : round(height * (row + 1) / grid_rows),
            round(width * column / grid_columns) : round(width * (column + 1) / grid_columns),
        ]
        cell_height, cell_width = cell.shape[-2:]
        if cell_height < 1 or cell_width < 1 or cell_height * cell_width > scale_resolution**2:
            return base_tile

        resolution = adaptive_upsize_resolution(cell.to("cpu", torch.uint8).numpy(), scale_resolution)
        if resolution == scale_resolution:
            return base_tile
        target_height, target_width = self.find_best_resize(
            (cell_height, cell_width),
            resolution or scale_resolution,
            patch_size,
            allow_upscale=resolution is not None,
            downsample_mode=downsample_mode,
        )
        if target_height * target_width >= base_tile.shape[-2] * base_tile.shape[-1]:
            return base_tile
        if (target_height, target_width) == (cell_height, cell_width):
            return cell
        return self.resize(
            cell, size=SizeDict(height=target_height, width=target_width), resample=PILImageResampling.BICUBIC
        )

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
        small_image_upsize: str,
        return_tensors: str | TensorType | None,
        disable_grouping: bool | None,
        downsample_mode: str = "4x",
        **kwargs,
    ) -> BatchFeature:
        if not do_resize:
            raise ValueError(
                "do_resize=False is not supported. MiniCPM-OCR always resizes images "
                "to align them to the patch grid; set do_resize=True."
            )
        all_pixel_values: list[torch.Tensor] = []
        all_target_sizes: list[list[int]] = []
        all_grids: list[list[int]] = []
        num_patches_per_image: list[int] = []

        for image in images:
            image_size = image.shape[-2:]
            best_grid = self.get_sliced_grid(image_size, max_slice_nums, scale_resolution) if slice_mode else None

            # Unsliced images may keep, or reduce, their native resolution; sliced overviews always use scale_resolution
            source_resolution, allow_upscale = scale_resolution, best_grid is None and small_image_upsize == "always"
            if best_grid is None and small_image_upsize == "adaptive":
                resolution = adaptive_upsize_resolution(image.to("cpu", torch.uint8).numpy(), scale_resolution)
                source_resolution, allow_upscale = resolution or scale_resolution, resolution is not None
            source_h, source_w = self.find_best_resize(
                image_size, source_resolution, patch_size, allow_upscale=allow_upscale, downsample_mode=downsample_mode
            )
            source_img = self.resize(image, size=SizeDict(height=source_h, width=source_w), resample=resample)

            # Collect all patches for this image: [source, *slices]
            image_patches: list[torch.Tensor] = [source_img]
            patch_height = patch_width = 0
            if best_grid is not None:
                refine_h, refine_w = self.get_refine_size(
                    image_size,
                    best_grid,
                    scale_resolution,
                    patch_size,
                    allow_upscale=True,
                    downsample_mode=downsample_mode,
                )
                refine_img = self.resize(image, size=SizeDict(height=refine_h, width=refine_w), resample=resample)
                refine_height, refine_width = refine_img.shape[-2:]
                grid_y, grid_x = best_grid
                patch_height = refine_height // grid_y
                patch_width = refine_width // grid_x

                slices = divide_to_patches(refine_img, (patch_height, patch_width))
                if small_image_upsize == "adaptive":
                    # Per-slice adaptive resolution, never above the baseline slice budget
                    slices = [
                        self._resolve_tile_size(
                            image,
                            best_grid,
                            i // grid_x,
                            i % grid_x,
                            tile,
                            scale_resolution,
                            patch_size,
                            downsample_mode,
                        )
                        for i, tile in enumerate(slices)
                    ]
                image_patches.extend(slices)

            # Group patches by shape and batch rescale + normalize
            grouped_patches, grouped_index = group_images_by_shape(image_patches, disable_grouping=disable_grouping)
            processed_grouped = {}
            for shape, stacked in grouped_patches.items():
                processed_grouped[shape] = self.rescale_and_normalize(
                    stacked.float(), do_rescale, rescale_factor, do_normalize, image_mean, image_std
                )
            processed_patches = reorder_images(processed_grouped, grouped_index)

            all_pixel_values.append(self.reshape_by_patch(processed_patches[0], patch_size))
            all_target_sizes.append([source_h // patch_size, source_w // patch_size])
            for processed_slice in processed_patches[1:]:
                all_pixel_values.append(self.reshape_by_patch(processed_slice, patch_size))
                all_target_sizes.append(
                    [processed_slice.shape[-2] // patch_size, processed_slice.shape[-1] // patch_size]
                )
            num_patches_per_image.append(len(processed_patches))
            all_grids.append(best_grid if best_grid is not None else [0, 0])

        pixel_values = torch.cat(all_pixel_values, dim=-1).unsqueeze(0)
        target_sizes = torch.tensor(all_target_sizes, dtype=torch.int32)

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
        which applies the full merge.
    use_image_id (`bool`, *optional*, defaults to `True`):
        Whether to prepend an image-id tag (``<image_id>N</image_id>``) before
        each image placeholder. Consumed by the Processor for placeholder
        generation, not by the image processing pipeline itself.
    """


class MiniCPMOcrVideoProcessor(MiniCPMV4_6VideoProcessor):
    downsample_mode = "4x"


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
    "MiniCPMOcrVideoProcessor",
    "MiniCPMOcrProcessor",
]
