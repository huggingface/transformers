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
from ..minicpmv4_6.video_processing_minicpmv4_6 import MiniCPMV4_6VideoProcessor


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
    downsample_mode (`str`, *optional*, defaults to `"4x"`):
        Visual token downsampling ratio. `"4x"` keeps 4× more tokens than `"16x"`, which is what
        text-dense document images are read with.
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
        downsample_mode: str,
        small_image_upsize: str,
        return_tensors: str | TensorType | None,
        disable_grouping: bool | None,
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

                if use_adaptive_tile:
                    # Per-tile adaptive resolution: classify each cell on the original
                    # image crop and use the cheapest resolution that does not exceed
                    # the baseline tile budget.
                    base_patches = divide_to_patches(refine_img, (patch_height, patch_width))
                    for row in range(grid_y):
                        for col in range(grid_x):
                            base_tile = base_patches[row * grid_x + col]
                            image_patches.append(
                                self._resolve_tile_size(
                                    image,
                                    image_size,
                                    (grid_y, grid_x),
                                    row,
                                    col,
                                    base_tile,
                                    scale_resolution,
                                    patch_size,
                                    downsample_mode,
                                )
                            )
                else:
                    image_patches.extend(divide_to_patches(refine_img, (patch_height, patch_width)))

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
    valid_kwargs = MiniCPMOcrVideoProcessorKwargs

    def find_best_resize(
        self,
        video_size: tuple[int, int],
        scale_resolution: int,
        patch_size: int,
        allow_upscale: bool = False,
        downsample_mode: str = "4x",
    ) -> tuple[int, int]:
        height, width = video_size
        if (height * width > scale_resolution * scale_resolution) or allow_upscale:
            aspect_ratio = width / height
            height = int(scale_resolution / math.sqrt(aspect_ratio))
            width = int(height * aspect_ratio)

        merge_factor = 2 if downsample_mode == "4x" else 4
        best_height = ensure_divide(height, patch_size * merge_factor)
        best_width = ensure_divide(width, patch_size * merge_factor)
        return best_height, best_width

    def get_refine_size(
        self,
        video_size: tuple[int, int],
        grid: list[int],
        scale_resolution: int,
        patch_size: int,
        allow_upscale: bool = False,
        downsample_mode: str = "4x",
    ) -> tuple[int, int]:
        height, width = video_size
        grid_y, grid_x = grid
        refine_width = ensure_divide(width, grid_x)
        refine_height = ensure_divide(height, grid_y)

        best_height, best_width = self.find_best_resize(
            video_size=(refine_height / grid_y, refine_width / grid_x),
            scale_resolution=scale_resolution,
            patch_size=patch_size,
            allow_upscale=allow_upscale,
            downsample_mode=downsample_mode,
        )
        return best_height * grid_y, best_width * grid_x

    def resize_and_split_patches(
        self,
        video: "torch.Tensor",
        resample,
        slice_mode: bool,
        max_slice_nums: int,
        scale_resolution: int,
        patch_size: int,
        downsample_mode: str = "4x",
    ):
        num_frames = video.shape[1]
        video_size = video.shape[-2:]
        best_grid = None

        if slice_mode:
            best_grid = self.get_sliced_grid(video_size, max_slice_nums, scale_resolution)

        # Always resize the source
        new_height, new_width = self.find_best_resize(
            video_size,
            scale_resolution,
            patch_size,
            allow_upscale=(best_grid is None),
            downsample_mode=downsample_mode,
        )
        source_videos = self.resize(video, size=SizeDict(height=new_height, width=new_width), resample=resample)

        # Collect all patches: [source, *slices]
        patches = [source_videos]
        if best_grid is not None:
            refine_height, refine_width = self.get_refine_size(
                video_size,
                best_grid,
                scale_resolution,
                patch_size,
                allow_upscale=True,
                downsample_mode=downsample_mode,
            )
            grid_y, grid_x = best_grid
            patch_height, patch_width = refine_height // grid_y, refine_width // grid_x

            refine_videos = self.resize(
                video, size=SizeDict(height=refine_height, width=refine_width), resample=resample
            )
            refine_videos = divide_to_patches(refine_videos, (patch_height, patch_width))
            patches.extend(refine_videos)

        # Reorder from `all_sources+all_patches` to represent each video as `source+patch`
        patches_grouped_by_batch = [
            [patches[patch][batch] for patch in range(len(patches))] for batch in range(len(video))
        ]

        # MiniCPM needs to process each video as a single frame instead of processing all together
        # So each video is represented as `source-frame, patch-frame, patch-frame, [...], source-frame, patch-frame, [...]`
        interleaved_frames = []
        for sublist in patches_grouped_by_batch:
            # Split all tensors into frames: each becomes T x (1, 3, H, W)
            all_frames = [patch.unsqueeze(1).unbind(0) for patch in sublist]
            # Interleave: for each timestep, yield source then all patch frames
            interleaved_frames.append([frame for t in zip(*all_frames) for frame in t])

        grid = best_grid if best_grid is not None else (0, 0)
        num_patches = (grid[0] * grid[1]) + 1
        # Expand by batch size per each video
        num_patches = [[num_patches] * num_frames] * len(video)
        grids = [[grid] * num_frames] * len(video)
        return interleaved_frames, grids, num_patches

    def _preprocess(
        self,
        videos: list[torch.Tensor],
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
        downsample_mode: str,
        stack_frames: int,
        max_num_frames: int,
        video_metadata: list[VideoMetadata],
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        # Different from MiniCPM-V 4.6 as we also pass `downsample_mode`.
        # Stage 1 — Build an ordered list of visual units.
        # Each unit is a [1, C, H, W] tensor representing either a single main
        # frame or a per-second composite of sub-frames.  When stack_frames > 1
        # the units are interleaved: [main_0, comp_0, main_1, comp_1, …]
        visual_units: list[torch.Tensor] = []
        num_frames_per_video: list[int] = []
        for video, metadata in zip(videos, video_metadata):
            units_before = len(visual_units)
            if stack_frames > 1:
                duration = metadata.duration
                num_seconds = math.ceil(duration)

                # Reconstruct sub_timestamps (same logic as sample_frames)
                sub_timestamps: list[float] = []
                for sec in range(num_seconds):
                    for j in range(1, stack_frames):
                        timestamp = sec + j / stack_frames
                        if timestamp < duration:
                            sub_timestamps.append(timestamp)

                # Apply the same downsampling that sample_frames would have applied
                max_num_frames_stack = max_num_frames * (stack_frames - 1)
                if len(sub_timestamps) > max_num_frames_stack:
                    sampling_idxs = np.linspace(0, len(sub_timestamps) - 1, max_num_frames_stack, dtype=int)
                    sub_timestamps = [sub_timestamps[int(i)] for i in sampling_idxs]

                n_sub = len(sub_timestamps)
                if n_sub > 0:
                    main_video, sub_video = video[:-n_sub], video[-n_sub:]
                else:
                    main_video, sub_video = video, video[:0]

                # Group sub-frames by second (matching _group_stacked_by_second)
                composites_by_sec: list[torch.Tensor | None] = []
                cursor = 0
                for sec in range(num_seconds):
                    group_start = cursor
                    while cursor < len(sub_timestamps) and sub_timestamps[cursor] < sec + 1:
                        cursor += 1
                    if cursor > group_start:
                        composites_by_sec.append(self.concat_frames_as_image(sub_video[group_start:cursor]))
                    else:
                        composites_by_sec.append(None)

                # Interleave: pair i-th main frame with i-th second's composite
                for i in range(len(main_video)):
                    visual_units.append(main_video[i : i + 1])
                    if i < len(composites_by_sec) and composites_by_sec[i] is not None:
                        visual_units.append(composites_by_sec[i])
            else:
                for i in range(len(video)):
                    visual_units.append(video[i : i + 1])
            num_frames_per_video.append(len(visual_units) - units_before)

        # Stage 2 — Resize, split, normalise and reshape each unit independently.
        grouped_videos, grouped_videos_index = group_videos_by_shape(visual_units)
        resized_videos_grouped = {}
        videos_grids = {}
        processed_num_patches_per_frame = {}
        for shape, stacked_videos in grouped_videos.items():
            batch_size, num_frames = stacked_videos.shape[:2]
            if do_resize:
                stacked_videos, grids, num_patches = self.resize_and_split_patches(
                    stacked_videos,
                    resample=resample,
                    slice_mode=slice_mode,
                    max_slice_nums=max_slice_nums,
                    scale_resolution=scale_resolution,
                    patch_size=patch_size,
                    downsample_mode=downsample_mode,
                )
            else:
                stacked_videos = [[video] for video in stacked_videos]
                grids = [[(0, 0)] * num_frames] * len(stacked_videos)
                num_patches = [[1] * num_frames] * batch_size
            resized_videos_grouped[shape] = stacked_videos
            videos_grids[shape] = grids
            processed_num_patches_per_frame[shape] = num_patches

        # Regroup back and flatten list
        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)
        resized_videos = [patch for patch_list in resized_videos for patch in patch_list]
        videos_grids = reorder_videos(videos_grids, grouped_videos_index)
        num_patches_per_frame = reorder_videos(processed_num_patches_per_frame, grouped_videos_index)

        # Group videos by size for further processing
        # Needed in case do_resize is False, or resize returns videos with different sizes
        grouped_videos, grouped_videos_index = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        processed_video_sizes = {}
        for shape, stacked_videos in grouped_videos.items():
            # Fused rescale and normalize
            stacked_videos = self.rescale_and_normalize(
                stacked_videos, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            patches = stacked_videos
            batch_size, time, channel, height, width = patches.shape
            patches = self.reshape_by_patch(patches, patch_size)  # [B, T, C, patch_size, H*W/patch_size]

            processed_videos_grouped[shape] = patches.permute(0, 2, 3, 4, 1).flatten(-2)
            processed_video_sizes[shape] = [[[height // patch_size, width // patch_size]] * time] * batch_size

        processed_videos = reorder_videos(processed_videos_grouped, grouped_videos_index)
        video_sizes = reorder_videos(processed_video_sizes, grouped_videos_index)

        # Stage 3 — Flatten into NaViT-packed format.
        pixel_values = torch.cat(processed_videos, dim=-1).unsqueeze(0)
        target_sizes = torch.tensor(video_sizes, dtype=torch.int32).reshape(-1, 2)
        videos_grids = torch.tensor(videos_grids, dtype=torch.int32).reshape(-1, 2)
        num_patches_per_frame = torch.tensor(num_patches_per_frame, dtype=torch.int32).flatten()

        return BatchFeature(
            data={
                "pixel_values_videos": pixel_values,
                "target_sizes_videos": target_sizes,
                "grids_videos": videos_grids,
                "num_patches_per_frame": num_patches_per_frame,
                "num_frames_per_video": num_frames_per_video,
            },
            tensor_type=return_tensors,
            skip_tensor_conversion=["grids_videos", "num_patches_per_frame", "num_frames_per_video"],
        )


class MiniCPMOcrProcessor(MiniCPMV4_6Processor):
    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        # Unlike the parent, slices carry individual token counts: `adaptive_tile_upsize` resizes
        # each tile independently, so a uniform count would desync text tokens from `pixel_values`.
        image_grids = image_inputs["grids"]
        num_patches_per_image = image_inputs["num_patches_per_image"]
        target_sizes = image_inputs["target_sizes"]

        cum_patches = np.cumsum(num_patches_per_image)
        start_idx = cum_patches[image_idx - 1] if image_idx > 0 else 0
        end_idx = cum_patches[image_idx]
        img_target_sizes = target_sizes[start_idx:end_idx]
        num_tokens_per_patch = img_target_sizes.prod(-1) // kwargs["image_token_divisor"]
        num_rows, num_cols = image_grids[image_idx]

        image_placeholder = (
            self.image_start_token + self.image_token * int(num_tokens_per_patch[0]) + self.image_end_token
        )

        if self.slice_mode and num_rows > 0 and num_cols > 0:
            slice_tokens = [int(n) for n in num_tokens_per_patch[1:]]
            if len(slice_tokens) != num_rows * num_cols:
                raise ValueError(f"Image {image_idx}: got {len(slice_tokens)} slices for a {num_rows}x{num_cols} grid")
            slices = []
            for row in range(num_rows):
                row_tokens = slice_tokens[row * num_cols : (row + 1) * num_cols]
                slices.append(
                    "".join(self.slice_start_token + self.image_token * n + self.slice_end_token for n in row_tokens)
                )
            image_placeholder += "\n".join(slices)

        return image_placeholder


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
