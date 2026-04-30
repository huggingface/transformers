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
"""Video processor for MiniCPM-V 4.6.

MiniCPM-V treats video as a sequence of images: frames are extracted and
optionally sub-second frames are stacked into composite images.
"""

import math
from functools import partial

import numpy as np
from huggingface_hub.dataclasses import validate_typed_dict

from ...image_processing_utils import BatchFeature
from ...image_transforms import divide_to_patches
from ...image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD, PILImageResampling, SizeDict, validate_kwargs
from ...processing_utils import Unpack, VideosKwargs
from ...utils import TensorType, add_start_docstrings, is_torch_available, logging
from ...video_processing_utils import BASE_VIDEO_PROCESSOR_DOCSTRING, BaseVideoProcessor
from ...video_utils import (
    VideoInput,
    VideoMetadata,
)


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


def ensure_divide(length: int, divisor: int) -> int:
    return max(round(length / divisor) * divisor, divisor)


class MiniCPMV4_6VideoProcessorKwargs(VideosKwargs, total=False):
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
    downsample_mode (`str`, *optional*, defaults to `"16x"`):
        Visual token downsampling mode. `"16x"` applies full merge; `"4x"` keeps
        4x more tokens.
    use_image_id (`bool`, *optional*, defaults to `True`):
        Whether to prepend an image-id tag (``<image_id>N</image_id>``) before
        each image placeholder. Consumed by the Processor for placeholder
        generation, not by the image processing pipeline itself.
    """

    max_num_frames: int
    stack_frames: int
    max_slice_nums: int
    scale_resolution: int
    patch_size: int
    slice_mode: bool
    downsample_mode: str
    use_image_id: bool


@add_start_docstrings(
    "Constructs a MiniCPM-V 4.6 video processor.",
    BASE_VIDEO_PROCESSOR_DOCSTRING,
)
class MiniCPMV4_6VideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BICUBIC
    do_resize = True
    do_rescale = True
    do_normalize = True
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    do_convert_rgb = True
    max_slice_nums = 9
    scale_resolution = 448
    patch_size = 14
    slice_mode = True
    downsample_mode = "16x"
    use_image_id = True
    do_sample_frames = True
    max_num_frames = 128
    stack_frames = 1
    valid_kwargs = MiniCPMV4_6VideoProcessorKwargs
    model_input_names = ["pixel_values_videos", "target_sizes_videos"]

    def __init__(self, **kwargs: Unpack[MiniCPMV4_6VideoProcessorKwargs]):
        super().__init__(**kwargs)

    def _validate_preprocess_kwargs(self, **kwargs):
        # Drop `do_resize`, model resizes based on auto-inferred size at run-time
        kwargs.pop("do_resize")
        super()._validate_preprocess_kwargs(**kwargs)

    def sample_frames(
        self, metadata: VideoMetadata, max_num_frames: int | None = None, stack_frames: int | None = None, **kwargs
    ):
        """
        Args:
            metadata (`VideoMetadata`):
                Metadata of the video containing information about total duration, fps and total number of frames.
            max_num_frames (`int`, *optional*):
                The maximum number of frames that can be sampled.
            stack_frames (`int`, *optional*):
                Sub-frames per second to stack. Value of `1` disables stacking.
        Returns:
            np.ndarray:
                Indices to sample video frames.
        """
        if metadata is None or metadata.duration is None or metadata.fps is None:
            raise ValueError(
                "MiniCPMV4_6 requires complete video metadata with `duration` and `fps` to sample frames. "
                "Please pass a complete `VideoMetadata` object or set `do_sample_frames=False`."
            )

        max_num_frames = max_num_frames if max_num_frames is not None else self.max_num_frames
        stack_frames = stack_frames if stack_frames is not None else self.stack_frames
        total_num_frames, avg_fps = metadata.total_num_frames, metadata.fps
        duration = metadata.duration

        num_seconds = math.ceil(duration)

        is_video_long = duration > max_num_frames
        if is_video_long:
            timestamps = [round(i * 0.1, 1) for i in range(int(duration / 0.1))]
            main_indices = [min(int(ts * avg_fps), total_num_frames - 1) for ts in timestamps]
            # Sample frames to keep the total length at `max_num_frames`
            if len(main_indices) > max_num_frames:
                sampling_idxs = np.linspace(0, len(main_indices) - 1, max_num_frames, dtype=int)
                main_indices = [main_indices[i] for i in sampling_idxs]
        else:
            main_indices = [int(i * avg_fps) for i in range(num_seconds)]

        indices_total = main_indices

        if stack_frames and stack_frames > 1:
            sub_timestamps = []
            for sec in range(num_seconds):
                for j in range(1, stack_frames):
                    timestamp = sec + j / stack_frames
                    if timestamp < duration:
                        sub_timestamps.append(timestamp)
            sub_indices = [min(int(timestamp * avg_fps), total_num_frames - 1) for timestamp in sub_timestamps]

            max_num_frames_stack = max_num_frames * (stack_frames - 1)
            if len(sub_indices) > max_num_frames_stack:
                # Sample frames to keep the total length at `max_num_frames`
                sampling_idxs = np.linspace(0, len(sub_indices) - 1, max_num_frames_stack, dtype=int)
                sub_indices = [sub_indices[i] for i in sampling_idxs]
            indices_total += sub_indices

        return indices_total

    def concat_frames_as_image(self, video: torch.Tensor) -> torch.Tensor:
        """
        Takes a video and concatenates it to a PIL canvas following
        a grid layout. Depending on video's size, the output (rows, cols)
        arrangement is picked automatically.
        """
        line_width = 6
        num_frames, channels, height, width = video.shape

        def canvas_ratio(rows, cols):
            canvas_width = cols * width + (cols - 1) * line_width
            canvas_height = rows * height + (rows - 1) * line_width
            return canvas_width / max(1, canvas_height)

        if num_frames == 4:
            rows, cols = 2, 2
        elif num_frames == 3:
            candidates = [(1, 3), (3, 1)]
            ratios = [abs(canvas_ratio(rows, cols) - 1.0) for rows, cols in candidates]
            rows, cols = candidates[int(np.argmin(ratios))]
        elif num_frames == 2:
            candidates = [(1, 2), (2, 1)]
            ratios = [abs(canvas_ratio(rows, cols) - 1.0) for rows, cols in candidates]
            if ratios[0] == ratios[1]:
                rows, cols = (1, 2) if width / height >= 1.0 else (2, 1)
            else:
                rows, cols = candidates[int(np.argmin(ratios))]
        else:
            rows, cols = 1, num_frames

        # Create a big canvas to fit in all time-frames
        canvas_width = cols * width + (cols - 1) * line_width
        canvas_height = rows * height + (rows - 1) * line_width
        canvas = torch.zeros((1, channels, canvas_height, canvas_width), dtype=torch.uint8)
        video = video.view(rows, cols, channels, height, width)

        for row in range(rows):
            for col in range(cols):
                h_start = row * (height + line_width)
                w_start = col * (width + line_width)
                canvas[..., h_start : h_start + height, w_start : w_start + width] = video[row, col]

        return canvas

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
        # factor 4 = two successive 2×2 spatial merges (ViT insert merger + downsample MLP)
        best_height = ensure_divide(height, patch_size * 4)
        best_width = ensure_divide(width, patch_size * 4)
        return best_height, best_width

    def get_refine_size(
        self,
        video_size: tuple[int, int],
        grid: list[int],
        scale_resolution: int,
        patch_size: int,
        allow_upscale: bool = False,
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
        )
        return best_height * grid_y, best_width * grid_x

    def get_sliced_grid(
        self,
        video_size: tuple[int, int],
        max_slice_nums: int,
        scale_resolution: int,
    ) -> list[int] | None:
        original_height, original_width = video_size
        log_ratio = math.log(original_width / original_height)
        ratio = original_width * original_height / (scale_resolution * scale_resolution)
        multiple = min(math.ceil(ratio), max_slice_nums)
        if multiple <= 1:
            return None

        best_grid = [1, 1]
        min_error = float("inf")
        for num_slices in [multiple - 1, multiple, multiple + 1]:
            if num_slices == 1 or num_slices > max_slice_nums:
                continue
            for num_rows in range(1, num_slices + 1):
                if num_slices % num_rows == 0:
                    num_cols = num_slices // num_rows
                    error = abs(log_ratio - math.log(num_rows / num_cols))
                    if error < min_error:
                        best_grid = [num_cols, num_rows]
                        min_error = error
        return best_grid

    def reshape_by_patch(self, videos: "torch.Tensor", patch_size: int) -> "torch.Tensor":
        "Reshape ``[B, T, C, H, W]`` into NaViT patchified format ``[B, T, C, patch_size, H*W/patch_size]``."
        batch, time, num_channels, height, width = videos.shape

        # merge B and T so unfold sees 4D (B*T, C, H, W)
        videos = videos.reshape(batch * time, num_channels, height, width)
        patches = torch.nn.functional.unfold(videos, (patch_size, patch_size), stride=(patch_size, patch_size))

        patches = patches.reshape(batch, time, num_channels, patch_size, patch_size, -1)
        patches = patches.permute(0, 1, 2, 3, 5, 4)  # (B, T, C, patch_size, num_patches, patch_size)
        patches = patches.reshape(batch, time, num_channels, patch_size, -1)  # (B, T, C, patch_size, H*W/patch_size)

        return patches

    @add_start_docstrings(
        BASE_VIDEO_PROCESSOR_DOCSTRING,
    )
    def preprocess(
        self,
        videos: VideoInput,
        **kwargs: Unpack[VideosKwargs],
    ) -> BatchFeature:
        validate_kwargs(
            captured_kwargs=kwargs.keys(),
            valid_processor_keys=list(self.valid_kwargs.__annotations__.keys()) + ["return_tensors"],
        )

        # Perform type validation on received kwargs
        validate_typed_dict(self.valid_kwargs, kwargs)

        # Set default kwargs from self. This ensures that if a kwarg is not provided
        # by the user, it gets its default value from the instance, or is set to None.
        for kwarg_name in self.valid_kwargs.__annotations__:
            kwargs.setdefault(kwarg_name, getattr(self, kwarg_name, None))

        input_data_format = kwargs.pop("input_data_format")
        do_sample_frames = kwargs.pop("do_sample_frames")
        device = kwargs.pop("device")
        video_metadata = kwargs.pop("video_metadata")

        sample_indices_fn = partial(self.sample_frames, **kwargs) if do_sample_frames else None
        videos, video_metadata = self._decode_and_sample_videos(
            videos,
            video_metadata=video_metadata,
            do_sample_frames=do_sample_frames,
            sample_indices_fn=sample_indices_fn,
        )
        videos = self._prepare_input_videos(
            videos=videos,
            input_data_format=input_data_format,
            device=device,
        )

        kwargs = self._standardize_kwargs(**kwargs)
        self._validate_preprocess_kwargs(**kwargs)

        # Pop kwargs that are not needed in _preprocess
        kwargs.pop("data_format")
        return_metadata = kwargs.pop("return_metadata")

        # Diff from base class, pass on `video_metadata` to infer subframes vs main frames
        preprocessed_videos = self._preprocess(videos=videos, video_metadata=video_metadata, **kwargs)
        if return_metadata:
            preprocessed_videos["video_metadata"] = video_metadata
        return preprocessed_videos

    def resize_and_split_patches(
        self,
        video: "torch.Tensor",
        resample,
        slice_mode: bool,
        max_slice_nums: int,
        scale_resolution: int,
        patch_size: int,
    ):
        video_size = video.shape[-2:]
        best_grid = None

        if slice_mode:
            best_grid = self.get_sliced_grid(video_size, max_slice_nums, scale_resolution)

        # Always resize the source
        new_height, new_width = self.find_best_resize(
            video_size, scale_resolution, patch_size, allow_upscale=(best_grid is None)
        )
        source_video = self.resize(video, size=SizeDict(height=new_height, width=new_width), resample=resample)

        # Collect all patches: [source, *slices]
        patches = [source_video]
        if best_grid is not None:
            refine_height, refine_width = self.get_refine_size(
                video_size, best_grid, scale_resolution, patch_size, allow_upscale=True
            )
            grid_y, grid_x = best_grid
            patch_height, patch_width = refine_height // grid_y, refine_width // grid_x

            refine_video = self.resize(
                video, size=SizeDict(height=refine_height, width=refine_width), resample=resample
            )
            refine_video = divide_to_patches(refine_video, (patch_height, patch_width))
            patches.extend(refine_video)
        grid = best_grid if best_grid is not None else (0, 0)
        return patches, grid

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
        stack_frames: int,
        max_num_frames: int,
        video_metadata: list[VideoMetadata],
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
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
        per_unit_pixel_values: list[list[torch.Tensor]] = []
        per_unit_target_sizes: list[list[list[int]]] = []
        all_grids: list[list[int]] = []

        for unit in visual_units:
            if do_resize:
                patches, grid = self.resize_and_split_patches(
                    video=unit,
                    resample=resample,
                    slice_mode=slice_mode,
                    max_slice_nums=max_slice_nums,
                    scale_resolution=scale_resolution,
                    patch_size=patch_size,
                )
            else:
                patches = [unit]
                grid = (0, 0)

            unit_pv: list[torch.Tensor] = []
            unit_ts: list[list[int]] = []
            for patch in patches:
                patch = self.rescale_and_normalize(
                    patch,
                    do_rescale,
                    rescale_factor,
                    do_normalize,
                    image_mean,
                    image_std,
                )
                height, width = patch.shape[-2:]
                patch = self.reshape_by_patch(patch.unsqueeze(0), patch_size)
                unit_pv.append(patch.squeeze(0).squeeze(0))
                unit_ts.append([height // patch_size, width // patch_size])

            per_unit_pixel_values.append(unit_pv)
            per_unit_target_sizes.append(unit_ts)
            all_grids.append(list(grid) if not isinstance(grid, list) else grid)

        # Stage 3 — Flatten into NaViT-packed format.
        all_pv = [pv for unit_pvs in per_unit_pixel_values for pv in unit_pvs]
        pixel_values = torch.cat(all_pv, dim=-1).unsqueeze(0)

        all_ts = [ts for unit_tss in per_unit_target_sizes for ts in unit_tss]
        target_sizes = torch.tensor(all_ts, dtype=torch.int32)

        num_patches_per_frame = [len(unit_pvs) for unit_pvs in per_unit_pixel_values]

        return BatchFeature(
            data={
                "pixel_values_videos": pixel_values,
                "target_sizes_videos": target_sizes,
                "grids_videos": all_grids,
                "num_patches_per_frame": num_patches_per_frame,
                "num_frames_per_video": num_frames_per_video,
            },
            tensor_type=return_tensors,
            skip_tensor_conversion=["grids_videos", "num_patches_per_frame", "num_frames_per_video"],
        )


__all__ = ["MiniCPMV4_6VideoProcessor"]
