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
from ...image_transforms import divide_to_patches
from ...image_processing_utils import BatchFeature
from ...image_utils import SizeDict, IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD, PILImageResampling, validate_kwargs
from ...processing_utils import Unpack, VideosKwargs
from ...utils import TensorType, add_start_docstrings, is_torch_available, logging
from ...video_processing_utils import BASE_VIDEO_PROCESSOR_DOCSTRING, BaseVideoProcessor
from ...video_utils import (
    VideoInput,
    VideoMetadata,
    group_videos_by_shape,
)


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class MiniCPMV4_6VideoProcessorKwargs(VideosKwargs, total=False):
    r"""
    max_frames (`int`, *optional*, defaults to 128):
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

    max_frames: int
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
    do_resize = False
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
    max_frames = 128
    stack_frames = 1
    valid_kwargs = MiniCPMV4_6VideoProcessorKwargs
    model_input_names = ["pixel_values_videos", "target_sizes_videos"]

    def __init__(self, **kwargs: Unpack[MiniCPMV4_6VideoProcessorKwargs]):
        super().__init__(**kwargs)

    def sample_frames(self,metadata: VideoMetadata,max_frames: int | None = None,stack_frames: int | None = None, **kwargs):
        """
        Args:
            metadata (`VideoMetadata`):
                Metadata of the video containing information about total duration, fps and total number of frames.
            max_frames (`int`, *optional*):
                The maximum number of frames that can be sampled.
            stack_frames (`int`, *optional*):
                Sub-frames per second to stack. Value of `1` disables stacking.
        Returns:
            np.ndarray:
                Indices to sample video frames.
        """
        if metadata is None:
            raise ValueError(
                "MiniCpm4_6 requires video metadata to sample frames but it wasn't found. "
                "Please pass in `VideoMetadata` object or set `do_sample_frames=False`"
            )

        max_frames = max_frames if max_frames is not None else self.max_frames
        stack_frames = stack_frames if stack_frames is not None else self.stack_frames
        total_num_frames, avg_fps = metadata.total_num_frames, metadata.fps
        duration = metadata.duration
        num_seconds = math.ceil(duration)

        is_video_long = duration > max_frames
        if is_video_long:
            timestamps = [round(i * 0.1, 1) for i in range(int(duration / 0.1))]
            main_indices = [min(int(ts * avg_fps), total_num_frames - 1) for ts in timestamps]
            # Sample frames to keep the total length at `max_frames`
            if len(main_indices) > max_frames:
                sampling_idxs = np.linspace(0, len(main_indices) - 1, max_frames, dtype=int)
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

            max_frames_stack = max_frames * (stack_frames - 1)
            if len(sub_indices) > max_frames_stack:
                # Sample frames to keep the total length at `max_frames`
                sampling_idxs = np.linspace(0, len(sub_indices) - 1, max_frames_stack, dtype=int)
                sub_indices = [sub_indices[i] for i in sampling_idxs]
            indices_total += sub_indices

        return indices_total

    def concat_frames_as_image(videos: torch.Tensor) -> torch.Tensor:
        """
        Takes a batch of videos and concatenates each subsampled video
        to a PIL canvas following a grid layout. Depending on video's size,
        the output (rows, cols) arrangement is picked automatically.
        """
        line_width = 6
        num_videos, _, channels, height, width = videos.shape

        def canvas_ratio(rows, cols):
            canvas_width = cols * width + (cols - 1) * line_width
            canvas_height = rows * height + (rows - 1) * line_width
            return canvas_width / max(1, canvas_height)

        if num_videos == 4:
            rows, cols = 2, 2
        elif num_videos == 3:
            candidates = [(1, 3), (3, 1)]
            ratios = [abs(canvas_ratio(rows, cols) - 1.0) for rows, cols in candidates]
            rows, cols = candidates[int(np.argmin(ratios))]
        elif num_videos == 2:
            candidates = [(1, 2), (2, 1)]
            ratios = [abs(canvas_ratio(rows, cols) - 1.0) for rows, cols in candidates]
            if ratios[0] == ratios[1]:
                rows, cols = (1, 2) if width / height >= 1.0 else (2, 1)
            else:
                rows, cols = candidates[int(np.argmin(ratios))]
        else:
            rows, cols = 1, num_videos

        # Create a big canvas to fit in all time-frames
        canvas_width = cols * width + (cols - 1) * line_width
        canvas_height = rows * height + (rows - 1) * line_width
        canvas = torch.full((num_videos, 1, channels, canvas_height, canvas_width), 255, dtype=torch.uint8)

        for sample_id, video in enumerate, videos:
            video = video.view(rows, col, channels, height, width)
            for row in range(rows):
                for col in range(cols):
                    h_start = row * (height + line_width)
                    w_start = col * (width + line_width)
                    canvas[sample_id, :, :, h_start:h_start+height, w_start:w_start+width] = video[row, col]

        return canvas

    @staticmethod
    def _ensure_divide(length: int, divisor: int) -> int:
        return max(round(length / divisor) * divisor, divisor)

    @classmethod
    def _find_best_resize(
        cls, original_size: tuple[int, int], scale_resolution: int, patch_size: int, allow_upscale: bool = False
    ) -> tuple[int, int]:
        width, height = original_size
        if (width * height > scale_resolution * scale_resolution) or allow_upscale:
            aspect_ratio = width / height
            height = int(scale_resolution / math.sqrt(aspect_ratio))
            width = int(height * aspect_ratio)
        # factor 4 = two successive 2×2 spatial merges (ViT insert merger + downsample MLP)
        best_width = cls._ensure_divide(width, patch_size * 4)
        best_height = cls._ensure_divide(height, patch_size * 4)
        return (best_width, best_height)

    @classmethod
    def _get_refine_size(
        cls,
        original_size: tuple[int, int],
        grid: list[int],
        scale_resolution: int,
        patch_size: int,
        allow_upscale: bool = False,
    ) -> tuple[int, int]:
        width, height = original_size
        grid_x, grid_y = grid
        refine_width = cls._ensure_divide(width, grid_x)
        refine_height = cls._ensure_divide(height, grid_y)
        grid_width = refine_width / grid_x
        grid_height = refine_height / grid_y
        best_grid_size = cls._find_best_resize(
            (grid_width, grid_height), scale_resolution, patch_size, allow_upscale=allow_upscale
        )
        return (best_grid_size[0] * grid_x, best_grid_size[1] * grid_y)

    @staticmethod
    def _get_sliced_grid(
        image_size: tuple[int, int], max_slice_nums: int, scale_resolution: int, never_split: bool = False
    ) -> list[int] | None:
        original_width, original_height = image_size
        log_ratio = math.log(original_width / original_height)
        ratio = original_width * original_height / (scale_resolution * scale_resolution)
        multiple = min(math.ceil(ratio), max_slice_nums)
        if multiple <= 1 or never_split:
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
                        best_grid = [num_rows, num_cols]
                        min_error = error
        return best_grid

    # ------------------------------------------------------------------
    # Tensor operations
    # ------------------------------------------------------------------

    @staticmethod
    def _reshape_by_patch(image: "torch.Tensor", patch_size: int) -> "torch.Tensor":
        """Reshape ``[C, H, W]`` into NaViT patchified format ``[C, patch_size, H*W/patch_size]``."""
        num_channels = image.shape[0]
        patches = torch.nn.functional.unfold(
            image.unsqueeze(0), (patch_size, patch_size), stride=(patch_size, patch_size)
        )
        patches = patches.reshape(num_channels, patch_size, patch_size, -1)
        patches = patches.permute(0, 1, 3, 2).reshape(num_channels, patch_size, -1)
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

    def _preprocess(
        self,
        videos: list[torch.Tensor],
        do_resize: bool,
        size,
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
        video_metadata: list[VideoMetadata],
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        token_divisor = 4 if downsample_mode == "4x" else 16

        # Group videos by size for batched resizing
        grouped_videos, grouped_metadata, grouped_videos_index = group_videos_by_shape(videos, video_metadata)
        for video in zip(videos, video_metadata):
            metadata = grouped_metadata[shape][0]

            if stack_frames > 1:
                num_seconds = math.ceil(metadata.duration)
                len_subframes = num_seconds * stack_frames
                len_subframes -= math.ceil((num_seconds - metadata.duration) / (1 / stack_frames))
                # Video alreqady in BTCHW format so just slice off on timeframe dim
                stacked_videos, sub_stacked_videos = stacked_videos[:, :-len_subframes], stacked_videos[:, -len_subframes:]
                # `sub_stacked_videos` shaped as BTCHW with T=1 and different HW than stacked_videos
                sub_stacked_videos = self.concat_frames_as_image(sub_stacked_videos)

            if do_resize:
                height, width = stacked_videos.shape[-2:]
                original_size = (width, height)
                best_grid = None

                if slice_mode:
                    best_grid = self._get_sliced_grid(original_size, max_slice_nums, scale_resolution)

                # Always resize the source
                new_width, new_height = self._find_best_resize(
                    original_size, scale_resolution, patch_size, allow_upscale=(best_grid is None)
                )
                source_videos = self.resize(stacked_videos, size=SizeDict(height=new_height, width=new_width), resample=resample)
                source_visual_tokens = new_height * new_width // (patch_size * patch_size * token_divisor)

                resized_videos_grouped[shape] = source_videos
                patches_grouped[shape] = refine_videos

                # Collect all patches: [source, *slices]
                patches = [source_videos]
                patch_visual_tokens = 0
                patch_height = patch_width = 0
                if best_grid is not None:
                    refine_width, refine_height = self._get_refine_size(
                        original_size, best_grid, scale_resolution, patch_size, allow_upscale=True
                    )
                    grid_x, grid_y = best_grid
                    patch_height, patch_width = refine_height // grid_y, refine_width // grid_x

                    refine_videos = self.resize(
                        stacked_videos, size=SizeDict(height=refine_height, width=refine_width), resample=resample
                    )
                    refine_videos = divide_to_patches(refine_videos, (patch_height, patch_width))
                    patch_visual_tokens = patch_height * patch_width // (patch_size * patch_size * token_divisor)
                    patches.extend(refine_videos)

                    stacked = self.rescale_and_normalize(
                        stacked.float(), do_rescale, rescale_factor, do_normalize, image_mean, image_std
                    )
                    processed_grouped[shape] = stacked
                processed_patches = reorder_images(processed_grouped, grouped_index)

                image_pv = [self._reshape_by_patch(processed_patches[0], patch_size)]
                image_ts = [[source_height // patch_size, source_width // patch_size]]
                for processed_slice in processed_patches[1:]:
                    image_pv.append(self._reshape_by_patch(processed_slice, patch_size))
                    image_ts.append([patch_height // patch_size, patch_width // patch_size])

                per_image_pixel_values.append(image_pv)
                per_image_target_sizes.append(image_ts)
                all_grids.append(best_grid if best_grid is not None else [0, 0])
                all_source_visual_tokens.append(source_visual_tokens)
                all_patch_visual_tokens.append(patch_visual_tokens)


        return BatchFeature(
            data={"pixel_values": torch.stack(all_frames) if all_frames else torch.empty(0)},
            tensor_type=return_tensors,
        )


__all__ = ["MiniCPMV4_6VideoProcessor"]
