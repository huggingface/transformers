# Copyright 2025 Baidu and HuggingFace Inc. team. All rights reserved.
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
import os.path
from functools import partial
from pathlib import Path
from shutil import SameFileError, copyfile
from typing import Any

import numpy as np
import torch
from huggingface_hub.dataclasses import validate_typed_dict
from PIL import ImageDraw, ImageFont
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from ...image_processing_utils import BatchFeature
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    PILImageResampling,
    SizeDict,
    get_image_size,
    validate_kwargs,
)
from ...processing_utils import Unpack, VideosKwargs
from ...utils import (
    TensorType,
    auto_docstring,
    logging,
)
from ...utils.import_utils import is_torchvision_available, is_tracing
from ...video_processing_utils import BaseVideoProcessor
from ...video_utils import (
    VideoInput,
    VideoMetadata,
    group_videos_by_shape,
    infer_channel_dimension_format,
    reorder_videos,
)
from .image_processing_ernie4_5_vl_moe import smart_resize


if is_torchvision_available():
    from torchvision.transforms.v2 import functional as tvF


logger = logging.get_logger(__name__)


class Ernie4_5_VLMoeVideoProcessorInitKwargs(VideosKwargs, total=False):
    r"""
    patch_size (`int`, *optional*, defaults to 14):
        The spatial patch size of the vision encoder.
    temporal_patch_size (`int`, *optional*, defaults to 1):
        The temporal patch size of the vision encoder.
    merge_size (`int`, *optional*, defaults to 2):
        The merge size of the vision encoder to llm encoder.
    min_frames (`int`, *optional*, defaults to 16):
        The minimum number of frames that will be sampled.
    max_frames (`int`, *optional*, defaults to 180):
        The maximum number of frames that will be sampled.
    draw_on_frames (`bool`, *optional*, defaults to `True`):
        Whether to draw timestamps on each video frame.
    font (`str`, *optional*, defaults to `True`):
        The font used to draw timestamps when `draw_on_frames` is set.
        Note that `torch.compile` is not compatible with drawing on frames.
    """

    patch_size: int
    temporal_patch_size: int
    merge_size: int
    min_frames: int
    max_frames: int
    draw_on_frames: bool
    font: str


@auto_docstring
class Ernie4_5_VLMoeVideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BICUBIC
    size = {"shortest_edge": 299 * 28 * 28, "longest_edge": 1196 * 28 * 28}
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    patch_size = 14
    temporal_patch_size = 2
    merge_size = 2
    min_frames = 16
    max_frames = 180
    do_sample_frames = True
    draw_on_frames = True
    font = "Roboto-Regular.ttf"
    valid_kwargs = Ernie4_5_VLMoeVideoProcessorInitKwargs
    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def __init__(self, **kwargs: Unpack[Ernie4_5_VLMoeVideoProcessorInitKwargs]):
        temporal_patch_size = kwargs.get("temporal_patch_size", 2)
        if temporal_patch_size is None or temporal_patch_size != 2:
            raise ValueError("`Ernie 4.5 VL` only supports a temporal patch size of 2")

        size = kwargs.pop("size", None)
        size = dict(self.size) if size is None else size
        if "shortest_edge" not in size or "longest_edge" not in size:
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")

        super().__init__(size=size, **kwargs)

    @classmethod
    def from_dict(cls, video_processor_dict: dict[str, Any], **kwargs):
        """
        Instantiates a type of [`~video_processing_utils.VideoProcessorBase`] from a Python dictionary of parameters.

        Args:
            video_processor_dict (`dict[str, Any]`):
                Dictionary that will be used to instantiate the video processor object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the
                [`~video_processing_utils.VideoProcessorBase.to_dict`] method.
            kwargs (`dict[str, Any]`):
                Additional parameters from which to initialize the video processor object.

        Returns:
            [`~video_processing_utils.VideoProcessorBase`]: The video processor object instantiated from those
            parameters.
        """
        path_to_repo = video_processor_dict.pop("_path_to_repo", None)

        # Specific to Ernie 4.5 VL Moe, we load the font file along the json (if we draw on frames)
        draws_on_frames = video_processor_dict.get("draw_on_frames")
        if (font_name := video_processor_dict.get("font")) is None and draws_on_frames:
            raise AttributeError(
                "Expected a `font` to be saved when using `draw_on_frames` in Ernie 4.5 VL Moe; found nothing."
            )
        if font_name is not None and draws_on_frames:
            video_processor_dict["font"] = f"{path_to_repo}/{font_name}"
            try:
                ImageFont.truetype(video_processor_dict["font"])
            except (TypeError, OSError):
                raise OSError(
                    f"Could not find an associated font file for {video_processor_dict['font']}. "
                    "Make sure to save a font file along for Ernie 4.5 VL Moe."
                )

        video_processor_dict = video_processor_dict.copy()
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        valid_keys = cls.valid_kwargs.__annotations__.keys()
        video_processor_dict.update({k: v for k, v in kwargs.items() if k in valid_keys})
        video_processor = cls(**video_processor_dict)

        # Apply extra kwargs to instance (BC for remote code, e.g. phi4_multimodal)
        extra_keys = []
        for key in reversed(list(kwargs.keys())):
            if hasattr(video_processor, key) and key not in cls.valid_kwargs.__annotations__:
                setattr(video_processor, key, kwargs.pop(key, None))
                extra_keys.append(key)
        if extra_keys:
            logger.warning_once(
                f"Image processor {cls.__name__}: kwargs {extra_keys} were applied for backward compatibility. "
                f"To avoid this warning, add them to valid_kwargs: create a custom TypedDict extending "
                f"ImagesKwargs with these keys and set it as the `valid_kwargs` class attribute."
            )

        logger.info(f"Video processor {video_processor}")
        if return_unused_kwargs:
            return video_processor, kwargs
        else:
            return video_processor

    def to_dict(self) -> dict[str, Any]:
        """Overridden to strip the prefix of the full path for the font, e.g. `tmp/folder/font.tff` -> `font.tff`"""
        output = super().to_dict()

        if os.path.isfile(output.get("font")):
            output["font"] = Path(output["font"]).name
        elif output.get("draw_on_frames"):
            raise ValueError(
                f"The video processor dict contains an invalid path to its font: {output['font']}. "
                "Please make sure to contain a valid path or disable `draw_on_frames`."
            )

        return output

    def save_pretrained(self, save_directory: str | os.PathLike, push_to_hub: bool = False, **kwargs):
        """We additionally save a copy of the font to the `save_directory` (if we found a file there)"""
        os.makedirs(save_directory, exist_ok=True)

        if os.path.isfile(self.font):
            try:
                copyfile(self.font, Path(save_directory, Path(self.font).name))
            except SameFileError:  # already exists which we allow (copy if needed)
                pass

        return super().save_pretrained(save_directory, push_to_hub, **kwargs)

    def _standardize_kwargs(
        self,
        size: SizeDict | None = None,
        **kwargs,
    ) -> dict:
        """
        Update kwargs that need further processing before being validated
        Can be overridden by subclasses to customize the processing of kwargs.
        """
        if size is not None and ("shortest_edge" not in size or "longest_edge" not in size):
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")

        return super()._standardize_kwargs(size=size, **kwargs)

    def sample_frames(
        self,
        metadata: VideoMetadata,
        min_frames: int | None = None,
        max_frames: int | None = None,
        num_frames: int | None = None,
        fps: int | float | None = None,
        **kwargs,
    ):
        if fps is not None and num_frames is not None:
            raise ValueError("`num_frames` and `fps` are mutually exclusive arguments, please use only one!")

        num_frames = num_frames if num_frames is not None else self.num_frames
        min_frames = min_frames if min_frames is not None else self.min_frames
        max_frames = max_frames if max_frames is not None else self.max_frames
        total_num_frames = metadata.total_num_frames

        if num_frames is not None:
            if num_frames < min_frames or num_frames > max_frames:
                raise ValueError(f"`num_frames` must be {min_frames} <= x <= {max_frames}. Got {num_frames} instead.")
        else:
            if fps is not None and (metadata is None or metadata.fps is None):
                raise ValueError(
                    "Asked to sample `fps` frames per second but no video metadata was provided which is required when sampling with `fps`. "
                    "Please pass in `VideoMetadata` object or use a fixed `num_frames` per input video"
                )
            num_frames = total_num_frames / metadata.fps * fps if fps is not None else total_num_frames
            num_frames = min(max(num_frames, min_frames), max_frames, total_num_frames)

        if num_frames > total_num_frames:
            raise ValueError(
                f"Video can't be sampled. The inferred `num_frames={num_frames}` exceeds `total_num_frames={total_num_frames}`. "
                "Decrease `num_frames` or `fps` for sampling."
            )

        indices = torch.arange(0, total_num_frames, total_num_frames / num_frames).int()

        return indices

    def _convert_timestamp(self, time_stamp_in_seconds):
        """Convert to `time: hr:min:sec` format"""
        hours = time_stamp_in_seconds // 3600
        time_stamp_in_seconds = time_stamp_in_seconds % 3600
        mins = time_stamp_in_seconds // 60
        time_stamp_in_seconds = time_stamp_in_seconds % 60
        return f"time: {int(hours):02d}:{int(mins):02d}:{time_stamp_in_seconds:05.02f}"

    def _render_image_with_timestamp(self, image: torch.Tensor, timestamp: str, size_factor: float = 0.1):
        """Draws a black timestamp with a white border on the corner of the frame"""
        if self.font is None:
            raise AttributeError("To draw on frames with Ernie 4.5 VL, you need an associated font; found nothing")

        # FIXME: conversion `torch->PIL->torch` is inefficient ~6ms per frame
        # Left for optimization if anyone want to pick it up
        #
        # This can take up to ~1s in preprocessing (if default sampling is used):
        #   180 (frames) x 6ms = 1080ms = ~1,1s
        image = to_pil_image(image)

        font_size = int(min(*image.size) * size_factor)
        outline_size = int(font_size * size_factor)
        font = ImageFont.truetype(self.font, font_size)

        # Draw a black text with a white border
        draw = ImageDraw.Draw(image)
        draw.text(
            (0, 0),
            timestamp,
            font=font,
            fill=(0, 0, 0),
            stroke_width=outline_size,
            stroke_fill=(255, 255, 255),
        )
        return pil_to_tensor(image)

    def _prepare_input_videos(
        self,
        videos: VideoInput,
        input_data_format: str | ChannelDimension | None = None,
        device: str | None = None,
        video_metadata: list[VideoMetadata] | None = None,
        draw_on_frames: bool = True,
    ) -> list["torch.Tensor"]:
        """
        Prepare the input videos for processing.
        """
        processed_videos = []
        for video, metadata in zip(videos, video_metadata):
            # Check for attributes that are necessary to draw timestamps on frames
            if draw_on_frames:
                if metadata is None:
                    raise ValueError("Need video metadata to process videos in Ernie 4.5 VL using `draw_on_frames`")
                elif metadata.fps is None:
                    metadata.fps = 24
                    logger.warning_once(
                        "Could not infer the fps of a video due to the metadata not being available, "
                        "defaulting to `24`. Please provide `video_metadata` for more accurate results."
                    )

            # `make_batched_videos` always returns a 4D array per video
            if isinstance(video, np.ndarray):
                # not using F.to_tensor as it doesn't handle (C, H, W) numpy arrays
                video = torch.from_numpy(video).contiguous()

            # Infer the channel dimension format if not provided
            if input_data_format is None:
                input_data_format = infer_channel_dimension_format(video)

            if input_data_format == ChannelDimension.LAST:
                video = video.permute(0, 3, 1, 2).contiguous()

            # specific to ernie, draws timestamps on each frame (if enabled)
            if draw_on_frames:
                if is_tracing(video):
                    raise RuntimeError(
                        "Using `torch.compile` is not compatible with drawing on frames. "
                        "Either don't use `torch.compile` or don't draw on frames via the kwarg `draw_on_frames=False`."
                    )

                for idx, frame in enumerate(video):
                    video[idx] = self._render_image_with_timestamp(
                        frame, self._convert_timestamp(metadata.timestamps[idx])
                    )

            # last frame is copied if uneven (mitigating issues for temporal patch size)
            if video.shape[0] % 2 != 0:
                video = torch.cat((video, video[-1].detach().clone()[None, ...]), dim=0)

            if device is not None:
                video = video.to(device)

            processed_videos.append(video)
        return processed_videos

    def get_num_of_video_patches(
        self, num_frames: int, height: int, width: int, videos_kwargs: dict | None = None
    ) -> int:
        """
        A utility that returns number of video patches for a given video size.

        Note: Do not remove this method! It is used by vLLM to infer the number of patches and placeholders
        without a video input.

        Args:
            num_frames (`int`):
                Number of frames in the input video.
            height (`int`):
                Height of the input video.
            width (`int`):
                Width of the input video.
            videos_kwargs (`dict`, *optional*)
                Any kwargs to override defaults of the video processor.
        Returns:
            `int`: Number of video patches per video.
        """
        videos_kwargs = videos_kwargs or {}
        size = videos_kwargs.get("size", self.size)
        patch_size = videos_kwargs.get("patch_size", self.patch_size)
        merge_size = videos_kwargs.get("merge_size", self.merge_size)
        factor = patch_size * merge_size
        resized_height, resized_width = smart_resize(
            height, width, factor, min_pixels=size["shortest_edge"], max_pixels=size["longest_edge"]
        )
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        # `_prepare_input_videos` copies the last frame if uneven, ignoring any kwarg
        grid_t = num_frames + -num_frames % self.temporal_patch_size
        return grid_t * grid_h * grid_w

    def _preprocess(
        self,
        videos: list[torch.Tensor],
        do_resize: bool = True,
        size: SizeDict | None = None,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None" = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255.0,
        do_normalize: bool = True,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        patch_size: int | None = None,
        merge_size: int | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ):
        # Group videos by size for batched resizing
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}
        for shape, stacked_videos in grouped_videos.items():
            height, width = get_image_size(stacked_videos[0], channel_dim=ChannelDimension.FIRST)
            resized_height, resized_width = height, width
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=patch_size * merge_size,
                    min_pixels=size["shortest_edge"],
                    max_pixels=size["longest_edge"],
                )
                stacked_videos = self.resize(
                    stacked_videos,
                    size=SizeDict(height=resized_height, width=resized_width),
                    resample=resample,
                )
            resized_videos_grouped[shape] = stacked_videos
        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)

        # Group videos by size for further processing
        # Needed in case do_resize is False, or resize returns videos with different sizes
        grouped_videos, grouped_videos_index = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        processed_grids = {}
        for shape, stacked_videos in grouped_videos.items():
            resized_height, resized_width = get_image_size(stacked_videos[0], channel_dim=ChannelDimension.FIRST)

            # Fused rescale and normalize
            stacked_videos = self.rescale_and_normalize(
                stacked_videos, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            patches = stacked_videos

            batch_size, grid_t, channel = patches.shape[:3]
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            patches = patches.view(
                batch_size,
                grid_t,
                channel,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
            # Reorder dimensions to group grid and patch information for subsequent flattening.
            # [batch, grid_t, grid_h/merge, grid_w/merge, merge, merge, channel, patch, patch]
            patches = patches.permute(0, 1, 3, 6, 4, 7, 2, 5, 8)

            flatten_patches = patches.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                channel * patch_size * patch_size,
            )

            processed_videos_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        processed_videos = reorder_videos(processed_videos_grouped, grouped_videos_index)
        processed_grids = reorder_videos(processed_grids, grouped_videos_index)
        pixel_values_videos = torch.cat(processed_videos, dim=0)
        video_grid_thw = torch.tensor(processed_grids)

        return BatchFeature(
            data={"pixel_values_videos": pixel_values_videos, "video_grid_thw": video_grid_thw},
            tensor_type=return_tensors,
        )

    @auto_docstring
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
        draw_on_frames = kwargs.pop("draw_on_frames")

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
            video_metadata=video_metadata,
            draw_on_frames=draw_on_frames,
        )

        kwargs = self._standardize_kwargs(**kwargs)
        self._validate_preprocess_kwargs(**kwargs)

        # Pop kwargs that are not needed in _preprocess
        kwargs.pop("data_format")
        return_metadata = kwargs.pop("return_metadata")

        preprocessed_videos = self._preprocess(videos=videos, **kwargs)
        if return_metadata:
            preprocessed_videos["video_metadata"] = video_metadata
        return preprocessed_videos


__all__ = ["Ernie4_5_VLMoeVideoProcessor"]
