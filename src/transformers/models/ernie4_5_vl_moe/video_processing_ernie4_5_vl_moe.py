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
from huggingface_hub import is_offline_mode
from huggingface_hub.dataclasses import validate_typed_dict
from PIL import Image, ImageDraw, ImageFont

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
    IMAGE_PROCESSOR_NAME,
    PROCESSOR_NAME,
    VIDEO_PROCESSOR_NAME,
    TensorType,
    add_start_docstrings,
    logging,
    safe_load_json_file,
)
from ...utils.hub import cached_file
from ...utils.import_utils import is_tracing, requires
from ...video_processing_utils import BASE_VIDEO_PROCESSOR_DOCSTRING, BaseVideoProcessor
from ...video_utils import (
    VideoInput,
    VideoMetadata,
    group_videos_by_shape,
    infer_channel_dimension_format,
    reorder_videos,
)
from .image_processing_ernie4_5_vl_moe import smart_resize


logger = logging.get_logger(__name__)


class _TimestampOverlayCache:
    """Cache for timestamp overlays to avoid slow torch->PIL->torch conversion."""

    def __init__(self, font_path: str, max_cache_size: int = 512):
        self.font_path = font_path
        self.max_cache_size = max_cache_size
        self._font_cache: dict[int, ImageFont.FreeTypeFont] = {}
        self._overlay_cache: dict[tuple, tuple[torch.Tensor, int, int]] = {}

    def _get_font(self, font_size: int) -> ImageFont.FreeTypeFont:
        if font_size not in self._font_cache:
            self._font_cache[font_size] = ImageFont.truetype(self.font_path, font_size)
        return self._font_cache[font_size]

    def _render_overlay(self, timestamp: str, font_size: int, outline_size: int):
        cache_key = (timestamp, font_size, outline_size)
        if cache_key in self._overlay_cache:
            return self._overlay_cache[cache_key]

        font = self._get_font(font_size)
        dummy_img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        dummy_draw = ImageDraw.Draw(dummy_img)
        bbox = dummy_draw.textbbox((0, 0), timestamp, font=font, stroke_width=outline_size)

        text_width = bbox[2] + outline_size + 2
        text_height = bbox[3] + outline_size + 2

        overlay = Image.new("RGBA", (text_width, text_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.text((0, 0), timestamp, font=font, fill=(0, 0, 0, 255),
                  stroke_width=outline_size, stroke_fill=(255, 255, 255))

        overlay_tensor = torch.from_numpy(np.array(overlay)).permute(2, 0, 1).contiguous()
        result = (overlay_tensor, text_width, text_height)

        if len(self._overlay_cache) >= self.max_cache_size:
            oldest_key = next(iter(self._overlay_cache))
            del self._overlay_cache[oldest_key]

        self._overlay_cache[cache_key] = result
        return result

    def apply(self, image: torch.Tensor, timestamp: str, size_factor: float = 0.1) -> torch.Tensor:
        C, H, W = image.shape
        font_size = int(min(H, W) * size_factor)
        outline_size = int(font_size * size_factor)

        overlay, ow, oh = self._render_overlay(timestamp, font_size, outline_size)
        paste_h, paste_w = min(oh, H), min(ow, W)

        result = image.clone()
        alpha = overlay[3:4, :paste_h, :paste_w].float() / 255.0
        rgb_overlay = overlay[:3, :paste_h, :paste_w].float()
        original_region = result[:, :paste_h, :paste_w].float()
        blended = alpha * rgb_overlay + (1.0 - alpha) * original_region
        result[:, :paste_h, :paste_w] = blended.to(result.dtype)

        return result


class Ernie4_5_VL_MoeVideoProcessorInitKwargs(VideosKwargs, total=False):
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    min_frames: int
    max_frames: int
    draw_on_frames: bool
    font: str


@add_start_docstrings(
    "Constructs a fast Ernie 4.5 VL image processor that dynamically resizes videos based on the original videos.",
    BASE_VIDEO_PROCESSOR_DOCSTRING,
    """
        patch_size (`int`, *optional*, defaults to 14):
            The spacial patch size of the vision encoder.
        temporal_patch_size (`int`, *optional*, defaults to 2):
            The temporal patch size of the vision encoder.
        merge_size (`int`, *optional*, defaults to 2):
            The merge size of the vision encoder to llm encoder.
        min_frames (`int`, *optional*, defaults to 16):
            The minimum number of frames that can be sampled.
        max_frames (`int`, *optional*, defaults to 180):
            The maximum number of frames that can be sampled.
        draw_on_frames (`bool`, *optional*, defaults to `True`):
            Whether to draw timestamps on each frame or not.
            This does not work with `torch.compile` but resembles
            the performance of the original model.
        font (`str`, *optional*, defaults to "Roboto-Regular.ttf"):
            The associated font name for drawing on frames.
            Defaults to "Roboto-Regular.ttf" and is expected to be
            saved along the processor as separate file.
    """,
)
@requires(backends=("torchvision",))
class Ernie4_5_VL_MoeVideoProcessor(BaseVideoProcessor):
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
    valid_kwargs = Ernie4_5_VL_MoeVideoProcessorInitKwargs
    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def __init__(self, **kwargs: Unpack[Ernie4_5_VL_MoeVideoProcessorInitKwargs]):
        temporal_patch_size = kwargs.get("temporal_patch_size", 2)
        if temporal_patch_size is None or temporal_patch_size != 2:
            raise ValueError("`Ernie 4.5 VL` only supports a temporal patch size of 2")

        size = kwargs.pop("size", None)
        size = self.size if size is None else size
        if "shortest_edge" not in size or "longest_edge" not in size:
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")

        super().__init__(size=size, **kwargs)

    @classmethod
    def get_video_processor_dict(
        cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Overriden to additionally load the font for drawing on frames."""
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", "")

        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)

        user_agent = {"file_type": "video processor", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        is_local = os.path.isdir(pretrained_model_name_or_path)
        if os.path.isfile(pretrained_model_name_or_path):
            resolved_video_processor_file = pretrained_model_name_or_path
            resolved_processor_file = None
            is_local = True
        else:
            video_processor_file = VIDEO_PROCESSOR_NAME
            try:
                # Try to load with a new config name first and if not successful try with the old file name
                # NOTE: we save all processor configs as nested dict in PROCESSOR_NAME from v5, which is the standard
                resolved_processor_file = cached_file(
                    pretrained_model_name_or_path,
                    filename=PROCESSOR_NAME,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder,
                    _raise_exceptions_for_missing_entries=False,
                )
                resolved_video_processor_files = [
                    resolved_file
                    for filename in [video_processor_file, IMAGE_PROCESSOR_NAME]
                    if (
                        resolved_file := cached_file(
                            pretrained_model_name_or_path,
                            filename=filename,
                            cache_dir=cache_dir,
                            force_download=force_download,
                            proxies=proxies,
                            local_files_only=local_files_only,
                            token=token,
                            user_agent=user_agent,
                            revision=revision,
                            subfolder=subfolder,
                            _raise_exceptions_for_missing_entries=False,
                        )
                    )
                    is not None
                ]
                resolved_video_processor_file = (
                    resolved_video_processor_files[0] if resolved_video_processor_files else None
                )
            except OSError:
                # Raise any OS error raise by `cached_file`. It will have a helpful error message adapted to
                # the original exception.
                raise
            except Exception:
                # For any other exception, we throw a generic error.
                raise OSError(
                    f"Can't load video processor for '{pretrained_model_name_or_path}'. If you were trying to load"
                    " it from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                    f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
                    f" directory containing a {video_processor_file} file"
                )

        # Load video_processor dict. Priority goes as (nested config if found -> video processor config -> image processor config)
        # We are downloading both configs because almost all models have a `processor_config.json` but
        # not all of these are nested. We need to check if it was saved recebtly as nested or if it is legacy style
        video_processor_dict = None
        if resolved_processor_file is not None:
            processor_dict = safe_load_json_file(resolved_processor_file)
            if "video_processor" in processor_dict:
                video_processor_dict = processor_dict["video_processor"]

        if resolved_video_processor_file is not None and video_processor_dict is None:
            video_processor_dict = safe_load_json_file(resolved_video_processor_file)

        if video_processor_dict is None:
            raise OSError(
                f"Can't load video processor for '{pretrained_model_name_or_path}'. If you were trying to load"
                " it from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
                f" directory containing a {video_processor_file} file"
            )

        # Specific to Ernie 4.5 VL Moe, we load the font file along the json (if we draw on frames)
        draws_on_frames = video_processor_dict.get("draw_on_frames")
        if (font_name := video_processor_dict.get("font")) is None and draws_on_frames:
            raise AttributeError(
                "Expected a `font` to be saved when using `draw_on_frames` in Ernie 4.5 VL Moe; found nothing."
            )
        if font_name is not None and draws_on_frames:
            video_processor_dict["font"] = cached_file(
                pretrained_model_name_or_path,
                filename=font_name,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                user_agent=user_agent,
                revision=revision,
                subfolder=subfolder,
                _raise_exceptions_for_missing_entries=False,
            )
            try:
                ImageFont.truetype(video_processor_dict["font"])
            except (TypeError, OSError):
                raise OSError(
                    f"Could not find an associated font file for {video_processor_dict['font']}. "
                    "Make sure to save a font file along for Ernie 4.5 VL Moe."
                )

        if is_local:
            logger.info(f"loading configuration file {resolved_video_processor_file}")
        else:
            logger.info(
                f"loading configuration file {video_processor_file} from cache at {resolved_video_processor_file}"
            )

        return video_processor_dict, kwargs

    def to_dict(self) -> dict[str, Any]:
        """Overriden to strip the prefix of the full path for the font, e.g. `tmp/folder/font.tff` -> `font.tff`"""
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

    def _further_process_kwargs(
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

        return super()._further_process_kwargs(size=size, **kwargs)

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

    _timestamp_cache: _TimestampOverlayCache = None

    @property
    def timestamp_cache(self) -> _TimestampOverlayCache:
        if self._timestamp_cache is None:
            self._timestamp_cache = _TimestampOverlayCache(font_path=self.font)
        return self._timestamp_cache

    def _render_image_with_timestamp(self, image: torch.Tensor, timestamp: str, size_factor: float = 0.1):
        """Draws a black timestamp with a white border on the corner of the frame"""
        if self.font is None:
            raise AttributeError("To draw on frames with Ernie 4.5 VL, you need an associated font; found nothing")
        return self.timestamp_cache.apply(image, timestamp, size_factor)

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

    def _preprocess(
        self,
        videos: list[torch.Tensor],
        do_convert_rgb: bool = True,
        do_resize: bool = True,
        size: SizeDict | None = None,
        interpolation: PILImageResampling = PILImageResampling.BICUBIC,
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
            if do_convert_rgb:
                stacked_videos = self.convert_to_rgb(stacked_videos)

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
                    image=stacked_videos,
                    size=SizeDict(height=resized_height, width=resized_width),
                    interpolation=interpolation,
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

        kwargs = self._further_process_kwargs(**kwargs)
        self._validate_preprocess_kwargs(**kwargs)

        # Pop kwargs that are not needed in _preprocess
        kwargs.pop("data_format")
        return_metadata = kwargs.pop("return_metadata")

        preprocessed_videos = self._preprocess(videos=videos, **kwargs)
        if return_metadata:
            preprocessed_videos["video_metadata"] = video_metadata
        return preprocessed_videos


__all__ = ["Ernie4_5_VL_MoeVideoProcessor"]
