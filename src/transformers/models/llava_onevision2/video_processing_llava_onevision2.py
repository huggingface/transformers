# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Video preprocessing for LLaVA-OneVision-2."""

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ...image_processing_utils import BatchFeature
from ...image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, ChannelDimension, PILImageResampling, SizeDict
from ...processing_utils import Unpack, VideosKwargs
from ...utils import auto_docstring, is_av_available, is_decord_available
from ...video_processing_utils import BaseVideoProcessor
from ...video_utils import VideoMetadata, load_video
from ...vision_utils import get_vision_position_ids
from .image_processing_pil_llava_onevision2 import LlavaOnevision2ImageProcessorPil, smart_resize


class LlavaOnevision2VideoProcessorKwargs(VideosKwargs, total=False):
    """
    patch_size (`int`, *optional*, defaults to 14):
        Spatial patch size.
    merge_size (`int`, *optional*, defaults to 2):
        Number of patches merged along each spatial dimension.
    temporal_patch_size (`int`, *optional*, defaults to 1):
        Number of frames represented by each input patch. OV2 requires one.
    max_frames (`int`, *optional*, defaults to 384):
        Maximum number of frames selected by duration-based sampling.
    video_backend (`str`, *optional*, defaults to `"frames"`):
        Use uniform frame sampling or the optional `"codec"` canvas preprocessing backend.
    codec_config (`dict`, *optional*):
        Codec canvas settings: `target_canvas`, `group_size`, `images_per_group`,
        `min_group_frames`, and `max_group_frames`.
    """

    patch_size: int
    merge_size: int
    temporal_patch_size: int
    max_frames: int
    video_backend: str
    codec_config: dict


@auto_docstring
class LlavaOnevision2VideoProcessor(BaseVideoProcessor):
    r"""
    do_convert_rgb (`bool`, *optional*, defaults to `True`):
        Whether to convert decoded frames to RGB before patch extraction.
    """

    valid_kwargs = LlavaOnevision2VideoProcessorKwargs
    model_input_names = ["pixel_values_videos", "video_grid_thw", "patch_positions"]
    size = {"shortest_edge": 256 * 28 * 28, "longest_edge": 1605632}
    resample = PILImageResampling.BICUBIC
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_sample_frames = True
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    patch_size = 14
    merge_size = 2
    temporal_patch_size = 1
    max_frames = 384
    video_backend = "frames"
    codec_config = None

    def __init__(self, do_convert_rgb: bool = True, **kwargs: Unpack[LlavaOnevision2VideoProcessorKwargs]):
        """
        Args:
            do_convert_rgb (`bool`, *optional*, defaults to `True`):
                Whether to convert decoded frames to RGB before patch extraction.
        """
        # Accept the field names used by the original published video configuration.
        size = dict(kwargs.pop("size", self.size))
        for legacy, canonical in (("min_pixels", "shortest_edge"), ("max_pixels", "longest_edge")):
            if legacy in kwargs:
                size[canonical] = kwargs.pop(legacy)
        for legacy, canonical in (
            ("spatial_merge_size", "merge_size"),
            ("fixed_num_frames", "num_frames"),
            ("target_fps", "fps"),
            ("resize_frames", "do_resize"),
        ):
            if legacy in kwargs:
                kwargs.setdefault(canonical, kwargs.pop(legacy))
        super().__init__(size=size, do_convert_rgb=do_convert_rgb, **kwargs)
        if self.temporal_patch_size != 1:
            raise ValueError("OV2 requires temporal_patch_size=1")

    def sample_frames(self, metadata: VideoMetadata, num_frames=None, fps=None, max_frames=None, **kwargs):
        if num_frames is not None and fps is not None:
            raise ValueError("num_frames and fps are mutually exclusive")
        max_frames = self.max_frames if max_frames is None else max_frames
        for name, value in (("num_frames", num_frames), ("fps", fps), ("max_frames", max_frames)):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")
        count = metadata.total_num_frames
        if count <= 0:
            raise ValueError("The video must contain at least one frame")
        duration = count / metadata.fps if metadata.fps else count
        if fps is not None:
            target = min(max(1, int(duration * fps)), max_frames)
        elif num_frames is not None:
            target = num_frames
        else:
            target = 8 if duration < 10 else 16 if duration < 30 else max_frames
        return torch.linspace(0, count - 1, min(count, target)).round().long().numpy()

    def fetch_videos(self, video_url_or_urls, sample_indices_fn=None):
        if not (is_decord_available() or is_av_available()):
            return super().fetch_videos(video_url_or_urls, sample_indices_fn=sample_indices_fn)
        if isinstance(video_url_or_urls, list):
            return list(zip(*[self.fetch_videos(video, sample_indices_fn) for video in video_url_or_urls]))
        return load_video(
            video_url_or_urls,
            backend="decord" if is_decord_available() else "pyav",
            sample_indices_fn=sample_indices_fn,
        )

    def preprocess(self, videos, **kwargs: Unpack[LlavaOnevision2VideoProcessorKwargs]):
        for name in ("patch_size", "merge_size", "temporal_patch_size"):
            if name in kwargs and kwargs[name] != getattr(self, name):
                raise ValueError(f"{name} must match the video processor configuration ({getattr(self, name)})")
        backend = kwargs.pop("video_backend", self.video_backend)
        if backend == "codec":
            return self._preprocess_codec(videos, **kwargs)
        if backend != "frames":
            raise ValueError("video_backend must be 'frames' or 'codec'")
        return_metadata = kwargs.pop("return_metadata", False)
        outputs = super().preprocess(videos, return_metadata=True, **kwargs)
        metadata = outputs["video_metadata"]
        grids = torch.as_tensor(outputs["video_grid_thw"])
        positions = []
        timestamps = []
        for grid, item in zip(grids, metadata):
            time, height, width = grid.tolist()
            indices = torch.as_tensor(item.frames_indices if item.frames_indices is not None else range(time))
            coordinates = get_vision_position_ids(grid[None], self.merge_size, include_temporal=True)
            coordinates[:, 0] = indices[coordinates[:, 0]]
            positions.append(coordinates)
            timestamps.append((indices / (item.fps or 1.0)).tolist())
        outputs["patch_positions"] = torch.cat(positions).numpy()
        outputs["frame_timestamps"] = timestamps
        if not return_metadata:
            outputs.pop("video_metadata")
        return BatchFeature(
            dict(outputs),
            tensor_type=kwargs.get("return_tensors"),
            skip_tensor_conversion=["frame_timestamps", "video_metadata"],
        )

    def _preprocess(
        self,
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
        merge_size,
        return_tensors=None,
        **kwargs,
    ):
        processor = LlavaOnevision2ImageProcessorPil(
            size=dict(size),
            patch_size=patch_size,
            merge_size=merge_size,
            do_resize=do_resize,
            resample=resample,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
        )
        pixels, grids = [], []
        for video in videos:
            if do_resize:
                height, width = smart_resize(
                    *video.shape[-2:],
                    factor=patch_size * merge_size,
                    min_pixels=size.shortest_edge,
                    max_pixels=size.longest_edge,
                )
                resize_size = SizeDict(height=height, width=width)
                if video.dtype == torch.uint8 and resample == PILImageResampling.BICUBIC:
                    # Match the float interpolation and rounding used by the reference video processor.
                    video = (
                        self.resize(video.float(), resize_size, resample=resample)
                        .clamp_(0, 255)
                        .round_()
                        .to(torch.uint8)
                    )
                else:
                    video = self.resize(video, resize_size, resample=resample)
            frames = list(video.cpu().numpy())
            # Frames are already aligned to the merge grid. A second resize would
            # change the rounded minimum-area result and its source coordinates.
            data = processor(frames, input_data_format=ChannelDimension.FIRST, do_resize=False, return_tensors="np")
            grid = data["image_grid_thw"]
            pixels.append(data["pixel_values"])
            grids.append([len(frames), *grid[0, 1:].tolist()])
        return BatchFeature(
            {"pixel_values_videos": np.concatenate(pixels), "video_grid_thw": np.array(grids)},
            tensor_type=return_tensors,
        )

    def _preprocess_codec(self, videos, codec_config=None, size=None, return_tensors=None, **kwargs):
        try:
            from codec_video_prep import run_preinfer
        except ImportError as error:
            raise ImportError(
                "Codec video preprocessing requires `pip install codec-video-prep` and ffmpeg on PATH"
            ) from error
        config = {
            "target_canvas": 32,
            "group_size": 32,
            "images_per_group": 4,
            "min_group_frames": 8,
            "max_group_frames": 64,
            **(self.codec_config or {}),
            **(codec_config or {}),
        }
        target = config.pop("target_canvas")
        if (
            min(target, config["images_per_group"], config["group_size"]) <= 0
            or target % config["images_per_group"]
            or config["group_size"] % config["images_per_group"]
        ):
            raise ValueError("target_canvas and group_size must be positive multiples of images_per_group")
        budget = size["longest_edge"] if size is not None else 150000
        paths = [videos] if isinstance(videos, (str, Path)) else videos
        pixels, grids, positions, timestamps = [], [], [], []
        for path in paths:
            if not isinstance(path, (str, Path)):
                raise ValueError("The codec backend requires encoded video file paths")
            with tempfile.TemporaryDirectory(prefix="ov2-codec-") as directory:
                result = run_preinfer(
                    str(path),
                    directory,
                    num_sampled_frames=target // config["images_per_group"] * config["group_size"],
                    patch=self.patch_size,
                    max_pixels=budget,
                    block_size=self.merge_size,
                    canvas_format="jpg",
                    **config,
                )
                meta = json.loads(Path(result.meta_path).read_text(encoding="utf-8"))
                source = np.load(Path(directory) / "src_patch_position.npy").reshape(len(result.canvas_files), -1, 3)
                keep = (source[:, :, 0] >= 0).all(axis=1)
                if ((source[:, :, 0] >= 0).any(axis=1) != keep).any():
                    raise ValueError("Codec canvases must contain either valid patches or padding, not both")
                images = [
                    Image.open(Path(directory) / name).convert("RGB")
                    for name, valid in zip(result.canvas_files, keep)
                    if valid
                ]
                if not images:
                    raise ValueError("Codec preprocessing produced no usable canvases")
                image_kwargs = {
                    key: value
                    for key, value in kwargs.items()
                    if key in ("do_rescale", "rescale_factor", "do_normalize", "image_mean", "image_std")
                }
                data = LlavaOnevision2ImageProcessorPil(patch_size=self.patch_size, merge_size=self.merge_size)(
                    images, do_resize=False, return_tensors="np", **image_kwargs
                )
                grid = data["image_grid_thw"]
                chunks = []
                for coordinates, (_, height, width) in zip(source[keep], grid):
                    chunks.append(
                        coordinates.reshape(
                            height // self.merge_size, self.merge_size, width // self.merge_size, self.merge_size, 3
                        )
                        .transpose(0, 2, 1, 3, 4)
                        .reshape(-1, 3)
                    )
                coordinates = np.concatenate(chunks)
                if len(coordinates) != len(data["pixel_values"]):
                    raise ValueError("Codec source positions do not match the canvas patch grid")
                pixels.append(data["pixel_values"])
                grids.append([len(images), *grid[0, 1:].tolist()])
                positions.append(coordinates)
                timestamps.append(float(meta.get("fps") or 30.0))
        return BatchFeature(
            {
                "pixel_values_videos": np.concatenate(pixels),
                "video_grid_thw": np.array(grids),
                "patch_positions": np.concatenate(positions),
                "frame_timestamps": timestamps,
                "is_codec": True,
            },
            tensor_type=return_tensors,
            skip_tensor_conversion=["frame_timestamps", "is_codec"],
        )


__all__ = ["LlavaOnevision2VideoProcessor"]
