# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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

import os
from contextlib import redirect_stdout
from io import BytesIO
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import requests

from .image_transforms import PaddingMode
from .image_utils import ChannelDimension, is_valid_image
from .utils import (
    is_av_available,
    is_cv2_available,
    is_decord_available,
    is_numpy_array,
    is_torch_available,
    is_torch_tensor,
    is_torchvision_available,
    is_vision_available,
    is_yt_dlp_available,
    logging,
)


if is_vision_available():
    import PIL.Image
    import PIL.ImageOps

    if is_torchvision_available():
        from torchvision import io as torchvision_io

if is_decord_available():
    from decord import VideoReader, cpu

if is_av_available():
    import av

if is_cv2_available():
    import cv2

if is_yt_dlp_available():
    from yt_dlp import YoutubeDL

if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


VideoInput = Union[
    List["PIL.Image.Image"],
    "np.ndarray",
    "torch.Tensor",
    List["np.ndarray"],
    List["torch.Tensor"],
    List[List["PIL.Image.Image"]],
    List[List["np.ndarrray"]],
    List[List["torch.Tensor"]],
]  # noqa


def is_valid_video_frame(frame):
    return isinstance(frame, PIL.Image.Image) or (
        (is_numpy_array(frame) or is_torch_tensor(frame)) and frame.ndim == 3
    )


def is_valid_video(video):
    if not isinstance(video, (list, tuple)):
        return (is_numpy_array(video) or is_torch_tensor(video)) and video.ndim == 4
    return all(is_valid_video_frame(frame) for frame in video)


def valid_videos(videos):
    # If we have a list of videos, it could be either one video as list of frames or a batch
    if isinstance(videos, (list, tuple)):
        for video_or_frame in videos:
            if not (is_valid_video(video_or_frame) or is_valid_video_frame(video_or_frame)):
                return False
    # If not a list, then we have a single 4D video or 5D batched tensor
    elif not is_valid_video(videos) or videos.ndim == 5:
        return False
    return True


def is_batched_video(videos):
    if isinstance(videos, (list, tuple)):
        return is_valid_video(videos[0])
    elif (is_numpy_array(videos) or is_torch_tensor(videos)) and videos.ndim == 5:
        return True
    return False


def is_scaled_video(video: np.ndarray) -> bool:
    """
    Checks to see whether the pixel values have already been rescaled to [0, 1].
    """
    # It's possible the video has pixel values in [0, 255] but is of floating type
    return np.min(video) >= 0 and np.max(video) <= 1


def convert_videos_to_array(videos: List[VideoInput]) -> List[Union["np.ndarray", "torch.Tensor"]]:
    """
    Given a batch of videos, converts each video to a 4D array. If video is already in array type,
    it is simply returned. We assume that all inputs in the list are in the same format, based on the type of the first element.

    Args:
        videos (`VideoInput`):
            Video inputs to turn into a list of videos.
    """

    if not isinstance(videos[0], (list, tuple)):
        return videos

    video_converted = []
    for video in videos:
        video = [np.array(frame) for frame in video]
        video = np.stack(video)
        video_converted.append(video)
    return video_converted


def make_batched_videos(videos) -> List[Union["np.ndarray", "torch.Tensor"]]:
    """
    Ensure that the input is a list of videos. If the input is a single video, it is converted to a list of length 1.
    If the input is a batch of videos, it is converted to a list of 4D video arrays. Videos passed as list `PIL.Image`
    frames are converted to 4D arrays.

    We assume that all inputs in the list are in the same format, based on the type of the first element.

    Args:
        videos (`VideoInput`):
            Video inputs to turn into a list of videos.
    """
    if not valid_videos:
        raise ValueError(
            f"Invalid video input. Expected either a list of video frames or an input of 4 or 5 dimensions, but got"
            f" type {type(videos)}."
        )

    if is_batched_video(videos):
        pass
    elif is_valid_video(videos):
        videos = [videos]
    # only one frame passed, thus we make a batched video
    elif is_valid_image(videos):
        videos = [[videos]]
    return convert_videos_to_array(videos)


def infer_channel_dimension_format(
    video: np.ndarray, num_channels: Optional[Union[int, Tuple[int, ...]]] = None
) -> ChannelDimension:
    """
    Infers the channel dimension format of `video`.

    Args:
        video (`np.ndarray`):
            The video to infer the channel dimension of.
        num_channels (`int` or `Tuple[int, ...]`, *optional*, defaults to `(1, 3)`):
            The number of channels of the video.

    Returns:
        The channel dimension of the video.
    """
    num_channels = num_channels if num_channels is not None else (1, 3)
    num_channels = (num_channels,) if isinstance(num_channels, int) else num_channels

    if video.ndim == 4:
        first_dim, last_dim = 1, 3
    elif video.ndim == 5:
        first_dim, last_dim = 2, 4
    else:
        raise ValueError(f"Unsupported number of video dimensions: {video.ndim}")

    if video.shape[first_dim] in num_channels and video.shape[last_dim] in num_channels:
        logger.warning(
            f"The channel dimension is ambiguous. Got video shape {video.shape}. Assuming channels are the first dimension."
        )
        return ChannelDimension.FIRST
    elif video.shape[first_dim] in num_channels:
        return ChannelDimension.FIRST
    elif video.shape[last_dim] in num_channels:
        return ChannelDimension.LAST
    raise ValueError("Unable to infer channel dimension format")


def get_channel_dimension_axis(
    video: np.ndarray, input_data_format: Optional[Union[ChannelDimension, str]] = None
) -> int:
    """
    Returns the channel dimension axis of the video.

    Args:
        video (`np.ndarray`):
            The video to get the channel dimension axis of.
        input_data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format of the video. If `None`, will infer the channel dimension from the video.

    Returns:
        The channel dimension axis of the video.
    """
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(video)
    if input_data_format == ChannelDimension.FIRST:
        return video.ndim - 3
    elif input_data_format == ChannelDimension.LAST:
        return video.ndim - 1
    raise ValueError(f"Unsupported data format: {input_data_format}")


def to_channel_dimension_format(
    video: np.ndarray,
    channel_dim: Union[ChannelDimension, str],
    input_channel_dim: Optional[Union[ChannelDimension, str]] = None,
) -> np.ndarray:
    """
    Converts `video` to the channel dimension format specified by `channel_dim`.

    Args:
        video (`numpy.ndarray`):
            The video to have its channel dimension set.
        channel_dim (`ChannelDimension`):
            The channel dimension format to use.
        input_channel_dim (`ChannelDimension`, *optional*):
            The channel dimension format of the input video. If not provided, it will be inferred from the input video.

    Returns:
        `np.ndarray`: The video with the channel dimension set to `channel_dim`.
    """
    if not isinstance(video, np.ndarray):
        raise TypeError(f"Input video must be of type np.ndarray, got {type(video)}")

    if input_channel_dim is None:
        input_channel_dim = infer_channel_dimension_format(video)

    target_channel_dim = ChannelDimension(channel_dim)
    if input_channel_dim == target_channel_dim:
        return video

    if target_channel_dim == ChannelDimension.FIRST:
        video = video.transpose((0, 3, 1, 2))
    elif target_channel_dim == ChannelDimension.LAST:
        video = video.transpose((0, 2, 3, 1))
    else:
        raise ValueError("Unsupported channel dimension format: {}".format(channel_dim))

    return video


def get_video_size(video: np.ndarray, channel_dim: ChannelDimension = None) -> Tuple[int, int]:
    """
    Returns the (height, width) dimensions of the video.

    Args:
        video (`np.ndarray`):
            The video to get the dimensions of.
        channel_dim (`ChannelDimension`, *optional*):
            Which dimension the channel dimension is in. If `None`, will infer the channel dimension from the video.

    Returns:
        A tuple of the video's height and width.
    """
    if channel_dim is None:
        channel_dim = infer_channel_dimension_format(video)

    if channel_dim == ChannelDimension.FIRST:
        return video.shape[-2], video.shape[-1]
    elif channel_dim == ChannelDimension.LAST:
        return video.shape[-3], video.shape[-2]
    else:
        raise ValueError(f"Unsupported data format: {channel_dim}")


def get_uniform_frame_indices(total_num_frames: int, num_frames: Optional[int] = None):
    """
    Creates a numpy array for uniform sampling of `num_frame` frames from `total_num_frames`
    when loading a video.

    Args:
        total_num_frames (`int`):
            Total number of frames that a video has.
        num_frames (`int`, *optional*):
            Number of frames to sample uniformly. If not specified, all frames are sampled.

    Returns:
        np.ndarray: np array of frame indices that will be sampled.
    """
    if num_frames is not None:
        indices = np.arange(0, total_num_frames, total_num_frames / num_frames).astype(int)
    else:
        indices = np.arange(0, total_num_frames).astype(int)
    return indices


def read_video_opencv(video_path: str, num_frames: Optional[int] = None, fps: Optional[int] = None):
    """
    Decode the video with open-cv decoder.

    Args:
        video_path (`str`):
            Path to the video file.
        num_frames (`int`, *optional*):
            Number of frames to sample uniformly. Should be passed only when `fps=None`.
            If not specified and `fps==None`, all frames are sampled.
        fps (`int`, *optional*):
            Number of frames to sample per second. Should be passed only when `num_frames=None`.
            If not specified and `num_frames==None`, all frames are sampled.

    Returns:
        np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
    """
    video = cv2.VideoCapture(video_path)
    total_num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = video.get(cv2.CAP_PROP_FPS)
    if num_frames is None and fps is not None:
        num_frames = int(total_num_frames / video_fps * fps)
        if num_frames > total_num_frames:
            raise ValueError(
                f"When loading the video with fps={fps}, we identified that num_frames ({num_frames}) > total_frames ({total_num_frames}) ."
                f"Make sure that fps of a video is less than the requested fps for loading. Detected video_fps={video_fps}"
            )
    indices = get_uniform_frame_indices(total_num_frames, num_frames=num_frames)

    index = 0
    frames = []
    while video.isOpened():
        success, frame = video.read()
        if index in indices:
            height, width, channel = frame.shape
            frames.append(frame[0:height, 0:width, 0:channel])
        if success:
            index += 1
        if index >= total_num_frames:
            break

    video.release()
    return np.stack(frames)


def read_video_decord(video_path: str, num_frames: Optional[int] = None, fps: Optional[int] = None):
    """
    Decode the video with Decord decoder.

    Args:
        video_path (`str`):
            Path to the video file.
        num_frames (`int`, *optional*):
            Number of frames to sample uniformly. Should be passed only when `fps=None`.
            If not specified and `fps==None`, all frames are sampled.
        fps (`int`, *optional*):
            Number of frames to sample per second. Should be passed only when `num_frames=None`.
            If not specified and `num_frames==None`, all frames are sampled.

    Returns:
        np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
    """
    vr = VideoReader(uri=video_path, ctx=cpu(0))  # decord has problems with gpu
    video_fps = vr.get_avg_fps()
    total_num_frames = len(vr)
    if num_frames is None and fps is not None:
        num_frames = int(total_num_frames / video_fps * fps)
        if num_frames > total_num_frames:
            raise ValueError(
                f"When loading the video with fps={fps}, we identified that num_frames ({num_frames}) > total_frames ({total_num_frames}) ."
                f"Make sure that fps of a video is less than the requested fps for loading. Detected video_fps={video_fps}"
            )
    indices = get_uniform_frame_indices(total_num_frames=total_num_frames, num_frames=num_frames)
    frames = vr.get_batch(indices).asnumpy()
    return frames


def read_video_pyav(video_path: str, num_frames: Optional[int] = None, fps: Optional[int] = None):
    """
    Decode the video with PyAV decoder.

    Args:
        video_path (`str`):
            Path to the video file.
        num_frames (`int`, *optional*):
            Number of frames to sample uniformly. Should be passed only when `fps=None`.
            If not specified and `fps==None`, all frames are sampled.
        fps (`int`, *optional*):
            Number of frames to sample per second. Should be passed only when `num_frames=None`.
            If not specified and `num_frames==None`, all frames are sampled.

    Returns:
        np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
    """
    container = av.open(video_path)

    total_num_frames = container.streams.video[0].frames
    video_fps = container.streams.video[0].average_rate  # should we better use `av_guess_frame_rate`?
    if num_frames is None and fps is not None:
        num_frames = int(total_num_frames / video_fps * fps)
        if num_frames > total_num_frames:
            raise ValueError(
                f"When loading the video with fps={fps}, we identified that num_frames ({num_frames}) > total_frames ({total_num_frames}) ."
                f"Make sure that fps of a video is less than the requested fps for loading. Detected video_fps={video_fps}"
            )
    indices = get_uniform_frame_indices(total_num_frames, num_frames=num_frames)

    frames = []
    container.seek(0)
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= 0 and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def read_video_torchvision(video_path: str, num_frames: Optional[int] = None, fps: Optional[int] = None):
    """
    Decode the video with torchvision decoder.

    Args:
        video_path (`str`):
            Path to the video file.
        num_frames (`int`, *optional*):
            Number of frames to sample uniformly. Should be passed only when `fps=None`.
            If not specified and `fps==None`, all frames are sampled.
        fps (`int`, *optional*):
            Number of frames to sample per second. Should be passed only when `num_frames=None`.
            If not specified and `num_frames==None`, all frames are sampled.

    Returns:
        np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
    """
    video, _, info = torchvision_io.read_video(
        video_path,
        start_pts=0.0,
        end_pts=None,
        pts_unit="sec",
        output_format="TCHW",
    )
    video_fps = info["video_fps"]
    total_num_frames = video.size(0) - 1
    if num_frames is None and fps is not None:
        num_frames = int(total_num_frames / video_fps * fps)
        if num_frames > total_num_frames:
            raise ValueError(
                f"When loading the video with fps={fps}, we identified that num_frames ({num_frames}) > total_frames ({total_num_frames}) ."
                f"Make sure that fps of a video is less than the requested fps for loading. Detected video_fps={video_fps}"
            )

    if num_frames is not None:
        idx = torch.linspace(0, video.size(0) - 1, num_frames, dtype=torch.int64)
        return video[idx]

    return video


VIDEO_DECODERS = {
    "decord": read_video_decord,
    "opencv": read_video_opencv,
    "pyav": read_video_pyav,
    "torchvision": read_video_torchvision,
}


def load_video(
    video: Union[str, "VideoInput"],
    num_frames: Optional[int] = None,
    fps: Optional[int] = None,
    backend: str = "pyav",
) -> np.array:
    """
    Loads `video` to a numpy array.

    Args:
        video (`str` or `VideoInput`):
            The video to convert to the numpy array format. Can be a link to video or local path.
        num_frames (`int`, *optional*):
            Number of frames to sample uniformly. If not passed, the whole video is loaded.
        fps (`int`, *optional*):
            Number of frames to sample per second. Should be passed only when `num_frames=None`.
            If not specified and `num_frames==None`, all frames are sampled.
        backend (`str`, *optional*, defaults to `"opencv"`):
            The backend to use when loading the video. Can be any of ["decord", "pyav", "opencv", "torchvision"]. Defaults to "pyav".

    Returns:
        `np.array`: A numpy array of shape (num_frames, channels, height, width).
    """

    if fps is not None and num_frames is not None:
        raise ValueError("`num_frames` and `fps` are mutually exclusive arguments, please use only one!")

    if video.startswith("https://www.youtube.com") or video.startswith("http://www.youtube.com"):
        if not is_yt_dlp_available():
            raise ImportError("To load a video from YouTube url you have  to install `yt_dlp` first.")
        buffer = BytesIO()
        with redirect_stdout(buffer), YoutubeDL() as f:
            f.download([video])
        bytes_obj = buffer.getvalue()
        file_obj = BytesIO(bytes_obj)
    elif video.startswith("http://") or video.startswith("https://"):
        file_obj = BytesIO(requests.get(video).content)
    elif os.path.isfile(video):
        file_obj = video
    elif is_valid_image(video) or (isinstance(video, (list, tuple) and is_valid_image(video[0]))):
        file_obj = None
    else:
        raise TypeError("Incorrect format used for video. Should be an url linking to an video or a local path.")

    # can also load with decord, but not cv2/torchvision
    # both will fail in case of url links
    video_is_url = video.startswith("http://") or video.startswith("https://")
    if video_is_url and backend in ["opencv", "torchvision"]:
        raise ValueError(
            "If you are trying to load a video from URL, you can decode the video only with `pyav` or `decord` as backend"
        )

    if file_obj is None:
        return video

    if (
        (not is_decord_available() and backend == "decord")
        or (not is_av_available() and backend == "pyav")
        or (not is_cv2_available() and backend == "opencv")
        or (not is_torchvision_available() and backend == "torchvision")
    ):
        raise ImportError(
            f"You chose backend={backend} for loading the video but the required library is not found in your environment "
            f"Make sure to install {backend} before loading the video."
        )

    video_decoder = VIDEO_DECODERS[backend]
    video = video_decoder(file_obj, num_frames=num_frames, fps=fps)
    return video


def convert_to_rgb(
    video: np.array,
    data_format: Optional[ChannelDimension] = None,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> np.array:
    """
    Convert video to RGB by blending the transparency layer if it's in RGBA format, otherwise simply returns it.

    Args:
        video (`np.array`):
            The video to convert.
        data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the output video. If unset, will use the inferred format from the input.
        input_data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the input video. If unset, will use the inferred format from the input.
    """
    if not isinstance(video, np.ndarray):
        raise ValueError(f"Video has to be a numpy array to convert to RGB format, but found {type(video)}")

    # np.array usually comes with ChannelDimension.LAST so leet's convert it
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(video)
    video = to_channel_dimension_format(video, ChannelDimension.FIRST, input_channel_dim=input_data_format)

    # 3 channels for RGB already
    if video.shape[-3] == 3:
        return video

    # Grayscale video so we repeat it 3 times for each channel
    if video.shape[-3] == 1:
        return video.repeat(3, -3)

    if not (video[..., 3, :, :] < 255).any():
        return video

    # There is a transparency layer, blend it with a white background.
    # Calculate the alpha proportion for blending.
    alpha = video[..., 3, :, :] / 255.0
    video = (1 - alpha[..., None, :, :]) * 255 + alpha[..., None, :, :] * video[..., 3, :, :]
    return video


def pad(
    video: np.ndarray,
    padding: Union[int, Tuple[int, int], Iterable[Tuple[int, int]]],
    mode: PaddingMode = PaddingMode.CONSTANT,
    constant_values: Union[float, Iterable[float]] = 0.0,
    data_format: Optional[Union[str, ChannelDimension]] = None,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> np.ndarray:
    """
    Pads the `video` with the specified (height, width) `padding` and `mode`.

    Args:
        video (`np.ndarray`):
            The video to pad.
        padding (`int` or `Tuple[int, int]` or `Iterable[Tuple[int, int]]`):
            Padding to apply to the edges of the height, width axes. Can be one of three formats:
            - `((before_height, after_height), (before_width, after_width))` unique pad widths for each axis.
            - `((before, after),)` yields same before and after pad for height and width.
            - `(pad,)` or int is a shortcut for before = after = pad width for all axes.
        mode (`PaddingMode`):
            The padding mode to use. Can be one of:
                - `"constant"`: pads with a constant value.
                - `"reflect"`: pads with the reflection of the vector mirrored on the first and last values of the
                  vector along each axis.
                - `"replicate"`: pads with the replication of the last value on the edge of the array along each axis.
                - `"symmetric"`: pads with the reflection of the vector mirrored along the edge of the array.
        constant_values (`float` or `Iterable[float]`, *optional*):
            The value to use for the padding if `mode` is `"constant"`.
        data_format (`str` or `ChannelDimension`, *optional*):
            The channel dimension format for the output video. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: video in (num_frames, num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: video in (num_frames, height, width, num_channels) format.
            If unset, will use same as the input video.
        input_data_format (`str` or `ChannelDimension`, *optional*):
            The channel dimension format for the input video. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: video in (num_frames, num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: video in (num_frames, height, width, num_channels) format.
            If unset, will use the inferred format of the input video.

    Returns:
        `np.ndarray`: The padded video.

    """
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(video)

    def _expand_for_data_format(values):
        """
        Convert values to be in the format expected by np.pad based on the data format.
        """
        if isinstance(values, (int, float)):
            values = ((values, values), (values, values))
        elif isinstance(values, tuple) and len(values) == 1:
            values = ((values[0], values[0]), (values[0], values[0]))
        elif isinstance(values, tuple) and len(values) == 2 and isinstance(values[0], int):
            values = (values, values)
        elif isinstance(values, tuple) and len(values) == 2 and isinstance(values[0], tuple):
            values = values
        else:
            raise ValueError(f"Unsupported format: {values}")

        # add 0 for channel dimension
        values = (
            ((0, 0), (0, 0), *values) if input_data_format == ChannelDimension.FIRST else ((0, 0), *values, (0, 0))
        )

        # Add additional padding if there's a batch dimension
        values = (0, *values) if video.ndim == 5 else values
        return values

    padding_map = {
        PaddingMode.CONSTANT: "constant",
        PaddingMode.REFLECT: "reflect",
        PaddingMode.REPLICATE: "replicate",
        PaddingMode.SYMMETRIC: "symmetric",
    }
    padding = _expand_for_data_format(padding)

    pad_kwargs = {}
    if mode not in padding_map:
        raise ValueError(f"Invalid padding mode: {mode}")
    elif mode == PaddingMode.CONSTANT:
        pad_kwargs["constant_values"] = _expand_for_data_format(constant_values)

    video = np.pad(video, padding, mode=padding_map[mode], **pad_kwargs)
    video = to_channel_dimension_format(video, data_format, input_data_format) if data_format is not None else video
    return video


def group_videos_by_shape(
    videos: List["torch.Tensor"],
) -> Tuple[Dict[Tuple[int, int], List["torch.Tensor"]], Dict[int, Tuple[Tuple[int, int], int]]]:
    """
    Groups videos by shape.
    Returns a dictionary with the shape as key and a list of videos with that shape as value,
    and a dictionary with the index of the video in the original list as key and the shape and index in the grouped list as value.
    """
    grouped_videos = {}
    grouped_videos_index = {}
    for i, video in enumerate(videos):
        shape = video.shape[-2::]
        if shape not in grouped_videos:
            grouped_videos[shape] = []
        grouped_videos[shape].append(video)
        grouped_videos_index[i] = (shape, len(grouped_videos[shape]) - 1)
    # stack videos with the same shape
    grouped_videos = {shape: torch.stack(videos, dim=0) for shape, videos in grouped_videos.items()}
    return grouped_videos, grouped_videos_index


def reorder_videos(
    processed_videos: Dict[Tuple[int, int], "torch.Tensor"], grouped_videos_index: Dict[int, Tuple[int, int]]
) -> List["torch.Tensor"]:
    """
    Reconstructs a list of videos in the original order.
    """
    return [
        processed_videos[grouped_videos_index[i][0]][grouped_videos_index[i][1]]
        for i in range(len(grouped_videos_index))
    ]


# class ConvertUint8ToFloat(torch.nn.Module):
#     """
#     Converts a video from dtype uint8 to dtype float32.
#     """
#     def __init__(self):
#         super().__init__()
#         self.convert_func = torchvision.transforms.ConvertImageDtype(torch.float32)
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x (torch.Tensor): video tensor with shape (C, T, H, W).
#         """
#         assert x.dtype == torch.uint8, "image must have dtype torch.uint8"
#         return self.convert_func(x)
