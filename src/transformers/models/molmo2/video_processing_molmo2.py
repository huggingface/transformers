"""Video processor class for Molmo2"""
from functools import partial
import os
import warnings
from contextlib import redirect_stdout
from io import BytesIO
from urllib.parse import urlparse
from typing import Optional, Union, Callable

import numpy as np
import requests
import einops
import torch
import torchvision.transforms

from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ImageInput,
    PILImageResampling,
    SizeDict,
    validate_kwargs,
)
from ...video_utils import (
    VideoInput,
    is_valid_video,
    make_batched_videos,
    make_batched_metadata,
    VideoMetadata,
)
from ...processing_utils import Unpack, VideosKwargs
from ...video_processing_utils import BaseVideoProcessor
from ...utils import logging
from ...feature_extraction_utils import BatchFeature
from ...utils import (
    is_av_available,
    is_decord_available,
    is_torchcodec_available,
    is_yt_dlp_available,
    TensorType,
    logging,
    to_numpy,
)


logger = logging.get_logger(__name__)

MAX_VIDEO_FPS = 8


def normalize_image(
    image: np.ndarray,
    image_mean: list[float],
    image_std: list[float],
) -> np.ndarray:
    image -= np.array(image_mean, dtype=np.float32)[None, None, :]
    image /= np.array(image_std, dtype=np.float32)[None, None, :]
    return image


def resize_image(
    image: np.ndarray,
    desired_output_size: list[int],
    resample: PILImageResampling,
) -> np.ndarray:
    if len(image.shape) == 3:
        is_video = False
        image = torch.permute(torch.from_numpy(image), [2, 0, 1])
    else:
        is_video = True
        image = torch.permute(torch.from_numpy(image), [0, 3, 1, 2])
    dtype = image.dtype
    if torch.is_floating_point(image):
        in_min = 0.0
        in_max = 1.0
        resized = torchvision.transforms.Resize(
            desired_output_size,
            resample,
            antialias=False,
        )(image)
        resized = torch.clip(resized, 0.0, 1.0).to(dtype)
    else:
        assert image.dtype == torch.uint8, "SigLIP expects float images or uint8 images, but got {}".format(image.dtype)
        in_min = 0.0
        in_max = 255.0
        resized = torchvision.transforms.Resize(
            desired_output_size,
            resample,
            antialias=False,
        )(image)
        resized = torch.clip(resized, 0, 255).to(dtype)

    resized = resized.to(torch.float32)
    resized = (resized - in_min) / (in_max - in_min)

    if is_video:
        resized = torch.permute(resized, [0, 2, 3, 1]).numpy()
    else:
        resized = torch.permute(resized, [1, 2, 0]).numpy()

    return resized


def build_resized_image(
    image: np.ndarray,
    base_image_input_size: list[int],
    resample: PILImageResampling,
    image_mean: list[float],
    image_std: list[float],
    image_patch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    resized = resize_image(
        image, base_image_input_size, resample,
    )
    resized = normalize_image(resized, image_mean, image_std)
    if len(resized.shape) == 3:
        resized = np.expand_dims(resized, 0)
    crop_patch_w = base_image_input_size[1] // image_patch_size
    crop_patch_h = base_image_input_size[0] // image_patch_size
    resize_idx = np.arange(crop_patch_w*crop_patch_h).reshape([crop_patch_h, crop_patch_w])
    return resized, resize_idx


def batch_pixels_to_patches(array: np.ndarray, patch_size: int) -> np.ndarray:
    """Reshape images of [n_images, h, w, 3] -> [n_images, n_patches, pixels_per_patch]"""
    if len(array.shape) == 3:
        n_crops, h, w = array.shape
        h_patches = h//patch_size
        w_patches = w//patch_size
        array = np.reshape(array, [n_crops, h_patches, patch_size, w_patches, patch_size])
        array = np.transpose(array, [0, 1, 3, 2, 4])
        array = np.reshape(array, [n_crops, h_patches*w_patches, patch_size*patch_size])
        return array
    else:
        n_crops, h, w, c = array.shape
        h_patches = h//patch_size
        w_patches = w//patch_size
        array = np.reshape(array, [n_crops, h_patches, patch_size, w_patches, patch_size, c])
        array = np.transpose(array, [0, 1, 3, 2, 4, 5])
        array = np.reshape(array, [n_crops, h_patches*w_patches, patch_size*patch_size*c])
        return array


def arange_for_pooling(
    idx_arr: np.ndarray,
    pool_h: int,
    pool_w: int,
) -> np.ndarray:
    h_pad = pool_h * ((idx_arr.shape[0] + pool_h - 1) // pool_h) - idx_arr.shape[0]
    w_pad = pool_w * ((idx_arr.shape[1] + pool_w - 1) // pool_w) - idx_arr.shape[1]
    idx_arr = np.pad(idx_arr, [[h_pad//2, (h_pad+1)//2], [w_pad//2, (w_pad+1)//2]],
                     mode='constant',constant_values=-1)
    return einops.rearrange(
        idx_arr, "(h dh) (w dw) -> h w (dh dw)", dh=pool_h, dw=pool_w)


def image_to_patches_and_grids(
    image: ImageInput,
    base_image_input_size: list[int],
    resample: PILImageResampling,
    image_mean: list[float],
    image_std: list[float],
    image_patch_size: int,
    image_pooling_w: int,
    image_pooling_h: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    :return image_grids, the shape of each image after pooling
    :return crops, the image crops to processes with the ViT
    :return pooled_patch_idx, for each patch_id tokens in `image_tokens`, the indices of the
                                patches in `crops` to pool for that token, masked with -1
    """
    if isinstance(base_image_input_size, int):
        base_image_input_size = (base_image_input_size, base_image_input_size)
    
    pooling_w = image_pooling_w
    pooling_h = image_pooling_h

    resized, resize_idx = build_resized_image(
        image,
        base_image_input_size,
        resample,
        image_mean,
        image_std,
        image_patch_size,
    )
    pooling_idx = arange_for_pooling(resize_idx, pooling_h, pooling_w)
    h, w = pooling_idx.shape[:2]
    pooling_idx = pooling_idx.reshape([-1, pooling_h*pooling_w])
    image_grid = [h, w]
    return (
        image_grid,
        batch_pixels_to_patches(resized, image_patch_size),
        pooling_idx,
    )


def get_candidate_target_fps(
    video_fps: Union[int, float],
    sampling_fps: Union[int, float],
    max_fps: Union[int, float] = MAX_VIDEO_FPS,
) -> list[float]:
    """
    Return the subset of `video_fps` factors that remain multiples of `sampling_fps`.

    Examples:
        >>> get_candidate_target_fps(video_fps=6, sampling_fps=2)
        [2, 6]
        >>> get_candidate_target_fps(video_fps=5, sampling_fps=1)
        [1, 5]
        >>> get_candidate_target_fps(video_fps=2, sampling_fps=2)
        [2]
        >>> get_candidate_target_fps(video_fps=5, sampling_fps=2)
        Traceback (most recent call last):
            ...
        ValueError: sampling_fps=2 must divide video_fps=5 to produce consistent frame steps.
    """
    video_fps = int(video_fps)
    sampling_fps = int(sampling_fps)
    max_fps = int(max_fps)

    if sampling_fps is None:
        raise ValueError("sampling_fps must be provided")
    if video_fps <= 0 or sampling_fps <= 0:
        raise ValueError(f"video_fps and sampling_fps must be positive (got {video_fps}, {sampling_fps})")
    if video_fps % sampling_fps != 0:
        raise ValueError(f"sampling_fps={sampling_fps} must divide video_fps={video_fps}.")

    candidates = []
    for candidate in range(sampling_fps, video_fps + 1, sampling_fps):
        if candidate > max_fps:
            break
        if video_fps % candidate == 0:
            candidates.append(float(candidate))
    
    return candidates


def read_video_decord(
    video_path,
    sample_timestamps_fn: Callable,
    **kwargs,
) -> np.ndarray:
    """
    Decode a video using the Decord backend.

    Args:
        video_path (`str`):
            Path to the video file.
        sample_timestamps_fn (`Callable`):
            A callable function that will return timestamps at which the video should be sampled.

    Returns:
        tuple[`np.array`, `VideoMetadata`]: A tuple containing:
            - Numpy array of frames in RGB (shape: [num_frames, height, width, 3]).
            - `VideoMetadata` object.
    """
    # Lazy import from decord
    import importlib
    decord = importlib.import_module("decord")

    vr = decord.VideoReader(uri=video_path, ctx=decord.cpu(0))  # decord has problems with gpu
    video_fps = vr.get_avg_fps()
    total_num_frames = len(vr)
    time_stamps = vr.get_frame_timestamp(list(range(len(vr))))
    duration = time_stamps[-1][1] - time_stamps[0][0]

    metadata = VideoMetadata(
        total_num_frames=int(total_num_frames),
        fps=float(video_fps),
        duration=float(duration),
        video_backend="decord",
    )

    target_timestamps = sample_timestamps_fn(metadata=metadata, **kwargs)
    target_timestamps = np.array(target_timestamps)
    offset = time_stamps[0, 0]

    ix = np.searchsorted(time_stamps[:, 1], target_timestamps + offset, side='right')
    ix = np.minimum(ix, len(time_stamps) - 1)

    video = vr.get_batch(ix).asnumpy()
    metadata.update(
        {
            "frames_indices": target_timestamps * video_fps,
            "height": video.shape[1],
            "width": video.shape[2],
        }
    )
    return video, metadata


def read_video_torchcodec(
    video_path,
    sample_timestamps_fn: Callable,
    **kwargs,
) -> np.ndarray:
    """
    Decode a video using torchcodec decoder.

    Args:
        video_path (`str`):
            Path to the video file.
        sample_timestamps_fn (`Callable`):
            A callable function that will return timestamps at which the video should be sampled.

    Returns:
        tuple[`np.array`, `VideoMetadata`]: A tuple containing:
            - Numpy array of frames in RGB (shape: [num_frames, height, width, 3]).
            - `VideoMetadata` object.
    """
    # Lazy import torchcodec
    import importlib
    torchcodec = importlib.import_module("torchcodec")

    decoder = torchcodec.decoders.VideoDecoder(
        video_path,
        # Interestingly `exact` mode takes less than approximate when we load the whole video
        seek_mode="exact",
        # Allow FFmpeg decide on the number of threads for efficiency
        num_ffmpeg_threads=0,
    )
    # If the first frame starts at > 0, we effectively clip the video starting at that time
    # since (most) video players would also skip to that time
    time_offset = decoder.metadata.begin_stream_seconds_from_content
    # Note this duration does assume we started playing at `time_offset`
    duration = decoder.metadata.duration_seconds

    metadata = VideoMetadata(
        total_num_frames=decoder.metadata.num_frames,
        fps=decoder.metadata.average_fps,
        duration=duration,
        video_backend="torchcodec",
        height=decoder.metadata.height,
        width=decoder.metadata.width,
    )

    target_timestamps = sample_timestamps_fn(metadata=metadata, **kwargs)

    # Floating point/rounding issues might cause `target_timestamps` to be very slightly
    # out-of-bounds, to handle this we sanity check then clip them
    assert all(x >= 0 for x in target_timestamps)
    assert all(x < duration+1e-6 for x in target_timestamps)
    # 1e-6 padding since torchcodec can throw out-of-bounds errors even if you ask for the
    # exact boundary value, we should still get the first/last frame anyway
    max_timestamp = decoder.metadata.end_stream_seconds_from_content - 1e-6
    min_timestamp = decoder.metadata.begin_stream_seconds_from_content + 1e-6
    # Note we avoid using numpy ops here to reduce floating precision issues
    timestamps = [x + time_offset for x in target_timestamps]
    timestamps = [max(min_timestamp, min(max_timestamp, x)) for x in timestamps]

    video = decoder.get_frames_played_at(timestamps).data.numpy().transpose(0, 2, 3, 1)  # Convert to THWC format
    target_timestamps = np.array(target_timestamps)
    metadata.frames_indices = target_timestamps * metadata.fps

    return video, metadata


def read_video_pyav(
    video_path,
    sample_timestamps_fn: Callable,
    **kwargs,
) -> np.ndarray:
    """
    Decode a video using the PyAV backend.

    Args:
        video_path (`str`):
            Path to the video file.
        sample_timestamps_fn (`Callable`):
            A callable function that will return timestamps at which the video should be sampled.

    Returns:
        tuple[`np.array`, `VideoMetadata`]: A tuple containing:
            - Numpy array of frames in RGB (shape: [num_frames, height, width, 3]).
            - `VideoMetadata` object.
    """
    # Lazy import torchcodec
    import importlib
    av = importlib.import_module("av")

    with av.open(video_path) as container:
        video_stream = container.streams.video[0]
        fps = video_stream.average_rate or video_stream.guessed_rate
        it = container.decode(video=0)
        frames = list(it)

        stream = container.streams.video[0]
        start = frames[0].pts * stream.time_base
        container_end = stream.duration
        if container_end is not None:
            container_end *= stream.time_base
        if container_end is None or container_end < frames[-1].pts:
            # Some problem with stream duration, so use the frame PTS directly
            # and guess the duration of the last frame
            end = frames[-1].pts * stream.time_base + 1/fps
        else:
            end = container_end
        duration = float(end - start)

        metadata = VideoMetadata(
            total_num_frames=len(frames),
            fps=float(fps),
            duration=float(duration),
            video_backend="pyav",
            height=video_stream.height,
            width=video_stream.width,
        )

        target_timestamps = sample_timestamps_fn(metadata=metadata, **kwargs)
        offset = float(start)

        target_timestamps = np.array(target_timestamps)
        end_time_stamps = np.array([float(frame.pts * stream.time_base) for frame in frames[1:]] + [duration])
        indices = np.searchsorted(end_time_stamps, target_timestamps + offset, side='right')
        indices = np.minimum(indices, len(end_time_stamps) - 1)

        video = np.stack(
            [frames[i].to_ndarray(format="rgb24", channel_last=True) for i in indices],
            axis=0,
        )

        metadata.frames_indices = target_timestamps * fps

        return video, metadata


VIDEO_DECODERS = {
    "decord": read_video_decord,
    "torchcodec": read_video_torchcodec,
    "pyav": read_video_pyav,
}


def load_video(
    video: VideoInput,
    backend: str = "decord",
    sample_timestamps_fn: Optional[Callable] = None,
    **kwargs,
):
    """
    Loads `video` to a numpy array.

    Args:
        video (`VideoInput`):
            The video to convert to the numpy array format. Can be a link to video or local path.
        backend (`str`, *optional*, defaults to `"decord"`):
            The backend to use when loading the video. Can be any of ["decord", "pyav", ""torchcodec"]. Defaults to "decord".
        sample_timestamps_fn (`Callable`):
            A callable function that will return timestamps at which the video should be sampled.
    """

    # Early exit if provided an array or `PIL` frames
    if not isinstance(video, str):
        metadata = [None] * len(video)
        return video, metadata

    if urlparse(video).netloc in ["www.youtube.com", "youtube.com"]:
        if not is_yt_dlp_available():
            raise ImportError("To load a video from YouTube url you have  to install `yt_dlp` first.")
        # Lazy import from yt_dlp
        import importlib
        yt_dlp = importlib.import_module("yt_dlp")

        buffer = BytesIO()
        with redirect_stdout(buffer), yt_dlp.YoutubeDL() as f:
            f.download([video])
        bytes_obj = buffer.getvalue()
        file_obj = BytesIO(bytes_obj)
    elif video.startswith("http://") or video.startswith("https://"):
        file_obj = BytesIO(requests.get(video).content)
    elif os.path.isfile(video):
        file_obj = video
    else:
        raise TypeError("Incorrect format used for video. Should be an url linking to an video or a local path.")

    # can also load with decord, but not cv2/torchvision
    # both will fail in case of url links
    video_is_url = video.startswith("http://") or video.startswith("https://")
    if video_is_url and backend == "opencv":
        raise ValueError("If you are trying to load a video from URL, you cannot use 'opencv' as backend")

    if (
        (not is_decord_available() and backend == "decord")
        or (not is_torchcodec_available() and backend == "torchcodec")
        or (not is_av_available() and backend == "pyav")
    ):
        raise ImportError(
            f"You chose backend={backend} for loading the video but the required library is not found in your environment "
            f"Make sure to install {backend} before loading the video."
        )
    
    video_decoder = VIDEO_DECODERS[backend]
    video, metadata = video_decoder(file_obj, sample_timestamps_fn, **kwargs)
    return video, metadata


def get_target_fps(
    video_fps: float,
    max_frames: int,
    total_frames: int,
    frame_sample_mode: str,
    candidate_target_fps: tuple[float],
) -> float:
    """
    Get the target fps that best spans the video and has the most frames sampled
    """
    num_frames_sampled = 0
    selected_target_fps = None
    for target_fps in candidate_target_fps:
        step_size = max(int(video_fps / target_fps), 1)
        num_frames_sampled_at_fps = int(total_frames / step_size)
        if num_frames_sampled == 0:
            if "uniform" in frame_sample_mode:
                if num_frames_sampled_at_fps > max_frames:
                    break
            selected_target_fps = target_fps
            num_frames_sampled = num_frames_sampled_at_fps

        else:
            # the candidate sampling fps increases so frame count can't decrease
            assert num_frames_sampled <= num_frames_sampled_at_fps
            if num_frames_sampled_at_fps > max_frames:
                # choose the sampling fps that spans the video
                continue

            elif num_frames_sampled_at_fps > num_frames_sampled:
                # both are less than max_frames, choose the one with higher density of frames sampled
                selected_target_fps = target_fps
                num_frames_sampled = num_frames_sampled_at_fps
    return selected_target_fps


def get_frame_times_and_chosen_fps(
    selected_target_fps,
    total_frames,
    max_frames,
    video_fps
):
    if selected_target_fps is None:
        frame_indices = np.linspace(0, total_frames, max_frames, endpoint=False, dtype=int)
    else:
        step_size = max(int(video_fps / selected_target_fps), 1)
        frame_indices = np.arange(0, total_frames, step_size)
    if len(frame_indices) > max_frames:
        frame_indices = frame_indices[:max_frames]
    return selected_target_fps, frame_indices


class Molmo2VideoProcessorKwargs(VideosKwargs, total=False):
    patch_size: Optional[int]
    pooling_size: Optional[list[int]]
    frame_sample_mode: Optional[str]
    max_fps: Optional[int]
    sampling_fps: Optional[int]


class Molmo2VideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BILINEAR
    size = {"height": 378, "width": 378}
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    patch_size = 14
    pooling_size = [3, 3]
    do_sample_frames = True
    frame_sample_mode = "uniform_last_frame"
    max_fps = 2
    sampling_fps = 2
    valid_kwargs = Molmo2VideoProcessorKwargs
    model_input_names = ["pixel_values_videos", "video_token_pooling", "video_grids"]

    def __init__(self, **kwargs: Unpack[Molmo2VideoProcessorKwargs]):
        super().__init__(**kwargs)
        if self.size is not None and (
            self.size.get("height", None) is None or self.size.get("width", None) is None
        ):
            raise ValueError("size must contain 'height' and 'width' keys.")

    def _further_process_kwargs(
        self,
        size: Optional[SizeDict] = None,
        **kwargs,
    ) -> dict:
        """
        Update kwargs that need further processing before being validated
        Can be overridden by subclasses to customize the processing of kwargs.
        """
        if size is not None and ("height" not in size or "width" not in size):
            raise ValueError("size must contain 'height' and 'width' keys.")

        return super()._further_process_kwargs(size=size, **kwargs)

    def sample_times(
        self,
        metadata: VideoMetadata,
        frame_sample_mode: str,
        num_frames: int,
        max_fps: Optional[int] = None,
        sampling_fps: Optional[int] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Time-based sampling if an array video is passed
        Args:
            metadata (`VideoMetadata`):
                Metadata of the video containing information about total duration, fps and total number of frames.
            frame_sample_mode (`str`, *optional*):
                Mode to sample frames. Defaults to `self.frame_sample_mode`.
            num_frames (`int`, *optional*):
                Maximum number of frames to sample. Defaults to `self.num_frames`.
            man_fps (`int`, *optional*):
                Maximum frames per second to sample.
            sampling_fps (`int`, *optional*):
                Sampling frames per second. Defaults to `self.sampling_fps`.
                Used when `frame_sample_mode` is `"fps"`.
        """
        frame_sample_mode = frame_sample_mode or self.frame_sample_mode
        num_frames = num_frames or self.num_frames
        sampling_fps = sampling_fps or self.sampling_fps

        duration = metadata.duration or metadata.total_num_frames / metadata.fps
        if frame_sample_mode == "fps":
            candidate_target_fps = get_candidate_target_fps(metadata.fps, sampling_fps)
            # Try larger and larger FPSs until we hit one that can't span the video
            target_fps = candidate_target_fps[0]
            for candidate_fps in candidate_target_fps[1:]:
                if num_frames / candidate_fps < duration:
                    break
                target_fps = candidate_fps
            times = np.arange(0, num_frames) / target_fps
            times = times[times < duration]
            return times
        elif frame_sample_mode == "uniform_last_frame":
            if max_fps is not None:
                max_duration = (num_frames-1) / max_fps  # -1 to include the last frame
                if max_duration < duration:
                    times = np.linspace(
                        0, duration, num=num_frames, endpoint=True, dtype=np.float64
                    )
                else:
                    times = np.arange(0.0, stop=duration, step=1/max_fps)
                    times = np.concatenate([times, [duration]], axis=0)
                    assert len(times) <= num_frames
            else:
                times = np.linspace(
                    0, duration, num=num_frames, endpoint=True, dtype=np.float64
                )
            return times
        else:
            raise NotImplementedError(frame_sample_mode)

    def sample_frames(
        self,
        metadata: VideoMetadata,
        frame_sample_mode: Optional[str] = None,
        num_frames: Optional[int] = None,
        max_fps: Optional[int] = None,
        sampling_fps: Optional[int] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Frame-based sampling if an array video is passed
        Args:
            metadata (`VideoMetadata`):
                Metadata of the video containing information about total duration, fps and total number of frames.
            frame_sample_mode (`str`, *optional*):
                Mode to sample frames. Defaults to `self.frame_sample_mode`.
            num_frames (`int`, *optional*):
                Maximum number of frames to sample. Defaults to `self.num_frames`.
            max_fps (`int`, *optional*):
                Maximum frames per second to sample.
            sampling_fps (`int`, *optional*):
                Sampling frames per second. Defaults to `self.sampling_fps`.
                Used when `frame_sample_mode` is `"fps"`.
        """
        frame_sample_mode = frame_sample_mode or self.frame_sample_mode
        num_frames = num_frames or self.num_frames
        sampling_fps = sampling_fps or self.sampling_fps

        total_num_frames = metadata.total_num_frames
        if frame_sample_mode == "uniform_last_frame" and max_fps is not None:
            duration = total_num_frames / metadata.fps
            if total_num_frames <= 2:
                return np.arange(total_num_frames).astype(int)
            if duration > (num_frames - 1) / max_fps:  # -1 to include the last frame
                # uniform fallback
                indices = np.linspace(
                    0,
                    total_num_frames - 1,
                    num=min(num_frames, total_num_frames),
                    endpoint=True,
                ).astype(int)
                return indices
            else:
                float_indices = np.arange(
                    0.0, stop=total_num_frames - 1, step=float(metadata.fps / max_fps),
                )
                if np.round(float_indices[-1]) != total_num_frames - 1:
                    float_indices = np.concatenate([float_indices, [total_num_frames - 1]], axis=0)
                indices = np.round(float_indices).astype(int)
                assert indices[-1] < total_num_frames
                assert len(float_indices) <= num_frames
                return indices
        elif frame_sample_mode == "uniform_last_frame":
            indices = np.linspace(
                0, total_num_frames - 1, num=min(num_frames, total_num_frames), endpoint=True,
            ).astype(int)
            return indices
        elif frame_sample_mode == "fps":
            candidate_target_fps = get_candidate_target_fps(metadata.fps, sampling_fps)
            selected_target_fps = get_target_fps(
                metadata.fps,
                num_frames,
                total_num_frames,
                frame_sample_mode,
                candidate_target_fps,
            )
            _, indices = get_frame_times_and_chosen_fps(
                selected_target_fps,
                total_num_frames,
                num_frames,
                metadata.fps,
            )
            return indices
        else:
            raise NotImplementedError(frame_sample_mode)
    
    def fetch_videos(
        self,
        video_url_or_urls: Union[str, list[str], list[list[str]]],
        sample_timestamps_fn=None
    ):
        """
        Convert a single or a list of urls into the corresponding `np.array` objects.

        If a single url is passed, the return value will be a single object. If a list is passed a list of objects is
        returned.
        """
        if (
            (not is_decord_available())
            and (not is_torchcodec_available())
            and (not is_av_available())
        ):
            raise ImportError(
                "Molmo2VideoProcessor requires `decord`, `torchcodec`, or `av` to be installed."
            )

        if is_decord_available():
            backend = "decord"
        elif is_torchcodec_available():
            warnings.warn(
                "`decord` is not installed and cannot be used to decode the video by default. "
                "Falling back to `torchcodec`."
            )
            backend = "torchcodec"
        else:
            warnings.warn(
                "`decord` is not installed and cannot be used to decode the video by default. "
                "Falling back to `PyAV`."
            )
            backend = "pyav"

        if isinstance(video_url_or_urls, list):
            return list(zip(*[self.fetch_videos(x, sample_timestamps_fn=sample_timestamps_fn) for x in video_url_or_urls]))
        else:
            return load_video(video_url_or_urls, backend=backend, sample_timestamps_fn=sample_timestamps_fn)

    def _decode_and_sample_videos(
        self,
        videos: VideoInput,
        video_metadata: Union[VideoMetadata, dict],
        do_sample_frames: Optional[bool] = None,
        sample_indices_fn: Optional[Callable] = None,
        sample_timestamps_fn: Optional[Callable] = None,
    ):
        """
        Decode input videos and sample frames if needed.
        """
        videos = make_batched_videos(videos)
        video_metadata = make_batched_metadata(videos, video_metadata=video_metadata)

        # Framed-based sampling if an array video is passed
        # Otherwise, time-based sampling with decoding
        if is_valid_video(videos[0]) and do_sample_frames:
            assert video_metadata[0].fps is not None, "FPS must be provided for video input"
            sampled_videos = []
            sampled_metadata = []
            for video, metadata in zip(videos, video_metadata):
                indices = sample_indices_fn(metadata=metadata)
                metadata.frames_indices = indices
                sampled_videos.append(video[indices])
                sampled_metadata.append(metadata)
            videos = sampled_videos
            video_metadata = sampled_metadata
        elif not is_valid_video(videos[0]):
            if sample_indices_fn is None:
                logger.warning(
                    "do_sample_frames is False, but video array is not provided: "
                    "Will decode the video and sample frames using Molmo2's default sampling mode"
                )
            if isinstance(videos[0], list):
                raise ValueError(
                    "A list of images is not supported for video input!"
                )
            else:
                videos, video_metadata = self.fetch_videos(videos, sample_timestamps_fn=sample_timestamps_fn)
        
        return videos, video_metadata
    
    def _prepare_input_videos(
        self,
        videos: VideoInput,
        **kwargs,
    ) -> list[np.ndarray]:
        processed_videos = [to_numpy(video) for video in videos]
        return processed_videos
    
    def preprocess(
        self,
        videos: VideoInput,
        **kwargs: Unpack[Molmo2VideoProcessorKwargs],
    ) -> BatchFeature:
        validate_kwargs(
            captured_kwargs=kwargs.keys(),
            valid_processor_keys=list(self.valid_kwargs.__annotations__.keys()) + ["return_tensors"],
        )

        # Set default kwargs from self. This ensures that if a kwarg is not provided
        # by the user, it gets its default value from the instance, or is set to None.
        for kwarg_name in self.valid_kwargs.__annotations__:
            kwargs.setdefault(kwarg_name, getattr(self, kwarg_name, None))
        
        do_sample_frames = kwargs.pop("do_sample_frames")
        video_metadata = kwargs.pop("video_metadata")

        sample_indices_fn = partial(self.sample_frames, **kwargs) if do_sample_frames else None
        sample_timestamps_fn = partial(self.sample_times, **kwargs)
        videos, video_metadata = self._decode_and_sample_videos(
            videos,
            video_metadata=video_metadata,
            do_sample_frames=do_sample_frames,
            sample_indices_fn=sample_indices_fn,
            sample_timestamps_fn=sample_timestamps_fn,
        )
        videos = self._prepare_input_videos(videos=videos)

        kwargs = self._further_process_kwargs(**kwargs)

        return_metadata = kwargs.pop("return_metadata")
        preprocessed_videos = self._preprocess(videos=videos, **kwargs)
        if return_metadata:
            preprocessed_videos["video_metadata"] = video_metadata
        return preprocessed_videos
    
    def _preprocess(
        self,
        videos: list[np.ndarray],
        size: Optional[SizeDict] = None,
        resample: Optional[PILImageResampling] = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        do_convert_rgb: Optional[bool] = None,
        patch_size: Optional[int] = None,
        pooling_size: Optional[list[int]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess a video for the model.
        Args:
            videos (`VideoInput`):
                Video to preprocess.
            size (`SizeDict`, *optional*, defaults to `self.size`):
                Size of the image after resizing.
            resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
                Resampling filter to use when resizing the image. This can be one of the enum `PILImageResampling`. Only
                has an effect if `do_resize` is set to `True`.
            image_mean (`float` or `list[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
            image_std (`float` or `list[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
                `True`.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            patch_size (`int`, *optional*, defaults to `self.patch_size`):
                The spatial patch size of the vision encoder.
            pooling_size (`list[int]`, *optional*, defaults to `self.pooling_size`):
                The pooling size of the vision adapter.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.

        Returns:
            A `BatchFeature` containing the following keys:
                - `pixel_values_videos`: The preprocessed videos.
                - `video_token_pooling`: The indices of the patches in `crops` to pool for each token in `video_tokens`.
                - `video_grids`: The video grids.
        """
        if size.height is None or size.width is None:
            raise ValueError("size must contain 'height' and 'width' keys.")
        
        base_image_input_size = [size.height, size.width]

        resample = resample or self.resample
        image_mean = image_mean or self.image_mean
        image_std = image_std or self.image_std
        do_convert_rgb = do_convert_rgb or self.do_convert_rgb

        patch_size = patch_size or self.patch_size
        pooling_size = pooling_size or self.pooling_size

        image_pooling_h, image_pooling_w = pooling_size

        batch_grids = []
        batch_crops = []
        batch_pooled_patches_idx = []

        for video in videos:
            all_crops = []
            pooled_patches_idx = []

            for frame in video:
                image_grid, crops, pooled_idx = image_to_patches_and_grids(
                    frame,
                    base_image_input_size,
                    resample,
                    image_mean,
                    image_std,
                    patch_size,
                    image_pooling_w,
                    image_pooling_h,
                )
                offset = sum(np.prod(x.shape[:2]) for x in all_crops)
                pooled_idx_with_offset = np.where(pooled_idx >= 0, pooled_idx + offset, pooled_idx)
                pooled_patches_idx.append(pooled_idx_with_offset)
                all_crops.append(crops)

            video_grid = np.array([len(video), image_grid[0], image_grid[1]])
            all_crops = np.concatenate(all_crops, 0)
            pooled_patches_idx = np.concatenate(pooled_patches_idx, 0)

            batch_grids.append(video_grid)
            batch_crops.append(all_crops)
            batch_pooled_patches_idx.append(pooled_patches_idx)
        
        video_grids = np.stack(batch_grids, 0)
        pixel_values_videos = np.concatenate(batch_crops, 0)
        video_token_pooling = np.concatenate(batch_pooled_patches_idx, 0)
        
        data =dict(
            pixel_values_videos=pixel_values_videos,
            video_token_pooling=video_token_pooling,
            video_grids=video_grids,
        )

        return BatchFeature(data, tensor_type=return_tensors)


Molmo2VideoProcessor.register_for_auto_class()