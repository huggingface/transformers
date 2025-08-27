"""video processor class for VideoLLaMA3."""

from typing import Optional, Union

from ...image_processing_utils import (
    BatchFeature,
)
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    SizeDict,
    get_image_size,
)
from ...processing_utils import VideosKwargs
from ...utils import (
    TensorType,
    is_torch_available,
    is_torchvision_available,
    is_torchvision_v2_available,
    is_vision_available,
)
from ...utils.import_utils import requires
from ...video_processing_utils import (
    BaseVideoProcessor,
)
from ...video_utils import VideoMetadata, group_videos_by_shape, reorder_videos


if is_vision_available():
    from ...image_utils import PILImageResampling
    from .image_processing_videollama3 import smart_resize

if is_torchvision_available():
    if is_torchvision_v2_available():
        from torchvision.transforms.v2 import functional as F
    else:
        from torchvision.transforms import functional as F


if is_torch_available():
    import torch


class Videollama3VideoProcessorInitKwargs(VideosKwargs):
    min_tokens: Optional[int]
    max_tokens: Optional[int]
    patch_size: Optional[int]
    image_merge_size: Optional[int]
    video_merge_size: Optional[int]
    max_frames: Optional[int]


@requires(backends=("torchvision",))
class Videollama3VideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    min_tokens = 16
    max_tokens = 16384
    patch_size = 14
    video_merge_size = 2
    max_frames = 180
    do_sample_frames = True
    valid_kwargs = Videollama3VideoProcessorInitKwargs
    model_input_names = ["pixel_values_videos", "grid_sizes_videos", "merge_sizes_videos"]

    def sample_frames(
        self,
        video: "torch.Tensor",
        max_frames: int,
        metadata: Optional[Union[VideoMetadata, dict]] = None,
        num_frames: Optional[int] = None,
        fps: Optional[Union[int, float]] = None,
    ):
        """
        Default sampling function which uniformly samples the desired number of frames between 0 and total number of frames.
        If `fps` is passed along with metadata, `fps` frames per second are sampled uniformty. Arguments `num_frames`
        and `fps` are mutually exclusive.

        Args:
            video (`torch.Tensor`):
                Video that need to be sampled.
            max_frames (`int`):
                The maximum number of frames that can be sampled.
            metadata (`VideoMetadata`, *optional*):
                Metadata of the video containing information about total duration, fps and total number of frames.
            num_frames (`int`, *optional*):
                Maximum number of frames to sample. Defaults to `self.num_frames`.
            fps (`int` or `float`, *optional*):
                Target frames to sample per second. Defaults to `self.fps`.

        Returns:
            torch.Tensor:
                Sampled video frames.
        """
        num_frames = num_frames if num_frames is not None else self.num_frames
        fps = fps if fps is not None else self.fps

        if fps is not None and num_frames is not None:
            raise ValueError("`num_frames` and `fps` are mutually exclusive arguments, please use only one!")

        total_num_frames = video.shape[0]

        # If num_frames is not given but fps is, calculate num_frames from fps
        if fps is not None:
            if metadata is None:
                raise ValueError(
                    "Asked to sample `fps` frames per second but no video metadata was provided which is required when sampling with `fps`. "
                    "Please pass in `VideoMetadata` object or use a fixed `num_frames` per input video"
                )
            max_frames = min(max_frames, total_num_frames)
            num_frames = total_num_frames / metadata["fps"] * fps
            num_frames = min(min(num_frames, max_frames), total_num_frames)

        if num_frames is not None and num_frames > total_num_frames:
            raise ValueError(
                f"Video can't be sampled. The inferred `num_frames={num_frames}` exceeds `total_num_frames={total_num_frames}`. "
                "Decrease `num_frames` or `fps` for sampling."
            )

        if num_frames is not None:
            indices = torch.arange(0, total_num_frames, total_num_frames / num_frames).int()
        else:
            indices = torch.arange(0, total_num_frames).int()
        video = video[indices].contiguous()
        timestamps = [int(idx / metadata["fps"]) for idx in indices]

        return video, timestamps

    def _further_process_kwargs(
        self,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        data_format: Optional[ChannelDimension] = None,
        **kwargs,
    ) -> dict:
        """
        Update kwargs that need further processing before being validated
        Can be overridden by subclasses to customize the processing of kwargs.
        """
        if kwargs is None:
            kwargs = {}
        if isinstance(image_mean, list):
            image_mean = tuple(image_mean)
        if isinstance(image_std, list):
            image_std = tuple(image_std)
        if data_format is None:
            data_format = ChannelDimension.FIRST

        kwargs["size"] = SizeDict()
        kwargs["image_mean"] = image_mean
        kwargs["image_std"] = image_std
        kwargs["data_format"] = data_format

        return kwargs

    def _preprocess(
        self,
        videos: list["torch.Tensor"],
        video_metadata: Union[list[VideoMetadata], list[dict]],
        do_convert_rgb: bool,
        do_resize: bool,
        interpolation: Optional["F.InterpolationMode"],
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        do_sample_frames: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        min_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        patch_size: Optional[int] = None,
        video_merge_size: Optional[int] = None,
        fps: Optional[Union[int, float]] = None,
        num_frames: Optional[int] = None,
        max_frames: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        device: Optional["torch.Tensor"] = None,
        **kwargs,
    ):
        timestamps_list = []
        if do_sample_frames:
            if video_metadata is None or (isinstance(video_metadata, list) and video_metadata[0] is None):
                raise ValueError(
                    "Frame sampling is enabled but no video metadata was found. "
                    "Please pass in `VideoMetadata` object per each input video or set `do_sample_frames=False`"
                )
            processed_videos = []
            for video, metadata in zip(videos, video_metadata):
                video, timestamps = self.sample_frames(
                    video,
                    max_frames=max_frames,
                    metadata=metadata,
                    num_frames=num_frames,
                    fps=fps,
                )
                processed_videos.append(video)
                timestamps_list.append(timestamps)
        else:
            # Assume 24 fps by default and prepare timestamps for the whole video when all frames are sampled
            processed_videos = videos
            timestamps_list = [[idx // 24 for idx in range(len(video))] for video in videos]

        # We need to sample frames first before moving to device, if `do_sample_frames=True`. Otherwise
        # moving the whole video incurs high GPU mem usage for long videos
        if device is not None:
            processed_videos = [video.to(device) for video in processed_videos]

        # Group videos by size for batched resizing
        resized_videos = []
        for video in processed_videos:
            height, width = get_image_size(video[0], channel_dim=ChannelDimension.FIRST)
            resized_height, resized_width = height, width
            num_frames = len(video)
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=patch_size * video_merge_size,
                    min_pixels=min_tokens * (patch_size * video_merge_size) ** 2 // num_frames,
                    max_pixels=max_tokens * (patch_size * video_merge_size) ** 2 // num_frames,
                )
                video = self.resize(
                    image=video,
                    size=SizeDict(height=resized_height, width=resized_width),
                    interpolation=interpolation,
                )
                resized_videos.append(video)

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

            batch_size, t, channel = patches.shape[:3]
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            patches = patches.view(
                batch_size,
                t,
                channel,
                grid_h // video_merge_size,
                video_merge_size,
                patch_size,
                grid_w // video_merge_size,
                video_merge_size,
                patch_size,
            )
            patches = patches.permute(0, 1, 3, 6, 4, 7, 2, 5, 8)
            flatten_patches = patches.reshape(
                batch_size,
                t * grid_h * grid_w,
                channel * patch_size * patch_size,
            )

            processed_videos_grouped[shape] = flatten_patches
            processed_grids[shape] = [[t, grid_h, grid_w]] * batch_size

        processed_videos = reorder_videos(processed_videos_grouped, grouped_videos_index)
        processed_grids = reorder_videos(processed_grids, grouped_videos_index)
        pixel_values = torch.cat(processed_videos, dim=0)
        grid_sizes = torch.tensor(processed_grids)
        merge_sizes = torch.tensor(
            [video_merge_size] * grid_sizes.size(0),
            dtype=grid_sizes.dtype,
            device=grid_sizes.device,
        )

        return BatchFeature(
            data={
                "pixel_values_videos": pixel_values,
                "grid_sizes_videos": grid_sizes,
                "merge_sizes_videos": merge_sizes,
                "timestamps": timestamps_list,
            },
            tensor_type=return_tensors,
        )


__all__ = ["Videollama3VideoProcessor"]
