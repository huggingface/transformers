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

import copy
import json
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .dynamic_module_utils import custom_object_save
from .image_processing_utils import (
    BatchFeature,
    get_size_dict,
)
from .image_processing_utils_fast import BaseImageProcessorFast
from .image_utils import (
    ChannelDimension,
    SizeDict,
    validate_kwargs,
)
from .processing_utils import Unpack, VideosKwargs
from .utils import (
    VIDEO_PROCESSOR_NAME,
    TensorType,
    add_model_info_to_auto_map,
    add_model_info_to_custom_pipelines,
    add_start_docstrings,
    cached_file,
    copy_func,
    download_url,
    is_offline_mode,
    is_remote_url,
    is_torch_available,
    is_torchvision_available,
    is_torchvision_v2_available,
    is_vision_available,
    logging,
)
from .utils.import_utils import requires
from .video_utils import (
    VideoInput,
    group_videos_by_shape,
    load_video,
    make_batched_videos,
    reorder_videos,
    to_channel_dimension_format,
)


if is_vision_available():
    from .image_utils import PILImageResampling

if is_torch_available():
    import torch

if is_torchvision_available():
    from .image_utils import pil_torch_interpolation_mapping

    if is_torchvision_v2_available():
        from torchvision.transforms.v2 import functional as F
    else:
        from torchvision.transforms import functional as F

logger = logging.get_logger(__name__)


BASE_VIDEO_PROCESSOR_DOCSTRING = r"""
    Args:
        do_resize (`bool`, *optional*, defaults to `self.do_resize`):
            Whether to resize the video's (height, width) dimensions to the specified `size`. Can be overridden by the
            `do_resize` parameter in the `preprocess` method.
        size (`dict`, *optional*, defaults to `self.size`):
            Size of the output video after resizing. Can be overridden by the `size` parameter in the `preprocess`
            method.
        size_divisor (`int`, *optional*, defaults to `self.size_divisor`):
            The size by which to make sure both the height and width can be divided.
        default_to_square (`bool`, *optional*, defaults to `self.default_to_square`):
            Whether to default to a square video when resizing, if size is an int.
        resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
            Resampling filter to use if resizing the video. Only has an effect if `do_resize` is set to `True`. Can be
            overridden by the `resample` parameter in the `preprocess` method.
        do_center_crop (`bool`, *optional*, defaults to `self.do_center_crop`):
            Whether to center crop the video to the specified `crop_size`. Can be overridden by `do_center_crop` in the
            `preprocess` method.
        do_pad (`bool`, *optional*):
            Whether to pad the video to the `(max_height, max_width)` of the videos in the batch.
        crop_size (`Dict[str, int]` *optional*, defaults to `self.crop_size`):
            Size of the output video after applying `center_crop`. Can be overridden by `crop_size` in the `preprocess`
            method.
        do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
            Whether to rescale the video by the specified scale `rescale_factor`. Can be overridden by the
            `do_rescale` parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `self.rescale_factor`):
            Scale factor to use if rescaling the video. Only has an effect if `do_rescale` is set to `True`. Can be
            overridden by the `rescale_factor` parameter in the `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
            Whether to normalize the video. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method. Can be overridden by the `do_normalize` parameter in the `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
            Mean to use if normalizing the video. This is a float or list of floats the length of the number of
            channels in the video. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
            overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
            Standard deviation to use if normalizing the video. This is a float or list of floats the length of the
            number of channels in the video. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_convert_rgb (`bool`, *optional*, defaults to `self.image_std`):
            Whether to convert the video to RGB.
        return_tensors (`str` or `TensorType`, *optional*):
            Returns stacked tensors if set to `pt, otherwise returns a list of tensors.
        data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
            The channel dimension format for the output video. Can be one of:
            - `"channels_first"` or `ChannelDimension.FIRST`: video in (num_channels, height, width) format.
            - `"channels_last"` or `ChannelDimension.LAST`: video in (height, width, num_channels) format.
            - Unset: Use the channel dimension format of the input video.
        input_data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format for the input video. If unset, the channel dimension format is inferred
            from the input video. Can be one of:
            - `"channels_first"` or `ChannelDimension.FIRST`: video in (num_channels, height, width) format.
            - `"channels_last"` or `ChannelDimension.LAST`: video in (height, width, num_channels) format.
            - `"none"` or `ChannelDimension.NONE`: video in (height, width) format.
        device (`torch.device`, *optional*):
            The device to process the videos on. If unset, the device is inferred from the input videos."""


@add_start_docstrings(
    "Constructs a base VideoProcessor.",
    BASE_VIDEO_PROCESSOR_DOCSTRING,
)
@requires(backends=("vision", "torchvision"))
class BaseVideoProcessor(BaseImageProcessorFast):
    _auto_class = None

    resample = None
    image_mean = None
    image_std = None
    size = None
    size_divisor = None
    default_to_square = True
    crop_size = None
    do_resize = None
    do_center_crop = None
    do_pad = None
    do_rescale = None
    rescale_factor = 1 / 255
    do_normalize = None
    do_convert_rgb = None
    valid_kwargs = VideosKwargs
    model_input_names = ["pixel_values_videos"]

    def __init__(self, **kwargs: Unpack[VideosKwargs]) -> None:
        super().__init__()

        self._processor_class = kwargs.pop("processor_class", None)

        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err

        # Prepare size related keys and turn then into `SizeDict`
        size = kwargs.pop("size", self.size)
        self.size = (
            get_size_dict(size=size, default_to_square=kwargs.pop("default_to_square", self.default_to_square))
            if size is not None
            else None
        )
        crop_size = kwargs.pop("crop_size", self.crop_size)
        self.crop_size = get_size_dict(crop_size, param_name="crop_size") if crop_size is not None else None

        # Save valid kwargs in a list for further processing
        self.model_valid_processing_keys = list(self.valid_kwargs.__annotations__.keys())
        for key in self.model_valid_processing_keys:
            if kwargs.get(key) is not None:
                setattr(self, key, kwargs[key])
            else:
                setattr(self, key, getattr(self, key, None))

    def __call__(self, videos, **kwargs) -> BatchFeature:
        return self.preprocess(videos, **kwargs)

    def convert_to_rgb(
        self,
        video: "torch.Tensor",
    ) -> VideoInput:
        """
        Converts a video to RGB format.

        Args:
            video (`"torch.Tensor"`):
                The video to convert.

        Returns:
            `torch.Tensor`: The converted video.
        """

        video = F.grayscale_to_rgb(video)
        if video.shape[-3] == 3 or not (video[..., 3, :, :] < 255).any():
            return video

        # There is a transparency layer, blend it with a white background.
        # Calculate the alpha proportion for blending.
        alpha = video[..., 3, :, :] / 255.0
        video = (1 - alpha[..., None, :, :]) * 255 + alpha[..., None, :, :] * video[..., :3, :, :]
        return video

    def _prepare_input_videos(
        self,
        videos: VideoInput,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        device: Optional["torch.device"] = None,
    ) -> List["torch.Tensor"]:
        """
        Prepare the input videos for processing.
        """
        videos = make_batched_videos(videos)
        processed_videos = []
        for video in videos:
            # `make_batched_videos` always returns a 4D array per video
            if isinstance(video, np.ndarray):
                video = to_channel_dimension_format(video, ChannelDimension.FIRST, input_data_format)
                # not using F.to_tensor as it doesn't handle (C, H, W) numpy arrays
                video = torch.from_numpy(video).contiguous()

            # Now that we have torch tensors, we can move them to the right device
            if device is not None:
                video = video.to(device)

            processed_videos.append(video)
        return processed_videos

    @add_start_docstrings(BASE_VIDEO_PROCESSOR_DOCSTRING)
    def preprocess(
        self,
        videos: VideoInput,
        **kwargs: Unpack[VideosKwargs],
    ) -> BatchFeature:
        validate_kwargs(captured_kwargs=kwargs.keys(), valid_processor_keys=self.valid_kwargs.__annotations__.keys())
        # Set default kwargs from self. This ensures that if a kwarg is not provided
        # by the user, it gets its default value from the instance, or is set to None.
        for kwarg_name in self.valid_kwargs.__annotations__:
            kwargs.setdefault(kwarg_name, getattr(self, kwarg_name, None))

        input_data_format = kwargs.pop("input_data_format")
        device = kwargs.pop("device")
        videos = self._prepare_input_videos(videos=videos, input_data_format=input_data_format, device=device)

        kwargs = self._further_process_kwargs(**kwargs)
        self._validate_preprocess_kwargs(**kwargs)

        # torch resize uses interpolation instead of resample
        resample = kwargs.pop("resample")
        kwargs["interpolation"] = (
            pil_torch_interpolation_mapping[resample] if isinstance(resample, (PILImageResampling, int)) else resample
        )

        # Pop kwargs that are not needed in _preprocess
        kwargs.pop("default_to_square")
        kwargs.pop("data_format")

        return self._preprocess(videos=videos, **kwargs)

    def _preprocess(
        self,
        videos: List["torch.Tensor"],
        do_convert_rgb: bool,
        do_resize: bool,
        size: SizeDict,
        size_divisor: Optional[int],
        interpolation: Optional["F.InterpolationMode"],
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        do_pad: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, List[float]]],
        image_std: Optional[Union[float, List[float]]],
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> BatchFeature:
        # Group videos by size for batched resizing
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}
        for shape, stacked_videos in grouped_videos.items():
            if do_convert_rgb:
                stacked_videos = self.convert_to_rgb(stacked_videos)
            if do_resize:
                stacked_videos = self.resize(
                    stacked_videos, size=size, size_divisor=size_divisor, interpolation=interpolation
                )
            resized_videos_grouped[shape] = stacked_videos
        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)

        # Group videos by size for further processing
        # Needed in case do_resize is False, or resize returns videos with different sizes
        grouped_videos, grouped_videos_index = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        for shape, stacked_videos in grouped_videos.items():
            if do_center_crop:
                stacked_videos = self.center_crop(stacked_videos, crop_size)
            # Fused rescale and normalize
            stacked_videos = self.rescale_and_normalize(
                stacked_videos, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_videos_grouped[shape] = stacked_videos

        processed_videos = reorder_videos(processed_videos_grouped, grouped_videos_index)
        processed_videos = torch.stack(processed_videos, dim=0) if return_tensors else processed_videos

        return BatchFeature(data={"pixel_values_videos": processed_videos}, tensor_type=return_tensors)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        **kwargs,
    ):
        r"""
        Instantiate a type of [`~video_processing_utils.VideoProcessorBase`] from an video processor.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained video hosted inside a model repo on
                  huggingface.co.
                - a path to a *directory* containing a video processor file saved using the
                  [`~video_processing_utils.VideoProcessorBase.save_pretrained`] method, e.g.,
                  `./my_model_directory/`.
                - a path or url to a saved video processor JSON *file*, e.g.,
                  `./my_model_directory/preprocessor_config.json`.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model video processor should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force to (re-)download the video processor files and override the cached versions if
                they exist.
            resume_download:
                Deprecated and ignored. All downloads are now resumed by default when possible.
                Will be removed in v5 of Transformers.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
                the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.


                <Tip>

                To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>"`.

                </Tip>

            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                If `False`, then this function returns just the final video processor object. If `True`, then this
                functions returns a `Tuple(video_processor, unused_kwargs)` where *unused_kwargs* is a dictionary
                consisting of the key/value pairs whose keys are not video processor attributes: i.e., the part of
                `kwargs` which has not been used to update `video_processor` and is otherwise ignored.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
                specify the folder name here.
            kwargs (`Dict[str, Any]`, *optional*):
                The values in kwargs of any keys which are video processor attributes will be used to override the
                loaded values. Behavior concerning key/value pairs whose keys are *not* video processor attributes is
                controlled by the `return_unused_kwargs` keyword parameter.

        Returns:
            A video processor of type [`~video_processing_utils.ImagVideoProcessorBase`].

        Examples:

        ```python
        # We can't instantiate directly the base class *VideoProcessorBase* so let's show the examples on a
        # derived class: *LlavaOnevisionVideoProcessor*
        video_processor = LlavaOnevisionVideoProcessor.from_pretrained(
            "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
        )  # Download video_processing_config from huggingface.co and cache.
        video_processor = LlavaOnevisionVideoProcessor.from_pretrained(
            "./test/saved_model/"
        )  # E.g. video processor (or model) was saved using *save_pretrained('./test/saved_model/')*
        video_processor = LlavaOnevisionVideoProcessor.from_pretrained("./test/saved_model/preprocessor_config.json")
        video_processor = LlavaOnevisionVideoProcessor.from_pretrained(
            "llava-hf/llava-onevision-qwen2-0.5b-ov-hf", do_normalize=False, foo=False
        )
        assert video_processor.do_normalize is False
        video_processor, unused_kwargs = LlavaOnevisionVideoProcessor.from_pretrained(
            "llava-hf/llava-onevision-qwen2-0.5b-ov-hf", do_normalize=False, foo=False, return_unused_kwargs=True
        )
        assert video_processor.do_normalize is False
        assert unused_kwargs == {"foo": False}
        ```"""
        kwargs["cache_dir"] = cache_dir
        kwargs["force_download"] = force_download
        kwargs["local_files_only"] = local_files_only
        kwargs["revision"] = revision

        use_auth_token = kwargs.pop("use_auth_token", None)
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        if token is not None:
            kwargs["token"] = token

        video_processor_dict, kwargs = cls.get_video_processor_dict(pretrained_model_name_or_path, **kwargs)

        return cls.from_dict(video_processor_dict, **kwargs)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        """
        Save an video processor object to the directory `save_directory`, so that it can be re-loaded using the
        [`~video_processing_utils.VideoProcessorBase.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the video processor JSON file will be saved (will be created if it does not exist).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        use_auth_token = kwargs.pop("use_auth_token", None)

        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if kwargs.get("token", None) is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            kwargs["token"] = use_auth_token

        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)

        # If we have a custom config, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=self)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_video_processor_file = os.path.join(save_directory, VIDEO_PROCESSOR_NAME)

        self.to_json_file(output_video_processor_file)
        logger.info(f"Video processor saved in {output_video_processor_file}")

        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("token"),
            )

        return [output_video_processor_file]

    @classmethod
    def get_video_processor_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
        video processor of type [`~video_processing_utils.VideoProcessorBase`] using `from_dict`.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
                specify the folder name here.

        Returns:
            `Tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the video processor object.
        """
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", None)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        use_auth_token = kwargs.pop("use_auth_token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", "")

        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)

        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

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
            is_local = True
        elif is_remote_url(pretrained_model_name_or_path):
            video_processor_file = pretrained_model_name_or_path
            resolved_video_processor_file = download_url(pretrained_model_name_or_path)
        else:
            try:
                # Try to load with a new config name first and if not successfull try with
                # the old file name. In case we can load with old name only, raise a deprecation warning
                # Deprecated until v5.0
                video_processor_file = VIDEO_PROCESSOR_NAME
                resolved_video_processor_file = cached_file(
                    pretrained_model_name_or_path,
                    video_processor_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder,
                )
            except EnvironmentError:
                video_processor_file = "preprocessor_config.json"
                resolved_video_processor_file = cached_file(
                    pretrained_model_name_or_path,
                    video_processor_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder,
                )
                logger.warning_once(
                    "You have video processor config saved in `preprocessor.json` file which is deprecated. "
                    "Video processor configs should be saved in their own `video_preprocessor.json` file. You can rename "
                    "the file or load and save the processor back which renames it automatically. "
                    "Loading from `preprocessor.json` will be removed in v5.0."
                )
            except EnvironmentError:
                # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted to
                # the original exception.
                raise
            except Exception:
                # For any other exception, we throw a generic error.
                raise EnvironmentError(
                    f"Can't load video processor for '{pretrained_model_name_or_path}'. If you were trying to load"
                    " it from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                    f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
                    f" directory containing a {VIDEO_PROCESSOR_NAME} file"
                )

        try:
            # Load video_processor dict
            with open(resolved_video_processor_file, "r", encoding="utf-8") as reader:
                text = reader.read()
            video_processor_dict = json.loads(text)

        except json.JSONDecodeError:
            raise EnvironmentError(
                f"It looks like the config file at '{resolved_video_processor_file}' is not a valid JSON file."
            )

        if is_local:
            logger.info(f"loading configuration file {resolved_video_processor_file}")
        else:
            logger.info(
                f"loading configuration file {video_processor_file} from cache at {resolved_video_processor_file}"
            )

        if not is_local:
            if "auto_map" in video_processor_dict:
                video_processor_dict["auto_map"] = add_model_info_to_auto_map(
                    video_processor_dict["auto_map"], pretrained_model_name_or_path
                )
            if "custom_pipelines" in video_processor_dict:
                video_processor_dict["custom_pipelines"] = add_model_info_to_custom_pipelines(
                    video_processor_dict["custom_pipelines"], pretrained_model_name_or_path
                )
        return video_processor_dict, kwargs

    @classmethod
    def from_dict(cls, video_processor_dict: Dict[str, Any], **kwargs):
        """
        Instantiates a type of [`~video_processing_utils.VideoProcessorBase`] from a Python dictionary of parameters.

        Args:
            video_processor_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the video processor object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the
                [`~video_processing_utils.VideoProcessorBase.to_dict`] method.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the video processor object.

        Returns:
            [`~video_processing_utils.VideoProcessorBase`]: The video processor object instantiated from those
            parameters.
        """
        video_processor_dict = video_processor_dict.copy()
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        # The `size` parameter is a dict and was previously an int or tuple in feature extractors.
        # We set `size` here directly to the `video_processor_dict` so that it is converted to the appropriate
        # dict within the video processor and isn't overwritten if `size` is passed in as a kwarg.
        if "size" in kwargs and "size" in video_processor_dict:
            video_processor_dict["size"] = kwargs.pop("size")
        if "crop_size" in kwargs and "crop_size" in video_processor_dict:
            video_processor_dict["crop_size"] = kwargs.pop("crop_size")

        video_processor = cls(**video_processor_dict)

        # Update video_processor with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(video_processor, key):
                setattr(video_processor, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        logger.info(f"Video processor {video_processor}")
        if return_unused_kwargs:
            return video_processor, kwargs
        else:
            return video_processor

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this video processor instance.
        """
        output = copy.deepcopy(self.__dict__)
        output["video_processor_type"] = self.__class__.__name__

        return output

    def to_json_string(self) -> str:
        """
        Serializes this instance to a JSON string.

        Returns:
            `str`: String containing all the attributes that make up this feature_extractor instance in JSON format.
        """
        dictionary = self.to_dict()

        for key, value in dictionary.items():
            if isinstance(value, np.ndarray):
                dictionary[key] = value.tolist()

        # make sure private name "_processor_class" is correctly
        # saved as "processor_class"
        _processor_class = dictionary.pop("_processor_class", None)
        if _processor_class is not None:
            dictionary["processor_class"] = _processor_class

        return json.dumps(dictionary, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this image_processor instance's parameters will be saved.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike]):
        """
        Instantiates a video processor of type [`~video_processing_utils.VideoProcessorBase`] from the path to a JSON
        file of parameters.

        Args:
            json_file (`str` or `os.PathLike`):
                Path to the JSON file containing the parameters.

        Returns:
            A video processor of type [`~video_processing_utils.VideoProcessorBase`]: The video_processor object
            instantiated from that JSON file.
        """
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        video_processor_dict = json.loads(text)
        return cls(**video_processor_dict)

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoVideoProcessor"):
        """
        Register this class with a given auto class. This should only be used for custom video processors as the ones
        in the library are already mapped with `AutoVideoProcessor `.

        <Tip warning={true}>

        This API is experimental and may have some slight breaking changes in the next releases.

        </Tip>

        Args:
            auto_class (`str` or `type`, *optional*, defaults to `"AutoVideoProcessor "`):
                The auto class to register this new video processor with.
        """
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        import transformers.models.auto as auto_module

        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        cls._auto_class = auto_class

    def fetch_videos(self, video_url_or_urls: Union[str, List[str]]):
        """
        Convert a single or a list of urls into the corresponding `np.array` objects.

        If a single url is passed, the return value will be a single object. If a list is passed a list of objects is
        returned.
        """
        if isinstance(video_url_or_urls, list):
            return [self.fetch_videos(x) for x in video_url_or_urls]
        elif isinstance(video_url_or_urls, str):
            return load_video(video_url_or_urls)
        else:
            raise TypeError(f"only a single or a list of entries is supported but got type={type(video_url_or_urls)}")


BaseVideoProcessor.push_to_hub = copy_func(BaseVideoProcessor.push_to_hub)
if BaseVideoProcessor.push_to_hub.__doc__ is not None:
    BaseVideoProcessor.push_to_hub.__doc__ = BaseVideoProcessor.push_to_hub.__doc__.format(
        object="video processor", object_class="AutoVideoProcessor", object_files="video processor file"
    )
