# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Fast Image processor class for TVP."""

from dataclasses import dataclass
from typing import Optional, Union, Unpack

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    group_images_by_shape,
    reorder_images,
    validate_kwargs,
)
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    PILImageResampling,
    SizeDict,
    ChannelDimension,
    ImageInput,
    is_valid_image,
    pil_torch_interpolation_mapping,
)
from ...utils import (
    TensorType,
    auto_docstring,
    is_torch_available,
    is_torchvision_available,
    is_torchvision_v2_available,
)


if is_torch_available():
    import torch

if is_torchvision_available():
    if is_torchvision_v2_available():
        from torchvision.transforms.v2 import functional as F
    else:
        from torchvision.transforms import functional as F


@dataclass
@auto_docstring(custom_intro="Fast image processor kwargs for TVP.")
class TvpFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    """Valid kwargs for TvpImageProcessorFast."""

    do_flip_channel_order: Optional[bool]
    pad_size: Optional[SizeDict] 
    constant_values: Optional[Union[float, list[float]]]
    pad_mode: Optional[str]


@auto_docstring
class TvpImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"longest_edge": 448}
    default_to_square = False
    crop_size = {"height": 448, "width": 448}
    do_resize = True
    do_center_crop = True
    do_rescale = True
    rescale_factor = 1 / 255
    do_pad = True
    pad_size = {"height": 448, "width": 448}
    constant_values = 0
    pad_mode = "constant"
    do_normalize = True
    do_flip_channel_order = True
    valid_kwargs = TvpFastImageProcessorKwargs
    model_input_names = ["pixel_values"]

    def __init__(self, **kwargs: Unpack[TvpFastImageProcessorKwargs]):
        super().__init__(**kwargs)

    def _process_image(
        self,
        image: ImageInput,
        do_convert_rgb: Optional[bool] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        device: Optional["torch.device"] = None,
    ) -> "torch.Tensor":
        """
        Process a single image to torch tensor.
        
        This fast implementation only accepts torch tensors as input to avoid PIL conversion.
        """
        if not isinstance(image, torch.Tensor):
            raise ValueError(
                f"TvpImageProcessorFast only accepts torch.Tensor inputs. Got {type(image)}. "
                "Please convert your images to torch tensors before passing to the fast processor."
            )
        
        # Ensure the tensor is in the correct format
        if image.dim() == 2:
            # Single channel image, add channel dimension
            image = image.unsqueeze(0)
        elif image.dim() == 3:
            # Check if channels are in the right place
            if input_data_format == ChannelDimension.LAST or (
                input_data_format is None and image.shape[-1] in [1, 3]
            ):
                # Channels last, convert to channels first
                image = image.permute(2, 0, 1).contiguous()
        elif image.dim() != 3:
            raise ValueError(f"Expected 2D or 3D tensor, got {image.dim()}D tensor")
        
        # Ensure float32 for processing
        if image.dtype != torch.float32:
            image = image.float()
        
        # Move to device if specified
        if device is not None:
            image = image.to(device)
        
        return image

    def _prepare_input_videos(
        self,
        videos: Union[ImageInput, list[ImageInput], list[list[ImageInput]]],
        do_convert_rgb: Optional[bool],
        input_data_format: Optional[Union[str, ChannelDimension]],
        device: Optional[Union[str, torch.device]],
    ) -> list[list["torch.Tensor"]]:
        """Prepare input videos for processing."""
        # Convert to batched format
        if isinstance(videos, (list, tuple)) and isinstance(videos[0], (list, tuple)) and is_valid_image(videos[0][0]):
            batched_videos = videos
        elif isinstance(videos, (list, tuple)) and is_valid_image(videos[0]):
            batched_videos = [videos]
        elif is_valid_image(videos):
            batched_videos = [[videos]]
        else:
            raise ValueError(f"Could not make batched video from {videos}")

        # Convert to tensors
        processed_videos = []
        for video in batched_videos:
            video_tensors = []
            for frame in video:
                frame_tensor = self._process_image(
                    frame, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format, device=device
                )
                video_tensors.append(frame_tensor)
            processed_videos.append(video_tensors)

        return processed_videos

    def _preprocess(
        self,
        videos: list[list["torch.Tensor"]],
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"],
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_pad: bool,
        pad_size: SizeDict,
        constant_values: Union[float, list[float]],
        pad_mode: str,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        do_flip_channel_order: bool,
        return_tensors: Optional[Union[str, TensorType]],
        disable_grouping: Optional[bool],
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess videos using the fast image processor.
        
        This method processes each video frame through the same pipeline as the original
        TVP image processor but uses torchvision operations for better performance.
        """
        processed_videos = []
        
        for video in videos:
            # Group frames by shape for efficient processing
            grouped_frames, grouped_frames_index = group_images_by_shape(video, disable_grouping=disable_grouping)
            processed_frames_grouped = {}
            
            for shape, stacked_frames in grouped_frames.items():
                # Resize if needed
                if do_resize:
                    stacked_frames = self.resize(stacked_frames, size, interpolation)
                
                # Center crop if needed
                if do_center_crop:
                    stacked_frames = self.center_crop(stacked_frames, crop_size)
                
                # Rescale and normalize separately (following original TVP behavior)
                if do_rescale:
                    stacked_frames = self.rescale(stacked_frames, rescale_factor)
                
                if do_normalize:
                    stacked_frames = self.normalize(stacked_frames, image_mean, image_std)
                
                # Pad if needed
                if do_pad:
                    stacked_frames = self._pad_frames(
                        stacked_frames, pad_size, constant_values, pad_mode
                    )
                
                # Flip channel order if needed (RGB to BGR)
                if do_flip_channel_order:
                    stacked_frames = self._flip_channel_order(stacked_frames)
                
                processed_frames_grouped[shape] = stacked_frames
            
            # Reorder frames back to original order
            processed_frames = reorder_images(processed_frames_grouped, grouped_frames_index)
            processed_videos.append(processed_frames)
        
        # Convert to tensor format if requested
        if return_tensors == "pt" and processed_videos:
            # Stack frames for each video
            stacked_videos = []
            for video in processed_videos:
                if video:
                    stacked_video = torch.stack(video)
                    stacked_videos.append(stacked_video)
                else:
                    # Handle empty video case
                    stacked_videos.append(torch.empty(0, 3, pad_size["height"], pad_size["width"]))
            
            # Stack videos if we have multiple videos
            if len(stacked_videos) > 1:
                pixel_values = torch.stack(stacked_videos)
            else:
                # Add batch dimension for single video
                pixel_values = stacked_videos[0].unsqueeze(0)
        else:
            pixel_values = processed_videos
        
        return BatchFeature(data={"pixel_values": pixel_values}, tensor_type=return_tensors)

    def _pad_frames(
        self,
        frames: "torch.Tensor",
        pad_size: SizeDict,
        constant_values: Union[float, list[float]],
        pad_mode: str,
    ) -> "torch.Tensor":
        """Pad frames to the specified size."""
        if frames.shape[-2:] == (pad_size["height"], pad_size["width"]):
            return frames
        
        # Calculate padding
        current_height, current_width = frames.shape[-2:]
        pad_bottom = pad_size["height"] - current_height
        pad_right = pad_size["width"] - current_width
        
        if pad_bottom < 0 or pad_right < 0:
            raise ValueError("The padding size must be greater than frame size")
        
        # Apply padding
        padding = [0, 0, pad_right, pad_bottom]  # [left, top, right, bottom]
        return F.pad(frames, padding, fill=constant_values, padding_mode=pad_mode)

    def resize(
        self,
        image: "torch.Tensor",
        size: SizeDict,
        interpolation: "F.InterpolationMode" = None,
        antialias: bool = True,
        **kwargs,
    ) -> "torch.Tensor":
        """
        Resize an image to the specified size.

        Args:
            image (`torch.Tensor`):
                Image to resize.
            size (`SizeDict`):
                Size dictionary. If `size` has `longest_edge`, resize the longest edge to that value
                while maintaining aspect ratio. Otherwise, use the base class resize method.
            interpolation (`F.InterpolationMode`, *optional*):
                Interpolation method to use.
            antialias (`bool`, *optional*, defaults to `True`):
                Whether to use antialiasing.

        Returns:
            `torch.Tensor`: The resized image.
        """
        interpolation = interpolation if interpolation is not None else F.InterpolationMode.BILINEAR
        
        # Handle longest_edge case (TVP-specific)
        if size["longest_edge"]:
            # Get current dimensions
            current_height, current_width = image.shape[-2:]
            
            # Calculate new dimensions maintaining aspect ratio
            if current_height >= current_width:
                ratio = current_width * 1.0 / current_height
                new_height = size["longest_edge"]
                new_width = int(new_height * ratio)
            else:
                ratio = current_height * 1.0 / current_width
                new_width = size["longest_edge"]
                new_height = int(new_width * ratio)
            
            new_size = (new_height, new_width)
            return F.resize(image, new_size, interpolation=interpolation, antialias=antialias)
        
        # Use base class resize method for other cases
        return super().resize(image, size, interpolation, antialias, **kwargs)

    def _flip_channel_order(self, frames: "torch.Tensor") -> "torch.Tensor":
        """Flip channel order from RGB to BGR."""
        # Assuming frames are in channels_first format (C, H, W)
        if frames.shape[-3] == 3:  # 3 channels in the last dimension
            return frames.flip(dims=[-3])  # Flip along channel dimension
        return frames

    @auto_docstring
    def preprocess(
        self,
        videos: Union[ImageInput, list[ImageInput], list[list[ImageInput]]],
        **kwargs: Unpack[TvpFastImageProcessorKwargs],
    ) -> BatchFeature:
        r"""
        videos (`ImageInput` or `list[ImageInput]` or `list[list[ImageInput]]`):
            The video frames to preprocess. Must be torch tensors for the fast processor.
        """
        validate_kwargs(captured_kwargs=kwargs.keys(), valid_processor_keys=self.valid_kwargs.__annotations__.keys())
        # Set default kwargs from self. This ensures that if a kwarg is not provided
        # by the user, it gets its default value from the instance, or is set to None.
        for kwarg_name in self.valid_kwargs.__annotations__:
            kwargs.setdefault(kwarg_name, getattr(self, kwarg_name, None))

        # Extract parameters that are only used for preparing the input images
        do_convert_rgb = kwargs.pop("do_convert_rgb")
        input_data_format = kwargs.pop("input_data_format")
        device = kwargs.pop("device")

        # Prepare input videos
        videos = self._prepare_input_videos(
            videos=videos, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format, device=device
        )

        # Update kwargs that need further processing before being validated
        kwargs = self._further_process_kwargs(**kwargs)

        # Validate kwargs
        self._validate_preprocess_kwargs(**kwargs)

        # torch resize uses interpolation instead of resample
        resample = kwargs.pop("resample")
        kwargs["interpolation"] = (
            pil_torch_interpolation_mapping[resample] if isinstance(resample, (PILImageResampling, int)) else resample
        )

        # Pop kwargs that are not needed in _preprocess
        kwargs.pop("default_to_square")
        kwargs.pop("data_format")

        # Extract required parameters for _preprocess
        do_resize = kwargs.pop("do_resize")
        size = kwargs.pop("size")
        interpolation = kwargs.pop("interpolation")
        do_center_crop = kwargs.pop("do_center_crop")
        crop_size = kwargs.pop("crop_size")
        do_rescale = kwargs.pop("do_rescale")
        rescale_factor = kwargs.pop("rescale_factor")
        do_pad = kwargs.pop("do_pad", self.do_pad)
        pad_size = kwargs.pop("pad_size", self.pad_size)
        constant_values = kwargs.pop("constant_values", self.constant_values)
        pad_mode = kwargs.pop("pad_mode", self.pad_mode)
        do_normalize = kwargs.pop("do_normalize")
        image_mean = kwargs.pop("image_mean")
        image_std = kwargs.pop("image_std")
        do_flip_channel_order = kwargs.pop("do_flip_channel_order", self.do_flip_channel_order)
        return_tensors = kwargs.pop("return_tensors")
        disable_grouping = kwargs.pop("disable_grouping")

        return self._preprocess(
            videos,
            do_resize=do_resize,
            size=size,
            interpolation=interpolation,
            do_center_crop=do_center_crop,
            crop_size=crop_size,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_pad=do_pad,
            pad_size=pad_size,
            constant_values=constant_values,
            pad_mode=pad_mode,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_flip_channel_order=do_flip_channel_order,
            return_tensors=return_tensors,
            disable_grouping=disable_grouping,
            **kwargs
        )


__all__ = ["TvpImageProcessorFast", "TvpFastImageProcessorKwargs"] 