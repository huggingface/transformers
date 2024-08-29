# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
"""Image processor class for ProPainter."""

from typing import Dict, List, Optional, Union

import numpy as np
import torch

from transformers.utils import is_vision_available
from transformers.utils.generic import TensorType

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
    get_resize_output_image_size,
    rescale,
    resize,
    to_channel_dimension_format,
)
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    is_valid_image,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from ...utils import filter_out_non_signature_kwargs, logging, requires_backends, is_scipy_available

if is_scipy_available():
    from scipy.ndimage import binary_dilation

if is_vision_available():
    import PIL

logger = logging.get_logger(__name__)


def make_batched(videos) -> List[List[ImageInput]]:
    if isinstance(videos, (list, tuple)) and isinstance(videos[0], (list, tuple)) and is_valid_image(videos[0][0]):
        return videos

    elif isinstance(videos, (list, tuple)) and is_valid_image(videos[0]):
        return [videos]

    elif is_valid_image(videos):
        return [[videos]]

    raise ValueError(f"Could not make batched video from {videos}")

# Copied from transformers.models.superpoint.image_processing_superpoint.convert_to_grayscale
def convert_to_grayscale_and_dilation(
    image: ImageInput,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
    mask_dilation: int  = 4,
) -> ImageInput:
    """
    Converts an image to grayscale format using the NTSC formula and performs binary dilation on an image. Only support numpy and PIL Image. TODO support torch
    and tensorflow grayscale conversion

    This function is supposed to return a 1-channel image, but it returns a 3-channel image with the same value in each
    channel, because of an issue that is discussed in :
    https://github.com/huggingface/transformers/pull/25786#issuecomment-1730176446

    Args:
        image (Image):
            The image to convert.
        input_data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format for the input image.
    """
    requires_backends(convert_to_grayscale_and_dilation, ["vision"])

    if isinstance(image, np.ndarray):
        if input_data_format == ChannelDimension.FIRST:
            gray_image = image[0, ...] * 0.2989 + image[1, ...] * 0.5870 + image[2, ...] * 0.1140
            gray_image = np.stack([gray_image] * 1, axis=0)
            gray_dilated_image = binary_dilation(gray_image, iterations=mask_dilation).astype(np.uint8)
        elif input_data_format == ChannelDimension.LAST:
            gray_image = image[..., 0] * 0.2989 + image[..., 1] * 0.5870 + image[..., 2] * 0.1140
            gray_image = np.stack([gray_image] * 1, axis=-1)
            gray_dilated_image = binary_dilation(gray_image, iterations=mask_dilation).astype(np.uint8)
        return gray_dilated_image

    if not isinstance(image, PIL.Image.Image):
        return image

    image = np.array(image.convert("L"))
    image = binary_dilation(image, iterations=mask_dilation).astype(np.uint8)

    return image

def extrapolation(
    image: ImageInput,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
    scale: Optional[tuple[float, float]] = None,
    num_frames: int = None,
):
    """Prepares the data for video outpainting.
    """
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image[0])

    height, width = get_image_size(image)

    # Defines new FOV.
    height_extr = int(scale[0] * height)
    width_extr = int(scale[1] * width)
    height_extr = height_extr - height_extr % 8
    width_extr = width_extr - width_extr % 8
    height_start = int((height_extr - height) / 2)
    width_start = int((width_extr - width) / 2)

    # Extrapolates the FOV for video.
    images = []
    if input_data_format == ChannelDimension.LAST:
        for v in image:
            frame = np.zeros(((height_extr, width_extr, 3)), dtype=np.uint8)
            frame[height_start: height_start + height, width_start: width_start + width, :] = v
            images.append(frame)
    elif input_data_format == ChannelDimension.FIRST:
        for v in image:
            frame = np.zeros((3, height_extr, width_extr), dtype=np.uint8)  # Adjusted shape
            frame[:, height_start: height_start + height, width_start: width_start + width] = v
            
            # Transpose to (height_extr, width_extr, num_channels)
            images.append(np.transpose(frame, (1, 2, 0)))  # Adjusted transpose

    # Generates the mask for missing region.
    masks_dilated = []
    flow_masks = []
    
    dilate_h = 4 if height_start > 10 else 0
    dilate_w = 4 if width_start > 10 else 0
    mask = np.ones(((height_extr, width_extr)), dtype=np.uint8)
    
    mask[height_start+dilate_h: height_start+height-dilate_h, 
        width_start+dilate_w: width_start+width-dilate_w] = 0
    flow_masks.append((mask * 255))

    mask[height_start: height_start+height, width_start: width_start+width] = 0
    masks_dilated.append((mask * 255))

    flow_masks = flow_masks * num_frames
    masks_dilated = masks_dilated * num_frames
    
    return images, flow_masks, masks_dilated, (height_extr, width_extr)

#Adapted from transformers.image_transforms.NumpyToTensor
class NumpyToTensor:
    """
    Convert a numpy array to a PyTorch tensor.
    Converts a numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    """

    def __call__(self, image: np.ndarray):
        # Same as in PyTorch, we assume incoming numpy images are in HWC format
        
        image =  np.stack(image, axis=2)
        image = torch.from_numpy(image).permute(2, 0, 1, 3).contiguous()
        image = image.float().div(255)
        return image

class ProPainterImageProcessor(BaseImageProcessor):
    r"""
    Constructs a ProPainter image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by the
            `do_resize` parameter in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 256}`):
            Size of the output image after resizing. The shortest edge of the image will be resized to
            `size["shortest_edge"]` while maintaining the aspect ratio of the original image. Can be overriden by
            `size` in the `preprocess` method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the
            `preprocess` method.
        do_center_crop (`bool`, *optional*, defaults to `False`):
            Whether to center crop the image to the specified `crop_size`. Can be overridden by the `do_center_crop`
            parameter in the `preprocess` method.
        crop_size (`Dict[str, int]`, *optional*, defaults to `{"height": 224, "width": 224}`):
            Size of the image after applying the center crop. Can be overridden by the `crop_size` parameter in the
            `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `False`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
            parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/127.5`):
            Defines the scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter
            in the `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `False`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = False,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_center_crop: bool = False,
        crop_size: Dict[str, int] = None,
        do_rescale: bool = False,
        rescale_factor: Union[int, float] = 255,
        do_normalize: bool = False,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        scale_hw: Optional[tuple[float, float]] = None,
        mask_dilation: int  = 4,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"shortest_edge": 256}
        size = get_size_dict(size, default_to_square=False)
        crop_size = crop_size if crop_size is not None else {"height": 224, "width": 224}
        crop_size = get_size_dict(crop_size, param_name="crop_size")

        self.do_resize = do_resize
        self.size = size
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD
        self.scale_hw = scale_hw
        self.mask_dilation = mask_dilation

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the output image. If `size` is of the form `{"height": h, "width": w}`, the output image will
                have the size `(h, w)`. If `size` is of the form `{"shortest_edge": s}`, the output image will have its
                shortest edge of length `s` while keeping the aspect ratio of the original image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                Resampling filter to use when resiizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        size = get_size_dict(size, default_to_square=False)
        if "shortest_edge" in size:
            output_size = get_resize_output_image_size(
                image, size["shortest_edge"], default_to_square=False, input_data_format=input_data_format
            )
        elif "height" in size and "width" in size:
            output_size = (size["height"], size["width"])
        else:
            raise ValueError(f"Size must have 'height' and 'width' or 'shortest_edge' as keys. Got {size.keys()}")
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def _preprocess_image(
        self,
        image: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_center_crop: bool = None,
        crop_size: Dict[str, int] = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        is_mask_frame: bool = None,
        mask_dilation: int  = None,
    ) -> np.ndarray:
        """Preprocesses a single image."""

        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_center_crop=do_center_crop,
            crop_size=crop_size,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )
        

        # All transformations expect numpy arrays.
        image = to_numpy_array(image)
        # print("checkkkkkk", image.shape)
        if is_scaled_image(image) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)

        if do_resize:
            image = self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format)

        if is_mask_frame:
            image = convert_to_grayscale_and_dilation(image, input_data_format=input_data_format, mask_dilation=mask_dilation)

        if do_center_crop:
            image = self.center_crop(image, size=crop_size, input_data_format=input_data_format)

        if do_rescale:
            image = self.rescale(image=image, scale=rescale_factor, dtype =np.uint8, input_data_format=input_data_format)

        if do_normalize:
            image = self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)

        image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
        return image

    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_center_crop: bool = None,
        crop_size: Dict[str, int] = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        scale_hw: Optional[tuple[float, float]] = None,
        mask_dilation: int = None,
    ) -> ImageInput:
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Video frames to preprocess. Expects a single or batch of video frames with pixel values ranging from 0
                to 255. If passing in frames with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after applying resize.
            resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`, Only
                has an effect if `do_resize` is set to `True`.
            do_center_crop (`bool`, *optional*, defaults to `self.do_centre_crop`):
                Whether to centre crop the image.
            crop_size (`Dict[str, int]`, *optional*, defaults to `self.crop_size`):
                Size of the image after applying the centre crop.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image values between [0 - 1].
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                    - Unset: Return a list of `np.ndarray`.
                    - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                    - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                    - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                    - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                    - `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                    - `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                    - Unset: Use the inferred channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        resample = resample if resample is not None else self.resample
        do_center_crop = do_center_crop if do_center_crop is not None else self.do_center_crop
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        scale_hw = scale_hw if scale_hw is not None else self.scale_hw
        mask_dilation = mask_dilation if mask_dilation is not None else self.mask_dilation

        size = size if size is not None else self.size
        size = get_size_dict(size, default_to_square=False)
        crop_size = crop_size if crop_size is not None else self.crop_size
        crop_size = get_size_dict(crop_size, param_name="crop_size")

        #This is to separate the video frames and the masks from one single variable
        masks = images[len(images)//2:]
        videos = images[:len(images)//2]

        to_tensors = NumpyToTensor()

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        videos = make_batched(videos)
        masks = make_batched(masks)

        video_size = get_image_size(videos[0][0])
        video_size = (video_size[0]-video_size[0]%8, video_size[1]-video_size[1]%8)

        videos = [
            [
                self._preprocess_image(
                    image=img,
                    do_resize=do_resize,
                    size=size,
                    resample=resample,
                    do_center_crop=do_center_crop,
                    crop_size=crop_size,
                    do_rescale=do_rescale,
                    rescale_factor=rescale_factor,
                    do_normalize=do_normalize,
                    image_mean=image_mean,
                    image_std=image_std,
                    data_format=data_format,
                    input_data_format=input_data_format,
                    is_mask_frame = False,
                )
                for img in video
            ]
            for video in videos
        ]

        pixel_values_masks = []
        if scale_hw is None:
            for mask in masks:
                pixel_values_mask = [
                        self._preprocess_image(
                            image=img,
                            do_resize=True,
                            size={"height": video_size[0], "width": video_size[1]},
                            resample=PILImageResampling.NEAREST,
                            do_center_crop=do_center_crop,
                            crop_size=crop_size,
                            do_rescale=True,
                            rescale_factor=255,
                            do_normalize=do_normalize,
                            image_mean=image_mean,
                            image_std=image_std,
                            data_format=data_format,
                            input_data_format=input_data_format,
                            is_mask_frame = True,
                            mask_dilation= mask_dilation,
                        )
                        for img in mask
                    ]
                if len(mask) == 1:
                  pixel_values_mask = pixel_values_mask * len(videos[0])
                pixel_values_masks.append(pixel_values_mask)
        else:
            #for outpainting of videos
            videos, flow_masks, masks_dilated, (height_extr, width_extr)= [extrapolation(
                        video, scale = scale_hw, num_frames=len(video))
                        for video in videos]

        if scale_hw is None:
            #masks is for both flow_masks, masks_dilated, just add the same data to both variables in case of inpainting
            flow_masks = masks_dilated = [to_tensors(mask).unsqueeze(0) for mask in pixel_values_masks]
            size = video_size #height, width
        else:
            flow_masks, = [to_tensors(mask).unsqueeze(0) for mask in flow_masks]
            masks_dilated = [to_tensors(mask).unsqueeze(0) for mask in masks_dilated]
            size = (height_extr, width_extr)
        videos_inp = [[np.array(frame).transpose(1,2,0).astype(np.uint8) for frame in video] for video in videos]
        videos = [to_tensors(video).unsqueeze(0) * 2 - 1 for video in videos]

        data = {"pixel_values_inp": videos_inp, "pixel_values": videos, "flow_masks": flow_masks, "masks_dilated": masks_dilated, "size": size}
        return BatchFeature(data=data, tensor_type=return_tensors)
