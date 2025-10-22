# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
"""Fast image processor class for GLPN."""

from typing import TYPE_CHECKING, Any, Optional, Union

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import BaseImageProcessorFast
from ...image_utils import ChannelDimension, ImageInput, PILImageResampling, infer_channel_dimension_format
from ...utils import TensorType, auto_docstring, is_torch_available, is_torchvision_available, logging
from ...utils.import_utils import requires


if TYPE_CHECKING:
    from ...modeling_outputs import DepthEstimatorOutput

if is_torch_available():
    import torch

if is_torchvision_available():
    from torchvision.transforms import functional as F


logger = logging.get_logger(__name__)


@auto_docstring
@requires(backends=("torchvision", "torch"))
class GLPNImageProcessorFast(BaseImageProcessorFast):
    r"""
    Constructs a fast GLPN image processor using PyTorch and TorchVision.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions, rounding them down to the closest multiple of
            `size_divisor`. Can be overridden by `do_resize` in `preprocess`.
        size_divisor (`int`, *optional*, defaults to 32):
            When `do_resize` is `True`, images are resized so their height and width are rounded down to the closest
            multiple of `size_divisor`. Can be overridden by `size_divisor` in `preprocess`.
        resample (`PIL.Image` resampling filter, *optional*, defaults to `PILImageResampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by `resample` in `preprocess`.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether or not to apply the scaling factor (to make pixel values floats between 0. and 1.). Can be
            overridden by `do_rescale` in `preprocess`.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size_divisor: int = 32,
        resample=PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        **kwargs,
    ) -> None:
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.size_divisor = size_divisor
        self.resample = resample
        self.rescale_factor = 1 / 255
        super().__init__(**kwargs)

    def resize(
        self,
        image: "torch.Tensor",
        size_divisor: int,
        interpolation: "F.InterpolationMode" = None,
        **kwargs,
    ) -> "torch.Tensor":
        """
        Resize the image, rounding the (height, width) dimensions down to the closest multiple of size_divisor.

        If the image is of dimension (3, 260, 170) and size_divisor is 32, the image will be resized to (3, 256, 160).

        Args:
            image (`torch.Tensor`):
                The image to resize.
            size_divisor (`int`):
                The image is resized so its height and width are rounded down to the closest multiple of
                `size_divisor`.
            interpolation (`F.InterpolationMode`, *optional*):
                Resampling filter to use when resizing the image e.g. `F.InterpolationMode.BILINEAR`.

        Returns:
            `torch.Tensor`: The resized image.
        """
        interpolation = interpolation if interpolation is not None else F.InterpolationMode.BILINEAR

        # Get current height and width from tensor shape (C, H, W)
        height, width = image.shape[-2:]

        # Round down to the closest multiple of size_divisor using torch operations
        new_h = (height // size_divisor) * size_divisor
        new_w = (width // size_divisor) * size_divisor

        if new_h != height or new_w != width:
            image = F.resize(image, (new_h, new_w), interpolation=interpolation, **kwargs)

        return image

    def _process_image(
        self,
        image: "torch.Tensor",
        do_convert_rgb: bool = True,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        device: Optional["torch.device"] = None,
    ) -> "torch.Tensor":
        """
        Process a single image tensor, supporting variable channel dimensions including 4-channel images.

        Overrides the base class method to support 1, 3, and 4 channel images.
        """
        if is_torch_available():
            import torch
            from torchvision.transforms.functional import pil_to_tensor

        # Convert PIL image to tensor if needed
        if isinstance(image, torch.Tensor):
            # Already a tensor, just ensure it's float
            if image.dtype != torch.float32:
                image = image.float()
        elif hasattr(image, "mode") and hasattr(image, "size"):  # PIL Image
            image = pil_to_tensor(image).float()
        else:
            # Assume it's numpy array
            image = torch.from_numpy(image)
            if image.dtype != torch.float32:
                image = image.float()

        # Infer the channel dimension format if not provided, supporting 1, 3, and 4 channels
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image, num_channels=(1, 3, 4))

        if input_data_format == ChannelDimension.LAST:
            # We force the channel dimension to be first for torch tensors as this is what torchvision expects.
            image = image.permute(2, 0, 1).contiguous()

        # Now that we have torch tensors, we can move them to the right device
        if device is not None:
            image = image.to(device)

        return image

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Override the default behavior to exclude
        attributes that are not relevant for GLPN processing to maintain compatibility with the slow processor.
        """
        output = super().to_dict()

        # Remove attributes that are only relevant for other processors to maintain
        # compatibility with the slow processor
        attrs_to_remove = [
            "crop_size",
            "do_center_crop",
            "do_normalize",
            "image_mean",
            "image_std",
            "do_convert_rgb",
            "size",
            "input_data_format",
            "device",
            "return_tensors",
            "disable_grouping",
            "rescale_factor",
        ]

        for attr in attrs_to_remove:
            output.pop(attr, None)

        return output

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size_divisor: Optional[int] = None,
        resample=None,
        do_rescale: Optional[bool] = None,
        return_tensors: Optional[Union[TensorType, str]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> BatchFeature:
        r"""
        Preprocess the given images.

        Args:
            images (`ImageInput`):
                Images to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the input such that the (height, width) dimensions are a multiple of `size_divisor`.
            size_divisor (`int`, *optional*, defaults to `self.size_divisor`):
                When `do_resize` is `True`, images are resized so their height and width are rounded down to the
                closest multiple of `size_divisor`.
            resample (`PIL.Image` resampling filter, *optional*, defaults to `self.resample`):
                `PIL.Image` resampling filter to use if resizing the image e.g. `PILImageResampling.BILINEAR`. Only has
                an effect if `do_resize` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether or not to apply the scaling factor (to make pixel values floats between 0. and 1.).
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                    - `None`: Return a list of `torch.Tensor`.
                    - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                    - `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                    - `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        size_divisor = size_divisor if size_divisor is not None else self.size_divisor
        resample = resample if resample is not None else self.resample

        # Convert PIL resampling to torchvision InterpolationMode
        if is_torchvision_available():
            from ...image_utils import pil_torch_interpolation_mapping

            interpolation = (
                pil_torch_interpolation_mapping[resample]
                if isinstance(resample, (PILImageResampling, int))
                else resample
            )
        else:
            interpolation = F.InterpolationMode.BILINEAR

        # Prepare images
        images = self._prepare_image_like_inputs(
            images=images,
            do_convert_rgb=False,  # Don't force RGB conversion to support variable channels
            input_data_format=input_data_format,
        )

        return self._preprocess(
            images=images,
            do_resize=do_resize,
            size_divisor=size_divisor,
            interpolation=interpolation,
            do_rescale=do_rescale,
            rescale_factor=self.rescale_factor,
            return_tensors=return_tensors,
            **kwargs,
        )

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size_divisor: Optional[int],
        interpolation: Optional["F.InterpolationMode"],
        do_rescale: bool,
        rescale_factor: float,
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess the images for GLPN.

        Args:
            images (`list[torch.Tensor]`):
                List of images to preprocess.
            do_resize (`bool`):
                Whether to resize the images.
            size_divisor (`int`, *optional*):
                Size divisor for resizing. If None, uses self.size_divisor.
            interpolation (`F.InterpolationMode`, *optional*):
                Interpolation mode for resizing.
            do_rescale (`bool`):
                Whether to rescale pixel values to [0, 1].
            rescale_factor (`float`):
                Factor to rescale pixel values by.
            return_tensors (`str` or `TensorType`, *optional*):
                Type of tensors to return.

        Returns:
            `BatchFeature`: Processed images in a BatchFeature.
        """
        if size_divisor is None:
            size_divisor = self.size_divisor

        processed_images = []

        for image in images:
            # Resize if needed
            if do_resize:
                image = self.resize(image, size_divisor=size_divisor, interpolation=interpolation)

            # Rescale to [0, 1] if needed
            if do_rescale:
                image = self.rescale(image, scale=rescale_factor)

            processed_images.append(image)

        # Stack images into a batch if return_tensors is specified
        if return_tensors:
            processed_images = torch.stack(processed_images, dim=0)

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)

    def post_process_depth_estimation(
        self,
        outputs: "DepthEstimatorOutput",
        target_sizes: Optional[Union[TensorType, list[tuple[int, int]], None]] = None,
    ) -> list[dict[str, TensorType]]:
        """
        Converts the raw output of [`DepthEstimatorOutput`] into final depth predictions and depth PIL images.
        Only supports PyTorch.

        Args:
            outputs ([`DepthEstimatorOutput`]):
                Raw outputs of the model.
            target_sizes (`TensorType` or `list[tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`tuple[int, int]`) containing the target size
                (height, width) of each image in the batch. If left to None, predictions will not be resized.

        Returns:
            `list[dict[str, TensorType]]`: A list of dictionaries of tensors representing the processed depth
            predictions.
        """
        predicted_depth = outputs.predicted_depth

        if (target_sizes is not None) and (len(predicted_depth) != len(target_sizes)):
            raise ValueError(
                "Make sure that you pass in as many target sizes as the batch dimension of the predicted depth"
            )

        results = []
        target_sizes = [None] * len(predicted_depth) if target_sizes is None else target_sizes
        for depth, target_size in zip(predicted_depth, target_sizes):
            if target_size is not None:
                depth = depth[None, None, ...]
                depth = torch.nn.functional.interpolate(depth, size=target_size, mode="bicubic", align_corners=False)
                depth = depth.squeeze()

            results.append({"predicted_depth": depth})

        return results


__all__ = ["GLPNImageProcessorFast"]
