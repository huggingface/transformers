from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.v2.functional import InterpolationMode

from ...configuration_utils import PreTrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...image_processing_utils import BaseImageProcessor
from ...image_processing_utils_fast import BaseImageProcessorFast
from ...image_transforms import flip_channel_order, to_channel_dimension_format
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    make_flat_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from ...modeling_outputs import BaseModelOutputWithNoAttention
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, filter_out_non_signature_kwargs
from ...utils.generic import TensorType


class UVDocConfig(PreTrainedConfig):
    model_type = "uvdoc"

    """
    This is the configuration class to store the configuration of a [`UVDocModel`]. It is used to instantiate a
    UVDoc model according to the specified arguments, defining the model architecture for document rectification
    (correcting perspective distortion, tilt, and geometric deformation of document images).
    Instantiating a configuration with the defaults will yield a similar configuration to that of the UVDoc
    [PaddlePaddle/UVDoc](https://huggingface.co/PaddlePaddle/UVDoc) baseline architecture for document rectification tasks.
    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.
    Args:
        num_filter (`int`, *optional*, defaults to 32):
            The number of convolutional filters (output channels) in the initial convolutional layers of the model,
            controlling the depth of feature maps extracted from input document images. Larger values increase
            model capacity but also computational cost.
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels of the model. Defaults to 3 for RGB document images; set to 1 for grayscale
            document images.
        kernel_size (`int`, *optional*, defaults to 5):
            The size of convolutional kernels used in the backbone network, typically an odd integer to ensure
            symmetric padding and preserve spatial dimensions of feature maps.
        stride (`List[int]`, *optional*, defaults to `[1, 2, 2, 2]`):
            The list of stride values for convolutional layers in the model, controlling the downsampling rate of
            feature maps at each stage. Stride=1 retains spatial resolution, while stride=2 halves the resolution
            to capture larger receptive fields.
        map_num (`List[int]`, *optional*, defaults to `[1, 2, 4, 8, 16]`):
            The scaling factors for feature map dimensions in multi-scale feature fusion modules, used to align
            feature maps of different resolutions for document structure restoration.
        block_nums (`List[int]`, *optional*, defaults to `[3, 4, 6, 3]`):
            The number of residual blocks in each stage of the model backbone, determining the depth of the network.
            More blocks enhance feature extraction capability but increase inference time.
        dilation_values (`Dict[str, Union[int, List[int]]]`, *optional*, defaults to `None`):
            A dictionary of dilation rates for dilated convolutional layers in bridge modules (e.g., "bridge_1": 1,
            "bridge_4": [8, 3, 2]). Dilated convolution expands the receptive field without increasing kernel size,
            critical for capturing long-range geometric dependencies in distorted documents. If `None`, default values
            will be used:{
                "bridge_1": 1,
                "bridge_2": 2,
                "bridge_3": 5,
                "bridge_4": [8, 3, 2],
                "bridge_5": [12, 7, 4],
                "bridge_6": [18, 12, 6]
            }
        padding_mode (`str`, *optional*, defaults to `"reflect"`):
        The padding mode for convolutional layers, used to handle boundary pixels of document images. Supported
        modes include `"reflect"` (recommended for document rectification to avoid edge artifacts), `"constant"`,
        and `"replicate"`.
        upsample_size (`List[int]`, *optional*, defaults to `[712, 488]`):
        The target spatial size (width, height) of the upsampled output image, matching the desired resolution
        of the rectified document. Adjust based on your input document size and task requirements.
        upsample_mode (`str`, *optional*, defaults to `"bilinear"`):
        The interpolation mode for upsampling layers to restore the original image resolution. Supported modes
        include `"bilinear"` (smooth upsampling, recommended for document images), `"nearest"`, and `"bicubic"`.
        Examples:
        ```python
        >>> from transformers import UVDocConfig, UVDocModelForImageToImage
        >>> # Initializing a UVDoc configuration
        >>> configuration = UVDocConfig()
        >>> # Customize configuration for grayscale document images
        >>> configuration = UVDocConfig(in_channels=1, upsample_size=[800, 600])
        >>> # Initializing a model (with random weights) from the configuration
        >>> model = UVDocModelForImageToImage(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    def __init__(
        self,
        num_filter: int = 32,
        in_channels: int = 3,
        kernel_size: int = 5,
        stride: list = [1, 2, 2, 2],
        map_num: list = [1, 2, 4, 8, 16],
        block_nums: list = [3, 4, 6, 3],
        dilation_values: dict | None = None,
        padding_mode: str = "reflect",
        upsample_size: list = [712, 488],
        upsample_mode: str = "bilinear",
        **kwargs,
    ):
        self.num_filter = num_filter
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.map_num = map_num
        self.block_nums = block_nums
        self.dilation_values = dilation_values
        self.padding_mode = padding_mode
        self.upsample_size = upsample_size
        self.upsample_mode = upsample_mode

        super().__init__(**kwargs)


class UVDocImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]

    """
    Image processor for the UVDoc model, tailored for document rectification tasks.
    This processor handles all preprocessing (resize, rescale, normalize, channel flip) and postprocessing
    steps required for UVDoc model inference on document images, using numpy-based operations for broad
    compatibility with PIL/numpy/torch image inputs.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the input images to the specified `size`. Disabling this is not recommended
            as UVDoc expects fixed-size inputs for document rectification.
        size (`Dict[str, int]`, *optional*, defaults to `{"height": 256, "width": 256}`):
            Target size for resizing images, specified as a dictionary with "height" and "width" keys.
            Adjust based on the UVDoc model's input requirements.
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
            Resampling filter to use for resizing. BICUBIC is recommended for document images to preserve
            text sharpness during resizing.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the pixel values from [0, 255] to [0, 1] using `rescale_factor`.
        rescale_factor (`Union[int, float]`, *optional*, defaults to `1/255`):
            Factor to scale pixel values by (e.g., 1/255 scales 0-255 to 0-1).
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the input images using `image_mean` and `image_std`. Normalization aligns
            input data with the statistics used during UVDoc model training.
        image_mean (`Union[float, List[float]]`, *optional*, defaults to `[0.406, 0.456, 0.485]`):
            Mean values for image normalization (RGB order), matching the training dataset statistics.
        image_std (`Union[float, List[float]]`, *optional*, defaults to `[0.225, 0.224, 0.229]`):
            Standard deviation values for image normalization (RGB order), matching the training dataset statistics.

    Attributes:
        model_input_names (`List[str]`):
            List of input names expected by the UVDoc model (only "pixel_values" for image inputs).
    """

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[dict[str, int]] = None,
        resample: Optional[PILImageResampling] = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, list[float]]] = [0.406, 0.456, 0.485],
        image_std: Optional[Union[float, list[float]]] = [0.225, 0.224, 0.229],
        **kwargs,
    ) -> None:
        """
        Initialize the UVDocImageProcessor with specified preprocessing parameters.
        Sets default size if not provided and stores all preprocessing hyperparameters.
        """
        super().__init__(**kwargs)
        size = size if size is not None else {"height": 256, "width": 256}

        self.do_resize = do_resize
        self.size = size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.resample = resample

    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        size: Optional[dict[str, int]] = None,
        do_resize: Optional[bool] = None,
        resample: Optional[PILImageResampling] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[Union[int, float]] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        return_tensors: Optional[Union[TensorType, str]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> BatchFeature:
        """
        Preprocess input images for UVDoc model inference (numpy-based implementation).

        Args:
            images (`ImageInput`):
                Input images to process (can be PIL images, numpy arrays, torch tensors, or lists thereof).
            size (`Dict[str, int]`, *optional*):
                Override the target resize size (defaults to self.size).
            do_resize (`bool`, *optional*):
                Override the do_resize flag (defaults to self.do_resize).
            resample (`PILImageResampling`, *optional*):
                Override the resampling filter (defaults to self.resample).
            do_rescale (`bool`, *optional*):
                Override the do_rescale flag (defaults to self.do_rescale).
            rescale_factor (`Union[int, float]`, *optional*):
                Override the rescale factor (defaults to self.rescale_factor).
            do_normalize (`bool`, *optional*):
                Override the do_normalize flag (defaults to self.do_normalize).
            image_mean (`Union[float, List[float]]`, *optional*):
                Override the normalization mean (defaults to self.image_mean).
            image_std (`Union[float, List[float]]`, *optional*):
                Override the normalization std (defaults to self.image_std).
            return_tensors (`Union[TensorType, str]`, *optional*):
                Type of tensors to return (e.g., "pt" for PyTorch tensors, "np" for numpy arrays).
            data_format (`Union[str, ChannelDimension]`, *optional*, defaults to `ChannelDimension.FIRST`):
                Output channel dimension format (FIRST = [C, H, W], LAST = [H, W, C]).
            input_data_format (`Union[str, ChannelDimension]`, *optional*):
                Channel dimension format of the input images (inferred if None).

        Returns:
            `BatchFeature`: A BatchFeature object containing the processed "pixel_values" tensor.

        Raises:
            ValueError: If input images are of invalid type (not PIL/numpy/torch).
        """
        size = self.size if size is None else size
        do_resize = self.do_resize if do_resize is None else do_resize
        resample = self.resample if resample is None else resample
        do_rescale = self.do_rescale if do_rescale is None else do_rescale
        rescale_factor = self.rescale_factor if rescale_factor is None else rescale_factor
        do_normalize = self.do_normalize if do_normalize is None else do_normalize
        image_mean = self.image_mean if image_mean is None else image_mean
        image_std = self.image_std if image_std is None else image_std

        images = make_flat_list_of_images(images)

        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            size=size,
            do_resize=do_resize,
            resample=resample,
        )

        if not valid_images(images):
            raise ValueError("Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, or torch.Tensor")

        # All transformations expect numpy arrays
        images = [to_numpy_array(image) for image in images]

        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(images[0])

        if do_rescale:
            images = [self.rescale(image, rescale_factor, input_data_format=input_data_format) for image in images]

        if do_normalize:
            images = [
                self.normalize(image, image_mean, image_std, input_data_format=input_data_format) for image in images
            ]
        # flip color channels from RGB to BGR
        images = [flip_channel_order(image, input_data_format=input_data_format) for image in images]
        images = [
            to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format) for image in images
        ]

        encoded_inputs = BatchFeature(data={"pixel_values": images}, tensor_type=return_tensors)

        return encoded_inputs

    def post_process_document_rectification(self, images, scale=None):
        """
        Postprocess UVDoc model outputs to get rectified document images (numpy-based).
        Converts model output tensors back to 0-255 RGB images ready for saving/display.

        Args:
            images (`Union[np.ndarray, Tuple[np.ndarray, ...], List[np.ndarray]]`):
                Raw model outputs (logits) to postprocess (batch of images).
            scale (`Union[str, float, int]`, *optional*, defaults to 255.0):
                Scaling factor to convert normalized outputs back to 0-255 pixel values.

        Returns:
            `List[np.ndarray]`: List of rectified document images (uint8, [H, W, C], RGB).
        """
        scale = np.float32(scale) if isinstance(scale, (str, float)) else np.float32(255.0)

        return [self.doctr(img, scale) for img in images]

    def doctr(self, pred: Union[np.ndarray, tuple[np.ndarray, ...]], scale) -> np.ndarray:
        """
        Core postprocessing logic for a single document image (numpy-based).
        Converts model output tensor to a valid RGB image (uint8, [H, W, C]).

        Args:
            pred (`Union[np.ndarray, Tuple[np.ndarray, ...]]`):
                Raw model output for a single image (or tuple containing the output).
            scale (`np.float32`):
                Scaling factor to convert normalized values to 0-255.

        Returns:
            `np.ndarray`: Rectified document image (uint8, [H, W, C], RGB).

        Raises:
            AssertionError: If input is not a numpy array.
        """
        if isinstance(pred, tuple):
            im = pred[0].cpu().detach().numpy()
        else:
            im = pred.cpu().detach().numpy()
        assert isinstance(im, np.ndarray), "Invalid input 'im' in DocTrPostProcess. Expected a numpy array."
        im = im.squeeze()
        im = im.transpose(1, 2, 0)
        im *= scale
        im = im[:, :, ::-1]
        im = im.astype("uint8", copy=False)
        return im


class UVDocImageProcessorFast(BaseImageProcessorFast):
    """
    Fast image processor for UVDoc models (PyTorch-optimized, inherits from `BaseImageProcessorFast`).
    Optimized for speed with torch tensor operations, skipping numpy conversions for low-latency inference.
    """

    resample = 2
    image_mean = [0, 0, 0]
    image_std = [1, 1, 1]
    size = {"height": 256, "width": 256}
    do_resize = True
    do_rescale = True
    do_normalize = True

    def __init__(self, **kwargs) -> None:
        """Initialize the fast UVDoc image processor (inherits class-level defaults)."""
        super().__init__(**kwargs)

    def _preprocess(
        self,
        images: list[torch.Tensor],
        size: Optional[list[dict[str, int]]],
        do_resize: bool,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        return_tensors: Optional[Union[str, TensorType]],
        interpolation: Optional[InterpolationMode] = None,
        resample: Optional[PILImageResampling] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Fast preprocessing for UVDoc model (pure PyTorch tensor operations).
        Optimized for GPU inference with minimal data conversion overhead.

        Args:
            images (`List[torch.Tensor]`):
                List of input images (PyTorch tensors, [C, H, W] format).
            size (`List[Dict[str, int]]`, *optional*):
                Ignored (class-level size is used for fast processing).
            do_resize (`bool`):
                Whether to resize images (ignored in fast implementation).
            do_rescale (`bool`):
                Whether to rescale pixel values from 0-255 to 0-1.
            rescale_factor (`float`):
                Factor to scale pixel values by (1/255 for 0-255 → 0-1).
            do_normalize (`bool`):
                Whether to normalize images with mean/std.
            image_mean (`Union[float, List[float]]`, *optional*):
                Override normalization mean (defaults to class-level image_mean).
            image_std (`Union[float, List[float]]`, *optional*):
                Override normalization std (defaults to class-level image_std).
            return_tensors (`Union[str, TensorType]`, *optional*):
                Type of tensors to return (only "pt" is supported for fast processing).
            interpolation (`InterpolationMode`, *optional*):
                Ignored (class-level resample is used).
            resample (`PILImageResampling`, *optional*):
                Ignored (class-level resample is used).

        Returns:
            `BatchFeature`: BatchFeature containing processed "pixel_values" (PyTorch tensor, [B, C, H, W]).
        """
        data = {}

        processed_images = []
        for image in images:
            image = self.rescale_and_normalize(image, do_rescale, rescale_factor, do_normalize, image_mean, image_std)
            processed_images.append(image)
        images = processed_images

        images = [image[[2, 1, 0], :, :] for image in images]
        data.update({"pixel_values": torch.stack(images, dim=0)})
        encoded_inputs = BatchFeature(data, tensor_type=return_tensors)

        return encoded_inputs

    def post_process_document_rectification(self, images, scale=None):
        """
        Fast postprocessing for UVDoc model outputs (pure PyTorch tensor operations).
        GPU-optimized conversion of model outputs to rectified document images (uint8).

        Args:
            images (`Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]]`):
                Raw model outputs (logits) to postprocess (batch of images, GPU tensors).
            scale (`Union[str, float, int]`, *optional*, defaults to 255.0):
                Scaling factor to convert normalized outputs back to 0-255 pixel values.

        Returns:
            `List[torch.Tensor]`: List of rectified document images (uint8, [H, W, C], RGB, same device as input).
        """
        if isinstance(scale, (str, float, int)):
            scale = torch.tensor(float(scale), device=images.device)
        else:
            scale = torch.tensor(255.0, device=images.device)

        return [self.doctr(img, scale) for img in images]

    def doctr(self, pred: Union[torch.Tensor, tuple[torch.Tensor, ...]], scale: torch.Tensor) -> torch.Tensor:
        """
        Core fast postprocessing logic for a single document image (pure PyTorch).
        Converts model output tensor to a valid RGB image (uint8, [H, W, C]) without CPU conversion.

        Args:
            pred (`Union[torch.Tensor, Tuple[torch.Tensor, ...]]`):
                Raw model output for a single image (or tuple containing the output, GPU tensor).
            scale (`torch.Tensor`):
                Scaling factor tensor (same device as input) to convert normalized values to 0-255.

        Returns:
            `torch.Tensor`: Rectified document image (uint8, [H, W, C], RGB, same device as input).

        Raises:
            AssertionError: If input is not a PyTorch tensor.
        """
        if isinstance(pred, tuple):
            im = pred[0]
        else:
            im = pred

        assert isinstance(im, torch.Tensor), "Invalid input 'im' in DocTrPostProcess. Expected a torch tensor."

        im = im.squeeze()
        im = im.permute(1, 2, 0)
        im = im * scale
        im = im.flip(dims=[-1])
        im = im.to(dtype=torch.uint8, non_blocking=True, copy=False)

        return im


def conv3x3(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1) -> nn.Conv2d:
    """
    Build a 3x3 standard convolutional layer with symmetric padding to maintain feature map dimensions.
    Used for basic ResNet blocks to ensure input/output size consistency when stride=1.

    Args:
        in_channels (`int`): Number of input channels
        out_channels (`int`): Number of output channels
        kernel_size (`int`): Size of convolutional kernel (typically 3)
        stride (`int`, *optional*, defaults to 1): Stride of the convolution

    Returns:
        `nn.Conv2d`: 3x3 convolutional layer instance
    """
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def dilated_conv(
    in_channels: int, out_channels: int, kernel_size: int, dilation: int, stride: int = 1
) -> nn.Sequential:
    """
    Build a dilated (atrous) convolutional layer to expand receptive field without increasing parameters/computation.
    Critical for capturing long-range geometric dependencies in distorted document images for rectification tasks.

    Args:
        in_channels (`int`): Number of input channels
        out_channels (`int`): Number of output channels
        kernel_size (`int`): Size of convolutional kernel (typically 3)
        dilation (`int`): Dilation rate (expansion factor) to control receptive field size
        stride (`int`, *optional*, defaults to 1): Stride of the convolution

    Returns:
        `nn.Sequential`: Sequential layer containing dilated convolution
    """
    model = nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation * (kernel_size // 2),
            dilation=dilation,
        )
    )
    return model


class ResidualBlockWithDilation(nn.Module):
    """
    Residual block with optional dilated convolution for UVDoc backbone.
    Uses standard convolution for downsampling layers and dilated convolution for middle layers
    to balance spatial resolution and receptive field size.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        downsample: Optional[nn.Sequential] = None,
        is_activation: bool = True,
        is_top: bool = False,
    ) -> None:
        """
        Initialize residual block with dilation support.

        Args:
            in_channels (`int`): Number of input channels
            out_channels (`int`): Number of output channels
            kernel_size (`int`): Size of convolutional kernel
            stride (`int`, *optional*, defaults to 1): Stride of the first convolution
            downsample (`nn.Sequential`, *optional*, defaults to None):
                Downsampling layer for residual connection (when stride != 1 or channel mismatch)
            is_activation (`bool`, *optional*, defaults to True):
                Whether to apply ReLU activation (always True for UVDoc)
            is_top (`bool`, *optional*, defaults to False):
                Whether this is the first block in the layer (uses standard conv instead of dilated conv)
        """
        super().__init__()
        self.stride = stride
        self.downsample = downsample
        self.is_activation = is_activation
        self.is_top = is_top
        if self.stride != 1 or self.is_top:
            self.conv1 = conv3x3(in_channels, out_channels, kernel_size, self.stride)
            self.conv2 = conv3x3(out_channels, out_channels, kernel_size)
        else:
            self.conv1 = dilated_conv(in_channels, out_channels, kernel_size, dilation=3)
            self.conv2 = dilated_conv(out_channels, out_channels, kernel_size, dilation=3)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of residual block with dilation.

        Args:
            x (`torch.Tensor`): Input tensor of shape [B, C, H, W]

        Returns:
            `torch.Tensor`: Output tensor of shape [B, C_out, H_out, W_out]
        """
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)
        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.bn2(self.conv2(out1))
        out2 += residual
        out = self.relu(out2)
        return out


class ResnetStraight(nn.Module):
    """
    Modified ResNet backbone for UVDoc with dilated residual blocks.
    Extracts multi-scale features from document images through progressive downsampling,
    using dilated convolution to maintain large receptive fields.
    """

    def __init__(
        self,
        num_filter: int,
        map_num: list[int],
        block_nums: list[int] = [3, 4, 6, 3],
        kernel_size: int = 5,
        stride: list[int] = [1, 1, 2, 2],
    ) -> None:
        """
        Initialize ResNet backbone for UVDoc downsampling.

        Args:
            num_filter (`int`): Base number of convolutional filters
            map_num (`List[int]`): Channel scaling factors for each ResNet layer
            block_nums (`List[int]`, *optional*, defaults to [3,4,6,3]):
                Number of residual blocks per layer
            kernel_size (`int`, *optional*, defaults to 5): Size of convolutional kernel
            stride (`List[int]`, *optional*, defaults to [1,1,2,2]):
                Stride values for each ResNet layer (controls downsampling rate)
        """
        super().__init__()
        self.in_channels = num_filter * map_num[0]
        self.stride = stride
        self.relu = nn.ReLU()
        self.block_nums = block_nums
        self.kernel_size = kernel_size
        for layer_idx, (map_num_val, block_num, stride_val) in enumerate(zip(map_num[:3], block_nums[:3], stride[:3])):
            layer = self.blocklayer(
                num_filter * map_num_val,
                block_num,
                kernel_size=self.kernel_size,
                stride=stride_val,
            )
            setattr(self, f"layer{layer_idx + 1}", layer)

    def blocklayer(self, out_channels: int, block_nums: int, kernel_size: int, stride: int = 1) -> nn.Sequential:
        """
        Build a single ResNet layer containing multiple residual blocks.

        Args:
            out_channels (`int`): Number of output channels for the layer
            block_nums (`int`): Number of residual blocks in the layer
            kernel_size (`int`): Size of convolutional kernel
            stride (`int`, *optional*, defaults to 1): Stride for the first block (downsampling)

        Returns:
            `nn.Sequential`: ResNet layer with multiple residual blocks
        """
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                conv3x3(
                    self.in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                ),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        for i in range(block_nums):
            layers.append(
                ResidualBlockWithDilation(
                    in_channels=self.in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride if i == 0 else 1,
                    downsample=downsample if i == 0 else None,
                    is_top=i == 0,
                )
            )
        self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of ResNet backbone (downsampling path).

        Args:
            x (`torch.Tensor`): Input tensor of shape [B, C, H, W]

        Returns:
            `torch.Tensor`: Output feature map from the last ResNet layer (layer3)
        """
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        return out3


@dataclass
class UVDocModelOutput(ModelOutput):
    """
    Output class for UVDoc model forward pass.

    Args:
        logits (`torch.FloatTensor`, *optional*):
            Rectified document image tensor of shape [B, C, H, W]
        last_hidden_state (`torch.FloatTensor`, *optional*):
            Last hidden state from bridge layers of shape [B, C, H, W]
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of hidden states from each bridge layer (if output_hidden_states=True)
    """

    logits: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None


class UVDocPreTrainedModel(PreTrainedModel):
    """
    Base class for all UVDoc pre-trained models.
    Inherits from Hugging Face PreTrainedModel and sets UVDoc-specific configurations.
    """

    config: UVDocConfig
    base_model_prefix = "uvdoc"
    main_input_name = "pixel_values"
    input_modalities = ("image",)


def dilated_conv_bn_act(in_channels: int, out_channels: int, dilation: int) -> nn.Sequential:
    """
    Build a dilated convolution block with BatchNorm and ReLU activation.
    Used for UVDoc bridge layers to extract multi-scale geometric features.

    Args:
        in_channels (`int`): Number of input channels
        out_channels (`int`): Number of output channels
        dilation (`int`): Dilation rate for dilated convolution

    Returns:
        `nn.Sequential`: Dilated conv → BN → ReLU block
    """
    model = nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            bias=False,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )
    return model


class UVDocModel(UVDocPreTrainedModel):
    """
    Core UVDoc model for document rectification.
    Combines ResNet backbone, multi-scale bridge layers, and spatial transformation
    to correct perspective distortion in document images.
    """

    def __init__(self, config: UVDocConfig) -> None:
        """
        Initialize UVDoc core model with configuration.

        Args:
            config (`UVDocConfig`): UVDoc model configuration
        """
        super().__init__(config)

        self.upsample_size = config.upsample_size
        self.upsample_mode = config.upsample_mode
        self.resnet_head = nn.Sequential(
            nn.Conv2d(
                config.in_channels,
                config.num_filter * config.map_num[0],
                bias=False,
                kernel_size=config.kernel_size,
                stride=2,
                padding=config.kernel_size // 2,
            ),
            nn.BatchNorm2d(config.num_filter * config.map_num[0]),
            nn.ReLU(),
            nn.Conv2d(
                config.num_filter * config.map_num[0],
                config.num_filter * config.map_num[0],
                bias=False,
                kernel_size=config.kernel_size,
                stride=2,
                padding=config.kernel_size // 2,
            ),
            nn.BatchNorm2d(config.num_filter * config.map_num[0]),
            nn.ReLU(),
        )

        self.resnet_down = ResnetStraight(
            config.num_filter,
            config.map_num,
            block_nums=config.block_nums,
            kernel_size=config.kernel_size,
            stride=config.stride,
        )

        bridge_in_channels = config.num_filter * config.map_num[2]

        def _build_bridge(bridge_key: str) -> nn.Sequential:
            """
            Build bridge layer with specified dilation values from config.
            Supports both single dilation rate and multiple dilation rates.

            Args:
                bridge_key (`str`): Key for dilation values in config (e.g., "bridge_1")

            Returns:
                `nn.Sequential`: Bridge layer with dilated convolution blocks
            """
            dilation = config.dilation_values[bridge_key]
            if isinstance(dilation, int):
                return nn.Sequential(dilated_conv_bn_act(bridge_in_channels, bridge_in_channels, dilation=dilation))
            else:
                return nn.Sequential(
                    *[dilated_conv_bn_act(bridge_in_channels, bridge_in_channels, dilation=d) for d in dilation]
                )

        self.bridge_1 = _build_bridge("bridge_1")
        self.bridge_2 = _build_bridge("bridge_2")
        self.bridge_3 = _build_bridge("bridge_3")
        self.bridge_4 = _build_bridge("bridge_4")
        self.bridge_5 = _build_bridge("bridge_5")
        self.bridge_6 = _build_bridge("bridge_6")

        self.bridge_concat = nn.Sequential(
            nn.Conv2d(
                config.num_filter * config.map_num[2] * 6,
                config.num_filter * config.map_num[2],
                bias=False,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(config.num_filter * config.map_num[2]),
            nn.ReLU(),
        )

        self.out_point_positions2D = nn.Sequential(
            nn.Conv2d(
                config.num_filter * config.map_num[2],
                config.num_filter * config.map_num[0],
                bias=False,
                kernel_size=config.kernel_size,
                stride=1,
                padding=config.kernel_size // 2,
                padding_mode=config.padding_mode,
            ),
            nn.BatchNorm2d(config.num_filter * config.map_num[0]),
            nn.PReLU(),
            nn.Conv2d(
                config.num_filter * config.map_num[0],
                2,
                kernel_size=config.kernel_size,
                stride=1,
                padding=config.kernel_size // 2,
                padding_mode=config.padding_mode,
            ),
        )

        self.post_init()

    def forward(
        self,
        hidden_state: torch.FloatTensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Union[tuple[torch.FloatTensor, ...], UVDocModelOutput]:
        """
        Forward pass of UVDoc core model for document rectification.

        Args:
            hidden_state (`torch.FloatTensor`): Input image tensor of shape [B, C, H, W]
            output_hidden_states (`bool`, *optional*):
                Whether to return hidden states from bridge layers
            return_dict (`bool`, *optional*):
                Whether to return a UVDocModelOutput object instead of a plain tuple

        Returns:
            `Union[Tuple[torch.FloatTensor, ...], UVDocModelOutput]`:
                - If return_dict=True: UVDocModelOutput with logits, last_hidden_state, hidden_states
                - If return_dict=False: Tuple containing (last_hidden_state, [hidden_states], logits)
        """
        hidden_states = () if output_hidden_states else None

        image = hidden_state
        h_ori, w_ori = hidden_state.shape[2:]
        hidden_state = F.interpolate(
            hidden_state,
            size=(self.upsample_size[0], self.upsample_size[1]),
            mode=self.upsample_mode,
            align_corners=True,
        )
        hidden_state = self.resnet_head(hidden_state)
        resnet_down = self.resnet_down(hidden_state)
        if output_hidden_states:
            hidden_states = hidden_states + (resnet_down,)
        bridge_1 = self.bridge_1(resnet_down)
        if output_hidden_states:
            hidden_states = hidden_states + (bridge_1,)
        bridge_2 = self.bridge_2(resnet_down)
        if output_hidden_states:
            hidden_states = hidden_states + (bridge_2,)
        bridge_3 = self.bridge_3(resnet_down)
        if output_hidden_states:
            hidden_states = hidden_states + (bridge_3,)
        bridge_4 = self.bridge_4(resnet_down)
        if output_hidden_states:
            hidden_states = hidden_states + (bridge_4,)
        bridge_5 = self.bridge_5(resnet_down)
        if output_hidden_states:
            hidden_states = hidden_states + (bridge_5,)
        bridge_6 = self.bridge_6(resnet_down)
        if output_hidden_states:
            hidden_states = hidden_states + (bridge_6,)
        last_hidden_state = bridge_6

        bridge_concat = torch.cat([bridge_1, bridge_2, bridge_3, bridge_4, bridge_5, bridge_6], dim=1)
        bridge = self.bridge_concat(bridge_concat)

        out_point_positions2D = self.out_point_positions2D(bridge)

        bm_up = F.interpolate(
            out_point_positions2D,
            size=(h_ori, w_ori),
            mode=self.upsample_mode,
            align_corners=True,
        )
        bm = bm_up.permute(0, 2, 3, 1)
        out = F.grid_sample(image, bm, align_corners=True)

        if not return_dict:
            output = (last_hidden_state,)
            if output_hidden_states:
                output += (hidden_states,)
            output += (out,)
            return output

        return UVDocModelOutput(
            logits=out,
            last_hidden_state=last_hidden_state,
            hidden_states=hidden_states if output_hidden_states else None,
        )


@dataclass
class UVDocForDocumentRectificationOutput(BaseModelOutputWithNoAttention):
    """
    Output class for UVDocForDocumentRectification forward pass.
    Extends BaseModelOutputWithNoAttention with document rectification logits.

    Args:
        logits (`torch.FloatTensor`, *optional*):
            Rectified document image tensor of shape [B, C, H, W]
        shape (`torch.FloatTensor`, *optional*):
            Reserved for future use (shape information)
    """

    logits: Optional[torch.FloatTensor] = None
    shape: Optional[torch.FloatTensor] = None


class UVDocForDocumentRectification(UVDocPreTrainedModel):
    """
    Wrapper class for UVDoc model focused on document rectification task.
    Provides a user-friendly interface for inference/training with standard Hugging Face API.
    """

    _keys_to_ignore_on_load_missing = ["num_batches_tracked"]

    def __init__(self, config: UVDocConfig) -> None:
        """
        Initialize UVDoc document rectification wrapper model.

        Args:
            config (`UVDocConfig`): UVDoc model configuration
        """
        super().__init__(config)
        self.model = UVDocModel(config)
        self.post_init()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[list[dict[str, Any]]] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Union[tuple[torch.FloatTensor, ...], UVDocForDocumentRectificationOutput]:
        """
        Forward pass of UVDoc document rectification model.

        Args:
            pixel_values (`torch.FloatTensor`): Input image tensor of shape [B, C, H, W] (preprocessed)
            labels (`List[Dict[str, Any]]`, *optional*):
                Training labels (not used in inference)
            output_hidden_states (`bool`, *optional*):
                Whether to return hidden states from the core model
            return_dict (`bool`, *optional*):
                Whether to return a UVDocForDocumentRectificationOutput object instead of a plain tuple

        Returns:
            `Union[Tuple[torch.FloatTensor, ...], UVDocForDocumentRectificationOutput]`:
                - If return_dict=True: Structured output with logits, last_hidden_state, hidden_states
                - If return_dict=False: Plain tuple with corresponding values
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

        if not return_dict:
            output = (outputs[0],)
            if output_hidden_states:
                output += (outputs[1], outputs[2])
            else:
                output += (outputs[1],)

            return output
        return UVDocForDocumentRectificationOutput(
            logits=outputs.logits,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
        )


__all__ = [
    "UVDocForDocumentRectification",
    "UVDocImageProcessor",
    "UVDocImageProcessorFast",
    "UVDocConfig",
    "UVDocModel",
    "UVDocPreTrainedModel",
]
