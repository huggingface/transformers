import math
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms.v2.functional import InterpolationMode

from ...configuration_utils import PreTrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...image_processing_utils import BaseImageProcessor
from ...image_processing_utils_fast import BaseImageProcessorFast
from ...image_transforms import flip_channel_order, resize, to_channel_dimension_format
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
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


class PPLCNetConfig(PreTrainedConfig):
    model_type = "pp_lcnet"

    """
    This is the configuration class to store the configuration of a [`PPLCNet`]. It is used to instantiate a
    PP-LCNet model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the PP-LCNet
    [PaddlePaddle/PP-LCNet_x1_0_doc_ori_safetensors](https://huggingface.co/PaddlePaddle/PP-LCNet_x1_0_doc_ori_safetensors) architecture.
    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.
    Args:
        scale (`float`, *optional*, defaults to 1.0):
            The scaling factor for the model's channel dimensions, used to adjust the model size and computational cost
            without changing the overall architecture (e.g., 0.25, 0.5, 1.0, 1.5).
        class_num (`int`, *optional*, defaults to 4):
            The number of output classes for the classification task. Typical values are 2 (binary classification) or
            4 (document orientation classification: 0°, 90°, 180°, 270°).
        stride_list (`List[int]`, *optional*, defaults to `[2, 2, 2, 2, 2]`):
            The list of stride values for convolutional layers in the backbone network, controlling the downsampling
            rate of feature maps at each stage to capture multi-scale visual information.
        reduction (`int`, *optional*, defaults to 4):
            The reduction factor for feature channel dimensions in the squeeze-and-excitation (SE) blocks, used to
            reduce the number of model parameters and computational complexity while maintaining feature representability.
        dropout_prob (`float`, *optional*, defaults to 0.2):
            The dropout probability for the classification head, used to prevent overfitting by randomly zeroing out
            a fraction of the neurons during training.
        class_expand (`int`, *optional*, defaults to 1280):
            The number of hidden units in the expansion layer of the classification head, used to enhance the model's
            feature representation capability before the final classification layer.
        use_last_conv (`bool`, *optional*, defaults to `True`):
            Whether to use the final convolutional layer in the classification head. Setting this to `True` helps
            extract more discriminative features for the classification task.
        act (`str`, *optional*, defaults to `"hardswish"`):
            The non-linear activation function used in the model's hidden layers. Supported functions include
            `"hardswish"`, `"relu"`, `"silu"`, and `"gelu"`. `"hardswish"` is preferred for lightweight and efficient
            inference on edge devices.
        backbone_config (`Union[dict, PreTrainedConfig]`, *optional*, defaults to `None`):
            The configuration of the backbone model. If `None`, the default backbone configuration for PP-LCNet
            will be used, which includes the standard block settings for feature extraction.

    Examples:
    ```python
    >>> from transformers import PPLCNetConfig, PPLCNetForImageClassification
    >>> # Initializing a PP-LCNet configuration
    >>> configuration = PPLCNetConfig()
    >>> # Initializing a model (with random weights) from the configuration
    >>> model = PPLCNetForImageClassification(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    """

    def __init__(
        self,
        scale: float = 1.0,
        class_num: int = 4,
        stride_list: list[int] = [2, 2, 2, 2, 2],
        reduction: int = 4,
        dropout_prob: float = 0.2,
        class_expand: int = 1280,
        use_last_conv: bool = True,
        act: str = "hardswish",
        backbone_config: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.scale = scale
        self.class_num = class_num
        self.stride_list = stride_list
        self.reduction = reduction
        self.dropout_prob = dropout_prob
        self.class_expand = class_expand
        self.use_last_conv = use_last_conv
        self.act = act
        self.backbone_config = backbone_config


class PPLCNetImageProcessor(BaseImageProcessor):
    """
    Image processor for PP-LCNet models, handling all preprocessing steps required for image classification:
    resizing, center cropping, rescaling, normalization, and channel order flipping (RGB → BGR).
    Inherits from Hugging Face `BaseImageProcessor` and follows the standard preprocessing pipeline for lightweight
    CNN models in document/image classification tasks.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        resize_short: Optional[int] = None,
        size_divisor: Optional[int] = None,
        do_resize: bool = True,
        size: Optional[dict[str, int]] = None,
        resample: Optional[PILImageResampling] = PILImageResampling.BICUBIC,
        do_center_crop: bool = True,
        crop_size: int = 224,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, list[float]]] = [0.406, 0.456, 0.485],
        image_std: Optional[Union[float, list[float]]] = [0.225, 0.224, 0.229],
        **kwargs,
    ) -> None:
        """
        Initialize the PPLCNetImageProcessor with preprocessing configuration.

        Args:
            resize_short (Optional[int]): Target length for the shorter edge of the image (overrides `size` if set).
                Defaults to None.
            size_divisor (Optional[int]): Divisor to align image dimensions (for hardware optimization, e.g., 32).
                Defaults to None.
            do_resize (bool): Whether to resize the input image. Defaults to True.
            size (Optional[Dict[str, int]]): Target size for resizing ({"height": H, "width": W}). Defaults to
                {"height": 256, "width": 256} if None.
            resample (Optional[PILImageResampling]): PIL resampling mode for resizing. Defaults to BICUBIC (high quality).
            do_center_crop (bool): Whether to perform center cropping after resizing. Defaults to True.
            crop_size (int): Target size for center cropping (square crop, H=Crop_size, W=Crop_size). Defaults to 224.
            do_rescale (bool): Whether to rescale pixel values from [0, 255] to [0, 1]. Defaults to True.
            rescale_factor (Union[int, float]): Factor to rescale pixel values (1/255 for [0,255] → [0,1]). Defaults to 1/255.
            do_normalize (bool): Whether to normalize pixel values with mean and std. Defaults to True.
            image_mean (Optional[Union[float, List[float]]]): Mean values for image normalization (BGR order, matching PP-LCNet defaults).
                Defaults to [0.406, 0.456, 0.485].
            image_std (Optional[Union[float, List[float]]]): Standard deviation values for image normalization (BGR order, matching PP-LCNet defaults).
                Defaults to [0.225, 0.224, 0.229].
            **kwargs: Additional keyword arguments passed to `BaseImageProcessor`.
        """
        super().__init__(**kwargs)
        size = size if size is not None else {"height": 256, "width": 256}

        self.resize_short = resize_short
        self.size_divisor = size_divisor

        self.do_resize = do_resize
        self.size = size
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
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
        resize_short: Optional[int] = None,
        size_divisor: Optional[int] = None,
        size: Optional[dict[str, int]] = None,
        do_resize: Optional[bool] = None,
        resample: Optional[PILImageResampling] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[Union[int, float]] = None,
        do_center_crop: Optional[bool] = None,
        crop_size: int = 224,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        return_tensors: Optional[Union[TensorType, str]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> BatchFeature:
        """
        Main preprocessing pipeline for input images: applies resize → center crop → rescale → normalize → channel flip.

        Args:
            images (ImageInput): Input images to preprocess (single image or list of images; supports PIL, numpy, or torch tensor).
            resize_short (Optional[int]): Override for the class-level `resize_short`. Defaults to None.
            size_divisor (Optional[int]): Override for the class-level `size_divisor`. Defaults to None.
            size (Optional[Dict[str, int]]): Override for the class-level `size`. Defaults to None.
            do_resize (Optional[bool]): Override for the class-level `do_resize`. Defaults to None.
            resample (Optional[PILImageResampling]): Override for the class-level `resample`. Defaults to None.
            do_rescale (Optional[bool]): Override for the class-level `do_rescale`. Defaults to None.
            rescale_factor (Optional[Union[int, float]]): Override for the class-level `rescale_factor`. Defaults to None.
            do_center_crop (Optional[bool]): Override for the class-level `do_center_crop`. Defaults to None.
            crop_size (int): Override for the class-level `crop_size` (only if `do_center_crop` is True). Defaults to 224.
            do_normalize (Optional[bool]): Override for the class-level `do_normalize`. Defaults to None.
            image_mean (Optional[Union[float, List[float]]]): Override for the class-level `image_mean`. Defaults to None.
            image_std (Optional[Union[float, List[float]]]): Override for the class-level `image_std`. Defaults to None.
            return_tensors (Optional[Union[TensorType, str]]): Type of tensors to return (e.g., "pt" for PyTorch, "np" for numpy).
                Defaults to None (returns raw numpy arrays).
            data_format (Union[str, ChannelDimension]): Output channel dimension format (FIRST = [C, H, W], LAST = [H, W, C]).
                Defaults to ChannelDimension.FIRST (compatible with PyTorch).
            input_data_format (Optional[Union[str, ChannelDimension]]): Channel dimension format of the input images.
                Defaults to None (auto-infer from input).

        Returns:
            BatchFeature: Preprocessed image batch with key "pixel_values" containing the processed tensors/arrays.

        Raises:
            ValueError: If input images are of an invalid type (not PIL, numpy array, or torch tensor).
            RuntimeError: If resizing fails for any input image.
        """
        size = self.size if size is None else size
        resize_short = self.resize_short if resize_short is None else resize_short
        size_divisor = self.size_divisor if size_divisor is None else size_divisor
        do_resize = self.do_resize if do_resize is None else do_resize
        resample = self.resample if resample is None else resample
        do_center_crop = self.do_center_crop if do_center_crop is None else do_center_crop
        crop_size = self.crop_size if crop_size is None else crop_size
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
            do_center_crop=do_center_crop,
            crop_size=crop_size,
        )

        if not valid_images(images):
            raise ValueError("Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, or torch.Tensor")

        # All transformations expect numpy arrays
        images = [to_numpy_array(image) for image in images]

        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(images[0])

        # transformations
        resize_imgs = []
        if do_resize:
            for image in images:
                if resize_short is not None:
                    size = self.get_image_size(image, target_short_edge=resize_short, size_divisor=size_divisor)
                try:
                    img = resize(
                        image,
                        size=(size["height"], size["width"]),
                        resample=resample,
                        input_data_format=input_data_format,
                    )
                except Exception as e:
                    print(size)
                    raise RuntimeError(f"Failed to resize image: {e}") from e
                resize_imgs.append(img)
            images = resize_imgs

        if do_center_crop:
            images = [
                self.center_crop(image=image, size=crop_size, input_data_format=input_data_format) for image in images
            ]

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

    def get_image_size(
        self,
        img: np.ndarray,
        target_short_edge: Union[int, None],
        size_divisor: Optional[int] = None,
    ) -> tuple[dict, np.ndarray]:
        """
        Calculate target image size while preserving aspect ratio (based on shorter edge) and aligning with size_divisor.

        Args:
            img (np.ndarray): Input image array (shape [H, W, C] or [C, H, W]).
            target_short_edge (Union[int, None]): Desired length for the shorter edge of the image.
            size_divisor (Optional[int]): Divisor to align image dimensions (for hardware optimization). Defaults to None.

        Returns:
            Dict[str, int]: Target size {"height": resize_h, "width": resize_w} with preserved aspect ratio.
        """
        h, w = img.shape[:2]
        scale = target_short_edge / min(h, w)
        resize_h = round(h * scale)
        resize_w = round(w * scale)
        if self.size_divisor is not None:
            resize_h = math.ceil(resize_h / size_divisor) * size_divisor
            resize_h = math.ceil(resize_h / size_divisor) * size_divisor

        return {"height": resize_h, "width": resize_w}


class PPLCNetImageProcessorFast(BaseImageProcessorFast):
    """
    Fast image processor for PP-LCNet models (PyTorch-optimized, inherits from `BaseImageProcessorFast`).
    Optimized for speed with torch tensor operations, skipping numpy conversions for low-latency inference.
    """

    resample = 2
    image_mean = [0.406, 0.456, 0.485]
    image_std = [0.225, 0.224, 0.229]
    size = {"height": 256, "width": 256}
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_center_crop = True
    crop_size = 224
    resize_short = None
    size_divisor = None

    def __init__(self, **kwargs) -> None:
        """
        Initialize the PPLCNetImageProcessorFast.

        Args:
            **kwargs: Additional keyword arguments passed to `BaseImageProcessorFast`.
        """
        super().__init__(**kwargs)

    def _preprocess(
        self,
        images: list[torch.Tensor],
        size: Optional[list[dict[str, int]]],
        do_resize: bool,
        do_center_crop: bool,
        crop_size: SizeDict,
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
        Fast preprocessing pipeline for PyTorch tensors (optimized for low latency): resize → center crop → rescale → normalize → channel flip.

        Args:
            images (List[torch.Tensor]): List of input PyTorch tensors (shape [C, H, W]).
            size (Optional[List[Dict[str, int]]]): List of target sizes for each image (one per image). Defaults to None.
            do_resize (bool): Whether to resize the input images.
            do_center_crop (bool): Whether to perform center cropping after resizing.
            crop_size (SizeDict): Target size for center cropping ({"height": H, "width": W}).
            do_rescale (bool): Whether to rescale pixel values from [0, 255] to [0, 1].
            rescale_factor (float): Factor to rescale pixel values (1/255 for [0,255] → [0,1]).
            do_normalize (bool): Whether to normalize pixel values with mean and std.
            image_mean (Optional[Union[float, List[float]]]): Mean values for image normalization (BGR order).
            image_std (Optional[Union[float, List[float]]]): Standard deviation values for image normalization (BGR order).
            return_tensors (Optional[Union[str, TensorType]]): Type of tensors to return (e.g., "pt" for PyTorch).
                Defaults to None (returns PyTorch tensors).
            interpolation (Optional[InterpolationMode]): TorchVision interpolation mode for resizing. Defaults to None.
            resample (Optional[PILImageResampling]): Unused (for compatibility with base class). Defaults to None.
            **kwargs: Additional unused keyword arguments (for compatibility).

        Returns:
            BatchFeature: Preprocessed image batch with key "pixel_values" containing the processed PyTorch tensors.
        """
        data = {}
        resize_imgs = []
        if do_resize:
            for image in images:
                if self.resize_short is not None:
                    size = self.get_image_size(
                        image, target_short_edge=self.resize_short, size_divisor=self.size_divisor
                    )

                img = self.resize(image, size=size, interpolation=interpolation)
                resize_imgs.append(img)
            images = resize_imgs

        crop_images = []
        if do_center_crop:
            for image in images:
                image = self.center_crop(image, crop_size)
                crop_images.append(image)
            images = crop_images

        processed_images = []
        for image in images:
            image = self.rescale_and_normalize(image, do_rescale, rescale_factor, do_normalize, image_mean, image_std)
            processed_images.append(image)
        images = processed_images

        images = [image[[2, 1, 0], :, :] for image in images]
        data.update({"pixel_values": torch.stack(images, dim=0)})
        encoded_inputs = BatchFeature(data, tensor_type=return_tensors)

        return encoded_inputs

    def get_image_size(
        self,
        img: torch.Tensor,
        target_short_edge: Union[int, None],
        size_divisor: Optional[int] = None,
    ) -> tuple[SizeDict, torch.Tensor]:
        """
        Calculate target image size for PyTorch tensors (preserve aspect ratio + align with size_divisor).

        Args:
            img (torch.Tensor): Input PyTorch tensor (shape [C, H, W]).
            target_short_edge (Union[int, None]): Desired length for the shorter edge of the image.
            size_divisor (Optional[int]): Divisor to align image dimensions (for hardware optimization). Defaults to None.

        Returns:
            SizeDict: Target size ({"height": resize_h, "width": resize_w}) with preserved aspect ratio.
        """
        c, h, w = img.shape
        scale = target_short_edge / min(h, w)
        resize_h = round(h * scale)
        resize_w = round(w * scale)
        if self.size_divisor is not None:
            resize_h = math.ceil(resize_h / size_divisor) * size_divisor
            resize_h = math.ceil(resize_h / size_divisor) * size_divisor

        return SizeDict(height=resize_h, width=resize_w)


def make_divisible(v, divisor=8, min_value=None):
    """
    Ensure that the number of channels is divisible by the given divisor (for hardware optimization).

    Args:
        v (float): Original number of channels.
        divisor (int, optional): Divisor for channel adjustment. Defaults to 8.
        min_value (Optional[int]): Minimum number of channels after adjustment. Defaults to None.

    Returns:
        int: Adjusted number of channels that is divisible by the divisor.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _create_act(act):
    """
    Create the corresponding PyTorch activation module based on activation name.

    Args:
        act (str): Name of activation function. Supported values: "hardswish", "relu", "relu6".

    Returns:
        nn.Module: Instantiated PyTorch activation module.

    Raises:
        RuntimeError: If the input activation name is not supported.
    """
    if act == "hardswish":
        return nn.Hardswish()
    elif act == "relu":
        return nn.ReLU()
    elif act == "relu6":
        return nn.ReLU6()
    else:
        raise RuntimeError(f"The activation function is not supported: {act}")


class ConvBNLayer(nn.Module):
    """
    Combined layer: Conv2d -> BatchNorm2d -> Activation
    A common basic component in lightweight models to reduce redundant code.
    """

    def __init__(
        self,
        num_channels,
        filter_size,
        num_filters,
        stride,
        num_groups=1,
        act="hardswish",
    ):
        """
        Initialize the ConvBNLayer module.

        Args:
            num_channels (int): Number of channels of the input feature map.
            filter_size (int): Size of the convolutional kernel (square kernel).
            num_filters (int): Number of channels of the output feature map.
            stride (int): Stride of the convolution operation.
            num_groups (int, optional): Number of groups for grouped convolution. Defaults to 1 (standard convolution).
            act (str, optional): Name of activation function. Defaults to "hardswish".
        """
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=num_groups,
            bias=False,
        )

        nn.init.kaiming_normal_(self.conv.weight)
        self.bn = nn.BatchNorm2d(
            num_filters,
            momentum=0.9,
        )
        self.act = _create_act(act)

    def forward(self, x):
        """
        Forward propagation logic.

        Args:
            x (FloatTensor): Input feature map with shape [B, C, H, W].

        Returns:
            FloatTensor: Output feature map with shape [B, num_filters, H', W'].
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class DepthwiseSeparable(nn.Module):
    """
    Depthwise Separable Convolution Layer: Depthwise Conv -> SE Module (optional) -> Pointwise Conv
    Core component of lightweight models (e.g., MobileNet, PP-LCNet) that significantly reduces
    the number of parameters and computational cost.
    """

    def __init__(
        self,
        num_channels,
        num_filters,
        stride,
        reduction: int,
        dw_size=3,
        use_se=False,
        act="hardswish",
    ):
        """
        Initialize the DepthwiseSeparable module.

        Args:
            num_channels (int): Number of channels of the input feature map.
            num_filters (int): Number of channels of the output feature map.
            stride (int): Stride of the depthwise convolution.
            reduction (int): Reduction ratio for SE module.
            dw_size (int, optional): Kernel size of depthwise convolution. Defaults to 3.
            use_se (bool, optional): Whether to use SE module. Defaults to False.
            act (str, optional): Name of activation function. Defaults to "hardswish".
        """
        super().__init__()
        self.use_se = use_se
        self.dw_conv = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_channels,
            filter_size=dw_size,
            stride=stride,
            num_groups=num_channels,
            act=act,
        )
        self.se = SEModule(num_channels, reduction) if use_se else nn.Identity()
        self.pw_conv = ConvBNLayer(
            num_channels=num_channels,
            filter_size=1,
            num_filters=num_filters,
            stride=1,
            act=act,
        )

    def forward(self, x):
        """
        Forward propagation logic.

        Args:
            x (FloatTensor): Input feature map with shape [B, C, H, W].

        Returns:
            FloatTensor: Output feature map with shape [B, num_filters, H', W'].
        """
        x = self.dw_conv(x)
        x = self.se(x)
        x = self.pw_conv(x)
        return x


class SEModule(nn.Module):
    """
    Squeeze-and-Excitation (SE) Module: Adaptive feature recalibration
    Enhances the model's ability to focus on important channels by learning channel-wise attention weights.
    """

    def __init__(self, channel, reduction=4):
        """
        Initialize the SEModule module.

        Args:
            channel (int): Number of channels of the input feature map.
            reduction (int, optional): Reduction ratio for bottleneck layer. Defaults to 4.
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        conv_kwargs = {"kernel_size": 1, "stride": 1, "padding": 0, "bias": True}
        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, **conv_kwargs)
        self.conv2 = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, **conv_kwargs)
        self.relu = nn.ReLU()

        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        """
        Forward propagation logic.

        Args:
            x (FloatTensor): Input feature map with shape [B, C, H, W].

        Returns:
            FloatTensor: Attention-weighted feature map with shape [B, C, H, W].
        """
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        x = identity * x
        return x


@dataclass
class PPLCNetModelOutput(ModelOutput):
    """
    Output class for PP-LCNet base model.

    Args:
        logits (Optional[FloatTensor]): Classification logits with shape [B, num_classes]. Defaults to None.
        last_hidden_state (Optional[FloatTensor]): Final hidden state from backbone with shape [B, C, H, W]. Defaults to None.
        hidden_states (Optional[Tuple[FloatTensor, ...]]): Tuple of hidden states at each layer. Defaults to None.
    """

    logits: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None


class PPLCNetPreTrainedModel(PreTrainedModel):
    """
    An abstract base class for PP-LCNet models that inherits from Hugging Face PreTrainedModel.
    Provides common functionality for weight initialization and loading.
    """

    config: PPLCNetConfig
    base_model_prefix = "pp_lcnet"
    main_input_name = "pixel_values"
    input_modalities = ("image",)


class PPLCNetModel(PPLCNetPreTrainedModel):
    """
    PP-LCNet base model: lightweight CNN backbone for image classification tasks.
    """

    def __init__(self, config: PPLCNetConfig):
        """
        Initialize the PPLCNetModel with given configuration.

        Args:
            config (PPLCNetConfig): Configuration class for PP-LCNet.
        """
        super().__init__(config)

        assert isinstance(config.stride_list, (list, tuple)), (
            f"stride_list should be in (list, tuple) but got {type(config.stride_list)}"
        )
        assert len(config.stride_list) == 5, f"stride_list length should be 5 but got {len(config.stride_list)}"

        self.dropout_prob = config.dropout_prob

        for i, stride in enumerate(config.stride_list[1:]):
            config.backbone_config[f"blocks{i + 3}"][0][3] = stride
        self.conv1 = ConvBNLayer(
            num_channels=3,
            filter_size=3,
            num_filters=make_divisible(16 * config.scale),
            stride=config.stride_list[0],
            act=config.act,
        )

        def _build_block(block_name):
            return nn.Sequential(
                *[
                    DepthwiseSeparable(
                        num_channels=make_divisible(in_c * config.scale),
                        num_filters=make_divisible(out_c * config.scale),
                        dw_size=k,
                        stride=s,
                        reduction=config.reduction,
                        use_se=se,
                        act=config.act,
                    )
                    for i, (k, in_c, out_c, s, se) in enumerate(config.backbone_config[block_name])
                ]
            )

        self.blocks2 = _build_block("blocks2")
        self.blocks3 = _build_block("blocks3")
        self.blocks4 = _build_block("blocks4")
        self.blocks5 = _build_block("blocks5")
        self.blocks6 = _build_block("blocks6")

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if config.use_last_conv:
            self.last_conv = nn.Conv2d(
                in_channels=make_divisible(config.backbone_config["blocks6"][-1][2] * config.scale),
                out_channels=config.class_expand,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            self.act = _create_act(config.act)
        else:
            self.last_conv = None
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

        if config.use_last_conv:
            fc_in_channels = config.class_expand
        else:
            fc_in_channels = make_divisible(config.backbone_config["blocks6"][-1][2] * config.scale)
        self.fc = nn.Linear(fc_in_channels, config.class_num)
        self.out_act = nn.Softmax(dim=-1)

        self.post_init()

    def forward(
        self,
        hidden_state: torch.FloatTensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple[torch.FloatTensor], PPLCNetModelOutput]:
        """
        Forward propagation of PP-LCNet base model.

        Args:
            hidden_state (FloatTensor): Input image tensor with shape [B, 3, H, W].
            output_hidden_states (Optional[bool]): Whether to return hidden states at each layer. Defaults to None.
            return_dict (Optional[bool]): Whether to return output as a dataclass. Defaults to None.

        Returns:
            Union[tuple[FloatTensor], PPLCNetModelOutput]: Model outputs.
        """
        hidden_states = () if output_hidden_states else None

        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)
        hidden_state = self.conv1(hidden_state)
        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)
        hidden_state = self.blocks2(hidden_state)
        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)
        hidden_state = self.blocks3(hidden_state)
        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)
        hidden_state = self.blocks4(hidden_state)
        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)
        hidden_state = self.blocks5(hidden_state)
        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)
        hidden_state = self.blocks6(hidden_state)
        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)
        last_hidden_state = hidden_states

        hidden_state = self.avg_pool(hidden_state)
        if self.last_conv is not None:
            hidden_state = self.last_conv(hidden_state)
            hidden_state = self.act(hidden_state)
            hidden_state = hidden_state * (1 - self.dropout_prob)  # dropout
        hidden_state = self.flatten(hidden_state)
        hidden_state = self.fc(hidden_state)

        hidden_state = self.out_act(hidden_state)

        if not return_dict:
            output = (last_hidden_state,)
            if output_hidden_states:
                output += (hidden_states,)
            output += (hidden_state,)
            return output

        return PPLCNetModelOutput(
            logits=hidden_state,
            last_hidden_state=last_hidden_state,
            hidden_states=hidden_states if output_hidden_states else None,
        )


@dataclass
class PPLCNetForImageClassificationOutput(BaseModelOutputWithNoAttention):
    """
    Output class for PP-LCNet image classification model.

    Args:
        logits (Optional[FloatTensor]): Classification logits with shape [B, num_classes]. Defaults to None.
        last_hidden_state (Optional[FloatTensor]): Final hidden state from backbone with shape [B, C, H, W]. Defaults to None.
        hidden_states (Optional[Tuple[FloatTensor, ...]]): Tuple of hidden states at each layer. Defaults to None.
    """

    logits: Optional[torch.FloatTensor] = None
    shape: Optional[torch.FloatTensor] = None


class PPLCNetForImageClassification(PPLCNetPreTrainedModel):
    """
    PP-LCNet model for image classification tasks.
    Wraps the base model with a classification head (compatible with Transformers pipeline).
    """

    _keys_to_ignore_on_load_missing = ["num_batches_tracked"]

    def __init__(self, config: PPLCNetConfig):
        """
        Initialize the PPLCNetForImageClassification model.

        Args:
            config (PPLCNetConfig): Configuration class for PP-LCNet.
        """
        super().__init__(config)
        self.model = PPLCNetModel(config)
        self.post_init()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[list[dict]] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple[torch.FloatTensor], PPLCNetForImageClassificationOutput]:
        """
        Forward propagation of PP-LCNet image classification model.

        Args:
            pixel_values (FloatTensor): Input image tensor with shape [B, 3, H, W].
            labels (Optional[List[Dict[str, Any]]]): Ground truth labels for loss calculation. Defaults to None.
            output_hidden_states (Optional[bool]): Whether to return hidden states at each layer. Defaults to None.
            return_dict (Optional[bool]): Whether to return output as a dataclass. Defaults to None.

        Returns:
            Union[tuple[FloatTensor], PPLCNetForImageClassificationOutput]: Classification outputs.
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
        return PPLCNetForImageClassificationOutput(
            logits=outputs.logits,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
        )


__all__ = [
    "PPLCNetForImageClassification",
    "PPLCNetImageProcessor",
    "PPLCNetImageProcessorFast",
    "PPLCNetConfig",
    "PPLCNetModel",
    "PPLCNetPreTrainedModel",
]
