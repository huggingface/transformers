from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...activations import ACT2FN
from ...configuration_utils import PreTrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...image_processing_utils import BaseImageProcessor
from ...image_processing_utils_fast import BaseImageProcessorFast
from ...image_transforms import flip_channel_order, to_channel_dimension_format
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    infer_channel_dimension_format,
    make_flat_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from ...modeling_outputs import BaseModelOutputWithNoAttention
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput, 
    filter_out_non_signature_kwargs, 
    auto_docstring, 
    can_return_tuple,
)
from ...utils.output_capturing import capture_outputs
from ...utils.generic import TensorType


@auto_docstring(
    custom_intro="""
    """
)
class UVDocConfig(PreTrainedConfig):
    r"""
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

    model_type = "uvdoc"

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


@auto_docstring(
    custom_intro="""
    """
)
class UVDocImageProcessorFast(BaseImageProcessorFast):
    """
    Fast image processor for UVDoc models (PyTorch-optimized, inherits from `BaseImageProcessorFast`).
    Optimized for speed with torch tensor operations, skipping numpy conversions for low-latency inference.
    """

    image_mean = [0, 0, 0]
    image_std = [1, 1, 1]
    do_rescale = True
    do_normalize = True

    def __init__(self, **kwargs) -> None:
        """Initialize the fast UVDoc image processor (inherits class-level defaults)."""
        super().__init__(**kwargs)

    def _preprocess(
        self,
        images: list[torch.Tensor],
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs,
    ) -> BatchFeature:
        """
        Fast preprocessing for UVDoc model (pure PyTorch tensor operations).
        Optimized for GPU inference with minimal data conversion overhead.

        Args:
            images (`List[torch.Tensor]`):
                List of input images (PyTorch tensors, [C, H, W] format).
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

        return [self.doctr(image, scale) for image in images]

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
            image = pred[0]
        else:
            image = pred

        assert isinstance(image, torch.Tensor), "Invalid input 'image' in DocTrPostProcess. Expected a torch tensor."

        image = image.squeeze()
        image = image.permute(1, 2, 0)
        image = image * scale
        image = image.flip(dims=[-1])
        image = image.to(dtype=torch.uint8, non_blocking=True, copy=False)

        return image


class UVDocResidualBlockWithDilation(nn.Module):
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
        downsample: Optional[bool] = None,
        is_top: bool = False,
    ) -> None:
        """
        Initialize residual block with dilation support.

        Args:
            in_channels (`int`): Number of input channels
            out_channels (`int`): Number of output channels
            kernel_size (`int`): Size of convolutional kernel
            stride (`int`, *optional*, defaults to 1): Stride of the first convolution
            downsample (`bool`, *optional*, defaults to None):
                Downsampling layer for residual connection (when stride != 1 or channel mismatch)
            is_top (`bool`, *optional*, defaults to False):
                Whether this is the first block in the layer (uses standard conv instead of dilated conv)
        """
        super().__init__()

        self.downsample = downsample
        if downsample:
            self.downsample_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            )
            self.downsample_bn = nn.BatchNorm2d(out_channels)

        if stride != 1 or is_top:
            stride1, padding, dilation = stride, kernel_size // 2, 1
        else:
            stride1, padding, dilation = 1, 3 * (kernel_size // 2), 3

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride1, padding, dilation=dilation)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding, dilation=dilation)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        identity = hidden_state
        if self.downsample:
            identity = self.downsample_conv(hidden_state)
            identity = self.downsample_bn(identity)

        hidden_state = self.conv1(hidden_state)
        hidden_state = self.bn1(hidden_state)
        hidden_state = self.relu(hidden_state)
        hidden_state = self.conv2(hidden_state)
        hidden_state = self.bn2(hidden_state)
        hidden_state += identity
        hidden_state = self.relu(hidden_state)
        return hidden_state


class UVDocResNetStraight(nn.Module):
    """
    Modified ResNet backbone for UVDoc with dilated residual blocks.
    Extracts multi-scale features from document images through progressive downsampling,
    using dilated convolution to maintain large receptive fields.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.in_channels = config.num_filter * config.map_num[0]
        self.relu = nn.ReLU()

        self.layers = nn.ModuleList([])
        for map_num, block_num, stride in zip(config.map_num[:3], config.block_nums[:3], config.stride[:3]):
            layers = nn.ModuleList([])
            out_channels = config.num_filter * map_num

            downsample = None
            if stride != 1 or self.in_channels != out_channels:
                downsample = True

            for i in range(block_num):
                layer = UVDocResidualBlockWithDilation(
                    in_channels=self.in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=config.kernel_size,
                    stride=stride if i == 0 else 1,
                    downsample=downsample if i == 0 else None,
                    is_top=i == 0,
                )
                layers.append(layer)
            self.layers.append(layers)
            self.in_channels = out_channels

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        for layers in self.layers:
            for layer in layers:
                hidden_state = layer(hidden_state)

        return hidden_state


class UVDocResNetHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_channels = config.in_channels
        num_filter = config.num_filter
        map_num_0 = config.map_num[0]
        kernel_size = config.kernel_size
        out_channels = num_filter * map_num_0

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=False,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            bias=False,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, hidden_state):
        hidden_state = self.conv1(hidden_state)
        hidden_state = self.bn1(hidden_state)
        hidden_state = self.relu1(hidden_state)

        hidden_state = self.conv2(hidden_state)
        hidden_state = self.bn2(hidden_state)
        hidden_state = self.relu2(hidden_state)
        return hidden_state


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
            Tuple of hidden states from each bridge layer
    """

    logits: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None


@auto_docstring(
    custom_intro="""
    """
)
class UVDocPreTrainedModel(PreTrainedModel):
    """
    Base class for all UVDoc pre-trained models.
    Inherits from Hugging Face PreTrainedModel and sets UVDoc-specific configurations.
    """

    config: UVDocConfig
    base_model_prefix = "uvdoc"
    main_input_name = "pixel_values"
    input_modalities = ("image",)


class UVDocConvLayer(nn.Module):
    """
    Convolutional layer used in UVDoc model.
    Consists of a convolutional operation followed by batch normalization and ReLU activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 3,
        activation: str = "relu",
    ):
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels,
            out_channels,
            bias=False,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.normalization = nn.BatchNorm2d(out_channels)
        self.activation = ACT2FN[activation] if activation is not None else nn.Identity()

    def forward(self, hidden_state):
        hidden_state = self.convolution(hidden_state)
        hidden_state = self.normalization(hidden_state)
        hidden_state = self.activation(hidden_state)
        return hidden_state


class UVDocBridgeBlock(nn.Module):
    def __init__(self, config, block_index):
        super().__init__()
        dilation_values = config.dilation_values[block_index]
        in_channels = config.num_filter * config.map_num[2]

        self.blocks = nn.ModuleList([])

        if isinstance(dilation_values, int):
            self.blocks.append(
                UVDocConvLayer(in_channels, in_channels, padding=dilation_values, dilation=dilation_values)
            )
        else:
            for dilation in dilation_values:
                self.blocks.append(UVDocConvLayer(in_channels, in_channels, padding=dilation, dilation=dilation))

    def forward(self, hidden_state):
        for block in self.blocks:
            hidden_state = block(hidden_state)
        return hidden_state


class UVDocPointPositions2D(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.convolution1 = nn.Conv2d(
            config.num_filter * config.map_num[2],
            config.num_filter * config.map_num[0],
            bias=False,
            kernel_size=config.kernel_size,
            stride=1,
            padding=config.kernel_size // 2,
            padding_mode=config.padding_mode,
        )
        self.normalization1 = nn.BatchNorm2d(config.num_filter * config.map_num[0])
        self.prelu = nn.PReLU()
        self.convolution2 = nn.Conv2d(
            config.num_filter * config.map_num[0],
            2,
            kernel_size=config.kernel_size,
            stride=1,
            padding=config.kernel_size // 2,
            padding_mode=config.padding_mode,
        )

    def forward(self, hidden_state):
        hidden_state = self.convolution1(hidden_state)
        hidden_state = self.normalization1(hidden_state)
        hidden_state = self.prelu(hidden_state)
        hidden_state = self.convolution2(hidden_state)
        return hidden_state


@auto_docstring(
    custom_intro="""
    """
)
class UVDocModel(UVDocPreTrainedModel):
    """
    Core UVDoc model for document rectification.
    Combines ResNet backbone, multi-scale bridge layers, and spatial transformation
    to correct perspective distortion in document images.
    """

    def __init__(self, config: UVDocConfig) -> None:
        super().__init__(config)

        self.upsample_size = config.upsample_size
        self.upsample_mode = config.upsample_mode

        self.resnet_head = UVDocResNetHead(config)
        self.resnet_down = UVDocResNetStraight(config)

        self.bridge = nn.ModuleList([])
        for block_index in config.dilation_values.keys():
            self.bridge.append(UVDocBridgeBlock(config, block_index))

        self.num_bridge_layers = len(self.bridge)

        self.bridge_concat = UVDocConvLayer(
            in_channels=config.num_filter * config.map_num[2] * self.num_bridge_layers,
            out_channels=config.num_filter * config.map_num[2],
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
        )

        self.out_point_positions2D = UVDocPointPositions2D(config)

        self.post_init()

    @capture_outputs
    @can_return_tuple
    def forward(
        self,
        hidden_state: torch.FloatTensor,
        **kwargs: Any,
    ) -> Union[tuple[torch.FloatTensor, ...], UVDocModelOutput]:
        """
        Forward pass of UVDoc core model for document rectification.

        Args:
            hidden_state (`torch.FloatTensor`): Input image tensor of shape [B, C, H, W]

        """

        identity = hidden_state
        original_height, original_width = hidden_state.shape[2:]
        hidden_state = F.interpolate(
            hidden_state,
            size=(self.upsample_size[0], self.upsample_size[1]),
            mode=self.upsample_mode,
            align_corners=True,
        )
        hidden_state = self.resnet_head(hidden_state)
        resnet_down = self.resnet_down(hidden_state)

        bridge_outputs = []
        for bridge_layer in self.bridge:
            bridge_out = bridge_layer(resnet_down)
            bridge_outputs.append(bridge_out)

        last_hidden_state = bridge_outputs[-1] if bridge_outputs else None

        bridge_concat = torch.cat(bridge_outputs, dim=1)
        bridge = self.bridge_concat(bridge_concat)

        out_point_positions2D = self.out_point_positions2D(bridge)

        upsampled_2d_bezier_mesh = F.interpolate(
            out_point_positions2D,
            size=(original_height, original_width),
            mode=self.upsample_mode,
            align_corners=True,
        )

        rearranged_bezier_mesh = upsampled_2d_bezier_mesh.permute(0, 2, 3, 1)
        rectified_image_output = F.grid_sample(identity, rearranged_bezier_mesh, align_corners=True)


        return UVDocModelOutput(
            logits=rectified_image_output,
            last_hidden_state=last_hidden_state,
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


@auto_docstring(
    custom_intro="""
    """
)
class UVDocForDocumentRectification(UVDocPreTrainedModel):
    """
    Wrapper class for UVDoc model focused on document rectification task.
    Provides a user-friendly interface for inference/training with standard Hugging Face API.
    """

    _keys_to_ignore_on_load_missing = ["num_batches_tracked"]

    def __init__(self, config: UVDocConfig) -> None:
        super().__init__(config)
        self.model = UVDocModel(config)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        **kwargs: Any,
    ) -> Union[tuple[torch.FloatTensor, ...], UVDocForDocumentRectificationOutput]:
        """
        Forward pass of UVDoc document rectification model.

        Args:
            pixel_values (`torch.FloatTensor`): Input image tensor of shape [B, C, H, W] (preprocessed)

        """

        outputs = self.model(pixel_values)

        return UVDocForDocumentRectificationOutput(
            logits=outputs.logits,
            last_hidden_state=outputs.last_hidden_state,
        )


__all__ = [
    "UVDocForDocumentRectification",
    "UVDocImageProcessorFast",
    "UVDocConfig",
    "UVDocModel",
    "UVDocPreTrainedModel",
]
