import math
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2.functional as tvF

from transformers.models.mobilenet_v2.modeling_mobilenet_v2 import make_divisible

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
from ...utils import (
    ModelOutput,
    auto_docstring,
    can_return_tuple,
    filter_out_non_signature_kwargs,
)
from ...utils.output_capturing import capture_outputs
from ...utils.generic import TensorType


@auto_docstring(
    custom_intro="""
    """
)
class PPLCNetConfig(PreTrainedConfig):
    r"""
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
        hidden_act (`str`, *optional*, defaults to `"hardswish"`):
            The non-linear activation function used in the model's hidden layers. Supported functions include
            `"hardswish"`, `"relu"`, `"silu"`, and `"gelu"`. `"hardswish"` is preferred for lightweight and efficient
            inference on edge devices.
        backbone_config (`Union[dict, PreTrainedConfig]`, *optional*, defaults to `None`):
            The configuration of the backbone model. If `None`, the default backbone configuration for PP-LCNet
            will be used, which includes the standard block settings for feature extraction.
        divisor (`int`, *optional*, defaults to 8):
            The divisor used to ensure that various model parameters (e.g., channel dimensions, kernel sizes) are
            multiples of this value, promoting efficient model implementation and resource utilization.

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

    model_type = "pp_lcnet"

    def __init__(
        self,
        scale=1.0,
        stride_list=[2, 2, 2, 2, 2],
        reduction=4,
        dropout_prob=0.2,
        class_expand=1280,
        use_last_convolution=True,
        hidden_act="hardswish",
        backbone_config=None,
        divisor=8,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.scale = scale
        self.stride_list = stride_list
        self.reduction = reduction
        self.dropout_prob = dropout_prob
        self.class_expand = class_expand
        self.use_last_convolution = use_last_convolution
        self.hidden_act = hidden_act
        self.backbone_config = backbone_config
        self.divisor = divisor


@auto_docstring(
    custom_intro="""
    """
)
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
        super().__init__(**kwargs)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        return_tensors: str | TensorType | None,
        interpolation: Optional["tvF.InterpolationMode"],
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
        resize_images = []
        if do_resize:
            for image in images:
                if self.resize_short is not None:
                    size = self.get_image_size(
                        image, target_short_edge=self.resize_short, size_divisor=self.size_divisor
                    )

                image = self.resize(image, size=size, interpolation=interpolation)
                resize_images.append(image)
            images = resize_images

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
        image: torch.Tensor,
        target_short_edge: Union[int, None],
        size_divisor: Optional[int] = None,
    ) -> tuple[SizeDict, torch.Tensor]:
        """
        Calculate target image size for PyTorch tensors (preserve aspect ratio + align with size_divisor).

        Args:
            image (torch.Tensor): Input PyTorch tensor (shape [C, H, W]).
            target_short_edge (Union[int, None]): Desired length for the shorter edge of the image.
            size_divisor (Optional[int]): Divisor to align image dimensions (for hardware optimization). Defaults to None.

        Returns:
            SizeDict: Target size ({"height": resized_height, "width": resized_width}) with preserved aspect ratio.
        """
        _, h, w = image.shape
        resize_scale = target_short_edge / min(h, w)
        resized_height = round(h * resize_scale)
        resized_width = round(w * resize_scale)
        if self.size_divisor is not None:
            resized_height = math.ceil(resized_height / size_divisor) * size_divisor
            resized_width = math.ceil(resized_width / size_divisor) * size_divisor

        return SizeDict(height=resized_height, width=resized_width)


def _create_act(hidden_act):
    if hidden_act == "hardswish":
        return nn.Hardswish()
    elif hidden_act == "relu":
        return nn.ReLU()
    elif hidden_act == "relu6":
        return nn.ReLU6()
    else:
        raise RuntimeError(f"The activation function is not supported: {hidden_act}")


class PPLCNetConvBNLayer(nn.Module):
    """
    Combined layer: Conv2d -> BatchNorm2d -> Activation
    A common basic component in lightweight models to reduce redundant code.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        groups=1,
        hidden_act="hardswish",
    ):
        """
        Initialize the PPLCNetConvBNLayer module.

        Args:
            in_channels (int): Number of channels of the input feature map.
            kernel_size (int): Size of the convolutional kernel (square kernel).
            out_channels (int): Number of channels of the output feature map.
            stride (int): Stride of the convolution operation.
            groups (int, optional): Number of groups for grouped convolution. Defaults to 1 (standard convolution).
            hidden_act (str, optional): Name of activation function. Defaults to "hardswish".
        """
        super().__init__()

        self.convolution = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            bias=False,
        )

        self.normalization = nn.BatchNorm2d(out_channels)
        self.activation = _create_act(hidden_act)

    def forward(self, hidden_state):
        """
        Forward propagation logic.

        Args:
            hidden_state (FloatTensor): Input feature map with shape [B, C, H, W].

        Returns:
            FloatTensor: Output feature map with shape [B, out_channels, H', W'].
        """
        hidden_state = self.convolution(hidden_state)
        hidden_state = self.normalization(hidden_state)
        hidden_state = self.activation(hidden_state)
        return hidden_state


class PPLCNetDepthwiseSeparable(nn.Module):
    """
    Depthwise Separable Convolution Layer: Depthwise Conv -> SE Module (optional) -> Pointwise Conv
    Core component of lightweight models (e.g., MobileNet, PP-LCNet) that significantly reduces
    the number of parameters and computational cost.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        reduction: int,
        kernel_size=3,
        use_squeeze_excitation=False,
        hidden_act="hardswish",
    ):
        """
        Initialize the PPLCNetDepthwiseSeparable module.

        Args:
            in_channels (int): Number of channels of the input feature map.
            out_channels (int): Number of channels of the output feature map.
            stride (int): Stride of the depthwise convolution.
            reduction (int): Reduction ratio for SE module.
            depthwise_size (int, optional): Kernel size of depthwise convolution. Defaults to 3.
            use_squeeze_excitation (bool, optional): Whether to use SE module. Defaults to False.
            hidden_act (str, optional): Name of activation function. Defaults to "hardswish".
        """
        super().__init__()
        self.use_squeeze_excitation = use_squeeze_excitation
        self.depthwise_convolution = PPLCNetConvBNLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
            hidden_act=hidden_act,
        )
        self.squeeze_excitation_module = (
            PPLCNetSEModule(in_channels, reduction) if use_squeeze_excitation else nn.Identity()
        )
        self.pointwise_convolution = PPLCNetConvBNLayer(
            in_channels=in_channels,
            kernel_size=1,
            out_channels=out_channels,
            stride=1,
            hidden_act=hidden_act,
        )

    def forward(self, hidden_state):
        """
        Forward propagation logic.

        Args:
            hidden_state (FloatTensor): Input feature map with shape [B, C, H, W].

        Returns:
            FloatTensor: Output feature map with shape [B, out_channels, H', W'].
        """
        hidden_state = self.depthwise_convolution(hidden_state)
        hidden_state = self.squeeze_excitation_module(hidden_state)
        hidden_state = self.pointwise_convolution(hidden_state)
        return hidden_state


class PPLCNetSEModule(nn.Module):
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
        self.convolution1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, **conv_kwargs)
        self.convolution2 = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, **conv_kwargs)
        self.activation1 = nn.ReLU()

        self.activation2 = nn.Hardsigmoid()

    def forward(self, hidden_state):
        """
        Forward propagation logic.

        Args:
            hidden_state (FloatTensor): Input feature map with shape [B, C, H, W].

        Returns:
            FloatTensor: Attention-weighted feature map with shape [B, C, H, W].
        """
        identity = hidden_state
        hidden_state = self.avg_pool(hidden_state)
        hidden_state = self.convolution1(hidden_state)
        hidden_state = self.activation1(hidden_state)
        hidden_state = self.convolution2(hidden_state)
        hidden_state = self.activation2(hidden_state)
        hidden_state = identity * hidden_state
        return hidden_state

class PPLCNetBlock(nn.Module):
    def __init__(self, config, block_index):
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList()
        for kernel_size, in_channels, out_channels, stride, squeeze_excitation in config.backbone_config[block_index]:
            scaled_in_channels = make_divisible(in_channels * config.scale, config.divisor)
            scaled_out_channels = make_divisible(out_channels * config.scale, config.divisor)
            
            depthwise_block = PPLCNetDepthwiseSeparable(
                in_channels=scaled_in_channels,
                out_channels=scaled_out_channels,
                kernel_size=kernel_size,
                stride=stride,
                reduction=config.reduction,
                use_squeeze_excitation=squeeze_excitation,
                hidden_act=config.hidden_act,
            )
            self.layers.append(depthwise_block)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
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


@auto_docstring(
    custom_intro="""
    """
)
class PPLCNetPreTrainedModel(PreTrainedModel):
    """
    An abstract base class for PP-LCNet models that inherits from Hugging Face PreTrainedModel.
    Provides common functionality for weight initialization and loading.
    """

    config: PPLCNetConfig
    base_model_prefix = "pp_lcnet"
    main_input_name = "pixel_values"
    input_modalities = ("image",)

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights"""
        super()._init_weights(module)
        if isinstance(module, PPLCNetConvBNLayer):
            nn.init.kaiming_normal_(module.convolution.weight)


@auto_docstring(
    custom_intro="""
    """
)
class PPLCNetModel(PPLCNetPreTrainedModel):
    """
    PP-LCNet base model: lightweight CNN backbone for image classification tasks.
    """

    def __init__(self, config: PPLCNetConfig):
        super().__init__(config)

        if config.stride_list is None:
            raise ValueError("stride_list cannot be None, please check your config.")

        if len(config.stride_list) != 5:
            raise ValueError(
                f"stride_list length should be 5 but got {len(config.stride_list)}, please check your config."
            )

        self.dropout_prob = config.dropout_prob

        for i, stride in enumerate(config.stride_list[1:]):
            config.backbone_config[f"blocks{i + 3}"][0][3] = stride
        self.convolution = PPLCNetConvBNLayer(
            in_channels=3,
            kernel_size=3,
            out_channels=make_divisible(16 * config.scale, config.divisor),
            stride=config.stride_list[0],
            hidden_act=config.hidden_act,
        )

        self.blocks = nn.ModuleList([])
        for block_index in config.backbone_config.keys():
            block = PPLCNetBlock(config, block_index)
            self.blocks.append(block)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if config.use_last_convolution:
            self.last_convolution = nn.Conv2d(
                in_channels=make_divisible(config.backbone_config["blocks6"][-1][2] * config.scale, config.divisor),
                out_channels=config.class_expand,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            self.activation = _create_act(config.hidden_act)
        else:
            self.last_convolution = None
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

        if config.use_last_convolution:
            fc_in_channels = config.class_expand
        else:
            fc_in_channels = make_divisible(config.backbone_config["blocks6"][-1][2] * config.scale, config.divisor)
        self.fc = nn.Linear(fc_in_channels, config.num_labels)
        self.out_activation = nn.Softmax(dim=-1)

        self.post_init()

    @capture_outputs
    @can_return_tuple
    def forward(
        self,
        hidden_state: torch.FloatTensor,
        **kwargs,
    ) -> Union[tuple[torch.FloatTensor], PPLCNetModelOutput]:
        """
        Forward propagation of PP-LCNet base model.

        Args:
            hidden_state (FloatTensor): Input image tensor with shape [B, 3, H, W].

        Returns:
            Union[tuple[FloatTensor], PPLCNetModelOutput]: Model outputs.
        """
        hidden_state = self.convolution(hidden_state)

        for block in self.blocks:
            hidden_state = block(hidden_state)

        last_hidden_state = hidden_state

        hidden_state = self.avg_pool(hidden_state)
        if self.last_convolution is not None:
            hidden_state = self.last_convolution(hidden_state)
            hidden_state = self.activation(hidden_state)
            hidden_state = hidden_state * (1 - self.dropout_prob)  # dropout
        hidden_state = self.flatten(hidden_state)
        hidden_state = self.fc(hidden_state)

        hidden_state = self.out_activation(hidden_state)

        return PPLCNetModelOutput(
            logits=hidden_state,
            last_hidden_state=last_hidden_state,
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


@auto_docstring(
    custom_intro="""
    """
)
class PPLCNetForImageClassification(PPLCNetPreTrainedModel):
    """
    PP-LCNet model for image classification tasks.
    Wraps the base model with a classification head (compatible with Transformers pipeline).
    """

    _keys_to_ignore_on_load_missing = ["num_batches_tracked"]

    def __init__(self, config: PPLCNetConfig):
        super().__init__(config)
        self.model = PPLCNetModel(config)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        **kwargs,
    ) -> Union[tuple[torch.FloatTensor], PPLCNetForImageClassificationOutput]:
        """
        Forward propagation of PP-LCNet image classification model.

        Args:
            pixel_values (FloatTensor): Input image tensor with shape [B, 3, H, W].
            labels (Optional[List[Dict[str, Any]]]): Ground truth labels for loss calculation. Defaults to None.

        Returns:
            Union[tuple[FloatTensor], PPLCNetForImageClassificationOutput]: Classification outputs.
        """

        outputs = self.model(pixel_values)

        return PPLCNetForImageClassificationOutput(
            logits=outputs.logits,
            last_hidden_state=outputs.last_hidden_state,
        )


__all__ = [
    "PPLCNetForImageClassification",
    "PPLCNetImageProcessorFast",
    "PPLCNetConfig",
    "PPLCNetModel",
    "PPLCNetPreTrainedModel",
]
