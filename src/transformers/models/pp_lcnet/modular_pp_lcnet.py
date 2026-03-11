# Copyright 2026 The PaddlePaddle Team and The HuggingFace Inc. team. All rights reserved.
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

import math
from typing import Optional

import torch
import torch.nn as nn
import torchvision.transforms.v2.functional as tvF

from ...activations import ACT2FN
from ...backbone_utils import BackboneConfigMixin, BackboneMixin
from ...configuration_utils import PreTrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...image_processing_utils_fast import BaseImageProcessorFast
from ...image_utils import (
    SizeDict,
)
from ...modeling_outputs import (
    BackboneOutput,
    BaseModelOutputWithNoAttention,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    auto_docstring,
    can_return_tuple,
)
from ...utils.generic import TensorType
from ...utils.output_capturing import capture_outputs
from ..mobilenet_v2.modeling_mobilenet_v2 import make_divisible
from ..resnet.modeling_resnet import ResNetConvLayer


@auto_docstring(checkpoint="PaddlePaddle/PP-LCNet_x1_0_doc_ori_safetensors")
class PPLCNetConfig(BackboneConfigMixin, PreTrainedConfig):
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
        hidden_act (`str`, *optional*, defaults to `"hardswish"`):
            The non-linear activation function used in the model's hidden layers. Supported functions include
            `"hardswish"`, `"relu"`, `"silu"`, and `"gelu"`. `"hardswish"` is preferred for lightweight and efficient
            inference on edge devices.
        out_features (`list[str]`, *optional*):
            If used as backbone, list of features to output. Can be any of `"stem"`, `"stage1"`, `"stage2"`, etc.
            (depending on how many stages the model has). If unset and `out_indices` is set, will default to the
            corresponding stages. If unset and `out_indices` is unset, will default to the last stage. Must be in the
            same order as defined in the `stage_names` attribute.
        out_indices (`list[int]`, *optional*):
            If used as backbone, list of indices of features to output. Can be any of 0, 1, 2, etc. (depending on how
            many stages the model has). If unset and `out_features` is set, will default to the corresponding stages.
            If unset and `out_features` is unset, will default to the last stage. Must be in the
            same order as defined in the `stage_names` attribute.
        stem_channels (`int`, *optional*, defaults to 16):
            The number of output channels for the stem layer.
        stem_stride (`int`, *optional*, defaults to 2):
            The stride for the stem convolution layer.
        block_configs (`list[list[tuple]]`, *optional*, defaults to `None`):
            Configuration for each block in each stage. Each tuple contains:
            (kernel_size, in_channels, out_channels, stride, use_squeeze_excitation).
            If `None`, uses the default PP-LCNet configuration.
        reduction (`int`, *optional*, defaults to 4):
            The reduction factor for feature channel dimensions in the squeeze-and-excitation (SE) blocks, used to
            reduce the number of model parameters and computational complexity while maintaining feature representability.
        dropout_prob (`float`, *optional*, defaults to 0.2):
            The dropout probability for the classification head, used to prevent overfitting by randomly zeroing out
            a fraction of the neurons during training.
        class_expand (`int`, *optional*, defaults to 1280):
            The number of hidden units in the expansion layer of the classification head, used to enhance the model's
            feature representation capability before the final classification layer.
        use_last_convolution (`bool`, *optional*, defaults to `True`):
            Whether to use the final convolutional layer in the classification head. Setting this to `True` helps
            extract more discriminative features for the classification task.
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
        hidden_act="hardswish",
        out_features=None,
        out_indices=None,
        stem_channels=16,
        stem_stride=2,
        block_configs=None,
        reduction=4,
        dropout_prob=0.2,
        class_expand=1280,
        use_last_convolution=True,
        divisor=8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scale = scale
        self.hidden_act = hidden_act
        self.stem_channels = stem_channels
        self.stem_stride = stem_stride
        self.reduction = reduction
        self.dropout_prob = dropout_prob
        self.class_expand = class_expand
        self.use_last_convolution = use_last_convolution
        self.divisor = divisor

        # Default block configs for PP-LCNet
        # Each tuple: (kernel_size, in_channels, out_channels, stride, use_squeeze_excitation)
        self.block_configs = (
            [
                # Stage 1 (blocks2)
                [[3, 16, 32, 1, False]],
                # Stage 2 (blocks3)
                [[3, 32, 64, 2, False], [3, 64, 64, 1, False]],
                # Stage 3 (blocks4)
                [[3, 64, 128, 2, False], [3, 128, 128, 1, False]],
                # Stage 4 (blocks5)
                [
                    [3, 128, 256, 2, False],
                    [5, 256, 256, 1, False],
                    [5, 256, 256, 1, False],
                    [5, 256, 256, 1, False],
                    [5, 256, 256, 1, False],
                    [5, 256, 256, 1, False],
                ],
                # Stage 5 (blocks6)
                [[5, 256, 512, 2, True], [5, 512, 512, 1, True]],
            ]
            if block_configs is None
            else block_configs
        )

        self.depths = [len(blocks) for blocks in self.block_configs]
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(self.block_configs) + 1)]
        self.set_output_features_output_indices(out_indices=out_indices, out_features=out_features)


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
        target_short_edge: int | None,
        size_divisor: int | None = None,
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


class PPLCNetConvLayer(ResNetConvLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        activation: str = "hardswish",
        groups: int = 1,
    ):
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
            groups=groups,
        )


class PPLCNetDepthwiseSeparableConvLayer(nn.Module):
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
        kernel_size,
        use_squeeze_excitation,
        config,
    ):
        super().__init__()
        self.depthwise_convolution = PPLCNetConvLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
            activation=config.hidden_act,
        )
        self.squeeze_excitation_module = (
            PPLCNetSEModule(in_channels, config.reduction) if use_squeeze_excitation else nn.Identity()
        )
        self.pointwise_convolution = PPLCNetConvLayer(
            in_channels=in_channels,
            kernel_size=1,
            out_channels=out_channels,
            stride=1,
            activation=config.hidden_act,
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
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.convolution1 = nn.Conv2d(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.convolution2 = nn.Conv2d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
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
        hidden_state = self.activation1(self.convolution1(hidden_state))
        hidden_state = self.activation2(self.convolution2(hidden_state))
        hidden_state = identity * hidden_state

        return hidden_state


class PPLCNetBlock(nn.Module):
    def __init__(self, config, stage_index):
        super().__init__()
        self.config = config

        blocks = config.block_configs[stage_index]

        self.layers = nn.ModuleList()
        for kernel_size, in_channels, out_channels, stride, use_se in blocks:
            scaled_in_channels = make_divisible(in_channels * config.scale, config.divisor)
            scaled_out_channels = make_divisible(out_channels * config.scale, config.divisor)

            depthwise_block = PPLCNetDepthwiseSeparableConvLayer(
                in_channels=scaled_in_channels,
                out_channels=scaled_out_channels,
                kernel_size=kernel_size,
                stride=stride,
                use_squeeze_excitation=use_se,
                config=config,
            )
            self.layers.append(depthwise_block)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


@auto_docstring
class PPLCNetPreTrainedModel(PreTrainedModel):
    """
    An abstract base class for PP-LCNet models that inherits from Hugging Face PreTrainedModel.
    Provides common functionality for weight initialization and loading.
    """

    config: PPLCNetConfig
    base_model_prefix = "pp_lcnet"
    main_input_name = "pixel_values"
    input_modalities = ("image",)
    _can_record_outputs = {
        "hidden_states": PPLCNetBlock,
    }


class PPLCNetEmbeddings(nn.Module):
    """
    PPLCNet Embeddings (Stem): Initial convolutional layer for processing input images.
    """

    def __init__(self, config: PPLCNetConfig):
        super().__init__()
        self.convolution = PPLCNetConvLayer(
            in_channels=3,
            kernel_size=3,
            out_channels=make_divisible(config.stem_channels * config.scale, config.divisor),
            stride=config.stem_stride,
            activation=config.hidden_act,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        embedding = self.convolution(pixel_values)

        return embedding


class PPLCNetEncoder(PPLCNetPreTrainedModel):
    def __init__(self, config: PPLCNetConfig):
        super().__init__(config)
        self.config = config
        self.blocks = nn.ModuleList([])
        for stage_index in range(len(config.block_configs)):
            block = PPLCNetBlock(config, stage_index)
            self.blocks.append(block)

        self.post_init()

    @capture_outputs
    def forward(self, hidden_state: torch.Tensor, **kwargs) -> BaseModelOutputWithNoAttention:
        for block in self.blocks:
            hidden_state = block(hidden_state)

        return BaseModelOutputWithNoAttention(last_hidden_state=hidden_state)


@auto_docstring(
    custom_intro="""
    PPLCNet backbone model for feature extraction.
    """
)
class PPLCNetBackbone(BackboneMixin, PPLCNetPreTrainedModel):
    has_attentions = False

    def __init__(self, config: PPLCNetConfig):
        super().__init__(config)
        num_features = [config.stem_channels]
        for block in config.block_configs:
            num_features.append(int(block[-1][2] * config.scale))
        self.num_features = num_features
        self.embedder = PPLCNetEmbeddings(config)
        self.encoder = PPLCNetEncoder(config)

        self.post_init()

    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor,
        **kwargs,
    ) -> BackboneOutput:
        r"""
        Examples:

        ```python
        >>> from transformers import PPLCNetConfig, PPLCNetBackbone
        >>> import torch

        >>> config = PPLCNetConfig()
        >>> model = PPLCNetBackbone(config)

        >>> pixel_values = torch.randn(1, 3, 224, 224)

        >>> with torch.no_grad():
        ...     outputs = model(pixel_values)

        >>> feature_maps = outputs.feature_maps
        >>> list(feature_maps[-1].shape)
        ```"""
        embedding_output = self.embedder(pixel_values)
        hidden_states = self.encoder(embedding_output, output_hidden_states=True).hidden_states

        feature_maps = ()
        for idx, stage in enumerate(self.stage_names):
            if stage in self.out_features:
                feature_maps += (hidden_states[idx],)

        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=hidden_states if kwargs.get("output_hidden_states", False) else None,
        )


@auto_docstring(
    custom_intro="""
    """
)
class PPLCNetForImageClassification(PPLCNetPreTrainedModel):
    _keys_to_ignore_on_load_missing = ["num_batches_tracked"]

    def __init__(self, config: PPLCNetConfig):
        super().__init__(config)
        self.embedder = PPLCNetEmbeddings(config)
        self.encoder = PPLCNetEncoder(config)

        self.config = config
        self.num_labels = config.num_labels
        last_block_out_channels = config.block_configs[-1][-1][2]
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if config.use_last_convolution:
            self.last_convolution = nn.Conv2d(
                in_channels=make_divisible(last_block_out_channels * config.scale, config.divisor),
                out_channels=config.class_expand,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            self.activation = ACT2FN[config.hidden_act]
            self.dropout_prob = config.dropout_prob
            fc_in_channels = config.class_expand
        else:
            self.last_convolution = None
            fc_in_channels = make_divisible(last_block_out_channels * config.scale, config.divisor)

        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.fc = nn.Linear(fc_in_channels, config.num_labels) if config.num_labels > 0 else nn.Identity()

        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithNoAttention:
        r"""
        Examples:

        ```python
        >>> import requests
        >>> from PIL import Image
        >>> from transformers import AutoModelForImageClassification, AutoImageProcessor

        >>> model_path = "PaddlePaddle/PP-LCNet_x1_0_table_cls_safetensors"
        >>> model = AutoModelForImageClassification.from_pretrained(model_path)
        >>> image_processor = AutoImageProcessor.from_pretrained(model_path)

        >>> url = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/img_rot180_demo.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> predicted_label = outputs.last_hidden_state.argmax(-1).item()
        >>> print(model.config.id2label[predicted_label])
        wireless_table
        ```"""
        embedding_output = self.embedder(pixel_values)
        outputs = self.encoder(embedding_output, **kwargs)

        last_hidden_state = self.avg_pool(outputs.last_hidden_state)

        if self.config.use_last_convolution:
            last_hidden_state = self.last_convolution(last_hidden_state)
            last_hidden_state = self.activation(last_hidden_state)
            last_hidden_state = last_hidden_state * (1 - self.dropout_prob)

        last_hidden_state = self.flatten(last_hidden_state)
        last_hidden_state = self.fc(last_hidden_state)

        return BaseModelOutputWithNoAttention(last_hidden_state=last_hidden_state, hidden_states=outputs.hidden_states)


__all__ = [
    "PPLCNetBackbone",
    "PPLCNetForImageClassification",
    "PPLCNetImageProcessorFast",
    "PPLCNetConfig",
    "PPLCNetPreTrainedModel",
]
