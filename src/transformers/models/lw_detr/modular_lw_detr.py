# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...configuration_utils import PreTrainedConfig
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BackboneOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...pytorch_utils import meshgrid
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, logging
from ...utils.generic import check_model_inputs
from ..auto.configuration_auto import AutoConfig
from ..convnext.modeling_convnext import ConvNextLayerNorm
from ..dab_detr.modeling_dab_detr import gen_sine_position_embeddings
from ..deformable_detr.modeling_deformable_detr import (
    DeformableDetrDecoderOutput,
    DeformableDetrForObjectDetection,
    DeformableDetrMLPPredictionHead,
    DeformableDetrModel,
    DeformableDetrMultiscaleDeformableAttention,
)
from ..llama.modeling_llama import eager_attention_forward
from ..rt_detr.configuration_rt_detr import CONFIG_MAPPING
from ..rt_detr.modeling_rt_detr import RTDetrConvNormLayer
from ..vit.modeling_vit import ViTAttention, ViTEncoder, ViTSelfAttention
from ..vitdet.configuration_vitdet import VitDetConfig
from ..vitdet.modeling_vitdet import (
    VitDetBackbone,
    VitDetEmbeddings,
    VitDetMlp,
    VitDetPreTrainedModel,
)


logger = logging.get_logger(__name__)


class LwDetrViTConfig(VitDetConfig):
    r"""
    This is the configuration class to store the configuration of a [`LwDetrViTModel`]. It is used to instantiate an
    LW-DETR ViT model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the LW-DETR ViT
    [AnnaZhang/lwdetr_small_60e_coco](https://huggingface.co/AnnaZhang/lwdetr_small_60e_coco) architecture.

    LW-DETR ViT is the Vision Transformer backbone used in the LW-DETR model for real-time object detection. It features
    interleaved window and global attention mechanisms to reduce computational complexity while maintaining high performance.
    The model uses a window-major feature map organization for efficient attention computation.

    Configuration objects inherit from [`VitDetConfig`] and can be used to control the model outputs. Read the
    documentation from [`VitDetConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        mlp_ratio (`int`, *optional*, defaults to 4):
            Ratio of mlp hidden dim to embedding dim.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        image_size (`int`, *optional*, defaults to 256):
            The size (resolution) of each image.
        pretrain_image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image during pretraining.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        window_block_indices (`list[int]`, *optional*, defaults to `[]`):
            List of indices of blocks that should have window attention instead of regular global self-attention.
        use_absolute_position_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to add absolute position embeddings to the patch embeddings.
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
        cae_init_values (`float`, *optional*, defaults to 0.1):
            Initialization value for CAE parameters when `use_cae` is enabled.
        num_windows (`int`, *optional*, defaults to 16):
            Number of windows for window-based attention. Must be a perfect square and the image size must be
            divisible by the square root of this value. This enables efficient window-major feature map organization.

    Example:

    ```python
    >>> from transformers import LwDetrViTConfig, LwDetrViTModel

    >>> # Initializing a LW-DETR ViT configuration
    >>> configuration = LwDetrViTConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = LwDetrViTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "lw_detr_vit"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        mlp_ratio=4,
        hidden_act="gelu",
        dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-6,
        image_size=256,
        pretrain_image_size=224,
        patch_size=16,
        num_channels=3,
        qkv_bias=True,
        window_block_indices=[],
        use_absolute_position_embeddings=True,
        out_features=None,
        out_indices=None,
        cae_init_values: float = 0.1,
        num_windows=16,
        **kwargs,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            mlp_ratio=mlp_ratio,
            hidden_act=hidden_act,
            dropout_prob=dropout_prob,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            image_size=image_size,
            pretrain_image_size=pretrain_image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            qkv_bias=qkv_bias,
            window_block_indices=window_block_indices,
            use_absolute_position_embeddings=use_absolute_position_embeddings,
            out_features=out_features,
            out_indices=out_indices,
            **kwargs,
        )
        del self.residual_block_indices
        del self.use_relative_position_embeddings
        del self.window_size
        del self.drop_path_rate

        self.cae_init_values = cae_init_values
        if num_windows % math.sqrt(num_windows) != 0:
            raise ValueError(
                f"`num_windows` has to be a perfect square, where num_windows % math.sqrt(num_windows) != 0, but got {num_windows}."
            )
        if image_size / num_windows % math.sqrt(num_windows) != 0:
            raise ValueError(
                f"`image_size` has to be divisible by `num_windows`, where image_size / num_windows % math.sqrt(num_windows) != 0,but got {image_size} and {num_windows}."
            )
        self.num_windows = num_windows
        self.num_windows_side = int(math.sqrt(num_windows))


class LwDetrConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LwDetrModel`]. It is used to instantiate
    a LW-DETR model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the LW-DETR
    [AnnaZhang/lwdetr_small_60e_coco](https://huggingface.co/AnnaZhang/lwdetr_small_60e_coco) architecture.

    LW-DETR (Lightweight Detection Transformer) is a transformer-based object detection model designed for real-time
    detection tasks. It replaces traditional CNN-based detectors like YOLO with a more efficient transformer architecture
    that achieves competitive performance while being computationally lightweight.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        backbone_config (`PretrainedConfig` or `dict`, *optional*):
            The configuration of the backbone model. If not provided, will default to `LwDetrViTConfig` with
            a small ViT architecture optimized for detection tasks.
        projector_scale_factors (`list[float]`, *optional*, defaults to `[]`):
            Scale factors for the feature pyramid network. Each scale factor determines the resolution of features
            at different levels. Supported values are 0.5, 1.0, and 2.0.
        hidden_expansion (`float`, *optional*, defaults to 0.5):
            Expansion factor for hidden dimensions in the projector layers.
        c2f_num_blocks (`int`, *optional*, defaults to 3):
            Number of blocks in the C2F layer.
        activation_function (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function in the projector. Supported values are `"silu"`, `"relu"`, `"gelu"`.
        batch_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon value for batch normalization layers.
        d_model (`int`, *optional*, defaults to 256):
            Dimension of the model layers and the number of expected features in the decoder inputs.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        decoder_ffn_dim (`int`, *optional*, defaults to 2048):
            Dimension of the "intermediate" (often named feed-forward) layer in decoder.
        decoder_n_points (`int`, *optional*, defaults to 4):
            The number of sampled keys in each feature level for each attention head in the decoder.
        decoder_layers (`int`, *optional*, defaults to 3):
            Number of decoder layers in the transformer.
        decoder_self_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the decoder self-attention.
        decoder_cross_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the decoder cross-attention.
        decoder_activation_function (`str`, *optional*, defaults to `"relu"`):
            The non-linear activation function in the decoder. Supported values are `"relu"`, `"silu"`, `"gelu"`.
        num_queries (`int`, *optional*, defaults to 300):
            Number of object queries, i.e. detection slots. This is the maximal number of objects
            [`LwDetrModel`] can detect in a single image.
        attention_bias (`bool`, *optional*, defaults to `True`):
            Whether to add bias to the attention layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        group_detr (`int`, *optional*, defaults to 13):
            Number of groups for Group DETR attention mechanism, which helps reduce computational complexity.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        disable_custom_kernels (`bool`, *optional*, defaults to `True`):
            Disable the use of custom CUDA and CPU kernels. This option is necessary for the ONNX export, as custom
            kernels are not supported by PyTorch ONNX export.
        class_cost (`float`, *optional*, defaults to 2):
            Relative weight of the classification error in the Hungarian matching cost.
        bbox_cost (`float`, *optional*, defaults to 5):
            Relative weight of the L1 error of the bounding box coordinates in the Hungarian matching cost.
        giou_cost (`float`, *optional*, defaults to 2):
            Relative weight of the generalized IoU loss of the bounding box in the Hungarian matching cost.
        class_loss_coefficient (`float`, *optional*, defaults to 1):
            Relative weight of the Cross Entropy loss in the object detection loss.
        mask_loss_coefficient (`float`, *optional*, defaults to 1):
            Relative weight of the Focal loss in the panoptic segmentation loss.
        dice_loss_coefficient (`float`, *optional*, defaults to 1):
            Relative weight of the DICE/F-1 loss in the panoptic segmentation loss.
        bbox_loss_coefficient (`float`, *optional*, defaults to 5):
            Relative weight of the L1 bounding box loss in the object detection loss.
        giou_loss_coefficient (`float`, *optional*, defaults to 2):
            Relative weight of the generalized IoU loss in the object detection loss.
        eos_coefficient (`float`, *optional*, defaults to 0.1):
            Relative classification weight of the 'no-object' class in the object detection loss.
        focal_alpha (`float`, *optional*, defaults to 0.25):
            Alpha parameter in the focal loss.
        auxiliary_loss (`bool`, *optional*, defaults to `True`):
            Whether auxiliary decoding losses (loss at each decoder layer) are to be used.

    Examples:

    ```python
    >>> from transformers import LwDetrConfig, LwDetrModel

    >>> # Initializing a LW-DETR AnnaZhang/lwdetr_small_60e_coco style configuration
    >>> configuration = LwDetrConfig()

    >>> # Initializing a model (with random weights) from the AnnaZhang/lwdetr_small_60e_coco style configuration
    >>> model = LwDetrModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "lw_detr"
    sub_configs = {"backbone_config": AutoConfig}

    def __init__(
        self,
        # backbone
        backbone_config=None,
        # projector
        projector_scale_factors: list[float] = [],
        hidden_expansion=0.5,
        c2f_num_blocks=3,
        activation_function="silu",
        batch_norm_eps=1e-5,
        # decoder
        d_model=256,
        dropout=0.1,
        decoder_ffn_dim=2048,
        decoder_n_points=4,
        decoder_layers: int = 3,
        decoder_self_attention_heads: int = 8,
        decoder_cross_attention_heads: int = 16,
        decoder_activation_function="relu",
        # model
        num_queries=300,
        attention_bias=True,
        attention_dropout=0.0,
        activation_dropout=0.0,
        group_detr: int = 13,
        init_std=0.02,
        disable_custom_kernels=True,
        # loss
        class_cost=2,
        bbox_cost=5,
        giou_cost=2,
        class_loss_coefficient=1,
        mask_loss_coefficient=1,
        dice_loss_coefficient=1,
        bbox_loss_coefficient=5,
        giou_loss_coefficient=2,
        eos_coefficient=0.1,
        focal_alpha=0.25,
        auxiliary_loss=True,
        **kwargs,
    ):
        self.batch_norm_eps = batch_norm_eps

        # backbone
        if backbone_config is None:
            logger.info(
                "`backbone_config` and `backbone` are `None`. Initializing the config with the default `LwDetrViT` backbone."
            )
            backbone_config = LwDetrViTConfig(
                image_size=1024,
                hidden_size=192,
                num_hidden_layers=10,
                num_attention_heads=12,
                window_block_indices=[0, 1, 3, 6, 7, 9],
                out_indices=[2, 4, 5, 9],
                **kwargs,
            )
        elif isinstance(backbone_config, dict):
            backbone_model_type = backbone_config.pop("model_type")
            config_class = CONFIG_MAPPING[backbone_model_type]
            backbone_config = config_class.from_dict(backbone_config)

        self.backbone_config = backbone_config
        # projector
        self.projector_scale_factors = projector_scale_factors
        for scale in projector_scale_factors:
            if scale not in [0.5, 1.0, 2.0]:
                raise ValueError(f"Unsupported scale factor: {scale}")
        self.projector_in_channels = [d_model] * len(projector_scale_factors)
        self.projector_out_channels = d_model
        self.activation_function = activation_function
        self.hidden_expansion = hidden_expansion
        self.c2f_num_blocks = c2f_num_blocks
        # decoder
        self.d_model = d_model
        self.dropout = dropout
        self.num_queries = num_queries
        self.decoder_ffn_dim = decoder_ffn_dim
        self.num_feature_levels = len(self.projector_scale_factors)
        self.decoder_n_points = decoder_n_points
        self.decoder_layers = decoder_layers
        self.decoder_activation_function = decoder_activation_function
        self.decoder_self_attention_heads = decoder_self_attention_heads
        self.decoder_cross_attention_heads = decoder_cross_attention_heads
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        # model
        self.init_std = init_std
        self.group_detr = group_detr
        # Loss
        self.auxiliary_loss = auxiliary_loss
        # Hungarian matcher
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        # Loss coefficients
        self.class_loss_coefficient = class_loss_coefficient
        self.dice_loss_coefficient = dice_loss_coefficient
        self.bbox_loss_coefficient = bbox_loss_coefficient
        self.giou_loss_coefficient = giou_loss_coefficient
        self.eos_coefficient = eos_coefficient
        self.focal_alpha = focal_alpha
        self.disable_custom_kernels = disable_custom_kernels
        super().__init__(**kwargs)


class LwDetrViTSelfAttention(ViTSelfAttention):
    def __init__(self, config: LwDetrViTConfig):
        super().__init__(config)
        del self.key
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.num_key_value_groups = 1
        self.dropout_prob = config.dropout_prob

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = hidden_states.shape[0]
        new_shape = batch_size, -1, self.num_attention_heads, self.attention_head_size

        key_layer = self.key(hidden_states).view(*new_shape).transpose(1, 2)
        value_layer = self.value(hidden_states).view(*new_shape).transpose(1, 2)
        query_layer = self.query(hidden_states).view(*new_shape).transpose(1, 2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        context_layer, attention_probs = attention_interface(
            self,
            query_layer,
            key_layer,
            value_layer,
            None,
            is_causal=self.is_causal,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.dropout_prob,
            **kwargs,
        )

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        return context_layer, attention_probs


class LwDetrViTAttention(ViTAttention):
    def __init__(self, config: LwDetrViTConfig):
        """
        Args:
            config (`LwDetrViTConfig`):
                Model configuration.
        """
        super().__init__(config)
        self.attention = LwDetrViTSelfAttention(config)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        self_attn_output, _ = self.attention(hidden_states, **kwargs)
        output = self.output(self_attn_output)
        return output


class LwDetrViTMlp(VitDetMlp):
    pass


class LwDetrViTLayer(GradientCheckpointingLayer):
    def __init__(
        self,
        config: LwDetrViTConfig,
        layer_idx,
    ) -> None:
        super().__init__()

        dim = config.hidden_size
        self.attention = LwDetrViTAttention(config)
        self.intermediate = LwDetrViTMlp(config=config, in_features=dim, hidden_features=int(dim * config.mlp_ratio))
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.gamma_1 = nn.Parameter(torch.Tensor(dim), requires_grad=True)
        self.gamma_2 = nn.Parameter(torch.Tensor(dim), requires_grad=True)

        self.window = layer_idx in config.window_block_indices
        self.num_windows = config.num_windows

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        batch_size, seq_len, channels = hidden_states.shape
        hidden_states_norm = self.layernorm_before(hidden_states)

        if not self.window:
            hidden_states_norm = hidden_states_norm.reshape(
                batch_size // self.num_windows, self.num_windows * seq_len, channels
            )

        attention_output = self.attention(hidden_states_norm, **kwargs)
        attention_output = attention_output * self.gamma_1

        if not self.window:
            attention_output = attention_output.reshape(batch_size, seq_len, channels)

        hidden_states = hidden_states + attention_output

        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = layer_output * self.gamma_2

        hidden_states = hidden_states + layer_output

        return hidden_states


class LwDetrViTEncoder(ViTEncoder):
    def __init__(self, config: LwDetrViTConfig) -> None:
        super().__init__(config)
        self.layer = nn.ModuleList([LwDetrViTLayer(config, i) for i in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> list[torch.Tensor]:
        list_hidden_states = [hidden_states]
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, **kwargs)
            list_hidden_states.append(hidden_states)
        return list_hidden_states


class LwDetrViTEmbeddings(VitDetEmbeddings):
    pass


class LwDetrViTPreTrainedModel(VitDetPreTrainedModel):
    config: LwDetrViTConfig
    base_model_prefix = "lw_detr_vit"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LwDetrViTEmbeddings", "LwDetrViTLayer"]
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": LwDetrViTLayer,
        "attentions": LwDetrViTSelfAttention,
    }

    def _init_weights(self, module) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            init.trunc_normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            init.zeros_(module.bias)
            init.ones_(module.weight)
        elif isinstance(module, LwDetrViTEmbeddings):
            init.trunc_normal_(module.position_embeddings, mean=0.0, std=self.config.initializer_range)
        if isinstance(module, LwDetrViTLayer):
            nn.init.constant_(module.gamma_1, self.config.cae_init_values)
            nn.init.constant_(module.gamma_2, self.config.cae_init_values)


@auto_docstring()
class LwDetrViTBackbone(VitDetBackbone):
    @check_model_inputs
    @auto_docstring
    def forward(self, pixel_values: torch.Tensor, **kwargs: Unpack[TransformersKwargs]) -> BackboneOutput:
        r"""
        Examples:

        ```python
        >>> from transformers import LwDetrViTConfig, LwDetrViTBackbone
        >>> import torch

        >>> config = LwDetrViTConfig()
        >>> model = LwDetrViTBackbone(config)

        >>> pixel_values = torch.randn(1, 3, 224, 224)

        >>> with torch.no_grad():
        ...     outputs = model(pixel_values)

        >>> feature_maps = outputs.feature_maps
        >>> list(feature_maps[-1].shape)
        [1, 768, 14, 14]
        ```"""
        embedding_output = self.embeddings(pixel_values)

        batch_size, channels, height, width = embedding_output.shape
        # (batch_size, channels, height, width) -> (batch_size, height, width, channels)
        hidden_states = embedding_output.permute(0, 2, 3, 1)

        window_height = height // self.config.num_windows_side
        window_width = width // self.config.num_windows_side
        # (batch_size, height, width, channels) -> (batch_size*num_windows_side**2, window_height*window_width, channels)
        hidden_states = (
            hidden_states.reshape(
                batch_size,
                self.config.num_windows_side,
                window_height,
                self.config.num_windows_side,
                window_width,
                channels,
            )
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(batch_size * self.config.num_windows_side**2, window_height * window_width, channels)
        )

        hidden_states = self.encoder(hidden_states, **kwargs)

        feature_maps = ()
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                hidden_state = (
                    hidden_state.reshape(
                        batch_size,
                        self.config.num_windows_side,
                        self.config.num_windows_side,
                        window_height,
                        window_width,
                        channels,
                    )
                    .permute(0, 5, 1, 3, 2, 4)
                    .reshape(batch_size, channels, height, width)
                )
                feature_maps += (hidden_state,)

        return BackboneOutput(feature_maps=feature_maps)


class LwDetrConvNormLayer(RTDetrConvNormLayer):
    def __init__(
        self,
        config: LwDetrConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        activation: str | None = None,
    ):
        super().__init__(config, in_channels, out_channels, kernel_size, stride, activation)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=kernel_size // 2,
            bias=False,
        )


class LwDetrRepVggBlock(nn.Module):
    def __init__(self, config: LwDetrConfig):
        super().__init__()
        hidden_channels = int(config.d_model * config.hidden_expansion)
        self.conv1 = LwDetrConvNormLayer(
            config, hidden_channels, hidden_channels, 3, 1, activation=config.activation_function
        )
        self.conv2 = LwDetrConvNormLayer(
            config, hidden_channels, hidden_channels, 3, 1, activation=config.activation_function
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.conv2(y)
        return y


class LwDetrC2FLayer(nn.Module):
    # Inspired by RTDetrCSPRepLayer
    def __init__(self, config: LwDetrConfig, in_channels: int):
        super().__init__()
        num_blocks = config.c2f_num_blocks
        activation = config.activation_function
        out_channels = config.d_model

        self.hidden_channels = int(out_channels * config.hidden_expansion)

        conv1_out_channels = 2 * self.hidden_channels
        self.conv1 = LwDetrConvNormLayer(config, in_channels, conv1_out_channels, 1, 1, activation=activation)

        conv2_in_channels = (2 + num_blocks) * self.hidden_channels
        self.conv2 = LwDetrConvNormLayer(config, conv2_in_channels, out_channels, 1, 1, activation=activation)

        self.bottlenecks = nn.ModuleList(LwDetrRepVggBlock(config) for _ in range(num_blocks))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv1(hidden_states)
        all_hidden_states = list(hidden_states.split(self.hidden_channels, 1))
        hidden_states = all_hidden_states[-1]

        for bottleneck in self.bottlenecks:
            hidden_states = bottleneck(hidden_states)
            all_hidden_states.append(hidden_states)

        hidden_states = torch.cat(all_hidden_states, 1)
        hidden_states = self.conv2(hidden_states)
        return hidden_states


class LwDetrLayerNorm(ConvNextLayerNorm):
    pass


class LwDetrSamplingLayer(nn.Module):
    def __init__(self, config: LwDetrConfig, channel_size: int, scale: float):
        super().__init__()

        self.scale = scale
        self.channel_size = channel_size

        layers = []
        if scale == 2.0:
            if channel_size > 512:
                layers.append(LwDetrConvNormLayer(config, channel_size, channel_size // 2, 1, 1, activation="relu"))
                layers.append(nn.ConvTranspose2d(channel_size // 2, channel_size // 4, kernel_size=2, stride=2))
            else:
                layers.append(nn.ConvTranspose2d(channel_size, channel_size // 2, 2, 2))
        elif scale == 0.5:
            layers.append(LwDetrConvNormLayer(config, channel_size, channel_size, 3, 2, activation="relu"))
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class LwDetrScaleProjector(nn.Module):
    def __init__(self, config: LwDetrConfig, scale: float):
        super().__init__()

        intermediate_dims = [config.backbone_config.hidden_size] * len(config.backbone_config.out_indices)
        sampling_layers = []
        for channel_size in intermediate_dims:
            sampling_layers.append(LwDetrSamplingLayer(config, channel_size, scale))
        self.sampling_layers = nn.ModuleList(sampling_layers)

        intermediate_dim = intermediate_dims[-1]
        if scale == 2.0:
            if intermediate_dim > 512:
                intermediate_dim = intermediate_dim // 4
            else:
                intermediate_dim = intermediate_dim // 2
        projector_input_dim = intermediate_dim * len(intermediate_dims)

        self.projector_layer = LwDetrC2FLayer(config, projector_input_dim)
        self.layer_norm = LwDetrLayerNorm(config.d_model, data_format="channels_first")

    def forward(self, hidden_states_tuple: tuple[torch.Tensor]) -> torch.Tensor:
        sampled_hidden_states = []
        for sampling_layer, hidden_states in zip(self.sampling_layers, hidden_states_tuple):
            hidden_states = sampling_layer(hidden_states)
            sampled_hidden_states.append(hidden_states)
        hidden_states = torch.cat(sampled_hidden_states, dim=1)
        hidden_states = self.projector_layer(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class LwDetrMultiScaleProjector(nn.Module):
    def __init__(self, config: LwDetrConfig):
        super().__init__()

        self.config = config
        scale_factors = config.projector_scale_factors

        self.scale_layers = nn.ModuleList([LwDetrScaleProjector(config, scale) for scale in scale_factors])

    def forward(self, hidden_states: tuple[torch.Tensor]) -> list[torch.Tensor]:
        output_hidden_states = []
        for scale_layer in self.scale_layers:
            output_hidden_states.append(scale_layer(hidden_states))
        return output_hidden_states


class LwDetrConvEncoder(nn.Module):
    def __init__(self, config: LwDetrConfig):
        super().__init__()
        self.backbone = LwDetrViTBackbone(config.backbone_config)
        self.projector = LwDetrMultiScaleProjector(config)

    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor):
        # send pixel_values through the model to get list of feature maps
        features = self.backbone(pixel_values).feature_maps
        features = self.projector(features)
        out = []
        for feature_map in features:
            # downsample pixel_mask to match shape of corresponding feature_map
            mask = nn.functional.interpolate(pixel_mask[None].float(), size=feature_map.shape[-2:]).to(torch.bool)[0]
            out.append((feature_map, mask))
        return out


class LwDetrAttention(nn.Module):
    def __init__(self, config: LwDetrConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.d_model // config.decoder_self_attention_heads)
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.num_key_value_groups = 1

        self.q_proj = nn.Linear(
            config.d_model, config.decoder_self_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.d_model, config.decoder_self_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.d_model, config.decoder_self_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.decoder_self_attention_heads * self.head_dim, config.d_model, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        hidden_states_original = hidden_states
        if position_embeddings is not None:
            hidden_states = hidden_states if position_embeddings is None else hidden_states + position_embeddings

        if self.training:
            # at training, we use group detr technique to add more supervision by using multiple weight-sharing decoders at once for faster convergence
            # at inference, we only use one decoder
            hidden_states_original = torch.cat(
                hidden_states_original.split(seq_len // self.config.group_detr, dim=1), dim=0
            )
            hidden_states = torch.cat(hidden_states.split(seq_len // self.config.group_detr, dim=1), dim=0)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states_original).view(hidden_shape).transpose(1, 2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=None,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if self.training:
            attn_output = torch.cat(torch.split(attn_output, batch_size, dim=0), dim=1)

        return attn_output, attn_weights


class LwDetrMultiscaleDeformableAttention(DeformableDetrMultiscaleDeformableAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_embeddings: torch.Tensor | None = None,
        reference_points=None,
        spatial_shapes=None,
        spatial_shapes_list=None,
        level_start_index=None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            **kwargs,
        )


class LwDetrMLP(nn.Module):
    def __init__(self, config: LwDetrConfig):
        super().__init__()
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.decoder_activation_function]
        self.fc1 = nn.Linear(config.d_model, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, config.d_model)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        return hidden_states


class LwDetrDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LwDetrConfig, layer_idx: int):
        nn.Module.__init__(self)

        # self-attention
        self.self_attn = LwDetrAttention(config, layer_idx=layer_idx)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.decoder_activation_function]
        self.activation_dropout = config.activation_dropout
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)

        # cross-attention
        self.cross_attn = LwDetrMultiscaleDeformableAttention(
            config,
            num_heads=config.decoder_cross_attention_heads,
            n_points=config.decoder_n_points,
        )
        self.cross_attn_layer_norm = nn.LayerNorm(config.d_model)

        # mlp
        self.mlp = LwDetrMLP(config)
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor | None = None,
        reference_points=None,
        spatial_shapes=None,
        spatial_shapes_list=None,
        level_start_index=None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        self_attention_output, self_attn_weights = self.self_attn(
            hidden_states, position_embeddings=position_embeddings, **kwargs
        )

        self_attention_output = nn.functional.dropout(self_attention_output, p=self.dropout, training=self.training)
        hidden_states = hidden_states + self_attention_output
        hidden_states = self.self_attn_layer_norm(hidden_states)

        cross_attention_output, cross_attn_weights = self.cross_attn(
            hidden_states=hidden_states,
            attention_mask=encoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            **kwargs,
        )
        cross_attention_output = nn.functional.dropout(cross_attention_output, p=self.dropout, training=self.training)
        hidden_states = hidden_states + cross_attention_output
        hidden_states = self.cross_attn_layer_norm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = self.layer_norm(hidden_states)

        return hidden_states


@auto_docstring
class LwDetrPreTrainedModel(PreTrainedModel):
    config: LwDetrConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"
    _no_split_modules = [
        r"LwDetrConvEncoder",
        r"LwDetrDecoderLayer",
    ]
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "attentions": [LwDetrAttention, LwDetrMultiscaleDeformableAttention],
        "hidden_states": [LwDetrDecoderLayer],
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)

        if isinstance(module, LwDetrMultiscaleDeformableAttention):
            init.constant_(module.sampling_offsets.weight, 0.0)
            thetas = torch.arange(module.n_heads, dtype=torch.int64).float() * (2.0 * math.pi / module.n_heads)
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = (
                (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
                .view(module.n_heads, 1, 1, 2)
                .repeat(1, module.n_levels, module.n_points, 1)
            )
            for i in range(module.n_points):
                grid_init[:, :, i, :] *= i + 1

            init.copy_(module.sampling_offsets.bias, grid_init.view(-1))
            init.constant_(module.attention_weights.weight, 0.0)
            init.constant_(module.attention_weights.bias, 0.0)
            init.xavier_uniform_(module.value_proj.weight)
            init.constant_(module.value_proj.bias, 0.0)
            init.xavier_uniform_(module.output_proj.weight)
            init.constant_(module.output_proj.bias, 0.0)
        if hasattr(module, "level_embed"):
            init.normal_(module.level_embed)
        if hasattr(module, "refpoint_embed") and module.refpoint_embed is not None:
            init.constant_(module.refpoint_embed.weight, 0)
        if hasattr(module, "class_embed") and module.class_embed is not None:
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            init.constant_(module.class_embed.bias, bias_value)
        if hasattr(module, "bbox_embed") and module.bbox_embed is not None:
            init.constant_(module.bbox_embed.layers[-1].weight, 0)
            init.constant_(module.bbox_embed.layers[-1].bias, 0)


def refine_bboxes(reference_points, deltas):
    reference_points = reference_points.to(deltas.device)
    new_reference_points_cxcy = deltas[..., :2] * reference_points[..., 2:] + reference_points[..., :2]
    new_reference_points_wh = deltas[..., 2:].exp() * reference_points[..., 2:]
    new_reference_points = torch.cat((new_reference_points_cxcy, new_reference_points_wh), -1)
    return new_reference_points


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for outputs of the LwDetrDecoder. This class adds two attributes to
    BaseModelOutputWithCrossAttentions, namely:
    - a stacked tensor of intermediate decoder hidden states (i.e. the output of each decoder layer)
    - a stacked tensor of intermediate reference points.
    """
)
class LwDetrDecoderOutput(DeformableDetrDecoderOutput):
    pass


class LwDetrDecoder(LwDetrPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`DeformableDetrDecoderLayer`].

    The decoder updates the query embeddings through multiple self-attention and deformable cross-attention layers.

    Some tweaks for LwDetr:

    - it uses group detr technique at training for faster convergence.

    Args:
        config: LwDetrConfig
    """

    def __init__(self, config: LwDetrConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layers = nn.ModuleList([LwDetrDecoderLayer(config, i) for i in range(config.decoder_layers)])
        self.layernorm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False

        self.ref_point_head = LwDetrMLPPredictionHead(2 * config.d_model, config.d_model, config.d_model, num_layers=2)

        self.post_init()

    def get_reference(self, reference_points, valid_ratios):
        # batch_size, num_queries, batch_size, 4
        obj_center = reference_points[..., :4]

        # batch_size, num_queries, num_levels, 4
        reference_points_inputs = obj_center[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[:, None]

        # batch_size, num_queries, d_model * 2
        query_sine_embed = gen_sine_position_embeddings(reference_points_inputs[:, :, 0, :], self.config.d_model)

        # batch_size, num_queries, d_model
        query_pos = self.ref_point_head(query_sine_embed)
        return reference_points_inputs, query_pos

    def forward(
        self,
        inputs_embeds: torch.Tensor | None = None,
        reference_points: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | None = None,
        spatial_shapes_list: torch.Tensor | None = None,
        level_start_index: torch.Tensor | None = None,
        valid_ratios: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        intermediate = ()
        intermediate_reference_points = (reference_points,)

        if inputs_embeds is not None:
            hidden_states = inputs_embeds

        reference_points_inputs, query_pos = self.get_reference(reference_points, valid_ratios)

        for idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                position_embeddings=query_pos,
                reference_points=reference_points_inputs,
                spatial_shapes=spatial_shapes,
                spatial_shapes_list=spatial_shapes_list,
                level_start_index=level_start_index,
                **kwargs,
            )
            intermediate_hidden_states = self.layernorm(hidden_states)
            intermediate += (intermediate_hidden_states,)

        intermediate = torch.stack(intermediate)
        last_hidden_state = intermediate[-1]
        intermediate_reference_points = torch.stack(intermediate_reference_points)

        return LwDetrDecoderOutput(
            last_hidden_state=last_hidden_state,
            intermediate_hidden_states=intermediate,
            intermediate_reference_points=intermediate_reference_points,
        )


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for outputs of the LwDetr backbone-decoder model.
    """
)
class LwDetrModelOutput(ModelOutput):
    r"""
    init_reference_points (`torch.FloatTensor` of shape  `(batch_size, num_queries, 4)`):
        Initial reference points sent through the Transformer decoder.
    intermediate_hidden_states (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`):
        Stacked intermediate hidden states (output of each layer of the decoder).
    intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`):
        Stacked intermediate reference points (reference points of each layer of the decoder).
    enc_outputs_class (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`):
        Predicted bounding boxes scores where the top `config.two_stage_num_proposals` scoring bounding boxes are
        picked as region proposals in the first stage. Output of bounding box binary classification (i.e.
        foreground and background).
    enc_outputs_coord_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, 4)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`):
        Logits of predicted bounding boxes coordinates in the first stage.
    """

    init_reference_points: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None
    intermediate_hidden_states: torch.FloatTensor | None = None
    intermediate_reference_points: torch.FloatTensor | None = None
    enc_outputs_class: torch.FloatTensor | None = None
    enc_outputs_coord_logits: torch.FloatTensor | None = None


@auto_docstring(
    custom_intro="""
    The bare LW Detr Model (consisting of a backbone and decoder Transformer) outputting raw
    hidden-states without any specific head on top.
    """
)
class LwDetrModel(DeformableDetrModel):
    def __init__(self, config: LwDetrConfig):
        LwDetrPreTrainedModel.__init__(config)

        # Create backbone + positional encoding
        self.backbone = LwDetrConvEncoder(config)

        self.group_detr = config.group_detr
        self.num_queries = config.num_queries
        hidden_dim = config.d_model
        self.reference_point_embed = nn.Embedding(self.num_queries * self.group_detr, 4)
        self.query_feat = nn.Embedding(self.num_queries * self.group_detr, hidden_dim)

        self.decoder = LwDetrDecoder(config)

        self.enc_output = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(self.group_detr)])
        self.enc_output_norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.group_detr)])
        # Should normally be None and then instantiated in the ForObjectDetection class
        self.enc_out_bbox_embed = nn.ModuleList(
            [LwDetrMLPPredictionHead(config.d_model, config.d_model, 4, num_layers=3) for _ in range(self.group_detr)]
        )
        self.enc_out_class_embed = nn.ModuleList(
            [nn.Linear(config.d_model, config.num_labels) for _ in range(self.group_detr)]
        )

        self.post_init()

    def gen_encoder_output_proposals(self, enc_output, padding_mask, spatial_shapes):
        """Generate the encoder output proposals from encoded enc_output.

        Args:
            enc_output (Tensor[batch_size, sequence_length, hidden_size]): Output of the encoder.
            padding_mask (Tensor[batch_size, sequence_length]): Padding mask for `enc_output`.
            spatial_shapes (list[tuple[int, int]]): Spatial shapes of the feature maps.

        Returns:
            `tuple(torch.FloatTensor)`: A tuple of feature map and bbox prediction.
                - object_query (Tensor[batch_size, sequence_length, hidden_size]): Object query features. Later used to
                  directly predict a bounding box. (without the need of a decoder)
                - output_proposals (Tensor[batch_size, sequence_length, 4]): Normalized proposals, after an inverse
                  sigmoid.
        """
        batch_size = enc_output.shape[0]
        proposals = []
        _cur = 0
        for level, (height, width) in enumerate(spatial_shapes):
            mask_flatten_ = padding_mask[:, _cur : (_cur + height * width)].view(batch_size, height, width, 1)
            valid_height = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_width = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = meshgrid(
                torch.linspace(
                    0,
                    height - 1,
                    height,
                    dtype=enc_output.dtype,
                    device=enc_output.device,
                ),
                torch.linspace(
                    0,
                    width - 1,
                    width,
                    dtype=enc_output.dtype,
                    device=enc_output.device,
                ),
                indexing="ij",
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_width.unsqueeze(-1), valid_height.unsqueeze(-1)], 1).view(batch_size, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(batch_size, -1, -1, -1) + 0.5) / scale
            width_height = torch.ones_like(grid) * 0.05 * (2.0**level)
            proposal = torch.cat((grid, width_height), -1).view(batch_size, -1, 4)
            proposals.append(proposal)
            _cur += height * width
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = output_proposals.masked_fill(padding_mask.unsqueeze(-1), float("inf"))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))

        # assign each pixel as an object query
        object_query = enc_output
        object_query = object_query.masked_fill(padding_mask.unsqueeze(-1), float(0))
        object_query = object_query.masked_fill(~output_proposals_valid, float(0))
        return object_query, output_proposals

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        pixel_mask: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> LwDetrModelOutput:
        r"""
        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, DeformableDetrModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("AnnaZhang/lwdetr_small_60e_coco")
        >>> model = DeformableDetrModel.from_pretrained("AnnaZhang/lwdetr_small_60e_coco")

        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)

        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 300, 256]
        ```"""
        batch_size, num_channels, height, width = pixel_values.shape
        device = pixel_values.device

        if pixel_mask is None:
            pixel_mask = torch.ones(((batch_size, height, width)), dtype=torch.long, device=device)

        # Extract multi-scale feature maps of same resolution `config.d_model` (cf Figure 4 in paper)
        # First, sent pixel_values + pixel_mask through Backbone to obtain the features
        # which is a list of tuples
        features = self.backbone(pixel_values, pixel_mask)

        # Then, apply 1x1 convolution to reduce the channel dimension to d_model (256 by default)
        sources = []
        masks = []
        for level, (source, mask) in enumerate(features):
            sources.append(source)
            masks.append(mask)
            if mask is None:
                raise ValueError("No attention mask was provided")

        if self.training:
            reference_points = self.reference_point_embed.weight
            query_feat = self.query_feat.weight
        else:
            # only use one group in inference
            reference_points = self.reference_point_embed.weight[: self.num_queries]
            query_feat = self.query_feat.weight[: self.num_queries]

        # Prepare encoder inputs (by flattening)
        source_flatten = []
        mask_flatten = []
        spatial_shapes_list = []
        for source, mask in zip(sources, masks):
            batch_size, num_channels, height, width = source.shape
            spatial_shape = (height, width)
            spatial_shapes_list.append(spatial_shape)
            source = source.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            source_flatten.append(source)
            mask_flatten.append(mask)
        source_flatten = torch.cat(source_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes_list, dtype=torch.long, device=source_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m, dtype=source_flatten.dtype) for m in masks], 1)

        target = query_feat.unsqueeze(0).expand(batch_size, -1, -1)
        reference_points = reference_points.unsqueeze(0).expand(batch_size, -1, -1)

        object_query_embedding, output_proposals = self.gen_encoder_output_proposals(
            source_flatten, ~mask_flatten, spatial_shapes_list
        )

        group_detr = self.group_detr if self.training else 1
        topk = self.num_queries
        topk_coords_logits = []
        topk_coords_logits_undetach = []
        object_query_undetach = []

        for group_id in range(group_detr):
            group_object_query = self.enc_output[group_id](object_query_embedding)
            group_object_query = self.enc_output_norm[group_id](group_object_query)

            group_enc_outputs_class = self.enc_out_class_embed[group_id](group_object_query)
            group_delta_bbox = self.enc_out_bbox_embed[group_id](group_object_query)
            group_enc_outputs_coord = refine_bboxes(output_proposals, group_delta_bbox)

            group_topk_proposals = torch.topk(group_enc_outputs_class.max(-1)[0], topk, dim=1)[1]
            group_topk_coords_logits_undetach = torch.gather(
                group_enc_outputs_coord,
                1,
                group_topk_proposals.unsqueeze(-1).repeat(1, 1, 4),
            )
            group_topk_coords_logits = group_topk_coords_logits_undetach.detach()
            group_object_query_undetach = torch.gather(
                group_object_query, 1, group_topk_proposals.unsqueeze(-1).repeat(1, 1, self.config.d_model)
            )

            topk_coords_logits.append(group_topk_coords_logits)
            topk_coords_logits_undetach.append(group_topk_coords_logits_undetach)
            object_query_undetach.append(group_object_query_undetach)

        topk_coords_logits = torch.cat(topk_coords_logits, 1)
        topk_coords_logits_undetach = torch.cat(topk_coords_logits_undetach, 1)
        object_query_undetach = torch.cat(object_query_undetach, 1)

        enc_outputs_class = object_query_undetach
        enc_outputs_coord_logits = topk_coords_logits

        reference_points = refine_bboxes(topk_coords_logits_undetach, reference_points)

        init_reference_points = reference_points
        decoder_outputs = self.decoder(
            inputs_embeds=target,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            encoder_hidden_states=source_flatten,
            encoder_attention_mask=mask_flatten,
            **kwargs,
        )

        return LwDetrModelOutput(
            init_reference_points=init_reference_points,
            last_hidden_state=decoder_outputs.last_hidden_state,
            intermediate_hidden_states=decoder_outputs.intermediate_hidden_states,
            intermediate_reference_points=decoder_outputs.intermediate_reference_points,
            enc_outputs_class=enc_outputs_class,
            enc_outputs_coord_logits=enc_outputs_coord_logits,
        )


class LwDetrMLPPredictionHead(DeformableDetrMLPPredictionHead):
    pass


@dataclass
@auto_docstring(
    custom_intro="""
    Output type of [`LwDetrForObjectDetection`].
    """
)
class LwDetrObjectDetectionOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` are provided)):
        Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
        bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
        scale-invariant IoU loss.
    loss_dict (`Dict`, *optional*):
        A dictionary containing the individual losses. Useful for logging.
    logits (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes + 1)`):
        Classification logits (including no-object) for all queries.
    pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`):
        Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
        values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
        possible padding). You can use [`~DeformableDetrProcessor.post_process_object_detection`] to retrieve the
        unnormalized bounding boxes.
    auxiliary_outputs (`list[Dict]`, *optional*):
        Optional, only returned when auxiliary losses are activated (i.e. `config.auxiliary_loss` is set to `True`)
        and labels are provided. It is a list of dictionaries containing the two above keys (`logits` and
        `pred_boxes`) for each decoder layer.
    init_reference_points (`torch.FloatTensor` of shape  `(batch_size, num_queries, 4)`):
        Initial reference points sent through the Transformer decoder.
    intermediate_hidden_states (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`):
        Stacked intermediate hidden states (output of each layer of the decoder).
    intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`):
        Stacked intermediate reference points (reference points of each layer of the decoder).
    enc_outputs_class (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`):
        Predicted bounding boxes scores where the top `config.two_stage_num_proposals` scoring bounding boxes are
        picked as region proposals in the first stage. Output of bounding box binary classification (i.e.
        foreground and background).
    enc_outputs_coord_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, 4)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`):
        Logits of predicted bounding boxes coordinates in the first stage.
    """

    loss: torch.FloatTensor | None = None
    loss_dict: dict | None = None
    logits: torch.FloatTensor | None = None
    pred_boxes: torch.FloatTensor | None = None
    auxiliary_outputs: list[dict] | None = None
    init_reference_points: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None
    intermediate_hidden_states: torch.FloatTensor | None = None
    intermediate_reference_points: torch.FloatTensor | None = None
    enc_outputs_class: Any = None
    enc_outputs_coord_logits: torch.FloatTensor | None = None


@auto_docstring(
    custom_intro="""
    LW DETR Model (consisting of a backbone and decoder Transformer) with object detection heads on
    top, for tasks such as COCO detection.
    """
)
class LwDetrForObjectDetection(DeformableDetrForObjectDetection):
    _tied_weights_keys = None

    def __init__(self, config: LwDetrConfig):
        PreTrainedModel.__init__(self, config)
        self.model = LwDetrModel(config)
        self.class_embed = nn.Linear(config.d_model, config.num_labels)
        self.bbox_embed = LwDetrMLPPredictionHead(config.d_model, config.d_model, 4, num_layers=3)

        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        pixel_mask: torch.LongTensor | None = None,
        labels: list[dict] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> LwDetrObjectDetectionOutput:
        r"""
        decoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, num_queries)`, *optional*):
            Not used by default. Can be used to mask object queries.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing the flattened feature map (output of the backbone + projection layer), you
            can choose to directly pass a flattened representation of an image.
        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
            Optionally, instead of initializing the queries with a tensor of zeros, you can choose to directly pass an
            embedded representation.
        labels (`list[Dict]` of len `(batch_size,)`, *optional*):
            Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
            following 2 keys: 'class_labels' and 'boxes' (the class labels and bounding boxes of an image in the batch
            respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes
            in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)`.

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, LwDetrForObjectDetection
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("AnnaZhang/lwdetr_small_60e_coco")
        >>> model = LwDetrForObjectDetection.from_pretrained("AnnaZhang/lwdetr_small_60e_coco")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> # convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
        >>> target_sizes = torch.tensor([image.size[::-1]])
        >>> results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[
        ...     0
        ... ]
        >>> for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        ...     box = [round(i, 2) for i in box.tolist()]
        ...     print(
        ...         f"Detected {model.config.id2label[label.item()]} with confidence "
        ...         f"{round(score.item(), 3)} at location {box}"
        ...     )
        Detected cat with confidence 0.8 at location [16.5, 52.84, 318.25, 470.78]
        Detected cat with confidence 0.789 at location [342.19, 24.3, 640.02, 372.25]
        Detected remote with confidence 0.633 at location [40.79, 72.78, 176.76, 117.25]
        ```"""
        outputs = self.model(
            pixel_values,
            pixel_mask=pixel_mask,
            **kwargs,
        )

        last_hidden_states = outputs.last_hidden_state
        intermediate_reference_points = outputs.intermediate_reference_points
        enc_outputs_class_logits = outputs.enc_outputs_class
        enc_outputs_boxes_logits = outputs.enc_outputs_coord_logits

        logits = self.class_embed(last_hidden_states)
        pred_boxes_delta = self.bbox_embed(last_hidden_states)
        pred_boxes = refine_bboxes(intermediate_reference_points[-1], pred_boxes_delta)

        enc_outputs_class_logits_list = enc_outputs_class_logits.split(self.config.num_queries, dim=1)
        pred_class = []
        group_detr = self.config.group_detr if self.training else 1
        for group_index in range(group_detr):
            group_pred_class = self.model.enc_out_class_embed[group_index](enc_outputs_class_logits_list[group_index])
            pred_class.append(group_pred_class)
        enc_outputs_class_logits = torch.cat(pred_class, dim=1)

        loss, loss_dict, auxiliary_outputs = None, None, None
        if labels is not None:
            outputs_class, outputs_coord = None, None
            if self.config.auxiliary_loss:
                intermediate_hidden_states = outputs.intermediate_hidden_states
                outputs_coord_delta = self.bbox_embed(intermediate_hidden_states)
                outputs_coord = refine_bboxes(intermediate_reference_points, outputs_coord_delta)
                outputs_class = self.class_embed(intermediate_hidden_states)

            loss, loss_dict, auxiliary_outputs = self.loss_function(
                logits,
                labels,
                self.device,
                pred_boxes,
                self.config,
                outputs_class,
                outputs_coord,
                enc_outputs_class_logits,
                enc_outputs_boxes_logits,
            )

        return LwDetrObjectDetectionOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes,
            auxiliary_outputs=auxiliary_outputs,
            last_hidden_state=outputs.last_hidden_state,
            intermediate_hidden_states=outputs.intermediate_hidden_states,
            intermediate_reference_points=outputs.intermediate_reference_points,
            init_reference_points=outputs.init_reference_points,
            enc_outputs_class=enc_outputs_class_logits,
            enc_outputs_coord_logits=enc_outputs_boxes_logits,
        )


__all__ = [
    "LwDetrConfig",
    "LwDetrPreTrainedModel",
    "LwDetrModel",
    "LwDetrForObjectDetection",
    "LwDetrViTConfig",
    "LwDetrViTPreTrainedModel",
    "LwDetrViTBackbone",
]
