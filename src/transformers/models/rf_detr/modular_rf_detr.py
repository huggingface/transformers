# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ...configuration_utils import PreTrainedConfig
from ...modeling_outputs import BackboneOutput, BaseModelOutput
from ...utils.generic import can_return_tuple
from ..auto import AutoConfig
from ..dinov2_with_registers.configuration_dinov2_with_registers import Dinov2WithRegistersConfig
from ..dinov2_with_registers.modeling_dinov2_with_registers import (
    Dinov2WithRegistersBackbone,
    Dinov2WithRegistersEmbeddings,
    Dinov2WithRegistersLayer,
    Dinov2WithRegistersPreTrainedModel,
)
from ..lw_detr.configuration_lw_detr import LwDetrConfig
from ..lw_detr.modeling_lw_detr import (
    LwDetrConvNormLayer,
    LwDetrDecoder,
    LwDetrForObjectDetection,
    LwDetrLayerNorm,
    LwDetrMLPPredictionHead,
    LwDetrModel,
    LwDetrMultiScaleProjector,
    LwDetrObjectDetectionOutput,
    LwDetrPreTrainedModel,
    refine_bboxes,
)


class RfDetrWindowedDinov2Config(Dinov2WithRegistersConfig):
    r"""
    This is the configuration class to store the configuration of a [`RfDetrWindowedDinov2Backbone`]. It is used to
    instantiate an RF-DETR windowed DINOv2 backbone according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the
    DINOv2 with Registers
    [facebook/dinov2-with-registers-base](https://huggingface.co/facebook/dinov2-with-registers-base) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        mlp_ratio (`int`, *optional*, defaults to 4):
            Ratio of the hidden size of the MLPs relative to the `hidden_size`.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        layerscale_value (`float`, *optional*, defaults to 1.0):
           Initial value to use for layer scale.
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            Stochastic depth rate per sample (when applied in the main path of residual layers).
        use_swiglu_ffn (`bool`, *optional*, defaults to `False`):
            Whether to use the SwiGLU feedforward neural network.
        num_register_tokens (`int`, *optional*, defaults to 4):
            Number of register tokens to use.
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
        apply_layernorm (`bool`, *optional*, defaults to `True`):
            Whether to apply layer normalization to the feature maps in case the model is used as backbone.
        reshape_hidden_states (`bool`, *optional*, defaults to `True`):
            Whether to reshape the feature maps to 4D tensors of shape `(batch_size, hidden_size, height, width)` in
            case the model is used as backbone. If `False`, the feature maps will be 3D tensors of shape `(batch_size,
            seq_len, hidden_size)`.
        num_windows (`int`, *optional*, defaults to 2):
            Number of windows to split the image into for windowed attention. The image patches are divided into a
            grid of `num_windows x num_windows` windows for local attention computation.
        window_block_indexes (`list[int]`, *optional*):
            List of encoder layer indices that use windowed (local) attention instead of global attention.
            If not provided, all layers use windowed attention by default.

    Example:

    ```python
    >>> from transformers import RfDetrWindowedDinov2Config, RfDetrWindowedDinov2Backbone

    >>> # Initializing a RfDetrWindowedDinov2 configuration
    >>> configuration = RfDetrWindowedDinov2Config()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = RfDetrWindowedDinov2Backbone(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "rf_detr_windowed_dinov2"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        mlp_ratio=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-6,
        image_size=224,
        patch_size=16,
        num_channels=3,
        qkv_bias=True,
        layerscale_value=1.0,
        drop_path_rate=0.0,
        use_swiglu_ffn=False,
        num_register_tokens=4,
        out_features=None,
        out_indices=None,
        apply_layernorm=True,
        reshape_hidden_states=True,
        num_windows: int = 2,
        window_block_indexes: list[int] | None = None,
        **kwargs,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            mlp_ratio=mlp_ratio,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            qkv_bias=qkv_bias,
            layerscale_value=layerscale_value,
            drop_path_rate=drop_path_rate,
            use_swiglu_ffn=use_swiglu_ffn,
            num_register_tokens=num_register_tokens,
            out_features=out_features,
            out_indices=out_indices,
            apply_layernorm=apply_layernorm,
            reshape_hidden_states=reshape_hidden_states,
            **kwargs,
        )
        self.num_windows = num_windows
        self.window_block_indexes = (
            list(range(self.num_hidden_layers)) if window_block_indexes is None else window_block_indexes
        )


class RfDetrWindowedDinov2Embeddings(Dinov2WithRegistersEmbeddings):
    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, _, height, width = pixel_values.shape
        target_dtype = self.patch_embeddings.projection.weight.dtype
        embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))

        if bool_masked_pos is not None:
            embeddings = torch.where(
                bool_masked_pos.unsqueeze(-1), self.mask_token.to(embeddings.dtype).unsqueeze(0), embeddings
            )

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)

        if self.config.num_windows > 1:
            num_h_patches = height // self.config.patch_size
            num_w_patches = width // self.config.patch_size
            cls_token_with_pos_embed = embeddings[:, :1]
            pixel_tokens_with_pos_embed = embeddings[:, 1:]
            pixel_tokens_with_pos_embed = pixel_tokens_with_pos_embed.view(
                batch_size, num_h_patches, num_w_patches, -1
            )

            num_w_patches_per_window = num_w_patches // self.config.num_windows
            num_h_patches_per_window = num_h_patches // self.config.num_windows
            num_windows = self.config.num_windows

            windowed_pixel_tokens = pixel_tokens_with_pos_embed.reshape(
                batch_size * num_windows,
                num_h_patches_per_window,
                num_windows,
                num_w_patches_per_window,
                -1,
            )
            windowed_pixel_tokens = windowed_pixel_tokens.permute(0, 2, 1, 3, 4)
            windowed_pixel_tokens = windowed_pixel_tokens.reshape(
                batch_size * num_windows**2,
                num_h_patches_per_window * num_w_patches_per_window,
                -1,
            )
            windowed_cls_token_with_pos_embed = cls_token_with_pos_embed.repeat(num_windows**2, 1, 1)
            embeddings = torch.cat((windowed_cls_token_with_pos_embed, windowed_pixel_tokens), dim=1)

        if self.register_tokens is not None:
            embeddings = torch.cat(
                (embeddings[:, :1], self.register_tokens.expand(embeddings.shape[0], -1, -1), embeddings[:, 1:]), dim=1
            )

        embeddings = self.dropout(embeddings)
        return embeddings


class RfDetrWindowedDinov2Layer(Dinov2WithRegistersLayer):
    def __init__(self, config: RfDetrWindowedDinov2Config):
        super().__init__(config)
        self.num_windows = config.num_windows

    def forward(
        self,
        hidden_states: torch.Tensor,
        run_full_attention: bool = False,
    ) -> torch.Tensor:
        shortcut = hidden_states

        if run_full_attention and self.num_windows > 1:
            batch_size, hidden_state_length, channels = hidden_states.shape
            num_windows_squared = self.num_windows**2
            hidden_states = hidden_states.view(
                batch_size // num_windows_squared, num_windows_squared * hidden_state_length, channels
            )

        hidden_states_norm = self.norm1(hidden_states)
        self_attention_output = self.attention(hidden_states_norm)

        if run_full_attention and self.num_windows > 1:
            batch_size, hidden_state_length, channels = hidden_states.shape
            num_windows_squared = self.num_windows**2
            self_attention_output = self_attention_output.view(
                batch_size * num_windows_squared,
                hidden_state_length // num_windows_squared,
                channels,
            )

        self_attention_output = self.layer_scale1(self_attention_output)
        hidden_states = self.drop_path(self_attention_output) + shortcut

        layer_output = self.norm2(hidden_states)
        layer_output = self.mlp(layer_output)
        layer_output = self.layer_scale2(layer_output)
        layer_output = self.drop_path(layer_output) + hidden_states

        return layer_output


class RfDetrWindowedDinov2Encoder(nn.Module):
    def __init__(self, config: RfDetrWindowedDinov2Config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([RfDetrWindowedDinov2Layer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states: torch.Tensor, output_hidden_states: bool = False) -> BaseModelOutput:
        all_hidden_states = [hidden_states] if output_hidden_states else None

        for layer_idx, layer_module in enumerate(self.layer):
            run_full_attention = layer_idx not in self.config.window_block_indexes
            hidden_states = layer_module(hidden_states, run_full_attention=run_full_attention)
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=tuple(all_hidden_states) if all_hidden_states is not None else None,
        )


class RfDetrWindowedDinov2PreTrainedModel(Dinov2WithRegistersPreTrainedModel):
    pass


class RfDetrWindowedDinov2Backbone(Dinov2WithRegistersBackbone):
    config_class = RfDetrWindowedDinov2Config

    def __init__(self, config: RfDetrWindowedDinov2Config):
        super().__init__(config)
        self.embeddings = RfDetrWindowedDinov2Embeddings(config)
        self.encoder = RfDetrWindowedDinov2Encoder(config)
        self.num_register_tokens = config.num_register_tokens
        self.post_init()

    def forward(
        self, pixel_values: torch.Tensor, output_hidden_states: bool | None = None, **kwargs
    ) -> BackboneOutput:
        if output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states

        embedding_output = self.embeddings(pixel_values)
        outputs = self.encoder(embedding_output, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        feature_maps = ()
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                if self.config.apply_layernorm:
                    hidden_state = self.layernorm(hidden_state)

                if self.config.reshape_hidden_states:
                    hidden_state = hidden_state[:, 1 + self.num_register_tokens :]
                    batch_size, _, height, width = pixel_values.shape
                    patch_size = self.config.patch_size

                    num_h_patches = height // patch_size
                    num_w_patches = width // patch_size

                    if self.config.num_windows > 1:
                        num_windows_squared = self.config.num_windows**2
                        hidden_batch_size, hidden_state_length, channels = hidden_state.shape
                        num_h_patches_per_window = num_h_patches // self.config.num_windows
                        num_w_patches_per_window = num_w_patches // self.config.num_windows

                        hidden_state = hidden_state.reshape(
                            hidden_batch_size // num_windows_squared,
                            num_windows_squared * hidden_state_length,
                            channels,
                        )
                        hidden_state = hidden_state.reshape(
                            (hidden_batch_size // num_windows_squared) * self.config.num_windows,
                            self.config.num_windows,
                            num_h_patches_per_window,
                            num_w_patches_per_window,
                            channels,
                        )
                        hidden_state = hidden_state.permute(0, 2, 1, 3, 4)

                    hidden_state = hidden_state.reshape(batch_size, num_h_patches, num_w_patches, -1)
                    hidden_state = hidden_state.permute(0, 3, 1, 2).contiguous()

                feature_maps += (hidden_state,)

        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
        )


class RfDetrConfig(LwDetrConfig):
    r"""
    This is the configuration class to store the configuration of a [`RfDetrModel`]. It is used to instantiate
    an RF-DETR model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the RF-DETR
    [AnnaZhang/RfDetr_small_60e_coco](https://huggingface.co/AnnaZhang/RfDetr_small_60e_coco) architecture.

    RF-DETR (Region-Focused Detection Transformer) is a transformer-based object detection model that uses a DINOv2
    backbone with windowed attention for efficient feature extraction and a deformable attention decoder for
    detection.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        backbone_config (`PretrainedConfig` or `dict`, *optional*):
            The configuration of the backbone model. If not provided, will default to
            `RfDetrWindowedDinov2Config` with a small DINOv2 architecture optimized for detection tasks.
        projector_scale_factors (`list[float]`, *optional*, defaults to `[1.0]`):
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
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        decoder_ffn_dim (`int`, *optional*, defaults to 2048):
            Dimension of the "intermediate" (often named feed-forward) layer in decoder.
        decoder_n_points (`int`, *optional*, defaults to 2):
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
            [`RfDetrModel`] can detect in a single image.
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
        mask_downsample_ratio (`int`, *optional*, defaults to 4):
            Downsample ratio for the segmentation mask predictions relative to the input image size.
        segmentation_bottleneck_ratio (`int`, *optional*, defaults to 1):
            Bottleneck ratio for the segmentation head. Controls the dimensionality reduction of features
            in the segmentation head interaction layers.

    Examples:

    ```python
    >>> from transformers import RfDetrConfig, RfDetrModel

    >>> # Initializing an RF-DETR configuration
    >>> configuration = RfDetrConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = RfDetrModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "rf_detr"
    sub_configs = {"backbone_config": AutoConfig}

    def __init__(
        self,
        backbone_config: PreTrainedConfig | dict | None = None,
        projector_scale_factors: list[float] | None = None,
        hidden_expansion=0.5,
        c2f_num_blocks=3,
        activation_function="silu",
        batch_norm_eps=1e-5,
        d_model=256,
        dropout: float = 0.0,
        decoder_ffn_dim=2048,
        decoder_n_points: int = 2,
        decoder_layers: int = 3,
        decoder_self_attention_heads: int = 8,
        decoder_cross_attention_heads: int = 16,
        decoder_activation_function="relu",
        num_queries: int = 300,
        attention_bias=True,
        attention_dropout=0.0,
        activation_dropout=0.0,
        group_detr: int = 13,
        init_std=0.02,
        disable_custom_kernels=True,
        class_cost=2,
        bbox_cost=5,
        giou_cost=2,
        mask_loss_coefficient=1,
        dice_loss_coefficient=1,
        bbox_loss_coefficient=5,
        giou_loss_coefficient=2,
        eos_coefficient=0.1,
        focal_alpha=0.25,
        auxiliary_loss=True,
        mask_downsample_ratio=4,
        segmentation_bottleneck_ratio=1,
        **kwargs,
    ):
        if backbone_config is None:
            backbone_config = RfDetrWindowedDinov2Config(
                image_size=512,
                patch_size=16,
                hidden_size=384,
                num_hidden_layers=12,
                num_attention_heads=6,
                mlp_ratio=4,
                out_indices=[3, 6, 9, 12],
                num_register_tokens=0,
                num_windows=2,
                window_block_indexes=[0, 1, 2, 4, 5, 7, 8, 10, 11],
            )
        elif isinstance(backbone_config, dict):
            backbone_config = dict(backbone_config)
            model_type = backbone_config.pop("model_type", "rf_detr_windowed_dinov2")
            if model_type == "rf_detr_windowed_dinov2":
                backbone_config = RfDetrWindowedDinov2Config(**backbone_config)
            else:
                backbone_config = AutoConfig.for_model(model_type, **backbone_config)

        if projector_scale_factors is None:
            projector_scale_factors = [1.0]
        if "num_labels" not in kwargs:
            kwargs["num_labels"] = 91

        super().__init__(
            backbone_config=backbone_config,
            projector_scale_factors=projector_scale_factors,
            hidden_expansion=hidden_expansion,
            c2f_num_blocks=c2f_num_blocks,
            activation_function=activation_function,
            batch_norm_eps=batch_norm_eps,
            d_model=d_model,
            dropout=dropout,
            decoder_ffn_dim=decoder_ffn_dim,
            decoder_n_points=decoder_n_points,
            decoder_layers=decoder_layers,
            decoder_self_attention_heads=decoder_self_attention_heads,
            decoder_cross_attention_heads=decoder_cross_attention_heads,
            decoder_activation_function=decoder_activation_function,
            num_queries=num_queries,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            group_detr=group_detr,
            init_std=init_std,
            disable_custom_kernels=disable_custom_kernels,
            class_cost=class_cost,
            bbox_cost=bbox_cost,
            giou_cost=giou_cost,
            mask_loss_coefficient=mask_loss_coefficient,
            dice_loss_coefficient=dice_loss_coefficient,
            bbox_loss_coefficient=bbox_loss_coefficient,
            giou_loss_coefficient=giou_loss_coefficient,
            eos_coefficient=eos_coefficient,
            focal_alpha=focal_alpha,
            auxiliary_loss=auxiliary_loss,
            **kwargs,
        )
        self.mask_downsample_ratio = mask_downsample_ratio
        self.segmentation_bottleneck_ratio = segmentation_bottleneck_ratio


class RfDetrPreTrainedModel(LwDetrPreTrainedModel):
    pass


class RfDetrDecoder(LwDetrDecoder):
    pass


class RfDetrMultiScaleProjector(LwDetrMultiScaleProjector):
    pass


class RfDetrLayerNorm(LwDetrLayerNorm):
    pass


class RfDetrConvNormLayer(LwDetrConvNormLayer):
    pass


class RfDetrMLPPredictionHead(LwDetrMLPPredictionHead):
    pass


class RfDetrDepthwiseConvBlock(nn.Module):
    def __init__(self, dim: int, layer_scale_init_value: float = 0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim,)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.dwconv(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 3, 1)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.pwconv1(hidden_states)
        hidden_states = self.act(hidden_states)
        if self.gamma is not None:
            hidden_states = self.gamma * hidden_states
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        return hidden_states + residual


class RfDetrMLPBlock(nn.Module):
    def __init__(self, dim: int, layer_scale_init_value: float = 0):
        super().__init__()
        self.norm_in = nn.LayerNorm(dim)
        self.layers = nn.ModuleList(
            [
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim),
            ]
        )
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim,)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm_in(hidden_states)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        if self.gamma is not None:
            hidden_states = self.gamma * hidden_states
        return hidden_states + residual


class RfDetrSegmentationHead(nn.Module):
    def __init__(self, in_dim: int, num_blocks: int, bottleneck_ratio: int | None = 1, downsample_ratio: int = 4):
        super().__init__()
        self.downsample_ratio = downsample_ratio
        self.interaction_dim = in_dim if bottleneck_ratio is None else in_dim // bottleneck_ratio
        self.blocks = nn.ModuleList([RfDetrDepthwiseConvBlock(in_dim) for _ in range(num_blocks)])
        self.spatial_features_proj = (
            nn.Identity() if bottleneck_ratio is None else nn.Conv2d(in_dim, self.interaction_dim, kernel_size=1)
        )
        self.query_features_block = RfDetrMLPBlock(in_dim)
        self.query_features_proj = (
            nn.Identity() if bottleneck_ratio is None else nn.Linear(in_dim, self.interaction_dim)
        )
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(
        self,
        spatial_features: torch.Tensor,
        query_features: list[torch.Tensor] | torch.Tensor,
        image_size: tuple[int, int],
        skip_blocks: bool = False,
    ) -> list[torch.Tensor]:
        target_size = (image_size[0] // self.downsample_ratio, image_size[1] // self.downsample_ratio)
        spatial_features = nn.functional.interpolate(
            spatial_features, size=target_size, mode="bilinear", align_corners=False
        )

        if isinstance(query_features, torch.Tensor):
            query_features = list(query_features.unbind(0))

        mask_logits = []
        if not skip_blocks:
            for block, query_feature in zip(self.blocks, query_features):
                spatial_features = block(spatial_features)
                spatial_features_proj = self.spatial_features_proj(spatial_features)
                query_feature = self.query_features_proj(self.query_features_block(query_feature))
                mask_logits.append(torch.einsum("bchw,bnc->bnhw", spatial_features_proj, query_feature) + self.bias)
        else:
            if len(query_features) != 1:
                raise ValueError("skip_blocks is only supported when `query_features` has length 1.")
            query_feature = self.query_features_proj(self.query_features_block(query_features[0]))
            mask_logits.append(torch.einsum("bchw,bnc->bnhw", spatial_features, query_feature) + self.bias)

        return mask_logits


class RfDetrConvEncoder(nn.Module):
    def __init__(self, config: RfDetrConfig):
        super().__init__()
        self.backbone = RfDetrWindowedDinov2Backbone(config.backbone_config)
        self.projector = RfDetrMultiScaleProjector(config)
        self._replace_projector_norms(self.projector)

    def _replace_projector_norms(self, module: nn.Module):
        for child in module.children():
            if isinstance(child, RfDetrConvNormLayer):
                child.norm = RfDetrLayerNorm(child.conv.out_channels, data_format="channels_first")
            else:
                self._replace_projector_norms(child)

    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor):
        features = self.backbone(pixel_values).feature_maps
        features = self.projector(features)
        out = []
        for feature_map in features:
            mask = nn.functional.interpolate(pixel_mask[None].float(), size=feature_map.shape[-2:]).to(torch.bool)[0]
            out.append((feature_map, mask))
        return out


class RfDetrModel(LwDetrModel):
    config_class = RfDetrConfig

    def __init__(self, config: RfDetrConfig):
        RfDetrPreTrainedModel.__init__(self, config)

        self.backbone = RfDetrConvEncoder(config)

        self.group_detr = config.group_detr
        self.num_queries = config.num_queries
        hidden_dim = config.d_model
        self.reference_point_embed = nn.Embedding(self.num_queries * self.group_detr, 4)
        self.query_feat = nn.Embedding(self.num_queries * self.group_detr, hidden_dim)

        self.decoder = RfDetrDecoder(config)

        self.enc_output = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(self.group_detr)])
        self.enc_output_norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.group_detr)])
        self.enc_out_bbox_embed = nn.ModuleList(
            [RfDetrMLPPredictionHead(config.d_model, config.d_model, 4, num_layers=3) for _ in range(self.group_detr)]
        )
        self.enc_out_class_embed = nn.ModuleList(
            [nn.Linear(config.d_model, config.num_labels) for _ in range(self.group_detr)]
        )

        self.post_init()


class RfDetrForObjectDetection(LwDetrForObjectDetection):
    config_class = RfDetrConfig

    def __init__(self, config: RfDetrConfig):
        RfDetrPreTrainedModel.__init__(self, config)

        self.model = RfDetrModel(config)
        self.class_embed = nn.Linear(config.d_model, config.num_labels)
        self.bbox_embed = RfDetrMLPPredictionHead(config.d_model, config.d_model, 4, num_layers=3)

        self.post_init()


@dataclass
class RfDetrInstanceSegmentationOutput(LwDetrObjectDetectionOutput):
    pred_masks: torch.FloatTensor | None = None


class RfDetrForInstanceSegmentation(RfDetrPreTrainedModel):
    config_class = RfDetrConfig
    _tied_weights_keys = None

    def __init__(self, config: RfDetrConfig):
        RfDetrPreTrainedModel.__init__(self, config)

        self.model = RfDetrModel(config)
        self.class_embed = nn.Linear(config.d_model, config.num_labels)
        self.bbox_embed = RfDetrMLPPredictionHead(config.d_model, config.d_model, 4, num_layers=3)
        self.segmentation_head = RfDetrSegmentationHead(
            in_dim=config.d_model,
            num_blocks=config.decoder_layers,
            bottleneck_ratio=config.segmentation_bottleneck_ratio,
            downsample_ratio=config.mask_downsample_ratio,
        )

        self.post_init()

    @can_return_tuple
    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        pixel_mask: torch.LongTensor | None = None,
        labels: list[dict] | None = None,
        **kwargs,
    ) -> RfDetrInstanceSegmentationOutput:
        if labels is not None:
            raise NotImplementedError("Loss computation for `RfDetrForInstanceSegmentation` is not implemented yet.")

        captured_features = {}

        def _capture_backbone_features(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output):
            captured_features["features"] = output

        feature_capture_handle = self.model.backbone.register_forward_hook(_capture_backbone_features)
        try:
            outputs = self.model(
                pixel_values,
                pixel_mask=pixel_mask,
                **kwargs,
            )
        finally:
            feature_capture_handle.remove()

        if "features" not in captured_features:
            raise RuntimeError("Could not capture RF-DETR backbone features required by the segmentation head.")
        features = captured_features["features"]

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

        intermediate_hidden_states = outputs.intermediate_hidden_states
        pred_masks_per_layer = self.segmentation_head(
            spatial_features=features[0][0],
            query_features=intermediate_hidden_states,
            image_size=(pixel_values.shape[-2], pixel_values.shape[-1]),
        )
        pred_masks = pred_masks_per_layer[-1]

        return RfDetrInstanceSegmentationOutput(
            logits=logits,
            pred_boxes=pred_boxes,
            pred_masks=pred_masks,
            init_reference_points=outputs.init_reference_points,
            last_hidden_state=outputs.last_hidden_state,
            intermediate_hidden_states=outputs.intermediate_hidden_states,
            intermediate_reference_points=outputs.intermediate_reference_points,
            enc_outputs_class=enc_outputs_class_logits,
            enc_outputs_coord_logits=enc_outputs_boxes_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


__all__ = [
    "RfDetrConfig",
    "RfDetrForInstanceSegmentation",
    "RfDetrForObjectDetection",
    "RfDetrInstanceSegmentationOutput",
    "RfDetrModel",
    "RfDetrPreTrainedModel",
    "RfDetrWindowedDinov2Backbone",
    "RfDetrWindowedDinov2Config",
    "RfDetrWindowedDinov2PreTrainedModel",
]
