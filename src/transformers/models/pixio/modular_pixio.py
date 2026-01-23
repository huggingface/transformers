# Copyright 2025 Meta AI and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Pixio model."""

import torch
from torch import nn

from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BackboneOutput, BaseModelOutput, BaseModelOutputWithPooling
from ...utils import auto_docstring, is_tracing, logging
from ...utils.generic import check_model_inputs
from ..dinov2.configuration_dinov2 import Dinov2Config
from ..dinov2.modeling_dinov2 import (
    Dinov2Backbone,
    Dinov2DropPath,
    Dinov2MLP,
)
from ..vit.modeling_vit import ViTAttention, ViTPatchEmbeddings, ViTPreTrainedModel


logger = logging.get_logger(__name__)


class PixioConfig(Dinov2Config):
    r"""
    This is the configuration class to store the configuration of a [`PixioModel`]. It is used to instantiate a
    Pixio model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the ViT
    [facebook/pixio-huge](https://huggingface.co/facebook/pixio-huge) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 1280):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        mlp_ratio (`int`, *optional*, defaults to 4):
            Ratio of the hidden size of the MLPs relative to the `hidden_size`.
        n_cls_tokens (`int`, *optional*, defaults to 8):
            Number of class tokens in the Transformer encoder.
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
        image_size (`int`, *optional*, defaults to 256):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            Stochastic depth rate per sample (when applied in the main path of residual layers).
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

    Example:

    ```python
    >>> from transformers import PixioConfig, PixioModel

    >>> # Initializing a Pixio pixio-huge style configuration
    >>> configuration = PixioConfig()

    >>> # Initializing a model (with random weights) from the pixio-huge style configuration
    >>> model = PixioModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "pixio"

    def __init__(
        self,
        hidden_size=1280,
        num_hidden_layers=32,
        num_attention_heads=16,
        mlp_ratio=4,
        n_cls_tokens=8,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-6,
        image_size=256,
        patch_size=16,
        num_channels=3,
        qkv_bias=True,
        drop_path_rate=0.0,
        out_features=None,
        out_indices=None,
        apply_layernorm=True,
        reshape_hidden_states=True,
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
            drop_path_rate=drop_path_rate,
            apply_layernorm=apply_layernorm,
            reshape_hidden_states=reshape_hidden_states,
        )

        self.n_cls_tokens = n_cls_tokens

        del self.layerscale_value
        del self.use_swiglu_ffn
        del self.use_mask_token


class PixioPatchEmbeddings(ViTPatchEmbeddings):
    pass


class PixioEmbeddings(nn.Module):
    """
    Construct the CLS tokens, position and patch embeddings.
    """

    def __init__(self, config: PixioConfig) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, config.n_cls_tokens, config.hidden_size))
        self.mask_token = None
        self.patch_embeddings = PixioPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + config.n_cls_tokens, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.n_cls_tokens = config.n_cls_tokens
        self.patch_size = config.patch_size
        self.config = config

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support tracing and interpolation at torch.float32 precision.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """
        num_patches = embeddings.shape[1] - self.n_cls_tokens
        num_positions = self.position_embeddings.shape[1] - self.n_cls_tokens

        if not is_tracing() and num_patches == num_positions and height == width:
            return self.position_embeddings

        class_pos_embed = self.position_embeddings[:, : self.n_cls_tokens]
        patch_pos_embed = self.position_embeddings[:, self.n_cls_tokens :]

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        target_dtype = patch_pos_embed.dtype
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.to(torch.float32),
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        ).to(dtype=target_dtype)

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = pixel_values.shape
        target_dtype = self.patch_embeddings.projection.weight.dtype
        embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)

        embeddings = self.dropout(embeddings)

        return embeddings


class PixioAttention(ViTAttention):
    pass


class PixioDropPath(Dinov2DropPath):
    pass


class PixioMLP(Dinov2MLP):
    pass


class PixioLayer(GradientCheckpointingLayer):
    def __init__(self, config: PixioConfig) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = PixioAttention(config)
        self.drop_path = PixioDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = PixioMLP(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states_norm = self.norm1(hidden_states)
        self_attention_output = self.attention(hidden_states_norm)

        hidden_states = self.drop_path(self_attention_output) + hidden_states

        layer_output = self.norm2(hidden_states)
        layer_output = self.mlp(layer_output)

        layer_output = self.drop_path(layer_output) + hidden_states

        return layer_output


class PixioEncoder(nn.Module):
    def __init__(self, config: PixioConfig):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([PixioLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor, output_hidden_states: bool = False) -> BaseModelOutput:
        all_hidden_states = [hidden_states] if output_hidden_states else None
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states)
            if all_hidden_states:
                all_hidden_states.append(hidden_states)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=tuple(all_hidden_states) if all_hidden_states else None,
        )


class PixioPreTrainedModel(ViTPreTrainedModel):
    pass


@auto_docstring
class PixioModel(PixioPreTrainedModel):
    def __init__(self, config: PixioConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = PixioEmbeddings(config)
        self.encoder = PixioEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.post_init()

    def get_input_embeddings(self) -> PixioPatchEmbeddings:
        return self.embeddings.patch_embeddings

    @check_model_inputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPooling:
        if output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        embedding_output = self.embeddings(pixel_values)

        encoder_outputs: BaseModelOutput = self.encoder(embedding_output, output_hidden_states=output_hidden_states)
        sequence_output = encoder_outputs.last_hidden_state
        sequence_output = self.layernorm(sequence_output)
        pooled_output = sequence_output[:, : self.embeddings.n_cls_tokens, :].mean(dim=1)

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )


@auto_docstring(
    custom_intro="""
    Pixio backbone, to be used with frameworks like DETR and MaskFormer.
    """
)
class PixioBackbone(Dinov2Backbone):
    @check_model_inputs
    @auto_docstring
    def forward(
        self, pixel_values: torch.Tensor, output_hidden_states: bool | None = None, **kwargs
    ) -> BackboneOutput:
        r"""
        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AutoBackbone
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> processor = AutoImageProcessor.from_pretrained("facebook/pixio-huge")
        >>> model = AutoBackbone.from_pretrained(
        ...     "facebook/pixio-huge", out_features=["stage7", "stage15", "stage23", "stage31"]
        ... )

        >>> inputs = processor(image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> feature_maps = outputs.feature_maps
        >>> list(feature_maps[-1].shape)
        [1, 1280, 16, 16]
        ```"""
        if output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states

        embedding_output = self.embeddings(pixel_values)
        output: BaseModelOutput = self.encoder(embedding_output, output_hidden_states=True)
        hidden_states = output.hidden_states

        feature_maps = []
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                if self.config.apply_layernorm:
                    hidden_state = self.layernorm(hidden_state)
                if self.config.reshape_hidden_states:
                    hidden_state = hidden_state[:, self.embeddings.n_cls_tokens :]
                    batch_size, _, height, width = pixel_values.shape
                    patch_size = self.config.patch_size
                    hidden_state = hidden_state.reshape(batch_size, height // patch_size, width // patch_size, -1)
                    hidden_state = hidden_state.permute(0, 3, 1, 2).contiguous()
                feature_maps.append(hidden_state)

        return BackboneOutput(
            feature_maps=tuple(feature_maps),
            hidden_states=hidden_states if output_hidden_states else None,
        )


__all__ = ["PixioConfig", "PixioModel", "PixioPreTrainedModel", "PixioBackbone"]
