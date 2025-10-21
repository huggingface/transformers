import math
from collections.abc import Callable
from typing import Optional

import torch
from torch import nn

from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BackboneOutput, BaseModelOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.generic import check_model_inputs
from ..vit.modeling_vit import ViTAttention, ViTEncoder, ViTSelfAttention, eager_attention_forward
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
    [stevenbucaille/lwdetr_small_60e_coco](https://huggingface.co/stevenbucaille/lwdetr_small_60e_coco) architecture.

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
    attribute_map = {
        "attention_probs_dropout_prob": "dropout_prob",
        "decoder_self_attention_heads": "num_attention_heads",
    }

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


class LwDetrViTSelfAttention(ViTSelfAttention):
    def __init__(self, config: LwDetrViTConfig):
        super().__init__(config)
        del self.key
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=False)

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
    ) -> BaseModelOutput:
        list_hidden_states = [hidden_states]
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, **kwargs)
            list_hidden_states.append(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=tuple(list_hidden_states))


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
        "hidden_states": ["LwDetrViTLayer", "LwDetrViTEncoder"],
        "attentions": LwDetrViTSelfAttention,
    }

    def _init_weights(self, module) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        elif isinstance(module, LwDetrViTEmbeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.position_embeddings.dtype)
        elif isinstance(module, LwDetrViTLayer):
            nn.init.constant_(module.gamma_1, self.config.cae_init_values)
            nn.init.constant_(module.gamma_2, self.config.cae_init_values)


class LwDetrViTBackbone(VitDetBackbone):
    @can_return_tuple
    @auto_docstring
    def forward(self, pixel_values: torch.Tensor = None, **kwargs: Unpack[TransformersKwargs]) -> BackboneOutput:
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
        # (batch_size, height, width, channels) -> (batch_size*16, window_height*window_width, channels)
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
            .reshape(batch_size * 16, window_height * window_width, channels)
        )

        encoder_outputs = self.encoder(hidden_states, **kwargs)
        hidden_states = encoder_outputs.hidden_states

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

        output_hidden_states = self.config.output_hidden_states or kwargs.get("output_hidden_states", False)
        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            attentions=encoder_outputs.attentions,
        )


__all__ = ["LwDetrViTConfig", "LwDetrViTPreTrainedModel", "LwDetrViTBackbone"]
