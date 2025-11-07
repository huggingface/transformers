from typing import Optional, Union

import torch
from torch import nn

from ...modeling_outputs import BackboneOutput, BaseModelOutput
from ...utils import torch_int
from ..dinov2.configuration_dinov2 import Dinov2Config
from ..dinov2.modeling_dinov2 import (
    Dinov2Backbone,
    Dinov2Embeddings,
    Dinov2Encoder,
    Dinov2Layer,
    Dinov2PreTrainedModel,
)


class RfDetrDinov2Config(Dinov2Config):
    r"""
    This is the configuration class to store the configuration of a [`RfDetrDinov2Model`]. It is used to instantiate an
    RfDetrDinov2 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the DINOv2
    [facebook/dinov2-base](https://huggingface.co/facebook/dinov2-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

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
        num_windows (`int`, *optional*, defaults to 4):
            Number of windows to use for windowed attention. If 1, no windowed attention is used.
    Example:

    ```python
    >>> from transformers import RfDetrDinov2Config, RfDetrDinov2Model

    >>> # Initializing a RfDetrDinov2 base style configuration
    >>> configuration = RfDetrDinov2Config()

    >>> # Initializing a model (with random weights) from the base style configuration
    >>> model = RfDetrDinov2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "rf_detr_dinov2"

    def __init__(self, num_windows: int = 4, **super_kwargs):
        super().__init__(**super_kwargs)

        self.num_windows = num_windows
        window_block_indexes = set(range(self._out_indices[-1] + 1))
        window_block_indexes.difference_update(self._out_indices)
        window_block_indexes = list(window_block_indexes)
        self.window_block_indexes = window_block_indexes


def window_partition(
    embeddings: torch.Tensor, num_windows: int, patch_size: int, height: int, width: int
) -> torch.Tensor:
    batch_size = embeddings.shape[0]
    num_h_patches = height // patch_size
    num_w_patches = width // patch_size
    cls_token_with_pos_embed = embeddings[:, :1]
    pixel_tokens_with_pos_embed = embeddings[:, 1:]
    pixel_tokens_with_pos_embed = pixel_tokens_with_pos_embed.view(batch_size, num_h_patches, num_w_patches, -1)
    num_w_patches_per_window = num_w_patches // num_windows
    num_h_patches_per_window = num_h_patches // num_windows
    windowed_pixel_tokens = pixel_tokens_with_pos_embed.view(
        batch_size, num_windows, num_h_patches_per_window, num_windows, num_h_patches_per_window, -1
    )
    windowed_pixel_tokens = windowed_pixel_tokens.permute(0, 1, 3, 2, 4, 5)
    windowed_pixel_tokens = windowed_pixel_tokens.reshape(
        batch_size * num_windows**2, num_h_patches_per_window * num_w_patches_per_window, -1
    )
    windowed_cls_token_with_pos_embed = cls_token_with_pos_embed.repeat(num_windows**2, 1, 1)
    embeddings = torch.cat((windowed_cls_token_with_pos_embed, windowed_pixel_tokens), dim=1)
    return embeddings


class RfDetrDinov2Embeddings(Dinov2Embeddings):
    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing and interpolation at torch.float32 precision.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """

        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embeddings

        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = torch_int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        target_dtype = patch_pos_embed.dtype
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.to(torch.float32),
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).to(dtype=target_dtype)

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, _, height, width = pixel_values.shape
        target_dtype = self.patch_embeddings.projection.weight.dtype
        embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))

        if bool_masked_pos is not None:
            embeddings = torch.where(
                bool_masked_pos.unsqueeze(-1), self.mask_token.to(embeddings.dtype).unsqueeze(0), embeddings
            )

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)

        if self.config.num_windows > 1:
            # reshape for windows
            embeddings = window_partition(embeddings, self.config.num_windows, self.config.patch_size, height, width)
        embeddings = self.dropout(embeddings)

        return embeddings


def window_unpartition_before_attention(hidden_states: torch.Tensor, num_windows: int) -> torch.Tensor:
    batch_size, seq_len, channels = hidden_states.shape
    num_windows_squared = num_windows**2
    hidden_states = hidden_states.view(batch_size // num_windows_squared, num_windows_squared * seq_len, channels)
    return hidden_states


def window_partition_after_attention(
    hidden_states: torch.Tensor, self_attention_output: torch.Tensor, num_windows: int
) -> torch.Tensor:
    batch_size, seq_len, channels = hidden_states.shape
    num_windows_squared = num_windows**2
    self_attention_output = self_attention_output.view(
        batch_size * num_windows_squared, seq_len // num_windows_squared, channels
    )
    return self_attention_output


class RfDetrDinov2Layer(Dinov2Layer):
    def __init__(self, config: RfDetrDinov2Config, layer_idx: int):
        super().__init__(config)
        self.num_windows = config.num_windows
        self.global_attention = layer_idx not in config.window_block_indexes

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Union[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor]]:
        shortcut = hidden_states
        if self.global_attention:
            hidden_states = window_unpartition_before_attention(hidden_states, self.num_windows)

        hidden_states_norm = self.norm1(hidden_states)
        self_attention_output = self.attention(hidden_states_norm)

        if self.global_attention:
            self_attention_output = window_partition_after_attention(
                hidden_states, self_attention_output, self.num_windows
            )

        self_attention_output = self.layer_scale1(self_attention_output)

        # first residual connection
        hidden_states = self.drop_path(self_attention_output) + shortcut

        # in Dinov2, layernorm is also applied after self-attention
        layer_output = self.norm2(hidden_states)
        layer_output = self.mlp(layer_output)
        layer_output = self.layer_scale2(layer_output)

        # second residual connection
        layer_output = self.drop_path(layer_output) + hidden_states

        return layer_output


class RfDetrDinov2Encoder(Dinov2Encoder):
    def __init__(self, config: RfDetrDinov2Config):
        super().__init__(config)
        self.layer = nn.ModuleList([RfDetrDinov2Layer(config, i) for i in range(config.num_hidden_layers)])


class RfDetrDinov2PreTrainedModel(Dinov2PreTrainedModel):
    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
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
        elif isinstance(module, RfDetrDinov2Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.cls_token.dtype)

            module.mask_token.data.zero_()
        elif isinstance(module, RfDetrDinov2LayerScale):  # noqa: F821
            module.lambda1.data.fill_(self.config.layerscale_value)


def window_unpartition(
    hidden_state: torch.Tensor,
    num_windows: int,
    num_h_patches: int,
    num_w_patches: int,
) -> torch.Tensor:
    hidden_batch_size, seq_len, channels = hidden_state.shape
    num_windows_squared = num_windows**2
    num_h_patches_per_window = num_h_patches // num_windows
    num_w_patches_per_window = num_w_patches // num_windows
    hidden_state = hidden_state.reshape(
        hidden_batch_size // num_windows_squared, num_windows_squared * seq_len, channels
    )
    hidden_state = hidden_state.view(
        hidden_batch_size // num_windows_squared,
        num_windows,
        num_windows,
        num_h_patches_per_window,
        num_w_patches_per_window,
        channels,
    )
    hidden_state = hidden_state.permute(0, 1, 3, 2, 4, 5)
    return hidden_state


class RfDetrDinov2Backbone(Dinov2Backbone):
    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> BackboneOutput:
        """
        Returns:

        Examples:
        Returns:

        Examples:


        ```python
        >>> from transformers import AutoImageProcessor, AutoBackbone
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        >>> model = AutoBackbone.from_pretrained(
        ...     "facebook/dinov2-base", out_features=["stage2", "stage5", "stage8", "stage11"]
        ... )

        >>> inputs = processor(image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> feature_maps = outputs.feature_maps
        >>> list(feature_maps[-1].shape)
        [1, 768, 16, 16]
        ```"""
        if output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states

        embedding_output = self.embeddings(pixel_values)

        output: BaseModelOutput = self.encoder(embedding_output, output_hidden_states=True)
        hidden_states = output.hidden_states

        feature_maps = ()
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                if self.config.apply_layernorm:
                    hidden_state = self.layernorm(hidden_state)
                if self.config.reshape_hidden_states:
                    hidden_state = hidden_state[:, 1:]
                    # this was actually a bug in the original implementation that we copied here,
                    # cause normally the order is height, width
                    batch_size, _, height, width = pixel_values.shape
                    patch_size = self.config.patch_size

                    num_h_patches = height // patch_size
                    num_w_patches = width // patch_size
                    hidden_batch_size, seq_len, channels = hidden_state.shape

                    if self.config.num_windows > 1:
                        hidden_state = window_unpartition(
                            hidden_state, self.config.num_windows, num_h_patches, num_w_patches
                        )

                    hidden_state = hidden_state.reshape(batch_size, num_h_patches, num_w_patches, -1)
                    hidden_state = hidden_state.permute(0, 3, 1, 2).contiguous()

                feature_maps += (hidden_state,)

        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=hidden_states if output_hidden_states else None,
        )


__all__ = [
    "RfDetrDinov2Config",
    "RfDetrDinov2Backbone",
    "RfDetrDinov2PreTrainedModel",
]
