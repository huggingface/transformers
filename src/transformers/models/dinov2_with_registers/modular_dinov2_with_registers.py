# Copyright 2024 Meta Inc. and the HuggingFace Inc. team. All rights reserved.
#
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


import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ....transformers.models.dinov2.modeling_dinov2 import (
    Dinov2Backbone,
    Dinov2Encoder,
    Dinov2ForImageClassification,
    Dinov2Model,
    Dinov2PatchEmbeddings,
    Dinov2PreTrainedModel,
)
from ... import initialization as init
from ...backbone_utils import BackboneConfigMixin
from ...configuration_utils import PreTrainedConfig
from ...modeling_outputs import BackboneOutput, BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging, torch_int


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="facebook/dinov2-with-registers-base")
@strict
class Dinov2WithRegistersConfig(BackboneConfigMixin, PreTrainedConfig):
    r"""
    layerscale_value (`float`, *optional*, defaults to 1.0):
        Initial value to use for layer scale.
    use_swiglu_ffn (`bool`, *optional*, defaults to `False`):
        Whether to use the SwiGLU feedforward neural network.
    num_register_tokens (`int`, *optional*, defaults to 4):
        Number of register tokens to use.
    apply_layernorm (`bool`, *optional*, defaults to `True`):
        Whether to apply layer normalization to the feature maps in case the model is used as backbone.
    reshape_hidden_states (`bool`, *optional*, defaults to `True`):
        Whether to reshape the feature maps to 4D tensors of shape `(batch_size, hidden_size, height, width)` in
        case the model is used as backbone. If `False`, the feature maps will be 3D tensors of shape `(batch_size,
        seq_len, hidden_size)`.

    Example:

    ```python
    >>> from transformers import Dinov2WithRegistersConfig, Dinov2WithRegistersModel

    >>> # Initializing a Dinov2WithRegisters base style configuration
    >>> configuration = Dinov2WithRegistersConfig()

    >>> # Initializing a model (with random weights) from the base style configuration
    >>> model = Dinov2WithRegistersModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "dinov2_with_registers"

    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    mlp_ratio: int = 4
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-6
    image_size: int | list[int] | tuple[int, int] = 224
    patch_size: int | list[int] | tuple[int, int] = 16
    num_channels: int = 3
    qkv_bias: bool = True
    layerscale_value: float = 1.0
    drop_path_rate: float = 0.0
    use_swiglu_ffn: bool = False
    num_register_tokens: int = 4
    _out_features: list[str] | None = None
    _out_indices: list[int] | None = None
    apply_layernorm: bool = True
    reshape_hidden_states: bool = True

    def __post_init__(self, **kwargs):
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, self.num_hidden_layers + 1)]
        self.set_output_features_output_indices(
            out_indices=kwargs.pop("out_indices", None), out_features=kwargs.pop("out_features", None)
        )
        super().__post_init__(**kwargs)


class Dinov2WithRegistersPatchEmbeddings(Dinov2PatchEmbeddings):
    pass


class Dinov2WithRegistersEmbeddings(nn.Module):
    """
    Construct the CLS token, mask token, register tokens, position and patch embeddings.
    """

    def __init__(self, config: Dinov2WithRegistersConfig) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1, config.hidden_size))
        self.register_tokens = nn.Parameter(torch.zeros(1, config.num_register_tokens, config.hidden_size))
        self.patch_embeddings = Dinov2WithRegistersPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.patch_size = config.patch_size
        self.config = config

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images. This implementation supports torch.jit tracing while maintaining backwards compatibility
        with the original implementation.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
        - https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py
        """
        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1

        # Skip interpolation for matching dimensions (unless tracing)
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embeddings

        # Handle class token and patch embeddings separately
        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]

        # Calculate new dimensions
        height = height // self.config.patch_size
        width = width // self.config.patch_size

        # Reshape for interpolation
        sqrt_num_positions = torch_int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        # Store original dtype for restoration after interpolation
        target_dtype = patch_pos_embed.dtype

        # Interpolate at float32 precision
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.to(dtype=torch.float32),
            size=(torch_int(height), torch_int(width)),  # Explicit size instead of scale_factor
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).to(dtype=target_dtype)

        # Validate output dimensions if not tracing
        if not torch.jit.is_tracing():
            if int(height) != patch_pos_embed.shape[-2] or int(width) != patch_pos_embed.shape[-1]:
                raise ValueError("Width or height does not match with the interpolated position embeddings")

        # Reshape back to original format
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        # Combine class and patch embeddings
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: torch.Tensor | None = None) -> torch.Tensor:
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

        # add register tokens
        embeddings = torch.cat(
            (embeddings[:, :1], self.register_tokens.expand(embeddings.shape[0], -1, -1), embeddings[:, 1:]), dim=1
        )

        embeddings = self.dropout(embeddings)

        return embeddings


class Dinov2WithRegistersPreTrainedModel(Dinov2PreTrainedModel):
    @torch.no_grad()
    def _init_weights(self, module: nn.Linear | nn.Conv2d | nn.LayerNorm) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            init.trunc_normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            init.zeros_(module.bias)
            init.ones_(module.weight)
        elif isinstance(module, Dinov2WithRegistersEmbeddings):
            init.trunc_normal_(module.position_embeddings, mean=0.0, std=self.config.initializer_range)
            init.trunc_normal_(module.cls_token, mean=0.0, std=self.config.initializer_range)
            init.zeros_(module.mask_token)
            init.zeros_(module.register_tokens)
        elif isinstance(module, Dinov2WithRegistersLayerScale):  # noqa: F821
            init.constant_(module.lambda1, self.config.layerscale_value)


class Dinov2WithRegistersEncoder(Dinov2Encoder):
    pass


class Dinov2WithRegistersModel(Dinov2Model):
    pass


class Dinov2WithRegistersForImageClassification(Dinov2ForImageClassification):
    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ImageClassifierOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        outputs: BaseModelOutputWithPooling = self.dinov2_with_registers(pixel_values, **kwargs)
        sequence_output = outputs.last_hidden_state  # batch_size, sequence_length, hidden_size

        cls_token = sequence_output[:, 0]
        # cls and register tokens should not be included in patch tokens variable
        patch_tokens = sequence_output[:, 1 + self.config.num_register_tokens :]

        linear_input = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)
        logits = self.classifier(linear_input)

        loss = None
        if labels is not None:
            loss = self.loss_function(labels, logits, self.config, **kwargs)

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Dinov2WithRegistersBackbone(Dinov2Backbone):
    def __init__(self, config):
        super().__init__(config)

        self.num_register_tokens = config.num_register_tokens
        self.num_features = [config.hidden_size for _ in range(config.num_hidden_layers + 1)]
        self.embeddings = Dinov2WithRegistersEmbeddings(config)
        self.encoder = Dinov2WithRegistersEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> Dinov2WithRegistersPatchEmbeddings:
        return self.embeddings.patch_embeddings

    def forward(
        self,
        pixel_values: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BackboneOutput:
        r"""
        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AutoBackbone
        >>> import torch
        >>> from PIL import Image
        >>> import httpx
        >>> from io import BytesIO

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read()))

        >>> processor = AutoImageProcessor.from_pretrained("facebook/dinov2-with-registers-base")
        >>> model = AutoBackbone.from_pretrained(
        ...     "facebook/dinov2-with-registers-base", out_features=["stage2", "stage5", "stage8", "stage11"]
        ... )

        >>> inputs = processor(image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> feature_maps = outputs.feature_maps
        >>> list(feature_maps[-1].shape)
        [1, 768, 16, 16]
        ```"""
        kwargs["output_hidden_states"] = True  # required to extract layers for the stages

        embedding_output = self.embeddings(pixel_values)
        output: BaseModelOutput = self.encoder(embedding_output, **kwargs)
        hidden_states = output.hidden_states

        feature_maps = []
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                if self.config.apply_layernorm:
                    hidden_state = self.layernorm(hidden_state)
                if self.config.reshape_hidden_states:
                    hidden_state = hidden_state[:, 1 + self.num_register_tokens :]
                    # this was actually a bug in the original implementation that we copied here,
                    # cause normally the order is height, width
                    batch_size, _, height, width = pixel_values.shape
                    patch_size = self.config.patch_size
                    hidden_state = hidden_state.reshape(batch_size, height // patch_size, width // patch_size, -1)
                    hidden_state = hidden_state.permute(0, 3, 1, 2).contiguous()
                feature_maps.append(hidden_state)

        return BackboneOutput(
            feature_maps=tuple(feature_maps),
            hidden_states=hidden_states,
            attentions=output.attentions,
        )


__all__ = [
    "Dinov2WithRegistersConfig",
    "Dinov2WithRegistersPreTrainedModel",
    "Dinov2WithRegistersModel",
    "Dinov2WithRegistersForImageClassification",
    "Dinov2WithRegistersBackbone",
]
