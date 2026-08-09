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
from huggingface_hub.dataclasses import strict
from torch import nn

from ...backbone_utils import BackboneMixin, filter_output_hidden_states
from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import BackboneOutput, BaseModelOutput, BaseModelOutputWithPooling
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, is_tracing
from ...utils.generic import can_return_tuple, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..dinov2.configuration_dinov2 import Dinov2Config
from ..dinov2.modeling_dinov2 import Dinov2MLP
from ..swin.modeling_swin import SwinDropPath
from ..vit.modeling_vit import ViTAttention, ViTLayer, ViTPatchEmbeddings, ViTPreTrainedModel


@auto_docstring(checkpoint="facebook/pixio-huge")
@strict
class PixioConfig(Dinov2Config):
    r"""
    apply_layernorm (`bool`, *optional*, defaults to `True`):
        Whether to apply layer normalization to the feature maps in case the model is used as backbone.
    reshape_hidden_states (`bool`, *optional*, defaults to `True`):
        Whether to reshape the feature maps to 4D tensors of shape `(batch_size, hidden_size, height, width)` in
        case the model is used as backbone. If `False`, the feature maps will be 3D tensors of shape `(batch_size,
        seq_len, hidden_size)`.
    n_cls_tokens (`int`, *optional*, defaults to 8):
        Number of class tokens in the Transformer encoder.

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

    hidden_size: int = 1280
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    n_cls_tokens: int = 8
    image_size: int | list[int] | tuple[int, int] = 256
    patch_size: int | list[int] | tuple[int, int] = 16

    layerscale_value = AttributeError()
    use_swiglu_ffn = AttributeError()
    use_mask_token = AttributeError()


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
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images. This method is also adapted to support tracing and interpolation at torch.float32 precision.

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


class PixioMLP(Dinov2MLP):
    pass


class PixioDropPath(SwinDropPath):
    pass


class PixioLayer(ViTLayer):
    def __init__(self, config: PixioConfig):
        super().__init__(config)
        self.drop_path = PixioDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layernorm_before(hidden_states)
        hidden_states, _ = self.attention(hidden_states, attention_mask, **kwargs)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.drop_path(hidden_states) + residual

        residual = hidden_states
        hidden_states = self.layernorm_after(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.drop_path(hidden_states) + residual

        return hidden_states


class PixioPreTrainedModel(ViTPreTrainedModel):
    pass


@auto_docstring
class PixioModel(PixioPreTrainedModel):
    def __init__(self, config: PixioConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = PixioEmbeddings(config)
        self.layers = nn.ModuleList([PixioLayer(config) for _ in range(config.num_hidden_layers)])

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.post_init()

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        embedding_output = self.embeddings(pixel_values)
        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=embedding_output,
            attention_mask=attention_mask,
        )
        hidden_states = embedding_output
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, **kwargs)
        hidden_states = self.layernorm(hidden_states)
        pooled_output = hidden_states[:, : self.embeddings.n_cls_tokens, :].mean(dim=1)

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=pooled_output,
        )


@auto_docstring(
    custom_intro="""
    Pixio backbone, to be used with frameworks like DETR and MaskFormer.
    """
)
class PixioBackbone(BackboneMixin, PixioPreTrainedModel):
    def __init__(self, config: PixioConfig):
        super().__init__(config)

        self.num_features = [config.hidden_size for _ in range(config.num_hidden_layers + 1)]
        self.pixio = PixioModel(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.post_init()

    @can_return_tuple
    @filter_output_hidden_states
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
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
        kwargs["output_hidden_states"] = True  # required to extract layers for the stages

        output: BaseModelOutput = self.pixio(pixel_values, attention_mask, **kwargs)
        hidden_states = output.hidden_states

        feature_maps = []
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                if self.config.apply_layernorm:
                    hidden_state = self.layernorm(hidden_state)
                if self.config.reshape_hidden_states:
                    hidden_state = hidden_state[:, self.pixio.embeddings.n_cls_tokens :]
                    batch_size, _, height, width = pixel_values.shape
                    patch_size = self.config.patch_size
                    hidden_state = hidden_state.reshape(batch_size, height // patch_size, width // patch_size, -1)
                    hidden_state = hidden_state.permute(0, 3, 1, 2).contiguous()
                feature_maps.append(hidden_state)

        return BackboneOutput(
            feature_maps=tuple(feature_maps),
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )


__all__ = ["PixioConfig", "PixioModel", "PixioPreTrainedModel", "PixioBackbone"]
