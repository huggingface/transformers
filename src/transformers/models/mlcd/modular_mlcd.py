# Copyright 2025 The HuggingFace Inc. team.
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
from collections.abc import Callable

import torch
import torch.nn as nn

from ... import initialization as init
from ...configuration_utils import PreTrainedConfig
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import check_model_inputs
from ..clip.modeling_clip import (
    CLIPMLP,
    CLIPAttention,
    CLIPEncoder,
    CLIPEncoderLayer,
    CLIPVisionEmbeddings,
    CLIPVisionModel,
    CLIPVisionTransformer,
)
from ..llama.modeling_llama import eager_attention_forward
from ..qwen2_vl.modeling_qwen2_vl import VisionRotaryEmbedding, apply_rotary_pos_emb_vision


logger = logging.get_logger(__name__)


class MLCDVisionConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MLCDVisionModel`]. It is used to instantiate a MLCD
    vision encoder according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the vision encoder of the MLCD
    [DeepGlint-AI/mlcd-vit-bigG-patch14-336](https://huggingface.co/DeepGlint-AI/mlcd-vit-bigG-patch14-336) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 1664):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 8192):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        projection_dim (`int`, *optional*, defaults to 1024):
            Dimensionality of text and vision projection layers.
        num_hidden_layers (`int`, *optional*, defaults to 48):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        image_size (`int`, *optional*, defaults to 336):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 14):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).

    Example:

    ```python
    >>> from transformers import MLCDVisionConfig, MLCDVisionModel

    >>> # Initializing a MLCDVisionConfig with DeepGlint-AI/mlcd-vit-bigG-patch14-336 style configuration
    >>> configuration = MLCDVisionConfig()

    >>> # Initializing a MLCDVisionModel (with random weights) from the DeepGlint-AI/mlcd-vit-bigG-patch14-336 style configuration
    >>> model = MLCDVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "mlcd_vision_model"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size=1664,
        intermediate_size=8192,
        num_hidden_layers=48,
        num_attention_heads=16,
        num_key_value_groups=1,
        num_channels=3,
        image_size=336,
        patch_size=14,
        hidden_act="gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_groups = num_key_value_groups
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act


class MLCDMLP(CLIPMLP):
    pass


class MLCDRotaryEmbedding(VisionRotaryEmbedding):
    def forward(self, num_patches_height: int, num_patches_width: int) -> torch.Tensor:
        """
        Calculate the Rotary Position Embedding (RoPE) for MLCDVisionModel based on the grid size.

        Args:
            num_patches_height (int): Number of patches in the height dimension.
            num_patches_width (int): Number of patches in the width dimension.

        Returns:
            torch.Tensor: Rotary positional embeddings for the given grid size.
        """
        # Generate position IDs for height and width dimensions
        hpos_ids = (
            torch.arange(num_patches_height, device=self.inv_freq.device).unsqueeze(1).expand(-1, num_patches_width)
        )
        wpos_ids = (
            torch.arange(num_patches_width, device=self.inv_freq.device).unsqueeze(0).expand(num_patches_height, -1)
        )

        # Flatten and stack the position IDs
        pos_ids = torch.stack([hpos_ids.flatten(), wpos_ids.flatten()], dim=-1)

        # Generate the full rotary positional embeddings for the maximum grid size
        max_grid_size = max(num_patches_height, num_patches_width)
        seq = torch.arange(max_grid_size, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        rotary_pos_emb_full = torch.outer(seq, self.inv_freq)

        # Select and flatten the embeddings based on the position IDs
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)

        return rotary_pos_emb


class MLCDVisionEmbeddings(CLIPVisionEmbeddings):
    def __init__(self, config: MLCDVisionConfig):
        super().__init__(config)
        del self.position_embedding

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        # patch_embeds -> shape = [batch, width, grid, grid]
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)

        return embeddings


class MLCDAttention(CLIPAttention):
    """Multi-headed attention with RoPE. Refer to papers:
    - Attention is all you need:
        https://huggingface.co/papers/1706.03762
    - RoFormer: Enhanced Transformer with Rotary Position Embedding:
        https://huggingface.co/papers/2104.09864
    """

    def __init__(self, config: MLCDVisionConfig):
        super().__init__(config)
        self.num_key_value_groups = config.num_key_value_groups
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_length = hidden_states.shape[:-1]

        # Each of shape: [batch_size, seq_length, num_heads, head_dim]
        query_states = self.q_proj(hidden_states).reshape((batch_size, seq_length, self.num_heads, self.head_dim))
        key_states = self.k_proj(hidden_states).reshape((batch_size, seq_length, self.num_heads, self.head_dim))
        value_states = self.v_proj(hidden_states).reshape((batch_size, seq_length, self.num_heads, self.head_dim))

        # Apply positional embeddings
        cos = position_embeddings[0].unsqueeze(0).float()
        sin = position_embeddings[1].unsqueeze(0).float()
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        # Each of shape: [batch_size, num_heads, seq_length, head_dim]
        query_states = query_states.permute(0, 2, 1, 3).contiguous()
        key_states = key_states.permute(0, 2, 1, 3).contiguous()
        value_states = value_states.permute(0, 2, 1, 3).contiguous()

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=self.scale,
            is_causal=self.is_causal,
            **kwargs,
        )

        attn_output = attn_output.permute(1, 0, 2, 3).contiguous()  # [seq_length, batch_size, num_heads, head_dim]
        attn_output = attn_output.view(seq_length, batch_size, -1)  # [seq_length, batch_size, embedding_dim]
        attn_output = self.out_proj(attn_output)
        attn_output = attn_output.permute(1, 0, 2).contiguous()  # [batch_size, seq_length, embedding_dim]
        return attn_output, attn_weights


class MLCDEncoderLayer(CLIPEncoderLayer):
    def __init__(self, config: MLCDVisionConfig):
        super().__init__(config)
        self.self_attn = MLCDAttention(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
                Represents the hidden states from the previous layer or the input embeddings.
            position_embeddings (`tuple[torch.Tensor, torch.Tensor]`):
                A tuple of two tensors, each of shape `(batch, seq_len, embed_dim)`.
                Represents absolute positional embeddings for the query and key in the attention mechanism.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(batch, 1, q_len, k_v_seq_len)` where padding elements are indicated by very large negative values.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class MLCDEncoder(CLIPEncoder):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`MLCDEncoderLayer`].

    Args:
        config: MLCDVisionConfig
    """

    def __init__(self, config: MLCDVisionConfig):
        """Overwrite dummy `MLCDConfig` to `MLCDVisionConfig`."""
        super().__init__(config)

    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutput:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            position_embeddings (`tuple[torch.Tensor, torch.Tensor]`):
                A tuple of two tensors, each of shape `(batch, seq_len, embed_dim)`.
                Represents absolute positional embeddings for the query and key in the attention mechanism.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
        """
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                position_embeddings,
                attention_mask,
                **kwargs,
            )

        return BaseModelOutput(
            last_hidden_state=hidden_states,
        )


@auto_docstring
class MLCDPreTrainedModel(PreTrainedModel):
    config: MLCDVisionConfig
    base_model_prefix = "mlcd"
    supports_gradient_checkpointing = True
    accepts_loss_kwargs = False
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": MLCDEncoderLayer,
        "attentions": MLCDAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if isinstance(module, MLCDVisionEmbeddings):
            factor = self.config.initializer_factor
            init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor)
            init.normal_(module.patch_embedding.weight, std=module.config.initializer_range * factor)
            init.copy_(module.position_ids, torch.arange(module.position_ids.shape[-1]).expand((1, -1)))
        elif isinstance(module, MLCDAttention):
            factor = self.config.initializer_factor
            in_proj_std = (module.embed_dim**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            out_proj_std = (module.embed_dim**-0.5) * factor
            init.normal_(module.q_proj.weight, std=in_proj_std)
            init.normal_(module.k_proj.weight, std=in_proj_std)
            init.normal_(module.v_proj.weight, std=in_proj_std)
            init.normal_(module.out_proj.weight, std=out_proj_std)
        elif isinstance(module, MLCDMLP):
            factor = self.config.initializer_factor
            in_proj_std = (module.config.hidden_size**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            init.normal_(module.fc1.weight, std=fc_std)
            init.normal_(module.fc2.weight, std=in_proj_std)
        elif isinstance(module, MLCDVisionTransformer):
            factor = self.config.initializer_factor
            pos_emb_std = (module.config.hidden_size // module.config.num_attention_heads // 2) ** -0.5 * factor
            init.normal_(module.class_pos_emb, mean=0.0, std=pos_emb_std)
        elif isinstance(module, nn.LayerNorm):
            init.zeros_(module.bias)
            init.ones_(module.weight)
        elif isinstance(module, nn.Linear) and module.bias is not None:
            init.zeros_(module.bias)
        elif isinstance(module, MLCDRotaryEmbedding):
            inv_freq = 1.0 / (module.theta ** (torch.arange(0, module.dim, 2, dtype=torch.float) / module.dim))
            init.copy_(module.inv_freq, inv_freq)


class MLCDVisionTransformer(CLIPVisionTransformer):
    def __init__(self, config: MLCDVisionConfig):
        super().__init__(config)
        self.vision_rotary_embedding = MLCDRotaryEmbedding(config.hidden_size // config.num_attention_heads // 2)
        self.class_pos_emb = nn.Parameter(torch.randn(1, config.hidden_size // config.num_attention_heads // 2))

    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        num_patches_height = pixel_values.shape[-2] // self.config.patch_size
        num_patches_width = pixel_values.shape[-1] // self.config.patch_size
        rotary_pos_emb = self.vision_rotary_embedding(num_patches_height, num_patches_width)
        rotary_pos_emb = rotary_pos_emb.to(self.class_pos_emb.device)
        rotary_pos_emb = torch.cat([self.class_pos_emb, rotary_pos_emb], dim=0)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
        )


class MLCDVisionModel(CLIPVisionModel):
    @check_model_inputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        Example:

        ```python
        >>> import httpx
        >>> from io import BytesIO
        >>> from PIL import Image
        >>> from transformers import AutoProcessor, MLCDVisionModel
        >>> model = MLCDVisionModel.from_pretrained("DeepGlint-AI/mlcd-vit-bigG-patch14-448")
        >>> processor = AutoProcessor.from_pretrained("DeepGlint-AI/mlcd-vit-bigG-patch14-448")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read()))
        >>> inputs = processor(images=image, return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs, output_attentions=True)

        >>> features = outputs.last_hidden_state
        >>> print(f"Extracted features shape: {features.shape}")
        >>> print(f"Number of attention layers: {len(outputs.attentions)}")
        >>> print(f"Attention shape: {outputs.attentions[0].shape}")
        ```"""
        return self.vision_model(
            pixel_values=pixel_values,
            **kwargs,
        )


__all__ = [
    "MLCDVisionConfig",
    "MLCDPreTrainedModel",
    "MLCDVisionModel",
]
