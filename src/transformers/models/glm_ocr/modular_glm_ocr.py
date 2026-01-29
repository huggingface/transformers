# Copyright 2026 the HuggingFace Team. All rights reserved.
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
import torch.nn.functional as F

from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ..glm4v.configuration_glm4v import Glm4vConfig, Glm4vTextConfig, Glm4vVisionConfig
from ..glm4v.modeling_glm4v import (
    Glm4vForConditionalGeneration,
    Glm4VisionMlp,
    Glm4vModel,
    Glm4vModelOutputWithPast,
    Glm4vPreTrainedModel,
    Glm4vRMSNorm,
    Glm4vTextAttention,
    Glm4vVisionAttention,
    Glm4vVisionBlock,
    Glm4vVisionModel,
    Glm4vVisionPatchMerger,
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
    is_flash_attention_requested,
)


class GlmOcrRMSNorm(Glm4vRMSNorm):
    pass


class GlmOcrVisionMlp(Glm4VisionMlp):
    def __init__(self, config, bias: bool = True):
        super().__init__(config)
        self.intermediate_size = config.intermediate_size


class GlmOcrVisionConfig(Glm4vVisionConfig):
    r"""
    This is the configuration class to store the configuration of a [`GlmOcrVisionConfig`]. It is used to instantiate a
    GLM-OCR model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of
    GLM-OCR [zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR).

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        depth (`int`, *optional*, defaults to 24):
            Number of layers (depth) in the model.
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the encoder layers and the pooler layer.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"silu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        attention_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability for attention weights.
        num_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer architecture.
        in_channels (`int`, *optional*, defaults to 3):
            Number of input channels.
        image_size (`int` or `list[int]`, *optional*, defaults to 336):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 14):
            The size (resolution) of each patch.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        spatial_merge_size (`int`, *optional*, defaults to 2):
            The size used for merging spatial dimensions.
        temporal_patch_size (`int`, *optional*, defaults to 2):
            The size used for patches along the temporal dimension.
        out_hidden_size (`int`, *optional*, defaults to 1536):
            The output hidden size of the vision model.
        intermediate_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    """

    def __init__(
        self,
        depth=24,
        hidden_size=1024,
        hidden_act="silu",
        attention_bias=True,
        num_heads=16,
        image_size=336,
        out_hidden_size=1536,
        intermediate_size=4096,
        **super_kwargs,
    ):
        super().__init__(**super_kwargs)


class GlmOcrTextConfig(Glm4vTextConfig):
    r"""
    This is the configuration class to store the configuration of a [`GlmOcrTextConfig`]. It is used to instantiate a
    GLM-OCR model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of
    GLM-OCR [zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR).

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 59392):
            Vocabulary size of the GlmOcr model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`GlmOcrModel`]
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 4096):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 16):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `32`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 131072):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        pad_token_id (`int`, *optional*):
            The id of the padding token.

    ```python
    >>> from transformers import GlmOcrTextModel, GlmOcrConfig

    >>> # Initializing a GLM-OCR style configuration
    >>> configuration = GlmOcrConfig()

    >>> # Initializing a model from the GLM-OCR style configuration
    >>> model = GlmOcrTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    def __init__(
        self,
        vocab_size: int | None = 59392,
        hidden_size: int | None = 1024,
        intermediate_size: int | None = 4096,
        num_hidden_layers: int | None = 16,
        num_attention_heads: int | None = 16,
        num_key_value_heads: int | None = 8,
        max_position_embeddings: int | None = 131072,
        **super_kwargs,
    ):
        super().__init__(**super_kwargs)


class GlmOcrConfig(Glm4vConfig):
    r"""
    This is the configuration class to store the configuration of a [`GlmOcrModel`]. It is used to instantiate a
    GLM-OCR model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of
    GLM-OCR [zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR).

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        text_config (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `GlmOcrTextConfig`):
            The config object or dictionary of the text backbone.
        vision_config (`Union[PreTrainedConfig, dict]`,  *optional*, defaults to `GlmOcrVisionConfig`):
            The config object or dictionary of the vision backbone.
        image_token_id (`int`, *optional*, defaults to 59280):
            The image token index to encode the image prompt.
        video_token_id (`int`, *optional*, defaults to 59281):
            The video token index to encode the image prompt.
        image_start_token_id (`int`, *optional*, defaults to 59256):
            The image start token index to encode the start of image.
        image_end_token_id (`int`, *optional*, defaults to 59257):
            The image end token index to encode the end of image.
        video_start_token_id (`int`, *optional*, defaults to 59258):
            The video start token index to encode the start of video.
        video_end_token_id (`int`, *optional*, defaults to 59259):
            The video end token index to encode the end of video.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.

    ```python
    >>> from transformers import GlmOcrForConditionalGeneration, GlmOcrConfig

    >>> # Initializing a GLM-OCR style configuration
    >>> configuration = GlmOcrConfig()

    >>> # Initializing a model from the GLM-OCR style configuration
    >>> model = GlmOcrForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=59280,
        video_token_id=59281,
        image_start_token_id=59256,
        image_end_token_id=59257,
        video_start_token_id=59258,
        video_end_token_id=59259,
        tie_word_embeddings=False,
        **super_kwargs,
    ):
        super().__init__(**super_kwargs)


class GlmOcrTextAttention(Glm4vTextAttention, nn.Module):
    def __init__(self, config: GlmOcrTextConfig, layer_idx: int | None = None):
        super().__init__()
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)


class GlmOcrPreTrainedModel(Glm4vPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"model\.language_model\.layers\.16.*"]


class GlmOcrModelOutputWithPast(Glm4vModelOutputWithPast):
    pass


class GlmOcrVisionAttention(Glm4vVisionAttention):
    def __init__(self, config: GlmOcrVisionConfig) -> None:
        super().__init__()
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=config.attention_bias)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.q_norm = GlmOcrRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GlmOcrRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if is_flash_attention_requested(self.config):
            # Flash Attention: Use cu_seqlens for variable length attention
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            # Other implementations: Process each chunk separately
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
            ]

            attn_outputs = [
                attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class GlmOcrVisionBlock(Glm4vVisionBlock):
    def __init__(self, config) -> None:
        super().__init__()
        self.mlp = GlmOcrVisionMlp(config, bias=config.attention_bias)


class GlmOcrVisionPatchMerger(Glm4vVisionPatchMerger):
    pass


class GlmOcrVisionModel(Glm4vVisionModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        del self.embeddings
        del self.post_conv_layernorm
        self.merger = GlmOcrVisionPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.out_hidden_size * config.in_channels,
            hidden_act=config.hidden_act,
        )

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""
        hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
            The final hidden states of the model.
        grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
            The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb, image_type_ids = self.rot_pos_emb(grid_thw)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.post_layernorm(hidden_states)

        hidden_states = hidden_states.view(
            -1, self.spatial_merge_size, self.spatial_merge_size, hidden_states.shape[-1]
        )
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        hidden_states = self.downsample(hidden_states).view(-1, self.config.out_hidden_size)

        merged_hidden_states = self.merger(hidden_states)
        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
        )


class GlmOcrModel(Glm4vModel):
    pass


class GlmOcrForConditionalGeneration(Glm4vForConditionalGeneration):
    pass


__all__ = [
    "GlmOcrConfig",
    "GlmOcrTextConfig",
    "GlmOcrVisionConfig",
    "GlmOcrTextModel",  # noqa: F822
    "GlmOcrVisionModel",
    "GlmOcrModel",
    "GlmOcrPreTrainedModel",
    "GlmOcrForConditionalGeneration",
]
