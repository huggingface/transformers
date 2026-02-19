# Copyright 2025 the HuggingFace Team. All rights reserved.
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
from typing import Any

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...generation import GenerationMixin
from ...image_utils import ImageInput
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import ImagesKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, is_torch_available, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..chameleon.modeling_chameleon import ChameleonVQVAE, ChameleonVQVAEModelOutput, ChameleonVQVAEVectorQuantizer
from ..glm4v.configuration_glm4v import Glm4vTextConfig, Glm4vVisionConfig
from ..glm4v.modeling_glm4v import (
    Glm4vCausalLMOutputWithPast,
    Glm4vModel,
    Glm4vModelOutputWithPast,
    Glm4vPreTrainedModel,
    Glm4vTextModel,
    Glm4vVisionAttention,
    Glm4vVisionBlock,
    Glm4vVisionEmbeddings,
    Glm4vVisionModel,
    Glm4vVisionPatchEmbed,
)
from ..glm4v_moe.modeling_glm4v_moe import Glm4vMoeTextAttention, eager_attention_forward
from ..qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from ..qwen2_vl.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast
from ..qwen2_vl.processing_qwen2_vl import Qwen2VLProcessorKwargs
from ..siglip.modeling_siglip import SiglipMLP


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class GlmImageVQVAEConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GlmImageVQModel`]. It is used to instantiate a
    `GlmImageVQModel` according to the specified arguments, defining the model architecture.
    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information. Instantiating a
    configuration with the defaults will yield a similar configuration to the VQModel of the
    [zai-org/GLM-Image](https://huggingface.co/zai-org/GLM-Image) architecture.

    Args:
        embed_dim (`int`, *optional*, defaults to 2048):
            Dimensionality of each embedding vector.
        num_embeddings (`int`, *optional*, defaults to 16384):
            Number of codebook embeddings.
        latent_channels (`int`, *optional*, defaults to 1536):
            Number of channels for the latent space.
        in_channels (`int`, *optional*, defaults to 3):
            Number of input channels.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    """

    model_type = "glm_image_vqmodel"
    base_config_key = "vq_config"

    def __init__(
        self,
        embed_dim: int = 2048,
        num_embeddings: int = 16384,
        latent_channels: int = 1536,
        in_channels: int = 3,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_embeddings = num_embeddings
        self.latent_channels = latent_channels
        self.in_channels = in_channels
        self.initializer_range = initializer_range


class GlmImageVisionConfig(Glm4vVisionConfig):
    r"""
    This is the configuration class to store the configuration of a [`GlmImageVisionModel`]. It is used to instantiate an GlmImageVisionModel
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield
    a similar configuration to that of
    GLM-Image [zai-org/GLM-Image](https://huggingface.co/zai-org/GLM-Image).

    Args:
        depth (`int`, *optional*, defaults to 40):
            Number of layers (depth) in the model.
        hidden_size (`int`, *optional*, defaults to 1536):
            Dimensionality of the encoder layers and the pooler layer.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        attention_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability for attention weights.
        num_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer architecture.
        in_channels (`int`, *optional*, defaults to 3):
            Number of input channels.
        image_size (`int` or `list[int]`, *optional*, defaults to 2048):
                The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        spatial_merge_size (`int`, *optional*, defaults to 1):
            The size used for merging spatial dimensions.
        intermediate_size (`int`, *optional*, defaults to 6144):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    """

    model_type = "glm_image_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=40,
        hidden_size=1536,
        hidden_act="gelu",
        attention_bias=True,
        attention_dropout=0.0,
        num_heads=16,
        in_channels=3,
        image_size=2048,
        patch_size=16,
        layer_norm_eps=1e-06,
        spatial_merge_size=1,
        intermediate_size=6144,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        del self.out_hidden_size
        del self.rms_norm_eps
        del self.temporal_patch_size
        self.layer_norm_eps = layer_norm_eps


class GlmImageTextConfig(Glm4vTextConfig):
    r"""
    This is the configuration class to store the configuration of a [`GlmImageTextModel`]. It is used to instantiate a
    GLM-Image model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of
    GLM-Image [zai-org/GLM-Image](https://huggingface.co/zai-org/GLM-Image).

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 168064):
            Vocabulary size of the GlmImage model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`GlmImageModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 13696):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 40):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 2):
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
        pad_token_id (`int`, *optional*, defaults to 167841):
            The id of the padding token.
        vision_vocab_size (`int`, *optional*, defaults to 16512):
            Vision vocabulary size of the GlmImage model. Defines the number of different tokens that can be
            represented by the `inputs_ids` passed when calling [`GlmImageVisionModel`]
        attention_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        eos_token_id (`int`, *optional*, defaults to 16385):
            The id of the end of sequence token.

    ```python
    >>> from transformers import GlmImageTextModel, GlmImageConfig

    >>> # Initializing a GlmImageConfig style configuration
    >>> configuration = GlmImageConfig()

    >>> # Initializing a model from the GlmImageConfig style configuration
    >>> model = GlmImageTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    def __init__(
        self,
        vocab_size: int = 168064,
        max_position_embeddings: int = 131072,
        vision_vocab_size: int = 16512,
        attention_bias: bool = True,
        pad_token_id: int = 167841,
        eos_token_id: int = 16385,
        **super_kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            pad_token_id=pad_token_id,
            **super_kwargs,
        )
        self.vision_vocab_size = vision_vocab_size
        self.attention_bias = attention_bias
        self.eos_token_id = eos_token_id


class GlmImageConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GlmImageModel`]. It is used to instantiate a
    GLM-Image model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of
    GLM-Image [zai-org/GLM-Image](https://huggingface.co/zai-org/GLM-Image) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        text_config (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `GlmImageTextConfig`):
            The config object or dictionary of the text backbone.
        vision_config (`Union[PreTrainedConfig, dict]`,  *optional*, defaults to `GlmImageVisionConfig`):
            The config object or dictionary of the vision backbone.
        vq_config (`Union[Dict, GlmImageVQVAEConfig]`, *optional*):
            GlmImageVQVAEConfig instance containing the configuration for the VQ-VAE model.
        image_token_id (`int`, *optional*, defaults to 167855):
            The image token index to encode the image prompt.
        image_start_token_id (`int`, *optional*, defaults to 16384):
            The image start token index to encode the start of image.
        image_end_token_id (`int`, *optional*, defaults to 16385):
            The image end token index to encode the end of image.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.

    ```python
    >>> from transformers import Glm4vForConditionalGeneration, Glm4vConfig

    >>> # Initializing a GLM-Image style configuration
    >>> configuration = Glm4vConfig()

    >>> # Initializing a model from the GLM-Image style configuration
    >>> model = Glm4vForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "glm_image"
    sub_configs = {
        "vision_config": GlmImageVisionConfig,
        "text_config": GlmImageTextConfig,
        "vq_config": GlmImageVQVAEConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        vq_config=None,
        image_token_id=167855,
        image_start_token_id=16384,
        image_end_token_id=16385,
        tie_word_embeddings: bool | None = False,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            vision_config = self.sub_configs["vision_config"](**kwargs)

        if isinstance(vq_config, dict):
            vq_config = self.sub_configs["vq_config"](**vq_config)
        elif vq_config is None:
            vq_config = self.sub_configs["vq_config"](**kwargs)

        if isinstance(text_config, dict):
            text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            text_config = self.sub_configs["text_config"](**kwargs)

        self.image_token_id = image_token_id
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.text_config = text_config
        self.vision_config = vision_config
        self.vq_config = vq_config
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(**kwargs)


class GlmImageVisionMLP(SiglipMLP):
    pass


class GlmImageVisionAttention(Glm4vVisionAttention):
    def __init__(self, config: GlmImageVisionConfig) -> None:
        super().__init__(config)
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=config.attention_bias)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        if "flash" in self.config._attn_implementation:
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


class GlmImageVisionPatchEmbed(Glm4vVisionPatchEmbed):
    def __init__(self, config: GlmImageVisionConfig) -> None:
        super().__init__(config)

        del self.temporal_patch_size
        kernel_size = [self.patch_size, self.patch_size]
        self.proj = nn.Conv2d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size)

    def forward(self, hidden_states):
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(-1, self.in_channels, self.patch_size, self.patch_size)
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class GlmImageVisionEmbeddings(Glm4vVisionEmbeddings):
    def __init__(self, config: GlmImageVisionConfig) -> None:
        super().__init__(config)
        self.interpolated_method = "bilinear"


class GlmImageVisionBlock(Glm4vVisionBlock):
    def __init__(self, config: GlmImageVisionConfig):
        super().__init__(config)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = GlmImageVisionAttention(config)
        self.mlp = GlmImageVisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        r"""
        cu_seqlens (`torch.Tensor` of shape `(num_images_or_videos + 1,)`):
            The cumulative sequence lengths of each image or video feature.
        position_embeddings (`tuple(torch.Tensor, torch.Tensor)` of shape `(num_patches, head_dim // 2)`):
            The cosine and sine position embeddings for vision attention.
        """
        residual = hidden_states

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn(
            hidden_states,
            cu_seqlens=cu_seqlens,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class GlmImageTextAttention(Glm4vMoeTextAttention):
    pass


class GlmImagePreTrainedModel(Glm4vPreTrainedModel):
    config: GlmImageConfig
    input_modalities = ("image", "text")

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(module)


class GlmImageModelOutputWithPast(Glm4vModelOutputWithPast):
    pass


class GlmImageVQVAEVectorQuantizer(ChameleonVQVAEVectorQuantizer):
    def __init__(self, config: GlmImageVQVAEConfig):
        super().__init__(config)
        self.num_embeddings = config.num_embeddings
        self.embedding_dim = config.embed_dim
        self.beta = getattr(config, "beta", 0.25)

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)

    def forward(self, hidden_state: torch.Tensor):
        hidden_state = hidden_state.permute(0, 2, 3, 1).contiguous()
        hidden_state_flattened = hidden_state.view(-1, self.embedding_dim)

        # L2 normalize
        hidden_state = F.normalize(hidden_state, p=2, dim=-1)
        hidden_state_flattened = F.normalize(hidden_state_flattened, p=2, dim=-1)
        embedding = F.normalize(self.embedding.weight, p=2, dim=-1)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        distances = (
            torch.sum(hidden_state_flattened**2, dim=1, keepdim=True)
            + torch.sum(embedding**2, dim=1)
            - 2 * torch.einsum("bd,dn->bn", hidden_state_flattened, embedding.transpose(0, 1))
        )

        min_encoding_indices = torch.argmin(distances, dim=1)
        hidden_state_quant = embedding[min_encoding_indices].view(hidden_state.shape)

        # compute loss for embedding
        loss = torch.mean((hidden_state_quant.detach() - hidden_state) ** 2) + self.beta * torch.mean(
            (hidden_state_quant - hidden_state.detach()) ** 2
        )

        # preserve gradients
        hidden_state_quant = hidden_state + (hidden_state_quant - hidden_state).detach()

        # reshape back to match original input shape
        hidden_state_quant = hidden_state_quant.permute(0, 3, 1, 2).contiguous()

        return hidden_state_quant, loss, min_encoding_indices


class GlmImageVQVAEModelOutput(ChameleonVQVAEModelOutput):
    pass


class GlmImageVQVAE(ChameleonVQVAE):
    _no_split_modules = [
        "GlmImageVQVAEVectorQuantizer",
    ]
    _can_record_outputs = {}

    def __init__(self, config: GlmImageVQVAEConfig):
        super().__init__(config)
        del self.encoder

    def encode(self, hidden_states):
        conv_hidden_states = self.quant_conv(hidden_states)
        quantized_last_hidden_state, emb_loss, indices = self.quantize(conv_hidden_states)
        return GlmImageVQVAEModelOutput(
            last_hidden_state=hidden_states,
            quantized_last_hidden_state=quantized_last_hidden_state,
            image_tokens=indices,
            embedding_loss=emb_loss,
        )


class GlmImageVisionModel(Glm4vVisionModel):
    config: GlmImageVisionConfig
    main_input_name = "pixel_values"
    input_modalities = ("image",)

    def __init__(self, config: GlmImageVisionConfig):
        super().__init__(config)

        head_dim = config.hidden_size // config.num_heads
        self.head_dim = head_dim

        del self.merger
        del self.rotary_pos_emb
        del self.post_conv_layernorm
        del self.downsample
        del self.post_layernorm

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        return pos_ids

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor, **kwargs: Unpack[TransformersKwargs]
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.Tensor` of shape `(total_patches, num_channels * patch_size * patch_size)`):
            Packed pixel values.
        grid_thw (`torch.Tensor` of shape `(num_images, 3)`):
            The temporal, height and width of feature shape of each image.

        Returns:
            `torch.Tensor` of shape `(total_patches, hidden_size)`: Hidden states.
        """

        hidden_states = self.patch_embed(pixel_values)
        image_type_ids = self.rot_pos_emb(grid_thw)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        hidden_states = self.embeddings(
            hidden_states,
            seqlens,
            grid_thw,
            image_type_ids[:, 0].to(hidden_states.device),
            image_type_ids[:, 1].to(hidden_states.device),
        )

        # Transformer blocks (no position_embeddings needed, already added above)
        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
            )

        return BaseModelOutputWithPooling(last_hidden_state=hidden_states)


class GlmImageTextModel(Glm4vTextModel):
    pass


class GlmImageModel(Glm4vModel):
    def __init__(self, config):
        super().__init__(config)
        self.visual = GlmImageVisionModel._from_config(config.vision_config)
        self.language_model = GlmImageTextModel._from_config(config.text_config)
        self.vqmodel = GlmImageVQVAE._from_config(config.vq_config)

        self.rope_deltas = None  # cache rope_deltas here

        # Per-sample caches for batch processing
        self._cached_decode_position_ids = None  # shape: [batch_size, 3, max_decode_len]
        self._prefill_len = None  # prefill sequence length (same for all samples in batch)

        # Initialize weights and apply final processing
        self.post_init()

    def get_rope_index(
        self,
        input_ids: torch.LongTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        images_per_sample: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index for image generation task with full batch support.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            image_grid_thw (`torch.LongTensor` of shape `(total_images_in_batch, 3)`, *optional*):
                The temporal, height and width of feature shape of each image.
                Images are packed across all samples in the batch.
            images_per_sample (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Number of images (including target grids) for each sample in the batch.
                Used to split image_grid_thw by sample.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`):
                Position IDs for temporal, height, and width dimensions.
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size, 1)`):
                Position deltas for multi-modal rotary position embedding.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        dtype = input_ids.dtype

        image_start_token_id = self.config.image_start_token_id
        image_end_token_id = self.config.image_end_token_id

        position_ids = torch.ones(3, batch_size, seq_len, dtype=dtype, device=device)
        text_positions = torch.arange(seq_len, device=device)[None, :].repeat(3, 1)

        # Split image_grid_thw by sample if images_per_sample is provided
        if image_grid_thw is not None and images_per_sample is not None:
            grids_per_sample = torch.split(image_grid_thw, images_per_sample.tolist())
        elif image_grid_thw is not None:
            # Fallback: assume all grids belong to first sample (batch_size=1)
            grids_per_sample = [image_grid_thw] * batch_size
        else:
            grids_per_sample = [None] * batch_size

        # Per-sample caches for decode stage
        all_decode_position_ids = []

        for batch_idx in range(batch_size):
            curr_input_ids = input_ids[batch_idx]
            curr_grids = grids_per_sample[batch_idx]

            if attention_mask is not None and attention_mask.shape[1] == seq_len:
                valid_mask = attention_mask[batch_idx] == 1
                curr_input_ids_valid = curr_input_ids[valid_mask]
            else:
                # attention_mask may have different length during assisted decoding
                curr_input_ids_valid = curr_input_ids
                valid_mask = None

            # Find image boundaries in this sample
            image_end_positions = torch.where(curr_input_ids_valid == image_end_token_id)[0]
            image_start_positions = torch.where(curr_input_ids_valid == image_start_token_id)[0] + 1
            num_complete_images = len(image_end_positions)

            current_pos = 0
            prev_image_end = 0
            curr_position_ids = []

            # Process complete images (source images in image-to-image task)
            for img_idx, (start, end) in enumerate(zip(image_start_positions, image_end_positions)):
                if curr_grids is None or img_idx >= len(curr_grids):
                    break
                grid = curr_grids[img_idx]
                # grid format is [temporal, height, width]
                _, height, width = grid.tolist()

                # Text tokens before this image
                llm_pos_length = start - prev_image_end
                llm_position_ids = text_positions[:, current_pos : current_pos + llm_pos_length].to(device=device)
                current_pos += llm_position_ids.shape[-1]

                # Image tokens with 2D spatial encoding
                # For an image with height H and width W:
                # - position_width cycles [0, 1, ..., W-1] for each row, repeated H times
                # - position_height stays constant per row, [0]*W, [1]*W, ..., [H-1]*W
                image_seq_length = height * width
                position_width = torch.arange(current_pos, current_pos + width, device=device).repeat(height)
                position_height = torch.arange(current_pos, current_pos + height, device=device).repeat_interleave(
                    width
                )
                position_temporal = torch.full((image_seq_length,), current_pos, device=device, dtype=torch.long)
                vision_position_ids = torch.stack([position_temporal, position_height, position_width], dim=0)
                current_pos += max(height, width)

                prev_image_end = end
                curr_position_ids.append(torch.cat([llm_position_ids, vision_position_ids], dim=-1))

            # Remaining text tokens (including the final image_start token for generation)
            end_position = len(curr_input_ids_valid) - prev_image_end
            llm_position_ids = text_positions[:, current_pos : current_pos + end_position].to(device=device)
            current_pos += llm_position_ids.shape[-1]
            curr_position_ids.append(llm_position_ids)

            # Concatenate all position ids for this sample
            curr_position_ids = torch.cat(curr_position_ids, dim=-1)

            # Store in the main position_ids tensor
            if valid_mask is not None:
                position_ids[:, batch_idx, valid_mask] = curr_position_ids
            else:
                position_ids[:, batch_idx, :] = curr_position_ids

            # Build decode position ids for this sample
            if curr_grids is not None and len(curr_grids) > 0:
                num_decode_grids = len(curr_grids) - num_complete_images
                num_decode_grids = max(num_decode_grids, 0)
                decode_pos = current_pos

                decode_temporal_list = []
                decode_height_list = []
                decode_width_list = []

                curr_grids_list = curr_grids.tolist()
                for i in range(1, num_decode_grids + 1):
                    grid_idx = -i
                    h = curr_grids_list[grid_idx][1]
                    w = curr_grids_list[grid_idx][2]
                    total_tokens = h * w

                    h_indices = torch.arange(h, device=device).unsqueeze(1).expand(h, w).flatten()
                    w_indices = torch.arange(w, device=device).unsqueeze(0).expand(h, w).flatten()

                    decode_temporal_list.append(
                        torch.full((total_tokens,), decode_pos, device=device, dtype=torch.long)
                    )
                    decode_height_list.append(decode_pos + h_indices)
                    decode_width_list.append(decode_pos + w_indices)
                    decode_pos = decode_pos + max(h, w)

                # End marker
                decode_temporal_list.append(torch.tensor([decode_pos], device=device, dtype=torch.long))
                decode_height_list.append(torch.tensor([decode_pos], device=device, dtype=torch.long))
                decode_width_list.append(torch.tensor([decode_pos], device=device, dtype=torch.long))

                sample_decode_pos_ids = torch.stack(
                    [
                        torch.cat(decode_temporal_list, dim=0),
                        torch.cat(decode_height_list, dim=0),
                        torch.cat(decode_width_list, dim=0),
                    ],
                    dim=0,
                )
                all_decode_position_ids.append(sample_decode_pos_ids)

        # Store prefill length (same for all samples since input_ids is padded to same length)
        self._prefill_len = seq_len

        # Pad decode position ids to same length and stack
        if all_decode_position_ids:
            max_decode_len = max(x.shape[1] for x in all_decode_position_ids)
            padded_decode_pos_ids = [
                F.pad(pos_ids, (0, max_decode_len - pos_ids.shape[1]), mode="replicate")
                for pos_ids in all_decode_position_ids
            ]
            self._cached_decode_position_ids = torch.stack(padded_decode_pos_ids, dim=0)  # [batch, 3, max_decode_len]
        else:
            self._cached_decode_position_ids = None

        mrope_position_deltas = torch.zeros([batch_size, 1], dtype=dtype, device=device)

        return position_ids, mrope_position_deltas

    def get_image_tokens(
        self,
        hidden_states: torch.FloatTensor,
        image_grid_thw: torch.LongTensor,
    ) -> torch.LongTensor:
        """
        Tokenizes image features into discrete tokens with VQVAE module.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(total_patches, hidden_size)`):
                The packed image features from vision encoder.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
                The temporal, height and width of feature shape of each image.

        Returns:
            image_tokens (`torch.LongTensor` of shape `(total_patches,)`):
                Discrete token indices from the VQVAE codebook.
        """
        hidden_size = hidden_states.shape[-1]
        split_sizes = (image_grid_thw.prod(dim=-1)).tolist()
        hidden_states_list = torch.split(hidden_states, split_sizes, dim=0)

        all_image_toks = []
        for i, hs in enumerate(hidden_states_list):
            grid_t, grid_h, grid_w = image_grid_thw[i].tolist()
            hs = hs.view(grid_t, grid_h, grid_w, hidden_size)
            hs = hs.permute(0, 3, 1, 2).contiguous()
            vqmodel_outputs: GlmImageVQVAEModelOutput = self.vqmodel.encode(hs)
            all_image_toks.append(vqmodel_outputs.image_tokens)
        return torch.cat(all_image_toks, dim=0)

    def get_video_features(self):
        raise AttributeError("Not needed for GlmImage")

    @can_return_tuple
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The tensors corresponding to the input images.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        """
        pixel_values = pixel_values.type(self.visual.dtype)
        vision_outputs = self.visual(pixel_values, grid_thw=image_grid_thw, return_dict=True, **kwargs)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(vision_outputs.last_hidden_state, split_sizes)
        vision_outputs.pooler_output = image_embeds

        return vision_outputs

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        image_ids: torch.LongTensor,
    ):
        """
        Replace image placeholder tokens in input_ids with actual image token ids from VQVAE.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, seq_len)`):
                Input token ids with image placeholders.
            image_ids (`torch.LongTensor` of shape `(num_images, num_tokens_per_image)` or flattened):
                Discrete token indices from the VQVAE codebook.

        Returns:
            special_image_mask (`torch.LongTensor` of shape `(batch_size, seq_len)`):
                Mask indicating positions in input ids that will be replaced by actual image tokens.
        """

        special_image_mask = input_ids == self.config.image_token_id
        n_placeholder_tokens = special_image_mask.sum()
        n_image_tokens = image_ids.shape[0]

        if n_placeholder_tokens != n_image_tokens:
            raise ValueError(
                f"Number of image placeholder tokens ({n_placeholder_tokens.item()}) does not match "
                f"number of image tokens from VQVAE ({n_image_tokens})"
            )

        return special_image_mask

    def compute_3d_position_ids(
        self,
        input_ids: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        images_per_sample: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        past_key_values: torch.Tensor | None,
        cache_position: torch.Tensor | None,
    ) -> torch.Tensor | None:
        past_key_values_length = 0 if past_key_values is None else past_key_values.get_seq_length()
        can_compute_mrope = input_ids is not None and image_grid_thw is not None

        if can_compute_mrope and (self.rope_deltas is None or past_key_values_length == 0):
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
                images_per_sample=images_per_sample,
            )
            self.rope_deltas = rope_deltas
        # Use pre-calculated rope-deltas to infer correct 3D position ids
        elif self.rope_deltas is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
            if self._cached_decode_position_ids is not None:
                step = cache_position[0].item() - self._prefill_len
                position_ids = self._cached_decode_position_ids[:, :, step : step + seq_length].permute(1, 0, 2)
            else:
                position_ids = cache_position.view(1, 1, -1).repeat(3, batch_size, 1)
        else:
            # Can't build correct 3D positions. Let the model infer it from `cache_position`
            position_ids = None
        return position_ids

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        images_per_sample: torch.LongTensor | None = None,
        rope_deltas: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | GlmImageModelOutputWithPast:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(total_images_in_batch, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
            Images are packed across all samples in the batch.
        images_per_sample (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Number of images (including target grids) for each sample in the batch.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]

        if pixel_values is not None:
            # Process source images (image-to-image mode)
            # Source images are identified by counting image_end_token_id in input_ids
            # Note: We must exclude padding tokens since pad_token_id == image_end_token_id
            if images_per_sample is not None:
                grids_per_sample = torch.split(image_grid_thw, images_per_sample.tolist())
                # Create mask for non-padding tokens (attention_mask=1 means non-padding)
                # Handle 4D attention mask (from static cache) by extracting diagonal
                if attention_mask is not None and attention_mask.ndim == 4:
                    non_pad_mask = torch.diagonal(attention_mask[:, 0], dim1=1, dim2=2)
                    if non_pad_mask.dtype.is_floating_point:
                        non_pad_mask = non_pad_mask / torch.finfo(non_pad_mask.dtype).min
                        non_pad_mask = (1.0 - non_pad_mask).int()
                    # Only keep columns matching input_ids length
                    non_pad_mask = non_pad_mask[:, -input_ids.shape[1] :]
                else:
                    non_pad_mask = attention_mask if attention_mask is not None else torch.ones_like(input_ids)

                source_grids_list = []
                is_image_end = input_ids == self.config.image_end_token_id
                is_non_pad = non_pad_mask == 1
                num_source_per_sample = (is_image_end & is_non_pad).sum(dim=1).tolist()
                for sample_idx in range(batch_size):
                    num_source = num_source_per_sample[sample_idx]
                    if num_source > 0:
                        source_grids_list.append(grids_per_sample[sample_idx][:num_source])
                if len(source_grids_list) == 0:
                    raise ValueError(
                        "pixel_values provided but no source images found in input_ids. "
                        "Ensure input_ids contains image_end_token_id for each source image."
                    )
                source_grids = torch.cat(source_grids_list, dim=0)
            else:
                # Fallback for batch_size=1: all but last grid are source images
                source_grids = image_grid_thw[:-1]

            image_features = self.get_image_features(pixel_values, source_grids, return_dict=True)
            image_embeds = torch.cat(image_features.pooler_output, dim=0)
            image_ids = self.get_image_tokens(image_embeds, source_grids)
            image_ids = image_ids.view(-1).to(input_ids.device)
            special_image_mask = self.get_placeholder_mask(input_ids, image_ids)
            input_ids = input_ids.masked_scatter(special_image_mask, image_ids)

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                images_per_sample=images_per_sample,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
            )

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        return GlmImageModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )


class GlmImageCausalLMOutputWithPast(Glm4vCausalLMOutputWithPast):
    pass


class GlmImageForConditionalGeneration(GlmImagePreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = {}
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    base_model_prefix = "model"
    config: GlmImageConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = GlmImageModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vision_vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The tensors corresponding to the input images.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        """
        return self.model.get_image_features(pixel_values, image_grid_thw, **kwargs)

    def get_image_tokens(self, hidden_states: torch.FloatTensor, image_grid_thw: torch.LongTensor | None = None):
        return self.model.get_image_tokens(hidden_states, image_grid_thw)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        images_per_sample: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | GlmImageCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(total_images_in_batch, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
            Images are packed across all samples in the batch.
        images_per_sample (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Number of images (including target grids) for each sample in the batch.

        Example:

        ```python
        >>> from PIL import Image
        >>> import httpx
        >>> from io import BytesIO
        >>> from transformers import AutoProcessor, GlmImageForConditionalGeneration

        >>> model = GlmImageForConditionalGeneration.from_pretrained("zai-org/GLM-Image")
        >>> processor = AutoProcessor.from_pretrained("zai-org/GLM-Image")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Add a truck of this photo.<sop>28 40<eop>"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read()))

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            images_per_sample=images_per_sample,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        return GlmImageCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        image_grid_thw=None,
        images_per_sample=None,
        is_first_iteration=False,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            is_first_iteration=is_first_iteration,
            use_cache=use_cache,
            **kwargs,
        )

        model_inputs["position_ids"] = None
        model_inputs["images_per_sample"] = images_per_sample

        if not is_first_iteration and use_cache:
            model_inputs["pixel_values"] = None

        return model_inputs

    def _get_image_nums(
        self,
        input_ids: torch.LongTensor | None,
    ) -> torch.Tensor:
        """
        Get the number of images for each sample.
        For GLM-Image, only input_ids allow us to get the number of images.

        Returns:
            image_counts (`torch.LongTensor` of shape `(batch_size,)`)
        """
        is_image = input_ids == self.config.image_start_token_id

        return is_image.sum(dim=1)

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: torch.LongTensor | None = None,
        **model_kwargs,
    ) -> tuple[torch.LongTensor, dict[str, Any]]:
        # Overwritten -- Support for expanding tensors without a batch size dimension
        # e.g., pixel_values, image_grid_thw
        # pixel_values.shape[0] is sum(seqlen_images for samples)
        # image_grid_thw.shape[0] is sum(num_images for samples)

        if expand_size == 1:
            return input_ids, model_kwargs

        visual_keys = ["pixel_values", "image_grid_thw", "images_per_sample"]

        def _expand_dict_for_generation_visual(dict_to_expand):
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            if image_grid_thw is None:
                return dict_to_expand

            images_per_sample = model_kwargs.get("images_per_sample", None)

            # Use images_per_sample if available
            if images_per_sample is not None:
                image_nums = images_per_sample.tolist()
            elif input_ids is not None:
                # Try to infer from image_grid_thw / batch_size
                batch_size = input_ids.shape[0]
                total_grids = image_grid_thw.shape[0]
                if total_grids % batch_size == 0:
                    grids_per_sample = total_grids // batch_size
                    image_nums = [grids_per_sample] * batch_size
                else:
                    # Cannot evenly distribute grids - fall back to simple repeat_interleave
                    # This handles test cases where image_grid_thw has (batch_size + 1) rows
                    dict_to_expand["image_grid_thw"] = image_grid_thw.repeat_interleave(expand_size, dim=0)
                    if dict_to_expand.get("pixel_values") is not None:
                        dict_to_expand["pixel_values"] = dict_to_expand["pixel_values"].repeat_interleave(
                            expand_size, dim=0
                        )
                    return dict_to_expand
            else:
                image_nums = self._get_image_nums(input_ids).tolist()

            # Get source image counts per sample from image_end_token_id count
            source_image_nums = (input_ids == self.config.image_end_token_id).sum(dim=1).tolist()

            def _repeat_interleave_samples(x, lengths, repeat_times):
                samples = torch.split(x, lengths)
                repeat_args = [repeat_times] + [1] * (x.dim() - 1)
                result = torch.cat([sample.repeat(*repeat_args) for sample in samples], dim=0)
                return result

            for key in dict_to_expand:
                if key == "pixel_values":
                    # Split images into samples based on source image counts
                    if sum(source_image_nums) > 0:
                        # Split grids by sample to compute pixel counts
                        grids_per_sample = torch.split(image_grid_thw, image_nums)
                        all_pixel_counts = image_grid_thw.prod(dim=1)
                        pixel_counts_per_sample = torch.split(all_pixel_counts, image_nums)
                        # Build source mask and compute per-sample source pixel counts in one sync
                        source_pixel_counts = torch.zeros(len(grids_per_sample), device=image_grid_thw.device)
                        for batch_idx in range(len(grids_per_sample)):
                            num_source = source_image_nums[batch_idx]
                            if num_source > 0:
                                source_pixel_counts[batch_idx] = pixel_counts_per_sample[batch_idx][:num_source].sum()
                        lengths = source_pixel_counts.to(torch.int64).tolist()

                        dict_to_expand[key] = _repeat_interleave_samples(
                            dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                        )
                elif key == "image_grid_thw":
                    # Expand all grids (source + target) per sample
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=image_nums, repeat_times=expand_size
                    )
                elif key == "images_per_sample":
                    # Simply repeat the counts
                    if dict_to_expand.get(key) is not None:
                        dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                    and key not in visual_keys
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        model_kwargs = _expand_dict_for_generation_visual(model_kwargs)

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs


def smart_resize(
    height: int,
    width: int,
    factor: int = 16,
    min_pixels: int = 512 * 512,
    max_pixels: int = 2048 * 2048,
) -> tuple[int, int]:
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 4:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 4, got {max(height, width) / min(height, width)}"
        )

    shortest_edge = int(round(math.sqrt(min_pixels)))
    longest_edge = int(round(math.sqrt(max_pixels)))
    min_side = min(height, width)
    max_side = max(height, width)

    scale = 1.0

    if min_side < shortest_edge:
        scale = shortest_edge / min_side

    if max_side * scale > longest_edge:
        scale = longest_edge / max_side

    height = height // 2
    width = width // 2

    h_bar = max(factor, int(round(height * scale / factor)) * factor)
    w_bar = max(factor, int(round(width * scale / factor)) * factor)

    if max(h_bar, w_bar) > longest_edge:
        beta = max(h_bar, w_bar) / longest_edge
        h_bar = max(factor, int(math.floor((h_bar / beta) / factor)) * factor)
        w_bar = max(factor, int(math.floor((w_bar / beta) / factor)) * factor)

    return h_bar, w_bar


class GlmImageImageProcessor(Qwen2VLImageProcessor):
    model_input_names = ["pixel_values", "image_grid_thw", "images_per_sample"]


class GlmImageImageProcessorFast(Qwen2VLImageProcessorFast):
    model_input_names = ["pixel_values", "image_grid_thw", "images_per_sample"]


class GlmImageImagesKwargs(ImagesKwargs, total=False):
    """
    target_h (`int`):
        Height of the target image to be generated.
    target_w (`int`):
        Width of the target image to be generated.
    """

    target_h: int
    target_w: int


class GlmImageProcessorKwargs(Qwen2VLProcessorKwargs):
    images_kwargs: GlmImageImagesKwargs

    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": False,
        },
        "images_kwargs": {
            "target_h": 1152,
            "target_w": 768,
        },
    }


class GlmImageProcessor(ProcessorMixin):
    r"""
    Constructs a GLM-Image processor which wraps a GLM-Image image processor and a GLM-Image tokenizer into a single processor.
    [`~GlmImageProcessor.__call__`] and [`~GlmImageProcessor.decode`] for more information.
    Args:
        image_processor ([`GlmImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`PreTrainedTokenizerFast`], *optional*):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    model_input_names = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw", "images_per_sample"]

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        self.image_token = tokenizer.image_token
        self.grid_bos_token = tokenizer.grid_bos_token
        self.grid_eos_token = tokenizer.grid_eos_token
        self.bos_token = tokenizer.bos_token
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        **kwargs: Unpack[GlmImageProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to PreTrainedTokenizerFast's [`~PreTrainedTokenizerFast.__call__`] if `text` is not `None` to encode
        the text.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **image_grid_thw** -- List of image 3D grid in LLM. Returned when `images` is not `None`.
        """
        output_kwargs = self._merge_kwargs(
            GlmImageProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        target_h = output_kwargs["images_kwargs"].pop("target_h", None)
        target_w = output_kwargs["images_kwargs"].pop("target_w", None)
        is_text_to_image = images is None

        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None

        # Handle text=None case (image-only processing)
        if text is None:
            if images is None:
                raise ValueError("You must provide at least one of `text` or `images`.")
            return image_inputs

        if not isinstance(text, list):
            text = [text]

        batch_size = len(text)
        text = text.copy()  # below lines change text in-place

        # Count images per sample by counting image tokens in each text
        images_per_sample = []
        for i in range(batch_size):
            images_per_sample.append(text[i].count(self.image_token))

        # Replace image tokens with the correct number of placeholder tokens
        if not is_text_to_image:
            index = 0
            for i in range(batch_size):
                while self.image_token in text[i]:
                    grid = image_grid_thw[index]
                    num_image_tokens = int(grid[1] * grid[2])
                    text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        # Build prompt with target shape and combine grids in a single loop
        # Format: [sample0_source_grids..., sample0_target_grids, sample1_source_grids..., sample1_target_grids, ...]
        # Note: In i2i mode, batches are homogeneous (same number of source images per sample)
        num_source_images = images_per_sample[0] if images_per_sample else 0

        # Validate homogeneity for i2i mode
        if not is_text_to_image and images_per_sample and len(set(images_per_sample)) != 1:
            raise ValueError(
                f"In image-to-image mode, all samples must have the same number of source images. "
                f"Got different counts: {images_per_sample}"
            )

        all_grids = []
        for i in range(batch_size):
            text[i], token_h, token_w, prev_h, prev_w = self._build_prompt_with_target_shape(
                text[i], height=target_h, width=target_w, is_text_to_image=is_text_to_image
            )
            # Add source grids for this sample (i2i mode only)
            if not is_text_to_image and num_source_images > 0:
                start_idx = i * num_source_images
                all_grids.append(image_grid_thw[start_idx : start_idx + num_source_images])
            # Add target grid for this sample
            all_grids.append(
                self._build_target_image_grid_thw(
                    token_h=token_h,
                    token_w=token_w,
                    prev_token_h=prev_h,
                    prev_token_w=prev_w,
                    is_text_to_image=is_text_to_image,
                )
            )
        image_inputs["image_grid_thw"] = torch.cat(all_grids, dim=0)

        # Store images_per_sample for later use (add target images count)
        # Each sample will have: source_images + target_images (typically 2 for t2i, 1 for i2i)
        num_target_grids = 2 if is_text_to_image else 1
        image_inputs["images_per_sample"] = torch.tensor(
            [num_source_images + num_target_grids] * batch_size, dtype=torch.long
        )

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        self._check_special_mm_tokens(text, text_inputs, modalities=["image"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()
        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)

    def _build_prompt_with_target_shape(
        self,
        prompt: str,
        height: int,
        width: int,
        is_text_to_image: bool,
    ) -> tuple[str, int, int, int, int]:
        factor = 32
        height = (height // factor) * factor
        width = (width // factor) * factor
        token_h = height // factor
        token_w = width // factor
        ratio = token_h / token_w
        prev_token_h = int(math.sqrt(ratio) * (factor // 2))
        prev_token_w = int(math.sqrt(1 / ratio) * (factor // 2))

        if is_text_to_image:
            expanded_prompt = f"{prompt}{self.grid_bos_token}{token_h} {token_w}{self.grid_eos_token}{self.grid_bos_token}{prev_token_h} {prev_token_w}{self.grid_eos_token}{self.bos_token}"
        else:
            expanded_prompt = f"{prompt}{self.grid_bos_token}{token_h} {token_w}{self.grid_eos_token}{self.bos_token}"

        return expanded_prompt, token_h, token_w, prev_token_h, prev_token_w

    @staticmethod
    def _build_target_image_grid_thw(
        token_h: int,
        token_w: int,
        prev_token_h: int,
        prev_token_w: int,
        is_text_to_image: bool = True,
    ):
        if is_text_to_image:
            # Text-to-image: 2 target grids (large + small preview)
            return torch.tensor(
                [
                    [1, token_h, token_w],
                    [1, prev_token_h, prev_token_w],
                ],
            )
        else:
            # Image-to-image: 1 target grid only
            return torch.tensor(
                [
                    [1, token_h, token_w],
                ],
            )


__all__ = [
    "GlmImageVQVAEConfig",
    "GlmImageVisionConfig",
    "GlmImageTextConfig",
    "GlmImageConfig",
    "GlmImagePreTrainedModel",
    "GlmImageVQVAE",
    "GlmImageVisionModel",
    "GlmImageTextModel",
    "GlmImageModel",
    "GlmImageForConditionalGeneration",
    "GlmImageImageProcessor",
    "GlmImageImageProcessorFast",
    "GlmImageProcessor",
]
