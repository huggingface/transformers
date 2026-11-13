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
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...configuration_utils import PreTrainedConfig
from ...masking_utils import create_bidirectional_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..blt.modeling_blt import apply_rotary_pos_emb
from ..clap.modeling_clap import ClapOutput, contrastive_loss
from ..cohere.modeling_cohere import CohereRotaryEmbedding
from ..dinov2.modeling_dinov2 import Dinov2LayerScale
from ..llama.modeling_llama import LlamaRMSNorm
from ..modernbert.modeling_modernbert import ModernBertMLP
from ..sam3.modeling_sam3 import Sam3ViTRotaryEmbedding
from ..siglip2.modeling_siglip2 import Siglip2Encoder, eager_attention_forward


@auto_docstring(checkpoint="laion/clap2-small-audioxl-s1B")
@strict
class Clap2TextConfig(PreTrainedConfig):
    r"""
    mlp_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for the MLP layers.
    use_qk_norm (`bool`, *optional*, defaults to `True`):
        Whether to apply an RMSNorm over the head dimension of the queries and keys.
    layerscale_value (`float`, *optional*):
        Initial value of the per-channel residual scaling. No scaling is applied when `None`.

    Example:

    ```python
    >>> from transformers import Clap2TextConfig, Clap2TextModel

    >>> configuration = Clap2TextConfig()
    >>> model = Clap2TextModel(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "clap2_text_model"
    base_config_key = "text_config"
    default_theta = 10000.0

    vocab_size: int = 50260
    hidden_size: int = 512
    intermediate_size: int = 1408
    num_hidden_layers: int = 11
    num_attention_heads: int = 8
    max_position_embeddings: int = 256
    hidden_activation: str = "silu"
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    mlp_bias: bool = False
    mlp_dropout: float | int = 0.0
    use_qk_norm: bool = True
    layerscale_value: float | None = None
    attention_dropout: float | int = 0.0
    rope_parameters: RopeParameters | dict | None = None
    initializer_range: float = 0.02
    pad_token_id: int | None = 50258
    bos_token_id: int | None = 50259
    eos_token_id: int | list[int] | None = 50257

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        super().validate_architecture()
        if self.hidden_size % (2 * self.num_attention_heads):
            raise ValueError(f"`hidden_size` must be a multiple of 2 * `num_attention_heads`, got {self.hidden_size}.")


@auto_docstring(checkpoint="laion/clap2-small-audioxl-s1B")
@strict
class Clap2AudioConfig(PreTrainedConfig):
    r"""
    num_mel_bins (`int`, *optional*, defaults to 128):
        Number of mel bins of the log-mel spectrogram the patches are cut from.
    num_register_tokens (`int`, *optional*, defaults to 1):
        Number of learned register tokens prepended to the patch sequence.
    pooler_hidden_act (`str`, *optional*, defaults to `"gelu"`):
        The activation function of the MLP in the attention pooling head.
    mlp_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for the MLP layers.
    use_qk_norm (`bool`, *optional*, defaults to `False`):
        Whether to apply an RMSNorm over the head dimension of the queries and keys.
    layerscale_value (`float`, *optional*, defaults to 1e-05):
        Initial value of the per-channel residual scaling. No scaling is applied when `None`.

    Example:

    ```python
    >>> from transformers import Clap2AudioConfig, Clap2AudioModel

    >>> configuration = Clap2AudioConfig()
    >>> model = Clap2AudioModel(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "clap2_audio_model"
    base_config_key = "audio_config"
    default_theta = 10000.0
    default_rope_type = "axial"

    hidden_size: int = 512
    intermediate_size: int = 1408
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    num_mel_bins: int = 128
    patch_size: int = 16
    num_register_tokens: int = 1
    hidden_activation: str = "silu"
    pooler_hidden_act: str = "gelu"
    rms_norm_eps: float = 1e-6
    attention_bias: bool = True
    mlp_bias: bool = True
    mlp_dropout: float | int = 0.0
    use_qk_norm: bool = False
    layerscale_value: float | None = 1e-5
    attention_dropout: float | int = 0.0
    rope_parameters: RopeParameters | dict | None = None
    initializer_range: float = 0.02

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        super().validate_architecture()
        if self.hidden_size % (4 * self.num_attention_heads) or self.num_mel_bins % self.patch_size:
            raise ValueError(
                "`hidden_size` must be a multiple of 4 * `num_attention_heads` and `num_mel_bins` a multiple of "
                f"`patch_size`, got {self.hidden_size}, {self.num_attention_heads}, {self.num_mel_bins}, {self.patch_size}."
            )


@auto_docstring(checkpoint="laion/clap2-small-audioxl-s1B")
@strict
class Clap2Config(PreTrainedConfig):
    r"""
    projection_dim (`int`, *optional*, defaults to 512):
        Dimension of the shared audio-text embedding space.
    projection_hidden_act (`str`, *optional*, defaults to `"gelu"`):
        The activation function of the audio projection MLP.
    logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
        Initial value of the logit scale (in log space).

    Example:

    ```python
    >>> from transformers import Clap2Config, Clap2Model

    >>> configuration = Clap2Config()
    >>> model = Clap2Model(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "clap2"
    sub_configs_defaults = {"text_config": Clap2TextConfig, "audio_config": Clap2AudioConfig}

    text_config: dict | PreTrainedConfig | None = None
    audio_config: dict | PreTrainedConfig | None = None
    projection_dim: int = 512
    projection_hidden_act: str = "gelu"
    logit_scale_init_value: float = 2.6592


class Clap2Output(ClapOutput):
    pass


class Clap2RMSNorm(LlamaRMSNorm):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # torch's RMSNorm, as in OpenCLIP and timm, so that outputs match them in every dtype and on every device
        return F.rms_norm(hidden_states, self.weight.shape, self.weight, self.variance_epsilon)


class Clap2MLP(ModernBertMLP):
    """Gated linear unit of each layer, as in ModernBERT (with the SiLU activation it is a SwiGLU)."""

    def __init__(self, config: Clap2AudioConfig | Clap2TextConfig):
        super().__init__(config)


class Clap2LayerScale(Dinov2LayerScale):
    pass


class Clap2TextRotaryEmbedding(CohereRotaryEmbedding):
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # cos/sin are taken before interleaving, as in OpenCLIP, which keeps the outputs bit-identical to it on CPU
        freqs = position_ids[..., None].float() * self.inv_freq.to(device=x.device, dtype=torch.float)
        cos = torch.repeat_interleave(freqs.cos() * self.attention_scaling, 2, dim=-1)
        sin = torch.repeat_interleave(freqs.sin() * self.attention_scaling, 2, dim=-1)
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Clap2AudioRotaryEmbedding(Sam3ViTRotaryEmbedding):
    """2D axial rotary embedding: half of the rotated pairs encode the frequency index of a patch, half its time index."""

    def __init__(self, config: Clap2AudioConfig):
        super().__init__(config)

    def compute_axial_rope_parameters(config: Clap2AudioConfig, device=None, **kwargs) -> tuple[torch.Tensor, float]:
        return super().compute_axial_rope_parameters(config, device, **kwargs)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: (batch_size, num_patches, 2) holding the (frequency, time) index of each patch. As in timm,
        # cos/sin are computed once on the patch grid of the longest clip and gathered per patch: on CPU, cos/sin of
        # another tensor layout can round differently.
        num_freq_patches = self.config.num_mel_bins // self.config.patch_size
        grid = torch.meshgrid(
            torch.arange(num_freq_patches, device=x.device),
            torch.arange(position_ids.shape[1] // num_freq_patches, device=x.device),
            indexing="ij",
        )
        freqs = torch.stack(grid, dim=-1)[..., None].float() * self.inv_freq.to(x.device)
        cos = torch.repeat_interleave(freqs.cos().flatten(-2), 2, dim=-1)[position_ids[..., 0], position_ids[..., 1]]
        sin = torch.repeat_interleave(freqs.sin().flatten(-2), 2, dim=-1)[position_ids[..., 0], position_ids[..., 1]]
        return cos.to(x.dtype), sin.to(x.dtype)

    def recomposition_frequencies(self, freq):
        raise AttributeError("Not needed for Clap2")


class Clap2Attention(nn.Module):
    def __init__(self, config: Clap2AudioConfig | Clap2TextConfig):
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        # fused, as in the original checkpoints: on GPU, cuBLAS rounds a 3 x hidden_size projection differently
        # from three hidden_size ones
        self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.q_norm = Clap2RMSNorm(self.head_dim, eps=config.rms_norm_eps) if config.use_qk_norm else nn.Identity()
        self.k_norm = Clap2RMSNorm(self.head_dim, eps=config.rms_norm_eps) if config.use_qk_norm else nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        qkv = self.qkv_proj(hidden_states).view(*input_shape, 3, -1, self.head_dim).permute(2, 0, 3, 1, 4)
        query_states, key_states, value_states = qkv.unbind(0)
        query_states, key_states = self.q_norm(query_states), self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights


class Clap2EncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Clap2AudioConfig | Clap2TextConfig):
        super().__init__()
        self.input_layernorm = Clap2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Clap2Attention(config)
        self.post_attention_layernorm = Clap2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Clap2MLP(config)
        use_layer_scale = config.layerscale_value is not None
        self.layer_scale1 = Clap2LayerScale(config) if use_layer_scale else nn.Identity()
        self.layer_scale2 = Clap2LayerScale(config) if use_layer_scale else nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states, _ = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings, attention_mask, **kwargs
        )
        hidden_states = residual + self.layer_scale1(hidden_states)
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + self.layer_scale2(hidden_states)


class Clap2Encoder(Siglip2Encoder):
    def __init__(self, config: Clap2AudioConfig | Clap2TextConfig):
        super().__init__(config)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states, position_embeddings, attention_mask, **kwargs)
        return BaseModelOutput(last_hidden_state=hidden_states)


class Clap2AttentionPoolingHead(nn.Module):
    """Pools a sequence into one vector with a learned query attending over the unmasked tokens."""

    def __init__(self, config: Clap2AudioConfig | Clap2TextConfig):
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = False
        self.probe = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.kv_proj = nn.Linear(config.hidden_size, 2 * config.hidden_size, bias=config.attention_bias)
        self.q_norm = Clap2RMSNorm(self.head_dim, eps=config.rms_norm_eps) if config.use_qk_norm else nn.Identity()
        self.k_norm = Clap2RMSNorm(self.head_dim, eps=config.rms_norm_eps) if config.use_qk_norm else nn.Identity()

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        probe = self.probe.expand(batch_size, -1, -1)
        query = self.q_proj(probe).view(batch_size, 1, -1, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(hidden_states).view(batch_size, hidden_states.shape[1], 2, -1, self.head_dim)
        key, value = kv.permute(2, 0, 3, 1, 4).unbind(0)
        query, key = self.q_norm(query), self.k_norm(key)
        # always materialized, as in OpenCLIP and timm: without a mask SDPA can pick another kernel in half precision
        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=probe,
            attention_mask=attention_mask,
            encoder_hidden_states=hidden_states,
            allow_is_bidirectional_skip=False,
        )
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        pooled_output, _ = attention_interface(self, query, key, value, attention_mask, scaling=self.scaling)
        return pooled_output.reshape(batch_size, -1)


class Clap2AudioAttentionPoolingHead(nn.Module):
    """Attention pooling as in [`Clap2AttentionPoolingHead`], followed by an output projection and a residual MLP."""

    def __init__(self, config: Clap2AudioConfig):
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = False
        self.probe = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.kv_proj = nn.Linear(config.hidden_size, 2 * config.hidden_size, bias=config.attention_bias)
        self.q_norm = Clap2RMSNorm(self.head_dim, eps=config.rms_norm_eps) if config.use_qk_norm else nn.Identity()
        self.k_norm = Clap2RMSNorm(self.head_dim, eps=config.rms_norm_eps) if config.use_qk_norm else nn.Identity()
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.layernorm = Clap2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act_fn = ACT2FN[config.pooler_hidden_act]
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        probe = self.probe.expand(batch_size, -1, -1)
        query = self.q_proj(probe).view(batch_size, 1, -1, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(hidden_states).view(batch_size, hidden_states.shape[1], 2, -1, self.head_dim)
        key, value = kv.permute(2, 0, 3, 1, 4).unbind(0)
        query, key = self.q_norm(query), self.k_norm(key)
        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=probe,
            attention_mask=attention_mask,
            encoder_hidden_states=hidden_states,
            allow_is_bidirectional_skip=False,
        )
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        pooled_output, _ = attention_interface(self, query, key, value, attention_mask, scaling=self.scaling)
        hidden_states = self.o_proj(pooled_output.reshape(batch_size, -1))
        return hidden_states + self.fc2(self.act_fn(self.fc1(self.layernorm(hidden_states))))


class Clap2AudioEmbeddings(nn.Module):
    def __init__(self, config: Clap2AudioConfig):
        super().__init__()
        self.patch_embedding = nn.Linear(config.patch_size**2, config.hidden_size, bias=False)
        self.register_tokens = nn.Parameter(torch.zeros(1, config.num_register_tokens, config.hidden_size))
        self.layernorm = Clap2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        embeddings = self.patch_embedding(input_features.to(self.patch_embedding.weight.dtype))
        register_tokens = self.register_tokens.expand(embeddings.shape[0], -1, -1)
        return self.layernorm(torch.cat([register_tokens, embeddings], dim=1))


class Clap2AudioProjection(nn.Module):
    def __init__(self, config: Clap2Config):
        super().__init__()
        self.linear1 = nn.Linear(config.audio_config.hidden_size, config.projection_dim)
        self.activation = ACT2FN[config.projection_hidden_act]
        self.linear2 = nn.Linear(config.projection_dim, config.projection_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.activation(self.linear1(hidden_states)))


@auto_docstring
class Clap2PreTrainedModel(PreTrainedModel):
    config: Clap2Config
    base_model_prefix = "clap2"
    input_modalities = ("audio", "text")
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "Clap2AudioEmbeddings",
        "Clap2EncoderLayer",
        "Clap2AttentionPoolingHead",
        "Clap2AudioAttentionPoolingHead",
    ]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_record_outputs = {"hidden_states": Clap2EncoderLayer, "attentions": Clap2Attention}

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Clap2AudioEmbeddings):
            init.normal_(module.register_tokens, std=1e-6)
        elif isinstance(module, (Clap2AttentionPoolingHead, Clap2AudioAttentionPoolingHead)):
            init.normal_(module.probe, std=module.probe.shape[-1] ** -0.5)
        elif isinstance(module, Clap2LayerScale):
            init.constant_(module.lambda1, self.config.layerscale_value)
        elif isinstance(module, Clap2Model):
            init.constant_(module.logit_scale, self.config.logit_scale_init_value)


@auto_docstring(
    custom_intro="""
    The text encoder of CLAP2, without the projection into the shared audio-text space.
    """
)
class Clap2TextModel(Clap2PreTrainedModel):
    config: Clap2TextConfig
    base_model_prefix = "text_model"
    input_modalities = ("text",)

    def __init__(self, config: Clap2TextConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.rotary_emb = Clap2TextRotaryEmbedding(config)
        self.encoder = Clap2Encoder(config)
        self.norm = Clap2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = Clap2AttentionPoolingHead(config)
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        hidden_states = self.embed_tokens(input_ids)
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device)[None]
        encoder_outputs: BaseModelOutput = self.encoder(
            hidden_states,
            position_embeddings=self.rotary_emb(hidden_states, position_ids),
            # always materialized, as in OpenCLIP: an all-ones mask would select another SDPA kernel in half precision
            attention_mask=create_bidirectional_mask(
                config=self.config,
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                allow_is_bidirectional_skip=False,
            ),
            **kwargs,
        )
        last_hidden_state = self.norm(encoder_outputs.last_hidden_state)
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state, pooler_output=self.head(last_hidden_state, attention_mask)
        )


@auto_docstring(
    custom_intro="""
    The audio encoder of CLAP2, without the projection into the shared audio-text space.
    """
)
class Clap2AudioModel(Clap2PreTrainedModel):
    config: Clap2AudioConfig
    base_model_prefix = "audio_model"
    main_input_name = "input_features"
    input_modalities = ("audio",)
    _input_embed_layer = "patch_embedding"

    def __init__(self, config: Clap2AudioConfig):
        super().__init__(config)
        self.embeddings = Clap2AudioEmbeddings(config)
        self.rotary_emb = Clap2AudioRotaryEmbedding(config)
        self.encoder = Clap2Encoder(config)
        self.norm = Clap2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = Clap2AudioAttentionPoolingHead(config)
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        input_features (`torch.Tensor` of shape `(batch_size, num_patches, patch_size * patch_size)`):
            Flattened log-mel patches, frequency-major, as returned by [`Clap2FeatureExtractor`].
        input_features_mask (`torch.Tensor` of shape `(batch_size, num_patches)`, *optional*):
            Mask to avoid attending to padding patches. Mask values selected in `[0, 1]`:

            - 1 for patches that are **not masked**,
            - 0 for patches that are **masked**.
        """
        if input_features_mask is None:
            input_features_mask = torch.ones(input_features.shape[:2], dtype=torch.long, device=input_features.device)
        # A clip with `num_time_patches` columns holds its patches frequency-major: its i-th patch sits at
        # (i // T, i % T). Counting only unmasked patches makes this independent of the padding side.
        num_freq_patches = self.config.num_mel_bins // self.config.patch_size
        num_time_patches = input_features_mask.sum(-1, keepdim=True) // num_freq_patches
        patch_index = input_features_mask.cumsum(-1) - 1
        position_ids = torch.stack([patch_index // num_time_patches, patch_index % num_time_patches], dim=-1)
        position_ids = position_ids * input_features_mask[..., None]  # padding patches point at (0, 0)

        hidden_states = self.embeddings(input_features)
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        num_registers = self.config.num_register_tokens
        # register tokens are not rotated
        position_embeddings = (F.pad(cos, (0, 0, num_registers, 0), value=1.0), F.pad(sin, (0, 0, num_registers, 0)))
        # always materialized, as in timm: an all-ones mask would select another SDPA kernel in half precision
        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=F.pad(input_features_mask, (num_registers, 0), value=1),
            allow_is_bidirectional_skip=False,
        )
        encoder_outputs: BaseModelOutput = self.encoder(
            hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask, **kwargs
        )
        last_hidden_state = self.norm(encoder_outputs.last_hidden_state)[:, num_registers:]
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state, pooler_output=self.head(last_hidden_state, input_features_mask)
        )


@auto_docstring
class Clap2Model(Clap2PreTrainedModel):
    def __init__(self, config: Clap2Config):
        super().__init__(config)
        self.text_model = Clap2TextModel._from_config(config.text_config)
        self.audio_model = Clap2AudioModel._from_config(config.audio_config)
        self.text_projection = nn.Linear(config.text_config.hidden_size, config.projection_dim, bias=False)
        self.audio_projection = Clap2AudioProjection(config)
        self.logit_scale = nn.Parameter(torch.tensor(config.logit_scale_init_value))
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.embed_tokens

    def set_input_embeddings(self, value: nn.Module):
        self.text_model.embed_tokens = value

    @can_return_tuple
    @auto_docstring
    def get_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        Examples:

        ```python
        >>> import torch
        >>> from transformers import AutoModel, AutoProcessor

        >>> model = AutoModel.from_pretrained("laion/clap2-small-audioxl-s1B")
        >>> processor = AutoProcessor.from_pretrained("laion/clap2-small-audioxl-s1B")

        >>> inputs = processor(text=["the sound of a cat", "the sound of a dog"], padding=True, return_tensors="pt")
        >>> with torch.inference_mode():
        ...     text_features = model.get_text_features(**inputs).pooler_output
        ```"""
        text_outputs: BaseModelOutputWithPooling = self.text_model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True, **kwargs
        )
        text_outputs.pooler_output = self.text_projection(text_outputs.pooler_output)
        return text_outputs

    @can_return_tuple
    @auto_docstring
    def get_audio_features(
        self,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        input_features (`torch.Tensor` of shape `(batch_size, num_patches, patch_size * patch_size)`):
            Flattened log-mel patches, frequency-major, as returned by [`Clap2FeatureExtractor`].
        input_features_mask (`torch.Tensor` of shape `(batch_size, num_patches)`, *optional*):
            Mask to avoid attending to padding patches. Mask values selected in `[0, 1]`:

            - 1 for patches that are **not masked**,
            - 0 for patches that are **masked**.

        Examples:

        ```python
        >>> import torch
        >>> from datasets import Audio, load_dataset
        >>> from transformers import AutoModel, AutoProcessor

        >>> model = AutoModel.from_pretrained("laion/clap2-small-audioxl-s1B")
        >>> processor = AutoProcessor.from_pretrained("laion/clap2-small-audioxl-s1B")

        >>> dataset = load_dataset("hf-internal-testing/ashraq-esc50-1-dog-example", split="train")
        >>> dataset = dataset.cast_column("audio", Audio(sampling_rate=32000))
        >>> audio = dataset[0]["audio"]["array"]

        >>> inputs = processor(audio=audio, sampling_rate=32000, return_tensors="pt")
        >>> with torch.inference_mode():
        ...     audio_features = model.get_audio_features(**inputs).pooler_output
        ```"""
        audio_outputs: BaseModelOutputWithPooling = self.audio_model(
            input_features=input_features, input_features_mask=input_features_mask, return_dict=True, **kwargs
        )
        audio_outputs.pooler_output = self.audio_projection(audio_outputs.pooler_output)
        return audio_outputs

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        return_loss: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Clap2Output:
        r"""
        input_features (`torch.Tensor` of shape `(batch_size, num_patches, patch_size * patch_size)`):
            Flattened log-mel patches, frequency-major, as returned by [`Clap2FeatureExtractor`].
        input_features_mask (`torch.Tensor` of shape `(batch_size, num_patches)`, *optional*):
            Mask to avoid attending to padding patches. Mask values selected in `[0, 1]`:

            - 1 for patches that are **not masked**,
            - 0 for patches that are **masked**.
        return_loss (`bool`, *optional*):
            Whether or not to return the contrastive loss.

        Examples:

        ```python
        >>> import torch
        >>> from datasets import Audio, load_dataset
        >>> from transformers import AutoModel, AutoProcessor

        >>> model = AutoModel.from_pretrained("laion/clap2-small-audioxl-s1B")
        >>> processor = AutoProcessor.from_pretrained("laion/clap2-small-audioxl-s1B")

        >>> dataset = load_dataset("hf-internal-testing/ashraq-esc50-1-dog-example", split="train")
        >>> dataset = dataset.cast_column("audio", Audio(sampling_rate=32000))
        >>> audio = dataset[0]["audio"]["array"]
        >>> texts = ["the sound of a dog", "the sound of a vacuum cleaner"]

        >>> inputs = processor(text=texts, audio=audio, sampling_rate=32000, padding=True, return_tensors="pt")
        >>> with torch.inference_mode():
        ...     outputs = model(**inputs)
        >>> probs = outputs.logits_per_audio.softmax(dim=-1)
        ```"""
        audio_outputs: BaseModelOutputWithPooling = self.audio_model(
            input_features=input_features, input_features_mask=input_features_mask, return_dict=True, **kwargs
        )
        text_outputs: BaseModelOutputWithPooling = self.text_model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True, **kwargs
        )
        audio_embeds = self.audio_projection(audio_outputs.pooler_output)
        text_embeds = self.text_projection(text_outputs.pooler_output)
        audio_embeds = audio_embeds / audio_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        logits_per_audio = (
            self.logit_scale.exp().to(audio_embeds.device) * audio_embeds @ text_embeds.t().to(audio_embeds.device)
        )
        logits_per_text = logits_per_audio.t()

        loss = None
        if return_loss:
            loss = (contrastive_loss(logits_per_audio) + contrastive_loss(logits_per_text)) / 2.0

        return Clap2Output(
            loss=loss,
            logits_per_audio=logits_per_audio,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            audio_embeds=audio_embeds,
            text_model_output=text_outputs,
            audio_model_output=audio_outputs,
        )


__all__ = [
    "Clap2AudioConfig",
    "Clap2Config",
    "Clap2TextConfig",
    "Clap2AudioModel",
    "Clap2Model",
    "Clap2PreTrainedModel",
    "Clap2TextModel",
]
