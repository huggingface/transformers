# Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Qwen3TTS model."""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from huggingface_hub.dataclasses import strict

from ...activations import ACT2FN
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...generation import GenerationMixin
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, ModelOutput
from ...modeling_rope_utils import RopeParameters, dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import auto_docstring, logging
from ...utils.generic import maybe_autocast
from ..qwen2.modeling_qwen2 import eager_attention_forward
from ..qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniDiTConfig
from ..qwen2_5_omni.modeling_qwen2_5_omni import (
    ECAPA_TimeDelayNet,
    Qwen2_5OmniTalkerModel,
)
from ..qwen2_vl.modeling_qwen2_vl import apply_multimodal_rotary_pos_emb
from ..qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3MLP,
    Qwen3Model,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
)
from ..voxtral.modeling_voxtral import VoxtralMultiModalProjector
from ..qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeRotaryEmbedding,
    Qwen3OmniMoeTalkerTextMLP,
)
from .generation_qwen3_tts import Qwen3TTSGenerationMixin


logger = logging.get_logger(__name__)


@strict
class Qwen3TTSSpeakerEncoderConfig(Qwen2_5OmniDiTConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen3TTSSpeakerEncoder`].
    It is used to instantiate a Qwen3-TTS speaker encoder model according to the specified arguments,
    defining the model architecture. The architecture is based on the ECAPA-TDNN model.

    Args:
        mel_dim (`int`, *optional*, defaults to 128):
            The dimension of the input mel-spectrogram.
        enc_dim (`int`, *optional*, defaults to 1024):
            The dimension of the final speaker embedding.
        enc_channels (`list[int]`, *optional*, defaults to `[512, 512, 512, 512, 1536]`):
            A list of output channels for each TDNN/SERes2Net layer in the encoder.
        enc_kernel_sizes (`list[int]`, *optional*, defaults to `[5, 3, 3, 3, 1]`):
            A list of kernel sizes for each layer in the encoder, corresponding to `enc_channels`.
        enc_dilations (`list[int]`, *optional*, defaults to `[1, 2, 3, 4, 1]`):
            A list of dilations for each layer in the encoder, corresponding to `enc_channels`.
        enc_attention_channels (`int`, *optional*, defaults to 128):
            The number of attention channels in the `AttentiveStatisticsPooling` layer.
        enc_res2net_scale (`int`, *optional*, defaults to 8):
            The scale of the `Res2NetBlock` in the encoder.
        enc_se_channels (`int`, *optional*, defaults to 128):
            The number of channels in the squeeze part of the `SqueezeExcitationBlock`.
        sample_rate (`int`, *optional*, defaults to 24000):
            The sample rate of the audio.
    """

    # Set an explicit `model_type` so the converter does not derive a mangled value from the class name; this follows
    # the qwen2_5_omni convention where every sub-config carries a descriptive (unregistered) `model_type`.
    model_type = "qwen3_tts_speaker_encoder"
    base_config_key = "speaker_encoder_config"

    # ECAPA-TDNN fields kept from the DiT config, re-declared with the Qwen3-TTS speaker encoder defaults.
    mel_dim: int = 128
    enc_dim: int = 1024
    enc_channels: list[int] | tuple[int, ...] = (512, 512, 512, 512, 1536)
    enc_kernel_sizes: list[int] | tuple[int, ...] = (5, 3, 3, 3, 1)
    enc_dilations: list[int] | tuple[int, ...] = (1, 2, 3, 4, 1)
    enc_attention_channels: int = 128
    enc_res2net_scale: int = 8
    enc_se_channels: int = 128
    sample_rate: int = 24000

    # DiT-only fields removed: the speaker encoder only instantiates the ECAPA-TDNN submodule.
    hidden_size = AttributeError()
    num_hidden_layers = AttributeError()
    num_attention_heads = AttributeError()
    ff_mult = AttributeError()
    emb_dim = AttributeError()
    head_dim = AttributeError()
    rope_parameters = AttributeError()
    max_position_embeddings = AttributeError()
    block_size = AttributeError()
    look_ahead_layers = AttributeError()
    look_backward_layers = AttributeError()
    repeats = AttributeError()
    num_embeds = AttributeError()
    dropout = AttributeError()
    enc_emb_dim = AttributeError()


@strict
class Qwen3TTSTalkerCodePredictorConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen3TTSTalkerCodePredictorModel`].
    It is used to instantiate a Qwen3-TTS code predictor model according to the specified arguments,
    defining the model architecture.

    Args:
        vocab_size (`int`, *optional*, defaults to 2048):
            Vocabulary size of the Qwen3-TTS code predictor model.
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 5):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            The number of key_value heads for Grouped Query Attention (GQA).
        head_dim (`int`, *optional*, defaults to 128):
            The attention head dimension.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie word embedding weights with output projection weights.
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers.
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention window size.
        max_window_layers (`int`, *optional*, defaults to 28):
            The number of layers using full attention.
        layer_types (`list[str]`, *optional*):
            List of attention layer types for each hidden layer. Defaults to alternating between `"full_attention"`
            and `"sliding_attention"` based on `max_window_layers`.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        num_code_groups (`int`, *optional*, defaults to 32):
            Number of code groups (codebooks).
        pad_token_id (`int`, *optional*):
            Padding token ID.
    """

    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int | None = 2048,
        hidden_size: int | None = 1024,
        intermediate_size: int | None = 3072,
        num_hidden_layers: int | None = 5,
        num_attention_heads: int | None = 16,
        num_key_value_heads: int | None = 8,
        head_dim: int | None = 128,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 32768,
        initializer_range: float | None = 0.02,
        rms_norm_eps: float | None = 1e-6,
        use_cache: bool | None = True,
        tie_word_embeddings: bool | None = False,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        attention_bias: bool | None = False,
        use_sliding_window: bool | None = False,
        sliding_window: int | None = 4096,
        max_window_layers: int | None = 28,
        layer_types: list[str] | None = None,
        attention_dropout: float | None = 0.0,
        num_code_groups: int | None = 32,
        pad_token_id: int | None = None,
        **kwargs,
    ):
        self.pad_token_id = pad_token_id
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if self.use_sliding_window else None
        self.max_window_layers = max_window_layers

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_parameters = (
            rope_parameters if rope_parameters is not None else {"rope_type": "default", "rope_theta": 500000.0}
        )
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        self.num_code_groups = num_code_groups


@strict
class Qwen3TTSTalkerConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen3TTSTalkerModel`].
    It is used to instantiate a Qwen3-TTS talker model according to the specified arguments,
    defining the model architecture.

    Args:
        code_predictor_config (`Qwen3TTSTalkerCodePredictorConfig`, *optional*):
            Configuration for the code predictor sub-model.
        vocab_size (`int`, *optional*, defaults to 3072):
            Vocabulary size of the Qwen3-TTS talker model.
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 2048):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 20):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer.
        num_key_value_heads (`int`, *optional*, defaults to 2):
            The number of key_value heads for Grouped Query Attention (GQA).
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie word embedding weights with output projection weights.
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers.
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention window size.
        max_window_layers (`int`, *optional*, defaults to 28):
            The number of layers using full attention.
        layer_types (`list[str]`, *optional*):
            List of attention layer types for each hidden layer. Defaults to `"full_attention"` for every layer
            unless sliding window attention is enabled.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        num_code_groups (`int`, *optional*, defaults to 32):
            Number of code groups (codebooks).
        text_hidden_size (`int`, *optional*, defaults to 2048):
            The dimension of the text embedding in the talker.
        codec_eos_token_id (`int`, *optional*, defaults to 4198):
            The end-of-sequence token ID for codec tokens.
        codec_think_id (`int`, *optional*, defaults to 4202):
            Token ID used to signal thinking mode in codec generation.
        codec_nothink_id (`int`, *optional*, defaults to 4203):
            Token ID used to signal non-thinking mode in codec generation.
        codec_think_bos_id (`int`, *optional*, defaults to 4204):
            Beginning-of-sequence token ID for codec thinking mode.
        codec_think_eos_id (`int`, *optional*, defaults to 4205):
            End-of-sequence token ID for codec thinking mode.
        codec_pad_id (`int`, *optional*, defaults to 4196):
            The padding token ID for codec tokens.
        codec_bos_id (`int`, *optional*, defaults to 4197):
            The beginning-of-sequence token ID for codec tokens.
        spk_id (`int`, *optional*):
            Speaker ID for built-in voice presets.
        spk_is_dialect (`bool`, *optional*):
            Whether the speaker uses a dialect variant.
        codec_language_id (`int`, *optional*):
            Language ID for codec generation.
        text_vocab_size (`int`, *optional*, defaults to 152064):
            Vocabulary size of the text tokenizer.
        pad_token_id (`int`, *optional*):
            Padding token ID.
    """

    base_config_key = "talker_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {"code_predictor_config": Qwen3TTSTalkerCodePredictorConfig}

    def __init__(
        self,
        code_predictor_config: dict | None = None,
        vocab_size: int | None = 3072,
        hidden_size: int | None = 1024,
        intermediate_size: int | None = 2048,
        num_hidden_layers: int | None = 20,
        num_attention_heads: int | None = 16,
        num_key_value_heads: int | None = 2,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 32768,
        initializer_range: float | None = 0.02,
        rms_norm_eps: float | None = 1e-6,
        use_cache: bool | None = True,
        tie_word_embeddings: bool | None = False,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        attention_bias: bool | None = False,
        use_sliding_window: bool | None = False,
        sliding_window: int | None = 4096,
        max_window_layers: int | None = 28,
        layer_types: list[str] | None = None,
        attention_dropout: float | None = 0.0,
        num_code_groups: int | None = 32,
        text_hidden_size: int | None = 2048,
        codec_eos_token_id: int | None = 4198,
        codec_think_id: int | None = 4202,
        codec_nothink_id: int | None = 4203,
        codec_think_bos_id: int | None = 4204,
        codec_think_eos_id: int | None = 4205,
        codec_pad_id: int | None = 4196,
        codec_bos_id: int | None = 4197,
        spk_id: int | None = None,
        spk_is_dialect: bool | None = None,
        codec_language_id: int | None = None,
        text_vocab_size: int | None = 152064,
        pad_token_id: int | None = None,
        **kwargs,
    ):
        self.pad_token_id = pad_token_id
        self.text_vocab_size = text_vocab_size

        # Build sub-config BEFORE super().__init__() so that the _attn_implementation
        # setter (triggered by super()) can propagate to it via sub_configs.
        if code_predictor_config is None:
            self.code_predictor_config = Qwen3TTSTalkerCodePredictorConfig()
            logger.info("code_predictor_config is None. Initializing code_predictor model with default values")
        elif isinstance(code_predictor_config, Qwen3TTSTalkerCodePredictorConfig):
            self.code_predictor_config = code_predictor_config
        else:
            self.code_predictor_config = Qwen3TTSTalkerCodePredictorConfig(**code_predictor_config)

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        self.max_window_layers = max_window_layers

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_parameters = (
            rope_parameters if rope_parameters is not None else {"rope_type": "default", "rope_theta": 500000.0}
        )
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        self.num_code_groups = num_code_groups
        self.text_hidden_size = text_hidden_size
        self.codec_eos_token_id = codec_eos_token_id
        self.codec_think_id = codec_think_id
        self.codec_language_id = codec_language_id
        self.codec_nothink_id = codec_nothink_id
        self.codec_think_bos_id = codec_think_bos_id
        self.codec_think_eos_id = codec_think_eos_id
        self.codec_pad_id = codec_pad_id
        self.codec_bos_id = codec_bos_id
        self.spk_id = spk_id
        self.spk_is_dialect = spk_is_dialect


@auto_docstring(checkpoint="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
@strict
class Qwen3TTSConfig(PreTrainedConfig):
    model_type = "qwen3_tts"
    sub_configs = {
        "talker_config": Qwen3TTSTalkerConfig,
        "speaker_encoder_config": Qwen3TTSSpeakerEncoderConfig,
    }

    def __init__(
        self,
        talker_config: dict | None = None,
        speaker_encoder_config: dict | None = None,
        tokenizer_type: str | None = None,
        tts_model_size: str | None = None,
        tts_model_type: str | None = None,
        im_start_token_id: int | None = 151644,
        im_end_token_id: int | None = 151645,
        tts_pad_token_id: int | None = 151671,
        tts_bos_token_id: int | None = 151672,
        tts_eos_token_id: int | None = 151673,
        **kwargs,
    ):
        r"""
        This is the configuration class to store the configuration of a [`Qwen3TTSForConditionalGeneration`].
        It is used to instantiate a Qwen3-TTS model according to the specified arguments, defining the model architecture.

        Args:
            talker_config (`Qwen3TTSTalkerConfig`, *optional*):
                Configuration for the talker sub-model (text-to-acoustic backbone).
            speaker_encoder_config (`Qwen3TTSSpeakerEncoderConfig`, *optional*):
                Configuration for the speaker encoder sub-model (extracts speaker embeddings).
            tokenizer_type (`str`, *optional*):
                Type of audio tokenizer to use (e.g., "12hz", "25hz").
            tts_model_size (`str`, *optional*):
                Size of the TTS model.
            tts_model_type (`str`, *optional*):
                Type of TTS model.
            im_start_token_id (`int`, *optional*, defaults to 151644):
                The beginning-of-image token ID (used as special marker in input).
            im_end_token_id (`int`, *optional*, defaults to 151645):
                The end-of-image token ID (used as special marker in input).
            tts_pad_token_id (`int`, *optional*, defaults to 151671):
                The padding token ID for TTS generation.
            tts_bos_token_id (`int`, *optional*, defaults to 151672):
                The beginning-of-sequence token ID for TTS generation.
            tts_eos_token_id (`int`, *optional*, defaults to 151673):
                The end-of-sequence token ID for TTS generation.
        """
        super().__init__(**kwargs)

        if talker_config is None:
            talker_config = {}
            logger.info("talker_config is None. Initializing talker model with default values")

        self.talker_config = Qwen3TTSTalkerConfig(**talker_config)
        # A speaker encoder is only present for voice-cloning ("base") checkpoints; leave it None otherwise.
        self.speaker_encoder_config = (
            Qwen3TTSSpeakerEncoderConfig(**speaker_encoder_config) if speaker_encoder_config is not None else None
        )

        self.tokenizer_type = tokenizer_type
        self.tts_model_size = tts_model_size
        self.tts_model_type = tts_model_type

        self.im_start_token_id = im_start_token_id
        self.im_end_token_id = im_end_token_id
        self.tts_pad_token_id = tts_pad_token_id
        self.tts_bos_token_id = tts_bos_token_id
        self.tts_eos_token_id = tts_eos_token_id


class Qwen3TTSRMSNorm(Qwen3RMSNorm):
    pass


class Qwen3TTSMlp(Qwen3MLP):
    pass


class Qwen3TTSTalkerTextMLP(Qwen3OmniMoeTalkerTextMLP):
    pass


class Qwen3TTSRotaryEmbedding(Qwen3OmniMoeRotaryEmbedding):
    pass


class Qwen3TTSTalkerRotaryEmbedding(Qwen3OmniMoeRotaryEmbedding):
    """3D multimodal rotary embedding (temporal / height / width) for Talker.

    position_ids expected shape: (3, batch, seq).
    """

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        # position_ids: (3, batch, seq) for temporal, height, width
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1).to(x.device)
        )
        position_ids_expanded = position_ids[:, :, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3TTSTalkerAttention(Qwen3Attention):
    """Talker attention with 3D multimodal RoPE.

    Reuses [`Qwen3Attention`] (projections + per-head q/k norm) and only swaps the 1D RoPE for the multimodal
    [`~models.qwen2_vl.modeling_qwen2_vl.apply_multimodal_rotary_pos_emb`]. The original checkpoints use an
    interleaved mRoPE layout; since the talker always feeds identical position ids to the temporal/height/width
    sections, the interleaved and non-interleaved layouts are numerically equivalent, so we use the standard
    (non-interleaved) implementation and the conversion script writes `interleaved=False` to the config.
    """

    def __init__(self, config: Qwen3TTSTalkerConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.sliding_window = getattr(config, "sliding_window", None)

        rope_params = config.rope_parameters if config.rope_parameters is not None else {}
        # mrope_section describes half-dimension splits (will be repeated *2 in apply function)
        half_dim = self.head_dim // 2
        self.mrope_section = rope_params.get(
            "mrope_section",
            [half_dim // 3, half_dim // 3, half_dim - 2 * (half_dim // 3)],
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.mrope_section
        )

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

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
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3TTSCodePredictorAttention(Qwen3Attention):
    """Code Predictor attention — inherited from Qwen3Attention (1D RoPE)."""

    pass


class Qwen3TTSTalkerDecoderLayer(Qwen3DecoderLayer):
    """Talker decoder layer. Reuses [`Qwen3DecoderLayer`]'s forward with a Talker mRoPE attention and text MLP."""

    def __init__(self, config: Qwen3TTSTalkerConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3TTSTalkerAttention(config, layer_idx)
        self.mlp = Qwen3TTSTalkerTextMLP(config, intermediate_size=config.intermediate_size)
        self.input_layernorm = Qwen3TTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3TTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]


class Qwen3TTSDecoderLayer(Qwen3DecoderLayer):
    """Code Predictor decoder layer. Reuses [`Qwen3DecoderLayer`]'s forward with the code-predictor attention."""

    def __init__(self, config: Qwen3TTSTalkerCodePredictorConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3TTSCodePredictorAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3TTSMlp(config)
        self.input_layernorm = Qwen3TTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3TTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]


# Components (TimeDelayNetBlock, SqueezeExcitationRes2NetBlock,
# AttentiveStatisticsPooling) imported from qwen2_5_omni.


class Qwen3TTSSpeakerEncoder(ECAPA_TimeDelayNet):
    """ECAPA-TDNN speaker encoder (inherited from qwen2_5_omni)."""

    def __init__(self, config: Qwen3TTSSpeakerEncoderConfig):
        super().__init__(config)


class Qwen3TTSTalkerResizeMLP(VoxtralMultiModalProjector):
    def __init__(self, config: Qwen3TTSTalkerConfig):
        super().__init__()
        self.linear_1 = nn.Linear(config.text_hidden_size, config.text_hidden_size, bias=True)
        self.linear_2 = nn.Linear(config.text_hidden_size, config.hidden_size, bias=True)
        self.act = ACT2FN[config.hidden_act]


@dataclass
class Qwen3TTSTalkerCodePredictorOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: list[torch.FloatTensor] | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    generation_steps: int | None = None


@dataclass
class Qwen3TTSTalkerOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: list[torch.FloatTensor] | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    past_hidden: torch.FloatTensor | None = None
    generation_step: int | None = None
    trailing_text_hidden: torch.FloatTensor | None = None
    tts_pad_embed: torch.FloatTensor | None = None


class Qwen3TTSBasePreTrainedModel(Qwen3PreTrainedModel):
    """Common base for all Qwen3TTS PreTrainedModel classes."""

    _no_split_modules = []
    # Qwen3TTS has separate Talker/CodePredictor attention classes, so the generic
    # output-recording hook from Qwen3 does not apply.
    _can_record_outputs = {}


class Qwen3TTSPreTrainedModel(Qwen3TTSBasePreTrainedModel):
    config_class = Qwen3TTSConfig
    _no_split_modules = ["Qwen3TTSTalkerDecoderLayer", "Qwen3TTSDecoderLayer"]
    _supports_cache_class = True
    _supports_static_cache = False


class Qwen3TTSTalkerTextPreTrainedModel(Qwen3TTSBasePreTrainedModel):
    """PreTrainedModel for Talker-related models."""

    _no_split_modules = []
    _supports_cache_class = True
    _supports_static_cache = False


class Qwen3TTSTalkerModel(Qwen2_5OmniTalkerModel):
    """Talker model: text encoder with dual codec+text embeddings.

    Reuses [`Qwen2_5OmniTalkerModel`]'s 3D-mRoPE decoder `forward`; only the embeddings (dual codec + text) and the
    Talker-specific decoder layer / rotary embedding differ.
    """

    config_class = Qwen3TTSTalkerConfig
    base_model_prefix = "talker.model"
    input_modalities = ("text",)
    # `generate` consumes the per-layer hidden states (the last entry, tied to `last_hidden_state`, feeds the
    # code predictor), so record them from the Talker-specific layer/attention classes.
    _can_record_outputs = {
        "hidden_states": Qwen3TTSTalkerDecoderLayer,
        "attentions": Qwen3TTSTalkerAttention,
    }

    def __init__(self, config: Qwen3TTSTalkerConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.layers = nn.ModuleList(
            [Qwen3TTSTalkerDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3TTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3TTSTalkerRotaryEmbedding(config)
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        self.gradient_checkpointing = False
        # The codec embedding is the talker's input embedding (named `embed_tokens` so the inherited
        # 3D-mRoPE forward and `get_input_embeddings` work unchanged); `text_embedding` is an extra branch.
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.text_embedding = nn.Embedding(config.text_vocab_size, config.text_hidden_size)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_text_embeddings(self):
        return self.text_embedding

    def set_input_embeddings(self, value):
        self.embed_tokens = value


class Qwen3TTSTalkerCodePredictorModel(Qwen3Model):
    """Code predictor model: sequential multi-codebook refinement.

    Reuses [`Qwen3Model`]'s decoder `forward`; only the input embedding differs (a per-codebook `codec_embedding`
    `ModuleList` instead of a single `embed_tokens`).
    """

    config_class = Qwen3TTSTalkerCodePredictorConfig
    base_model_prefix = "talker.code_predictor.model"
    _can_record_outputs = {
        "hidden_states": Qwen3TTSDecoderLayer,
        "attentions": Qwen3TTSCodePredictorAttention,
    }

    def __init__(self, config: Qwen3TTSTalkerCodePredictorConfig, embedding_dim: int):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.layers = nn.ModuleList(
            [Qwen3TTSDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3TTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3TTSRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        # Per-codebook input embeddings; the talker sums these externally and feeds `inputs_embeds`, so there is no
        # single `embed_tokens`.
        del self.embed_tokens
        self.codec_embedding = nn.ModuleList(
            [nn.Embedding(config.vocab_size, embedding_dim) for _ in range(config.num_code_groups - 1)]
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.codec_embedding

    def set_input_embeddings(self, value):
        self.codec_embedding = value


class Qwen3TTSTalkerCodePredictorModelForConditionalGeneration(Qwen3TTSPreTrainedModel, GenerationMixin):
    """Wrapper for CodePredictorModel with generation capabilities."""

    config_class = Qwen3TTSTalkerCodePredictorConfig
    base_model_prefix = "talker.code_predictor"

    def __init__(self, config: Qwen3TTSTalkerCodePredictorConfig, talker_config: Qwen3TTSTalkerConfig):
        super().__init__(config)
        self.model = Qwen3TTSTalkerCodePredictorModel(config, talker_config.hidden_size)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, (config.num_code_groups - 1) * config.vocab_size, bias=False)

        if config.hidden_size != talker_config.hidden_size:
            self.small_to_mtp_projection = nn.Linear(talker_config.hidden_size, config.hidden_size, bias=True)
        else:
            self.small_to_mtp_projection = nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        generation_steps: int | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # Prefill stage: derive generation_steps from sequence length
        if inputs_embeds is not None and inputs_embeds.shape[1] > 1:
            generation_steps = inputs_embeds.shape[1] - 2
        # Generation stage: look up step-specific embedding
        else:
            inputs_embeds = self.model.get_input_embeddings()[generation_steps - 1](input_ids)

        inputs_embeds = self.small_to_mtp_projection(inputs_embeds)

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states).view(
            *hidden_states.shape[:-1], self.config.num_code_groups - 1, self.config.vocab_size
        )
        logits = logits[..., generation_steps, :]

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.reshape(-1, self.config.vocab_size), labels.reshape(-1))

        return Qwen3TTSTalkerCodePredictorOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            generation_steps=generation_steps + 1,
        )

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder=False, num_new_tokens=1):
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder, num_new_tokens
        )
        model_kwargs["generation_steps"] = outputs.generation_steps
        return model_kwargs

class Qwen3TTSForConditionalGeneration(Qwen3TTSPreTrainedModel, Qwen3TTSGenerationMixin):
    """Main Qwen3-TTS model for text-to-acoustic generation."""

    config_class = Qwen3TTSConfig
    main_input_name = "input_ids"

    def __init__(self, config: Qwen3TTSConfig):
        super().__init__(config)
        self.config = config
        talker_config = config.talker_config

        # Talker: text encoder + codec head + code predictor
        self.model = Qwen3TTSTalkerModel(talker_config)
        self.vocab_size = talker_config.vocab_size
        self.text_projection = Qwen3TTSTalkerResizeMLP(talker_config)
        self.codec_head = nn.Linear(talker_config.hidden_size, talker_config.vocab_size, bias=False)
        self.code_predictor = Qwen3TTSTalkerCodePredictorModelForConditionalGeneration(
            config=talker_config.code_predictor_config,
            talker_config=talker_config,
        )
        self.rope_deltas = None

        # Optional speaker encoder for voice cloning (present only when a speaker_encoder_config is set)
        if config.speaker_encoder_config is not None:
            self.speaker_encoder = Qwen3TTSSpeakerEncoder(config.speaker_encoder_config)
        else:
            self.speaker_encoder = None

        # Optional: speech_tokenizer and generate_config loaded separately
        self.speech_tokenizer = None
        self.generate_config = None

        # Model metadata
        self.supported_speakers = (
            list(talker_config.spk_id.keys())
            if hasattr(talker_config, "spk_id") and talker_config.spk_id is not None
            else []
        )
        self.supported_languages = ["auto"]
        if hasattr(talker_config, "codec_language_id") and talker_config.codec_language_id is not None:
            for language_id in talker_config.codec_language_id.keys():
                if "dialect" not in language_id:
                    self.supported_languages.append(language_id)

        self.speaker_encoder_sample_rate = (
            config.speaker_encoder_config.sample_rate if config.speaker_encoder_config is not None else 24000
        )
        self.tokenizer_type = getattr(config, "tokenizer_type", "qwen2")
        self.tts_model_size = getattr(config, "tts_model_size", "base")
        self.tts_model_type = getattr(config, "tts_model_type", "base")

        # Initialize weights and apply final processing
        self.post_init()

    def load_speech_tokenizer(self, speech_tokenizer):
        """Load the speech tokenizer for audio encoding/decoding."""
        self.speech_tokenizer = speech_tokenizer

    def load_generate_config(self, generate_config):
        """Load the generation configuration."""
        if isinstance(generate_config, str):
            import json

            with open(generate_config, encoding="utf-8") as f:
                generate_config = json.load(f)
        self.generate_config = generate_config

    def get_supported_speakers(self):
        """Get list of supported speakers."""
        return list(self.supported_speakers)

    def get_supported_languages(self):
        """Get list of supported languages."""
        return self.supported_languages

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_text_embeddings(self):
        return self.model.get_text_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.codec_head

    def set_output_embeddings(self, new_embeddings):
        self.codec_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        past_hidden: torch.FloatTensor | None = None,
        trailing_text_hidden: torch.FloatTensor | None = None,
        tts_pad_embed: torch.FloatTensor | None = None,
        generation_step: int | None = None,
        subtalker_dosample: bool | None = None,
        subtalker_top_p: float | None = None,
        subtalker_top_k: int | None = None,
        subtalker_temperature: float | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        # Prefill stage
        if inputs_embeds is not None and inputs_embeds.shape[1] > 1:
            generation_step = -1
            codec_ids = None
        # Generation stage
        else:
            last_id_hidden = self.get_input_embeddings()(input_ids)
            predictor_result = self.code_predictor.generate(
                inputs_embeds=torch.cat((past_hidden, last_id_hidden), dim=1),
                max_new_tokens=self.config.talker_config.num_code_groups - 1,
                do_sample=subtalker_dosample,
                top_p=subtalker_top_p,
                top_k=subtalker_top_k,
                temperature=subtalker_temperature,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            codec_ids = torch.cat((input_ids, predictor_result.sequences), dim=-1)
            codec_hiddens = torch.cat(
                [last_id_hidden]
                + [
                    self.code_predictor.get_input_embeddings()[i](predictor_result.sequences[..., i : i + 1])
                    for i in range(self.config.talker_config.num_code_groups - 1)
                ],
                dim=1,
            )
            inputs_embeds = codec_hiddens.sum(1, keepdim=True)

            if generation_step < trailing_text_hidden.shape[1]:
                inputs_embeds = inputs_embeds + trailing_text_hidden[:, generation_step].unsqueeze(1)
            else:
                inputs_embeds = inputs_embeds + tts_pad_embed

        if attention_mask is not None:
            if (
                cache_position is None
                or (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
            ):
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.get_rope_index(attention_mask)
                rope_deltas = rope_deltas - delta0
                self.rope_deltas = rope_deltas
                # Trim to match actual input length (avoids broadcast during decode when
                # attention_mask covers all past tokens but inputs_embeds is 1 token)
                position_ids = position_ids[:, :, -inputs_embeds.shape[1] :]
            else:
                batch_size, seq_length = input_ids.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.codec_head(hidden_states)

        loss = None
        if labels is not None:
            # Use standard loss computation for now
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.talker_config.vocab_size), labels.view(-1))

        return Qwen3TTSTalkerOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=(outputs.hidden_states, codec_ids),
            attentions=outputs.attentions,
            past_hidden=hidden_states[:, -1:, :],
            generation_step=generation_step + 1,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
        )

    def get_rope_index(
        self,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the 3D rope index based on temporal, height and width."""
        position_ids = attention_mask.float().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        mrope_position_deltas = max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)
        return position_ids, mrope_position_deltas

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder=False, num_new_tokens=1):
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder, num_new_tokens
        )
        model_kwargs["past_hidden"] = outputs.past_hidden
        model_kwargs["generation_step"] = outputs.generation_step
        model_kwargs["trailing_text_hidden"] = outputs.trailing_text_hidden
        model_kwargs["tts_pad_embed"] = outputs.tts_pad_embed
        return model_kwargs


__all__ = [
    "Qwen3TTSConfig",
    "Qwen3TTSTalkerConfig",
    "Qwen3TTSSpeakerEncoderConfig",
    "Qwen3TTSTalkerCodePredictorConfig",
    "Qwen3TTSBasePreTrainedModel",
    "Qwen3TTSPreTrainedModel",
    "Qwen3TTSSpeakerEncoder",
    "Qwen3TTSTalkerModel",
    "Qwen3TTSTalkerTextPreTrainedModel",
    "Qwen3TTSTalkerCodePredictorModel",
    "Qwen3TTSTalkerCodePredictorModelForConditionalGeneration",
    "Qwen3TTSForConditionalGeneration",
]
