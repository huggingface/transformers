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

from dataclasses import dataclass

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...generation import GenerationMixin
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_rope_utils import RopeParameters
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging, torch_compilable_check
from ...utils.import_utils import is_torchdynamo_compiling
from ..qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniDiTConfig
from ..qwen2_5_omni.modeling_qwen2_5_omni import (
    ECAPA_TimeDelayNet,
    Qwen2_5OmniTalkerModel,
)
from ..qwen2_vl.modeling_qwen2_vl import Qwen2VLRotaryEmbedding
from ..qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3MLP,
    Qwen3Model,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
)
from ..qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeRotaryEmbedding,
    Qwen3OmniMoeTalkerTextMLP,
)
from ..voxtral.modeling_voxtral import VoxtralMultiModalProjector
from .generation_qwen3_tts import Qwen3TTSGenerationMixin


logger = logging.get_logger(__name__)


@auto_docstring
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


@auto_docstring
@strict
class Qwen3TTSTalkerCodePredictorConfig(PreTrainedConfig):
    r"""
    num_code_groups (`int`, *optional*, defaults to 32):
        Number of code groups (codebooks).
    """

    keys_to_ignore_at_inference = ["past_key_values"]

    vocab_size: int | None = 2048
    hidden_size: int | None = 1024
    intermediate_size: int | None = 3072
    num_hidden_layers: int | None = 5
    num_attention_heads: int | None = 16
    num_key_value_heads: int | None = 8
    head_dim: int | None = 128
    hidden_act: str | None = "silu"
    max_position_embeddings: int | None = 32768
    initializer_range: float | None = 0.02
    rms_norm_eps: float | None = 1e-6
    use_cache: bool | None = True
    tie_word_embeddings: bool | None = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool | None = False
    use_sliding_window: bool | None = False
    sliding_window: int | None = 4096
    max_window_layers: int | None = 28
    layer_types: list[str] | None = None
    attention_dropout: float | int | None = 0.0
    num_code_groups: int | None = 32
    pad_token_id: int | None = None

    def __post_init__(self, **kwargs):
        self.sliding_window = self.sliding_window if self.use_sliding_window else None
        self.num_key_value_heads = self.num_key_value_heads or self.num_attention_heads
        if self.rope_parameters is None:
            self.rope_parameters = {"rope_type": "default", "rope_theta": 500000.0}
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        super().__post_init__(**kwargs)


@auto_docstring
@strict
class Qwen3TTSTalkerConfig(PreTrainedConfig):
    r"""
    code_predictor_config (`Union[Qwen3TTSTalkerCodePredictorConfig, dict]`, *optional*):
        Configuration for the code predictor sub-model.
    num_code_groups (`int`, *optional*, defaults to 32):
        Number of code groups (codebooks).
    text_hidden_size (`int`, *optional*, defaults to 2048):
        The dimension of the text embedding in the talker.
    codec_eos_token_id (`int`, *optional*, defaults to 2150):
        The end-of-sequence token ID for codec tokens.
    codec_think_id (`int`, *optional*, defaults to 4202):
        Token ID used to signal thinking mode in codec generation.
    codec_nothink_id (`int`, *optional*, defaults to 4203):
        Token ID used to signal non-thinking mode in codec generation.
    codec_think_bos_id (`int`, *optional*, defaults to 4204):
        Beginning-of-sequence token ID for codec thinking mode.
    codec_think_eos_id (`int`, *optional*, defaults to 4205):
        End-of-sequence token ID for codec thinking mode.
    codec_pad_id (`int`, *optional*, defaults to 2148):
        The padding token ID for codec tokens.
    codec_bos_id (`int`, *optional*, defaults to 2149):
        The beginning-of-sequence token ID for codec tokens.
    spk_id (`dict[str, int]`, *optional*):
        Mapping from speaker names to IDs for built-in voice presets.
    spk_is_dialect (`dict[str, bool | str]`, *optional*):
        Mapping from speaker names to dialect names, or `False` for speakers without a dialect variant.
    codec_language_id (`dict[str, int]`, *optional*):
        Mapping from language names to codec generation IDs.
    text_vocab_size (`int`, *optional*, defaults to 152064):
        Vocabulary size of the text tokenizer.
    """

    base_config_key = "talker_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {"code_predictor_config": Qwen3TTSTalkerCodePredictorConfig}
    # The talker applies mRoPE on top of a standard rotary embedding, so `mrope_section` rides along
    # in `rope_parameters` without being part of the `default` rope schema. Same handling as Qwen2-VL.
    ignore_keys_at_rope_validation = {"mrope_section"}

    code_predictor_config: dict | PreTrainedConfig | None = None
    vocab_size: int | None = 3072
    hidden_size: int | None = 1024
    intermediate_size: int | None = 2048
    num_hidden_layers: int | None = 20
    num_attention_heads: int | None = 16
    num_key_value_heads: int | None = 2
    hidden_act: str | None = "silu"
    max_position_embeddings: int | None = 32768
    initializer_range: float | None = 0.02
    rms_norm_eps: float | None = 1e-6
    use_cache: bool | None = True
    tie_word_embeddings: bool | None = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool | None = False
    use_sliding_window: bool | None = False
    sliding_window: int | None = 4096
    max_window_layers: int | None = 28
    layer_types: list[str] | None = None
    attention_dropout: float | int | None = 0.0
    num_code_groups: int | None = 32
    text_hidden_size: int | None = 2048
    codec_eos_token_id: int | None = 2150
    codec_think_id: int | None = 4202
    codec_nothink_id: int | None = 4203
    codec_think_bos_id: int | None = 4204
    codec_think_eos_id: int | None = 4205
    codec_pad_id: int | None = 2148
    codec_bos_id: int | None = 2149
    spk_id: dict[str, int] | None = None
    spk_is_dialect: dict[str, bool | str] | None = None
    codec_language_id: dict[str, int] | None = None
    text_vocab_size: int | None = 152064
    pad_token_id: int | None = None

    def __post_init__(self, **kwargs):
        if self.code_predictor_config is None:
            self.code_predictor_config = Qwen3TTSTalkerCodePredictorConfig()
            logger.info("code_predictor_config is None. Initializing code_predictor model with default values")
        elif isinstance(self.code_predictor_config, dict):
            self.code_predictor_config = Qwen3TTSTalkerCodePredictorConfig(**self.code_predictor_config)

        self.sliding_window = self.sliding_window if self.use_sliding_window else None
        self.num_key_value_heads = self.num_key_value_heads or self.num_attention_heads
        if self.rope_parameters is None:
            self.rope_parameters = {"rope_type": "default", "rope_theta": 500000.0}
        half_dim = (getattr(self, "head_dim", None) or self.hidden_size // self.num_attention_heads) // 2
        self.rope_parameters.setdefault(
            "mrope_section", [half_dim // 3, half_dim // 3, half_dim - 2 * (half_dim // 3)]
        )
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
@strict
class Qwen3TTSConfig(PreTrainedConfig):
    r"""
    talker_config (`Union[Qwen3TTSTalkerConfig, dict]`, *optional*):
        Configuration for the talker sub-model (text-to-acoustic backbone).
    speaker_encoder_config (`Union[Qwen3TTSSpeakerEncoderConfig, dict]`, *optional*):
        Configuration for the speaker encoder sub-model (extracts speaker embeddings).
    tokenizer_type (`str`, *optional*):
        Type of audio tokenizer to use (e.g., "12hz", "25hz").
    tts_model_size (`str`, *optional*):
        Size of the TTS model.
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
    code_predictor_loss_weight (`float`, *optional*, defaults to 0.3):
        Weight of the residual-codebook loss in the combined training objective.
    """

    model_type = "qwen3_tts"
    sub_configs = {
        "talker_config": Qwen3TTSTalkerConfig,
        "speaker_encoder_config": Qwen3TTSSpeakerEncoderConfig,
    }

    talker_config: dict | PreTrainedConfig | None = None
    speaker_encoder_config: dict | PreTrainedConfig | None = None
    tokenizer_type: str | None = None
    tts_model_size: str | None = None
    im_start_token_id: int | None = 151644
    im_end_token_id: int | None = 151645
    tts_pad_token_id: int | None = 151671
    tts_bos_token_id: int | None = 151672
    tts_eos_token_id: int | None = 151673
    code_predictor_loss_weight: float = 0.3

    def __post_init__(self, **kwargs):
        if self.talker_config is None:
            self.talker_config = Qwen3TTSTalkerConfig()
            logger.info("talker_config is None. Initializing talker model with default values")
        elif isinstance(self.talker_config, dict):
            self.talker_config = Qwen3TTSTalkerConfig(**self.talker_config)

        # A speaker encoder is only present for voice-cloning ("base") checkpoints; leave it None otherwise.
        if isinstance(self.speaker_encoder_config, dict):
            self.speaker_encoder_config = Qwen3TTSSpeakerEncoderConfig(**self.speaker_encoder_config)
        super().__post_init__(**kwargs)

    def get_text_config(self, *args, **kwargs):
        """Defaulting to the talker config: it is the decoder that generates codec tokens and owns the cache."""
        # `super().__init__` runs the config validators before `talker_config` is assigned, so this has to
        # stay callable on a partially constructed config.
        talker_config = getattr(self, "talker_config", None)
        if talker_config is None:
            return super().get_text_config(*args, **kwargs)
        return talker_config


class Qwen3TTSRMSNorm(Qwen3RMSNorm):
    pass


class Qwen3TTSMlp(Qwen3MLP):
    pass


class Qwen3TTSTalkerTextMLP(Qwen3OmniMoeTalkerTextMLP):
    pass


class Qwen3TTSRotaryEmbedding(Qwen3OmniMoeRotaryEmbedding):
    pass


class Qwen3TTSTalkerRotaryEmbedding(Qwen2VLRotaryEmbedding):
    pass


class Qwen3TTSTalkerAttention(Qwen3Attention):
    pass


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


@auto_docstring
@dataclass
class Qwen3TTSTalkerCodePredictorOutputWithPast(CausalLMOutputWithPast):
    r"""
    generation_steps (`int`, *optional*):
        Index of the next residual codebook to predict during generation.
    """

    generation_steps: int | None = None


@auto_docstring
@dataclass
class Qwen3TTSTalkerOutputWithPast(CausalLMOutputWithPast):
    r"""
    talker_loss (`torch.FloatTensor`, *optional*):
        Causal first-codebook loss, including codec EOS for supervised examples.
    code_predictor_loss (`torch.FloatTensor`, *optional*):
        Teacher-forced residual-codebook loss.
    past_hidden (`torch.FloatTensor`, *optional*):
        Last talker hidden state used to condition the next generation step.
    """

    talker_loss: torch.FloatTensor | None = None
    code_predictor_loss: torch.FloatTensor | None = None
    past_hidden: torch.FloatTensor | None = None


class Qwen3TTSBasePreTrainedModel(Qwen3PreTrainedModel):
    """Common base for all Qwen3TTS PreTrainedModel classes."""

    _no_split_modules = []
    # Qwen3TTS has separate Talker/CodePredictor attention classes, so the generic
    # output-recording hook from Qwen3 does not apply.
    _can_record_outputs = {}


@auto_docstring
class Qwen3TTSPreTrainedModel(Qwen3TTSBasePreTrainedModel):
    config_class = Qwen3TTSConfig
    _no_split_modules = ["Qwen3TTSTalkerDecoderLayer", "Qwen3TTSDecoderLayer"]
    _supports_cache_class = True
    _supports_static_cache = False


@auto_docstring
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
    # Record only talker outputs, not those of the nested code predictor.
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
        r"""
        embedding_dim (`int`):
            Dimension of each per-codebook input embedding in `codec_embedding`.
        """
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


@auto_docstring
class Qwen3TTSTalkerCodePredictorModelForConditionalGeneration(Qwen3TTSPreTrainedModel, GenerationMixin):
    """Wrapper for CodePredictorModel with generation capabilities."""

    config_class = Qwen3TTSTalkerCodePredictorConfig
    base_model_prefix = "talker.code_predictor"

    def __init__(self, config: Qwen3TTSTalkerCodePredictorConfig, talker_config: Qwen3TTSTalkerConfig):
        r"""
        talker_config (`Qwen3TTSTalkerConfig`):
            Configuration of the talker model whose hidden size is used by the code predictor projection.
        """
        super().__init__(config)
        self.model = Qwen3TTSTalkerCodePredictorModel(config, talker_config.hidden_size)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, (config.num_code_groups - 1) * config.vocab_size, bias=False)

        # CODEPATH: the 1.7B checkpoints carry this projection; on the 0.6B ones the two sizes already match
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

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        generation_steps: int | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        generation_steps (`int`, *optional*):
            Residual codebook index for cached decoding. When omitted, `inputs_embeds` contains the talker
            conditioning state followed by ground-truth codebook embeddings, and each position uses its matching head.
        labels (`torch.LongTensor` of shape `(batch_size, num_residual_codebooks)`, *optional*):
            Residual codebook targets aligned with the returned logits. Values of `-100` are ignored.
        """
        if inputs_embeds is None:
            if generation_steps is None:
                raise ValueError("`generation_steps` is required when providing code predictor `input_ids`.")
            inputs_embeds = self.model.get_input_embeddings()[generation_steps - 1](input_ids)

        inputs_embeds = self.small_to_mtp_projection(inputs_embeds)

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        if generation_steps is None:
            num_predictions = hidden_states.shape[1] - 1
            if not 1 <= num_predictions <= self.config.num_code_groups - 1:
                raise ValueError("Code predictor inputs must contain a conditioning state and at least one codebook.")
            logits = torch.stack(
                [
                    nn.functional.linear(
                        hidden_states[:, index + 1],
                        self.lm_head.weight[index * self.vocab_size : (index + 1) * self.vocab_size],
                    )
                    for index in range(num_predictions)
                ],
                dim=1,
            )
            next_generation_step = num_predictions
        else:
            logits = nn.functional.linear(
                hidden_states,
                self.lm_head.weight[generation_steps * self.vocab_size : (generation_steps + 1) * self.vocab_size],
            )
            next_generation_step = generation_steps + 1

        loss = None
        if labels is not None:
            if labels.shape != logits.shape[:-1]:
                raise ValueError("Code predictor labels must match the batch and codebook dimensions of the logits.")
            loss = self.loss_function(
                logits=logits,
                labels=None,
                shift_labels=labels.contiguous(),
                vocab_size=self.vocab_size,
                num_items_in_batch=(labels != -100).sum().clamp_min(1),
            )

        return Qwen3TTSTalkerCodePredictorOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            generation_steps=next_generation_step,
        )

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder=False, num_new_tokens=1):
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder, num_new_tokens
        )
        model_kwargs["generation_steps"] = outputs.generation_steps
        return model_kwargs


@auto_docstring
class Qwen3TTSForConditionalGeneration(Qwen3TTSPreTrainedModel, Qwen3TTSGenerationMixin):
    """Main Qwen3-TTS model for text-to-acoustic generation."""

    config_class = Qwen3TTSConfig
    main_input_name = "input_ids"
    accepts_loss_kwargs = False

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

        # CODEPATH: base checkpoints only; CustomVoice and VoiceDesign ship no speaker encoder
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
            # CODEPATH: base checkpoints only; the others never resample a reference clip
            config.speaker_encoder_config.sample_rate if config.speaker_encoder_config is not None else 24000
        )
        self.tokenizer_type = getattr(config, "tokenizer_type", "qwen2")
        self.tts_model_size = getattr(config, "tts_model_size", "base")

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

    def _prepare_teacher_forcing_inputs(
        self,
        input_ids,
        attention_mask,
        audio_codes,
        audio_attention_mask,
        speaker_embeddings,
    ):
        """Pack the Base/Auto non-streaming prompt and ground-truth audio without padding gaps."""
        config = self.config.talker_config
        if input_ids.ndim != 2 or input_ids.shape[1] < 8:
            raise ValueError("`input_ids` must include the three-token role prefix and five-token text suffix.")
        batch_size, text_length = input_ids.shape
        if (
            audio_codes.ndim != 3
            or audio_codes.shape[0] != batch_size
            or audio_codes.shape[2] != config.num_code_groups
        ):
            raise ValueError("`audio_codes` must have shape (batch_size, audio_length, num_code_groups).")
        if config.num_code_groups != self.code_predictor.config.num_code_groups:
            raise ValueError("Talker and code predictor must use the same number of codebooks.")
        if speaker_embeddings.shape != (batch_size, config.hidden_size):
            raise ValueError("`speaker_embeddings` must have shape (batch_size, talker_hidden_size).")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if audio_attention_mask is None:
            audio_attention_mask = torch.ones_like(audio_codes[..., 0], dtype=torch.bool)
        if attention_mask.shape != input_ids.shape or audio_attention_mask.shape != audio_codes.shape[:2]:
            raise ValueError("Text and audio attention masks must match their respective input sequence shapes.")
        attention_mask = attention_mask.bool()
        audio_attention_mask = audio_attention_mask.bool()
        text_lengths = attention_mask.sum(-1)
        audio_lengths = audio_attention_mask.sum(-1)
        torch_compilable_check(text_lengths >= 8, "Each text prompt must contain its role prefix and text suffix.")
        torch_compilable_check(audio_lengths > 0, "Each example must contain at least one audio frame.")

        # Ignore padded IDs before embedding and compact either left- or right-padded text.
        input_ids = input_ids.masked_fill(~attention_mask, 0)
        torch_compilable_check(
            (input_ids >= 0) & (input_ids < config.text_vocab_size), "Text IDs must be in the text vocabulary."
        )
        text_order = (~attention_mask).int().argsort(dim=1, stable=True)
        input_ids = input_ids.gather(1, text_order)
        text_embeds = self.text_projection(self.get_text_embeddings()(input_ids))
        text_mask = torch.arange(text_length - 8, device=input_ids.device)[None, :] < (text_lengths - 8)[:, None]
        text_embeds = text_embeds[:, 3:-5]

        special_text = self.text_projection(
            self.get_text_embeddings()(
                input_ids.new_tensor(
                    [[self.config.tts_pad_token_id, self.config.tts_bos_token_id, self.config.tts_eos_token_id]]
                )
            )
        )
        text_pad, text_bos, text_eos = special_text.chunk(3, dim=1)
        codec_special = self.get_input_embeddings()(
            input_ids.new_tensor(
                [
                    [
                        config.codec_nothink_id,
                        config.codec_think_bos_id,
                        config.codec_think_eos_id,
                        config.codec_pad_id,
                        config.codec_bos_id,
                        config.codec_eos_token_id,
                    ]
                ]
            )
        )
        codec_pad, codec_bos, codec_eos = codec_special[:, 3:].chunk(3, dim=1)
        codec_prefix = torch.cat(
            [
                codec_special[:, :3].expand(batch_size, -1, -1),
                speaker_embeddings[:, None].to(codec_special),
                codec_pad.expand(batch_size, -1, -1),
            ],
            dim=1,
        )
        prefix = codec_prefix + torch.cat([text_pad.expand(1, 4, -1), text_bos], dim=1)

        audio_codes = audio_codes.masked_fill(~audio_attention_mask[..., None], 0)
        torch_compilable_check(
            (audio_codes[..., 0] >= 0) & (audio_codes[..., 0] < config.vocab_size),
            "Primary audio codes must be in the talker vocabulary.",
        )
        torch_compilable_check(
            (audio_codes[..., 1:] >= 0) & (audio_codes[..., 1:] < self.code_predictor.vocab_size),
            "Residual audio codes must be in the code predictor vocabulary.",
        )
        audio_embeds = self.get_input_embeddings()(audio_codes[..., 0])
        for index, embedding in enumerate(self.code_predictor.get_input_embeddings()):
            audio_embeds = audio_embeds + embedding(audio_codes[..., index + 1])

        inputs_embeds = torch.cat(
            [
                self.text_projection(self.get_text_embeddings()(input_ids[:, :3])),
                prefix,
                text_embeds + codec_pad,
                (text_eos + codec_pad).expand(batch_size, -1, -1),
                (text_pad + codec_bos).expand(batch_size, -1, -1),
                audio_embeds + text_pad,
                (text_pad + codec_eos).expand(batch_size, -1, -1),
            ],
            dim=1,
        )
        packed_mask = torch.cat(
            [
                attention_mask.new_ones(batch_size, 8),
                text_mask,
                attention_mask.new_ones(batch_size, 2),
                audio_attention_mask,
                attention_mask.new_ones(batch_size, 1),
            ],
            dim=1,
        )
        order = (~packed_mask).int().argsort(dim=1, stable=True)
        inputs_embeds = inputs_embeds.gather(1, order[..., None].expand_as(inputs_embeds))
        packed_mask = packed_mask.gather(1, order)
        inputs_embeds = inputs_embeds.masked_fill(~packed_mask[..., None], 0)
        audio_positions = text_lengths[:, None] + 2 + audio_attention_mask.long().cumsum(-1) - 1
        eos_positions = text_lengths + 2 + audio_lengths
        return inputs_embeds, packed_mask, audio_positions, eos_positions

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        audio_codes: torch.LongTensor | None = None,
        audio_attention_mask: torch.Tensor | None = None,
        speaker_embeddings: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, text_length)`, *optional*):
            Formatted text prompts, including the three-token role prefix and five-token suffix, when `audio_codes`
            is supplied. During cached generation, contains the latest primary codec token instead.
        audio_codes (`torch.LongTensor` of shape `(batch_size, audio_length, num_code_groups)`, *optional*):
            Ground-truth codec frames for Base/Auto non-streaming teacher forcing.
        audio_attention_mask (`torch.Tensor` of shape `(batch_size, audio_length)`, *optional*):
            Mask with 1 for valid audio frames and 0 for padding.
        speaker_embeddings (`torch.FloatTensor` of shape `(batch_size, hidden_size)`, *optional*):
            Precomputed speaker conditioning required for teacher forcing.
        labels (`torch.LongTensor`, *optional*):
            Unshifted targets with `-100` for ignored values. With `audio_codes`, has shape
            `(batch_size, audio_length, num_code_groups)`; codec EOS is supervised only for examples with at least
            one supervised primary target, and primary logits follow the packed text/audio sequence. Otherwise,
            has shape `(batch_size, sequence_length)` for causal primary-code prediction.
        """
        teacher_forcing = audio_codes is not None
        if teacher_forcing:
            if input_ids is None or speaker_embeddings is None:
                raise ValueError("Teacher forcing requires `input_ids` and `speaker_embeddings`.")
            if inputs_embeds is not None or past_key_values is not None:
                raise ValueError("Teacher forcing cannot be combined with `inputs_embeds` or a generation cache.")
            inputs_embeds, attention_mask, audio_positions, eos_positions = self._prepare_teacher_forcing_inputs(
                input_ids, attention_mask, audio_codes, audio_attention_mask, speaker_embeddings
            )
            use_cache = False if use_cache is None else use_cache
        else:
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)

        if position_ids is None and isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids[:, -inputs_embeds.shape[1] :]

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.codec_head(hidden_states)

        loss = talker_loss = code_predictor_loss = None
        if teacher_forcing and labels is not None:
            if labels.shape != audio_codes.shape:
                raise ValueError("`labels` must match the shape of `audio_codes`.")
            if audio_attention_mask is None:
                audio_attention_mask = torch.ones_like(audio_codes[..., 0], dtype=torch.bool)
            labels = labels.masked_fill(~audio_attention_mask.bool()[..., None], -100)
            torch_compilable_check(
                (labels[..., 0] == -100) | ((labels[..., 0] >= 0) & (labels[..., 0] < self.vocab_size)),
                "Primary labels must be vocabulary IDs or -100.",
            )
            torch_compilable_check(
                (labels[..., 1:] == -100)
                | ((labels[..., 1:] >= 0) & (labels[..., 1:] < self.code_predictor.vocab_size)),
                "Residual labels must be vocabulary IDs or -100.",
            )
            primary_labels = labels.new_full(logits.shape[:2], -100)
            # Padded frames write only to the ignored role prefix, never over a valid target or EOS.
            frame_positions = audio_positions.masked_fill(~audio_attention_mask.bool(), 0)
            primary_labels.scatter_(1, frame_positions, labels[..., 0])
            eos_labels = labels.new_full((labels.shape[0], 1), self.config.talker_config.codec_eos_token_id)
            eos_labels = eos_labels.masked_fill(~(labels[..., 0] != -100).any(-1, keepdim=True), -100)
            primary_labels.scatter_(1, eos_positions[:, None], eos_labels)
            talker_loss = self.loss_function(
                logits=logits,
                labels=primary_labels,
                vocab_size=self.vocab_size,
                num_items_in_batch=(primary_labels != -100).sum().clamp_min(1),
            )

            safe_codes = audio_codes.masked_fill(~audio_attention_mask.bool()[..., None], 0).flatten(0, 1)
            previous_hidden = hidden_states.gather(
                1, (audio_positions - 1)[..., None].expand(-1, -1, hidden_states.shape[-1])
            ).flatten(0, 1)
            predictor_labels = labels[..., 1:].flatten(0, 1)
            # Keep fixed frame dimensions under compilation; eager training drops ignored frames.
            if not is_torchdynamo_compiling():
                train_mask = (predictor_labels != -100).any(-1)
                # Retain one ignored frame for an autograd-connected zero loss when all targets are ignored.
                train_mask[0] |= ~train_mask.any()
                safe_codes = safe_codes[train_mask]
                previous_hidden = previous_hidden[train_mask]
                predictor_labels = predictor_labels[train_mask]
            predictor_inputs = [previous_hidden, self.get_input_embeddings()(safe_codes[:, 0])]
            for index in range(self.config.talker_config.num_code_groups - 2):
                predictor_inputs.append(self.code_predictor.get_input_embeddings()[index](safe_codes[:, index + 1]))
            predictor_inputs = torch.stack(predictor_inputs, dim=1)
            predictor_output = self.code_predictor(
                inputs_embeds=predictor_inputs,
                labels=predictor_labels,
                use_cache=False,
            )
            code_predictor_loss = predictor_output.loss
            loss = talker_loss + self.config.code_predictor_loss_weight * code_predictor_loss
        elif labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.vocab_size,
                num_items_in_batch=(labels[..., 1:] != -100).sum().clamp_min(1),
            )

        return Qwen3TTSTalkerOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            talker_loss=talker_loss,
            code_predictor_loss=code_predictor_loss,
            past_hidden=None if teacher_forcing else hidden_states[:, -1:, :],
        )


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
