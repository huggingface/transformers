# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
from dataclasses import dataclass

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...generation import GenerationMixin
from ...masking_utils import create_bidirectional_mask, create_causal_mask, create_recurrent_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ...utils.type_validators import interval
from ..audioflamingo3.modeling_audioflamingo3 import (
    AudioFlamingo3CausalLMOutputWithPast,
    AudioFlamingo3ForConditionalGeneration,
    AudioFlamingo3Model,
    AudioFlamingo3ModelOutputWithPast,
    AudioFlamingo3PreTrainedModel,
)
from ..auto import AutoConfig, AutoModelForAudioFrameClassification
from ..glmasr.modeling_glmasr import apply_rotary_pos_emb, eager_attention_forward
from ..nemotron3_diarization.configuration_nemotron3_diarization import Nemotron3DiarizationAudioConfig
from ..nemotron3_diarization.modeling_nemotron3_diarization import (
    Nemotron3DiarizationAttention,
    Nemotron3DiarizationAudioLayer,
    Nemotron3DiarizationAudioModel,
    Nemotron3DiarizationFeatureStacking,
    Nemotron3DiarizationMLP,
    Nemotron3DiarizationRotaryEmbedding,
)
from .generation_nemotron3_5_transcribe import Nemotron3_5TranscribeGenerateOutput


@auto_docstring(checkpoint="nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B")
@strict
class Nemotron3_5TranscribeAsrAudioConfig(Nemotron3DiarizationAudioConfig):
    r"""
    subsampling_factor (`int`, *optional*, defaults to 8):
        Number of consecutive spectrogram frames stacked into one encoder frame.

    Example:

    ```python
    >>> from transformers import Nemotron3_5TranscribeAsrAudioModel, Nemotron3_5TranscribeAsrAudioConfig

    >>> # Initializing a Nemotron3_5TranscribeAsrAudioModel configuration
    >>> configuration = Nemotron3_5TranscribeAsrAudioConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = Nemotron3_5TranscribeAsrAudioModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "nemotron3_5_transcribe_asr_audio"
    base_config_key = "asr_config"

    hidden_size: int = 1280
    intermediate_size: int = 5120
    num_hidden_layers: int = 32
    num_attention_heads: int = 10

    def __post_init__(self, **kwargs):
        if self.rope_parameters is None:
            self.rope_parameters = {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5}
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B")
@strict
class Nemotron3_5TranscribeWordTimestampConfig(PreTrainedConfig):
    r"""
    vocab_size (`int`, *optional*, defaults to 23863):
        Number of classes of the word timestamp head: the pieces of its alignment tokenizer, then the blank.
    blank_token_id (`int`, *optional*, defaults to 23862):
        Index of the blank class, after the pieces of the alignment tokenizer.
    residual_scale (`float`, *optional*, defaults to 0.1):
        Scale of the transformer output added to the speech recognition encoder states before the classifier.
    upsampling_factor (`int`, *optional*, defaults to 2):
        Number of head frames per encoder frame, made by a transposed convolution before the transformer: 2 gives
        40 ms frames from the 80 ms encoder frames.

    Example:

    ```python
    >>> from transformers import Nemotron3_5TranscribeConfig, Nemotron3_5TranscribeWordTimestampConfig

    >>> # Initializing a Nemotron3_5TranscribeWordTimestamp configuration
    >>> configuration = Nemotron3_5TranscribeWordTimestampConfig()

    >>> # Using it as the word timestamp head of the model
    >>> model_configuration = Nemotron3_5TranscribeConfig(word_timestamp_config=configuration)
    ```
    """

    model_type = "nemotron3_5_transcribe_word_timestamp"
    base_config_key = "word_timestamp_config"

    hidden_size: int = 1280
    intermediate_size: int = 640
    num_hidden_layers: int = 2
    num_attention_heads: int = 16
    num_key_value_heads: int | None = None
    hidden_act: str = "gelu"
    max_position_embeddings: int = 5000
    initializer_range: float = 0.02
    rope_parameters: dict | None = None
    attention_dropout: float | int = 0.0
    vocab_size: int = 23863
    blank_token_id: int = 23862
    residual_scale: float = interval(min=0.0)(default=0.1)
    upsampling_factor: int = interval(min=1)(default=2)

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.rope_parameters is None:
            self.rope_parameters = {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5}
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"`hidden_size` ({self.hidden_size}) must be divisible by `num_attention_heads` "
                f"({self.num_attention_heads})."
            )
        if not 0 <= self.blank_token_id < self.vocab_size:
            raise ValueError(
                f"`blank_token_id` ({self.blank_token_id}) must be a class below `vocab_size` ({self.vocab_size})."
            )


@auto_docstring(checkpoint="nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B")
@strict
class Nemotron3_5TranscribeAudioConfig(PreTrainedConfig):
    r"""
    asr_config (`Union[dict, Nemotron3_5TranscribeAsrAudioConfig]`, *optional*):
        Configuration of the speech recognition encoder.
    diarization_config (`Union[dict, Nemotron3DiarizationConfig]`, *optional*):
        Configuration of the speaker diarization model, whose speaker activities are added to the speech recognition
        encoder outputs.
    speaker_activity_threshold (`float`, *optional*, defaults to 0.5):
        A speaker is active in an encoder frame when its probability is above this threshold.

    Example:

    ```python
    >>> from transformers import Nemotron3_5TranscribeAudioModel, Nemotron3_5TranscribeAudioConfig

    >>> # Initializing a Nemotron3_5TranscribeAudioModel configuration
    >>> configuration = Nemotron3_5TranscribeAudioConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = Nemotron3_5TranscribeAudioModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "nemotron3_5_transcribe_audio"
    base_config_key = "audio_config"
    sub_configs_defaults = {
        "asr_config": SubConfigSpec(config_class=AutoConfig, model_type="nemotron3_5_transcribe_asr_audio"),
        "diarization_config": SubConfigSpec(
            config_class=AutoConfig,
            model_type="nemotron3_diarization",
            init_kwargs={
                "chunk_length": 500,
                "chunk_left_context": 50,
                "chunk_right_context": 50,
                "fifo_length": 40,
                "speaker_cache_update_period": 300,
                "streaming_config": {"speaker_cache_length": 188},
            },
        ),
    }

    asr_config: dict | PreTrainedConfig | None = None
    diarization_config: dict | PreTrainedConfig | None = None
    speaker_activity_threshold: float = interval(min=0.0, max=1.0)(default=0.5)

    @property
    def hidden_size(self) -> int:
        return self.asr_config.hidden_size

    @property
    def num_speakers(self) -> int:
        return self.diarization_config.head_config.num_speakers

    @property
    def subsampling_factor(self) -> int:
        return self.asr_config.subsampling_factor

    def validate_architecture(self):
        if self.asr_config.subsampling_factor != self.diarization_config.audio_config.subsampling_factor:
            raise ValueError(
                f"The speech recognition encoder (`subsampling_factor={self.asr_config.subsampling_factor}`) and the "
                "diarization model (`subsampling_factor="
                f"{self.diarization_config.audio_config.subsampling_factor}`) must have the same frame rate."
            )


@auto_docstring(checkpoint="nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B")
@strict
class Nemotron3_5TranscribeConfig(PreTrainedConfig):
    r"""
    audio_token_id (`int`, *optional*, defaults to 131072):
        The audio token index used as a placeholder for the audio embeddings. The original checkpoint appends it after
        the language model's vocabulary, with no embedding row: it is embedded as token 0, then replaced by the audio.
    word_timestamp_config (`Union[dict, Nemotron3_5TranscribeWordTimestampConfig]`, *optional*):
        Configuration of the word timestamp encoder and head, which predict alignment tokenizer pieces from the speech
        recognition encoder states.

    Example:

    ```python
    >>> from transformers import Nemotron3_5TranscribeForConditionalGeneration, Nemotron3_5TranscribeConfig

    >>> # Initializing a Nemotron3_5Transcribe configuration
    >>> configuration = Nemotron3_5TranscribeConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = Nemotron3_5TranscribeForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "nemo_speechlm"
    sub_configs_defaults = {
        "audio_config": SubConfigSpec(config_class=AutoConfig, model_type="nemotron3_5_transcribe_audio"),
        "text_config": SubConfigSpec(
            config_class=AutoConfig,
            model_type="nemotron_h",
            init_kwargs={
                "hidden_size": 2688,
                "intermediate_size": 1856,
                "hybrid_override_pattern": "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME",
                "mamba_num_heads": 64,
                "max_position_embeddings": 262144,
                "moe_intermediate_size": 1856,
                "moe_shared_expert_intermediate_size": 3712,
                "n_routed_experts": 128,
                "num_experts_per_tok": 6,
                "num_key_value_heads": 2,
                "routed_scaling_factor": 2.5,
            },
        ),
        "word_timestamp_config": SubConfigSpec(
            config_class=AutoConfig, model_type="nemotron3_5_transcribe_word_timestamp"
        ),
    }

    audio_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    audio_token_id: int = 131072
    word_timestamp_config: dict | PreTrainedConfig | None = None


@auto_docstring
@dataclass
class Nemotron3_5TranscribeModelOutputWithPast(AudioFlamingo3ModelOutputWithPast):
    pass


@auto_docstring(
    custom_intro="""
    Base class for Nemotron3_5Transcribe causal language model (or autoregressive) outputs.
    """
)
@dataclass
class Nemotron3_5TranscribeCausalLMOutputWithPast(AudioFlamingo3CausalLMOutputWithPast):
    pass


@auto_docstring
@dataclass
class Nemotron3_5TranscribeAudioModelOutput(BaseModelOutputWithPooling):
    r"""
    pooler_output (`torch.FloatTensor` of shape `(num_audio_tokens, text_config.hidden_size)`, *optional*):
        Projected audio embeddings of the valid encoder frames of every recording, in batch order: the embeddings that
        replace the audio tokens of the prompt. Returned by `get_audio_features`.
    asr_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_frames, audio_config.hidden_size)`):
        Output of the speech recognition encoder, before its normalization and the speaker embeddings: the input of
        [`~Nemotron3_5TranscribeForConditionalGeneration.get_word_timestamp_logits`].
    speaker_logits (`torch.FloatTensor` of shape `(batch_size, feature_sequence_length, audio_config.num_speakers)`):
        Logits of the diarization model, one frame per spectrogram frame.
    """

    asr_last_hidden_state: torch.FloatTensor | None = None
    speaker_logits: torch.FloatTensor | None = None


class Nemotron3_5TranscribePreTrainedModel(AudioFlamingo3PreTrainedModel):
    _no_split_modules = None
    _is_stateful = True
    _keys_to_ignore_on_load_unexpected = [
        r"^llm\.mtp\.",
        r"^perception\.preprocessor\.",
        r"^perception\.encoder\.diarization_model\.preprocessor\.",
        r"^perception\.encoder\.diarization_model\.sortformer_modules\.hidden_to_spks\.",
        r"\._extra_state$",
        r"^ctc_timestamp\.decoder\.residual_scale$",
        r"^ctc_timestamp\.decoder\.upsample_factor$",
        r"^ctc_timestamp\.tokenizer\.model_proto$",
    ]

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, Nemotron3_5TranscribeSpeakerEmbedding):
            init.copy_(module.weight, module.compute_embeddings(module.config))


class Nemotron3_5TranscribeFeatureStacking(Nemotron3DiarizationFeatureStacking):
    def __init__(self, config: Nemotron3_5TranscribeAsrAudioConfig):
        super().__init__(config)


class Nemotron3_5TranscribeRotaryEmbedding(Nemotron3DiarizationRotaryEmbedding):
    def __init__(self, config: Nemotron3_5TranscribeAsrAudioConfig):
        super().__init__(config)


class Nemotron3_5TranscribeAttention(Nemotron3DiarizationAttention):
    def __init__(self, config: Nemotron3_5TranscribeAsrAudioConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

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
            attention_mask=attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Nemotron3_5TranscribeMLP(Nemotron3DiarizationMLP): ...


class Nemotron3_5TranscribeAsrAudioLayer(Nemotron3DiarizationAudioLayer):
    def __init__(self, config: Nemotron3_5TranscribeAsrAudioConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = Nemotron3_5TranscribeAttention(config, layer_idx)


@auto_docstring(
    custom_intro="""
    The Nemotron 3.5 Transcribe speech recognition encoder: the Nemotron 3 Diarization audio encoder with the queries
    and keys of each attention head normalized, and rotary position embeddings on part of each head.
    """
)
class Nemotron3_5TranscribeAsrAudioModel(Nemotron3DiarizationAudioModel):
    config: Nemotron3_5TranscribeAsrAudioConfig
    main_input_name = "input_features"
    input_modalities = "audio"
    _no_split_modules = ["Nemotron3_5TranscribeAsrAudioLayer"]
    _can_record_outputs = {
        "hidden_states": Nemotron3_5TranscribeAsrAudioLayer,
        "attentions": Nemotron3_5TranscribeAttention,
    }

    def __init__(self, config: Nemotron3_5TranscribeAsrAudioConfig):
        super().__init__(config)
        self.embedder = Nemotron3_5TranscribeFeatureStacking(config)
        self.layers = nn.ModuleList(
            [Nemotron3_5TranscribeAsrAudioLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.rotary_emb = Nemotron3_5TranscribeRotaryEmbedding(config)


@auto_docstring(
    custom_intro="""
    The transformer of the Nemotron 3.5 Transcribe word timestamp head.
    [`Nemotron3_5TranscribeForConditionalGeneration.word_timestamp_head`] classifies its output.
    """
)
class Nemotron3_5TranscribeWordTimestampModel(Nemotron3_5TranscribePreTrainedModel):
    config: Nemotron3_5TranscribeWordTimestampConfig
    main_input_name = "inputs_embeds"
    input_modalities = "audio"
    _no_split_modules = ["Nemotron3_5TranscribeAsrAudioLayer"]
    _can_record_outputs = {
        "hidden_states": Nemotron3_5TranscribeAsrAudioLayer,
        "attentions": Nemotron3_5TranscribeAttention,
    }

    def __init__(self, config: Nemotron3_5TranscribeWordTimestampConfig):
        super().__init__(config)
        self.upsampler = nn.ConvTranspose1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.upsampling_factor,
            stride=config.upsampling_factor,
        )
        self.input_layer_norm = nn.LayerNorm(config.hidden_size)
        self.layers = nn.ModuleList(
            [Nemotron3_5TranscribeAsrAudioLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.rotary_emb = Nemotron3_5TranscribeRotaryEmbedding(config)
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        r"""
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_frames, hidden_size)`):
            Speech recognition encoder states, before the audio model's `asr_norm` and speaker embeddings.
        attention_mask (`torch.Tensor` of shape `(batch_size, num_frames)`, *optional*):
            Mask of the valid encoder frames.

        Returns `num_frames * upsampling_factor` frames.
        """
        inputs_embeds = self.upsampler(inputs_embeds.transpose(1, 2)).transpose(1, 2)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(self.config.upsampling_factor, dim=1)
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)[None, :]
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        attention_mask = create_bidirectional_mask(
            config=self.config, inputs_embeds=inputs_embeds, attention_mask=attention_mask
        )
        hidden_states = self.input_layer_norm(inputs_embeds)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, attention_mask=attention_mask, position_embeddings=position_embeddings, **kwargs
            )
        # the original casts its residual scale to the dtype of the states before the product
        residual_scale = inputs_embeds.new_tensor(self.config.residual_scale)
        return BaseModelOutput(last_hidden_state=inputs_embeds + residual_scale * self.layer_norm(hidden_states))


def normalize_per_feature(input_features: torch.Tensor, input_features_mask: torch.Tensor) -> torch.Tensor:
    """Normalizes each mel bin to zero mean and unit variance over the valid frames of each sample."""
    mask = input_features_mask.bool()[..., None]
    input_features = input_features.float().masked_fill(~mask, 0.0)
    num_frames = mask.sum(1)
    mean = input_features.sum(1, keepdim=True) / num_frames[:, None].clamp(min=1)
    centered = (input_features - mean).masked_fill(~mask, 0.0)
    # the original uses the unbiased variance (0 for a single frame) and adds its epsilon to the standard deviation
    std = (centered.pow(2).sum(1, keepdim=True) / (num_frames[:, None] - 1).clamp(min=1)).sqrt() + 1e-5
    return centered / std


class Nemotron3_5TranscribeSpeakerEmbedding(nn.Module):
    """Fixed sinusoidal embedding of each speaker slot, which the speaker activities select."""

    def __init__(self, config: Nemotron3_5TranscribeAudioConfig):
        super().__init__()
        self.config = config
        self.weight = nn.Buffer(self.compute_embeddings(config), persistent=False)

    @staticmethod
    def compute_embeddings(config: Nemotron3_5TranscribeAudioConfig) -> torch.Tensor:
        position = torch.arange(config.num_speakers, dtype=torch.float32)[:, None]
        inv_freq = torch.exp(
            torch.arange(0, config.hidden_size, 2, dtype=torch.float32) * -(math.log(10000.0) / config.hidden_size)
        )
        # float32 whatever the default dtype of the loading context: the buffer is a fixed table, not a weight
        embeddings = torch.zeros(config.num_speakers, config.hidden_size, dtype=torch.float32)
        embeddings[:, 0::2] = torch.sin(position * inv_freq)
        embeddings[:, 1::2] = torch.cos(position * inv_freq)
        return embeddings

    def forward(self, speaker_activity: torch.Tensor) -> torch.Tensor:
        return speaker_activity @ self.weight.to(speaker_activity.dtype)


@auto_docstring(
    custom_intro="""
    The Nemotron 3.5 Transcribe audio model: a speech recognition encoder and a speaker diarization model run in
    parallel on the same spectrogram, and the speaker activities are added to the speech recognition encoder outputs.
    """
)
class Nemotron3_5TranscribeAudioModel(Nemotron3_5TranscribePreTrainedModel):
    config: Nemotron3_5TranscribeAudioConfig
    main_input_name = "input_features"
    input_modalities = "audio"

    def __init__(self, config: Nemotron3_5TranscribeAudioConfig):
        super().__init__(config)
        self.asr_encoder = Nemotron3_5TranscribeAsrAudioModel(config.asr_config)
        self.diarization_model = AutoModelForAudioFrameClassification.from_config(config.diarization_config)
        self.asr_norm = nn.LayerNorm(config.hidden_size)
        self.speaker_norm = nn.LayerNorm(config.num_speakers)
        self.speaker_embedding = Nemotron3_5TranscribeSpeakerEmbedding(config)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_features: torch.FloatTensor,
        input_features_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Nemotron3_5TranscribeAudioModelOutput:
        r"""
        input_features (`torch.FloatTensor` of shape `(batch_size, feature_sequence_length, num_mel_bins)`):
            Log mel spectrogram, padded to a multiple of `config.subsampling_factor` frames. As in the original, audio
            longer than `config.diarization_config.chunk_length` encoder frames is encoded in chunks with context.
        input_features_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`, *optional*):
            Mask of the valid spectrogram frames. Padding must be on the right.
        """
        subsampling_factor = self.config.subsampling_factor
        if input_features.shape[1] % subsampling_factor != 0:
            raise ValueError(
                f"The number of spectrogram frames ({input_features.shape[1]}) must be a multiple of "
                f"`subsampling_factor` ({subsampling_factor}): pad `input_features` and `input_features_mask` on the "
                "right, as the processor does."
            )
        if input_features_mask is None:
            input_features_mask = torch.ones(input_features.shape[:2], dtype=torch.long, device=input_features.device)

        num_frames = input_features.shape[1] // subsampling_factor
        asr_input = normalize_per_feature(input_features, input_features_mask).to(self.asr_encoder.dtype)
        hidden_states = asr_input.new_empty(asr_input.shape[0], num_frames, self.config.asr_config.hidden_size)
        for context, chunk in self._get_chunks(num_frames):
            spectrogram_context = slice(context.start * subsampling_factor, context.stop * subsampling_factor)
            chunk_states = self.asr_encoder(
                asr_input[:, spectrogram_context], input_features_mask[:, spectrogram_context]
            )
            hidden_states[:, context.start + chunk.start : context.start + chunk.stop] = (
                chunk_states.last_hidden_state[:, chunk]
            )

        speaker_logits = self.diarization_model(
            input_features.to(self.diarization_model.dtype), attention_mask=input_features_mask
        ).logits
        speaker_probabilities = speaker_logits.sigmoid().unflatten(1, (num_frames, -1)).mean(2)
        speaker_probs = speaker_probabilities.to(hidden_states.dtype)
        speaker_activity = (speaker_probs > self.config.speaker_activity_threshold).to(hidden_states.dtype)
        speaker_states = self.speaker_embedding(self.speaker_norm(speaker_activity))
        last_hidden_state = self.asr_norm(hidden_states) + speaker_states

        return Nemotron3_5TranscribeAudioModelOutput(
            last_hidden_state=last_hidden_state,
            asr_last_hidden_state=hidden_states,
            speaker_logits=speaker_logits,
        )

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        return input_lengths, (input_lengths + self.config.subsampling_factor - 1) // self.config.subsampling_factor

    def _get_chunks(self, num_frames: int) -> list[tuple[slice, slice]]:
        """
        The chunks of the original's online inference, which the speech recognition encoder and the word timestamp
        encoder run on (NeMo `ParallelExpertEncoder._forward_online`, `TimestampAligner._iter_ctc_windows`):
        `chunk_length` encoder frames each, seeing `chunk_left_context` and `chunk_right_context` more frames, as the
        diarization model's chunks. Per chunk, `context`, the encoder frames of the sequence the encoder runs on (the
        chunk with its context), and `chunk`, the chunk's frames in the output of that run.
        """
        config = self.config.diarization_config
        chunks = []
        for start in range(0, num_frames, config.chunk_length):
            end = min(start + config.chunk_length, num_frames)
            context = slice(
                max(start - config.chunk_left_context, 0), min(end + config.chunk_right_context, num_frames)
            )
            chunks.append((context, slice(start - context.start, end - context.start)))
        return chunks


@auto_docstring(
    custom_intro="""
    The Nemotron 3.5 Transcribe model (speech recognition encoder and speaker diarization model in parallel, linear
    projector and Nemotron-H language model), without a language modeling head.
    """
)
class Nemotron3_5TranscribeModel(AudioFlamingo3Model):
    _keep_in_fp32_modules_strict = ["audio_tower", "multi_modal_projector"]

    def __init__(self, config: Nemotron3_5TranscribeConfig):
        super().__init__(config)
        self.multi_modal_projector = nn.Linear(config.audio_config.hidden_size, config.text_config.hidden_size)
        self.word_timestamp_encoder = Nemotron3_5TranscribeWordTimestampModel(config.word_timestamp_config)

    def _embed_input_ids(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        # the audio token can be past the language model's vocabulary (NVIDIA's tokenizer appends it): it is looked up
        # as 0, its positions take the audio embeddings
        if self.config.audio_token_id >= self.config.text_config.vocab_size:
            input_ids = input_ids.masked_fill(input_ids == self.config.audio_token_id, 0)
        return self.get_input_embeddings()(input_ids)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Nemotron3_5TranscribeModelOutputWithPast:
        r"""
        input_features_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`):
            Mask to avoid performing attention on padding feature indices.
        """
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self._embed_input_ids(input_ids)
        return super().forward(
            input_ids=input_ids,
            input_features=input_features,
            input_features_mask=input_features_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

    @auto_docstring(
        custom_intro="""
        Runs the word timestamp encoder on the speech recognition encoder states, in the chunks of the encoder: the
        states the word timestamp head classifies.
        """
    )
    def get_word_timestamp_hidden_states(
        self, asr_last_hidden_state: torch.FloatTensor, input_features_mask: torch.Tensor
    ) -> torch.FloatTensor:
        r"""
        asr_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_frames, audio_config.hidden_size)`):
            Output of the speech recognition encoder before its normalization and the speaker embeddings, as returned
            by [`~Nemotron3_5TranscribeModel.get_audio_features`] for `input_features_mask`.
        input_features_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`):
            Mask of the valid spectrogram frames. Padding must be on the right.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_frames * word_timestamp_config.upsampling_factor,
            audio_config.hidden_size)`: `upsampling_factor` frames per encoder frame, in the dtype of the word timestamp
            encoder.
        """
        states = asr_last_hidden_state.to(self.word_timestamp_encoder.dtype)
        frame_mask = input_features_mask[:, :: self.config.audio_config.subsampling_factor].bool()
        factor = self.config.word_timestamp_config.upsampling_factor
        hidden_states = states.new_empty(
            states.shape[0], states.shape[1] * factor, self.config.word_timestamp_config.hidden_size
        )
        for context, chunk in self.audio_tower._get_chunks(states.shape[1]):
            chunk_states = self.word_timestamp_encoder(states[:, context], frame_mask[:, context])
            hidden_states[:, (context.start + chunk.start) * factor : (context.start + chunk.stop) * factor] = (
                chunk_states.last_hidden_state[:, chunk.start * factor : chunk.stop * factor]
            )
        return hidden_states


@auto_docstring(
    custom_intro="""
    The Nemotron 3.5 Transcribe model (speech recognition encoder and speaker diarization model in parallel, linear
    projector and Nemotron-H language model) with a language modeling head.
    """
)
class Nemotron3_5TranscribeForConditionalGeneration(AudioFlamingo3ForConditionalGeneration):
    def __init__(self, config: Nemotron3_5TranscribeConfig):
        super().__init__(config)
        self.word_timestamp_head = nn.Conv1d(
            config.word_timestamp_config.hidden_size, config.word_timestamp_config.vocab_size, kernel_size=1
        )
        self.post_init()

    @auto_docstring
    def get_audio_features(self, input_features, input_features_mask, **kwargs):
        r"""
        input_features_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`):
            Mask of the valid spectrogram frames. Padding must be on the right.
        """
        return self.model.get_audio_features(input_features, input_features_mask, **kwargs)

    @auto_docstring(
        custom_intro="""
        Logits of the word timestamp head on the speech recognition encoder states, which
        [`~Nemotron3_5TranscribeProcessor.align_word_timestamps`] turns into word times.
        """
    )
    def get_word_timestamp_logits(
        self, asr_last_hidden_state: torch.FloatTensor, input_features_mask: torch.Tensor
    ) -> torch.FloatTensor:
        r"""
        asr_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_frames, audio_config.hidden_size)`):
            Output of the speech recognition encoder before its normalization and the speaker embeddings, as returned
            by [`~Nemotron3_5TranscribeForConditionalGeneration.get_audio_features`] for `input_features_mask`.
        input_features_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`):
            Mask of the valid spectrogram frames. Padding must be on the right.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_frames * word_timestamp_config.upsampling_factor,
            word_timestamp_config.vocab_size)`: raw logits over the pieces of the alignment tokenizer and the blank,
            `upsampling_factor` frames per encoder frame (the processor applies the log-softmax).
        """
        hidden_states = self.model.get_word_timestamp_hidden_states(asr_last_hidden_state, input_features_mask)
        return self.word_timestamp_head(hidden_states.transpose(1, 2)).transpose(1, 2)

    @torch.no_grad()
    def generate(self, input_ids=None, input_features=None, input_features_mask=None, **kwargs):
        r"""
        return_word_timestamp_logits (`bool`, *optional*, defaults to `False`):
            Whether to also return the logits of the word timestamp head and of the diarization model, for
            [`~Nemotron3_5TranscribeProcessor.align_word_timestamps`], in a [`Nemotron3_5TranscribeGenerateOutput`]. The
            audio model runs once. Greedy or sampled decoding with the cache, one sequence per recording.
        """
        if not kwargs.pop("return_word_timestamp_logits", False):
            return GenerationMixin.generate(
                self,
                input_ids=input_ids,
                input_features=input_features,
                input_features_mask=input_features_mask,
                **kwargs,
            )
        generation_config = kwargs.get("generation_config", self.generation_config)
        if kwargs.get("use_cache") is False or kwargs.get(
            "num_return_sequences", generation_config.num_return_sequences
        ) not in (None, 1):
            raise ValueError("`return_word_timestamp_logits=True` needs the cache and one sequence per recording.")

        # the audio model runs once: its embeddings fill the prompt, its states feed the word timestamp head
        audio_outputs = self.get_audio_features(input_features, input_features_mask)
        inputs_embeds = self.model._embed_input_ids(input_ids)
        audio_embeds = audio_outputs.pooler_output.to(inputs_embeds.device, inputs_embeds.dtype)
        special_audio_mask = self.model.get_placeholder_mask(input_ids, inputs_embeds, audio_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(special_audio_mask, audio_embeds)

        outputs = GenerationMixin.generate(
            self, input_ids=input_ids, inputs_embeds=inputs_embeds, return_dict_in_generate=True, **kwargs
        )
        return Nemotron3_5TranscribeGenerateOutput(
            **outputs,
            word_timestamp_logits=self.get_word_timestamp_logits(
                audio_outputs.asr_last_hidden_state, input_features_mask
            ),
            speaker_logits=audio_outputs.speaker_logits,
        )

    @staticmethod
    def create_masks_for_generate(config, inputs_embeds, attention_mask, past_key_values, position_ids=None, **_):
        # TODO: @eustlb, remove once `masking_utils.create_masks_for_generate` skips the layers without attention
        # (Nemotron-H `moe` / `mlp`), which also removes `NemotronHForCausalLM.create_masks_for_generate`
        # as `NemotronHForCausalLM`: the default dispatch does not know the Nemotron-H `moe` layer type, so both masks
        # that the language model needs are returned
        mask_kwargs = {
            "config": config.get_text_config(),
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        return {
            "full_attention": create_causal_mask(**mask_kwargs),
            "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
        }


__all__ = [
    "Nemotron3_5TranscribeAsrAudioModel",
    "Nemotron3_5TranscribeAsrAudioConfig",
    "Nemotron3_5TranscribeAudioConfig",
    "Nemotron3_5TranscribeAudioModel",
    "Nemotron3_5TranscribeConfig",
    "Nemotron3_5TranscribeForConditionalGeneration",
    "Nemotron3_5TranscribeModel",
    "Nemotron3_5TranscribePreTrainedModel",
    "Nemotron3_5TranscribeWordTimestampConfig",
    "Nemotron3_5TranscribeWordTimestampModel",
]
