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
from ...configuration_utils import PreTrainedConfig
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import ProcessingKwargs, Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ..audioflamingo3.modeling_audioflamingo3 import (
    AudioFlamingo3ForConditionalGeneration,
    AudioFlamingo3Model,
    AudioFlamingo3ModelOutputWithPast,
    AudioFlamingo3PreTrainedModel,
)
from ..audioflamingo3.processing_audioflamingo3 import AudioFlamingo3Processor
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel, AutoModelForAudioFrameClassification
from ..glmasr.modeling_glmasr import GlmAsrRotaryEmbedding, apply_rotary_pos_emb, eager_attention_forward
from ..nemotron3_diarization.configuration_nemotron3_diarization import Nemotron3DiarizationAudioConfig
from ..nemotron3_diarization.modeling_nemotron3_diarization import (
    Nemotron3DiarizationAttention,
    Nemotron3DiarizationAudioLayer,
    Nemotron3DiarizationAudioModel,
    Nemotron3DiarizationFeatureStacking,
    Nemotron3DiarizationMLP,
)


@auto_docstring(checkpoint="nvidia/Nemotron-3.5-Transcribe")
@strict
class Nemotron3_5TranscribeAsrEncoderConfig(Nemotron3DiarizationAudioConfig):
    r"""
    subsampling_factor (`int`, *optional*, defaults to 8):
        Number of consecutive spectrogram frames stacked into one encoder frame.

    Example:

    ```python
    >>> from transformers import Nemotron3_5TranscribeAsrEncoder, Nemotron3_5TranscribeAsrEncoderConfig

    >>> # Initializing a Nemotron3_5TranscribeAsrEncoder configuration
    >>> configuration = Nemotron3_5TranscribeAsrEncoderConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = Nemotron3_5TranscribeAsrEncoder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "nemotron3_5_transcribe_asr_encoder"
    base_config_key = "asr_config"

    hidden_size: int = 1280
    intermediate_size: int = 5120
    num_hidden_layers: int = 32
    num_attention_heads: int = 10

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        kwargs.setdefault("partial_rotary_factor", 0.5)
        PreTrainedConfig.__post_init__(self, **kwargs)


@auto_docstring(checkpoint="nvidia/Nemotron-3.5-Transcribe")
@strict
class Nemotron3_5TranscribeEncoderConfig(PreTrainedConfig):
    r"""
    asr_config (`Union[dict, Nemotron3_5TranscribeAsrEncoderConfig]`, *optional*):
        Configuration of the speech recognition encoder.
    diarization_config (`Union[dict, Nemotron3DiarizationConfig]`, *optional*):
        Configuration of the speaker diarization model, whose speaker activities are added to the speech recognition
        encoder outputs.
    speaker_activity_threshold (`float`, *optional*, defaults to 0.5):
        A speaker is active in an encoder frame when its probability is above this threshold.

    Example:

    ```python
    >>> from transformers import Nemotron3_5TranscribeEncoder, Nemotron3_5TranscribeEncoderConfig

    >>> # Initializing a Nemotron3_5TranscribeEncoder configuration
    >>> configuration = Nemotron3_5TranscribeEncoderConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = Nemotron3_5TranscribeEncoder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "nemotron3_5_transcribe_encoder"
    base_config_key = "audio_config"
    sub_configs = {"asr_config": AutoConfig, "diarization_config": AutoConfig}
    # the original runs the diarization model on 40 s chunks with 4 s of left and right context, and a smaller speaker
    # cache than Nemotron 3 Diarization
    _default_diarization_config_kwargs = {
        "chunk_length": 500,
        "chunk_left_context": 50,
        "chunk_right_context": 50,
        "fifo_length": 40,
        "speaker_cache_update_period": 300,
        "streaming_config": {"speaker_cache_length": 188},
    }

    asr_config: dict | PreTrainedConfig | None = None
    diarization_config: dict | PreTrainedConfig | None = None
    speaker_activity_threshold: float = 0.5

    def __post_init__(self, **kwargs):
        if isinstance(self.asr_config, dict):
            self.asr_config["model_type"] = self.asr_config.get("model_type", "nemotron3_5_transcribe_asr_encoder")
            self.asr_config = CONFIG_MAPPING[self.asr_config["model_type"]](**self.asr_config)
        elif self.asr_config is None:
            self.asr_config = CONFIG_MAPPING["nemotron3_5_transcribe_asr_encoder"]()

        if isinstance(self.diarization_config, dict):
            self.diarization_config["model_type"] = self.diarization_config.get("model_type", "nemotron3_diarization")
            self.diarization_config = CONFIG_MAPPING[self.diarization_config["model_type"]](
                **{**self._default_diarization_config_kwargs, **self.diarization_config}
            )
        elif self.diarization_config is None:
            self.diarization_config = CONFIG_MAPPING["nemotron3_diarization"](
                **self._default_diarization_config_kwargs
            )

        super().__post_init__(**kwargs)

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


@auto_docstring(checkpoint="nvidia/Nemotron-3.5-Transcribe")
@strict
class Nemotron3_5TranscribeConfig(PreTrainedConfig):
    r"""
    audio_token_id (`int`, *optional*, defaults to 131072):
        The audio token index used as a placeholder for the audio embeddings. It is outside of the text vocabulary.

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

    model_type = "nemotron3_5_transcribe"
    sub_configs = {"audio_config": AutoConfig, "text_config": AutoConfig}
    _default_text_config_kwargs = {
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
    }

    audio_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    audio_token_id: int = 131072

    def __post_init__(self, **kwargs):
        if isinstance(self.audio_config, dict):
            self.audio_config["model_type"] = self.audio_config.get("model_type", "nemotron3_5_transcribe_encoder")
            self.audio_config = CONFIG_MAPPING[self.audio_config["model_type"]](**self.audio_config)
        elif self.audio_config is None:
            self.audio_config = CONFIG_MAPPING["nemotron3_5_transcribe_encoder"]()

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "nemotron_h")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](
                **{**self._default_text_config_kwargs, **self.text_config}
            )
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["nemotron_h"](**self._default_text_config_kwargs)

        super().__post_init__(**kwargs)


@auto_docstring
@dataclass
class Nemotron3_5TranscribeModelOutputWithPast(AudioFlamingo3ModelOutputWithPast):
    pass


class Nemotron3_5TranscribePreTrainedModel(AudioFlamingo3PreTrainedModel):
    _no_split_modules = None
    _is_stateful = True

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, Nemotron3_5TranscribeEncoder):
            init.copy_(module.speaker_embeddings, module.compute_speaker_embeddings(module.config))


class Nemotron3_5TranscribeFeatureStacking(Nemotron3DiarizationFeatureStacking):
    def __init__(self, config: Nemotron3_5TranscribeAsrEncoderConfig):
        super().__init__(config)


class Nemotron3_5TranscribeRotaryEmbedding(GlmAsrRotaryEmbedding): ...


class Nemotron3_5TranscribeAttention(Nemotron3DiarizationAttention):
    def __init__(self, config: Nemotron3_5TranscribeAsrEncoderConfig, layer_idx: int):
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


class Nemotron3_5TranscribeAsrEncoderLayer(Nemotron3DiarizationAudioLayer):
    def __init__(self, config: Nemotron3_5TranscribeAsrEncoderConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = Nemotron3_5TranscribeAttention(config, layer_idx)


@auto_docstring(
    custom_intro="""
    The Nemotron 3.5 Transcribe speech recognition encoder: the Nemotron 3 Diarization audio encoder with the queries
    and keys of each attention head normalized, and rotary position embeddings on part of each head.
    """
)
class Nemotron3_5TranscribeAsrEncoder(Nemotron3DiarizationAudioModel):
    config: Nemotron3_5TranscribeAsrEncoderConfig
    main_input_name = "input_features"
    input_modalities = "audio"
    _no_split_modules = ["Nemotron3_5TranscribeAsrEncoderLayer"]
    _can_record_outputs = {
        "hidden_states": Nemotron3_5TranscribeAsrEncoderLayer,
        "attentions": Nemotron3_5TranscribeAttention,
    }

    def __init__(self, config: Nemotron3_5TranscribeAsrEncoderConfig):
        super().__init__(config)
        self.embedder = Nemotron3_5TranscribeFeatureStacking(config)
        self.layers = nn.ModuleList(
            [Nemotron3_5TranscribeAsrEncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.rotary_emb = Nemotron3_5TranscribeRotaryEmbedding(config)


@auto_docstring(
    custom_intro="""
    The Nemotron 3.5 Transcribe audio encoder: a speech recognition encoder and a speaker diarization model run in
    parallel on the same spectrogram, and the speaker activities are added to the speech recognition encoder outputs.
    """
)
class Nemotron3_5TranscribeEncoder(Nemotron3_5TranscribePreTrainedModel):
    config: Nemotron3_5TranscribeEncoderConfig
    main_input_name = "input_features"
    input_modalities = "audio"
    _no_split_modules = ["Nemotron3_5TranscribeEncoder"]

    def __init__(self, config: Nemotron3_5TranscribeEncoderConfig):
        super().__init__(config)
        self.asr_encoder = AutoModel.from_config(config.asr_config)
        self.diarization_model = AutoModelForAudioFrameClassification.from_config(config.diarization_config)
        self.asr_norm = nn.LayerNorm(config.hidden_size)
        self.speaker_norm = nn.LayerNorm(config.num_speakers)
        self.speaker_embeddings = nn.Buffer(self.compute_speaker_embeddings(config), persistent=False)
        self.post_init()

    @staticmethod
    def compute_speaker_embeddings(config: Nemotron3_5TranscribeEncoderConfig) -> torch.Tensor:
        """Fixed sinusoidal embedding of each speaker slot, which the speaker activities select."""
        position = torch.arange(config.num_speakers, dtype=torch.float32)[:, None]
        inv_freq = torch.exp(
            torch.arange(0, config.hidden_size, 2, dtype=torch.float32) * -(math.log(10000.0) / config.hidden_size)
        )
        embeddings = torch.zeros(config.num_speakers, config.hidden_size)
        embeddings[:, 0::2] = torch.sin(position * inv_freq)
        embeddings[:, 1::2] = torch.cos(position * inv_freq)
        return embeddings

    @staticmethod
    def normalize_features(input_features: torch.Tensor, input_features_mask: torch.Tensor) -> torch.Tensor:
        """Normalizes each mel bin to zero mean and unit variance over the valid frames of each sample."""
        mask = input_features_mask.bool()[..., None]
        input_features = input_features.float().masked_fill(~mask, 0.0)
        num_frames = mask.sum(1)
        mean = input_features.sum(1, keepdim=True) / num_frames[:, None].clamp(min=1)
        centered = (input_features - mean).masked_fill(~mask, 0.0)
        # the original uses the unbiased variance and adds its epsilon to the standard deviation
        std = (centered.pow(2).sum(1, keepdim=True) / (num_frames[:, None] - 1)).sqrt().nan_to_num(0.0) + 1e-5
        return centered / std

    @auto_docstring
    def forward(
        self,
        input_features: torch.FloatTensor,
        input_features_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        r"""
        input_features_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`, *optional*):
            Mask of the valid spectrogram frames. Padding must be on the right.
        """
        if input_features_mask is None:
            input_features_mask = torch.ones(input_features.shape[:2], dtype=torch.long, device=input_features.device)

        asr_features = self.normalize_features(input_features, input_features_mask).to(input_features.dtype)
        asr_outputs = self.asr_encoder(asr_features, attention_mask=input_features_mask, **kwargs)
        hidden_states = asr_outputs.last_hidden_state

        # the diarization model reads the raw spectrogram, in chunks with its speaker cache, and predicts speaker
        # activities at the spectrogram frame rate, averaged back over the spectrogram frames of each encoder frame.
        # Its logits stop at the last input frame, so the input is padded to whole encoder frames.
        padding = -input_features.shape[1] % self.config.subsampling_factor
        speaker_logits = self.diarization_model(
            nn.functional.pad(input_features, (0, 0, 0, padding)),
            attention_mask=nn.functional.pad(input_features_mask, (0, padding)),
        ).logits
        speaker_probs = speaker_logits.float().sigmoid().unflatten(1, (hidden_states.shape[1], -1)).mean(2)

        speaker_activity = (speaker_probs > self.config.speaker_activity_threshold).to(hidden_states.dtype)
        speaker_states = self.speaker_norm(speaker_activity) @ self.speaker_embeddings.to(hidden_states.dtype)
        return BaseModelOutput(
            last_hidden_state=self.asr_norm(hidden_states) + speaker_states,
            hidden_states=asr_outputs.hidden_states,
            attentions=asr_outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    The Nemotron 3.5 Transcribe model (speech recognition encoder and speaker diarization model in parallel, linear
    projector and Nemotron-H language model), without a language modeling head.
    """
)
class Nemotron3_5TranscribeModel(AudioFlamingo3Model):
    def __init__(self, config: Nemotron3_5TranscribeConfig):
        super().__init__(config)
        self.multi_modal_projector = nn.Linear(config.audio_config.hidden_size, config.text_config.hidden_size)

    @can_return_tuple
    @auto_docstring(
        custom_intro="This method is used to get the audio embeddings from input features (a log mel spectrogram), meaning inferring the audio encoder and the multi-modal projector."
    )
    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        input_features_mask: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        input_features (`torch.FloatTensor`):
            Float values of mel features extracted from the raw speech waveform.
        input_features_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`):
            Mask of the valid spectrogram frames. Padding must be on the right.
        """
        audio_output = self.audio_tower(input_features, input_features_mask=input_features_mask, **kwargs)
        audio_embeds = self.multi_modal_projector(audio_output.last_hidden_state)

        valid_mask = input_features_mask[:, :: self.config.audio_config.subsampling_factor].bool()
        return BaseModelOutputWithPooling(
            last_hidden_state=audio_output.last_hidden_state,
            pooler_output=audio_embeds[valid_mask.to(audio_embeds.device)],
            hidden_states=audio_output.hidden_states,
            attentions=audio_output.attentions,
        )

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
    ) -> Nemotron3_5TranscribeModelOutputWithPast:
        r"""
        input_features_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`):
            Mask of the valid spectrogram frames. Padding must be on the right.
        """
        if inputs_embeds is None:
            # the audio token is outside of the text vocabulary, its embeddings are replaced below
            inputs_embeds = self.get_input_embeddings()(
                input_ids.masked_fill(input_ids == self.config.audio_token_id, 0)
            )

        audio_embeds = None
        if input_features is not None and input_ids is not None:
            audio_embeds = self.get_audio_features(input_features, input_features_mask, return_dict=True).pooler_output
            special_audio_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, audio_features=audio_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                special_audio_mask, audio_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            )

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        return Nemotron3_5TranscribeModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            audio_hidden_states=audio_embeds,
        )


@auto_docstring(
    custom_intro="""
    The Nemotron 3.5 Transcribe model (speech recognition encoder and speaker diarization model in parallel, linear
    projector and Nemotron-H language model) with a language modeling head.
    """
)
class Nemotron3_5TranscribeForConditionalGeneration(AudioFlamingo3ForConditionalGeneration):
    _no_split_modules = ["Nemotron3_5TranscribeEncoder"]

    def forward(**super_kwargs):
        r"""
        input_features_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`):
            Mask to avoid performing attention on padding feature indices.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss.

        Example:

        ```python
        >>> from transformers import AutoProcessor, Nemotron3_5TranscribeForConditionalGeneration

        >>> model_id = "nvidia/Nemotron-3.5-Transcribe"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> model = Nemotron3_5TranscribeForConditionalGeneration.from_pretrained(model_id, device_map="auto")
        ```"""
        super().forward(**super_kwargs)


class Nemotron3_5TranscribeProcessorKwargs(ProcessingKwargs, total=False):  # trf-ignore: TRF019
    _defaults = {
        "text_kwargs": {"padding": True},
        "audio_kwargs": {"sampling_rate": 16000},
        "common_kwargs": {"return_tensors": "pt"},
    }


@auto_docstring
class Nemotron3_5TranscribeProcessor(AudioFlamingo3Processor):
    valid_processor_kwargs = Nemotron3_5TranscribeProcessorKwargs

    def __init__(
        self,
        feature_extractor,
        tokenizer,
        chat_template=None,
        audio_token="<|audio|>",
        default_transcription_prompt="Audio transcription. Language: as spoken. Wording: verbatim, including fillers and repetitions. Formatting: punctuation and capitalization. Output: transcript only.",
    ):
        r"""
        audio_token (`str`, *optional*, defaults to `"<|audio|>"`):
            Special token used to represent audio inputs in the chat template.
        default_transcription_prompt (`str`, *optional*):
            Default prompt used by [`~Nemotron3_5TranscribeProcessor.apply_transcription_request`].
        """
        super().__init__(
            feature_extractor,
            tokenizer,
            chat_template=chat_template,
            audio_token=audio_token,
            default_transcription_prompt=default_transcription_prompt,
        )
        del self.max_audio_len

    def _get_audio_token_length(self, audio_lengths: torch.Tensor) -> torch.Tensor:
        # one audio token per `subsampling_factor` spectrogram frames, the last group may be incomplete
        subsampling_factor = 8
        return (audio_lengths + subsampling_factor - 1) // subsampling_factor

    def _process_audio(self, audio, **kwargs):
        audio = self.feature_extractor.fetch_audio(audio)
        audio_inputs = self.feature_extractor(audio, **kwargs)
        audio_inputs["input_features_mask"] = audio_inputs.pop("attention_mask")
        audio_inputs["num_audio_tokens"] = self._get_audio_token_length(audio_inputs["input_features_mask"].sum(-1))
        audio_replacements = [self.replace_audio_token(audio_inputs, audio_idx=idx) for idx in range(len(audio))]
        return audio_inputs, audio_replacements

    def apply_transcription_request(**super_kwargs):
        r"""
        Prepare inputs for automatic speech recognition without manually writing the default transcription prompt.

        Args:
            audio (`str`, `list[str]`, `np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`):
                Audio to transcribe. Strings are interpreted as local paths or URLs and will be loaded automatically by
                the chat template loader; NumPy arrays and PyTorch tensors are forwarded directly.
            prompt (`str` or `list[str]`, *optional*):
                Custom prompt(s) to include in the user turn. A list must be the same length as the batch. When `None`,
                each sample uses `default_transcription_prompt`.
            **kwargs:
                Additional keyword arguments forwarded to
                [`~Nemotron3_5TranscribeProcessor.apply_chat_template`] (for example `text_kwargs`, `audio_kwargs`, ...).

        Returns:
            [`BatchFeature`]: Processor outputs ready to be passed to
            [`Nemotron3_5TranscribeForConditionalGeneration.generate`].
        """
        return super().apply_transcription_request(**super_kwargs)

    def decode(self, *args, **kwargs):
        raise AttributeError("Not needed")

    def batch_decode(self, *args, **kwargs):
        raise AttributeError("Not needed")

    def _strip_assistant_prefix_and_quotes(self, text):
        raise AttributeError("Not needed")


__all__ = [
    "Nemotron3_5TranscribeAsrEncoder",
    "Nemotron3_5TranscribeAsrEncoderConfig",
    "Nemotron3_5TranscribeConfig",
    "Nemotron3_5TranscribeEncoderConfig",
    "Nemotron3_5TranscribeEncoder",
    "Nemotron3_5TranscribeForConditionalGeneration",
    "Nemotron3_5TranscribeModel",
    "Nemotron3_5TranscribePreTrainedModel",
    "Nemotron3_5TranscribeProcessor",
]
