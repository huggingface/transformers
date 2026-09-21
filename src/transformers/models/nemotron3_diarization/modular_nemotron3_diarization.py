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
from dataclasses import dataclass
from itertools import pairwise

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...configuration_utils import PreTrainedConfig
from ...masking_utils import create_bidirectional_mask
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, is_torchdynamo_compiling
from ...utils.generic import can_return_tuple, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..clip.modeling_clip import CLIPMLP, CLIPEncoderLayer
from ..glmasr.configuration_glmasr import GlmAsrEncoderConfig
from ..glmasr.modeling_glmasr import GlmAsrAttention
from ..llama.modeling_llama import LlamaRotaryEmbedding


@auto_docstring(checkpoint="nvidia/Nemotron-3-Diarization-preview")
@strict
class Nemotron3DiarizationEncoderConfig(GlmAsrEncoderConfig):
    r"""
    subsampling_factor (`int`, *optional*, defaults to 8):
        Number of consecutive spectrogram frames stacked into one encoder frame. The classifier upsamples its
        outputs by the same factor, so speaker activity is predicted at the spectrogram frame rate.
    """

    model_type = "nemotron3_diarization_encoder"
    base_config_key = "encoder_config"

    hidden_size: int = 512
    num_hidden_layers: int = 31
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    subsampling_factor: int = 8
    max_position_embeddings: int = 5000

    def __post_init__(self, **kwargs):
        kwargs.setdefault("partial_rotary_factor", 1.0)
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"`hidden_size` ({self.hidden_size}) must be divisible by `num_attention_heads` "
                f"({self.num_attention_heads})."
            )
        if self.subsampling_factor < 1:
            raise ValueError(f"`subsampling_factor` must be a positive integer, got {self.subsampling_factor}.")


@auto_docstring(checkpoint="nvidia/Nemotron-3-Diarization-preview")
@strict
class Nemotron3DiarizationConfig(PreTrainedConfig):
    r"""
    encoder_config (`Nemotron3DiarizationEncoderConfig` or `dict`, *optional*):
        Configuration of the transformer encoder. Defaults to `Nemotron3DiarizationEncoderConfig()`.
    speaker_hidden_size (`int`, *optional*, defaults to 192):
        Hidden size of the speaker head (projection of the encoder output, upsampler and classifier).
    num_speakers (`int`, *optional*, defaults to 8):
        Maximum number of speakers, i.e. the number of per-frame activity outputs. Speakers are ordered by their first
        arrival in the audio.
    speaker_cache_length (`int`, *optional*, defaults to 264):
        Capacity of the Arrival-Order Speaker Cache, in encoder frames. Must be at least
        `(1 + speaker_cache_silence_frames_per_speaker) * num_speakers`.
    fifo_length (`int`, *optional*, defaults to 40):
        Capacity of the FIFO queue of the most recent encoder frames, in encoder frames.
    chunk_length (`int`, *optional*, defaults to 340):
        Number of encoder frames processed per streaming step.
    chunk_right_context (`int`, *optional*, defaults to 40):
        Number of look-ahead encoder frames appended to each chunk. The input buffer latency of a step is
        `(chunk_length + chunk_right_context)` encoder frames.
    speaker_cache_update_period (`int`, *optional*, defaults to 300):
        Number of encoder frames moved from the FIFO queue to the speaker cache when the queue overflows.
    speaker_cache_silence_frames_per_speaker (`int`, *optional*, defaults to 1):
        Number of speaker-cache slots per speaker reserved for the learned silence embedding when the cache is
        compressed.
    prediction_score_threshold (`float`, *optional*, defaults to 0.25):
        Lower clamp of the speaker probabilities before taking their log in the speaker-cache frame scores.
    latest_frames_score_boost (`float`, *optional*, defaults to 0.05):
        Score bonus given to the frames newly added to the speaker cache when it is compressed.
    strong_boost_rate (`float`, *optional*, defaults to 0.75):
        Fraction of the per-speaker cache budget whose best frames get a strong score boost, so that every speaker
        keeps at least that many frames in the cache.
    weak_boost_rate (`float`, *optional*, defaults to 1.5):
        Fraction of the per-speaker cache budget whose best frames get a weak score boost, which prevents one speaker
        from dominating the cache.
    min_positive_scores_rate (`float`, *optional*, defaults to 0.5):
        Fraction of the per-speaker cache budget: a speaker with at least that many positively scored frames has its
        non-positive (overlapped speech) frames excluded from the cache.
    """

    model_type = "nemotron3_diarization"
    sub_configs = {"encoder_config": Nemotron3DiarizationEncoderConfig}

    encoder_config: Nemotron3DiarizationEncoderConfig | dict | None = None
    initializer_range: float = 0.02
    speaker_hidden_size: int = 192
    num_speakers: int = 8
    speaker_cache_length: int = 264
    fifo_length: int = 40
    chunk_length: int = 340
    chunk_right_context: int = 40
    speaker_cache_update_period: int = 300
    speaker_cache_silence_frames_per_speaker: int = 1
    prediction_score_threshold: float = 0.25
    latest_frames_score_boost: float = 0.05
    strong_boost_rate: float = 0.75
    weak_boost_rate: float = 1.5
    min_positive_scores_rate: float = 0.5

    def __post_init__(self, **kwargs):
        if self.encoder_config is None:
            self.encoder_config = Nemotron3DiarizationEncoderConfig()
        elif isinstance(self.encoder_config, dict):
            self.encoder_config = Nemotron3DiarizationEncoderConfig(**self.encoder_config)
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        min_speaker_cache_length = (1 + self.speaker_cache_silence_frames_per_speaker) * self.num_speakers
        if self.speaker_cache_length < min_speaker_cache_length:
            raise ValueError(
                f"`speaker_cache_length` ({self.speaker_cache_length}) must be at least "
                f"`(1 + speaker_cache_silence_frames_per_speaker) * num_speakers` ({min_speaker_cache_length})."
            )
        for name in ["chunk_length", "speaker_cache_update_period"]:
            if getattr(self, name) < 1:
                raise ValueError(f"`{name}` must be a positive integer, got {getattr(self, name)}.")
        for name in ["fifo_length", "chunk_right_context", "speaker_cache_silence_frames_per_speaker"]:
            if getattr(self, name) < 0:
                raise ValueError(f"`{name}` must be a non-negative integer, got {getattr(self, name)}.")
        if self.chunk_right_context >= self.chunk_length:
            raise ValueError(
                f"`chunk_right_context` ({self.chunk_right_context}) must be smaller than "
                f"`chunk_length` ({self.chunk_length})."
            )

    def get_text_config(self, *args, **kwargs):
        return self.encoder_config


class Nemotron3DiarizationSpeakerCache:
    def __init__(self, config: Nemotron3DiarizationConfig):
        self.config = config
        self.embeds: torch.Tensor | None = None
        self.probs: torch.Tensor | None = None
        self.fifo: torch.Tensor | None = None
        self.cache_length = 0
        self.fifo_length = 0
        self.is_compressed: bool = False
        self.is_initialized: bool = False

    def lazy_initialization(self, chunk_embeds: torch.Tensor):
        batch_size, _, hidden_size = chunk_embeds.shape
        tensor_kwargs = {"device": chunk_embeds.device, "dtype": chunk_embeds.dtype}
        self.embeds = torch.zeros(batch_size, self.config.speaker_cache_length, hidden_size, **tensor_kwargs)
        self.probs = torch.zeros(
            batch_size, self.config.speaker_cache_length, self.config.num_speakers, **tensor_kwargs
        )
        self.fifo = torch.zeros(batch_size, self.config.fifo_length, hidden_size, **tensor_kwargs)
        if not is_torchdynamo_compiling():
            torch._dynamo.mark_static_address(self.embeds)
            torch._dynamo.mark_static_address(self.probs)
            torch._dynamo.mark_static_address(self.fifo)
        self.is_initialized = True

    def get_embeds(self, chunk_embeds: torch.Tensor) -> torch.Tensor:
        if not self.is_initialized:
            self.lazy_initialization(chunk_embeds)
        return torch.cat([self.embeds[:, : self.cache_length], self.fifo[:, : self.fifo_length]], dim=1)

    @property
    def num_frames_per_speaker(self) -> int:
        """Share of the speaker cache every speaker is budgeted, excluding its reserved silence slots."""
        capacity = self.config.speaker_cache_length // self.config.num_speakers
        return capacity - self.config.speaker_cache_silence_frames_per_speaker

    @staticmethod
    def _copy_into(buffer: torch.Tensor, values: torch.Tensor):
        buffer.index_copy_(1, torch.arange(values.shape[1], device=buffer.device), values)

    def _pool_probs(self, logits: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Speaker probabilities at the encoder frame rate, zeroed on the padding frames."""
        factor = self.config.encoder_config.subsampling_factor
        probs = nn.functional.avg_pool1d(logits.sigmoid().transpose(1, 2), factor, factor).transpose(1, 2)
        if mask is not None:
            probs = probs * mask.to(device=probs.device, dtype=probs.dtype)[..., None]
        return probs.to(self.probs)

    def _num_popped_frames(self, fifo_length: int) -> int:
        """
        No frames move to the speaker cache until the FIFO overflows, then at least `speaker_cache_update_period`
        oldest frames are moved, plus any additional frames needed to restore fifo capacity.
        """
        if fifo_length <= self.config.fifo_length:
            return 0

        num_popped = max(self.config.speaker_cache_update_period, fifo_length - self.config.fifo_length)
        return min(num_popped, fifo_length)

    def update(
        self,
        chunk_input_embeds: torch.Tensor,
        chunk_logits: torch.Tensor,
        silence_embeds: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        if not self.is_initialized:
            self.lazy_initialization(chunk_input_embeds)

        cache_length, fifo_length = self.cache_length, self.fifo_length
        probs = self._pool_probs(chunk_logits, mask)

        # the right context is re-fed at the next step, so only the chunk itself joins fifo
        chunk_start = cache_length + fifo_length
        chunk_embeds = chunk_input_embeds[:, chunk_start : chunk_start + self.config.chunk_length]
        fifo_embeds = torch.cat([self.fifo[:, :fifo_length], chunk_embeds], dim=1)

        num_popped = self._num_popped_frames(fifo_embeds.shape[1])
        if num_popped:
            fifo_probs = probs[:, cache_length : cache_length + fifo_embeds.shape[1]]
            # an uncompressed cache still holds plain chunk frames, whose probabilities this step re-estimates
            # a compressed one is out of order, so the probs stored alongside its frames are the only ones
            stored_probs = self.probs[:, :cache_length] if self.is_compressed else probs[:, :cache_length]
            cache_embeds = torch.cat([self.embeds[:, :cache_length], fifo_embeds[:, :num_popped]], dim=1)
            cache_probs = torch.cat([stored_probs, fifo_probs[:, :num_popped]], dim=1)
            fifo_embeds = fifo_embeds[:, num_popped:]

            if cache_embeds.shape[1] > self.config.speaker_cache_length:
                cache_embeds, cache_probs = self._compress(cache_embeds, cache_probs, silence_embeds)
                self.is_compressed = True
            self.cache_length = cache_embeds.shape[1]

            self._copy_into(self.embeds, cache_embeds)
            self._copy_into(self.probs, cache_probs)

        self.fifo_length = fifo_embeds.shape[1]
        self._copy_into(self.fifo, fifo_embeds)

    def _get_frame_scores(self, probs: torch.Tensor) -> torch.Tensor:
        threshold = self.config.prediction_score_threshold
        log_probs = torch.log(probs.clamp(min=threshold))
        log_complements = torch.log((1.0 - probs).clamp(min=threshold))

        scores = log_probs - log_complements + log_complements.sum(dim=-1, keepdim=True) - math.log(0.5)

        is_speech = probs > 0.5
        scores = scores.masked_fill(~is_speech, float("-inf"))
        min_positive_scores = math.floor(self.num_frames_per_speaker * self.config.min_positive_scores_rate)
        is_positive = scores > 0
        has_enough_positive = is_positive.sum(dim=1, keepdim=True) >= min_positive_scores

        scores = scores.masked_fill(~is_positive & is_speech & has_enough_positive, float("-inf"))
        return scores

    def _boost_scores(self, scores: torch.Tensor, rate: float, boost: float) -> torch.Tensor:
        num_boosted = math.floor(self.num_frames_per_speaker * rate)
        _, topk_indices = torch.topk(scores, num_boosted, dim=1, sorted=False)
        scores = scores.scatter_add(1, topk_indices, scores.new_full(topk_indices.shape, boost))
        return scores

    def _compress(self, embeds: torch.Tensor, probs: torch.Tensor, silence_embeds: torch.Tensor):
        """
        Keeps the `speaker_cache_length` most important frames, grouped by speaker and in their original order within
        a speaker. `speaker_cache_silence_frames_per_speaker` slots per speaker are filled with `silence_embeds`.
        """
        batch_size, num_frames, num_speakers = probs.shape
        num_silence_frames = self.config.speaker_cache_silence_frames_per_speaker

        scores = self._get_frame_scores(probs)
        # frames beyond the cache capacity are the ones popped from fifo
        scores[:, self.config.speaker_cache_length :] += self.config.latest_frames_score_boost

        scores = self._boost_scores(scores, self.config.strong_boost_rate, boost=-2.0 * math.log(0.5))
        scores = self._boost_scores(scores, self.config.weak_boost_rate, boost=-math.log(0.5))
        scores = nn.functional.pad(scores, (0, 0, 0, num_silence_frames), value=float("inf"))
        embeds = torch.cat([embeds, silence_embeds.to(embeds).view(1, 1, -1).expand(batch_size, 1, -1)], dim=1)
        probs = nn.functional.pad(probs, (0, 0, 0, 1))

        num_scored_frames = num_frames + num_silence_frames
        sentinel = num_scored_frames * num_speakers
        flat_scores = scores.transpose(1, 2).reshape(batch_size, -1)

        topk_scores, topk_indices = torch.topk(flat_scores, self.config.speaker_cache_length, dim=1, sorted=False)
        topk_indices = topk_indices.masked_fill(topk_scores == float("-inf"), sentinel)
        topk_indices, _ = torch.sort(topk_indices, dim=1)
        frame_indices = torch.where(
            topk_indices == sentinel, num_frames, (topk_indices % num_scored_frames).clamp(max=num_frames)
        )

        batch_indices = torch.arange(batch_size, device=probs.device)[:, None]
        return (
            embeds[batch_indices.to(embeds.device), frame_indices.to(embeds.device)],
            probs[batch_indices, frame_indices],
        )


@auto_docstring(
    custom_intro="""
    Output of [`Nemotron3DiarizationForAudioFrameClassification`].
    """
)
@dataclass
class Nemotron3DiarizationOutput(ModelOutput):
    r"""
    logits (`torch.FloatTensor` of shape `(batch_size, num_frames, config.num_speakers)`):
        Per-frame speaker activity logits at the spectrogram frame rate. `logits.sigmoid()` gives the probability
        that each speaker is active in each frame; speakers are ordered by their first arrival in the audio.
    speaker_cache (`Nemotron3DiarizationSpeakerCache`, *optional*, returned when `use_cache=True`):
        Updated streaming state, to pass to the forward of the next audio chunk.
    """

    logits: torch.Tensor | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None
    speaker_cache: Nemotron3DiarizationSpeakerCache | None = None


class Nemotron3DiarizationFeatureStacking(nn.Module):
    """Stacks `subsampling_factor` consecutive spectrogram frames and projects them to the encoder hidden size."""

    def __init__(self, config: Nemotron3DiarizationEncoderConfig):
        super().__init__()
        self.subsampling_factor = config.subsampling_factor
        self.projection = nn.Linear(config.subsampling_factor * config.num_mel_bins, config.hidden_size, bias=False)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, num_mel_bins = input_features.shape
        # The original zero-pads the last incomplete group of frames.
        padding = -num_frames % self.subsampling_factor
        input_features = nn.functional.pad(input_features, (0, 0, 0, padding))
        stacked = input_features.reshape(
            batch_size, (num_frames + padding) // self.subsampling_factor, num_mel_bins * self.subsampling_factor
        )
        return self.projection(stacked)


class Nemotron3DiarizationRotaryEmbedding(LlamaRotaryEmbedding): ...


class Nemotron3DiarizationAttention(GlmAsrAttention):
    def __init__(self, config: Nemotron3DiarizationEncoderConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        # The original fused query/key/value projection has no bias while the output projection has one.
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)


class Nemotron3DiarizationMLP(CLIPMLP): ...


class Nemotron3DiarizationEncoderLayer(CLIPEncoderLayer):
    def __init__(self, config: Nemotron3DiarizationEncoderConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = Nemotron3DiarizationAttention(config, layer_idx)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)


@auto_docstring
class Nemotron3DiarizationPreTrainedModel(PreTrainedModel):
    config: Nemotron3DiarizationConfig
    base_model_prefix = "model"
    main_input_name = "input_features"
    input_modalities = "audio"
    _no_split_modules = ["Nemotron3DiarizationEncoderLayer"]
    _skip_keys_device_placement = ["speaker_cache"]

    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _can_compile_fullgraph = True
    _can_record_outputs = {
        "hidden_states": Nemotron3DiarizationEncoderLayer,
        "attentions": Nemotron3DiarizationAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Nemotron3DiarizationForAudioFrameClassification):
            init.zeros_(module.silence_embeds)
        elif isinstance(module, Nemotron3DiarizationSubpixelUpsampler):
            # The original initializes the sub-pixel convolution as nearest-neighbour upsampling: zero weights except
            # a repeated identity on the center tap, zero bias.
            weight = torch.zeros_like(module.conv.weight)
            hidden_size = module.conv.in_channels
            weight[:, :, 1] = torch.eye(hidden_size, device=weight.device, dtype=weight.dtype).repeat(
                module.upsample_factor, 1
            )
            init.copy_(module.conv.weight, weight)
            init.zeros_(module.conv.bias)

    def _get_feat_extract_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Number of encoder frames produced by feature stacking for `input_lengths` spectrogram frames."""
        factor = self.config.get_text_config().subsampling_factor
        return (input_lengths + factor - 1) // factor


class Nemotron3DiarizationEncoder(Nemotron3DiarizationPreTrainedModel):
    config: Nemotron3DiarizationEncoderConfig

    def __init__(self, config: Nemotron3DiarizationEncoderConfig):
        super().__init__(config)
        self.feature_stacking = Nemotron3DiarizationFeatureStacking(config)
        self.input_layer_norm = nn.LayerNorm(config.hidden_size)
        self.layers = nn.ModuleList(
            [Nemotron3DiarizationEncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.rotary_emb = Nemotron3DiarizationRotaryEmbedding(config)
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_features: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        if (input_features is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of `input_features` and `inputs_embeds`.")

        if inputs_embeds is None:
            inputs_embeds = self.feature_stacking(input_features)
            if attention_mask is not None:
                lengths = self._get_feat_extract_output_lengths(attention_mask.sum(dim=-1))
                attention_mask = (
                    torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)[None, :] < lengths[:, None]
                )

        if position_embeddings is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)[None, :]
            position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        attention_mask = create_bidirectional_mask(
            config=self.config, inputs_embeds=inputs_embeds, attention_mask=attention_mask
        )
        hidden_states = self.input_layer_norm(inputs_embeds)
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.layer_norm(hidden_states)


class Nemotron3DiarizationSubpixelUpsampler(nn.Module):
    """Upsamples the encoder frame rate back to the spectrogram frame rate with a sub-pixel convolution."""

    def __init__(self, config: Nemotron3DiarizationConfig):
        super().__init__()
        self.upsample_factor = config.encoder_config.subsampling_factor
        self.conv = nn.Conv1d(
            config.speaker_hidden_size,
            config.speaker_hidden_size * config.encoder_config.subsampling_factor,
            kernel_size=3,
            padding=1,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, hidden_size = hidden_states.shape
        hidden_states = self.conv(hidden_states.transpose(1, 2)).transpose(1, 2)
        return hidden_states.reshape(batch_size, num_frames * self.upsample_factor, hidden_size)


class Nemotron3DiarizationClassificationHead(nn.Module):
    def __init__(self, config: Nemotron3DiarizationConfig):
        super().__init__()
        self.dense = nn.Linear(config.speaker_hidden_size, config.speaker_hidden_size)
        self.out_proj = nn.Linear(config.speaker_hidden_size, config.num_speakers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(nn.functional.relu(hidden_states))
        return self.out_proj(nn.functional.relu(hidden_states))


@auto_docstring
class Nemotron3DiarizationModel(Nemotron3DiarizationPreTrainedModel):
    def __init__(self, config: Nemotron3DiarizationConfig):
        super().__init__(config)
        self.encoder = Nemotron3DiarizationEncoder(config.encoder_config)
        self.speaker_projection = nn.Linear(config.encoder_config.hidden_size, config.speaker_hidden_size)
        self.upsampler = Nemotron3DiarizationSubpixelUpsampler(config)
        self.classifier = Nemotron3DiarizationClassificationHead(config)
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_features: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Nemotron3DiarizationOutput:
        hidden_states = self.encoder(
            input_features=input_features,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = self.speaker_projection(hidden_states)
        hidden_states = self.upsampler(hidden_states)
        logits = self.classifier(hidden_states)

        return Nemotron3DiarizationOutput(logits=logits)


@auto_docstring(
    custom_intro="""
    Streaming Sortformer speaker diarization model: predicts, for every spectrogram frame, the activity of up to
    `config.num_speakers` speakers ordered by first arrival. Audio is processed in chunks of
    `config.chunk_length` encoder frames (plus `config.chunk_right_context` look-ahead frames) that attend to the
    Arrival-Order Speaker Cache and FIFO queue carried in a [`Nemotron3DiarizationSpeakerCache`].
    """
)
class Nemotron3DiarizationForAudioFrameClassification(Nemotron3DiarizationPreTrainedModel):
    def __init__(self, config: Nemotron3DiarizationConfig):
        super().__init__(config)
        self.model = Nemotron3DiarizationModel(config)
        self.silence_embeds = nn.Parameter(torch.zeros(config.encoder_config.hidden_size))
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        speaker_cache: Nemotron3DiarizationSpeakerCache | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Nemotron3DiarizationOutput:
        r"""
        speaker_cache (`Nemotron3DiarizationSpeakerCache`, *optional*):
            Streaming state returned by a previous forward, to continue diarizing the same audio streams.
        use_cache (`bool`, *optional*):
            Whether more audio will follow. When `True`, the input must hold complete chunks of
            `config.chunk_length * config.encoder_config.subsampling_factor` frames followed by at most
            `config.chunk_right_context * config.encoder_config.subsampling_factor` look-ahead frames; only the chunk frames are
            returned in `logits`, the look-ahead frames must be passed again at the start of the next call, and the
            updated `speaker_cache` is returned. When `False` (the default), the input is the end of the audio: all
            frames are returned and complete chunks take their look-ahead from the following frames.

        Example:

        ```python
        >>> from transformers import AutoModelForAudioFrameClassification, AutoProcessor
        >>> from transformers.audio_utils import load_audio

        >>> model_id = "nvidia/Nemotron-3-Diarization-preview"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> model = AutoModelForAudioFrameClassification.from_pretrained(model_id, device_map="auto")

        >>> sampling_rate = processor.feature_extractor.sampling_rate
        >>> audio = load_audio(
        ...     "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/en-Alice_woman.wav",
        ...     sampling_rate=sampling_rate,
        ... )
        >>> inputs = processor(audio, sampling_rate=sampling_rate).to(model.device, dtype=model.dtype)
        >>> probabilities = model(**inputs).logits.sigmoid()  # (1, num_frames, 8), one frame every 10 ms
        ```
        """
        batch_size, num_frames, _ = input_features.shape
        inputs_embeds = self.model.encoder.feature_stacking(input_features)

        num_embeds = inputs_embeds.shape[1]
        embed_mask = None
        if attention_mask is not None:
            embed_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(dim=-1))
            embed_mask = torch.arange(num_embeds, device=inputs_embeds.device)[None, :] < embed_lengths[:, None]

        chunk_length = self.config.chunk_length
        right_context = self.config.chunk_right_context
        subsampling_factor = self.config.encoder_config.subsampling_factor
        if use_cache:
            num_chunks = num_embeds // chunk_length
            num_lookahead = num_embeds - num_chunks * chunk_length
            if num_chunks == 0 or num_lookahead > right_context:
                raise ValueError(
                    "With `use_cache=True`, the input must hold at least one complete chunk of "
                    f"{chunk_length * subsampling_factor} spectrogram frames followed by at most "
                    f"{right_context * subsampling_factor} look-ahead frames, got {num_frames} frames."
                )
        else:
            num_chunks = math.ceil(num_embeds / chunk_length)

        if speaker_cache is None and (use_cache or num_chunks > 1):
            speaker_cache = Nemotron3DiarizationSpeakerCache(self.config)

        max_step_length = self.config.speaker_cache_length + self.config.fifo_length + chunk_length + right_context
        position_ids = torch.arange(max_step_length, device=inputs_embeds.device)[None, :]
        position_embeddings = self.model.encoder.rotary_emb(inputs_embeds, position_ids)

        logits = []
        all_hidden_states = []
        all_attentions = []
        chunk_boundaries = range(0, (num_chunks + 1) * chunk_length, chunk_length)
        for start_idx, end_idx in pairwise(chunk_boundaries):
            chunk_embeds = inputs_embeds[:, start_idx : end_idx + right_context]

            cached_length = 0
            chunk_input_embeds = chunk_embeds
            if speaker_cache is not None:
                cached_embeds = speaker_cache.get_embeds(chunk_embeds)
                cached_length = cached_embeds.shape[1]
                chunk_input_embeds = torch.cat([cached_embeds, chunk_embeds], dim=1)

            step_mask = None
            if embed_mask is not None:
                chunk_mask = embed_mask[:, start_idx : end_idx + right_context]
                step_mask = chunk_mask
                if cached_length:
                    step_mask = torch.cat([chunk_mask.new_ones(batch_size, cached_length), chunk_mask], dim=1)

            # positions restart at every chunk
            chunk_position_embeddings = tuple(
                embedding[:, : chunk_input_embeds.shape[1]] for embedding in position_embeddings
            )

            outputs = self.model(
                inputs_embeds=chunk_input_embeds,
                attention_mask=step_mask,
                position_embeddings=chunk_position_embeddings,
                return_dict=True,
                **kwargs,
            )
            chunk_logits = outputs.logits
            if outputs.hidden_states is not None:
                all_hidden_states.extend(outputs.hidden_states)
            if outputs.attentions is not None:
                all_attentions.extend(outputs.attentions)

            if speaker_cache is not None:
                speaker_cache.update(chunk_input_embeds, chunk_logits, self.silence_embeds, mask=step_mask)

            start_logit_idx = cached_length * subsampling_factor
            end_logit_idx = (cached_length + min(chunk_length, chunk_embeds.shape[1])) * subsampling_factor
            logits.append(chunk_logits[:, start_logit_idx:end_logit_idx])

        # remove padding added by feature stacking
        logits = torch.cat(logits, dim=1)[:, :num_frames]

        return Nemotron3DiarizationOutput(
            logits=logits,
            hidden_states=tuple(all_hidden_states) if all_hidden_states else None,
            attentions=tuple(all_attentions) if all_attentions else None,
            speaker_cache=speaker_cache if use_cache else None,
        )


__all__ = [
    "Nemotron3DiarizationConfig",
    "Nemotron3DiarizationEncoderConfig",
    "Nemotron3DiarizationForAudioFrameClassification",
    "Nemotron3DiarizationModel",
    "Nemotron3DiarizationOutput",
    "Nemotron3DiarizationPreTrainedModel",
    "Nemotron3DiarizationSpeakerCache",
]
