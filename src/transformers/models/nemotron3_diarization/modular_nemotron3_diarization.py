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

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...configuration_utils import PreTrainedConfig
from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    ModelOutput,
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    is_torchdynamo_compiling,
)
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ...utils.type_validators import interval, positive_int_field
from ..clip.modeling_clip import CLIPMLP, CLIPEncoderLayer
from ..glmasr.configuration_glmasr import GlmAsrEncoderConfig
from ..glmasr.modeling_glmasr import GlmAsrAttention
from ..llama.modeling_llama import LlamaRotaryEmbedding
from ..pe_audio.modeling_pe_audio import PeAudioPreTrainedModel


@auto_docstring(checkpoint="nvidia/Nemotron-3-Diarization-preview")
@strict
class Nemotron3DiarizationAudioConfig(GlmAsrEncoderConfig):
    r"""
    subsampling_factor (`int`, *optional*, defaults to 8):
        Number of consecutive spectrogram frames stacked into one encoder frame. The classifier upsamples its
        outputs by the same factor, so speaker activity is predicted at the spectrogram frame rate.
    """

    model_type = "nemotron3_diarization_audio"
    base_config_key = "audio_config"

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


@auto_docstring(checkpoint="nvidia/Nemotron-3-Diarization-preview")
@strict
class Nemotron3DiarizationHeadConfig(PreTrainedConfig):
    r"""
    hidden_size (`int`, *optional*, defaults to 192):
        Hidden size of the speaker head: the encoder output is projected to it, and the upsampler and the classifier
        keep it.
    num_speakers (`int`, *optional*, defaults to 8):
        Maximum number of speakers, i.e. the number of per-frame activity outputs. Speakers are ordered by their first
        arrival in the audio.
    audio_hidden_size (`int`, *optional*, defaults to 512):
        Hidden size of the encoder output the head projects from. Must match
        `Nemotron3DiarizationAudioConfig.hidden_size`.
    subsampling_factor (`int`, *optional*, defaults to 8):
        Upsampling factor of the head, back to the spectrogram frame rate. Must match
        `Nemotron3DiarizationAudioConfig.subsampling_factor`.
    """

    base_config_key = "head_config"

    hidden_size: int = positive_int_field(default=192)
    num_speakers: int = positive_int_field(default=8)
    audio_hidden_size: int = positive_int_field(default=512)
    subsampling_factor: int = positive_int_field(default=8)


@auto_docstring(checkpoint="nvidia/Nemotron-3-Diarization-preview")
@strict
class Nemotron3DiarizationStreamingConfig(PreTrainedConfig):
    r"""
    fifo_length (`int`, *optional*, defaults to 264):
        Capacity of the FIFO queue of the most recent encoder frames in streaming mode (offline mode uses
        `Nemotron3DiarizationConfig.fifo_length`).
    speaker_cache_update_period (`int`, *optional*, defaults to 222):
        Number of encoder frames moved from the FIFO queue to the speaker cache when the queue overflows, in
        streaming mode (offline mode uses `Nemotron3DiarizationConfig.speaker_cache_update_period`).
    speaker_cache_length (`int`, *optional*, defaults to 264):
        Capacity of the Arrival-Order Speaker Cache. Must be at least
        `(1 + speaker_cache_silence_frames_per_speaker) * num_speakers`.
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
    num_speakers (`int`, *optional*, defaults to 8):
        Number of speakers tracked by the speaker cache. Must match `Nemotron3DiarizationHeadConfig.num_speakers`.
    subsampling_factor (`int`, *optional*, defaults to 8):
        Number of speaker-probability frames per encoder frame. Must match
        `Nemotron3DiarizationAudioConfig.subsampling_factor`.
    """

    fifo_length: int = positive_int_field(default=264)
    speaker_cache_update_period: int = positive_int_field(default=222)
    speaker_cache_length: int = positive_int_field(default=264)
    speaker_cache_silence_frames_per_speaker: int = interval(min=0)(default=1)
    prediction_score_threshold: float = 0.25
    latest_frames_score_boost: float = 0.05
    strong_boost_rate: float = 0.75
    weak_boost_rate: float = 1.5
    min_positive_scores_rate: float = 0.5
    num_speakers: int = positive_int_field(default=8)
    subsampling_factor: int = positive_int_field(default=8)


@auto_docstring(checkpoint="nvidia/Nemotron-3-Diarization-preview")
@strict
class Nemotron3DiarizationConfig(PreTrainedConfig):
    r"""
    audio_config (`Nemotron3DiarizationAudioConfig` or `dict`, *optional*):
        Configuration of the transformer audio encoder. Defaults to `Nemotron3DiarizationAudioConfig()`.
    head_config (`Nemotron3DiarizationHeadConfig` or `dict`, *optional*):
        Configuration of the speaker head. Defaults to `Nemotron3DiarizationHeadConfig()`.
    streaming_config (`Nemotron3DiarizationStreamingConfig` or `dict`, *optional*):
        Speaker-cache policy, and the FIFO sizes of streaming mode. Defaults to
        `Nemotron3DiarizationStreamingConfig()`.
    chunk_length (`int`, *optional*, defaults to 340):
        Offline mode: number of encoder frames per chunk when a whole recording is diarized in one forward. In
        streaming mode the chunk is the input of each forward.
    chunk_right_context (`int`, *optional*, defaults to 40):
        Offline mode: number of look-ahead encoder frames each chunk takes from the following ones. In streaming mode
        the look-ahead is `num_lookahead_frames` of each forward.
    fifo_length (`int`, *optional*, defaults to 40):
        Offline mode: capacity of the FIFO queue of the most recent encoder frames. Streaming mode uses
        `streaming_config.fifo_length`.
    speaker_cache_update_period (`int`, *optional*, defaults to 300):
        Offline mode: number of encoder frames moved from the FIFO queue to the speaker cache when the queue
        overflows. Streaming mode uses `streaming_config.speaker_cache_update_period`.
    """

    model_type = "nemotron3_diarization"
    sub_configs = {
        "audio_config": Nemotron3DiarizationAudioConfig,
        "head_config": Nemotron3DiarizationHeadConfig,
        "streaming_config": Nemotron3DiarizationStreamingConfig,
    }

    audio_config: Nemotron3DiarizationAudioConfig | dict | None = None
    head_config: Nemotron3DiarizationHeadConfig | dict | None = None
    streaming_config: Nemotron3DiarizationStreamingConfig | dict | None = None
    chunk_length: int = 340
    chunk_right_context: int = 40
    fifo_length: int = 40
    speaker_cache_update_period: int = 300
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if self.audio_config is None:
            self.audio_config = Nemotron3DiarizationAudioConfig()
        elif isinstance(self.audio_config, dict):
            self.audio_config = Nemotron3DiarizationAudioConfig(**self.audio_config)
        if self.head_config is None:
            self.head_config = Nemotron3DiarizationHeadConfig()
        elif isinstance(self.head_config, dict):
            self.head_config = Nemotron3DiarizationHeadConfig(**self.head_config)
        if self.streaming_config is None:
            self.streaming_config = Nemotron3DiarizationStreamingConfig()
        elif isinstance(self.streaming_config, dict):
            self.streaming_config = Nemotron3DiarizationStreamingConfig(**self.streaming_config)
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        if self.chunk_right_context >= self.chunk_length:
            raise ValueError(
                f"`chunk_right_context` ({self.chunk_right_context}) must be smaller than "
                f"`chunk_length` ({self.chunk_length})."
            )

        if self.head_config.audio_hidden_size != self.audio_config.hidden_size:
            raise ValueError(
                f"`head_config.audio_hidden_size` ({self.head_config.audio_hidden_size}) must match "
                f"`audio_config.hidden_size` ({self.audio_config.hidden_size})."
            )
        if self.head_config.subsampling_factor != self.audio_config.subsampling_factor:
            raise ValueError(
                f"`head_config.subsampling_factor` ({self.head_config.subsampling_factor}) must match "
                f"`audio_config.subsampling_factor` ({self.audio_config.subsampling_factor})."
            )
        if self.streaming_config.num_speakers != self.head_config.num_speakers:
            raise ValueError(
                f"`streaming_config.num_speakers` ({self.streaming_config.num_speakers}) must match "
                f"`head_config.num_speakers` ({self.head_config.num_speakers})."
            )
        if self.streaming_config.subsampling_factor != self.audio_config.subsampling_factor:
            raise ValueError(
                f"`streaming_config.subsampling_factor` ({self.streaming_config.subsampling_factor}) must match "
                f"`audio_config.subsampling_factor` ({self.audio_config.subsampling_factor})."
            )

        silence_frames = self.streaming_config.speaker_cache_silence_frames_per_speaker
        min_speaker_cache_length = (1 + silence_frames) * self.streaming_config.num_speakers
        if self.streaming_config.speaker_cache_length < min_speaker_cache_length:
            raise ValueError(
                f"`streaming_config.speaker_cache_length` ({self.streaming_config.speaker_cache_length}) must be at "
                "least `(1 + streaming_config.speaker_cache_silence_frames_per_speaker) * "
                f"streaming_config.num_speakers` ({min_speaker_cache_length})."
            )


class Nemotron3DiarizationSpeakerCache:
    """
    Streaming state of [`Nemotron3DiarizationForAudioFrameClassification`]: the Arrival-Order Speaker Cache and the
    FIFO queue of the most recent encoder frames, that every chunk attends to.

    Args:
        config (`Nemotron3DiarizationStreamingConfig`):
            Speaker-cache policy, and the FIFO sizes of streaming mode.
        fifo_length (`int`, *optional*):
            Capacity of the FIFO queue of the most recent encoder frames. Defaults to `config.fifo_length`.
        speaker_cache_update_period (`int`, *optional*):
            Number of encoder frames moved from the FIFO queue to the speaker cache when the queue overflows. Defaults
            to `config.speaker_cache_update_period`.
    """

    def __init__(
        self,
        config: Nemotron3DiarizationStreamingConfig,
        fifo_length: int | None = None,
        speaker_cache_update_period: int | None = None,
    ):
        self.fifo_length = fifo_length if fifo_length is not None else config.fifo_length
        self.speaker_cache_update_period = (
            speaker_cache_update_period
            if speaker_cache_update_period is not None
            else config.speaker_cache_update_period
        )
        self.speaker_cache_length = config.speaker_cache_length
        self.num_silence_frames = config.speaker_cache_silence_frames_per_speaker
        self.prediction_score_threshold = config.prediction_score_threshold
        self.latest_frames_score_boost = config.latest_frames_score_boost
        self.num_speakers = config.num_speakers
        self.subsampling_factor = config.subsampling_factor

        # share of the speaker cache every speaker is budgeted, excluding its reserved silence slots, and the frame
        # counts the score policy spends it on when the cache is compressed
        budget = self.speaker_cache_length // self.num_speakers - self.num_silence_frames
        self.min_positive_scores = math.floor(budget * config.min_positive_scores_rate)
        self.num_strong_boosted_frames = math.floor(budget * config.strong_boost_rate)
        self.num_weak_boosted_frames = math.floor(budget * config.weak_boost_rate)

        self.embeds: torch.Tensor | None = None
        self.probs: torch.Tensor | None = None
        self.fifo: torch.Tensor | None = None
        self.num_cache_frames = 0
        self.num_fifo_frames = 0
        self.is_compressed: bool = False
        self.is_initialized: bool = False

    def lazy_initialization(self, chunk_embeds: torch.Tensor):
        batch_size, _, hidden_size = chunk_embeds.shape
        tensor_kwargs = {"device": chunk_embeds.device, "dtype": chunk_embeds.dtype}
        self.embeds = torch.zeros(batch_size, self.speaker_cache_length, hidden_size, **tensor_kwargs)
        self.probs = torch.zeros(batch_size, self.speaker_cache_length, self.num_speakers, **tensor_kwargs)
        self.fifo = torch.zeros(batch_size, self.fifo_length, hidden_size, **tensor_kwargs)
        if not is_torchdynamo_compiling():
            torch._dynamo.mark_static_address(self.embeds)
            torch._dynamo.mark_static_address(self.probs)
            torch._dynamo.mark_static_address(self.fifo)
        self.is_initialized = True

    def get_embeds(self, chunk_embeds: torch.Tensor) -> torch.Tensor:
        if not self.is_initialized:
            self.lazy_initialization(chunk_embeds)
        return torch.cat([self.embeds[:, : self.num_cache_frames], self.fifo[:, : self.num_fifo_frames]], dim=1)

    def _pool_probs(self, logits: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Speaker probabilities at the encoder frame rate, zeroed on the padding frames."""
        factor = self.subsampling_factor
        probs = nn.functional.avg_pool1d(logits.sigmoid().transpose(1, 2), factor, factor).transpose(1, 2)
        if mask is not None:
            probs = probs * mask.to(device=probs.device, dtype=probs.dtype)[..., None]
        return probs.to(self.probs)

    def _num_popped_frames(self, num_fifo_frames: int) -> int:
        """
        No frames move to the speaker cache until the FIFO overflows, then at least `speaker_cache_update_period`
        oldest frames are moved, plus any additional frames needed to restore fifo capacity.
        """
        if num_fifo_frames <= self.fifo_length:
            return 0

        num_popped = max(self.speaker_cache_update_period, num_fifo_frames - self.fifo_length)
        return min(num_popped, num_fifo_frames)

    def update(
        self,
        chunk_input_embeds: torch.Tensor,
        chunk_logits: torch.Tensor,
        silence_embeds: torch.Tensor,
        num_chunk_frames: int,
        mask: torch.Tensor | None = None,
    ):
        """
        Pushes a processed chunk to the FIFO queue, moving its oldest frames to the speaker cache when it overflows.

        Args:
            chunk_input_embeds (`torch.Tensor` of shape `(batch_size, num_input_frames, hidden_size)`):
                Encoder input of the step: the cached frames returned by `get_embeds`, the chunk and its look-ahead.
            chunk_logits (`torch.Tensor` of shape `(batch_size, num_input_frames * subsampling_factor, num_speakers)`):
                Speaker logits of the step, used to score the frames when the speaker cache is compressed.
            silence_embeds (`torch.Tensor` of shape `(hidden_size,)`):
                Learned silence embedding filling the reserved silence slots of a compressed cache.
            num_chunk_frames (`int`):
                Number of chunk frames following the cached frames in `chunk_input_embeds`. Only those join the
                FIFO queue: the look-ahead frames after them are fed again at the next step.
            mask (`torch.Tensor` of shape `(batch_size, num_input_frames)`, *optional*):
                Valid frames of `chunk_input_embeds`, whose padding frames are given zero speaker probabilities.
        """
        if not self.is_initialized:
            self.lazy_initialization(chunk_input_embeds)

        num_cache_frames, num_fifo_frames = self.num_cache_frames, self.num_fifo_frames
        probs = self._pool_probs(chunk_logits, mask)

        chunk_start = num_cache_frames + num_fifo_frames
        chunk_embeds = chunk_input_embeds[:, chunk_start : chunk_start + num_chunk_frames]
        fifo_embeds = torch.cat([self.fifo[:, :num_fifo_frames], chunk_embeds], dim=1)

        num_popped = self._num_popped_frames(fifo_embeds.shape[1])
        if num_popped:
            fifo_probs = probs[:, num_cache_frames : num_cache_frames + fifo_embeds.shape[1]]
            # an uncompressed cache still holds plain chunk frames, whose probabilities this step re-estimates
            # a compressed one is out of order, so the probs stored alongside its frames are the only ones
            stored_probs = self.probs[:, :num_cache_frames] if self.is_compressed else probs[:, :num_cache_frames]
            cache_embeds = torch.cat([self.embeds[:, :num_cache_frames], fifo_embeds[:, :num_popped]], dim=1)
            cache_probs = torch.cat([stored_probs, fifo_probs[:, :num_popped]], dim=1)
            fifo_embeds = fifo_embeds[:, num_popped:]

            if cache_embeds.shape[1] > self.speaker_cache_length:
                cache_embeds, cache_probs = self._compress(cache_embeds, cache_probs, silence_embeds)
                self.is_compressed = True
            self.num_cache_frames = cache_embeds.shape[1]

            cache_indices = torch.arange(self.num_cache_frames, device=self.embeds.device)
            self.embeds.index_copy_(1, cache_indices, cache_embeds)
            self.probs.index_copy_(1, cache_indices, cache_probs)

        self.num_fifo_frames = fifo_embeds.shape[1]
        self.fifo.index_copy_(1, torch.arange(self.num_fifo_frames, device=self.fifo.device), fifo_embeds)

    def _get_frame_scores(self, probs: torch.Tensor) -> torch.Tensor:
        threshold = self.prediction_score_threshold
        log_probs = torch.log(probs.clamp(min=threshold))
        log_complements = torch.log((1.0 - probs).clamp(min=threshold))

        scores = log_probs - log_complements + log_complements.sum(dim=-1, keepdim=True) - math.log(0.5)

        is_speech = probs > 0.5
        scores = scores.masked_fill(~is_speech, float("-inf"))
        is_positive = scores > 0
        has_enough_positive = is_positive.sum(dim=1, keepdim=True) >= self.min_positive_scores

        scores = scores.masked_fill(~is_positive & is_speech & has_enough_positive, float("-inf"))
        return scores

    def _boost_scores(self, scores: torch.Tensor, num_boosted: int, boost: float) -> torch.Tensor:
        _, topk_indices = torch.topk(scores, num_boosted, dim=1, sorted=False)
        scores = scores.scatter_add(1, topk_indices, scores.new_full(topk_indices.shape, boost))
        return scores

    def _compress(self, embeds: torch.Tensor, probs: torch.Tensor, silence_embeds: torch.Tensor):
        """
        Keeps the `speaker_cache_length` most important frames, grouped by speaker and in their original order within
        a speaker. `speaker_cache_silence_frames_per_speaker` slots per speaker are filled with `silence_embeds`.
        """
        batch_size, num_frames, num_speakers = probs.shape

        scores = self._get_frame_scores(probs)
        # frames beyond the cache capacity are the ones popped from fifo
        scores[:, self.speaker_cache_length :] += self.latest_frames_score_boost

        scores = self._boost_scores(scores, self.num_strong_boosted_frames, boost=-2.0 * math.log(0.5))
        scores = self._boost_scores(scores, self.num_weak_boosted_frames, boost=-math.log(0.5))
        scores = nn.functional.pad(scores, (0, 0, 0, self.num_silence_frames), value=float("inf"))
        embeds = torch.cat([embeds, silence_embeds.to(embeds).view(1, 1, -1).expand(batch_size, 1, -1)], dim=1)
        probs = nn.functional.pad(probs, (0, 0, 0, 1))

        num_scored_frames = num_frames + self.num_silence_frames
        sentinel = num_scored_frames * num_speakers
        flat_scores = scores.transpose(1, 2).reshape(batch_size, -1)

        topk_scores, topk_indices = torch.topk(flat_scores, self.speaker_cache_length, dim=1, sorted=False)
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
    logits (`torch.FloatTensor` of shape `(batch_size, num_frames, config.head_config.num_speakers)`):
        Per-frame speaker activity logits at the spectrogram frame rate. `logits.sigmoid()` gives the probability
        that each speaker is active in each frame; speakers are ordered by their first arrival in the audio.
    hidden_states (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True`):
        Encoder hidden states of every chunk, in chunk order: the encoder runs once per chunk, so the tuple holds
        `config.audio_config.num_hidden_layers + 1` tensors per chunk. Their sequence length is the chunk's, cache
        and look-ahead frames included.
    attentions (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True`):
        Encoder attention weights of every chunk, in chunk order, `config.audio_config.num_hidden_layers` tensors
        per chunk.
    speaker_cache (`Nemotron3DiarizationSpeakerCache`, *optional*, returned in streaming mode):
        Updated streaming state, to pass to the forward of the next audio chunk of the same streams.
    """

    logits: torch.Tensor | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None
    speaker_cache: Nemotron3DiarizationSpeakerCache | None = None


class Nemotron3DiarizationFeatureStacking(nn.Module):
    """Stacks `subsampling_factor` consecutive spectrogram frames and projects them to the encoder hidden size."""

    def __init__(self, config: Nemotron3DiarizationAudioConfig):
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
    def __init__(self, config: Nemotron3DiarizationAudioConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        # The original fused query/key/value projection has no bias while the output projection has one.
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)


class Nemotron3DiarizationMLP(CLIPMLP): ...


class Nemotron3DiarizationAudioLayer(CLIPEncoderLayer):
    def __init__(self, config: Nemotron3DiarizationAudioConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = Nemotron3DiarizationAttention(config, layer_idx)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)


@auto_docstring
class Nemotron3DiarizationPreTrainedModel(PeAudioPreTrainedModel):
    config: Nemotron3DiarizationConfig
    base_model_prefix = "model"
    main_input_name = "input_features"
    input_modalities = "audio"
    _no_split_modules = ["Nemotron3DiarizationAudioLayer"]
    _skip_keys_device_placement = ["speaker_cache"]
    _can_record_outputs = {
        "hidden_states": Nemotron3DiarizationAudioLayer,
        "attentions": Nemotron3DiarizationAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
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


@auto_docstring
class Nemotron3DiarizationAudioModel(Nemotron3DiarizationPreTrainedModel):
    config: Nemotron3DiarizationAudioConfig

    def __init__(self, config: Nemotron3DiarizationAudioConfig):
        super().__init__(config)
        self.embedder = Nemotron3DiarizationFeatureStacking(config)
        self.input_layer_norm = nn.LayerNorm(config.hidden_size)
        self.layers = nn.ModuleList(
            [Nemotron3DiarizationAudioLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.rotary_emb = Nemotron3DiarizationRotaryEmbedding(config)
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_features: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        if (input_features is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of `input_features` and `inputs_embeds`.")

        if inputs_embeds is None:
            inputs_embeds = self.embedder(input_features)
            # if inputs_embeds is provided, we expect attention_mask already downsampled
            if attention_mask is not None:
                attention_mask = attention_mask[:, :: self.config.subsampling_factor].bool()

        if position_ids is None:
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
        return BaseModelOutput(last_hidden_state=self.layer_norm(hidden_states))


class Nemotron3DiarizationSubpixelUpsampler(nn.Module):
    """Upsamples the encoder frame rate back to the spectrogram frame rate with a sub-pixel convolution."""

    def __init__(self, config: Nemotron3DiarizationHeadConfig):
        super().__init__()
        self.upsample_factor = config.subsampling_factor
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size * config.subsampling_factor,
            kernel_size=3,
            padding=1,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, hidden_size = hidden_states.shape
        hidden_states = self.conv(hidden_states.transpose(1, 2)).transpose(1, 2)
        return hidden_states.reshape(batch_size, num_frames * self.upsample_factor, hidden_size)


class Nemotron3DiarizationClassificationHead(nn.Module):
    def __init__(self, config: Nemotron3DiarizationHeadConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.num_speakers)
        self.act_fn = ACT2FN["relu"]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(self.act_fn(hidden_states))
        return self.out_proj(self.act_fn(hidden_states))


class Nemotron3DiarizationSpeakerHead(nn.Module):
    """Turns encoder frames into per-speaker activity logits at the spectrogram frame rate."""

    def __init__(self, config: Nemotron3DiarizationHeadConfig):
        super().__init__()
        self.proj = nn.Linear(config.audio_hidden_size, config.hidden_size)
        self.upsampler = Nemotron3DiarizationSubpixelUpsampler(config)
        self.classifier = Nemotron3DiarizationClassificationHead(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.upsampler(self.proj(hidden_states)))


@auto_docstring(
    custom_intro="""
    Streaming Sortformer speaker diarization model: predicts, for every spectrogram frame, the activity of up to
    `config.head_config.num_speakers` speakers ordered by first arrival. Audio is processed chunk by chunk, each
    chunk attending to a few look-ahead frames and to the Arrival-Order Speaker Cache and FIFO queue carried in a
    [`Nemotron3DiarizationSpeakerCache`]. A whole recording is chunked by the forward itself (offline mode); a stream
    is fed one chunk per forward (streaming mode).
    """
)
class Nemotron3DiarizationForAudioFrameClassification(Nemotron3DiarizationPreTrainedModel):
    def __init__(self, config: Nemotron3DiarizationConfig):
        super().__init__(config)
        self.model = Nemotron3DiarizationAudioModel(config.audio_config)
        self.head = Nemotron3DiarizationSpeakerHead(config.head_config)
        self.silence_embeds = nn.Parameter(torch.zeros(config.audio_config.hidden_size))
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        speaker_cache: Nemotron3DiarizationSpeakerCache | None = None,
        num_lookahead_frames: int | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Nemotron3DiarizationOutput:
        r"""
        speaker_cache (`Nemotron3DiarizationSpeakerCache`, *optional*):
            Streaming state returned by the forward of the previous chunk of the same audio streams.
        num_lookahead_frames (`int`, *optional*):
            Streaming mode: number of trailing encoder frames of the input that are look-ahead only. They are attended
            to, but their logits are not returned and they do not join the FIFO queue, as they open the next chunk.
            [`Nemotron3DiarizationProcessor`] sets it for every chunk but the last one of a session.

            The two arguments select the mode. Streaming mode, one chunk per forward: `num_lookahead_frames` given
            (a first chunk creates the `speaker_cache`, later chunks receive it), or `speaker_cache` given alone (the
            last chunk of the session, no look-ahead). The input minus its look-ahead is one chunk, whatever its
            length, pushed as a whole to the FIFO queue sized by `config.streaming_config`. Offline mode, neither
            given: the input is a whole recording, split by the forward into chunks of `config.chunk_length` encoder
            frames that take up to `config.chunk_right_context` look-ahead frames from the following ones, with a
            FIFO queue sized by `config.fifo_length`; no cache is returned.

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
        is_streaming = num_lookahead_frames is not None or speaker_cache is not None
        if speaker_cache is None:
            # the cache defaults to the streaming FIFO sizes, offline mode overrides them
            offline_sizes = {}
            if not is_streaming:
                offline_sizes = {
                    "fifo_length": self.config.fifo_length,
                    "speaker_cache_update_period": self.config.speaker_cache_update_period,
                }
            speaker_cache = Nemotron3DiarizationSpeakerCache(self.config.streaming_config, **offline_sizes)
        if num_lookahead_frames is None:
            num_lookahead_frames = 0

        batch_size, num_frames, _ = input_features.shape
        inputs_embeds = self.model.embedder(input_features)
        num_embeds = inputs_embeds.shape[1]
        subsampling_factor = self.config.audio_config.subsampling_factor

        num_chunk_embeds = num_embeds - num_lookahead_frames
        if not is_torchdynamo_compiling() and (num_lookahead_frames < 0 or num_chunk_embeds < 1):
            raise ValueError(
                f"`num_lookahead_frames` ({num_lookahead_frames}) must be between 0 and one less than the number of "
                f"encoder frames of the input ({num_embeds})."
            )

        embed_mask = None
        if attention_mask is not None:
            embed_mask = attention_mask[:, ::subsampling_factor].bool()

        if is_streaming:
            chunk_length, chunk_right_context = num_chunk_embeds, num_lookahead_frames
        else:
            chunk_length, chunk_right_context = self.config.chunk_length, self.config.chunk_right_context

        logits = []
        # the encoder runs once per chunk, so its recorded outputs are concatenated in chunk order
        all_hidden_states, all_attentions = (), ()
        for start_idx in range(0, num_chunk_embeds, chunk_length):
            end_idx = min(start_idx + chunk_length, num_chunk_embeds)
            num_chunk_frames = end_idx - start_idx
            chunk_embeds = inputs_embeds[:, start_idx : min(end_idx + chunk_right_context, num_embeds)]

            cached_embeds = speaker_cache.get_embeds(chunk_embeds)
            cached_length = cached_embeds.shape[1]
            chunk_input_embeds = torch.cat([cached_embeds, chunk_embeds], dim=1)

            step_mask = None
            if embed_mask is not None:
                chunk_mask = embed_mask[:, start_idx : start_idx + chunk_embeds.shape[1]]
                step_mask = torch.cat([chunk_mask.new_ones(batch_size, cached_length), chunk_mask], dim=1)

            # positions restart at every chunk
            position_ids = torch.arange(chunk_input_embeds.shape[1], device=chunk_input_embeds.device)[None, :]
            encoder_outputs: BaseModelOutput = self.model(
                inputs_embeds=chunk_input_embeds,
                attention_mask=step_mask,
                position_ids=position_ids,
                **kwargs,
            )
            all_hidden_states += encoder_outputs.hidden_states or ()
            all_attentions += encoder_outputs.attentions or ()
            chunk_logits = self.head(encoder_outputs.last_hidden_state)
            speaker_cache.update(
                chunk_input_embeds, chunk_logits, self.silence_embeds, num_chunk_frames, mask=step_mask
            )

            start_logit_idx = cached_length * subsampling_factor
            end_logit_idx = (cached_length + num_chunk_frames) * subsampling_factor
            logits.append(chunk_logits[:, start_logit_idx:end_logit_idx])

        # with no look-ahead, the last encoder frame may be the padding added by feature stacking
        logits = torch.cat(logits, dim=1)[:, :num_frames]

        return Nemotron3DiarizationOutput(
            logits=logits,
            hidden_states=all_hidden_states or None,
            attentions=all_attentions or None,
            speaker_cache=speaker_cache if is_streaming else None,
        )


__all__ = [
    "Nemotron3DiarizationAudioConfig",
    "Nemotron3DiarizationAudioModel",
    "Nemotron3DiarizationConfig",
    "Nemotron3DiarizationForAudioFrameClassification",
    "Nemotron3DiarizationHeadConfig",
    "Nemotron3DiarizationOutput",
    "Nemotron3DiarizationPreTrainedModel",
    "Nemotron3DiarizationSpeakerCache",
    "Nemotron3DiarizationStreamingConfig",
]
