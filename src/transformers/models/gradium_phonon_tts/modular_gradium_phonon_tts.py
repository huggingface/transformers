# Copyright 2026 Gradium and The HuggingFace Inc. team. All rights reserved.
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

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...processing_utils import ProcessorMixin
from ...utils import auto_docstring
from ...utils.type_validators import activation_fn_key, positive_int_field
from ..pocket_tts.configuration_pocket_tts import PocketTTSConfig, PocketTTSFlowHeadConfig
from ..pocket_tts.modeling_pocket_tts import (
    PocketTTSDecoderLayer,
    PocketTTSForConditionalGeneration,
    PocketTTSModel,
    PocketTTSPreTrainedModel,
    PocketTTSRotaryEmbedding,
    PocketTTSTimestepNorm,
)
from ..pocket_tts.processing_pocket_tts import PocketTTSProcessor
from .generation_gradium_phonon_tts import GradiumPhononTTSGenerationMixin


@auto_docstring(checkpoint="eustlb/gradium-phonon-tts")
@strict
class GradiumPhononTTSFlowHeadConfig(PocketTTSFlowHeadConfig):
    r"""
    latent_dim (`int`, *optional*, defaults to 32):
        Dimension of the continuous audio latents the head predicts (the latent dimension of the audio tokenizer).
    condition_dim (`int`, *optional*, defaults to 768):
        Dimension of the backbone hidden state the head is conditioned on (the backbone `hidden_size`).
    frequency_embedding_size (`int`, *optional*, defaults to 256):
        Size of the sinusoidal embedding of the flow times.
    layer_norm_eps (`float`, *optional*, defaults to 1e-06):
        Epsilon of the layer normalizations inside the residual blocks and the final layer.
    rms_norm_eps (`float`, *optional*, defaults to 1e-05):
        Epsilon of the normalization closing the time embeddings.
    max_period (`float`, *optional*, defaults to 10000.0):
        Maximum period of the sinusoidal time embedding, which the checkpoint stores as `freqs`.
    flow_type (`str`, *optional*, defaults to `"lsd"`):
        Objective the head was trained with, which sets how it is conditioned and sampled. `"lsd"` (Lagrangian
        self-distillation): conditioned on the start and end times of a flow step, sampled by Euler steps from Gaussian
        noise. `"drifting"`: no time conditioning, the head maps noise to the latent in one call.
    """

    condition_dim: int = positive_int_field(default=768)


@auto_docstring(checkpoint="eustlb/gradium-phonon-tts")
@strict
class GradiumPhononTTSConfig(PocketTTSConfig):
    r"""
    flow_head_config (`Union[GradiumPhononTTSFlowHeadConfig, dict]`, *optional*):
        The config of the flow head that samples each latent from the backbone hidden state.
    layer_norm_eps (`float`, *optional*, defaults to 1e-05):
        Epsilon of the layer normalizations of the backbone.
    latent_dim (`int`, *optional*, defaults to 32):
        Dimension of the audio latents the backbone takes as input (`audio_latents`); equals the flow head's and the audio
        tokenizer's `latent_dim`.
    audio_prompt_token_id (`int`, *optional*, defaults to 4000):
        Placeholder token id of the voice-prompt frames in `input_ids`.
    audio_latent_token_id (`int`, *optional*, defaults to 4001):
        Placeholder token id of the audio latent positions in `input_ids`.
    audio_prompt_dim (`int`, *optional*, defaults to 512):
        Dimension of the voice-prompt latents (`audio_prompt_latents`), the unquantized latents of Gradium's speaker
        codec (a Mimi encoder, not the audio tokenizer's), which the backbone projects to `hidden_size`.
    """

    model_type = "gradium_phonon_tts"
    sub_configs = {"flow_head_config": GradiumPhononTTSFlowHeadConfig}

    vocab_size: int = positive_int_field(default=4002)
    hidden_size: int = positive_int_field(default=768)
    num_attention_heads: int = positive_int_field(default=12)
    intermediate_size: int = positive_int_field(default=3072)
    hidden_act: str = activation_fn_key(default="gelu")
    audio_prompt_dim: int = positive_int_field(default=512)
    # Difference to Pocket TTS: no learnt vector before the voice prompt
    audio_prompt_bos_token_id = AttributeError()


class GradiumPhononTTSDecoderLayer(PocketTTSDecoderLayer):
    pass


class GradiumPhononTTSRotaryEmbedding(PocketTTSRotaryEmbedding):
    pass


class GradiumPhononTTSTimestepNorm(PocketTTSTimestepNorm):
    """
    The normalization closing the time embeddings. Difference to Pocket TTS (whose Python original divides by
    `sqrt(unbiased variance + eps)`): Gradium's runtime divides by `sqrt(biased variance + eps)` and rescales by
    `sqrt((n - 1) / n)`, the same up to where `eps` sits.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__(hidden_size, eps=eps)
        self.unbiased_scale = math.sqrt((hidden_size - 1) / hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.var(-1, unbiased=False, keepdim=True)
        return self.weight * hidden_states * torch.rsqrt(variance + self.variance_epsilon) * self.unbiased_scale


class GradiumPhononTTSPreTrainedModel(PocketTTSPreTrainedModel):
    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, GradiumPhononTTSModel):
            init.zeros_(module.num_speakers_embedding)


@auto_docstring(
    custom_intro="""
    The Gradium Phonon TTS backbone: a causal transformer over one stream made of the voice prompt (speaker-codec
    latents), the text tokens and the audio latents, returning the normalised hidden states.
    """
)
class GradiumPhononTTSModel(PocketTTSModel):
    def __init__(self, config: GradiumPhononTTSConfig):
        # `embed_tokens` keeps Llama's `padding_idx` (the pad token's row gets no gradient); the original's table has
        # none, which only matters for training.
        super().__init__(config)
        self.layers = nn.ModuleList(
            [GradiumPhononTTSDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.rotary_emb = GradiumPhononTTSRotaryEmbedding(config=config)
        # Difference to Pocket TTS: the voice prompt is made of speaker-codec latents (`audio_prompt_dim`), not the audio
        # tokenizer's.
        self.audio_prompt_proj = nn.Linear(config.audio_prompt_dim, config.hidden_size, bias=False)
        self.audio_latent_proj = nn.Linear(config.latent_dim, config.hidden_size, bias=False)
        # The learnt latent of the first audio position (the processor's trailing `audio_latent_token`).
        self.bos_latent = nn.Parameter(torch.zeros(config.latent_dim))
        # The original's `num_speakers` conditioner, a lookup table whose only released entry (one speaker) is folded
        # into this vector at conversion; it is added to every audio latent position.
        self.num_speakers_embedding = nn.Parameter(torch.zeros(config.hidden_size))
        self.post_init()

    def embed_audio_latents(self, audio_latents: torch.Tensor) -> torch.Tensor:
        """
        Embeds normalised audio latents (the learnt BOS latent or the generated ones) into the backbone stream, with the
        speaker conditioning added to every audio latent position.
        """
        latent_embeds = self.audio_latent_proj(audio_latents)
        # `.to`: with `device_map`, the vector can sit on another device than the projection
        return latent_embeds + self.num_speakers_embedding.to(latent_embeds.device)


@auto_docstring(
    custom_intro="""
    The Gradium Phonon TTS text-to-speech model: the backbone, the flow head that samples one audio latent per step and
    the end-of-speech head. It generates audio tokenizer latents, which the processor decodes into 24 kHz audio.
    """
)
class GradiumPhononTTSForConditionalGeneration(PocketTTSForConditionalGeneration, GradiumPhononTTSGenerationMixin):
    pass


@auto_docstring
class GradiumPhononTTSProcessor(PocketTTSProcessor):
    def __init__(
        self,
        feature_extractor,
        tokenizer,
        audio_tokenizer=None,
        chat_template=None,
        short_text_max_words: int = 4,
        frames_after_eos_short_text: int = 3,
        frames_after_eos: int = 1,
        fade_in_duration: float = 0.0,
    ):
        r"""
        audio_tokenizer (`PocketTTSAudioTokenizerModel`, *optional*):
            The codec: it decodes the generated latents into audio ([`~GradiumPhononTTSProcessor.decode`]). The released
            checkpoint has no encoder: the voice is passed as precomputed `audio_prompt_latents`.
        short_text_max_words (`int`, *optional*, defaults to 4):
            Texts of at most this many words get `frames_after_eos_short_text` frames after the end-of-speech frame.
        frames_after_eos_short_text (`int`, *optional*, defaults to 3):
            Number of frames generated after the end-of-speech frame for short texts.
        frames_after_eos (`int`, *optional*, defaults to 1):
            Number of frames generated after the end-of-speech frame for longer texts.
        fade_in_duration (`float`, *optional*, defaults to 0.0):
            Duration in seconds of a linear fade-in [`~GradiumPhononTTSProcessor.decode`] applies to the start of every
            generated waveform. Gradium's runtime applies none (Pocket TTS's original fades the first 5 ms in).
        """
        self.short_text_max_words = short_text_max_words
        self.frames_after_eos_short_text = frames_after_eos_short_text
        self.frames_after_eos = frames_after_eos
        self.fade_in_duration = fade_in_duration
        # registered on the tokenizer at conversion (`model_specific_special_tokens`); the voice-prompt placeholder is the
        # processor's `audio_token`, expanded to one token per voice frame
        self.audio_token = tokenizer.audio_prompt_token
        self.audio_token_id = tokenizer.convert_tokens_to_ids(self.audio_token)
        self.audio_latent_token = tokenizer.audio_latent_token
        # Difference to Pocket TTS: no voice BOS, the chat template writes the voice placeholder first
        self.voice_prompt = self.audio_token
        self.placeholder_tokens = [self.audio_token, self.audio_latent_token]
        ProcessorMixin.__init__(
            self, feature_extractor, tokenizer, audio_tokenizer=audio_tokenizer, chat_template=chat_template
        )


__all__ = [
    "GradiumPhononTTSConfig",
    "GradiumPhononTTSFlowHeadConfig",
    "GradiumPhononTTSPreTrainedModel",
    "GradiumPhononTTSModel",
    "GradiumPhononTTSForConditionalGeneration",
    "GradiumPhononTTSProcessor",
]
