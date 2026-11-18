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

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ...utils import auto_docstring
from ...utils.type_validators import activation_fn_key, positive_int_field
from ..pocket_tts.configuration_pocket_tts import PocketTTSConfig, PocketTTSFlowHeadConfig
from ..pocket_tts.modeling_pocket_tts import (
    PocketTTSForConditionalGeneration,
    PocketTTSModel,
    PocketTTSPreTrainedModel,
    PocketTTSTimestepNorm,
)
from .generation_gradium_phonon_tts import GradiumPhononTTSGenerationMixin


@auto_docstring(checkpoint="gradium/phonon-31m-en")
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
    loss_weighting_hidden_size (`int`, *optional*, defaults to 32):
        Width of the MLP that learns the uncertainty weighting of the LSD loss terms from their flow times (training
        only).
    loss_weighting_num_layers (`int`, *optional*, defaults to 3):
        Number of hidden layers of that MLP.
    """

    condition_dim: int = positive_int_field(default=768)


@auto_docstring(checkpoint="gradium/phonon-31m-en")
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
    audio_prompt_token_id (`int`, *optional*, defaults to 5):
        Placeholder token id of the voice-prompt frames in `input_ids` (`<|audio_prompt|>`).
    audio_latent_token_id (`int`, *optional*, defaults to 6):
        Placeholder token id of the audio latent positions in `input_ids` (`<|audio_latent|>`).
    audio_bos_token_id (`int`, *optional*, defaults to 4):
        Token id of the first audio position, closing the prompt: it holds the learnt BOS latent, from which the first
        frame is generated (`<|audio_bos|>`).
    frame_rate (`float`, *optional*, defaults to 12.5):
        Number of audio latents per second of audio: `generate` derives its default frame budget from it.
    eos_loss_weight (`float`, *optional*, defaults to 0.1):
        Weight of the end-of-speech loss in the training loss.
    flow_loss_num_samples (`int`, *optional*, defaults to 4):
        Number of noise draws per target frame in the flow loss.
    lsd_distill_probability (`float`, *optional*, defaults to 0.25):
        Probability, per training step, of computing the self-distillation term of the LSD loss (reweighted to keep the
        expected loss); 1 computes it at every step.
    lsd_diagonal_weight (`float`, *optional*, defaults to 0.75):
        Weight of the diagonal (instantaneous flow) term of the LSD loss; the self-distillation term weighs the rest.
    lsd_time_logit_mean (`float`, *optional*, defaults to 0.4):
        Mean of the logit-normal distribution of the flow times drawn by the LSD loss.
    lsd_time_logit_std (`float`, *optional*, defaults to 1.0):
        Standard deviation of the logit-normal distribution of the flow times drawn by the LSD loss.
    latent_stats_ema_steps (`int`, *optional*, defaults to 1000):
        Number of first training steps during which `latent_mean` and `latent_std` follow an exponential moving average
        of the target latents; 0 keeps them frozen.
    latent_stats_ema_decay (`float`, *optional*, defaults to 0.999):
        Decay of that moving average.
    audio_prompt_dim (`int`, *optional*, defaults to 512):
        Dimension of the voice-prompt latents (`audio_prompt_latents`), the unquantized latents of Gradium's speaker
        codec (a Mimi encoder, not the audio tokenizer's), which the backbone projects to `hidden_size`.
    num_speakers (`int`, *optional*, defaults to 1):
        Value of the `num_speakers` condition: the row of its lookup table added to every audio latent position.
    num_speakers_vocab_size (`int`, *optional*, defaults to 32):
        Number of rows of the `num_speakers` lookup table.
    num_speakers_embedding_dim (`int`, *optional*, defaults to 16):
        Width of the `num_speakers` lookup table, projected to `hidden_size`.
    duration_delta (`float`, *optional*, defaults to 0.0):
        Value of the `duration_delta` condition, embedded sinusoidally and added to every audio latent position.
    duration_delta_embedding_dim (`int`, *optional*, defaults to 128):
        Size of the sinusoidal embedding of `duration_delta`, projected to `hidden_size`.
    duration_delta_scale (`float`, *optional*, defaults to 1000.0):
        Factor applied to `duration_delta` before its sinusoidal embedding.
    duration_delta_max_period (`float`, *optional*, defaults to 10000.0):
        Maximum period of the sinusoidal embedding of `duration_delta`.
    """

    vocab_size: int = positive_int_field(default=4001)
    hidden_size: int = positive_int_field(default=768)
    num_attention_heads: int = positive_int_field(default=12)
    intermediate_size: int = positive_int_field(default=3072)
    hidden_act: str = activation_fn_key(default="gelu")
    audio_prompt_dim: int = positive_int_field(default=512)
    audio_prompt_token_id: int = 5
    audio_latent_token_id: int = 6
    audio_bos_token_id: int = 4
    num_speakers: int = 1
    num_speakers_vocab_size: int = positive_int_field(default=32)
    num_speakers_embedding_dim: int = positive_int_field(default=16)
    duration_delta: float = 0.0
    duration_delta_embedding_dim: int = positive_int_field(default=128)
    duration_delta_scale: float = 1000.0
    duration_delta_max_period: float = 10000.0

    # Difference to Pocket TTS: no learnt vector before the voice prompt
    audio_prompt_bos_token_id = AttributeError()


class GradiumPhononTTSTimestepNorm(PocketTTSTimestepNorm):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        mean_square = hidden_states.pow(2).mean(-1, keepdim=True)
        return self.weight * hidden_states * torch.rsqrt(mean_square + self.variance_epsilon)


class GradiumPhononTTSPreTrainedModel(PocketTTSPreTrainedModel): ...


@auto_docstring(
    custom_intro="""
    The Gradium Phonon TTS backbone: a causal transformer over one stream made of the voice prompt (speaker-codec
    latents), the text tokens and the audio latents, returning the normalised hidden states.
    """
)
class GradiumPhononTTSModel(PocketTTSModel):
    def __init__(self, config: GradiumPhononTTSConfig):
        super().__init__(config)
        self.audio_prompt_proj = nn.Linear(config.audio_prompt_dim, config.hidden_size, bias=False)

        self.num_speakers_embed = nn.Embedding(config.num_speakers_vocab_size, config.num_speakers_embedding_dim)
        self.num_speakers_proj = nn.Linear(config.num_speakers_embedding_dim, config.hidden_size, bias=False)
        self.duration_delta_proj = nn.Linear(config.duration_delta_embedding_dim, config.hidden_size, bias=False)

    def get_condition_embeds(self, device: torch.device) -> torch.Tensor:
        """The sum of the conditions, one `hidden_size` vector (built on `device`, the modules called for offloading)."""
        num_speakers = torch.full((), self.config.num_speakers, dtype=torch.long, device=device)
        condition_embeds = self.num_speakers_proj(self.num_speakers_embed(num_speakers))
        half_dim = self.config.duration_delta_embedding_dim // 2
        exponents = torch.arange(half_dim, device=device, dtype=torch.float32) / (half_dim - 1)
        phases = self.config.duration_delta_scale * self.config.duration_delta
        phases = phases / self.config.duration_delta_max_period**exponents
        duration_embeds = torch.cat([phases.cos(), phases.sin()]).to(condition_embeds.dtype)
        condition_embeds = condition_embeds + self.duration_delta_proj(duration_embeds).to(condition_embeds.device)
        return condition_embeds

    def embed_audio_latents(self, audio_latents: torch.Tensor) -> torch.Tensor:
        """
        Embeds normalised audio latents (the learnt BOS latent or the generated ones) into the backbone stream, with the
        speaker conditioning added to every audio latent position.
        """
        # the generated latents are fp32 whatever the model's dtype
        latent_embeds = self.audio_latent_proj(audio_latents.to(self.audio_latent_proj.weight.dtype))
        return latent_embeds + self.get_condition_embeds(latent_embeds.device).to(latent_embeds.device)


@auto_docstring(
    custom_intro="""
    The Gradium Phonon TTS text-to-speech model: the backbone, the flow head that samples one audio latent per step and
    the end-of-speech head. It generates audio tokenizer latents, which the processor decodes into 24 kHz audio.
    """
)
class GradiumPhononTTSForConditionalGeneration(PocketTTSForConditionalGeneration, GradiumPhononTTSGenerationMixin):
    # Gradium's runtime checkpoint also holds the codec (the processor's audio tokenizer) and training-only tensors
    _keys_to_ignore_on_load_unexpected = [r"^mimi\.", r"^flow_lm\.flow\.w_s_t\.", r"\.learnt_padding$"]


__all__ = [
    "GradiumPhononTTSConfig",
    "GradiumPhononTTSFlowHeadConfig",
    "GradiumPhononTTSPreTrainedModel",
    "GradiumPhononTTSModel",
    "GradiumPhononTTSForConditionalGeneration",
]
