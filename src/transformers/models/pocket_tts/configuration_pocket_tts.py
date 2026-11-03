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

from typing import Literal

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring
from ...utils.type_validators import activation_fn_key, interval, positive_int_field


@auto_docstring(checkpoint="eustlb/pocket-tts")
@strict
class PocketTTSFlowHeadConfig(PreTrainedConfig):
    r"""
    latent_dim (`int`, *optional*, defaults to 32):
        Dimension of the continuous audio latents the head predicts (the latent dimension of the audio tokenizer).
    condition_dim (`int`, *optional*, defaults to 1024):
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

    base_config_key = "flow_head_config"

    hidden_size: int = positive_int_field(default=512)
    num_hidden_layers: int = positive_int_field(default=6)
    latent_dim: int = positive_int_field(default=32)
    condition_dim: int = positive_int_field(default=1024)
    frequency_embedding_size: int = positive_int_field(default=256)
    hidden_act: str = activation_fn_key(default="silu")
    layer_norm_eps: float = interval(min=0.0)(default=1e-6)
    rms_norm_eps: float = interval(min=0.0)(default=1e-5)
    max_period: float = interval(min=0.0)(default=10000.0)
    flow_type: Literal["lsd", "drifting"] = "lsd"
    loss_weighting_hidden_size: int = positive_int_field(default=32)
    loss_weighting_num_layers: int = positive_int_field(default=3)

    @property
    def num_time_conditions(self) -> int:
        """Number of flow times the head is conditioned on: the start and end of the step for LSD, none for drifting."""
        return 2 if self.flow_type == "lsd" else 0


@auto_docstring(checkpoint="eustlb/pocket-tts")
@strict
class PocketTTSConfig(PreTrainedConfig):
    r"""
    flow_head_config (`Union[PocketTTSFlowHeadConfig, dict]`, *optional*):
        The config of the flow head that samples each latent from the backbone hidden state.
    layer_norm_eps (`float`, *optional*, defaults to 1e-05):
        Epsilon of the layer normalizations of the backbone.
    latent_dim (`int`, *optional*, defaults to 32):
        Dimension of the audio latents the backbone takes as input (`audio_prompt_latents`, `audio_latents`); equals the
        flow head's and the audio tokenizer's `latent_dim`.
    audio_prompt_token_id (`int`, *optional*, defaults to 4000):
        Placeholder token id of the voice-prompt frames in `input_ids`.
    audio_latent_token_id (`int`, *optional*, defaults to 4001):
        Placeholder token id of the audio latent positions in `input_ids`.
    audio_prompt_bos_token_id (`int`, *optional*, defaults to 4002):
        Token id placed before the voice prompt; its embedding is the learnt vector the original prepends to the voice.
    audio_bos_token_id (`int`, *optional*, defaults to 4003):
        Token id of the first audio position, closing the prompt: it holds the learnt BOS latent, from which the first
        frame is generated.
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
    """

    model_type = "pocket_tts"
    sub_configs_defaults = {"flow_head_config": SubConfigSpec(config_class=PocketTTSFlowHeadConfig)}

    flow_head_config: dict | PreTrainedConfig | None = None
    vocab_size: int = positive_int_field(default=4004)
    hidden_size: int = positive_int_field(default=1024)
    num_hidden_layers: int = positive_int_field(default=6)
    num_attention_heads: int = positive_int_field(default=16)
    num_key_value_heads: int | None = None
    head_dim: int | None = None
    intermediate_size: int = positive_int_field(default=4096)
    hidden_act: str = activation_fn_key(default="gelu_pytorch_tanh")
    layer_norm_eps: float = interval(min=0.0)(default=1e-5)
    attention_bias: bool = False
    attention_dropout: float = interval(min=0.0, max=1.0)(default=0.0)
    max_position_embeddings: int = positive_int_field(default=8192)
    rope_parameters: RopeParameters | dict | None = None
    latent_dim: int = positive_int_field(default=32)
    pad_token_id: int = 3
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    audio_prompt_token_id: int = 4000
    audio_latent_token_id: int = 4001
    audio_prompt_bos_token_id: int = 4002
    audio_bos_token_id: int = 4003
    frame_rate: float = interval(min=0.0)(default=12.5)
    initializer_range: float = interval(min=0.0, max=1.0)(default=0.02)
    use_cache: bool = True
    eos_loss_weight: float = interval(min=0.0)(default=0.1)
    flow_loss_num_samples: int = positive_int_field(default=4)
    lsd_distill_probability: float = interval(min=0.0, max=1.0)(default=0.25)
    lsd_diagonal_weight: float = interval(min=0.0, max=1.0)(default=0.75)
    lsd_time_logit_mean: float = 0.4
    lsd_time_logit_std: float = interval(min=0.0)(default=1.0)
    latent_stats_ema_steps: int = interval(min=0)(default=1000)
    latent_stats_ema_decay: float = interval(min=0.0, max=1.0)(default=0.999)

    def __post_init__(self, **kwargs):
        self.head_dim = self.head_dim or self.hidden_size // self.num_attention_heads
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})."
            )
        if self.flow_head_config.condition_dim != self.hidden_size:
            raise ValueError(
                f"`flow_head_config.condition_dim` ({self.flow_head_config.condition_dim}) must equal the backbone "
                f"`hidden_size` ({self.hidden_size})."
            )
        if self.flow_head_config.latent_dim != self.latent_dim:
            raise ValueError(
                f"`flow_head_config.latent_dim` ({self.flow_head_config.latent_dim}) must equal `latent_dim` "
                f"({self.latent_dim})."
            )


__all__ = ["PocketTTSConfig", "PocketTTSFlowHeadConfig"]
