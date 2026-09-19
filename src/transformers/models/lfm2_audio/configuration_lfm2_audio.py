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

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..lfm2.configuration_lfm2 import Lfm2Config
from ..parakeet.configuration_parakeet import ParakeetEncoderConfig


@auto_docstring(checkpoint="LiquidAI/LFM2.5-Audio-1.5B")
@strict
class Lfm2AudioDepthConfig(PreTrainedConfig):
    r"""
    dim (`int`, *optional*, defaults to 1024):
        Hidden size of the codebook-depth transformer.
    tie (`bool`, *optional*, defaults to `True`):
        Whether each codebook's input embedding and output projection share weights.
    multiple_of (`int`, *optional*, defaults to 256):
        Multiple used when deriving the SwiGLU intermediate size.
    rope_theta (`float`, *optional*, defaults to 1000000.0):
        Base period of the depth transformer's rotary position embeddings.
    """

    model_type = "lfm2_audio_depth"
    attribute_map = {"hidden_size": "dim", "num_hidden_layers": "layers", "rms_norm_eps": "norm_eps"}

    attention_bias: bool = False
    attention_dropout: float = 0.0

    layers: int = 6
    dim: int = 1024
    tie: bool = True
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    intermediate_size: int | None = None
    multiple_of: int = 256
    norm_eps: float = 1e-5
    rope_theta: float = 1_000_000.0

    def __post_init__(self, **kwargs):
        if self.intermediate_size is None:
            swiglu_size = int(2 * (4 * self.dim) / 3)
            self.intermediate_size = self.multiple_of * math.ceil(swiglu_size / self.multiple_of)
        if self.dim % self.num_attention_heads != 0:
            raise ValueError("`depthformer.dim` must be divisible by `num_attention_heads`.")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("`num_attention_heads` must be divisible by `num_key_value_heads`.")
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="LiquidAI/LFM2.5-Audio-1.5B")
@strict
class Lfm2AudioConfig(PreTrainedConfig):
    r"""
    codebooks (`int`, *optional*, defaults to 8):
        Number of Mimi codebooks generated for every audio timestep.
    semantic_codebook_factor (`float`, *optional*, defaults to 100.0):
        Relative loss weight assigned to the first, semantic Mimi codebook.
    codebook_weight (`str`, *optional*, defaults to `"log"`):
        Strategy used to interpolate loss weights between the semantic and remaining codebooks. Must be either
        `"log"` or `"linear"`.
    text_loss_multiplier (`float`, *optional*, defaults to 1.0):
        Multiplier applied to the text loss when text and audio targets are trained together.
    audio_loss_multiplier (`float`, *optional*, defaults to 1.0):
        Multiplier applied to the audio loss when text and audio targets are trained together.
    interleaved_n_text (`int`, *optional*, defaults to 6):
        Number of consecutive text tokens in interleaved generation.
    interleaved_n_audio (`int`, *optional*, defaults to 12):
        Number of consecutive audio frames in interleaved generation.
    encoder (`dict` or [`ParakeetEncoderConfig`], *optional*):
        Configuration of the FastConformer audio encoder.
    lfm (`dict` or [`Lfm2Config`], *optional*):
        Configuration of the LFM2 language-model backbone.
    depthformer (`dict` or [`Lfm2AudioDepthConfig`], *optional*):
        Configuration of the depth transformer that predicts the Mimi codebooks within an audio frame.
    audio_vocab_size (`int`, *optional*, defaults to 2049):
        Vocabulary size of each audio codebook, including the end-of-audio token.
    audio_token_id (`int`, *optional*, defaults to 133):
        Token used as an audio-input feature placeholder.
    audio_start_token_id (`int`, *optional*, defaults to 128):
        Text token that switches sequential generation from text to audio.
    text_end_token_id (`int`, *optional*, defaults to 130):
        Token that marks completion of the text stream during interleaved generation.
    audio_eos_token_id (`int`, *optional*, defaults to 2048):
        End-of-audio code used by every Mimi codebook.
    """

    model_type = "lfm2_audio"
    sub_configs = {
        "encoder": ParakeetEncoderConfig,
        "lfm": Lfm2Config,
        "depthformer": Lfm2AudioDepthConfig,
    }

    codebooks: int = 8
    semantic_codebook_factor: float | int = 100.0
    codebook_weight: str = "log"
    text_loss_multiplier: float | int | None = 1.0
    audio_loss_multiplier: float | int | None = 1.0
    interleaved_n_text: int = 6
    interleaved_n_audio: int = 12
    encoder: dict | PreTrainedConfig | None = None
    lfm: dict | PreTrainedConfig | None = None
    depthformer: dict | PreTrainedConfig | None = None
    audio_vocab_size: int = 2049
    audio_token_id: int = 133
    audio_start_token_id: int = 128
    text_end_token_id: int = 130
    audio_eos_token_id: int = 2048
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        # Older checkpoints stored a flag for an audio output projection that was never used.
        kwargs.pop("tie_audio_embeddings", None)
        self.semantic_codebook_factor = float(self.semantic_codebook_factor)
        self.text_loss_multiplier = 1.0 if self.text_loss_multiplier is None else float(self.text_loss_multiplier)
        self.audio_loss_multiplier = 1.0 if self.audio_loss_multiplier is None else float(self.audio_loss_multiplier)

        if isinstance(self.encoder, dict):
            self.encoder = ParakeetEncoderConfig(**self.encoder)
        elif self.encoder is None:
            self.encoder = ParakeetEncoderConfig(
                hidden_size=512,
                num_hidden_layers=17,
                num_attention_heads=8,
                intermediate_size=2048,
                num_mel_bins=128,
                conv_kernel_size=9,
                subsampling_factor=8,
                subsampling_conv_channels=256,
                dropout=0.1,
                attention_dropout=0.1,
                activation_dropout=0.1,
                max_position_embeddings=5000,
                layerdrop=0.0,
                scale_input=False,
            )

        if isinstance(self.lfm, dict):
            self.lfm = Lfm2Config(**self.lfm)
        elif self.lfm is None:
            self.lfm = Lfm2Config()

        if isinstance(self.depthformer, dict):
            self.depthformer = Lfm2AudioDepthConfig(**self.depthformer)
        elif self.depthformer is None:
            self.depthformer = Lfm2AudioDepthConfig()

        if self.codebook_weight not in {"log", "linear"}:
            raise ValueError("`codebook_weight` must be either 'log' or 'linear'.")
        if self.audio_vocab_size != self.audio_eos_token_id + 1:
            raise ValueError("`audio_vocab_size` must include all Mimi tokens and the end-of-audio token.")

        kwargs.setdefault("bos_token_id", 1)
        kwargs.setdefault("eos_token_id", self.lfm.eos_token_id)
        kwargs.setdefault("pad_token_id", 0)
        self.hidden_size = self.lfm.hidden_size
        self.vocab_size = self.lfm.vocab_size
        self.initializer_range = self.lfm.initializer_range
        super().__post_init__(**kwargs)
        if self.encoder._attn_implementation is None:
            self.encoder._attn_implementation = "eager"
        if self.depthformer._attn_implementation is None:
            self.depthformer._attn_implementation = "sdpa"

    @property
    def text_config(self) -> Lfm2Config:
        if not isinstance(self.lfm, Lfm2Config):
            raise ValueError("`lfm` was not initialized as an Lfm2Config.")
        return self.lfm

    @property
    def encoder_config(self) -> ParakeetEncoderConfig:
        if not isinstance(self.encoder, ParakeetEncoderConfig):
            raise ValueError("`encoder` was not initialized as a ParakeetEncoderConfig.")
        return self.encoder

    @property
    def depth_config(self) -> Lfm2AudioDepthConfig:
        if not isinstance(self.depthformer, Lfm2AudioDepthConfig):
            raise ValueError("`depthformer` was not initialized as an Lfm2AudioDepthConfig.")
        return self.depthformer


__all__ = [
    "Lfm2AudioConfig",
    "Lfm2AudioDepthConfig",
]
