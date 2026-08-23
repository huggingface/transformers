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
class Lfm2AudioPreprocessorConfig(PreTrainedConfig):
    r"""
    normalize (`str`, *optional*, defaults to `"per_feature"`):
        Feature-normalization strategy used by the released checkpoint.
    window_size (`float`, *optional*, defaults to 0.025):
        Analysis-window duration in seconds.
    window_stride (`float`, *optional*, defaults to 0.01):
        Distance between consecutive analysis windows in seconds.
    window (`str`, *optional*, defaults to `"hann"`):
        Window function used by the short-time Fourier transform.
    features (`int`, *optional*, defaults to 128):
        Number of log-mel bins.
    n_fft (`int`, *optional*, defaults to 512):
        Fourier-transform size.
    log (`bool`, *optional*, defaults to `True`):
        Whether to return logarithmic mel energies.
    frame_splicing (`int`, *optional*, defaults to 1):
        Number of adjacent frames to splice. Only 1 is supported by the native frontend.
    dither (`float`, *optional*, defaults to 1e-5):
        Dither value stored by the released checkpoint.
    pad_to (`int`, *optional*, defaults to 0):
        Frame multiple used for padding. A value of 0 disables extra padding.
    pad_value (`float`, *optional*, defaults to 0.0):
        Value used to pad waveforms and features.
    """

    model_type = "lfm2_audio_preprocessor"

    sample_rate: int = 16_000
    normalize: str = "per_feature"
    window_size: float = 0.025
    window_stride: float = 0.01
    window: str = "hann"
    features: int = 128
    n_fft: int = 512
    log: bool = True
    frame_splicing: int = 1
    dither: float = 1e-5
    pad_to: int = 0
    pad_value: float = 0.0


@auto_docstring(checkpoint="LiquidAI/LFM2.5-Audio-1.5B")
@strict
class Lfm2AudioEncoderConfig(PreTrainedConfig):
    r"""
    feat_in (`int`, *optional*, defaults to 128):
        Number of input log-mel features.
    feat_out (`int`, *optional*, defaults to -1):
        Optional encoder output size in the legacy NeMo config. A value of -1 keeps `hidden_size`.
    subsampling (`str`, *optional*, defaults to `"dw_striding"`):
        FastConformer subsampling implementation.
    subsampling_factor (`int`, *optional*, defaults to 8):
        Temporal reduction applied before the encoder layers.
    subsampling_conv_channels (`int`, *optional*, defaults to 256):
        Number of channels in the subsampling convolutions.
    causal_downsampling (`bool`, *optional*, defaults to `False`):
        Whether the legacy frontend uses causal downsampling.
    reduction (`str`, *optional*):
        Optional extra reduction stage. Native LFM2-Audio checkpoints do not use one.
    reduction_position (`int`, *optional*):
        Encoder layer at which an extra reduction stage would be applied.
    reduction_factor (`int`, *optional*, defaults to 1):
        Factor of the optional extra reduction stage.
    ff_expansion_factor (`int`, *optional*, defaults to 4):
        Expansion factor of each Conformer feed-forward block.
    self_attention_model (`str`, *optional*, defaults to `"rel_pos"`):
        Attention variant. LFM2-Audio uses relative-position attention.
    att_context_size (`list[int]`, *optional*):
        Optional left and right attention-context limits.
    xscaling (`bool`, *optional*, defaults to `False`):
        Whether to scale encoder inputs by the square root of `hidden_size`.
    untie_biases (`bool`, *optional*, defaults to `True`):
        Whether relative-position biases are independent in each encoder layer.
    pos_emb_max_len (`int`, *optional*, defaults to 5000):
        Maximum length of the relative-position embedding.
    conv_norm_type (`str`, *optional*, defaults to `"batch_norm"`):
        Normalization used by Conformer convolution modules.
    conv_context_size (`list[int]`, *optional*):
        Optional left and right convolution-context limits.
    dropout_pre_encoder (`float`, *optional*, defaults to 0.1):
        Dropout applied after subsampling.
    dropout_emb (`float`, *optional*, defaults to 0.0):
        Dropout applied to relative-position embeddings.
    dropout_att (`float`, *optional*, defaults to 0.1):
        Attention-probability dropout.
    """

    model_type = "lfm2_audio_encoder"

    feat_in: int = 128
    feat_out: int = -1
    num_hidden_layers: int = 17
    hidden_size: int = 512
    subsampling: str = "dw_striding"
    subsampling_factor: int = 8
    subsampling_conv_channels: int = 256
    causal_downsampling: bool = False
    reduction: str | None = None
    reduction_position: int | None = None
    reduction_factor: int = 1
    ff_expansion_factor: int = 4
    self_attention_model: str = "rel_pos"
    num_attention_heads: int = 8
    att_context_size: list[int] | None = None
    xscaling: bool = False
    untie_biases: bool = True
    pos_emb_max_len: int = 5000
    conv_kernel_size: int = 9
    conv_norm_type: str = "batch_norm"
    conv_context_size: list[int] | None = None
    dropout: float | int = 0.1
    dropout_pre_encoder: float | int = 0.1
    dropout_emb: float | int = 0.0
    dropout_att: float | int = 0.1

    def to_parakeet_config(self) -> ParakeetEncoderConfig:
        """Translate the released NeMo-style config to Transformers' FastConformer config."""
        if self.subsampling != "dw_striding":
            raise ValueError(f"LFM2-Audio only supports `subsampling='dw_striding'`, got {self.subsampling!r}.")
        if self.self_attention_model != "rel_pos":
            raise ValueError(
                f"LFM2-Audio only supports relative-position attention, got {self.self_attention_model!r}."
            )
        if self.conv_norm_type != "batch_norm":
            raise ValueError(f"LFM2-Audio only supports batch-norm convolutions, got {self.conv_norm_type!r}.")
        if self.reduction is not None or self.reduction_factor != 1:
            raise ValueError("LFM2-Audio checkpoints with an additional encoder reduction stage are not supported.")

        return ParakeetEncoderConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.hidden_size * self.ff_expansion_factor,
            hidden_act="silu",
            attention_bias=True,
            convolution_bias=True,
            conv_kernel_size=self.conv_kernel_size,
            subsampling_factor=self.subsampling_factor,
            subsampling_conv_channels=self.subsampling_conv_channels,
            num_mel_bins=self.feat_in,
            subsampling_conv_kernel_size=3,
            subsampling_conv_stride=2,
            dropout=self.dropout_pre_encoder,
            dropout_positions=self.dropout_emb,
            layerdrop=0.0,
            activation_dropout=self.dropout,
            attention_dropout=self.dropout_att,
            max_position_embeddings=self.pos_emb_max_len,
            scale_input=self.xscaling,
        )


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
    tie_audio_embeddings (`bool`, *optional*, defaults to `False`):
        Whether to tie the input and output weights of the audio embedding table.
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
    preprocessor (`dict` or [`Lfm2AudioPreprocessorConfig`], *optional*):
        Configuration of the log-mel audio frontend.
    encoder (`dict`, [`Lfm2AudioEncoderConfig`] or [`ParakeetEncoderConfig`], *optional*):
        Configuration of the FastConformer audio encoder. The legacy NeMo-style configuration in the released
        checkpoint is translated to [`ParakeetEncoderConfig`] automatically.
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
        "preprocessor": Lfm2AudioPreprocessorConfig,
        "encoder": ParakeetEncoderConfig,
        "lfm": Lfm2Config,
        "depthformer": Lfm2AudioDepthConfig,
    }

    codebooks: int = 8
    tie_audio_embeddings: bool = False
    semantic_codebook_factor: float | int = 100.0
    codebook_weight: str = "log"
    text_loss_multiplier: float | int | None = 1.0
    audio_loss_multiplier: float | int | None = 1.0
    interleaved_n_text: int = 6
    interleaved_n_audio: int = 12
    preprocessor: dict | PreTrainedConfig | None = None
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
        self.semantic_codebook_factor = float(self.semantic_codebook_factor)
        self.text_loss_multiplier = 1.0 if self.text_loss_multiplier is None else float(self.text_loss_multiplier)
        self.audio_loss_multiplier = 1.0 if self.audio_loss_multiplier is None else float(self.audio_loss_multiplier)

        if isinstance(self.preprocessor, dict):
            self.preprocessor = Lfm2AudioPreprocessorConfig(**self.preprocessor)
        elif self.preprocessor is None:
            self.preprocessor = Lfm2AudioPreprocessorConfig()

        if isinstance(self.encoder, dict):
            if self.encoder.get("model_type") == "parakeet_encoder":
                self.encoder = ParakeetEncoderConfig(**self.encoder)
            else:
                encoder = dict(self.encoder)
                for legacy_name, canonical_name in {
                    "n_layers": "num_hidden_layers",
                    "d_model": "hidden_size",
                    "n_heads": "num_attention_heads",
                }.items():
                    if legacy_name in encoder:
                        encoder.setdefault(canonical_name, encoder.pop(legacy_name))
                self.encoder = Lfm2AudioEncoderConfig(**encoder).to_parakeet_config()
        elif isinstance(self.encoder, Lfm2AudioEncoderConfig):
            self.encoder = self.encoder.to_parakeet_config()
        elif self.encoder is None:
            self.encoder = Lfm2AudioEncoderConfig().to_parakeet_config()

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
    def preprocessor_config(self) -> Lfm2AudioPreprocessorConfig:
        if not isinstance(self.preprocessor, Lfm2AudioPreprocessorConfig):
            raise ValueError("`preprocessor` was not initialized as an Lfm2AudioPreprocessorConfig.")
        return self.preprocessor

    @property
    def depth_config(self) -> Lfm2AudioDepthConfig:
        if not isinstance(self.depthformer, Lfm2AudioDepthConfig):
            raise ValueError("`depthformer` was not initialized as an Lfm2AudioDepthConfig.")
        return self.depthformer


__all__ = [
    "Lfm2AudioConfig",
    "Lfm2AudioDepthConfig",
    "Lfm2AudioEncoderConfig",
    "Lfm2AudioPreprocessorConfig",
]
