# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""PyTorch NemotronAsr model."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ...configuration_utils import PreTrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMode
from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    logging,
)
from ...utils.generic import maybe_autocast, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..auto import AutoModel
from ..fastspeech2_conformer.modeling_fastspeech2_conformer import FastSpeech2ConformerConvolutionModule
from ..llama.modeling_llama import eager_attention_forward
from ..parakeet.configuration_parakeet import ParakeetEncoderConfig, ParakeetRNNTConfig
from ..parakeet.feature_extraction_parakeet import ParakeetFeatureExtractor
from ..parakeet.modeling_parakeet import (
    ParakeetEncoder,
    ParakeetEncoderAttention,
    ParakeetEncoderBlock,
    ParakeetEncoderFeedForward,
    ParakeetEncoderModelOutput,
    ParakeetEncoderRelPositionalEncoding,
    ParakeetForRNNT,
    ParakeetPreTrainedModel,
    ParakeetRNNTDecoder,
    ParakeetRNNTJointNetwork,
    ParakeetRNNTOutput,
)
from .generation_nemotron_asr import NemotronAsrGenerationMixin, NemotronAsrRNNTDecoderCache


LOG_ZERO_GUARD_VALUE = 2**-24

logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="nvidia/nemotron-speech-streaming-en-0.6b")
@strict
class NemotronAsrEncoderConfig(ParakeetEncoderConfig):
    r"""
    convolution_bias (`bool`, *optional*, defaults to `True`):
        Whether to use bias in convolutions of the conformer's convolution module.
    conv_kernel_size (`int`, *optional*, defaults to 9):
        The kernel size of the convolution layers in the Conformer block.
    subsampling_factor (`int`, *optional*, defaults to 8):
        The factor by which the input sequence is subsampled.
    subsampling_conv_channels (`int`, *optional*, defaults to 256):
        The number of channels in the subsampling convolution layers.
    num_mel_bins (`int`, *optional*, defaults to 80):
        Number of mel features.
    subsampling_conv_kernel_size (`int`, *optional*, defaults to 3):
        The kernel size of the subsampling convolution layers.
    subsampling_conv_stride (`int`, *optional*, defaults to 2):
        The stride of the subsampling convolution layers.
    dropout_positions (`float`, *optional*, defaults to 0.0):
        The dropout ratio for the positions in the input sequence.
    scale_input (`bool`, *optional*, defaults to `True`):
        Whether to scale the input embeddings.
    att_context_size (`list[int]` or `list[list[int]]`, *optional*, defaults to `None`):
        Attention context window `[left, right]` (in subsampled encoder frames). `None` (or `[-1, -1]`)
        means full bidirectional context. A single pair like `[70, 13]` constrains attention to
        ±frames per-position (cache-aware models). A list of pairs `[[70, 13], [70, 0]]` enables
        multi-lookahead training; the first entry is the inference default.
    att_context_style (`str`, *optional*, defaults to `"regular"`):
        Attention context style. `"regular"` masks per-position with the chosen `att_context_size`.
        `"chunked_limited"` groups frames into fixed chunks (size `right + 1`) and masks at chunk
        boundaries — matches NeMo's cache-aware streaming semantics.
    conv_context_size (`str` or `list[int]`, *optional*, defaults to `None`):
        Padding for the depthwise Conformer convolution. `None` uses symmetric `[(k-1)//2, (k-1)//2]`.
        `"causal"` uses left-only `[k-1, 0]`. A `[left, right]` pair (with `left + right + 1 == conv_kernel_size`)
        applies custom asymmetric padding.
    causal_downsampling (`bool`, *optional*, defaults to `False`):
        Whether the input subsampling Conv2d uses causal (left-only) padding in the time dimension.
        Required for cache-aware checkpoints.
    conv_norm_type (`str`, *optional*, defaults to `"batch_norm"`):
        Normalization for the depthwise convolution in the Conformer block: `"batch_norm"` or `"layer_norm"`.

    Example:
    ```python
    >>> from transformers import NemotronAsrEncoder, NemotronAsrEncoderConfig

    >>> # Initializing a `NemotronAsrEncoder` configuration
    >>> configuration = NemotronAsrEncoderConfig()

    >>> # Initializing a model from the configuration
    >>> model = NemotronAsrEncoder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "nemotron_asr_encoder"
    keys_to_ignore_at_inference = ["past_key_values"]

    att_context_size: list | None = None
    att_context_style: str = "regular"
    conv_context_size: str | list | None = None
    causal_downsampling: bool = False
    conv_norm_type: str = "batch_norm"

    def __post_init__(self, **kwargs):
        self.num_key_value_heads = self.num_attention_heads
        if isinstance(self.conv_context_size, list):
            left, right = self.conv_context_size
            if left + right + 1 != self.conv_kernel_size:
                raise ValueError(
                    f"conv_context_size {self.conv_context_size} must satisfy "
                    f"left + right + 1 == conv_kernel_size ({self.conv_kernel_size})."
                )
        if self.att_context_style not in {"regular", "chunked_limited"}:
            raise ValueError(
                f"att_context_style must be 'regular' or 'chunked_limited', got {self.att_context_style!r}."
            )
        if self.conv_norm_type not in {"batch_norm", "layer_norm"}:
            raise ValueError(f"conv_norm_type must be 'batch_norm' or 'layer_norm', got {self.conv_norm_type!r}.")
        PreTrainedConfig.__post_init__(self, **kwargs)


class NemotronAsrRNNTConfig(ParakeetRNNTConfig):
    r"""
    This is the base NemotronAsr transducer configuration. A conventional RNN-T (RNN Transducer) joint network
    emits token logits only (so the joint head outputs just `vocab_size` logits), and during greedy decoding
    the encoder frame pointer advances by exactly one frame on each blank emission. The duration-aware
    [`NemotronAsrRNNTConfig`] extends this configuration with a `durations` field.

    decoder_hidden_size (`int`, *optional*, defaults to 640):
        Hidden size of the LSTM prediction network and joint network.
    num_decoder_layers (`int`, *optional*, defaults to 2):
        Number of LSTM layers in the prediction network.
    max_symbols_per_step (`int`, *optional*, defaults to 10):
        Maximum number of symbols to emit per encoder time step during greedy decoding.
    encoder_config (`Union[dict, NemotronAsrEncoderConfig]`, *optional*):
        The config object or dictionary of the encoder.
    blank_token_id (`int`, *optional*, defaults to 8192):
        Blank token id. Different from `pad_token_id` for RNN-T.

    Example:
    ```python
    >>> from transformers import NemotronAsrForRNNT, NemotronAsrRNNTConfig

    >>> # Initializing a NemotronAsr RNN-T configuration
    >>> configuration = NemotronAsrRNNTConfig()

    >>> # Initializing a model from the configuration
    >>> model = NemotronAsrForRNNT(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "nemotron_asr_rnnt"
    sub_configs = {"encoder_config": NemotronAsrEncoderConfig}

    def __post_init__(self, **kwargs):
        if isinstance(self.encoder_config, dict):
            self.encoder_config = NemotronAsrEncoderConfig(**self.encoder_config)
        elif self.encoder_config is None:
            self.encoder_config = NemotronAsrEncoderConfig()
        self.initializer_range = self.encoder_config.initializer_range
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="nvidia/nemotron-speech-streaming-en-0.6b")
@strict
class NemotronAsrConfig(NemotronAsrRNNTConfig):
    r"""
    decoder_hidden_size (`int`, *optional*, defaults to 640):
        Hidden size of the LSTM prediction network (NeMo's `pred_hidden`).
    joint_hidden_size (`int`, *optional*, defaults to 640):
        Hidden size of the joint network's encoder/decoder projections (NeMo's `joint_hidden`).
        Encoder and decoder outputs are projected to this size before being summed and activated.
    num_decoder_layers (`int`, *optional*, defaults to 2):
        Number of LSTM layers in the prediction network.
    hidden_act (`str`, *optional*, defaults to `"relu"`):
        Activation in the joint network.
    max_symbols_per_step (`int`, *optional*, defaults to 10):
        Maximum number of non-blank symbols emitted per encoder time step during greedy decoding.
    durations (`list[int]`, *optional*, defaults to `()`):
        Pinned to the empty tuple for RNN-T: no token durations are predicted, so the joint head outputs
        only `vocab_size` logits.
    encoder_config (`Union[dict, NemotronAsrEncoderConfig]`, *optional*):
        The config object or dictionary of the encoder.
    blank_token_id (`int`, *optional*, defaults to 1024):
        Blank token id. Different from `pad_token_id` for RNN-T.

    Example:
    ```python
    >>> from transformers import NemotronAsrForRNNT, NemotronAsrConfig

    >>> # Initializing a NemotronAsr RNN-T configuration
    >>> configuration = NemotronAsrConfig()

    >>> # Initializing a model from the configuration
    >>> model = NemotronAsrForRNNT(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "nemotron_asr"
    sub_configs = {"encoder_config": NemotronAsrEncoderConfig}

    vocab_size: int = 1025
    joint_hidden_size: int = 640
    durations: list[int] | tuple[int, ...] = ()
    pad_token_id: int = 0
    blank_token_id: int = 1024

    def __post_init__(self, **kwargs):
        if self.decoder_hidden_size != self.joint_hidden_size:
            raise ValueError(
                "NemotronAsrConfig currently requires decoder_hidden_size == joint_hidden_size "
                f"(got {self.decoder_hidden_size} and {self.joint_hidden_size}). All known NeMo "
                "RNNT checkpoints satisfy this; if you have a checkpoint where they differ, please "
                "open an issue."
            )
        # The decoder starts on the blank token at frame 0 (NeMo's blank_as_pad convention).
        kwargs.setdefault("decoder_start_token_id", self.blank_token_id)
        if isinstance(self.encoder_config, dict):
            self.encoder_config = NemotronAsrEncoderConfig(**self.encoder_config)
        elif self.encoder_config is None:
            self.encoder_config = NemotronAsrEncoderConfig()
        self.initializer_range = self.encoder_config.initializer_range
        PreTrainedConfig.__post_init__(self, **kwargs)


class NemotronAsrFeatureExtractor(ParakeetFeatureExtractor):
    def _torch_extract_fbank_features(self, waveform, device="cpu", center=True):
        # spectrogram
        window = torch.hann_window(self.win_length, periodic=False, device=device)
        stft = torch.stft(
            waveform,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
            pad_mode="constant",
            center=center,
        )
        # Let's math original implementation
        # magnitudes = torch.abs(stft) ** 2
        magnitudes = torch.view_as_real(stft)
        magnitudes = torch.sqrt(magnitudes.pow(2).sum(-1))
        magnitudes = magnitudes.pow(2)

        # log mel spectrogram
        mel_filters = self.mel_filters.to(device)
        mel_spec = mel_filters @ magnitudes
        mel_spec = torch.log(mel_spec + LOG_ZERO_GUARD_VALUE)

        # (batch_size, num_mel_filters, num_frames) -> (batch_size, num_frames, num_mel_filters)
        mel_spec = mel_spec.permute(0, 2, 1)

        return mel_spec

    def __call__(
        self,
        raw_speech: "np.ndarray | list[float] | list[np.ndarray] | list[list[float]]",
        truncation: bool = False,
        pad_to_multiple_of: int | None = None,
        return_tensors: "str | TensorType | None" = None,
        return_attention_mask: bool | None = None,
        padding: str | None = "longest",
        max_length: int | None = None,
        sampling_rate: int | None = None,
        device: str | None = "cpu",
        return_token_timestamps: bool | None = None,
        center: bool = True,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s). Implementation uses PyTorch for
        the STFT computation if available, otherwise a slower NumPy based one.

        Args:
            raw_speech (`np.ndarray`, `list[float]`, `list[np.ndarray]`, `list[list[float]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values. Must be mono channel audio, not
                stereo, i.e. single float per timestep.
            truncation (`bool`, *optional*, default to `True`):
                Activates truncation to cut input sequences longer than *max_length* to *max_length*.
            pad_to_multiple_of (`int`, *optional*, defaults to None):
                If set will pad the sequence to a multiple of the provided value.
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific feature_extractor's default.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors and allow automatic speech recognition
                pipeline.
            device (`str`, *optional*, defaults to `'cpu'`):
                Specifies the device for computation of the log-mel spectrogram of audio signals in the
                `_torch_extract_fbank_features` method. (e.g., "cpu", "cuda")
            return_token_timestamps (`bool`, *optional*, defaults to `None`):
                Deprecated. Use `return_attention_mask` instead from which the number of frames can be inferred.
            center (`bool`, *optional*, defaults to `True`):
                Whether to pad the audio on both sides so STFT frames are centered (`torch.stft(center=True)`). Use
                `True` for offline extraction and for the first chunk of a streaming session. Use `False` for
                subsequent streaming chunks: feeding `audio[hop * frame - n_fft // 2 : ...]` with `center=False`
                reproduces, frame-for-frame, the features that a single `center=True` pass over the whole utterance
                would have produced for those frames.
        """
        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self.__class__.__name__} was trained using a"
                    f" sampling rate of {self.sampling_rate}. Please make sure that the provided `raw_speech` input"
                    f" was sampled with {self.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning(
                f"It is strongly recommended to pass the `sampling_rate` argument to `{self.__class__.__name__}()`. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        # Convert to torch tensor
        if isinstance(raw_speech, np.ndarray):
            raw_speech = torch.tensor(raw_speech)
        elif isinstance(raw_speech, (list, tuple)) and isinstance(raw_speech[0], np.ndarray):
            raw_speech = [torch.tensor(speech) for speech in raw_speech]

        is_batched_torch = isinstance(raw_speech, torch.Tensor) and len(raw_speech.shape) > 1
        if is_batched_torch and len(raw_speech.shape) > 2:
            logger.warning(
                f"Only mono-channel audio is supported for input to {self.__class__.__name__}. "
                "We will take the mean of the channels to convert to mono."
            )
            raw_speech = raw_speech.mean(-1)

        is_batched_sequence = isinstance(raw_speech, (list, tuple))
        if is_batched_sequence:
            for speech in raw_speech:
                if len(speech.shape) > 1:
                    logger.warning(
                        f"Only mono-channel audio is supported for input to {self.__class__.__name__}. "
                        "We will take the mean of the channels to convert to mono."
                    )
                    speech = speech.mean(-1)

        if is_batched_torch or is_batched_sequence:
            raw_speech = [speech[:, None].to(torch.float32) for speech in raw_speech]
        else:
            raw_speech = [raw_speech[:, None].to(torch.float32)]

        audio_lengths = [len(speech) for speech in raw_speech]
        batched_speech = BatchFeature({"input_features": raw_speech, "audio_lengths": audio_lengths})

        padded_inputs = self.pad(
            batched_speech,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )
        input_features = padded_inputs.input_features.squeeze(-1)

        # preemphasis
        if self.preemphasis is not None:
            timemask = torch.arange(input_features.shape[1], device=input_features.device).unsqueeze(
                0
            ) < padded_inputs.audio_lengths.unsqueeze(1)
            input_features = torch.cat(
                [input_features[:, :1], input_features[:, 1:] - self.preemphasis * input_features[:, :-1]], dim=1
            )
            input_features = input_features.masked_fill(~timemask, 0.0)

        input_features = self._torch_extract_fbank_features(input_features, device, center=center)
        if center:
            # `center=True` pads `n_fft // 2` on each side, so the number of valid frames is `floor(L / hop)`.
            features_lengths = torch.floor_divide(
                padded_inputs.audio_lengths + self.n_fft // 2 * 2 - self.n_fft, self.hop_length
            )
        else:
            # `center=False` does no padding: `floor((L - n_fft) / hop) + 1` frames.
            features_lengths = torch.floor_divide(padded_inputs.audio_lengths - self.n_fft, self.hop_length) + 1
        attention_mask = torch.arange(input_features.shape[1], device=device)[None, :] < features_lengths[:, None]

        # NemotronAsr never normalizes the mel features (the NeMo checkpoint uses `normalize="NA"`);
        # we only zero out the padded frames.
        input_features *= attention_mask.unsqueeze(-1)

        return BatchFeature(
            data={
                "input_features": input_features,
                "attention_mask": attention_mask,
            },
            tensor_type=return_tensors,
        )


@auto_docstring(
    custom_intro="""
    Extends [`ParakeetEncoderModelOutput`] with optional streaming caches. Caches are only populated for
    cache-aware models when `use_cache=True`.
    """
)
@dataclass
@dataclass
class NemotronAsrEncoderModelOutput(BaseModelOutputWithPooling):
    r"""
    attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
        Mask to avoid performing attention on padding token indices after sequence compression. Returned because the
        sequence length may differ from the input sequence length. Mask values selected in `[0, 1]`:

        - 1 for tokens that are **not masked**,
        - 0 for tokens that are **masked**.
    past_key_values (`Cache`, *optional*):
        Updated attention K/V sliding-window cache from the encoder. Pass to the next chunk call.
    """

    attention_mask: torch.Tensor | None = None

    past_key_values: Cache | None = None



class NemotronAsrEncoderRelPositionalEncoding(ParakeetEncoderRelPositionalEncoding):
    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor, context_length: int | None = None):
        # `context_length` overrides hidden_states.shape[1] for streaming (cache + chunk).
        seq_length = context_length if context_length is not None else hidden_states.shape[1]
        if seq_length > self.max_position_embeddings:
            raise ValueError(
                f"Sequence Length: {seq_length} has to be less or equal than "
                f"config.max_position_embeddings {self.max_position_embeddings}."
            )
        position_ids = torch.arange(seq_length - 1, -seq_length, -1, device=hidden_states.device)
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(hidden_states.shape[0], -1, 1).to(hidden_states.device)
        )
        position_ids_expanded = position_ids[None, None, :].float()

        device_type = (
            hidden_states.device.type
            if isinstance(hidden_states.device.type, str) and hidden_states.device.type != "mps"
            else "cpu"
        )
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            sin = freqs.sin()
            cos = freqs.cos()
            # interleave sin and cos
            pos_embed = torch.stack([sin, cos], dim=-1)
            pos_embed = pos_embed.reshape(*pos_embed.shape[:-2], -1)

        return pos_embed.to(dtype=hidden_states.dtype)


class NemotronAsrEncoderFeedForward(ParakeetEncoderFeedForward):
    pass


class NemotronAsrEncoderConvolutionModule(FastSpeech2ConformerConvolutionModule):
    def __init__(self, config: NemotronAsrEncoderConfig, module_config=None, layer_idx: int | None = None):
        """
        Args:
            config (NemotronAsrEncoderConfig): Configuration for the model.
            module_config (dict): Configuration for the module (e.g., encoder or decoder).
            layer_idx (int): Index of the conformer block; used to build a unique conv padding cache key.
        """
        super().__init__(config, module_config)
        kernel_size = config.conv_kernel_size
        channels = config.hidden_size

        # Replace BatchNorm with LayerNorm for cache-aware checkpoints.
        if config.conv_norm_type == "layer_norm":
            self.norm = nn.LayerNorm(channels)

        # Resolve depthwise conv left/right padding.
        ctx = config.conv_context_size
        if ctx is None:
            self._conv_left = (kernel_size - 1) // 2
            self._conv_right = (kernel_size - 1) // 2
        elif ctx == "causal":
            self._conv_left = kernel_size - 1
            self._conv_right = 0
        else:  # explicit [left, right]
            self._conv_left, self._conv_right = ctx

        # A purely causal depthwise conv (left = kernel - 1, right = 0) supports streaming through the
        # unified `NemotronAsrEncoderCausalConvPaddingCache`. Other (symmetric / asymmetric-with-right)
        # configs keep a plain `Conv1d` with manual padding and are not streaming-capable.
        sym = (kernel_size - 1) // 2
        self._is_causal_streaming = self._conv_left == kernel_size - 1 and self._conv_right == 0
        if self._is_causal_streaming:
            self.depthwise_conv = NemotronAsrEncoderCausalConv1d(
                channels,
                channels,
                kernel_size,
                cache_key=f"conv.{layer_idx}",
                stride=1,
                groups=channels,
                bias=config.convolution_bias,
            )
        elif self._conv_left != sym or self._conv_right != sym:
            self.depthwise_conv = nn.Conv1d(
                channels,
                channels,
                kernel_size,
                stride=1,
                padding=0,
                groups=channels,
                bias=config.convolution_bias,
            )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        padding_cache: "NemotronAsrEncoderCausalConvPaddingCache | None" = None,
    ):
        """
        Compute convolution module.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch, time, channels)`): Input tensor.
            attention_mask (`torch.Tensor` of shape `(batch, 1, time, time)`): Attention mask.
            padding_cache (`NemotronAsrEncoderCausalConvPaddingCache`, *optional*): Streaming conv cache.

        Returns:
            `torch.Tensor`: Output tensor of shape `(batch, time, channels)`.

        """
        # Override the parent forward to support asymmetric (causal/custom) conv padding
        # and LayerNorm (channel-last layout), plus an optional streaming padding cache.
        # exchange the temporal dimension and the feature dimension
        hidden_states = hidden_states.transpose(1, 2)  # (B, C, T)

        # GLU mechanism
        hidden_states = self.pointwise_conv1(hidden_states)
        hidden_states = nn.functional.glu(hidden_states, dim=1)

        # Apply padding mask before convolution. The mask may be 3D (B, T_q, T_k) for offline or
        # streaming use; reduce over the key dim to get a per-query padding indicator.
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                all_masked_rows = torch.all(~attention_mask, dim=-1)
            else:
                all_masked_rows = torch.all(attention_mask == 0.0, dim=-1)
            hidden_states = hidden_states.masked_fill(all_masked_rows, 0.0)

        if self._is_causal_streaming:
            # Causal depthwise conv: left context from `padding_cache` when streaming, else left-padded.
            hidden_states = self.depthwise_conv(hidden_states, padding_cache=padding_cache)
        else:
            sym = (self.depthwise_conv.kernel_size[0] - 1) // 2
            if self._conv_left != sym or self._conv_right != sym:
                padded = nn.functional.pad(hidden_states, (self._conv_left, self._conv_right))
                hidden_states = self.depthwise_conv(padded)
            else:
                hidden_states = self.depthwise_conv(hidden_states)

        # Norm: BatchNorm1d expects (B,C,T); LayerNorm expects (B,T,C).
        if isinstance(self.norm, nn.LayerNorm):
            hidden_states = hidden_states.transpose(1, 2)
            hidden_states = self.norm(hidden_states)
            hidden_states = hidden_states.transpose(1, 2)
        else:
            hidden_states = self.norm(hidden_states)

        hidden_states = self.activation(hidden_states)
        hidden_states = self.pointwise_conv2(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)  # (B, T, C)

        return hidden_states


class NemotronAsrEncoderAttention(ParakeetEncoderAttention):
    """Multi-head attention with relative positional encoding. See section 3.3 of https://huggingface.co/papers/1901.02860."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        batch_size, seq_length = input_shape
        hidden_shape = (batch_size, seq_length, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Streaming: prepend cached K/V from previous chunks and update the order-preserving sliding window.
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        total_key_length = key_states.shape[2]

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        query_states_with_bias_u = query_states + self.bias_u.view(
            1, self.config.num_attention_heads, 1, self.head_dim
        )
        query_states_with_bias_v = query_states + self.bias_v.view(
            1, self.config.num_attention_heads, 1, self.head_dim
        )

        relative_key_states = self.relative_k_proj(position_embeddings)
        relative_key_states = relative_key_states.view(batch_size, -1, self.config.num_attention_heads, self.head_dim)

        # terms (b) and (d) — slice to total_key_length to cover cache + current chunk
        matrix_bd = query_states_with_bias_v @ relative_key_states.permute(0, 2, 3, 1)
        matrix_bd = self._rel_shift(matrix_bd)
        matrix_bd = matrix_bd[..., :total_key_length]
        matrix_bd = matrix_bd * self.scaling

        if attention_mask is not None:
            # here the original codebase uses -10000.0 rather than float("-inf") and then manual masked fill with 0.0s
            # see: https://github.com/NVIDIA-NeMo/NeMo/blob/8cfedd7203462cb251a914e700e5605444277561/nemo/collections/asr/parts/submodules/multi_head_attention.py#L320-L340
            # we rather went for a straight-forward approach with float("-inf")
            matrix_bd = matrix_bd.masked_fill_(attention_mask.logical_not(), float("-inf"))

        # will compute matrix_ac - terms (a) and (c) - and add matrix_bd
        attn_output, attn_weights = attention_interface(
            self,
            query=query_states_with_bias_u,
            key=key_states,
            value=value_states,
            attention_mask=matrix_bd,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class NemotronAsrEncoderCausalConvCacheLayer:
    """
    Per-convolution streaming cache holding the left time-context of one causal conv.

    Works for **both** `Conv1d` (`(B, C, T)`) and `Conv2d` (`(B, C, T, F)`): the time axis is `dim=2`
    in both layouts, so the cache always slices and concatenates along `dim=2`. For the `Conv2d` case
    the (non-streamed) frequency axis is padded *before* the cache is consulted, so the cached tensor
    keeps a constant frequency size across chunks.

    The cache carries `conv_module.left_pad` frames between chunks, which for a causal conv is
    `effective_kernel_size - stride` (`1` for the stride-2 kernel-3 subsampling convs, `8` for the
    stride-1 kernel-9 conformer depthwise conv). The very first chunk instead needs the full offline
    left padding `conv_module.left_pad_init` (`effective_kernel_size - 1`), which only differs from the
    carry for strided convs; the extra `stride - 1` leading zeros reproduce the offline causal padding.
    """

    def __init__(self):
        self.cache: torch.Tensor | None = None
        self.is_initialized: bool = False

    def lazy_initialization(self, hidden_states, conv_module):
        self.left_pad = conv_module.left_pad
        cache_shape = list(hidden_states.shape)
        cache_shape[2] = conv_module.left_pad_init  # time axis (offline left padding for the first chunk)
        self.cache = torch.zeros(cache_shape, device=hidden_states.device, dtype=hidden_states.dtype)
        self.is_initialized = True

    def update(self, hidden_states, conv_module=None):
        if not self.is_initialized and conv_module is not None:
            self.lazy_initialization(hidden_states, conv_module)
        elif not self.is_initialized:
            raise ValueError(
                "NemotronAsrEncoderCausalConvCacheLayer is not initialized. "
                "Make sure to provide conv_module to the update method."
            )

        # The current cache is prepended to this chunk; the new cache keeps the last `left_pad` frames of
        # [previous_cache | hidden_states] along the time axis (== last `left_pad` frames the next chunk's
        # leftmost conv window needs from this chunk).
        prepend = self.cache
        combined = torch.cat([self.cache, hidden_states], dim=2)
        if self.left_pad > 0:
            self.cache = combined.narrow(2, combined.shape[2] - self.left_pad, self.left_pad).clone()
        else:
            self.cache = combined.narrow(2, 0, 0)
        return prepend


class NemotronAsrEncoderCausalConvPaddingCache:
    """
    Unified streaming padding cache for **all** causal convolutions in the encoder: the depthwise
    `Conv2d` layers of the subsampling **and** the depthwise `Conv1d` of every conformer block's
    convolution module. Each conv registers under a unique `cache_key`.

    This is the first transformers cache that backs both `Conv1d` and `Conv2d` layers. It mirrors
    [`VoxtralRealtimeConv1dPaddingCache`] but caches along the time axis (`dim=2`) so the same logic
    serves the 4D subsampling tensors and the 3D conformer-conv tensors.
    """

    def __init__(self):
        self.layers: dict[str, NemotronAsrEncoderCausalConvCacheLayer] = {}

    def update(self, hidden_states, cache_key, conv_module):
        if cache_key not in self.layers:
            self.layers[cache_key] = NemotronAsrEncoderCausalConvCacheLayer()

        padding_states = self.layers[cache_key].update(hidden_states, conv_module)
        return torch.cat([padding_states, hidden_states], dim=2)


class NemotronAsrEncoderCausalConv1d(nn.Conv1d):
    """
    Causal `Conv1d` (left-only padding) used as the depthwise conv of the conformer convolution module,
    with optional streaming support through [`NemotronAsrEncoderCausalConvPaddingCache`].
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        cache_key: str,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, groups=groups, bias=bias
        )
        self.cache_key = cache_key

    @cached_property
    def left_pad(self):
        effective_kernel_size = (self.kernel_size[0] - 1) * self.dilation[0] + 1
        return effective_kernel_size - self.stride[0]

    @cached_property
    def left_pad_init(self):
        effective_kernel_size = (self.kernel_size[0] - 1) * self.dilation[0] + 1
        return effective_kernel_size - 1

    def forward(self, x: torch.Tensor, padding_cache: NemotronAsrEncoderCausalConvPaddingCache | None = None):
        # x: (B, C, T). Conv1d caches along dim=2 (== last dim here).
        if padding_cache is not None:
            x = padding_cache.update(x, self.cache_key, self)
        else:
            x = nn.functional.pad(x, (self.left_pad, 0))
        return super().forward(x)


class NemotronAsrEncoderCausalConv2D(nn.Conv2d):
    """
    Causal `Conv2d` for the subsampling. The frequency axis is never streamed and is always padded the
    same way; the time axis is left-padded causally offline, or sourced from a
    [`NemotronAsrEncoderCausalConvPaddingCache`] when streaming.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        cache_key: str,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups
        )
        self.cache_key = cache_key

    @cached_property
    def left_pad(self):
        # Streaming cache size along time: kernel - stride. With stride 2 the conv output windows advance
        # by `stride`, so the boundary window needs only `kernel - stride` (== 1 for the k=3, s=2
        # subsampling convs) cached frames from the previous chunk; a larger cache would land windows on
        # the wrong stride parity.
        return self.kernel_size[0] - self.stride[0]

    @cached_property
    def left_pad_init(self):
        # First-chunk left padding (offline causal pad) along time: kernel - 1.
        return self.kernel_size[0] - 1

    @cached_property
    def time_pad(self):
        # Offline causal padding on the time axis: (left = kernel - 1, right = stride - 1).
        return (self.kernel_size[0] - 1, self.stride[0] - 1)

    @cached_property
    def freq_pad(self):
        # Frequency axis is not streamed: always padded (left = kernel - 1, right = stride - 1).
        return (self.kernel_size[1] - 1, self.stride[1] - 1)

    def forward(
        self,
        x: torch.Tensor,
        padding_cache: NemotronAsrEncoderCausalConvPaddingCache | None = None,
    ) -> torch.Tensor:
        # x: (B, C, T, F). Always pad the (non-streamed) frequency axis (the last dim).
        x = nn.functional.pad(x, (self.freq_pad[0], self.freq_pad[1]))
        if padding_cache is not None:
            # Streaming: the left time-context comes from the cache (size = left_pad = kernel - stride).
            x = padding_cache.update(x, self.cache_key, self)
        else:
            # Offline: causal padding on the time axis (dim 2).
            x = nn.functional.pad(x, (0, 0, self.time_pad[0], self.time_pad[1]))
        return super().forward(x)


class NemotronAsrEncoderSubsamplingConv2D(nn.Module):
    def __init__(self, config: NemotronAsrEncoderConfig):
        super().__init__()

        self.kernel_size = config.subsampling_conv_kernel_size
        self.stride = config.subsampling_conv_stride
        self.channels = config.subsampling_conv_channels
        self.padding = (self.kernel_size - 1) // 2
        self.num_layers = int(math.log2(config.subsampling_factor))

        self.layers = nn.ModuleList()
        self.layers.append(
            NemotronAsrEncoderCausalConv2D(
                1,
                self.channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=0,
                cache_key="subsampling.0",
            )
        )
        self.layers.append(nn.ReLU())
        for idx in range(self.num_layers - 1):
            # depthwise conv
            self.layers.append(
                NemotronAsrEncoderCausalConv2D(
                    self.channels,
                    self.channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    groups=self.channels,
                    cache_key=f"subsampling.{idx + 1}",
                )
            )
            # pointwise conv
            self.layers.append(nn.Conv2d(self.channels, self.channels, kernel_size=1))
            # activation
            self.layers.append(nn.ReLU())

        # Compute output freq length by simulating the conv chain with the actual padding applied.
        out_length = config.num_mel_bins
        total_pad = self._pad_left + self._pad_right
        for _ in range(self.num_layers):
            out_length = (out_length + total_pad - self.kernel_size) // self.stride + 1
        self.linear = nn.Linear(config.subsampling_conv_channels * out_length, config.hidden_size, bias=True)

    def _get_output_length(self, input_lengths: torch.Tensor, conv_layer: nn.Conv2d, streaming: bool = False):
        if not (hasattr(conv_layer, "stride") and conv_layer.stride != (1, 1)):
            return input_lengths

        kernel_size = conv_layer.kernel_size[0]
        stride = conv_layer.stride[0]
        if isinstance(conv_layer, NemotronAsrEncoderCausalConv2D):
            # Streaming consumes `left_pad` cached frames on the left and no right padding; offline uses
            # the full causal padding `(kernel - 1, stride - 1)` on the time axis.
            left, right = (conv_layer.left_pad, 0) if streaming else conv_layer.time_pad
        else:
            left = right = conv_layer.padding[0]
        return (input_lengths + left + right - kernel_size) // stride + 1

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor = None,
        padding_cache: NemotronAsrEncoderCausalConvPaddingCache | None = None,
    ):
        hidden_states = input_features.unsqueeze(1)
        current_lengths = attention_mask.sum(-1) if attention_mask is not None else None
        streaming = padding_cache is not None

        for layer in self.layers:
            if isinstance(layer, NemotronAsrEncoderCausalConv2D):
                hidden_states = layer(hidden_states, padding_cache=padding_cache)
            else:
                hidden_states = layer(hidden_states)

            # mask the hidden states
            if isinstance(layer, nn.Conv2d) and attention_mask is not None:
                current_lengths = self._get_output_length(current_lengths, layer, streaming=streaming)
                current_seq_length = hidden_states.shape[2]
                channel_mask = (
                    torch.arange(current_seq_length, device=attention_mask.device) < current_lengths[:, None]
                )
                hidden_states *= channel_mask[:, None, :, None]

        hidden_states = hidden_states.transpose(1, 2).reshape(hidden_states.shape[0], hidden_states.shape[2], -1)
        hidden_states = self.linear(hidden_states)

        return hidden_states


class NemotronAsrEncoderBlock(ParakeetEncoderBlock):
    def __init__(self, config: NemotronAsrEncoderConfig, layer_idx: int | None = None):
        super().__init__()
        self.gradient_checkpointing = False

        self.feed_forward1 = NemotronAsrEncoderFeedForward(config)
        self.self_attn = NemotronAsrEncoderAttention(config, layer_idx)
        self.conv = NemotronAsrEncoderConvolutionModule(config, layer_idx=layer_idx)
        self.feed_forward2 = NemotronAsrEncoderFeedForward(config)

        self.norm_feed_forward1 = nn.LayerNorm(config.hidden_size)
        self.norm_self_att = nn.LayerNorm(config.hidden_size)
        self.norm_conv = nn.LayerNorm(config.hidden_size)
        self.norm_feed_forward2 = nn.LayerNorm(config.hidden_size)
        self.norm_out = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        padding_cache: NemotronAsrEncoderCausalConvPaddingCache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.feed_forward1(self.norm_feed_forward1(hidden_states))
        hidden_states = residual + 0.5 * hidden_states  # the conformer architecture uses a factor of 0.5

        normalized_hidden_states = self.norm_self_att(hidden_states)
        attn_output, _ = self.self_attn(
            hidden_states=normalized_hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = hidden_states + attn_output

        conv_output = self.conv(
            self.norm_conv(hidden_states), attention_mask=attention_mask, padding_cache=padding_cache
        )
        hidden_states = hidden_states + conv_output

        ff2_output = self.feed_forward2(self.norm_feed_forward2(hidden_states))
        hidden_states = hidden_states + 0.5 * ff2_output  # the conformer architecture uses a factor of 0.5

        hidden_states = self.norm_out(hidden_states)

        return hidden_states


@auto_docstring
class NemotronAsrPreTrainedModel(ParakeetPreTrainedModel):
    config: NemotronAsrConfig

    def _get_subsampling_output_length(self, input_lengths: torch.Tensor):
        encoder_config = getattr(self.config, "encoder_config", self.config)

        kernel_size = encoder_config.subsampling_conv_kernel_size
        stride = encoder_config.subsampling_conv_stride
        num_layers = int(math.log2(encoder_config.subsampling_factor))

        if getattr(encoder_config, "causal_downsampling", False):
            # NeMo's CausalConv2D pads (left=kernel-1, right=stride-1) → total = kernel-1 + stride-1.
            all_paddings = (kernel_size - 1) + (stride - 1)
        else:
            # Symmetric same-padding: total = (kernel-1)//2 * 2.
            all_paddings = (kernel_size - 1) // 2 * 2
        add_pad = all_paddings - kernel_size
        lengths = input_lengths

        for _ in range(num_layers):
            lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + 1.0
            lengths = torch.floor(lengths)

        return lengths.to(dtype=torch.int)


def chunked_limited_mask_function(left_ctx: int, right_ctx: int) -> Callable:
    """
    Build the `chunked_limited` attention mask function used by NeMo cache-aware FastConformer models.

    From the `[left, right]` attention context, frames are grouped into fixed chunks of size
    `right + 1` by integer division of their position: `chunk_idx = position // (right + 1)`. A query
    attends to a key iff the key lies in the query's own chunk or in one of the `left // (right + 1)`
    chunks immediately to its left. Because membership is by chunk index, every frame in a chunk shares
    identical boundaries — a frame sees future frames only up to its chunk boundary. This is NOT a
    per-frame sliding window (which would let each frame peek a fixed number of frames into the next
    chunk).
    """
    chunk_size = right_ctx + 1
    left_context_chunks = left_ctx // chunk_size if left_ctx >= 0 else 10_000

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        q_chunk = torch.div(q_idx, chunk_size, rounding_mode="trunc")
        kv_chunk = torch.div(kv_idx, chunk_size, rounding_mode="trunc")
        chunk_diff = q_chunk - kv_chunk
        return (chunk_diff >= 0) & (chunk_diff <= left_context_chunks)

    return inner_mask


@auto_docstring(
    custom_intro="""
    The NemotronAsr Encoder model, based on the [Fast Conformer architecture](https://huggingface.co/papers/2305.05084).
    """
)
class NemotronAsrEncoder(ParakeetEncoder):
    @auto_docstring
    @merge_with_config_defaults
    @capture_outputs
    @can_return_tuple
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        output_attention_mask: bool = True,
        use_cache: bool | None = None,
        padding_cache: NemotronAsrEncoderCausalConvPaddingCache | None = None,
        att_context_size: list | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        r"""
        output_attention_mask (`bool`, *optional*, defaults to `True`):
            Whether to return the output attention mask. Only effective when `attention_mask` is provided.
        past_key_values (`Cache`, *optional*):
            Sliding-window K/V cache (`DynamicCache` built from `config.sliding_window`) for cache-aware
            streaming attention.
        padding_cache (`NemotronAsrEncoderCausalConvPaddingCache`, *optional*):
            Unified streaming cache backing the subsampling Conv2d layers and the conformer depthwise Conv1d.
        att_context_size (`list[int]`, *optional*):
            Override of the `[left, right]` attention context for this forward pass.

        Example:

        ```python
        >>> from transformers import AutoProcessor, NemotronAsrEncoder
        >>> from datasets import load_dataset, Audio

        >>> model_id = "nvidia/nemotron_asr-ctc-1.1b"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> encoder = NemotronAsrEncoder.from_pretrained(model_id)

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

        >>> inputs = processor(ds[0]["audio"]["array"])
        >>> encoder_outputs = encoder(**inputs)

        >>> print(encoder_outputs.last_hidden_state.shape)
        ```
        """
        # Lazily allocate the streaming caches when requested without one. The K/V cache is a standard
        # `DynamicCache` built from the config: `config.sliding_window` makes it a sliding-window cache of
        # order-preserving `DynamicSlidingWindowLayer`s (exactly what the relative-position attention needs).
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if use_cache and padding_cache is None:
            padding_cache = NemotronAsrEncoderCausalConvPaddingCache()

        inputs_embeds = self.subsampling(input_features, attention_mask, padding_cache=padding_cache)
        inputs_embeds *= self.input_scale

        seq_length = inputs_embeds.shape[1]
        # In streaming, attention also attends to the cached frames on the left (sliding window). The
        # rel-pos encoding spans the full key length = cached frames (capped at the window) + current chunk.
        if past_key_values is not None:
            kv_length, _ = past_key_values.get_mask_sizes(seq_length, 0)
            past_seen = past_key_values.get_seq_length()
        else:
            kv_length, past_seen = seq_length, 0
        position_embeddings = self.encode_positions(inputs_embeds, context_length=kv_length)

        inputs_embeds = nn.functional.dropout(inputs_embeds, p=self.dropout, training=self.training)
        position_embeddings = nn.functional.dropout(
            position_embeddings, p=self.dropout_positions, training=self.training
        )

        if position_ids is None:
            position_ids = (torch.arange(seq_length, device=inputs_embeds.device) + past_seen).unsqueeze(0)

        output_mask = None
        if attention_mask is not None:
            output_mask = self._get_output_attention_mask(attention_mask, target_length=seq_length)

        # The standard masking machinery handles the cache offset: `create_bidirectional_mask` reads
        # `q_offset = past_key_values.get_seq_length()` and `kv_length, kv_offset = get_mask_sizes(...)` from
        # the sliding-window cache and offsets the `chunked_limited` overlay with the absolute frame positions
        # accordingly. `past_key_values=None` (offline) is the full-sequence case.
        left_ctx, right_ctx = self._resolve_att_context_size(att_context_size)
        attention_mask_4d = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=output_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            and_mask_function=chunked_limited_mask_function(left_ctx, right_ctx),
        )

        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if not to_drop:
                hidden_states = encoder_layer(
                    hidden_states,
                    attention_mask=attention_mask_4d,
                    position_embeddings=position_embeddings,
                    past_key_values=past_key_values,
                    padding_cache=padding_cache,
                    **kwargs,
                )

        return NemotronAsrEncoderModelOutput(
            last_hidden_state=hidden_states,
            attention_mask=output_mask.int() if output_mask is not None and output_attention_mask else None,
            past_key_values=past_key_values,
        )

    def _resolve_att_context_size(self, att_context_size: list | None = None) -> list | None:
        """
        Resolve the effective `[left, right]` attention context for this forward pass.

        - If the model is offline (config.att_context_size is None) → returns None.
        - If `att_context_size` is provided by the caller → uses it (and warns once if the
          requested context is outside the model's trained set).
        - Otherwise → uses the first entry from config (the inference default).
        """
        configured = self.config.att_context_size
        if configured is None:
            return None
        if att_context_size is not None:
            if isinstance(configured[0], list) and att_context_size not in configured:
                logger.warning_once(
                    f"att_context_size {att_context_size} was not used during training "
                    f"(trained contexts: {configured}). The model may still produce reasonable "
                    f"output, but quality is not guaranteed."
                )
            return att_context_size
        if isinstance(configured[0], list):
            return configured[0]
        return configured


class NemotronAsrRNNTDecoder(ParakeetRNNTDecoder):
    def __init__(self, config: NemotronAsrRNNTConfig):
        super().__init__(config)


class NemotronAsrRNNTJointNetwork(ParakeetRNNTJointNetwork):
    def __init__(self, config: NemotronAsrRNNTConfig):
        super().__init__(config)


@dataclass
class NemotronAsrRNNTOutput(ParakeetRNNTOutput):
    pass


@auto_docstring(
    custom_intro="""
    NemotronAsr Encoder with an RNN-T (Recurrent Neural Network Transducer) head.
    """
)
class NemotronAsrForRNNT(ParakeetForRNNT, NemotronAsrPreTrainedModel, NemotronAsrGenerationMixin):
    config: NemotronAsrRNNTConfig

    def __init__(self, config: NemotronAsrRNNTConfig):
        super().__init__(config)
        self.encoder = AutoModel.from_config(config.encoder_config)
        self.encoder_projector = nn.Linear(config.encoder_config.hidden_size, config.decoder_hidden_size)
        self.decoder = NemotronAsrRNNTDecoder(config)
        self.joint = NemotronAsrRNNTJointNetwork(config)
        self.max_symbols_per_step = config.max_symbols_per_step  # used in generation

        self.post_init()

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_features: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        decoder_cache: NemotronAsrRNNTDecoderCache | None = None,
        use_decoder_cache: bool | None = None,
        encoder_outputs: NemotronAsrEncoderModelOutput | tuple[torch.FloatTensor] | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> NemotronAsrRNNTOutput:
        r"""
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, 1)`, *optional*):
            Decoder input token ids for single-step inference.
        decoder_cache (`NemotronAsrRNNTDecoderCache`, *optional*):
            Decoder LSTM cache. Reused on blank predictions to skip the LSTM step.
        use_decoder_cache (`bool`, *optional*):
            Whether to allocate and use a decoder cache when none is provided.
        encoder_outputs (`tuple(torch.FloatTensor)`, *optional*):
            Pre-computed encoder outputs (last_hidden_state, pooler_output, ...).

        Example:

        ```python
        >>> from transformers import AutoProcessor, NemotronAsrForRNNT
        >>> from datasets import load_dataset, Audio

        >>> model_id = "nvidia/nemotron-speech-streaming-en-0.6b"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> model = NemotronAsrForRNNT.from_pretrained(model_id)

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

        >>> inputs = processor(ds[0]["audio"]["array"])
        >>> outputs = model(**inputs)
        ```
        """
        if encoder_outputs is None:
            encoder_outputs = self.get_audio_features(
                input_features=input_features,
                attention_mask=attention_mask,
                **kwargs,
            )
        elif not isinstance(encoder_outputs, NemotronAsrEncoderModelOutput):
            encoder_outputs = NemotronAsrEncoderModelOutput(
                last_hidden_state=encoder_outputs[0] if len(encoder_outputs) > 0 else None,
                pooler_output=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                hidden_states=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                attentions=encoder_outputs[3] if len(encoder_outputs) > 3 else None,
                attention_mask=encoder_outputs[4] if len(encoder_outputs) > 4 else None,
            )

        if use_decoder_cache and decoder_cache is None:
            decoder_cache = NemotronAsrRNNTDecoderCache()

        decoder_hidden_states = self.decoder(decoder_input_ids, cache=decoder_cache)
        logits = self.joint(
            encoder_hidden_states=encoder_outputs.pooler_output[:, :, None, :],
            decoder_hidden_states=decoder_hidden_states[:, None, :, :],
        ).squeeze(2)

        if labels is not None:
            raise NotImplementedError(
                "RNN-T training loss is not yet implemented for NemotronAsrForRNNT. Inference (greedy decoding) works."
            )

        return NemotronAsrRNNTOutput(
            loss=None,
            logits=logits,
            last_hidden_state=encoder_outputs.last_hidden_state,
            pooler_output=encoder_outputs.pooler_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            decoder_cache=decoder_cache,
        )


__all__ = [
    "NemotronAsrConfig",
    "NemotronAsrEncoderConfig",
    "NemotronAsrFeatureExtractor",
    "NemotronAsrEncoderModelOutput",
    "NemotronAsrRNNTOutput",
    "NemotronAsrForRNNT",
    "NemotronAsrEncoder",
    "NemotronAsrPreTrainedModel",
]
