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

import numpy as np
import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...feature_extraction_utils import BatchFeature
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
from ..parakeet.configuration_parakeet import ParakeetEncoderConfig, ParakeetTDTConfig
from ..parakeet.feature_extraction_parakeet import ParakeetFeatureExtractor
from ..parakeet.modeling_parakeet import (
    ParakeetEncoder,
    ParakeetEncoderAttention,
    ParakeetEncoderBlock,
    ParakeetEncoderFeedForward,
    ParakeetEncoderModelOutput,
    ParakeetEncoderRelPositionalEncoding,
    ParakeetEncoderSubsamplingConv2D,
    ParakeetForTDT,
    ParakeetPreTrainedModel,
    ParakeetTDTDecoder,
    ParakeetTDTJointNetwork,
    ParakeetTDTOutput,
)
from .generation_nemotron_asr import NemotronAsrGenerationMixin, NemotronAsrTDTDecoderCache


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


@auto_docstring(checkpoint="nvidia/nemotron-speech-streaming-en-0.6b")
@strict
class NemotronAsrConfig(ParakeetTDTConfig):
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
        super().__post_init__(**kwargs)


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
class NemotronAsrEncoderModelOutput(ParakeetEncoderModelOutput):
    r"""
    attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
        Mask to avoid performing attention on padding token indices after sequence compression. Returned because the
        sequence length may differ from the input sequence length. Mask values selected in `[0, 1]`:

        - 1 for tokens that are **not masked**,
        - 0 for tokens that are **masked**.
    cache_last_channel (`torch.Tensor` of shape `(num_layers, batch, left_ctx, hidden_size)`, *optional*):
        Updated attention cache from the encoder (sliding KV window). Pass to the next chunk call.
    cache_last_time (`torch.Tensor` of shape `(num_layers, batch, hidden_size, conv_left_ctx)`, *optional*):
        Updated convolution cache from the encoder. Pass to the next chunk call.
    cache_last_channel_len (`torch.Tensor` of shape `(batch,)`, *optional*):
        Number of valid frames currently stored in `cache_last_channel`.
    """

    cache_last_channel: torch.Tensor | None = None
    cache_last_time: torch.Tensor | None = None
    cache_last_channel_len: torch.Tensor | None = None


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
    def __init__(self, config: NemotronAsrEncoderConfig, module_config=None):
        super().__init__(config, module_config)
        channels = config.hidden_size
        kernel_size = config.conv_kernel_size

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

        # Symmetric padding is the default; override depthwise conv if asymmetric.
        sym = (kernel_size - 1) // 2
        if self._conv_left != sym or self._conv_right != sym:
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
        cache_last_time: torch.Tensor | None = None,
    ):
        # Override the parent forward to support asymmetric (causal/custom) conv padding
        # and LayerNorm (channel-last layout), plus an optional time-domain cache for streaming.
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

        # Asymmetric / causal padding: optionally use cache from previous chunk on the left.
        sym = (self.depthwise_conv.kernel_size[0] - 1) // 2
        new_cache = None
        if self._conv_left != sym or self._conv_right != sym:
            if cache_last_time is not None:
                # cache_last_time: (B, C, _conv_left)
                padded = torch.cat([cache_last_time, hidden_states], dim=2)
                # Sliding-window update: keep the last `_conv_left` frames of [cache | current]
                new_cache = padded[:, :, -self._conv_left :]
            else:
                padded = nn.functional.pad(hidden_states, (self._conv_left, 0))
            if self._conv_right > 0:
                padded = nn.functional.pad(padded, (0, self._conv_right))
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

        if new_cache is not None:
            return hidden_states, new_cache
        return hidden_states


class NemotronAsrEncoderAttention(ParakeetEncoderAttention):
    """Multi-head attention with relative positional encoding. See section 3.3 of https://huggingface.co/papers/1901.02860."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        cache_last_channel: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        batch_size, seq_length = input_shape
        hidden_shape = (batch_size, seq_length, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Streaming: prepend cached K/V from previous chunks and update the sliding window cache.
        new_cache = None
        if cache_last_channel is not None:
            cache_len = cache_last_channel.shape[1]
            cache_shape = (batch_size, cache_len, -1, self.head_dim)
            k_cache = self.k_proj(cache_last_channel).view(cache_shape).transpose(1, 2)
            v_cache = self.v_proj(cache_last_channel).view(cache_shape).transpose(1, 2)
            key_states = torch.cat([k_cache, key_states], dim=2)
            value_states = torch.cat([v_cache, value_states], dim=2)
            new_cache = torch.cat([cache_last_channel, hidden_states], dim=1)[:, -cache_len:]

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
        return attn_output, attn_weights, new_cache


class NemotronAsrEncoderSubsamplingConv2D(ParakeetEncoderSubsamplingConv2D): ...


class NemotronAsrEncoderBlock(ParakeetEncoderBlock):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: torch.Tensor | None = None,
        cache_last_channel: torch.Tensor | None = None,
        cache_last_time: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        residual = hidden_states
        hidden_states = self.feed_forward1(self.norm_feed_forward1(hidden_states))
        hidden_states = residual + 0.5 * hidden_states  # the conformer architecture uses a factor of 0.5

        normalized_hidden_states = self.norm_self_att(hidden_states)
        attn_output, _, new_cache_channel = self.self_attn(
            hidden_states=normalized_hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            cache_last_channel=cache_last_channel,
            **kwargs,
        )
        hidden_states = hidden_states + attn_output

        conv_result = self.conv(
            self.norm_conv(hidden_states), attention_mask=attention_mask, cache_last_time=cache_last_time
        )
        if isinstance(conv_result, tuple):
            conv_output, new_cache_time = conv_result
        else:
            conv_output, new_cache_time = conv_result, None
        hidden_states = hidden_states + conv_output

        ff2_output = self.feed_forward2(self.norm_feed_forward2(hidden_states))
        hidden_states = hidden_states + 0.5 * ff2_output  # the conformer architecture uses a factor of 0.5

        hidden_states = self.norm_out(hidden_states)

        return hidden_states, new_cache_channel, new_cache_time


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
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        r"""
        output_attention_mask (`bool`, *optional*, defaults to `True`):
            Whether to return the output attention mask. Only effective when `attention_mask` is provided.

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

        inputs_embeds = self.subsampling(input_features, attention_mask)
        inputs_embeds *= self.input_scale
        position_embeddings = self.encode_positions(inputs_embeds)

        inputs_embeds = nn.functional.dropout(inputs_embeds, p=self.dropout, training=self.training)
        position_embeddings = nn.functional.dropout(
            position_embeddings, p=self.dropout_positions, training=self.training
        )

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        output_mask = None
        if attention_mask is not None:
            output_mask = self._get_output_attention_mask(attention_mask, target_length=inputs_embeds.shape[1])

        left_ctx, right_ctx = self._resolve_att_context_size()
        chunk_aware_bidirectional_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=output_mask,
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
                hidden_states, _, _ = encoder_layer(
                    hidden_states,
                    attention_mask=chunk_aware_bidirectional_mask,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

        return NemotronAsrEncoderModelOutput(
            last_hidden_state=hidden_states,
            attention_mask=output_mask.int() if output_mask is not None and output_attention_mask else None,
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

    def get_initial_cache_state(
        self,
        batch_size: int = 1,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Returns zeroed cache tensors to start a streaming session.

        Returns a dict with keys `cache_last_channel`, `cache_last_time`, `cache_last_channel_len`
        that can be passed directly as `**cache` to `forward()`.
        """
        ctx = self._resolve_att_context_size()
        if ctx is None:
            raise ValueError("get_initial_cache_state() is only valid for cache-aware (streaming) models.")
        left_ctx = ctx[0]
        conv_ctx = self.config.conv_context_size
        if conv_ctx is None:
            conv_left = (self.config.conv_kernel_size - 1) // 2
        elif conv_ctx == "causal":
            conv_left = self.config.conv_kernel_size - 1
        else:
            conv_left = conv_ctx[0]
        n = self.config.num_hidden_layers
        d = self.config.hidden_size
        return {
            "cache_last_channel": torch.zeros(n, batch_size, left_ctx, d, dtype=dtype, device=device),
            "cache_last_time": torch.zeros(n, batch_size, d, conv_left, dtype=dtype, device=device),
            "cache_last_channel_len": torch.zeros(batch_size, dtype=torch.long, device=device),
        }


class NemotronAsrTDTDecoder(ParakeetTDTDecoder):
    """LSTM-based prediction network for RNN-T."""

    def __init__(self, config: NemotronAsrConfig):
        super().__init__(config)


class NemotronAsrTDTJointNetwork(ParakeetTDTJointNetwork):
    """Joint network that combines encoder and decoder outputs to predict tokens (no duration head)."""

    def __init__(self, config: NemotronAsrConfig):
        super().__init__(config)


@dataclass
class NemotronAsrTDTOutput(ParakeetTDTOutput):
    pass


@auto_docstring(
    custom_intro="""
    NemotronAsr Encoder with an RNN-T (Recurrent Neural Network Transducer) head.
    """
)
class NemotronAsrForRNNT(ParakeetForTDT, NemotronAsrPreTrainedModel, NemotronAsrGenerationMixin):
    config: NemotronAsrConfig

    def __init__(self, config: NemotronAsrConfig):
        super().__init__(config)
        self.encoder = AutoModel.from_config(config.encoder_config)
        self.encoder_projector = nn.Linear(config.encoder_config.hidden_size, config.joint_hidden_size)
        self.decoder = NemotronAsrTDTDecoder(config)
        self.joint = NemotronAsrTDTJointNetwork(config)
        self.max_symbols_per_step = config.max_symbols_per_step

        self.post_init()

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_features: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        decoder_cache: NemotronAsrTDTDecoderCache | None = None,
        use_decoder_cache: bool | None = None,
        encoder_outputs: NemotronAsrEncoderModelOutput | tuple[torch.FloatTensor] | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> NemotronAsrTDTOutput:
        r"""
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, 1)`, *optional*):
            Decoder input token ids for single-step inference.
        decoder_cache (`NemotronAsrTDTDecoderCache`, *optional*):
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
            decoder_cache = NemotronAsrTDTDecoderCache()

        decoder_hidden_states = self.decoder(decoder_input_ids, cache=decoder_cache)
        logits = self.joint(
            encoder_hidden_states=encoder_outputs.pooler_output[:, :, None, :],
            decoder_hidden_states=decoder_hidden_states[:, None, :, :],
        ).squeeze(2)

        if labels is not None:
            raise NotImplementedError(
                "RNN-T training loss is not yet implemented for NemotronAsrForRNNT. Inference (greedy decoding) works."
            )

        return NemotronAsrTDTOutput(
            loss=None,
            logits=logits,
            last_hidden_state=encoder_outputs.last_hidden_state,
            pooler_output=encoder_outputs.pooler_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            decoder_cache=decoder_cache,
        )

    def get_initial_streaming_state(
        self,
        batch_size: int = 1,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> dict:
        """
        Returns the streaming state dict to seed `streaming_step`.

        Holds:
        - encoder cache (`cache_last_channel`, `cache_last_time`, `cache_last_channel_len`)
        - decoder LSTM `(h, c)` state from BEFORE processing `last_token` (`dec_h`, `dec_c`)
        - decoder output `g` for the most recent committed token (`last_dec_g`) — reused on
          blank predictions so the LSTM is not re-stepped
        - last committed (non-blank) token (`last_token`: `(B, 1)` LongTensor, seeded with blank)

        The state mirrors NeMo's `(last_label, hidden, hidden_prime)` triple in
        `_greedy_decode_blank_as_pad_loop_frames`.

        Args:
            batch_size: Number of audio streams to decode in parallel.
        """
        enc_state = self.encoder.get_initial_cache_state(batch_size=batch_size, dtype=dtype, device=device)
        return {
            "cache_last_channel": enc_state["cache_last_channel"],
            "cache_last_time": enc_state["cache_last_time"],
            "cache_last_channel_len": enc_state["cache_last_channel_len"],
            # Decoder LSTM state from BEFORE last_token. None means "freshly initialized".
            "dec_h": None,
            "dec_c": None,
            # Cached decoder output `g` (after projector); reused on blank predictions.
            # Lazily allocated on the first non-blank emission.
            "last_dec_g": None,
            "last_token": torch.full((batch_size, 1), self.config.blank_token_id, dtype=torch.long, device=device),
        }

    def _decoder_pred_step(self, last_token, dec_h, dec_c):
        """
        Run a single LSTM + projector step manually so we can decide AFTER the joint+argmax
        whether to commit the resulting state. Mirrors NeMo's `RNNTDecoder.predict`.

        Args:
            last_token: `(B, 1)` LongTensor input.
            dec_h, dec_c: optional `(num_layers, B, hidden)` LSTM states from BEFORE last_token.
                          `None` means "no prior state" (initial).

        Returns:
            (g, new_h, new_c) where `g` is `(B, 1, decoder_hidden_size)` and `new_h/new_c` are
            the post-`last_token` LSTM states.
        """
        emb = self.decoder.embedding(last_token)
        if dec_h is None:
            lstm_output, (new_h, new_c) = self.decoder.lstm(emb)
        else:
            lstm_output, (new_h, new_c) = self.decoder.lstm(emb, (dec_h, dec_c))
        g = self.decoder.decoder_projector(lstm_output)
        return g, new_h, new_c

    @torch.no_grad()
    def streaming_step(
        self,
        inputs: dict,
        drop_extra_pre_encoded: int,
        state: dict,
        att_context_size: list | None = None,
    ) -> list[list[int]]:
        """
        Process one audio chunk and emit non-blank tokens, threading encoder + decoder state.

        Mirrors NeMo's `_greedy_decode_blank_as_pad_loop_frames`: encoder runs cache-aware on
        the chunk, then a per-frame inner loop emits up to `max_symbols_per_step` non-blank tokens
        before advancing. The decoder LSTM is run only when the prediction would advance state
        (non-blank emission); on blank predictions we reuse `last_dec_g` so the state is not
        corrupted by re-stepping the same token.

        Args:
            inputs: dict with `input_features` and `attention_mask` for this chunk
                    (typically yielded by `NemotronAsrCacheAwareStreamingBuffer`).
            drop_extra_pre_encoded: number of pre-encoded mel frames to drop after subsampling
                    (0 for the first chunk, otherwise from the buffer).
            state: dict from `get_initial_streaming_state`. Mutated in place.
            att_context_size: optional override of `[left, right]` attention window.

        Returns:
            list of length `batch_size`, each a list of token IDs (excluding blanks) emitted
            during this chunk.
        """
        encoder_outputs = self.encoder(
            input_features=inputs["input_features"],
            attention_mask=inputs.get("attention_mask"),
            cache_last_channel=state["cache_last_channel"],
            cache_last_time=state["cache_last_time"],
            cache_last_channel_len=state["cache_last_channel_len"],
            att_context_size=att_context_size,
            use_cache=True,
            drop_extra_pre_encoded=drop_extra_pre_encoded,
            output_attention_mask=True,
        )
        state["cache_last_channel"] = encoder_outputs.cache_last_channel
        state["cache_last_time"] = encoder_outputs.cache_last_time
        state["cache_last_channel_len"] = encoder_outputs.cache_last_channel_len

        encoded = encoder_outputs.last_hidden_state  # (B, T_chunk, enc_hidden)

        pooled = self.encoder_projector(encoded)  # (B, T_chunk, joint_hidden)
        chunk_attn = encoder_outputs.attention_mask
        if chunk_attn is not None:
            chunk_attn = chunk_attn.bool()

        batch_size, time_steps, _ = pooled.shape
        blank_id = self.config.blank_token_id
        last_token = state["last_token"]
        dec_h = state["dec_h"]
        dec_c = state["dec_c"]
        last_dec_g = state["last_dec_g"]
        emitted: list[list[int]] = [[] for _ in range(batch_size)]

        for time_idx in range(time_steps):
            f = pooled[:, time_idx : time_idx + 1, :]  # (B, 1, D)
            if chunk_attn is not None:
                frame_invalid = ~chunk_attn[:, time_idx]
            else:
                frame_invalid = torch.zeros(batch_size, dtype=torch.bool, device=pooled.device)

            symbols_added = 0
            advance_now = frame_invalid.clone()
            while not advance_now.all() and symbols_added < self.max_symbols_per_step:
                # Run LSTM with last_token + (dec_h, dec_c). Compute lookahead WITHOUT
                # committing the new state until we see whether the prediction is non-blank.
                g, new_h, new_c = self._decoder_pred_step(last_token, dec_h, dec_c)
                logits = (
                    self.joint(
                        encoder_hidden_states=f[:, :, None, :],
                        decoder_hidden_states=g[:, None, :, :],
                    )
                    .squeeze(2)
                    .squeeze(1)
                )
                tokens = logits.argmax(-1)  # (B,)

                is_blank = tokens == blank_id
                effective_blank = is_blank | advance_now

                # Per-element commit: only update last_token / state for non-blank emissions.
                # For blank predictions, last_token, dec_h, dec_c remain unchanged so the next
                # iter's pred_step recomputes the same g (or we'd just break out of the loop).
                if not effective_blank.all():
                    # Build a (B, 1) gather mask for which elements should update.
                    update = (~effective_blank).view(-1, 1, 1)  # for state shape
                    update_h = update.permute(2, 0, 1)  # (1, B, 1) for (L, B, H) state shape
                    if dec_h is None:
                        # No prior state means this is the very first emission for at least
                        # one batch element. Initialize with new state where committed,
                        # zeros elsewhere.
                        committed_h = torch.where(update_h, new_h, torch.zeros_like(new_h))
                        committed_c = torch.where(update_h, new_c, torch.zeros_like(new_c))
                    else:
                        committed_h = torch.where(update_h, new_h, dec_h)
                        committed_c = torch.where(update_h, new_c, dec_c)
                    dec_h, dec_c = committed_h, committed_c

                    # Emit + update last_token for non-blank items.
                    for b in range(batch_size):
                        if not effective_blank[b]:
                            emitted[b].append(int(tokens[b].item()))
                            last_token[b, 0] = tokens[b]

                    # Cache the latest g (used as fallback if last_token stays unchanged on
                    # subsequent blank predictions). Per-element commit on g too:
                    if last_dec_g is None:
                        last_dec_g = torch.where(update.expand_as(g), g, torch.zeros_like(g))
                    else:
                        last_dec_g = torch.where(update.expand_as(g), g, last_dec_g)

                advance_now = advance_now | is_blank
                symbols_added += 1

        state["last_token"] = last_token
        state["dec_h"] = dec_h
        state["dec_c"] = dec_c
        state["last_dec_g"] = last_dec_g
        return emitted


class NemotronAsrCacheAwareStreamingBuffer:
    """
    Streaming audio buffer for cache-aware NemotronAsr models (the **Nemotron Speech Streaming**
    family). Works with the RNN-T ([`NemotronAsrForRNNT`]) cache-aware checkpoints.

    Splits audio into correctly-sized chunks and yields processor inputs ready to pass directly
    to the encoder, handling the pre-encode cache, STFT lookahead, and mel frame trimming internally.

    Args:
        model ([`NemotronAsrForRNNT`]):
            The streaming model.
        processor ([`ParakeetProcessor`]):
            Matching processor (provides the feature extractor and tokenizer).
        att_context_size (`list[int]`, *optional*):
            `[left, right]` attention context to use. Defaults to the first (largest lookahead)
            entry in `model.encoder.config.att_context_size`.

    Example (RNN-T streaming with the public Nemotron Speech Streaming checkpoint):

    ```python
    import torch
    import soundfile as sf
    from transformers import AutoProcessor, NemotronAsrForRNNT, NemotronAsrCacheAwareStreamingBuffer

    model_id = "nvidia/nemotron-speech-streaming-en-0.6b"
    processor = AutoProcessor.from_pretrained(model_id)
    model = NemotronAsrForRNNT.from_pretrained(model_id).eval()

    audio, _ = sf.read("audio.wav", dtype="float32")  # resample to 16 kHz if needed

    buffer = NemotronAsrCacheAwareStreamingBuffer(model, processor, att_context_size=[70, 6])
    buffer.append_audio(audio)

    state = model.get_initial_streaming_state(batch_size=1, device=model.device, dtype=model.dtype)
    all_tokens = []
    for inputs, drop in buffer:
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            chunk_tokens = model.streaming_step(inputs, drop, state)
        all_tokens.extend(chunk_tokens[0])

    print(processor.batch_decode(torch.tensor([all_tokens]), skip_special_tokens=True)[0])
    ```
    """

    def __init__(self, model: "NemotronAsrForRNNT", processor, att_context_size=None, pad_and_drop_preencoded=True):
        encoder = model.encoder
        cfg_ctx = encoder.config.att_context_size
        if att_context_size is not None:
            ctx = list(att_context_size)
        elif isinstance(cfg_ctx[0], list):
            ctx = list(cfg_ctx[0])
        else:
            ctx = list(cfg_ctx)
        self.att_context_size = ctx
        right_ctx = ctx[1]

        hop = processor.feature_extractor.hop_length
        S = encoder.config.subsampling_factor

        self._processor = processor
        self._sr = processor.feature_extractor.sampling_rate
        self._n_fft_half = processor.feature_extractor.n_fft // 2

        pre_encode_cache_mel = S + 1
        self._pre_cache_mel = pre_encode_cache_mel
        self._pre_cache_samples = pre_encode_cache_mel * hop
        self._drop = 1 + (pre_encode_cache_mel - 1) // S
        # When pad_and_drop_preencoded is True (NeMo's reference inference mode), chunk 0
        # uses the same layout as subsequent chunks (steady-state size, zero pre-encode
        # pad, drop applied). When False, chunk 0 is smaller (no pad, no drop) which
        # mismatches what the encoder saw during training and degrades early-utterance WER.
        self.pad_and_drop_preencoded = pad_and_drop_preencoded
        if pad_and_drop_preencoded:
            self._first_chunk_mel = S * (right_ctx + 1)
        else:
            self._first_chunk_mel = 1 + S * right_ctx
        self._first_chunk_samples = self._first_chunk_mel * hop
        self._chunk_mel = S * (right_ctx + 1)
        self._chunk_samples = self._chunk_mel * hop
        self._target_mel_len = pre_encode_cache_mel + self._chunk_mel

        self._audio = None
        self._chunks = None

    def append_audio(self, audio):
        """
        Queue audio for streaming.

        Args:
            audio (`numpy.ndarray`):
                Float32 audio array sampled at `processor.feature_extractor.sampling_rate`.
        """
        self._audio = audio
        self._chunks = [audio[: self._first_chunk_samples]]
        rem = audio[self._first_chunk_samples :]
        i = 0
        while i < len(rem):
            self._chunks.append(rem[i : i + self._chunk_samples])
            i += self._chunk_samples

    def __iter__(self):
        """Yield `(processor_inputs, drop_extra_pre_encoded)` for each chunk."""
        if self._chunks is None:
            return
        past = np.zeros(self._pre_cache_samples, dtype=np.float32)
        for idx, chunk in enumerate(self._chunks):
            if idx == 0 and self.pad_and_drop_preencoded:
                # NeMo-style chunk 0: real audio occupies the steady-state chunk_size, and the
                # pre-encode cache is filled with zero mel frames (in normalized space). This
                # matches the encoder's training-time input layout so the first chunk's encoder
                # output is consistent with subsequent chunks.
                ns = self._first_chunk_samples
                lookahead = self._audio[ns : ns + self._n_fft_half]
                ext = np.concatenate([chunk, lookahead])
                inputs = self._processor([ext], return_tensors="pt", sampling_rate=self._sr)
                if inputs["input_features"].shape[1] > self._chunk_mel:
                    inputs["input_features"] = inputs["input_features"][:, : self._chunk_mel, :]
                    inputs["attention_mask"] = inputs["attention_mask"][:, : self._chunk_mel]
                feats = inputs["input_features"]
                mask = inputs["attention_mask"]
                pad_feats = torch.zeros(feats.shape[0], self._pre_cache_mel, feats.shape[-1], dtype=feats.dtype)
                pad_mask = torch.ones(mask.shape[0], self._pre_cache_mel, dtype=mask.dtype)
                inputs["input_features"] = torch.cat([pad_feats, feats], dim=1)
                inputs["attention_mask"] = torch.cat([pad_mask, mask], dim=1)
                drop = self._drop
            elif idx == 0:
                inputs = self._processor([chunk], return_tensors="pt", sampling_rate=self._sr)
                if inputs["input_features"].shape[1] > self._first_chunk_mel:
                    inputs["input_features"] = inputs["input_features"][:, : self._first_chunk_mel, :]
                    inputs["attention_mask"] = inputs["attention_mask"][:, : self._first_chunk_mel]
                drop = 0
            else:
                ns = self._first_chunk_samples + idx * self._chunk_samples
                lookahead = self._audio[ns : ns + self._n_fft_half]
                ext = np.concatenate([past, chunk, lookahead])
                inputs = self._processor([ext], return_tensors="pt", sampling_rate=self._sr)
                if inputs["input_features"].shape[1] > self._target_mel_len:
                    inputs["input_features"] = inputs["input_features"][:, : self._target_mel_len, :]
                    inputs["attention_mask"] = inputs["attention_mask"][:, : self._target_mel_len]
                drop = self._drop
            past = chunk[-self._pre_cache_samples :]
            yield inputs, drop


__all__ = [
    "NemotronAsrConfig",
    "NemotronAsrEncoderConfig",
    "NemotronAsrFeatureExtractor",
    "NemotronAsrEncoderModelOutput",
    "NemotronAsrTDTOutput",
    "NemotronAsrCacheAwareStreamingBuffer",
    "NemotronAsrForRNNT",
    "NemotronAsrEncoder",
    "NemotronAsrPreTrainedModel",
]
