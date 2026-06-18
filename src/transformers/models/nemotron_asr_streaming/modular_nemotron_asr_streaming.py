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
from functools import cached_property

import numpy as np
import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ...audio_utils import AudioInput, make_list_of_audio
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import ProcessingKwargs, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import (
    TensorType,
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    is_torchdynamo_compiling,
    logging,
)
from ...utils.generic import maybe_autocast, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..fastspeech2_conformer.modeling_fastspeech2_conformer import FastSpeech2ConformerConvolutionModule
from ..llama.modeling_llama import eager_attention_forward
from ..parakeet.configuration_parakeet import ParakeetEncoderConfig, ParakeetRNNTConfig
from ..parakeet.feature_extraction_parakeet import ParakeetFeatureExtractor
from ..parakeet.modeling_parakeet import (
    ParakeetEncoder,
    ParakeetEncoderAttention,
    ParakeetEncoderBlock,
    ParakeetEncoderRelPositionalEncoding,
    ParakeetForRNNT,
    ParakeetPreTrainedModel,
    ParakeetRNNTDecoder,
    ParakeetRNNTJointNetwork,
    ParakeetRNNTOutput,
)
from ..parakeet.processing_parakeet import ParakeetProcessor
from ..voxtral_realtime.modeling_voxtral_realtime import (
    VoxtralRealtimeCausalConv1d,
    VoxtralRealtimeConv1dCacheLayer,
)
from .generation_nemotron_asr_streaming import (
    NemotronAsrStreamingGenerationMixin,
    NemotronAsrStreamingRNNTDecoderCache,
)


LOG_ZERO_GUARD_VALUE = 2**-24

logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="nvidia/nemotron-speech-streaming-en-0.6b")
@strict
class NemotronAsrStreamingEncoderConfig(ParakeetEncoderConfig):
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
    sliding_window (`int`, *optional*, defaults to 71):
        Size of the K/V attention sliding window (in subsampled encoder frames). It equals
        `left_context + 1` (the current frame plus the left context), so the left attention context is
        `sliding_window - 1` — the same across all supported lookaheads.
    default_num_lookahead_tokens (`int`, *optional*, defaults to 13):
        The right attention context (lookahead, in subsampled encoder frames) used when none is passed to the
        forward. The supported set the model was trained with lives on [`NemotronAsrStreamingProcessor`].

    Example:
    ```python
    >>> from transformers import NemotronAsrStreamingEncoder, NemotronAsrStreamingEncoderConfig

    >>> # Initializing a `NemotronAsrStreamingEncoder` configuration
    >>> configuration = NemotronAsrStreamingEncoderConfig()

    >>> # Initializing a model from the configuration
    >>> model = NemotronAsrStreamingEncoder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "nemotron_asr_streaming_encoder"
    keys_to_ignore_at_inference = ["past_key_values"]

    sliding_window: int = 71
    default_num_lookahead_tokens: int = 13

    def __post_init__(self, **kwargs):
        self.num_key_value_heads = self.num_attention_heads
        PreTrainedConfig.__post_init__(self, **kwargs)


@auto_docstring(checkpoint="nvidia/nemotron-speech-streaming-en-0.6b")
@strict
class NemotronAsrStreamingConfig(ParakeetRNNTConfig):
    r"""
    This is the NemotronAsrStreaming transducer configuration. The RNN-T (RNN Transducer) joint network emits token
    logits only (so the joint head outputs just `vocab_size` logits), and during greedy decoding the encoder
    frame pointer advances by exactly one frame on each blank emission.

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
    encoder_config (`Union[dict, NemotronAsrStreamingEncoderConfig]`, *optional*):
        The config object or dictionary of the encoder.
    blank_token_id (`int`, *optional*, defaults to 1024):
        Blank token id. Different from `pad_token_id` for RNN-T.

    Example:
    ```python
    >>> from transformers import NemotronAsrStreamingForRNNT, NemotronAsrStreamingConfig

    >>> # Initializing a NemotronAsrStreaming RNN-T configuration
    >>> configuration = NemotronAsrStreamingConfig()

    >>> # Initializing a model from the configuration
    >>> model = NemotronAsrStreamingForRNNT(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "nemotron_asr_streaming"
    sub_configs = {"encoder_config": NemotronAsrStreamingEncoderConfig}

    vocab_size: int = 1025
    joint_hidden_size: int = 640
    durations: list[int] | tuple[int, ...] = ()
    pad_token_id: int = 0
    blank_token_id: int = 1024

    def __post_init__(self, **kwargs):
        if self.decoder_hidden_size != self.joint_hidden_size:
            raise ValueError(
                "NemotronAsrStreamingConfig currently requires decoder_hidden_size == joint_hidden_size "
                f"(got {self.decoder_hidden_size} and {self.joint_hidden_size}). All known NeMo "
                "RNNT checkpoints satisfy this; if you have a checkpoint where they differ, please "
                "open an issue."
            )
        # The decoder starts on the blank token at frame 0 (NeMo's blank_as_pad convention).
        kwargs.setdefault("decoder_start_token_id", self.blank_token_id)
        if isinstance(self.encoder_config, dict):
            self.encoder_config = NemotronAsrStreamingEncoderConfig(**self.encoder_config)
        elif self.encoder_config is None:
            self.encoder_config = NemotronAsrStreamingEncoderConfig()
        self.initializer_range = self.encoder_config.initializer_range
        PreTrainedConfig.__post_init__(self, **kwargs)


class NemotronAsrStreamingProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "audio_kwargs": {
            "sampling_rate": 16000,
            "padding": "longest",
            "return_attention_mask": True,
            "subsampling_factor": 8,
        },
        "text_kwargs": {
            "padding": True,
            "padding_side": "right",
            "add_special_tokens": False,
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


# Default supported right attention contexts (lookaheads, in subsampled encoder frames) of the NeMo
# cache-aware streaming FastConformer checkpoint. The first entry is the default.
DEFAULT_NUM_LOOKAHEAD_TOKENS = [13, 6, 1, 0]


@auto_docstring
class NemotronAsrStreamingProcessor(ParakeetProcessor):
    def __init__(
        self,
        feature_extractor,
        tokenizer,
        blank_token="<blank>",
        decoder_type=None,
        supported_num_lookahead_tokens=None,
        default_num_lookahead_tokens=None,
    ):
        r"""
        blank_token (`str`, *optional*, defaults to `"<blank>"`):
            Blank token for RNN-T decoding.
        decoder_type (`str`, *optional*):
            Decoding/timestamp emission mode (e.g. `"ctc"`, `"rnnt"`, `"tdt"`). If `None` (older checkpoints)
            the decoder type is inferred automatically for backward compatibility.
        supported_num_lookahead_tokens (`list[int]`, *optional*):
            Right attention contexts (lookaheads, in subsampled encoder frames) the model was trained with.
            The processor is the single source of truth for this set: [`~NemotronAsrStreamingProcessor.set_num_lookahead_tokens`]
            validates against it. Defaults to the NeMo cache-aware set `[13, 6, 1, 0]`.
        default_num_lookahead_tokens (`int`, *optional*):
            The right context used to size streaming chunks and emitted by [`~NemotronAsrStreamingProcessor.__call__`];
            change it with [`~NemotronAsrStreamingProcessor.set_num_lookahead_tokens`]. Defaults to the first entry of
            `supported_num_lookahead_tokens`.
        """
        self.supported_num_lookahead_tokens = (
            supported_num_lookahead_tokens
            if supported_num_lookahead_tokens is not None
            else DEFAULT_NUM_LOOKAHEAD_TOKENS
        )
        self.default_num_lookahead_tokens = (
            default_num_lookahead_tokens
            if default_num_lookahead_tokens is not None
            else self.supported_num_lookahead_tokens[0]
        )
        super().__init__(feature_extractor, tokenizer, blank_token=blank_token, decoder_type=decoder_type)

    def set_num_lookahead_tokens(self, num_lookahead_tokens: int):
        """
        Select the right attention context (lookahead, in subsampled encoder frames) used for streaming.

        Sets `default_num_lookahead_tokens`, so every derived streaming property
        (`num_mel_frames_first_audio_chunk`, `num_mel_frames_per_audio_chunk`, `num_samples_first_audio_chunk`,
        `num_samples_per_audio_chunk`) re-derives from the new value. `num_lookahead_tokens` must be one of
        `supported_num_lookahead_tokens`.

        Pass the same `num_lookahead_tokens` to `model.generate` so the attention right context used in the
        forward matches the chunk sizes produced here; otherwise streaming `generate` raises.
        """
        if num_lookahead_tokens not in self.supported_num_lookahead_tokens:
            raise ValueError(
                f"`num_lookahead_tokens={num_lookahead_tokens}` is not supported by this model. Supported "
                f"values: {list(self.supported_num_lookahead_tokens)}."
            )
        self.default_num_lookahead_tokens = num_lookahead_tokens

    @property
    def _subsampling_factor(self) -> int:
        output_kwargs = self._merge_kwargs(
            NemotronAsrStreamingProcessorKwargs, tokenizer_init_kwargs=self.tokenizer.init_kwargs
        )
        return output_kwargs["audio_kwargs"]["subsampling_factor"]

    @property
    def _encoder_frame_ms(self) -> float:
        """Duration in milliseconds of one subsampled encoder frame (`subsampling_factor * hop_length / sampling_rate`)."""
        return (
            self._subsampling_factor * self.feature_extractor.hop_length / self.feature_extractor.sampling_rate * 1000
        )

    @property
    def streaming_latency_ms(self) -> int:
        """
        Streaming latency (ms) of the currently-selected right attention context
        (`default_num_lookahead_tokens`, settable via [`~NemotronAsrStreamingProcessor.set_num_lookahead_tokens`]).

        The model emits a chunk only once its last frame has its full lookahead, so the delay of a right
        context `r` is `(r + 1)` encoder frames, i.e. `(r + 1) * encoder_frame_ms`.
        """
        return round((self.default_num_lookahead_tokens + 1) * self._encoder_frame_ms)

    @property
    def supported_streaming_latencies_ms(self) -> dict[int, int]:
        """
        Mapping from each supported right attention context (`supported_num_lookahead_tokens`) to its streaming
        latency in milliseconds (`(num_lookahead_tokens + 1) * encoder_frame_ms`).
        """
        frame_ms = self._encoder_frame_ms
        return {right: round((right + 1) * frame_ms) for right in self.supported_num_lookahead_tokens}

    @property
    def num_mel_frames_first_audio_chunk(self) -> int:
        """
        Number of mel frames the first cache-aware streaming chunk must carry, for the model's
        `default_num_lookahead_tokens`: `1 + subsampling_factor * num_lookahead_tokens`.
        """
        return 1 + self._subsampling_factor * self.default_num_lookahead_tokens

    @property
    def num_mel_frames_per_audio_chunk(self) -> int:
        """
        Number of mel frames each subsequent cache-aware streaming chunk must carry, for the model's
        `default_num_lookahead_tokens`: `subsampling_factor * (num_lookahead_tokens + 1)`.
        """
        return self._subsampling_factor * (self.default_num_lookahead_tokens + 1)

    @property
    def num_samples_first_audio_chunk(self) -> int:
        """
        Number of raw audio samples to feed the processor (with `is_first_audio_chunk=True`, i.e. `center=True`)
        so it returns exactly `num_mel_frames_first_audio_chunk` frames.
        """
        return (
            self.num_mel_frames_first_audio_chunk - 1
        ) * self.feature_extractor.hop_length + self.feature_extractor.win_length // 2

    @property
    def num_samples_per_audio_chunk(self) -> int:
        """
        Number of raw audio samples to feed the processor (with `is_first_audio_chunk=False`, i.e. `center=False`)
        so it returns exactly `num_mel_frames_per_audio_chunk` frames.
        """
        return (
            self.num_mel_frames_per_audio_chunk * self.feature_extractor.hop_length + self.feature_extractor.win_length
        )

    @auto_docstring
    def __call__(
        self,
        audio: AudioInput,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        sampling_rate: int | None = None,
        is_streaming: bool = False,
        is_first_audio_chunk: bool | None = True,
        **kwargs: Unpack[NemotronAsrStreamingProcessorKwargs],
    ):
        r"""
        sampling_rate (`int`, *optional*):
            The sampling rate of the input audio in Hz. This should match the sampling rate expected by the feature
            extractor (defaults to 16000 Hz). If provided, it will be validated against the processor's expected
            sampling rate, and an error will be raised if they don't match. If not provided, a warning will be
            issued and the default sampling rate will be assumed.
        is_streaming (`bool`, *optional*, defaults to `False`):
            Whether to process audio in streaming mode. When `True`, audio can be passed in chunks, using
            `is_first_audio_chunk` to distinguish the first chunk from subsequent ones.
        is_first_audio_chunk (`bool`, *optional*, defaults to `True`):
            Whether the current audio is the first chunk of a streaming session. The feature extractor uses
            `center=True` for the first chunk (and for offline use) and `center=False` for subsequent chunks,
            so that the per-chunk STFT reproduces, frame-for-frame, a single full-utterance pass. Must be
            `True` when `is_streaming=False`.

        Returns:
            [`BatchFeature`]: the feature-extractor (and optional tokenizer) outputs, augmented with:

            - **num_lookahead_tokens** -- The right attention context (lookahead, in subsampled encoder frames),
              i.e. `default_num_lookahead_tokens` (set via [`~NemotronAsrStreamingProcessor.set_num_lookahead_tokens`]).
              Pass it to the model/encoder forward (or `generate`); it plays the role of Voxtral Realtime's
              `num_delay_tokens`.
        """
        if not is_streaming and not is_first_audio_chunk:
            raise ValueError("In non-streaming mode (`is_streaming=False`), `is_first_audio_chunk` must be `True`.")

        audio = make_list_of_audio(audio)

        output_kwargs = self._merge_kwargs(
            NemotronAsrStreamingProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if sampling_rate is None:
            logger.warning_once(
                f"You've provided audio without specifying the sampling rate. It will be assumed to be {output_kwargs['audio_kwargs']['sampling_rate']}, which can result in silent errors."
            )
        elif sampling_rate != output_kwargs["audio_kwargs"]["sampling_rate"]:
            raise ValueError(
                f"The sampling rate of the audio ({sampling_rate}) does not match the sampling rate of the processor ({output_kwargs['audio_kwargs']['sampling_rate']}). Please provide resampled the audio to the expected sampling rate."
            )

        if audio is not None:
            # `center=True` for the first/offline chunk, `center=False` for subsequent streaming chunks.
            inputs = self.feature_extractor(audio, center=bool(is_first_audio_chunk), **output_kwargs["audio_kwargs"])
        if text is not None:
            encodings = self.tokenizer(text, **output_kwargs["text_kwargs"])

        # The right attention context (akin to Voxtral Realtime's `num_delay_tokens`), selected via
        # `set_num_lookahead_tokens`; pass it to the model/encoder forward or `generate`.
        inputs["num_lookahead_tokens"] = self.default_num_lookahead_tokens

        if text is None:
            return inputs
        else:
            inputs["labels"] = encodings["input_ids"]
            # Prepend blank token to labels to form decoder_input_ids.
            # The RNN-T decoder expects [blank, label_0, ..., label_{U-1}] as input,
            if isinstance(text, str):
                text = [text]
            decoder_text = [self.blank_token + t for t in text]
            decoder_encodings = self.tokenizer(decoder_text, **output_kwargs["text_kwargs"])
            inputs["decoder_input_ids"] = decoder_encodings["input_ids"]
            return inputs


class NemotronAsrStreamingFeatureExtractor(ParakeetFeatureExtractor):
    def _torch_extract_fbank_features(self, waveform, device="cpu", center=True):
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

        # NemotronAsrStreaming never normalizes the mel features
        input_features *= attention_mask.unsqueeze(-1)

        return BatchFeature(
            data={
                "input_features": input_features,
                "attention_mask": attention_mask,
            },
            tensor_type=return_tensors,
        )


class NemotronAsrStreamingEncoderCausalConv1dCacheLayer(VoxtralRealtimeConv1dCacheLayer): ...


class NemotronAsrStreamingEncoderCausalConv2dCacheLayer:
    def __init__(self):
        self.cache: torch.Tensor | None = None
        self.is_initialized: bool = False

    def lazy_initialization(self, hidden_states, conv_module):
        self.left_pad = conv_module.left_pad
        self.init_pad = conv_module.left_pad_init - conv_module.left_pad
        cache_shape = list(hidden_states.shape)
        cache_shape[2] = self.left_pad
        self.cache = torch.zeros(cache_shape, device=hidden_states.device, dtype=hidden_states.dtype)

        if not is_torchdynamo_compiling():
            torch._dynamo.mark_static_address(self.cache)

        self.is_first_chunk = True
        self.is_initialized = True

    def update(self, hidden_states, conv_module=None):
        if not self.is_initialized and conv_module is not None:
            self.lazy_initialization(hidden_states, conv_module)
        elif not self.is_initialized:
            raise ValueError(
                "NemotronAsrStreamingEncoderCausalConv2dCacheLayer is not initialized. Make sure to provide conv_module to the update method."
            )

        # new cache: the last `left_pad` time frames (dim 2), keeping the old cache tail on shortfall
        shortfall = max(0, self.left_pad - hidden_states.shape[2])
        if shortfall > 0:
            new_cache = torch.cat([self.cache[:, :, -shortfall:], hidden_states], dim=2)
        else:
            new_cache = hidden_states[:, :, -self.left_pad :]

        # left context to prepend: the old cache, plus `init_pad` leading zeros on the first chunk
        current_cache = self.cache.clone()
        if self.is_first_chunk and self.init_pad > 0:
            init_shape = list(current_cache.shape)
            init_shape[2] = self.init_pad
            current_cache = torch.cat([current_cache.new_zeros(init_shape), current_cache], dim=2)
        self.is_first_chunk = False

        self.cache.copy_(new_cache)
        return current_cache


class NemotronAsrStreamingEncoderCausalConvPaddingCache:
    def __init__(self):
        self.layers: dict[str, NemotronAsrStreamingEncoderCausalConv1dCacheLayer] = {}

    def update(self, hidden_states, cache_key, conv_module):
        if cache_key not in self.layers:
            if isinstance(conv_module, NemotronAsrStreamingEncoderCausalConv2D):
                self.layers[cache_key] = NemotronAsrStreamingEncoderCausalConv2dCacheLayer()
            elif isinstance(conv_module, NemotronAsrStreamingEncoderCausalConv1d):
                self.layers[cache_key] = NemotronAsrStreamingEncoderCausalConv1dCacheLayer()
            else:
                raise NotImplementedError(f"Unsupported conv_module type: {type(conv_module)}")

        padding_states = self.layers[cache_key].update(hidden_states, conv_module)
        return torch.cat([padding_states, hidden_states], dim=2)


class NemotronAsrStreamingEncoderCausalConv1d(VoxtralRealtimeCausalConv1d): ...


class NemotronAsrStreamingEncoderCausalConv2D(nn.Conv2d):
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
        return self.kernel_size[0] - self.stride[0]

    @cached_property
    def left_pad_init(self):
        return self.kernel_size[0] - 1

    @cached_property
    def time_pad(self):
        return (self.kernel_size[0] - 1, self.stride[0] - 1)

    @cached_property
    def freq_pad(self):
        return (self.kernel_size[1] - 1, self.stride[1] - 1)

    def forward(
        self,
        x: torch.Tensor,
        padding_cache: NemotronAsrStreamingEncoderCausalConvPaddingCache | None = None,
    ) -> torch.Tensor:
        x = nn.functional.pad(x, (self.freq_pad[0], self.freq_pad[1]))
        if padding_cache is not None:
            x = padding_cache.update(x, self.cache_key, self)
        else:
            x = nn.functional.pad(x, (0, 0, self.time_pad[0], self.time_pad[1]))

        return super().forward(x)


@auto_docstring(
    custom_intro="""
    Extends [`ParakeetEncoderModelOutput`] with optional streaming caches. Caches are only populated for
    cache-aware models when `use_cache=True`.
    """
)
@dataclass
class NemotronAsrStreamingEncoderModelOutput(BaseModelOutputWithPooling):
    r"""
    attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
        Mask to avoid performing attention on padding token indices after sequence compression. Returned because the
        sequence length may differ from the input sequence length. Mask values selected in `[0, 1]`:

        - 1 for tokens that are **not masked**,
        - 0 for tokens that are **masked**.
    past_key_values (`Cache`, *optional*):
        Updated attention K/V sliding-window cache from the encoder. Pass to the next chunk call.
    padding_cache (`NemotronAsrStreamingEncoderCausalConvPaddingCache`, *optional*):
        Unified streaming cache backing the subsampling Conv2d layers and the conformer depthwise Conv1d.
    """

    attention_mask: torch.Tensor | None = None
    past_key_values: Cache | None = None
    padding_cache: NemotronAsrStreamingEncoderCausalConvPaddingCache | None = None


class NemotronAsrStreamingEncoderRelPositionalEncoding(ParakeetEncoderRelPositionalEncoding):
    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor, cached_frames: int | None = None):
        # `cached_frames` is the number of cached left-context frames (0 offline). This Transformer-XL
        # style relative encoding spans the full key length `L = current chunk + cached_frames`, with
        # relative distances running from `L - 1` down to `-(L - 1)`.
        seq_length = hidden_states.shape[1] + (cached_frames if cached_frames is not None else 0)
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


class NemotronAsrStreamingEncoderConvolutionModule(FastSpeech2ConformerConvolutionModule):
    def __init__(self, config: NemotronAsrStreamingEncoderConfig, module_config=None, layer_idx: int | None = None):
        super().__init__(config, module_config)
        kernel_size = config.conv_kernel_size
        channels = config.hidden_size

        self.norm = nn.LayerNorm(channels)
        self.depthwise_conv = NemotronAsrStreamingEncoderCausalConv1d(
            channels,
            channels,
            kernel_size,
            cache_key=f"conv.{layer_idx}",
            stride=1,
            groups=channels,
            bias=config.convolution_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        padding_cache: NemotronAsrStreamingEncoderCausalConvPaddingCache | None = None,
    ):
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

        # Causal depthwise conv: left context from `padding_cache` when streaming, else left-padded.
        hidden_states = self.depthwise_conv(hidden_states, padding_cache=padding_cache)

        # LayerNorm expects (B, T, C).
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.activation(hidden_states)
        hidden_states = self.pointwise_conv2(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)  # (B, T, C)

        return hidden_states


class NemotronAsrStreamingEncoderAttention(ParakeetEncoderAttention):
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


class NemotronAsrStreamingEncoderSubsamplingConv2D(nn.Module):
    def __init__(self, config: NemotronAsrStreamingEncoderConfig):
        super().__init__()

        self.kernel_size = config.subsampling_conv_kernel_size
        self.stride = config.subsampling_conv_stride
        self.channels = config.subsampling_conv_channels
        self.padding = (self.kernel_size - 1) // 2
        self.num_layers = int(math.log2(config.subsampling_factor))

        self.layers = nn.ModuleList()
        self.layers.append(
            NemotronAsrStreamingEncoderCausalConv2D(
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
                NemotronAsrStreamingEncoderCausalConv2D(
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
        pad_left, pad_right = self.kernel_size - 1, self.stride - 1
        out_length = config.num_mel_bins
        total_pad = pad_left + pad_right
        for _ in range(self.num_layers):
            out_length = (out_length + total_pad - self.kernel_size) // self.stride + 1
        self.linear = nn.Linear(config.subsampling_conv_channels * out_length, config.hidden_size, bias=True)

    def _get_output_length(self, input_lengths: torch.Tensor, conv_layer: nn.Conv2d, streaming: bool = False):
        if not (hasattr(conv_layer, "stride") and conv_layer.stride != (1, 1)):
            return input_lengths

        kernel_size = conv_layer.kernel_size[0]
        stride = conv_layer.stride[0]
        if isinstance(conv_layer, NemotronAsrStreamingEncoderCausalConv2D):
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
        padding_cache: NemotronAsrStreamingEncoderCausalConvPaddingCache | None = None,
    ):
        hidden_states = input_features.unsqueeze(1)
        current_lengths = attention_mask.sum(-1) if attention_mask is not None else None
        streaming = padding_cache is not None

        for layer in self.layers:
            if isinstance(layer, NemotronAsrStreamingEncoderCausalConv2D):
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


class NemotronAsrStreamingEncoderBlock(ParakeetEncoderBlock):
    def __init__(self, config: NemotronAsrStreamingEncoderConfig, layer_idx: int | None = None):
        super().__init__(config, layer_idx)
        self.conv = NemotronAsrStreamingEncoderConvolutionModule(config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        padding_cache: NemotronAsrStreamingEncoderCausalConvPaddingCache | None = None,
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
class NemotronAsrStreamingPreTrainedModel(ParakeetPreTrainedModel):
    config: NemotronAsrStreamingConfig

    def _get_subsampling_output_length(self, input_lengths: torch.Tensor):
        encoder_config = getattr(self.config, "encoder_config", self.config)

        kernel_size = encoder_config.subsampling_conv_kernel_size
        stride = encoder_config.subsampling_conv_stride
        num_layers = int(math.log2(encoder_config.subsampling_factor))

        # The subsampling Conv2d is always causal: NeMo's CausalConv2D pads (left=kernel-1, right=stride-1).
        all_paddings = (kernel_size - 1) + (stride - 1)
        add_pad = all_paddings - kernel_size
        lengths = input_lengths

        for _ in range(num_layers):
            lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + 1.0
            lengths = torch.floor(lengths)

        return lengths.to(dtype=torch.int)


def chunked_limited_mask_function(left_ctx: int, right_ctx: int) -> Callable:
    """
    `chunked_limited` attention mask.
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
    The NemotronAsrStreaming Encoder model, based on the [Fast Conformer architecture](https://huggingface.co/papers/2305.05084).
    """
)
class NemotronAsrStreamingEncoder(ParakeetEncoder):
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
        padding_cache: NemotronAsrStreamingEncoderCausalConvPaddingCache | None = None,
        num_lookahead_tokens: int | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        r"""
        output_attention_mask (`bool`, *optional*, defaults to `True`):
            Whether to return the output attention mask. Only effective when `attention_mask` is provided.
        past_key_values (`Cache`, *optional*):
            Sliding-window K/V cache (`DynamicCache` built from `config.sliding_window`) for cache-aware
            streaming attention.
        padding_cache (`NemotronAsrStreamingEncoderCausalConvPaddingCache`, *optional*):
            Unified streaming cache backing the subsampling Conv2d layers and the conformer depthwise Conv1d.
        num_lookahead_tokens (`int`, *optional*):
            Override of the right attention context (lookahead, in subsampled encoder frames) for this
            forward pass. Combined with the left context `config.sliding_window - 1`. Defaults to
            `config.default_num_lookahead_tokens`.

        Example:

        ```python
        >>> from transformers import AutoProcessor, NemotronAsrStreamingEncoder
        >>> from datasets import load_dataset, Audio

        >>> model_id = "nvidia/nemotron-speech-streaming-en-0.6b"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> encoder = NemotronAsrStreamingEncoder.from_pretrained(model_id)

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

        >>> inputs = processor(ds[0]["audio"]["array"])
        >>> encoder_outputs = encoder(**inputs)

        >>> print(encoder_outputs.last_hidden_state.shape)
        ```
        """
        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache(config=self.config)

            if padding_cache is None:
                padding_cache = NemotronAsrStreamingEncoderCausalConvPaddingCache()

        inputs_embeds = self.subsampling(input_features, attention_mask, padding_cache=padding_cache)
        inputs_embeds *= self.input_scale

        seq_length = inputs_embeds.shape[1]
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        output_mask = None
        if attention_mask is not None:
            output_mask = self._get_output_attention_mask(attention_mask, target_length=seq_length)

        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=output_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            and_mask_function=chunked_limited_mask_function(*self._resolve_attn_context(num_lookahead_tokens)),
        )

        cached_frames = (
            past_key_values.get_mask_sizes(seq_length, 0)[0] - seq_length if past_key_values is not None else 0
        )
        position_embeddings = self.encode_positions(inputs_embeds, cached_frames=cached_frames)

        inputs_embeds = nn.functional.dropout(inputs_embeds, p=self.dropout, training=self.training)
        position_embeddings = nn.functional.dropout(
            position_embeddings, p=self.dropout_positions, training=self.training
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
                    attention_mask=attention_mask,
                    position_embeddings=position_embeddings,
                    past_key_values=past_key_values,
                    padding_cache=padding_cache,
                    use_cache=use_cache,
                    **kwargs,
                )

        return NemotronAsrStreamingEncoderModelOutput(
            last_hidden_state=hidden_states,
            attention_mask=output_mask.int() if output_mask is not None and output_attention_mask else None,
            past_key_values=past_key_values,
            padding_cache=padding_cache,
        )

    def _resolve_attn_context(self, num_lookahead_tokens: int | None = None) -> tuple[int, int]:
        if num_lookahead_tokens is None:
            num_lookahead_tokens = self.config.default_num_lookahead_tokens
            logger.warning_once(
                f"`num_lookahead_tokens` was not provided. "
                f"Falling back to `config.default_num_lookahead_tokens={num_lookahead_tokens}`. "
                f"Consider preparing inputs with [`~NemotronAsrStreamingProcessor.__call__`] which automatically sets "
                f"this parameter."
            )

        left_context = self.config.sliding_window - 1
        return left_context, num_lookahead_tokens


@dataclass
class NemotronAsrStreamingRNNTOutput(ParakeetRNNTOutput):
    r"""
    encoder_past_key_values (`Cache`, *optional*):
        Updated encoder attention K/V sliding-window cache, returned when encoding audio with `use_cache=True`
        (cache-aware streaming). Pass it to the next chunk's forward.
    padding_cache (`NemotronAsrStreamingEncoderCausalConvPaddingCache`, *optional*):
        Updated unified streaming conv cache (subsampling Conv2d + conformer depthwise Conv1d), returned when
        encoding audio with `use_cache=True`. Pass it to the next chunk's forward.
    """

    encoder_past_key_values: Cache | None = None
    padding_cache: NemotronAsrStreamingEncoderCausalConvPaddingCache | None = None


class NemotronAsrStreamingRNNTDecoder(ParakeetRNNTDecoder):
    def __init__(self, config: NemotronAsrStreamingConfig):
        super().__init__(config)


class NemotronAsrStreamingRNNTJointNetwork(ParakeetRNNTJointNetwork):
    def __init__(self, config: NemotronAsrStreamingConfig):
        super().__init__(config)


@auto_docstring(
    custom_intro="""
    NemotronAsrStreaming Encoder with an RNN-T (Recurrent Neural Network Transducer) head.
    """
)
class NemotronAsrStreamingForRNNT(ParakeetForRNNT, NemotronAsrStreamingPreTrainedModel, NemotronAsrStreamingGenerationMixin):
    config: NemotronAsrStreamingConfig

    def __init__(self, config: NemotronAsrStreamingConfig):
        super().__init__(config)

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_features: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        decoder_cache: NemotronAsrStreamingRNNTDecoderCache | None = None,
        use_decoder_cache: bool | None = None,
        encoder_outputs: NemotronAsrStreamingEncoderModelOutput | tuple[torch.FloatTensor] | None = None,
        labels: torch.Tensor | None = None,
        num_lookahead_tokens: int | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> NemotronAsrStreamingRNNTOutput:
        r"""
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, 1)`, *optional*):
            Decoder input token ids for single-step inference.
        decoder_cache (`NemotronAsrStreamingRNNTDecoderCache`, *optional*):
            Decoder LSTM cache. Reused on blank predictions to skip the LSTM step.
        use_decoder_cache (`bool`, *optional*):
            Whether to allocate and use a decoder cache when none is provided.
        encoder_outputs (`tuple(torch.FloatTensor)`, *optional*):
            Pre-computed encoder outputs (last_hidden_state, pooler_output, ...).
        num_lookahead_tokens (`int`, *optional*):
            Right attention context (lookahead, in subsampled encoder frames) forwarded to the encoder.
            Defaults to `config.encoder_config.default_num_lookahead_tokens`.

        Example:

        ```python
        >>> from transformers import AutoProcessor, NemotronAsrStreamingForRNNT
        >>> from datasets import load_dataset, Audio

        >>> model_id = "nvidia/nemotron-speech-streaming-en-0.6b"
        >>> revision = "refs/pr/17"
        >>> processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        >>> model = NemotronAsrStreamingForRNNT.from_pretrained(model_id, revision=revision)

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
                num_lookahead_tokens=num_lookahead_tokens,
                **kwargs,
            )
        elif not isinstance(encoder_outputs, NemotronAsrStreamingEncoderModelOutput):
            encoder_outputs = NemotronAsrStreamingEncoderModelOutput(
                last_hidden_state=encoder_outputs[0] if len(encoder_outputs) > 0 else None,
                pooler_output=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                hidden_states=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                attentions=encoder_outputs[3] if len(encoder_outputs) > 3 else None,
                attention_mask=encoder_outputs[4] if len(encoder_outputs) > 4 else None,
            )

        if use_decoder_cache and decoder_cache is None:
            decoder_cache = NemotronAsrStreamingRNNTDecoderCache()

        decoder_hidden_states = self.decoder(decoder_input_ids, cache=decoder_cache)
        logits = self.joint(
            encoder_hidden_states=encoder_outputs.pooler_output[:, :, None, :],
            decoder_hidden_states=decoder_hidden_states[:, None, :, :],
        ).squeeze(2)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, encoder_outputs=encoder_outputs)

        return NemotronAsrStreamingRNNTOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=encoder_outputs.last_hidden_state,
            pooler_output=encoder_outputs.pooler_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            decoder_cache=decoder_cache,
            encoder_past_key_values=encoder_outputs.past_key_values,
            padding_cache=encoder_outputs.padding_cache,
        )


__all__ = [
    "NemotronAsrStreamingConfig",
    "NemotronAsrStreamingEncoderConfig",
    "NemotronAsrStreamingFeatureExtractor",
    "NemotronAsrStreamingProcessor",
    "NemotronAsrStreamingEncoderModelOutput",
    "NemotronAsrStreamingRNNTOutput",
    "NemotronAsrStreamingForRNNT",
    "NemotronAsrStreamingEncoder",
    "NemotronAsrStreamingPreTrainedModel",
]
