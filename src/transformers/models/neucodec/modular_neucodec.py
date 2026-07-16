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
from dataclasses import dataclass

import numpy as np
import torch
from huggingface_hub.dataclasses import strict

from ...audio_utils import AudioInput, make_list_of_audio
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import Unpack
from ...utils import PaddingStrategy, TensorType, TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.import_utils import is_torch_available, is_torchaudio_available
from ..xcodec2.configuration_xcodec2 import Xcodec2Config
from ..xcodec2.feature_extraction_xcodec2 import Xcodec2FeatureExtractor
from ..xcodec2.modeling_xcodec2 import (
    Xcodec2DecoderOutput,
    Xcodec2EncoderOutput,
    Xcodec2Model,
    Xcodec2Output,
    Xcodec2PreTrainedModel,
)


if is_torch_available():
    import torch.nn.functional as F

if is_torchaudio_available():
    import torchaudio


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="neuphonic/neucodec")
@strict
class NeuCodecConfig(Xcodec2Config):
    r"""
    downsampling_ratios (`list[int]`, *optional*, defaults to `[2, 2, 4, 4, 5]`):
        Ratios for downsampling in the encoder.
    semantic_model_config (`Union[Dict, Wav2Vec2BertConfig]`, *optional*):
        An instance of the configuration object for the semantic (Wav2Vec2BertConfig) model.
    quantization_dim (`int`, *optional*, defaults to 2048):
        Dimension for the vector quantization codebook.
    quantization_levels (`list[int]`, *optional*, defaults to `[4, 4, 4, 4, 4, 4, 4, 4]`):
        Levels for the vector quantization codebook.
    input_sampling_rate (`int`, *optional*, defaults to 16000):
        Sampling rate, in hertz (Hz), of the decoder's input audio waveform. NeuCodec encodes audio sampled at
        `input_sampling_rate` (16kHz) but its decoder upsamples the reconstruction to a higher-fidelity
        `sampling_rate` (24kHz), while keeping the same 50Hz code frame rate.

    Example:

    ```python
    >>> from transformers import NeuCodecConfig, NeuCodecModel

    >>> # Initializing configuration
    >>> configuration = NeuCodecConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = NeuCodecModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "neucodec"
    input_sampling_rate: int = 16_000

    @property
    def encoder_hop_length(self) -> int:
        """Hop length, in samples at `sampling_rate`, between successive codes (i.e. the encoder frame rate)."""
        return int(np.prod(self.downsampling_ratios))

    @property
    def hop_length(self) -> int:
        # The ISTFT head (which reads `hop_length`/`n_fft` off the config) synthesizes audio at
        # `sampling_rate`, so the encoder's native hop_length is rescaled into that domain.
        return int(self.encoder_hop_length * self.sampling_rate / self.input_sampling_rate)


@auto_docstring
@dataclass
class NeuCodecOutput(Xcodec2Output):
    pass


@auto_docstring
@dataclass
class NeuCodecEncoderOutput(Xcodec2EncoderOutput):
    pass


@auto_docstring
@dataclass
class NeuCodecDecoderOutput(Xcodec2DecoderOutput):
    pass


class NeuCodecPreTrainedModel(Xcodec2PreTrainedModel):
    config: NeuCodecConfig
    base_model_prefix = "neucodec"


@auto_docstring(custom_intro="NeuCodec neural audio codec model.")
class NeuCodecModel(Xcodec2Model):
    def __init__(self, config: NeuCodecConfig):
        super().__init__(config)
        # `Xcodec2Model.hop_length` mirrors `config.hop_length`, which for NeuCodec is expressed in the decoder's
        # (24kHz) domain. The mask arithmetic in `encode()` operates on `input_values` in the encoder's (16kHz)
        # domain, so it must use the un-rescaled hop length instead.
        self.hop_length = config.encoder_hop_length

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_values: torch.Tensor,
        input_features: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        output_latents: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | NeuCodecOutput:
        r"""
        input_values (`torch.Tensor` of shape `(batch_size, 1, sequence_length)`):
            Input audio waveform, sampled at `config.sampling_rate` (16kHz).
        input_features (`torch.Tensor` of shape `(batch_size, mel_bins, time_steps)`):
            Input audio mel spectrogram for semantic encoding.
        padding_mask (`torch.Tensor` of shape `(batch_size, 1, sequence_length)`):
            Padding mask used to pad `input_values`.
        input_features_mask (`torch.Tensor` of shape `(batch_size, time_steps)`, *optional*):
            Attention mask for the spectrogram input to the semantic encoder. `1` for valid frames, `0` for padding.
        output_latents (`bool`, *optional*, defaults to `False`):
            Whether to return the continuous latent representation from the quantizer.

        Examples:

        ```python
        >>> from datasets import load_dataset
        >>> from transformers import AutoFeatureExtractor, NeuCodecModel

        >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> audio = dataset["train"]["audio"][0]["array"]

        >>> model_id = "neuphonic/neucodec"
        >>> model = NeuCodecModel.from_pretrained(model_id)
        >>> feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

        >>> inputs = feature_extractor(audio=audio, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> audio_codes = outputs.audio_codes
        >>> audio_values = outputs.audio_values  # sampled at 24kHz
        ```"""
        # NeuCodec's decoder outputs audio at `sampling_rate`, which differs from the `input_sampling_rate` of
        # `input_values`, so the truncation length must be rescaled accordingly.
        input_length = input_values.shape[-1]
        output_length = int(input_length * self.config.sampling_rate / self.config.input_sampling_rate)

        encoder_outputs = self.encode(
            input_values,
            input_features=input_features,
            padding_mask=padding_mask,
            input_features_mask=input_features_mask,
            output_latents=True,
            return_dict=True,
        )
        audio_values = self.decode(latents=encoder_outputs.latents, return_dict=True, **kwargs)[0][
            ..., :output_length
        ]

        return NeuCodecOutput(
            audio_values=audio_values,
            audio_codes=encoder_outputs.audio_codes,
            latents=encoder_outputs.latents if output_latents else None,
            audio_codes_mask=encoder_outputs.audio_codes_mask,
        )


class NeuCodecFeatureExtractor(Xcodec2FeatureExtractor):
    def __call__(
        self,
        audio: AudioInput,
        padding: bool | str | PaddingStrategy = True,
        max_length: int | None = None,
        truncation: bool = False,
        return_tensors: str | TensorType | None = None,
        sampling_rate: int | None = None,
        device: str = "cpu",
        **kwargs,
    ) -> BatchFeature:
        """
        Args:
            audio (`np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`):
                Numpy array or torch tensor with shape (num_channels, sequence_length). A list of such arrays or
                tensors can also be provided for a batch of inputs.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            truncation (`bool`):
                Activates truncation to cut input sequences longer than *max_length* to *max_length*.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            sampling_rate (`int`, *optional*):
                The sample rate at which the `audio` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors.
            device (`str`, *optional*, defaults to `"cpu"`):
                Device for PyTorch tensors during mel-filter bank feature extraction.
            kwargs (*optional*):
                Remaining dictionary of keyword arguments that will be passed to the tokenizer or the feature
                extractor.
        """
        if not is_torch_available():
            raise ImportError("PyTorch is required for mel-filter bank feature extraction.")

        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self} was trained using a sampling rate of"
                    f" {self.sampling_rate}. Please make sure that the provided `audio` input was sampled with"
                    f" {self.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning(
                f"It is strongly recommended to pass the `sampling_rate` argument to `{self.__class__.__name__}()`. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        audio = make_list_of_audio(audio)
        for example in audio:
            if example.ndim > 2:
                raise ValueError(f"Expected input shape (channels, length) but got shape {example.shape}")
        batch_size = len(audio)

        # 1) Acoustic encoder padding
        audio = [F.pad(torch.as_tensor(a), (0, 1), value=0.0) for a in audio]
        padded_inputs = self.acoustic_encoder_padder.pad(
            BatchFeature({"audio": audio}),
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            return_attention_mask=padding,
            pad_to_multiple_of=self.hop_length,
            return_tensors="pt",
        )
        padding_mask = padded_inputs.pop("attention_mask")
        padded_audio = padded_inputs["audio"][:, None, :]

        # 2) Semantic encoder feature extraction (mel spectrogram). Unlike Xcodec2FeatureExtractor, NeuCodec's
        # reference implementation (https://github.com/neuphonic/neucodec/blob/main/neucodec/model.py) feeds the
        # acoustic-padded waveform directly into the mel computation, with no extra "valid_len" trimming or
        # symmetric hop-sized padding: https://github.com/neuphonic/neucodec/blob/main/neucodec/model.py#L128
        mel_features = []
        for i in range(batch_size):
            waveform = padded_audio[i].to(device)
            features = torchaudio.compliance.kaldi.fbank(
                waveform * (2**15),
                num_mel_bins=self.num_mel_bins,
                frame_length=self.frame_length / self.sampling_rate * 1000,
                frame_shift=self.frame_shift / self.sampling_rate * 1000,
                sample_frequency=self.sampling_rate,
                window_type="povey",
                preemphasis_coefficient=0.97,
                remove_dc_offset=True,
                use_log_fbank=True,
                use_energy=False,
                dither=0.0,
                snip_edges=True,
                low_freq=20,
                high_freq=self.sampling_rate // 2,
            )
            features = (features - features.mean(0)) / torch.sqrt(features.var(0, unbiased=True) + 1e-7)
            mel_features.append(features)
        encoded_inputs = BatchFeature({"input_features": mel_features})
        padded_mel = self.pad(
            encoded_inputs,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            pad_to_multiple_of=self.stride,
            return_attention_mask=padding,
            return_tensors="pt",
        )
        audio_spectrogram = padded_mel["input_features"]
        spectrogram_mask = padded_mel.get("attention_mask")
        trimmed_frames = audio_spectrogram.shape[1] - (audio_spectrogram.shape[1] % self.stride)
        audio_spectrogram = audio_spectrogram[:, :trimmed_frames, :].reshape(
            batch_size, trimmed_frames // self.stride, self.num_mel_bins * self.stride
        )
        if spectrogram_mask is not None:
            spectrogram_mask = (
                spectrogram_mask[:, :trimmed_frames]
                .reshape(batch_size, trimmed_frames // self.stride, self.stride)
                .min(dim=-1)
                .values
            )

        return BatchFeature(
            {
                "input_values": padded_audio,
                "padding_mask": padding_mask,
                "input_features": audio_spectrogram,
                "input_features_mask": spectrogram_mask,
            },
            tensor_type=return_tensors,
        )


__all__ = ["NeuCodecConfig", "NeuCodecModel", "NeuCodecPreTrainedModel", "NeuCodecFeatureExtractor"]
