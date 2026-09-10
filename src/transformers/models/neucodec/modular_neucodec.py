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

import numpy as np
from huggingface_hub.dataclasses import strict

from ...audio_utils import AudioInput, make_list_of_audio
from ...feature_extraction_utils import BatchFeature
from ...masking_utils import create_bidirectional_mask
from ...processing_utils import Unpack
from ...utils import (
    PaddingStrategy,
    TensorType,
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    is_torch_available,
    logging,
)
from ...utils.import_utils import is_torchaudio_available, requires
from ..xcodec2.configuration_xcodec2 import Xcodec2Config
from ..xcodec2.modeling_xcodec2 import (
    Xcodec2DecoderOutput,
    Xcodec2EncoderOutput,
    Xcodec2Model,
    Xcodec2Output,
    Xcodec2PreTrainedModel,
)


if is_torch_available():
    import torch
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
        Sampling rate, in hertz (Hz), of the encoder's input audio waveform.
    output_sampling_rate (`int`, *optional*, defaults to 24000):
        Sampling rate, in hertz (Hz), of the decoder's output audio waveform.

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
    output_sampling_rate: int = 24_000
    sampling_rate = AttributeError()

    @property
    def encoder_hop_length(self) -> int:
        """Hop length, in samples at `input_sampling_rate`, between successive codes (i.e. the encoder frame rate)."""
        return int(np.prod(self.downsampling_ratios))

    @property
    def hop_length(self) -> int:
        # The ISTFT head (which reads `hop_length`/`n_fft` off the config) synthesizes audio at
        # `output_sampling_rate`, so the encoder's native hop_length is rescaled into that domain.
        return int(self.encoder_hop_length * self.output_sampling_rate / self.input_sampling_rate)


class NeuCodecOutput(Xcodec2Output):
    pass


class NeuCodecEncoderOutput(Xcodec2EncoderOutput):
    pass


class NeuCodecDecoderOutput(Xcodec2DecoderOutput):
    pass


class NeuCodecPreTrainedModel(Xcodec2PreTrainedModel):
    pass


@auto_docstring(custom_intro="NeuCodec neural audio codec model.")
class NeuCodecModel(Xcodec2Model):
    def __init__(self, config: NeuCodecConfig):
        super().__init__(config)
        # `Xcodec2Model.hop_length` mirrors `config.hop_length`, which for NeuCodec is expressed in the decoder's
        # (24kHz) domain. The mask arithmetic in `encode()` operates on `input_values` in the encoder's (16kHz)
        # domain, so it must use the un-rescaled hop length instead.
        self.hop_length = config.encoder_hop_length
        self.sample_rate_conversion_factor = config.output_sampling_rate / config.input_sampling_rate

    @auto_docstring
    @can_return_tuple
    def encode(
        self,
        input_values: torch.Tensor,
        input_features: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        output_latents: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | NeuCodecEncoderOutput:
        r"""
        input_values (`torch.Tensor` of shape `(batch_size, 1, sequence_length)`):
            Input audio waveform.
        input_features (`torch.Tensor` of shape `(batch_size, mel_bins, time_steps)`):
            Input audio mel spectrogram for semantic encoding.
        padding_mask (`torch.Tensor` of shape `(batch_size, 1, sequence_length)`):
            Padding mask used to pad `input_values`.
        input_features_mask (`torch.Tensor` of shape `(batch_size, time_steps)`, *optional*):
            Attention mask for the spectrogram input to the semantic encoder. `1` for valid frames, `0` for padding.
        output_latents (`bool`, *optional*, defaults to `False`):
            Whether to return the continuous latent representation from the quantizer.
        """

        # Semantic embedding
        with torch.no_grad():
            semantic_output = self.semantic_encoder(input_features, attention_mask=input_features_mask)
        semantic_hidden_states = semantic_output.last_hidden_state.transpose(1, 2)
        semantic_hidden_states = self.semantic_adapter(semantic_hidden_states)

        # Acoustic embedding
        acoustic_hidden_states = self.acoustic_encoder(input_values)

        # The two branches downsample independently and can differ by a frame; trim to the shorter one, matching
        # the reference: https://github.com/neuphonic/neucodec/blob/ed3e6cd1bdc374ce14a21355e5eee66a777149ce/neucodec/model.py#L173
        min_length = min(acoustic_hidden_states.shape[-1], semantic_hidden_states.shape[-1])
        acoustic_hidden_states = acoustic_hidden_states[..., :min_length]
        semantic_hidden_states = semantic_hidden_states[..., :min_length]

        hidden_states = torch.cat([semantic_hidden_states, acoustic_hidden_states], dim=1)
        hidden_states = self.fc_encoder(hidden_states.transpose(1, 2))

        # Quantize
        latents, audio_codes = self.quantizer(hidden_states)
        latents = latents.transpose(1, 2)
        audio_codes = audio_codes.transpose(1, 2)

        # If provided, compute corresponding padding mask for audio codes
        audio_codes_mask = None
        if padding_mask is not None:
            audio_length = padding_mask.sum(dim=-1, keepdim=True)
            audio_length = audio_length * self.sample_rate_conversion_factor
            token_length = audio_length // self.hop_length
            idx = torch.arange(audio_codes.shape[-1], device=padding_mask.device).view(1, -1)
            audio_codes_mask = (idx < token_length).to(padding_mask.dtype)

        return NeuCodecEncoderOutput(
            audio_codes=audio_codes,
            latents=latents if output_latents else None,
            audio_codes_mask=audio_codes_mask,
        )

    @auto_docstring
    @can_return_tuple
    def decode(
        self,
        audio_codes: torch.Tensor | None = None,
        latents: torch.Tensor | None = None,
        audio_codes_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | NeuCodecDecoderOutput:
        r"""
        audio_codes (`torch.LongTensor`  of shape `(batch_size, 1, codes_length)`):
            Discrete code indices computed using `model.encode`.
        latents (torch.Tensor of shape `(batch_size, dimension, time_steps)`, *optional*):
            Quantized continuous representation of input.
        audio_codes_mask (`torch.Tensor` of shape `(batch_size, codes_length)`, *optional*):
            Mask marking valid (non-padding) positions, as returned by `model.encode`. Needed for correct
            batched decoding of variable-length audio.
        """
        if latents is None and audio_codes is None:
            raise ValueError("Either `latents` or `audio_codes` must be provided.")

        if audio_codes is not None:
            latents = self.quantizer.from_codes(audio_codes.transpose(1, 2))
        else:
            latents = latents.transpose(1, 2)

        # Difference to xcodec2 is the additional mask
        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=latents,
            attention_mask=audio_codes_mask,
        )

        recon_audio = self.acoustic_decoder(latents, attention_mask=attention_mask, **kwargs)
        return NeuCodecDecoderOutput(audio_values=recon_audio)

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
            Input audio waveform, sampled at `config.input_sampling_rate`.
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
        # NeuCodec's decoder outputs audio at `output_sampling_rate`, which differs from the `input_sampling_rate` of
        input_length = input_values.shape[-1]
        output_length = int(input_length * self.sample_rate_conversion_factor)

        encoder_outputs = self.encode(
            input_values,
            input_features=input_features,
            padding_mask=padding_mask,
            input_features_mask=input_features_mask,
            output_latents=True,
            return_dict=True,
        )
        audio_values = self.decode(
            latents=encoder_outputs.latents,
            audio_codes_mask=encoder_outputs.audio_codes_mask,
            return_dict=True,
            **kwargs,
        )[0][..., :output_length]

        return NeuCodecOutput(
            audio_values=audio_values,
            audio_codes=encoder_outputs.audio_codes,
            latents=encoder_outputs.latents if output_latents else None,
            audio_codes_mask=encoder_outputs.audio_codes_mask,
        )


__all__ = ["NeuCodecConfig", "NeuCodecModel", "NeuCodecPreTrainedModel"]
