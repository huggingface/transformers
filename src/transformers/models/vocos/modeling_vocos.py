# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Transformers vocos model."""

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring
from .configuration_vocos import VocosConfig


@dataclass
class VocosOutput(ModelOutput):
    """
    Args:
        audio (`torch.FloatTensor` of shape `(batch_size, time)`):
            Reconstructed audio waveform.
    """

    audio: torch.FloatTensor


def _vocos_inverse_stft(spectrogram, padding, n_fft, hop_length, win_length, window):
    """
    Performs the Inverse Short Time Fourier Transform (ISTFT) on a STFT coefficients to reconstruct audio in the time domain.
    It computes ISTFT differently depending on padding:
        if `center` : uses PyTorch's built-in ISTFT implementation since it uses `center=True` by default.
        if `same` : uses custom implementation of ISTFT with the overlap-add method, since the Pytorch version fails the
        Nonzero Overlap Add (NOLA) condition when center is False. See issue: https://github.com/pytorch/pytorch/issues/62323
        You can find the original vocos implementation here: https://github.com/gemelo-ai/vocos/blob/c859e3b7b534f3776a357983029d34170ddd6fc3/vocos/spectral_ops.py#L7
    """
    if padding == "center":
        audio = torch.istft(
            spectrogram,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
        )

    else:
        batch_size, num_freq_bins, num_time_frames = spectrogram.shape
        pad = (win_length - hop_length) // 2
        # the inverse FFT of each frame
        inverse_fft = torch.fft.irfft(spectrogram, n=n_fft, dim=1, norm="backward")
        inverse_fft = inverse_fft * window[None, :, None]

        # combine the overlapping frame with windowing and normalizing by the sum of squared window values across overlapping frames
        # to make sure the reconstruction of the audio is accurate
        output_length = (num_time_frames - 1) * hop_length + win_length
        audio = F.fold(
            inverse_fft,
            output_size=(1, output_length),
            kernel_size=(1, win_length),
            stride=(1, hop_length),
        )[:, 0, 0, pad:-pad]
        window_sqrt = window.square().expand(1, num_time_frames, -1).transpose(1, 2)
        norm = F.fold(
            window_sqrt,
            output_size=(1, output_length),
            kernel_size=(1, win_length),
            stride=(1, hop_length),
        ).squeeze()[pad:-pad]

        if torch.any(norm <= 1e-11):
            raise ValueError(
                "Normalization tensor `norm` contains values ≤ 1e-11, it would cause division by zero. check the n_fft, hop_length and padding parameters."
            )

        audio = audio / norm

    return audio


class VocosAdaptiveLayerNorm(nn.Module):
    """
    Weight and bias parameters come from a lookup table based on the target bandwidth.
    """

    def __init__(self, config: VocosConfig):
        super().__init__()
        self.eps = config.layer_norm_eps
        self.hidden_dim = config.hidden_dim
        adanorm_num_embeddings = len(config.bandwidths)
        self.weight = nn.Parameter(torch.ones(adanorm_num_embeddings, config.hidden_dim))
        self.bias = nn.Parameter(torch.zeros(adanorm_num_embeddings, config.hidden_dim))

    def forward(self, hidden_states: torch.Tensor, cond_embedding_id: torch.LongTensor):
        hidden_states = F.layer_norm(hidden_states, (self.hidden_dim,), weight=None, bias=None, eps=self.eps)
        return hidden_states * self.weight[cond_embedding_id].unsqueeze(0) + self.bias[cond_embedding_id].unsqueeze(0)


class VocosConvNeXtBlock(nn.Module):
    """ConvNeXt block adapted for 1D convolutions in the Vocos architecture."""

    def __init__(self, config: VocosConfig):
        super().__init__()
        self.dwconv = nn.Conv1d(
            config.hidden_dim,
            config.hidden_dim,
            kernel_size=config.kernel_size,
            padding=config.padding,
            groups=config.hidden_dim,
        )
        if config.use_adaptive_norm:
            self.norm = VocosAdaptiveLayerNorm(config)
        else:
            self.norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.pwconv1 = nn.Linear(config.hidden_dim, config.intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(config.intermediate_dim, config.hidden_dim)
        if config.layer_scale_init_value > 0:
            self.layer_scale_parameter = nn.Parameter(
                config.layer_scale_init_value * torch.ones(config.hidden_dim), requires_grad=True
            )
        else:
            self.layer_scale_parameter = None

    def forward(self, hidden_states: torch.Tensor, bandwidth_id: Optional[torch.LongTensor] = None) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.dwconv(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        if isinstance(self.norm, VocosAdaptiveLayerNorm):
            hidden_states = self.norm(hidden_states, cond_embedding_id=bandwidth_id)
        else:
            hidden_states = self.norm(hidden_states)
        hidden_states = self.pwconv1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.pwconv2(hidden_states)
        if self.layer_scale_parameter is not None:
            hidden_states = self.layer_scale_parameter * hidden_states
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = residual + hidden_states
        return hidden_states


class VocosBackbone(nn.Module):
    """The convolutional backbone of Vocos based on ConvNeXt blocks."""

    def __init__(self, config):
        super().__init__()
        self.embed = nn.Conv1d(
            config.input_channels, config.hidden_dim, kernel_size=config.kernel_size, padding=config.padding
        )
        if config.use_adaptive_norm:
            self.norm = VocosAdaptiveLayerNorm(config)
        else:
            self.norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.layers = nn.ModuleList([VocosConvNeXtBlock(config) for _ in range(config.num_layers)])
        self.final_layer_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states, bandwidth_id=None):
        hidden_states = self.embed(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        if isinstance(self.norm, VocosAdaptiveLayerNorm):
            hidden_states = self.norm(hidden_states, bandwidth_id)
        else:
            hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        for layer in self.layers:
            hidden_states = layer(hidden_states, bandwidth_id)
        hidden_states = self.final_layer_norm(hidden_states.transpose(1, 2))
        return hidden_states


class VocosISTFTHead(nn.Module):
    """
    Projects hidden states to magnitude and phase predictions, combines them into complex
    STFT coefficients, and applies ISTFT to reconstruct the audio waveform.
    """

    def __init__(self, config: VocosConfig):
        super().__init__()
        self.out_proj = nn.Linear(config.hidden_dim, config.n_fft + 2)
        self.padding = config.spec_padding
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.win_length = config.n_fft
        window = torch.hann_window(self.win_length)
        self.register_buffer("window", window)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        spectrogram = self.out_proj(hidden_states)
        spectrogram = spectrogram.transpose(1, 2)
        magnitude, phase = spectrogram.chunk(2, dim=1)
        # safeguard to prevent excessively large magnitudes
        magnitude = torch.exp(magnitude).clamp(max=1e2)
        real = torch.cos(phase)
        imag = torch.sin(phase)
        stft_complex_coeffs = magnitude * (real + 1j * imag)
        audio = _vocos_inverse_stft(
            stft_complex_coeffs,
            padding=self.padding,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
        )
        return audio


class VocosPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = VocosConfig
    base_model_prefix = "vocos"
    main_input_name = "features"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            std = getattr(self.config, "initializer_range", 0.02)
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, VocosAdaptiveLayerNorm):
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.data.fill_(1.0)


@auto_docstring(
    custom_intro="""
    Main Vocos model for neural vocoding for high-quality audio generation. This model can be paired with [`VocosProcessor`] to generate audio from either mel-spectrograms or EnCodec embeddings.
    """
)
class VocosModel(VocosPreTrainedModel):
    config_class = VocosConfig
    base_model_prefix = "vocos"
    main_input_name = "features"

    def __init__(self, config: VocosConfig):
        super().__init__(config)
        self.backbone = VocosBackbone(config)
        self.head = VocosISTFTHead(config)
        self._bandwidth_to_id = {bandwidth: id for id, bandwidth in enumerate(config.bandwidths)}
        self.post_init()

    @auto_docstring
    def forward(
        self, features: torch.FloatTensor, bandwidth: Optional[float] = None, return_dict: Optional[bool] = None
    ) -> Union[VocosOutput, tuple[torch.FloatTensor]]:
        r"""
        features (`torch.FloatTensor` of shape `(batch_size, feature_dim, time)`):
            Output of [`VocosProcessor`] is either:
                - Mel-spectrogram features: computed directly from audio via (`processor(audio=waveform)`)
                - EnCodec neural audio codec features: computed either from precomputed EnCodec RVQ codes via `processor(codes=codes, bandwidth=1.5)`
                            or from raw audio via `processor(audio=waveform, bandwidth=1.5)`, you need to provide bandwidth for both.

        bandwidth (`float`, *optional*):
            Target bandwidth for EnCodec quantizer, e.g. one of [1.5, 3, 6, 12] kbps, or `None` for Mel-spectrogram features.

        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`]

        Returns:
            `VocosOutput` or tuple `(audio,)`:
            - `audio` of shape (batch_size, time): Reconstructed audio waveform.

        Example:

        ```python
        >>> from datasets import load_dataset, Audio
        >>> from transformers import VocosProcessor, VocosModel

        >>> processor = VocosProcessor.from_pretrained("Manel/vocos-mel-24khz")
        >>> model = VocosModel.from_pretrained("Manel/vocos-mel-24khz")

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
        >>> audio_sample= ds[0]["audio"]["array"]

        >>> # extract mel-spectrogram features from audio and reconstruct high-quality audio
        >>> inputs = processor(audio=audio_sample)
        >>> outputs = model(**inputs)
        >>> reconstructed_audio = outputs.audio


        >>> # Encode audio using EnCodec neural codec and reconstruct from audio from that
        >>> processor = VocosProcessor.from_pretrained("Manel/vocos-encodec-24khz")
        >>> model = VocosModel.from_pretrained("Manel/vocos-encodec-24khz")

        >>> bandwidth = 6.0
        >>> inputs = processor(audio=audio_sample, bandwidth=bandwidth)
        >>> outputs = model(**inputs)
        >>> reconstructed_audio = outputs.audio

        >>> # Reconstruct audio directly from pre-computed EnCodec quantized codes
        >>> inputs = processor(codes=audio_codes, bandwidth=bandwidth)
        >>> outputs = model(**inputs)
        >>> reconstructed_audio = outputs.audio

        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if bandwidth is not None:
            bandwidth_id = self._bandwidth_to_id[float(bandwidth)]
        else:
            bandwidth_id = None
        hidden_states = self.backbone(features, bandwidth_id)
        audio = self.head(hidden_states)

        if not return_dict:
            return (audio,)
        return VocosOutput(audio=audio)


__all__ = ["VocosModel", "VocosPreTrainedModel"]
