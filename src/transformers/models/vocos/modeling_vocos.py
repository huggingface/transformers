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


class VocosAdaptiveLayerNorm(nn.Module):
    """
    Weight and bias parameters come from a lookup table based on the target bandwidth.
    """

    def __init__(self, config: VocosConfig):
        super().__init__()
        self.eps = config.layer_norm_eps
        self.hidden_size = config.hidden_size
        adanorm_num_embeddings = len(config.bandwidths)
        self.weight = nn.Parameter(torch.ones(adanorm_num_embeddings, config.hidden_size))
        self.bias = nn.Parameter(torch.zeros(adanorm_num_embeddings, config.hidden_size))

    def forward(self, hidden_states: torch.Tensor, cond_embedding_id: torch.LongTensor):
        hidden_states = F.layer_norm(hidden_states, (self.hidden_size,), weight=None, bias=None, eps=self.eps)
        return hidden_states * self.weight[cond_embedding_id].unsqueeze(0) + self.bias[cond_embedding_id].unsqueeze(0)


class VocosConvNeXtBlock(nn.Module):
    """ConvNeXt block adapted for 1D convolutions in the Vocos architecture."""

    def __init__(self, config: VocosConfig):
        super().__init__()
        self.dwconv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.kernel_size,
            padding=config.padding,
            groups=config.hidden_size,
        )
        if config.use_adaptive_norm:
            self.norm = VocosAdaptiveLayerNorm(config)
        else:
            self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pwconv1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(config.intermediate_size, config.hidden_size)
        if config.layer_scale_init_value > 0:
            self.layer_scale_parameter = nn.Parameter(
                config.layer_scale_init_value * torch.ones(config.hidden_size), requires_grad=True
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
            config.input_channels, config.hidden_size, kernel_size=config.kernel_size, padding=config.padding
        )
        if config.use_adaptive_norm:
            self.norm = VocosAdaptiveLayerNorm(config)
        else:
            self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layers = nn.ModuleList([VocosConvNeXtBlock(config) for _ in range(config.num_layers)])
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

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


class VocosISTFT(nn.Module):
    """
    As in original Vocos code:
    https://github.com/gemelo-ai/vocos/blob/c859e3b7b534f3776a357983029d34170ddd6fc3/vocos/spectral_ops.py#L7

    Custom ISTFT implementation to support "same" padding as in Vocos.
    """

    def __init__(self, config: VocosConfig):
        super().__init__()
        if config.istft_padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = config.istft_padding
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.win_length = getattr(config, "win_length", config.n_fft)
        window = torch.hann_window(self.win_length)
        self.register_buffer("window", window)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Compute the Inverse Short Time Fourier Transform (ISTFT) of a complex spectrogram.

        Args:
            spec (Tensor): Input complex spectrogram of shape (B, N, T), where B is the batch size,
                  is the number of frequency bins, and T is the number of time frames.

        Returns:
            Tensor: Reconstructed time-domain signal of shape (B, L), where L is the length of the output signal.
        """
        if spec.dim() != 3:
            raise ValueError("Expected a 3D tensor as input")

        if self.padding == "center":
            # Fallback to pytorch native implementation
            return torch.istft(spec, self.n_fft, self.hop_length, self.win_length, self.window, center=True)

        elif self.padding == "same":
            # Custom implementation from Vocos codebase
            pad = (self.win_length - self.hop_length) // 2
            n_frames = spec.shape[-1]

            # Inverse FFT
            ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
            ifft = ifft * self.window[None, :, None]

            # Overlap and Add
            output_size = (n_frames - 1) * self.hop_length + self.win_length
            y = F.fold(
                ifft,
                output_size=(1, output_size),
                kernel_size=(1, self.win_length),
                stride=(1, self.hop_length),
            )[:, 0, 0, pad:-pad]

            # Window envelope
            window_sq = self.window.square().expand(1, n_frames, -1).transpose(1, 2)
            window_envelope = F.fold(
                window_sq,
                output_size=(1, output_size),
                kernel_size=(1, self.win_length),
                stride=(1, self.hop_length),
            ).squeeze()[pad:-pad]

            # Normalize
            if not (window_envelope > 1e-11).all():
                raise ValueError("Window envelope values are too small (<=1e-11)")
            return y / window_envelope

        else:
            raise ValueError("Padding must be 'center' or 'same'.")


class VocosISTFTHead(nn.Module):
    """
    As in original Vocos code:
    https://github.com/gemelo-ai/vocos/blob/c859e3b7b534f3776a357983029d34170ddd6fc3/vocos/heads.py#L26
    - Projects the hidden states to STFT coefficients (magnitude and phase)
    - Applies ISTFT to reconstruct the time-domain audio signal
    """

    def __init__(self, config: VocosConfig):
        super().__init__()
        self.out = torch.nn.Linear(config.hidden_size, config.n_fft + 2)
        self.istft = VocosISTFT(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ISTFTHead module.

        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
            Tensor: Predicted STFT coefficients of shape (B, L, N+2), where N is the number of frequency bins.
        """
        x_pred = self.out(x).transpose(1, 2)
        mag, p = x_pred.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(mag, max=1e2)  # safeguard to prevent excessively large magnitudes
        # wrapping happens here. These two lines produce real and imaginary value
        spectrogram_real = torch.cos(p)
        spectrogram_imag = torch.sin(p)
        spectrogram_complex = mag * (spectrogram_real + 1j * spectrogram_imag)
        audio = self.istft(spectrogram_complex)
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
            Output of [`VocosProcessor`] which can be either:
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
