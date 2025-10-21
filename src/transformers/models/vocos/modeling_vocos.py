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

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring, can_return_tuple
from .configuration_vocos import VocosConfig


@dataclass
class VocosOutput(ModelOutput):
    """
    Args:
        audio (`torch.FloatTensor` of shape `(batch_size, time)`):
            Reconstructed audio waveform.
    """

    audio: torch.FloatTensor


def vocos_istft(input, n_fft: int, padding=None, **kwargs) -> "torch.Tensor":
    """
    Performs the Inverse Short Time Fourier Transform (ISTFT) on STFT coefficients to reconstruct audio in the time domain.

    Adds support for `same` padding as in Vocos:
    https://github.com/gemelo-ai/vocos/blob/c859e3b7b534f3776a357983029d34170ddd6fc3/vocos/spectral_ops.py#L7

    Otherwise falls back to PyTorch's built-in ISTFT implementation `torch.istft`.

    TODO (ebezzam): with sufficient tests, this more general function could be moved to `src/transformers/audio_utils.py`.

    Args:
        input (`torch.Tensor`): Complex-valued STFT coefficients of shape (batch_size, freq_bins, time_frames).
        n_fft (`int`): Size of the FFT.
        padding (`str`, *optional*): Padding mode. Either "center" or "same".
        **kwargs: Additional arguments passed to torch.istft or used for "same" padding:
            - win_length (`int`, *optional*): Window length. Defaults to n_fft.
            - hop_length (`int`, *optional*): Hop length. Defaults to n_fft // 4.
            - window (`torch.Tensor`, *optional*): Window function. Defaults to Hann window.
            - center (`bool`, *optional*): Used only for "center" padding mode.

    Returns:
        `torch.Tensor`: Reconstructed audio waveform.

    It computes ISTFT differently depending on padding:
        if `center` : uses PyTorch's built-in ISTFT implementation since it uses `center=True` by default.
        if `same` : uses custom implementation of ISTFT with the overlap-add method, since the Pytorch version fails the
        Nonzero Overlap Add (NOLA) condition when center is False. See issue: https://github.com/pytorch/pytorch/issues/62323
    """

    if padding == "center" or padding is None:
        # user may provide center=False in kwargs
        center = kwargs.get("center", True)
        audio = torch.istft(
            input,
            n_fft=n_fft,
            center=center,
            **kwargs,
        )

    elif padding == "same":
        win_length = kwargs.get("win_length", n_fft)
        hop_length = kwargs.get("hop_length", n_fft // 4)
        window = kwargs.get("window", torch.hann_window(win_length))

        _, _, num_time_frames = input.shape
        pad = (win_length - hop_length) // 2
        # the inverse FFT of each frame
        inverse_fft = torch.fft.irfft(input, n=n_fft, dim=1, norm="backward")
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

    else:
        raise ValueError(f"Unsupported padding mode: {padding}. Supported modes are 'center' and 'same'.")

    return audio


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
        self.act_fn = nn.GELU()
        self.pwconv2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layer_scale_parameter = nn.Parameter(
            config.layer_scale_init_value * torch.ones(config.hidden_size), requires_grad=True
        )

    def forward(self, hidden_states: torch.Tensor, bandwidth_id: Optional[torch.LongTensor] = None) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.dwconv(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        if isinstance(self.norm, VocosAdaptiveLayerNorm):
            hidden_states = self.norm(hidden_states, cond_embedding_id=bandwidth_id)
        else:
            hidden_states = self.norm(hidden_states)
        hidden_states = self.pwconv1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.pwconv2(hidden_states)
        hidden_states = self.layer_scale_parameter.to(hidden_states.device) * hidden_states
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = residual.to(hidden_states.device) + hidden_states
        return hidden_states


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
        # ISTFT parameters
        if config.istft_padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = config.istft_padding
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.win_length = getattr(config, "win_length", config.n_fft)
        window = torch.hann_window(self.win_length)
        self.register_buffer("window", window)

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
        audio = vocos_istft(
            spectrogram_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            padding=self.padding,
        )
        return audio


class VocosPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = VocosConfig
    base_model_prefix = "vocos"
    main_input_name = "audio_spectrogram"
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
    def __init__(self, config: VocosConfig):
        super().__init__(config)
        
        # `VocosBackbone` in original: https://github.com/gemelo-ai/vocos/blob/c859e3b7b534f3776a357983029d34170ddd6fc3/vocos/models.py#L26
        self.embed = nn.Conv1d(
            config.n_mels, config.hidden_size, kernel_size=config.kernel_size, padding=config.padding
        )
        if config.use_adaptive_norm:
            self.norm = VocosAdaptiveLayerNorm(config)
        else:
            self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layers = nn.ModuleList([VocosConvNeXtBlock(config) for _ in range(config.num_layers)])
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Decoder (Linear + ISTFT)
        self.decoder = VocosISTFTHead(config)
        self._bandwidth_to_id = {bandwidth: id for id, bandwidth in enumerate(config.bandwidths)}
        
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        audio_spectrogram: Optional[torch.FloatTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        bandwidth: Optional[float] = None,
        padding_mask: Optional[torch.BoolTensor] = None,
    ) -> Union[VocosOutput, tuple[torch.FloatTensor]]:
        r"""
        audio_spectrogram (`torch.FloatTensor` of shape `(batch_size, feature_dim, time_dim)`):
            Mel-spectrogram features: computed directly from audio via (`processor(audio=waveform)`).
        input_features (`torch.FloatTensor` of shape `(batch_size, feature_dim, time_dim)`):
            EnCodec neural audio codec features which can be computed either from precomputed EnCodec RVQ codes via
            `processor(codes=codes, bandwidth=1.5)` or from raw audio via `processor(audio=waveform, bandwidth=1.5)`.
            It must be provided with the corresponding EnCodec bandwidth.
        bandwidth (`float`, *optional*):
            Target bandwidth for EnCodec quantizer, e.g. one of [1.5, 3, 6, 12] kbps, to be provided if
            `input_features`is not None.
        padding_mask (`torch.BoolTensor` of shape `(batch_size, time_dim)`, *optional*):
            Padding mask. Not used, but kept so processor outputs can be passed directly.

        Returns:
            `VocosOutput` or tuple `(audio,)`:
            - `audio` of shape (batch_size, time): Reconstructed audio waveform.

        Example:

        ```python
        >>> from datasets import load_dataset, Audio
        >>> from transformers import VocosProcessor, VocosModel

        >>> processor = VocosProcessor.from_pretrained("hf-audio/vocos-mel-24khz")
        >>> model = VocosModel.from_pretrained("hf-audio/vocos-mel-24khz")

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
        >>> audio_sample= ds[0]["audio"]["array"]

        >>> # extract mel-spectrogram features from audio and reconstruct high-quality audio
        >>> inputs = processor(audio=audio_sample)
        >>> outputs = model(**inputs)
        >>> reconstructed_audio = outputs.audio


        >>> # Encode audio using EnCodec neural codec and reconstruct from audio from that
        >>> processor = VocosProcessor.from_pretrained("hf-audio/vocos-encodec-24khz")
        >>> model = VocosModel.from_pretrained("hf-audio/vocos-encodec-24khz")

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

        if input_features is None and audio_spectrogram is None:
            raise ValueError("One of `input_features` or `audio_spectrogram` should be provided.")

        if input_features is not None:
            if bandwidth is None:
                raise ValueError("When passing `input_features`, `bandwidth` must be also be provided.")
            bandwidth_id = self._bandwidth_to_id[float(bandwidth)]
            hidden_states = input_features
        else:
            bandwidth_id = None
            hidden_states = audio_spectrogram
        
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
        
        # Decode back to audio (linear + ISTFT)
        audio = self.decoder(hidden_states)
        return VocosOutput(audio=audio)


__all__ = ["VocosModel", "VocosPreTrainedModel"]
