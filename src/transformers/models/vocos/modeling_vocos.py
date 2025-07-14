# coding=utf-8
# Copyright 2024 Descript and The HuggingFace Inc. team. All rights reserved.
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

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring
from ..auto import AutoModel
from .configuration_vocos import VocosConfig, VocosWithEncodecConfig


class VocosLayerNorm(nn.Module):
    def __init__(self, config: VocosConfig):
        super().__init__()
        self.eps = config.layer_norm_eps
        self.hidden_dim = config.hidden_dim
        self.use_adaptive_norm = config.use_adaptive_norm
        if self.use_adaptive_norm:
            # only used in Encodec variant
            self.weight = nn.Parameter(torch.ones(config.adanorm_num_embeddings, config.hidden_dim))
            self.bias = nn.Parameter(torch.zeros(config.adanorm_num_embeddings, config.hidden_dim))
        else:
            self.weight = nn.Parameter(torch.ones(config.hidden_dim))
            self.bias = nn.Parameter(torch.zeros(config.hidden_dim))

    def forward(self, hidden_states: torch.Tensor, cond_embedding_id: torch.LongTensor = None):
        if self.use_adaptive_norm:
            if cond_embedding_id is None:
                # the index used to select the target bandwidth is used to to index into Adaptive Normalization lookup table (adanorm_num_embeddings, hidden_dim)
                raise ValueError(
                    "When using adaptive LayerNorm `use_adaptive_norm=True`, you must pass conditional id via `bandwidth_id`."
                )

            hidden_states = F.layer_norm(hidden_states, (self.hidden_dim,), weight=None, bias=None, eps=self.eps)
            return hidden_states * self.weight[cond_embedding_id].unsqueeze(0) + self.bias[
                cond_embedding_id
            ].unsqueeze(0)

        return F.layer_norm(hidden_states, (self.hidden_dim,), weight=self.weight, bias=self.bias, eps=self.eps)


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
        self.norm = VocosLayerNorm(config)
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
        hidden_states = self.norm(hidden_states, cond_embedding_id=bandwidth_id)
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
        self.norm = VocosLayerNorm(config)
        self.layers = nn.ModuleList([VocosConvNeXtBlock(config) for _ in range(config.num_layers)])
        self.final_layer_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states, bandwidth_id=None):
        hidden_states = self.embed(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.norm(hidden_states, bandwidth_id)
        hidden_states = hidden_states.transpose(1, 2)
        for layer in self.layers:
            hidden_states = layer(hidden_states, bandwidth_id)
        hidden_states = self.final_layer_norm(hidden_states.transpose(1, 2))
        return hidden_states


class VocosISTFT(nn.Module):
    """
    Performs the Inverse Short Time Fourier Transform (ISTFT) on a STFT coefficients to reconstruct audio in the time domain.
    It computes ISTFT differently depending on padding:
        if `center` : uses PyTorch's built-in ISTFT implementation since it uses `center=True` by default.
        if `same` : uses custom implementation of ISTFT with the overlap-add method, since the Pytorch version fails the
        Nonzero Overlap Add (NOLA) condition when center is False. See issue: https://github.com/pytorch/pytorch/issues/62323
    """

    def __init__(self, config: VocosConfig):
        super().__init__()
        if config.spec_padding not in ["center", "same"]:
            raise ValueError("padding must be `center` or `same`")
        self.padding = config.spec_padding
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.win_length = config.n_fft
        window = torch.hann_window(self.win_length)
        self.register_buffer("window", window)

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        if self.padding == "center":
            audio = torch.istft(
                spectrogram,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=True,
            )

        else:
            batch_size, num_freq_bins, num_time_frames = spectrogram.shape
            pad = (self.win_length - self.hop_length) // 2
            # the inverse FFT of each frame
            inverse_fft = torch.fft.irfft(spectrogram, n=self.n_fft, dim=1, norm="backward")
            inverse_fft = inverse_fft * self.window[None, :, None]

            # combine the overlapping frame with windowing and normalizing by the sum of squared window values across overlapping frames
            # to make sure the reconstruction of the audio is accurate
            output_length = (num_time_frames - 1) * self.hop_length + self.win_length
            audio = F.fold(
                inverse_fft,
                output_size=(1, output_length),
                kernel_size=(1, self.win_length),
                stride=(1, self.hop_length),
            )[:, 0, 0, pad:-pad]
            window_sqrt = self.window.square().expand(1, num_time_frames, -1).transpose(1, 2)
            norm = F.fold(
                window_sqrt,
                output_size=(1, output_length),
                kernel_size=(1, self.win_length),
                stride=(1, self.hop_length),
            ).squeeze()[pad:-pad]

            if torch.any(norm <= 1e-11):
                raise ValueError(
                    "Normalization tensor `norm` contains values ≤ 1e-11, it would cause division by zero. check the n_fft, hop_length and padding parameters."
                )
            audio = audio / norm

        return audio


class VocosISTFTHead(nn.Module):
    """
    Projects hidden states to magnitude and phase predictions, combines them into complex
    STFT coefficients, and applies ISTFT to reconstruct the audio waveform.
    """

    def __init__(self, config: VocosConfig):
        super().__init__()
        self.out_proj = nn.Linear(config.hidden_dim, config.n_fft + 2)
        self.istft = VocosISTFT(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        spectrogram = self.out_proj(hidden_states)
        spectrogram = spectrogram.transpose(1, 2)
        magnitude, phase = spectrogram.chunk(2, dim=1)
        # safeguard to prevent excessively large magnitudes
        magnitude = torch.exp(magnitude).clamp(max=1e2)
        real = torch.cos(phase)
        imag = torch.sin(phase)
        stft_complex_coeffs = magnitude * (real + 1j * imag)
        audio = self.istft(stft_complex_coeffs)
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
        elif isinstance(module, VocosLayerNorm):
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.data.fill_(1.0)


@auto_docstring(
    custom_intro="""
    Main Vocos model for neural vocoding. This model takes mel-spectrogram or other acoustic feature inputs and generates high-quality
    audio waveforms.
    """
)
class VocosModel(VocosPreTrainedModel):
    config_class = VocosConfig
    base_model_prefix = "vocos"

    def __init__(self, config: VocosConfig):
        super().__init__(config)
        self.backbone = VocosBackbone(config)
        self.head = VocosISTFTHead(config)
        self.post_init()

    @auto_docstring
    def forward(self, features: torch.FloatTensor) -> torch.FloatTensor:
        r"""
        Convert input spectrogram features to audio waveform.
        Args:
            features (`torch.FloatTensor`):
                Mel-spectrogram of shape (batch_size, n_mels, num_time_frames).

        Returns:
            `torch.FloatTensor` of shape (batch_size, time): Reconstructed audio waveform .

        Example:

        ```python
        >>> from datasets import load_dataset, Audio
        >>> from transformers import VocosModel, VocosFeatureExtractor

        >>> model = VocosModel.from_pretrained("Manel/Vocos")
        >>> feature_extractor = VocosFeatureExtractor.from_pretrained("Manel/Vocos")

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
        >>> audio_sample= ds[0]["audio"]["array"]

        >>> inputs = feature_extractor(audio_sample, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")
        >>> audio = model(inputs.input_features)
        ```
        """
        hidden_states = self.backbone(features)
        audio = self.head(hidden_states)
        return audio


@auto_docstring(
    custom_intro="""
    Vocos model with integrated EnCodec neural audio codec for end-to-end audio compression and reconstruction.
    It accepts raw audio or pre-computed EnCodec RVQ codes (quantized audio features) and reconstructs a higher quality audio.
    """
)
class VocosWithEncodecModel(VocosPreTrainedModel):
    config_class = VocosWithEncodecConfig

    def __init__(self, config: VocosWithEncodecConfig):
        super().__init__(config)

        self.encodec_model = AutoModel.from_config(config.encodec_config)
        self.num_quantizers = self.encodec_model.quantizer.get_num_quantizers_for_bandwidth(
            bandwidth=max(config.bandwidths)
        )
        codebook_weights = torch.cat(
            [layer.codebook.embed for layer in self.encodec_model.quantizer.layers[: self.num_quantizers]], dim=0
        )
        self.codebook_weights = nn.Parameter(codebook_weights, requires_grad=config.train_codebooks)
        self.backbone = VocosBackbone(config)
        self.head = VocosISTFTHead(config)
        self.post_init()

    def _audio_codes_to_features(self, codes: torch.LongTensor) -> torch.FloatTensor:
        r"""
        Transform a sequence of discrete tokens (codes) into feature embeddings using codebook weights.

        Args:
            codes (torch.LongTensor): Discrete codes of shape (n_codebooks, seq_len) or
                (n_codebooks, batch_size, seq_len).

        Returns:
            torch.FloatTensor: Feature embeddings of shape (batch_size, hidden_dim, seq_len).
        """
        if codes.dim() == 2:
            codes = codes.unsqueeze(1)
        num_bins = self.encodec_model.quantizer.codebook_size
        offsets = torch.arange(0, num_bins * codes.size(0), num_bins, device=codes.device).reshape(-1, 1, 1)
        embeddings_idxs = codes + offsets
        features = F.embedding(embeddings_idxs, self.codebook_weights).sum(dim=0)
        return features.transpose(1, 2)

    @auto_docstring
    def forward(
        self,
        bandwidth_id: torch.LongTensor,
        audio: Optional[torch.FloatTensor] = None,
        codes: Optional[torch.LongTensor] = None,
        return_codes: bool = False,
    ):
        r"""
        Forward pass through the VocosWithEncodec model, it accepts raw audio or discrete codes from EnCodec model
        and returns a reconstructed audio and optionally the Encodec quantized codes.


        Args:
            bandwidth_id (torch.LongTensor):
                Index in [0, 1, 2, 3] used to select the desired bandwidth for EnCodec
                quantizer [1.5, 3, 6, 12] kbps respectively. This determines
                the number of RVQ codebooks used[2, 4, 6, 8].
            audio (torch.FloatTensor, optional):
                Raw audio input of shape (batch_size, n_samples).
            codes (torch.LongTensor, optional):
                Pre-computed RVQ discrete codes of shape
                (n_codebooks, seq_len) or (n_codebooks, batch_size, seq_len).
            return_codes (bool):
                Whether to return the codes along with reconstructed audio.

        Returns:
            torch.FloatTensor:
                Reconstructed audio waveform of shape (batch_size, n_samples).
                tuple: If return_codes=True, returns (audio, codes).

        Example:

        ```python
        >>> import torch
        >>> from datasets import load_dataset, Audio
        >>> from transformers import VocosWithEncodecModel, VocosWithEncodecConfig

        >>> model = VocosWithEncodecModel.from_pretrained("Manel/Vocos-Encodec")
        >>> config = VocosWithEncodecConfig.from_pretrained("Manel/Vocos-Encodec")

        >>> # load audio sample
        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.cast_column("audio", Audio(sampling_rate=24000))

        >>> # reconstructing audio from raw audio
        >>> bandwidth_id = torch.tensor([0], dtype=torch.long)
        >>> audio_sample = torch.tensor(ds[0]["audio"]["array"], dtype=torch.float32).unsqueeze(0)
        >>> audio = model(audio=audio_sample, bandwidth_id=bandwidth_id)

        >>> # reconstructing audio from encoded codes
        >>> codes = torch.randint(low=0, high=1024, size=(8, 200))
        >>> audio = model(codes=codes, bandwidth_id=bandwidth_id)
        ```
        """
        if audio is None and codes is None:
            raise ValueError("One of `audio` or `codes` must be provided as input.")

        if codes is not None and codes.ndim not in (2, 3):
            raise ValueError(
                f"`codes` must have shape (num_codebooks, sequence_length) or (num_codebooks, batch_size, sequence_length), but got {codes.shape}."
            )

        if audio is not None:
            embedding = self.encodec_model.encoder(audio.unsqueeze(1))
            bandwidth = self.config.encodec_config.target_bandwidths[bandwidth_id.item()]
            codes = self.encodec_model.quantizer.encode(embedding, bandwidth=bandwidth)

        hidden_states = self._audio_codes_to_features(codes)
        hidden_states = self.backbone(hidden_states, bandwidth_id)
        reconstructed_audio = self.head(hidden_states)

        if return_codes:
            return reconstructed_audio, codes
        return reconstructed_audio


__all__ = ["VocosModel", "VocosWithEncodecModel", "VocosPreTrainedModel"]
