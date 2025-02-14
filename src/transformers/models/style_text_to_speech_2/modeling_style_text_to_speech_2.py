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
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, logging
from .configuration_style_text_to_speech_2 import (
    StyleTextToSpeech2AcousticTextEncoderConfig,
    StyleTextToSpeech2PredictorConfig,
    StyleTextToSpeech2DecoderConfig,
    StyleTextToSpeech2Config,
)
from ..auto import AutoModel


logger = logging.get_logger(__name__)


@dataclass
class StyleTextToSpeech2PredictorOutput(ModelOutput):
    durations: torch.Tensor = None
    pitch: torch.Tensor = None
    energy: torch.Tensor = None


@dataclass
class StyleTextToSpeech2ModelOutput(ModelOutput):
    durations: Optional[torch.Tensor] = None
    pitch: Optional[torch.Tensor] = None
    energy: Optional[torch.Tensor] = None
    waveform: torch.Tensor = None


class StyleTextToSpeech2PretrainedModel(PreTrainedModel):
    config_class = StyleTextToSpeech2Config

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.constant_(param, 0.0)


class AcousticTextEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.kernel_size,
            padding=config.kernel_size // 2,
        )
        self.norm = nn.LayerNorm(config.hidden_size)
        self.leaky_relu_slope = config.leaky_relu_slope
        self.dropout = nn.Dropout(config.dropout)

        # apply weight norm
        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        weight_norm(self.conv)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states.transpose(1, -1))
        hidden_states = self.norm(hidden_states.transpose(1, -1))
        hidden_states = F.leaky_relu(hidden_states, self.leaky_relu_slope)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class StyleTextToSpeech2AcousticTextEncoder(StyleTextToSpeech2PretrainedModel):
    base_model_prefix = "acoustic_text_encoder"
    config_class = StyleTextToSpeech2AcousticTextEncoderConfig
    main_input_name = "input_ids"

    def __init__(self, config: StyleTextToSpeech2AcousticTextEncoderConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [AcousticTextEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.lstm = nn.LSTM(
            config.hidden_size,
            config.hidden_size // 2,
            batch_first=True,
            bidirectional=True,
        )

    def forward(
        self, 
        input_ids: torch.Tensor, 
        mask: Optional[torch.Tensor] = None, 
        input_lengths: Optional[torch.Tensor] = None
    ):
        if mask is None:
            mask = torch.full((*input_ids.shape, 1), 1, dtype=torch.int, device=input_ids.device)
        else:
            mask = mask.unsqueeze(-1) 

        if input_lengths is None:
            input_lengths = mask.sum(dim=1).view(-1).tolist()

        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states)
            hidden_states = hidden_states.masked_fill(~mask.bool(), 0)

        hidden_states = nn.utils.rnn.pack_padded_sequence(
            hidden_states, input_lengths, batch_first=True, enforce_sorted=False
        )
        self.lstm.flatten_parameters()
        hidden_states, _ = self.lstm(hidden_states)
        hidden_states, _ = nn.utils.rnn.pad_packed_sequence(hidden_states, batch_first=True)

        return hidden_states


class StyleTextToSpeech2ProsodicTextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert_model = AutoModel.from_config(config.prosodic_text_encoder_sub_model_config)
        self.proj_out = nn.Linear(
            config.prosodic_text_encoder_sub_model_config.hidden_size, 
            config.hidden_size
        )

    def forward(self, input_ids, mask=None):
        outputs = self.bert_model(input_ids, attention_mask=mask)
        return self.proj_out(outputs.last_hidden_state)


class StyleTextToSpeech2AdaLayerNorm(nn.Module):
    def __init__(self, hidden_size, style_size, use_instance_norm=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.style_size = style_size
        self.proj = nn.Linear(style_size, hidden_size * 2)
        self.use_instance_norm = use_instance_norm

    def forward(self, hidden_states, style):     
        hidden_style = self.proj(style)
        gamma, beta = torch.chunk(hidden_style, chunks=2, dim=-1)
        if self.use_instance_norm:
            hidden_states = F.instance_norm(hidden_states.transpose(1, -1)).transpose(1, -1)
        else:
            hidden_states = F.layer_norm(hidden_states, (self.hidden_size,))
        hidden_states = (1 + gamma) * hidden_states + beta
        return hidden_states
     

class StyleTextToSpeech2ProsodyEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lstm = nn.LSTM(config.hidden_size + config.style_hidden_size, config.hidden_size // 2, batch_first=True, bidirectional=True)
        self.ada_layer_norm = StyleTextToSpeech2AdaLayerNorm(config.hidden_size, config.style_hidden_size)
        self.dropout = nn.Dropout(config.prosody_encoder_dropout)

    def forward(self, hidden_states, style, input_lengths):
        hidden_states = nn.utils.rnn.pack_padded_sequence(
            hidden_states, input_lengths, batch_first=True, enforce_sorted=False
        )
        self.lstm.flatten_parameters()
        hidden_states, _ = self.lstm(hidden_states)
        hidden_states, _ = nn.utils.rnn.pad_packed_sequence(hidden_states, batch_first=True)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.ada_layer_norm(hidden_states, style)

        return hidden_states


class StyleTextToSpeech2ProsodyEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(
            [StyleTextToSpeech2ProsodyEncoderLayer(config) for _ in range(config.prosody_encoder_num_layers)]
        )  
    
    def _concat_style(self, hidden_states, style, mask):
        bsz, seq_len, _ = hidden_states.shape
        hidden_states = torch.cat(
            (hidden_states, style.expand(bsz, seq_len, -1)),
            dim=-1
        ) 
        hidden_states = hidden_states.masked_fill(~mask.bool(), 0)
        return hidden_states

    def forward(self, hidden_states, style, mask=None):
        bsz, seq_len, _ = hidden_states.shape
        if mask is None:
            mask = torch.full((bsz, seq_len, 1), 1, dtype=torch.int, device=hidden_states.device)
        else:
            mask = mask.unsqueeze(-1)

        input_lengths = mask.sum(dim=1).view(-1).tolist()

        hidden_states = self._concat_style(hidden_states, style, mask)
        for layer in self.layers:
            hidden_states = layer(hidden_states, style, input_lengths)
            hidden_states = self._concat_style(hidden_states, style, mask)

        return hidden_states
    

class StyleTextToSpeech2DurationProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lstm = nn.LSTM(config.hidden_size + config.style_hidden_size, config.hidden_size // 2, 1, batch_first=True, bidirectional=True)
        self.duration_proj = nn.Linear(config.hidden_size, config.duration_projector_max_duration)

    def forward(self, hidden_states, speed):
        hidden_states, _ = self.lstm(hidden_states)
        hidden_states = self.duration_proj(hidden_states)
        durations = torch.sigmoid(hidden_states).sum(axis=-1) / speed  
        durations = torch.round(durations).clamp(min=1).long().squeeze()
        return durations


class StyleTextToSpeech2AdainResBlock1d(nn.Module):
    def __init__(self, hidden_size_in, hidden_size_out, style_size, upsample=False, learned_shortcut=False, dropout_p=0.0):
        super().__init__()
        self.do_upsample = upsample

        self.conv1 = nn.Conv1d(hidden_size_in, hidden_size_out, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(hidden_size_out, hidden_size_out, kernel_size=3, stride=1, padding=1)
        self.norm1 = StyleTextToSpeech2AdaLayerNorm(hidden_size_in, style_size, use_instance_norm=True)
        self.norm2 = StyleTextToSpeech2AdaLayerNorm(hidden_size_out, style_size, use_instance_norm=True)
        
        # apply weight norm
        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        weight_norm(self.conv1)
        weight_norm(self.conv2)

        if upsample:
            self.upsample = lambda x: F.interpolate(x, scale_factor=2, mode='nearest')
            self.pool = nn.ConvTranspose1d(hidden_size_in, hidden_size_in, kernel_size=3, stride=2, groups=hidden_size_in, padding=1, output_padding=1)
            weight_norm(self.pool)
        else:
            self.upsample = lambda x: x
            self.pool = nn.Identity()
        
        if learned_shortcut:
            self.conv1_shortcut = nn.Conv1d(hidden_size_in, hidden_size_out, kernel_size=1, stride=1, padding=0, bias=False)
            weight_norm(self.conv1_shortcut)
        else:
            self.conv1_shortcut = nn.Identity()

        self.dropout = nn.Dropout(dropout_p)

    def _residual(self, hidden_states, style):
        hidden_states = self.norm1(hidden_states, style)
        hidden_states = F.leaky_relu(hidden_states, 0.2)
        hidden_states = self.pool(hidden_states.transpose(1, -1)).transpose(1, -1)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv1(hidden_states.transpose(1, -1)).transpose(1, -1)
        hidden_states = self.norm2(hidden_states, style)
        hidden_states = F.leaky_relu(hidden_states, 0.2)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states.transpose(1, -1)).transpose(1, -1)
        return hidden_states

    def _shortcut(self, hidden_states):
        hidden_states = self.upsample(hidden_states.transpose(1, -1))
        hidden_states = self.conv1_shortcut(hidden_states).transpose(1, -1)
        return hidden_states

    def forward(self, hidden_states, style):
        hidden_states_residual = self._residual(hidden_states, style)
        hidden_states_shortcut = self._shortcut(hidden_states)
        hidden_states = (hidden_states_residual + hidden_states_shortcut) / 2.0**0.5
        return hidden_states


class StyleTextToSpeech2ProsodyBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.adain_res_1d_1 = StyleTextToSpeech2AdainResBlock1d(
            config.hidden_size, 
            config.hidden_size,
            config.style_hidden_size, 
            dropout_p=config.prosody_predictor_dropout,
        )
        self.adain_res_1d_2 = StyleTextToSpeech2AdainResBlock1d(
            config.hidden_size, 
            config.hidden_size // 2, 
            config.style_hidden_size, 
            dropout_p=config.prosody_predictor_dropout,
            upsample=True, 
            learned_shortcut=True,
        )
        self.adain_res_1d_3 = StyleTextToSpeech2AdainResBlock1d(
            config.hidden_size // 2, 
            config.hidden_size // 2, 
            config.style_hidden_size, 
            dropout_p=config.prosody_predictor_dropout
        )
        self.conv_out = nn.Conv1d(config.hidden_size // 2, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, hidden_states, style):
        hidden_states = self.adain_res_1d_1(hidden_states, style)
        hidden_states = self.adain_res_1d_2(hidden_states, style)
        hidden_states = self.adain_res_1d_3(hidden_states, style)
        return self.conv_out(hidden_states.transpose(1, -1))


class StyleTextToSpeech2ProsodyPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lstm = nn.LSTM(config.hidden_size + config.style_hidden_size, config.hidden_size // 2, 1, batch_first=True, bidirectional=True)
        self.pitch_block = StyleTextToSpeech2ProsodyBlock(config)
        self.energy_block = StyleTextToSpeech2ProsodyBlock(config)
    
    def forward(self, hidden_states, style):
        hidden_states, _ = self.lstm(hidden_states)
        pitch = self.pitch_block(hidden_states, style)
        energy = self.energy_block(hidden_states, style)
        return pitch, energy


class StyleTextToSpeech2AdainResBlockLayer(nn.Module):
    def __init__(self, channels, style_size, kernel_size, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation, padding=self._get_padding(kernel_size, dilation))
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=self._get_padding(kernel_size, 1))
        self.norm1 = StyleTextToSpeech2AdaLayerNorm(channels, style_size, use_instance_norm=True)
        self.norm2 = StyleTextToSpeech2AdaLayerNorm(channels, style_size, use_instance_norm=True)
        self.alpha1 = nn.Parameter(torch.ones(channels))
        self.alpha2 = nn.Parameter(torch.ones(channels))

        # apply weight norm
        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        weight_norm(self.conv1)
        weight_norm(self.conv2)

    def _get_padding(self, kernel_size, dilation):
        return int((kernel_size * dilation - dilation) / 2)
    
    def forward(self, hidden_states, style):
        x = self.norm1(hidden_states, style)
        x = x + (1 / self.alpha1) * (torch.sin(self.alpha1 * x) ** 2)
        x = self.conv1(x.transpose(1, -1)).transpose(1, -1)
        x = self.norm2(x, style)
        x = x + (1 / self.alpha2) * (torch.sin(self.alpha2 * x) ** 2)
        x = self.conv2(x.transpose(1, -1)).transpose(1, -1)
        return x + hidden_states


class StyleTextToSpeech2AdainResBlock(nn.Module):
    def __init__(self, channels, style_size, kernel_size, dilations):
        super().__init__()
        self.layers = nn.ModuleList([
            StyleTextToSpeech2AdainResBlockLayer(channels, style_size, kernel_size, dilation) for dilation in dilations
        ])

    def forward(self, hidden_states, style):
        for layer in self.layers:
            hidden_states = layer(hidden_states, style)
        return hidden_states


class StyleTextToSpeech2HarmonicNoiseSourceFilter(nn.Module):
    """
    Harmonic plus Noise Neural Source Filter. 
    See: https://arxiv.org/abs/2309.09493
    Adapted from: https://github.com/yl4579/StyleTT2
    """
    def __init__(
        self,
        sampling_rate,
        upsample_scale,
        harmonic_num=0,
        sine_amplitude=0.1,
        add_noise_std=0.003,
        voiced_threshold=0,
    ):
        """
        Args:
            samp_rate: (`int`):
                Sampling rate in Hz.
            upsample_scale: (`int`):
                Upsampling scale.
            harmonic_num: (`int`, *optional*, defaults to 0):
                Number of harmonic overtones.
            sine_amplitude: (`float`, *optional*, defaults to 0.1):
                Amplitude of sine-waveform.
            add_noise_std: (`float`, *optional*, defaults to 0.003):
                Standard deviation of Gaussian noise.
            voiced_threshold: (`float`, *optional*, defaults to 0):
                F0 threshold for U/V classification.     
        """
        super().__init__()
        self.sampling_rate = sampling_rate
        self.upsample_scale = upsample_scale
        self.harmonic_num = harmonic_num
        self.sine_amplitude = sine_amplitude
        self.add_noise_std = add_noise_std
        self.voiced_threshold = voiced_threshold

        # to merge source harmonics into a single excitation
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)

    def _f02sine(self, f0_values):
        rad_values = (f0_values / self.sampling_rate) % 1
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] += rand_ini

        rad_values = F.interpolate(rad_values.transpose(1, 2), scale_factor= 1 / self.upsample_scale, mode="linear").transpose(1, 2)
        phase = rad_values.cumsum(dim=1) * 2 * torch.pi
        phase = F.interpolate(phase.transpose(1, 2)  * self.upsample_scale, scale_factor=self.upsample_scale, mode="linear").transpose(1, 2)
        sines = phase.sin()

        return sines
    
    def _sine_gen(self, f0):
        # generate sine waveforms
        fn = f0 * torch.arange(1, self.harmonic_num + 2, device=f0.device)
        sine_waves = self._f02sine(fn) * self.sine_amplitude

        # generate uv signal
        uv = (f0 > self.voiced_threshold).float()
        noise_amp = uv * self.add_noise_std + (1 - uv) * self.sine_amplitude / 3
        noise = noise_amp * torch.randn_like(sine_waves)
        sine_waves = sine_waves * uv + noise

        return sine_waves
        
    def forward(self, hidden_states):
        with torch.no_grad():
            sine_wavs = self._sine_gen(hidden_states)
        sine_merge = F.tanh(self.l_linear(sine_wavs))

        return sine_merge


class StyleTextToSpeech2GeneratorLayer(nn.Module):
    def __init__(self, layer_idx, style_size, upsample_rates, resblock_kernel_sizes, upsample_initial_channel, resblock_dilation_sizes, upsample_kernel_sizes, n_fft, reflection_pad=False):
        super().__init__()
        self.layer_idx = layer_idx

        c_cur = upsample_initial_channel // (2 ** (layer_idx + 1))
        if layer_idx + 1 < len(upsample_rates):
            noise_conv_stride = math.prod(upsample_rates[layer_idx + 1:])
            noise_conv_padding = (noise_conv_stride + 1) // 2
            noise_conv_kernel_size = noise_conv_stride * 2
            noise_res_kernel_size = 7
        else:
            noise_conv_stride = 1
            noise_conv_padding = 0
            noise_conv_kernel_size = 1
            noise_res_kernel_size = 11

        self.up = nn.ConvTranspose1d(
            upsample_initial_channel // (2**layer_idx), 
            c_cur,
            upsample_kernel_sizes[layer_idx], 
            upsample_rates[layer_idx],
            padding=(upsample_kernel_sizes[layer_idx] - upsample_rates[layer_idx])//2
        )

        self.noise_conv = nn.Conv1d(
            n_fft + 2, 
            c_cur, 
            kernel_size=noise_conv_kernel_size, 
            stride=noise_conv_stride, 
            padding=noise_conv_padding
        )

        self.noise_res = StyleTextToSpeech2AdainResBlock(
            c_cur, 
            style_size, 
            noise_res_kernel_size, 
            (1, 3, 5)
        )

        self.resblocks = nn.ModuleList([
            StyleTextToSpeech2AdainResBlock(
                c_cur, 
                style_size, 
                kernel_size, 
                dilation
            )
            for kernel_size, dilation in zip(resblock_kernel_sizes, resblock_dilation_sizes)
        ])

        self.reflection_pad = nn.ReflectionPad1d((1, 0)) if reflection_pad else nn.Identity()

        # apply weight norm
        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        weight_norm(self.up)

    def forward(self, hidden_states, hidden_states_source, style):
        hidden_states = F.leaky_relu(hidden_states, 0.1)
        hidden_states_source = self.noise_conv(hidden_states_source.transpose(1, -1)).transpose(1, -1)
        hidden_states_source = self.noise_res(hidden_states_source, style)

        hidden_states = self.up(hidden_states.transpose(1, -1))
        hidden_states = self.reflection_pad(hidden_states).transpose(1, -1)
        hidden_states = hidden_states + hidden_states_source
        hidden_states = sum(resblock(hidden_states, style) for resblock in self.resblocks) / len(self.resblocks)

        return hidden_states


class StyleTextToSpeech2Generator(nn.Module):
    def __init__(
        self,
        style_size,
        resblock_kernel_sizes,
        upsample_rates,
        upsample_initial_channel,
        resblock_dilation_sizes,
        upsample_kernel_sizes,
        n_fft,
        hop_size,
        sampling_rate
    ):
        super().__init__()

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.n_fft = n_fft
        self.hop_length = hop_size
        self.win_length = n_fft
        self.window = torch.hann_window(n_fft)
        self.scale_factor = math.prod(upsample_rates) * hop_size

        self.f0_upsamp = nn.Upsample(scale_factor=self.scale_factor)
        self.m_source = StyleTextToSpeech2HarmonicNoiseSourceFilter(
            sampling_rate=sampling_rate,
            upsample_scale=self.scale_factor,
            harmonic_num=8, 
            voiced_threshold=10
        )

        self.layers = nn.ModuleList([
            StyleTextToSpeech2GeneratorLayer(
                layer_idx, 
                style_size, 
                upsample_rates, 
                resblock_kernel_sizes, 
                upsample_initial_channel, 
                resblock_dilation_sizes, 
                upsample_kernel_sizes, 
                n_fft, 
                reflection_pad=True if layer_idx == len(upsample_rates) - 1 else False
            )
            for layer_idx in range(len(upsample_rates))
        ])    
                
        self.conv_post = nn.Conv1d(
            upsample_initial_channel // (2 ** (len(upsample_rates))), 
            n_fft + 2,
            7, 
            padding=3
        )

        # apply weight norm
        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        weight_norm(self.conv_post)
        
    def forward(self, hidden_states, style, f0):
        with torch.no_grad():
            f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # bs,n,t
            har_source = self.m_source(f0)
            har_source = har_source.transpose(1, 2).squeeze(1)
            har_transform = torch.stft(
                har_source,
                self.n_fft,
                self.hop_length,
                self.win_length,
                self.window.to(har_source.device),
                return_complex=True
            )
            har_spec, har_phase = har_transform.abs(), har_transform.angle()
            har = torch.cat([har_spec, har_phase], dim=1).transpose(1, -1)

        for layer in self.layers:
            hidden_states = layer(hidden_states, har, style)

        hidden_states = F.leaky_relu(hidden_states)
        hidden_states = self.conv_post(hidden_states.transpose(1, -1))
        spec = torch.exp(hidden_states[:,:self.n_fft // 2 + 1, :])
        phase = torch.sin(hidden_states[:, self.n_fft // 2 + 1:, :])
        inverse_transform = torch.istft(
            spec * torch.exp(phase * 1j),
            self.n_fft,
            self.hop_length,
            self.win_length,
            window=self.window.to(har_source.device)
        )
        return inverse_transform


class StyleTextToSpeech2Predictor(StyleTextToSpeech2PretrainedModel):
    base_model_prefix = "predictor"
    config_class = StyleTextToSpeech2PredictorConfig
    main_input_name = "input_ids"

    def __init__(self, config):
        super().__init__(config)
        self.prosodic_text_encoder = StyleTextToSpeech2ProsodicTextEncoder(config)
        self.prosody_encoder = StyleTextToSpeech2ProsodyEncoder(config)
        self.duration_projector = StyleTextToSpeech2DurationProjector(config)
        self.prosody_predictor = StyleTextToSpeech2ProsodyPredictor(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        prosodic_style: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        speed: Optional[float] = 1.0,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # encode prosody
        prosodic_text_hidden_states = self.prosodic_text_encoder(input_ids, mask)
        prosodic_hidden_states = self.prosody_encoder(prosodic_text_hidden_states, prosodic_style, mask)

        # predict durations
        durations = self.duration_projector(prosodic_hidden_states, speed)

        # predic pitch and energy
        prosodic_hidden_states = prosodic_hidden_states.repeat_interleave(durations, dim=1)
        pitch, energy = self.prosody_predictor(prosodic_hidden_states, prosodic_style)

        if not return_dict:
            return (durations, pitch, energy)

        return StyleTextToSpeech2PredictorOutput(
            pitch=pitch,
            energy=energy,
            durations=durations
        )


class StyleTextToSpeech2Decoder(StyleTextToSpeech2PretrainedModel):
    """
    iSTFTNet based decoder
    """
    base_model_prefix = "decoder"
    config_class = StyleTextToSpeech2DecoderConfig
    main_input_name = "hidden_states" # ???

    def __init__(self, config):
        super().__init__(config)
        self.pitch_conv = nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1)
        self.energy_conv = nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1)
        self.acoustic_encoder = StyleTextToSpeech2AdainResBlock1d(config.hidden_size + 2, 1024, config.style_hidden_size, learned_shortcut=True)
        self.acoustic_residual = nn.Conv1d(512, 64, kernel_size=1)
        self.decoder = nn.ModuleList([
            StyleTextToSpeech2AdainResBlock1d(1024 + 2 + 64, 1024, config.style_hidden_size, learned_shortcut=True),
            StyleTextToSpeech2AdainResBlock1d(1024 + 2 + 64, 1024, config.style_hidden_size, learned_shortcut=True),
            StyleTextToSpeech2AdainResBlock1d(1024 + 2 + 64, 1024, config.style_hidden_size, learned_shortcut=True),
            StyleTextToSpeech2AdainResBlock1d(1024 + 2 + 64, 512, config.style_hidden_size, learned_shortcut=True, upsample=True)
        ])
        self.generator = StyleTextToSpeech2Generator(
            config.style_hidden_size, 
            config.resblock_kernel_sizes, 
            config.upsample_rates, 
            config.upsample_initial_channel, 
            config.resblock_dilation_sizes, 
            config.upsample_kernel_sizes, 
            config.gen_istft_n_fft, 
            config.gen_istft_hop_size,
            config.sampling_rate
        )

        # apply weight norm
        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        weight_norm(self.pitch_conv)
        weight_norm(self.energy_conv)
        weight_norm(self.acoustic_residual)

    def forward(
        self,
        acoustic_hidden_states: torch.Tensor, 
        pitch: torch.Tensor, 
        energy: torch.Tensor, 
        style: torch.Tensor
    ) -> torch.Tensor:
        
        pitch_processed = self.pitch_conv(pitch).transpose(1, -1)
        energy_processed = self.energy_conv(energy).transpose(1, -1)

        acoustic_hidden_states_res = self.acoustic_residual(acoustic_hidden_states.transpose(1, -1)).transpose(1, -1)
        acoustic_hidden_states = torch.cat([acoustic_hidden_states, pitch_processed, energy_processed], dim=-1)
        acoustic_hidden_states = self.acoustic_encoder(acoustic_hidden_states, style)
        
        residual = True
        for block in self.decoder:
            if residual:
                acoustic_hidden_states = torch.cat([acoustic_hidden_states, acoustic_hidden_states_res, pitch_processed, energy_processed], dim=-1)
            acoustic_hidden_states = block(acoustic_hidden_states, style)
            if block.do_upsample:
                residual = False
        
        waveform = self.generator(acoustic_hidden_states, style, pitch.squeeze(0))
        return waveform


class StyleTextToSpeech2Model(StyleTextToSpeech2PretrainedModel):
    config_class = StyleTextToSpeech2Config
    
    def __init__(self, config):
        super().__init__(config)
        self.acoustic_text_encoder = StyleTextToSpeech2AcousticTextEncoder(config.acoustic_text_encoder_config)
        self.predictor = StyleTextToSpeech2Predictor(config.predictor_config)
        self.decoder = StyleTextToSpeech2Decoder(config.decoder_config)

    @torch.no_grad()
    def generate(
        self, 
        input_ids: torch.Tensor,
        style: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        speed: Optional[float] = 1.0,
        return_dict: Optional[bool] = None,
        output_durations: Optional[bool] = None,
        output_pitch: Optional[bool] = None,
        output_energy: Optional[bool] = None,
    ) -> Union[Tuple, StyleTextToSpeech2ModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        prosodic_style = style[:, self.config.style_size // 2:]
        acoustic_style = style[:, :self.config.style_size // 2]

        # predict durations, pitch and energy
        predictor_output = self.predictor(input_ids, prosodic_style, mask, speed)
        if not return_dict:
            durations, pitch, energy = predictor_output
        else:
            durations = predictor_output.durations
            pitch = predictor_output.pitch
            energy = predictor_output.energy

        # encode and align phoneme representations
        acoustic_hidden_states = self.acoustic_text_encoder(input_ids)
        acoustic_aligned_hidden_states = acoustic_hidden_states.repeat_interleave(durations, dim=1)

        # decode waveform
        waveform = self.decoder(acoustic_aligned_hidden_states, pitch, energy, acoustic_style)

        outputs = {
            "durations": durations if output_durations else None,
            "pitch": pitch if output_pitch else None,
            "energy": energy if output_energy else None,
            "waveform": waveform,
        }

        if not return_dict:
            return (output for output in outputs.values() if output is not None)

        return StyleTextToSpeech2ModelOutput(**outputs)


__all__ = ["StyleTextToSpeech2Predictor", "StyleTextToSpeech2Decoder", "StyleTextToSpeech2Model"]
