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
import copy
import math
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm, remove_weight_norm
import numpy as np

from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import (
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    is_torch_greater_or_equal_than_2_2,
    prune_linear_layer,
)
from ...utils import add_start_docstrings, logging
from .configuration_style_text_to_speech_2 import StyleTextToSpeech2Config
from ..albert.modeling_albert import AlbertModel, AlbertConfig


logger = logging.get_logger(__name__)

# to be replaced with https://github.com/huggingface/transformers/blob/4d3b1076a17b72f68c7332008b667c22e81d8f94/src/transformers/audio_utils.py#L1064
from scipy.signal import get_window
class TorchSTFT(torch.nn.Module):
    def __init__(self, filter_length=800, hop_length=200, win_length=800, window='hann'):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.from_numpy(get_window(window, win_length, fftbins=True).astype(np.float32))

    def transform(self, input_data):
        forward_transform = torch.stft(
            input_data,
            self.filter_length, self.hop_length, self.win_length, window=self.window.to(input_data.device),
            return_complex=True)

        return torch.abs(forward_transform), torch.angle(forward_transform)

    def inverse(self, magnitude, phase):
        inverse_transform = torch.istft(
            magnitude * torch.exp(phase * 1j),
            self.filter_length, self.hop_length, self.win_length, window=self.window.to(magnitude.device))

        return inverse_transform.unsqueeze(-2)  # unsqueeze to stay consistent with conv_transpose1d implementation

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction

class AcousticTextEncoderLayer(nn.Module):
    def __init__(self, config: StyleTextToSpeech2Config):
        super().__init__()
        self.conv = nn.Conv1d(
            config.acoustic_text_encoder_hidden_size,
            config.acoustic_text_encoder_hidden_size,
            kernel_size=config.acoustic_text_encoder_kernel_size,
            padding=config.acoustic_text_encoder_kernel_size // 2,
        )
        self.norm = nn.LayerNorm(config.acoustic_text_encoder_hidden_size)
        self.leaky_relu_slope = config.acoustic_text_encoder_leaky_relu_slope
        self.dropout = nn.Dropout(config.acoustic_text_encoder_dropout)

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


class StyleTextToSpeech2AcousticTextEncoderPretrainedModel(PreTrainedModel):
    config_class = StyleTextToSpeech2Config
    base_model_prefix = "acoustic_text_encoder"

    def _init_weights(self, module):
        """Initialize the weights"""
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


class StyleTextToSpeech2AcousticTextEncoder(StyleTextToSpeech2AcousticTextEncoderPretrainedModel):
    def __init__(self, config: StyleTextToSpeech2Config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.acoustic_text_encoder_hidden_size)
        self.layers = nn.ModuleList(
            [AcousticTextEncoderLayer(config) for _ in range(config.acoustic_text_encoder_num_hidden_layers)]
        )
        self.lstm = nn.LSTM(
            config.acoustic_text_encoder_hidden_size,
            config.acoustic_text_encoder_hidden_size // 2,
            1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, input_ids, mask=None, input_lengths=None):
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
        bert_config = AlbertConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.prosodic_encoder_hidden_size,
            num_attention_heads=config.prosodic_encoder_num_attention_heads,
            intermediate_size=config.prosodic_encoder_intermediate_size,
            max_position_embeddings=config.prosodic_encoder_max_position_embeddings,
            dropout=config.prosodic_encoder_dropout,
        )
        self.bert_model = AlbertModel(bert_config)
        self.proj_out = nn.Linear(config.prosodic_encoder_hidden_size, config.acoustic_text_encoder_hidden_size)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert_model(input_ids, attention_mask=attention_mask)
        return self.proj_out(outputs.last_hidden_state)


class StyleTextToSpeech2AdaLayerNorm(nn.Module):
    def __init__(self, hidden_size, style_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.style_size = style_size
        self.proj = nn.Linear(style_size, hidden_size*2)

    def forward(self, hidden_states, style):     
        hidden_style = self.proj(style)
        gamma, beta = torch.chunk(hidden_style, chunks=2, dim=-1)
        hidden_states = F.layer_norm(hidden_states, (self.hidden_size,))
        hidden_states = (1 + gamma) * hidden_states + beta
        return hidden_states
 

class StyleTextToSpeech2DurationEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lstm = nn.LSTM(config.style_hidden_size + config.duration_encoder_hidden_size, config.duration_encoder_hidden_size // 2, 1, batch_first=True, bidirectional=True, dropout=config.duration_encoder_dropout)
        self.ada_layer_norm = StyleTextToSpeech2AdaLayerNorm(config.duration_encoder_hidden_size, config.style_hidden_size)
        self.dropout = nn.Dropout(config.acoustic_text_encoder_dropout)

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

class StyleTextToSpeech2DurationEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(
            [StyleTextToSpeech2DurationEncoderLayer(config) for _ in range(config.duration_encoder_num_layers)]
        )  
    
    def _concat_style(self, hidden_states, style, mask):
        bsz, seq_len, _ = hidden_states.shape
        hidden_states = torch.cat(
            (hidden_states, style.expand(bsz, seq_len, -1)),
            dim=-1
        ) 
        hidden_states = hidden_states.masked_fill(~mask.bool(), 0)
        return hidden_states

    def forward(self, hidden_states, style, mask=None, input_lengths=None):
        bsz, seq_len, _ = hidden_states.shape
        if mask is None:
            mask = torch.full((bsz, seq_len, 1), 1, dtype=torch.int, device=hidden_states.device)
        else:
            mask = mask.unsqueeze(-1)

        if input_lengths is None:
            input_lengths = mask.sum(dim=1).view(-1).tolist()

        hidden_states = self._concat_style(hidden_states, style, mask)
        for layer in self.layers:
            hidden_states = layer(hidden_states, style, input_lengths)
            hidden_states = self._concat_style(hidden_states, style, mask)

        return hidden_states

class StyleTextToSpeech2DurationPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lstm = nn.LSTM(config.acoustic_text_encoder_hidden_size + config.style_hidden_size, config.acoustic_text_encoder_hidden_size // 2, 1, batch_first=True, bidirectional=True)
        self.duration_proj = nn.Linear(config.acoustic_text_encoder_hidden_size, 1)

    def forward(self, hidden_states):
        hidden_states, _ = self.lstm(hidden_states)
        return self.duration_proj(hidden_states)
    

class StyleTextToSpeech2AdainResBlockLayer(nn.Module):
    def __init__(self, channels, style_size, kernel_size, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation, padding=self._get_padding(kernel_size, dilation))
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=self._get_padding(kernel_size, 1))
        self.norm1 = StyleTextToSpeech2AdaLayerNorm(channels, style_size)
        self.norm2 = StyleTextToSpeech2AdaLayerNorm(channels, style_size)
        self.alpha1 = nn.Parameter(torch.ones(1, channels, 1))
        self.alpha2 = nn.Parameter(torch.ones(1, channels, 1))

    def _get_padding(self, kernel_size, dilation):
        return int((kernel_size * dilation - dilation) / 2)
    
    def forward(self, hidden_states, style):
        x = self.norm1(hidden_states, style)
        x = x + (1 / self.alpha1) * (torch.sin(self.alpha1 * x) ** 2)
        x = self.conv1(x)
        x = self.norm2(x, style)
        x = x + (1 / self.alpha2) * (torch.sin(self.alpha2 * x) ** 2)
        x = self.conv2(x)
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


class StyleTextToSpeech2AdainResBlock1d(nn.Module):
    def __init__(self, hidden_size_in, hidden_size_out, style_size, upsample=False, learned_shortcut=False, dropout_p=0.0):
        super().__init__()
        self.upsample = upsample

        self.conv1 = nn.Conv1d(hidden_size_in, hidden_size_out, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(hidden_size_out, hidden_size_out, kernel_size=3, stride=1, padding=1)
        self.norm1 = StyleTextToSpeech2AdaLayerNorm(hidden_size_in, style_size)
        self.norm2 = StyleTextToSpeech2AdaLayerNorm(hidden_size_out, style_size)

        if upsample:
            self.upsample = lambda x: F.interpolate(x, scale_factor=2, mode='nearest')
            self.pool = nn.ConvTranspose1d(hidden_size_in, hidden_size_in, kernel_size=3, stride=2, groups=hidden_size_in, padding=1, output_padding=1)
        else:
            self.upsample = lambda x: x
            self.pool = nn.Identity()
        
        if learned_shortcut:
            self.conv1_shortcut = nn.Conv1d(hidden_size_in, hidden_size_out, kernel_size=1, stride=1, padding=0, bias=False)
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
            config.acoustic_text_encoder_hidden_size, 
            config.acoustic_text_encoder_hidden_size,
            config.style_hidden_size, 
            dropout_p=config.acoustic_text_encoder_dropout,
        )
        self.adain_res_1d_2 = StyleTextToSpeech2AdainResBlock1d(
            config.acoustic_text_encoder_hidden_size, 
            config.acoustic_text_encoder_hidden_size // 2, 
            config.style_hidden_size, 
            dropout_p=config.acoustic_text_encoder_dropout,
            upsample=True, 
            learned_shortcut=True,
        )
        self.adain_res_1d_3 = StyleTextToSpeech2AdainResBlock1d(
            config.acoustic_text_encoder_hidden_size // 2, 
            config.acoustic_text_encoder_hidden_size // 2, 
            config.style_hidden_size, 
            dropout_p=config.acoustic_text_encoder_dropout
        )
        self.conv_out = nn.Conv1d(config.acoustic_text_encoder_hidden_size // 2, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, hidden_states, style):
        hidden_states = self.adain_res_1d_1(hidden_states, style)
        hidden_states = self.adain_res_1d_2(hidden_states, style)
        hidden_states = self.adain_res_1d_3(hidden_states, style)
        return self.conv_out(hidden_states.transpose(1, -1))


class StyleTextToSpeech2ProsodyPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lstm = nn.LSTM(config.acoustic_text_encoder_hidden_size + config.style_hidden_size, config.acoustic_text_encoder_hidden_size // 2, 1, batch_first=True, bidirectional=True)
        self.pitch_block = StyleTextToSpeech2ProsodyBlock(config)
        self.energy_block = StyleTextToSpeech2ProsodyBlock(config)
    
    def forward(self, hidden_states, style):
        hidden_states, _ = self.lstm(hidden_states)
        pitch = self.pitch_block(hidden_states, style)
        energy = self.energy_block(hidden_states, style)
        return pitch, energy


class StyleTextToSpeech2HarmonicNoiseSourceFilter(nn.Module):
    """
    Harmonic plus Noise Neural Source Filter. https://arxiv.org/abs/2309.09493
    Adapted from https://github.com/yl4579/StyleTT2
    """
    def __init__(
        self,
        sampling_rate,
        upsample_scale,
        harmonic_num=0,
        sine_amplitude=0.1,
        add_noise_std=0.003,
        voiced_threshold=0,
        flag_for_pulse=False,
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.upsample_scale = upsample_scale
        self.harmonic_num = harmonic_num
        self.sine_amplitude = sine_amplitude
        self.add_noise_std = add_noise_std
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse
        # to merge source harmonics into a single excitation
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def _f02uv(self, f0):
        uv = (f0 > self.voiced_threshold).to(torch.float32)
        return uv

    def _f02sine(self, f0_values):
        """ f0_values: (batchsize, length, dim)
            where dim indicates fundamental tone and overtones
        """
        rad_values = (f0_values / self.sampling_rate) % 1

        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        if not self.flag_for_pulse:
            rad_values = torch.nn.functional.interpolate(rad_values.transpose(1, 2), scale_factor=1/self.upsample_scale, mode="linear").transpose(1, 2)
            phase = torch.cumsum(rad_values, dim=1) * 2 * np.pi
            phase = torch.nn.functional.interpolate(phase.transpose(1, 2) * self.upsample_scale, scale_factor=self.upsample_scale, mode="linear").transpose(1, 2)
            sines = torch.sin(phase)
        else:
            uv = self._f02uv(f0_values)
            uv_1 = torch.roll(uv, shifts=-1, dims=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)
            tmp_cumsum = torch.cumsum(rad_values, dim=1)
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum

            i_phase = torch.cumsum(rad_values - tmp_cumsum, dim=1)
            sines = torch.cos(i_phase * 2 * np.pi)

        return sines
    
    def _sine_gen(self, f0):
        """ sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        fn = torch.multiply(f0, torch.FloatTensor([[range(1, self.harmonic_num + 2)]]).to(f0.device))

        # generate sine waveforms
        sine_waves = self._f02sine(fn) * self.sine_amp

        # generate uv signal
        uv = self._f02uv(f0)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)
        sine_waves = sine_waves * uv + noise

        return sine_waves, uv
        
    def forward(self, hidden_states):
        sine_waves, uv = self._sine_gen(hidden_states)
        sine_merge = self.l_tanh(self.l_linear(sine_waves))
        noise = torch.randn_like(uv) * self.sine_amp / 3

        return sine_merge, noise, uv
    

class StyleTextToSpeech2GeneratorLayer(nn.Module):
    def __init__(self, layer_idx, style_size, upsample_rates, resblock_kernel_sizes, upsample_initial_channel, resblock_dilation_sizes, upsample_kernel_sizes, gen_istft_n_fft, reflection_pad=True):
        super().__init__()

        c_cur = upsample_initial_channel // (2 ** (layer_idx + 1))
        if layer_idx + 1 < len(upsample_rates):
            noise_conv_stride = np.prod(upsample_rates[layer_idx + 1:])
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
            gen_istft_n_fft + 2, 
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

    def forward(self, hidden_states, hidden_states_source, style):
        hidden_states = F.leaky_relu(hidden_states, 0.1)
        hidden_states_source = self.noise_conv(hidden_states_source)
        hidden_states_source = self.noise_res(hidden_states_source, style)

        hidden_states = self.up(hidden_states)
        hidden_states = self.reflection_pad(hidden_states)
        hidden_states = hidden_states + hidden_states_source
        
        hidden_states = self.resblocks[0](hidden_states, style)
        for resblock in self.resblocks[1:]:
            hidden_states += resblock(hidden_states, style)
        hidden_states /= len(self.resblocks)

        return hidden_states


class StyleTextToSpeech2Generator(nn.Module):
    def __init__(self, style_size, resblock_kernel_sizes, upsample_rates, upsample_initial_channel, resblock_dilation_sizes, upsample_kernel_sizes, gen_istft_n_fft, gen_istft_hop_size):
        super().__init__()

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.m_source = StyleTextToSpeech2HarmonicNoiseSourceFilter(
            sampling_rate=24000,
            upsample_scale=np.prod(upsample_rates) * gen_istft_hop_size,
            harmonic_num=8, 
            voiced_threshod=10
        )

        self.f0_upsamp = torch.nn.Upsample(scale_factor=np.prod(upsample_rates) * gen_istft_hop_size)

        self.layers = nn.ModuleList([
            StyleTextToSpeech2GeneratorLayer(
                layer_idx, 
                style_size, 
                upsample_rates, 
                resblock_kernel_sizes, 
                upsample_initial_channel, 
                resblock_dilation_sizes, 
                upsample_kernel_sizes, 
                gen_istft_n_fft, 
                reflection_pad=True if layer_idx == len(upsample_rates) - 1 else False
            )
            for layer_idx in range(len(upsample_rates))
        ])    
                
        self.post_n_fft = gen_istft_n_fft

        self.conv_post = nn.Conv1d(
            upsample_initial_channel // (2 ** (len(upsample_rates))), 
            self.post_n_fft + 2,
            7, 
            1, 
            padding=3
        )
        self.stft = TorchSTFT(filter_length=gen_istft_n_fft, hop_length=gen_istft_hop_size, win_length=gen_istft_n_fft)
        
    def forward(self, hidden_states, style, f0):
        with torch.no_grad():
            f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # bs,n,t
            har_source, noi_source, uv = self.m_source(f0)
            har_source = har_source.transpose(1, 2).squeeze(1)
            har_spec, har_phase = self.stft.transform(har_source)
            har = torch.cat([har_spec, har_phase], dim=1)

        for layer in self.layers:
            hidden_states = layer(hidden_states, har, style)

        hidden_states = F.leaky_relu(hidden_states)
        hidden_states = self.conv_post(hidden_states)
        spec = torch.exp(hidden_states[:,:self.post_n_fft // 2 + 1, :])
        phase = torch.sin(hidden_states[:, self.post_n_fft // 2 + 1:, :])
        return self.stft.inverse(spec, phase)


class StyleTextToSpeech2Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pitch_conv = nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1)
        self.energy_conv = nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1)
        self.encode = StyleTextToSpeech2AdainResBlock1d(config.acoustic_text_encoder_hidden_size + 2, 1024, config.style_hidden_size)
        self.asr_res = nn.Conv1d(512, 64, kernel_size=1)
        self.decode = nn.ModuleList([
            StyleTextToSpeech2AdainResBlock1d(1024 + 2 + 64, 1024, config.style_hidden_size),
            StyleTextToSpeech2AdainResBlock1d(1024 + 2 + 64, 1024, config.style_hidden_size),
            StyleTextToSpeech2AdainResBlock1d(1024 + 2 + 64, 1024, config.style_hidden_size),
            StyleTextToSpeech2AdainResBlock1d(1024 + 2 + 64, 512, config.style_hidden_size, upsample=True)
        ])
        self.generator = StyleTextToSpeech2Generator(config.style_hidden_size, config.resblock_kernel_sizes, config.upsample_rates, config.upsample_initial_channel, config.resblock_dilation_sizes, config.upsample_kernel_sizes, config.gen_istft_n_fft, config.gen_istft_hop_size)

    def forward(self, hidden_states, pitch, energy, style):
        pitch = self.pitch_conv(pitch)
        energy = self.energy_conv(energy)

        hidden_states = torch.cat([hidden_states, pitch, energy], dim=-1)
        hidden_states = self.encode(hidden_states, style)
        asr_res = self.asr_res(hidden_states)

        residual = True
        for block in self.decode:
            if residual:
                hidden_states = torch.cat([hidden_states, asr_res, pitch, energy], axis=1)
            hidden_states = block(hidden_states, style)
            if block.upsample:
                residual = False

        return self.generator(hidden_states, style, pitch)


class StyleTextToSpeech2ForConditionalGeneration(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.acoustic_text_encoder = StyleTextToSpeech2AcousticTextEncoder(config)
        self.prosodic_text_encoder = StyleTextToSpeech2ProsodicTextEncoder(config)
        self.duration_encoder = StyleTextToSpeech2DurationEncoder(config)
        self.duration_predictor = StyleTextToSpeech2DurationPredictor(config)
        self.prosody_predictor = StyleTextToSpeech2ProsodyPredictor(config)
        self.decoder = StyleTextToSpeech2Decoder(config)

    def generate(self, input_ids, style, speed=1):
        style = style[:, :128]

        hidden_states_text = self.acoustic_text_encoder(input_ids)
        hidden_states_prosodic = self.prosodic_text_encoder(input_ids)

        hidden_states_duration = self.duration_encoder(hidden_states_prosodic, style)
        durations = self.duration_predictor(hidden_states_duration)
        durations = torch.sigmoid(durations).sum(axis=-1) / speed
        durations = torch.round(durations).clamp(min=1).long().squeeze()

        hidden_states_duration = hidden_states_duration.repeat_interleave(durations, dim=1)
        hidden_states_text = hidden_states_text.repeat_interleave(durations, dim=1)

        pitch, energy = self.prosody_predictor(hidden_states_duration, style)

        return hidden_states_text, pitch, energy, style

__all__ = ["StyleTextToSpeech2AcousticTextEncoder", "StyleTextToSpeech2ProsodicEncoder", "StyleTextToSpeech2DurationEncoder", "StyleTextToSpeech2ForConditionalGeneration"]
