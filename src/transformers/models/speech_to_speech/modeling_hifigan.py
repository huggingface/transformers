import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d

from configuration_hifigan import CodeHiFiGANConfig, HiFiGANConfig
from transformers import PreTrainedModel


class HiFiGANResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), leaky_relu_slope=0.1):
        super(HiFiGANResidualBlock, self).__init__()
        self.convs1 = nn.ModuleList(
            [
                Conv1d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation[i],
                    padding=self.get_padding(kernel_size, dilation[i]),
                )
                for i in range(len(dilation))
            ]
        )

        self.convs2 = nn.ModuleList(
            [
                Conv1d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=1,
                    padding=self.get_padding(kernel_size, 1),
                )
                for _ in range(len(dilation))
            ]
        )
        self.leaky_relu_slope = leaky_relu_slope

    def get_padding(self, kernel_size, dilation=1):
        return (kernel_size * dilation - dilation) // 2

    def forward(self, hidden_states):
        for conv1, conv2 in zip(self.convs1, self.convs2):
            residual = hidden_states
            hidden_states = F.leaky_relu(hidden_states, self.leaky_relu_slope)
            hidden_states = conv1(hidden_states)
            hidden_states = F.leaky_relu(hidden_states, self.leaky_relu_slope)
            hidden_states = conv2(hidden_states)
            hidden_states = hidden_states + residual
        return hidden_states


class HiFiGANModel(PreTrainedModel):
    config_class = HiFiGANConfig

    def __init__(self, config: HiFiGANConfig):
        super().__init__(config)
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)
        self.conv_pre = Conv1d(
            config.model_in_dim,
            config.upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding=3,
        )

        self.upsampler = nn.ModuleList()
        for i, (upsample_rate, kernel_size) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            self.upsampler.append(
                ConvTranspose1d(
                    config.upsample_initial_channel // (2**i),
                    config.upsample_initial_channel // (2 ** (i + 1)),
                    kernel_size=kernel_size,
                    stride=upsample_rate,
                    padding=(kernel_size - upsample_rate) // 2,
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.upsampler)):
            channels = config.upsample_initial_channel // (2 ** (i + 1))
            for kernel_size, dilation in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes):
                self.resblocks.append(HiFiGANResidualBlock(channels, kernel_size, dilation, config.leaky_relu_slope))

        self.conv_post = Conv1d(channels, 1, kernel_size=7, stride=1, padding=3)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, hidden_states):
        hidden_states = self.conv_pre(hidden_states)
        for i in range(self.num_upsamples):
            hidden_states = F.leaky_relu(hidden_states, self.config.leaky_relu_slope)
            hidden_states = self.upsampler[i](hidden_states)

            res_state = self.resblocks[i * self.num_kernels](hidden_states)
            for j in range(1, self.num_kernels):
                res_state += self.resblocks[i * self.num_kernels + j](hidden_states)
            hidden_states = res_state / self.num_kernels

        hidden_states = F.leaky_relu(hidden_states)
        hidden_states = self.conv_post(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        return hidden_states


class CodeHiFiGANVariancePredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv1 = nn.Conv1d(
            config.encoder_embed_dim,
            config.variance_predictor_hidden_dim,
            kernel_size=config.variance_predictor_kernel_size,
            padding=(config.variance_predictor_kernel_size - 1) // 2,
        )
        self.ln1 = nn.LayerNorm(config.variance_predictor_hidden_dim)
        self.dropout_module = nn.Dropout(p=config.variance_predictor_dropout)
        self.conv2 = nn.Conv1d(
            config.variance_predictor_hidden_dim,
            config.variance_predictor_hidden_dim,
            kernel_size=config.variance_predictor_kernel_size,
            padding=1,
        )
        self.ln2 = nn.LayerNorm(config.variance_predictor_hidden_dim)
        self.proj = nn.Linear(config.variance_predictor_hidden_dim, 1)

    def forward(self, hidden_states):
        # hidden_states: (batch_size, sequence_length, channels)
        hidden_states = F.relu(self.conv1(hidden_states.transpose(1, 2))).transpose(1, 2)
        hidden_states = self.dropout_module(self.ln1(hidden_states))
        hidden_states = F.relu(self.conv2(hidden_states.transpose(1, 2))).transpose(1, 2)
        hidden_states = self.dropout_module(self.ln2(hidden_states))
        out = self.proj(hidden_states).squeeze(dim=2)
        # out: (batch_size, sequence_length)
        return out


class CodeHiFiGANModel(HiFiGANModel):
    config_class = CodeHiFiGANConfig

    def __init__(self, config):
        super().__init__(config)
        self.dict = nn.Embedding(config.num_embeddings, config.embedding_dim)
        self.multispeaker = config.multispeaker
        self.speaker_embedding = config.speaker_embedding

        if self.multispeaker and not self.speaker_embedding:
            self.speaker = nn.Embedding(config.num_speakers, config.embedding_dim)
        elif self.speaker_embedding:
            self.speaker = nn.Linear(config.speaker_embedding_dim, config.embedding_dim)

        self.duration_predictor = None
        if config.duration_predictor:
            self.duration_predictor = CodeHiFiGANVariancePredictor(config)

        self.f0 = config.f0
        n_f0_bin = config.f0_quant_num_bin

        self.f0_quant_embed = None
        if n_f0_bin > 0:
            self.f0_quant_embed = nn.Embedding(n_f0_bin, config.embedding_dim)

    @staticmethod
    def _upsample(signal, max_frames):
        if signal.dim() == 3:
            batch_size, channels, cond_length = signal.size()
        elif signal.dim() == 2:
            signal = signal.unsqueeze(2)
            batch_size, channels, cond_length = signal.size()
        else:
            signal = signal.view(-1, 1, 1)
            batch_size, channels, cond_length = signal.size()

        signal = signal.unsqueeze(3).repeat(1, 1, 1, max_frames // cond_length)

        # pad zeros as needed (if signal's shape does not divide completely with max_frames)
        remainder = (max_frames - signal.shape[2] * signal.shape[3]) // signal.shape[3]
        if remainder > 0:
            raise NotImplementedError("Padding condition signal - misalignment between condition features.")

        signal = signal.view(batch_size, channels, max_frames)
        return signal

    def forward(self, hidden_states, duration_prediction=False, f0=None, speaker=None):
        hidden_states = self.dict(hidden_states).transpose(1, 2)

        if self.duration_predictor and duration_prediction:
            assert hidden_states.size(0) == 1, "only support single sample"
            log_dur_pred = self.duration_predictor(hidden_states.transpose(1, 2))
            dur_out = torch.clamp(torch.round((torch.exp(log_dur_pred) - 1)).long(), min=1)
            # hidden_states: (batch_size, channels, sequence_length)
            hidden_states = torch.repeat_interleave(hidden_states, dur_out.view(-1), dim=2)

        if self.f0:
            assert f0 is not None, 'require "f0" if "config.f0" is True'
            if self.f0_quant_embed:
                f0 = self.f0_quant_embed(f0.long()).transpose(1, 2)
            else:
                f0 = f0.unsqueeze(1)

            if hidden_states.shape[-1] < f0.shape[-1]:
                hidden_states = self._upsample(hidden_states, f0.shape[-1])
            elif hidden_states.shape[-1] > f0.shape[-1]:
                f0 = self._upsample(f0, hidden_states.shape[-1])
            hidden_states = torch.cat([hidden_states, f0], dim=1)

        if self.multispeaker:
            assert speaker is not None, 'require "speaker" input for multispeaker CodeHiFiGAN vocoder'
            speaker = self.speaker(speaker).transpose(1, 2)
            speaker = self._upsample(speaker, hidden_states.shape[-1])
            hidden_states = torch.cat([hidden_states, speaker], dim=1)

        return super().forward(hidden_states)
