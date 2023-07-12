import math
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, L1Loss

from ...activations import ACT2FN
from ...deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqSpectrogramOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_univnet import UnivNetGanConfig


logger = logging.get_logger(__name__)


class UnivNetKernelPredictorResidualBlock(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size,
        dropout=0.0,
        leaky_relu_slope=0.1,
    ):
        super().__init__()
        self.leaky_relu_slope = leaky_relu_slope

        self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=self.get_padding(kernel_size),
            bias=True,
        )
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=self.get_padding(kernel_size),
            bias=True,
        )
    
    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv1(hidden_states)
        hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
        hidden_states = self.conv2(hidden_states)
        hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
        return hidden_states + residual
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def get_padding(self, kernel_size, dilation=1):
        return dilation * (kernel_size - 1) // 2

    def apply_weight_norm(self):
        nn.utils.weight_norm(self.conv1)
        nn.utils.weight_norm(self.conv2)

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv1)
        nn.utils.remove_weight_norm(self.conv2)


# For now, only support leaky ReLU as activation function
class UnivNetKernelPredictor(nn.Module):
    def __init__(
        self,
        input_channels,
        conv_in_channels,
        conv_out_channels,
        conv_layers,
        conv_kernel_size=3,
        num_blocks=3,
        resnet_hidden_channels=64,
        resnet_kernel_size=3,
        dropout=0.0,
        leaky_relu_slope=0.1,
    ):
        super().__init__()

        self.conv_in_channels = conv_in_channels
        self.conv_out_channels = conv_out_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_layers = conv_layers
        self.leaky_relu_slope = leaky_relu_slope

        kernel_channels = conv_in_channels * conv_out_channels * conv_kernel_size * conv_layers
        bias_channels = conv_out_channels * conv_layers
        padding = self.get_padding(resnet_kernel_size)

        self.input_conv = nn.Conv1d(input_channels, resnet_hidden_channels, 5, padding=2, bias=True)

        self.resblocks = nn.ModuleList(
            [
                UnivNetKernelPredictorResidualBlock(
                    channels=resnet_hidden_channels,
                    kernel_size=resnet_kernel_size,
                    dropout=dropout,
                    leaky_relu_slope=leaky_relu_slope
                )
                for _ in range(len(num_blocks))
            ]
        )

        self.kernel_conv = nn.Conv1d(
            resnet_hidden_channels, kernel_channels, resnet_kernel_size, padding=padding, bias=True
        )
        self.bias_conv = nn.Conv1d(
            resnet_hidden_channels, bias_channels, resnet_kernel_size, padding=padding, bias=True
        )
    
    def forward(self, spectrogram):
        # spectrogram should have shape (batch_size, input_channels, seq_length)
        batch_size, _, seq_length = spectrogram.shape

        hidden_states = self.input_conv(spectrogram)
        for resblock in self.resblocks:
            hidden_states = resblock(hidden_states)
        kernel_hidden_states = self.kernel_conv(hidden_states)
        bias_hidden_states = self.bias_conv(hidden_states)
        # Reshape kernels and biases to appropriate shape
        kernels = kernel_hidden_states.contiguous().view(
            batch_size,
            self.conv_layers,
            self.conv_in_channels,
            self.conv_out_channels,
            self.conv_kernel_size,
            seq_length,
        )
        biases = bias_hidden_states.contiguous().view(
            batch_size,
            self.conv_layers,
            self.conv_out_channels,
            seq_length,
        )
        return kernels, biases
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def get_padding(self, kernel_size, dilation=1):
        return dilation * (kernel_size - 1) // 2

    def apply_weight_norm(self):
        nn.utils.weight_norm(self.input_conv)
        for layer in self.resblocks:
            layer.apply_weight_norm()
        nn.utils.weight_norm(self.kernel_conv)
        nn.utils.weight_norm(self.bias_conv)

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.input_conv)
        for layer in self.resblocks:
            layer.remove_weight_norm()
        nn.utils.remove_weight_norm(self.kernel_conv)
        nn.utils.remove_weight_norm(self.bias_conv)


class UnivNetLVCResidualBlock(nn.Module):
    def __init__(
        self,
        input_channels,
        kernel_size,
        dilation,
        leaky_relu_slope=0.2,
    ):
        super().__init__()
        self.in_channels = input_channels
        self.leaky_relu_slope = leaky_relu_slope

        self.conv = nn.Conv1d(
            input_channels,
            input_channels,
            kernel_size,
            padding=self.get_padding(kernel_size, dilation),
            dilation=dilation,
        )
    
    def forward(self, hidden_states, kernel, bias, hop_size=256):
        residual = hidden_states
        hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
        hidden_states = self.conv(hidden_states)
        hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
        hidden_states = self.location_variable_convolution(hidden_states, kernel, bias, hop_size=hop_size)
        # Gated activation unit
        hidden_states = torch.sigmoid(hidden_states[:, :self.in_channels, :]) * torch.tanh(hidden_states[:, self.in_channels:, :])
        # Skip connection
        hidden_states = residual + hidden_states

        return hidden_states
    
    def location_variable_convolution(self, x, kernel, bias, dilation=1, hop_size=256):
        ''' perform location-variable convolution operation on the input sequence (x) using the local convolution kernl. 
        Time: 414 μs ± 309 ns per loop (mean ± std. dev. of 7 runs, 1000 loops each), test on NVIDIA V100. 
        Args:
            x (Tensor): the input sequence (batch, in_channels, in_length). 
            kernel (Tensor): the local convolution kernel (batch, in_channel, out_channels, kernel_size, kernel_length) 
            bias (Tensor): the bias for the local convolution (batch, out_channels, kernel_length) 
            dilation (int): the dilation of convolution. 
            hop_size (int): the hop_size of the conditioning sequence. 
        Returns:
            (Tensor): the output sequence after performing local convolution. (batch, out_channels, in_length).
        '''
        batch, _, in_length = x.shape
        batch, _, out_channels, kernel_size, kernel_length = kernel.shape
        assert in_length == (kernel_length * hop_size), "length of (x, kernel) is not matched"

        padding = dilation * int((kernel_size - 1) / 2)
        x = nn.functional.pad(x, (padding, padding), 'constant', 0)  # (batch, in_channels, in_length + 2*padding)
        x = x.unfold(2, hop_size + 2 * padding, hop_size)  # (batch, in_channels, kernel_length, hop_size + 2*padding)

        if hop_size < dilation:
            x = nn.functional.pad(x, (0, dilation), 'constant', 0)
        x = x.unfold(3, dilation, dilation)     # (batch, in_channels, kernel_length, (hop_size + 2*padding)/dilation, dilation)
        x = x[:, :, :, :, :hop_size]          
        x = x.transpose(3, 4)                   # (batch, in_channels, kernel_length, dilation, (hop_size + 2*padding)/dilation)  
        x = x.unfold(4, kernel_size, 1)         # (batch, in_channels, kernel_length, dilation, _, kernel_size)

        o = torch.einsum('bildsk,biokl->bolsd', x, kernel)
        o = o.to(memory_format=torch.channels_last_3d)
        bias = bias.unsqueeze(-1).unsqueeze(-1).to(memory_format=torch.channels_last_3d)
        o = o + bias
        o = o.contiguous().view(batch, out_channels, -1)

        return o
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def get_padding(self, kernel_size, dilation=1):
        return dilation * (kernel_size - 1) // 2
    
    def apply_weight_norm(self):
        nn.utils.weight_norm(self.conv)

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv)


class UnivNetLVCBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        cond_channels,
        conv_kernel_size=3,
        stride=8,
        dilations=[1, 3, 9, 27],
        leaky_relu_slope=0.2,
        cond_hop_length=256,
        kernel_pred_net_hidden_channels=64,
        kernel_pred_net_conv_size=3,
        kernel_pred_net_dropout=0.0,
    ):
        super().__init__()
        self.cond_hop_length = cond_hop_length
        self.leaky_relu_slope = leaky_relu_slope

        self.convt_pre = nn.ConvTranspose1d(
            in_channels,
            in_channels,
            2 * stride,
            stride=stride,
            padding=stride // 2 + stride % 2,
            output_padding=stride % 2,
        )

        self.kernel_predictor = UnivNetKernelPredictor(
            input_channels=cond_channels,
            conv_in_channels=in_channels,
            conv_out_channels=2 * in_channels,
            conv_layers=len(dilations),
            conv_kernel_size=conv_kernel_size,
            resnet_hidden_channels=kernel_pred_net_hidden_channels,
            resnet_kernel_size=kernel_pred_net_conv_size,
            dropout=kernel_pred_net_dropout,
            leaky_relu_slope=leaky_relu_slope,
        )

        self.resblocks = nn.ModuleList(
            [
                UnivNetLVCResidualBlock(
                    input_channels=in_channels,
                    kernel_size=conv_kernel_size,
                    dilation=dilations[i],
                    leaky_relu_slope=leaky_relu_slope,
                )
                for i in range(len(dilations))
            ]
        )
    
    def forward(self, hidden_states, spectrogram):
        # hidden_states: (batch_size, in_channels, in_length)
        # spectrogram: (batch_size, cond_channels, cond_length)
        hidden_states = self.convt_pre(hidden_states)
        kernels, biases = self.kernel_predictor(spectrogram)

        for i, resblock in enumerate(self.resblocks):
            kernel = kernels[:, i, :, :, :, :]
            bias = biases[:, i, :, :]
            hidden_states = resblock(hidden_states, kernel, bias, hop_size=self.cond_hop_length)
        
        return hidden_states
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def apply_weight_norm(self):
        nn.utils.weight_norm(self.convt_pre)
        self.kernel_predictor.apply_weight_norm()
        for layer in self.resblocks:
            layer.apply_weight_norm()

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.convt_pre)
        self.kernel_predictor.remove_weight_norm()
        for layer in self.resblocks:
            layer.remove_weight_norm()


class UnivNetGan(PreTrainedModel):
    config_class = UnivNetGanConfig
    main_input_name = "spectrogram"

    def __init__(self, config: UnivNetGanConfig):
        super().__init__(config)

        self.num_kernels = len(config.resblock_kernel_sizes)
        self.leaky_relu_slope = config.leaky_relu_slope

        self.conv_pre = nn.Conv1d(
            config.model_in_dim,
            config.channel_size,
            kernel_size=7,
            stride=1,
            padding=3,
            padding_mode="reflect",
        )

        # TODO: no upsampler in UnivNet GAN?

        # Initialize location-variable convolution ResNet Blocks.
        num_layers = len(config.resblock_stride_sizes)
        hop_length = 1
        hop_lengths = []
        for stride in config.resblock_stride_sizes:
            hop_length = hop_length * stride
            hop_lengths.append(hop_length)
        
        self.resblocks = nn.ModuleList(
            [
                UnivNetLVCBlock(
                    in_channels=config.model_in_dim,
                    cond_channels=config.num_mel_channels,
                    conv_kernel_size=config.resblock_kernel_sizes[i],
                    stride=config.resblock_stride_sizes[i],
                    dilations=config.resblock_dilation_sizes[i],
                    leaky_relu_slope=config.leaky_relu_slope,
                    cond_hop_length=hop_lengths[i],
                    kernel_pred_net_hidden_channels=config.kernel_predictor_hidden_channels,
                    kernel_pred_net_conv_size=config.kernel_predictor_conv_size,
                    kernel_pred_net_dropout=config.kernel_predictor_dropout,
                )
                for i in range(num_layers)
            ]
        )

        self.conv_post = nn.Conv1d(config.channel_size, 1, 7, padding=3, padding_mode="reflect")
    
    def forward(self, noise_waveform, cond_spectrogram):
        # noise_waveform: (batch_size, noise_dim, in_length)
        # cond_spectrogram: (batch_size, mel_channels, in_length)
        hidden_states = self.conv_pre(noise_waveform)

        for resblock in self.resblocks:
            hidden_states = resblock(hidden_states, cond_spectrogram)
        
        hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
        hidden_states = self.conv_post(hidden_states)
        hidden_states = torch.tanh(hidden_states)

        return hidden_states

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def apply_weight_norm(self):
        nn.utils.weight_norm(self.conv_pre)
        for layer in self.resblocks:
            layer.apply_weight_norm()
        nn.utils.weight_norm(self.conv_post)

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv_pre)
        for layer in self.resblocks:
            layer.remove_weight_norm()
        nn.utils.remove_weight_norm(self.conv_post)

