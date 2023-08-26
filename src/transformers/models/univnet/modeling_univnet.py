# Copyright 2023 The HuggingFace Team. All rights reserved.
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

from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_start_docstrings,
    logging,
)
from .configuration_univnet import UnivNetGanConfig


logger = logging.get_logger(__name__)


_CHECKPOINT_FOR_DOC = "dg845/univnet-dev"

UNIVNET_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "dg845/univnet-dev",
    # See all UnivNet models at https://huggingface.co/models?filter=univnet
]


class UnivNetKernelPredictorResidualBlock(nn.Module):
    """
    Implementation of the residual block for the kernel predictor network inside each location variable convolution
    block (LVCBlock).

    Parameters:
        channels (`int`, *optional*, defaults to 64):
            The number of hidden channels for the residual block.
        kernel_size (`int`, *optional*, defaults to 3):
            The kernel size for the 1D convolutional layers in the residual block.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for the Dropout layer in the residual block.
        leaky_relu_slope (`float`, *optional*, defaults to 0.1):
            The angle of the negative slope used by the leaky ReLU activation.
    """

    def __init__(
        self,
        channels: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.0,
        leaky_relu_slope: float = 0.1,
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

    def forward(self, hidden_states: torch.FloatTensor):
        # hidden_states should have shape (batch_size, channels, seq_length)
        residual = hidden_states
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv1(hidden_states)
        hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
        hidden_states = self.conv2(hidden_states)
        hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
        return hidden_states + residual

    def get_padding(self, kernel_size: int, dilation: int = 1):
        return dilation * (kernel_size - 1) // 2

    def apply_weight_norm(self):
        nn.utils.weight_norm(self.conv1)
        nn.utils.weight_norm(self.conv2)

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv1)
        nn.utils.remove_weight_norm(self.conv2)


# For now, only support leaky ReLU as activation function (reference implementation supports arbitrary act fns)
class UnivNetKernelPredictor(nn.Module):
    """
    Implementation of the kernel predictor network which supplies the kernel and bias for the location variable
    convolutional layers (LVCs) in each UnivNet LVCBlock.

    Based on the KernelPredictor implementation in
    [mindslab-ai/univnet](https://github.com/mindslab-ai/univnet/blob/master/model/lvcnet.py#L7).

    Parameters:
        conv_layers (`int`):
            The number of location variable convolutional layers to output kernels and biases for.
        conv_in_channels (`int`):
            The number of input channels for the location variable convolutional layer kernels (convolutional weight
            tensor).
        conv_out_channels (`int`):
            The number of output channels for the location variable convolutional layer kernels (convolutional weight
            tensor).
        conv_kernel_size (`int`, *optional*, defaults to 3):
            The kernel size for the location variable convolutional layer kernels (convolutional weight tensor).
        num_blocks (`int`, *optional*, defaults to 3):
            The number of residual blocks in the kernel predictor residual network.
        resnet_in_channels (`int`, *optional*, defaults to 100):
            The number of input channels to the kernel predictor residual network. This should be the same as the
            number of frequency bins in the conditioning log-mel spectrogram.
        resnet_hidden_channels (`int`, *optional*, defaults to 64):
            The number of hidden channels for each residual block in the kernel predictor residual network.
        resnet_kernel_size (`int`, *optional*, defaults to 3):
            The kernel size of each 1D convolutional layer in the kernel predictor residual network.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for the kernel predictor.
        leaky_relu_slope (`float`, *optional*, defaults 0.1):
            The angle of the negative slope used by the leaky ReLU activation.
    """

    def __init__(
        self,
        conv_layers: int,
        conv_in_channels: int,
        conv_out_channels: int,
        conv_kernel_size: int = 3,
        num_blocks: int = 3,
        resnet_in_channels: int = 100,
        resnet_hidden_channels: int = 64,
        resnet_kernel_size: int = 3,
        dropout: float = 0.0,
        leaky_relu_slope: float = 0.1,
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

        self.input_conv = nn.Conv1d(resnet_in_channels, resnet_hidden_channels, 5, padding=2, bias=True)

        self.resblocks = nn.ModuleList(
            [
                UnivNetKernelPredictorResidualBlock(
                    channels=resnet_hidden_channels,
                    kernel_size=resnet_kernel_size,
                    dropout=dropout,
                    leaky_relu_slope=leaky_relu_slope,
                )
                for _ in range(num_blocks)
            ]
        )

        self.kernel_conv = nn.Conv1d(
            resnet_hidden_channels, kernel_channels, resnet_kernel_size, padding=padding, bias=True
        )
        self.bias_conv = nn.Conv1d(
            resnet_hidden_channels, bias_channels, resnet_kernel_size, padding=padding, bias=True
        )

    def forward(self, spectrogram: torch.FloatTensor):
        """
        Maps a conditioning log-mel spectrogram to a tensor of convolutional kernels and biases, for use in location
        variable convolutional layers. Note that the input spectrogram should have shape (batch_size, input_channels,
        seq_length).

        Args:
            spectrogram (`torch.FloatTensor` of shape `(batch_size, input_channels, seq_length)`):
                Tensor containing the log-mel spectrograms.

        Returns:
            Tuple[`torch.FloatTensor, `torch.FloatTensor`]: tuple of tensors where the first element is the tensor of
            location variable convolution kernels of shape `(batch_size, self.conv_layers, self.conv_in_channels,
            self.conv_out_channels, self.conv_kernel_size, seq_length)` and the second element is the tensor of
            location variable convolution biases of shape `(batch_size, self.conv_layers. self.conv_out_channels,
            seq_length)`.
        """
        batch_size, _, seq_length = spectrogram.shape

        hidden_states = self.input_conv(spectrogram)
        hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)

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

    def get_padding(self, kernel_size: int, dilation: int = 1):
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
    """
    Implementation of the location variable convolution (LVC) residual block for the UnivNet residual network.

    Parameters:
        hidden_channels (`int`):
            The number of hidden channels for the residual block.
        kernel_size (`int`):
            The kernel size for the dilated 1D convolutional layer.
        dilation (`int`):
            The dilation for the dilated 1D convolutional layer.
        leaky_relu_slope (`float`, *optional*, defaults to 0.2):
            The angle of the negative slope used by the leaky ReLU activation.
    """

    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation: int,
        leaky_relu_slope: float = 0.2,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.leaky_relu_slope = leaky_relu_slope

        self.conv = nn.Conv1d(
            hidden_channels,
            hidden_channels,
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
        hidden_states = torch.sigmoid(hidden_states[:, : self.hidden_channels, :]) * torch.tanh(
            hidden_states[:, self.hidden_channels :, :]
        )
        # Skip connection
        hidden_states = residual + hidden_states

        return hidden_states

    # Based on https://github.com/mindslab-ai/univnet/blob/master/model/lvcnet.py#L171
    def location_variable_convolution(
        self,
        hidden_states: torch.FloatTensor,
        kernel: torch.FloatTensor,
        bias: torch.FloatTensor,
        dilation: int = 1,
        hop_size: int = 256,
    ):
        """
        Performs location-variable convolution operation on the input sequence (hidden_states) using the local
        convolution kernal. This was introduced in [LVCNet: Efficient Condition-Dependent Modeling Network for Waveform
        Generation](https://arxiv.org/abs/2102.10815) by Zhen Zheng, Jianzong Wang, Ning Cheng, and Jing Xiao.

        Time: 414 μs ± 309 ns per loop (mean ± std. dev. of 7 runs, 1000 loops each), test on NVIDIA V100.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, in_channels, in_length)`):
                The input sequence of shape (batch, in_channels, in_length).
            kernel (`torch.FloatTensor` of shape `(batch_size, in_channels, out_channels, kernel_size, kernel_length)`):
                The local convolution kernel of shape (batch, in_channels, out_channels, kernel_size, kernel_length).
            bias (`torch.FloatTensor` of shape `(batch_size, out_channels, kernel_length)`):
                The bias for the local convolution of shape (batch, out_channels, kernel_length).
            dilation (`int`, *optional*, defaults to 1):
                The dilation of convolution.
            hop_size (`int`, *optional*, defaults to 256):
                The hop_size of the conditioning sequence.
        Returns:
            `torch.FloatTensor`: the output sequence after performing local convolution with shape (batch_size,
            out_channels, in_length).
        """
        batch, _, in_length = hidden_states.shape
        batch, _, out_channels, kernel_size, kernel_length = kernel.shape
        assert in_length == (kernel_length * hop_size), "length of (hidden_states, kernel) is not matched"

        padding = dilation * int((kernel_size - 1) / 2)
        hidden_states = nn.functional.pad(
            hidden_states, (padding, padding), "constant", 0
        )  # (batch, in_channels, in_length + 2*padding)
        hidden_states = hidden_states.unfold(
            2, hop_size + 2 * padding, hop_size
        )  # (batch, in_channels, kernel_length, hop_size + 2*padding)

        if hop_size < dilation:
            hidden_states = nn.functional.pad(hidden_states, (0, dilation), "constant", 0)
        hidden_states = hidden_states.unfold(
            3, dilation, dilation
        )  # (batch, in_channels, kernel_length, (hop_size + 2*padding)/dilation, dilation)
        hidden_states = hidden_states[:, :, :, :, :hop_size]
        hidden_states = hidden_states.transpose(
            3, 4
        )  # (batch, in_channels, kernel_length, dilation, (hop_size + 2*padding)/dilation)
        hidden_states = hidden_states.unfold(
            4, kernel_size, 1
        )  # (batch, in_channels, kernel_length, dilation, _, kernel_size)

        # Apply local convolutional kernel to hidden_states.
        output_hidden_states = torch.einsum("bildsk,biokl->bolsd", hidden_states, kernel)

        output_hidden_states = output_hidden_states.to(memory_format=torch.channels_last_3d)
        bias = bias.unsqueeze(-1).unsqueeze(-1).to(memory_format=torch.channels_last_3d)
        output_hidden_states = output_hidden_states + bias
        output_hidden_states = output_hidden_states.contiguous().view(batch, out_channels, -1)

        return output_hidden_states

    def get_padding(self, kernel_size: int, dilation: int = 1):
        return dilation * (kernel_size - 1) // 2

    def apply_weight_norm(self):
        nn.utils.weight_norm(self.conv)

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv)


class UnivNetLVCBlock(nn.Module):
    """
    Implementation of the location variable convolution (LVC) residual block of the UnivNet residual block. Includes a
    `UnivNetKernelPredictor` inside to predict the kernels and biases of the LVC layers.

    Based on LVCBlock in [mindslab-ai/univnet](https://github.com/mindslab-ai/univnet/blob/master/model/lvcnet.py#L98)

    Parameters:
        hidden_channels (`int`, *optional*, defaults to 32):
            The number of hidden channels for the `UnivNetLVCResidualBlock`s.
        kernel_size (`int`, *optional*, defaults to 3):
            The kernel size for the dilated 1D convolutional layers in the residual blocks.
        stride (`int`, *optional*, defaults to 8):
            The stride for the dilated 1D convolutional layers in the residual blocks.
        dilations (`Union[Tuple[int], List[int]], *optional*, defaults to `[1, 3, 9, 27]`):
            A tuple of list of dilations for the dilated 1D convolutional layers in the residual blocks. The length of
            `dilations` determines how many residual blocks are in `UnivNetLVCBlock`.
        lvc_hop_size (`int`, *optional*, defaults to 256):
            The hop size for the location variable convolutional layers.
        resnet_leaky_relu_slope (`float`, *optional*, defaults to 0.2):
            The angle of the negative slope used by the leaky ReLU activation for the resnet blocks.
        cond_channels (`int`, *optional*, defaults to 100):
            The number of frequency bins in the conditioning log-mel spectrogram.
        kernel_predictor_num_blocks (`int`, *optional*, defaults to 3):
            The number of residual blocks for the kernel predictor.
        kernel_predictor_hidden_channels (`int`, *optional*, defaults to 64):
            The number of hidden channels for the residual blocks of the kernel predictor.
        kernel_predictor_conv_size (`int`, *optional*, defaults to 3):
            The kernel size for the 1D convolutional layers in the kernel predictor.
        kernel_predictor_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for the Dropout layers in the kernel predictor.
        kernel_predictor_leaky_relu_slope (`float`, *optional*, defaults to 0.2):
            The angle of the negative slope used by the leaky ReLU activation for the kernel predictor.
    """

    def __init__(
        self,
        hidden_channels: int = 32,
        kernel_size: int = 3,
        stride: int = 8,
        dilations: Union[Tuple[int], List[int]] = [1, 3, 9, 27],
        lvc_hop_size: int = 256,
        resnet_leaky_relu_slope: float = 0.2,
        cond_channels: int = 100,
        kernel_predictor_num_blocks: int = 3,
        kernel_predictor_hidden_channels: int = 64,
        kernel_predictor_conv_size: int = 3,
        kernel_predictor_dropout: float = 0.0,
        kernel_predictor_leaky_relu_slope: float = 0.2,
    ):
        super().__init__()
        self.cond_hop_length = lvc_hop_size
        self.leaky_relu_slope = resnet_leaky_relu_slope

        self.convt_pre = nn.ConvTranspose1d(
            hidden_channels,
            hidden_channels,
            2 * stride,
            stride=stride,
            padding=stride // 2 + stride % 2,
            output_padding=stride % 2,
        )

        self.kernel_predictor = UnivNetKernelPredictor(
            conv_layers=len(dilations),
            conv_in_channels=hidden_channels,
            conv_out_channels=2 * hidden_channels,
            conv_kernel_size=kernel_size,
            num_blocks=kernel_predictor_num_blocks,
            resnet_in_channels=cond_channels,
            resnet_hidden_channels=kernel_predictor_hidden_channels,
            resnet_kernel_size=kernel_predictor_conv_size,
            dropout=kernel_predictor_dropout,
            leaky_relu_slope=kernel_predictor_leaky_relu_slope,
        )

        self.resblocks = nn.ModuleList(
            [
                UnivNetLVCResidualBlock(
                    hidden_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilations[i],
                    leaky_relu_slope=resnet_leaky_relu_slope,
                )
                for i in range(len(dilations))
            ]
        )

    def forward(self, hidden_states: torch.FloatTensor, spectrogram: torch.FloatTensor):
        # hidden_states: (batch_size, hidden_channels, seq_length)
        # spectrogram: (batch_size, cond_channels, cond_length)
        hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
        hidden_states = self.convt_pre(hidden_states)

        kernels, biases = self.kernel_predictor(spectrogram)

        for i, resblock in enumerate(self.resblocks):
            kernel = kernels[:, i, :, :, :, :]
            bias = biases[:, i, :, :]
            hidden_states = resblock(hidden_states, kernel, bias, hop_size=self.cond_hop_length)

        return hidden_states

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


UNIVNET_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`UnivNetGanConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    """UnivNet GAN vocoder.""",
    UNIVNET_START_DOCSTRING,
)
class UnivNetGan(PreTrainedModel):
    config_class = UnivNetGanConfig
    main_input_name = "spectrogram"

    def __init__(self, config: UnivNetGanConfig):
        super().__init__(config)

        self.num_kernels = len(config.resblock_kernel_sizes)
        self.leaky_relu_slope = config.leaky_relu_slope

        self.conv_pre = nn.Conv1d(
            config.model_in_channels,
            config.model_hidden_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            padding_mode="reflect",
        )

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
                    hidden_channels=config.model_hidden_channels,
                    kernel_size=config.resblock_kernel_sizes[i],
                    stride=config.resblock_stride_sizes[i],
                    dilations=config.resblock_dilation_sizes[i],
                    lvc_hop_size=hop_lengths[i],
                    resnet_leaky_relu_slope=config.leaky_relu_slope,
                    cond_channels=config.num_mel_channels,
                    kernel_predictor_num_blocks=config.kernel_predictor_num_blocks,
                    kernel_predictor_hidden_channels=config.kernel_predictor_hidden_channels,
                    kernel_predictor_conv_size=config.kernel_predictor_conv_size,
                    kernel_predictor_dropout=config.kernel_predictor_dropout,
                    kernel_predictor_leaky_relu_slope=config.leaky_relu_slope,
                )
                for i in range(num_layers)
            ]
        )

        self.conv_post = nn.Conv1d(config.model_hidden_channels, 1, 7, padding=3, padding_mode="reflect")

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        spectrogram: torch.FloatTensor,
        noise_waveform: Optional[torch.FloatTensor] = None,
        generator: Optional[torch.Generator] = None,
    ):
        r"""
        Converts a noise waveform and a conditioning spectrogram to a speech waveform. Passing a batch of log-mel
        spectrograms returns a batch of speech waveforms. Passing a single, un-batched log-mel spectrogram returns a
        single, un-batched speech waveform.

        Args:
            spectrogram (`torch.FloatTensor`):
                Tensor containing the log-mel spectrograms. Can be batched and of shape `(batch_size, sequence_length,
                config.num_mel_channels)`, or un-batched and of shape `(sequence_length, config.num_mel_channels)`.
            noise_waveform (`torch.FloatTensor`, *optional*):
                Tensor containing a noise waveform sequence of standard Gaussian noise. Can be batched and of shape
                `(batch_size, sequence_length, config.model_in_channels)`, or un-batched and of shape (sequence_length,
                config.model_in_channels)`. If not supplied, will be randomly generated.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.

        Returns:
            `torch.FloatTensor`: Tensor containing the speech waveform. IF the input spectrogram is batched, wil be of
            shape `(batch_size, num_frames)`. If un-batched, will be of shape `(num_frames,)`.
        """
        # Resolve batch sizes for noise_waveform and spectrogram
        spectrogram_batched = spectrogram.dim() == 3
        if not spectrogram_batched:
            spectrogram = spectrogram.unsqueeze(0)
        spectrogram_batch_size, spectrogram_length, _ = spectrogram.shape

        if noise_waveform is not None:
            noise_waveform_batched = noise_waveform.dim() == 3
            if not noise_waveform_batched:
                noise_waveform = noise_waveform.unsqueeze(0)
        else:
            # Randomly generate noise_waveform
            noise_waveform_shape = (spectrogram_batch_size, spectrogram_length, self.config.model_in_channels)
            noise_waveform = torch.randn(
                noise_waveform_shape, generator=generator, dtype=spectrogram.dtype, device=spectrogram.device
            )
            noise_waveform_batched = True
        noise_waveform_batch_size = noise_waveform.shape[0]

        if spectrogram_batch_size > 1 and noise_waveform_batch_size == 1:
            # Repeat noise_waveform spectrogram_batch_size times
            noise_waveform = noise_waveform.repeat(spectrogram_batch_size, 1, 1)
        elif noise_waveform_batch_size > 1 and spectrogram_batch_size == 1:
            # Repeat spectrogram noise_waveform_batch_size times
            spectrogram = spectrogram.repeat(noise_waveform_batch_size, 1, 1)

        if noise_waveform_batch_size != spectrogram_batch_size:
            raise ValueError(
                f"The batch size of `noise_waveform` is {noise_waveform_batch_size} and the batch size of"
                f" `spectrogram` is {spectrogram_batch_size}, but the two are expected to be equal."
            )

        # Change shapes to have channels before sequence lengths
        hidden_states = noise_waveform.transpose(2, 1)
        spectrogram = spectrogram.transpose(2, 1)

        hidden_states = self.conv_pre(hidden_states)

        for resblock in self.resblocks:
            hidden_states = resblock(hidden_states, spectrogram)

        hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
        hidden_states = self.conv_post(hidden_states)
        hidden_states = torch.tanh(hidden_states)

        if spectrogram_batch_size > 1:
            # remove seq-len dim since this collapses to 1
            waveform = hidden_states.squeeze(1)
        else:
            # remove batch dim and collapse tensor to 1-d audio waveform
            waveform = hidden_states.squeeze(0).transpose(1, 0).view(-1)

        return waveform

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
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
