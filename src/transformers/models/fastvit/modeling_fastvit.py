# coding=utf-8
# Copyright 2023 Apple, Ross Wightman, The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch FastViT model."""


import collections.abc 
from collections import OrderedDict
import math
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_fastvit import FastViTConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "FastViTConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "apple/fastvit-t8"
_EXPECTED_OUTPUT_SHAPE = [1, 48, 64, 64]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "apple/fastvit-t8" 
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby cat"

#TODO: Add more new models there are atleast 5 more
FASTVIT_PRETRAINED_MODEL_ARCHIVE_LIST = [ 
    "apple/fastvit-t8",
    # See all FastViT models at https://huggingface.co/models?filter=fastvit
]

class FastViTEmbeddings(nn.Module):
    """
    Construct the patch embeddings. Optionally, also the mask token.
    """

    def __init__(self, config: FastViTConfig, inference_mode: bool = False) -> None:
        super().__init__()
        self.inference_mode = inference_mode

        self.patch_embeddings = FastViTPatchEmbeddings(config, inference_mode=inference_mode)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        pixel_values: torch.Tensor,
        interpolate_pos_encoding: bool = False
    ) -> torch.Tensor:

        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        embeddings = self.dropout(embeddings)

        return embeddings


class FastViTPatchEmbeddings(nn.Module):
    """
    Construction of the Stem Block, following paper structure here <https://arxiv.org/abs/2303.14189>.
    """
    def __init__(self, config: FastViTConfig, inference_mode: bool = False) -> None:
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_sizes[0]

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.inference_mode = inference_mode
        self.config = config

        self.projection = nn.Sequential(
            FastViTConvLayer(
            in_channels=num_channels, 
            out_channels=hidden_size, 
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            inference_mode = self.inference_mode),
            FastViTConvLayer(
            in_channels=hidden_size, 
            out_channels=hidden_size, 
            kernel_size=3,
            stride=2,
            padding=1,
            groups=hidden_size,
            inference_mode = self.inference_mode),
            FastViTConvLayer(
            in_channels=hidden_size, 
            out_channels=hidden_size, 
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            inference_mode = self.inference_mode)
        )

    def forward(self,
        pixel_values: torch.Tensor,
        interpolate_pos_encoding: bool = False
    ) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape

        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )

        embeddings = self.projection(pixel_values)

        return embeddings


class FastViTConvLayer(nn.Module):
    """
    Build of Convolution Layer following structure proposed in <https://arxiv.org/pdf/2303.14189.pdf>
    This block has a multi-branched architecture at train-time and plain-CNN style architecture 
    at inference time
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        inference_mode: bool = False,
        num_conv_branches: int = 1,
        use_scale_branch: bool = True,
        use_act: bool = True,
        use_se: bool = False,
        activation: nn.Module = nn.GELU(),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.inference_mode = inference_mode
        self.num_conv_branches = num_conv_branches
        self.use_scale_branch = use_scale_branch
        self.use_act = use_act
        self.use_se = use_se

        # Use of SE layer
        if use_se:
            self.se = FastViTSEBlock(out_channels)

        # Use of Activation layer
        if use_act:
            self.activation = activation

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
            )
        else:
            # skip connection
            self.rbr_skip = None
            if out_channels == in_channels and stride == 1:
                self.rbr_skip = nn.BatchNorm2d(num_features=in_channels)


            # Conv branches
            self.rbr_scale = None
            if kernel_size > 1 and self.use_scale_branch:
                self.rbr_scale = nn.Sequential(collections.OrderedDict({
                    'conv': nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=stride,
                        padding=0,
                        groups=groups,
                        bias=False,
                    ),
                    'bn': nn.BatchNorm2d(num_features=out_channels),
                })
                )

            self.rbr_conv = None
            if num_conv_branches > 0:
                self.rbr_conv = nn.Sequential(collections.OrderedDict({
                    'conv': nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        groups=groups,
                        bias=False,
                    ),
                    'bn': nn.BatchNorm2d(num_features=out_channels)
                })
                )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:

        # inference step
        if self.inference_mode:
            features =  self.reparam_conv(embeddings)
            # SE block
            if self.use_se:
                features = self.se(features)
            # Activation
            if self.use_act:
                features = self.activation(features)
            return features

        # Skip branch output
        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(embeddings)

        # Scale branch output
        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(embeddings)

        out = scale_out + identity_out
        if self.rbr_conv is not None:
            out = out + self.rbr_conv(embeddings)

        # SE block
        if self.use_se:
                out = self.se(out)
        # Activation
        if self.use_act:
            out = self.activation(out)

        return out
    
    def reparameterize(self):
        """Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        if self.inference_mode:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=True,
        )
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        # Delete un-used branches
        for para in self.parameters():
            para.detach_()
        self.__delattr__("rbr_conv")
        self.__delattr__("rbr_scale")
        if hasattr(self, "rbr_skip"):
            self.__delattr__("rbr_skip")

        self.inference_mode = True

    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        Returns:
            Tuple of (kernel, bias) after fusing branches.
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale, [pad, pad, pad, pad])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        if self.rbr_conv is not None:
            for ix in range(self.num_conv_branches):
                _kernel, _bias = self._fuse_bn_tensor(self.rbr_conv)
                kernel_conv += _kernel
                bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final
    
    def _fuse_bn_tensor(
        self, branch: Union[nn.Sequential, nn.BatchNorm2d]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Method to fuse batchnorm layer with preceeding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95

        Args:
            branch: Sequence of ops to be fused.

        Returns:
            Tuple of (kernel, bias) after fusing batchnorm.
        """
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros(
                    (self.in_channels, input_dim, self.kernel_size, self.kernel_size),
                    dtype=branch.weight.dtype,
                    device=branch.weight.device,
                )
                for i in range(self.in_channels):
                    kernel_value[
                        i, i % input_dim, self.kernel_size // 2, self.kernel_size // 2
                    ] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class FastViTSEBlock(nn.Module):
    """Squeeze and Excite module.

    Pytorch implementation of `Squeeze-and-Excitation Networks` -
    https://arxiv.org/pdf/1709.01507.pdf
    """

    def __init__(self, in_channels: int, rd_ratio: float = 0.0625) -> None:
        """Construct a Squeeze and Excite Module.

        Args:
            in_channels: Number of input channels.
            rd_ratio: Input channel reduction ratio.
        """
        super(FastViTSEBlock, self).__init__()
        self.reduce = nn.Conv2d(
            in_channels=in_channels,
            out_channels=int(in_channels * rd_ratio),
            kernel_size=1,
            stride=1,
            bias=True,
        )
        self.expand = nn.Conv2d(
            in_channels=int(in_channels * rd_ratio),
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply forward pass."""
        b, c, h, w = inputs.size()
        x = nn.functional.avg_pool2d(inputs, kernel_size=[h, w])
        x = self.reduce(x)
        x = nn.functional.relu(x)
        x = self.expand(x)
        x = torch.sigmoid(x)
        x = x.view(-1, c, 1, 1)
        return inputs * x


class FastViTReparamLKConv(nn.Module):
    """Building Block of RepLKNet

    This class defines overparameterized large kernel conv block
    introduced in `RepLKNet <https://arxiv.org/abs/2203.06717>`_

    Reference: https://github.com/DingXiaoH/RepLKNet-pytorch
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int,
        small_kernel: int,
        inference_mode: bool = False,
    ) -> None:
        super().__init__()

        self.stride = stride
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        self.padding = kernel_size // 2
        self.inference_mode = inference_mode

        if inference_mode:
            self.lkb_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=self.padding,
                dilation=1,
                groups=groups,
                bias=True,
            )
        else:
            self.large_conv = nn.Sequential(collections.OrderedDict({
                    "conv": nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=self.padding,
                        groups=groups,
                        bias=False,
                    ),
                    "bn": nn.BatchNorm2d(num_features=out_channels)
            })
                )

            if small_kernel is not None:
                assert (
                    small_kernel <= kernel_size
                ), "The kernel size for re-param cannot be larger than the large kernel!"
                
                self.small_conv = nn.Sequential(collections.OrderedDict({
                    "conv": nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=small_kernel,
                        stride=stride,
                        padding=small_kernel // 2,
                        groups=groups,
                        bias=False,
                    ),
                    "bn": nn.BatchNorm2d(num_features=out_channels)
                })
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward pass."""
        if self.inference_mode:
            out = self.lkb_reparam(x)
        else:
            out = self.large_conv(x)
            if hasattr(self, "small_conv"):
                out += self.small_conv(x)

        return out

    def get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepLKNet-pytorch

        Returns:
            Tuple of (kernel, bias) after fusing branches.
        """
        eq_k, eq_b = self._fuse_bn(self.large_conv.conv, self.large_conv.bn)
        if hasattr(self, "small_conv"):
            small_k, small_b = self._fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            eq_k += nn.functional.pad(
                small_k, [(self.kernel_size - self.small_kernel) // 2] * 4
            )
        return eq_k, eq_b

    def reparameterize(self) -> None:
        """
        Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        eq_k, eq_b = self.get_kernel_bias()
        self.lkb_reparam = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.large_conv.conv.dilation,
            groups=self.groups,
            bias=True,
        )

        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__("large_conv")
        if hasattr(self, "small_conv"):
            self.__delattr__("small_conv")

    @staticmethod
    def _fuse_bn(
        conv: torch.Tensor, bn: nn.BatchNorm2d
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Method to fuse batchnorm layer with conv layer.

        Args:
            conv: Convolutional kernel weights.
            bn: Batchnorm 2d layer.

        Returns:
            Tuple of (kernel, bias) after fusing batchnorm.
        """
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class FastViTSelfAttention(nn.Module):
    def __init__(self, config: FastViTConfig, stage: str) -> None:
        super().__init__()
        self.hidden_size = config.hidden_sizes[stage]
        self.attention_head_dim = config.attention_head_dim
        self.num_heads = int(self.hidden_size / self.attention_head_dim)
        self.scale = self.attention_head_dim ** -0.5
        
        self.all_head_size = self.attention_head_dim * self.num_heads

        if self.hidden_size % config.attention_head_dim != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {self.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.attention_head_dim}."
            )

        self.qkv = nn.Linear(self.hidden_size, self.all_head_size * 3, bias=config.qkv_bias)
        self.proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.dropout_proj = nn.Dropout(config.hidden_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.attention_head_dim, self.num_heads)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        B, C, H, W = hidden_states.shape
        N = H * W
        # Added this line, we expect always 4D Input convert
        # from shape (batch_size, channels, orig_height, orig_width)
        # to the shape (batch_size * patch_area, num_patches, channels)
        hidden_states = torch.flatten(hidden_states, start_dim=2).transpose(-2, -1) #B N C

        qkv = (
            self.qkv(hidden_states)
            .reshape(B, N, 3, self.num_heads, self.attention_head_dim)
            .permute(2,0,3,1,4)
        )
        query_layer, key_layer, value_layer = qkv.unbind(0)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = (query_layer * self.scale) @ key_layer.transpose(-2, -1)

        # Normalize the attention scores to probabilities.
        attention_probs = attention_scores.softmax(dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = (attention_probs @ value_layer).transpose(1,2).reshape(B, N, C)

        attention_outputs = self.proj(context_layer)
        attention_outputs = self.dropout_proj(attention_outputs)

        # Convert to 4D tensor
        attention_outputs = attention_outputs.transpose(-2, -1).reshape(B, C, H, W)

        return attention_outputs


class FastViTRepMixer(nn.Module):
    """
    Part of Metaformer block with RepMixer as token mixer, uses structural 
    reparameterization to lower the memory access cost by removing skip-connections in the network.
    For more info: `MetaFormer Is Actually What You Need for Vision <https://arxiv.org/pdf/2111.11418.pdf>`_
    """
    def __init__(self, config: FastViTConfig, stage: str, inference_mode: bool = False) -> None:
        super().__init__()
        dimension = config.hidden_sizes[stage]
        kernel_size = 3
        layer_norm_eps = config.layer_norm_eps
        self.inference_mode = inference_mode
        self.dimension = dimension
        self.kernel_size = kernel_size

        if self.inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=dimension,
                out_channels=dimension,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                groups=dimension,
                bias=True,
            )
        else:
            self.norm = FastViTConvLayer(            
                    in_channels=dimension, 
                    out_channels=dimension, 
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    groups=dimension,
                    use_act=False,
                    use_scale_branch=False,
                    num_conv_branches = 0
                )

            self.mixer = FastViTConvLayer(            
                    in_channels=dimension, 
                    out_channels=dimension, 
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    groups=dimension,
                    use_act=False)
            
            self.layer_scale = nn.Parameter(
                layer_norm_eps * torch.ones((dimension, 1, 1)), requires_grad=True
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.inference_mode:
            features_probs = self.reparam_conv(features)
        else:
            features_norm = self.norm(features)
            features_mixer = self.mixer(features)
            features_probs = features + self.layer_scale * (features_mixer - features_norm)

        return features_probs
    
    def reparameterize(self) -> None:
        """Reparameterize mixer and norm into a single
        convolutional layer for efficient inference.
        """
        if self.inference_mode:
            return
        
        self.mixer.reparameterize()
        self.norm.reparameterize()

        w = self.mixer.id_tensor + self.layer_scale.unsqueeze(-1) * (
            self.mixer.reparam_conv.weight - self.norm.reparam_conv.weight
        )
        b = torch.squeeze(self.layer_scale) * (
            self.mixer.reparam_conv.bias - self.norm.reparam_conv.bias
        )

        self.reparam_conv = nn.Conv2d(
            in_channels=self.dimension,
            out_channels=self.dimension,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
            groups=self.dimension,
            bias=True,
        )
        self.reparam_conv.weight.data = w
        self.reparam_conv.bias.data = b

        for para in self.parameters():
            para.detach_()
        self.__delattr__("mixer")
        self.__delattr__("norm")
        self.__delattr__("layer_scale")


class FastViTConvFFN(nn.Module):
    def __init__(self, config: FastViTConfig, stage: str) -> None:
        super().__init__()
        mlp_ratio = config.mlp_ratio
        hidden_size = config.hidden_sizes[stage]
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        act_layer = ACT2FN[config.hidden_act]
        dropout_rate = config.attention_probs_dropout_prob

        self.conv = nn.Conv2d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=7,
            padding=3,
            groups=hidden_size,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(num_features=hidden_size)
        self.fc1 = nn.Conv2d(
            in_channels=hidden_size,
            out_channels=mlp_hidden_dim,
            kernel_size = 1
        )
        self.act = act_layer
        self.fc2 = nn.Conv2d(
            in_channels=mlp_hidden_dim,
            out_channels=hidden_size,
            kernel_size = 1
        )
        self.drop = nn.Dropout(dropout_rate)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x    


class FastViTDownsample(nn.Module):
    def __init__(self, config: FastViTConfig, stage: str, inference_mode : bool = False) -> None:
        super().__init__()
        hidden_size = config.hidden_sizes[stage]
        hidden_size_next = config.hidden_sizes[stage+1]
        
        self.reparam_large_conv = FastViTReparamLKConv(
            in_channels=hidden_size,
            out_channels=hidden_size_next,
            kernel_size=7,
            stride=2,
            groups=hidden_size,
            small_kernel=3,
            inference_mode=inference_mode
        )
        self.conv = FastViTConvLayer(
            in_channels=hidden_size_next,
            out_channels=hidden_size_next,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            inference_mode=inference_mode
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reparam_large_conv(x)
        x = self.conv(x)
        return x


class FastViTAttention(nn.Module):
    def __init__(self, config, stage: str) -> None:
        super().__init__()
        hidden_size = config.hidden_sizes[stage]
        self.patch_size = config.patch_size

        self.norm = nn.BatchNorm2d(num_features=hidden_size)
        self.attention = FastViTSelfAttention(config, stage)
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Apply Layernorm
        hidden_states = self.norm(hidden_states)

        # Apply attention
        attention = self.attention(hidden_states)
        return attention


class FastViTMixer(nn.Module):
    """
    This class is an implementation of Metaformer block with RepMixer as token mixer.
    For more info: `MetaFormer Is Actually What You Need for Vision <https://arxiv.org/pdf/2111.11418.pdf>`_
    
    """
    def __init__(self, config: FastViTConfig, stage: str) -> None:
        super().__init__()
        self.token_mixer = FastViTRepMixer(config, stage)

    def forward(self, x : torch.tensor) -> torch.tensor:
        x = self.token_mixer(x)
        return x


class FastViTCPE(nn.Module):
    """Implementation of conditional positional encoding.

    For more details refer to paper:
    `Conditional Positional Encodings for Vision Transformers <https://arxiv.org/abs/2102.10882>`

    In our implementation, we can reparameterize this module to eliminate a skip connection for Inference step.

    """
    def __init__(self, 
                in_channels: int, 
                embed_dim: int = 768,
                spatial_shape: Union[int, Tuple[int, int]] = (7, 7),
                inference_mode: bool = False
                ) -> None:
        super().__init__()
        if isinstance(spatial_shape, int):
            spatial_shape = tuple([spatial_shape] * 2)

        assert isinstance(spatial_shape, Tuple), (
            f'"spatial_shape" must by a sequence or int, '
            f"get {type(spatial_shape)} instead."
        )

        assert len(spatial_shape) == 2, (
            f'Length of "spatial_shape" should be 2, '
            f"got {len(spatial_shape)} instead."
        )
        
        self.spatial_shape = spatial_shape
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.groups = embed_dim
        self.inference_mode = inference_mode
        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.embed_dim,
                kernel_size=self.spatial_shape,
                stride=1,
                padding=int(self.spatial_shape[0] // 2),
                groups=self.embed_dim,
                bias=True,
            )
        else:
            self.pos_enc = nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.embed_dim,
                    kernel_size=self.spatial_shape,
                    stride=1,
                    padding=int(self.spatial_shape[0] // 2),
                    groups=self.embed_dim,
                    bias=True,
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.inference_mode:
            CPE = self.reparam_conv(hidden_states)
        else:
            CPE = self.pos_enc(hidden_states) + hidden_states

        return CPE

    def reparameterize(self) -> None:
        # Build equivalent Id tensor
        input_dim = self.in_channels // self.groups
        kernel_value = torch.zeros(
            (
                self.in_channels,
                input_dim,
                self.spatial_shape[0],
                self.spatial_shape[1],
            ),
            dtype=self.pos_enc.weight.dtype,
            device=self.pos_enc.weight.device,
        )
        for i in range(self.in_channels):
            kernel_value[
                i,
                i % input_dim,
                self.spatial_shape[0] // 2,
                self.spatial_shape[1] // 2,
            ] = 1
        id_tensor = kernel_value

        # Reparameterize Id tensor and conv
        w_final = id_tensor + self.pos_enc.weight
        b_final = self.pos_enc.bias

        # Introduce reparam conv
        self.reparam_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.embed_dim,
            kernel_size=self.spatial_shape,
            stride=1,
            padding=int(self.spatial_shape[0] // 2),
            groups=self.embed_dim,
            bias=True,
        )
        self.reparam_conv.weight.data = w_final
        self.reparam_conv.bias.data = b_final

        for para in self.parameters():
            para.detach_()
        self.__delattr__("pe")


class FastViTIntermediate(nn.Module):
    def __init__(self, config: FastViTConfig, stage: str) -> None:
        super().__init__()
        token_mixer_type = config.token_mixers[stage]
        drop_path = config.attention_probs_dropout_prob
        layer_scale_init_value = config.layer_norm_eps
        hidden_size = config.hidden_sizes[stage]
        self.depth = config.depths[stage]
        self.token_mixer_type = token_mixer_type

        self.drop_path = nn.Dropout(drop_path) if drop_path > 0.0 else nn.Identity()
        if token_mixer_type == "repmixer":
            self.token_mixer_block = FastViTMixer(config, stage)
            self.layer_scale = nn.Parameter(
                    layer_scale_init_value * torch.ones((hidden_size, 1, 1)), requires_grad=True
                )
        elif token_mixer_type == "attention":
            self.token_mixer_block = FastViTAttention(config, stage)
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((hidden_size, 1, 1)), requires_grad=True
            )
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((hidden_size, 1, 1)), requires_grad=True
            )
        else:
            raise ValueError(
                "Token mixer type: {} not supported".format(token_mixer_type)
            )
        self.convffn = FastViTConvFFN(config, stage)


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output_token_mixer = self.token_mixer_block(hidden_states)
        if self.token_mixer_type == "repmixer":
            output = output_token_mixer + self.drop_path(self.layer_scale * self.convffn(output_token_mixer))
        else:            
            output = hidden_states + self.drop_path(self.layer_scale_1 * output_token_mixer)
            output = output + self.drop_path(self.layer_scale_2 * self.convffn(output))
        return output


class FastViTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: FastViTConfig, stage: int, inference_mode: bool = False) -> None:
        super().__init__()
        self.stage = stage
        pos_embeds = config.pos_embeds
        depth = config.depths[stage]

        if pos_embeds is None:
            pos_embeds = [None] * len(config.depths)

        self.pos_emb = None
        if pos_embeds[stage] is not None:
            self.pos_emb = FastViTCPE(config.hidden_sizes[stage], 
                                                config.hidden_sizes[stage], 
                                                spatial_shape=(7, 7), 
                                                inference_mode = inference_mode)

        self.stage_conv = nn.ModuleList()
        for _ in range(depth):
            stage_layer = FastViTIntermediate(config, stage)
            self.stage_conv.append(stage_layer)

        self.downsample = None
        if stage+1 < len(config.depths):
            self.downsample = FastViTDownsample(config, stage)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        if self.pos_emb:
            hidden_states = self.pos_emb(hidden_states)

        features = hidden_states
        for layer_module in self.stage_conv:
            features = layer_module(features)

        if self.downsample:
            features = self.downsample(features)

        return features


class FastViTEncoder(nn.Module):
    def __init__(self, config: FastViTConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([FastViTLayer(config, i) for i in range(len(config.depths))])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                )
            else:
                layer_outputs = layer_module(hidden_states)

            hidden_states = layer_outputs

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class FastViTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = FastViTConfig
    base_model_prefix = "fastvit"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = []

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    def _set_gradient_checkpointing(self, module: FastViTEncoder, value: bool = False) -> None:
        if isinstance(module, FastViTEncoder):
            module.gradient_checkpointing = value


FASTVIT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`FastViTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

FASTVIT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`ViTImageProcessor.__call__`]
            for details.

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        interpolate_pos_encoding (`bool`, *optional*):
            Whether to interpolate the pre-trained position encodings.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare FastViT Model transformer outputting raw hidden-states without any specific head on top.",
    FASTVIT_START_DOCSTRING,
)
class FastViTModel(FastViTPreTrainedModel):
    def __init__(self, config: FastViTConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = FastViTEmbeddings(config)

        self.encoder = FastViTEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> FastViTPatchEmbeddings:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(FASTVIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
        if hasattr(self, "rbr_scale"):
            expected_dtype = self.embeddings.patch_embeddings.projection[0].rbr_scale[0].weight.dtype
            if pixel_values.dtype != expected_dtype:
                pixel_values = pixel_values.to(expected_dtype)

        embedding_output = self.embeddings(
            pixel_values, interpolate_pos_encoding=interpolate_pos_encoding
        )
        
        encoder_outputs = self.encoder(
            embedding_output,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = encoder_outputs[0]

        if not return_dict:
            head_outputs = (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=None,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@add_start_docstrings(
    """
    FastViT Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
    the [CLS] token) e.g. for ImageNet.

    <Tip>

        Note that it's possible to fine-tune FastViT on higher resolution images than the ones it has been trained on, by
        setting `interpolate_pos_encoding` to `True` in the forward of the model. This will interpolate the pre-trained
        position embeddings to the higher resolution.

    </Tip>
    """,
    FASTVIT_START_DOCSTRING,
)
class FastViTForImageClassification(FastViTPreTrainedModel):
    def __init__(self, config: FastViTConfig, inference_mode: bool = False) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.inference_mode = inference_mode
        self.fastvit = FastViTModel(config)

        # Classifier head
        hidden_size = config.hidden_sizes[-1]
        self.final_conv = FastViTConvLayer(
                in_channels=hidden_size,
                out_channels=int(hidden_size * 2),
                kernel_size=3,
                stride=1,
                padding=1,
                groups=hidden_size,
                inference_mode=self.inference_mode,
                use_se=True,
                num_conv_branches=1,
            )
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Linear(int(hidden_size * 2), config.num_labels) if config.num_labels > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(FASTVIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.fastvit(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.final_conv(sequence_output)
        sequence_output = self.gap(sequence_output)

        logits = self.classifier(sequence_output[:, :, 0].flatten(1))

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
