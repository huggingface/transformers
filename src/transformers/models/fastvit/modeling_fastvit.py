# coding=utf-8
# Copyright 2023 Google AI, Ross Wightman, The HuggingFace Inc. team. All rights reserved.
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
    ImageClassifierOutput,
    MaskedImageModelingOutput,
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
_IMAGE_CLASS_EXPECTED_OUTPUT = "Egyptian cat"

#TODO: Add more new models there are atleast 5 more
FASTVIT_PRETRAINED_MODEL_ARCHIVE_LIST = [ 
    "apple/fastvit-t8",
    # See all FastViT models at https://huggingface.co/models?filter=fastvit
]


class FastViTEmbeddings(nn.Module):
    """
    Construct the patch embeddings. Optionally, also the mask token.
    """

    def __init__(self, config: FastViTConfig, use_mask_token: bool = False) -> None:
        super().__init__()

        self.patch_embeddings = FastViTPatchEmbeddings(config)
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
    def __init__(self, config: FastViTConfig) -> None:
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
        self.inference_mode = config.inference_mode
        self.config = config

        self.projection_first_conv = FastViTConvLayer(
            in_channels=num_channels, 
            out_channels=hidden_size, 
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            inference_mode = self.inference_mode)
        self.projection_second_conv = FastViTConvLayer(
            in_channels=hidden_size, 
            out_channels=hidden_size, 
            kernel_size=3,
            stride=2,
            padding=1,
            groups=hidden_size,
            inference_mode = self.inference_mode)
        self.projection_third_conv = FastViTConvLayer(
            in_channels=hidden_size, 
            out_channels=hidden_size, 
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            inference_mode = self.inference_mode)

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

        embeddings = self.projection_first_conv(pixel_values)
        embeddings = self.projection_second_conv(embeddings)
        embeddings = self.projection_third_conv(embeddings)

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
                self.rbr_scale = nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=stride,
                        padding=0,
                        groups=groups,
                        bias=False,
                    ),
                    nn.BatchNorm2d(num_features=out_channels)
                )

            self.rbr_conv = None
            if num_conv_branches > 0:
                self.rbr_conv = nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        groups=groups,
                        bias=False,
                    ),
                    nn.BatchNorm2d(num_features=out_channels)
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
        activation: nn.Module = nn.GELU(),
    ) -> None:
        super().__init__()

        self.stride = stride
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
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
            self.lkb_origin = nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=self.padding,
                        groups=groups,
                        bias=False,
                    ),
                    nn.BatchNorm2d(num_features=out_channels)
                )

            if small_kernel is not None:
                assert (
                    small_kernel <= kernel_size
                ), "The kernel size for re-param cannot be larger than the large kernel!"
                
                self.small_conv = nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=small_kernel,
                        stride=stride,
                        padding=small_kernel // 2,
                        groups=groups,
                        bias=False,
                    ),
                    nn.BatchNorm2d(num_features=out_channels)
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward pass."""
        if self.inference_mode:
            out = self.lkb_reparam(x)
        else:
            out = self.lkb_origin(x)
            if hasattr(self, "small_conv"):
                out += self.small_conv(x)

        out = self.activation(out)
        return out


class FastViTSelfAttention(nn.Module):
    def __init__(self, config: FastViTConfig, stage: str) -> None:
        super().__init__()
        self.hidden_size = config.hidden_sizes[stage]
        self.num_attention_heads = config.num_attention_heads
        

        if self.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {self.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(self.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(self.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(self.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class FastViTRepMixer(nn.Module):
    """
    Part of Metaformer block with RepMixer as token mixer, uses structural 
    reparameterization to lower the memory access cost by removing skip-connections in the network.
    For more info: `MetaFormer Is Actually What You Need for Vision <https://arxiv.org/pdf/2111.11418.pdf>`_
    """
    def __init__(self, config: FastViTConfig, stage: str) -> None:
        super().__init__()
        dimension = config.hidden_sizes[stage]
        kernel_size = 3
        layer_norm_eps = config.layer_norm_eps
        self.inference_mode = config.inference_mode

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
        self.batch_norm = nn.BatchNorm2d(num_features=hidden_size)
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
        x = self.batch_norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x    


class FastViTDownsample(nn.Module):
    def __init__(self, config: FastViTConfig, stage: str) -> None:
        super().__init__()
        hidden_size = config.hidden_sizes[stage]
        hidden_size_next = config.hidden_sizes[stage+1]
        inference_mode = config.inference_mode

        self.reparam_LargeKernel_conv = FastViTReparamLKConv(
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
        x = self.reparam_LargeKernel_conv(x)
        x = self.conv(x)
        return x


class FastViTAttention(nn.Module):
    def __init__(self, config, stage: str) -> None:
        super().__init__()
        hidden_size = config.hidden_sizes[stage]
        drop_path = config.hidden_dropout_prob
        layer_scale_init_value = config.layer_norm_eps
        self.patch_size = config.patch_size

        self.layer_norm = nn.BatchNorm2d(num_features=hidden_size)
        self.token_mixer = FastViTSelfAttention(config, stage)

        self.convffn = FastViTConvFFN(config, stage)

        self.drop_path = nn.Dropout(drop_path) if drop_path > 0.0 else nn.Identity()

        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((hidden_size, 1, 1)), requires_grad=True
        )
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((hidden_size, 1, 1)), requires_grad=True
        )

        self.pruned_heads = set()

    def unfolding(self, features: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        patch_height = int(self.patch_size / 2)
        patch_width = int(self.patch_size / 2)

        batch_size, channels, orig_height, orig_width = features.shape
        new_height = int(math.ceil(orig_height / patch_height) * patch_height)
        new_width = int(math.ceil(orig_width / patch_width) * patch_width)

        interpolate = False
        if new_width != orig_width or new_height != orig_height:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            features = nn.functional.interpolate(
                features, size=(new_height, new_width), mode="bilinear", align_corners=False
            )
            interpolate = True

        num_patch_width = new_width // patch_width
        num_patch_height = new_height // patch_height
        num_patches = num_patch_height * num_patch_width

        # convert from shape (batch_size, channels, orig_height, orig_width)
        # to the shape (batch_size * patch_area, num_patches, channels)
        patches = features.reshape(
            batch_size * channels * num_patch_height, patch_height, num_patch_width, patch_width
        )
        patches = patches.transpose(1, 2)
        patches = patches.reshape(batch_size, channels, num_patches, self.patch_size)
        patches = patches.transpose(1, 3)
        patches = patches.reshape(batch_size * self.patch_size, num_patches, -1)
        info_dict = {
            "orig_size": (orig_height, orig_width),
            "batch_size": batch_size,
            "channels": channels,
            "interpolate": interpolate,
            "num_patches": num_patches,
            "num_patches_width": num_patch_width,
            "num_patches_height": num_patch_height,
        }
        return patches, info_dict

    def folding(self, patches: torch.Tensor, info_dict: Dict) -> torch.Tensor:
        patch_height = int(self.patch_size / 2)
        patch_width = int(self.patch_size / 2)

        patch_area = int(patch_width * patch_height)

        batch_size = info_dict["batch_size"]
        channels = info_dict["channels"]
        num_patches = info_dict["num_patches"]
        num_patch_height = info_dict["num_patches_height"]
        num_patch_width = info_dict["num_patches_width"]

        # convert from shape (batch_size * patch_area, num_patches, channels)
        # back to shape (batch_size, channels, orig_height, orig_width)
        features = patches.contiguous().view(batch_size, patch_area, num_patches, -1)
        features = features.transpose(1, 3)
        features = features.reshape(
            batch_size * channels * num_patch_height, num_patch_width, patch_height, patch_width
        )
        features = features.transpose(1, 2)
        features = features.reshape(
            batch_size, channels, num_patch_height * patch_height, num_patch_width * patch_width
        )

        if info_dict["interpolate"]:
            features = nn.functional.interpolate(
                features, size=info_dict["orig_size"], mode="bilinear", align_corners=False
            )

        return features

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
        residutal_features = hidden_states

        before_layer_norm = self.layer_norm(hidden_states)

        # convert feature maps back to patches
        patches, info_dict = self.unfolding(before_layer_norm)

        # Apply token mixer 
        token_mixer = self.token_mixer(patches)

        # convert patches back to feature maps
        features = self.folding(token_mixer[0], info_dict)

        features_probs = self.drop_path(self.layer_scale_1 * features)
        embeddings = residutal_features + features_probs 

        residutal_features = embeddings
        features_conv = self.convffn(embeddings)
        features_probs = self.drop_path(self.layer_scale_2 * features_conv)

        embeddings = residutal_features + features_probs

        return embeddings


class FastViTMixer(nn.Module):
    """
    This class is an implementation of Metaformer block with RepMixer as token mixer.
    For more info: `MetaFormer Is Actually What You Need for Vision <https://arxiv.org/pdf/2111.11418.pdf>`_
    
    """
    def __init__(self, config: FastViTConfig, stage: str) -> None:
        super().__init__()

        hidden_size = config.hidden_sizes[stage]
        drop_path = config.hidden_dropout_prob
        layer_scale_init_value = config.layer_norm_eps

        self.token_mixer = FastViTRepMixer(config, stage)

        self.convffn = FastViTConvFFN(config, stage)

        self.drop_path = nn.Dropout(drop_path) if drop_path > 0.0 else nn.Identity()


        self.layer_scale = nn.Parameter(
            layer_scale_init_value * torch.ones((hidden_size, 1, 1)), requires_grad=True
        )

    def forward(self, x : torch.tensor) -> torch.tensor:
        x = self.token_mixer(x)
        x = x + self.drop_path(self.layer_scale * self.convffn(x))
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

        self.pe = nn.Conv2d(
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
            CPE = self.pe(hidden_states)
        else:
            CPE = self.pe(hidden_states) + hidden_states

        return CPE


class FastViTIntermediate(nn.Module):
    def __init__(self, config: FastViTConfig, stage: str) -> None:
        super().__init__()
        token_mixer_type = config.token_mixers[stage]
        self.depth = config.depths[stage]
        if token_mixer_type == "repmixer":
            self.token_mixer_block = FastViTMixer(config, stage)
        elif token_mixer_type == "attention":
            self.token_mixer_block = FastViTAttention(config, stage)
        else:
            raise ValueError(
                "Token mixer type: {} not supported".format(token_mixer_type)
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.token_mixer_block(hidden_states)

        return hidden_states


class FastViTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: FastViTConfig, stage: int) -> None:
        super().__init__()
        self.stage = stage
        pos_embeds = config.pos_embeds
        depth = config.depths[stage]
        inference_mode = config.inference_mode

        if pos_embeds is None:
            pos_embeds = [None] * len(config.depths)

        self.position_embeddings = None
        if pos_embeds[stage] is not None:
            self.position_embeddings = FastViTCPE(config.hidden_sizes[stage], 
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
        if self.position_embeddings:
            hidden_states = self.position_embeddings(hidden_states)

        for layer_module in self.stage_conv:
            features = layer_module(hidden_states)

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
        expected_dtype = self.embeddings.patch_embeddings.projection_first_conv.rbr_scale[0].weight.dtype
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
    def __init__(self, config: FastViTConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.inference_mode = config.inference_mode
        self.fastvit = FastViTModel(config)

        # Classifier head
        hidden_size = config.hidden_sizes[-1]
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv_exp = FastViTConvLayer(
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
        sequence_output = self.conv_exp(sequence_output)
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
