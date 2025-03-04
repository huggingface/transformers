# Copyright 2025 The HuggingFace Team. All rights reserved.
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
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from ...activations import ACT2CLS, ACT2FN
from ...configuration_utils import PretrainedConfig
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, rope_config_validation
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from ..cohere.modeling_cohere import apply_rotary_pos_emb
from ..llama.modeling_llama import LlamaAttention, eager_attention_forward
from ..rt_detr_v2.modeling_rt_detr_v2 import RTDetrV2ConvNormLayer
from ..superpoint.modeling_superpoint import SuperPointPreTrainedModel


"""PyTorch EfficientLoFTR model."""

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC_ = "EfficientLoFTRConfig"
_CHECKPOINT_FOR_DOC_ = "stevenbucaille/efficient_loftr"


def create_meshgrid(
    height: int,
    width: int,
    normalized_coordinates: bool = False,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    Copied from kornia library : kornia/kornia/utils/grid.py:26

    Generate a coordinate grid for an image.

    When the flag ``normalized_coordinates`` is set to True, the grid is
    normalized to be in the range :math:`[-1,1]` to be consistent with the pytorch
    function :py:func:`torch.nn.functional.grid_sample`.

    Args:
        height (`int`):
            The image height (rows).
        width (`int`):
            The image width (cols).
        normalized_coordinates (`bool`):
            Whether to normalize coordinates in the range :math:`[-1,1]` in order to be consistent with the
            PyTorch function :py:func:`torch.nn.functional.grid_sample`.
        device (`torch.device`):
            The device on which the grid will be generated.
        dtype (`torch.dtype`):
            The data type of the generated grid.

    Return:
        grid (`torch.Tensor` of shape `(1, height, width, 2)`):
            The grid tensor.

    Example:
        >>> create_meshgrid(2, 2)
        tensor([[[[-1., -1.],
                  [ 1., -1.]],
        <BLANKLINE>
                 [[-1.,  1.],
                  [ 1.,  1.]]]])

        >>> create_meshgrid(2, 2, normalized_coordinates=False)
        tensor([[[[0., 0.],
                  [1., 0.]],
        <BLANKLINE>
                 [[0., 1.],
                  [1., 1.]]]])

    """
    xs = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
    ys = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
    if normalized_coordinates:
        xs = (xs / (width - 1) - 0.5) * 2
        ys = (ys / (height - 1) - 0.5) * 2
    grid = torch.stack(torch.meshgrid(ys, xs, indexing="ij"), dim=-1)
    grid = grid.permute(1, 0, 2).unsqueeze(0)
    return grid


def spatial_expectation2d(input: torch.Tensor, normalized_coordinates: bool = True) -> torch.Tensor:
    r"""
    Copied from kornia library : kornia/geometry/subpix/dsnt.py:76
    Compute the expectation of coordinate values using spatial probabilities.

    The input heatmap is assumed to represent a valid spatial probability distribution,
    which can be achieved using :func:`~kornia.geometry.subpixel.spatial_softmax2d`.

    Args:
        input (`torch.Tensor` of shape `(batch_size, channels, height, width)`):
            The input tensor representing dense spatial probabilities.
        normalized_coordinates (`bool`):
            Whether to return the coordinates normalized in the range of :math:`[-1, 1]`. Otherwise, it will return
            the coordinates in the range of the input shape.

    Returns:
        output (`torch.Tensor` of shape `(batch_size, channels, 2)`)
            Expected value of the 2D coordinates. Output order of the coordinates is (x, y).

    Examples:
        >>> heatmaps = torch.tensor([[[
        ... [0., 0., 0.],
        ... [0., 0., 0.],
        ... [0., 1., 0.]]]])
        >>> spatial_expectation2d(heatmaps, False)
        tensor([[[1., 2.]]])

    """
    batch_size, channels, height, width = input.shape

    # Create coordinates grid.
    grid = create_meshgrid(height, width, normalized_coordinates, input.device)
    grid = grid.to(input.dtype)

    pos_x = grid[..., 0].reshape(-1)
    pos_y = grid[..., 1].reshape(-1)

    input_flat = input.view(batch_size, channels, -1)

    # Compute the expectation of the coordinates.
    expected_y = torch.sum(pos_y * input_flat, -1, keepdim=True)
    expected_x = torch.sum(pos_x * input_flat, -1, keepdim=True)

    output = torch.cat([expected_x, expected_y], -1)

    return output.view(batch_size, channels, 2)


def mask_border(tensor: torch.Tensor, border_margin: int, value: Union[bool, float, int]) -> torch.Tensor:
    """
    Mask a tensor border with a given value

    Args:
        tensor (`torch.Tensor` of shape `(batch_size, height_0, width_0, height_1, width_1)`):
            The tensor to mask
        border_margin (`int`) :
            The size of the border
        value (`Union[bool, int, float]`):
            The value to place in the tensor's borders

    Returns:
        tensor (`torch.Tensor` of shape `(batch_size, height_0, width_0, height_1, width_1)`):
            The masked tensor
    """
    if border_margin <= 0:
        return tensor

    tensor[:, :border_margin] = value
    tensor[:, :, :border_margin] = value
    tensor[:, :, :, :border_margin] = value
    tensor[:, :, :, :, :border_margin] = value
    tensor[:, -border_margin:] = value
    tensor[:, :, -border_margin:] = value
    tensor[:, :, :, -border_margin:] = value
    tensor[:, :, :, :, -border_margin:] = value
    return tensor


class EfficientLoFTRConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`EffientLoFTRFromKeypointMatching`].
    It is used to instantiate a EfficientLoFTR model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the
    EfficientLoFTR [stevenbucaille/efficient_loftr](https://huggingface.co/stevenbucaille/efficient_loftr) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        stage_block_dims (`List`, *optional*, defaults to [64, 64, 128, 256]):
            The hidden size of the features in the blocks of each stage
        stage_num_blocks (`List`, *optional*, defaults to [1, 2, 4, 14]):
            The number of blocks in each stages
        stage_hidden_expansion (`List`, *optional*, defaults to [1, 1, 1, 1]):
            The rate of expansion of hidden size in each stage
        stage_stride (`List`, *optional*, defaults to [2, 1, 2, 2]):
            The stride used in each stage
        hidden_size (`int`, *optional*, defaults to 256):
            The dimension of the descriptors.
        activation_function (`str`, *optional*, defaults to `"relu"`):
            The activation function used in the backbone
        aggregation_sizes (`List`, *optional*, defaults to [4, 4]):
            The size of each aggregation for the fusion network
        num_attention_layers (`int`, *optional*, defaults to 4):
            Number of attention layers in the LocalFeatureTransformer
        num_attention_heads (`int`, *optional*, defaults to 8):
            The number of heads in the GNN layers.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during attention.
        mlp_activation_function (`str`, *optional*, defaults to `"leaky_relu"`):
            Activation function used in the attention mlp layer.
        coarse_matching_skip_softmax (`bool`, *optional*, defaults to `False`):
            Whether to skip softmax or not at the coarse matching step.
        coarse_matching_threshold (`float`, *optional*, defaults to 0.2):
            The threshold for the minimum score required for a match.
        coarse_matching_temperature (`float`, *optional*, defaults to 0.1):
            The temperature to apply to the coarse similarity matrix
        coarse_matching_border_removal (`int`, *optional*, defaults to 2):
            The size of the border to remove during coarse matching
        fine_kernel_size (`int`, *optional*, defaults to 8):
            Kernel size used for the fine feature matching
        batch_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the batch normalization layers.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3', '2d'], with 'default' being the original RoPE implementation.
                `dim` (`int`): The dimension of the RoPE embeddings.
        fine_matching_slicedim (`int`, *optional*, defaults to 8):
            The size of the slice used to divide the fine features for the first and second fine matching stages.
        fine_matching_regress_temperature (`float`, *optional*, defaults to 10.0):
            The temperature to apply to the fine similarity matrix
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    Examples:
        ```python
        >>> from transformers import EfficientLoFTRConfig, EfficientLoFTRForKeypointMatching

        >>> # Initializing a SuperGlue superglue style configuration
        >>> configuration = EfficientLoFTRConfig()

        >>> # Initializing a model from the superglue style configuration
        >>> model = EfficientLoFTRForKeypointMatching(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
        ```
    """

    model_type = "efficientloftr"

    def __init__(
        self,
        stage_block_dims: List[int] = None,
        stage_num_blocks: List[int] = None,
        stage_hidden_expansion: List[float] = None,
        stage_stride: List[int] = None,
        hidden_size: int = 256,
        activation_function: str = "relu",
        aggregation_sizes: List[int] = None,
        num_attention_layers: int = 4,
        num_attention_heads: int = 8,
        num_key_value_heads: int = None,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        mlp_activation_function: str = "leaky_relu",
        coarse_matching_skip_softmax: bool = False,
        coarse_matching_threshold: float = 0.2,
        coarse_matching_temperature: float = 0.1,
        coarse_matching_border_removal: int = 2,
        fine_kernel_size: int = 8,
        batch_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        rope_scaling: Dict = None,
        fine_matching_slicedim: int = 8,
        fine_matching_regress_temperature: float = 10.0,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        self.stage_block_dims = stage_block_dims if stage_block_dims is not None else [64, 64, 128, 256]
        self.stage_num_blocks = stage_num_blocks if stage_num_blocks is not None else [1, 2, 4, 14]
        self.stage_hidden_expansion = stage_hidden_expansion if stage_hidden_expansion is not None else [1, 1, 1, 1]
        self.stage_stride = stage_stride if stage_stride is not None else [2, 1, 2, 2]
        self.hidden_size = hidden_size
        if self.hidden_size != self.stage_block_dims[-1]:
            raise ValueError(
                f"hidden_size should be equal to the last value in stage_block_dims. hidden_size = {self.hidden_size}, stage_blck_dims = {self.stage_block_dims}"
            )

        self.activation_function = activation_function
        self.aggregation_sizes = aggregation_sizes if aggregation_sizes is not None else [4, 4]
        self.num_attention_layers = num_attention_layers
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.mlp_activation_function = mlp_activation_function
        self.coarse_matching_skip_softmax = coarse_matching_skip_softmax
        self.coarse_matching_threshold = coarse_matching_threshold
        self.coarse_matching_temperature = coarse_matching_temperature
        self.coarse_matching_border_removal = coarse_matching_border_removal
        self.fine_kernel_size = fine_kernel_size
        self.batch_norm_eps = batch_norm_eps
        self.fine_matching_slicedim = fine_matching_slicedim
        self.fine_matching_regress_temperature = fine_matching_regress_temperature

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads

        self.rope_theta = rope_theta
        self.rope_scaling = (
            rope_scaling if rope_scaling is not None else {"rope_type": "2d", "dim": self.hidden_size // 4}
        )
        rope_config_validation(self)

        self.initializer_range = initializer_range

        super().__init__(**kwargs)


@dataclass
class KeypointMatchingOutput(ModelOutput):
    """
    Base class for outputs of keypoint matching models. Due to the nature of keypoint detection and matching, the number
    of keypoints is not fixed and can vary from image to image, which makes batching non-trivial. In the batch of
    images, the maximum number of matches is set as the dimension of the matches and matching scores. The mask tensor is
    used to indicate which values in the keypoints, matches and matching_scores tensors are keypoint matching
    information.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Loss computed during training.
        mask (`torch.IntTensor` of shape `(batch_size, num_keypoints)`):
            Mask indicating which values in matches and matching_scores are keypoint matching information.
        matches (`torch.FloatTensor` of shape `(batch_size, 2, num_matches)`):
            Index of keypoint matched in the other image.
        matching_scores (`torch.FloatTensor` of shape `(batch_size, 2, num_matches)`):
            Scores of predicted matches.
        keypoints (`torch.FloatTensor` of shape `(batch_size, num_keypoints, 2)`):
            Absolute (x, y) coordinates of predicted keypoints in a given image.
        hidden_states (`Tuple[torch.FloatTensor, ...]`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of each stage) of shape `(batch_size, 2, num_channels,
            num_keypoints)`, returned when `output_hidden_states=True` is passed or when
            `config.output_hidden_states=True`)
        attentions (`Tuple[torch.FloatTensor, ...]`, *optional*):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, 2, num_heads, num_keypoints,
            num_keypoints)`, returned when `output_attentions=True` is passed or when `config.output_attentions=True`)
    """

    loss: Optional[torch.FloatTensor] = None
    matches: Optional[torch.FloatTensor] = None
    matching_scores: Optional[torch.FloatTensor] = None
    keypoints: Optional[torch.FloatTensor] = None
    mask: Optional[torch.IntTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class EfficientLoFTRRotaryEmbedding(nn.Module):
    def __init__(self, config: EfficientLoFTRConfig, device="cpu") -> None:
        super().__init__()
        self.config = config
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type")
        else:
            self.rope_type = "2d"
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = x.shape

        i_position_indices = torch.ones(h, w, device=x.device).cumsum(0).float().unsqueeze(-1)
        j_position_indices = torch.ones(h, w, device=x.device).cumsum(1).float().unsqueeze(-1)
        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, None, None, :].float().expand(1, 1, 1, -1)
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            emb = torch.zeros(1, h, w, self.config.hidden_size // 2)
            emb[:, :, :, 0::2] = i_position_indices * inv_freq_expanded
            emb[:, :, :, 1::2] = j_position_indices * inv_freq_expanded

        sin = emb.sin()
        cos = emb.cos()

        sin = sin.repeat_interleave(2, dim=-1)
        cos = cos.repeat_interleave(2, dim=-1)

        sin = sin.to(device=x.device, dtype=x.dtype)
        cos = cos.to(device=x.device, dtype=x.dtype)

        return cos, sin


class EfficientLoFTRConvNormLayer(RTDetrV2ConvNormLayer):
    pass


class EfficientLoFTRRepVGGBlock(nn.Module):
    """
    RepVGG architecture block introduced by the work "RepVGG: Making VGG-style ConvNets Great Again".
    """

    def __init__(self, config: EfficientLoFTRConfig, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        activation = config.activation_function
        self.conv1 = EfficientLoFTRConvNormLayer(
            config, in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.conv2 = EfficientLoFTRConvNormLayer(
            config, in_channels, out_channels, kernel_size=1, stride=stride, padding=0
        )
        self.identity = nn.BatchNorm2d(in_channels) if in_channels == out_channels and stride == 1 else None
        self.activation = nn.Identity() if activation is None else ACT2FN[activation]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.identity is not None:
            identity_out = self.identity(hidden_states)
        else:
            identity_out = 0
        hidden_states = self.conv1(hidden_states) + self.conv2(hidden_states) + identity_out
        hidden_states = self.activation(hidden_states)
        return hidden_states


class EfficientLoFTRRepVGGStage(nn.Module):
    def __init__(self, config: EfficientLoFTRConfig, in_channels, out_channels, num_blocks, stride) -> None:
        super().__init__()

        strides = [stride] + [1] * (num_blocks - 1)
        current_channel_dim = in_channels
        blocks = []
        for stride in strides:
            blocks.append(
                EfficientLoFTRRepVGGBlock(
                    config,
                    current_channel_dim,
                    out_channels,
                    stride,
                )
            )
            current_channel_dim = out_channels
        self.blocks = nn.ModuleList(blocks)

    def forward(
        self, hidden_states: torch.Tensor, output_hidden_states: Optional[bool] = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        all_hidden_states = () if output_hidden_states else None
        for block in self.blocks:
            hidden_states = block(hidden_states)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
        return hidden_states, all_hidden_states


class EfficientLoFTRepVGG(nn.Module):
    def __init__(self, config: EfficientLoFTRConfig) -> None:
        super().__init__()

        self.stages = nn.ModuleList([])
        num_stages = len(config.stage_block_dims)
        current_in_channels = 1

        for i in range(num_stages):
            out_channels = int(config.stage_block_dims[i] * config.stage_hidden_expansion[i])
            stage = EfficientLoFTRRepVGGStage(
                config, current_in_channels, out_channels, config.stage_num_blocks[i], config.stage_stride[i]
            )
            current_in_channels = out_channels
            self.stages.append(stage)

    def forward(
        self, hidden_states: torch.Tensor, output_hidden_states: Optional[bool] = False
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        outputs = []
        all_hidden_states = () if output_hidden_states else None
        for stage in self.stages:
            stage_outputs = stage(hidden_states, output_hidden_states=output_hidden_states)
            hidden_states = stage_outputs[0]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + stage_outputs[1]
            outputs.append(hidden_states)

        # Exclude first stage in outputs
        outputs = outputs[1:]
        # Last stage outputs are coarse outputs
        coarse_features = outputs[-1]
        # Rest is residual features used in EfficientLoFTRFineFusionLayer
        residual_features = outputs[:-1]
        return coarse_features, residual_features, all_hidden_states


class EfficientLoFTRAggregationLayer(nn.Module):
    def __init__(self, config: EfficientLoFTRConfig) -> None:
        super().__init__()

        hidden_size = config.hidden_size
        aggregation_sizes = config.aggregation_sizes
        self.q_aggregation = (
            nn.Conv2d(
                hidden_size,
                hidden_size,
                kernel_size=aggregation_sizes[0],
                padding=0,
                stride=aggregation_sizes[0],
                bias=False,
                groups=hidden_size,
            )
            if aggregation_sizes[0] != 1
            else nn.Identity()
        )

        self.kv_aggregation = (
            torch.nn.MaxPool2d(kernel_size=aggregation_sizes[1], stride=aggregation_sizes[1])
            if aggregation_sizes[1] != 1
            else nn.Identity()
        )

        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        query_states = hidden_states
        is_cross_attention = encoder_hidden_states is not None
        kv_states = encoder_hidden_states if is_cross_attention else hidden_states

        query_states = self.q_aggregation(query_states)
        kv_states = self.kv_aggregation(kv_states)
        query_states = query_states.permute(0, 2, 3, 1)
        kv_states = kv_states.permute(0, 2, 3, 1)
        hidden_states = self.norm(query_states)
        encoder_hidden_states = self.norm(kv_states)
        if attention_mask is not None:
            current_mask = encoder_attention_mask if is_cross_attention else attention_mask
            attention_mask = self.kv_aggregation(attention_mask.float()).bool()
            encoder_attention_mask = self.kv_aggregation(current_mask.float()).bool()
        return hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask


class EfficientLoFTRAttention(LlamaAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_len, dim = hidden_states.shape
        input_shape = hidden_states.shape[:-1]

        query_states = self.q_proj(hidden_states).view(batch_size, seq_len, -1, dim)

        is_cross_attention = encoder_hidden_states is not None
        current_states = encoder_hidden_states if is_cross_attention else hidden_states
        current_attention_mask = encoder_attention_mask if is_cross_attention else attention_mask

        key_states = self.k_proj(current_states).view(batch_size, seq_len, -1, dim)
        value_states = self.v_proj(current_states).view(batch_size, seq_len, -1, self.head_dim).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=2)

        query_states = query_states.view(batch_size, seq_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, -1, self.head_dim).transpose(1, 2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            current_attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class EfficientLoFTRMLP(nn.Module):
    def __init__(self, config: EfficientLoFTRConfig):
        super().__init__()
        hidden_size = config.hidden_size
        self.fc1 = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)
        self.activation = ACT2FN[config.mlp_activation_function]
        self.fc2 = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


def get_positional_embeddings_slice(
    hidden_states: torch.Tensor, positional_embeddings: Tuple[torch.Tensor, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, h, w, _ = hidden_states.shape
    positional_embeddings = tuple(
        tensor[:, :h, :w, :].expand(batch_size, -1, -1, -1) for tensor in positional_embeddings
    )
    return positional_embeddings


class EfficientLoFTRAggregatedAttention(nn.Module):
    def __init__(self, config: EfficientLoFTRConfig, layer_idx: int) -> None:
        super().__init__()

        self.aggregation_sizes = config.aggregation_sizes
        self.aggregation = EfficientLoFTRAggregationLayer(config)
        self.attention = EfficientLoFTRAttention(config, layer_idx)
        self.mlp = EfficientLoFTRMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        batch_size, channels, h, w = hidden_states.shape

        # Aggregate features
        aggregated_hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask = self.aggregation(
            hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask
        )

        attention_hidden_states = aggregated_hidden_states.reshape(batch_size, -1, channels)
        encoder_hidden_states = encoder_hidden_states.reshape(batch_size, -1, channels)

        if position_embeddings is not None:
            position_embeddings = get_positional_embeddings_slice(aggregated_hidden_states, position_embeddings)
            position_embeddings = tuple(tensor.reshape(batch_size, -1, channels) for tensor in position_embeddings)

        # Multi-head attention
        attention_outputs = self.attention(
            attention_hidden_states,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            position_embeddings=position_embeddings,
        )
        message = attention_outputs[0]

        # Upsample features
        _, aggregated_h, aggregated_w, _ = aggregated_hidden_states.shape
        # (batch_size, seq_len, channels) -> (batch_size, channels, h, w) with seq_len = h * w
        message = message.permute(0, 2, 1)
        message = message.reshape(batch_size, channels, aggregated_h, aggregated_w)
        if self.aggregation_sizes[0] != 1:
            message = torch.nn.functional.interpolate(
                message, scale_factor=self.aggregation_sizes[0], mode="bilinear", align_corners=False
            )
        intermediate_states = torch.cat([hidden_states, message], dim=1)
        intermediate_states = intermediate_states.permute(0, 2, 3, 1)
        output_states = self.mlp(intermediate_states)
        output_states = output_states.permute(0, 3, 1, 2)

        hidden_states = hidden_states + output_states

        outputs = (hidden_states,) + attention_outputs[1:]
        return outputs


class EfficientLoFTRLocalFeatureTransformerLayer(nn.Module):
    def __init__(self, config: EfficientLoFTRConfig, layer_idx: int) -> None:
        super().__init__()

        self.self_attention = EfficientLoFTRAggregatedAttention(config, layer_idx)
        self.cross_attention = EfficientLoFTRAggregatedAttention(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        all_attentions = () if output_attentions else None
        batch_size, _, c, h, w = hidden_states.shape

        hidden_states = hidden_states.reshape(-1, c, h, w)
        if attention_mask is not None:
            attention_mask = attention_mask.reshape(-1, c, h, w)

        self_attention_outputs = self.self_attention(
            hidden_states, attention_mask, position_embeddings=position_embeddings
        )
        hidden_states = self_attention_outputs[0]

        encoder_hidden_states = hidden_states.reshape(-1, 2, c, h, w).flip(1).reshape(-1, c, h, w)
        encoder_attention_mask = None
        if attention_mask is not None:
            encoder_attention_mask = attention_mask.reshape(-1, 2, c, h, w).flip(1).reshape(-1, c, h, w)

        cross_attention_outputs = self.cross_attention(
            hidden_states,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
        )

        hidden_states = cross_attention_outputs[0]
        hidden_states = hidden_states.reshape(batch_size, -1, c, h, w)

        if output_attentions:
            all_attentions = all_attentions + (self_attention_outputs[1], cross_attention_outputs[1])

        return hidden_states, all_attentions


class EfficientLoFTRLocalFeatureTransformer(nn.Module):
    def __init__(self, config: EfficientLoFTRConfig) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [
                EfficientLoFTRLocalFeatureTransformerLayer(config, layer_idx=i)
                for i in range(config.num_attention_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        all_attentions = () if output_attentions else None

        for layer in self.layers:
            layer_outputs = layer(
                hidden_states, position_embeddings=position_embeddings, output_attentions=output_attentions
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + layer_outputs[1]
        return hidden_states, all_attentions


class EfficientLoFTROutConvBlock(nn.Module):
    def __init__(self, config: EfficientLoFTRConfig, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()

        self.out_conv1 = nn.Conv2d(hidden_size, intermediate_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.out_conv2 = nn.Conv2d(
            intermediate_size, intermediate_size, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.batch_norm = nn.BatchNorm2d(intermediate_size)
        self.activation = ACT2CLS[config.mlp_activation_function]()
        self.out_conv3 = nn.Conv2d(intermediate_size, hidden_size, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, hidden_states: torch.Tensor, residual_states: List[torch.Tensor]) -> torch.Tensor:
        residual_states = self.out_conv1(residual_states)
        residual_states = residual_states + hidden_states
        residual_states = self.out_conv2(residual_states)
        residual_states = self.batch_norm(residual_states)
        residual_states = self.activation(residual_states)
        residual_states = self.out_conv3(residual_states)
        residual_states = nn.functional.interpolate(
            residual_states, scale_factor=2.0, mode="bilinear", align_corners=False
        )
        return residual_states


class EfficientLoFTRFineFusionLayer(nn.Module):
    def __init__(self, config: EfficientLoFTRConfig) -> None:
        super().__init__()

        self.fine_kernel_size = config.fine_kernel_size

        stage_block_dims = config.stage_block_dims
        stage_block_dims = list(reversed(stage_block_dims))[:-1]
        self.out_conv = nn.Conv2d(
            stage_block_dims[0], stage_block_dims[0], kernel_size=1, stride=1, padding=0, bias=False
        )
        self.out_conv_layers = nn.ModuleList()
        for i in range(1, len(stage_block_dims)):
            out_conv = EfficientLoFTROutConvBlock(config, stage_block_dims[i], stage_block_dims[i - 1])
            self.out_conv_layers.append(out_conv)

    def forward_pyramid(
        self,
        hidden_states: torch.Tensor,
        residual_states: List[torch.Tensor],
        output_hidden_states: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        all_hidden_states = () if output_hidden_states else None
        hidden_states = self.out_conv(hidden_states)
        hidden_states = nn.functional.interpolate(
            hidden_states, scale_factor=2.0, mode="bilinear", align_corners=False
        )

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        for i, layer in enumerate(self.out_conv_layers):
            hidden_states = self.out_conv_layers[i](hidden_states, residual_states[i])
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        return hidden_states, all_hidden_states

    def forward(
        self,
        coarse_features: torch.Tensor,
        residual_features: List[torch.Tensor],
        output_hidden_states: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        """
        For each image pair, compute the fine features of pixels.
        In both images, compute a patch of fine features center cropped around each coarse pixel.
        In the first image, the feature patch is kernel_size large and long.
        In the second image, it is (kernel_size + 2) large and long.
        """
        batch_size, _, channels, coarse_height, coarse_width = coarse_features.shape

        coarse_features = coarse_features.reshape(-1, channels, coarse_height, coarse_width)
        residual_features = list(reversed(residual_features))

        # 1. Fine feature extraction
        pyramid_outputs = self.forward_pyramid(
            coarse_features, residual_features, output_hidden_states=output_hidden_states
        )
        fine_features = pyramid_outputs[0]
        _, fine_channels, fine_height, fine_width = fine_features.shape

        fine_features = fine_features.reshape(batch_size, 2, fine_channels, fine_height, fine_width)
        fine_features_0 = fine_features[:, 0]
        fine_features_1 = fine_features[:, 1]

        # 2. Unfold all local windows in crops
        stride = int(fine_height // coarse_height)
        fine_features_0 = nn.functional.unfold(
            fine_features_0, kernel_size=self.fine_kernel_size, stride=stride, padding=0
        )
        _, _, seq_len = fine_features_0.shape
        fine_features_0 = fine_features_0.reshape(batch_size, -1, self.fine_kernel_size**2, seq_len)
        fine_features_0 = fine_features_0.permute(0, 3, 2, 1)

        fine_features_1 = nn.functional.unfold(
            fine_features_1, kernel_size=self.fine_kernel_size + 2, stride=stride, padding=1
        )
        fine_features_1 = fine_features_1.reshape(batch_size, -1, (self.fine_kernel_size + 2) ** 2, seq_len)
        fine_features_1 = fine_features_1.permute(0, 3, 2, 1)

        return fine_features_0, fine_features_1, pyramid_outputs[1]


class EfficientLoFTRPreTrainedModel(SuperPointPreTrainedModel):
    config_class = EfficientLoFTRConfig
    base_model_prefix = "efficientloftr"
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


EFFICIENTLOFTR_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`EfficientLoFTRConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    """

EFFICIENTLOFTR_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`SuperGlueImageProcessor`]. See
            [`SuperGlueImageProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors. See `attentions` under returned tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "EfficientLoFTR model taking images as inputs and outputting the matching of them.",
    EFFICIENTLOFTR_START_DOCSTRING,
)
class EfficientLoFTRForKeypointMatching(EfficientLoFTRPreTrainedModel):
    """EfficientLoFTR dense image matcher

    Given two images, we determine the correspondences by:
      1. Extracting coarse and fine features through a backbone
      2. Transforming coarse features through self and cross attention
      3. Matching coarse features to obtain coarse coordinates of matches
      4. Obtaining full resolution fine features by fusing transformed and backbone coarse features
      5. Refining the coarse matches using fine feature patches centered at each coarse match in a two-stage refinement

    Yifan Wang, Xingyi He, Sida Peng, Dongli Tan and Xiaowei Zhou.
    Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed
    In CVPR, 2024. https://arxiv.org/abs/2403.04765
    """

    def __init__(self, config: EfficientLoFTRConfig) -> None:
        super().__init__(config)

        self.config = config
        self.backbone = EfficientLoFTRepVGG(config)
        self.local_feature_transformer = EfficientLoFTRLocalFeatureTransformer(config)
        self.refinement_layer = EfficientLoFTRFineFusionLayer(config)

        self.rotary_emb = EfficientLoFTRRotaryEmbedding(config=config)

        self.post_init()

    def get_matches_from_scores(self, scores: torch.Tensor):
        """
        Based on a keypoint score matrix, compute the best keypoint matches between the first and second image.
        Since each image pair can have different number of matches, the matches are concatenated together for all pair
        in the batch and a batch_indices tensor is returned to specify which match belong to which element in the batch.
        Args:
            scores (`torch.Tensor` of shape `(batch_size, height_0, width_0, height_1, width_1)`):
                Scores of keypoints

        Returns:
            matched_indices (`torch.Tensor` of shape `(2, num_matches)`):
                Indices representing which pixel in the first image matches which pixel in the second image
            matching_scores (`torch.Tensor` of shape `(num_matches,)`):
                Scores of each match
            batch_indices (`torch.Tensor` of shape `(num_matches,)`):
                Batch correspondences of matches
        """
        batch_size, height0, width0, height1, width1 = scores.shape

        scores = scores.reshape(batch_size, height0 * width0, height1 * width1)

        # For each keypoint, get the best match
        max_0 = scores.max(2, keepdim=True).values
        max_1 = scores.max(1, keepdim=True).values

        # 1. Thresholding
        mask = scores > self.config.coarse_matching_threshold

        # 2. Border removal
        mask = mask.reshape(batch_size, height0, width0, height1, width1)
        mask = mask_border(mask, self.config.coarse_matching_border_removal, False)
        mask = mask.reshape(batch_size, height0 * width0, height1 * width1)

        # 3. Mutual nearest neighbors
        mask = mask * (scores == max_0) * (scores == max_1)

        # 4. Fine coarse matches
        mask_values, mask_indices = mask.max(dim=2)
        batch_indices, matched_indices_0 = torch.where(mask_values)
        matched_indices_1 = mask_indices[batch_indices, matched_indices_0]
        matching_scores = scores[batch_indices, matched_indices_0, matched_indices_1]

        matched_indices = torch.stack([matched_indices_0, matched_indices_1], dim=0)
        return matched_indices, matching_scores, batch_indices

    def coarse_matching(
        self, coarse_features: torch.Tensor, coarse_scale: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        For each image pair, compute the matching confidence between each coarse element (by default (image_height / 8)
        * (image_width / 8 elements)) from the first image to the second image. Since the number of matches can vary
        with different image pairs, the matches are concatenated together in a dimension. A batch_indices tensor is
        returned to inform which keypoint is part of which image pair.

        Args:
            coarse_features (`torch.Tensor` of shape `(batch_size, 2, hidden_size, coarse_height, coarse_width)`):
                Coarse features
            coarse_scale (`float`): Scale between the image size and the coarse size

        Returns:
            matched_keypoints (`torch.Tensor` of shape `(2, num_matches, 2)`):
                Matched keypoint between the first and the second image. All matched keypoints are concatenated in the
                second dimension.
            matching_scores (`torch.Tensor` of shape `(batch_size, num_matches)`):
                The confidence score of each matched keypoint.
            batch_indices (`torch.Tensor` of shape `(num_matches,)`):
                Indices of batches for each matched keypoint found.
        """
        batch_size, _, channels, height, width = coarse_features.shape

        # (batch_size, 2, channels, height, width) -> (batch_size, 2, height * width, channels)
        coarse_features = coarse_features.permute(0, 1, 3, 4, 2)
        coarse_features = coarse_features.reshape(batch_size, 2, -1, channels)

        coarse_features = coarse_features / coarse_features.shape[-1] ** 0.5
        coarse_features_0 = coarse_features[:, 0]
        coarse_features_1 = coarse_features[:, 1]

        similarity = coarse_features_0 @ coarse_features_1.transpose(-1, -2)
        similarity = similarity / self.config.coarse_matching_temperature

        if self.config.coarse_matching_skip_softmax:
            confidence = similarity
        else:
            confidence = nn.functional.softmax(similarity, 1) * nn.functional.softmax(similarity, 2)

        confidence = confidence.reshape(batch_size, height, width, height, width)
        matched_indices, matching_scores, batch_indices = self.get_matches_from_scores(confidence)

        matched_keypoints = torch.stack([matched_indices % width, matched_indices // width], dim=-1) * coarse_scale

        return (
            matched_keypoints,
            matching_scores,
            batch_indices,
            matched_indices,
        )

    def get_first_stage_fine_matching(
        self,
        fine_confidence: torch.Tensor,
        coarse_matched_keypoints: torch.Tensor,
        fine_window_size: int,
        fine_scale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        For each coarse pixel, retrieve the highest fine confidence score and index.
        The index represents the matching between a pixel position in the fine window in the first image and a pixel
        position in the fine window of the second image.
        For example, for a fine_window_size of 64 (8 * 8), the index 2474 represents the matching between the index 38
        (2474 // 64) in the fine window of the first image, and the index 42 in the second image. This means that 38
        which corresponds to the position (4, 6) (4 // 8 and 4 % 8) is matched with the position (5, 2). In this example
        the coarse matched coordinate will be shifted to the matched fine coordinates in the first and second image.

        Args:
            fine_confidence (`torch.Tensor` of shape `(num_matches, fine_window_size, fine_window_size)`):
                First stage confidence of matching fine features between the first and the second image
            coarse_matched_keypoints (`torch.Tensor` of shape `(2, num_matches, 2)`):
                Coarse matched keypoint between the first and the second image.
            fine_window_size (`int`):
                Size of the window used to refine matches
            fine_scale (`float`):
                Scale between the size of fine features and coarse features

        Returns:
            indices (`torch.Tensor` of shape `(2, num_matches, 1)`):
                Indices of the fine coordinate matched in the fine window
            fine_matches (`torch.Tensor` of shape `(2, num_matches, 2)`):
                Coordinates of matched keypoints after the first fine stage
        """
        num_matches, _, _ = fine_confidence.shape
        fine_kernel_size = int(math.sqrt(fine_window_size))

        fine_confidence = fine_confidence.reshape(num_matches, -1)
        values, indices = torch.max(fine_confidence, dim=-1)
        indices = indices[..., None]
        indices_0 = indices // fine_window_size
        indices_1 = indices % fine_window_size

        grid = create_meshgrid(
            fine_kernel_size,
            fine_kernel_size,
            normalized_coordinates=False,
            device=fine_confidence.device,
            dtype=fine_confidence.dtype,
        )
        grid = grid - (fine_kernel_size // 2) + 0.5
        grid = grid.reshape(1, -1, 2).expand(num_matches, -1, -1)
        delta_0 = torch.gather(grid, 1, indices_0.unsqueeze(-1).expand(-1, -1, 2)).squeeze(1)
        delta_1 = torch.gather(grid, 1, indices_1.unsqueeze(-1).expand(-1, -1, 2)).squeeze(1)

        fine_matches_0 = coarse_matched_keypoints[0] + delta_0 * fine_scale
        fine_matches_0 = fine_matches_0.reshape(num_matches, 2)
        fine_matches_1 = coarse_matched_keypoints[1] + delta_1 * fine_scale
        fine_matches_1 = fine_matches_1.reshape(num_matches, 2)

        indices = torch.stack([indices_0, indices_1], dim=0)
        fine_matches = torch.stack([fine_matches_0, fine_matches_1], dim=0)

        return indices, fine_matches

    def get_second_stage_fine_matching(
        self,
        indices: torch.Tensor,
        fine_matches: torch.Tensor,
        fine_confidence: torch.Tensor,
        fine_window_size: int,
        fine_scale: float,
    ) -> torch.Tensor:
        """
        For the given position in their respective fine windows, retrieve the 3x3 fine confidences around this position.
        After applying softmax to these confidences, compute the 2D spatial expected coordinates.
        Shift the first stage fine matching with these expected coordinates.

        Args:
            indices (`torch.Tensor` of shape `(batch_size, 2, num_keypoints)`):
                Indices representing the position of each keypoint in the fine window
            fine_matches (`torch.Tensor` of shape `(2, num_matches, 2)`):
                Coordinates of matched keypoints after the first fine stage
            fine_confidence (`torch.Tensor` of shape `(num_matches, fine_window_size, fine_window_size)`):
                Second stage confidence of matching fine features between the first and the second image
            fine_window_size (`int`):
                Size of the window used to refine matches
            fine_scale (`float`):
                Scale between the size of fine features and coarse features
        Returns:
            fine_matches (`torch.Tensor` of shape `(2, num_matches, 2)`):
                Coordinates of matched keypoints after the second fine stage
        """
        num_matches, _, _ = fine_confidence.shape
        fine_kernel_size = int(math.sqrt(fine_window_size))

        indices_0 = indices[0]
        indices_1 = indices[1]
        indices_1_i = indices_1 // fine_kernel_size
        indices_1_j = indices_1 % fine_kernel_size

        matches_indices = torch.arange(num_matches, device=indices_0.device)

        # matches_indices, indices_0, indices_1_i, indices_1_j of shape (num_matches, 3, 3)
        matches_indices = matches_indices[..., None, None].expand(-1, 3, 3)
        indices_0 = indices_0[..., None].expand(-1, 3, 3)
        indices_1_i = indices_1_i[..., None].expand(-1, 3, 3)
        indices_1_j = indices_1_j[..., None].expand(-1, 3, 3)

        delta = create_meshgrid(3, 3, normalized_coordinates=True, device=indices_0.device).to(torch.long)
        delta = delta[None, ...]

        indices_1_i = indices_1_i + delta[..., 1]
        indices_1_j = indices_1_j + delta[..., 0]

        fine_confidence = fine_confidence.reshape(
            num_matches, fine_window_size, fine_kernel_size + 2, fine_kernel_size + 2
        )
        # (batch_size, seq_len, fine_window_size, fine_kernel_size + 2, fine_kernel_size + 2) -> (batch_size, seq_len, 3, 3)
        fine_confidence = fine_confidence[matches_indices, indices_0, indices_1_i, indices_1_j]
        fine_confidence = fine_confidence.reshape(num_matches, 9)
        fine_confidence = nn.functional.softmax(
            fine_confidence / self.config.fine_matching_regress_temperature, dim=-1
        )

        heatmap = fine_confidence.reshape(1, -1, 3, 3)
        fine_coordinates_normalized = spatial_expectation2d(heatmap, True)[0]

        fine_matches_0 = fine_matches[0]
        fine_matches_1 = fine_matches[1] + (fine_coordinates_normalized * (3 // 2) * fine_scale)

        fine_matches = torch.stack([fine_matches_0, fine_matches_1], dim=0)

        return fine_matches

    def fine_matching(
        self,
        fine_features_0: torch.Tensor,
        fine_features_1: torch.Tensor,
        coarse_matched_keypoints: torch.Tensor,
        fine_scale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        For each coarse pixel with a corresponding window of fine features, compute the matching confidence between fine
        features in the first image and the second image.

        Fine features are sliced in two part :
        - The first part used for the first stage are the first fine_hidden_size - config.fine_matching_slicedim (64 - 8
         = 56 by default) features.
        - The second part used for the second stage are the last config.fine_matching_slicedim (8 by default) features.

        Each part is used to compute a fine confidence tensor of the following shape :
        (batch_size, (coarse_height * coarse_width), fine_window_size, fine_window_size)
        They correspond to the score between each fine pixel in the first image and each fine pixel in the second image.

        Args:
            fine_features_0 (`torch.Tensor` of shape `(num_matches, fine_kernel_size ** 2, fine_kernel_size ** 2)`):
                Fine features from the first image
            fine_features_1 (`torch.Tensor` of shape `(num_matches, (fine_kernel_size + 2) ** 2, (fine_kernel_size + 2)
            ** 2)`):
                Fine features from the second image
            coarse_matched_keypoints (`torch.Tensor` of shape `(2, num_matches, 2)`):
                Keypoint coordinates found in coarse matching for the first and second image
            fine_scale (`int`):
                Scale between the size of fine features and coarse features

        Returns:
            fine_coordinates (`torch.Tensor` of shape `(2, num_matches, 2)`):
                Matched keypoint between the first and the second image. All matched keypoints are concatenated in the
                second dimension.
            first_stage_fine_confidence (`torch.Tensor` of shape `(num_matches, fine_kernel_size ** 2, fine_kernel_size
            ** 2)`):
                Scores of fine matching in the first stage
            second_stage_fine_confidence (`torch.Tensor` of shape `(num_matches, fine_kernel_size ** 2,
            (fine_kernel_size + 2) ** 2)`):
                Scores of fine matching in the second stage

        """
        num_matches, fine_window_size, _ = fine_features_0.shape

        if num_matches == 0:
            fine_confidence = torch.empty(0, fine_window_size, fine_window_size, device=fine_features_0.device)
            return coarse_matched_keypoints, fine_confidence, fine_confidence

        fine_kernel_size = int(math.sqrt(fine_window_size))

        first_stage_fine_features_0 = fine_features_0[..., : -self.config.fine_matching_slicedim]
        first_stage_fine_features_1 = fine_features_1[..., : -self.config.fine_matching_slicedim]
        first_stage_fine_features_0 = first_stage_fine_features_0 / first_stage_fine_features_0.shape[-1] ** 0.5
        first_stage_fine_features_1 = first_stage_fine_features_1 / first_stage_fine_features_1.shape[-1] ** 0.5
        first_stage_fine_confidence = first_stage_fine_features_0 @ first_stage_fine_features_1.transpose(-1, -2)
        first_stage_fine_confidence = nn.functional.softmax(first_stage_fine_confidence, 1) * nn.functional.softmax(
            first_stage_fine_confidence, 2
        )
        first_stage_fine_confidence = first_stage_fine_confidence.reshape(
            num_matches, fine_window_size, fine_kernel_size + 2, fine_kernel_size + 2
        )
        first_stage_fine_confidence = first_stage_fine_confidence[..., 1:-1, 1:-1]
        first_stage_fine_confidence = first_stage_fine_confidence.reshape(
            num_matches, fine_window_size, fine_window_size
        )

        fine_indices, fine_matches = self.get_first_stage_fine_matching(
            first_stage_fine_confidence,
            coarse_matched_keypoints,
            fine_window_size,
            fine_scale,
        )

        second_stage_fine_features_0 = fine_features_0[..., -self.config.fine_matching_slicedim :]
        second_stage_fine_features_1 = fine_features_1[..., -self.config.fine_matching_slicedim :]
        second_stage_fine_features_1 = second_stage_fine_features_1 / self.config.fine_matching_slicedim**0.5
        second_stage_fine_confidence = second_stage_fine_features_0 @ second_stage_fine_features_1.transpose(-1, -2)

        fine_coordinates = self.get_second_stage_fine_matching(
            fine_indices,
            fine_matches,
            second_stage_fine_confidence,
            fine_window_size,
            fine_scale,
        )

        return fine_coordinates, first_stage_fine_confidence, second_stage_fine_confidence

    @add_start_docstrings_to_model_forward(EFFICIENTLOFTR_INPUTS_DOCSTRING)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, "KeypointMatchingOutput"]:
        """
        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AutoModel
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/assets/phototourism_sample_images/london_bridge_78916675_4568141288.jpg?raw=true"
        >>> image1 = Image.open(requests.get(url, stream=True).raw)
        >>> url = "https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/assets/phototourism_sample_images/london_bridge_19481797_2295892421.jpg?raw=true"
        >>> image2 = Image.open(requests.get(url, stream=True).raw)
        >>> images = [image1, image2]

        >>> processor = AutoImageProcessor.from_pretrained("stevenbucaille/efficient_loftr")
        >>> model = AutoModel.from_pretrained("stevenbucaille/efficient_loftr")

        >>> with torch.no_grad():
        >>>     inputs = processor(images, return_tensors="pt")
        >>>     outputs = model(**inputs)
        ```"""
        loss = None
        if labels is not None:
            raise ValueError("SuperGlue is not trainable, no labels should be provided.")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values.ndim != 5 or pixel_values.size(1) != 2:
            raise ValueError("Input must be a 5D tensor of shape (batch_size, 2, num_channels, height, width)")

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        batch_size, _, channels, height, width = pixel_values.shape
        pixel_values = pixel_values.reshape(batch_size * 2, channels, height, width)
        pixel_values = self.extract_one_channel_pixel_values(pixel_values)

        # 1. Local Feature CNN
        backbone_outputs = self.backbone(pixel_values, output_hidden_states=output_hidden_states)
        coarse_features, residual_features = backbone_outputs[:2]
        coarse_channels, coarse_height, coarse_width = coarse_features.shape[-3:]

        # 2. Coarse-level LoFTR module
        position_embeddings = self.rotary_emb(coarse_features)
        coarse_features = coarse_features.reshape(batch_size, 2, coarse_channels, coarse_height, coarse_width)
        local_feature_transformer_outputs = self.local_feature_transformer(
            coarse_features, position_embeddings=position_embeddings, output_attentions=output_attentions
        )
        coarse_features = local_feature_transformer_outputs[0]

        # 3. Compute coarse-level matching
        coarse_scale = height / coarse_height
        (
            coarse_matched_keypoints,
            coarse_matching_scores,
            batch_indices,
            matched_indices,
        ) = self.coarse_matching(coarse_features, coarse_scale)

        # 4. Fine-level refinement
        refinement_layer_outputs = self.refinement_layer(
            coarse_features, residual_features, output_hidden_states=output_hidden_states
        )
        fine_features_0, fine_features_1 = refinement_layer_outputs[:2]
        fine_features_0 = fine_features_0[batch_indices, matched_indices[0]]
        fine_features_1 = fine_features_1[batch_indices, matched_indices[1]]

        # 5. Computer fine-level matching
        fine_height = int(coarse_height * coarse_scale)
        fine_scale = height / fine_height
        matching_keypoints, first_stage_matching_scores, second_stage_matching_scores = self.fine_matching(
            fine_features_0,
            fine_features_1,
            coarse_matched_keypoints,
            fine_scale,
        )

        matching_keypoints[:, :, 0] = matching_keypoints[:, :, 0] / width
        matching_keypoints[:, :, 1] = matching_keypoints[:, :, 1] / height

        unique_values, counts = torch.unique_consecutive(batch_indices, return_counts=True)

        if len(unique_values) > 0:
            matching_keypoints_0 = matching_keypoints[0]
            matching_keypoints_1 = matching_keypoints[1]
            split_keypoints_0 = torch.split(matching_keypoints_0, counts.tolist())
            split_keypoints_1 = torch.split(matching_keypoints_1, counts.tolist())
            split_scores = torch.split(coarse_matching_scores, counts.tolist())

            split_mask = [torch.ones(size, device=matching_keypoints.device) for size in counts.tolist()]
            split_indices = [torch.arange(size, device=matching_keypoints.device) for size in counts.tolist()]

            keypoints_0 = pad_sequence(split_keypoints_0, batch_first=True)
            keypoints_1 = pad_sequence(split_keypoints_1, batch_first=True)
            matching_scores = pad_sequence(split_scores, batch_first=True)
            mask = pad_sequence(split_mask, batch_first=True)
            matches = pad_sequence(split_indices, batch_first=True)

            keypoints = torch.stack([keypoints_0, keypoints_1], dim=1)
            matching_scores = torch.stack([matching_scores, matching_scores], dim=1)
            mask = torch.stack([mask, mask], dim=1)
            matches = torch.stack([matches, matches], dim=1)

        else:
            keypoints = matching_keypoints.unsqueeze(0)
            matching_scores = torch.stack([coarse_matching_scores, coarse_matching_scores], dim=0).unsqueeze(0)
            mask = torch.ones_like(keypoints)
            matches = torch.stack([matched_indices, matched_indices], dim=0).unsqueeze(0)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + backbone_outputs[2] + refinement_layer_outputs[2]

        if output_attentions:
            all_attentions = all_attentions + local_feature_transformer_outputs[1]

        if not return_dict:
            return tuple(
                v
                for v in [loss, matches, matching_scores, keypoints, mask, all_hidden_states, all_attentions]
                if v is not None
            )

        return KeypointMatchingOutput(
            loss=loss,
            matches=matches,
            matching_scores=matching_scores,
            keypoints=keypoints,
            mask=mask,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


__all__ = ["EfficientLoFTRConfig", "EfficientLoFTRPreTrainedModel", "EfficientLoFTRForKeypointMatching"]
