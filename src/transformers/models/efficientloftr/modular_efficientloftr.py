import math
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

import torch
from torch import nn

from ...activations import ACT2CLS, ACT2FN
from ...configuration_utils import PretrainedConfig
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from ..llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, eager_attention_forward
from ..rt_detr_v2.modeling_rt_detr_v2 import RTDetrV2ConvNormLayer
from ..superpoint.modeling_superpoint import SuperPointPreTrainedModel


if TYPE_CHECKING:
    from ..superglue.modeling_superglue import KeypointMatchingOutput

logger = logging.get_logger(__name__)


def get_matches_from_scores(scores: torch.Tensor, threshold: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """obtain matches from a score matrix [Bx M+1 x N+1]"""
    batch_size, _, _ = scores.shape
    # For each keypoint, get the best match
    max0 = scores.max(2)
    max1 = scores.max(1)
    matches0 = max0.indices
    matches1 = max1.indices

    # Mutual check for matches
    indices0 = torch.arange(matches0.shape[1], device=matches0.device)[None]
    indices1 = torch.arange(matches1.shape[1], device=matches1.device)[None]
    mutual0 = indices0 == matches1.gather(1, matches0)
    mutual1 = indices1 == matches0.gather(1, matches1)

    # Get matching scores and filter based on mutual check and thresholding
    max0 = max0.values.exp()
    zero = max0.new_tensor(0)
    matching_scores0 = torch.where(mutual0, max0, zero)
    matching_scores1 = torch.where(mutual1, matching_scores0.gather(1, matches1), zero)
    valid0 = mutual0 & (matching_scores0 > threshold)
    valid1 = mutual1 & valid0.gather(1, matches1)

    # Filter matches based on mutual check and thresholding of scores
    matches0 = torch.where(valid0, matches0, -1)
    matches1 = torch.where(valid1, matches1, -1)
    matches = torch.stack([matches0, matches1]).transpose(0, 1)
    matching_scores = torch.stack([matching_scores0, matching_scores1]).transpose(0, 1)

    return matches, matching_scores


class EfficientLoFTRConfig(PretrainedConfig):
    model_type = "efficientloftr"

    def __init__(
        self,
        stage_block_dims: List[int] = None,
        stage_num_blocks: List[int] = None,
        stage_hidden_expansion: List[int | float] = None,
        stage_stride: List[int | float] = None,
        activation_function: str = "relu",
        resolution: List[int] = None,
        aggregation_sizes: List[int] = None,
        num_attention_layers: int = 4,
        num_attention_heads: int = 8,
        num_key_value_heads: int = None,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        mlp_activation_function: str = "leaky_relu",
        coarse_matching_skip_softmax: bool = False,
        coarse_matching_threshold: float = 0.2,
        fine_kernel_size: int = 8,
        batch_norm_eps: float = 0.0,
        hidden_size: int = 256,
        rope_type: str = "2d",
        rope_theta=10000.0,
        fine_matching_slicedim=8,
        fine_matching_regress_temperature=10.0,
        **kwargs,
    ):
        self.stage_block_dims = stage_block_dims if stage_block_dims is not None else [64, 64, 128, 256]
        self.stage_num_blocks = stage_num_blocks if stage_num_blocks is not None else [1, 2, 4, 14]
        self.stage_hidden_expansion = stage_hidden_expansion if stage_hidden_expansion is not None else [1, 1, 1, 1]
        self.stage_stride = stage_stride if stage_stride is not None else [2, 1, 2, 2]
        self.activation_function = activation_function
        self.resolution = resolution if resolution is not None else [8, 1]
        self.aggregation_sizes = aggregation_sizes if aggregation_sizes is not None else [4, 4]
        self.num_attention_layers = num_attention_layers
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.mlp_activation_function = mlp_activation_function
        self.coarse_matching_skip_softmax = coarse_matching_skip_softmax
        self.coarse_matching_threshold = coarse_matching_threshold
        self.fine_kernel_size = fine_kernel_size
        self.batch_norm_eps = batch_norm_eps
        self.hidden_size = hidden_size
        self.fine_matching_slicedim = fine_matching_slicedim
        self.fine_matching_regress_temperature = fine_matching_regress_temperature

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads

        self.rope_type = rope_type
        self.rope_theta = rope_theta

        # TODO checks on config size compatibilities
        assert self.hidden_size == self.stage_block_dims[-1]
        super().__init__(**kwargs)


class EfficientLoFTRRotaryEmbedding(nn.Module):
    def __init__(self, config: EfficientLoFTRConfig, device=None):
        super().__init__()
        self.config = config
        self.rope_type = config.rope_type
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, x):
        b, _, h, w = x.shape

        i_position_ids = torch.ones(h, w, device=x.device).cumsum(0).float().unsqueeze(-1)
        j_position_ids = torch.ones(h, w, device=x.device).cumsum(1).float().unsqueeze(-1)
        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, None, None, :].float().expand(b, 1, 1, -1)
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            emb = torch.zeros(b, h, w, self.config.hidden_size // 2, device=x.device)
            emb[:, :, :, 0::2] = i_position_ids * inv_freq_expanded
            emb[:, :, :, 1::2] = j_position_ids * inv_freq_expanded

        sin = emb.sin()
        cos = emb.cos()

        sin = sin.repeat_interleave(2, dim=-1)
        cos = cos.repeat_interleave(2, dim=-1)

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class EfficientLoFTRConvNormLayer(RTDetrV2ConvNormLayer):
    pass


class EfficientLoFTRRepVGGBlock(nn.Module):
    """
    RepVGG architecture block introduced by the work "RepVGG: Making VGG-style ConvNets Great Again".
    """

    def __init__(self, config: EfficientLoFTRConfig, in_channels: int, out_channels: int, stride: int = 1):
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

    def forward(self, hidden_states):
        hidden_states = self.conv1(hidden_states) + self.conv2(hidden_states)
        if self.identity is not None:
            hidden_states = hidden_states + self.identity(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class EfficientLoFTRRepVGGStage(nn.Module):
    def __init__(self, config: EfficientLoFTRConfig, in_channels, out_channels, num_blocks, stride):
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

    def forward(self, hidden_states):
        for block in self.blocks:
            hidden_states = block(hidden_states)

        return hidden_states


class EfficientLoFTRepVGG(nn.Module):
    def __init__(self, config: EfficientLoFTRConfig):
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

    def forward(self, hidden_states):
        outputs = []
        for stage in self.stages:
            hidden_states = stage(hidden_states)
            outputs.append(hidden_states)

        # Exclude first stage in outputs
        outputs = outputs[1:]
        # Last stage outputs are coarse outputs
        coarse_features = outputs[-1]
        # Rest is residual features used in EfficientLoFTRFineFusionLayer
        residual_features = outputs[:-1]
        return coarse_features, residual_features


class EfficientLoFTRAggregationLayer(nn.Module):
    def __init__(self, config: EfficientLoFTRConfig):
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

    def forward(self, hidden_states, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
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
        value_states = self.v_proj(current_states).view(batch_size, seq_len, -1, self.head_dim)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=2)

        query_states = query_states.view(batch_size, seq_len, -1, self.head_dim)
        key_states = key_states.view(batch_size, seq_len, -1, self.head_dim)

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


class EfficientLoFTRAggregatedAttention(nn.Module):
    def __init__(self, config: EfficientLoFTRConfig, layer_idx: int):
        super().__init__()

        self.aggregation_sizes = config.aggregation_sizes
        self.aggregation = EfficientLoFTRAggregationLayer(config)
        self.attention = EfficientLoFTRAttention(config, layer_idx)
        self.mlp = EfficientLoFTRMLP(config)

    def get_positional_embeddings_slice(self, hidden_states, positional_embeddings):
        _, h, w, _ = hidden_states.shape
        positional_embeddings = tuple(tensor[:, :h, :w, :] for tensor in positional_embeddings)
        return positional_embeddings

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, channels, h, w = hidden_states.shape

        aggregated_hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask = self.aggregation(
            hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask
        )

        # (batch_size, channels, h, w) -> (batch_size, h, w, channels)
        position_embeddings = tuple(tensor.permute(0, 2, 3, 1) for tensor in position_embeddings)
        position_embeddings = self.get_positional_embeddings_slice(aggregated_hidden_states, position_embeddings)

        attention_hidden_states = aggregated_hidden_states.reshape(batch_size, -1, channels)
        encoder_hidden_states = encoder_hidden_states.reshape(batch_size, -1, channels)
        position_embeddings = tuple(tensor.reshape(batch_size, -1, channels) for tensor in position_embeddings)

        # Multi head attention
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
        # (batch_size, seq_len, channels) -> (batch_size, channels, seq_len)
        message = message.permute(0, 2, 1)
        # (batch_size, channels, seq_len) -> (batch_size, channels, h, w) with seq_len = h * w
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

        return (hidden_states,)


class EfficientLoFTRLocalFeatureTransformerLayer(nn.Module):
    def __init__(self, config: EfficientLoFTRConfig, layer_idx: int):
        super().__init__()

        self.self_attention = EfficientLoFTRAggregatedAttention(config, layer_idx)
        self.cross_attention = EfficientLoFTRAggregatedAttention(config, layer_idx)

    def get_positional_embeddings_slice(self, hidden_states, positional_embeddings):
        _, h, w, _ = hidden_states.shape
        positional_embeddings = tuple(tensor[:, :h, :w, :] for tensor in positional_embeddings)
        return positional_embeddings

    def forward(self, hidden_states, position_embeddings, attention_mask=None, **kwargs):
        batch_size, _, c, h, w = hidden_states.shape

        hidden_states = hidden_states.reshape(-1, c, h, w)
        if attention_mask is not None:
            attention_mask = attention_mask.reshape(-1, c, h, w)

        # hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask = self.self_aggregation(
        #     hidden_states, attention_mask
        # )

        # (batch_size, 2, channels, h, w) -> (batch_size * 2, channels, h, w)
        position_embeddings = tuple(tensor.reshape(-1, c, h, w) for tensor in position_embeddings)

        self_attention_outputs = self.self_attention(
            hidden_states,
            attention_mask,
            # encoder_hidden_states,
            # encoder_attention_mask,
            position_embeddings=position_embeddings,
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
            position_embeddings=position_embeddings,
        )

        hidden_states = cross_attention_outputs[0]

        # TODO Feature cropped

        hidden_states = hidden_states.reshape(batch_size, -1, c, h, w)

        return hidden_states


class EfficientLoFTRLocalFeatureTransformer(nn.Module):
    def __init__(self, config: EfficientLoFTRConfig):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                EfficientLoFTRLocalFeatureTransformerLayer(config, layer_idx=i)
                for i in range(config.num_attention_layers)
            ]
        )

    def forward(self, hidden_states, position_embeddings):
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings=position_embeddings)
        return hidden_states


class EfficientLoFTROutConvBlock(nn.Module):
    def __init__(self, config: EfficientLoFTRConfig, hidden_size: int, intermediate_size: int):
        super().__init__()

        self.out_conv1 = nn.Conv2d(hidden_size, intermediate_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.out_conv2 = nn.Conv2d(
            intermediate_size, intermediate_size, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.batch_norm = nn.BatchNorm2d(intermediate_size)
        self.activation = ACT2CLS[config.mlp_activation_function]()
        self.out_conv3 = nn.Conv2d(intermediate_size, hidden_size, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, hidden_states, residual_states):
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
    def __init__(self, config: EfficientLoFTRConfig):
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

    def forward_pyramid(self, hidden_states, residual_states):
        hidden_states = self.out_conv(hidden_states)
        hidden_states = nn.functional.interpolate(
            hidden_states, scale_factor=2.0, mode="bilinear", align_corners=False
        )

        for i, layer in enumerate(self.out_conv_layers):
            hidden_states = self.out_conv_layers[i](hidden_states, residual_states[i])

        return hidden_states

    def forward(self, coarse_features, residual_features):
        """
        For each image pair, compute the fine features of pixels.
        For each coarse pixel, in the first image, unfold the fine features around it by a kernel of size
        self.fine_kernel_size (8 by default). In the second image, unfold the fine features around it by a kernel of
        size self.fine_kernel_size + 2 (10 by default).
        Thus, for the first image, for each coarse pixel, we have (self.fine_kernel_size * self.fine_kernel_size) (64
        by default). In the second image, for each coarse pixel, we have ((self.fine_kernel_size + 2) *
        (self.fine_kernel_size + 2)) (100 by default).
        The fine feature dim being config.stage_block_dims[0].

        Args:
            coarse_features:
            residual_features:

        Returns:

        """
        batch_size, _, channels, h, w = coarse_features.shape

        coarse_features = coarse_features.reshape(-1, channels, h, w)
        residual_features = list(reversed(residual_features))
        # 1. fine feature extraction
        fine_features = self.forward_pyramid(coarse_features, residual_features)
        _, fine_channels, fine_height, fine_width = fine_features.shape

        fine_features = fine_features.reshape(batch_size, 2, fine_channels, fine_height, fine_width)
        fine_features0 = fine_features[:, 0]
        fine_features1 = fine_features[:, 1]

        # 2. unfold(crop) all local windows
        stride = fine_height // h
        fine_features0 = nn.functional.unfold(
            fine_features0, kernel_size=self.fine_kernel_size, stride=stride, padding=0
        )
        _, _, l = fine_features0.shape
        fine_features0 = fine_features0.reshape(batch_size, -1, self.fine_kernel_size**2, l)
        fine_features0 = fine_features0.permute(0, 3, 2, 1)

        fine_features1 = nn.functional.unfold(
            fine_features1, kernel_size=self.fine_kernel_size + 2, stride=stride, padding=1
        )
        fine_features1 = fine_features1.reshape(batch_size, -1, (self.fine_kernel_size + 2) ** 2, l)
        fine_features1 = fine_features1.permute(0, 3, 2, 1)

        return fine_features0, fine_features1


class EfficientLoFTRPreTrainedModel(SuperPointPreTrainedModel):
    config_class = EfficientLoFTRConfig
    base_model_prefix = "efficientloftr"


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


def create_grid(height: int, width: int, normalized_coordinates: bool = False, device: str = None, dtype=None):
    xs = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
    ys = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
    if normalized_coordinates:
        xs = (xs / (width - 1) - 0.5) * 2
        ys = (ys / (height - 1) - 0.5) * 2
    grid = torch.stack(torch.meshgrid(ys, xs, indexing="ij"), dim=-1)
    grid = grid.permute(1, 0, 2).unsqueeze(0)
    return grid


def spatial_expectation2d(input: torch.Tensor, normalized_coordinates: bool = True) -> torch.Tensor:
    r"""Compute the expectation of coordinate values using spatial probabilities.

    The input heatmap is assumed to represent a valid spatial probability distribution,
    which can be achieved using :func:`~kornia.geometry.subpixel.spatial_softmax2d`.

    Args:
        input: the input tensor representing dense spatial probabilities with shape :math:`(B, N, H, W)`.
        normalized_coordinates: whether to return the coordinates normalized in the range
          of :math:`[-1, 1]`. Otherwise, it will return the coordinates in the range of the input shape.

    Returns:
       expected value of the 2D coordinates with shape :math:`(B, N, 2)`. Output order of the coordinates is (x, y).

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
    grid = create_grid(height, width, normalized_coordinates, input.device)
    grid = grid.to(input.dtype)

    pos_x = grid[..., 0].reshape(-1)
    pos_y = grid[..., 1].reshape(-1)

    input_flat = input.view(batch_size, channels, -1)

    # Compute the expectation of the coordinates.
    expected_y = torch.sum(pos_y * input_flat, -1, keepdim=True)
    expected_x = torch.sum(pos_x * input_flat, -1, keepdim=True)

    output = torch.cat([expected_x, expected_y], -1)

    return output.view(batch_size, channels, 2)  # BxNx2


@add_start_docstrings(
    "SuperGlue model taking images as inputs and outputting the matching of them.",
    EFFICIENTLOFTR_START_DOCSTRING,
)
class EfficientLoFTRForKeypointMatching(EfficientLoFTRPreTrainedModel):
    """
    TODO
    """

    def __init__(self, config: EfficientLoFTRConfig) -> None:
        super().__init__(config)

        self.config = config
        self.backbone = EfficientLoFTRepVGG(config)
        self.local_feature_transformer = EfficientLoFTRLocalFeatureTransformer(config)
        self.refinement_layer = EfficientLoFTRFineFusionLayer(config)

        self.rotary_emb = EfficientLoFTRRotaryEmbedding(config=config)

        # self.post_init()

    def coarse_matching(self, coarse_features, mask=None):
        """
        For each image pair, compute the matching confidence between each coarse element (by default (image_height / 8)
        * (image_width / 8 elements)) from the first image to the second image.

        Args:
            coarse_features: tensor of shape (batch_size, 2, channels, coarse_height, coarse_width)
            mask:

        Returns:
            matches: tensor of shape (batch_size, (coarse_height * coarse_width), 2) representing x and y values
            matching_scores: tensor of shape ( TODO
        """
        batch_size, _, channels, h, w = coarse_features.shape
        # (batch_size, 2, channels, h, w) -> (batch_size, 2, h, w, channels)
        coarse_features = coarse_features.permute(0, 1, 3, 4, 2)
        # (batch_size, 2, h, w, channels) -> (batch_size, 2, h * w, channels)
        coarse_features = coarse_features.reshape(batch_size, 2, -1, channels)

        coarse_features = coarse_features / coarse_features.shape[-1] ** 0.5
        coarse_features0 = coarse_features[:, 0]
        coarse_features1 = coarse_features[:, 1]

        similarity = coarse_features0 @ coarse_features1.transpose(-1, -2)
        # TODO mask

        if self.config.coarse_matching_skip_softmax:
            confidence = similarity
        else:
            confidence = nn.functional.softmax(similarity, 1) * nn.functional.softmax(similarity, 2)

        # TODO mask
        matches, matching_scores = get_matches_from_scores(confidence, self.config.coarse_matching_threshold)

        return matches, matching_scores

    def get_first_stage_fine_matching(self, fine_confidence, coarse_grid, fine_kernel_size, fine_window_size):
        """
        For each coarse pixel, retrieve the highest fine confidence score and index.
        The index represents the matching between a pixel position in the fine window in the first image and a pixel
        position in the fine window of the second image.
        For example, for a fine_window_size of 64 (8 * 8), the index 2474 represents the matching between the index 38
        (2474 // 64) in the fine window of the first image, and the index 42 in the second image. This means that 38
        which corresponds to the position (4, 6) (4 // 8 and 4 % 8) is matched with the position (5, 2).

        Args:
            fine_confidence:
            coarse_grid:
            fine_kernel_size:
            fine_window_size:

        Returns:

        """
        batch_size, seq_len, _, _ = fine_confidence.shape

        fine_confidence = fine_confidence.reshape(batch_size, seq_len, -1)
        values, indices = torch.max(fine_confidence, dim=-1)
        indices = indices[..., None]
        indices_0 = indices // fine_window_size
        indices_1 = indices % fine_window_size

        grid = create_grid(
            fine_kernel_size,
            fine_kernel_size,
            normalized_coordinates=False,
            device=fine_confidence.device,
            dtype=fine_confidence.dtype,
        )
        grid = grid.reshape(batch_size, 1, -1, 2).expand(-1, seq_len, -1, -1)
        delta_0 = torch.gather(grid, 2, indices_0.unsqueeze(-1).expand(-1, -1, -1, 2))
        delta_1 = torch.gather(grid, 2, indices_1.unsqueeze(-1).expand(-1, -1, -1, 2))

        coarse_grid = coarse_grid.reshape(batch_size, -1, 2)
        fine_matches_0 = coarse_grid[:, :, None, :] + delta_0
        fine_matches_0 = fine_matches_0.reshape(batch_size, -1, 2)
        fine_matches_1 = coarse_grid[:, :, None, :] + delta_1
        fine_matches_1 = fine_matches_1.reshape(batch_size, -1, 2)

        indices = torch.stack([indices_0, indices_1], dim=0)
        fine_matches = torch.stack([fine_matches_0, fine_matches_1], dim=0)

        return indices, fine_matches

    def get_second_stage_fine_matching(
        self, indices, confidence, fine_kernel_size, fine_window_size, coordinates, fine_scale
    ):
        """
        For the given position in their respective fine windows, retrieve the 3x3 fine confidences around this position.
        After applying softmax to these confidences, compute the 2D spatial expected coordinates.

        Args:
            indices_0:
            indices_1:
            confidence:
            fine_kernel_size:
            fine_window_size:
            coarse_coordinates:
            fine_scale:

        Returns:

        """
        batch_size, seq_len, _, _ = confidence.shape
        indices_0 = indices[:, 0]
        indices_1 = indices[:, 1]
        indices_1_i = indices_1 // fine_kernel_size
        indices_1_j = indices_1 % fine_kernel_size

        # batch_ids, seq_ids and indices_0 of shape (batch_size, seq_len, 3, 3)
        batch_ids = torch.arange(batch_size, device=indices_0.device)[..., None, None, None].expand(-1, seq_len, 3, 3)
        seq_ids = torch.arange(seq_len, device=indices_0.device)[None, ..., None, None].expand(batch_size, -1, 3, 3)
        indices_0 = indices_0[..., None].expand(-1, -1, 3, 3)

        delta = create_grid(3, 3, normalized_coordinates=True, device=indices_0.device).to(torch.long)

        # indices_1_i and indices_1_j of shape (batch_size, seq_len, 3, 3)
        indices_1_i = indices_1_i[..., None].expand(-1, -1, 3, 3) + delta[None, ..., 1]
        indices_1_j = indices_1_j[..., None].expand(-1, -1, 3, 3) + delta[None, ..., 0]

        confidence = confidence.reshape(
            batch_size, seq_len, fine_window_size, fine_kernel_size + 2, fine_kernel_size + 2
        )
        # (batch_size, seq_len, fine_window_size, fine_kernel_size + 2, fine_kernel_size + 2) -> (batch_size, seq_len, 3, 3)
        confidence = confidence[batch_ids, seq_ids, indices_0, indices_1_i, indices_1_j]
        confidence = confidence.reshape(batch_size, seq_len, 9)
        confidence = nn.functional.softmax(confidence / self.config.fine_matching_regress_temperature, dim=-1)

        heatmap = confidence.reshape(batch_size, seq_len, 3, 3)
        fine_coordinates_normalized = spatial_expectation2d(heatmap, True)

        coordinates = coordinates.reshape(batch_size, seq_len, 2)
        fine_coordinates_0 = coordinates[:, 0]
        fine_coordinates_1 = coordinates[:, 1] + (fine_coordinates_normalized * (3 // 2) * fine_scale)

        fine_coordinates = torch.stack([fine_coordinates_0, fine_coordinates_1], dim=0)

        return fine_coordinates, confidence

    def fine_matching(
        self, fine_features0, fine_features1, coarse_matches, coarse_matching_scores, coarse_coordinates, fine_scale
    ):
        """
        For each coarse pixel with a corresponding window of fine features, compute the matching confidence between fine
        features in the first image to the second image.

        Fine features are sliced in two part. The first part used for the first stage are the first fine_hidden_size -
        config.fine_matching_slicedim (64 - 8 = 56 by default) features. The second part used for the second stage are
        the last config.fine_matching_slicedim (8 by default) features.

        Each part is used to compute a fine confidence tensor of the following shape : (batch_size, (coarse_height *
        coarse_width), fine_window_size, fine_window_size). They correspond to the score between each fine pixel in the
        first image and each fine pixel in the second image.

        **1st Stage** :
            In the first stage, we take the first fine confidence tensor and compute

        Args:
            fine_features0:
            fine_features1:
            coarse_matches:
            coarse_matching_scores:
            coarse_coordinates:
            fine_scale:

        Returns:

        """
        if torch.all(coarse_matches == -1):
            return coarse_matches, coarse_matching_scores

        batch_size, seq_len, fine_window_size, fine_hidden_size = fine_features0.shape
        fine_kernel_size = int(math.sqrt(fine_window_size))

        fine_features0 = fine_features0[..., : -self.config.fine_matching_slicedim]
        fine_features1 = fine_features1[..., : -self.config.fine_matching_slicedim]
        fine_features0 = fine_features0 / fine_features0.shape[-1] ** 0.5
        fine_features1 = fine_features1 / fine_features1.shape[-1] ** 0.5
        first_stage_fine_confidence = fine_features0 @ fine_features1.transpose(-1, -2)
        first_stage_fine_confidence = nn.functional.softmax(first_stage_fine_confidence, 2) * nn.functional.softmax(
            first_stage_fine_confidence, 3
        )
        first_stage_fine_confidence = first_stage_fine_confidence.reshape(
            batch_size, seq_len, fine_window_size, fine_kernel_size + 2, fine_kernel_size + 2
        )
        first_stage_fine_confidence = first_stage_fine_confidence[..., 1:-1, 1:-1]
        first_stage_fine_confidence = first_stage_fine_confidence.reshape(
            batch_size, seq_len, fine_window_size, fine_window_size
        )

        fine_indices, fine_coordinates = self.get_first_stage_fine_matching(
            first_stage_fine_confidence, coarse_coordinates, fine_kernel_size, fine_window_size
        )

        f_fine_features0 = fine_features0[..., -self.config.fine_matching_slicedim :]
        f_fine_features1 = fine_features1[..., -self.config.fine_matching_slicedim :]
        f_fine_features1 = f_fine_features1 / self.config.fine_matching_slicedim**0.5
        second_stage_fine_confidence = f_fine_features0 @ f_fine_features1.transpose(-1, -2)

        fine_coordinates, second_stage_fine_confidence = self.get_second_stage_fine_matching(
            fine_indices,
            second_stage_fine_confidence,
            fine_kernel_size,
            fine_window_size,
            fine_coordinates,
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
        loss = None

        batch_size, _, channels, height, width = pixel_values.shape
        pixel_values = pixel_values.reshape(batch_size * 2, channels, height, width)
        pixel_values = self.extract_one_channel_pixel_values(pixel_values)

        # 1. Local Feature CNN
        coarse_features, residual_features = self.backbone(pixel_values)
        mul = self.config.resolution[0] // self.config.resolution[1]  # TODO
        coarse_channels, coarse_height, coarse_width = coarse_features.shape[-3:]
        fine_height = coarse_height * mul
        fine_width = coarse_width * mul
        fine_scale = height / fine_height

        # 2. Coarse-level LoFTR module
        coarse_mask0 = coarse_mask1 = None  # mask is useful in training
        position_embeddings = self.rotary_emb(coarse_features)
        coarse_features = coarse_features.reshape(batch_size, 2, coarse_channels, coarse_height, coarse_width)
        position_embeddings = tuple(
            tensor.reshape(batch_size, 2, coarse_channels, coarse_height, coarse_width)
            for tensor in position_embeddings
        )

        coarse_features = self.local_feature_transformer(coarse_features, position_embeddings=position_embeddings)
        coarse_matches, coarse_matching_scores = self.coarse_matching(coarse_features)

        # 4. fine-level refinement
        fine_features0, fine_features1 = self.refinement_layer(coarse_features, residual_features)

        # 5. match fine-level
        coarse_coordinates = create_grid(
            coarse_height, coarse_width, device=coarse_features.device, dtype=coarse_features.dtype
        )
        coarse_coordinates = coarse_coordinates.reshape(batch_size, 1, coarse_height, coarse_width, 2)
        matching_keypoints, first_stage_matching_scores, second_stage_matching_scores = self.fine_matching(
            fine_features0, fine_features1, coarse_coordinates, coarse_matching_scores, coarse_coordinates, fine_scale
        )

        matching_keypoints = matching_keypoints * mul

        return matching_keypoints, first_stage_matching_scores
