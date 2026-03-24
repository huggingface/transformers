# Copyright 2025 Baidu Inc and The HuggingFace Inc. team.
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
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import Tensor

from ... import initialization as init
from ...backbone_utils import consolidate_backbone_kwargs_to_config
from ...configuration_utils import PreTrainedConfig
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging, torch_compilable_check
from ..auto import AutoConfig
from ..rt_detr.modeling_rt_detr import (
    RTDetrDecoder,
    RTDetrDecoderLayer,
    RTDetrForObjectDetection,
    RTDetrMLPPredictionHead,
    RTDetrModel,
    RTDetrPreTrainedModel,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="PekingU/rtdetr_r18vd")
@strict
class RTDetrV2Config(PreTrainedConfig):
    r"""
    initializer_bias_prior_prob (`float`, *optional*):
        The prior probability used by the bias initializer to initialize biases for `enc_score_head` and `class_embed`.
        If `None`, `prior_prob` computed as `prior_prob = 1 / (num_labels + 1)` while initializing model weights.
    freeze_backbone_batch_norms (`bool`, *optional*, defaults to `True`):
        Whether to freeze the batch normalization layers in the backbone.
    encoder_in_channels (`list`, *optional*, defaults to `[512, 1024, 2048]`):
        Multi level features input for encoder.
    feat_strides (`list[int]`, *optional*, defaults to `[8, 16, 32]`):
        Strides used in each feature map.
    encode_proj_layers (`list[int]`, *optional*, defaults to `[2]`):
        Indexes of the projected layers to be used in the encoder.
    positional_encoding_temperature (`int`, *optional*, defaults to 10000):
        The temperature parameter used to create the positional encodings.
    encoder_activation_function (`str`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    activation_function (`str`, *optional*, defaults to `"silu"`):
        The non-linear activation function (function or string) in the general layer. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    eval_size (`tuple[int, int]`, *optional*):
        Height and width used to compute the effective height and width of the position embeddings after taking
        into account the stride.
    normalize_before (`bool`, *optional*, defaults to `False`):
        Determine whether to apply layer normalization in the transformer encoder layer before self-attention and
        feed-forward modules.
    hidden_expansion (`float`, *optional*, defaults to 1.0):
        Expansion ratio to enlarge the dimension size of RepVGGBlock and CSPRepLayer.
    num_queries (`int`, *optional*, defaults to 300):
        Number of object queries.
    decoder_in_channels (`list`, *optional*, defaults to `[256, 256, 256]`):
        Multi level features dimension for decoder
    num_feature_levels (`int`, *optional*, defaults to 3):
        The number of input feature levels.
    decoder_n_points (`int`, *optional*, defaults to 4):
        The number of sampled keys in each feature level for each attention head in the decoder.
    decoder_activation_function (`str`, *optional*, defaults to `"relu"`):
        The non-linear activation function (function or string) in the decoder. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    num_denoising (`int`, *optional*, defaults to 100):
        The total number of denoising tasks or queries to be used for contrastive denoising.
    label_noise_ratio (`float`, *optional*, defaults to 0.5):
        The fraction of denoising labels to which random noise should be added.
    box_noise_scale (`float`, *optional*, defaults to 1.0):
        Scale or magnitude of noise to be added to the bounding boxes.
    learn_initial_query (`bool`, *optional*, defaults to `False`):
        Indicates whether the initial query embeddings for the decoder should be learned during training
    anchor_image_size (`tuple[int, int]`, *optional*):
        Height and width of the input image used during evaluation to generate the bounding box anchors. If None, automatic generate anchor is applied.
    with_box_refine (`bool`, *optional*, defaults to `True`):
        Whether to apply iterative bounding box refinement, where each decoder layer refines the bounding boxes
        based on the predictions from the previous layer.
    matcher_alpha (`float`, *optional*, defaults to 0.25):
        Parameter alpha used by the Hungarian Matcher.
    matcher_gamma (`float`, *optional*, defaults to 2.0):
        Parameter gamma used by the Hungarian Matcher.
    matcher_class_cost (`float`, *optional*, defaults to 2.0):
        The relative weight of the class loss used by the Hungarian Matcher.
    matcher_bbox_cost (`float`, *optional*, defaults to 5.0):
        The relative weight of the bounding box loss used by the Hungarian Matcher.
    matcher_giou_cost (`float`, *optional*, defaults to 2.0):
        The relative weight of the giou loss of used by the Hungarian Matcher.
    use_focal_loss (`bool`, *optional*, defaults to `True`):
        Parameter informing if focal loss should be used.
    focal_loss_alpha (`float`, *optional*, defaults to 0.75):
        Parameter alpha used to compute the focal loss.
    focal_loss_gamma (`float`, *optional*, defaults to 2.0):
        Parameter gamma used to compute the focal loss.
    weight_loss_vfl (`float`, *optional*, defaults to 1.0):
        Relative weight of the varifocal loss in the object detection loss.
    weight_loss_bbox (`float`, *optional*, defaults to 5.0):
        Relative weight of the L1 bounding box loss in the object detection loss.
    weight_loss_giou (`float`, *optional*, defaults to 2.0):
        Relative weight of the generalized IoU loss in the object detection loss.
    decoder_n_levels (`int`, *optional*, defaults to 3):
        The number of feature levels used by the decoder.
    decoder_offset_scale (`float`, *optional*, defaults to 0.5):
        Scaling factor applied to the attention offsets in the decoder.
    decoder_method (`str`, *optional*, defaults to `"default"`):
        The method to use for the decoder: `"default"` or `"discrete"`.

    Examples:

    ```python
    >>> from transformers import RTDetrV2Config, RTDetrV2Model

    >>> # Initializing a RT-DETR configuration
    >>> configuration = RTDetrV2Config()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = RTDetrV2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "rt_detr_v2"
    sub_configs = {"backbone_config": AutoConfig}
    layer_types = ["basic", "bottleneck"]
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
    }

    initializer_range: float = 0.01
    initializer_bias_prior_prob: float | None = None
    layer_norm_eps: float = 1e-5
    batch_norm_eps: float = 1e-5
    backbone_config: dict | PreTrainedConfig | None = None
    freeze_backbone_batch_norms: bool = True
    encoder_hidden_dim: int = 256
    encoder_in_channels: list[int] | tuple[int, ...] = (512, 1024, 2048)
    feat_strides: list[int] | tuple[int, ...] = (8, 16, 32)
    encoder_layers: int = 1
    encoder_ffn_dim: int = 1024
    encoder_attention_heads: int = 8
    dropout: float | int = 0.0
    activation_dropout: float | int = 0.0
    encode_proj_layers: list[int] | tuple[int, ...] = (2,)
    positional_encoding_temperature: int = 10000
    encoder_activation_function: str = "gelu"
    activation_function: str = "silu"
    eval_size: int | None = None
    normalize_before: bool = False
    hidden_expansion: float = 1.0
    d_model: int = 256
    num_queries: int = 300
    decoder_in_channels: list[int] | tuple[int, ...] = (256, 256, 256)
    decoder_ffn_dim: int = 1024
    num_feature_levels: int = 3
    decoder_n_points: int = 4
    decoder_layers: int = 6
    decoder_attention_heads: int = 8
    decoder_activation_function: str = "relu"
    attention_dropout: float | int = 0.0
    num_denoising: int = 100
    label_noise_ratio: float = 0.5
    box_noise_scale: float = 1.0
    learn_initial_query: bool = False
    anchor_image_size: int | list[int] | None = None
    with_box_refine: bool = True
    is_encoder_decoder: bool = True
    matcher_alpha: float = 0.25
    matcher_gamma: float = 2.0
    matcher_class_cost: float = 2.0
    matcher_bbox_cost: float = 5.0
    matcher_giou_cost: float = 2.0
    use_focal_loss: bool = True
    auxiliary_loss: bool = True
    focal_loss_alpha: float = 0.75
    focal_loss_gamma: float = 2.0
    weight_loss_vfl: float = 1.0
    weight_loss_bbox: float = 5.0
    weight_loss_giou: float = 2.0
    eos_coefficient: float = 1e-4
    decoder_n_levels: int = 3
    decoder_offset_scale: float = 0.5
    decoder_method: str = "default"
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_config_type="rt_detr_resnet",
            default_config_kwargs={"out_indices": [2, 3, 4]},
            **kwargs,
        )
        super().__post_init__(**kwargs)


def multi_scale_deformable_attention_v2(
    value: Tensor,
    value_spatial_shapes: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
    num_points_list: list[int],
    method="default",
) -> Tensor:
    batch_size, _, num_heads, hidden_dim = value.shape
    _, num_queries, num_heads, num_levels, num_points = sampling_locations.shape
    value_list = (
        value.permute(0, 2, 3, 1)
        .flatten(0, 1)
        .split([height * width for height, width in value_spatial_shapes], dim=-1)
    )
    # sampling_offsets [8, 480, 8, 12, 2]
    if method == "default":
        sampling_grids = 2 * sampling_locations - 1
    elif method == "discrete":
        sampling_grids = sampling_locations
    sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1)
    sampling_grids = sampling_grids.split(num_points_list, dim=-2)
    sampling_value_list = []
    for level_id, (height, width) in enumerate(value_spatial_shapes):
        # batch_size, height*width, num_heads, hidden_dim
        # -> batch_size, height*width, num_heads*hidden_dim
        # -> batch_size, num_heads*hidden_dim, height*width
        # -> batch_size*num_heads, hidden_dim, height, width
        value_l_ = value_list[level_id].reshape(batch_size * num_heads, hidden_dim, height, width)
        # batch_size, num_queries, num_heads, num_points, 2
        # -> batch_size, num_heads, num_queries, num_points, 2
        # -> batch_size*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[level_id]
        # batch_size*num_heads, hidden_dim, num_queries, num_points
        if method == "default":
            sampling_value_l_ = nn.functional.grid_sample(
                value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
            )
        elif method == "discrete":
            sampling_coord = (sampling_grid_l_ * torch.tensor([[width, height]], device=value.device) + 0.5).to(
                torch.int64
            )

            # Separate clamping for x and y coordinates
            sampling_coord_x = sampling_coord[..., 0].clamp(0, width - 1)
            sampling_coord_y = sampling_coord[..., 1].clamp(0, height - 1)

            # Combine the clamped coordinates
            sampling_coord = torch.stack([sampling_coord_x, sampling_coord_y], dim=-1)
            sampling_coord = sampling_coord.reshape(batch_size * num_heads, num_queries * num_points_list[level_id], 2)
            sampling_idx = (
                torch.arange(sampling_coord.shape[0], device=value.device)
                .unsqueeze(-1)
                .repeat(1, sampling_coord.shape[1])
            )
            sampling_value_l_ = value_l_[sampling_idx, :, sampling_coord[..., 1], sampling_coord[..., 0]]
            sampling_value_l_ = sampling_value_l_.permute(0, 2, 1).reshape(
                batch_size * num_heads, hidden_dim, num_queries, num_points_list[level_id]
            )
        sampling_value_list.append(sampling_value_l_)
    # (batch_size, num_queries, num_heads, num_levels, num_points)
    # -> (batch_size, num_heads, num_queries, num_levels, num_points)
    # -> (batch_size, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.permute(0, 2, 1, 3).reshape(
        batch_size * num_heads, 1, num_queries, sum(num_points_list)
    )
    output = (
        (torch.concat(sampling_value_list, dim=-1) * attention_weights)
        .sum(-1)
        .view(batch_size, num_heads * hidden_dim, num_queries)
    )
    return output.transpose(1, 2).contiguous()


# the main change
class RTDetrV2MultiscaleDeformableAttention(nn.Module):
    """
    RTDetrV2 version of multiscale deformable attention, extending the base implementation
    with improved offset handling and initialization.
    """

    def __init__(self, config: RTDetrV2Config):
        super().__init__()
        num_heads = config.decoder_attention_heads
        n_points = config.decoder_n_points

        if config.d_model % num_heads != 0:
            raise ValueError(
                f"embed_dim (d_model) must be divisible by num_heads, but got {config.d_model} and {num_heads}"
            )
        dim_per_head = config.d_model // num_heads
        # check if dim_per_head is power of 2
        if not ((dim_per_head & (dim_per_head - 1) == 0) and dim_per_head != 0):
            warnings.warn(
                "You'd better set embed_dim (d_model) in RTDetrV2MultiscaleDeformableAttention to make the"
                " dimension of each attention head a power of 2 which is more efficient in the authors' CUDA"
                " implementation."
            )

        self.im2col_step = 64

        self.d_model = config.d_model

        # V2-specific attributes
        self.n_levels = config.decoder_n_levels
        self.n_heads = num_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(config.d_model, num_heads * self.n_levels * n_points * 2)
        self.attention_weights = nn.Linear(config.d_model, num_heads * self.n_levels * n_points)
        self.value_proj = nn.Linear(config.d_model, config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.d_model)

        self.offset_scale = config.decoder_offset_scale
        self.method = config.decoder_method

        # Initialize n_points list and scale
        n_points_list = [self.n_points for _ in range(self.n_levels)]
        self.n_points_list = n_points_list
        n_points_scale = [1 / n for n in n_points_list for _ in range(n)]
        self.register_buffer("n_points_scale", torch.tensor(n_points_scale, dtype=torch.float32))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_embeddings: torch.Tensor | None = None,
        reference_points=None,
        spatial_shapes=None,
        spatial_shapes_list=None,
        level_start_index=None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        # Process inputs up to sampling locations calculation using parent class logic
        if position_embeddings is not None:
            hidden_states = hidden_states + position_embeddings

        batch_size, num_queries, _ = hidden_states.shape
        batch_size, sequence_length, _ = encoder_hidden_states.shape
        torch_compilable_check(
            (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == sequence_length,
            "Make sure to align the spatial shapes with the sequence length of the encoder hidden states",
        )

        value = self.value_proj(encoder_hidden_states)
        if attention_mask is not None:
            value = value.masked_fill(~attention_mask[..., None], float(0))
        value = value.view(batch_size, sequence_length, self.n_heads, self.d_model // self.n_heads)

        # V2-specific sampling offsets shape
        sampling_offsets = self.sampling_offsets(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels * self.n_points, 2
        )

        attention_weights = self.attention_weights(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1)

        # V2-specific sampling locations calculation
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            n_points_scale = self.n_points_scale.to(dtype=hidden_states.dtype).unsqueeze(-1)
            offset = sampling_offsets * n_points_scale * reference_points[:, :, None, :, 2:] * self.offset_scale
            sampling_locations = reference_points[:, :, None, :, :2] + offset
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}")

        # V2-specific attention implementation choice
        output = multi_scale_deformable_attention_v2(
            value, spatial_shapes_list, sampling_locations, attention_weights, self.n_points_list, self.method
        )

        output = self.output_proj(output)
        return output, attention_weights


class RTDetrV2DecoderLayer(RTDetrDecoderLayer):
    def __init__(self, config: RTDetrV2Config):
        # initialize parent class
        super().__init__(config)
        # override only the encoder attention module with v2 version
        self.encoder_attn = RTDetrV2MultiscaleDeformableAttention(config)


class RTDetrV2PreTrainedModel(RTDetrPreTrainedModel):
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, RTDetrV2MultiscaleDeformableAttention):
            n_points_scale = [1 / n for n in module.n_points_list for _ in range(n)]
            init.copy_(module.n_points_scale, torch.tensor(n_points_scale, dtype=torch.float32))


class RTDetrV2Decoder(RTDetrDecoder):
    def __init__(self, config: RTDetrV2Config):
        super().__init__(config)
        self.layers = nn.ModuleList([RTDetrV2DecoderLayer(config) for _ in range(config.decoder_layers)])


class RTDetrV2Model(RTDetrModel):
    def __init__(self, config: RTDetrV2Config):
        super().__init__(config)
        # decoder
        self.decoder = RTDetrV2Decoder(config)


class RTDetrV2MLPPredictionHead(RTDetrMLPPredictionHead):
    pass


class RTDetrV2ForObjectDetection(RTDetrForObjectDetection, RTDetrV2PreTrainedModel):
    _tied_weights_keys = {
        r"bbox_embed.(?![0])\d+": r"bbox_embed.0",
        r"class_embed.(?![0])\d+": r"^class_embed.0",
        "class_embed": "model.decoder.class_embed",
        "bbox_embed": "model.decoder.bbox_embed",
    }

    def __init__(self, config: RTDetrV2Config):
        RTDetrV2PreTrainedModel.__init__(self, config)
        # RTDETR encoder-decoder model
        self.model = RTDetrV2Model(config)
        self.class_embed = nn.ModuleList(
            [torch.nn.Linear(config.d_model, config.num_labels) for _ in range(config.decoder_layers)]
        )
        self.bbox_embed = nn.ModuleList(
            [
                RTDetrV2MLPPredictionHead(config.d_model, config.d_model, 4, num_layers=3)
                for _ in range(config.decoder_layers)
            ]
        )
        self.model.decoder.class_embed = self.class_embed
        self.model.decoder.bbox_embed = self.bbox_embed

        # Initialize weights and apply final processing
        self.post_init()


__all__ = [
    "RTDetrV2Config",
    "RTDetrV2Model",
    "RTDetrV2PreTrainedModel",
    "RTDetrV2ForObjectDetection",
]
