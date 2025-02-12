# coding=utf-8
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
import functools
import math
from typing import List, Optional

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn

from ...activations import ACT2CLS
from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ...utils.backbone_utils import (
    verify_backbone_config_arguments,
)
from ..auto import CONFIG_MAPPING
from ..rt_detr.modeling_rt_detr import (
    RTDetrConvEncoder,
    RTDetrConvNormLayer,
    RTDetrDecoder,
    RTDetrDecoderLayer,
    RTDetrDecoderOutput,
    RTDetrEncoder,
    RTDetrForObjectDetection,
    RTDetrHybridEncoder,
    RTDetrMLPPredictionHead,
    RTDetrModel,
    RTDetrPreTrainedModel,
    RTDetrRepVggBlock,
    inverse_sigmoid,
)


logger = logging.get_logger(__name__)


class DFineConfig(PretrainedConfig):
    """
    Configuration class for D-FINE (Distribution-guided Fine-grained Object Detection).
    Extends RTDetrConfig with additional parameters specific to D-FINE architecture.

    Args:
        initializer_range (`float`, *optional*, defaults to 0.01):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_bias_prior_prob (`float`, *optional*):
            The prior probability used by the bias initializer to initialize biases for `enc_score_head` and `class_embed`.
            If `None`, `prior_prob` computed as `prior_prob = 1 / (num_labels + 1)` while initializing model weights.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        batch_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the batch normalization layers.
        backbone_config (`Dict`, *optional*, defaults to `RTDetrV2ResNetConfig()`):
            The configuration of the backbone model.
        backbone (`str`, *optional*):
            Name of backbone to use when `backbone_config` is `None`. If `use_pretrained_backbone` is `True`, this
            will load the corresponding pretrained weights from the timm or transformers library. If `use_pretrained_backbone`
            is `False`, this loads the backbone's config and uses that to initialize the backbone with random weights.
        use_pretrained_backbone (`bool`, *optional*, defaults to `False`):
            Whether to use pretrained weights for the backbone.
        use_timm_backbone (`bool`, *optional*, defaults to `False`):
            Whether to load `backbone` from the timm library. If `False`, the backbone is loaded from the transformers
            library.
        freeze_backbone_batch_norms (`bool`, *optional*, defaults to `True`):
            Whether to freeze the batch normalization layers in the backbone.
        backbone_kwargs (`dict`, *optional*):
            Keyword arguments to be passed to AutoBackbone when loading from a checkpoint
            e.g. `{'out_indices': (0, 1, 2, 3)}`. Cannot be specified if `backbone_config` is set.
        encoder_hidden_dim (`int`, *optional*, defaults to 256):
            Dimension of the layers in hybrid encoder.
        encoder_in_channels (`list`, *optional*, defaults to `[512, 1024, 2048]`):
            Multi level features input for encoder.
        feat_strides (`List[int]`, *optional*, defaults to `[8, 16, 32]`):
            Strides used in each feature map.
        encoder_layers (`int`, *optional*, defaults to 1):
            Total of layers to be used by the encoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 1024):
            Dimension of the "intermediate" (often named feed-forward) layer in decoder.
        encoder_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        dropout (`float`, *optional*, defaults to 0.0):
            The ratio for all dropout layers.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        encode_proj_layers (`List[int]`, *optional*, defaults to `[2]`):
            Indexes of the projected layers to be used in the encoder.
        positional_encoding_temperature (`int`, *optional*, defaults to 10000):
            The temperature parameter used to create the positional encodings.
        encoder_activation_function (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        activation_function (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the general layer. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        eval_size (`Tuple[int, int]`, *optional*):
            Height and width used to compute the effective height and width of the position embeddings after taking
            into account the stride.
        normalize_before (`bool`, *optional*, defaults to `False`):
            Determine whether to apply layer normalization in the transformer encoder layer before self-attention and
            feed-forward modules.
        hidden_expansion (`float`, *optional*, defaults to 1.0):
            Expansion ratio to enlarge the dimension size of RepVGGBlock and CSPRepLayer.
        d_model (`int`, *optional*, defaults to 256):
            Dimension of the layers exclude hybrid encoder.
        num_queries (`int`, *optional*, defaults to 300):
            Number of object queries.
        decoder_in_channels (`list`, *optional*, defaults to `[256, 256, 256]`):
            Multi level features dimension for decoder
        decoder_ffn_dim (`int`, *optional*, defaults to 1024):
            Dimension of the "intermediate" (often named feed-forward) layer in decoder.
        num_feature_levels (`int`, *optional*, defaults to 3):
            The number of input feature levels.
        decoder_n_points (`int`, *optional*, defaults to 4):
            The number of sampled keys in each feature level for each attention head in the decoder.
        decoder_layers (`int`, *optional*, defaults to 6):
            Number of decoder layers.
        decoder_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer decoder.
        decoder_activation_function (`str`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the decoder. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        num_denoising (`int`, *optional*, defaults to 100):
            The total number of denoising tasks or queries to be used for contrastive denoising.
        label_noise_ratio (`float`, *optional*, defaults to 0.5):
            The fraction of denoising labels to which random noise should be added.
        box_noise_scale (`float`, *optional*, defaults to 1.0):
            Scale or magnitude of noise to be added to the bounding boxes.
        learn_initial_query (`bool`, *optional*, defaults to `False`):
            Indicates whether the initial query embeddings for the decoder should be learned during training
        anchor_image_size (`Tuple[int, int]`, *optional*):
            Height and width of the input image used during evaluation to generate the bounding box anchors. If None, automatic generate anchor is applied.
        disable_custom_kernels (`bool`, *optional*, defaults to `True`):
            Whether to disable custom kernels.
        with_box_refine (`bool`, *optional*, defaults to `True`):
            Whether to apply iterative bounding box refinement, where each decoder layer refines the bounding boxes
            based on the predictions from the previous layer.
        is_encoder_decoder (`bool`, *optional*, defaults to `True`):
            Whether the architecture has an encoder decoder structure.
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
        auxiliary_loss (`bool`, *optional*, defaults to `True`):
            Whether auxiliary decoding losses (loss at each decoder layer) are to be used.
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
        eos_coefficient (`float`, *optional*, defaults to 0.0001):
            Relative classification weight of the 'no-object' class in the object detection loss.
        decoder_n_levels (`int`, *optional*, defaults to 3):
            The number of feature levels used by the decoder.
        decoder_offset_scale (`float`, *optional*, defaults to 0.5):
            Scaling factor applied to the attention offsets in the decoder.
        decoder_method (`str`, *optional*, defaults to `"default"`):
            The method to use for the decoder: `"default"` or `"discrete"`.
        decoder_offset_scale (`float`, *optional*, defaults to 0.5):
            Scaling factor for the decoder's sampling offset. Controls how far the attention
            mechanism can look relative to the reference point's bounding box size.

        eval_idx (`int`, *optional*, defaults to -1):
            Index of the decoder layer to use for evaluation. If negative, counts from the end
            (e.g., -1 means use the last layer). This allows for early prediction in the decoder
            stack while still training later layers.

        layer_scale (`float`, *optional*, defaults to 1.0):
            Scaling factor for the hidden dimension in later decoder layers. Used to adjust the
            model capacity after the evaluation layer.

        reg_max (`int`, *optional*, defaults to 32):
            Maximum number of bins for the distribution-guided bounding box refinement.
            Higher values allow for more fine-grained localization but increase computation.

        reg_scale (`float`, *optional*, defaults to 4.0):
            Scale factor for the regression distribution. Controls the range and granularity
            of the bounding box refinement process.

        depth_mult (`float`, *optional*, defaults to 1.0):
            Multiplier for the number of blocks in RepNCSPELAN4 layers. Used to scale the model's
            depth while maintaining its architecture.

        **super_kwargs:
            Additional arguments from RTDetrConfig, including standard parameters like
            hidden_size, num_attention_heads, etc.
    """

    model_type = "d_fine"
    layer_types = ["basic", "bottleneck"]
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
    }

    def __init__(
        self,
        initializer_range=0.01,
        initializer_bias_prior_prob=None,
        layer_norm_eps=1e-5,
        batch_norm_eps=1e-5,
        # backbone
        backbone_config=None,
        backbone=None,
        use_pretrained_backbone=False,
        use_timm_backbone=False,
        freeze_backbone_batch_norms=True,
        backbone_kwargs=None,
        # encoder HybridEncoder
        encoder_hidden_dim=256,
        encoder_in_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        encoder_layers=1,
        encoder_ffn_dim=1024,
        encoder_attention_heads=8,
        dropout=0.0,
        activation_dropout=0.0,
        encode_proj_layers=[2],
        positional_encoding_temperature=10000,
        encoder_activation_function="gelu",
        activation_function="silu",
        eval_size=None,
        normalize_before=False,
        hidden_expansion=1.0,
        # decoder DFineTransformer
        d_model=256,
        num_queries=300,
        decoder_in_channels=[256, 256, 256],
        decoder_ffn_dim=1024,
        num_feature_levels=3,
        decoder_n_points=4,
        decoder_layers=6,
        decoder_attention_heads=8,
        decoder_activation_function="relu",
        attention_dropout=0.0,
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learn_initial_query=False,
        anchor_image_size=None,
        disable_custom_kernels=True,
        with_box_refine=True,
        is_encoder_decoder=True,
        # Loss
        matcher_alpha=0.25,
        matcher_gamma=2.0,
        matcher_class_cost=2.0,
        matcher_bbox_cost=5.0,
        matcher_giou_cost=2.0,
        use_focal_loss=True,
        auxiliary_loss=True,
        focal_loss_alpha=0.75,
        focal_loss_gamma=2.0,
        weight_loss_vfl=1.0,
        weight_loss_bbox=5.0,
        weight_loss_giou=2.0,
        eos_coefficient=1e-4,
        decoder_offset_scale=0.5,
        eval_idx=-1,
        layer_scale=1,
        reg_max=32,
        reg_scale=4.0,
        depth_mult=1.0,
        **kwargs,
    ):
        super().__init__(is_encoder_decoder=is_encoder_decoder, **kwargs)
        self.initializer_range = initializer_range
        self.initializer_bias_prior_prob = initializer_bias_prior_prob
        self.layer_norm_eps = layer_norm_eps
        self.batch_norm_eps = batch_norm_eps
        # backbone
        if backbone_config is None and backbone is None:
            logger.info(
                "`backbone_config` and `backbone` are `None`. Initializing the config with the default `DFine-ResNet` backbone."
            )
            backbone_model_type = "d_fine_resnet"
            config_class = CONFIG_MAPPING[backbone_model_type]
            backbone_config = config_class(
                num_channels=3,
                embedding_size=64,
                hidden_sizes=[256, 512, 1024, 2048],
                depths=[3, 4, 6, 3],
                layer_type="bottleneck",
                hidden_act="relu",
                downsample_in_first_stage=False,
                downsample_in_bottleneck=False,
                out_features=None,
                out_indices=[2, 3, 4],
            )
        elif isinstance(backbone_config, dict):
            backbone_model_type = backbone_config.pop("model_type")
            config_class = CONFIG_MAPPING[backbone_model_type]
            backbone_config = config_class.from_dict(backbone_config)

        verify_backbone_config_arguments(
            use_timm_backbone=use_timm_backbone,
            use_pretrained_backbone=use_pretrained_backbone,
            backbone=backbone,
            backbone_config=backbone_config,
            backbone_kwargs=backbone_kwargs,
        )

        self.backbone_config = backbone_config
        self.backbone = backbone
        self.use_pretrained_backbone = use_pretrained_backbone
        self.use_timm_backbone = use_timm_backbone
        self.freeze_backbone_batch_norms = freeze_backbone_batch_norms
        self.backbone_kwargs = backbone_kwargs
        # encoder
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_in_channels = encoder_in_channels
        self.feat_strides = feat_strides
        self.encoder_ffn_dim = encoder_ffn_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.encode_proj_layers = encode_proj_layers
        self.encoder_layers = encoder_layers
        self.positional_encoding_temperature = positional_encoding_temperature
        self.eval_size = eval_size
        self.normalize_before = normalize_before
        self.encoder_activation_function = encoder_activation_function
        self.activation_function = activation_function
        self.hidden_expansion = hidden_expansion
        # decoder
        self.num_queries = num_queries
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_in_channels = decoder_in_channels
        self.num_feature_levels = num_feature_levels
        self.decoder_n_points = decoder_n_points
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_activation_function = decoder_activation_function
        self.attention_dropout = attention_dropout
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        self.learn_initial_query = learn_initial_query
        self.anchor_image_size = anchor_image_size
        self.auxiliary_loss = auxiliary_loss
        self.disable_custom_kernels = disable_custom_kernels
        self.with_box_refine = with_box_refine
        self.decoder_offset_scale = decoder_offset_scale
        # Loss
        self.matcher_alpha = matcher_alpha
        self.matcher_gamma = matcher_gamma
        self.matcher_class_cost = matcher_class_cost
        self.matcher_bbox_cost = matcher_bbox_cost
        self.matcher_giou_cost = matcher_giou_cost
        self.use_focal_loss = use_focal_loss
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.weight_loss_vfl = weight_loss_vfl
        self.weight_loss_bbox = weight_loss_bbox
        self.weight_loss_giou = weight_loss_giou
        self.eos_coefficient = eos_coefficient

        if not hasattr(self, "d_model"):
            self.d_model = d_model

        if not hasattr(self, "encoder_attention_heads"):
            self.encoder_attention_heads = encoder_attention_heads

        # add the new attributes with the given values or defaults
        self.eval_idx = eval_idx
        self.layer_scale = layer_scale
        self.reg_max = reg_max
        self.reg_scale = reg_scale
        self.depth_mult = depth_mult

    @classmethod
    def from_backbone_configs(cls, backbone_config: PretrainedConfig, **kwargs):
        """Instantiate a [`DFineConfig`] (or a derived class) from a pre-trained backbone model configuration and DETR model
        configuration.

            Args:
                backbone_config ([`PretrainedConfig`]):
                    The backbone configuration.

            Returns:
                [`DFineConfig`]: An instance of a configuration object
        """
        return cls(
            backbone_config=backbone_config,
            **kwargs,
        )


def deformable_attention_core_func_v2(
    value: torch.Tensor,
    value_spatial_shapes,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    num_points_list: List[int],
    method="default",
):
    """
    Args:
        value (Tensor): [bs, value_length, n_head, c]
        value_spatial_shapes (Tensor|List): [n_levels, 2]
        value_level_start_index (Tensor|List): [n_levels]
        sampling_locations (Tensor): [bs, query_length, n_head, n_levels * n_points, 2]
        attention_weights (Tensor): [bs, query_length, n_head, n_levels * n_points]

    Returns:
        output (Tensor): [bs, Length_{query}, C]
    """
    bs, n_head, c, _ = value[0].shape
    _, Len_q, _, _, _ = sampling_locations.shape

    # sampling_offsets [8, 480, 8, 12, 2]
    if method == "default":
        sampling_grids = 2 * sampling_locations - 1

    elif method == "discrete":
        sampling_grids = sampling_locations

    sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1)
    sampling_locations_list = sampling_grids.split(num_points_list, dim=-2)

    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        value_l = value[level].reshape(bs * n_head, c, h, w)
        sampling_grid_l: torch.Tensor = sampling_locations_list[level]

        if method == "default":
            sampling_value_l = F.grid_sample(
                value_l, sampling_grid_l, mode="bilinear", padding_mode="zeros", align_corners=False
            )

        elif method == "discrete":
            # n * m, seq, n, 2
            sampling_coord = (sampling_grid_l * torch.tensor([[w, h]], device=value_l.device) + 0.5).to(torch.int64)

            # FIX ME? for rectangle input
            sampling_coord = sampling_coord.clamp(0, h - 1)
            sampling_coord = sampling_coord.reshape(bs * n_head, Len_q * num_points_list[level], 2)

            s_idx = (
                torch.arange(sampling_coord.shape[0], device=value_l.device)
                .unsqueeze(-1)
                .repeat(1, sampling_coord.shape[1])
            )
            sampling_value_l: torch.Tensor = value_l[s_idx, :, sampling_coord[..., 1], sampling_coord[..., 0]]  # n l c

            sampling_value_l = sampling_value_l.permute(0, 2, 1).reshape(bs * n_head, c, Len_q, num_points_list[level])

        sampling_value_list.append(sampling_value_l)

    attn_weights = attention_weights.permute(0, 2, 1, 3).reshape(bs * n_head, 1, Len_q, sum(num_points_list))
    weighted_sample_locs = torch.concat(sampling_value_list, dim=-1) * attn_weights
    output = weighted_sample_locs.sum(-1).reshape(bs, n_head * c, Len_q)

    return output.permute(0, 2, 1)


class DFineMultiscaleDeformableAttention(nn.Module):
    def __init__(
        self,
        config: DFineConfig,
        method="default",
    ):
        """
        D-Fine version of multiscale deformable attention
        """
        super(DFineMultiscaleDeformableAttention, self).__init__()
        self.d_model = config.d_model
        self.n_levels = config.num_feature_levels
        self.n_heads = config.decoder_attention_heads
        self.n_points = config.decoder_n_points
        self.offset_scale = config.decoder_offset_scale

        if isinstance(self.n_points, list):
            assert len(self.n_points) == self.n_levels, ""
            num_points_list = self.n_points
        else:
            num_points_list = [self.n_points for _ in range(self.n_levels)]

        self.num_points_list = num_points_list

        num_points_scale = [1 / n for n in num_points_list for _ in range(n)]
        self.register_buffer("num_points_scale", torch.tensor(num_points_scale, dtype=torch.float32))

        self.total_points = self.n_heads * sum(num_points_list)
        self.method = method

        self.head_dim = self.d_model // self.n_heads
        assert self.head_dim * self.n_heads == self.d_model, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(self.d_model, self.total_points * 2)
        self.attention_weights = nn.Linear(self.d_model, self.total_points)

        self.ms_deformable_attn_core = functools.partial(deformable_attention_core_func_v2, method=self.method)

        self._reset_parameters()

        if method == "discrete":
            for p in self.sampling_offsets.parameters():
                p.requires_grad = False

    def _reset_parameters(self):
        # sampling_offsets
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.n_heads, 1, 2).tile([1, sum(self.num_points_list), 1])
        scaling = torch.concat([torch.arange(1, n + 1) for n in self.num_points_list]).reshape(1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        reference_points: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        spatial_shapes: List[int],
    ):
        bs, Len_q = hidden_states.shape[:2]

        sampling_offsets: torch.Tensor = self.sampling_offsets(hidden_states)
        sampling_offsets = sampling_offsets.reshape(bs, Len_q, self.n_heads, sum(self.num_points_list), 2)

        attention_weights = self.attention_weights(hidden_states).reshape(
            bs, Len_q, self.n_heads, sum(self.num_points_list)
        )
        attention_weights = F.softmax(attention_weights, dim=-1)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.n_levels, 1, 2)
            sampling_locations = (
                reference_points.reshape(bs, Len_q, 1, self.n_levels, 1, 2) + sampling_offsets / offset_normalizer
            )
        elif reference_points.shape[-1] == 4:
            # reference_points [8, 480, None, 1,  4]
            # sampling_offsets [8, 480, 8,    12, 2]
            num_points_scale = self.num_points_scale.to(dtype=hidden_states.dtype).unsqueeze(-1)
            offset = sampling_offsets * num_points_scale * reference_points[:, :, None, :, 2:] * self.offset_scale
            sampling_locations = reference_points[:, :, None, :, :2] + offset
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".format(reference_points.shape[-1])
            )

        output = self.ms_deformable_attn_core(
            encoder_hidden_states, spatial_shapes, sampling_locations, attention_weights, self.num_points_list
        )

        return output, attention_weights


class DFineGate(nn.Module):
    def __init__(self, d_model):
        super(DFineGate, self).__init__()
        self.gate = nn.Linear(2 * d_model, 2 * d_model)
        bias = self._bias_init_with_prob(0.5)
        init.constant_(self.gate.bias, bias)
        init.constant_(self.gate.weight, 0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x1, x2):
        gate_input = torch.cat([x1, x2], dim=-1)
        gates = torch.sigmoid(self.gate(gate_input))
        gate1, gate2 = gates.chunk(2, dim=-1)
        return self.norm(gate1 * x1 + gate2 * x2)

    def _bias_init_with_prob(self, prior_prob=0.01):
        """initialize conv/fc bias value according to a given probability value."""
        bias_init = float(-math.log((1 - prior_prob) / prior_prob))
        return bias_init


class DFineDecoderLayer(RTDetrDecoderLayer):
    def __init__(self, config: DFineConfig):
        # initialize parent class
        super().__init__(config)

        # override the encoder attention module with d-fine version
        self.encoder_attn = DFineMultiscaleDeformableAttention(config=config)
        # gate
        self.gateway = DFineGate(config.d_model)

        del self.encoder_attn_layer_norm
        self._reset_parameters()

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[torch.Tensor] = None,
        reference_points=None,
        spatial_shapes=None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ):
        residual = hidden_states

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=encoder_attention_mask,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        second_residual = hidden_states

        # Cross-Attention
        cross_attn_weights = None
        hidden_states, cross_attn_weights = self.encoder_attn(
            hidden_states=self.with_pos_embed(hidden_states, position_embeddings),
            encoder_hidden_states=encoder_hidden_states,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
        )

        hidden_states = self.gateway(
            second_residual, nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        )

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states.clamp(min=-65504, max=65504))

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs

    def _reset_parameters(self):
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)


class DFineConvEncoder(RTDetrConvEncoder):
    def __init__(self, config: DFineConfig):
        super().__init__(config)
        self.intermediate_channel_sizes = config.encoder_in_channels


class DFinePreTrainedModel(RTDetrPreTrainedModel):
    def _init_weights(self, module):
        """Initalize the weights"""

        """initialize linear layer bias value according to a given probability value."""
        if isinstance(module, (DFineForObjectDetection, DFineDecoder)):
            if module.class_embed is not None:
                for layer in module.class_embed:
                    prior_prob = self.config.initializer_bias_prior_prob or 1 / (self.config.num_labels + 1)
                    bias = float(-math.log((1 - prior_prob) / prior_prob))
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, bias)

            if module.bbox_embed is not None:
                for layer in module.bbox_embed:
                    nn.init.constant_(layer.layers[-1].weight, 0)
                    nn.init.constant_(layer.layers[-1].bias, 0)

        if isinstance(module, DFineMultiscaleDeformableAttention):
            nn.init.constant_(module.sampling_offsets.weight.data, 0.0)
            default_dtype = torch.get_default_dtype()
            thetas = torch.arange(module.n_heads, dtype=torch.int64).to(default_dtype) * (
                2.0 * math.pi / module.n_heads
            )
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
            grid_init = grid_init.reshape(module.n_heads, 1, 2).tile([1, sum(module.num_points_list), 1])
            scaling = torch.concat([torch.arange(1, n + 1) for n in module.num_points_list]).reshape(1, -1, 1)
            grid_init *= scaling
            with torch.no_grad():
                module.sampling_offsets.bias.data[...] = grid_init.flatten()

            nn.init.constant_(module.attention_weights.weight.data, 0.0)
            nn.init.constant_(module.attention_weights.bias.data, 0.0)

        if isinstance(module, DFineModel):
            prior_prob = self.config.initializer_bias_prior_prob or 1 / (self.config.num_labels + 1)
            bias = float(-math.log((1 - prior_prob) / prior_prob))
            nn.init.xavier_uniform_(module.enc_score_head.weight)
            nn.init.constant_(module.enc_score_head.bias, bias)

        if isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

        if hasattr(module, "weight_embedding") and self.config.learn_initial_query:
            nn.init.xavier_uniform_(module.weight_embedding.weight)
        if hasattr(module, "denoising_class_embed") and self.config.num_denoising > 0:
            nn.init.xavier_uniform_(module.denoising_class_embed.weight)


class DFineIntegral(nn.Module):
    """
    A static layer that calculates integral results from a distribution.

    This layer computes the target location using the formula: `sum{Pr(n) * W(n)}`,
    where Pr(n) is the softmax probability vector representing the discrete
    distribution, and W(n) is the non-uniform Weighting Function.

    Args:
        reg_max (int): Max number of the discrete bins. Default is 32.
                       It can be adjusted based on the dataset or task requirements.
    """

    def __init__(self, reg_max=32):
        super(DFineIntegral, self).__init__()
        self.reg_max = reg_max

    def forward(self, x, project):
        shape = x.shape
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, project.to(x.device)).reshape(-1, 4)
        return x.reshape(list(shape[:-1]) + [-1])


class DFineDecoder(RTDetrDecoder):
    """
    D-FINE Decoder implementing Fine-grained Distribution Refinement (FDR).

    This decoder refines object detection predictions through iterative updates across multiple layers,
    utilizing attention mechanisms, location quality estimators, and distribution refinement techniques
    to improve bounding box accuracy and robustness.
    """

    def __init__(self, config: DFineConfig):
        self.eval_idx = config.eval_idx if config.eval_idx >= 0 else config.decoder_layers + config.eval_idx
        super().__init__(config=config)
        self.reg_scale = nn.Parameter(torch.tensor([config.reg_scale]), requires_grad=False)
        self.reg_max = config.reg_max
        self.d_model = config.d_model
        self.layer_scale = config.layer_scale
        self.pre_bbox_head = DFineMLP(config.hidden_size, config.hidden_size, 4, 3)
        self.integral = DFineIntegral(self.reg_max)
        self.num_head = config.decoder_attention_heads
        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.layers = nn.ModuleList(
            [DFineDecoderLayer(config=config) for _ in range(config.decoder_layers)]
            + [DFineDecoderLayer(config=config) for _ in range(config.decoder_layers - self.eval_idx - 1)]
        )
        self.lqe_layers = nn.ModuleList([DFineLQE(4, 64, 2, config.reg_max) for _ in range(config.decoder_layers)])

    def value_op(self, memory, value_proj, value_scale, memory_mask, memory_spatial_shapes):
        """
        Preprocess values for MSDeformableAttention.
        """
        value = value_proj(memory) if value_proj is not None else memory
        value = F.interpolate(memory, size=value_scale) if value_scale is not None else value
        if memory_mask is not None:
            value = value * memory_mask.to(value.dtype).unsqueeze(-1)
        value = value.reshape(value.shape[0], value.shape[1], self.num_head, -1)
        split_shape = [h * w for h, w in memory_spatial_shapes]
        return value.permute(0, 2, 3, 1).split(split_shape, dim=-1)

    def forward(
        self,
        encoder_hidden_states,
        reference_points,
        inputs_embeds,
        spatial_shapes,
        level_start_index=None,
        spatial_shapes_list=None,
        output_hidden_states=None,
        encoder_attention_mask=None,
        memory_mask=None,
        output_attentions=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is not None:
            hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        intermediate = ()
        intermediate_reference_points = ()
        intermediate_logits = ()

        output_detach = pred_corners_undetach = 0
        value = self.value_op(encoder_hidden_states, None, None, memory_mask, spatial_shapes_list)

        project = weighting_function(self.reg_max, self.up, self.reg_scale)
        ref_points_detach = F.sigmoid(reference_points)

        for i, decoder_layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = self.query_pos_head(ref_points_detach).clamp(min=-10, max=10)

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            output = decoder_layer(
                hidden_states=hidden_states,
                position_embeddings=query_pos_embed,
                reference_points=ref_points_input,
                spatial_shapes=spatial_shapes_list,
                encoder_hidden_states=value,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )

            hidden_states = output[0]

            if i == 0:
                # Initial bounding box predictions with inverse sigmoid refinement
                new_reference_points = F.sigmoid(self.pre_bbox_head(output[0]) + inverse_sigmoid(ref_points_detach))
                ref_points_initial = new_reference_points.detach()

            # Refine bounding box corners using FDR, integrating previous layer's corrections
            if self.bbox_embed is not None:
                pred_corners = self.bbox_embed[i](hidden_states + output_detach) + pred_corners_undetach
                inter_ref_bbox = distance2bbox(
                    ref_points_initial, self.integral(pred_corners, project), self.reg_scale
                )
                pred_corners_undetach = pred_corners
                ref_points_detach = inter_ref_bbox.detach()

            output_detach = hidden_states.detach()

            intermediate += (hidden_states,)

            if self.class_embed is not None and i == self.eval_idx:
                scores = self.class_embed[i](hidden_states)
                # Lqe does not affect the performance here.
                scores = self.lqe_layers[i](scores, pred_corners)
                intermediate_logits += (scores,)
                intermediate_logits = torch.stack(intermediate_logits, dim=1)
                intermediate_reference_points += (
                    (inter_ref_bbox,) if self.bbox_embed is not None else (reference_points,)
                )
                intermediate_reference_points = torch.stack(intermediate_reference_points, dim=1)

            if output_attentions:
                all_self_attns += (output[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (output[2],)

        # Keep batch_size as first dimension
        intermediate = torch.stack(intermediate, dim=1)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    intermediate,
                    intermediate_logits,
                    intermediate_reference_points,
                    all_hidden_states,
                    all_self_attns,
                    all_cross_attentions,
                ]
                if v is not None
            )

        return DFineDecoderOutput(
            last_hidden_state=hidden_states,
            intermediate_logits=intermediate_logits,
            intermediate_reference_points=intermediate_reference_points,
            intermediate_hidden_states=intermediate,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class DFineModel(RTDetrModel):
    def __init__(self, config: DFineConfig):
        super().__init__(config)
        del self.decoder_input_proj
        self.encoder = DFineHybridEncoder(config=config)
        num_backbone_outs = len(config.decoder_in_channels)
        decoder_input_proj = []
        in_channels = config.decoder_in_channels[-1]
        for _ in range(num_backbone_outs):
            if config.hidden_size == config.decoder_in_channels[-1]:
                decoder_input_proj.append(nn.Identity())
            else:
                decoder_input_proj.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, config.d_model, kernel_size=1, bias=False),
                        nn.BatchNorm2d(config.d_model, config.batch_norm_eps),
                    )
                )
        for _ in range(config.num_feature_levels - num_backbone_outs):
            if config.hidden_size == config.decoder_in_channels[-1]:
                decoder_input_proj.append(nn.Identity())
            else:
                decoder_input_proj.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, config.d_model, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(config.d_model, config.batch_norm_eps),
                    )
                )
        self.decoder_input_proj = nn.ModuleList(decoder_input_proj)
        self.decoder = DFineDecoder(config)


class DFineForObjectDetection(RTDetrForObjectDetection, DFinePreTrainedModel):
    def __init__(self, config: DFineConfig):
        DFinePreTrainedModel.__init__(config)

        # D-FINE encoder-decoder model
        self.eval_idx = config.eval_idx if config.eval_idx >= 0 else config.decoder_layers + config.eval_idx
        self.model = DFineModel(config)
        scaled_dim = round(config.layer_scale * config.hidden_size)
        num_pred = config.decoder_layers
        self.class_embed = functools.partial(nn.Linear, config.d_model, config.num_labels)
        self.class_embed = nn.ModuleList([self.class_embed() for _ in range(num_pred)])
        self.bbox_embed = nn.ModuleList(
            [
                DFineMLP(config.hidden_size, config.hidden_size, 4 * (config.reg_max + 1), 3)
                for _ in range(self.eval_idx + 1)
            ]
            + [
                DFineMLP(scaled_dim, scaled_dim, 4 * (config.reg_max + 1), 3)
                for _ in range(config.decoder_layers - self.eval_idx - 1)
            ]
        )

        # here self.model.decoder.bbox_embed is null, but not self.bbox_embed
        self.model.decoder.class_embed = self.class_embed
        self.model.decoder.bbox_embed = self.bbox_embed

        # Initialize weights and apply final processing
        self.post_init()


def weighting_function(reg_max, up, reg_scale):
    """
    Generates the non-uniform Weighting Function W(n) for bounding box regression.

    Args:
        reg_max (int): Max number of the discrete bins.
        up (Tensor): Controls upper bounds of the sequence,
                     where maximum offset is ±up * H / W.
        reg_scale (float): Controls the curvature of the Weighting Function.
                           Larger values result in flatter weights near the central axis W(reg_max/2)=0
                           and steeper weights at both ends.
    Returns:
        Tensor: Sequence of Weighting Function.
    """
    upper_bound1 = abs(up[0]) * abs(reg_scale)
    upper_bound2 = abs(up[0]) * abs(reg_scale) * 2
    step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
    left_values = [-((step) ** i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
    right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
    values = [-upper_bound2] + left_values + [torch.zeros_like(up[0][None])] + right_values + [upper_bound2]
    return torch.cat(values, 0)


class DFineMLPPredictionHead(RTDetrMLPPredictionHead):
    pass


def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def distance2bbox(points, distance, reg_scale):
    """
    Decodes edge-distances into bounding box coordinates.

    Args:
        points (Tensor): (B, N, 4) or (N, 4) format, representing [x, y, w, h],
                         where (x, y) is the center and (w, h) are width and height.
        distance (Tensor): (B, N, 4) or (N, 4), representing distances from the
                           point to the left, top, right, and bottom boundaries.

        reg_scale (float): Controls the curvature of the Weighting Function.

    Returns:
        Tensor: Bounding boxes in (N, 4) or (B, N, 4) format [cx, cy, w, h].
    """
    reg_scale = abs(reg_scale)
    x1 = points[..., 0] - (0.5 * reg_scale + distance[..., 0]) * (points[..., 2] / reg_scale)
    y1 = points[..., 1] - (0.5 * reg_scale + distance[..., 1]) * (points[..., 3] / reg_scale)
    x2 = points[..., 0] + (0.5 * reg_scale + distance[..., 2]) * (points[..., 2] / reg_scale)
    y2 = points[..., 1] + (0.5 * reg_scale + distance[..., 3]) * (points[..., 3] / reg_scale)

    bboxes = torch.stack([x1, y1, x2, y2], -1)

    return box_xyxy_to_cxcywh(bboxes)


class DFineMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act="relu"):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = ACT2CLS[act]()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DFineLQE(nn.Module):
    def __init__(self, k, hidden_dim, num_layers, reg_max):
        super(DFineLQE, self).__init__()
        self.k = k
        self.reg_max = reg_max
        self.reg_conf = DFineMLP(4 * (k + 1), hidden_dim, 1, num_layers)
        init.constant_(self.reg_conf.layers[-1].bias, 0)
        init.constant_(self.reg_conf.layers[-1].weight, 0)

    def forward(self, scores, pred_corners):
        B, L, _ = pred_corners.size()
        prob = F.softmax(pred_corners.reshape(B, L, 4, self.reg_max + 1), dim=-1)
        prob_topk, _ = prob.topk(self.k, dim=-1)
        stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)
        quality_score = self.reg_conf(stat.reshape(B, L, -1))
        return scores + quality_score


class DFineDecoderOutput(RTDetrDecoderOutput):
    pass


class DFineVggBlock(RTDetrRepVggBlock):
    pass


class DFineConvNormLayer(RTDetrConvNormLayer):
    def __init__(
        self, config, in_channels, out_channels, kernel_size, stride, groups=1, padding=None, activation=None
    ):
        super().__init__(config, in_channels, out_channels, kernel_size, stride, padding=None, activation=activation)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            groups=groups,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=False,
        )


class DFineRepVggBlock(nn.Module):
    def __init__(self, config: DFineConfig, in_channels: int, out_channels: int):
        super().__init__()

        activation = config.activation_function
        self.conv1 = DFineConvNormLayer(config, in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = DFineConvNormLayer(config, in_channels, out_channels, 1, 1, padding=0)
        self.activation = nn.Identity() if activation is None else ACT2CLS[activation]()

    def forward(self, x):
        y = self.conv1(x) + self.conv2(x)
        return self.activation(y)


class DFineCSPRepLayer(nn.Module):
    """
    Cross Stage Partial (CSP) network layer with RepVGG blocks.
    """

    def __init__(self, config: DFineConfig, in_channels: int, out_channels: int, num_blocks: int, expansion=1.0):
        super().__init__()
        in_channels = in_channels
        out_channels = out_channels
        activation = config.activation_function

        hidden_channels = int(out_channels * expansion)
        self.conv1 = DFineConvNormLayer(config, in_channels, hidden_channels, 1, 1, activation=activation)
        self.conv2 = DFineConvNormLayer(config, in_channels, hidden_channels, 1, 1, activation=activation)
        self.bottlenecks = nn.Sequential(
            *[DFineRepVggBlock(config, hidden_channels, hidden_channels) for _ in range(num_blocks)]
        )
        if hidden_channels != out_channels:
            self.conv3 = DFineConvNormLayer(config, hidden_channels, out_channels, 1, 1, activation=activation)
        else:
            self.conv3 = nn.Identity()

    def forward(self, hidden_state):
        device = hidden_state.device
        hidden_state_1 = self.conv1(hidden_state)
        hidden_state_1 = self.bottlenecks(hidden_state_1).to(device)
        hidden_state_2 = self.conv2(hidden_state).to(device)
        return self.conv3(hidden_state_1 + hidden_state_2)


class DFineRepNCSPELAN4(nn.Module):
    # csp-elan
    def __init__(self, config: DFineConfig, act="silu", numb_blocks=3):
        super().__init__()
        c1 = config.encoder_hidden_dim * 2
        c2 = config.encoder_hidden_dim
        c3 = config.encoder_hidden_dim * 2
        c4 = round(config.hidden_expansion * config.encoder_hidden_dim // 2)
        self.c = c3 // 2
        self.cv1 = DFineConvNormLayer(config, c1, c3, 1, 1, activation=act)
        self.cv2 = nn.Sequential(
            DFineCSPRepLayer(config, c3 // 2, c4, num_blocks=numb_blocks),
            DFineConvNormLayer(config, c4, c4, 3, 1, activation=act),
        )
        self.cv3 = nn.Sequential(
            DFineCSPRepLayer(config, c4, c4, num_blocks=numb_blocks),
            DFineConvNormLayer(config, c4, c4, 3, 1, activation=act),
        )
        self.cv4 = DFineConvNormLayer(config, c3 + (2 * c4), c2, 1, 1, activation=act)

    def forward_chunk(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class DFineSCDown(nn.Module):
    def __init__(self, config: DFineConfig, c1, c2, k, s):
        super().__init__()
        self.cv1 = DFineConvNormLayer(config, c1, c2, 1, 1)
        self.cv2 = DFineConvNormLayer(config, c2, c2, k, s, c2)

    def forward(self, x):
        return self.cv2(self.cv1(x))


class DFineEncoder(RTDetrEncoder):
    pass


class DFineHybridEncoder(RTDetrHybridEncoder):
    def __init__(self, config: DFineConfig):
        nn.Module.__init__(self)
        self.config = config
        self.in_channels = config.encoder_in_channels
        self.feat_strides = config.feat_strides
        self.encoder_hidden_dim = config.encoder_hidden_dim
        self.encode_proj_layers = config.encode_proj_layers
        self.positional_encoding_temperature = config.positional_encoding_temperature
        self.eval_size = config.eval_size
        self.out_channels = [self.encoder_hidden_dim for _ in self.in_channels]
        self.out_strides = self.feat_strides

        # encoder transformer
        self.encoder = nn.ModuleList([DFineEncoder(config) for _ in range(len(self.encode_proj_layers))])
        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(self.in_channels) - 1, 0, -1):
            self.lateral_convs.append(
                DFineConvNormLayer(config, self.encoder_hidden_dim, self.encoder_hidden_dim, 1, 1)
            )
            self.fpn_blocks.append(DFineRepNCSPELAN4(config, numb_blocks=round(3 * config.depth_mult)))

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(self.in_channels) - 1):
            self.downsample_convs.append(
                nn.Sequential(
                    DFineSCDown(config, self.encoder_hidden_dim, self.encoder_hidden_dim, 3, 2),
                )
            )
            self.pan_blocks.append(DFineRepNCSPELAN4(config, numb_blocks=round(3 * config.depth_mult)))


__all__ = [
    "DFineConfig",
    "DFineModel",
    "DFinePreTrainedModel",
    "DFineForObjectDetection",
]
