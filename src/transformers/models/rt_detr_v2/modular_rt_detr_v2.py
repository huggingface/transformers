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
from functools import partial
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...configuration_utils import PretrainedConfig
from ...utils import is_torchdynamo_compiling, logging
from ...utils.backbone_utils import (
    verify_backbone_config_arguments,
)
from ..auto import CONFIG_MAPPING
from ..rt_detr.modeling_rt_detr import (
    RTDetrDecoder,
    RTDetrDecoderLayer,
    RTDetrForObjectDetection,
    RTDetrMLPPredictionHead,
    RTDetrModel,
    RTDetrPreTrainedModel,
)


logger = logging.get_logger(__name__)


class RTDetrV2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`RTDetrV2Model`]. It is used to instantiate a
    RT-DETR model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the RT-DETR architecture.

    e.g. [PekingU/rtdetr_r18vd](https://huggingface.co/PekingU/rtdetr_r18vd)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

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
        # decoder RTDetrV2Transformer
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
        decoder_n_levels=3,  # default value
        decoder_offset_scale=0.5,  # default value
        decoder_method="default",
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
                "`backbone_config` and `backbone` are `None`. Initializing the config with the default `RTDetrV2-ResNet` backbone."
            )
            backbone_model_type = "rt_detr_resnet"
            config_class = CONFIG_MAPPING[backbone_model_type]
            # this will map it to RTDetrResNetConfig
            # note: we can instead create RTDetrV2ResNetConfig but it will be exactly the same as V1
            # and we would need to create RTDetrV2ResNetModel
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
        self.with_box_refine = with_box_refine
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
        self.decoder_n_levels = decoder_n_levels
        self.decoder_offset_scale = decoder_offset_scale
        self.decoder_method = decoder_method

    @classmethod
    def from_backbone_configs(cls, backbone_config: PretrainedConfig, **kwargs):
        """Instantiate a [`RTDetrV2Config`] (or a derived class) from a pre-trained backbone model configuration and DETR model
        configuration.

            Args:
                backbone_config ([`PretrainedConfig`]):
                    The backbone configuration.

            Returns:
                [`RTDetrV2Config`]: An instance of a configuration object
        """
        return cls(
            backbone_config=backbone_config,
            **kwargs,
        )


def multi_scale_deformable_attention_v2(
    value: Tensor,
    value_spatial_shapes: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
    num_points_list: List[int],
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
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_embeddings: Optional[torch.Tensor] = None,
        reference_points=None,
        spatial_shapes=None,
        spatial_shapes_list=None,
        level_start_index=None,
        output_attentions: bool = False,
    ):
        # Process inputs up to sampling locations calculation using parent class logic
        if position_embeddings is not None:
            hidden_states = hidden_states + position_embeddings

        batch_size, num_queries, _ = hidden_states.shape
        batch_size, sequence_length, _ = encoder_hidden_states.shape
        if not is_torchdynamo_compiling() and (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() != sequence_length:
            raise ValueError(
                "Make sure to align the spatial shapes with the sequence length of the encoder hidden states"
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
    pass


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
    def __init__(self, config: RTDetrV2Config):
        RTDetrV2PreTrainedModel.__init__(config)
        # RTDETR encoder-decoder model
        self.model = RTDetrV2Model(config)

        # Detection heads on top
        class_embed = partial(nn.Linear, config.d_model, config.num_labels)
        bbox_embed = partial(RTDetrV2MLPPredictionHead, config, config.d_model, config.d_model, 4, num_layers=3)

        self.class_embed = nn.ModuleList([class_embed() for _ in range(config.decoder_layers)])
        self.bbox_embed = nn.ModuleList([bbox_embed() for _ in range(config.decoder_layers)])

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
