from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F  # noqa: F401
from torch import nn

from ...activations import ACT2FN
from ...configuration_utils import PretrainedConfig
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    logging,
)
from ...utils.generic import check_model_inputs
from ..conditional_detr.modeling_conditional_detr import (
    ConditionalDetrConvEncoder,
    ConditionalDetrConvModel,
    build_position_encoding,
)
from ..convnext.modeling_convnext import ConvNextLayerNorm
from ..dab_detr.modeling_dab_detr import gen_sine_position_embeddings
from ..deformable_detr.image_processing_deformable_detr import DeformableDetrImageProcessor
from ..deformable_detr.image_processing_deformable_detr_fast import DeformableDetrImageProcessorFast
from ..deformable_detr.modeling_deformable_detr import (
    DeformableDetrDecoderLayer,
    DeformableDetrDecoderOutput,
    DeformableDetrForObjectDetection,
    DeformableDetrMLPPredictionHead,
    DeformableDetrModel,
    DeformableDetrModelOutput,
    DeformableDetrMultiscaleDeformableAttention,
    DeformableDetrObjectDetectionOutput,
    DeformableDetrPreTrainedModel,
    _get_clones,
)
from ..llama.modeling_llama import eager_attention_forward
from ..rt_detr.configuration_rt_detr import CONFIG_MAPPING, verify_backbone_config_arguments
from ..rt_detr.modeling_rt_detr import RTDetrConvNormLayer, RTDetrRepVggBlock
from .configuration_lw_detr_vit import LwDetrVitConfig


logger = logging.get_logger(__name__)


class LwDetrImageProcessor(DeformableDetrImageProcessor):
    pass


class LwDetrImageProcessorFast(DeformableDetrImageProcessorFast):
    def __init__(self, **super_kwargs):
        super().__init__(**super_kwargs)


class LwDetrDecoderOutput(DeformableDetrDecoderOutput):
    pass


class LwDetrModelOutput(DeformableDetrModelOutput):
    pass


class LwDetrConfig(PretrainedConfig):
    model_type = "lw_detr"
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "decoder_self_attention_heads",
        "num_key_value_heads": "decoder_self_attention_heads",
    }

    def __init__(
        self,
        # backbone
        backbone_config=None,
        backbone=None,
        use_pretrained_backbone=False,
        use_timm_backbone=False,
        freeze_backbone_batch_norms=True,
        backbone_kwargs=None,
        # projector
        projector_scale_factors: list[float] = [],
        hidden_expansion=0.5,
        activation_function="silu",
        batch_norm_eps=1e-5,
        # decoder
        d_model=256,
        dropout=0.1,
        decoder_ffn_dim=2048,
        decoder_n_points=4,
        decoder_layers: int = 3,
        decoder_self_attention_heads: int = 8,
        decoder_cross_attention_heads: int = 16,
        decoder_activation_function="relu",
        # model
        num_queries=300,
        attention_bias=True,
        attention_dropout=0.0,
        activation_dropout=0.0,
        position_embedding_type="sine",
        two_stage=True,
        group_detr: int = 13,
        init_std=0.02,
        init_xavier_std=1.0,
        disable_custom_kernels=True,
        bbox_reparam=True,
        is_encoder_decoder=False,
        # loss
        class_cost=2,
        bbox_cost=5,
        giou_cost=2,
        mask_loss_coefficient=1,
        dice_loss_coefficient=1,
        bbox_loss_coefficient=5,
        giou_loss_coefficient=2,
        eos_coefficient=0.1,
        focal_alpha=0.25,
        auxiliary_loss=True,
        **kwargs,
    ):
        self.batch_norm_eps = batch_norm_eps

        # backbone
        if backbone_config is None and backbone is None:
            logger.info(
                "`backbone_config` and `backbone` are `None`. Initializing the config with the default `LwDetrViT` backbone."
            )
            backbone_config = LwDetrVitConfig(
                # TODO: add default config
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

        # projector
        self.projector_scale_factors = projector_scale_factors
        for scale in projector_scale_factors:
            if scale not in [0.5, 1.0, 2.0]:
                raise ValueError(f"Unsupported scale factor: {scale}")
        self.projector_in_channels = [d_model] * len(projector_scale_factors)
        self.projector_out_channels = d_model
        self.activation_function = activation_function
        self.hidden_expansion = hidden_expansion

        # decoder
        self.d_model = d_model
        self.dropout = dropout
        self.num_queries = num_queries
        self.decoder_ffn_dim = decoder_ffn_dim
        self.num_feature_levels = len(self.projector_scale_factors)
        self.decoder_n_points = decoder_n_points
        self.decoder_layers = decoder_layers
        self.decoder_activation_function = decoder_activation_function
        self.decoder_self_attention_heads = decoder_self_attention_heads
        self.decoder_cross_attention_heads = decoder_cross_attention_heads
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout

        self.position_embedding_type = position_embedding_type
        self.two_stage = two_stage

        self.disable_custom_kernels = disable_custom_kernels

        # Loss
        self.auxiliary_loss = auxiliary_loss

        # Hungarian matcher
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        # Loss coefficients
        self.mask_loss_coefficient = mask_loss_coefficient
        self.dice_loss_coefficient = dice_loss_coefficient
        self.bbox_loss_coefficient = bbox_loss_coefficient
        self.giou_loss_coefficient = giou_loss_coefficient
        self.eos_coefficient = eos_coefficient
        self.focal_alpha = focal_alpha

        self.init_std = init_std
        self.init_xavier_std = init_xavier_std
        self.bbox_reparam = bbox_reparam
        self.group_detr = group_detr

        super().__init__(is_encoder_decoder=is_encoder_decoder, **kwargs)


class LwDetrConvNormLayer(RTDetrConvNormLayer):
    def __init__(self, **super_kwargs):
        super().__init__(**super_kwargs)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=kernel_size // 2,
            bias=False,
        )


class LwDetrRepVggBlock(RTDetrRepVggBlock):
    def __init__(self, config: LwDetrConfig, hidden_channels: int):
        super().__init__(config)
        del hidden_channels
        del self.activation
        del activation
        self.conv1 = LwDetrConvNormLayer(config, hidden_channels, hidden_channels, 3, 1, padding=1, activation="silu")
        self.conv2 = LwDetrConvNormLayer(config, hidden_channels, hidden_channels, 3, 1, padding=0, activation="silu")

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        return y


class LwDetrCSPRepLayer(nn.Module):
    # Inspired by RTDetrCSPRepLayer
    def __init__(self, config: LwDetrConfig, in_channels: int, out_channels: int):
        super().__init__()
        num_blocks = 3
        activation = config.activation_function

        self.hidden_channels = int(out_channels * config.hidden_expansion)
        conv1_out_channels = 2 * self.hidden_channels
        conv2_in_channels = (2 + num_blocks) * self.hidden_channels
        self.conv1 = LwDetrConvNormLayer(config, in_channels, conv1_out_channels, 1, 1, activation=activation)
        self.conv2 = LwDetrConvNormLayer(config, conv2_in_channels, out_channels, 1, 1, activation=activation)
        self.bottlenecks = nn.ModuleList(LwDetrRepVggBlock(config, self.hidden_channels) for _ in range(num_blocks))

    def forward(self, hidden_state):
        hidden_state = self.conv1(hidden_state)
        all_hidden_states = list(hidden_state.split(self.hidden_channels, 1))
        last_hidden_state = all_hidden_states[-1]

        bottleneck_hidden_states = []
        for bottleneck in self.bottlenecks:
            bottleneck_hidden_states.append(bottleneck(last_hidden_state))

        all_hidden_states.extend(bottleneck_hidden_states)
        hidden_state = torch.cat(all_hidden_states, 1)
        hidden_state = self.conv2(hidden_state)
        return hidden_state


class LwDetrLayerNorm(ConvNextLayerNorm):
    pass


class LwDetrSamplingLayer(nn.Module):
    def __init__(self, config: LwDetrConfig, channel_size: int, scale: float):
        super().__init__()

        self.scale = scale
        self.channel_size = channel_size

        layers = []
        if scale == 2.0:
            if channel_size > 512:
                layers.extend(
                    [
                        LwDetrConvNormLayer(config, channel_size, channel_size // 2, 1, 1, activation="relu"),
                        nn.ConvTranspose2d(channel_size // 2, channel_size // 4, kernel_size=2, stride=2),
                    ]
                )
            else:
                layers.append(nn.ConvTranspose2d(channel_size, channel_size // 2, 2, 2))
        elif scale == 0.5:
            layers.append(LwDetrConvNormLayer(config, channel_size, channel_size, 3, 2, activation="relu"))
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class LwDetrScaleProjector(nn.Module):
    def __init__(self, config: LwDetrConfig, intermediate_dims: list[int], scale: float, output_dim: int):
        super().__init__()

        sampling_layers = []
        for channel_size in intermediate_dims:
            sampling_layers.append(LwDetrSamplingLayer(config, channel_size, scale))
        self.sampling_layers = nn.ModuleList(sampling_layers)

        intermediate_dim = intermediate_dims[-1]
        if scale == 2.0:
            if intermediate_dim > 512:
                intermediate_dim = intermediate_dim // 4
            else:
                intermediate_dim = intermediate_dim // 2
        projector_input_dim = intermediate_dim * len(intermediate_dims)

        self.projector_layer = LwDetrCSPRepLayer(config, projector_input_dim, output_dim)
        self.layer_norm = LwDetrLayerNorm(output_dim, data_format="channels_first")

    def forward(self, hidden_states_tuple: tuple[torch.Tensor]):
        sampled_hidden_states = []
        for sampling_layer, hidden_states in zip(self.sampling_layers, hidden_states_tuple):
            hidden_states = sampling_layer(hidden_states)
            sampled_hidden_states.append(hidden_states)
        hidden_states = torch.cat(sampled_hidden_states, dim=1)
        hidden_states = self.projector_layer(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class LwDetrMultiScaleProjector(nn.Module):
    def __init__(self, config: LwDetrConfig, intermediate_channel_sizes: list[int]):
        super().__init__()

        self.config = config
        scale_factors = config.projector_scale_factors
        output_channels = config.d_model

        self.scale_layers = nn.ModuleList(
            [
                LwDetrScaleProjector(config, intermediate_channel_sizes, scale, output_channels)
                for scale in scale_factors
            ]
        )

    def forward(self, hidden_states: tuple[torch.Tensor]):
        output_hidden_states = []
        for scale_layer in self.scale_layers:
            output_hidden_states.append(scale_layer(hidden_states))
        return output_hidden_states


class LwDetrConvEncoder(ConditionalDetrConvEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.projector = LwDetrMultiScaleProjector(config, self.intermediate_channel_sizes)

    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor):
        # send pixel_values through the model to get list of feature maps
        features = self.model(pixel_values) if self.config.use_timm_backbone else self.model(pixel_values).feature_maps
        features = self.projector(features)
        out = []
        for feature_map in features:
            # downsample pixel_mask to match shape of corresponding feature_map
            mask = nn.functional.interpolate(pixel_mask[None].float(), size=feature_map.shape[-2:]).to(torch.bool)[0]
            out.append((feature_map, mask))
        return out


class LwDetrConvModel(ConditionalDetrConvModel):
    pass


class LwDetrMultiheadAttention(nn.Module):
    def __init__(
        self,
        config: LwDetrConfig,
    ):
        super().__init__()
        self.config = config
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]):
        return tensor if position_embeddings is None else tensor + position_embeddings

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        batch_size, seq_len, embed_dim = hidden_states.shape
        input_shape = hidden_states.shape[:-1]

        hidden_states_original = hidden_states
        if position_embeddings is not None:
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)

        if self.training:
            hidden_states_original = torch.cat(
                hidden_states_original.split(seq_len // self.config.group_detr, dim=1), dim=0
            )
            hidden_states = torch.cat(hidden_states.split(seq_len // self.config.group_detr, dim=1), dim=0)

        query_states = self.q_proj(hidden_states).view(batch_size, seq_len, -1, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(batch_size, seq_len, -1, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states_original).view(batch_size, seq_len, -1, self.head_dim).transpose(1, 2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=None,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if self.training:
            attn_output = torch.cat(torch.split(attn_output, batch_size, dim=0), dim=1)

        return attn_output, attn_weights


class LwDetrMultiscaleDeformableAttention(DeformableDetrMultiscaleDeformableAttention):
    pass


class LwDetrDecoderLayer(DeformableDetrDecoderLayer):
    def __init__(self, config: LwDetrConfig):
        GradientCheckpointingLayer.__init__(self)
        self.embed_dim = config.d_model

        # self-attention
        self.self_attn = LwDetrMultiheadAttention(config)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.decoder_activation_function]
        self.activation_dropout = config.activation_dropout
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # cross-attention
        self.cross_attn = LwDetrMultiscaleDeformableAttention(
            config,
            num_heads=config.decoder_cross_attention_heads,
            n_points=config.decoder_n_points,
        )
        self.cross_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # feedforward neural networks
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[torch.Tensor] = None,
        reference_points=None,
        spatial_shapes=None,
        spatial_shapes_list=None,
        level_start_index=None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        self_attention_output, self_attn_weights = self.self_attn(
            hidden_states, position_embeddings=position_embeddings, **kwargs
        )

        self_attention_output = nn.functional.dropout(self_attention_output, p=self.dropout, training=self.training)
        hidden_states = hidden_states + self_attention_output
        hidden_states = self.self_attn_layer_norm(hidden_states)

        cross_attention_output, cross_attn_weights = self.cross_attn(
            hidden_states=hidden_states,
            attention_mask=encoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            **kwargs,
        )
        cross_attention_output = nn.functional.dropout(cross_attention_output, p=self.dropout, training=self.training)
        hidden_states = hidden_states + cross_attention_output
        hidden_states = self.cross_attn_layer_norm(hidden_states)

        # FFN
        residual = hidden_states
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class LwDetrPreTrainedModel(DeformableDetrPreTrainedModel):
    _no_split_modules = [
        r"LwDetrConvEncoder",
        r"LwDetrDecoderLayer",
    ]
    _can_record_outputs = {
        "attentions": [LwDetrMultiheadAttention, LwDetrMultiscaleDeformableAttention],
        "hidden_states": [LwDetrDecoderLayer],
    }
    # TODO Add other features


def refine_bboxes(reference_points, deltas, bbox_reparam):
    if bbox_reparam:
        new_reference_points_cxcy = deltas[..., :2] * reference_points[..., 2:] + reference_points[..., :2]
        new_reference_points_wh = deltas[..., 2:].exp() * reference_points[..., 2:]
        new_reference_points = torch.cat((new_reference_points_cxcy, new_reference_points_wh), -1)
    else:
        new_reference_points = deltas + reference_points
    return new_reference_points


class LwDetrDecoder(LwDetrPreTrainedModel):
    def __init__(self, config: LwDetrConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layers = nn.ModuleList([LwDetrDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False

        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

        self.ref_point_head = LwDetrMLPPredictionHead(2 * config.d_model, config.d_model, config.d_model, num_layers=2)

        # Initialize weights and apply final processing
        self.post_init()

    def get_reference(self, reference_points, valid_ratios):
        # [num_queries, batch_size, 4]
        obj_center = reference_points[..., :4]

        refpoints_input = (
            obj_center[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
        )  # bs, nq, nlevel, 4
        query_sine_embed = gen_sine_position_embeddings(
            refpoints_input[:, :, 0, :], self.config.d_model
        )  # bs, nq, 256*2
        query_pos = self.ref_point_head(query_sine_embed)
        return obj_center, refpoints_input, query_pos, query_sine_embed

    def forward(
        self,
        inputs_embeds: Optional[torch.Tensor] = None,
        reference_points: Optional[torch.Tensor] = None,
        query_position_embeddings: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
        spatial_shapes_list: Optional[torch.Tensor] = None,
        level_start_index: Optional[torch.Tensor] = None,
        valid_ratios: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        intermediate = ()
        intermediate_reference_points = (reference_points,)

        if inputs_embeds is not None:
            hidden_states = inputs_embeds
            input_shape = inputs_embeds.size()[:-1]

        if self.config.bbox_reparam:
            get_reference_points_input = reference_points
        else:
            get_reference_points_input = reference_points.sigmoid()
        obj_center, reference_points_inputs, query_pos, query_sine_embed = self.get_reference(
            get_reference_points_input, valid_ratios
        )

        for idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                position_embeddings=query_pos,
                reference_points=reference_points_inputs,
                spatial_shapes=spatial_shapes,
                spatial_shapes_list=spatial_shapes_list,
                level_start_index=level_start_index,
                **kwargs,
            )
            intermediate_hidden_states = self.layernorm(hidden_states)
            intermediate += (intermediate_hidden_states,)

        intermediate = torch.stack(intermediate)
        intermediate_reference_points = torch.stack(intermediate_reference_points)

        return LwDetrDecoderOutput(
            last_hidden_state=hidden_states,
            intermediate_hidden_states=intermediate,
            intermediate_reference_points=intermediate_reference_points,
        )


class LwDetrModel(DeformableDetrModel):
    def __init__(self, config: LwDetrConfig):
        LwDetrPreTrainedModel.__init__(config)

        # Create backbone + positional encoding
        backbone = LwDetrConvEncoder(config)
        position_embeddings = build_position_encoding(config)
        self.backbone = LwDetrConvModel(backbone, position_embeddings)

        self.group_detr = config.group_detr
        self.num_queries = config.num_queries
        hidden_dim = config.d_model
        query_dim = 4
        self.refpoint_embed = nn.Embedding(self.num_queries * self.group_detr, query_dim)
        self.query_feat = nn.Embedding(self.num_queries * self.group_detr, hidden_dim)

        self.decoder = LwDetrDecoder(config)

        if config.two_stage:
            self.enc_output = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(self.group_detr)])
            self.enc_output_norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.group_detr)])
            self.enc_out_bbox_embed = None
            self.enc_out_class_embed = None

        self.post_init()

    def gen_encoder_output_proposals(self, enc_output, padding_mask, spatial_shapes):
        """Generate the encoder output proposals from encoded enc_output.

        Args:
            enc_output (Tensor[batch_size, sequence_length, hidden_size]): Output of the encoder.
            padding_mask (Tensor[batch_size, sequence_length]): Padding mask for `enc_output`.
            spatial_shapes (list[tuple[int, int]]): Spatial shapes of the feature maps.

        Returns:
            `tuple(torch.FloatTensor)`: A tuple of feature map and bbox prediction.
                - object_query (Tensor[batch_size, sequence_length, hidden_size]): Object query features. Later used to
                  directly predict a bounding box. (without the need of a decoder)
                - output_proposals (Tensor[batch_size, sequence_length, 4]): Normalized proposals, after an inverse
                  sigmoid.
        """
        batch_size = enc_output.shape[0]
        proposals = []
        _cur = 0
        for level, (height, width) in enumerate(spatial_shapes):
            mask_flatten_ = padding_mask[:, _cur : (_cur + height * width)].view(batch_size, height, width, 1)
            valid_height = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_width = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = meshgrid(
                torch.linspace(
                    0,
                    height - 1,
                    height,
                    dtype=enc_output.dtype,
                    device=enc_output.device,
                ),
                torch.linspace(
                    0,
                    width - 1,
                    width,
                    dtype=enc_output.dtype,
                    device=enc_output.device,
                ),
                indexing="ij",
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_width.unsqueeze(-1), valid_height.unsqueeze(-1)], 1).view(batch_size, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(batch_size, -1, -1, -1) + 0.5) / scale
            width_height = torch.ones_like(grid) * 0.05 * (2.0**level)
            proposal = torch.cat((grid, width_height), -1).view(batch_size, -1, 4)
            proposals.append(proposal)
            _cur += height * width
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        # TODO careful
        # output_proposals = torch.log(output_proposals / (1 - output_proposals))  # inverse sigmoid
        output_proposals = output_proposals.masked_fill(padding_mask.unsqueeze(-1), float("inf"))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))

        # assign each pixel as an object query
        object_query = enc_output
        object_query = object_query.masked_fill(padding_mask.unsqueeze(-1), float(0))
        object_query = object_query.masked_fill(~output_proposals_valid, float(0))
        return object_query, output_proposals

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple[torch.FloatTensor], LwDetrModelOutput]:
        r"""
        decoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, num_queries)`, *optional*):
            Not used by default. Can be used to mask object queries.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing the flattened feature map (output of the backbone + projection layer), you
            can choose to directly pass a flattened representation of an image.
        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
            Optionally, instead of initializing the queries with a tensor of zeros, you can choose to directly pass an
            embedded representation.

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, DeformableDetrModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")
        >>> model = DeformableDetrModel.from_pretrained("SenseTime/deformable-detr")

        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)

        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 300, 256]
        ```"""
        batch_size, num_channels, height, width = pixel_values.shape
        device = pixel_values.device

        if pixel_mask is None:
            pixel_mask = torch.ones(((batch_size, height, width)), dtype=torch.long, device=device)

        # Extract multi-scale feature maps of same resolution `config.d_model` (cf Figure 4 in paper)
        # First, sent pixel_values + pixel_mask through Backbone to obtain the features
        # which is a list of tuples
        features, position_embeddings_list = self.backbone(pixel_values, pixel_mask)

        # Then, apply 1x1 convolution to reduce the channel dimension to d_model (256 by default)
        sources = []
        masks = []
        for level, (source, mask) in enumerate(features):
            sources.append(source)
            masks.append(mask)
            if mask is None:
                raise ValueError("No attention mask was provided")

        if self.training:
            reference_points = self.refpoint_embed.weight
            query_feat = self.query_feat.weight
        else:
            # only use one group in inference
            reference_points = self.refpoint_embed.weight[: self.num_queries]
            query_feat = self.query_feat.weight[: self.num_queries]

        # Prepare encoder inputs (by flattening)
        source_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes_list = []
        for level, (source, mask, pos_embed) in enumerate(zip(sources, masks, position_embeddings_list)):
            batch_size, num_channels, height, width = source.shape
            spatial_shape = (height, width)
            spatial_shapes_list.append(spatial_shape)
            source = source.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed_flatten.append(pos_embed)
            source_flatten.append(source)
            mask_flatten.append(mask)
        source_flatten = torch.cat(source_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes_list, dtype=torch.long, device=source_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m, dtype=source_flatten.dtype) for m in masks], 1)

        target = query_feat.unsqueeze(0).expand(batch_size, -1, -1)
        reference_points = reference_points.unsqueeze(0).expand(batch_size, -1, -1)

        enc_outputs_class = None
        enc_outputs_coord_logits = None
        if self.config.two_stage:
            object_query_embedding, output_proposals = self.gen_encoder_output_proposals(
                source_flatten, ~mask_flatten, spatial_shapes_list
            )

            group_detr = self.group_detr if self.training else 1
            topk = self.num_queries
            topk_coords_logits = []
            topk_coords_logits_undetach = []
            object_query_undetach = []

            for group_id in range(group_detr):
                group_object_query = self.enc_output[group_id](object_query_embedding)
                group_object_query = self.enc_output_norm[group_id](group_object_query)

                group_enc_outputs_class = self.enc_out_class_embed[group_id](group_object_query)
                group_delta_bbox = self.enc_out_bbox_embed[group_id](group_object_query)
                group_enc_outputs_coord = refine_bboxes(output_proposals, group_delta_bbox, self.config.bbox_reparam)

                group_topk_proposals = torch.topk(group_enc_outputs_class.max(-1)[0], topk, dim=1)[1]
                group_topk_coords_logits_undetach = torch.gather(
                    group_enc_outputs_coord,
                    1,
                    group_topk_proposals.unsqueeze(-1).repeat(1, 1, 4),
                )
                group_topk_coords_logits = group_topk_coords_logits_undetach.detach()
                group_object_query_undetach = torch.gather(
                    group_object_query, 1, group_topk_proposals.unsqueeze(-1).repeat(1, 1, self.config.d_model)
                )

                topk_coords_logits.append(group_topk_coords_logits)
                topk_coords_logits_undetach.append(group_topk_coords_logits_undetach)
                object_query_undetach.append(group_object_query_undetach)

            topk_coords_logits = torch.cat(topk_coords_logits, 1)
            topk_coords_logits_undetach = torch.cat(topk_coords_logits_undetach, 1)
            object_query_undetach = torch.cat(object_query_undetach, 1)

            if not self.config.bbox_reparam:
                topk_coords_logits = topk_coords_logits.sigmoid()
            enc_outputs_class = object_query_undetach
            enc_outputs_coord_logits = topk_coords_logits

            reference_points = refine_bboxes(topk_coords_logits_undetach, reference_points, self.config.bbox_reparam)

        init_reference_points = reference_points
        decoder_outputs = self.decoder(
            inputs_embeds=target,
            query_position_embeddings=lvl_pos_embed_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            encoder_hidden_states=source_flatten,
            encoder_attention_mask=mask_flatten,
            **kwargs,
        )

        return LwDetrModelOutput(
            init_reference_points=init_reference_points,
            last_hidden_state=decoder_outputs.last_hidden_state,
            intermediate_hidden_states=decoder_outputs.intermediate_hidden_states,
            intermediate_reference_points=decoder_outputs.intermediate_reference_points,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            # TODO encoder_last_hidden_state, encoder_hidden_states, encoder_attentions
            enc_outputs_class=enc_outputs_class,
            enc_outputs_coord_logits=enc_outputs_coord_logits,
        )


class LwDetrMLPPredictionHead(DeformableDetrMLPPredictionHead):
    pass


class LwDetrObjectDetectionOutput(DeformableDetrObjectDetectionOutput):
    pass


class LwDetrForObjectDetection(DeformableDetrForObjectDetection):
    _tied_weights_keys = [r"bbox_embed.[1-9]*", r"class_embed.[1-9]*"]

    def __init__(self, config: LwDetrConfig):
        LwDetrPreTrainedModel.__init__(config)
        self.model = LwDetrModel(config)
        self.class_embed = nn.Linear(config.d_model, config.num_labels)
        self.bbox_embed = LwDetrMLPPredictionHead(config.d_model, config.d_model, 4, num_layers=3)

        if config.two_stage:
            self.model.enc_out_bbox_embed = _get_clones(self.bbox_embed, config.group_detr)
            self.model.enc_out_class_embed = _get_clones(self.class_embed, config.group_detr)

        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[list[dict]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple[torch.FloatTensor], LwDetrObjectDetectionOutput]:
        outputs = self.model(
            pixel_values,
            pixel_mask=pixel_mask,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
        )

        hidden_states = outputs.intermediate_hidden_states
        reference_points = outputs.intermediate_reference_points
        enc_outputs_class_logits = outputs.enc_outputs_class
        enc_outputs_boxes_logits = outputs.enc_outputs_coord_logits

        outputs_coord_delta = self.bbox_embed(hidden_states)
        outputs_coord = refine_bboxes(reference_points, outputs_coord_delta, self.config.bbox_reparam)
        if not self.config.bbox_reparam:
            outputs_coord = outputs_coord.sigmoid()

        outputs_class = self.class_embed(hidden_states)

        logits = outputs_class[-1]
        pred_boxes = outputs_coord[-1]

        if self.config.two_stage:
            enc_outputs_class_logits_list = enc_outputs_class_logits.split(self.config.num_queries, dim=1)
            pred_class = []
            group_detr = self.config.group_detr if self.training else 1
            for group_index in range(group_detr):
                group_pred_class = self.model.enc_out_class_embed[group_index](
                    enc_outputs_class_logits_list[group_index]
                )
                pred_class.append(group_pred_class)
            enc_outputs_class_logits = torch.cat(pred_class)

        loss, loss_dict, auxiliary_outputs = None, None, None
        if labels is not None:
            loss, loss_dict, auxiliary_outputs = self.loss_function(
                logits,
                labels,
                self.device,
                pred_boxes,
                self.config,
                outputs_class,
                outputs_coord,
                enc_outputs_class_logits,
                enc_outputs_boxes_logits,
            )

        dict_outputs = LwDetrObjectDetectionOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes,
            auxiliary_outputs=auxiliary_outputs,
            last_hidden_state=outputs.last_hidden_state,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            intermediate_hidden_states=outputs.intermediate_hidden_states,
            intermediate_reference_points=outputs.intermediate_reference_points,
            init_reference_points=outputs.init_reference_points,
            enc_outputs_class=enc_outputs_class_logits,
            enc_outputs_coord_logits=enc_outputs_boxes_logits,
        )

        return dict_outputs


__all__ = [
    "LwDetrConfig",
    "LwDetrPreTrainedModel",
    "LwDetrModel",
    "LwDetrForObjectDetection",
    "LwDetrImageProcessorFast",
    "LwDetrImageProcessor",
]
