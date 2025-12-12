from typing import Optional, Union

import torch
from torch import nn

from ...configuration_utils import PreTrainedConfig
from ...modeling_outputs import BackboneOutput, BaseModelOutput
from ...processing_utils import Unpack
from ...utils import logging, torch_int
from ...utils.backbone_utils import verify_backbone_config_arguments
from ...utils.generic import TransformersKwargs
from ..auto import CONFIG_MAPPING
from ..dinov2.configuration_dinov2 import Dinov2Config
from ..dinov2.modeling_dinov2 import (
    Dinov2Backbone,
    Dinov2Embeddings,
    Dinov2Encoder,
    Dinov2Layer,
    Dinov2PreTrainedModel,
    Dinov2SelfAttention,
)
from ..lw_detr.configuration_lw_detr import LwDetrConfig
from ..lw_detr.modeling_lw_detr import (
    LwDetrC2FLayer,
    LwDetrConvEncoder,
    LwDetrConvNormLayer,
    LwDetrForObjectDetection,
    LwDetrLayerNorm,
    LwDetrModel,
    LwDetrModelOutput,
    LwDetrPreTrainedModel,
    LwDetrSamplingLayer,
    LwDetrScaleProjector,
    refine_bboxes,
)


logger = logging.get_logger(__name__)


class RfDetrModelOutput(LwDetrModelOutput):
    pass


class RfDetrConfig(LwDetrConfig):
    r"""
    This is the configuration class to store the configuration of a [`RfDetrModel`]. It is used to instantiate
    a LW-DETR model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the LW-DETR
    [stevenbucaille/RfDetr_small_60e_coco](https://huggingface.co/stevenbucaille/RfDetr_small_60e_coco) architecture.

    LW-DETR (Lightweight Detection Transformer) is a transformer-based object detection model designed for real-time
    detection tasks. It replaces traditional CNN-based detectors like YOLO with a more efficient transformer architecture
    that achieves competitive performance while being computationally lightweight.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        backbone_config (`PretrainedConfig` or `dict`, *optional*):
            The configuration of the backbone model. If not provided, will default to `RfDetrDinov2Config`
            with a small ViT architecture optimized for detection tasks.
        backbone (`str`, *optional*):
            Name of backbone to use when `backbone_config` is `None`. Only used when `use_timm_backbone` is `True`.
        use_pretrained_backbone (`bool`, *optional*, defaults to `False`):
            Whether to use pretrained weights for the backbone.
        use_timm_backbone (`bool`, *optional*, defaults to `False`):
            Whether to use the `timm` library for the backbone. If set to `False`, will use the [`AutoBackbone`] API.
        backbone_kwargs (`dict`, *optional*):
            Keyword arguments to be passed to AutoBackbone when loading from a checkpoint.
        projector_scale_factors (`list[float]`, *optional*, defaults to `[]`):
            Scale factors for the feature pyramid network. Each scale factor determines the resolution of features
            at different levels. Supported values are 0.5, 1.0, and 2.0.
        hidden_expansion (`float`, *optional*, defaults to 0.5):
            Expansion factor for hidden dimensions in the projector layers.
        c2f_num_blocks (`int`, *optional*, defaults to 3):
            Number of blocks in the C2F layer.
        activation_function (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function in the projector. Supported values are `"silu"`, `"relu"`, `"gelu"`.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon value for layer normalization layers.
        d_model (`int`, *optional*, defaults to 256):
            Dimension of the model layers and the number of expected features in the decoder inputs.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        decoder_ffn_dim (`int`, *optional*, defaults to 2048):
            Dimension of the "intermediate" (often named feed-forward) layer in decoder.
        decoder_n_points (`int`, *optional*, defaults to 4):
            The number of sampled keys in each feature level for each attention head in the decoder.
        decoder_layers (`int`, *optional*, defaults to 3):
            Number of decoder layers in the transformer.
        decoder_self_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the decoder self-attention.
        decoder_cross_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the decoder cross-attention.
        decoder_activation_function (`str`, *optional*, defaults to `"relu"`):
            The non-linear activation function in the decoder. Supported values are `"relu"`, `"silu"`, `"gelu"`.
        num_queries (`int`, *optional*, defaults to 300):
            Number of object queries, i.e. detection slots. This is the maximal number of objects
            [`RfDetrModel`] can detect in a single image.
        attention_bias (`bool`, *optional*, defaults to `True`):
            Whether to add bias to the attention layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        group_detr (`int`, *optional*, defaults to 13):
            Number of groups for Group DETR attention mechanism, which helps reduce computational complexity.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        disable_custom_kernels (`bool`, *optional*, defaults to `True`):
            Disable the use of custom CUDA and CPU kernels. This option is necessary for the ONNX export, as custom
            kernels are not supported by PyTorch ONNX export.
        class_cost (`float`, *optional*, defaults to 2):
            Relative weight of the classification error in the Hungarian matching cost.
        bbox_cost (`float`, *optional*, defaults to 5):
            Relative weight of the L1 error of the bounding box coordinates in the Hungarian matching cost.
        giou_cost (`float`, *optional*, defaults to 2):
            Relative weight of the generalized IoU loss of the bounding box in the Hungarian matching cost.
        mask_loss_coefficient (`float`, *optional*, defaults to 1):
            Relative weight of the Focal loss in the panoptic segmentation loss.
        dice_loss_coefficient (`float`, *optional*, defaults to 1):
            Relative weight of the DICE/F-1 loss in the panoptic segmentation loss.
        bbox_loss_coefficient (`float`, *optional*, defaults to 5):
            Relative weight of the L1 bounding box loss in the object detection loss.
        giou_loss_coefficient (`float`, *optional*, defaults to 2):
            Relative weight of the generalized IoU loss in the object detection loss.
        eos_coefficient (`float`, *optional*, defaults to 0.1):
            Relative classification weight of the 'no-object' class in the object detection loss.
        focal_alpha (`float`, *optional*, defaults to 0.25):
            Alpha parameter in the focal loss.
        auxiliary_loss (`bool`, *optional*, defaults to `True`):
            Whether auxiliary decoding losses (loss at each decoder layer) are to be used.

    Examples:

    ```python
    >>> from transformers import RfDetrConfig, RfDetrModel

    >>> # Initializing a LW-DETR stevenbucaille/RfDetr_small_60e_coco style configuration
    >>> configuration = RfDetrConfig()

    >>> # Initializing a model (with random weights) from the stevenbucaille/RfDetr_small_60e_coco style configuration
    >>> model = RfDetrModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "rf_detr"

    def __init__(
        self,
        # backbone
        backbone_config=None,
        backbone=None,
        use_pretrained_backbone=False,
        use_timm_backbone=False,
        backbone_kwargs=None,
        # projector
        projector_scale_factors: list[float] = [],
        hidden_expansion=0.5,
        c2f_num_blocks=3,
        activation_function="silu",
        layer_norm_eps=1e-5,
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
        group_detr: int = 13,
        init_std=0.02,
        disable_custom_kernels=True,
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
        self.layer_norm_eps = layer_norm_eps

        # backbone
        if backbone_config is None and backbone is None:
            logger.info(
                "`backbone_config` and `backbone` are `None`. Initializing the config with the default `RfDetrDinov2` backbone."
            )
            backbone_config = RfDetrDinov2Config(
                attention_probs_dropout_prob=0.0,
                drop_path_rate=0.0,
                hidden_act="gelu",
                hidden_dropout_prob=0.0,
                initializer_range=0.02,
                layer_norm_eps=1e-06,
                layerscale_value=1.0,
                mlp_ratio=4,
                num_attention_heads=6,
                num_channels=3,
                num_hidden_layers=12,
                qkv_bias=True,
                use_swiglu_ffn=False,
                out_features=["stage2", "stage5", "stage8", "stage11"],
                hidden_size=384,
                patch_size=14,
                num_windows=4,
                num_register_tokens=0,
                image_size=518,
                **kwargs,
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
        self.c2f_num_blocks = c2f_num_blocks
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
        # model
        self.init_std = init_std
        self.group_detr = group_detr
        # Loss
        self.auxiliary_loss = auxiliary_loss
        # Hungarian matcher
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        # Loss coefficients
        self.dice_loss_coefficient = dice_loss_coefficient
        self.bbox_loss_coefficient = bbox_loss_coefficient
        self.giou_loss_coefficient = giou_loss_coefficient
        self.eos_coefficient = eos_coefficient
        self.focal_alpha = focal_alpha
        self.disable_custom_kernels = disable_custom_kernels
        PreTrainedConfig.__init__(self, **kwargs)


class RfDetrLayerNorm(LwDetrLayerNorm):
    pass


class RfDetrConvNormLayer(LwDetrConvNormLayer):
    def __init__(
        self,
        config: RfDetrConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        activation: Optional[str] = None,
    ):
        super().__init__(
            config,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            activation,
        )
        self.norm = RfDetrLayerNorm(out_channels, data_format="channels_first", eps=config.layer_norm_eps)


class RfDetrC2FLayer(LwDetrC2FLayer):
    pass


class RfDetrSamplingLayer(LwDetrSamplingLayer):
    def __init__(self, config: RfDetrConfig, channel_size: int, scale: float):
        nn.Module.__init__(self)

        self.scale = scale
        self.channel_size = channel_size

        layers = []
        if scale == 2.0:
            layers.append(nn.ConvTranspose2d(channel_size, channel_size // 2, 2, 2))
        elif scale == 0.5:
            layers.append(RfDetrConvNormLayer(config, channel_size, channel_size, 3, 2, activation="relu"))
        self.layers = nn.ModuleList(layers)


class RfDetrScaleProjector(LwDetrScaleProjector):
    def __init__(self, config: RfDetrConfig, intermediate_dims: list[int], scale: float, output_dim: int):
        nn.Module.__init__(self)

        sampling_layers = []
        for channel_size in intermediate_dims:
            sampling_layers.append(RfDetrSamplingLayer(config, channel_size, scale))
        self.sampling_layers = nn.ModuleList(sampling_layers)

        projector_input_dim = int(sum(intermediate_dim // max(1, scale) for intermediate_dim in intermediate_dims))
        self.projector_layer = RfDetrC2FLayer(config, projector_input_dim, output_dim)
        self.layer_norm = RfDetrLayerNorm(output_dim, data_format="channels_first")


class RfDetrPreTrainedModel(LwDetrPreTrainedModel):
    pass


class RfDetrConvEncoder(LwDetrConvEncoder):
    def __init__(self, config: RfDetrConfig):
        super().__init__(config)
        self.backbone = RfDetrDinov2Backbone(config.backbone_config)


class RfDetrModel(LwDetrModel):
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> RfDetrModelOutput:
        r"""
        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, DeformableDetrModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("stevenbucaille/RfDetr_small_60e_coco")
        >>> model = DeformableDetrModel.from_pretrained("stevenbucaille/RfDetr_small_60e_coco")

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
        features = self.backbone(pixel_values, pixel_mask)

        # Then, apply 1x1 convolution to reduce the channel dimension to d_model (256 by default)
        sources = []
        masks = []
        for level, (source, mask) in enumerate(features):
            sources.append(source)
            masks.append(mask)
            if mask is None:
                raise ValueError("No attention mask was provided")

        if self.training:
            reference_points = self.reference_point_embed.weight
            query_feat = self.query_feat.weight
        else:
            # only use one group in inference
            reference_points = self.reference_point_embed.weight[: self.num_queries]
            query_feat = self.query_feat.weight[: self.num_queries]

        # Prepare encoder inputs (by flattening)
        source_flatten = []
        mask_flatten = []
        spatial_shapes_list = []
        for source, mask in zip(sources, masks):
            batch_size, num_channels, height, width = source.shape
            spatial_shape = (height, width)
            spatial_shapes_list.append(spatial_shape)
            source = source.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            source_flatten.append(source)
            mask_flatten.append(mask)
        source_flatten = torch.cat(source_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes_list, dtype=torch.long, device=source_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m, dtype=source_flatten.dtype) for m in masks], 1)

        target = query_feat.unsqueeze(0).expand(batch_size, -1, -1)
        reference_points = reference_points.unsqueeze(0).expand(batch_size, -1, -1)

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
            group_enc_outputs_coord = refine_bboxes(output_proposals, group_delta_bbox)

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

        enc_outputs_class = object_query_undetach
        enc_outputs_coord_logits = topk_coords_logits

        two_stage_len = topk_coords_logits.shape[-2]
        reference_points_two_stage_subset = reference_points[..., :two_stage_len, :]
        reference_points_subset = reference_points[..., two_stage_len:, :]
        reference_points_two_stage_subset = refine_bboxes(topk_coords_logits, reference_points_two_stage_subset)
        reference_points = torch.cat([reference_points_two_stage_subset, reference_points_subset], dim=-2)
        init_reference_points = reference_points
        decoder_outputs = self.decoder(
            inputs_embeds=target,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            encoder_hidden_states=source_flatten,
            encoder_attention_mask=mask_flatten,
            **kwargs,
        )

        return RfDetrModelOutput(
            init_reference_points=init_reference_points,
            last_hidden_state=decoder_outputs.last_hidden_state,
            intermediate_hidden_states=decoder_outputs.intermediate_hidden_states,
            intermediate_reference_points=decoder_outputs.intermediate_reference_points,
            enc_outputs_class=enc_outputs_class,
            enc_outputs_coord_logits=enc_outputs_coord_logits,
        )


class RfDetrForObjectDetection(LwDetrForObjectDetection):
    pass


class RfDetrDinov2Config(Dinov2Config):
    r"""
    This is the configuration class to store the configuration of a [`RfDetrDinov2Model`]. It is used to instantiate an
    RfDetrDinov2 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the DINOv2
    [facebook/dinov2-base](https://huggingface.co/facebook/dinov2-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        mlp_ratio (`int`, *optional*, defaults to 4):
            Ratio of the hidden size of the MLPs relative to the `hidden_size`.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 14):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        layerscale_value (`float`, *optional*, defaults to 1.0):
           Initial value to use for layer scale.
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            Stochastic depth rate per sample (when applied in the main path of residual layers).
        use_swiglu_ffn (`bool`, *optional*, defaults to `False`):
            Whether to use the SwiGLU feedforward neural network.
        out_features (`list[str]`, *optional*):
            If used as backbone, list of features to output. Can be any of `"stem"`, `"stage1"`, `"stage2"`, etc.
            (depending on how many stages the model has). If unset and `out_indices` is set, will default to the
            corresponding stages. If unset and `out_indices` is unset, will default to the last stage. Must be in the
            same order as defined in the `stage_names` attribute.
        out_indices (`list[int]`, *optional*):
            If used as backbone, list of indices of features to output. Can be any of 0, 1, 2, etc. (depending on how
            many stages the model has). If unset and `out_features` is set, will default to the corresponding stages.
            If unset and `out_features` is unset, will default to the last stage. Must be in the
            same order as defined in the `stage_names` attribute.
        apply_layernorm (`bool`, *optional*, defaults to `True`):
            Whether to apply layer normalization to the feature maps in case the model is used as backbone.
        reshape_hidden_states (`bool`, *optional*, defaults to `True`):
            Whether to reshape the feature maps to 4D tensors of shape `(batch_size, hidden_size, height, width)` in
            case the model is used as backbone. If `False`, the feature maps will be 3D tensors of shape `(batch_size,
            seq_len, hidden_size)`.
        use_mask_token (`bool`, *optional*, defaults to `True`):
            Whether to use mask_token in embeddings.
        num_windows (`int`, *optional*, defaults to 4):
            Number of windows to use for windowed attention. If 1, no windowed attention is used.
    Example:

    ```python
    >>> from transformers import RfDetrDinov2Config, RfDetrDinov2Backbone

    >>> # Initializing a RfDetrDinov2 base style configuration
    >>> configuration = RfDetrDinov2Config()

    >>> # Initializing a model (with random weights) from the base style configuration
    >>> model = RfDetrDinov2Backbone(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "rf_detr_dinov2"

    def __init__(self, num_windows: int = 4, **super_kwargs):
        super().__init__(**super_kwargs)

        self.num_windows = num_windows
        window_block_indexes = set(range(self._out_indices[-1] + 1))
        window_block_indexes.difference_update(self._out_indices)
        window_block_indexes = list(window_block_indexes)
        self.window_block_indexes = window_block_indexes


def window_partition(
    embeddings: torch.Tensor, num_windows: int, patch_size: int, height: int, width: int
) -> torch.Tensor:
    batch_size = embeddings.shape[0]
    num_h_patches = height // patch_size
    num_w_patches = width // patch_size
    cls_token_with_pos_embed = embeddings[:, :1]
    pixel_tokens_with_pos_embed = embeddings[:, 1:]
    pixel_tokens_with_pos_embed = pixel_tokens_with_pos_embed.view(batch_size, num_h_patches, num_w_patches, -1)
    num_w_patches_per_window = num_w_patches // num_windows
    num_h_patches_per_window = num_h_patches // num_windows
    windowed_pixel_tokens = pixel_tokens_with_pos_embed.view(
        batch_size, num_windows, num_h_patches_per_window, num_windows, num_h_patches_per_window, -1
    )
    windowed_pixel_tokens = windowed_pixel_tokens.permute(0, 1, 3, 2, 4, 5)
    windowed_pixel_tokens = windowed_pixel_tokens.reshape(
        batch_size * num_windows**2, num_h_patches_per_window * num_w_patches_per_window, -1
    )
    windowed_cls_token_with_pos_embed = cls_token_with_pos_embed.repeat(num_windows**2, 1, 1)
    embeddings = torch.cat((windowed_cls_token_with_pos_embed, windowed_pixel_tokens), dim=1)
    return embeddings


class RfDetrDinov2Embeddings(Dinov2Embeddings):
    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing and interpolation at torch.float32 precision.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """

        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embeddings

        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = torch_int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        target_dtype = patch_pos_embed.dtype
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.to(torch.float32),
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).to(dtype=target_dtype)

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, _, height, width = pixel_values.shape
        target_dtype = self.patch_embeddings.projection.weight.dtype
        embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))

        if bool_masked_pos is not None:
            embeddings = torch.where(
                bool_masked_pos.unsqueeze(-1), self.mask_token.to(embeddings.dtype).unsqueeze(0), embeddings
            )

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)

        if self.config.num_windows > 1:
            # reshape for windows
            embeddings = window_partition(embeddings, self.config.num_windows, self.config.patch_size, height, width)
        embeddings = self.dropout(embeddings)

        return embeddings


def window_unpartition_before_attention(hidden_states: torch.Tensor, num_windows: int) -> torch.Tensor:
    batch_size, seq_len, channels = hidden_states.shape
    num_windows_squared = num_windows**2
    hidden_states = hidden_states.view(batch_size // num_windows_squared, num_windows_squared * seq_len, channels)
    return hidden_states


def window_partition_after_attention(
    hidden_states: torch.Tensor, self_attention_output: torch.Tensor, num_windows: int
) -> torch.Tensor:
    batch_size, seq_len, channels = hidden_states.shape
    num_windows_squared = num_windows**2
    self_attention_output = self_attention_output.view(
        batch_size * num_windows_squared, seq_len // num_windows_squared, channels
    )
    return self_attention_output


class RfDetrDinov2SelfAttention(Dinov2SelfAttention):
    def __init__(self, config: RfDetrDinov2Config):
        super().__init__(config)
        self.num_key_value_groups = 1


class RfDetrDinov2Layer(Dinov2Layer):
    def __init__(self, config: RfDetrDinov2Config, layer_idx: int):
        super().__init__(config)
        self.num_windows = config.num_windows
        self.global_attention = layer_idx not in config.window_block_indexes

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Union[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor]]:
        shortcut = hidden_states
        if self.global_attention:
            hidden_states = window_unpartition_before_attention(hidden_states, self.num_windows)

        hidden_states_norm = self.norm1(hidden_states)
        self_attention_output = self.attention(hidden_states_norm)

        if self.global_attention:
            self_attention_output = window_partition_after_attention(
                hidden_states, self_attention_output, self.num_windows
            )

        self_attention_output = self.layer_scale1(self_attention_output)

        # first residual connection
        hidden_states = self.drop_path(self_attention_output) + shortcut

        # in Dinov2, layernorm is also applied after self-attention
        layer_output = self.norm2(hidden_states)
        layer_output = self.mlp(layer_output)
        layer_output = self.layer_scale2(layer_output)

        # second residual connection
        layer_output = self.drop_path(layer_output) + hidden_states

        return layer_output


class RfDetrDinov2Encoder(Dinov2Encoder):
    def __init__(self, config: RfDetrDinov2Config):
        super().__init__(config)
        self.layer = nn.ModuleList([RfDetrDinov2Layer(config, i) for i in range(config.num_hidden_layers)])


class RfDetrDinov2PreTrainedModel(Dinov2PreTrainedModel):
    pass


def window_unpartition(
    hidden_state: torch.Tensor,
    num_windows: int,
    num_h_patches: int,
    num_w_patches: int,
) -> torch.Tensor:
    hidden_batch_size, seq_len, channels = hidden_state.shape
    num_windows_squared = num_windows**2
    num_h_patches_per_window = num_h_patches // num_windows
    num_w_patches_per_window = num_w_patches // num_windows
    hidden_state = hidden_state.reshape(
        hidden_batch_size // num_windows_squared, num_windows_squared * seq_len, channels
    )
    hidden_state = hidden_state.view(
        hidden_batch_size // num_windows_squared,
        num_windows,
        num_windows,
        num_h_patches_per_window,
        num_w_patches_per_window,
        channels,
    )
    hidden_state = hidden_state.permute(0, 1, 3, 2, 4, 5)
    return hidden_state


class RfDetrDinov2Backbone(Dinov2Backbone):
    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> BackboneOutput:
        r"""
        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AutoBackbone
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        >>> model = AutoBackbone.from_pretrained(
        ...     "facebook/dinov2-base", out_features=["stage2", "stage5", "stage8", "stage11"]
        ... )

        >>> inputs = processor(image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> feature_maps = outputs.feature_maps
        >>> list(feature_maps[-1].shape)
        [1, 768, 16, 16]
        ```"""
        if output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states

        embedding_output = self.embeddings(pixel_values)

        output: BaseModelOutput = self.encoder(embedding_output, output_hidden_states=True)
        hidden_states = output.hidden_states

        feature_maps = ()
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                if self.config.apply_layernorm:
                    hidden_state = self.layernorm(hidden_state)
                if self.config.reshape_hidden_states:
                    hidden_state = hidden_state[:, 1:]
                    # this was actually a bug in the original implementation that we copied here,
                    # cause normally the order is height, width
                    batch_size, _, height, width = pixel_values.shape
                    patch_size = self.config.patch_size

                    num_h_patches = height // patch_size
                    num_w_patches = width // patch_size
                    hidden_batch_size, seq_len, channels = hidden_state.shape

                    if self.config.num_windows > 1:
                        hidden_state = window_unpartition(
                            hidden_state, self.config.num_windows, num_h_patches, num_w_patches
                        )

                    hidden_state = hidden_state.reshape(batch_size, num_h_patches, num_w_patches, -1)
                    hidden_state = hidden_state.permute(0, 3, 1, 2).contiguous()

                feature_maps += (hidden_state,)

        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=hidden_states if output_hidden_states else None,
        )


__all__ = [
    "RfDetrConfig",
    "RfDetrModel",
    "RfDetrForObjectDetection",
    "RfDetrPreTrainedModel",
    "RfDetrDinov2Config",
    "RfDetrDinov2Backbone",
    "RfDetrDinov2PreTrainedModel",
]
