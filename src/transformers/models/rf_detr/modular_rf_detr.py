from typing import Optional

from torch import nn

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ...utils.backbone_utils import verify_backbone_config_arguments
from ..auto import CONFIG_MAPPING
from ..lw_detr.modeling_lw_detr import (
    LwDetrConvNormLayer,
    LwDetrCSPRepLayer,
    LwDetrForObjectDetection,
    LwDetrLayerNorm,
    LwDetrModel,
    LwDetrSamplingLayer,
    LwDetrScaleProjector,
)
from .modeling_rf_detr_dinov2_with_registers import RfDetrDinov2WithRegistersConfig


logger = logging.get_logger(__name__)


class RfDetrConfig(PretrainedConfig):
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
            The configuration of the backbone model. If not provided, will default to `RfDetrDinov2WithRegistersConfig`
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
        activation_function (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function in the projector. Supported values are `"silu"`, `"relu"`, `"gelu"`.
        batch_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon value for batch normalization layers.
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
        position_embedding_type (`str`, *optional*, defaults to `"sine"`):
            Type of position embeddings to be used on top of the image features. One of `"sine"` or `"learned"`.
        two_stage (`bool`, *optional*, defaults to `True`):
            Whether to apply a two-stage detection approach, where region proposals are generated first
            and then refined by the decoder.
        group_detr (`int`, *optional*, defaults to 13):
            Number of groups for Group DETR attention mechanism, which helps reduce computational complexity.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        disable_custom_kernels (`bool`, *optional*, defaults to `True`):
            Disable the use of custom CUDA and CPU kernels. This option is necessary for the ONNX export, as custom
            kernels are not supported by PyTorch ONNX export.
        bbox_reparam (`bool`, *optional*, defaults to `True`):
            Whether to use bounding box reparameterization for better training stability.
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
        disable_custom_kernels=True,
        bbox_reparam=True,
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
        super().__init__(**kwargs)
        self.batch_norm_eps = batch_norm_eps

        # backbone
        if backbone_config is None and backbone is None:
            logger.info(
                "`backbone_config` and `backbone` are `None`. Initializing the config with the default `RfDetrDinov2WithRegisters` backbone."
            )
            backbone_config = RfDetrDinov2WithRegistersConfig(
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
        self.position_embedding_type = position_embedding_type
        self.two_stage = two_stage
        self.init_std = init_std
        self.bbox_reparam = bbox_reparam
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
        self.norm = RfDetrLayerNorm(out_channels, data_format="channels_first")


class RfDetrCSPRepLayer(LwDetrCSPRepLayer):
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
        self.projector_layer = RfDetrCSPRepLayer(config, projector_input_dim, output_dim)
        self.layer_norm = RfDetrLayerNorm(output_dim, data_format="channels_first")


class RfDetrModel(LwDetrModel):
    pass


class RfDetrForObjectDetection(LwDetrForObjectDetection):
    pass


__all__ = ["RfDetrConfig", "RfDetrModel", "RfDetrForObjectDetection"]
