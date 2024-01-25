# coding=utf-8
# Copyright 2023 Facebook AI Research and The HuggingFace Inc. team. All rights reserved.
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
""" RT-DETR model configuration"""


from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..timm_backbone import TimmBackboneConfig


logger = logging.get_logger(__name__)

RT_DETR_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "rafaelpadilla/porting_rt_detr": "https://huggingface.co/rafaelpadilla/porting_rt_detr/raw/main/config.json",
}


class RTDetrConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`RTDetrModel`]. It is used to instantiate a
    RT_DETR model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the RT_DETR
    [checkpoing/todo](https://huggingface.co/checkpoing/todo) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        batch_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the batch normalization layers.
        backbone_config (`Union[Dict[str, Any], PretrainedConfig]`, *optional*):
            The configuration of the backbone in a dictionary or the config object of the backbone.
        feat_strides (`List[int]`, *optional*, defaults to `[8, 16, 32]`):
            Strides used in each feature map.
        hidden_dim (`int`, *optional*, defaults to 256):
            Dimension for hidden states in transformer encoder and decoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the transformer encoder and decoder.
        dim_feedforward (`int`, *optional*, defaults to 1024):
            Dimension for feedforward network layer in transformer encoder and decoder.
        dropout (`float`, *optional*, defaults to 0.0):
            The ratio for all dropout layers.
        encode_proj_layers (`List[int]`, *optional*, defaults to `[2]`):
            Indexes of the projected layers to be used in the encoder.
        num_encoder_layers (`int`, *optional*, defaults to 1):
            Total of layers to be used by the encoder.
        pe_temperature (`int`, *optional*, defaults to 10000):
            The temperature parameter used to create the positional encodings.
        act_encoder (`str`, *optional*, defaults to `"silu"`):
            Activation function of the encoder used in the top-down Feature Pyramid Network and the bottom-up Path
            Aggregation Network.
        eval_size (`Tuple[int, int]`, *optional*):
            Height and width used to computes the effective height and width of the position embeddings after taking
            into account the stride.
        normalize_before (`bool`, *optional*, defaults to `False`):
            Determine whether to apply layer normalization in the transformer encoder layer before self-attention and
            feed-forward modules.
        num_queries (`int`, *optional*, defaults to 300):
            Number of object queries.
        feat_channels (`List[int]`, *optional*, defaults to `[256, 256, 256]`):
            A list of integers representing the number of feature channels at various layers or stages of the network
        num_levels (`int`, *optional*, defaults to 3):
            The number of feature levels used by the `RTDetrTransformers`.
        num_decoder_points (`int`, *optional*, defaults to 4):
            Number of points used by the `TransformerDecoderLayer`.
        num_decoder_layers (`int`, *optional*, defaults to 6):
            Number of layers of the decoder.
        num_denoising (`int`, *optional*, defaults to 100):
            The total number of denoising tasks or queries to be used for contrastive denoising.
        label_noise_ratio (`float`, *optional*, defaults to 0.5):
            The fraction of denoising labels to which random noise should be added.
        box_noise_scale (`float`, *optional*, defaults to 1.0):
            Scale or magnitude of noise to be added to the bounding boxes.
        learnt_init_query (`bool`, *optional*, defaults to `False`):
            Indicates whether the initial query embeddings for the decoder should be learned during training
        image_size (`Tuple[int, int]`, *optional*, defaults to `[640, 640]`):
            Height and width of the input image used during evaluation to generate the bounding box anchors.
        eval_idx (`int`, *optional*, defaults to -1):
            Id of the decoder layer used to obtain the logits and bounding boxes.
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
            Parameter informing if focal focal should be used.
        use_aux_loss (`bool`, *optional*, defaults to `True`):
            Parameter informing if auxiliary focal loss should be used.
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
        eos_coefficient (`float`, *optional*, defaults to 0.1):
            Relative classification weight of the 'no-object' class in the object detection loss.

    Examples:

    ```python
    >>> from transformers import RTDetrConfig, RTDetrModel

    >>> # Initializing a RT_DETR checkpoing/todo style configuration
    >>> configuration = RTDetrConfig()

    >>> # Initializing a model (with random weights) from the checkpoing/todo style configuration
    >>> model = RTDetrModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "rt_detr"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
    }

    def __init__(
        self,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        batch_norm_eps=1e-5,
        # backbone
        backbone_config=None,
        # encoder HybridEncoder
        feat_strides=[8, 16, 32],
        hidden_dim=256,
        num_attention_heads=8,
        dim_feedforward=1024,
        dropout=0.0,
        encode_proj_layers=[2],
        num_encoder_layers=1,
        pe_temperature=10000,
        act_encoder="silu",
        eval_size=None,
        normalize_before=False,
        # decoder RTDetrTransformer
        num_queries=300,
        feat_channels=[256, 256, 256],
        num_levels=3,
        num_decoder_points=4,
        num_decoder_layers=6,
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learnt_init_query=False,
        image_size=[640, 640],
        eval_idx=-1,
        # Loss
        matcher_alpha=0.25,
        matcher_gamma=2.0,
        matcher_class_cost=2.0,
        matcher_bbox_cost=5.0,
        matcher_giou_cost=2.0,
        use_focal_loss=True,
        use_aux_loss=True,
        focal_loss_alpha=0.75,
        focal_loss_gamma=2.0,
        weight_loss_vfl=1.0,
        weight_loss_bbox=5.0,
        weight_loss_giou=2.0,
        eos_coefficient=0.1,
        **kwargs,
    ):
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.batch_norm_eps = batch_norm_eps

        # backbone
        if backbone_config is None:
            logger.info("Initializing the config with a `TimmBackbone` backbone.")
            backbone_config = {
                "backbone": "resnet50d",
                "out_indices": [2, 3, 4],
                "freeze_batch_norm_2d": True,
            }
            self.backbone_config = TimmBackboneConfig(**backbone_config)
        elif isinstance(backbone_config, dict):
            logger.info("Initializing the config with a `TimmBackbone` backbone.")
            self.backbone_config = TimmBackboneConfig(**backbone_config)
        elif isinstance(backbone_config, PretrainedConfig):
            self.backbone_config = backbone_config
        else:
            raise ValueError(
                f"backbone_config must be a dictionary or a `PretrainedConfig`, got {backbone_config.__class__}."
            )

        # encoder
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.num_attention_heads = num_attention_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.encode_proj_layers = encode_proj_layers
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.act_encoder = act_encoder
        self.eval_size = eval_size
        self.normalize_before = normalize_before
        # decoder
        self.num_queries = num_queries
        self.feat_channels = feat_channels
        self.num_levels = num_levels
        self.num_decoder_points = num_decoder_points
        self.num_decoder_layers = num_decoder_layers
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        self.learnt_init_query = learnt_init_query
        self.image_size = image_size
        self.eval_idx = eval_idx
        self.use_aux_loss = use_aux_loss
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
        super().__init__(**kwargs)
