# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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

import torch
from torch import nn

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ...utils.backbone_utils import verify_backbone_config_arguments
from ..auto import CONFIG_MAPPING
from ..auto.modeling_auto import AutoModel
from ..grounding_dino.configuration_grounding_dino import GroundingDinoConfig
from ..grounding_dino.modeling_grounding_dino import (
    GroundingDinoContrastiveEmbedding,
    GroundingDinoConvEncoder,
    GroundingDinoConvModel,
    GroundingDinoDecoder,
    GroundingDinoEncoder,
    GroundingDinoForObjectDetection,
    GroundingDinoMLPPredictionHead,
    GroundingDinoModel,
    GroundingDinoPreTrainedModel,
    build_position_encoding,
)


logger = logging.get_logger(__name__)


class MMGroundingDinoConfig(GroundingDinoConfig, PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MMGroundingDinoModel`]. It is used to instantiate a
    MM Grounding DINO model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the MM Grounding DINO tiny architecture
    [openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det](https://huggingface.co/openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        backbone_config (`PretrainedConfig` or `dict`, *optional*, defaults to `ResNetConfig()`):
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
        backbone_kwargs (`dict`, *optional*):
            Keyword arguments to be passed to AutoBackbone when loading from a checkpoint
            e.g. `{'out_indices': (0, 1, 2, 3)}`. Cannot be specified if `backbone_config` is set.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `BertConfig`):
            The config object or dictionary of the text backbone.
        num_queries (`int`, *optional*, defaults to 900):
            Number of object queries, i.e. detection slots. This is the maximal number of objects
            [`MMGroundingDinoModel`] can detect in a single image.
        encoder_layers (`int`, *optional*, defaults to 6):
            Number of encoder layers.
        encoder_ffn_dim (`int`, *optional*, defaults to 2048):
            Dimension of the "intermediate" (often named feed-forward) layer in decoder.
        encoder_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_layers (`int`, *optional*, defaults to 6):
            Number of decoder layers.
        decoder_ffn_dim (`int`, *optional*, defaults to 2048):
            Dimension of the "intermediate" (often named feed-forward) layer in decoder.
        decoder_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer decoder.
        is_encoder_decoder (`bool`, *optional*, defaults to `True`):
            Whether the model is used as an encoder/decoder or not.
        activation_function (`str` or `function`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        d_model (`int`, *optional*, defaults to 256):
            Dimension of the layers.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        auxiliary_loss (`bool`, *optional*, defaults to `False`):
            Whether auxiliary decoding losses (loss at each decoder layer) are to be used.
        position_embedding_type (`str`, *optional*, defaults to `"sine"`):
            Type of position embeddings to be used on top of the image features. One of `"sine"` or `"learned"`.
        num_feature_levels (`int`, *optional*, defaults to 4):
            The number of input feature levels.
        encoder_n_points (`int`, *optional*, defaults to 4):
            The number of sampled keys in each feature level for each attention head in the encoder.
        decoder_n_points (`int`, *optional*, defaults to 4):
            The number of sampled keys in each feature level for each attention head in the decoder.
        two_stage (`bool`, *optional*, defaults to `True`):
            Whether to apply a two-stage deformable DETR, where the region proposals are also generated by a variant of
            Grounding DINO, which are further fed into the decoder for iterative bounding box refinement.
        class_cost (`float`, *optional*, defaults to 1.0):
            Relative weight of the classification error in the Hungarian matching cost.
        bbox_cost (`float`, *optional*, defaults to 5.0):
            Relative weight of the L1 error of the bounding box coordinates in the Hungarian matching cost.
        giou_cost (`float`, *optional*, defaults to 2.0):
            Relative weight of the generalized IoU loss of the bounding box in the Hungarian matching cost.
        bbox_loss_coefficient (`float`, *optional*, defaults to 5.0):
            Relative weight of the L1 bounding box loss in the object detection loss.
        giou_loss_coefficient (`float`, *optional*, defaults to 2.0):
            Relative weight of the generalized IoU loss in the object detection loss.
        focal_alpha (`float`, *optional*, defaults to 0.25):
            Alpha parameter in the focal loss.
        disable_custom_kernels (`bool`, *optional*, defaults to `False`):
            Disable the use of custom CUDA and CPU kernels. This option is necessary for the ONNX export, as custom
            kernels are not supported by PyTorch ONNX export.
        max_text_len (`int`, *optional*, defaults to 256):
            The maximum length of the text input.
        text_enhancer_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the text enhancer.
        fusion_droppath (`float`, *optional*, defaults to 0.1):
            The droppath ratio for the fusion module.
        fusion_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the fusion module.
        embedding_init_target (`bool`, *optional*, defaults to `True`):
            Whether to initialize the target with Embedding weights.
        query_dim (`int`, *optional*, defaults to 4):
            The dimension of the query vector.
        positional_embedding_temperature (`float`, *optional*, defaults to 20):
            The temperature for Sine Positional Embedding that is used together with vision backbone.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.

    Examples:

    ```python
    >>> from transformers import MMGroundingDinoConfig, MMGroundingDinoModel

    >>> # Initializing a MM Grounding DINO configuration
    >>> configuration = MMGroundingDinoConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = MMGroundingDinoModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "mm-grounding-dino"

    def __init__(
        self,
        backbone_config=None,
        backbone=None,
        use_pretrained_backbone=False,
        use_timm_backbone=False,
        backbone_kwargs=None,
        text_config=None,
        num_queries=900,
        encoder_layers=6,
        encoder_ffn_dim=2048,
        encoder_attention_heads=8,
        decoder_layers=6,
        decoder_ffn_dim=2048,
        decoder_attention_heads=8,
        is_encoder_decoder=True,
        activation_function="relu",
        d_model=256,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        auxiliary_loss=False,
        position_embedding_type="sine",
        num_feature_levels=4,
        encoder_n_points=4,
        decoder_n_points=4,
        two_stage=True,
        class_cost=1.0,
        bbox_cost=5.0,
        giou_cost=2.0,
        bbox_loss_coefficient=5.0,
        giou_loss_coefficient=2.0,
        focal_alpha=0.25,
        disable_custom_kernels=False,
        # other parameters
        max_text_len=256,
        text_enhancer_dropout=0.0,
        fusion_droppath=0.1,
        fusion_dropout=0.0,
        embedding_init_target=True,
        query_dim=4,
        positional_embedding_temperature=20,
        init_std=0.02,
        layer_norm_eps=1e-5,
        **kwargs,
    ):
        PretrainedConfig.__init__(is_encoder_decoder=is_encoder_decoder, **kwargs)
        if backbone_config is None and backbone is None:
            logger.info("`backbone_config` is `None`. Initializing the config with the default `Swin` backbone.")
            backbone_config = CONFIG_MAPPING["swin"](
                window_size=7,
                image_size=224,
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
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

        if text_config is None:
            text_config = {}
            logger.info("text_config is None. Initializing the text config with default values (`BertConfig`).")

        self.backbone_config = backbone_config
        self.backbone = backbone
        self.use_pretrained_backbone = use_pretrained_backbone
        self.use_timm_backbone = use_timm_backbone
        self.backbone_kwargs = backbone_kwargs
        self.num_queries = num_queries
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.auxiliary_loss = auxiliary_loss
        self.position_embedding_type = position_embedding_type
        # deformable attributes
        self.num_feature_levels = num_feature_levels
        self.encoder_n_points = encoder_n_points
        self.decoder_n_points = decoder_n_points
        self.two_stage = two_stage
        # Hungarian matcher
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        # Loss coefficients
        self.bbox_loss_coefficient = bbox_loss_coefficient
        self.giou_loss_coefficient = giou_loss_coefficient
        self.focal_alpha = focal_alpha
        self.disable_custom_kernels = disable_custom_kernels
        # Text backbone
        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "bert"
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["bert"]()

        self.text_config = text_config
        self.max_text_len = max_text_len

        # Text Enhancer
        self.text_enhancer_dropout = text_enhancer_dropout
        # Fusion
        self.fusion_droppath = fusion_droppath
        self.fusion_dropout = fusion_dropout
        # Others
        self.embedding_init_target = embedding_init_target
        self.query_dim = query_dim
        self.positional_embedding_temperature = positional_embedding_temperature
        self.init_std = init_std
        self.layer_norm_eps = layer_norm_eps


class MMGroundingDinoContrastiveEmbedding(GroundingDinoContrastiveEmbedding):
    def __init__(self, config):
        super().__init__(config)
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        vision_hidden_state: torch.FloatTensor,
        text_hidden_state: torch.FloatTensor,
        text_token_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        res = vision_hidden_state @ text_hidden_state.transpose(-1, -2)
        res = res / math.sqrt(vision_hidden_state.shape[-1])
        res = res + self.bias
        res.masked_fill_(~text_token_mask[:, None, :], float("-inf"))

        # padding to max_text_len
        new_res = torch.full((*res.shape[:-1], self.max_text_len), float("-inf"), device=res.device)
        new_res[..., : res.shape[-1]] = res

        return new_res


class MMGroundingDinoPreTrainedModel(GroundingDinoPreTrainedModel):
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, MMGroundingDinoContrastiveEmbedding):
            nn.init.constant_(module.bias, -math.log((1 - 0.01) / 0.01))


class MMGroundingDinoConvEncoder(GroundingDinoConvEncoder):
    pass


class MMGroundingDinoConvModel(GroundingDinoConvModel):
    pass


class MMGroundingDinoEncoder(GroundingDinoEncoder):
    pass


class MMGroundingDinoDecoder(GroundingDinoDecoder):
    pass


class MMGroundingDinoModel(GroundingDinoModel, MMGroundingDinoPreTrainedModel):
    def __init__(self, config: MMGroundingDinoConfig):
        MMGroundingDinoPreTrainedModel.__init__(config)

        # Create backbone + positional encoding
        backbone = MMGroundingDinoConvEncoder(config)
        position_embeddings = build_position_encoding(config)
        self.backbone = MMGroundingDinoConvModel(backbone, position_embeddings)

        # Create input projection layers
        num_backbone_outs = len(backbone.intermediate_channel_sizes)
        input_proj_list = []
        for i in range(num_backbone_outs):
            in_channels = backbone.intermediate_channel_sizes[i]
            input_proj_list.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, config.d_model, kernel_size=1),
                    nn.GroupNorm(32, config.d_model),
                )
            )
        for _ in range(config.num_feature_levels - num_backbone_outs):
            input_proj_list.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, config.d_model, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, config.d_model),
                )
            )
            in_channels = config.d_model
        self.input_proj_vision = nn.ModuleList(input_proj_list)

        # Create text backbone
        self.text_backbone = AutoModel.from_config(config.text_config, add_pooling_layer=False)
        self.text_projection = nn.Linear(config.text_config.hidden_size, config.d_model)

        if config.embedding_init_target or not config.two_stage:
            self.query_position_embeddings = nn.Embedding(config.num_queries, config.d_model)

        self.encoder = MMGroundingDinoEncoder(config)
        self.decoder = MMGroundingDinoDecoder(config)

        self.level_embed = nn.Parameter(torch.Tensor(config.num_feature_levels, config.d_model))

        self.enc_output = nn.Linear(config.d_model, config.d_model)
        self.enc_output_norm = nn.LayerNorm(config.d_model, config.layer_norm_eps)
        self.encoder_output_bbox_embed = MMGroundingDinoMLPPredictionHead(
            input_dim=config.d_model, hidden_dim=config.d_model, output_dim=4, num_layers=3
        )
        self.encoder_output_class_embed = MMGroundingDinoContrastiveEmbedding(config)

        self.post_init()


class MMGroundingDinoMLPPredictionHead(GroundingDinoMLPPredictionHead):
    pass


class MMGroundingDinoForObjectDetection(GroundingDinoForObjectDetection, MMGroundingDinoPreTrainedModel):
    _tied_weights_keys = [
        r"bbox_embed\.[1-9]\d*",
        r"model\.decoder\.bbox_embed\.[0-9]\d*",
        r"class_embed\.[1-9]\d*",
        r"model\.decoder\.class_embed\.[0-9]\d*",
    ]

    def __init__(self, config: MMGroundingDinoConfig):
        MMGroundingDinoPreTrainedModel.__init__(config)

        self.model = MMGroundingDinoModel(config)

        self.class_embed = nn.ModuleList(
            [MMGroundingDinoContrastiveEmbedding(config) for _ in range(config.decoder_layers)]
        )

        self.bbox_embed = nn.ModuleList(
            [
                MMGroundingDinoMLPPredictionHead(
                    input_dim=config.d_model, hidden_dim=config.d_model, output_dim=4, num_layers=3
                )
                for _ in range(config.decoder_layers)
            ]
        )

        # hack for box-refinement
        self.model.decoder.bbox_embed = self.bbox_embed
        # hack implementation for two-stage
        self.model.decoder.class_embed = self.class_embed

        # Initialize weights and apply final processing
        self.post_init()


__all__ = [
    "MMGroundingDinoConfig",
    "MMGroundingDinoForObjectDetection",
    "MMGroundingDinoModel",
    "MMGroundingDinoPreTrainedModel",
]
