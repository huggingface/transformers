import math

import torch
from torch import nn

from ..grounding_dino.configuration_grounding_dino import GroundingDinoConfig
from ..grounding_dino.modeling_grounding_dino import (
    GroundingDinoContrastiveEmbedding,
    GroundingDinoDecoder,
    GroundingDinoForObjectDetection,
    GroundingDinoMLPPredictionHead,
    GroundingDinoModel,
    GroundingDinoPreTrainedModel,
)


# --- config --- #


class MMGroundingDinoConfig(GroundingDinoConfig):
    r"""
    This is the configuration class to store the configuration of a [`MMGroundingDinoModel`]. It is used to instantiate a
    MM Grounding DINO model according to the specified arguments, defining the model architecture.

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
        decoder_bbox_embed_share (`bool`, *optional*, defaults to `False`):
            Whether to share the bbox regression head for all decoder layers.
        decoder_cls_embed_share (`bool`, *optional*, defaults to `False`):
            Whether to share the class head for all decoder layers.
        two_stage_bbox_embed_share (`bool`, *optional*, defaults to `False`):
            Whether to share the bbox embedding between the two-stage bbox generator and the region proposal
            generation.
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
        decoder_bbox_embed_share=False,  # set this to false by default
        decoder_cls_embed_share=False,  # add this argument
        two_stage_bbox_embed_share=False,
        positional_embedding_temperature=20,
        init_std=0.02,
        layer_norm_eps=1e-5,
        **kwargs,
    ):
        super().__init__(
            backbone_config=backbone_config,
            backbone=backbone,
            use_pretrained_backbone=use_pretrained_backbone,
            use_timm_backbone=use_timm_backbone,
            backbone_kwargs=backbone_kwargs,
            text_config=text_config,
            num_queries=num_queries,
            encoder_layers=encoder_layers,
            encoder_ffn_dim=encoder_ffn_dim,
            encoder_attention_heads=encoder_attention_heads,
            decoder_layers=decoder_layers,
            decoder_ffn_dim=decoder_ffn_dim,
            decoder_attention_heads=decoder_attention_heads,
            is_encoder_decoder=is_encoder_decoder,
            activation_function=activation_function,
            d_model=d_model,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            auxiliary_loss=auxiliary_loss,
            position_embedding_type=position_embedding_type,
            num_feature_levels=num_feature_levels,
            encoder_n_points=encoder_n_points,
            decoder_n_points=decoder_n_points,
            two_stage=two_stage,
            class_cost=class_cost,
            bbox_cost=bbox_cost,
            giou_cost=giou_cost,
            bbox_loss_coefficient=bbox_loss_coefficient,
            giou_loss_coefficient=giou_loss_coefficient,
            focal_alpha=focal_alpha,
            disable_custom_kernels=disable_custom_kernels,
            # other parameters
            max_text_len=max_text_len,
            text_enhancer_dropout=text_enhancer_dropout,
            fusion_droppath=fusion_droppath,
            fusion_dropout=fusion_dropout,
            embedding_init_target=embedding_init_target,
            query_dim=query_dim,
            decoder_bbox_embed_share=decoder_bbox_embed_share,
            two_stage_bbox_embed_share=two_stage_bbox_embed_share,
            positional_embedding_temperature=positional_embedding_temperature,
            init_std=init_std,
            layer_norm_eps=layer_norm_eps,
            **kwargs,
        )
        self.decoder_cls_embed_share = decoder_cls_embed_share


# --- modeling --- #


class MMGroundingDinoContrastiveEmbedding(GroundingDinoContrastiveEmbedding):
    def __init__(self, config):
        super().__init__(config)
        self.bias = nn.Parameter(torch.tensor(0.0))
        nn.init.constant_(self.bias, -math.log((1 - 0.01) / 0.01))

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
    pass


# TODO: this one is useless, but without it class order in modeling gets messed up
class MMGroundingDinoDecoder(GroundingDinoDecoder):
    pass


class MMGroundingDinoModel(GroundingDinoModel):
    def __init__(self, config: MMGroundingDinoConfig):
        super().__init__(config)
        self.encoder_output_class_embed = MMGroundingDinoContrastiveEmbedding(config)


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

        if config.decoder_cls_embed_share:
            _class_embed = MMGroundingDinoContrastiveEmbedding(config)
            self.class_embed = nn.ModuleList([_class_embed for _ in range(config.decoder_layers)])
        else:
            module_list = []
            for _ in range(config.decoder_layers):
                _class_embed = MMGroundingDinoContrastiveEmbedding(config)
                module_list.append(_class_embed)
            self.class_embed = nn.ModuleList(module_list)

        if config.decoder_bbox_embed_share:
            _bbox_embed = MMGroundingDinoMLPPredictionHead(
                input_dim=config.d_model, hidden_dim=config.d_model, output_dim=4, num_layers=3
            )
            self.bbox_embed = nn.ModuleList([_bbox_embed for _ in range(config.decoder_layers)])
        else:
            module_list = []
            for _ in range(config.decoder_layers):
                _bbox_embed = MMGroundingDinoMLPPredictionHead(
                    input_dim=config.d_model, hidden_dim=config.d_model, output_dim=4, num_layers=3
                )
                module_list.append(_bbox_embed)
            self.bbox_embed = nn.ModuleList(module_list)

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
