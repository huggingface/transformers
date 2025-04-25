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

    def forward(self, vision_hidden_state, text_hidden_state, text_token_mask):
        """Forward function.

        Args:
            visual_feat (Tensor): Visual features.
            text_feat (Tensor): Text features.
            text_token_mask (Tensor): A mask used for text feats.

        Returns:
            Tensor: Classification score.
        """
        y = text_hidden_state
        text_token_mask = text_token_mask
        res = vision_hidden_state @ y.transpose(-1, -2)
        res = res / math.sqrt(vision_hidden_state.shape[-1])
        res = res + self.bias
        res.masked_fill_(~text_token_mask[:, None, :], float("-inf"))

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
        r"bbox_embed\.[1-9]\d*", r"model\.decoder\.bbox_embed\.[0-9]\d*",
        r"class_embed\.[1-9]\d*", r"model\.decoder\.class_embed\.[0-9]\d*",
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
