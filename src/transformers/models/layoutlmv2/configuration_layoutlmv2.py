# coding=utf-8
from transformers.models.layoutlm.configuration_layoutlm import LayoutLMConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

LAYOUTLMV2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "layoutlmv2-base-uncased": "https://huggingface.co/microsoft/layoutlmv2-base-uncased/resolve/main/config.json",
    "layoutlmv2-large-uncased": "https://huggingface.co/microsoft/layoutlmv2-large-uncased/resolve/main/config.json",
}


class LayoutLMv2Config(LayoutLMConfig):
    model_type = "layoutlmv2"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        gradient_checkpointing=False,
        max_2d_position_embeddings=1024,
        max_rel_pos=128,
        rel_pos_bins=32,
        fast_qkv=False,
        max_rel_2d_pos=256,
        rel_2d_pos_bins=64,
        convert_sync_batchnorm=True,
        image_feature_pool_shape=[7, 7, 256],
        coordinate_size=128,
        shape_size=128,
        has_relative_attention_bias=False,
        has_spatial_attention_bias=False,
        has_visual_segment_embedding=False,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            gradient_checkpointing=gradient_checkpointing,
            **kwargs,
        )
        self.max_2d_position_embeddings = max_2d_position_embeddings
        self.max_rel_pos = max_rel_pos
        self.rel_pos_bins = rel_pos_bins
        self.fast_qkv = fast_qkv
        self.max_rel_2d_pos = max_rel_2d_pos
        self.rel_2d_pos_bins = rel_2d_pos_bins
        self.convert_sync_batchnorm = convert_sync_batchnorm
        self.image_feature_pool_shape = image_feature_pool_shape
        self.coordinate_size = coordinate_size
        self.shape_size = shape_size
        self.has_relative_attention_bias = has_relative_attention_bias
        self.has_spatial_attention_bias = has_spatial_attention_bias
        self.has_visual_segment_embedding = has_visual_segment_embedding