# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Dict, Optional, Sequence

from ...configuration_utils import PretrainedConfig


UDOP_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class UdopConfig(PretrainedConfig):
    pretrained_config_archive_map = UDOP_PRETRAINED_CONFIG_ARCHIVE_MAP

    attribute_map = {"hidden_size": "d_model", "num_attention_heads": "num_heads", "num_hidden_layers": "num_layers"}

    def __init__(
        self,
        vocab_size=32128,
        d_model=512,
        d_kv=64,
        d_ff=2048,
        num_layers=6,
        num_decoder_layers=None,
        num_heads=8,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        feed_forward_proj="relu",
        is_encoder_decoder=True,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        max_2d_position_embeddings=1024,
        max_bbox_length=1001,
        mae_version="mae_vit_large_patch16",
        mae_checkpoint="mae-models/mae_pretrain_vit_large_full.pth",
        image_size: int = 224,
        relative_bias_args: Optional[Sequence[Dict[str, Any]]] = [
            {"type": "1d"},
            {"type": "horizontal"},
            {"type": "vertical"},
        ],
        truncate_decoder_after_layer: Optional[int] = None,
        truncate_encoder_after_layer: Optional[int] = None,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_decoder_layers = (
            num_decoder_layers if num_decoder_layers is not None else self.num_layers
        )  # default = symmetry
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.feed_forward_proj = feed_forward_proj
        self.use_cache = use_cache
        self.max_2d_position_embeddings = max_2d_position_embeddings
        self.max_bbox_length = max_bbox_length
        self.relative_bias_args = [] if relative_bias_args is None else relative_bias_args
        self.image_size = image_size
        self.mae_version = mae_version
        self.mae_checkpoint = mae_checkpoint
        self.truncate_decoder_after_layer = truncate_decoder_after_layer
        self.truncate_encoder_after_layer = truncate_encoder_after_layer
        self.is_encoder_decoder = is_encoder_decoder
        self.pad_token_id = (pad_token_id,)
        self.eos_token_id = eos_token_id

        act_info = self.feed_forward_proj.split("-")
        self.dense_act_fn = act_info[-1]
        self.is_gated_act = act_info[0] == "gated"

        if len(act_info) > 1 and act_info[0] != "gated" or len(act_info) > 2:
            raise ValueError(
                f"`feed_forward_proj`: {feed_forward_proj} is not a valid activation function of the dense layer."
                "Please make sure `feed_forward_proj` is of the format `gated-{ACT_FN}` or `{ACT_FN}`, e.g. "
                "'gated-gelu' or 'relu'"
            )

        # for backwards compatibility
        if feed_forward_proj == "gated-gelu":
            self.dense_act_fn = "gelu_new"

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )
