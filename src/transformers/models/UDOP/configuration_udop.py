# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Dict, Optional, Sequence, Tuple

from ...configuration_utils import PretrainedConfig


Udop_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class UdopConfig(PretrainedConfig):
    pretrained_config_archive_map = Udop_PRETRAINED_CONFIG_ARCHIVE_MAP

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
        schema_version: Optional[str] = None,
        max_context_weight: float = 1.0,
        context_weight_update_time: float = 0.5,
        max_pos_dropout: float = 0.0,
        image_size: int = 224,
        pos_dropout_update_time: float = 0.5,
        positional_dropout_type: str = "random",
        context_embeddings: Optional[Sequence[Dict[str, Any]]] = None,
        relative_bias_args: Optional[Sequence[Dict[str, Any]]] = [
            {"type": "1d"},
            {"type": "horizontal"},
            {"type": "vertical"},
        ],
        vision_augmentation: Optional[Sequence[Dict[str, Any]]] = None,
        do_lower_case: bool = False,
        disable_sequential_embeddings: bool = False,
        word_dropout: float = 0.0,
        locked_dropout: float = 0.0,
        attention_dropout: Optional[float] = None,
        context_residual: Optional[str] = None,
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
        self.mae_version = "mae_vit_large_patch16"
        self.mae_checkpoint = "mae-models/mae_pretrain_vit_large_full.pth"
        self.data_dir = "."

        self.schema_version = schema_version
        self.max_context_weight = max_context_weight
        self.context_weight_update_time = context_weight_update_time
        self.max_pos_dropout = max_pos_dropout
        self.pos_dropout_update_time = pos_dropout_update_time
        self.positional_dropout_type = positional_dropout_type
        self.context_embeddings = [] if context_embeddings is None else context_embeddings
        self.relative_bias_args = [] if relative_bias_args is None else relative_bias_args
        self.vision_augmentation = [] if vision_augmentation is None else vision_augmentation
        self.do_lower_case = do_lower_case
        self.disable_sequential_embeddings = disable_sequential_embeddings
        self.word_dropout = word_dropout
        self.locked_dropout = locked_dropout
        self.attention_dropout = attention_dropout
        self.context_residual = context_residual
        self.truncate_decoder_after_layer = truncate_decoder_after_layer
        self.truncate_encoder_after_layer = truncate_encoder_after_layer
        self.image_size = image_size

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
