# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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

from typing import Optional

from ...configuration_utils import PretrainedConfig


UDOP_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class UdopConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`UdopDualForConditionalGeneration`] or a
    [`UdopUnimodelForConditionalGeneration`]. It is used to instantiate a UdopDualForConditionalGeneration or
    UdopUnimodelForConditionalGeneration model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Arguments:
        vocab_size (`int`, *optional*, defaults to 32128):
            Vocabulary size of the Udop model. Defines the number of different tokens that can be represented by the
            `inputs_ids`.
        d_model (`int`, *optional*, defaults to 512):
            Size of the encoder layers and the pooler layer.
        d_kv (`int`, *optional*, defaults to 64):
            Size of the key, query, value projections per attention head. `d_kv` has to be equal to `d_model //
            num_heads`.
        d_ff (`int`, *optional*, defaults to 2048):
            Size of the intermediate feed forward layer in each `UdopBlock`.
        num_layers (`int`, *optional*, defaults to 6):
            Number of hidden layers in the Transformer encoder.
        num_decoder_layers (`int`, *optional*):
            Number of hidden layers in the Transformer decoder. Will use the same value as `num_layers` if not set.
        num_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        relative_attention_num_buckets (`int`, *optional*, defaults to 32):
            The number of buckets to use for each attention layer.
        relative_attention_max_distance (`int`, *optional*, defaults to 128):
            The maximum distance of the longer sequences for the bucket separation.
        dropout_rate (`float`, *optional*, defaults to 0.1):
            The ratio for all dropout layers.
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the layer normalization layers.
        initializer_factor (`float`, *optional*, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        feed_forward_proj (`string`, *optional*, defaults to `"relu"`):
            Type of feed forward layer to be used. Should be one of `"relu"` or `"gated-gelu"`.
        is_encoder_decoder (`bool`, *optional*, defaults to `False`):
            Is the encoder and decoder same
        pad_token_id (`int`, *optional*, defaults to 0): The id of the _padding_ token.
        eos_token_id (`int`, *optional*, defaults to 0): The id of the _end-of-stream_ token.
        max_2d_position_embeddings (`int`, *optional*, defaults to 1024):
            The maximum value that the 2D position embedding might ever used. Typically set this to something large
            just in case (e.g., 1024).
        max_bbox_length (`int`, *optional*, defaults to 1001):
            The maximum length of the bbox coordinates.
        mae_config (`dict`, *optional*):
            Config to be used for Masked autoencoder VIT model various options are :
                patch_size (`int`, *optional*, defaults to 0): Patch size to use embedding layer. in_channels (`int`,
                *optional*, defaults to 0): Number of channels in the image. embed_dim (`int`, *optional*, defaults to
                0): Size of the Embedding layer to be used for bbox depth (`int`, *optional*, defaults to 0): Depth to
                be used in the MAE VIT. num_heads (`int`, *optional*, defaults to 0): Number of head to be used in the
                MAE VIT. decoder_embed_dim (`int`, *optional*, defaults to 0): Embedding dim to be used in decoder of
                MAE VIT decoder_depth (`int`, *optional*, defaults to 0): Depth to be used in the decoder of MAE VIT.
                decoder_num_heads (`int`, *optional*, defaults to 0): Number of head to be used in decoder of MAE VIT.
                mlp_ratio (`int`, *optional*, defaults to 0): Number of head to be used in decoder of MAE VIT.
        image_size (`int`, *optional*, defaults to `224`):
            The size (resolution) of each image.
        ccat (`bool`, *optional*, defaults to `False`): The parameter controlling the embedding dim.
        relative_bias_args ('list', *optional*,
            defaults to '[{"type": "1d"}{"type": "horizontal"}{"type": "vertical"}]'): Denotes the order of relative
            bias to be used
        truncate_decoder_after_layer (`int`, *optional*, defaults to `None`):
            Denotes the number of the layers in decoder
        truncate_encoder_after_layer (`int`, *optional*, defaults to `None`):
            Denotes the number of the layers in encoder.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
    """
    pretrained_config_archive_map = UDOP_PRETRAINED_CONFIG_ARCHIVE_MAP
    model_type = "udop"
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
        pad_token_id=0,
        eos_token_id=1,
        max_2d_position_embeddings=1024,
        max_bbox_length=1001,
        mae_config={
            "patch_size": 16,
            "in_channels": 3,
            "embed_dim": 1024,
            "depth": 24,
            "num_heads": 16,
            "decoder_embed_dim": 512,
            "decoder_depth": 8,
            "decoder_num_heads": 16,
            "mlp_ratio": 4,
        },
        image_size=224,
        ccat=False,
        relative_bias_args=[
            {"type": "1d"},
            {"type": "horizontal"},
            {"type": "vertical"},
        ],
        truncate_decoder_after_layer: Optional[int] = None,
        truncate_encoder_after_layer: Optional[int] = None,
        use_cache=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_decoder_layers = num_decoder_layers if num_decoder_layers is not None else self.num_layers
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
        self.mae_config = mae_config
        self.truncate_decoder_after_layer = truncate_decoder_after_layer
        self.truncate_encoder_after_layer = truncate_encoder_after_layer
        self.is_encoder_decoder = is_encoder_decoder
        self.pad_token_id = (pad_token_id,)
        self.eos_token_id = eos_token_id
        self.ccat = ccat
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
