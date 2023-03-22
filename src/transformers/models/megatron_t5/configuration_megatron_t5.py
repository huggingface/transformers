# coding=utf-8
# Copyright 2021- NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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
""" MEGATRON-T5 model configuration"""
from typing import Mapping

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

MEGATRON_T5_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "t5-small": "https://huggingface.co/t5-small/resolve/main/config.json",
    "t5-base": "https://huggingface.co/t5-base/resolve/main/config.json",
    "t5-large": "https://huggingface.co/t5-large/resolve/main/config.json",
    "t5-3b": "https://huggingface.co/t5-3b/resolve/main/config.json",
    "t5-11b": "https://huggingface.co/t5-11b/resolve/main/config.json",
}


class MegatronT5Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MegatronT5Model`]. It is used to
    instantiate a MegatronT5 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the T5
    [t5-small](https://huggingface.co/t5-small) architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Arguments:
        vocab_size (`int`, *optional*, defaults to 32128):
            Vocabulary size of the T5 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`MegatronT5Model`].
        d_model (`int`, *optional*, defaults to 512):
            Size of the encoder layers and the pooler layer.
        d_kv (`int`, *optional*, defaults to 64):
            Size of the key, query, value projections per attention head. `d_kv` has to be equal to `d_model //
            num_heads`.
        d_ff (`int`, *optional*, defaults to 2048):
            Size of the intermediate feed forward layer in each `T5Block`.
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
        feed_forward_proj (`string`, *optional*, defaults to `"gated-gelu"`):
            Type of feed forward layer to be used. Should be one of `"relu"` or `"gated-gelu"`. T5v1.1 uses the
            `"gated-gelu"` feed forward projection. Original T5 uses `"relu"`.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        use_bias (`bool`, *optional*, defaults to `True`):
            Flag for determining whether to use bias in dense and attention layer.
        apply_query_key_layer_scaling (`bool`, *optional*, defaults to `True`):
            Flag for determining whether to use query_key_layer_scaling in T5Attention.
        use_rescale_tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Flag for determining whether to rescale output before projecting on vocab.
        position_embedding_type (`string`, *optional*, defaults to `learned_absolute`):
            Type of position embedding to be used. Should be one of `"learned_absolute"` or `"relative"`.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum distance of the longer sequences for `learned_absolute` position embedding.
        normalization (`string`, *optional*, defaults to `layernorm`):
            Type of layer normalization to be used. Should be one of `"layernorm"` or `"rmsnorm"`.
    """
    model_type = "megatron-t5"
    keys_to_ignore_at_inference = ["past_key_values"]
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
        feed_forward_proj="gated-gelu",
        is_encoder_decoder=True,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        use_bias=True,
        apply_query_key_layer_scaling=True,
        use_rescale_tie_word_embeddings=False,
        position_embedding_type="learned_absolute",
        max_position_embeddings=512,
        normalization="layernorm",
        **kwargs,
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
        self.use_bias = use_bias
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.use_rescale_tie_word_embeddings = use_rescale_tie_word_embeddings
        self.position_embedding_type = position_embedding_type
        self.max_position_embeddings = max_position_embeddings
        self.normalization = normalization

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )
