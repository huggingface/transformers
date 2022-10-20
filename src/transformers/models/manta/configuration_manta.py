# coding=utf-8
# Copyright 2020, The Manta Authors and HuggingFace Inc.
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
""" Manta model configuration"""
from typing import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxSeq2SeqConfigWithPast
from ...utils import logging


logger = logging.get_logger(__name__)

MANTA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "nthngdy/manta-base": "https://huggingface.co/nthngdy/manta-base/resolve/main/config.json",
}



class MantaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MantaModel`] or a [`TFMantaModel`]. It is used to
    instantiate a Manta model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Manta-base architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Arguments:
        vocab_size (`int`, *optional*, defaults to 32128):
            Vocabulary size of the Manta model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`MantaModel`] or [`TFMantaModel`].
        byte_embedding_dim (`int`, *optional*, defaults to 64):
            Size of the input byte embeddings fed to the MANTa tokenization module.
        frontier_predictor_num_layers (`int`, *optional*, defaults to 1):
            Number of sliding window attention layers in the frontier predictor of the tokenization module.
        frontier_predictor_num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads in the frontier predictor of the tokenization module.
        frontier_predictor_attention_window (`int`, *optional*, defaults to 16):
            Size of the sliding attention window along the byte sequence.
        pooling_variance_regularization (`float`, *optional*, defaults to 1.0e-6):
            Variance regularization term used in the computation of the byte-block assignment map.
        pooling_kernel_size (`int` or `List[List[int]]`, *optional*, defaults to 3):
            Size(s) of the 1D-convolution kernel(s) used for the byte pooling operation in the tokenization module. Providing an integer
            will imply using a convolution filter of `(pooling_kernel_size, byte_embedding_dim)`. Several kernel sizes can be provided
            in the form `[(kernel_size, num_channels), ...]`. These will be concatenated in the style of [Character BERT](https://arxiv.org/pdf/2010.10392.pdf).
        pooling_n_highway_layers (`int`, *optional*, defaults to 0):
            Number of highway layers after the convolution in the pooling operation.
            This allows to mimic the pooling operation of [Character BERT](https://arxiv.org/pdf/2010.10392.pdf).
        pooling_highway_activation (`string`, *optional*, defaults to `"relu"`):
            Activation function used for the highway layers in the pooling operation. Any function name from `torch.nn.functional` can be used.
        pooling_depthwise_convolution (`bool`, *optional*, defaults to `True`):
            Activates depthwise convolution in the pooling operation of the tokenization module. Depthwise convolution will be faster, but might be
            less powerful than normal convolution, and impedes using different number of channels.
        pooling_mean_pool (`bool`, *optional*, defaults to `False`):
            Activates mean-pooling instead of default max-pooling as the reduction operation for each block.
        max_length_encoder_decoder (`int`, *optional*, defaults to 256):
            Maximum output sequence length of the tokenization module. This allows to control the length of the sequences that the encoder-decoder model receives.
        d_model (`int`, *optional*, defaults to 512):
            Size of the encoder layers and the pooler layer.
        d_kv (`int`, *optional*, defaults to 64):
            Size of the key, query, value projections per attention head. `d_kv` has to be equal to `d_model //
            num_heads`.
        d_ff (`int`, *optional*, defaults to 2048):
            Size of the intermediate feed forward layer in each `MantaBlock`.
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
            Type of feed forward layer to be used. Should be one of `"relu"` or `"gated-gelu"`. Mantav1.1 uses the
            `"gated-gelu"` feed forward projection. Original Manta uses `"relu"`.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
    """
    model_type = "manta"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"hidden_size": "d_model", "num_attention_heads": "num_heads", "num_hidden_layers": "num_layers"}

    def __init__(
        self,
        vocab_size=384,
        byte_embedding_dim=64,
        frontier_predictor_num_layers=1,
        frontier_predictor_num_attention_heads=8,
        frontier_predictor_attention_window=16,
        pooling_variance_regularization=1.0e-6,
        pooling_kernel_size=3,
        pooling_n_highway_layers=0,
        pooling_highway_activation="relu",
        pooling_depthwise_convolution=True,
        pooling_mean_pool=False,
        max_length_encoder_decoder=256,
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
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.byte_embedding_dim=byte_embedding_dim
        self.frontier_predictor_num_layers=frontier_predictor_num_layers
        self.frontier_predictor_num_attention_heads=frontier_predictor_num_attention_heads
        self.frontier_predictor_attention_window=frontier_predictor_attention_window
        self.pooling_variance_regularization=pooling_variance_regularization
        self.pooling_kernel_size=pooling_kernel_size
        self.pooling_n_highway_layers=pooling_n_highway_layers
        self.pooling_highway_activation=pooling_highway_activation
        self.pooling_depthwise_convolution=pooling_depthwise_convolution
        self.pooling_mean_pool=pooling_mean_pool
        self.max_length_encoder_decoder=max_length_encoder_decoder
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

        act_info = self.feed_forward_proj.split("-")
        self.dense_act_fn = act_info[-1]
        self.is_gated_act = act_info[0] == "gated"

        if len(act_info) > 1 and act_info[0] != "gated" or len(act_info) > 2:
            raise ValueError(
                f"`feed_forward_proj`: {feed_forward_proj} is not a valid activation function of the dense layer."
                "Please make sure `feed_forward_proj` is of the format `gated-{ACT_FN}` or `{ACT_FN}`, e.g. "
                "'gated-gelu' or 'relu'"
            )
        
        if pooling_depthwise_convolution and isinstance(pooling_kernel_size, list) and any(size!=byte_embedding_dim for _, size in pooling_kernel_size):
            raise ValueError(
                f"`pooling_kernel_size`: {pooling_kernel_size} is not a valid list of kernels when `pooling_depthwise_convolution` is True. Please set all"
                f"kernel dimensions to {byte_embedding_dim} (=`byte_embedding_dim`) or `pooling_depthwise_convolution` to False."
            )

        # for backwards compatibility
        if feed_forward_proj == "gated-gelu":
            self.dense_act_fn = "gelu_new"

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            tie_word_embeddings=False,
            **kwargs,
        )
