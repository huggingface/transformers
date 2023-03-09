# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
""" Pop2Piano model configuration """

from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Mapping, Optional, Union

from ...configuration_utils import PretrainedConfig
from ...utils import logging

logger = logging.get_logger(__name__)

POP2PIANO_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "sweetcocoa/pop2piano": "https://huggingface.co/sweetcocoa/pop2piano/blob/main/config.json"
}

COMPOSER_TO_FEATURE_TOKEN = {'composer1': 2052,
                             'composer2': 2053,
                             'composer3': 2054,
                             'composer4': 2055,
                             'composer5': 2056,
                             'composer6': 2057,
                             'composer7': 2058,
                             'composer8': 2059,
                             'composer9': 2060,
                             'composer10': 2061,
                             'composer11': 2062,
                             'composer12': 2063,
                             'composer13': 2064,
                             'composer14': 2065,
                             'composer15': 2066,
                             'composer16': 2067,
                             'composer17': 2068,
                             'composer18': 2069,
                             'composer19': 2070,
                             'composer20': 2071,
                             'composer21': 2072
}

# Adapted from transformers.models.t5.configuration_t5.T5Config with T5->Pop2Piano,T5Model->Pop2PianoModel,t5->pop2piano,T5Block->Pop2PianoBlock
class Pop2PianoConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Pop2PianoModel`]. It is used to instantiate a
    Pop2Piano model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Pop2Piano
    [sweetcocoa/pop2piano](https://huggingface.co/sweetcocoa/pop2piano) architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Arguments:
        vocab_size (`int`, *optional*, defaults to 32128):
            Vocabulary size of the Pop2Piano model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Pop2PianoModel`].
        d_model (`int`, *optional*, defaults to 512):
            Size of the encoder layers and the pooler layer.
        d_kv (`int`, *optional*, defaults to 64):
            Size of the key, query, value projections per attention head. The `inner_dim` of the projection layer will
            be defined as `num_heads * d_kv`.
        d_ff (`int`, *optional*, defaults to 2048):
            Size of the intermediate feed forward layer in each `Pop2PianoBlock`.
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
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the layer normalization layers.
        initializer_factor (`float`, *optional*, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        feed_forward_proj (`string`, *optional*, defaults to `"gated-gelu"`):
            Type of feed forward layer to be used. Should be one of `"relu"` or `"gated-gelu"`.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        n_fft (`int`, *optional*, defaults to 4096):
            Size of Fast Fourier Transform, creates n_fft // 2 + 1 bins.
        hop_length (`int`, *optional*, defaults to 1024):
            Length of hop between Short-Time Fourier Transform windows.
        f_min (`float`, *optional*, defaults to 10.0):
            Minimum frequency.
        n_mels (`int`, *optional*, defaults to 512):
            Number of mel filterbanks.
        dense_act_fn (`string`, *optional*, defaults to `"relu"`):
            Type of Activation Function to be used in `Pop2PianoDenseActDense` and in `Pop2PianoDenseGatedActDense`.
        dataset_sample_rate (`int` *optional*, defaults to 22050):
            Sample rate of audio signal.
        dataset_mel_is_conditioned (`bool`, *optional*, defaults to `True`):
            Whether to use `ConcatEmbeddingToMel` or not.
        dataset_target_length (`int`, *optional*, defaults to 256):
            Determines `max_length` for transformer `generate` function along with `dataset_n_bars`.
        dataset_n_bars (`int`, *optional*, defaults to 2):
            Determines `max_length` for transformer `generate` function along with `dataset_target_length`.
    """

    model_type = "pop2piano"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"hidden_size": "d_model", "num_attention_heads": "num_heads", "num_hidden_layers": "num_layers"}

    def __init__(
        self,
        vocab_size=2400,
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
        dense_act_fn="relu",
        dataset_target_length=256,
        dataset_n_bars=2,
        dataset_sample_rate=22050,
        dataset_mel_is_conditioned=True,
        n_fft=4096,
        hop_length=1024,
        f_min=10.0,
        n_mels=512,
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

        act_info = self.feed_forward_proj.split("-")
        self.dense_act_fn = dense_act_fn
        self.is_gated_act = act_info[0] == "gated"

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )

        self.composer_to_feature_token = COMPOSER_TO_FEATURE_TOKEN
        self.dataset = {'target_length': dataset_target_length,
                        'n_bars': dataset_n_bars,
                        'sample_rate': dataset_sample_rate,
                        'mel_is_conditioned': dataset_mel_is_conditioned
                        }
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.f_min = f_min
        self.n_mels = n_mels
