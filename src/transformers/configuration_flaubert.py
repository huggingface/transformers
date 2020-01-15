# coding=utf-8
# Copyright 2019-present CNRS, Facebook Inc. and the HuggingFace Inc. team.
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
""" Flaubert configuration, based on XLM. """


import logging

from .configuration_xlm import XLMConfig


logger = logging.getLogger(__name__)

FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "flaubert-small-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/flaubert/flaubert_small_cased/config.json",
    "flaubert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/flaubert/flaubert_base_uncased/config.json",
    "flaubert-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/flaubert/flaubert_base_cased/config.json",
    "flaubert-large-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/flaubert/flaubert_large_cased/config.json",
}


class FlaubertConfig(XLMConfig):
    """Configuration class to store the configuration of a `FlaubertModel`.

    Args:
        vocab_size: Vocabulary size of `inputs_ids` in `FlaubertModel`.
        d_model: Size of the encoder layers and the pooler layer.
        n_layer: Number of hidden layers in the Transformer encoder.
        n_head: Number of attention heads for each attention layer in
            the Transformer encoder.
        d_inner: The size of the "intermediate" (i.e., feed-forward)
            layer in the Transformer encoder.
        ff_activation: The non-linear activation function (function or string) in the
            encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
        untie_r: untie relative position biases
        attn_type: 'bi' for Flaubert, 'uni' for Transformer-XL

        dropout: The dropout probabilitiy for all fully connected
            layers in the embeddings, encoder, and pooler.
        max_position_embeddings: The maximum sequence length that this model might
            ever be used with. Typically set this to something large just in case
            (e.g., 512 or 1024 or 2048).
        initializer_range: The sttdev of the truncated_normal_initializer for
            initializing all weight matrices.
        layer_norm_eps: The epsilon used by LayerNorm.

        dropout: float, dropout rate.
        init: str, the initialization scheme, either "normal" or "uniform".
        init_range: float, initialize the parameters with a uniform distribution
            in [-init_range, init_range]. Only effective when init="uniform".
        init_std: float, initialize the parameters with a normal distribution
            with mean 0 and stddev init_std. Only effective when init="normal".
        mem_len: int, the number of tokens to cache.
        reuse_len: int, the number of tokens in the currect batch to be cached
            and reused in the future.
        bi_data: bool, whether to use bidirectional input pipeline.
            Usually set to True during pretraining and False during finetuning.
        clamp_len: int, clamp all relative distances larger than clamp_len.
            -1 means no clamping.
        same_length: bool, whether to use the same attention length for each token.
    """

    pretrained_config_archive_map = FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP
    model_type = "flaubert"

    def __init__(self, layerdrop=0.0, pre_norm=False, **kwargs):
        """Constructs FlaubertConfig.
        """
        super().__init__(**kwargs)
        self.layerdrop = layerdrop
        self.pre_norm = pre_norm
