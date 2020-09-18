#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the;
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
# LICENSE file in the root directory of this source tree.
from .modeling_bart import BartConfig


BLENDERBOT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/blenderbot-3B": "https://cdn.huggingface.co/facebook/blenderbot-3B/config.json",
    "facebook/blenderbot-90M": "https://cdn.huggingface.co/facebook/blenderbot-/config.json",
}


class BlenderbotConfig(BartConfig):
    """
    This is the configuration class to store the configuration of a :class:`~transformers.BlenderbotForConditionalGeneration`.
    Instantiating a configuration with the defaults will yield a similar configuration to that of
    the `blenderbot <https://huggingface.co/blenderbot>`__ architecture.

    Configuration objects inherit from  :class:`~transformers.BartConfig` and can be used
    to control the model outputs. Read the documentation from  :class:`~transformers.BartConfig`
    for more information. The

    Args:
        d_model: (:obj:`int`, default to 2560), dimension of the embeddings vector
        encoder_layers: (:obj:`int`, default to 2), number of layers in the encoder
        encoder_ffn_size: (:obj:`int`, default to 10240), size of hidden layers in the FFN in the encoder
        decoder_layers: (:obj:`int`, default to 24), number of layers in the decoder
        decoder_ffn_size: (:obj:`int`, default to 10240), size of hidden layers in the FFN in the decoder
        dropout: (:obj:`float`, default to 0.1), embedding dropout
        activation_dropout: (:obj:`float`, default to 0.0), dropout after activation function
        encoder_layerdrop: (:obj:`float`, default to 0.0,
        decoder_layerdrop: (:obj:`float`, default to 0.0),
        encoder_attention_heads:(:obj:`int`, default to 32),  number of multi heads attention in the encoder
        decoder_attention_heads:(:obj:`int`, default to 32),  number of multi heads attention in the encoder
        max_positions_embeddings:(:obj:`int`, default to 128), size of the position embeddings
        activation: (:obj:`string`, default to 'gelu'), activation function to use
        attention_dropout: (:obj:`float`, default to 0.0), multi head attention dropout
        relu_dropout: (:obj:`float`, default to 0.0), relu dropout
        vocab_size: (:obj:`int`, default to 8008), the size of the vocabulary
        variant: (obj: str, default to "prelayernorm") defines when to apply a layernorm
        init_std: (obj: float, default to 0.02): The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        is_encoder_decoder: (obj:`boolean`, default to True)
        pad_token_id: (obj:`int`, default to 1): token id used to pad sequences.
        bos_token_id: (obj:`int`, default to 0): begginning of sequence token id.
        eos_token_id: (obj:`int`, default to 2): end of sequence token id.
        add_final_layer_norm: (obj:`boolean`, default to False): if set to true a final Layernorm is added
        scale_embedding: (obj:`boolean`, default to False):  Scale embeddings by diving by sqrt(d_model)
        normalize_embedding: (obj:`boolean`, default to False): apply Layernorm to the embedding layer output
        static_position_embeddings: (:obj:`boolean`, default to False): if set to True  positional embeddings are learnt otherwise use sinusoidal

    Attributes:
        pretrained_config_archive_map (Dict[str, str]): A dictionary containing all the available pre-trained checkpoints.
    """

    model_type = "blenderbot"
