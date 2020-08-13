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
from .configuration_utils import PretrainedConfig



BLENDERBOT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/blenderbot-3B": "https://cdn.huggingface.co/facebook/blenderbot-3B/config.json",
    "facebook/blenderbot-90M": "https://cdn.huggingface.co/facebook/blenderbot-/config.json",
}


class BlenderbotConfig(PretrainedConfig):
    """
        This is the configuration class to store the configuration of a :class:`~transformers.BlenderbotModel`.
        It is used to instantiate an Blenderbot model according to the specified arguments, defining the model
        architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
        the `blenderbot <https://huggingface.co/blenderbot>`__ architecture.

        Configuration objects inherit from  :class:`~transformers.PretrainedConfig` and can be used
        to control the model outputs. Read the documentation from  :class:`~transformers.PretrainedConfig`
        for more information.

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
            static_position_embeddings: (:obj:`boolean`, default to False),  if yes or no the positional embeddings will be learn
            variant: (obj: str, default to "prelayernorm") defines when to apply a layernorm
            init_std: (obj: float, default to 0.02)
            is_encoder_decoder: (obj:`boolean`, default to True)
            pad_token_id: (obj:`int`, default to 1),
            bos_token_id: (obj:`int`, default to 0),
            eos_token_id: (obj:`int`, default to 2),
            normalize_before: (:obj:`boolean`, default to True),
            add_final_layer_norm: (obj:`boolean`, default to False),
            scale_embedding: (obj:`boolean`, default to False),
            normalize_embedding: (obj:`boolean`, default to False),
            static_position_embeddings: (:obj:`boolean`, default to False),
            add_bias_logits: (:obj:`boolean`, default to False),

        Attributes:
            pretrained_config_archive_map (Dict[str, str]): A dictionary containing all the available pre-trained checkpoints.
    """

    model_type = "blenderbot"

    def __init__(
        self,
        activation_dropout=0.0,
        extra_pos_embeddings=0,
        activation_function="gelu",
        vocab_size=8008,
        d_model=2560,
        encoder_ffn_dim=10240,
        encoder_layers=2,
        encoder_attention_heads=32,
        decoder_ffn_dim=10240,
        decoder_layers=24,
        decoder_attention_heads=32,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        attention_dropout=0.0,
        dropout=0.1,
        max_position_embeddings=128,
        init_std=0.02,
        is_encoder_decoder=True,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        normalize_before=True,
        add_final_layer_norm=False,
        scale_embedding=False,
        normalize_embedding=False,
        static_position_embeddings=False,
        add_bias_logits=False,
        variant="prelayernorm",
        **kwargs
    ):
        r"""
            :class:`~transformers.BlenderbotConfig` is the configuration class for `BlenderbotForConditionalGeneration`.

            Examples::

                >>> from transformers import BlenderbotConfig, BlenderbotForConditionalGeneration

                >>> config = BlenderbotConfig.from_pretrained('facebook/blenderbot-3B')
                >>> model = BlenderbotForComditionalGeneration(config)
        """
        if "hidden_size" in kwargs:
            raise ValueError("hidden size is called d_model")
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = self.num_hidden_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.init_std = init_std
        self.activation_function = activation_function
        self.variant = variant
        self.scale_embedding = scale_embedding
        self.normalize_embedding = normalize_embedding
        self.normalize_before = normalize_before
        self.add_final_layer_norm = add_final_layer_norm
        self.add_bias_logits = add_bias_logits
        self.static_position_embeddings = static_position_embeddings

        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.dropout = dropout
        self.extra_pos_embeddings = extra_pos_embeddings

    @property
    def num_attention_heads(self) -> int:
        return self.encoder_attention_heads

    @property
    def hidden_size(self) -> int:
        return self.d_model
