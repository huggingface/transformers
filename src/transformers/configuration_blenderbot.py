#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Facebook, Inc. and Huggingface, 2020
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
"""BlenderbotConfig has the same signature as BartConfig. We only rewrite the signature in order to document blenderbot-90M defaults."""
from .configuration_bart import BartConfig


BLENDERBOT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/blenderbot-3B": "https://cdn.huggingface.co/facebook/blenderbot-3B/config.json",
    "facebook/blenderbot-90M": "https://cdn.huggingface.co/facebook/blenderbot-90M/config.json",
}


class BlenderbotConfig(BartConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.BlenderbotForConditionalGeneration`.
    It inherits from :class:`~transformers.BartConfig` and has the same signature with different defaults.

    Configuration objects inherit from  :class:`~transformers.PretrainedConfig` and can be used
    to control the model outputs. Read the documentation from  :class:`~transformers.PretrainedConfig`
    for more information.

    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 54944):
            Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.BlenderbotForConditionalGeneration`.
        d_model (:obj:`int`, `optional`, defaults to 512):
            Dimensionality of the layers and the pooler layer.
        encoder_layers (:obj:`int`, `optional`, defaults to 8):
            Number of encoder layers, 6 are used for the `blenderbot-90M` model.
        decoder_layers (:obj:`int`, `optional`, defaults to 8):
            Number of decoder layers, 6 are used for the `blenderbot-90M` model.
        encoder_attention_heads (:obj:`int`, `optional`, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_attention_heads (:obj:`int`, `optional`, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        decoder_ffn_dim (:obj:`int`, `optional`, defaults to 2048):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        encoder_ffn_dim (:obj:`int`, `optional`, defaults to 2048):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        activation_function (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler.
            If string, :obj:`"gelu"`, :obj:`"relu"`, :obj:`"swish"` and :obj:`"gelu_new"` are supported.
        dropout (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        classifier_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for classifier.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with.
            Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
        init_std (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        add_bias_logits (:obj:`bool`, `optional`, defaults to :obj:`False`):
            This should be completed, specific to marian.
        normalize_before (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Call layernorm before attention ops.
        normalize_embedding (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Call layernorm after embeddings.
        static_position_embeddings (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Don't learn positional embeddings, use sinusoidal.
        add_final_layer_norm (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Why not add another layernorm?
        do_blenderbot_90_layernorm (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Blenderbot-90m checkpoint uses `layernorm_embedding` one line earlier in the decoder.
        scale_embedding (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Scale embeddings by diving by sqrt(d_model).
        eos_token_id (:obj:`int`, `optional`, defaults to 2)
            End of stream token id.
        pad_token_id (:obj:`int`, `optional`, defaults to 1)
            Padding token id.
        bos_token_id (:obj:`int`, `optional`, defaults to 0)
            Beginning of stream token id.
        encoder_layerdrop: (:obj:`float`, `optional`, defaults to 0.0):
            The LayerDrop probability for the encoder. See the `LayerDrop paper
            <see https://arxiv.org/abs/1909.11556>`__ for more details.
        decoder_layerdrop: (:obj:`float`, `optional`, defaults to 0.0):
            The LayerDrop probability for the decoder. See the `LayerDrop paper
            <see https://arxiv.org/abs/1909.11556>`__ for more details.
        extra_pos_embeddings: (:obj:`int`, `optional`, defaults to 2):
            How many extra learned positional embeddings to use. Should be set to :obj:`pad_token_id+1`.
        is_encoder_decoder (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether this is an encoder/decoder model.
        force_bos_token_to_be_generated (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to force BOS token to be generated at step 1 (after ``decoder_start_token_id``),
    """
    model_type = "blenderbot"

    def __init__(
        self,
        activation_dropout=0.0,
        extra_pos_embeddings=0,
        activation_function="gelu",
        vocab_size=54944,
        d_model=512,
        encoder_ffn_dim=2048,
        encoder_layers=8,
        encoder_attention_heads=16,
        decoder_ffn_dim=2048,
        decoder_layers=8,
        decoder_attention_heads=16,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        attention_dropout=0.0,
        dropout=0.1,
        max_position_embeddings=512,
        classifier_dropout=0.0,
        is_encoder_decoder=True,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        normalize_before=False,
        add_final_layer_norm=False,
        do_blenderbot_90_layernorm=True,
        scale_embedding=False,
        normalize_embedding=True,
        static_position_embeddings=False,
        add_bias_logits=False,
        force_bos_token_to_be_generated=False,
        **common_kwargs
    ):
        r"""
        Examples::

            >>> from transformers import BlenderbotConfig
            >>> config = BlenderbotConfig.from_pretrained('facebook/blenderbot-90M')

        """
        if "hidden_size" in common_kwargs:
            raise ValueError("hidden size is called d_model")
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            vocab_size=vocab_size,
            d_model=d_model,
            encoder_ffn_dim=encoder_ffn_dim,
            encoder_layers=encoder_layers,
            encoder_layerdrop=encoder_layerdrop,
            encoder_attention_heads=encoder_attention_heads,
            decoder_layerdrop=decoder_layerdrop,
            decoder_ffn_dim=decoder_ffn_dim,
            decoder_layers=decoder_layers,
            normalize_before=normalize_before,
            normalize_embedding=normalize_embedding,
            static_position_embeddings=static_position_embeddings,
            add_bias_logits=add_bias_logits,
            force_bos_token_to_be_generated=force_bos_token_to_be_generated,
            do_blenderbot_90_layernorm=do_blenderbot_90_layernorm,
            add_final_layer_norm=add_final_layer_norm,
            scale_embedding=scale_embedding,
            attention_dropout=attention_dropout,
            dropout=dropout,
            classifier_dropout=classifier_dropout,
            activation_dropout=activation_dropout,
            max_position_embeddings=max_position_embeddings,
            extra_pos_embeddings=extra_pos_embeddings,
            activation_function=activation_function,
            decoder_attention_heads=decoder_attention_heads,
            **common_kwargs,
        )
