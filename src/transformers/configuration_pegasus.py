# coding=utf-8
# Copyright 2020 Google and The HuggingFace Inc. team.
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
""" PEGASUS model configuration """

from .configuration_bart import BartConfig
from .utils import logging


logger = logging.get_logger(__name__)

# These config values do not vary between checkpoints
DEFAULTS = dict(
    vocab_size=96103,
    max_position_embeddings=512,
    d_model=1024,
    encoder_ffn_dim=4096,
    decoder_ffn_dim=4096,
    encoder_attention_heads=16,
    decoder_attention_heads=16,
    encoder_layers=16,
    decoder_layers=16,
    dropout=0.1,
    attention_dropout=0.1,
    activation_dropout=0.1,
    pad_token_id=0,
    eos_token_id=1,
    is_encoder_decoder=True,
    normalize_before=True,
    scale_embedding=True,
    normalize_embedding=False,
    add_final_layer_norm=True,
    static_position_embeddings=True,
    num_beams=8,
    activation_function="relu",
)
# Config values that vary between checkpoints: for testing and conversion
task_specific_params = {
    # These are task specific params for pegasus-large and normal params for finetuned checkpoints
    "summarization_xsum": {"length_penalty": 0.6, "max_length": 64, "max_position_embeddings": 512},
    "summarization_cnn_dailymail": {"length_penalty": 0.8, "max_length": 128, "max_position_embeddings": 1024},
    "summarization_newsroom": {"length_penalty": 0.8, "max_length": 128, "max_position_embeddings": 512},
    "summarization_wikihow": {"length_penalty": 0.6, "max_length": 256, "max_position_embeddings": 512},
    "summarization_multi_news": {"length_penalty": 0.8, "max_length": 256, "max_position_embeddings": 1024},
    "summarization_reddit_tifu": {"length_penalty": 0.6, "max_length": 128, "max_position_embeddings": 512},
    "summarization_big_patent": {"length_penalty": 0.7, "max_length": 256, "max_position_embeddings": 1024},
    "summarization_arxiv": {"length_penalty": 0.8, "max_length": 256, "max_position_embeddings": 1024},
    "summarization_pubmed": {"length_penalty": 0.8, "max_length": 256, "max_position_embeddings": 1024},
    "summarization_gigaword": {"length_penalty": 0.6, "max_length": 32, "max_position_embeddings": 128},
    "summarization_aeslc": {"length_penalty": 0.6, "max_length": 32, "max_position_embeddings": 512},
    "summarization_billsum": {"length_penalty": 0.6, "max_length": 256, "max_position_embeddings": 1024},
    # this last entry is useless -- just for consistency
    "summarization_large": {"length_penalty": 0.8, "max_length": 256, "max_position_embeddings": 1024},
}


class PegasusConfig(BartConfig):
    """
    This is the configuration class to store the configuration of a
    :class:`~transformers.PegasusForConditionalGeneration`. It is used to instantiate a Pegasus model according to the
    specified arguments, defining the model architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 96103):
            Vocabulary size of the Pegasus model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.PegasusForConditionalGeneration`.
        d_model (:obj:`int`, `optional`, defaults to 1024):
            Dimensionality of the layers and the pooler layer.
        encoder_layers (:obj:`int`, `optional`, defaults to 16):
            Number of encoder layers.
        decoder_layers (:obj:`int`, `optional`, defaults to 16):
            Number of decoder layers.
        encoder_attention_heads (:obj:`int`, `optional`, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_attention_heads (:obj:`int`, `optional`, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        decoder_ffn_dim (:obj:`int`, `optional`, defaults to 4096):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in decoder.
        encoder_ffn_dim (:obj:`int`, `optional`, defaults to 4096):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in decoder.
        activation_function (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        dropout (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        classifier_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for classifier.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        init_std (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        add_bias_logits (:obj:`bool`, `optional`, defaults to :obj:`False`):
            This should be completed, specific to marian.
        normalize_before (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Call layernorm before attention ops.
        normalize_embedding (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Call layernorm after embeddings.
        static_position_embeddings (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Don't learn positional embeddings, use sinusoidal.
        add_final_layer_norm (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Why not add another layernorm?
        scale_embedding (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Scale embeddings by diving by sqrt(d_model).
        eos_token_id (:obj:`int`, `optional`, defaults to 2)
            End of stream token id.
        pad_token_id (:obj:`int`, `optional`, defaults to 1)
            Padding token id.
        bos_token_id (:obj:`int`, `optional`, defaults to 0)
            Beginning of stream token id.
        encoder_layerdrop: (:obj:`float`, `optional`, defaults to 0.0):
            The LayerDrop probability for the encoder. See the `LayerDrop paper <see
            https://arxiv.org/abs/1909.11556>`__ for more details.
        decoder_layerdrop: (:obj:`float`, `optional`, defaults to 0.0):
            The LayerDrop probability for the decoder. See the `LayerDrop paper <see
            https://arxiv.org/abs/1909.11556>`__ for more details.
        extra_pos_embeddings: (:obj:`int`, `optional`, defaults to 2):
            How many extra learned positional embeddings to use. Should be pad_token_id+1 for bart.
        is_encoder_decoder (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether this is an encoder/decoder model
        force_bos_token_to_be_generated (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to force BOS token to be generated at step 1 (after ``decoder_start_token_id``).
    """

    model_type = "pegasus"
    # The implementation of the config object is in BartConfig
