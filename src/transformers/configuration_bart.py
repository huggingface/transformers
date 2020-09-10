# coding=utf-8
# Copyright 2020 The Fairseq Authors and The HuggingFace Inc. team.
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
""" BART configuration """

from .configuration_utils import PretrainedConfig
from .file_utils import add_start_docstrings_to_callable
from .utils import logging


logger = logging.get_logger(__name__)

BART_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/bart-base": "https://s3.amazonaws.com/models.huggingface.co/bert/facebook/bart-base/config.json",
    "facebook/bart-large": "https://s3.amazonaws.com/models.huggingface.co/bert/facebook/bart-large/config.json",
    "facebook/bart-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/facebook/bart-large-mnli/config.json",
    "facebook/bart-large-cnn": "https://s3.amazonaws.com/models.huggingface.co/bert/facebook/bart-large-cnn/config.json",
    "facebook/bart-large-xsum": "https://s3.amazonaws.com/models.huggingface.co/bert/facebook/bart-large-xsum/config.json",
    "facebook/mbart-large-en-ro": "https://s3.amazonaws.com/models.huggingface.co/bert/facebook/mbart-large-en-ro/config.json",
    "yjernite/bart_eli5": "https://s3.amazonaws.com/models.huggingface.co/bert/yjernite/bart_eli5/config.json",
}

BART_CONFIG_ARGS_DOC = r"""
    Args:
        vocab_size (:obj:`int`, optional, defaults to 50265):
            defines the different tokens that can be represented by `inputs_ids` passed to the forward method.
        d_model (:obj:`int`, optional, defaults to 1024):
            Dimensionality of the layers and the pooler layer.
        encoder_layers (:obj:`int`, optional, defaults to 12):
            Number of encoder layers, 16 for pegasus, 6 for bart-base and marian
        decoder_layers (:obj:`int`, optional, defaults to 12):
            Number of decoder layers, 16 for pegasus, 6 for bart-base and marian
        encoder_attention_heads (:obj:`int`, optional, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_attention_heads (:obj:`int`, optional, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        decoder_ffn_dim (:obj:`int`, optional, defaults to 4096):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in decoder.
        encoder_ffn_dim (:obj:`int`, optional, defaults to 4096):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in decoder.
        activation_function (:obj:`str` or :obj:`function`, optional, defaults to "gelu"):
            The non-linear activation function (function or string) in the encoder and pooler.
            If string, "gelu", "relu", "swish" and "gelu_new" are supported.
        dropout (:obj:`float`, optional, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (:obj:`float`, optional, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (:obj:`float`, optional, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        classifier_dropout (:obj:`float`, optional, defaults to 0.0):
            The dropout ratio for classifier.
        max_position_embeddings (:obj:`int`, optional, defaults to 1024):
            The maximum sequence length that this model might ever be used with.
            Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
        init_std (:obj:`float`, optional, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        add_bias_logits (:obj:`bool`, optional, defaults to :obj:`False`):
            True for marian only.
        normalize_before (:obj:`bool`, optional, defaults to :obj:`False`):
            Call layernorm before attention ops. True for pegasus, mbart. False for bart. FIXME: marian?
        normalize_embedding (:obj:`bool`, optional, defaults to :obj:`True`):
            Call layernorm after embeddings. Only True for Bart.
        static_position_embeddings (:obj:`bool`, optional, defaults to :obj:`False`):
            Don't learn positional embeddings, use sinusoidal. True for marian, pegasus.
        add_final_layer_norm (:obj:`bool`, optional, defaults to :obj:`False`):
            Why not add another layernorm?
        scale_embedding (:obj:`bool`, optional, defaults to :obj:`False`):
            Scale embeddings by diving by sqrt(d_model).
        eos_token_id (:obj:`int`, optional, defaults to 2)
            End of stream token id.
        pad_token_id (:obj:`int`, optional, defaults to 1)
            Padding token id.
        bos_token_id (:obj:`int`, optional, defaults to 0)
            Beginning of stream token id.
        encoder_layerdrop: (:obj:`float`, optional, defaults to 0.0):
            Google "layerdrop arxiv", as its not explainable in one line.
        decoder_layerdrop: (:obj:`float`, optional, defaults to 0.0):
            Google "layerdrop arxiv", as its not explainable in one line.
        extra_pos_embeddings: (:obj:`int`, optional, defaults to 2):
            How many extra learned positional embeddings to use. Should be pad_token_id+1 for bart.
        num_labels: (:obj:`int`, optional, defaults to 3):
            for SequenceClassification
        is_encoder_decoder (:obj:`bool`, optional, defaults to :obj:`True`):
            Whether this is an encoder/decoder model
        force_bos_token_to_be_generated (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to force BOS token to be generated at step 1 (after ``decoder_start_token_id``), only true for `bart-large-cnn`.

"""


@add_start_docstrings_to_callable(BART_CONFIG_ARGS_DOC)
class BartConfig(PretrainedConfig):
    r"""
    Configuration class for Bart. Parameters are renamed from the fairseq implementation
    """
    model_type = "bart"

    def __init__(
        self,
        activation_dropout=0.0,
        extra_pos_embeddings=2,  # FIXME(@sshleifer): delete?
        activation_function="gelu",
        vocab_size=50265,
        d_model=1024,
        encoder_ffn_dim=4096,
        encoder_layers=12,
        encoder_attention_heads=16,
        decoder_ffn_dim=4096,
        decoder_layers=12,
        decoder_attention_heads=16,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        attention_dropout=0.0,
        dropout=0.1,
        max_position_embeddings=1024,
        init_std=0.02,
        classifier_dropout=0.0,
        num_labels=3,
        is_encoder_decoder=True,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        normalize_before=False,
        add_final_layer_norm=False,
        scale_embedding=False,
        normalize_embedding=True,
        static_position_embeddings=False,
        add_bias_logits=False,
        force_bos_token_to_be_generated=False,
        **common_kwargs
    ):
        r"""
        :class:`~transformers.BartConfig` is the configuration class for `BartModel`.

        Examples::

            >>> from transformers import BartConfig, BartModel

            >>> config = BartConfig.from_pretrained('facebook/bart-large')
            >>> model = BartModel(config)

        """
        if "hidden_size" in common_kwargs:
            raise ValueError("hidden size is called d_model")
        super().__init__(
            num_labels=num_labels,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            **common_kwargs,
        )
        self.vocab_size = vocab_size
        self.d_model = d_model  # encoder_embed_dim and decoder_embed_dim
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = self.num_hidden_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.init_std = init_std  # Normal(0, this parameter)
        self.activation_function = activation_function

        # Params introduced for Mbart
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        self.normalize_embedding = normalize_embedding  # True for mbart, False otherwise
        self.normalize_before = normalize_before  # combo of fairseq's encoder_ and decoder_normalize_before
        self.add_final_layer_norm = add_final_layer_norm

        # Params introduced for Marian
        self.add_bias_logits = add_bias_logits
        self.static_position_embeddings = static_position_embeddings

        # 3 Types of Dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.dropout = dropout

        # Classifier stuff
        self.classif_dropout = classifier_dropout

        # pos embedding offset
        self.extra_pos_embeddings = self.pad_token_id + 1

        self.force_bos_token_to_be_generated = force_bos_token_to_be_generated

    @property
    def num_attention_heads(self) -> int:
        return self.encoder_attention_heads

    @property
    def hidden_size(self) -> int:
        return self.d_model

    def is_valid_mbart(self) -> bool:
        """Is the configuration aligned with the MBART paper."""
        if self.normalize_before and self.add_final_layer_norm and self.scale_embedding:
            return True
        if self.normalize_before or self.add_final_layer_norm or self.scale_embedding:
            logger.info("This configuration is a mixture of MBART and BART settings")
        return False
