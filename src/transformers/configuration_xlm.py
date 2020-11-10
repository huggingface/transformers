# coding=utf-8
# Copyright 2019-present, Facebook, Inc and the HuggingFace Inc. team.
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
""" XLM configuration """

from .configuration_utils import PretrainedConfig
from .utils import logging


logger = logging.get_logger(__name__)

XLM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "xlm-mlm-en-2048": "https://huggingface.co/xlm-mlm-en-2048/resolve/main/config.json",
    "xlm-mlm-ende-1024": "https://huggingface.co/xlm-mlm-ende-1024/resolve/main/config.json",
    "xlm-mlm-enfr-1024": "https://huggingface.co/xlm-mlm-enfr-1024/resolve/main/config.json",
    "xlm-mlm-enro-1024": "https://huggingface.co/xlm-mlm-enro-1024/resolve/main/config.json",
    "xlm-mlm-tlm-xnli15-1024": "https://huggingface.co/xlm-mlm-tlm-xnli15-1024/resolve/main/config.json",
    "xlm-mlm-xnli15-1024": "https://huggingface.co/xlm-mlm-xnli15-1024/resolve/main/config.json",
    "xlm-clm-enfr-1024": "https://huggingface.co/xlm-clm-enfr-1024/resolve/main/config.json",
    "xlm-clm-ende-1024": "https://huggingface.co/xlm-clm-ende-1024/resolve/main/config.json",
    "xlm-mlm-17-1280": "https://huggingface.co/xlm-mlm-17-1280/resolve/main/config.json",
    "xlm-mlm-100-1280": "https://huggingface.co/xlm-mlm-100-1280/resolve/main/config.json",
}


class XLMConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~transformers.XLMModel` or a
    :class:`~transformers.TFXLMModel`. It is used to instantiate a XLM model according to the specified arguments,
    defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration
    to that of the `xlm-mlm-en-2048 <https://huggingface.co/xlm-mlm-en-2048>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30145):
            Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.XLMModel` or :class:`~transformers.TFXLMModel`.
        emb_dim (:obj:`int`, `optional`, defaults to 2048):
            Dimensionality of the encoder layers and the pooler layer.
        n_layer (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        n_head (:obj:`int`, `optional`, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        dropout (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for the attention mechanism
        gelu_activation (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to use `gelu` for the activations instead of `relu`.
        sinusoidal_embeddings (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to use sinusoidal positional embeddings instead of absolute positional embeddings.
        causal (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the model should behave in a causal manner. Causal models use a triangular attention mask in
            order to only attend to the left-side context instead if a bidirectional context.
        asm (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to use an adaptive log softmax projection layer instead of a linear layer for the prediction
            layer.
        n_langs (:obj:`int`, `optional`, defaults to 1):
            The number of languages the model handles. Set to 1 for monolingual models.
        use_lang_emb (:obj:`bool`, `optional`, defaults to :obj:`True`)
            Whether to use language embeddings. Some models use additional language embeddings, see `the multilingual
            models page <http://huggingface.co/transformers/multilingual.html#xlm-language-embeddings>`__ for
            information on how to use them.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        embed_init_std (:obj:`float`, `optional`, defaults to 2048^-0.5):
            The standard deviation of the truncated_normal_initializer for initializing the embedding matrices.
        init_std (:obj:`int`, `optional`, defaults to 50257):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices except the
            embedding matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        bos_index (:obj:`int`, `optional`, defaults to 0):
            The index of the beginning of sentence token in the vocabulary.
        eos_index (:obj:`int`, `optional`, defaults to 1):
            The index of the end of sentence token in the vocabulary.
        pad_index (:obj:`int`, `optional`, defaults to 2):
            The index of the padding token in the vocabulary.
        unk_index (:obj:`int`, `optional`, defaults to 3):
            The index of the unknown token in the vocabulary.
        mask_index (:obj:`int`, `optional`, defaults to 5):
            The index of the masking token in the vocabulary.
        is_encoder(:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the initialized model should be a transformer encoder or decoder as seen in Vaswani et al.
        summary_type (:obj:`string`, `optional`, defaults to "first"):
            Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.

            Has to be one of the following options:

                - :obj:`"last"`: Take the last token hidden state (like XLNet).
                - :obj:`"first"`: Take the first token hidden state (like BERT).
                - :obj:`"mean"`: Take the mean of all tokens hidden states.
                - :obj:`"cls_index"`: Supply a Tensor of classification token position (like GPT/GPT-2).
                - :obj:`"attn"`: Not implemented now, use multi-head attention.
        summary_use_proj (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.

            Whether or not to add a projection after the vector extraction.
        summary_activation (:obj:`str`, `optional`):
            Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.

            Pass :obj:`"tanh"` for a tanh activation to the output, any other value will result in no activation.
        summary_proj_to_labels (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Used in the sequence classification and multiple choice models.

            Whether the projection outputs should have :obj:`config.num_labels` or :obj:`config.hidden_size` classes.
        summary_first_dropout (:obj:`float`, `optional`, defaults to 0.1):
            Used in the sequence classification and multiple choice models.

            The dropout ratio to be used after the projection and activation.
        start_n_top (:obj:`int`, `optional`, defaults to 5):
            Used in the SQuAD evaluation script.
        end_n_top (:obj:`int`, `optional`, defaults to 5):
            Used in the SQuAD evaluation script.
        mask_token_id (:obj:`int`, `optional`, defaults to 0):
            Model agnostic parameter to identify masked tokens when generating text in an MLM context.
        lang_id (:obj:`int`, `optional`, defaults to 1):
            The ID of the language used by the model. This parameter is used when generating text in a given language.

    Examples::

        >>> from transformers import XLMConfig, XLMModel

        >>> # Initializing a XLM configuration
        >>> configuration = XLMConfig()

        >>> # Initializing a model from the configuration
        >>> model = XLMModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "xlm"

    def __init__(
        self,
        vocab_size=30145,
        emb_dim=2048,
        n_layers=12,
        n_heads=16,
        dropout=0.1,
        attention_dropout=0.1,
        gelu_activation=True,
        sinusoidal_embeddings=False,
        causal=False,
        asm=False,
        n_langs=1,
        use_lang_emb=True,
        max_position_embeddings=512,
        embed_init_std=2048 ** -0.5,
        layer_norm_eps=1e-12,
        init_std=0.02,
        bos_index=0,
        eos_index=1,
        pad_index=2,
        unk_index=3,
        mask_index=5,
        is_encoder=True,
        summary_type="first",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        start_n_top=5,
        end_n_top=5,
        mask_token_id=0,
        lang_id=0,
        pad_token_id=2,
        bos_token_id=0,
        **kwargs
    ):
        """Constructs XLMConfig."""
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.gelu_activation = gelu_activation
        self.sinusoidal_embeddings = sinusoidal_embeddings
        self.causal = causal
        self.asm = asm
        self.n_langs = n_langs
        self.use_lang_emb = use_lang_emb
        self.layer_norm_eps = layer_norm_eps
        self.bos_index = bos_index
        self.eos_index = eos_index
        self.pad_index = pad_index
        self.unk_index = unk_index
        self.mask_index = mask_index
        self.is_encoder = is_encoder
        self.max_position_embeddings = max_position_embeddings
        self.embed_init_std = embed_init_std
        self.init_std = init_std
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_proj_to_labels = summary_proj_to_labels
        self.summary_first_dropout = summary_first_dropout
        self.start_n_top = start_n_top
        self.end_n_top = end_n_top
        self.mask_token_id = mask_token_id
        self.lang_id = lang_id

        if "n_words" in kwargs:
            self.n_words = kwargs["n_words"]

    @property
    def n_words(self):  # For backward compatibility
        return self.vocab_size

    @n_words.setter
    def n_words(self, value):  # For backward compatibility
        self.vocab_size = value

    @property
    def hidden_size(self):
        return self.emb_dim

    @property
    def num_attention_heads(self):
        return self.n_heads

    @property
    def num_hidden_layers(self):
        return self.n_layers
