# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" OpenAI GPT configuration """

from .configuration_utils import PretrainedConfig
from .utils import logging


logger = logging.get_logger(__name__)

OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "openai-gpt": "https://s3.amazonaws.com/models.huggingface.co/bert/openai-gpt-config.json"
}


class OpenAIGPTConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~transformers.OpenAIGPTModel` or a
    :class:`~transformers.TFOpenAIGPTModel`. It is used to instantiate a GPT model according to the specified
    arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar
    configuration to that of the `GPT <https://huggingface.co/openai-gpt>`__ architecture from OpenAI.

    Configuration objects inherit from  :class:`~transformers.PretrainedConfig` and can be used
    to control the model outputs. Read the documentation from  :class:`~transformers.PretrainedConfig`
    for more information.

    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 40478):
            Vocabulary size of the GPT-2 model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.OpenAIGPTModel` or
            :class:`~transformers.TFOpenAIGPTModel`.
        n_positions (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with.
            Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
        n_ctx (:obj:`int`, `optional`, defaults to 512):
            Dimensionality of the causal mask (usually same as n_positions).
        n_embd (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the embeddings and hidden states.
        n_layer (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        n_head (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        afn (:obj:`str` or :obj:`Callable`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler.
            If string, :obj:`"gelu"`, :obj:`"relu"`, :obj:`"swish"` and :obj:`"gelu_new"` are supported.
        resid_pdrop (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (:obj:`int`, `optional`, defaults to 0.1):
            The dropout ratio for the embeddings.
        attn_pdrop (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention.
        layer_norm_epsilon (:obj:`float`, `optional`, defaults to 1e-5):
            The epsilon to use in the layer normalization layers
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        predict_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not special tokens should be predicted when the model has a language modeling head.
        summary_type (:obj:`str`, `optional`, defaults to :obj:`"cls_index"`):
            Argument used when doing sequence summary, used in the models
            :class:`~transformers.OpenAIGPTDoubleHeadsModel` and :class:`~transformers.OpenAIGPTDoubleHeadsModel`.

            Has to be one of the following options:

                - :obj:`"last"`: Take the last token hidden state (like XLNet).
                - :obj:`"first"`: Take the first token hidden state (like BERT).
                - :obj:`"mean"`: Take the mean of all tokens hidden states.
                - :obj:`"cls_index"`: Supply a Tensor of classification token position (like GPT/GPT-2).
                - :obj:`"attn"`: Not implemented now, use multi-head attention.
        summary_use_proj (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Argument used when doing sequence summary, used in the models
            :class:`~transformers.OpenAIGPTDoubleHeadsModel` and :class:`~transformers.OpenAIGPTDoubleHeadsModel`.

            Whether or not to add a projection after the vector extraction.
        summary_activation (:obj:`str`, `optional`):
            Argument used when doing sequence summary, used in the models
            :class:`~transformers.OpenAIGPTDoubleHeadsModel` and :class:`~transformers.OpenAIGPTDoubleHeadsModel`.

            Pass :obj:`"tanh"` for a tanh activation to the output, any other value will result in no activation.
        summary_proj_to_labels (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Argument used when doing sequence summary, used in the models
            :class:`~transformers.OpenAIGPTDoubleHeadsModel` and :class:`~transformers.OpenAIGPTDoubleHeadsModel`.

            Whether the projection outputs should have :obj:`config.num_labels` or :obj:`config.hidden_size` classes.
        summary_first_dropout (:obj:`float`, `optional`, defaults to 0.1):
            Argument used when doing sequence summary, used in the models
            :class:`~transformers.OpenAIGPTDoubleHeadsModel` and :class:`~transformers.OpenAIGPTDoubleHeadsModel`.

            The dropout ratio to be used after the projection and activation.

    Examples::

        >>> from transformers import OpenAIGPTConfig, OpenAIGPTModel

        >>> # Initializing a GPT configuration
        >>> configuration = OpenAIGPTConfig()

        >>> # Initializing a model from the configuration
        >>> model = OpenAIGPTModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "openai-gpt"

    def __init__(
        self,
        vocab_size=40478,
        n_positions=512,
        n_ctx=512,
        n_embd=768,
        n_layer=12,
        n_head=12,
        afn="gelu",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        predict_special_tokens=True,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.afn = afn
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.predict_special_tokens = predict_special_tokens
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_first_dropout = summary_first_dropout
        self.summary_proj_to_labels = summary_proj_to_labels

    @property
    def max_position_embeddings(self):
        return self.n_positions

    @property
    def hidden_size(self):
        return self.n_embd

    @property
    def num_attention_heads(self):
        return self.n_head

    @property
    def num_hidden_layers(self):
        return self.n_layer
