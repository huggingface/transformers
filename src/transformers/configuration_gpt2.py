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
""" OpenAI GPT-2 configuration """


import logging

from .configuration_utils import PretrainedConfig


logger = logging.getLogger(__name__)

GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "gpt2": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-config.json",
    "gpt2-medium": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-medium-config.json",
    "gpt2-large": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-large-config.json",
    "gpt2-xl": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-xl-config.json",
    "distilgpt2": "https://s3.amazonaws.com/models.huggingface.co/bert/distilgpt2-config.json",
}


class GPT2Config(PretrainedConfig):
    """
        This is the configuration class to store the configuration of a :class:`~transformers.GPT2Model`.
        It is used to instantiate an GPT-2 model according to the specified arguments, defining the model
        architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
        the GPT-2 `small <https://huggingface.co/gpt2>`__ architecture.

        Configuration objects inherit from  :class:`~transformers.PretrainedConfig` and can be used
        to control the model outputs. Read the documentation from  :class:`~transformers.PretrainedConfig`
        for more information.


        Args:
            vocab_size (:obj:`int`, optional, defaults to 50257):
                Vocabulary size of the GPT-2 model. Defines the different tokens that
                can be represented by the `inputs_ids` passed to the forward method of :class:`~transformers.GPT2Model`.
            n_positions (:obj:`int`, optional, defaults to 1024):
                The maximum sequence length that this model might ever be used with.
                Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
            n_ctx (:obj:`int`, optional, defaults to 1024):
                Dimensionality of the causal mask (usually same as n_positions).
            n_embd (:obj:`int`, optional, defaults to 768):
                Dimensionality of the embeddings and hidden states.
            n_layer (:obj:`int`, optional, defaults to 12):
                Number of hidden layers in the Transformer encoder.
            n_head (:obj:`int`, optional, defaults to 12):
                Number of attention heads for each attention layer in the Transformer encoder.
            resid_pdrop (:obj:`float`, optional, defaults to 0.1):
                The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
            embd_pdrop (:obj:`int`, optional, defaults to 0.1):
                The dropout ratio for the embeddings.
            attn_pdrop (:obj:`float`, optional, defaults to 0.1):
                The dropout ratio for the attention.
            layer_norm_epsilon (:obj:`float`, optional, defaults to 1e-5):
                The epsilon to use in the layer normalization layers
            initializer_range (:obj:`float`, optional, defaults to 16):
                The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            summary_type (:obj:`string`, optional, defaults to "cls_index"):
                Argument used when doing sequence summary. Used in for the multiple choice head in
                :class:`~transformers.GPT2DoubleHeadsModel`.
                Is one of the following options:
                    - 'last' => take the last token hidden state (like XLNet)
                    - 'first' => take the first token hidden state (like Bert)
                    - 'mean' => take the mean of all tokens hidden states
                    - 'cls_index' => supply a Tensor of classification token position (GPT/GPT-2)
                    - 'attn' => Not implemented now, use multi-head attention
            summary_use_proj (:obj:`boolean`, optional, defaults to :obj:`True`):
                Argument used when doing sequence summary. Used in for the multiple choice head in
                :class:`~transformers.GPT2DoubleHeadsModel`.
                Add a projection after the vector extraction
            summary_activation (:obj:`string` or :obj:`None`, optional, defaults to :obj:`None`):
                Argument used when doing sequence summary. Used in for the multiple choice head in
                :class:`~transformers.GPT2DoubleHeadsModel`.
                'tanh' => add a tanh activation to the output, Other => no activation.
            summary_proj_to_labels (:obj:`boolean`, optional, defaults to :obj:`True`):
                Argument used when doing sequence summary. Used in for the multiple choice head in
                :class:`~transformers.GPT2DoubleHeadsModel`.
                If True, the projection outputs to config.num_labels classes (otherwise to hidden_size). Default: False.
            summary_first_dropout (:obj:`float`, optional, defaults to 0.1):
                Argument used when doing sequence summary. Used in for the multiple choice head in
                :class:`~transformers.GPT2DoubleHeadsModel`.
                Add a dropout before the projection and activation

        Example::

            from transformers import GPT2Model, GPT2Config

            # Initializing a GPT2 configuration
            configuration = GPT2Config()

            # Initializing a model from the configuration
            model = GPT2Model(configuration)

            # Accessing the model configuration
            configuration = model.config

        Attributes:
            pretrained_config_archive_map (Dict[str, str]):
                A dictionary containing all the available pre-trained checkpoints.
    """

    pretrained_config_archive_map = GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP
    model_type = "gpt2"

    def __init__(
        self,
        vocab_size=50257,
        n_positions=1024,
        n_ctx=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
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
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
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
