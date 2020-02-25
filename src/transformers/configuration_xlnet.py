# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
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
""" XLNet configuration """


import logging

from .configuration_utils import PretrainedConfig


logger = logging.getLogger(__name__)

XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "xlnet-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-base-cased-config.json",
    "xlnet-large-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-large-cased-config.json",
}


class XLNetConfig(PretrainedConfig):
    """
        This is the configuration class to store the configuration of a :class:`~transformers.XLNetModel`.
        It is used to instantiate an XLNet model according to the specified arguments, defining the model
        architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
        the `xlnet-large-cased <https://huggingface.co/xlnet-large-cased>`__ architecture.

        Configuration objects inherit from  :class:`~transformers.PretrainedConfig` and can be used
        to control the model outputs. Read the documentation from  :class:`~transformers.PretrainedConfig`
        for more information.

        Args:
            vocab_size (:obj:`int`, optional, defaults to 32000):
                Vocabulary size of the XLNet model. Defines the different tokens that
                can be represented by the `inputs_ids` passed to the forward method of :class:`~transformers.XLNetModel`.
            d_model (:obj:`int`, optional, defaults to 1024):
                Dimensionality of the encoder layers and the pooler layer.
            n_layer (:obj:`int`, optional, defaults to 24):
                Number of hidden layers in the Transformer encoder.
            n_head (:obj:`int`, optional, defaults to 16):
                Number of attention heads for each attention layer in the Transformer encoder.
            d_inner (:obj:`int`, optional, defaults to 4096):
                Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
            ff_activation (:obj:`string`, optional, defaults to "gelu"):
                The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            untie_r (:obj:`boolean`, optional, defaults to :obj:`True`):
                Untie relative position biases
            attn_type (:obj:`string`, optional, defaults to "bi"):
                The attention type used by the model. Set 'bi' for XLNet, 'uni' for Transformer-XL.
            initializer_range (:obj:`float`, optional, defaults to 0.02):
                The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            layer_norm_eps (:obj:`float`, optional, defaults to 1e-12):
                The epsilon used by the layer normalization layers.
            dropout (:obj:`float`, optional, defaults to 0.1):
                The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
            mem_len (:obj:`int` or :obj:`None`, optional, defaults to :obj:`None`):
                The number of tokens to cache. The key/value pairs that have already been pre-computed
                in a previous forward pass won't be re-computed. See the
                `quickstart <https://huggingface.co/transformers/quickstart.html#using-the-past>`__
                for more information.
            reuse_len (:obj:`int` or :obj:`None`, optional, defaults to :obj:`None`):
                The number of tokens in the current batch to be cached and reused in the future.
            bi_data (:obj:`boolean`, optional, defaults to :obj:`False`):
                Whether to use bidirectional input pipeline. Usually set to `True` during
                pretraining and `False` during finetuning.
            clamp_len (:obj:`int`, optional, defaults to -1):
                Clamp all relative distances larger than clamp_len.
                Setting this attribute to -1 means no clamping.
            same_length (:obj:`boolean`, optional, defaults to :obj:`False`):
                Whether to use the same attention length for each token.
            summary_type (:obj:`string`, optional, defaults to "last"):
                Argument used when doing sequence summary. Used in for the multiple choice head in
                :class:transformers.XLNetForSequenceClassification` and :class:`~transformers.XLNetForMultipleChoice`.
                Is one of the following options:
                    - 'last' => take the last token hidden state (like XLNet)
                    - 'first' => take the first token hidden state (like Bert)
                    - 'mean' => take the mean of all tokens hidden states
                    - 'cls_index' => supply a Tensor of classification token position (GPT/GPT-2)
                    - 'attn' => Not implemented now, use multi-head attention
            summary_use_proj (:obj:`boolean`, optional, defaults to :obj:`True`):
                Argument used when doing sequence summary. Used in for the multiple choice head in
                :class:`~transformers.XLNetForSequenceClassification` and :class:`~transformers.XLNetForMultipleChoice`.
                Add a projection after the vector extraction
            summary_activation (:obj:`string` or :obj:`None`, optional, defaults to :obj:`None`):
                Argument used when doing sequence summary. Used in for the multiple choice head in
                :class:`~transformers.XLNetForSequenceClassification` and :class:`~transformers.XLNetForMultipleChoice`.
                'tanh' => add a tanh activation to the output, Other => no activation.
            summary_proj_to_labels (:obj:`boolean`, optional, defaults to :obj:`True`):
                Argument used when doing sequence summary. Used in for the multiple choice head in
                :class:`~transformers.XLNetForSequenceClassification` and :class:`~transformers.XLNetForMultipleChoice`.
                If True, the projection outputs to config.num_labels classes (otherwise to hidden_size). Default: False.
            summary_last_dropout (:obj:`float`, optional, defaults to 0.1):
                Argument used when doing sequence summary. Used in for the multiple choice head in
                :class:`~transformers.XLNetForSequenceClassification` and :class:`~transformers.XLNetForMultipleChoice`.
                Add a dropout after the projection and activation
            start_n_top (:obj:`int`, optional, defaults to 5):
                Used in the SQuAD evaluation script for XLM and XLNet.
            end_n_top (:obj:`int`, optional, defaults to 5):
                Used in the SQuAD evaluation script for XLM and XLNet.

        Example::

            from transformers import XLNetConfig, XLNetModel

            # Initializing a XLNet configuration
            configuration = XLNetConfig()

            # Initializing a model from the configuration
            model = XLNetModel(configuration)

            # Accessing the model configuration
            configuration = model.config

        Attributes:
            pretrained_config_archive_map (Dict[str, str]):
                A dictionary containing all the available pre-trained checkpoints.
    """

    pretrained_config_archive_map = XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP
    model_type = "xlnet"

    def __init__(
        self,
        vocab_size=32000,
        d_model=1024,
        n_layer=24,
        n_head=16,
        d_inner=4096,
        ff_activation="gelu",
        untie_r=True,
        attn_type="bi",
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        dropout=0.1,
        mem_len=None,
        reuse_len=None,
        bi_data=False,
        clamp_len=-1,
        same_length=False,
        summary_type="last",
        summary_use_proj=True,
        summary_activation="tanh",
        summary_last_dropout=0.1,
        start_n_top=5,
        end_n_top=5,
        bos_token_id=1,
        pad_token_id=5,
        eos_token_id=2,
        **kwargs
    ):
        """Constructs XLNetConfig.
        """
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_head = n_head
        assert d_model % n_head == 0
        self.d_head = d_model // n_head
        self.ff_activation = ff_activation
        self.d_inner = d_inner
        self.untie_r = untie_r
        self.attn_type = attn_type

        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

        self.dropout = dropout
        self.mem_len = mem_len
        self.reuse_len = reuse_len
        self.bi_data = bi_data
        self.clamp_len = clamp_len
        self.same_length = same_length

        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_last_dropout = summary_last_dropout
        self.start_n_top = start_n_top
        self.end_n_top = end_n_top

        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_ids = [eos_token_id]

    @property
    def max_position_embeddings(self):
        return -1

    @property
    def n_token(self):  # Backward compatibility
        return self.vocab_size

    @n_token.setter
    def n_token(self, value):  # Backward compatibility
        self.vocab_size = value

    @property
    def hidden_size(self):
        return self.d_model

    @property
    def num_attention_heads(self):
        return self.n_head

    @property
    def num_hidden_layers(self):
        return self.n_layer
