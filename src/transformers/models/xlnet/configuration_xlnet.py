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

import warnings

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "xlnet-base-cased": "https://huggingface.co/xlnet-base-cased/resolve/main/config.json",
    "xlnet-large-cased": "https://huggingface.co/xlnet-large-cased/resolve/main/config.json",
}


class XLNetConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~transformers.XLNetModel` or a
    :class:`~transformers.TFXLNetModel`. It is used to instantiate a XLNet model according to the specified arguments,
    defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration
    to that of the `xlnet-large-cased <https://huggingface.co/xlnet-large-cased>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 32000):
            Vocabulary size of the XLNet model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.XLNetModel` or
            :class:`~transformers.TFXLNetModel`.
        d_model (:obj:`int`, `optional`, defaults to 1024):
            Dimensionality of the encoder layers and the pooler layer.
        n_layer (:obj:`int`, `optional`, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        n_head (:obj:`int`, `optional`, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        d_inner (:obj:`int`, `optional`, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        ff_activation (:obj:`str` or :obj:`Callable`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the If string, :obj:`"gelu"`, :obj:`"relu"`,
            :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        untie_r (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to untie relative position biases
        attn_type (:obj:`str`, `optional`, defaults to :obj:`"bi"`):
            The attention type used by the model. Set :obj:`"bi"` for XLNet, :obj:`"uni"` for Transformer-XL.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        dropout (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        mem_len (:obj:`int` or :obj:`None`, `optional`):
            The number of tokens to cache. The key/value pairs that have already been pre-computed in a previous
            forward pass won't be re-computed. See the `quickstart
            <https://huggingface.co/transformers/quickstart.html#using-the-past>`__ for more information.
        reuse_len (:obj:`int`, `optional`):
            The number of tokens in the current batch to be cached and reused in the future.
        bi_data (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to use bidirectional input pipeline. Usually set to :obj:`True` during pretraining and
            :obj:`False` during finetuning.
        clamp_len (:obj:`int`, `optional`, defaults to -1):
            Clamp all relative distances larger than clamp_len. Setting this attribute to -1 means no clamping.
        same_length (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to use the same attention length for each token.
        summary_type (:obj:`str`, `optional`, defaults to "last"):
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
        summary_proj_to_labels (:obj:`boo`, `optional`, defaults to :obj:`True`):
            Used in the sequence classification and multiple choice models.

            Whether the projection outputs should have :obj:`config.num_labels` or :obj:`config.hidden_size` classes.
        summary_last_dropout (:obj:`float`, `optional`, defaults to 0.1):
            Used in the sequence classification and multiple choice models.

            The dropout ratio to be used after the projection and activation.
        start_n_top (:obj:`int`, `optional`, defaults to 5):
            Used in the SQuAD evaluation script.
        end_n_top (:obj:`int`, `optional`, defaults to 5):
            Used in the SQuAD evaluation script.
        use_mems_eval (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should make use of the recurrent memory mechanism in evaluation mode.
        use_mems_train (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the model should make use of the recurrent memory mechanism in train mode.

            .. note::
                For pretraining, it is recommended to set ``use_mems_train`` to :obj:`True`. For fine-tuning, it is
                recommended to set ``use_mems_train`` to :obj:`False` as discussed `here
                <https://github.com/zihangdai/xlnet/issues/41#issuecomment-505102587>`__. If ``use_mems_train`` is set
                to :obj:`True`, one has to make sure that the train batches are correctly pre-processed, `e.g.`
                :obj:`batch_1 = [[This line is], [This is the]]` and :obj:`batch_2 = [[ the first line], [ second
                line]]` and that all batches are of equal size.

    Examples::

        >>> from transformers import XLNetConfig, XLNetModel

        >>> # Initializing a XLNet configuration
        >>> configuration = XLNetConfig()

        >>> # Initializing a model from the configuration
        >>> model = XLNetModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "xlnet"
    keys_to_ignore_at_inference = ["mems"]

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
        mem_len=512,
        reuse_len=None,
        use_mems_eval=True,
        use_mems_train=False,
        bi_data=False,
        clamp_len=-1,
        same_length=False,
        summary_type="last",
        summary_use_proj=True,
        summary_activation="tanh",
        summary_last_dropout=0.1,
        start_n_top=5,
        end_n_top=5,
        pad_token_id=5,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs
    ):
        """Constructs XLNetConfig."""
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_head = n_head
        assert d_model % n_head == 0
        if "d_head" in kwargs:
            assert (
                kwargs["d_head"] == d_model // n_head
            ), f"`d_head` ({kwargs['d_head']}) should be equal to `d_model // n_head` ({d_model // n_head})"
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
        self.eos_token_id = eos_token_id

        if "use_cache" in kwargs:
            warnings.warn(
                "The `use_cache` argument is deprecated and will be removed in a future version, use `use_mems_eval` instead.",
                FutureWarning,
            )
            use_mems_eval = kwargs["use_cache"]

        self.use_mems_eval = use_mems_eval
        self.use_mems_train = use_mems_train

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
