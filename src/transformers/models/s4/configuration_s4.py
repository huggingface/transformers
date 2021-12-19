# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
""" S4 model configuration """

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class S4Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.S4Model`. It is used to
    instantiate an S4 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the S4 `s4
    <https://huggingface.co/s4>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 267735):
            Vocabulary size of the S4 model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.S4Model` or :class:`~transformers.TFS4Model`.
        d_embed (:obj:`int`, `optional`, defaults to 1024):
            Dimensionality of the embeddings
        embedding_dropout_prob (:obj:`float`, `optional`, defaults to 0.25):
            The dropout probability for the embeddings.
        div_val (:obj:`int`, `optional`, defaults to 4):
            Divident value for adapative input and softmax
        cutoffs (:obj:`List[int]`, `optional`, defaults to :obj:`[19997, 39997, 199997]`):
            The cutoffs used to calculate the vocabulary size of the model.
        tie_weights (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to share the input and output embeddings.
        tie_projs (:obj:`List[bool]`, `optional`, defaults to :obj:`[True, True, True]`):
            Whether to share the input and output embedding projections.
        initializer_scale (:obj:`float`, `optional`, defaults to 0.5):
            The scale used to initialize the weights.
        bias_scale (:obj:`float`, `optional`, defaults to 1.0):
            The scale used to initialize the biases.
        input_dropout_prob (:obj:`float`, `optional`, defaults to 0.0):
            The dropout probability for the inputs.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.25):
            The dropout probability for the outputs.
        pre_norm (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to apply LayerNorm to the input instead of the output in the blocks.
        transposed (:obj:`bool`, `optional`, defaults to :obj:`True`):
            choose backbone axis ordering of (B, L, D) or (B, D, L) [B=batch size, L=sequence length, D=feature
            dimension]
        d_model (:obj:`int`, `optional`, defaults to 1024):
            Dimensionality of the s4 model's hidden states.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 16):
            Number of hidden layers in the s4 model.
        residual_connections (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to use residual connections.
        pool_size (:obj:`int`, `optional`, defaults to :obj:`1`):
            DownSample pooling size
        pool_expand (:obj:`int`, `optional`, defaults to :obj:`1`):
            DownSample expand size
        normalize_type (:obj:`str`, `optional`, defaults to :obj:`"layer"`):
            normalization type: "layer" or "batch"
        d_state (:obj:`int`, `optional`, defaults to 64):
            The dimension of the state used in S4.
        measure (:obj:`str`, `optional`, defaults to :obj:`"legs"`):
            Specifies the measure to use for the S4 model. legt - Legendre (translated), legs - Legendre (scaled),
            glagt - generalized Laguerre (translated), lagt, tlagt - previous versions of (tilted) Laguerre with
            slightly different normalization
        rank (:obj:`int`, `optional`, defaults to :obj:`1`):
            The rank of the measure.
        dt_min (:obj:`float`, `optional`, defaults to :obj:`0.001`):
            The discretization step size (minimum).
        dt_max (:obj:`float`, `optional`, defaults to :obj:`0.1`):
            The discretization step size (maximum).
        trainable_s4_params (:obj:`dict`, `optional`, defaults to :obj:`{"A":1, "B":2, "C":1, "dt":1}`):
            The parameters of the S4 model that are trainable.
        learning_rate_s4_params (:obj:`dict`, `optional`, defaults to :obj:`{"A":0.0005, "B":0.0005, "C":null "dt":0.0005}`):
            The learning rate of the S4 model parameters.
        cache (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to cache the SS Kernel during evaluation.
        weight_decay (:obj:`float`, `optional`, defaults to :obj:`0.0`):
            The weight decay on the SS Kernel.
        l_max (:obj:Union[:obj: 'int', :obj: 'bool'], `optional`, defaults to :obj:`1024`):
            The maximum sequence length, if known at model build time, otherwise :obj:`False`.
        activation_function (:obj:`str`, `optional`, defaults to :obj:`"gelu"`):
            The activation function to use.
        post_activation_function (:obj:`str`, `optional`, defaults to :obj:`"glu"`):
            The activation function to use after the linear projection.
        s4_dropout (:obj:`float`, `optional`, defaults to :obj:`0.25`):
            The dropout probability for the S4 model.
        ff_expand (:obj:`int`, `optional`, defaults to :obj:`4`):
            The expansion factor for the feed forward layer.
        ff_activation_function (:obj:`str`, `optional`, defaults to :obj:`"gelu"`):
            The activation function to use for the feed forward layer.
        ff_dropout (:obj:`float`, `optional`, defaults to :obj:`0.25`):
            The dropout probability for the feed forward layer.
        softmax_dropout_prob (:obj:`float`, `optional`, defaults to 0.25):
            The dropout probability for the softmax layer.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

        Example::

        >>> from transformers import S4Model, S4Config

        >>> # Initializing a S4 s4 style configuration
        >>> configuration = S4Config()

        >>> # Initializing a model from the s4 style configuration
        >>> model = S4Model(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = "s4"

    def __init__(
        self,
        vocab_size=267735,
        d_embed=1024,
        embedding_dropout_prob=0.25,
        div_val=4,
        cutoffs=[19997, 39997, 199997],
        tie_weights=True,
        tie_projs=[True, True, True],
        initializer_scale=0.5,
        bias_scale=1.0,
        input_dropout_prob=0.0,
        hidden_dropout_prob=0.25,
        pre_norm=True,
        transposed=True,
        d_model=1024,
        num_hidden_layers=16,
        residual_connections=True,
        pool_size=1,
        pool_expand=1,
        normalize_type="layer",
        d_state=64,
        measure="legs",
        rank=1,
        dt_min=0.001,
        dt_max=0.1,
        trainable_s4_params={"A": 1, "B": 2, "C": 1, "dt": 1},
        learning_rate_s4_params={"A": 0.0005, "B": 0.0005, "C": None, "dt": 0.0005},
        cache=False,
        weight_decay=0.0,
        l_max=1024,
        activation_function="gelu",
        post_activation_function="glu",
        s4_dropout=0.25,
        ff_expand=4,
        ff_activation_function="gelu",
        ff_dropout=0.25,
        softmax_dropout_prob=0.25,
        initializer_range=0.02,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_embed = d_embed
        self.embedding_dropout_prob = embedding_dropout_prob
        self.div_val = div_val
        self.cutoffs = cutoffs
        self.tie_weights = tie_weights
        self.tie_projs = tie_projs
        self.initializer_scale = initializer_scale
        self.bias_scale = bias_scale
        self.input_dropout_prob = input_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.pre_norm = pre_norm
        self.transposed = transposed
        self.d_model = d_model
        self.num_hidden_layers = num_hidden_layers
        self.residual_connections = residual_connections
        self.pool_size = pool_size
        self.pool_expand = pool_expand
        self.normalize_type = normalize_type
        self.d_state = d_state
        self.measure = measure
        self.rank = rank
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.trainable_s4_params = trainable_s4_params
        self.learning_rate_s4_params = learning_rate_s4_params
        self.cache = cache
        self.weight_decay = weight_decay
        self.l_max = l_max
        self.activation_function = activation_function
        self.post_activation_function = post_activation_function
        self.s4_dropout = s4_dropout
        self.ff_expand = ff_expand
        self.ff_activation_function = ff_activation_function
        self.ff_dropout = ff_dropout
        self.softmax_dropout_prob = softmax_dropout_prob
        self.initializer_range = initializer_range
        super().__init__(**kwargs)
