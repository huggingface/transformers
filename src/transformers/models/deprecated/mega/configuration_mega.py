# coding=utf-8
# Copyright 2023 The Mega Authors and The HuggingFace Inc. team.
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
"""MEGA configuration"""

from collections import OrderedDict
from typing import Mapping

from ....configuration_utils import PretrainedConfig
from ....onnx import OnnxConfig
from ....utils import logging


logger = logging.get_logger(__name__)


class MegaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MegaModel`]. It is used to instantiate a Mega
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Mega
    [mnaylor/mega-base-wikitext](https://huggingface.co/mnaylor/mega-base-wikitext) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the Mega model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`MegaModel`].
        hidden_size (`int`, *optional*, defaults to 128):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 4):
            Number of hidden layers in the Mega encoder.
        intermediate_size (`int`, *optional*, defaults to 256):
            Dimensionality of the hidden size (self-attention value projection) within the Mega encoder
        ema_projection_size (`int`, *optional*, defaults to 16):
            Dimensionality of the MegaMultiDimensionDampedEma
        bidirectional (`bool`, *optional*, defaults to `True`):
            Whether the MegaMultiDimensionDampedEma used in Mega's self-attention should work bidirectionally (`True`)
            or unidirectionally (`False`). Bidirectional EMA is incompatible with causal decoding, so this should be
            False if you intend to use the model as a decoder.
        shared_representation_size (`int`, *optional*, defaults to 64):
            Dimensionality of the linear projection for shared representation of self-attention queries and keys
        use_chunking (`bool`, *optional*, defaults to `False`):
            Whether to chunk inputs for linear self-attention complexity (described as Mega-chunk in the paper)
        chunk_size (`int`, *optional*, defaults to -1):
            If `use_chunking` is set to `True`, determines the size of the chunks to apply to the input sequence. If
            chunking is used, input sequences must be padded to a multiple of `chunk_size`
        truncation (`int`, *optional*):
            If specified, the sequence length for which to truncate MegaMultiDimensionDampedEma
        normalize_before_mega (`bool`, *optional*, defaults to `True`):
            Whether to normalize before (`True`) or after (`False`) passing through Mega encoder blocks
        normalization_type (`str`, *optional*, defaults to `"scalenorm"`):
            Type of normalization to use in Mega encoder blocks. Choose one of `"scalenorm"`, `"layernorm"`,
            `"rmsnorm"`, `"batchnorm"`, or `"syncbatchnorm"` (GPU required for syncbatchnorm)
        norm_affine (`bool`, *optional*, defaults to `True`):
            If `True`, applies a parameterized affine transformation to inputs during normalization
        activation (`str`, *optional*, defaults to `"silu"`):
            Activation function to apply within Mega encoder blocks. Choose one of `"silu"`, `"relu"`, `"linear"`,
            `"gelu"`, or `"gelu_accurate"`
        attention_activation (`str`, *optional*, defaults to `"softmax"`):
            Activation function to apply for single-headed self-attention (a la Transformer). Choose one of
            `"softmax"`, `"laplace"`, or `"relu2"`
        dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for EMA self-attention
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        use_feature_dropout (`bool`, *optional*, defaults to `False`):
            Whether to use feature-based (`True`) or standard dropout (`False`)
        use_normalized_ffn (`bool`, *optional*, defaults to `True`):
            Whether to use the normalized feed-forward sub-layer in Mega blocks (`True`) or pass Mega encoder output
            as-is (`False`)
        nffn_hidden_size (`int`, *optional*, defaults to 256):
            If using the normalized feed-forward network (NFFN) layer within Mega (`use_normalized_ffn = True`), this
            is the hidden size of the NFFN
        normalize_before_ffn (`bool`, *optional*, defaults to `True`):
            Whether to normalize before (`True`) or after (`False`) the feed-forward portion of NFFN
        nffn_activation_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the NFFN component.
        max_positions (`int`, *optional*, defaults to 2048):
            The maximum sequence length to use for positional representations. For `"simple"` relative positional bias,
            this is a hard limit on input length; `"rotary"` relative positional bias will extrapolate to longer
            sequences
        add_token_type_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to account for token types in embeddings. Left as optional to maintain compatibility with original
            implementation while adding support for token types.
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`MegaModel`]. Only used if
            `add_token_type_embeddings = True`
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        ema_delta_alpha_range (`float`, *optional*, defaults to 0.2):
            The standard deviation for initializing the delta (damping factor) and alpha (decay factor) parameters in
            MegaMultiDimensionDampedEma.
        ema_beta_range (`float`, *optional*, defaults to 0.02):
            The standard deviation for initializing the beta parameter (expansion matrix) in
            MegaMultiDimensionDampedEma.
        ema_gamma_omega_range (`float`, *optional*, defaults to 1.0):
            The standard deviation for initializing the gamma (projection matrix) and omega (residual weight)
            parameters in MultiDimensionEMA.
        relative_positional_bias (`str`, *optional*, defaults to `"rotary"`):
            Type of relative positional encoding. Choose one of `"rotary"` or `"simple"`. If `"simple"` is selected,
            `max_positions` is used as a limit on input size, while `"rotary"` extrapolates beyond `max_positions`.
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.
        add_lm_hidden_dense_layer (`bool`, *optional*, defaults to `True`):
            Whether to include a hidden layer for projection between encoder outputs and LM heads (`True`) or pass
            hidden states directly to LM head (`False`). Remains optional for compatibility with original
            implementation

    Examples:

    ```python
    >>> from transformers import MegaConfig, MegaModel

    >>> # Initializing a Mega configuration
    >>> configuration = MegaConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = MegaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "mega"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=128,
        num_hidden_layers=4,
        intermediate_size=256,
        ema_projection_size=16,
        bidirectional=True,
        shared_representation_size=64,
        use_chunking=False,
        chunk_size=-1,
        truncation=None,
        normalize_before_mega=True,
        normalization_type="scalenorm",
        norm_affine=True,
        activation="silu",
        attention_activation="softmax",
        dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        use_feature_dropout=False,
        use_normalized_ffn=True,
        nffn_hidden_size=256,
        normalize_before_ffn=True,
        nffn_activation_dropout_prob=0.1,
        max_positions=2048,
        add_token_type_embeddings=False,
        type_vocab_size=2,
        initializer_range=0.02,
        ema_delta_alpha_range=0.2,
        ema_beta_range=0.02,
        ema_gamma_omega_range=1.0,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        relative_positional_bias="rotary",
        classifier_dropout=None,
        use_cache=True,
        add_lm_hidden_dense_layer=True,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.activation = activation
        self.attention_activation = attention_activation
        self.intermediate_size = intermediate_size
        self.ema_projection_size = ema_projection_size
        self.bidirectional = bidirectional
        self.shared_representation_size = shared_representation_size
        self.use_chunking = use_chunking
        self.chunk_size = chunk_size
        self.truncation = truncation
        self.normalize_before_mega = normalize_before_mega
        self.normalization_type = normalization_type
        self.norm_affine = norm_affine
        self.dropout_prob = dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.use_feature_dropout = use_feature_dropout
        self.use_normalized_ffn = use_normalized_ffn
        self.nffn_hidden_size = nffn_hidden_size
        self.normalize_before_ffn = normalize_before_ffn
        self.nffn_activation_dropout_prob = nffn_activation_dropout_prob
        self.max_positions = max_positions
        self.add_token_type_embeddings = add_token_type_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.ema_delta_alpha_range = ema_delta_alpha_range
        self.ema_beta_range = ema_beta_range
        self.ema_gamma_omega_range = ema_gamma_omega_range
        self.relative_positional_bias = relative_positional_bias
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.add_lm_hidden_dense_layer = add_lm_hidden_dense_layer
        self.num_attention_heads = 1  # not used but required by Hugging Face


class MegaOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            dynamic_axis = {0: "batch", 1: "sequence"}
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),
                ("attention_mask", dynamic_axis),
            ]
        )


__all__ = ["MegaConfig", "MegaOnnxConfig"]
