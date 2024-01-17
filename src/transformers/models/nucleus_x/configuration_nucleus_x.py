# This code was adapted from Sehyun Choi's https://github.com/syncdoth/retnet
# licensed under the MIT License. Above repository builds on Microsoft's
# torchscale library and the RetNet implementations in this library, also
# licensed under the MIT License. It has been modified from its original forms
# to accommodate minor architectural differences compared to NucleusX used by
# the NucleusAI team that trained the model.

# MIT License
#
# Copyright (c) 2023  NucleusAI and The HuggingFace Inc. team and Sehyun Choi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
""" NucleusX model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

NUCLEUS_X_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "NucleusAI/Nucleus-X": "https://huggingface.co/NucleusAI/Nucleus-X/resolve/main/config.json",
}


class NucleusXConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~NucleusXModel`]. It is used to instantiate an
    NucleusX model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the NucleusX-7B
    [NucleusAI/Nucleus-X](https://huggingface.co/NucleusAI/Nucleus-X) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the NucleusX model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`~NucleusXModel`]
        initializer_factor (`float`, *optional*, defaults to `2**-2.5`):
            The gain for initializing linear projection weights using xavier uniform.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        lm_head_initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing the lm_head weights.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        forward_mode (`str`, *optional*, defaults to `"parallel"`):
            The three modes of Retentive Network: `"parallel"`, `"recurrent"`, or `"chunkwise"`.
        activation_fn (`str` or `function`, *optional*, defaults to `"swish"`):
            The non-linear activation function (function or string) in the decoder.
        dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability
        activation_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability after activation in FFN.
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            Drop Path (or Stochastic Depth, https://arxiv.org/abs/1603.09382) rate. Uses timm implementation (see
            https://github.com/huggingface/pytorch-image-models).
        decoder_embed_dim (`int`, *optional*, defaults to 4096):
            Decoder embedding dimension.
        decoder_value_embed_dim (`int`, *optional*, defaults to 6912):
            Decoder value embedding dimension.
        decoder_ffn_embed_dim (`int`, *optional*, defaults to 6912):
            Decoder embedding dimension for FFN.
        decoder_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        decoder_retention_heads (`int`, *optional*, defaults to 16):
            Number of decoder retention heads.
        decoder_normalize_before (`bool`, *optional*, defaults to `True`):
            Apply rms_norm before each decoder block
        rms_norm_embedding (`bool`, *optional*, defaults to `True`):
            Add rms_norm to embedding
        no_scale_embedding (`bool`, *optional*, defaults to `False`):
            If True, dont scale embeddings
        recurrent_chunk_size (`int`, *optional*, defaults to 512):
            The chunk size for `"chunkwise"` mode.
        use_lm_decay (`bool`, *optional*, defaults to `False`):
            Whether to use language model decay. (Found in https://arxiv.org/pdf/2307.08621.pdf, Page 7, Section 3.1,
            Paragraph **Parameter Allocation**, last sentence)
        z_loss_coeff (`float`, *optional*, defaults to 0.0):
            coefficient for z-loss (Used in PaLM, https://arxiv.org/pdf/2204.02311.pdf)
        deepnorm (`bool`, *optional*, defaults to `False`):
            DeepNorm (https://arxiv.org/abs/2203.00555). Disables `decoder_normalize_before`.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        groupnorm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the group_norm layer in NucleusXMultiScaleRetention.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings


    ```python
    >>> from transformers import NucleusXModel, NucleusXConfig

    >>> # Initializing a NucleusX-7B style configuration
    >>> configuration = NucleusXConfig(decoder_layers=2)  # only 2 layers for quick & small example

    >>> # Initializing a model from the NucleusX-7B style configuration
    >>> model = NucleusXModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "nucleus_x"
    attribute_map = {
        "hidden_size": "decoder_embed_dim",
        "intermediate_size": "decoder_ffn_embed_dim",
        "num_attention_heads": "decoder_retention_heads",
        "num_hidden_layers": "decoder_layers",
    }

    def __init__(
        self,
        vocab_size: int = 32000,
        initializer_factor: float = 2**-2.5,
        initializer_range: float = 0.02,
        lm_head_initializer_range: float = 4096**-0.5,
        pad_token_id: int = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        use_cache: bool = True,
        forward_mode: str = "parallel",
        activation_fn: str = "swish",
        dropout: float = 0.0,
        activation_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        decoder_embed_dim: int = 4096,
        decoder_value_embed_dim: int = 6912,
        decoder_ffn_embed_dim: int = 6912,
        decoder_layers: int = 32,
        decoder_retention_heads: int = 16,
        decoder_normalize_before: bool = True,
        rms_norm_embedding: bool = True,
        no_scale_embedding: bool = False,
        recurrent_chunk_size: int = 512,
        use_lm_decay: bool = False,
        z_loss_coeff: float = 0.0,
        deepnorm: bool = False,
        rms_norm_eps: float = 1e-6,
        groupnorm_eps: float = 1e-6,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.lm_head_initializer_range = lm_head_initializer_range
        # retentive network related
        self.use_lm_decay = use_lm_decay
        self.recurrent_chunk_size = recurrent_chunk_size
        self.forward_mode = forward_mode
        # size related
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_value_embed_dim = decoder_value_embed_dim
        self.decoder_retention_heads = decoder_retention_heads
        self.decoder_ffn_embed_dim = decoder_ffn_embed_dim
        self.decoder_layers = decoder_layers
        # normalization related
        self.decoder_normalize_before = decoder_normalize_before
        self.activation_fn = activation_fn
        self.dropout = dropout
        self.drop_path_rate = drop_path_rate
        self.activation_dropout = activation_dropout
        self.no_scale_embedding = no_scale_embedding
        self.rms_norm_embedding = rms_norm_embedding
        self.deepnorm = deepnorm
        self.rms_norm_eps = rms_norm_eps
        self.groupnorm_eps = groupnorm_eps
        self.z_loss_coeff = z_loss_coeff

        if self.deepnorm:
            self.decoder_normalize_before = False

        super().__init__(
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            use_cache=use_cache,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
