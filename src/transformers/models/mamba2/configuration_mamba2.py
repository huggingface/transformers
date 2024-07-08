# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
"""MAMBA2 configuration"""

import math

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class Mamba2Config(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`Mamba2Model`]. It is used to instantiate a MAMBA2
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the MAMBA2
    [state-spaces/mamba2-2.7b](https://huggingface.co/state-spaces/mamba2-2.7b) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 50280):
            Vocabulary size of the MAMBA2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Mamba2Model`].
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 0):
            The id of the beginning of sentence token in the vocabulary.
        eos_token_id (`int`, *optional*, defaults to 0):
            The id of the end of sentence token in the vocabulary.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the embeddings and hidden states.
        state_size (`int`, *optional*, defaults to 128):
            Shape of the state space latents.
        head_dim (`int`, *optional*, defaults to 64):
            Multi-input SSM head dimension.
        chunk_size (`int`, *optional*, defaults to 256):
            Block / Chunk size for the HW-efficient algorithm which parallelizes on intra- and inter-chunk calculations.
        expand (`int`, *optional*, defaults to 2):
            Expanding factor used to determine the intermediate size.
        conv_kernel (`int`, *optional*, defaults to 4):
            Size of the convolution kernel.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the model.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-05):
            The epsilon to use in the layer normalization layers.
        use_bias (`bool`, *optional*, defaults to `False`):
            Whether or not to use bias in ["in_proj", "out_proj"] of the mixer block
        use_conv_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to use bias in the convolution layer of the mixer block.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        emb_initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing the embedding weight matrix.
        conv_initializer_range (`float`, *optional*, defaults to None):
            The range for uniformly initializing the convolution weights.
        A_initializer_range (`Tuple[int]`, *optional*, defaults to (1, 16)):
            The range for uniformly initializing the 1-SS(a) scalar.
        time_step_min (`float`, *optional*, defaults to 0.001):
            Minimum `time_step` used to bound `dt_proj.bias`.
        time_step_max (`float`, *optional*, defaults to 0.1):
            Maximum `time_step` used to bound `dt_proj.bias`.
        time_step_floor (`float`, *optional*, defaults to 0.0001):
            Minimum clamping value of the `dt_proj.bias` layer initialization.
        time_step_limit (`Tuple[float]`, *optional*, defaults to (0.0, float("inf"))):
            Clapping values for the dt weights.
        residual_in_fp32 (`bool`, *optional*, defaults to `True`):
            Whether or not residuals should be in `float32`. If set to `False` residuals will keep the same `dtype` as the rest of the model
        rescale_prenorm_residual (`bool`, *optional*, defaults to `False`):
            Whether or not to rescale `out_proj` weights when initializing.
        tie_embedding_weights (`bool`, *optional*, defaults to `True`):
            Whether or not to tie the lm head to the input embeddings.
        output_last_ssm_states (`bool`, *optional*, defaults to `False`):
            Whether or not return the last ssm states of each layer.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the cache should be used.
        use_triton_kernels (`bool`, *optional*, defaults to `True`):
            Whether or not to use the triton kernels.

    Example:

    ```python
    >>> from transformers import Mamba2Config, Mamba2Model

    >>> # Initializing a Mamba2 configuration
    >>> configuration = Mamba2Config()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = Mamba2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "mamba2"

    def __init__(
            self,
            vocab_size=50280,
            pad_token_id=0,
            bos_token_id=0,
            eos_token_id=0,
            hidden_size=768,
            state_size=128,
            head_dim=64,
            chunk_size=256,
            expand=2,
            conv_kernel=4,
            num_hidden_layers=32,
            layer_norm_epsilon=1e-5,
            use_bias=False,
            use_conv_bias=True,
            hidden_act="silu",
            emb_initializer_range=0.02,
            conv_initializer_range=None,
            A_initializer_range=(1, 16),
            time_step_min=0.001,
            time_step_max=0.1,
            time_step_floor=1e-4,
            time_step_limit=(0.0, float("inf")),
            residual_in_fp32=True,
            rescale_prenorm_residual=False,
            tie_embedding_weights=True,
            output_last_ssm_states=False,
            use_cache=True,
            use_triton_kernels=True,
            **kwargs,
    ):
        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.expand = expand
        self.conv_kernel = conv_kernel
        self.intermediate_size = int(expand * self.hidden_size)
        self.num_heads = self.intermediate_size // self.head_dim
        self.num_hidden_layers = num_hidden_layers
        self.layer_norm_epsilon = layer_norm_epsilon
        self.use_bias = use_bias
        self.use_conv_bias = use_conv_bias
        self.hidden_act = hidden_act
        self.emb_initializer_range = emb_initializer_range
        self.conv_initializer_range = conv_initializer_range
        self.A_initializer_range = A_initializer_range
        self.time_step_min = time_step_min
        self.time_step_max = time_step_max
        self.time_step_floor = time_step_floor
        self.time_step_limit = time_step_limit
        self.residual_in_fp32 = residual_in_fp32
        self.rescale_prenorm_residual = rescale_prenorm_residual
        self.tie_embedding_weights = tie_embedding_weights
        self.output_last_ssm_states = output_last_ssm_states
        self.use_cache = use_cache
        self.use_triton_kernels = use_triton_kernels

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, pad_token_id=pad_token_id, **kwargs)
