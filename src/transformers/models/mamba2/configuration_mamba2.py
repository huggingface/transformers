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

from typing import List

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class Mamba2Config(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`Mamba2Model`]. It is used to instantiate a MAMBA2
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the MAMBA2
    [state-spaces/mamba2-130m](https://huggingface.co/state-spaces/mamba2-130m) architecture.

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
        expand (`int`, *optional*, defaults to 2):
            Expanding factor used to determine the intermediate size.
        chunk_size (`int`, *optional*, defaults to 256):
            Block / Chunk size for the HW-efficient algorithm which parallelizes on intra- and inter-chunk calculations.
        mamba2_conv_kernel (`int`, *optional*, defaults to 4):
            Size of the convolution kernel in the mamba2 mixer.
        attention_conv_kernel (`int`, *optional*, defaults to 4):
            Size of the convolution kernel in the attention block.
        mlp_intermediate_size (`int`, *optional*, defaults to 0):
            Dimensionality of up-projections within the MLP blocks. If set to <=0, then MLP blocks are disabled.
        mlp_padding_size (`int`, *optional*, defaults to 128):
            Padding `mlp_intermediate_size` to a multiple of this.
        mamba2_head_dim (`int`, *optional*, defaults to 64):
            Multi-input SSM head dimension.
        attention_head_dim (`int`, *optional*, defaults to 128):
            Multi-head attention's head dimension.
        num_attention_heads (`int`, *optional*, defaults to 30):
            The number of heads in multi-head attention.
        num_key_value_heads (`int`, *optional*, defaults to 30):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `attention_num_key_value_heads=attention_num_heads`, the model will use Multi Head Attention (MHA), if
            `attention_num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `attention_num_heads`.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the model.
        attention_layers_idx (`List[int]`, *optional*, defaults to `[]`):
            The specific layers that exchange the mamba2 mixer block with the attention equivalent.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-05):
            The epsilon to use in the layer normalization layers.
        use_conv_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to use bias in the convolution layer of the mixer block.
        use_mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether or not to use a bias in the up- and downprojections of the MLP block.
        use_mamba2_bias (`bool`, *optional*, defaults to `False`):
            Whether or not to use bias in ["in_proj", "out_proj"] of the mamba2 mixer block.
        use_attention_qkv_bias (`bool`, *optional*, defaults to `False`):
            Whether or not to use bias in the qkv projection of the attention block.
        use_attention_out_bias (`bool`, *optional*, defaults to `False`):
            Whether or not to use bias in the out projection of the attention block.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        emb_initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing the embedding weight matrix.
        conv_initializer_range (`float`, *optional*):
            The range for uniformly initializing the convolution weights.
        A_initializer_range (`List[int]`, *optional*, defaults to `[1, 16]`):
            The range for uniformly initializing the 1-SS(a) scalar.
        time_step_min (`float`, *optional*, defaults to 0.001):
            Minimum `time_step` used to bound `dt_proj.bias`.
        time_step_max (`float`, *optional*, defaults to 0.1):
            Maximum `time_step` used to bound `dt_proj.bias`.
        time_step_floor (`float`, *optional*, defaults to 0.0001):
            Minimum clamping value of the `dt_proj.bias` layer initialization.
        time_step_limit (`List[float]`, *optional*, defaults to `[0.0, inf]`):
            Clapping values for the dt weights.
        residual_in_fp32 (`bool`, *optional*, defaults to `True`):
            Whether or not residuals should be in `float32`. If set to `False` residuals will keep the same `dtype` as the rest of the model
        rescale_prenorm_residual (`bool`, *optional*, defaults to `False`):
            Whether or not to rescale `out_proj` weights when initializing.
        rope_emb_dim (`int`, *optional*, defaults to 64):
            Embedding dimension of the RoPE embeddings.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
            these scaling strategies behave:
            https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
            experimental feature, subject to breaking API changes in future versions.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. This is based on the context length the
            Mamba2 models have been trained on. Also necessary when using any sort of RoPE embeddings.
        tie_embedding_weights (`bool`, *optional*, defaults to `True`):
            Whether or not to tie the lm head to the input embeddings.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the cache should be used.
        classifier_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the classification head in [`Mamba2ForSequenceClassification`] model.

    Example:

    ```python
    >>> from transformers import Mamba2Config, Mamba2Model

    >>> # Initializing a Mamba2 configuration
    >>> configuration = Mamba2Config()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = Mamba2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "mamba2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=50280,
        pad_token_id=0,
        bos_token_id=0,
        eos_token_id=0,
        hidden_size=768,
        state_size=128,
        expand=2,
        chunk_size=256,
        mamba2_conv_kernel=4,
        attention_conv_kernel=4,
        mlp_intermediate_size=0,
        mlp_padding_size=128,
        mamba2_head_dim=64,
        attention_head_dim=128,
        num_attention_heads=30,
        num_key_value_heads=30,
        num_hidden_layers=24,
        attention_layers_idx=None,
        layer_norm_epsilon=1e-5,
        use_conv_bias=True,
        use_mlp_bias=False,
        use_mamba2_bias=False,
        use_attention_qkv_bias=False,
        use_attention_out_bias=False,
        hidden_act="silu",
        emb_initializer_range=0.02,
        conv_initializer_range=None,
        A_initializer_range=None,
        time_step_min=0.001,
        time_step_max=0.1,
        time_step_floor=1e-4,
        time_step_limit=None,
        residual_in_fp32=True,
        rescale_prenorm_residual=False,
        rope_emb_dim=64,
        rope_theta=10000.0,
        rope_scaling=None,
        max_position_embeddings=2048,
        tie_embedding_weights=True,
        use_cache=True,
        classifier_dropout=0.1,
        **kwargs,
    ):
        # Avoid mutable default args
        attention_layers_idx = [] if attention_layers_idx is None else attention_layers_idx
        A_initializer_range = [1, 16] if A_initializer_range is None else A_initializer_range
        time_step_limit = [0.0, float("inf")] if time_step_limit is None else time_step_limit

        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.expand = expand
        self.intermediate_size = int(expand * self.hidden_size)
        self.chunk_size = chunk_size
        self.mamba2_conv_kernel = mamba2_conv_kernel
        self.attention_conv_kernel = attention_conv_kernel
        self.mlp_padding_size = mlp_padding_size
        self.mlp_intermediate_size = mlp_intermediate_size
        self.mamba2_head_dim = mamba2_head_dim
        self.mamba2_num_heads = self.intermediate_size // self.mamba2_head_dim
        self.attention_head_dim = attention_head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.attention_layers_idx = attention_layers_idx
        self._attention_layers_idx_validation()
        self.layer_norm_epsilon = layer_norm_epsilon
        self.use_conv_bias = use_conv_bias
        self.use_mlp_bias = use_mlp_bias
        self.use_mamba2_bias = use_mamba2_bias
        self.use_attention_qkv_bias = use_attention_qkv_bias
        self.use_attention_out_bias = use_attention_out_bias
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
        self.rope_emb_dim = rope_emb_dim
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        if self.rope_emb_dim > 0:
            self._rope_scaling_validation()
        self.max_position_embeddings = max_position_embeddings
        self.tie_embedding_weights = tie_embedding_weights
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_embedding_weights=tie_embedding_weights,
            **kwargs,
        )

    # Copied from transformers.models.llama.configuration_llama.LlamaConfig._rope_scaling_validation
    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with two fields, `type` and `factor`, " f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")

    def _attention_layers_idx_validation(self):
        """
        Validate the `attention_layers_idx` configuration.
        """
        if isinstance(self.attention_layers_idx, list) and len(self.attention_layers_idx) == 0:
            return

        if not isinstance(self.attention_layers_idx, List) and all(
            isinstance(x, int) for x in self.attention_layers_idx
        ):
            raise ValueError(
                "`attention_layers_idx` must be a list of integers indicating the attention layers, "
                f"got {self.attention_layers_idx}"
            )

        if min(self.attention_layers_idx) < 0 or max(self.attention_layers_idx) >= self.num_hidden_layers:
            raise ValueError(
                "`attention_layers_idx` has out-of-range indices, "
                f"got {self.attention_layers_idx}, but expected indices in {list(range(self.num_hidden_layers))}"
            )
