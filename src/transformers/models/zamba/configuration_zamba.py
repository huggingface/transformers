# Copyright 2024 Zyphra Technologies and the HuggingFace Inc. team. All rights reserved.
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
"""Zamba model configuration"""

import math

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="Zyphra/Zamba-7B-v1")
class ZambaConfig(PreTrainedConfig):
    r"""
    attention_hidden_size (`int`, *optional*):
        Dimension of the hidden representations of the inputs to the Attention layer.
    attention_head_dim (`int`, *optional*):
        Dimension of the attention head in the Transformer decoder.
    mamba_dt_rank (`Union[int,str]`, *optional*, defaults to `"auto"`):
        Rank of the mamba discretization projection matrix. `"auto"` means that it will default to `math.ceil(self.hidden_size / 16)`
    n_mamba_heads (`int`, *optional*, defaults to 2):
        Number of mamba heads for each mamba layer.
    hidden_mamba_act (`str` or `function`, *optional*, defaults to `"silu"`):
        The non-linear activation function (function or string) in the mamba layer.
    num_logits_to_keep (`int` or `None`, *optional*, defaults to 1):
        Number of prompt logits to calculate during generation. If `None`, all logits will be calculated. If an
        integer value, only last `num_logits_to_keep` logits will be calculated. Default is 1 because only the
        logits of the last prompt token are needed for generation. For long sequences, the logits for the entire
        sequence may use a lot of memory so, setting `num_logits_to_keep=1` will reduce memory footprint
        significantly.
    attn_layer_period (`int`, *optional*, defaults to 6):
        Once in this many layers, we will have a shared attention layer
    attn_layer_offset (`int`, *optional*, defaults to 4):
        Offset of the shared attention layer
    use_mamba_kernels (`bool`, *optional*, defaults to `True`):
        Flag indicating whether or not to use the fast mamba kernels. These are available only if `mamba-ssm` and
        `causal-conv1d` are installed, and the mamba modules are running on a CUDA device. Raises ValueError if
        `True` and kernels are not available
    """

    model_type = "zamba"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        tie_word_embeddings=True,
        hidden_size=3712,
        attention_hidden_size=None,
        intermediate_size=14848,
        num_hidden_layers=76,
        num_attention_heads=16,
        attention_head_dim=None,
        num_key_value_heads=16,
        n_mamba_heads=2,
        hidden_act="gelu",
        hidden_mamba_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        num_logits_to_keep=1,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        max_position_embeddings=4096,
        attention_dropout=0.0,
        attn_layer_period=6,
        attn_layer_offset=4,
        use_mamba_kernels=True,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_dt_rank="auto",
        time_step_min=0.001,
        time_step_max=0.1,
        time_step_floor=1e-4,
        mamba_conv_bias=True,
        mamba_proj_bias=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.tie_word_embeddings = tie_word_embeddings
        self.hidden_size = hidden_size
        if attention_hidden_size is None:
            self.attention_hidden_size = 2 * hidden_size
        else:
            self.attention_hidden_size = attention_hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        if attention_head_dim is None:
            self.attention_head_dim = 2 * self.hidden_size // self.num_attention_heads
        else:
            self.attention_head_dim = attention_head_dim
        self.max_position_embeddings = max_position_embeddings
        self.attention_dropout = attention_dropout

        self.num_key_value_heads = num_key_value_heads
        self.n_mamba_heads = n_mamba_heads
        self.hidden_act = hidden_act
        self.hidden_mamba_act = hidden_mamba_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps

        self.use_cache = use_cache
        self.num_logits_to_keep = num_logits_to_keep

        self.attn_layer_period = attn_layer_period
        self.attn_layer_offset = attn_layer_offset

        self.use_mamba_kernels = use_mamba_kernels
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_dt_rank = math.ceil(self.hidden_size / 16) if mamba_dt_rank == "auto" else mamba_dt_rank
        self.time_step_min = time_step_min
        self.time_step_max = time_step_max
        self.time_step_floor = time_step_floor
        self.mamba_conv_bias = mamba_conv_bias
        self.mamba_proj_bias = mamba_proj_bias

        self.layers_block_type = self._layers_block_type(num_hidden_layers, attn_layer_period, attn_layer_offset)

        assert (self.mamba_expand * self.hidden_size) % self.n_mamba_heads == 0, (
            "`intermediate_size` should be divisible by `n_mamba_heads`."
        )

        self.tie_word_embeddings = tie_word_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        super().__init__(**kwargs)

    def _layers_block_type(self, num_hidden_layers, attn_layer_period, attn_layer_offset):
        layers = [
            "mamba",
            "mamba",
            "hybrid",
        ] + ["hybrid" if i % attn_layer_period == attn_layer_offset else "mamba" for i in range(num_hidden_layers - 3)]
        return layers


__all__ = ["ZambaConfig"]
