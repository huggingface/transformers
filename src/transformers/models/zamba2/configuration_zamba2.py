# Copyright 2024 Zyphra Technologies and the HuggingFace Inc. team. All rights reserved.
#
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


from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="Zyphra/Zamba2-2.7B")
class Zamba2Config(PreTrainedConfig):
    r"""
    mamba_ngroups (`int`, *optional*, defaults to 1):
        Number of groups for the evolution matrices of mamba 2.
    n_mamba_heads (`int`, *optional*, defaults to 8):
        Number of heads for the evolution matrices of mamba 2.
    use_conv_bias (`bool`, *optional*, defaults to `True`):
        Whether or not to use bias in the convolution layer of the mixer block.
    chunk_size (`int`, *optional*, defaults to 256):
        Size of the chunks that will comprise the sequence.
    use_mem_eff_path (`bool`, *optional*, defaults to `False`):
        Whether or not to use the fused conv1d and scan in mamba2 layers.
    add_bias_linear (`bool`, *optional*, defaults to `False`):
        Flag indicating whether or not to use bias in various layers
    num_mem_blocks (`int`, *optional*, defaults to 1):
        Number of unshared transformer blocks.
    use_shared_attention_adapter (`bool`, *optional*, defaults to `False`):
        If True, unshared adapters (formally the same as LoRA but used in the base model) will be added to the q, k, v projectors in the shared attention layers.
    adapter_rank (`int`, *optional*, defaults to 128):
        Rank of the adapter in the shared MLP and shared attention layers.
    use_mem_rope (`bool`, *optional*, defaults to `False`):
        If True, includes RoPE in the shared attention layers.
    num_logits_to_keep (`int` or `None`, *optional*, defaults to 1):
        Number of prompt logits to calculate during generation. If `None`, all logits will be calculated. If an
        integer value, only last `num_logits_to_keep` logits will be calculated. Default is 1 because only the
        logits of the last prompt token are needed for generation. For long sequences, the logits for the entire
        sequence may use a lot of memory so, setting `num_logits_to_keep=1` will reduce memory footprint
        significantly.
    use_long_context (`bool`, *optional*, defaults to `False`):
        Activates the context-extended version of Zamba by modifying RoPE.

    Example:
    ```python
    >>> from transformers import Zamba2Model, Zamba2Config
    >>> # Initializing a Zamba2-2.7B style configuration
    >>> configuration = Zamba2Config()
    >>> # Initializing a model from the Zamba2-2.7B style configuration
    >>> model = Zamba2Model(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "zamba2"
    attribute_map = {"head_dim": "attention_head_dim"}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int | None = 32000,
        max_position_embeddings: int | None = 4096,
        hidden_size: int | None = 2560,
        num_hidden_layers: int | None = 54,
        layers_block_type: list[str] | None = None,
        mamba_d_state: int | None = 64,
        mamba_d_conv: int | None = 4,
        mamba_expand: int | None = 2,
        mamba_ngroups: int | None = 1,
        time_step_min: float | None = 0.001,
        time_step_max: float | None = 0.1,
        time_step_floor: int | None = 1e-4,
        time_step_limit: int | None = None,
        n_mamba_heads: int | None = 8,
        use_conv_bias: bool | None = True,
        chunk_size: int | None = 256,
        use_mem_eff_path: bool | None = False,
        add_bias_linear: bool | None = False,
        intermediate_size: int | None = None,
        hidden_act: str | None = "gelu",
        num_attention_heads: int | None = 32,
        num_key_value_heads: int | None = None,
        attention_dropout: float | None = 0.0,
        num_mem_blocks: int | None = 1,
        use_shared_attention_adapter: bool | None = False,
        adapter_rank: int | None = 128,
        use_mem_rope: bool | None = False,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        initializer_range: float | None = 0.02,
        rms_norm_eps: int | None = 1e-5,
        use_cache: bool | None = True,
        num_logits_to_keep: int | None = 1,
        pad_token_id: int | None = 0,
        bos_token_id: int | None = 1,
        eos_token_id: int | None = 2,
        use_long_context: bool | None = False,
        tie_word_embeddings: bool | None = True,
        **kwargs,
    ):
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        if intermediate_size is None:
            self.intermediate_size = 4 * hidden_size
        else:
            self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_mem_blocks = num_mem_blocks
        self.attention_hidden_size = 2 * hidden_size
        self.attention_head_dim = 2 * self.hidden_size // self.num_attention_heads
        self.attention_dropout = attention_dropout
        self.use_mem_rope = use_mem_rope
        self.use_long_context = use_long_context
        self.rope_parameters = rope_parameters

        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.add_bias_linear = add_bias_linear
        self.mamba_ngroups = mamba_ngroups
        self.n_mamba_heads = n_mamba_heads
        self.mamba_headdim = int(mamba_expand * hidden_size) // n_mamba_heads
        self.use_conv_bias = use_conv_bias
        self.chunk_size = chunk_size
        self.time_step_limit = time_step_limit
        self.use_shared_attention_adapter = use_shared_attention_adapter
        self.adapter_rank = adapter_rank
        self.time_step_min = time_step_min
        self.time_step_max = time_step_max
        self.time_step_floor = time_step_floor
        if use_long_context:
            self.max_position_embeddings = 16384
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_attention_heads = num_attention_heads
        self.kv_channels = self.hidden_size // self.num_attention_heads
        self.num_query_groups = self.num_attention_heads
        # Below, "mamba" stands for mamba layer, "hybrid" stands for hybrid layer (composed by a shared transformer followed by mamba layer)
        if layers_block_type is None:
            self.layers_block_type = (
                ["mamba"]
                + (["mamba"] * 5 + ["hybrid"]) * 7
                + ["mamba"] * 4
                + ["hybrid"]
                + ["mamba"] * 3
                + ["hybrid"]
                + ["mamba"] * 2
            )
        else:
            self.layers_block_type = layers_block_type
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.num_logits_to_keep = num_logits_to_keep
        self.hybrid_layer_ids = [index for index, type in enumerate(self.layers_block_type) if type == "hybrid"]
        self.use_mem_eff_path = use_mem_eff_path
        super().__init__(**kwargs)


__all__ = ["Zamba2Config"]
