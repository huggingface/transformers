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


from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="Zyphra/Zamba2-2.7B")
@strict
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
    attribute_map = {"layer_types": "layers_block_type", "head_dim": "attention_head_dim"}
    keys_to_ignore_at_inference = ["past_key_values"]

    vocab_size: int = 32000
    max_position_embeddings: int = 4096
    hidden_size: int = 2560
    num_hidden_layers: int = 54
    layers_block_type: list[str] | None = None
    mamba_d_state: int = 64
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_ngroups: int = 1
    time_step_min: float = 0.001
    time_step_max: float = 0.1
    time_step_floor: float = 1e-4
    time_step_limit: list[float] | tuple[float, ...] | None = None
    n_mamba_heads: int = 8
    use_conv_bias: bool = True
    chunk_size: int = 256
    use_mem_eff_path: bool = False
    add_bias_linear: bool = False
    intermediate_size: int | None = None
    hidden_act: str = "gelu"
    num_attention_heads: int = 32
    num_key_value_heads: int | None = None
    attention_dropout: float | int = 0.0
    num_mem_blocks: int = 1
    use_shared_attention_adapter: bool = False
    adapter_rank: int = 128
    use_mem_rope: bool = False
    rope_parameters: RopeParameters | dict | None = None
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    num_logits_to_keep: int = 1
    pad_token_id: int | None = 0
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    use_long_context: bool = False
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        self.intermediate_size = self.intermediate_size or 4 * self.hidden_size
        self.attention_hidden_size = 2 * self.hidden_size
        self.attention_head_dim = 2 * self.hidden_size // self.num_attention_heads
        self.mamba_headdim = int(self.mamba_expand * self.hidden_size) // self.n_mamba_heads
        if self.use_long_context:
            self.max_position_embeddings = 16384

        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        self.kv_channels = self.hidden_size // self.num_attention_heads
        self.num_query_groups = self.num_attention_heads

        # Below, "mamba" stands for mamba layer, "hybrid" stands for hybrid layer (composed by a shared transformer followed by mamba layer)
        if self.layers_block_type is None:
            self.layers_block_type = (
                ["mamba"]
                + (["mamba"] * 5 + ["hybrid"]) * 7
                + ["mamba"] * 4
                + ["hybrid"]
                + ["mamba"] * 3
                + ["hybrid"]
                + ["mamba"] * 2
            )
        self.hybrid_layer_ids = [index for index, type in enumerate(self.layers_block_type) if type == "hybrid"]
        super().__post_init__(**kwargs)


__all__ = ["Zamba2Config"]
