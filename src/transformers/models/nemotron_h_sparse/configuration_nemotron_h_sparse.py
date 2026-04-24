# Copyright 2024-2026 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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
"""NemotronHSparse model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


# Pattern characters in the released Nemotron-3 config:
#   `M` = mamba mixer, `*` = attention mixer, `E` = moe FFN tail.
# Each `M` / `*` becomes one decoder layer (with an moe tail baked in); standalone `E`
# is absorbed as the FFN of the preceding `M` / `*`.
_SPARSE_VALID_CHARS = {"M", "*", "E"}


def _collapse_pattern_to_layer_types(pattern: str) -> list[str]:
    """Convert a per-char `hybrid_override_pattern` into a per-decoder-layer list."""
    layer_types: list[str] = []
    for char in pattern:
        if char == "M":
            layer_types.append("mamba")
        elif char == "*":
            layer_types.append("attention")
        elif char == "E":
            continue
        else:
            raise ValueError(
                f"NemotronHSparseConfig `hybrid_override_pattern` got invalid char {char!r}; "
                f"must only contain {sorted(_SPARSE_VALID_CHARS)} (M=mamba, *=attention, E=moe). "
                "Use NemotronHDenseConfig for MLP (-) patterns."
            )
    return layer_types


@auto_docstring(checkpoint="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
@strict
class NemotronHSparseConfig(PreTrainedConfig):
    r"""
    hybrid_override_pattern (`str`, *optional*, defaults to `"M*EM*E"`):
        Per-sub-block sequence (from the released Nemotron-3 config). `M` = mamba,
        `*` = attention, `E` = moe FFN tail. Each `M` and `*` becomes one decoder
        layer (with an moe tail baked in); `E` is absorbed.
    num_logits_to_keep (`int`, *optional*, defaults to 1):
        Number of prompt logits to calculate during generation. If `None`, all logits will be calculated.
    use_mamba_kernels (`bool`, *optional*, defaults to `True`):
        Flag indicating whether or not to use the fast mamba kernels.
    ssm_state_size (`int`, *optional*, defaults to 128):
        The dimension of the mamba state space latents.
    mamba_hidden_act (`str`, *optional*, defaults to `"silu"`):
        The non-linear activation function in the Mamba layers.
    n_groups (`int`, *optional*, defaults to 8):
        Number of groups for the evolution matrices of the Mamba layers.
    expand (`int`, *optional*, defaults to 2):
        Expanding factor used to determine the intermediate size in the Mamba layers.
    use_conv_bias (`bool`, *optional*, defaults to `True`):
        Whether or not to use bias in the convolution layer of the Mamba mixer block.
    chunk_size (`int`, *optional*, defaults to 128):
        Size of the chunks that will comprise the sequence in the Mamba layers.
    mamba_ssm_cache_dtype (`str`, *optional*, defaults to `"float32"`):
        Data type for Mamba SSM cache states.
    moe_shared_expert_intermediate_size (`int`, *optional*, defaults to 7688):
        Dimension of the MLP representations in shared experts.
    moe_shared_expert_overlap (`bool`, *optional*, defaults to `True`):
        Whether shared experts overlap with routed experts.
    n_group (`int`, *optional*, defaults to 1):
        Number of groups for expert routing.
    use_bias (`bool`, *optional*, defaults to `False`):
        Whether to use bias in the model.
    residual_in_fp32 (`bool`, *optional*, defaults to `False`):
        Whether or not residuals should be in `float32`.
    rescale_prenorm_residual (`bool`, *optional*, defaults to `True`):
        Whether to rescale the pre-normalization residual connections.

    ```python
    >>> from transformers import NemotronHSparseModel, NemotronHSparseConfig

    >>> configuration = NemotronHSparseConfig()
    >>> model = NemotronHSparseModel(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "nemotron_h_sparse"
    keys_to_ignore_at_inference = ["past_key_values"]

    vocab_size: int = 131072
    hidden_size: int = 4096
    hybrid_override_pattern: str = "M*EM*E"
    tie_word_embeddings: bool = False
    use_cache: bool = True
    num_logits_to_keep: int = 1
    pad_token_id: int | None = 0
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 4096
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    sliding_window: int | None = None
    mlp_hidden_act: str = "relu2"
    mlp_bias: bool = False
    use_mamba_kernels: bool = True
    ssm_state_size: int = 128
    mamba_num_heads: int = 128
    mamba_head_dim: int = 64
    mamba_hidden_act: str = "silu"
    n_groups: int = 8
    conv_kernel: int = 4
    expand: int = 2
    time_step_min: float = 0.001
    time_step_max: float = 0.1
    time_step_limit: list[float] | tuple[float, ...] = (0.0, float("inf"))
    time_step_floor: float = 1e-4
    use_conv_bias: bool = True
    chunk_size: int = 128
    mamba_proj_bias: bool = False
    mamba_ssm_cache_dtype: str = "float32"
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    moe_intermediate_size: int = 7688
    moe_shared_expert_intermediate_size: int = 7688
    moe_shared_expert_overlap: bool = True
    num_experts_per_tok: int = 2
    routed_scaling_factor: float | int = 1.0
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True
    use_bias: bool = False
    initializer_range: float = 0.02
    layer_norm_epsilon: float = 1e-5
    residual_in_fp32: bool = False
    hidden_dropout: float | int = 0.0
    rescale_prenorm_residual: bool = True

    def __post_init__(self, **kwargs):
        # BC: legacy mamba_* kwarg aliases.
        self.n_groups = kwargs.pop("mamba_n_groups", self.n_groups)
        self.conv_kernel = kwargs.pop("mamba_d_conv", self.conv_kernel)
        self.expand = kwargs.pop("mamba_expand", self.expand)
        self.time_step_min = kwargs.pop("mamba_dt_min", self.time_step_min)
        self.time_step_max = kwargs.pop("mamba_dt_max", self.time_step_max)
        self.time_step_limit = kwargs.pop("mamba_dt_limit", self.time_step_limit)
        self.time_step_floor = kwargs.pop("mamba_dt_init_floor", self.time_step_floor)
        self.use_conv_bias = kwargs.pop("mamba_conv_bias", self.use_conv_bias)
        self.chunk_size = kwargs.pop("mamba_chunk_size", self.chunk_size)

        # Drop unsupported kwargs (MTP / latent expert projections / layer counts).
        kwargs.pop("num_nextn_predict_layers", None)
        kwargs.pop("mtp_hybrid_override_pattern", None)
        kwargs.pop("mtp_layers_block_type", None)
        kwargs.pop("moe_latent_size", None)
        kwargs.pop("num_hidden_layers", None)
        kwargs.pop("layers_block_type", None)

        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        # Validate by parsing: raises on unknown chars.
        _collapse_pattern_to_layer_types(self.hybrid_override_pattern)

        super().__post_init__(**kwargs)

    @property
    def layer_types(self) -> list[str]:
        return _collapse_pattern_to_layer_types(self.hybrid_override_pattern)

    @property
    def layers_block_type(self) -> list[str]:
        return self.layer_types

    @property
    def num_hidden_layers(self) -> int:
        return len(self.layer_types)

    @num_hidden_layers.setter
    def num_hidden_layers(self, value):
        # BC: ignore; length is derived from `hybrid_override_pattern`.
        pass


__all__ = ["NemotronHSparseConfig"]
