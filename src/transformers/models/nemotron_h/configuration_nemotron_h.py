# Copyright 2024-2025 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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
"""NemotronH model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
@strict
class NemotronHConfig(PreTrainedConfig):
    r"""
    layers_block_type (`list`, *optional*):
        Explicit list of layer types for each layer. Each element must be one of: "mamba", "attention", or "moe".
        The number of layers is determined by the length of this list.
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
    moe_latent_size (`int`, *optional*):
        Latent size for MoE expert projections. If `None`, uses `hidden_size`.
    moe_shared_expert_overlap (`bool`, *optional*, defaults to `True`):
        Whether shared experts overlap with routed experts.
    n_group (`int`, *optional*, defaults to 1):
        Number of groups for expert routing.
    num_nextn_predict_layers (`int`, *optional*, defaults to 0):
        Number of additional layers for multi-token prediction. If 0, multi-token prediction is disabled.
    mtp_layers_block_type (`list`, *optional*, defaults to `['attention', 'moe']`):
        Explicit list of layer types for multi-token prediction layers when `num_nextn_predict_layers` > 0.
    use_bias (`bool`, *optional*, defaults to `False`):
        Whether to use bias in the model.
    residual_in_fp32 (`bool`, *optional*, defaults to `False`):
        Whether or not residuals should be in `float32`.
    rescale_prenorm_residual (`bool`, *optional*, defaults to `True`):
        Whether to rescale the pre-normalization residual connections.

    ```python
    >>> from transformers import NemotronHModel, NemotronHConfig

    >>> # Initializing a NemotronH configuration
    >>> configuration = NemotronHConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = NemotronHModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "nemotron_h"
    keys_to_ignore_at_inference = ["past_key_values"]

    vocab_size: int = 131072
    hidden_size: int = 4096
    layers_block_type: list[str] | None = None
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
    attention_dropout: float = 0.0
    sliding_window: int | None = None
    intermediate_size: int = 21504
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
    moe_latent_size: int | None = None
    moe_shared_expert_overlap: bool = True
    num_experts_per_tok: int = 2
    routed_scaling_factor: float | int = 1.0
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True
    num_nextn_predict_layers: int = 0
    mtp_layers_block_type: list[str] | None = None
    use_bias: bool = False
    initializer_range: float = 0.02
    layer_norm_epsilon: float = 1e-5
    residual_in_fp32: bool = False
    hidden_dropout: float | int = 0.0
    rescale_prenorm_residual: bool = True

    def __post_init__(self, **kwargs):
        # Backward compatibility; configs expect different names for these fields when init
        # but they have to be re-names when creating/saving the config.
        self.n_groups = kwargs.pop("mamba_n_groups") if "mamba_n_groups" in kwargs else self.n_groups
        self.conv_kernel = kwargs.pop("mamba_d_conv") if "mamba_d_conv" in kwargs else self.conv_kernel
        self.expand = kwargs.pop("mamba_expand") if "mamba_expand" in kwargs else self.expand
        self.time_step_min = kwargs.pop("mamba_dt_min") if "mamba_dt_min" in kwargs else self.time_step_min
        self.time_step_max = kwargs.pop("mamba_dt_max") if "mamba_dt_max" in kwargs else self.time_step_max
        self.time_step_limit = kwargs.pop("mamba_dt_limit") if "mamba_dt_limit" in kwargs else self.time_step_limit
        self.time_step_floor = (
            kwargs.pop("mamba_dt_init_floor") if "mamba_dt_init_floor" in kwargs else self.time_step_floor
        )
        self.use_conv_bias = kwargs.pop("mamba_conv_bias") if "mamba_conv_bias" in kwargs else self.use_conv_bias
        self.chunk_size = kwargs.pop("mamba_chunk_size") if "mamba_chunk_size" in kwargs else self.chunk_size

        # Backward compatibility: convert hybrid_override_pattern to layers_block_type
        # Always pop hybrid_override_pattern from kwargs to prevent it from being set as an attribute
        if "hybrid_override_pattern" in kwargs:
            pattern = kwargs.pop("hybrid_override_pattern")
            if self.layers_block_type is None:
                self.layers_block_type = self._pattern_to_list(pattern)
        elif self.layers_block_type is None:
            # Default layers_block_type if not provided
            self.layers_block_type = ["mamba", "moe", "attention", "moe"]

        # Note: num_hidden_layers is deprecated and ignored if layers_block_type is explicitly provided
        # It's only kept for backward compatibility when loading old configs
        if self.num_hidden_layers is not None:
            # Warn if num_hidden_layers is provided but doesn't match layers_block_type
            if len(self.layers_block_type) != self.num_hidden_layers:
                logger.warning(
                    f"num_hidden_layers ({self.num_hidden_layers}) is deprecated and doesn't match "
                    f"layers_block_type length ({len(self.layers_block_type)}). Using layers_block_type length."
                )

        # Backward compatibility: convert mtp_hybrid_override_pattern to mtp_layers_block_type
        # Always pop mtp_hybrid_override_pattern from kwargs to prevent it from being set as an attribute
        if self.mtp_layers_block_type is None:
            self.mtp_layers_block_type = ["attention", "moe"]

        if "mtp_hybrid_override_pattern" in kwargs:
            pattern = kwargs.pop("mtp_hybrid_override_pattern")
            if self.mtp_layers_block_type == ["attention", "moe"]:
                self.mtp_layers_block_type = self._pattern_to_list(pattern)

        # for backward compatibility
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        super().__post_init__(**kwargs)

    @staticmethod
    def validate_layers_block_type(self):
        """
        Validate layers_block_type list.
        """
        if not isinstance(self.layers_block_type, list):
            raise ValueError(
                f"`layers_block_type` must be a list of strings. Got type: {type(self.layers_block_type)}"
            )

        valid_types = {"mamba", "attention", "moe"}
        if not all(block_type in valid_types for block_type in self.layers_block_type):
            invalid = set(self.layers_block_type) - valid_types
            raise ValueError(f"`layers_block_type` contains invalid types: {invalid}. Must be one of: {valid_types}")

        if self.num_nextn_predict_layers > 0:
            if self.mtp_layers_block_type is None:
                raise ValueError(
                    "mtp_layers_block_type is required when num_nextn_predict_layers > 0. "
                    "Please provide an explicit list of layer types for MTP layers. "
                    "Example: mtp_layers_block_type=['attention', 'moe']"
                )

            if not isinstance(self.mtp_layers_block_type, list):
                raise ValueError(
                    f"`mtp_layers_block_type` must be a list of strings. Got type: {type(self.mtp_layers_block_type)}"
                )

            valid_types = {"mamba", "attention", "moe"}
            if not all(block_type in valid_types for block_type in self.mtp_layers_block_type):
                invalid = set(self.mtp_layers_block_type) - valid_types
                raise ValueError(
                    f"`mtp_layers_block_type` contains invalid types: {invalid}. Must be one of: {valid_types}"
                )

    @property
    def num_hidden_layers(self) -> int:
        """
        Number of hidden layers derived from the length of layers_block_type.
        This property replaces the deprecated num_hidden_layers parameter.
        """
        return len(self.layers_block_type)

    @num_hidden_layers.setter
    def num_hidden_layers(self, value):
        """
        Setter for backward compatibility when loading configs.
        The value is ignored since num_hidden_layers is computed from layers_block_type.
        """
        # Ignore the value - num_hidden_layers is always derived from layers_block_type
        pass

    @property
    def hybrid_override_pattern(self) -> str:
        """
        Backward compatibility property.
        Returns the pattern string representation of layers_block_type.
        """
        return self._list_to_pattern(self.layers_block_type)

    @property
    def mtp_hybrid_override_pattern(self) -> str:
        """
        Backward compatibility property.
        Returns the pattern string representation of mtp_layers_block_type.
        """
        return self._list_to_pattern(self.mtp_layers_block_type)

    @staticmethod
    def _list_to_pattern(layers_list: list) -> str:
        """Convert list of layer types back to pattern string (for backward compatibility)."""
        reverse_mapping = {"mamba": "M", "moe": "E", "attention": "*"}
        return "".join(reverse_mapping[layer_type] for layer_type in layers_list)

    @staticmethod
    def _pattern_to_list(pattern: str) -> list:
        """Convert pattern string to list of layer types (for backward compatibility)."""
        pattern_mapping = {"M": "mamba", "E": "moe", "*": "attention"}
        return [pattern_mapping[char] for char in pattern]


__all__ = ["NemotronHConfig"]
