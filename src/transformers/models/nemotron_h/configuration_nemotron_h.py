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

from ...configuration_utils import PretrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
class NemotronHConfig(PretrainedConfig):
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
    n_group (`int`, *optional*, defaults to 1):
        Number of groups for expert routing.
    mamba_hidden_act (`str`, *optional*, defaults to `"silu"`):
        The non-linear activation function in the Mamba layers.
    mamba_dt_min (`float`, *optional*, defaults to 0.001):
        Minimum value for the time step in Mamba.
    mamba_dt_max (`float`, *optional*, defaults to 0.1):
        Maximum value for the time step in Mamba.
    mamba_dt_limit (`tuple`, *optional*, defaults to `(0.0, inf)`):
        Limits for the time step in Mamba.
    mamba_dt_init_floor (`float`, *optional*, defaults to 0.0001):
        Floor value for time step initialization in Mamba.
    mamba_ssm_cache_dtype (`str`, *optional*, defaults to `"float32"`):
        Data type for Mamba SSM cache states.
    moe_shared_expert_intermediate_size (`int`, *optional*, defaults to 7688):
        Dimension of the MLP representations in shared experts.
    moe_latent_size (`int`, *optional*):
        Latent size for MoE expert projections. If `None`, uses `hidden_size`.
    moe_shared_expert_overlap (`bool`, *optional*, defaults to `True`):
        Whether shared experts overlap with routed experts.
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
    ```"""

    model_type = "nemotron_h"
    keys_to_ignore_at_inference = ["past_key_values"]

    @staticmethod
    def _validate_layers_block_type(layers_block_type, expected_length=None, param_name="layers_block_type"):
        """
        Validate layers_block_type list.

        Args:
            layers_block_type: List of layer types to validate
            expected_length: If provided, validate the list has this length
            param_name: Parameter name for error messages

        Raises:
            ValueError: If validation fails
        """
        if not isinstance(layers_block_type, list):
            raise ValueError(f"{param_name} must be a list of strings. Got type: {type(layers_block_type)}")

        if expected_length is not None and len(layers_block_type) != expected_length:
            raise ValueError(f"{param_name} must have length {expected_length}. Got length {len(layers_block_type)}.")

        valid_types = {"mamba", "attention", "moe"}
        if not all(block_type in valid_types for block_type in layers_block_type):
            invalid = set(layers_block_type) - valid_types
            raise ValueError(f"{param_name} contains invalid types: {invalid}. Must be one of: {valid_types}")

    def __init__(
        self,
        # General model config
        vocab_size=131072,
        hidden_size=4096,
        layers_block_type=None,
        num_hidden_layers=None,  # Deprecated, only for backward compatibility
        tie_word_embeddings=False,
        use_cache=True,
        num_logits_to_keep=1,
        # Token IDs
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        # Attention layer config
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=4096,
        attention_bias=False,
        attention_dropout=0.0,
        sliding_window=None,
        # MLP layer config
        intermediate_size=21504,
        mlp_hidden_act="relu2",
        mlp_bias=False,
        # Mamba layer config
        use_mamba_kernels=True,
        ssm_state_size=128,
        mamba_num_heads=128,
        mamba_n_groups=8,
        mamba_head_dim=64,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_hidden_act="silu",
        mamba_dt_min=0.001,
        mamba_dt_max=0.1,
        mamba_dt_limit=(0.0, float("inf")),
        mamba_dt_init_floor=1e-4,
        mamba_conv_bias=True,
        mamba_proj_bias=False,
        mamba_chunk_size=128,
        mamba_ssm_cache_dtype="float32",
        # MoE config
        n_routed_experts=8,
        n_shared_experts=1,
        moe_intermediate_size=7688,
        moe_shared_expert_intermediate_size=7688,
        moe_latent_size=None,
        moe_shared_expert_overlap=True,
        num_experts_per_tok=2,
        routed_scaling_factor=1.0,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        # Multi-token prediction config
        num_nextn_predict_layers=0,
        mtp_layers_block_type=["attention", "moe"],
        # General training config
        use_bias=False,
        initializer_range=0.02,
        layer_norm_epsilon=1e-5,
        residual_in_fp32=False,
        hidden_dropout=0.0,
        rescale_prenorm_residual=True,
        **kwargs,
    ):
        # Backward compatibility: convert hybrid_override_pattern to layers_block_type
        # Always pop hybrid_override_pattern from kwargs to prevent it from being set as an attribute
        if "hybrid_override_pattern" in kwargs:
            pattern = kwargs.pop("hybrid_override_pattern")
            if layers_block_type is None:
                layers_block_type = self._pattern_to_list(pattern)
        elif layers_block_type is None:
            # Default layers_block_type if not provided
            layers_block_type = ["mamba", "moe", "attention", "moe"]

        # Note: num_hidden_layers is deprecated and ignored if layers_block_type is explicitly provided
        # It's only kept for backward compatibility when loading old configs
        if num_hidden_layers is not None:
            # Warn if num_hidden_layers is provided but doesn't match layers_block_type
            if len(layers_block_type) != num_hidden_layers:
                logger.warning(
                    f"num_hidden_layers ({num_hidden_layers}) is deprecated and doesn't match "
                    f"layers_block_type length ({len(layers_block_type)}). Using layers_block_type length."
                )

        # Backward compatibility: convert mtp_hybrid_override_pattern to mtp_layers_block_type
        # Always pop mtp_hybrid_override_pattern from kwargs to prevent it from being set as an attribute
        if "mtp_hybrid_override_pattern" in kwargs:
            pattern = kwargs.pop("mtp_hybrid_override_pattern")
            if mtp_layers_block_type is None or mtp_layers_block_type == ["attention", "moe"]:
                mtp_layers_block_type = self._pattern_to_list(pattern)

        self.vocab_size = vocab_size
        self.tie_word_embeddings = tie_word_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        self.max_position_embeddings = max_position_embeddings
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout

        # Validate layers_block_type (no longer checking length against num_hidden_layers)
        self._validate_layers_block_type(layers_block_type, expected_length=None, param_name="layers_block_type")
        self.layers_block_type = layers_block_type

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.mlp_hidden_act = mlp_hidden_act
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.use_bias = use_bias
        self.initializer_range = initializer_range
        self.layer_norm_epsilon = layer_norm_epsilon
        self.residual_in_fp32 = residual_in_fp32

        self.use_cache = use_cache
        self.num_logits_to_keep = num_logits_to_keep

        self.use_mamba_kernels = use_mamba_kernels
        self.n_groups = mamba_n_groups
        self.mamba_head_dim = mamba_head_dim
        self.ssm_state_size = ssm_state_size
        self.mamba_num_heads = mamba_num_heads
        self.conv_kernel = mamba_d_conv
        self.expand = mamba_expand
        self.mamba_hidden_act = mamba_hidden_act
        self.time_step_min = mamba_dt_min
        self.time_step_max = mamba_dt_max
        self.time_step_limit = mamba_dt_limit
        self.time_step_floor = mamba_dt_init_floor
        self.use_conv_bias = mamba_conv_bias
        self.mamba_proj_bias = mamba_proj_bias
        self.chunk_size = mamba_chunk_size
        self.rescale_prenorm_residual = rescale_prenorm_residual
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_shared_expert_intermediate_size = moe_shared_expert_intermediate_size
        self.moe_latent_size = moe_latent_size
        self.moe_shared_expert_overlap = moe_shared_expert_overlap
        self.num_experts_per_tok = num_experts_per_tok
        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.mamba_ssm_cache_dtype = mamba_ssm_cache_dtype

        # MTP config
        self.num_nextn_predict_layers = num_nextn_predict_layers

        # Validate mtp_layers_block_type is provided when MTP is enabled
        if self.num_nextn_predict_layers > 0:
            if mtp_layers_block_type is None:
                raise ValueError(
                    "mtp_layers_block_type is required when num_nextn_predict_layers > 0. "
                    "Please provide an explicit list of layer types for MTP layers. "
                    "Example: mtp_layers_block_type=['attention', 'moe']"
                )
            self._validate_layers_block_type(mtp_layers_block_type, None, "mtp_layers_block_type")
        self.mtp_layers_block_type = mtp_layers_block_type

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
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
