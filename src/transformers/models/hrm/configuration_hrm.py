# coding=utf-8
# Copyright 2025 The HRM Team and HuggingFace Inc. team. All rights reserved.
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
"""HRM (Hierarchical Reasoning Model) configuration"""

from ...configuration_utils import PretrainedConfig


class HrmConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`HrmModel`]. It is used to instantiate a HRM
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the HRM base model.

    The Hierarchical Reasoning Model (HRM) is a novel recurrent neural network architecture for sequential reasoning
    tasks featuring:
    - Two-level hierarchical processing inspired by human cognition
    - High-level (H) module: Slow, abstract planning and reasoning
    - Low-level (L) module: Fast, detailed computations
    - Adaptive Computation Time (ACT) mechanism with Q-learning based halting

    This model was introduced in the paper "Hierarchical Reasoning Model" by Guan Wang, Jin Li, Yuhao Sun,
    Xing Chen, Changling Liu, Yue Wu, Meng Lu, Sen Song, and Yasin Abbasi Yadkori.
    For more details, see: https://arxiv.org/abs/2506.21734

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 11):
            Vocabulary size of the HRM model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`HrmModel`]. For reasoning tasks like Sudoku, this is typically 11
            (digits 0-9 plus a padding token).
        hidden_size (`int`, *optional*, defaults to 512):
            Dimension of the hidden representations and embeddings.
        num_hidden_layers (`int`, *optional*, defaults to 4):
            Number of transformer layers in both high-level (H) and low-level (L) modules.
            Use `h_layers` and `l_layers` to set them independently.
        h_layers (`int`, *optional*):
            Number of transformer layers in the high-level (H) module for abstract planning.
            If not specified, defaults to `num_hidden_layers`.
        l_layers (`int`, *optional*):
            Number of transformer layers in the low-level (L) module for detailed computations.
            If not specified, defaults to `num_hidden_layers`.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the transformer blocks.
        intermediate_size (`int`, *optional*):
            Dimension of the MLP representations. If not specified, defaults to `hidden_size * expansion`.
        expansion (`float`, *optional*, defaults to 4.0):
            MLP expansion ratio for SwiGLU feed-forward layers. Used to calculate `intermediate_size`
            if not explicitly provided.
        max_position_embeddings (`int`, *optional*, defaults to 81):
            The maximum sequence length that this model might ever be used with. For Sudoku, this is 81 (9x9 grid).
            For ARC tasks, this can be up to 900 (30x30 grid).
        h_cycles (`int`, *optional*, defaults to 2):
            Number of high-level reasoning cycles per forward pass. Controls the depth of abstract planning.
        l_cycles (`int`, *optional*, defaults to 2):
            Number of low-level computation cycles per high-level cycle. Controls granularity of detailed processing.
        pos_encodings (`str`, *optional*, defaults to `"rope"`):
            Type of positional encoding to use. Options are "rope" (Rotary Position Embeddings) or "learned".
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings. Only used when `pos_encodings="rope"`.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the RMS normalization layers for numerical stability.
        puzzle_emb_ndim (`int`, *optional*, defaults to 0):
            Dimension of per-puzzle sparse embeddings. Set to 0 to disable puzzle-specific embeddings.
            When > 0, each unique puzzle gets a learned embedding of this dimension.
        num_puzzle_identifiers (`int`, *optional*, defaults to 1):
            Total number of unique puzzle types/IDs for which to learn separate embeddings.
            Only used when `puzzle_emb_ndim > 0`.
        halt_max_steps (`int`, *optional*, defaults to 16):
            Maximum number of computation steps before forcing the ACT mechanism to halt.
            Controls the computational budget per sequence.
        halt_exploration_prob (`float`, *optional*, defaults to 0.1):
            Probability of exploration during ACT training. Used for Q-learning based adaptive halting.
        dtype (`str` or `torch.dtype`, *optional*, defaults to `"bfloat16"`):
            The dtype of the model's forward pass computations. Can be "bfloat16", "float32", or "float16".
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        use_cache (`bool`, *optional*, defaults to `False`):
            Whether or not the model should return the carry state for recurrent computation.
            HRM uses a unique carry state system for hierarchical processing.

    Example:
    ```python
    >>> from transformers import HrmConfig, HrmModel

    >>> # Initializing a HRM configuration for Sudoku solving
    >>> configuration = HrmConfig(
    ...     vocab_size=11,  # 0-9 digits + padding
    ...     hidden_size=512,
    ...     num_hidden_layers=4,
    ...     max_position_embeddings=81,  # 9x9 grid
    ...     h_cycles=2,
    ...     l_cycles=2,
    ... )

    >>> # Initializing a model from the configuration
    >>> model = HrmModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "hrm"

    def __init__(
        self,
        vocab_size=11,
        hidden_size=512,
        num_hidden_layers=4,
        h_layers=None,
        l_layers=None,
        num_attention_heads=8,
        intermediate_size=None,
        expansion=4.0,
        max_position_embeddings=81,
        h_cycles=2,
        l_cycles=2,
        pos_encodings="rope",
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        puzzle_emb_ndim=0,
        num_puzzle_identifiers=1,
        halt_max_steps=16,
        halt_exploration_prob=0.1,
        dtype="bfloat16",
        initializer_range=0.02,
        use_cache=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.h_layers = h_layers if h_layers is not None else num_hidden_layers
        self.l_layers = l_layers if l_layers is not None else num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.expansion = expansion
        self.intermediate_size = intermediate_size if intermediate_size is not None else int(hidden_size * expansion)
        self.max_position_embeddings = max_position_embeddings
        self.h_cycles = h_cycles
        self.l_cycles = l_cycles
        self.pos_encodings = pos_encodings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.puzzle_emb_ndim = puzzle_emb_ndim
        self.num_puzzle_identifiers = num_puzzle_identifiers
        self.halt_max_steps = halt_max_steps
        self.halt_exploration_prob = halt_exploration_prob
        self.initializer_range = initializer_range
        self.use_cache = use_cache

        super().__init__(dtype=dtype, **kwargs)
