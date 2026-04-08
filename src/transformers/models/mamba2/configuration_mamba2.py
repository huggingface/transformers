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

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="state-spaces/mamba2-2.8b")
@strict
class Mamba2Config(PreTrainedConfig):
    r"""
    layer_norm_epsilon (`float`, *optional*, defaults to 1e-05):
        The epsilon to use in the layer normalization layers..
    expand (`int`, *optional*, defaults to 2):
        Expanding factor used to determine the intermediate size.
    n_groups (`int`, *optional*, defaults to 8):
        Number of groups for the evolution matrices of mamba 2.
    use_bias (`bool`, *optional*, defaults to `False`):
        Whether or not to use bias in ["in_proj", "out_proj"] of the mixer block
    use_conv_bias (`bool`, *optional*, defaults to `True`):
        Whether or not to use bias in the convolution layer of the mixer block.
    residual_in_fp32 (`bool`, *optional*, defaults to `True`):
        Whether or not residuals should be in `float32`. If set to `False` residuals will keep the same `dtype` as the rest of the model
    rescale_prenorm_residual (`bool`, *optional*, defaults to `False`):
        Whether or not to rescale `out_proj` weights when initializing.
    chunk_size (`int`, *optional*, defaults to 256):
        Size of the chunks that will comprise the sequence.

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

    num_heads: int = 128
    head_dim: int = 64
    vocab_size: int = 32768
    hidden_size: int = 4096
    state_size: int = 128
    num_hidden_layers: int = 64
    layer_norm_epsilon: float = 1e-5
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
    expand: int = 2
    conv_kernel: int = 4
    n_groups: int = 8
    use_bias: bool = False
    use_conv_bias: bool = True
    hidden_act: str = "silu"
    initializer_range: float = 0.1
    residual_in_fp32: bool = True
    time_step_rank: str | int = "auto"
    time_step_min: float = 0.001
    time_step_max: float = 0.1
    time_step_floor: float = 1e-4
    time_step_limit: list[float] | tuple[float, ...] = (0.0, float("inf"))
    rescale_prenorm_residual: bool = False
    use_cache: bool = True
    rms_norm: bool = True
    chunk_size: int = 256
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        self.time_step_rank = (
            math.ceil(self.hidden_size / 16) if self.time_step_rank == "auto" else self.time_step_rank
        )
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if (self.hidden_size * self.expand) != (self.num_heads * self.head_dim):
            raise ValueError(
                "Inconsistent configuration: hidden_size * expand "
                f"({self.hidden_size * self.expand}) must equal num_heads * head_dim "
                f"({self.num_heads * self.head_dim})."
            )

    @property
    def layer_types(self):
        return ["mamba"] * self.num_hidden_layers


__all__ = ["Mamba2Config"]
