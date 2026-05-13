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
"""MAMBA configuration"""

import math

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="state-spaces/mamba-2.8b")
@strict
class MambaConfig(PreTrainedConfig):
    r"""
    layer_norm_epsilon (`float`, *optional*, defaults to 1e-05):
        The epsilon to use in the layer normalization layers.
    expand (`int`, *optional*, defaults to 2):
        Expanding factor used to determine the intermediate size.
    use_bias (`bool`, *optional*, defaults to `False`):
        Whether or not to use bias in ["in_proj", "out_proj"] of the mixer block
    use_conv_bias (`bool`, *optional*, defaults to `True`):
        Whether or not to use bias in the convolution layer of the mixer block.
    residual_in_fp32 (`bool`, *optional*, defaults to `True`):
        Whether or not residuals should be in `float32`. If set to `False` residuals will keep the same `dtype` as the rest of the model
    rescale_prenorm_residual (`bool`, *optional*, defaults to `False`):
        Whether or not to rescale `out_proj` weights when initializing.
    use_mambapy (`bool`, *optional*, defaults to `False`):
        Determines the fallback strategy during training if the CUDA-based official implementation of Mamba is not available. If `True`,
        the mamba.py implementation is used. If `False`, the naive and slower implementation is used. Consider switching to the naive
        version if memory is limited.
    use_associative_scan (`bool`, *optional*, defaults to `True`):
        Whether to use PyTorch's `torch._higher_order_ops.associative_scan` for the parallel scan instead of the naive
        sequential implementation. The associative scan is only active during `torch.compile` tracing and
        requires torch >= 2.9.0. Both paths are tested to produce numerically identical results (see
        `test_associative_scan_matches_sequential`). Set to `False` to fall back to the sequential loop.

    Example:

    ```python
    >>> from transformers import MambaConfig, MambaModel

    >>> # Initializing a Mamba configuration
    >>> configuration = MambaConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = MambaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "mamba"

    vocab_size: int = 50280
    hidden_size: int = 768
    state_size: int = 16
    num_hidden_layers: int = 32
    layer_norm_epsilon: float = 1e-5
    pad_token_id: int | None = 0
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 0
    expand: int = 2
    conv_kernel: int = 4
    use_bias: bool = False
    use_conv_bias: bool = True
    hidden_act: str = "silu"
    initializer_range: float = 0.1
    residual_in_fp32: bool = True
    time_step_rank: str | int = "auto"
    time_step_scale: float = 1.0
    time_step_min: float = 0.001
    time_step_max: float = 0.1
    time_step_init_scheme: str = "random"
    time_step_floor: float = 1e-4
    rescale_prenorm_residual: bool = False
    use_cache: bool = True
    use_mambapy: bool = False
    use_associative_scan: bool = True
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        self.intermediate_size = int(self.expand * self.hidden_size)
        self.time_step_rank = (
            math.ceil(self.hidden_size / 16) if self.time_step_rank == "auto" else self.time_step_rank
        )
        super().__post_init__(**kwargs)

    @property
    def layer_types(self):
        return ["mamba"] * self.num_hidden_layers


__all__ = ["MambaConfig"]
