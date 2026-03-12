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

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="state-spaces/mamba-2.8b")
class MambaConfig(PreTrainedConfig):
    """
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

    def __init__(
        self,
        vocab_size=50280,
        hidden_size=768,
        state_size=16,
        num_hidden_layers=32,
        layer_norm_epsilon=1e-5,
        pad_token_id=0,
        bos_token_id=0,
        eos_token_id=0,
        expand=2,
        conv_kernel=4,
        use_bias=False,
        use_conv_bias=True,
        hidden_act="silu",
        initializer_range=0.1,
        residual_in_fp32=True,
        time_step_rank="auto",
        time_step_scale=1.0,
        time_step_min=0.001,
        time_step_max=0.1,
        time_step_init_scheme="random",
        time_step_floor=1e-4,
        rescale_prenorm_residual=False,
        use_cache=True,
        use_mambapy=False,
        use_associative_scan=True,
        tie_word_embeddings=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.num_hidden_layers = num_hidden_layers
        self.layer_norm_epsilon = layer_norm_epsilon
        self.conv_kernel = conv_kernel
        self.expand = expand
        self.intermediate_size = int(expand * self.hidden_size)
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.use_bias = use_bias
        self.use_conv_bias = use_conv_bias
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.time_step_rank = math.ceil(self.hidden_size / 16) if time_step_rank == "auto" else time_step_rank
        self.time_step_scale = time_step_scale
        self.time_step_min = time_step_min
        self.time_step_max = time_step_max
        self.time_step_init_scheme = time_step_init_scheme
        self.time_step_floor = time_step_floor
        self.rescale_prenorm_residual = rescale_prenorm_residual
        self.residual_in_fp32 = residual_in_fp32
        self.use_cache = use_cache
        self.use_mambapy = use_mambapy
        self.use_associative_scan = use_associative_scan
        self.tie_word_embeddings = tie_word_embeddings

        super().__init__(**kwargs)


__all__ = ["MambaConfig"]
