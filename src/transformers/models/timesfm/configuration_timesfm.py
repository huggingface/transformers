# Copyright 2025 Google LLC and HuggingFace Inc. team.
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
"""TimesFM model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="google/timesfm-2.0-500m-pytorch")
class TimesFmConfig(PreTrainedConfig):
    r"""
    patch_length (`int`, *optional*, defaults to 32):
        The length of one patch in the input sequence.
    context_length (`int`, *optional*, defaults to 512):
        The length of the input context.
    horizon_length (`int`, *optional*, defaults to 128):
        The length of the prediction horizon.
    freq_size (`int`, *optional*, defaults to 3):
        The number of frequency embeddings.
    tolerance (`float`, *optional*, defaults to 1e-06):
        The tolerance for the quantile loss.
    quantiles (`list[float]`, *optional*, defaults to `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`):
        The quantiles to predict.
    pad_val (`float`, *optional*, defaults to 1123581321.0):
        The value used to pad the predictions.
    attention_dropout (`float`, *optional*, defaults to 0.0):
        The dropout probability for the attention scores.
    use_positional_embedding (`bool`, *optional*, defaults to `False`):
        Whether to add positional embeddings.
    min_timescale (`int`, *optional*, defaults to 1):
        The start of the geometric positional index. Determines the periodicity of
        the added signal.
    max_timescale (`int`, *optional*, defaults to 10000):
        The end of the geometric positional index. Determines the frequency of the
        added signal.
    """

    model_type = "timesfm"
    keys_to_ignore_at_inference = []
    is_encoder_decoder = False

    def __init__(
        self,
        patch_length: int = 32,
        context_length: int = 512,
        horizon_length: int = 128,
        freq_size: int = 3,
        num_hidden_layers: int = 50,
        hidden_size: int = 1280,
        intermediate_size: int = 1280,
        head_dim: int = 80,
        num_attention_heads: int = 16,
        tolerance: float = 1e-6,
        rms_norm_eps: float = 1e-6,
        quantiles: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        pad_val: float = 1123581321.0,
        attention_dropout: float = 0.0,
        use_positional_embedding: bool = False,
        initializer_range: float = 0.02,
        min_timescale: int = 1,
        max_timescale: int = 10_000,
        **kwargs,
    ):
        self.patch_length = patch_length
        self.context_length = context_length
        self.horizon_length = horizon_length
        self.quantiles = quantiles
        self.pad_val = pad_val
        self.freq_size = freq_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.head_dim = head_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.tolerance = tolerance
        self.rms_norm_eps = rms_norm_eps
        self.attention_dropout = attention_dropout
        self.use_positional_embedding = use_positional_embedding
        self.initializer_range = initializer_range
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale

        kwargs["is_encoder_decoder"] = self.is_encoder_decoder
        super().__init__(**kwargs)


__all__ = ["TimesFmConfig"]
