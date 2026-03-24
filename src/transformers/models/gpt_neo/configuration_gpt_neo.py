# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""GPT Neo model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="EleutherAI/gpt-neo-1.3B")
@strict
class GPTNeoConfig(PreTrainedConfig):
    r"""
    attention_types (`list`, *optional*, defaults to `[[['global', 'local'], 12]]`):
        The type of attention for each layer in a `List` of the following format `[[["attention_type"],
        num_layerss]]` e.g. for a 24 layer model `[[["global"], 24]]` or `[[["global", "local"], 12]]` Choose the
        value of `attention_type` from `["global", "local"]
    window_size (`int`, *optional*, defaults to 256):
        The size of the sliding window for local attention.

    Example:

    ```python
    >>> from transformers import GPTNeoConfig, GPTNeoModel

    >>> # Initializing a GPTNeo EleutherAI/gpt-neo-1.3B style configuration
    >>> configuration = GPTNeoConfig()

    >>> # Initializing a model (with random weights) from the EleutherAI/gpt-neo-1.3B style configuration
    >>> model = GPTNeoModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "gpt_neo"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_attention_heads": "num_heads", "num_hidden_layers": "num_layers"}

    vocab_size: int = 50257
    max_position_embeddings: int = 2048
    hidden_size: int = 2048
    num_layers: int = 24
    attention_types: list | tuple | None = None
    num_heads: int = 16
    intermediate_size: int | None = None
    window_size: int = 256
    activation_function: str = "gelu_new"
    resid_dropout: float | int = 0.0
    embed_dropout: float | int = 0.0
    attention_dropout: float | int = 0.0
    classifier_dropout: float | int = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    use_cache: bool = True
    bos_token_id: int | None = 50256
    eos_token_id: int | list[int] | None = 50256
    pad_token_id: int | None = None
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if self.attention_types is None:
            self.attention_types = [[["global", "local"], 12]]
        self.attention_layers = self.expand_attention_types_params(self.attention_types)
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if len(self.attention_layers) != self.num_layers:
            raise ValueError(
                "Configuration for convolutional module is incorrect. "
                "It is required that `len(config.attention_layers)` == `config.num_layers` "
                f"but is `len(config.attention_layers) = {len(self.attention_layers)}`, "
                f"`config.num_layers = {self.num_layers}`. "
                "`config.attention_layers` is prepared using `config.attention_types`. "
                "Please verify the value of `config.attention_types` argument."
            )

    @staticmethod
    def expand_attention_types_params(attention_types):
        attentions = []
        for item in attention_types:
            for _ in range(item[1]):
                attentions.extend(item[0])
        return attentions


def custom_unfold(input, dimension, size, step):
    """Custom torch.Tensor.unfold implementation to enable the export to ONNX."""
    import torch

    shape = input.size()
    rank = len(shape)
    sizedim = shape[dimension]

    low_indices = torch.arange(0, sizedim, step)
    min_length = torch.div(sizedim - size, step, rounding_mode="floor") + 1
    indices = torch.arange(size) + low_indices[:min_length][:, None]

    s = [slice(None)] * rank
    s[dimension] = indices
    sliced = input[s]

    perm = list(range(0, rank + 1))
    perm.append(perm.pop(dimension + 1))

    return sliced.permute(perm)


def custom_get_block_length_and_num_blocks(seq_length, window_size):
    """
    Custom implementation for GPTNeoAttentionMixin._get_block_length_and_num_blocks to enable the export to ONNX as
    original implementation uses Python variables and control flow.
    """
    import torch

    candidates = torch.arange(1, window_size)
    remainders = torch.remainder(seq_length, candidates)
    divisor_indices = remainders == 0
    divisors = candidates[divisor_indices]
    largest_divisor = torch.max(divisors)
    return largest_divisor, torch.div(seq_length, largest_divisor, rounding_mode="floor")


__all__ = ["GPTNeoConfig"]
