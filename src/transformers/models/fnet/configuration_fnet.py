# Copyright 2021 Google AI and The HuggingFace Inc. team. All rights reserved.
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
"""FNet model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="google/fnet-base")
@strict
class FNetConfig(PreTrainedConfig):
    r"""
    use_tpu_fourier_optimizations (`bool`, *optional*, defaults to `False`):
        Determines whether to use TPU optimized FFTs. If `True`, the model will favor axis-wise FFTs transforms.
        Set to `False` for GPU/CPU hardware, in which case n-dimensional FFTs are used.
    tpu_short_seq_length (`int`, *optional*, defaults to 512):
        The sequence length that is expected by the model when using TPUs. This will be used to initialize the DFT
        matrix only when *use_tpu_fourier_optimizations* is set to `True` and the input sequence is shorter than or
        equal to 4096 tokens.

    Example:

    ```python
    >>> from transformers import FNetConfig, FNetModel

    >>> # Initializing a FNet fnet-base style configuration
    >>> configuration = FNetConfig()

    >>> # Initializing a model (with random weights) from the fnet-base style configuration
    >>> model = FNetModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "fnet"

    vocab_size: int = 32000
    hidden_size: int = 768
    num_hidden_layers: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu_new"
    hidden_dropout_prob: float | int = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 4
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    use_tpu_fourier_optimizations: bool = False
    tpu_short_seq_length: int = 512
    pad_token_id: int | None = 3
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    tie_word_embeddings: bool = True


__all__ = ["FNetConfig"]
