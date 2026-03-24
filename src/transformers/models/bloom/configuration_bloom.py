# Copyright 2022 the Big Science Workshop and HuggingFace Inc. team.  All rights reserved.
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
"""Bloom configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="bigscience/bloom")
@strict
class BloomConfig(PreTrainedConfig):
    r"""
    apply_residual_connection_post_layernorm (`bool`, *optional*, defaults to `False`):
        If enabled, use the layer norm of the hidden states as the residual in the transformer blocks
    slow_but_exact (`bool`, *optional*, defaults to `False`):
        Experimental feature. Whether to use slow but exact implementation of the attention mechanism. While
        merging the TP rank tensors, due to slicing operations the results may be slightly different between the
        model trained on Megatron and our model. Please refer to [this
        issue](https://github.com/pytorch/pytorch/issues/76232). A solution to obtain more accurate results is to
        enable this feature. Enabling this will hurt the computational time of the inference. Will be probably
        resolved in the future once the main model has been fine-tuned with TP_rank=1.

    Example:

    ```python
    >>> from transformers import BloomConfig, BloomModel

    >>> # Initializing a Bloom configuration
    >>> configuration = BloomConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = BloomModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "bloom"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_hidden_layers": "n_layer",
        "num_attention_heads": "n_head",
    }

    vocab_size: int = 250880
    hidden_size: int = 64
    n_layer: int = 2
    n_head: int = 8
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    use_cache: bool = True
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    pad_token_id: int | None = None
    apply_residual_connection_post_layernorm: bool = False
    hidden_dropout: float | int = 0.0
    attention_dropout: float | int = 0.0
    pretraining_tp: int = 1  # TP rank used when training with megatro
    slow_but_exact: bool = False
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        # Backward compatibility with n_embed kwarg
        n_embed = kwargs.pop("n_embed", None)
        self.hidden_size = self.hidden_size if n_embed is None else n_embed
        super().__post_init__(**kwargs)


__all__ = ["BloomConfig"]
