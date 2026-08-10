# Copyright 2023, HuggingFace Inc.
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
"""NLLB-MoE model configuration"""

from typing import Literal

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="facebook/nllb-moe-54b")
@strict
class NllbMoeConfig(PreTrainedConfig):
    r"""
    router_bias (`bool`, *optional*, defaults to `False`):
        Whether or not the classifier of the router should have a bias.
    router_dtype (`str`, *optional*, default to `"float32"`):
        The `dtype` used for the routers. It is preferable to keep the `dtype` to `"float32"` as specified in the
        *selective precision* discussion in [the paper](https://huggingface.co/papers/2101.03961).
    router_ignore_padding_tokens (`bool`, *optional*, defaults to `False`):
        Whether to ignore padding tokens when routing. if `False`, the padding tokens are not routed to any
        experts.
    expert_capacity (`int`, *optional*, defaults to 64):
        Number of tokens that can be stored in each expert.
    encoder_sparse_step (`int`, *optional*, defaults to 4):
        Frequency of the sparse layers in the encoder. 4 means that one out of 4 layers will be sparse.
    decoder_sparse_step (`int`, *optional*, defaults to 4):
        Frequency of the sparse layers in the decoder. 4 means that one out of 4 layers will be sparse.
    second_expert_policy (`str`, *optional*, default to `"all"`):
        The policy used for the sampling the probability of being sampled to a second expert for each token.
    normalize_router_prob_before_dropping (`bool`, *optional*, defaults to `True`):
        Whether or not to normalize the router probabilities before applying a mask based on the experts capacity
        (capacity dropping).
    batch_prioritized_routing (`bool`, *optional*, defaults to `True`):
        Whether or not to orders the tokens by their router probabilities before capacity dropping. This means that
        the tokens that have the highest probabilities will be routed before other tokens that might be further in
        the sequence.
    moe_eval_capacity_token_fraction (`float`, *optional*, defaults to 1.0):
        Fraction of tokens as capacity during validation, if set to negative, uses the same as training. Should be
        in range: (0.0, 1.0].
    moe_token_dropout (`float`, *optional*, default to 0.2):
        Masking rate for MoE expert output masking (EOM), which is implemented via a Dropout2d on the expert
        outputs.

    Example:

    ```python
    >>> from transformers import NllbMoeModel, NllbMoeConfig

    >>> # Initializing a NllbMoe facebook/nllb-moe-54b style configuration
    >>> configuration = NllbMoeConfig()

    >>> # Initializing a model from the facebook/nllb-moe-54b style configuration
    >>> model = NllbMoeModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "nllb-moe"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_attention_heads": "encoder_attention_heads",
        "hidden_size": "d_model",
        "num_hidden_layers": "encoder_layers",
    }

    vocab_size: int = 128112
    max_position_embeddings: int = 1024
    encoder_layers: int = 12
    encoder_ffn_dim: int = 4096
    encoder_attention_heads: int = 16
    decoder_layers: int = 12
    decoder_ffn_dim: int = 4096
    decoder_attention_heads: int = 16
    encoder_layerdrop: float | int = 0.05
    decoder_layerdrop: float | int = 0.05
    use_cache: bool = True
    is_encoder_decoder: bool = True
    activation_function: str = "relu"
    d_model: int = 1024
    dropout: float | int = 0.1
    attention_dropout: float | int = 0.1
    activation_dropout: float | int = 0.0
    init_std: float = 0.02
    decoder_start_token_id: int | None = 2
    scale_embedding: bool = True
    router_bias: bool = False
    router_dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    router_ignore_padding_tokens: bool = False
    num_experts: int = 128
    expert_capacity: int = 64
    encoder_sparse_step: int = 4
    decoder_sparse_step: int = 4
    router_z_loss_coef: float = 0.001
    router_aux_loss_coef: float = 0.001
    second_expert_policy: str = "all"
    normalize_router_prob_before_dropping: bool = False
    batch_prioritized_routing: bool = False
    moe_eval_capacity_token_fraction: float = 1.0
    moe_token_dropout: float | int = 0.2
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
    tie_word_embeddings: bool = True
    output_router_logits: bool = False


__all__ = ["NllbMoeConfig"]
