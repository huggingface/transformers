# Copyright 2022, Google and HuggingFace Inc.
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
"""Switch Transformers model configuration"""

from typing import Literal

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="google/switch-base-8")
@strict
class SwitchTransformersConfig(PreTrainedConfig):
    r"""
    num_sparse_encoder_layers (`int`, *optional*, defaults to 3):
        Number of sparse (MoE) dense hidden layers in the Transformer encoder layer.
        Note: When set to 0 with `num_layers=1`, the current implementation may still create a sparse layer
        due to the sparse step calculation. This edge case is not encountered in existing checkpoints.
    num_sparse_decoder_layers (`int`, *optional*, defaults to 3):
        Number of sparse (MoE) dense hidden layers in the Transformer decoder layer.
        Note: When set to 0 with `num_decoder_layers=1`, the current implementation may still create a sparse
        layer due to the sparse step calculation. This edge case is not encountered in existing checkpoints.
    router_bias (`bool`, *optional*, defaults to `False`):
        Whether to add a bias to the router.
    router_dtype (`str`, *optional*, default to `"float32"`):
        The `dtype` used for the routers. It is preferable to keep the `dtype` to `"float32"` as specified in the
        *selective precision* discussion in [the paper](https://huggingface.co/papers/2101.03961).
    router_ignore_padding_tokens (`bool`, *optional*, defaults to `False`):
        Whether to ignore padding tokens when routing.
    relative_attention_num_buckets (`int`, *optional*, defaults to 32):
        The number of buckets to use for each attention layer.
    relative_attention_max_distance (`int`, *optional*, defaults to 128):
        The maximum distance of the longer sequences for the bucket separation.
    dense_act_fn (`string`, *optional*, defaults to `"relu"`):
        Type of feed forward layer to be used. Should be one of `"relu"` or `"gated-gelu"`. SwitchTransformersv1.1
        uses the `"gated-gelu"` feed forward projection. Original SwitchTransformers uses `"relu"`.
    add_router_probs (`bool`, *optional*, defaults to `False`):
        Whether to output router probabilities to compute router auxiliary loss.
    """

    model_type = "switch_transformers"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"hidden_size": "d_model", "num_attention_heads": "num_heads", "num_hidden_layers": "num_layers"}

    vocab_size: int = 32128
    d_model: int = 768
    d_kv: int = 64
    d_ff: int = 2048
    expert_capacity: int = 64
    num_layers: int = 12
    num_sparse_encoder_layers: int = 3
    num_decoder_layers: int | None = 12
    num_sparse_decoder_layers: int = 3
    num_heads: int = 12
    num_experts: int = 8
    router_bias: bool = False
    router_jitter_noise: int | float = 0.01
    router_dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    router_ignore_padding_tokens: bool = False
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    dropout_rate: float = 0.1
    layer_norm_epsilon: float = 1e-6
    router_z_loss_coef: float = 0.001
    router_aux_loss_coef: float = 0.001
    initializer_factor: float = 1.0
    dense_act_fn: str = "relu"
    is_encoder_decoder: bool = True
    add_router_probs: bool = False
    use_cache: bool = True
    pad_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    bos_token_id: int | None = None
    tie_word_embeddings: bool = True
    is_decoder: bool = False
    add_cross_attention: bool = False

    def __post_init__(self, **kwargs):
        self.num_decoder_layers = (
            self.num_decoder_layers if self.num_decoder_layers is not None else self.num_layers
        )  # default = symmetry

        # This tells us, each how many encoder layer we'll have to set a sparse layer.
        if self.num_sparse_encoder_layers > 0:
            self.encoder_sparse_step = self.num_layers // self.num_sparse_encoder_layers
        else:
            self.encoder_sparse_step = self.num_layers  # HACK: this will create 0 sparse layers

        # This tells us, each how many decoder layer we'll have to set a sparse layer.
        if self.num_sparse_decoder_layers > 0:
            self.decoder_sparse_step = self.num_decoder_layers // self.num_sparse_decoder_layers
        else:
            self.decoder_sparse_step = self.num_decoder_layers  # HACK: this will create 0 sparse layers

        super().__post_init__(**kwargs)


__all__ = ["SwitchTransformersConfig"]
