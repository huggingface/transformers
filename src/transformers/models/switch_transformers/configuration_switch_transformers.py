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

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="google/switch-base-8")
class SwitchTransformersConfig(PreTrainedConfig):
    r"""
    num_sparse_encoder_layers (`int`, *optional*, defaults to 3):
        Number of sparse (MoE) dense hidden layers in the Transformer encoder layer.
        Note: When set to 0 with `num_layers=1`, the current implementation may still create a sparse layer
        due to the sparse step calculation. This edge case is not encountered in existing checkpoints.
    num_decoder_layers (`int`, *optional*, defaults to 12):
        Number of hidden layers in the Transformer decoder. Will use the same value as `num_layers` if not set.
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

    def __init__(
        self,
        vocab_size=32128,
        d_model=768,
        d_kv=64,
        d_ff=2048,
        expert_capacity=64,
        num_layers=12,
        num_sparse_encoder_layers=3,
        num_decoder_layers=12,
        num_sparse_decoder_layers=3,
        num_heads=12,
        num_experts=8,
        router_bias=False,
        router_jitter_noise=0.01,
        router_dtype="float32",
        router_ignore_padding_tokens=False,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        router_z_loss_coef=0.001,
        router_aux_loss_coef=0.001,
        initializer_factor=1.0,
        dense_act_fn="relu",
        is_encoder_decoder=True,
        add_router_probs=False,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=None,
        tie_word_embeddings=True,
        is_decoder=False,
        add_cross_attention=False,
        **kwargs,
    ):
        self.is_decoder = is_decoder
        self.add_cross_attention = add_cross_attention
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff

        self.num_sparse_encoder_layers = num_sparse_encoder_layers

        self.num_layers = num_layers
        self.num_decoder_layers = (
            num_decoder_layers if num_decoder_layers is not None else self.num_layers
        )  # default = symmetry
        self.num_sparse_decoder_layers = num_sparse_decoder_layers

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

        self.num_heads = num_heads
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        self.router_bias = router_bias
        self.router_jitter_noise = router_jitter_noise
        if router_dtype not in ["float32", "float16", "bfloat16"]:
            raise ValueError(f"`router_dtype` must be one of 'float32', 'float16' or 'bfloat16', got {router_dtype}")
        self.router_dtype = router_dtype

        self.router_ignore_padding_tokens = router_ignore_padding_tokens
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance

        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.use_cache = use_cache
        self.add_router_probs = add_router_probs

        self.router_z_loss_coef = router_z_loss_coef
        self.router_aux_loss_coef = router_aux_loss_coef
        self.dense_act_fn = dense_act_fn
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings

        super().__init__(is_encoder_decoder=is_encoder_decoder, **kwargs)


__all__ = ["SwitchTransformersConfig"]
