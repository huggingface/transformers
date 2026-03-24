# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Pop2Piano model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="sweetcocoa/pop2piano")
@strict
class Pop2PianoConfig(PreTrainedConfig):
    r"""
    composer_vocab_size (`int`, *optional*, defaults to 21):
        Denotes the number of composers.
    relative_attention_num_buckets (`int`, *optional*, defaults to 32):
        The number of buckets to use for each attention layer.
    relative_attention_max_distance (`int`, *optional*, defaults to 128):
        The maximum distance of the longer sequences for the bucket separation.
    feed_forward_proj (`string`, *optional*, defaults to `"gated-gelu"`):
        Type of feed forward layer to be used. Should be one of `"relu"` or `"gated-gelu"`.
    dense_act_fn (`string`, *optional*, defaults to `"relu"`):
        Type of Activation Function to be used in `Pop2PianoDenseActDense` and in `Pop2PianoDenseGatedActDense`.
    """

    model_type = "pop2piano"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_hidden_layers": "num_layers", "hidden_size": "d_model", "num_attention_heads": "num_heads"}

    vocab_size: int = 2400
    composer_vocab_size: int = 21
    d_model: int = 512
    d_kv: int = 64
    d_ff: int = 2048
    num_layers: int = 6
    num_decoder_layers: int | None = None
    num_heads: int = 8
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    dropout_rate: float = 0.1
    layer_norm_epsilon: float = 1e-6
    initializer_factor: float = 1.0
    feed_forward_proj: str = "gated-gelu"
    is_encoder_decoder: bool = True
    use_cache: bool = True
    pad_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    dense_act_fn: str = "relu"
    is_decoder: bool = False
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        self.num_decoder_layers = self.num_decoder_layers if self.num_decoder_layers is not None else self.num_layers
        self.is_gated_act = self.feed_forward_proj.split("-")[0] == "gated"
        super().__post_init__(**kwargs)


__all__ = ["Pop2PianoConfig"]
