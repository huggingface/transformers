# Copyright 2022, The LongT5 Authors and HuggingFace Inc.
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
"""LongT5 model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="google/long-t5-local-base")
@strict
class LongT5Config(PreTrainedConfig):
    r"""
    d_ff (`int`, *optional*, defaults to 2048):
        Size of the intermediate feed forward layer in each `LongT5Block`.
    local_radius (`int`, *optional*, defaults to 127):
        Number of tokens to the left/right for each token to locally self-attend in a local attention mechanism.
    global_block_size (`int`, *optional*, defaults to 16):
        Length of blocks an input sequence is divided into for a global token representation. Used only for
        `encoder_attention_type = "transient-global"`.
    relative_attention_num_buckets (`int`, *optional*, defaults to 32):
        The number of buckets to use for each attention layer.
    relative_attention_max_distance (`int`, *optional*, defaults to 128):
        The maximum distance of the longer sequences for the bucket separation.
    feed_forward_proj (`string`, *optional*, defaults to `"relu"`):
        Type of feed forward layer to be used. Should be one of `"relu"` or `"gated-gelu"`. LongT5v1.1 uses the
        `"gated-gelu"` feed forward projection. Original LongT5 implementation uses `"gated-gelu"`.
    encoder_attention_type (`string`, *optional*, defaults to `"local"`):
        Type of encoder attention to be used. Should be one of `"local"` or `"transient-global"`, which are
        supported by LongT5 implementation.
    """

    model_type = "longt5"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
        "head_dim": "d_kv",
    }

    vocab_size: int = 32128
    d_model: int = 512
    d_kv: int = 64
    d_ff: int = 2048
    num_layers: int = 6
    num_decoder_layers: int | None = None
    num_heads: int = 8
    local_radius: int = 127
    global_block_size: int = 16
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    dropout_rate: float | int = 0.1
    layer_norm_epsilon: float = 1e-6
    initializer_factor: float = 1.0
    feed_forward_proj: str = "relu"
    is_encoder_decoder: bool = True
    encoder_attention_type: str = "local"
    use_cache: bool = True
    pad_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    bos_token_id: int | None = None
    is_decoder: bool = False
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        self.num_decoder_layers = self.num_decoder_layers if self.num_decoder_layers is not None else self.num_layers
        act_info = self.feed_forward_proj.split("-")
        self.dense_act_fn = act_info[-1]
        self.is_gated_act = act_info[0] == "gated"

        if self.feed_forward_proj == "gated-gelu":
            self.dense_act_fn = "gelu_new"

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        act_info = self.feed_forward_proj.split("-")
        if len(act_info) > 1 and act_info[0] != "gated" or len(act_info) > 2:
            raise ValueError(
                f"`feed_forward_proj`: {self.feed_forward_proj} is not a valid activation function of the dense layer. "
                "Please make sure `feed_forward_proj` is of the format `gated-{ACT_FN}` or `{ACT_FN}`, e.g. "
                "'gated-gelu' or 'relu'"
            )


__all__ = ["LongT5Config"]
