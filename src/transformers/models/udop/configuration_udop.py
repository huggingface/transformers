# Copyright 2024 HuggingFace Inc.
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
"""UDOP model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="microsoft/udop-large")
@strict
class UdopConfig(PreTrainedConfig):
    r"""
    relative_attention_num_buckets (`int`, *optional*, defaults to 32):
        The number of buckets to use for each attention layer.
    relative_attention_max_distance (`int`, *optional*, defaults to 128):
        The maximum distance of the longer sequences for the bucket separation.
    relative_bias_args (`list[dict]`, *optional*, defaults to `[{'type': '1d'}, {'type': 'horizontal'}, {'type': 'vertical'}]`):
        A list of dictionaries containing the arguments for the relative bias layers.
    feed_forward_proj (`string`, *optional*, defaults to `"relu"`):
        Type of feed forward layer to be used. Should be one of `"relu"` or `"gated-gelu"`. Udopv1.1 uses the
        `"gated-gelu"` feed forward projection. Original Udop uses `"relu"`.
    max_2d_position_embeddings (`int`, *optional*, defaults to 1024):
        The maximum absolute position embeddings for relative position encoding.
    """

    model_type = "udop"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"hidden_size": "d_model", "num_attention_heads": "num_heads", "num_hidden_layers": "num_layers"}

    vocab_size: int = 33201
    d_model: int = 1024
    d_kv: int = 64
    d_ff: int = 4096
    num_layers: int = 24
    num_decoder_layers: int | None = None
    num_heads: int = 16
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    relative_bias_args: list[dict] | None = None
    dropout_rate: float = 0.1
    layer_norm_epsilon: float = 1e-6
    initializer_factor: float = 1.0
    feed_forward_proj: str = "relu"
    is_encoder_decoder: bool = True
    use_cache: bool = True
    pad_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    max_2d_position_embeddings: int = 1024
    image_size: int | list[int] | tuple[int, int] = 224
    patch_size: int | list[int] | tuple[int, int] = 16
    num_channels: int = 3
    is_decoder: bool = False
    add_cross_attention: bool = False
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if self.relative_bias_args is None:
            self.relative_bias_args = [{"type": "1d"}, {"type": "horizontal"}, {"type": "vertical"}]

        self.num_decoder_layers = (
            self.num_decoder_layers if self.num_decoder_layers is not None else self.num_layers
        )  # default = symmetry

        act_info = self.feed_forward_proj.split("-")
        self.dense_act_fn = act_info[-1]
        self.is_gated_act = act_info[0] == "gated"

        kwargs.pop("tie_word_embeddings", None)
        self.tie_word_embeddings = True
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        act_info = self.feed_forward_proj.split("-")
        if len(act_info) > 1 and act_info[0] != "gated" or len(act_info) > 2:
            raise ValueError(
                f"`feed_forward_proj`: {self.feed_forward_proj} is not a valid activation function of the dense layer."
                "Please make sure `feed_forward_proj` is of the format `gated-{ACT_FN}` or `{ACT_FN}`, e.g. "
                "'gated-gelu' or 'relu'"
            )


__all__ = ["UdopConfig"]
