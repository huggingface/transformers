# Copyright 2023, The T5 Authors and HuggingFace Inc.
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
"""UMT5 model configuration"""

from dataclasses import dataclass

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig


@strict(accept_kwargs=True)
@dataclass(repr=False)
class UMT5Config(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`UMT5Model`]. It is used to instantiate a UMT5
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the UMT5
    [google/umt5-small](https://huggingface.co/google/umt5-small) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Arguments:
        vocab_size (`int`, *optional*, defaults to 250112):
            Vocabulary size of the UMT5 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`UMT5Model`].
        d_model (`int`, *optional*, defaults to 512):
            Size of the encoder layers and the pooler layer.
        d_kv (`int`, *optional*, defaults to 64):
            Size of the key, query, value projections per attention head. `d_kv` has to be equal to `d_model //
            num_heads`.
        d_ff (`int`, *optional*, defaults to 1024):
            Size of the intermediate feed forward layer in each `UMT5Block`.
        num_layers (`int`, *optional*, defaults to 8):
            Number of hidden layers in the Transformer encoder.
        num_decoder_layers (`int`, *optional*):
            Number of hidden layers in the Transformer decoder. Will use the same value as `num_layers` if not set.
        num_heads (`int`, *optional*, defaults to 6):
            Number of attention heads for each attention layer in the Transformer encoder.
        relative_attention_num_buckets (`int`, *optional*, defaults to 32):
            The number of buckets to use for each attention layer.
        relative_attention_max_distance (`int`, *optional*, defaults to 128):
            The maximum distance of the longer sequences for the bucket separation.
        dropout_rate (`float`, *optional*, defaults to 0.1):
            The ratio for all dropout layers.
        classifier_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for classifier.
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the layer normalization layers.
        initializer_factor (`float`, *optional*, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        feed_forward_proj (`string`, *optional*, defaults to `"gated-gelu"`):
            Type of feed forward layer to be used. Should be one of `"relu"` or `"gated-gelu"`.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
    """

    model_type = "umt5"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
        "head_dim": "d_kv",
    }

    vocab_size: int = 250112
    d_model: int = 512
    d_kv: int = 64
    d_ff: int = 1024
    num_layers: int = 8
    num_decoder_layers: int | None = None
    num_heads: int = 6
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    dropout_rate: float = 0.1
    layer_norm_epsilon: float = 1e-6
    initializer_factor: float = 1.0
    feed_forward_proj: str = "gated-gelu"
    is_encoder_decoder: bool = True
    use_cache: bool = True
    tokenizer_class: str = "T5Tokenizer"
    pad_token_id: int | None = 0
    eos_token_id: int | None = 1
    decoder_start_token_id: int | None = 0
    classifier_dropout: float | int = 0.0
    is_decoder: bool = False

    def __post_init__(self, **kwargs):
        self.num_decoder_layers = (
            self.num_decoder_layers if self.num_decoder_layers is not None else self.num_layers
        )  # default = symmetry
        act_info = self.feed_forward_proj.split("-")
        self.dense_act_fn = act_info[-1]
        self.is_gated_act = act_info[0] == "gated"

        if self.feed_forward_proj == "gated-gelu":
            self.dense_act_fn = "gelu_new"

        kwargs.pop("tie_word_embeddings", None)
        self.tie_word_embeddings = True  # force it for T5 family

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


__all__ = ["UMT5Config"]
