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

from dataclasses import dataclass

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig


@strict(accept_kwargs=True)
@dataclass(repr=False)
class UdopConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`UdopForConditionalGeneration`]. It is used to
    instantiate a UDOP model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the UDOP
    [microsoft/udop-large](https://huggingface.co/microsoft/udop-large) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Arguments:
        vocab_size (`int`, *optional*, defaults to 33201):
            Vocabulary size of the UDOP model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`UdopForConditionalGeneration`].
        d_model (`int`, *optional*, defaults to 1024):
            Size of the encoder layers and the pooler layer.
        d_kv (`int`, *optional*, defaults to 64):
            Size of the key, query, value projections per attention head. The `inner_dim` of the projection layer will
            be defined as `num_heads * d_kv`.
        d_ff (`int`, *optional*, defaults to 4096):
            Size of the intermediate feed forward layer in each `UdopBlock`.
        num_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder and decoder.
        num_decoder_layers (`int`, *optional*):
            Number of hidden layers in the Transformer decoder. Will use the same value as `num_layers` if not set.
        num_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder and decoder.
        relative_attention_num_buckets (`int`, *optional*, defaults to 32):
            The number of buckets to use for each attention layer.
        relative_attention_max_distance (`int`, *optional*, defaults to 128):
            The maximum distance of the longer sequences for the bucket separation.
        relative_bias_args (`list[dict]`, *optional*, defaults to `[{'type': '1d'}, {'type': 'horizontal'}, {'type': 'vertical'}]`):
            A list of dictionaries containing the arguments for the relative bias layers.
        dropout_rate (`float`, *optional*, defaults to 0.1):
            The ratio for all dropout layers.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        feed_forward_proj (`string`, *optional*, defaults to `"relu"`):
            Type of feed forward layer to be used. Should be one of `"relu"` or `"gated-gelu"`. Udopv1.1 uses the
            `"gated-gelu"` feed forward projection. Original Udop uses `"relu"`.
        is_encoder_decoder (`bool`, *optional*, defaults to `True`):
            Whether the model should behave as an encoder/decoder or not.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        pad_token_id (`int`, *optional*, defaults to 0):
            The id of the padding token in the vocabulary.
        eos_token_id (`int`, *optional*, defaults to 1):
            The id of the end-of-sequence token in the vocabulary.
        max_2d_position_embeddings (`int`, *optional*, defaults to 1024):
            The maximum absolute position embeddings for relative position encoding.
        image_size (`int`, *optional*, defaults to 224):
            The size of the input images.
        patch_size (`int`, *optional*, defaults to 16):
            The patch size used by the vision encoder.
        num_channels (`int`, *optional*, defaults to 3):
            The number of channels in the input images.
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether to only use the decoder in an encoder-decoder architecture, otherwise it has no effect on
            decoder-only or encoder-only architectures.
        add_cross_attention (`bool`, *optional*, defaults to `False`):
            Whether cross-attention layers should be added to the model.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie weight embeddings
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
    eos_token_id: int | None = 1
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
