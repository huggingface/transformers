# Copyright 2020, Hugging Face
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
"""Funnel Transformer model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="funnel-transformer/small")
@strict
class FunnelConfig(PreTrainedConfig):
    r"""
    block_sizes (`list[int]`, *optional*, defaults to `[4, 4, 4]`):
        The sizes of the blocks used in the model.
    block_repeats (`list[int]`, *optional*):
        If passed along, each layer of each block is repeated the number of times indicated.
    num_decoder_layers (`int`, *optional*, defaults to 2):
        The number of layers in the decoder (when not using the base model).
    pooling_type (`str`, *optional*, defaults to `"mean"`):
        Possible values are `"mean"` or `"max"`. The way pooling is performed at the beginning of each block.
    attention_type (`str`, *optional*, defaults to `"relative_shift"`):
        Possible values are `"relative_shift"` or `"factorized"`. The former is faster on CPU/GPU while the latter
        is faster on TPU.
    separate_cls (`bool`, *optional*, defaults to `True`):
        Whether or not to separate the cls token when applying pooling.
    truncate_seq (`bool`, *optional*, defaults to `True`):
        When using `separate_cls`, whether or not to truncate the last token when pooling, to avoid getting a
        sequence length that is not a multiple of 2.
    pool_q_only (`bool`, *optional*, defaults to `True`):
        Whether or not to apply the pooling only to the query or to query, key and values for the attention layers.
    """

    model_type = "funnel"
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "n_head",
    }

    vocab_size: int = 30522
    block_sizes: list[int] | tuple[int, ...] = (4, 4, 4)
    block_repeats: list[int] | None = None
    num_decoder_layers: int = 2
    d_model: int = 768
    n_head: int = 12
    d_head: int = 64
    d_inner: int = 3072
    hidden_act: str = "gelu_new"
    hidden_dropout: float | int = 0.1
    attention_dropout: float | int = 0.1
    activation_dropout: float | int = 0.0
    initializer_range: float = 0.1
    initializer_std: float | None = None
    layer_norm_eps: float = 1e-9
    pooling_type: str = "mean"
    attention_type: str = "relative_shift"
    separate_cls: bool = True
    truncate_seq: bool = True
    pool_q_only: bool = True
    pad_token_id: int | None = None
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        self.block_repeats = [1] * len(self.block_sizes) if self.block_repeats is None else self.block_repeats
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if len(self.block_sizes) != len(self.block_repeats):
            raise ValueError("`block_sizes` and `block_repeats` should have the same length.")
        if self.pooling_type not in [
            "mean",
            "max",
        ]:
            raise ValueError(f"Got {self.pooling_type} for `pooling_type` but only 'mean' and 'max' are supported.")
        if self.attention_type not in [
            "relative_shift",
            "factorized",
        ]:
            raise ValueError(
                f"Got {self.attention_type} for `attention_type` but only 'relative_shift' and 'factorized' are supported."
            )

    @property
    def num_hidden_layers(self):
        return sum(self.block_sizes)

    @num_hidden_layers.setter
    def num_hidden_layers(self, value):
        raise NotImplementedError(
            "This model does not support the setting of `num_hidden_layers`. Please set `block_sizes`."
        )

    @property
    def num_blocks(self):
        return len(self.block_sizes)

    @num_blocks.setter
    def num_blocks(self, value):
        raise NotImplementedError("This model does not support the setting of `num_blocks`. Please set `block_sizes`.")


__all__ = ["FunnelConfig"]
