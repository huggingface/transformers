# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""CharacterBERT model configuration."""

from huggingface_hub.dataclasses import strict

from ...utils import auto_docstring
from ..bert.configuration_bert import BertConfig


@auto_docstring(checkpoint="helboukkouri/character-bert-base-uncased")
@strict(accept_kwargs=True)
class CharacterBertConfig(BertConfig):
    r"""
    character_embedding_dim (`int`, *optional*, defaults to 16):
        Character embedding dimension used before the convolution stack.
    character_vocab_size (`int`, *optional*, defaults to 262):
        CharacterBERT byte-level tokenization expects exactly 262 character IDs before applying the +1 offset for
        masking and padding (256 byte values + 6 special markers).
    max_characters_per_token (`int`, *optional*, defaults to 50):
        Maximum number of characters represented for each token. Must be at least the width of the widest
        convolution in `character_cnn_filters`.
    character_cnn_filters (`tuple[tuple[int, int], ...]`, *optional*, defaults to `((1, 32), (2, 32), (3, 64), (4, 128), (5, 256), (6, 512), (7, 1024))`):
        Convolution widths and output channels used in the character CNN.
    num_highway_layers (`int`, *optional*, defaults to 2):
        Number of highway layers applied after the convolution outputs.

    Examples:

    ```python
    >>> from transformers import CharacterBertConfig, CharacterBertModel

    >>> configuration = CharacterBertConfig()
    >>> model = CharacterBertModel(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "character_bert"
    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int | None = 0
    use_cache: bool = True
    classifier_dropout: float | int | None = None
    is_decoder: bool = False
    add_cross_attention: bool = False
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    tie_word_embeddings: bool = False
    character_embedding_dim: int = 16
    character_vocab_size: int = 262
    max_characters_per_token: int = 50
    character_cnn_filters: tuple[tuple[int, int], ...] | list[tuple[int, int]] | list[list[int]] = (
        (1, 32),
        (2, 32),
        (3, 64),
        (4, 128),
        (5, 256),
        (6, 512),
        (7, 1024),
    )
    num_highway_layers: int = 2

    def __post_init__(self, **kwargs):
        legacy_character_embedding_dim = kwargs.pop("character_embeddings_dim", None)
        legacy_character_cnn_filters = kwargs.pop("cnn_filters", None)
        legacy_max_characters_per_token = kwargs.pop("max_word_length", None)
        legacy_mlm_vocab_size = kwargs.pop("mlm_vocab_size", None)

        if legacy_character_embedding_dim is not None and self.character_embedding_dim == 16:
            self.character_embedding_dim = legacy_character_embedding_dim
        if legacy_character_cnn_filters is not None and self.character_cnn_filters == (
            (1, 32),
            (2, 32),
            (3, 64),
            (4, 128),
            (5, 256),
            (6, 512),
            (7, 1024),
        ):
            self.character_cnn_filters = legacy_character_cnn_filters
        if legacy_max_characters_per_token is not None and self.max_characters_per_token == 50:
            self.max_characters_per_token = legacy_max_characters_per_token

        if legacy_mlm_vocab_size is not None and self.vocab_size == 30522:
            self.vocab_size = legacy_mlm_vocab_size

        self.character_cnn_filters = tuple(
            (int(width), int(channels)) for width, channels in self.character_cnn_filters
        )

        super().__post_init__(**kwargs)

        if self.character_vocab_size != 262:
            raise ValueError(
                "`character_vocab_size` must be 262 for CharacterBERT byte-level tokenization "
                "(256 bytes + 6 special characters)."
            )
        if len(self.character_cnn_filters) == 0:
            raise ValueError("`character_cnn_filters` must contain at least one convolution specification.")

        widest_filter = max(width for width, _ in self.character_cnn_filters)
        if self.max_characters_per_token < widest_filter:
            raise ValueError(
                "`max_characters_per_token` must be at least the width of the widest character CNN filter "
                f"({widest_filter})."
            )

    @property
    def mlm_vocab_size(self) -> int:
        # Legacy alias kept for backward compatibility with older CharacterBERT checkpoints.
        return self.vocab_size

    @mlm_vocab_size.setter
    def mlm_vocab_size(self, value: int) -> None:
        self.vocab_size = value


__all__ = ["CharacterBertConfig"]
